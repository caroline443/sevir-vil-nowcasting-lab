"""Losses targeting diagnosed SEVIR VIL forecast failure modes."""

from __future__ import annotations

from collections.abc import Sequence
import math
import random

import torch
from torch import nn
from torch.nn import functional as F


class SoftExceedanceAreaLoss(nn.Module):
    """Match severe-threshold area while remaining insensitive to displacement.

    Each hard exceedance indicator is replaced by a sigmoid.  The spatially
    summed soft areas are compared after ``log1p`` compression, separately for
    every sample, lead time and threshold.  Unlike pixelwise overlap losses,
    translating a core without changing its intensity distribution leaves this
    objective unchanged.

    Inputs are expected to be normalized VIL tensors with shape ``[B,T,C,H,W]``.
    Threshold and temperature arguments retain raw SEVIR VIL units for readable
    experiment configuration.
    """

    def __init__(
        self,
        thresholds_raw: Sequence[float] = (160.0, 181.0, 219.0),
        temperature_raw: float = 2.0,
    ) -> None:
        super().__init__()
        if not thresholds_raw:
            raise ValueError("at least one threshold is required")
        if temperature_raw <= 0:
            raise ValueError("temperature_raw must be positive")
        thresholds = torch.as_tensor(tuple(thresholds_raw), dtype=torch.float32)
        if not torch.isfinite(thresholds).all() or (thresholds < 0).any():
            raise ValueError("thresholds_raw must be finite and non-negative")
        self.register_buffer("thresholds", thresholds / 255.0)
        self.temperature = float(temperature_raw) / 255.0

    def soft_counts(self, values: torch.Tensor) -> torch.Tensor:
        """Return ``[B,T,K]`` differentiable exceedance pixel counts."""
        if values.ndim != 5:
            raise ValueError(f"expected [B,T,C,H,W], got shape {tuple(values.shape)}")
        # Tail calculations stay in float32 under AMP: a two-VIL-unit sigmoid
        # is intentionally sharp and float16 would discard useful gradients.
        values = values.float().unsqueeze(2)
        thresholds = self.thresholds.view(1, 1, -1, 1, 1, 1)
        probabilities = torch.sigmoid((values - thresholds) / self.temperature)
        return probabilities.sum(dim=(3, 4, 5))

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                f"prediction and target shapes differ: {tuple(prediction.shape)} "
                f"vs {tuple(target.shape)}"
            )
        predicted_log_area = torch.log1p(self.soft_counts(prediction))
        target_log_area = torch.log1p(self.soft_counts(target).detach())
        return F.smooth_l1_loss(predicted_log_area, target_log_area)


class ProbabilityMatchingLoss(nn.Module):
    """Match each forecast field's empirical intensity distribution.

    This implements the probability-matching constraint from Cao et al.
    (GRL, 2025): flatten each sample/lead field, sort forecast and target
    intensities independently, and compute MSE between the ordered values.
    Spatial locations are therefore ignored by this term while every
    intensity quantile is constrained.
    """

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                f"prediction and target shapes differ: {tuple(prediction.shape)} "
                f"vs {tuple(target.shape)}"
            )
        if prediction.ndim != 5:
            raise ValueError(
                f"expected [B,T,C,H,W], got shape {tuple(prediction.shape)}"
            )
        # The published constraint is applied to each forecast/observation
        # field. Keep sorting and the loss in FP32 under mixed precision.
        predicted_ordered = torch.sort(
            prediction.float().flatten(start_dim=2), dim=-1
        ).values
        target_ordered = torch.sort(
            target.float().flatten(start_dim=2), dim=-1
        ).values.detach()
        return F.mse_loss(predicted_ordered, target_ordered)


class FourierAmplitudeCorrelationLoss(nn.Module):
    """Official FACL training schedule adapted to the local trainer.

    Yan et al. (NeurIPS 2024) alternate stochastically between global Fourier
    correlation loss (FCL) and Fourier amplitude loss (FAL). The probability
    of selecting FAL rises linearly during training and is one throughout the
    final ``constant_ratio`` of updates. The spatial loss is scaled by
    ``sqrt(H * W)`` as in the official MIT-licensed implementation.
    """

    def __init__(self, total_steps: int, constant_ratio: float = 0.1) -> None:
        super().__init__()
        if total_steps < 2:
            raise ValueError("total_steps must be at least two")
        if not 0 <= constant_ratio < 1:
            raise ValueError("constant_ratio must be in [0, 1)")
        self.total_steps = int(total_steps)
        self.constant_ratio = float(constant_ratio)
        self.transition_steps = max(
            2, self.total_steps - int(self.total_steps * self.constant_ratio)
        )
        self.step = 0
        self.last_term = "fcl"

    @staticmethod
    def _transforms(
        prediction: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prediction_fft = torch.fft.fftn(
            prediction.float(), dim=(-2, -1), norm="ortho"
        )
        target_fft = torch.fft.fftn(
            target.float(), dim=(-2, -1), norm="ortho"
        )
        return prediction_fft, target_fft

    @staticmethod
    def _fcl(
        prediction_fft: torch.Tensor, target_fft: torch.Tensor
    ) -> torch.Tensor:
        numerator = (torch.conj(prediction_fft) * target_fft).sum().real
        denominator = torch.sqrt(
            target_fft.abs().square().sum()
            * prediction_fft.abs().square().sum()
        ).clamp_min(torch.finfo(torch.float32).tiny)
        return 1.0 - numerator / denominator

    @staticmethod
    def _fal(
        prediction_fft: torch.Tensor, target_fft: torch.Tensor
    ) -> torch.Tensor:
        return F.mse_loss(prediction_fft.abs(), target_fft.abs())

    def fal_probability(self) -> float:
        if self.step >= self.transition_steps:
            return 1.0
        # Official code uses a threshold decreasing from one to zero and
        # selects FAL when a uniform random draw exceeds that threshold.
        threshold = 1.0 - self.step / (self.transition_steps - 1)
        return 1.0 - threshold

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        if prediction.shape != target.shape or prediction.ndim != 5:
            raise ValueError("FACL expects matching [B,T,C,H,W] tensors")
        prediction_fft, target_fft = self._transforms(prediction, target)
        use_fal = random.random() < self.fal_probability()
        if use_fal:
            loss = self._fal(prediction_fft, target_fft)
            self.last_term = "fal"
        else:
            loss = self._fcl(prediction_fft, target_fft)
            self.last_term = "fcl"
        self.step += 1
        return loss * math.sqrt(prediction.shape[-2] * prediction.shape[-1])
