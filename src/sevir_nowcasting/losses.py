"""Losses targeting diagnosed SEVIR VIL forecast failure modes."""

from __future__ import annotations

from collections.abc import Sequence

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
