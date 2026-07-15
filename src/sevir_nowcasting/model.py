"""A compact SimVP-style baseline for asymmetric input/output sequences.

The implementation follows the high-level SimVP design: a frame-wise spatial
encoder, a channel-mixed temporal translator, and a frame-wise decoder. It is
kept local and small so experiments do not depend on an older OpenSTL stack.
It is not intended to reproduce OpenSTL weights or scores bit-for-bit.
"""

from __future__ import annotations

from math import gcd

import torch
from torch import Tensor, nn


def _group_count(channels: int, preferred: int = 8) -> int:
    """Return a valid GroupNorm group count."""

    return max(1, gcd(channels, preferred))


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int | None = None,
    ) -> None:
        if padding is None:
            padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


class InceptionResidualBlock(nn.Module):
    """Multi-kernel depth-grouped spatial mixing with a residual connection."""

    def __init__(
        self,
        channels: int,
        *,
        kernels: tuple[int, ...] = (3, 5, 7, 11),
        groups: int = 8,
    ) -> None:
        super().__init__()
        groups = max(1, gcd(channels, groups))
        self.pre = nn.Sequential(
            nn.GroupNorm(_group_count(channels), channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
        )
        self.branches = nn.ModuleList(
            nn.Conv2d(
                channels,
                channels,
                kernel,
                padding=kernel // 2,
                groups=groups,
                bias=False,
            )
            for kernel in kernels
        )
        self.mix = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        hidden = self.pre(x)
        mixed = torch.stack([branch(hidden) for branch in self.branches]).mean(0)
        return x + self.mix(mixed)


class FrameEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            ConvNormAct(in_channels, hidden_channels, 3),
            ConvNormAct(hidden_channels, hidden_channels, 3, stride=2),
            ConvNormAct(hidden_channels, hidden_channels, 3, stride=2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class FrameDecoder(nn.Module):
    def __init__(self, hidden_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_channels, hidden_channels, 4, stride=2, padding=1, bias=False
            ),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(
                hidden_channels, hidden_channels, 4, stride=2, padding=1, bias=False
            ),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, 3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class SimVP(nn.Module):
    """SimVP-style sequence predictor.

    Input shape: ``[batch, input_length, channels, height, width]``.
    Output shape: ``[batch, output_length, channels, height, width]``.
    Height and width must be divisible by four.
    """

    def __init__(
        self,
        *,
        input_length: int = 13,
        output_length: int = 12,
        in_channels: int = 1,
        hidden_spatial: int = 32,
        hidden_temporal: int = 192,
        temporal_depth: int = 4,
        inception_groups: int = 8,
    ) -> None:
        super().__init__()
        if input_length < 1 or output_length < 1:
            raise ValueError("input_length and output_length must be positive")
        if temporal_depth < 1:
            raise ValueError("temporal_depth must be positive")

        self.input_length = input_length
        self.output_length = output_length
        self.in_channels = in_channels
        self.hidden_spatial = hidden_spatial

        self.encoder = FrameEncoder(in_channels, hidden_spatial)
        self.temporal_in = ConvNormAct(
            input_length * hidden_spatial,
            hidden_temporal,
            1,
            padding=0,
        )
        self.temporal_blocks = nn.Sequential(
            *[
                InceptionResidualBlock(
                    hidden_temporal,
                    groups=inception_groups,
                )
                for _ in range(temporal_depth)
            ]
        )
        self.temporal_out = nn.Conv2d(
            hidden_temporal, output_length * hidden_spatial, 1
        )
        self.decoder = FrameDecoder(hidden_spatial, in_channels)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 5:
            raise ValueError(f"expected [B,T,C,H,W], got shape {tuple(x.shape)}")
        batch, steps, channels, height, width = x.shape
        if steps != self.input_length:
            raise ValueError(
                f"expected {self.input_length} input frames, received {steps}"
            )
        if channels != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, received {channels}"
            )
        if height % 4 or width % 4:
            raise ValueError("height and width must be divisible by four")

        encoded = self.encoder(x.reshape(batch * steps, channels, height, width))
        _, hidden_channels, reduced_h, reduced_w = encoded.shape
        encoded = encoded.reshape(
            batch, steps * hidden_channels, reduced_h, reduced_w
        )
        translated = self.temporal_out(
            self.temporal_blocks(self.temporal_in(encoded))
        )
        translated = translated.reshape(
            batch * self.output_length,
            self.hidden_spatial,
            reduced_h,
            reduced_w,
        )
        decoded = self.decoder(translated)
        return decoded.reshape(
            batch,
            self.output_length,
            self.in_channels,
            height,
            width,
        )


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
