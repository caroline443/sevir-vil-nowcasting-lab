"""Streaming deterministic metrics for normalized SEVIR VIL forecasts."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


SEVIR_THRESHOLDS = (16, 74, 133, 160, 181, 219)


@dataclass
class VILMetrics:
    thresholds: tuple[int, ...] = SEVIR_THRESHOLDS
    squared_error: float = 0.0
    element_count: int = 0
    hits: Tensor = field(init=False)
    misses: Tensor = field(init=False)
    false_alarms: Tensor = field(init=False)

    def __post_init__(self) -> None:
        size = len(self.thresholds)
        self.hits = torch.zeros(size, dtype=torch.float64)
        self.misses = torch.zeros(size, dtype=torch.float64)
        self.false_alarms = torch.zeros(size, dtype=torch.float64)

    @torch.no_grad()
    def update(self, prediction: Tensor, target: Tensor) -> None:
        prediction = prediction.detach().clamp(0.0, 1.0).cpu()
        target = target.detach().clamp(0.0, 1.0).cpu()
        difference = prediction - target
        self.squared_error += float(torch.sum(difference * difference))
        self.element_count += difference.numel()

        for index, raw_threshold in enumerate(self.thresholds):
            threshold = raw_threshold / 255.0
            forecast = prediction >= threshold
            observed = target >= threshold
            self.hits[index] += torch.sum(forecast & observed)
            self.misses[index] += torch.sum(~forecast & observed)
            self.false_alarms[index] += torch.sum(forecast & ~observed)

    def compute(self) -> dict[str, float]:
        result = {
            "mse": self.squared_error / max(1, self.element_count),
        }
        csi_values: list[float] = []
        for index, threshold in enumerate(self.thresholds):
            denominator = (
                self.hits[index] + self.misses[index] + self.false_alarms[index]
            )
            csi = float(self.hits[index] / denominator) if denominator > 0 else 0.0
            csi_values.append(csi)
            result[f"csi_{threshold}"] = csi
        result["csi_mean"] = sum(csi_values) / len(csi_values)
        return result
