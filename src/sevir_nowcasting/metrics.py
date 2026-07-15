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


@dataclass
class LeadTimeVILMetrics:
    """Streaming SEVIR metrics that preserve the forecast lead-time axis."""

    output_length: int = 12
    thresholds: tuple[int, ...] = SEVIR_THRESHOLDS
    squared_error: Tensor = field(init=False)
    value_sum_prediction: Tensor = field(init=False)
    value_sum_target: Tensor = field(init=False)
    element_count: Tensor = field(init=False)
    hits: Tensor = field(init=False)
    misses: Tensor = field(init=False)
    false_alarms: Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.squared_error = torch.zeros(self.output_length, dtype=torch.float64)
        self.value_sum_prediction = torch.zeros(
            self.output_length, dtype=torch.float64
        )
        self.value_sum_target = torch.zeros(self.output_length, dtype=torch.float64)
        self.element_count = torch.zeros(self.output_length, dtype=torch.float64)
        shape = (len(self.thresholds), self.output_length)
        self.hits = torch.zeros(shape, dtype=torch.float64)
        self.misses = torch.zeros(shape, dtype=torch.float64)
        self.false_alarms = torch.zeros(shape, dtype=torch.float64)

    @torch.no_grad()
    def update(self, prediction: Tensor, target: Tensor) -> None:
        if prediction.shape != target.shape or prediction.ndim != 5:
            raise ValueError(
                "prediction and target must share shape [B,T,C,H,W], got "
                f"{tuple(prediction.shape)} and {tuple(target.shape)}"
            )
        if prediction.shape[1] != self.output_length:
            raise ValueError(
                f"expected {self.output_length} lead times, got {prediction.shape[1]}"
            )

        prediction = prediction.detach().float()
        target = target.detach().float()
        reduce_dims = (0, 2, 3, 4)
        difference = prediction - target
        self.squared_error += torch.sum(
            difference * difference, dim=reduce_dims
        ).double().cpu()
        self.value_sum_prediction += torch.sum(
            prediction, dim=reduce_dims
        ).double().cpu()
        self.value_sum_target += torch.sum(target, dim=reduce_dims).double().cpu()
        values_per_lead = prediction[:, 0].numel()
        self.element_count += values_per_lead

        for index, raw_threshold in enumerate(self.thresholds):
            threshold = raw_threshold / 255.0
            forecast = prediction >= threshold
            observed = target >= threshold
            self.hits[index] += torch.sum(
                forecast & observed, dim=reduce_dims
            ).double().cpu()
            self.misses[index] += torch.sum(
                ~forecast & observed, dim=reduce_dims
            ).double().cpu()
            self.false_alarms[index] += torch.sum(
                forecast & ~observed, dim=reduce_dims
            ).double().cpu()

    def compute(self) -> dict[str, object]:
        mse = self.squared_error / self.element_count.clamp_min(1.0)
        mean_prediction = self.value_sum_prediction / self.element_count.clamp_min(1.0)
        mean_target = self.value_sum_target / self.element_count.clamp_min(1.0)
        result: dict[str, object] = {
            "lead_minutes": [5 * (index + 1) for index in range(self.output_length)],
            "mse_by_lead": mse.tolist(),
            "mean_prediction_by_lead": mean_prediction.tolist(),
            "mean_target_by_lead": mean_target.tolist(),
            "csi_by_threshold": {},
            "pod_by_threshold": {},
            "sucr_by_threshold": {},
            "observed_pixels_by_threshold": {},
            "forecast_pixels_by_threshold": {},
        }
        csi_by_threshold = result["csi_by_threshold"]
        pod_by_threshold = result["pod_by_threshold"]
        sucr_by_threshold = result["sucr_by_threshold"]
        observed_by_threshold = result["observed_pixels_by_threshold"]
        forecast_by_threshold = result["forecast_pixels_by_threshold"]
        assert isinstance(csi_by_threshold, dict)
        assert isinstance(pod_by_threshold, dict)
        assert isinstance(sucr_by_threshold, dict)
        assert isinstance(observed_by_threshold, dict)
        assert isinstance(forecast_by_threshold, dict)

        csi_stack: list[Tensor] = []
        for index, threshold in enumerate(self.thresholds):
            hits = self.hits[index]
            misses = self.misses[index]
            false_alarms = self.false_alarms[index]
            csi_denominator = hits + misses + false_alarms
            pod_denominator = hits + misses
            sucr_denominator = hits + false_alarms
            csi = torch.where(
                csi_denominator > 0, hits / csi_denominator, 0.0
            )
            pod = torch.where(pod_denominator > 0, hits / pod_denominator, 0.0)
            sucr = torch.where(
                sucr_denominator > 0, hits / sucr_denominator, 0.0
            )
            csi_by_threshold[str(threshold)] = csi.tolist()
            pod_by_threshold[str(threshold)] = pod.tolist()
            sucr_by_threshold[str(threshold)] = sucr.tolist()
            observed_by_threshold[str(threshold)] = (hits + misses).tolist()
            forecast_by_threshold[str(threshold)] = (hits + false_alarms).tolist()
            csi_stack.append(csi)

        result["csi_mean_by_lead"] = torch.stack(csi_stack).mean(0).tolist()
        result["mse"] = float(self.squared_error.sum() / self.element_count.sum())
        result["csi_mean"] = float(torch.stack(csi_stack).mean())
        return result
