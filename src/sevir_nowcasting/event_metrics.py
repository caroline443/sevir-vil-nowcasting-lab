"""Event-grouped sufficient statistics and paired bootstrap utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from sevir_nowcasting.metrics import SEVIR_THRESHOLDS


@dataclass
class EventVILStats:
    """Accumulate forecast sufficient statistics by SEVIR storm event."""

    output_length: int = 12
    thresholds: tuple[int, ...] = SEVIR_THRESHOLDS
    _events: dict[str, dict[str, Tensor]] = field(default_factory=dict)

    @torch.no_grad()
    def update(
        self,
        event_ids: Sequence[str],
        prediction: Tensor,
        target: Tensor,
    ) -> None:
        if prediction.shape != target.shape or prediction.ndim != 5:
            raise ValueError(
                "prediction and target must share shape [B,T,C,H,W], got "
                f"{tuple(prediction.shape)} and {tuple(target.shape)}"
            )
        if prediction.shape[0] != len(event_ids):
            raise ValueError("event ID count must equal batch size")
        if prediction.shape[1] != self.output_length:
            raise ValueError(
                f"expected {self.output_length} lead times, got {prediction.shape[1]}"
            )

        prediction = prediction.detach().float()
        target = target.detach().float()
        reduce_dims = (2, 3, 4)
        difference = prediction - target
        squared_error = torch.sum(
            difference * difference, dim=reduce_dims
        ).double().cpu()
        absolute_error = torch.sum(
            difference.abs(), dim=reduce_dims
        ).double().cpu()
        values_per_lead = prediction[0, 0].numel()
        element_count = torch.full(
            (prediction.shape[0], self.output_length),
            values_per_lead,
            dtype=torch.float64,
        )

        shape = (
            prediction.shape[0],
            len(self.thresholds),
            self.output_length,
        )
        hits = torch.zeros(shape, dtype=torch.float64)
        misses = torch.zeros_like(hits)
        false_alarms = torch.zeros_like(hits)
        for threshold_index, raw_threshold in enumerate(self.thresholds):
            threshold = raw_threshold / 255.0
            forecast = prediction >= threshold
            observed = target >= threshold
            hits[:, threshold_index] = torch.sum(
                forecast & observed, dim=reduce_dims
            ).double().cpu()
            misses[:, threshold_index] = torch.sum(
                ~forecast & observed, dim=reduce_dims
            ).double().cpu()
            false_alarms[:, threshold_index] = torch.sum(
                forecast & ~observed, dim=reduce_dims
            ).double().cpu()

        for batch_index, event_id in enumerate(event_ids):
            key = str(event_id)
            if key not in self._events:
                self._events[key] = {
                    "squared_error": torch.zeros(
                        self.output_length, dtype=torch.float64
                    ),
                    "absolute_error": torch.zeros(
                        self.output_length, dtype=torch.float64
                    ),
                    "element_count": torch.zeros(
                        self.output_length, dtype=torch.float64
                    ),
                    "hits": torch.zeros(
                        len(self.thresholds),
                        self.output_length,
                        dtype=torch.float64,
                    ),
                    "misses": torch.zeros(
                        len(self.thresholds),
                        self.output_length,
                        dtype=torch.float64,
                    ),
                    "false_alarms": torch.zeros(
                        len(self.thresholds),
                        self.output_length,
                        dtype=torch.float64,
                    ),
                }
            event = self._events[key]
            event["squared_error"] += squared_error[batch_index]
            event["absolute_error"] += absolute_error[batch_index]
            event["element_count"] += element_count[batch_index]
            event["hits"] += hits[batch_index]
            event["misses"] += misses[batch_index]
            event["false_alarms"] += false_alarms[batch_index]

    def save(self, path: str | Path) -> None:
        output = Path(path)
        if output.suffix != ".npz":
            raise ValueError("event statistics output must use the .npz suffix")
        if not self._events:
            raise ValueError("cannot save empty event statistics")
        output.parent.mkdir(parents=True, exist_ok=True)
        event_ids = sorted(self._events)

        def stacked(name: str) -> np.ndarray:
            return np.stack(
                [self._events[event_id][name].numpy() for event_id in event_ids]
            )

        np.savez_compressed(
            output,
            format_version=np.asarray(1, dtype=np.int64),
            event_ids=np.asarray(event_ids),
            thresholds=np.asarray(self.thresholds, dtype=np.int64),
            lead_minutes=np.arange(
                5, 5 * (self.output_length + 1), 5, dtype=np.int64
            ),
            squared_error=stacked("squared_error"),
            absolute_error=stacked("absolute_error"),
            element_count=stacked("element_count"),
            hits=stacked("hits"),
            misses=stacked("misses"),
            false_alarms=stacked("false_alarms"),
        )


def load_event_stats(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as archive:
        required = {
            "format_version",
            "event_ids",
            "thresholds",
            "lead_minutes",
            "squared_error",
            "absolute_error",
            "element_count",
            "hits",
            "misses",
            "false_alarms",
        }
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"event statistics missing arrays: {sorted(missing)}")
        result = {name: archive[name].copy() for name in required}
    if int(result["format_version"]) != 1:
        raise ValueError("unsupported event statistics format")
    event_count = len(result["event_ids"])
    if result["hits"].shape[0] != event_count:
        raise ValueError("event statistics arrays have inconsistent event counts")
    return result


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > 0,
    )


def _weighted_metrics(
    stats: dict[str, np.ndarray],
    weights: np.ndarray,
) -> dict[str, np.ndarray]:
    squared_error = weights @ stats["squared_error"].sum(axis=1)
    absolute_error = weights @ stats["absolute_error"].sum(axis=1)
    element_count = weights @ stats["element_count"].sum(axis=1)
    hits = weights @ stats["hits"].sum(axis=2)
    misses = weights @ stats["misses"].sum(axis=2)
    false_alarms = weights @ stats["false_alarms"].sum(axis=2)
    csi = _safe_ratio(hits, hits + misses + false_alarms)
    pod = _safe_ratio(hits, hits + misses)
    sucr = _safe_ratio(hits, hits + false_alarms)
    return {
        "mse": _safe_ratio(squared_error, element_count),
        "mae": _safe_ratio(absolute_error, element_count),
        "csi": csi,
        "pod": pod,
        "sucr": sucr,
        "mcsi_global": csi.mean(axis=1),
    }


def _validate_paired_stats(
    baseline: Sequence[dict[str, np.ndarray]],
    proposed: Sequence[dict[str, np.ndarray]],
) -> None:
    if not baseline or len(baseline) != len(proposed):
        raise ValueError("baseline and proposed statistics require paired seeds")
    reference = baseline[0]
    for stats in [*baseline, *proposed]:
        if not np.array_equal(stats["event_ids"], reference["event_ids"]):
            raise ValueError("paired statistics do not contain identical event IDs")
        if not np.array_equal(stats["thresholds"], reference["thresholds"]):
            raise ValueError("paired statistics use different thresholds")
        if not np.array_equal(stats["lead_minutes"], reference["lead_minutes"]):
            raise ValueError("paired statistics use different lead times")
        if not np.array_equal(
            stats["element_count"], reference["element_count"]
        ):
            raise ValueError("paired statistics use different element counts")
        if not np.array_equal(
            stats["hits"] + stats["misses"],
            reference["hits"] + reference["misses"],
        ):
            raise ValueError(
                "paired statistics do not contain identical observations"
            )


def paired_event_bootstrap(
    baseline_paths: Sequence[str | Path],
    proposed_paths: Sequence[str | Path],
    *,
    repetitions: int = 10000,
    seed: int = 2027,
    chunk_size: int = 128,
) -> dict[str, object]:
    """Bootstrap paired metric differences by resampling complete events."""

    if repetitions < 1 or chunk_size < 1:
        raise ValueError("repetitions and chunk size must be positive")
    baseline = [load_event_stats(path) for path in baseline_paths]
    proposed = [load_event_stats(path) for path in proposed_paths]
    _validate_paired_stats(baseline, proposed)
    event_count = len(baseline[0]["event_ids"])
    thresholds = baseline[0]["thresholds"].tolist()
    point_weights = np.ones((1, event_count), dtype=np.float64)

    point_baseline = [_weighted_metrics(stats, point_weights) for stats in baseline]
    point_proposed = [_weighted_metrics(stats, point_weights) for stats in proposed]
    metric_names = ("mse", "mae", "mcsi_global")
    point: dict[str, object] = {}
    for name in metric_names:
        baseline_value = float(
            np.mean([metrics[name][0] for metrics in point_baseline])
        )
        proposed_value = float(
            np.mean([metrics[name][0] for metrics in point_proposed])
        )
        point[name] = {
            "baseline": baseline_value,
            "proposed": proposed_value,
            "difference": proposed_value - baseline_value,
        }
    for metric_name in ("csi", "pod", "sucr"):
        metric_result: dict[str, object] = {}
        for threshold_index, threshold in enumerate(thresholds):
            baseline_value = float(
                np.mean(
                    [
                        metrics[metric_name][0, threshold_index]
                        for metrics in point_baseline
                    ]
                )
            )
            proposed_value = float(
                np.mean(
                    [
                        metrics[metric_name][0, threshold_index]
                        for metrics in point_proposed
                    ]
                )
            )
            metric_result[str(threshold)] = {
                "baseline": baseline_value,
                "proposed": proposed_value,
                "difference": proposed_value - baseline_value,
            }
        point[f"global_{metric_name}_by_threshold"] = metric_result

    rng = np.random.default_rng(seed)
    bootstrap_values: dict[str, list[np.ndarray]] = {
        name: [] for name in metric_names
    }
    for metric_name in ("csi", "pod", "sucr"):
        for threshold in thresholds:
            bootstrap_values[f"{metric_name}_{threshold}"] = []

    remaining = repetitions
    uniform_probability = np.full(event_count, 1.0 / event_count)
    while remaining:
        current = min(chunk_size, remaining)
        weights = rng.multinomial(
            event_count, uniform_probability, size=current
        ).astype(np.float64)
        baseline_metrics = [
            _weighted_metrics(stats, weights) for stats in baseline
        ]
        proposed_metrics = [
            _weighted_metrics(stats, weights) for stats in proposed
        ]
        for name in metric_names:
            baseline_mean = np.mean(
                [metrics[name] for metrics in baseline_metrics], axis=0
            )
            proposed_mean = np.mean(
                [metrics[name] for metrics in proposed_metrics], axis=0
            )
            bootstrap_values[name].append(proposed_mean - baseline_mean)
        for metric_name in ("csi", "pod", "sucr"):
            baseline_mean = np.mean(
                [metrics[metric_name] for metrics in baseline_metrics], axis=0
            )
            proposed_mean = np.mean(
                [metrics[metric_name] for metrics in proposed_metrics], axis=0
            )
            difference = proposed_mean - baseline_mean
            for threshold_index, threshold in enumerate(thresholds):
                bootstrap_values[f"{metric_name}_{threshold}"].append(
                    difference[:, threshold_index]
                )
        remaining -= current

    intervals: dict[str, object] = {}
    for name, chunks in bootstrap_values.items():
        values = np.concatenate(chunks)
        intervals[name] = {
            "lower_95": float(np.quantile(values, 0.025)),
            "median": float(np.quantile(values, 0.5)),
            "upper_95": float(np.quantile(values, 0.975)),
            "probability_difference_gt_zero": float(np.mean(values > 0)),
        }
    return {
        "ok": True,
        "purpose": "paired_event_bootstrap",
        "paired_seed_count": len(baseline),
        "event_count": event_count,
        "repetitions": repetitions,
        "random_seed": seed,
        "thresholds_raw": thresholds,
        "point_estimates": point,
        "difference_intervals": intervals,
        "interpretation": "Differences are proposed minus baseline.",
    }
