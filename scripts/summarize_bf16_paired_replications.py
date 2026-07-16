#!/usr/bin/env python3
"""Summarize three-seed BF16 baseline/proposed metrics and paired changes."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from run_bf16_paired_replications import load_valid_summary


THRESHOLDS = ("16", "74", "133", "160", "181", "219")
LEAD_MINUTES = tuple(range(5, 61, 5))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts/local"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def describe(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def load_metrics(directory: Path) -> dict[str, Any]:
    path = directory / "metrics.json"
    if not path.exists():
        raise RuntimeError(f"missing metrics file: {path}")
    metrics = json.loads(path.read_text(encoding="utf-8"))
    if metrics.get("lead_minutes") != list(LEAD_MINUTES):
        raise RuntimeError(f"unexpected lead minutes in {path}")
    for family in ("csi_by_threshold", "pod_by_threshold", "sucr_by_threshold"):
        if set(metrics.get(family, {})) != set(THRESHOLDS):
            raise RuntimeError(f"unexpected thresholds for {family} in {path}")
    return metrics


def paired_value(baseline: float, proposed: float) -> dict[str, float]:
    return {
        "baseline": baseline,
        "proposed": proposed,
        "absolute_change": proposed - baseline,
        "relative_change": proposed / baseline - 1.0,
    }


def threshold_lead_mean(metrics: dict[str, Any], family: str, threshold: str) -> float:
    return statistics.fmean(float(value) for value in metrics[family][threshold])


def summarize_seed(
    seed: int,
    baseline_dir: Path,
    proposed_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    load_valid_summary(baseline_dir, seed)
    load_valid_summary(proposed_dir, seed)
    baseline = load_metrics(baseline_dir)
    proposed = load_metrics(proposed_dir)
    result: dict[str, Any] = {
        "seed": seed,
        "baseline_dir": str(baseline_dir),
        "proposed_dir": str(proposed_dir),
        "overall": {
            "mse": paired_value(float(baseline["mse"]), float(proposed["mse"])),
            "csi_mean": paired_value(
                float(baseline["csi_mean"]), float(proposed["csi_mean"])
            ),
        },
        "thresholds": {},
        "at_60_minutes": {},
    }
    for threshold in THRESHOLDS:
        baseline_csi = [float(value) for value in baseline["csi_by_threshold"][threshold]]
        proposed_csi = [float(value) for value in proposed["csi_by_threshold"][threshold]]
        result["thresholds"][threshold] = {
            "lead_mean_csi": paired_value(
                statistics.fmean(baseline_csi), statistics.fmean(proposed_csi)
            ),
            "lead_mean_pod": paired_value(
                threshold_lead_mean(baseline, "pod_by_threshold", threshold),
                threshold_lead_mean(proposed, "pod_by_threshold", threshold),
            ),
            "lead_mean_sucr": paired_value(
                threshold_lead_mean(baseline, "sucr_by_threshold", threshold),
                threshold_lead_mean(proposed, "sucr_by_threshold", threshold),
            ),
            "csi_improved_leads": sum(
                proposed_value > baseline_value
                for baseline_value, proposed_value in zip(
                    baseline_csi, proposed_csi, strict=True
                )
            ),
            "csi_change_by_lead": [
                proposed_value - baseline_value
                for baseline_value, proposed_value in zip(
                    baseline_csi, proposed_csi, strict=True
                )
            ],
        }
        observed = float(proposed["observed_pixels_by_threshold"][threshold][-1])
        baseline_forecast = float(
            baseline["forecast_pixels_by_threshold"][threshold][-1]
        )
        proposed_forecast = float(
            proposed["forecast_pixels_by_threshold"][threshold][-1]
        )
        result["at_60_minutes"][threshold] = {
            "observed_pixels": observed,
            "baseline_forecast_pixels": baseline_forecast,
            "proposed_forecast_pixels": proposed_forecast,
            "baseline_forecast_to_observed_ratio": baseline_forecast / observed,
            "proposed_forecast_to_observed_ratio": proposed_forecast / observed,
        }
    baseline_target = float(baseline["mean_target_by_lead"][-1])
    proposed_target = float(proposed["mean_target_by_lead"][-1])
    if abs(baseline_target - proposed_target) > 1e-12:
        raise RuntimeError(f"target mismatch for seed {seed}")
    result["at_60_minutes"]["mean_vil"] = {
        "target": baseline_target,
        "baseline_relative_bias": float(baseline["mean_prediction_by_lead"][-1])
        / baseline_target
        - 1.0,
        "proposed_relative_bias": float(proposed["mean_prediction_by_lead"][-1])
        / baseline_target
        - 1.0,
    }
    return result, baseline, proposed


def main() -> int:
    args = parse_args()
    directories = {
        0: (
            args.artifacts_root / "exp010_bf16_baseline_seed0",
            args.artifacts_root / "exp010_bf16_tail_area_seed0",
        ),
        1: (
            args.artifacts_root / "exp009_bf16_protocol_gate_seed1",
            args.artifacts_root / "exp010_bf16_tail_area_seed1",
        ),
        2: (
            args.artifacts_root / "exp010_bf16_baseline_seed2",
            args.artifacts_root / "exp010_bf16_tail_area_seed2",
        ),
    }
    per_seed: list[dict[str, Any]] = []
    baseline_metrics: list[dict[str, Any]] = []
    proposed_metrics: list[dict[str, Any]] = []
    for seed, (baseline_dir, proposed_dir) in directories.items():
        seed_result, baseline, proposed = summarize_seed(
            seed, baseline_dir, proposed_dir
        )
        per_seed.append(seed_result)
        baseline_metrics.append(baseline)
        proposed_metrics.append(proposed)

    aggregate: dict[str, Any] = {"overall": {}, "thresholds": {}}
    for metric in ("mse", "csi_mean"):
        baseline_values = [float(item[metric]) for item in baseline_metrics]
        proposed_values = [float(item[metric]) for item in proposed_metrics]
        paired_changes = [
            proposed - baseline
            for baseline, proposed in zip(
                baseline_values, proposed_values, strict=True
            )
        ]
        paired_relative_changes = [
            proposed / baseline - 1.0
            for baseline, proposed in zip(
                baseline_values, proposed_values, strict=True
            )
        ]
        aggregate["overall"][metric] = {
            "baseline": describe(baseline_values),
            "proposed": describe(proposed_values),
            "paired_absolute_change": describe(paired_changes),
            "paired_relative_change": describe(paired_relative_changes),
        }

    for threshold in THRESHOLDS:
        baseline_means = [
            threshold_lead_mean(item, "csi_by_threshold", threshold)
            for item in baseline_metrics
        ]
        proposed_means = [
            threshold_lead_mean(item, "csi_by_threshold", threshold)
            for item in proposed_metrics
        ]
        relative_changes = [
            proposed / baseline - 1.0
            for baseline, proposed in zip(
                baseline_means, proposed_means, strict=True
            )
        ]
        changes_by_lead = []
        wins_by_lead = []
        for lead_index in range(len(LEAD_MINUTES)):
            changes = [
                float(proposed["csi_by_threshold"][threshold][lead_index])
                - float(baseline["csi_by_threshold"][threshold][lead_index])
                for baseline, proposed in zip(
                    baseline_metrics, proposed_metrics, strict=True
                )
            ]
            changes_by_lead.append(describe(changes))
            wins_by_lead.append(sum(change > 0 for change in changes))
        aggregate["thresholds"][threshold] = {
            "baseline_lead_mean_csi": describe(baseline_means),
            "proposed_lead_mean_csi": describe(proposed_means),
            "paired_relative_change": describe(relative_changes),
            "paired_csi_change_by_lead": changes_by_lead,
            "proposed_wins_out_of_3_by_lead": wins_by_lead,
        }

    output = {
        "ok": True,
        "protocol": "frozen_bf16_128_seed_0_1_2",
        "lead_minutes": list(LEAD_MINUTES),
        "per_seed": per_seed,
        "aggregate": aggregate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
