#!/usr/bin/env python3
"""Create the compact quantitative figure for the GRSL manuscript."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, nargs=2, required=True)
    parser.add_argument("--proposed", type=Path, nargs=2, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_metrics(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    metrics = payload.get("metrics", payload)
    if not isinstance(metrics, dict):
        raise ValueError(f"{path} does not contain a metrics object")
    return metrics


def main() -> int:
    args = parse_args()
    import matplotlib.pyplot as plt

    baseline = [load_metrics(path) for path in args.baseline]
    proposed = [load_metrics(path) for path in args.proposed]
    thresholds = np.asarray([16, 74, 133, 160, 181, 219])
    lead_minutes = np.asarray(baseline[0]["lead_minutes"])

    def lead_array(group: list[dict[str, object]]) -> np.ndarray:
        return np.asarray([metrics["csi_mean_by_lead"] for metrics in group])

    def threshold_array(group: list[dict[str, object]]) -> np.ndarray:
        return np.asarray(
            [
                [
                    metrics["global_csi_by_threshold"][str(threshold)]
                    for threshold in thresholds
                ]
                for metrics in group
            ]
        )

    baseline_lead = lead_array(baseline)
    proposed_lead = lead_array(proposed)
    baseline_threshold = threshold_array(baseline)
    proposed_threshold = threshold_array(proposed)

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.1, 2.55))

    for values, label, color, marker in (
        (baseline_lead, "SimVP (MSE)", "#4c78a8", "o"),
        (proposed_lead, "SimVP + SEA", "#e45756", "s"),
    ):
        mean = values.mean(axis=0)
        std = values.std(axis=0, ddof=1)
        axes[0].plot(
            lead_minutes,
            mean,
            label=label,
            color=color,
            marker=marker,
            markersize=3,
            linewidth=1.3,
        )
        axes[0].fill_between(
            lead_minutes, mean - std, mean + std, color=color, alpha=0.14
        )
    axes[0].set_xlabel("Forecast lead (min)")
    axes[0].set_ylabel("Mean CSI")
    axes[0].set_xticks(lead_minutes[::2])
    axes[0].grid(alpha=0.2, linewidth=0.5)
    axes[0].legend(frameon=False)
    axes[0].text(
        0.01, 0.98, "(a)", transform=axes[0].transAxes, va="top", fontweight="bold"
    )

    for values, label, color, marker in (
        (baseline_threshold, "SimVP (MSE)", "#4c78a8", "o"),
        (proposed_threshold, "SimVP + SEA", "#e45756", "s"),
    ):
        mean = values.mean(axis=0)
        std = values.std(axis=0, ddof=1)
        axes[1].errorbar(
            thresholds,
            mean,
            yerr=std,
            label=label,
            color=color,
            marker=marker,
            markersize=3.5,
            linewidth=1.3,
            capsize=2,
        )
    relative_gain = (
        proposed_threshold.mean(axis=0) / baseline_threshold.mean(axis=0) - 1
    ) * 100
    for threshold_index in (3, 4, 5):
        axes[1].annotate(
            f"+{relative_gain[threshold_index]:.1f}%",
            (
                thresholds[threshold_index],
                proposed_threshold.mean(axis=0)[threshold_index],
            ),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=6.5,
        )
    axes[1].set_xlabel("Raw VIL threshold")
    axes[1].set_ylabel("Global CSI")
    axes[1].set_xticks(thresholds)
    axes[1].grid(alpha=0.2, linewidth=0.5)
    axes[1].text(
        0.01, 0.98, "(b)", transform=axes[1].transAxes, va="top", fontweight="bold"
    )

    figure.tight_layout(pad=0.6, w_pad=1.4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        args.output_dir / "grsl-quantitative-results.pdf",
        bbox_inches="tight",
    )
    figure.savefig(
        args.output_dir / "grsl-quantitative-results.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
