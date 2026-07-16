#!/usr/bin/env python3
"""Measure how often severe VIL persists, initiates, grows, or extinguishes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
from torch.utils.data import DataLoader

from sevir_nowcasting.data import SevirVILWindowDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Zero evaluates the complete split.",
    )
    parser.add_argument(
        "--thresholds-raw", type=float, nargs="+", default=[160, 181, 219]
    )
    parser.add_argument("--growth-ratio", type=float, default=1.25)
    parser.add_argument("--decay-ratio", type=float, default=0.8)
    args = parser.parse_args()
    if args.batch_size < 1 or args.resolution < 1 or args.max_batches < 0:
        parser.error("batch-size and resolution must be positive; max-batches >= 0")
    if not 0 < args.decay_ratio < 1 < args.growth_ratio:
        parser.error("require 0 < decay-ratio < 1 < growth-ratio")
    return args


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def quantiles(values: list[torch.Tensor]) -> dict[str, float]:
    if not values:
        return {key: 0.0 for key in ("q50", "q75", "q90", "q95", "q99")}
    tensor = torch.cat(values).float()
    levels = torch.tensor([0.50, 0.75, 0.90, 0.95, 0.99])
    result = torch.quantile(tensor, levels)
    return {
        key: float(value)
        for key, value in zip(("q50", "q75", "q90", "q95", "q99"), result)
    }


def main() -> int:
    args = parse_args()
    dataset = SevirVILWindowDataset(
        args.manifest,
        args.data_root,
        split=args.split,
        input_length=13,
        output_length=12,
        resolution=args.resolution,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )

    accumulators: dict[str, dict[str, object]] = {}
    for threshold_raw in args.thresholds_raw:
        accumulators[str(int(threshold_raw))] = {
            "samples": 0,
            "last_input_active": 0,
            "any_future_active": 0,
            "lead_active": torch.zeros(12, dtype=torch.int64),
            "no_input_no_future": 0,
            "initiation_any": 0,
            "late_initiation_60": 0,
            "extinction_by_60": 0,
            "persistent_to_60": 0,
            "growth_to_60": 0,
            "stable_to_60": 0,
            "decay_to_60": 0,
            "last_area_active": [],
            "max_future_area_active": [],
            "minute_60_area_active": [],
        }

    completed_batches = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if args.max_batches and batch_index >= args.max_batches:
                break
            inputs = batch["inputs"]
            targets = batch["targets"]
            batch_size = inputs.shape[0]
            for threshold_raw in args.thresholds_raw:
                key = str(int(threshold_raw))
                acc = accumulators[key]
                threshold = threshold_raw / 255.0
                last_area = (inputs[:, -1] >= threshold).sum(dim=(1, 2, 3))
                future_area = (targets >= threshold).sum(dim=(2, 3, 4))
                last_active = last_area > 0
                future_active = future_area > 0
                any_future = future_active.any(dim=1)
                active_60 = future_active[:, -1]

                acc["samples"] += batch_size
                acc["last_input_active"] += int(last_active.sum())
                acc["any_future_active"] += int(any_future.sum())
                acc["lead_active"] += future_active.sum(dim=0).cpu()
                acc["no_input_no_future"] += int((~last_active & ~any_future).sum())
                acc["initiation_any"] += int((~last_active & any_future).sum())
                acc["late_initiation_60"] += int((~last_active & active_60).sum())
                acc["extinction_by_60"] += int((last_active & ~active_60).sum())
                persistent = last_active & active_60
                acc["persistent_to_60"] += int(persistent.sum())

                if last_active.any():
                    acc["last_area_active"].append(last_area[last_active].cpu())
                if any_future.any():
                    acc["max_future_area_active"].append(
                        future_area.max(dim=1).values[any_future].cpu()
                    )
                if active_60.any():
                    acc["minute_60_area_active"].append(
                        future_area[active_60, -1].cpu()
                    )
                if persistent.any():
                    ratio = (
                        future_area[persistent, -1].float()
                        / last_area[persistent].float()
                    )
                    acc["growth_to_60"] += int((ratio >= args.growth_ratio).sum())
                    acc["decay_to_60"] += int((ratio <= args.decay_ratio).sum())
                    acc["stable_to_60"] += int(
                        ((ratio > args.decay_ratio) & (ratio < args.growth_ratio)).sum()
                    )
            completed_batches += 1

    result_by_threshold: dict[str, object] = {}
    for key, acc in accumulators.items():
        samples = int(acc["samples"])
        last_active = int(acc["last_input_active"])
        persistent = int(acc["persistent_to_60"])
        lead_active = [int(value) for value in acc["lead_active"]]
        result_by_threshold[key] = {
            "sample_count": samples,
            "last_input_active_count": last_active,
            "last_input_active_fraction": safe_ratio(last_active, samples),
            "any_future_active_count": int(acc["any_future_active"]),
            "any_future_active_fraction": safe_ratio(acc["any_future_active"], samples),
            "active_fraction_by_lead": [safe_ratio(value, samples) for value in lead_active],
            "no_input_no_future_count": int(acc["no_input_no_future"]),
            "initiation_any_count": int(acc["initiation_any"]),
            "initiation_any_fraction": safe_ratio(acc["initiation_any"], samples),
            "late_initiation_60_count": int(acc["late_initiation_60"]),
            "late_initiation_60_fraction": safe_ratio(acc["late_initiation_60"], samples),
            "extinction_by_60_count": int(acc["extinction_by_60"]),
            "extinction_by_60_given_input_active": safe_ratio(
                acc["extinction_by_60"], last_active
            ),
            "persistent_to_60_count": persistent,
            "persistent_to_60_fraction": safe_ratio(persistent, samples),
            "persistent_to_60_given_input_active": safe_ratio(persistent, last_active),
            "growth_to_60_given_persistent": safe_ratio(acc["growth_to_60"], persistent),
            "stable_to_60_given_persistent": safe_ratio(acc["stable_to_60"], persistent),
            "decay_to_60_given_persistent": safe_ratio(acc["decay_to_60"], persistent),
            "last_input_area_pixels_quantiles_given_active": quantiles(
                acc["last_area_active"]
            ),
            "max_future_area_pixels_quantiles_given_active": quantiles(
                acc["max_future_area_active"]
            ),
            "minute_60_area_pixels_quantiles_given_active": quantiles(
                acc["minute_60_area_active"]
            ),
        }

    result = {
        "ok": True,
        "purpose": "gate_future_tail_trajectory_balancing",
        "manifest": str(args.manifest),
        "data_root": str(args.data_root),
        "split": args.split,
        "resolution": args.resolution,
        "batch_size": args.batch_size,
        "completed_batches": completed_batches,
        "max_batches": args.max_batches,
        "lead_minutes": [5 * (index + 1) for index in range(12)],
        "thresholds_raw": args.thresholds_raw,
        "growth_ratio": args.growth_ratio,
        "decay_ratio": args.decay_ratio,
        "caveat": (
            "Endpoint categories use threshold presence and area, not storm-object "
            "identity tracking; initiation and persistence are sequence-level labels."
        ),
        "result_by_threshold": result_by_threshold,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
