#!/usr/bin/env python3
"""Measure whether severe false alarms are near-miss displacements."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from sevir_nowcasting.data import SevirVILWindowDataset
from train_openstl_simvp import build_model, predict_12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=200)
    parser.add_argument(
        "--thresholds-raw", type=float, nargs="+", default=[160, 181, 219]
    )
    parser.add_argument(
        "--radii", type=int, nargs="+", default=[0, 1, 2, 4, 8]
    )
    parser.add_argument(
        "--amp-dtype", choices=("float16", "bfloat16"), default="bfloat16"
    )
    args = parser.parse_args()
    if args.max_batches < 1 or args.batch_size < 1:
        parser.error("max-batches and batch-size must be positive")
    if any(radius < 0 for radius in args.radii):
        parser.error("radii must be non-negative")
    if len(set(args.radii)) != len(args.radii):
        parser.error("radii must be unique")
    return args


def dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Dilate a ``[B,T,C,H,W]`` Boolean mask by a square radius."""
    if radius == 0:
        return mask
    batch, time, channels, height, width = mask.shape
    flat = mask.float().reshape(batch * time, channels, height, width)
    expanded = F.max_pool2d(
        flat,
        kernel_size=2 * radius + 1,
        stride=1,
        padding=radius,
    )
    return expanded.reshape_as(mask).bool()


def safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> list[float]:
    result = torch.where(
        denominator > 0,
        numerator / denominator,
        torch.zeros_like(numerator),
    )
    return [float(value) for value in result]


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this diagnostic expects a CUDA-capable machine")
    device = torch.device("cuda:0")
    amp_dtype = (
        torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    )
    dataset = SevirVILWindowDataset(
        args.manifest,
        args.data_root,
        split="val",
        input_length=13,
        output_length=12,
        resolution=args.resolution,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    model = build_model(args.resolution).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    training_loss = checkpoint.get("summary", {}).get("args", {}).get(
        "training_loss", "mse"
    )
    output_sigmoid = training_loss == "facl"

    shape = (len(args.thresholds_raw), len(args.radii), 12)
    matched_forecast = torch.zeros(shape, dtype=torch.float64, device=device)
    matched_observed = torch.zeros(shape, dtype=torch.float64, device=device)
    forecast_count = torch.zeros(
        (len(args.thresholds_raw), 12), dtype=torch.float64, device=device
    )
    observed_count = torch.zeros_like(forecast_count)
    completed = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= args.max_batches:
                break
            inputs = batch["inputs"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=amp_dtype):
                predictions = predict_12(model, inputs)
                if output_sigmoid:
                    predictions = torch.sigmoid(predictions)
            for threshold_index, threshold_raw in enumerate(args.thresholds_raw):
                threshold = threshold_raw / 255.0
                predicted_mask = predictions >= threshold
                observed_mask = targets >= threshold
                forecast_count[threshold_index] += predicted_mask.sum(
                    dim=(0, 2, 3, 4)
                )
                observed_count[threshold_index] += observed_mask.sum(
                    dim=(0, 2, 3, 4)
                )
                for radius_index, radius in enumerate(args.radii):
                    observed_envelope = dilate(observed_mask, radius)
                    predicted_envelope = dilate(predicted_mask, radius)
                    matched_forecast[threshold_index, radius_index] += (
                        predicted_mask & observed_envelope
                    ).sum(dim=(0, 2, 3, 4))
                    matched_observed[threshold_index, radius_index] += (
                        observed_mask & predicted_envelope
                    ).sum(dim=(0, 2, 3, 4))
            completed += 1

    result_by_threshold: dict[str, object] = {}
    for threshold_index, threshold_raw in enumerate(args.thresholds_raw):
        threshold_result: dict[str, object] = {
            "forecast_pixels_by_lead": [
                float(value) for value in forecast_count[threshold_index]
            ],
            "observed_pixels_by_lead": [
                float(value) for value in observed_count[threshold_index]
            ],
            "by_radius": {},
        }
        for radius_index, radius in enumerate(args.radii):
            precision = safe_ratio(
                matched_forecast[threshold_index, radius_index],
                forecast_count[threshold_index],
            )
            recall = safe_ratio(
                matched_observed[threshold_index, radius_index],
                observed_count[threshold_index],
            )
            threshold_result["by_radius"][str(radius)] = {
                "tolerant_precision_by_lead": precision,
                "tolerant_recall_by_lead": recall,
                "tolerant_precision_lead_mean": sum(precision) / len(precision),
                "tolerant_recall_lead_mean": sum(recall) / len(recall),
            }
        result_by_threshold[str(int(threshold_raw))] = threshold_result

    result = {
        "ok": True,
        "purpose": "separate_near_miss_displacement_from_remote_false_alarms",
        "checkpoint": str(args.checkpoint),
        "resolution": args.resolution,
        "validation_batches": completed,
        "lead_minutes": [5 * (index + 1) for index in range(12)],
        "thresholds_raw": args.thresholds_raw,
        "radii_pixels_at_working_resolution": args.radii,
        "checkpoint_training_loss": training_loss,
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

