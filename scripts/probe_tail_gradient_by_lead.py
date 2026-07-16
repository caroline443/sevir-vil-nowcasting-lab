#!/usr/bin/env python3
"""Probe whether soft tail gradients vanish or delocalize at long leads."""

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
from sevir_nowcasting.losses import SoftExceedanceAreaLoss
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
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument(
        "--thresholds-raw", type=float, nargs="+", default=[160, 181, 219]
    )
    parser.add_argument(
        "--temperatures-raw", type=float, nargs="+", default=[5, 10, 20, 30]
    )
    parser.add_argument("--radii", type=int, nargs="+", default=[0, 2, 4])
    parser.add_argument(
        "--amp-dtype", choices=("float16", "bfloat16"), default="bfloat16"
    )
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_batches < 1 or args.resolution < 1:
        parser.error("batch-size, max-batches and resolution must be positive")
    if any(value <= 0 for value in args.temperatures_raw):
        parser.error("temperatures must be positive")
    if any(radius < 0 for radius in args.radii):
        parser.error("radii must be non-negative")
    return args


def dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius == 0:
        return mask
    batch, time, channels, height, width = mask.shape
    flat = mask.float().reshape(batch * time, channels, height, width)
    result = F.max_pool2d(
        flat, kernel_size=2 * radius + 1, stride=1, padding=radius
    )
    return result.reshape_as(mask).bool()


def zeros(device: torch.device) -> torch.Tensor:
    return torch.zeros(12, dtype=torch.float64, device=device)


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this probe expects a CUDA-capable machine")
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

    accumulators: dict[str, dict[str, dict[str, object]]] = {}
    criteria: dict[tuple[str, str], SoftExceedanceAreaLoss] = {}
    for temperature_raw in args.temperatures_raw:
        temperature_key = f"{temperature_raw:g}"
        accumulators[temperature_key] = {}
        for threshold_raw in args.thresholds_raw:
            threshold_key = f"{threshold_raw:g}"
            criteria[(temperature_key, threshold_key)] = SoftExceedanceAreaLoss(
                thresholds_raw=[threshold_raw], temperature_raw=temperature_raw
            ).to(device)
            accumulators[temperature_key][threshold_key] = {
                "absolute_gradient": zeros(device),
                "squared_gradient": zeros(device),
                "upward_gradient": zeros(device),
                "downward_gradient": zeros(device),
                "predicted_soft_count": zeros(device),
                "target_soft_count": zeros(device),
                "upward_inside_radius": {
                    str(radius): zeros(device) for radius in args.radii
                },
            }

    completed_batches = 0
    completed_samples = 0
    for batch_index, batch in enumerate(loader):
        if batch_index >= args.max_batches:
            break
        inputs = batch["inputs"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype):
            predictions = predict_12(model, inputs)
        predictions = predictions.detach().float().requires_grad_(True)
        batch_size = predictions.shape[0]

        envelopes: dict[tuple[str, str], torch.Tensor] = {}
        for threshold_raw in args.thresholds_raw:
            threshold_key = f"{threshold_raw:g}"
            target_mask = targets >= threshold_raw / 255.0
            for radius in args.radii:
                envelopes[(threshold_key, str(radius))] = dilate(
                    target_mask, radius
                )

        for temperature_raw in args.temperatures_raw:
            temperature_key = f"{temperature_raw:g}"
            for threshold_raw in args.thresholds_raw:
                threshold_key = f"{threshold_raw:g}"
                criterion = criteria[(temperature_key, threshold_key)]
                predicted_count = criterion.soft_counts(predictions).squeeze(-1)
                target_count = criterion.soft_counts(targets).squeeze(-1).detach()
                loss = F.smooth_l1_loss(
                    torch.log1p(predicted_count), torch.log1p(target_count)
                )
                gradient = torch.autograd.grad(loss, predictions)[0]
                absolute = gradient.abs()
                upward = (-gradient).clamp_min(0)
                downward = gradient.clamp_min(0)
                acc = accumulators[temperature_key][threshold_key]
                acc["absolute_gradient"] += absolute.sum(dim=(0, 2, 3, 4)).double()
                acc["squared_gradient"] += gradient.square().sum(
                    dim=(0, 2, 3, 4)
                ).double()
                acc["upward_gradient"] += upward.sum(dim=(0, 2, 3, 4)).double()
                acc["downward_gradient"] += downward.sum(
                    dim=(0, 2, 3, 4)
                ).double()
                acc["predicted_soft_count"] += predicted_count.sum(dim=0).double()
                acc["target_soft_count"] += target_count.sum(dim=0).double()
                for radius in args.radii:
                    envelope = envelopes[(threshold_key, str(radius))]
                    acc["upward_inside_radius"][str(radius)] += (
                        upward * envelope
                    ).sum(dim=(0, 2, 3, 4)).double()
        completed_batches += 1
        completed_samples += batch_size

    pixels_per_lead = completed_samples * args.resolution * args.resolution
    result_by_temperature: dict[str, object] = {}
    for temperature_key, by_threshold in accumulators.items():
        result_by_temperature[temperature_key] = {}
        for threshold_key, acc in by_threshold.items():
            absolute = acc["absolute_gradient"]
            squared = acc["squared_gradient"]
            upward = acc["upward_gradient"]
            effective_support = torch.where(
                squared > 0,
                absolute.square() / (pixels_per_lead * squared),
                torch.zeros_like(squared),
            )
            inside: dict[str, list[float]] = {}
            for radius in args.radii:
                matched = acc["upward_inside_radius"][str(radius)]
                fraction = torch.where(
                    upward > 0, matched / upward, torch.zeros_like(upward)
                )
                inside[str(radius)] = [float(value) for value in fraction]
            result_by_temperature[temperature_key][threshold_key] = {
                "predicted_soft_count_mean_by_lead": [
                    float(value / completed_samples)
                    for value in acc["predicted_soft_count"]
                ],
                "target_soft_count_mean_by_lead": [
                    float(value / completed_samples)
                    for value in acc["target_soft_count"]
                ],
                "absolute_gradient_l1_by_lead": [float(value) for value in absolute],
                "upward_gradient_l1_by_lead": [float(value) for value in upward],
                "downward_gradient_l1_by_lead": [
                    float(value) for value in acc["downward_gradient"]
                ],
                "effective_gradient_support_fraction_by_lead": [
                    float(value) for value in effective_support
                ],
                "upward_gradient_fraction_inside_target_radius_by_lead": inside,
            }

    result = {
        "ok": True,
        "purpose": "gate_lead_adaptive_tail_temperature",
        "checkpoint": str(args.checkpoint),
        "resolution": args.resolution,
        "validation_batches": completed_batches,
        "validation_samples": completed_samples,
        "lead_minutes": [5 * (index + 1) for index in range(12)],
        "thresholds_raw": args.thresholds_raw,
        "temperatures_raw": args.temperatures_raw,
        "radii_pixels": args.radii,
        "result_by_temperature": result_by_temperature,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
