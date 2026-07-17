#!/usr/bin/env python3
"""Audit hard/soft severe-area calibration of trained ConvLSTM checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import torch
from torch.utils.data import DataLoader

from sevir_nowcasting.data import SevirVILWindowDataset
from sevir_nowcasting.losses import SoftExceedanceAreaLoss
from train_openstl_convlstm import build_model, predict_future


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument(
        "--thresholds-raw", type=float, nargs="+", default=[160, 181, 219]
    )
    parser.add_argument(
        "--temperatures-raw", type=float, nargs="+", default=[2, 5, 10]
    )
    args = parser.parse_args()
    if min(
        args.resolution,
        args.batch_size,
        args.workers + 1,
        args.max_batches,
        args.log_every,
        args.patch_size,
    ) < 1:
        parser.error("resolution, batch size, batch limit and patch size are invalid")
    if args.resolution % args.patch_size:
        parser.error("resolution must be divisible by patch size")
    if not args.thresholds_raw or min(args.thresholds_raw) < 0:
        parser.error("thresholds must be non-negative")
    if not args.temperatures_raw or min(args.temperatures_raw) <= 0:
        parser.error("temperatures must be positive")
    return args


def nested_list(tensor: torch.Tensor) -> list[list[float]]:
    return tensor.detach().cpu().double().tolist()


def safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return torch.where(
        denominator > 0,
        numerator / denominator,
        torch.zeros_like(numerator),
    )


def build_loader(args: argparse.Namespace) -> DataLoader:
    dataset = SevirVILWindowDataset(
        args.manifest,
        args.data_root,
        split="val",
        input_length=13,
        output_length=12,
        resolution=args.resolution,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )


@torch.no_grad()
def audit_checkpoint(
    checkpoint_path: Path,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    model_args = SimpleNamespace(
        resolution=args.resolution,
        patch_size=args.patch_size,
    )
    model = build_model(model_args).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    thresholds = torch.tensor(
        args.thresholds_raw, device=device, dtype=torch.float32
    ) / 255.0
    criteria = {
        str(float(temperature)): SoftExceedanceAreaLoss(
            thresholds_raw=args.thresholds_raw,
            temperature_raw=temperature,
        ).to(device)
        for temperature in args.temperatures_raw
    }
    shape = (12, len(args.thresholds_raw))
    hard_forecast = torch.zeros(shape, device=device, dtype=torch.float64)
    hard_observed = torch.zeros_like(hard_forecast)
    soft_forecast = {
        key: torch.zeros_like(hard_forecast) for key in criteria
    }
    soft_observed = {
        key: torch.zeros_like(hard_forecast) for key in criteria
    }
    sample_log_area_absolute_error = {
        key: torch.zeros(shape, device=device, dtype=torch.float64)
        for key in criteria
    }
    samples = 0
    completed_batches = 0

    for batch_index, batch in enumerate(loader):
        if batch_index >= args.max_batches:
            break
        inputs = batch["inputs"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        mask = torch.zeros(inputs.shape[0], 11, 1, 1, 1, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predictions = predict_future(
                model, inputs, targets, mask, args.patch_size
            )

        expanded_thresholds = thresholds.view(1, 1, -1, 1, 1, 1)
        prediction_values = predictions.float().unsqueeze(2)
        target_values = targets.float().unsqueeze(2)
        hard_forecast += (
            prediction_values >= expanded_thresholds
        ).sum(dim=(0, 3, 4, 5), dtype=torch.float64)
        hard_observed += (
            target_values >= expanded_thresholds
        ).sum(dim=(0, 3, 4, 5), dtype=torch.float64)

        for key, criterion in criteria.items():
            predicted_counts = criterion.soft_counts(predictions)
            observed_counts = criterion.soft_counts(targets)
            soft_forecast[key] += predicted_counts.sum(dim=0, dtype=torch.float64)
            soft_observed[key] += observed_counts.sum(dim=0, dtype=torch.float64)
            sample_log_area_absolute_error[key] += (
                torch.log1p(predicted_counts) - torch.log1p(observed_counts)
            ).abs().sum(dim=0, dtype=torch.float64)

        samples += inputs.shape[0]
        completed_batches += 1
        if completed_batches == 1 or completed_batches % args.log_every == 0:
            print(
                json.dumps(
                    {
                        "checkpoint": str(checkpoint_path),
                        "completed_batches": completed_batches,
                        "max_batches": args.max_batches,
                        "samples": samples,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    hard_ratio = safe_ratio(hard_forecast, hard_observed)
    temperatures: dict[str, object] = {}
    for key in criteria:
        soft_ratio = safe_ratio(soft_forecast[key], soft_observed[key])
        temperatures[key] = {
            "soft_forecast_to_observed_by_lead": nested_list(soft_ratio),
            "hard_minus_soft_ratio_by_lead": nested_list(hard_ratio - soft_ratio),
            "mean_sample_absolute_log_area_error_by_lead": nested_list(
                sample_log_area_absolute_error[key] / samples
            ),
        }

    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_summary": checkpoint.get("summary", {}),
        "samples": samples,
        "validation_batches": completed_batches,
        "hard_forecast_pixels_by_lead": nested_list(hard_forecast),
        "hard_observed_pixels_by_lead": nested_list(hard_observed),
        "hard_forecast_to_observed_by_lead": nested_list(hard_ratio),
        "temperatures": temperatures,
    }


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("this audit requires CUDA with BF16 support")
    device = torch.device("cuda:0")
    loader = build_loader(args)
    results = [
        audit_checkpoint(checkpoint, loader, args, device)
        for checkpoint in args.checkpoint
    ]
    output = {
        "ok": True,
        "purpose": "convlstm_hard_soft_area_surrogate_audit",
        "thresholds_raw": args.thresholds_raw,
        "temperatures_raw": args.temperatures_raw,
        "lead_minutes": list(range(5, 65, 5)),
        "results": results,
        "caveat": (
            "This is a read-only diagnostic. It tests whether soft-area "
            "calibration at the training temperature agrees with hard-threshold "
            "area calibration; it is not a model comparison."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
