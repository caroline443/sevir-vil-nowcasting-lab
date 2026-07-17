#!/usr/bin/env python3
"""Audit ConvLSTM calibration across teacher-forcing probabilities."""

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
from sevir_nowcasting.metrics import LeadTimeVILMetrics
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
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument(
        "--teacher-forcing-probabilities",
        type=float,
        nargs="+",
        default=[1.0, 0.92, 0.5, 0.0],
    )
    parser.add_argument("--mask-seed", type=int, default=2026)
    args = parser.parse_args()
    if min(
        args.resolution,
        args.batch_size,
        args.workers + 1,
        args.max_batches,
        args.log_every,
        args.patch_size,
    ) < 1:
        parser.error("resolution, batch size and batch limits must be positive")
    if args.resolution % args.patch_size:
        parser.error("resolution must be divisible by patch size")
    if not args.teacher_forcing_probabilities or any(
        probability < 0 or probability > 1
        for probability in args.teacher_forcing_probabilities
    ):
        parser.error("teacher-forcing probabilities must lie in [0, 1]")
    return args


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


def make_mask(
    batch_size: int,
    probability: float,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    if probability == 0:
        return torch.zeros(batch_size, 11, 1, 1, 1, device=device)
    if probability == 1:
        return torch.ones(batch_size, 11, 1, 1, 1, device=device)
    return (
        torch.rand(
            batch_size,
            11,
            1,
            1,
            1,
            device=device,
            generator=generator,
        )
        < probability
    ).float()


def compact_metrics(metrics: dict[str, object]) -> dict[str, object]:
    forecast = metrics["forecast_pixels_by_threshold"]
    observed = metrics["observed_pixels_by_threshold"]
    csi = metrics["csi_by_threshold"]
    assert isinstance(forecast, dict)
    assert isinstance(observed, dict)
    assert isinstance(csi, dict)
    ratios: dict[str, list[float]] = {}
    severe_csi: dict[str, list[float]] = {}
    for threshold in ("160", "181", "219"):
        forecast_values = forecast[threshold]
        observed_values = observed[threshold]
        assert isinstance(forecast_values, list)
        assert isinstance(observed_values, list)
        ratios[threshold] = [
            predicted / actual if actual > 0 else 0.0
            for predicted, actual in zip(forecast_values, observed_values)
        ]
        severe_csi[threshold] = csi[threshold]
    mean_prediction = metrics["mean_prediction_by_lead"]
    mean_target = metrics["mean_target_by_lead"]
    assert isinstance(mean_prediction, list)
    assert isinstance(mean_target, list)
    return {
        "mse": metrics["mse"],
        "csi_mean": metrics["csi_mean"],
        "severe_csi_by_threshold": severe_csi,
        "severe_forecast_to_observed_by_threshold": ratios,
        "mean_prediction_minus_target_by_lead": [
            prediction - target
            for prediction, target in zip(mean_prediction, mean_target)
        ],
    }


@torch.no_grad()
def evaluate_probability(
    model: torch.nn.Module,
    loader: DataLoader,
    checkpoint_path: Path,
    probability: float,
    probability_index: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    metrics = LeadTimeVILMetrics(output_length=12)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.mask_seed + probability_index)
    completed_batches = 0
    samples = 0
    for batch_index, batch in enumerate(loader):
        if batch_index >= args.max_batches:
            break
        inputs = batch["inputs"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        mask = make_mask(
            inputs.shape[0], probability, device, generator
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predictions = predict_future(
                model, inputs, targets, mask, args.patch_size
            )
        metrics.update(predictions, targets)
        completed_batches += 1
        samples += inputs.shape[0]
        if completed_batches == 1 or completed_batches % args.log_every == 0:
            print(
                json.dumps(
                    {
                        "checkpoint": str(checkpoint_path),
                        "teacher_forcing_probability": probability,
                        "completed_batches": completed_batches,
                        "max_batches": args.max_batches,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return {
        "teacher_forcing_probability": probability,
        "samples": samples,
        "validation_batches": completed_batches,
        **compact_metrics(metrics.compute()),
    }


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("this audit requires CUDA with BF16 support")
    device = torch.device("cuda:0")
    loader = build_loader(args)
    results: list[dict[str, object]] = []
    for checkpoint_path in args.checkpoint:
        model_args = SimpleNamespace(
            resolution=args.resolution,
            patch_size=args.patch_size,
        )
        model = build_model(model_args).to(device)
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        model.load_state_dict(checkpoint["model"])
        model.eval()
        probability_results = [
            evaluate_probability(
                model,
                loader,
                checkpoint_path,
                probability,
                probability_index,
                args,
                device,
            )
            for probability_index, probability in enumerate(
                args.teacher_forcing_probabilities
            )
        ]
        results.append(
            {
                "checkpoint": str(checkpoint_path),
                "checkpoint_summary": checkpoint.get("summary", {}),
                "probabilities": probability_results,
            }
        )
        del model
        torch.cuda.empty_cache()

    output = {
        "ok": True,
        "purpose": "convlstm_teacher_forcing_exposure_audit",
        "lead_minutes": list(range(5, 65, 5)),
        "teacher_forcing_probabilities": args.teacher_forcing_probabilities,
        "mask_seed": args.mask_seed,
        "results": results,
        "caveat": (
            "Probabilities above zero use future observations and are diagnostic "
            "counterfactuals, not valid nowcasts. Only probability zero is a "
            "deployable forecast."
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
