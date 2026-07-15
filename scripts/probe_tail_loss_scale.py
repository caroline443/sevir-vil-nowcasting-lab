#!/usr/bin/env python3
"""Estimate a principled tail-area loss weight from a trained checkpoint."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

# Editable installs become stale if a WSL repository is moved.  Resolve the
# in-repository package directly so the documented root-level command is robust.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
from torch import nn
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--tail-thresholds", type=float, nargs="+", default=[160, 181, 219]
    )
    parser.add_argument("--tail-temperature-raw", type=float, default=2.0)
    parser.add_argument(
        "--amp-dtype",
        choices=("float16", "bfloat16"),
        default="float16",
        help="autocast dtype used for checkpoint inference",
    )
    parser.add_argument(
        "--target-gradient-fraction",
        type=float,
        default=0.1,
        help="desired weighted tail gradient norm divided by MSE gradient norm",
    )
    args = parser.parse_args()
    if args.max_batches < 1 or args.target_gradient_fraction <= 0:
        parser.error("max-batches and target-gradient-fraction must be positive")
    return args


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this probe expects a CUDA-capable machine")
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")
    amp_dtype = (
        torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    )
    if amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("this CUDA device/runtime does not support bfloat16 AMP")
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
    mse_fn = nn.MSELoss()
    tail_fn = SoftExceedanceAreaLoss(
        args.tail_thresholds, args.tail_temperature_raw
    ).to(device)

    rows: list[dict[str, float | int]] = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= args.max_batches:
            break
        inputs = batch["inputs"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype):
            prediction = predict_12(model, inputs)
        prediction = prediction.detach().float().requires_grad_(True)
        targets = targets.float()
        mse = mse_fn(prediction, targets)
        tail = tail_fn(prediction, targets)
        mse_grad = torch.autograd.grad(mse, prediction, retain_graph=True)[0]
        tail_grad = torch.autograd.grad(tail, prediction)[0]
        mse_norm = float(torch.linalg.vector_norm(mse_grad))
        tail_norm = float(torch.linalg.vector_norm(tail_grad))
        denominator = max(mse_norm * tail_norm, torch.finfo(torch.float32).tiny)
        cosine = float(torch.sum(mse_grad * tail_grad)) / denominator
        suggested = args.target_gradient_fraction * mse_norm / max(
            tail_norm, torch.finfo(torch.float32).tiny
        )
        rows.append(
            {
                "batch": batch_index,
                "mse": float(mse),
                "tail_area_loss": float(tail),
                "mse_gradient_norm": mse_norm,
                "tail_gradient_norm": tail_norm,
                "gradient_cosine": cosine,
                "suggested_weight": suggested,
            }
        )

    if not rows:
        raise RuntimeError("no validation batches were processed")
    result = {
        "ok": True,
        "purpose": "tail_loss_gradient_scale_probe",
        "checkpoint": str(args.checkpoint),
        "amp_dtype": args.amp_dtype,
        "validation_batches": len(rows),
        "target_gradient_fraction": args.target_gradient_fraction,
        "thresholds_raw": args.tail_thresholds,
        "temperature_raw": args.tail_temperature_raw,
        "recommended_tail_area_weight": statistics.median(
            float(row["suggested_weight"]) for row in rows
        ),
        "median_gradient_cosine": statistics.median(
            float(row["gradient_cosine"]) for row in rows
        ),
        "batches": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
