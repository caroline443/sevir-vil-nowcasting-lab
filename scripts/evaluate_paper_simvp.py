#!/usr/bin/env python3
"""Evaluate a frozen paper-protocol SimVP checkpoint on validation or test."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import torch
from torch.utils.data import DataLoader

from sevir_nowcasting.data import SevirVILWindowDataset
from sevir_nowcasting.metrics import LeadTimeVILMetrics
from train_openstl_simvp import build_model, predict_12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="zero evaluates the complete selected split",
    )
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--confirm-final-test",
        action="store_true",
        help="required for test split to make the one-time evaluation explicit",
    )
    args = parser.parse_args()
    if min(args.batch_size, args.workers + 1, args.log_every) < 1:
        parser.error("batch size, workers and log interval are invalid")
    if args.max_batches < 0:
        parser.error("max-batches must be non-negative")
    if args.split == "test" and not args.confirm_final_test:
        parser.error("test evaluation requires --confirm-final-test")
    if args.output.exists():
        parser.error(f"refusing to overwrite existing evaluation: {args.output}")
    return args


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("paper evaluation requires CUDA with BF16 support")
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    signature = checkpoint.get("configuration_signature")
    if not isinstance(signature, dict):
        raise RuntimeError("checkpoint lacks a paper-protocol signature")
    resolution = int(signature["resolution"])
    model_type = str(signature["model_type"])

    dataset = SevirVILWindowDataset(
        args.manifest,
        args.data_root,
        split=args.split,
        input_length=13,
        output_length=12,
        resolution=resolution,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    batch_limit = len(loader) if args.max_batches == 0 else min(
        args.max_batches, len(loader)
    )
    device = torch.device("cuda:0")
    model = build_model(resolution, model_type).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    metrics = LeadTimeVILMetrics(output_length=12)
    samples = 0
    torch.cuda.reset_peak_memory_stats(device)
    started_at = time.perf_counter()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= batch_limit:
                break
            inputs = batch["inputs"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                predictions = predict_12(model, inputs)
            metrics.update(predictions, targets)
            samples += inputs.shape[0]
            completed = batch_index + 1
            if completed == 1 or completed % args.log_every == 0:
                print(
                    json.dumps(
                        {
                            "split": args.split,
                            "completed_batches": completed,
                            "batch_limit": batch_limit,
                            "samples": samples,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    torch.cuda.synchronize(device)
    result = {
        "ok": True,
        "purpose": "paper_protocol_checkpoint_evaluation",
        "split": args.split,
        "deployable_test_evaluation": args.split == "test",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "configuration_signature": signature,
        "samples": samples,
        "batches": batch_limit,
        "device": torch.cuda.get_device_name(device),
        "amp_dtype": "bfloat16",
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "wall_seconds": time.perf_counter() - started_at,
        "metrics": metrics.compute(),
        "caveat": (
            "Validation may be repeated for engineering checks. Test output "
            "must be generated only after the configuration is frozen."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
