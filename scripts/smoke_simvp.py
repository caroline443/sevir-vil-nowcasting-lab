#!/usr/bin/env python3
"""Run one synthetic SimVP optimization step and report CUDA memory."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch import nn

from sevir_nowcasting.model import SimVP, parameter_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--hidden-spatial", type=int, default=32)
    parser.add_argument("--hidden-temporal", type=int, default=192)
    parser.add_argument("--temporal-depth", type=int, default=4)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the A4000 smoke test")
    if args.resolution % 4:
        raise ValueError("resolution must be divisible by four")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    device = torch.device("cuda:0")
    model = SimVP(
        hidden_spatial=args.hidden_spatial,
        hidden_temporal=args.hidden_temporal,
        temporal_depth=args.temporal_depth,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=not args.no_amp)
    criterion = nn.MSELoss()

    inputs = torch.rand(
        args.batch_size, 13, 1, args.resolution, args.resolution, device=device
    )
    targets = torch.rand(
        args.batch_size, 12, 1, args.resolution, args.resolution, device=device
    )
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16, enabled=not args.no_amp):
        predictions = model(inputs)
        loss = criterion(predictions, targets)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize(device)

    report = {
        "ok": True,
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "batch_size": args.batch_size,
        "resolution": args.resolution,
        "input_shape": list(inputs.shape),
        "output_shape": list(predictions.shape),
        "parameters": parameter_count(model),
        "loss": float(loss.detach()),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "step_seconds": time.perf_counter() - start,
        "amp": not args.no_amp,
    }
    serialized = json.dumps(report, indent=2, sort_keys=True)
    print(serialized)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
