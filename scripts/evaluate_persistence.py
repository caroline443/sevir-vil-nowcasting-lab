#!/usr/bin/env python3
"""Evaluate last-observation persistence with lead-time SEVIR metrics."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from torch.utils.data import DataLoader

from sevir_nowcasting.data import SevirVILWindowDataset
from sevir_nowcasting.metrics import LeadTimeVILMetrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-val-batches", type=int, default=200)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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
    metrics = LeadTimeVILMetrics(output_length=12)
    batches = 0
    start = time.perf_counter()
    for batch_index, batch in enumerate(loader):
        if batch_index >= args.max_val_batches:
            break
        inputs = batch["inputs"]
        targets = batch["targets"]
        persistence = inputs[:, -1:].repeat(1, 12, 1, 1, 1)
        metrics.update(persistence, targets)
        batches += 1

    result = metrics.compute()
    result["baseline"] = "last_observation_persistence"
    result["resolution"] = args.resolution
    result["validation_batches"] = batches
    result["validation_samples_upper_bound"] = batches * args.batch_size
    result["wall_seconds"] = time.perf_counter() - start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
