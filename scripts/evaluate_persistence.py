#!/usr/bin/env python3
"""Evaluate last-observation persistence with lead-time SEVIR metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from torch.utils.data import DataLoader

from sevir_nowcasting.data import SevirVILWindowDataset
from sevir_nowcasting.event_metrics import EventVILStats
from sevir_nowcasting.metrics import LeadTimeVILMetrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--event-stats-output",
        type=Path,
        help="optional .npz output for event-level comparisons",
    )
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--max-batches",
        "--max-val-batches",
        dest="max_batches",
        type=int,
        default=0,
        help="zero evaluates the complete selected split",
    )
    parser.add_argument(
        "--confirm-final-test",
        action="store_true",
        help="required for test split to make the one-time evaluation explicit",
    )
    args = parser.parse_args()
    if args.max_batches < 0:
        parser.error("--max-batches must be non-negative")
    if args.split == "test" and not args.confirm_final_test:
        parser.error("test evaluation requires --confirm-final-test")
    if args.output.exists():
        parser.error(f"refusing to overwrite existing evaluation: {args.output}")
    if args.event_stats_output is not None:
        if args.event_stats_output.suffix != ".npz":
            parser.error("--event-stats-output must use the .npz suffix")
        if args.event_stats_output.exists():
            parser.error(
                "refusing to overwrite existing event statistics: "
                f"{args.event_stats_output}"
            )
    return args


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    metrics = LeadTimeVILMetrics(output_length=12)
    event_stats = EventVILStats(output_length=12)
    batches = 0
    start = time.perf_counter()
    for batch_index, batch in enumerate(loader):
        if args.max_batches > 0 and batch_index >= args.max_batches:
            break
        inputs = batch["inputs"]
        targets = batch["targets"]
        persistence = inputs[:, -1:].repeat(1, 12, 1, 1, 1)
        metrics.update(persistence, targets)
        event_stats.update(batch["event_id"], persistence, targets)
        batches += 1

    result = metrics.compute()
    if args.event_stats_output is not None:
        event_stats.save(args.event_stats_output)
    result["baseline"] = "last_observation_persistence"
    result["split"] = args.split
    result["deployable_test_evaluation"] = args.split == "test"
    result["resolution"] = args.resolution
    result["batches"] = batches
    result["samples_upper_bound"] = batches * args.batch_size
    result["wall_seconds"] = time.perf_counter() - start
    result["event_stats_output"] = (
        str(args.event_stats_output)
        if args.event_stats_output is not None
        else None
    )
    result["event_stats_sha256"] = (
        file_sha256(args.event_stats_output)
        if args.event_stats_output is not None
        else None
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
