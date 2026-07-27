#!/usr/bin/env python3
"""Compute paired storm-event bootstrap intervals from frozen evaluations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sevir_nowcasting.event_metrics import paired_event_bootstrap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-stats", type=Path, nargs="+", required=True
    )
    parser.add_argument(
        "--proposed-stats", type=Path, nargs="+", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    if len(args.baseline_stats) != len(args.proposed_stats):
        parser.error("baseline and proposed paths must form seed-matched pairs")
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    return args


def main() -> int:
    args = parse_args()
    result = paired_event_bootstrap(
        args.baseline_stats,
        args.proposed_stats,
        repetitions=args.repetitions,
        seed=args.seed,
    )
    result["baseline_stats"] = [str(path) for path in args.baseline_stats]
    result["proposed_stats"] = [str(path) for path in args.proposed_stats]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
