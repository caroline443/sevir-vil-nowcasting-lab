#!/usr/bin/env python3
"""Freeze and validate a paper-facing SEVIR window manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expected-start-frames", type=int, nargs="+", default=[0, 12, 24]
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_manifest(
    manifest: Path,
    expected_start_frames: list[int],
) -> dict[str, object]:
    required = {
        "event_id",
        "file_path",
        "file_index",
        "start_frame",
        "split",
        "time_utc",
    }
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"manifest missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("manifest is empty")

    row_counts = Counter(row["split"] for row in rows)
    allowed_splits = {"train", "val", "test"}
    unexpected_splits = sorted(set(row_counts) - allowed_splits)
    if unexpected_splits:
        raise ValueError(f"unexpected splits: {unexpected_splits}")
    missing_splits = sorted(allowed_splits - set(row_counts))
    if missing_splits:
        raise ValueError(f"manifest lacks splits: {missing_splits}")

    event_splits: dict[str, set[str]] = defaultdict(set)
    event_starts: dict[tuple[str, str], list[int]] = defaultdict(list)
    time_values: dict[str, list[str]] = defaultdict(list)
    window_keys: set[tuple[str, str, int]] = set()
    duplicate_windows = 0
    for row in rows:
        event_id = row["event_id"]
        split = row["split"]
        start = int(row["start_frame"])
        event_splits[event_id].add(split)
        event_starts[(split, event_id)].append(start)
        time_values[split].append(row["time_utc"])
        key = (split, event_id, start)
        if key in window_keys:
            duplicate_windows += 1
        window_keys.add(key)

    overlapping_events = sorted(
        event_id
        for event_id, splits in event_splits.items()
        if len(splits) > 1
    )
    expected = sorted(expected_start_frames)
    malformed_events = [
        {
            "split": split,
            "event_id": event_id,
            "observed_start_frames": sorted(starts),
        }
        for (split, event_id), starts in event_starts.items()
        if sorted(starts) != expected
    ]
    unique_event_counts = {
        split: len(
            {
                event_id
                for (event_split, event_id) in event_starts
                if event_split == split
            }
        )
        for split in sorted(allowed_splits)
    }
    time_ranges = {
        split: {
            "min": min(time_values[split]),
            "max": max(time_values[split]),
        }
        for split in sorted(allowed_splits)
    }
    ok = (
        not overlapping_events
        and not malformed_events
        and duplicate_windows == 0
    )
    return {
        "ok": ok,
        "manifest": str(manifest),
        "sha256": file_sha256(manifest),
        "total_windows": len(rows),
        "window_counts": dict(sorted(row_counts.items())),
        "unique_event_counts": unique_event_counts,
        "expected_start_frames": expected,
        "time_ranges": time_ranges,
        "duplicate_window_count": duplicate_windows,
        "cross_split_event_count": len(overlapping_events),
        "cross_split_event_examples": overlapping_events[:20],
        "malformed_event_window_count": len(malformed_events),
        "malformed_event_window_examples": malformed_events[:20],
    }


def main() -> int:
    args = parse_args()
    result = summarize_manifest(args.manifest, args.expected_start_frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
