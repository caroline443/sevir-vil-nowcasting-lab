#!/usr/bin/env python3
"""Create an event-disjoint, time-split VIL window manifest from CATALOG.csv."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path


def parse_date(value: str) -> datetime:
    normalized = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    return parsed.replace(tzinfo=None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-end", type=parse_date, required=True)
    parser.add_argument("--val-end", type=parse_date, required=True)
    parser.add_argument(
        "--start-frames",
        default="0,12,24",
        help="Comma-separated 25-frame window starts for each 49-frame event.",
    )
    parser.add_argument("--max-missing-percent", type=float, default=0.0)
    return parser.parse_args()


def choose_split(timestamp: datetime, train_end: datetime, val_end: datetime) -> str:
    if timestamp < train_end:
        return "train"
    if timestamp < val_end:
        return "val"
    return "test"


def main() -> int:
    args = parse_args()
    if args.train_end >= args.val_end:
        raise ValueError("train-end must be earlier than val-end")
    starts = [int(value) for value in args.start_frames.split(",")]
    if any(start < 0 or start + 25 > 49 for start in starts):
        raise ValueError("each start frame must define a valid 25-frame window in 49 frames")

    output_rows: list[dict[str, str | int]] = []
    with args.catalog.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "file_name", "file_index", "img_type", "time_utc"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"catalog missing columns: {sorted(missing)}")

        for row in reader:
            if row["img_type"].lower() != "vil":
                continue
            missing_percent = float(row.get("pct_missing") or 0.0)
            if missing_percent > args.max_missing_percent:
                continue
            timestamp = parse_date(row["time_utc"])
            split = choose_split(timestamp, args.train_end, args.val_end)
            for start in starts:
                output_rows.append(
                    {
                        "event_id": row["id"],
                        "file_path": row["file_name"],
                        "file_index": int(row["file_index"]),
                        "start_frame": start,
                        "split": split,
                        "time_utc": row["time_utc"],
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["event_id", "file_path", "file_index", "start_frame", "split", "time_utc"],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    counts = {split: 0 for split in ("train", "val", "test")}
    for row in output_rows:
        counts[str(row["split"])] += 1
    print(f"wrote {len(output_rows)} windows to {args.output}: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
