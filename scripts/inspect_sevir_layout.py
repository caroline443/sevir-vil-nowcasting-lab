#!/usr/bin/env python3
"""Inspect one raw SEVIR VIL event without reporting private absolute paths."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--catalog",
        type=Path,
        help="Catalog path; defaults to DATA_ROOT/CATALOG.csv.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def failure(message: str, **details: Any) -> dict[str, Any]:
    return {"ok": False, "error": message, **details}


def inspect(args: argparse.Namespace) -> dict[str, Any]:
    catalog = args.catalog or args.data_root / "CATALOG.csv"
    if not catalog.is_file():
        return failure(
            "catalog_not_found",
            expected_catalog_name=catalog.name,
            hint="Pass --catalog if the catalog is stored outside DATA_ROOT.",
        )

    first_vil: dict[str, str] | None = None
    vil_rows = 0
    with catalog.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        required = {"id", "file_name", "file_index", "img_type", "time_utc"}
        missing = sorted(required - fields)
        if missing:
            return failure(
                "catalog_columns_missing",
                catalog_name=catalog.name,
                missing_columns=missing,
                available_columns=sorted(fields),
            )
        for row in reader:
            if row["img_type"].lower() == "vil":
                vil_rows += 1
                if first_vil is None:
                    first_vil = row

    if first_vil is None:
        return failure("no_vil_rows", catalog_name=catalog.name)

    relative_file = Path(first_vil["file_name"])
    h5_path = args.data_root / relative_file
    if not h5_path.is_file():
        return failure(
            "hdf5_file_not_found",
            catalog_name=catalog.name,
            relative_file=str(relative_file),
            hint="DATA_ROOT must be the directory relative to file_name in CATALOG.csv.",
        )

    try:
        import h5py
    except ImportError:
        return failure(
            "h5py_not_installed",
            hint="Run: python -m pip install -r requirements-data.txt",
        )

    with h5py.File(h5_path, "r") as handle:
        keys = sorted(handle.keys())
        if "vil" not in handle:
            return failure(
                "vil_dataset_missing",
                relative_file=str(relative_file),
                hdf5_keys=keys,
            )
        dataset = handle["vil"]
        file_index = int(first_vil["file_index"])
        if file_index >= dataset.shape[0]:
            return failure(
                "file_index_out_of_range",
                relative_file=str(relative_file),
                file_index=file_index,
                dataset_shape=list(dataset.shape),
            )
        dataset_shape = list(dataset.shape)
        dataset_dtype = str(dataset.dtype)
        sample = dataset[file_index]

    sample_shape = list(sample.shape)
    expected_shape = sample_shape in ([384, 384, 49], [49, 384, 384])
    return {
        "ok": expected_shape,
        "catalog_name": catalog.name,
        "catalog_vil_rows": vil_rows,
        "catalog_columns": sorted(fields),
        "example_event_id": first_vil["id"],
        "example_time_utc": first_vil["time_utc"],
        "relative_file": str(relative_file),
        "file_index": int(first_vil["file_index"]),
        "hdf5_keys": keys,
        "vil_dataset_shape": dataset_shape,
        "vil_dtype": dataset_dtype,
        "sample_shape": sample_shape,
        "expected_sample_shape": expected_shape,
    }


def main() -> int:
    args = parse_args()
    report = inspect(args)
    serialized = json.dumps(report, indent=2, sort_keys=True)
    print(serialized)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
