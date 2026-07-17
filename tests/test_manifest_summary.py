from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "summarize_sevir_manifest",
    ROOT / "scripts" / "summarize_sevir_manifest.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_manifest(path: Path, *, overlap: bool = False) -> None:
    fields = [
        "event_id",
        "file_path",
        "file_index",
        "start_frame",
        "split",
        "time_utc",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for split_index, split in enumerate(("train", "val", "test")):
            event_id = "shared" if overlap and split != "train" else f"event-{split}"
            if overlap and split == "train":
                event_id = "shared"
            for start in (0, 12, 24):
                writer.writerow(
                    {
                        "event_id": event_id,
                        "file_path": f"{split}.h5",
                        "file_index": split_index,
                        "start_frame": start,
                        "split": split,
                        "time_utc": f"2019-0{split_index + 1}-01 00:00:00",
                    }
                )


def test_valid_manifest_summary(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest)
    result = MODULE.summarize_manifest(manifest, [0, 12, 24])
    assert result["ok"] is True
    assert result["total_windows"] == 9
    assert result["window_counts"] == {"test": 3, "train": 3, "val": 3}
    assert result["cross_split_event_count"] == 0


def test_cross_split_event_is_rejected(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, overlap=True)
    result = MODULE.summarize_manifest(manifest, [0, 12, 24])
    assert result["ok"] is False
    assert result["cross_split_event_count"] == 1
