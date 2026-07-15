from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class ManifestBuilderTest(unittest.TestCase):
    def test_time_split_occurs_before_window_expansion(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            catalog = temp / "CATALOG.csv"
            output = temp / "manifest.csv"
            rows = [
                {
                    "id": "train-event",
                    "file_name": "data/vil/train.h5",
                    "file_index": "0",
                    "img_type": "vil",
                    "time_utc": "2018-12-31T00:00:00",
                    "pct_missing": "0",
                },
                {
                    "id": "train-boundary-event",
                    "file_name": "data/vil/train-boundary.h5",
                    "file_index": "3",
                    "img_type": "vil",
                    "time_utc": "2019-01-01T00:00:00",
                    "pct_missing": "0",
                },
                {
                    "id": "val-event",
                    "file_name": "data/vil/val.h5",
                    "file_index": "1",
                    "img_type": "vil",
                    "time_utc": "2019-03-01T00:00:00",
                    "pct_missing": "0",
                },
                {
                    "id": "test-event",
                    "file_name": "data/vil/test.h5",
                    "file_index": "2",
                    "img_type": "vil",
                    "time_utc": "2019-07-01T00:00:00",
                    "pct_missing": "0",
                },
                {
                    "id": "val-boundary-event",
                    "file_name": "data/vil/val-boundary.h5",
                    "file_index": "4",
                    "img_type": "vil",
                    "time_utc": "2019-06-01T00:00:00",
                    "pct_missing": "0",
                },
                {
                    "id": "duplicate-event",
                    "file_name": "data/vil/duplicate-a.h5",
                    "file_index": "5",
                    "img_type": "vil",
                    "time_utc": "2018-06-01T00:00:00",
                    "pct_missing": "0",
                },
                {
                    "id": "duplicate-event",
                    "file_name": "data/vil/duplicate-b.h5",
                    "file_index": "6",
                    "img_type": "vil",
                    "time_utc": "2018-06-01T00:00:00",
                    "pct_missing": "0",
                },
            ]
            with catalog.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

            subprocess.run(
                [
                    sys.executable,
                    str(root / "scripts" / "build_sevir_manifest.py"),
                    "--catalog",
                    str(catalog),
                    "--output",
                    str(output),
                    "--train-end",
                    "2019-01-01",
                    "--val-end",
                    "2019-06-01",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            with output.open(newline="", encoding="utf-8") as handle:
                manifest = list(csv.DictReader(handle))

        self.assertEqual(len(manifest), 15)
        event_splits: dict[str, set[str]] = {}
        for row in manifest:
            event_splits.setdefault(row["event_id"], set()).add(row["split"])
        self.assertEqual(event_splits["train-event"], {"train"})
        self.assertEqual(event_splits["train-boundary-event"], {"train"})
        self.assertEqual(event_splits["val-event"], {"val"})
        self.assertEqual(event_splits["val-boundary-event"], {"val"})
        self.assertEqual(event_splits["test-event"], {"test"})
        self.assertNotIn("duplicate-event", event_splits)


if __name__ == "__main__":
    unittest.main()
