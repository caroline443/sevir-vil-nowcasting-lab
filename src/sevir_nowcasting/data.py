"""Manifest-driven loader for raw SEVIR VIL HDF5 files."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor
from torch.utils.data import Dataset


class SevirVILWindowDataset(Dataset[dict[str, Any]]):
    """Load event-disjoint VIL windows described by a CSV manifest.

    Required manifest columns are ``event_id``, ``file_path``, ``file_index``,
    ``start_frame`` and ``split``. Raw VIL arrays may be ``[H,W,T]`` or
    ``[T,H,W]`` after selecting ``file_index`` from the HDF5 ``vil`` dataset.
    """

    def __init__(
        self,
        manifest: str | Path,
        data_root: str | Path,
        *,
        split: str,
        input_length: int = 13,
        output_length: int = 12,
        resolution: int = 128,
    ) -> None:
        self.manifest = Path(manifest)
        self.data_root = Path(data_root)
        self.split = split
        self.input_length = input_length
        self.output_length = output_length
        self.total_length = input_length + output_length
        self.resolution = resolution
        self.rows = self._read_manifest()
        self._handles: dict[Path, Any] = {}

        if not self.rows:
            raise ValueError(f"manifest contains no rows for split={split!r}")

    def _read_manifest(self) -> list[dict[str, str]]:
        required = {"event_id", "file_path", "file_index", "start_frame", "split"}
        with self.manifest.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            missing = required - fields
            if missing:
                raise ValueError(f"manifest missing columns: {sorted(missing)}")
            return [row for row in reader if row["split"] == self.split]

    def _get_handle(self, path: Path) -> Any:
        if path not in self._handles:
            try:
                import h5py
            except ImportError as exc:
                raise RuntimeError(
                    "h5py is required for raw SEVIR data; install requirements-data.txt"
                ) from exc
            self._handles[path] = h5py.File(path, "r")
        return self._handles[path]

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        path = self.data_root / row["file_path"]
        handle = self._get_handle(path)
        if "vil" not in handle:
            raise KeyError(f"{path} does not contain an HDF5 dataset named 'vil'")

        raw = handle["vil"][int(row["file_index"])]
        sequence = torch.as_tensor(raw)
        if sequence.ndim != 3:
            raise ValueError(f"expected a 3D VIL event, got {tuple(sequence.shape)}")

        # Raw SEVIR commonly stores one event as [H, W, T].
        if sequence.shape[-1] >= self.total_length and sequence.shape[0] == sequence.shape[1]:
            sequence = sequence.permute(2, 0, 1)
        elif sequence.shape[0] < self.total_length:
            raise ValueError(
                f"cannot identify time axis for event shape {tuple(sequence.shape)}"
            )

        start = int(row["start_frame"])
        stop = start + self.total_length
        if stop > sequence.shape[0]:
            raise IndexError(
                f"window [{start}:{stop}] exceeds {sequence.shape[0]} frames"
            )

        sequence = sequence[start:stop].to(torch.float32).unsqueeze(1) / 255.0
        if sequence.shape[-2:] != (self.resolution, self.resolution):
            sequence = functional.interpolate(
                sequence,
                size=(self.resolution, self.resolution),
                mode="bilinear",
                align_corners=False,
            )

        return {
            "inputs": sequence[: self.input_length],
            "targets": sequence[self.input_length :],
            "event_id": row["event_id"],
            "start_frame": start,
        }
