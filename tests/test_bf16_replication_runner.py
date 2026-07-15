import json
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_bf16_paired_replications import load_valid_summary  # noqa: E402


def write_summary(directory: Path, **overrides: object) -> None:
    summary: dict[str, object] = {
        "ok": True,
        "amp_dtype": "bfloat16",
        "args": {"seed": 1},
        "train_steps": 4000,
        "optimizer_updates": 4000,
        "amp_fp32_fallbacks": 0,
        "skipped_optimizer_updates": 0,
    }
    summary.update(overrides)
    directory.mkdir()
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_accepts_frozen_bf16_summary(tmp_path: Path) -> None:
    output_dir = tmp_path / "valid"
    write_summary(output_dir)
    assert load_valid_summary(output_dir, 1)["optimizer_updates"] == 4000


def test_rejects_run_with_fallback(tmp_path: Path) -> None:
    output_dir = tmp_path / "invalid"
    write_summary(output_dir, amp_fp32_fallbacks=1)
    with pytest.raises(RuntimeError, match="amp_fp32_fallbacks"):
        load_valid_summary(output_dir, 1)
