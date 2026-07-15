#!/usr/bin/env python3
"""Run the remaining frozen BF16 baseline/proposed replications safely."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_openstl_simvp.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts/local"))
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def load_valid_summary(output_dir: Path, expected_seed: int) -> dict[str, object]:
    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"missing completed summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checks = {
        "ok": summary.get("ok") is True,
        "amp_dtype": summary.get("amp_dtype") == "bfloat16",
        "seed": summary.get("args", {}).get("seed") == expected_seed,
        "train_steps": summary.get("train_steps") == 4000,
        "optimizer_updates": summary.get("optimizer_updates") == 4000,
        "amp_fp32_fallbacks": summary.get("amp_fp32_fallbacks") == 0,
        "skipped_optimizer_updates": summary.get("skipped_optimizer_updates") == 0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            f"invalid BF16 run in {output_dir}; failed checks: {', '.join(failed)}"
        )
    return summary


def run_job(
    args: argparse.Namespace,
    *,
    name: str,
    seed: int,
    output_dir: Path,
    tail_area_weight: float,
) -> None:
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        load_valid_summary(output_dir, seed)
        print(f"SKIP valid completed job: {name}", flush=True)
        return
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--data-root",
        str(args.data_root),
        "--manifest",
        str(args.manifest),
        "--output-dir",
        str(output_dir),
        "--resolution",
        "128",
        "--batch-size",
        "8",
        "--epochs",
        "1",
        "--max-train-batches",
        "4000",
        "--max-val-batches",
        "200",
        "--learning-rate",
        "0.005",
        "--tail-area-weight",
        str(tail_area_weight),
        "--amp-dtype",
        "bfloat16",
        "--seed",
        str(seed),
        "--workers",
        str(args.workers),
    ]
    if tail_area_weight > 0:
        command.extend(
            [
                "--tail-thresholds",
                "160",
                "181",
                "219",
                "--tail-temperature-raw",
                "10",
            ]
        )
    print(f"START job: {name}", flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    load_valid_summary(output_dir, seed)
    print(f"PASS job: {name}", flush=True)


def main() -> int:
    args = parse_args()
    # Seed-1 baseline was already accepted by EXP-009 and is part of the final
    # paired set. Validate it before spending compute on the remaining arms.
    seed1_baseline = args.artifacts_root / "exp009_bf16_protocol_gate_seed1"
    load_valid_summary(seed1_baseline, 1)

    jobs = [
        ("baseline-seed0", 0, "exp010_bf16_baseline_seed0", 0.0),
        ("tail-area-seed0", 0, "exp010_bf16_tail_area_seed0", 0.0003),
        ("tail-area-seed1", 1, "exp010_bf16_tail_area_seed1", 0.0003),
        ("baseline-seed2", 2, "exp010_bf16_baseline_seed2", 0.0),
        ("tail-area-seed2", 2, "exp010_bf16_tail_area_seed2", 0.0003),
    ]
    for name, seed, directory, weight in jobs:
        run_job(
            args,
            name=name,
            seed=seed,
            output_dir=args.artifacts_root / directory,
            tail_area_weight=weight,
        )
    print("All frozen BF16 paired replications passed numerical checks.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
