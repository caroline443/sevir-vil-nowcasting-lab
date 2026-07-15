#!/usr/bin/env python3
"""Collect a minimal, privacy-conscious environment report for GPU experiments."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any


def collect_torch(cuda_smoke_test: bool) -> dict[str, Any]:
    report: dict[str, Any] = {}
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on target machine
        report["import_ok"] = False
        report["import_error"] = f"{type(exc).__name__}: {exc}"
        return report

    report.update(
        {
            "import_ok": True,
            "version": torch.__version__,
            "compiled_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cudnn_version": torch.backends.cudnn.version(),
        }
    )

    if not torch.cuda.is_available():
        report["device_count"] = 0
        return report

    report["device_count"] = torch.cuda.device_count()
    devices: list[dict[str, Any]] = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
                "compute_capability": [properties.major, properties.minor],
                "multiprocessor_count": properties.multi_processor_count,
            }
        )
    report["devices"] = devices

    if cuda_smoke_test:
        try:
            device = torch.device("cuda:0")
            generator = torch.Generator(device=device).manual_seed(0)
            left = torch.randn((512, 512), generator=generator, device=device)
            right = torch.randn((512, 512), generator=generator, device=device)
            result = left @ right
            torch.cuda.synchronize(device)
            report["smoke_test"] = {
                "ok": bool(torch.isfinite(result).all().item()),
                "shape": list(result.shape),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            }
        except Exception as exc:  # pragma: no cover - depends on target machine
            report["smoke_test"] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path. The report is always printed to stdout.",
    )
    parser.add_argument(
        "--cuda-smoke-test",
        action="store_true",
        help="Run a small CUDA tensor allocation and matrix multiplication.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable_name": Path(sys.executable).name,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "torch": collect_torch(args.cuda_smoke_test),
    }
    serialized = json.dumps(report, indent=2, sort_keys=True)
    print(serialized)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

