#!/usr/bin/env python3
"""Run one real-data AMP step with the pinned official OpenSTL SimVP model."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import platform
import time
from pathlib import Path
from types import ModuleType

import torch
from torch import nn
from torch.utils.data import DataLoader

from sevir_nowcasting.data import SevirVILWindowDataset


OPENSTL_COMMIT = "eecf8a3078f0a178dbc7b28723da20f94ce36985"
OPENSTL_MODEL_SHA256 = "1de78c0de74c89131470a3538498e301a9b2c2ca7c2598c9187576a7a60aa46c"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def load_official_simvp_module() -> ModuleType:
    """Load the upstream model file without importing OpenSTL's training stack.

    Importing ``openstl.models`` eagerly imports every OpenSTL model and the
    Lightning utilities.  The compatibility probe needs only the unmodified
    upstream SimVP model and modules, so loading this file directly avoids
    unrelated optional dependencies while preserving the official code.
    """

    package_spec = importlib.util.find_spec("openstl")
    if package_spec is None or not package_spec.submodule_search_locations:
        raise RuntimeError(
            "OpenSTL is not installed; follow experiments/EXP-002-openstl-simvp/README.md"
        )
    model_file = (
        Path(next(iter(package_spec.submodule_search_locations)))
        / "models"
        / "simvp_model.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_pinned_openstl_simvp_model", model_file
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load official SimVP model from {model_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("EXP-002 expects a CUDA-capable machine")

    dataset = SevirVILWindowDataset(
        args.manifest,
        args.data_root,
        split=args.split,
        input_length=13,
        output_length=12,
        resolution=args.resolution,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    batch = next(iter(loader))
    inputs = batch["inputs"]
    targets = batch["targets"]

    upstream = load_official_simvp_module()
    model_source_sha256 = file_sha256(upstream.__file__)
    if model_source_sha256 != OPENSTL_MODEL_SHA256:
        raise RuntimeError(
            "installed OpenSTL SimVP source does not match the pinned upstream commit: "
            f"expected {OPENSTL_MODEL_SHA256}, got {model_source_sha256}"
        )
    model = upstream.SimVP_Model(
        in_shape=(13, 1, args.resolution, args.resolution),
        hid_S=64,
        hid_T=256,
        N_S=2,
        N_T=4,
        model_type="IncepU",
        spatio_kernel_enc=3,
        spatio_kernel_dec=3,
        drop_path=0.1,
    )
    device = torch.device("cuda:0")
    model = model.to(device)
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
    criterion = nn.MSELoss()
    use_amp = not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
        raw_predictions = model(inputs)
        # This is exactly the aft_seq_length < pre_seq_length branch in
        # OpenSTL's SimVP method wrapper: predict 13 frames, retain the first 12.
        predictions = raw_predictions[:, :12]
        loss = criterion(predictions, targets)
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite loss: {float(loss.detach())}")
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    estimated_epoch_steps = math.ceil(len(dataset) / args.batch_size)

    result = {
        "ok": True,
        "amp": use_amp,
        "batch_size": args.batch_size,
        "dataset_samples": len(dataset),
        "dataset_split": args.split,
        "device": torch.cuda.get_device_name(device),
        "input_shape": list(inputs.shape),
        "target_shape": list(targets.shape),
        "raw_output_shape": list(raw_predictions.shape),
        "output_shape": list(predictions.shape),
        "loss": float(loss.detach()),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "resolution": args.resolution,
        "step_seconds": elapsed,
        "single_step_epoch_estimate_seconds": estimated_epoch_steps * elapsed,
        "estimated_epoch_steps": estimated_epoch_steps,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torchvision_version": package_version("torchvision"),
        "timm_version": package_version("timm"),
        "scipy_version": package_version("scipy"),
        "openstl_version": package_version("OpenSTL"),
        "openstl_commit": OPENSTL_COMMIT,
        "openstl_model_sha256": model_source_sha256,
        "model_config": {
            "model_type": "IncepU",
            "hid_S": 64,
            "hid_T": 256,
            "N_S": 2,
            "N_T": 4,
            "spatio_kernel_enc": 3,
            "spatio_kernel_dec": 3,
            "drop_path": 0.1,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
