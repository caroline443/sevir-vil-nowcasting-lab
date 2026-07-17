#!/usr/bin/env python3
"""Run one real-data BF16 step with the official OpenSTL ConvLSTM."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
from torch import nn
from torch.utils.data import DataLoader

from sevir_nowcasting.data import SevirVILWindowDataset


OPENSTL_COMMIT = "eecf8a3078f0a178dbc7b28723da20f94ce36985"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.steps < 1 or args.patch_size < 1:
        parser.error("batch-size, steps and patch-size must be positive")
    if args.resolution % args.patch_size:
        parser.error("resolution must be divisible by patch-size")
    return args


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


def load_official_convlstm_module() -> ModuleType:
    package_spec = importlib.util.find_spec("openstl")
    if package_spec is None or not package_spec.submodule_search_locations:
        raise RuntimeError("OpenSTL is not installed in the active environment")
    model_file = (
        Path(next(iter(package_spec.submodule_search_locations)))
        / "models"
        / "convlstm_model.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_pinned_openstl_convlstm_model", model_file
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load official ConvLSTM from {model_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def patchify(values: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Convert [B,T,C,H,W] to OpenSTL [B,T,H/p,W/p,p*p*C]."""
    batch, time_steps, channels, height, width = values.shape
    patched = values.reshape(
        batch,
        time_steps,
        channels,
        height // patch_size,
        patch_size,
        width // patch_size,
        patch_size,
    )
    return (
        patched.permute(0, 1, 3, 5, 4, 6, 2)
        .contiguous()
        .reshape(
            batch,
            time_steps,
            height // patch_size,
            width // patch_size,
            patch_size * patch_size * channels,
        )
    )


def unpatchify(values: torch.Tensor, patch_size: int, channels: int = 1) -> torch.Tensor:
    """Convert OpenSTL patched frames back to [B,T,C,H,W]."""
    batch, time_steps, height, width, _ = values.shape
    expanded = values.reshape(
        batch,
        time_steps,
        height,
        width,
        patch_size,
        patch_size,
        channels,
    )
    return (
        expanded.permute(0, 1, 6, 2, 4, 3, 5)
        .contiguous()
        .reshape(
            batch,
            time_steps,
            channels,
            height * patch_size,
            width * patch_size,
        )
    )


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this smoke test expects a CUDA-capable machine")
    dataset = SevirVILWindowDataset(
        args.manifest,
        args.data_root,
        split="train",
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
    device = torch.device("cuda:0")
    inputs = batch["inputs"].to(device, non_blocking=True)
    targets = batch["targets"].to(device, non_blocking=True)

    upstream = load_official_convlstm_module()
    source_hash = file_sha256(upstream.__file__)
    config = SimpleNamespace(
        in_shape=(13, 1, args.resolution, args.resolution),
        pre_seq_length=13,
        aft_seq_length=12,
        patch_size=args.patch_size,
        filter_size=5,
        stride=1,
        layer_norm=0,
        reverse_scheduled_sampling=0,
    )
    model = upstream.ConvLSTM_Model(
        num_layers=4,
        num_hidden=[128, 128, 128, 128],
        configs=config,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.MSELoss()
    zeros = torch.zeros_like(targets)
    full_frames = patchify(torch.cat((inputs, zeros), dim=1), args.patch_size)
    mask = torch.zeros(
        inputs.shape[0],
        11,
        args.resolution // args.patch_size,
        args.resolution // args.patch_size,
        args.patch_size * args.patch_size,
        device=device,
    )

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started_at = time.perf_counter()
    losses: list[float] = []
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            patched_predictions, _ = model(
                full_frames, mask, return_loss=False
            )
            predictions = unpatchify(
                patched_predictions[:, 12:], args.patch_size
            )
            loss = criterion(predictions.float(), targets.float())
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss: {float(loss.detach())}")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started_at

    result = {
        "ok": True,
        "purpose": "official_openstl_convlstm_a4000_smoke",
        "device": torch.cuda.get_device_name(device),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "openstl_version": package_version("OpenSTL"),
        "openstl_commit_expected": OPENSTL_COMMIT,
        "openstl_convlstm_sha256_observed": source_hash,
        "batch_size": args.batch_size,
        "resolution": args.resolution,
        "patch_size": args.patch_size,
        "steps": args.steps,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "total_seconds": elapsed,
        "step_seconds": elapsed / args.steps,
        "loss": losses[-1],
        "input_shape": list(inputs.shape),
        "target_shape": list(targets.shape),
        "patched_input_shape": list(full_frames.shape),
        "patched_output_shape": list(patched_predictions.shape),
        "output_shape": list(predictions.shape),
        "model_config": {
            "num_hidden": [128, 128, 128, 128],
            "filter_size": 5,
            "stride": 1,
            "layer_norm": 0,
            "pure_autoregressive_mask": True,
        },
        "caveat": (
            "This smoke uses a zero scheduled-sampling mask. It measures shape, "
            "memory and throughput only; it is not a training-protocol result."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
