#!/usr/bin/env python3
"""Train the pinned OpenSTL ConvLSTM under the bounded SEVIR protocol."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from sevir_nowcasting.data import SevirVILWindowDataset
from sevir_nowcasting.losses import SoftExceedanceAreaLoss
from sevir_nowcasting.metrics import LeadTimeVILMetrics
from smoke_openstl_convlstm import (
    OPENSTL_CONVLSTM_SHA256,
    file_sha256,
    load_official_convlstm_module,
    patchify,
    unpatchify,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-train-batches", type=int, default=4000)
    parser.add_argument("--max-val-batches", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--tail-area-weight", type=float, default=0.0)
    parser.add_argument("--tail-temperature-raw", type=float, default=10.0)
    parser.add_argument(
        "--tail-thresholds", type=float, nargs="+", default=[160, 181, 219]
    )
    parser.add_argument(
        "--sampling-stop-iter",
        type=int,
        default=50000,
        help="official OpenSTL scheduled-sampling stop update",
    )
    parser.add_argument(
        "--sampling-changing-rate",
        type=float,
        default=0.00002,
        help="official per-update teacher-forcing probability decrement",
    )
    args = parser.parse_args()
    if min(
        args.batch_size,
        args.epochs,
        args.max_train_batches,
        args.max_val_batches,
        args.patch_size,
    ) < 1:
        parser.error("batch, epoch, batch limits and patch size must be positive")
    if args.resolution % args.patch_size:
        parser.error("resolution must be divisible by patch-size")
    if args.tail_area_weight < 0 or args.tail_temperature_raw <= 0:
        parser.error("tail weight must be non-negative and temperature positive")
    if args.sampling_stop_iter < 1 or args.sampling_changing_rate < 0:
        parser.error("scheduled-sampling arguments are invalid")
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(
    args: argparse.Namespace, split: str, *, shuffle: bool, drop_last: bool
) -> DataLoader:
    dataset = SevirVILWindowDataset(
        args.manifest,
        args.data_root,
        split=split,
        input_length=13,
        output_length=12,
        resolution=args.resolution,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )


def build_model(args: argparse.Namespace) -> nn.Module:
    upstream = load_official_convlstm_module()
    source_hash = file_sha256(upstream.__file__)
    if source_hash != OPENSTL_CONVLSTM_SHA256:
        raise RuntimeError(
            f"OpenSTL ConvLSTM hash mismatch: expected {OPENSTL_CONVLSTM_SHA256}, "
            f"got {source_hash}"
        )
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
    return upstream.ConvLSTM_Model(
        num_layers=4,
        num_hidden=[128, 128, 128, 128],
        configs=config,
    )


def scheduled_mask(
    batch_size: int,
    eta: float,
    device: torch.device,
) -> torch.Tensor:
    # OpenSTL samples one teacher-forcing decision per sample and future time,
    # then tiles it over space and channels. Broadcasting is equivalent and
    # avoids materializing a 32x32x16 mask.
    return (
        torch.rand(batch_size, 11, 1, 1, 1, device=device) < eta
    ).float()


def predict_future(
    model: nn.Module,
    inputs: torch.Tensor,
    future_frames: torch.Tensor,
    mask: torch.Tensor,
    patch_size: int,
) -> torch.Tensor:
    frames = patchify(torch.cat((inputs, future_frames), dim=1), patch_size)
    patched_predictions, _ = model(frames, mask, return_loss=False)
    return unpatchify(patched_predictions[:, 12:], patch_size)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[dict[str, object], int]:
    model.eval()
    metrics = LeadTimeVILMetrics(output_length=12)
    completed = 0
    for batch_index, batch in enumerate(loader):
        if batch_index >= args.max_val_batches:
            break
        inputs = batch["inputs"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        mask = torch.zeros(inputs.shape[0], 11, 1, 1, 1, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predictions = predict_future(
                model, inputs, torch.zeros_like(targets), mask, args.patch_size
            )
        metrics.update(predictions, targets)
        completed += 1
    return metrics.compute(), completed


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("this trainer requires CUDA with BF16 support")
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_loader = make_loader(args, "train", shuffle=True, drop_last=True)
    val_loader = make_loader(args, "val", shuffle=False, drop_last=False)
    steps_per_epoch = min(args.max_train_batches, len(train_loader))
    total_steps = args.epochs * steps_per_epoch
    device = torch.device("cuda:0")
    model = build_model(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        total_steps=total_steps,
        final_div_factor=1e4,
    )
    mse_criterion = nn.MSELoss()
    tail_criterion = SoftExceedanceAreaLoss(
        thresholds_raw=args.tail_thresholds,
        temperature_raw=args.tail_temperature_raw,
    ).to(device)

    torch.cuda.reset_peak_memory_stats(device)
    started_at = time.perf_counter()
    global_step = 0
    eta = 1.0
    objective_sum = 0.0
    mse_sum = 0.0
    tail_sum = 0.0
    train_log: list[dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_index, batch in enumerate(train_loader):
            if batch_index >= steps_per_epoch:
                break
            inputs = batch["inputs"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            if global_step < args.sampling_stop_iter:
                eta = max(0.0, eta - args.sampling_changing_rate)
            else:
                eta = 0.0
            mask = scheduled_mask(inputs.shape[0], eta, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                predictions = predict_future(
                    model, inputs, targets, mask, args.patch_size
                )
            mse_loss = mse_criterion(predictions.float(), targets.float())
            if args.tail_area_weight > 0:
                tail_loss = tail_criterion(predictions, targets)
            else:
                tail_loss = mse_loss.new_zeros(())
            objective = mse_loss + args.tail_area_weight * tail_loss
            if not torch.isfinite(objective):
                raise RuntimeError(
                    f"non-finite objective at epoch={epoch}, batch={batch_index}"
                )
            objective.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1
            objective_sum += float(objective.detach())
            mse_sum += float(mse_loss.detach())
            tail_sum += float(tail_loss.detach())
            if global_step == 1 or global_step % args.log_every == 0:
                entry = {
                    "epoch": epoch,
                    "step": global_step,
                    "objective": float(objective.detach()),
                    "mse_loss": float(mse_loss.detach()),
                    "tail_area_loss": float(tail_loss.detach()),
                    "teacher_forcing_probability": eta,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
                train_log.append(entry)
                print(json.dumps(entry, sort_keys=True), flush=True)

    validation, val_batches = evaluate(model, val_loader, device, args)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started_at
    summary = {
        "ok": True,
        "purpose": "bounded_convlstm_cross_backbone_gate",
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "amp_dtype": "bfloat16",
        "model": "OpenSTL ConvLSTM",
        "openstl_convlstm_sha256": OPENSTL_CONVLSTM_SHA256,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "wall_seconds": elapsed,
        "train_steps": global_step,
        "optimizer_updates": global_step,
        "train_samples": global_step * args.batch_size,
        "mean_train_objective": objective_sum / global_step,
        "mean_train_mse": mse_sum / global_step,
        "mean_train_tail_area_loss": tail_sum / global_step,
        "initial_teacher_forcing_probability": 1.0,
        "final_teacher_forcing_probability": eta,
        "scheduled_sampling_policy": "OpenSTL linear decrement",
        "validation_batches": val_batches,
        "validation_samples_upper_bound": val_batches * args.batch_size,
        "optimizer": "Adam",
        "scheduler": "OneCycleLR",
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "train_log.json").write_text(
        json.dumps(train_log, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    torch.save(
        {"model": model.state_dict(), "summary": summary},
        args.output_dir / "last.pt",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
