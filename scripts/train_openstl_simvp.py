#!/usr/bin/env python3
"""Run a bounded official-SimVP pilot with lead-time SEVIR diagnostics."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from sevir_nowcasting.data import SevirVILWindowDataset
from sevir_nowcasting.metrics import LeadTimeVILMetrics
from smoke_openstl_simvp import (
    OPENSTL_MODEL_SHA256,
    file_sha256,
    load_official_simvp_module,
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
    parser.add_argument("--max-train-batches", type=int, default=1000)
    parser.add_argument("--max-val-batches", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.max_train_batches < 1 or args.max_val_batches < 1:
        parser.error("epochs and maximum batch counts must be positive")
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
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


def build_model(resolution: int) -> nn.Module:
    upstream = load_official_simvp_module()
    actual_hash = file_sha256(upstream.__file__)
    if actual_hash != OPENSTL_MODEL_SHA256:
        raise RuntimeError(
            f"OpenSTL source hash mismatch: expected {OPENSTL_MODEL_SHA256}, "
            f"got {actual_hash}"
        )
    return upstream.SimVP_Model(
        in_shape=(13, 1, resolution, resolution),
        hid_S=64,
        hid_T=256,
        N_S=2,
        N_T=4,
        model_type="IncepU",
        spatio_kernel_enc=3,
        spatio_kernel_dec=3,
        drop_path=0.1,
    )


def predict_12(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    return model(inputs)[:, :12]


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
    use_amp: bool,
) -> tuple[dict[str, object], int]:
    model.eval()
    metrics = LeadTimeVILMetrics(output_length=12)
    completed = 0
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        inputs = batch["inputs"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            predictions = predict_12(model, inputs)
        metrics.update(predictions, targets)
        completed += 1
    return metrics.compute(), completed


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("EXP-005 expects a CUDA-capable machine")
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_loader = make_loader(args, "train", shuffle=True, drop_last=True)
    val_loader = make_loader(args, "val", shuffle=False, drop_last=False)
    steps_per_epoch = min(args.max_train_batches, len(train_loader))
    total_steps = args.epochs * steps_per_epoch

    device = torch.device("cuda:0")
    model = build_model(args.resolution).to(device)
    # OpenSTL defaults to Adam with zero weight decay. Its SEVIR config sets
    # max_lr=5e-3 and sched='onecycle'. The pilot compresses that schedule into
    # its bounded number of updates; it is not a full-score reproduction.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        total_steps=total_steps,
        final_div_factor=1e4,
    )
    criterion = nn.MSELoss()
    use_amp = not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    global_step = 0
    train_loss_sum = 0.0
    train_log: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_index, batch in enumerate(train_loader):
            if batch_index >= steps_per_epoch:
                break
            inputs = batch["inputs"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                predictions = predict_12(model, inputs)
                loss = criterion(predictions, targets)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite loss at epoch={epoch}, batch={batch_index}: "
                    f"{float(loss.detach())}"
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1
            loss_value = float(loss.detach())
            train_loss_sum += loss_value
            if global_step == 1 or global_step % args.log_every == 0:
                entry = {
                    "step": global_step,
                    "epoch": epoch,
                    "loss": loss_value,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
                train_log.append(entry)
                print(json.dumps(entry, sort_keys=True))

    validation, val_batches = evaluate(
        model, val_loader, device, args.max_val_batches, use_amp
    )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    summary = {
        "ok": True,
        "purpose": "bounded_diagnostic_pilot_not_full_reproduction",
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "wall_seconds": elapsed,
        "train_steps": global_step,
        "train_samples": global_step * args.batch_size,
        "mean_train_mse": train_loss_sum / global_step,
        "validation_batches": val_batches,
        "validation_samples_upper_bound": val_batches * args.batch_size,
        "scheduler": "OneCycleLR",
        "optimizer": "Adam",
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
