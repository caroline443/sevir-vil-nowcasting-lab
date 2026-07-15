"""Train the compact SimVP baseline on a manifest-defined SEVIR split."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import SevirVILWindowDataset
from .metrics import VILMetrics
from .model import SimVP, parameter_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--input-length", type=int, default=13)
    parser.add_argument("--output-length", type=int, default=12)
    parser.add_argument("--hidden-spatial", type=int, default=32)
    parser.add_argument("--hidden-temporal", type=int, default=192)
    parser.add_argument("--temporal-depth", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(
    args: argparse.Namespace, split: str, *, shuffle: bool
) -> DataLoader:
    dataset = SevirVILWindowDataset(
        args.manifest,
        args.data_root,
        split=split,
        input_length=args.input_length,
        output_length=args.output_length,
        resolution=args.resolution,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None,
) -> dict[str, float]:
    model.eval()
    metrics = VILMetrics()
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        inputs = batch["inputs"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        metrics.update(model(inputs), targets)
    return metrics.compute()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("EXP-001 expects a CUDA-capable machine")
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0")
    use_amp = not args.no_amp
    train_loader = make_loader(args, "train", shuffle=True)
    val_loader = make_loader(args, "val", shuffle=False)
    model = SimVP(
        input_length=args.input_length,
        output_length=args.output_length,
        hidden_spatial=args.hidden_spatial,
        hidden_temporal=args.hidden_temporal,
        temporal_depth=args.temporal_depth,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = nn.MSELoss()
    torch.cuda.reset_peak_memory_stats(device)

    log_path = args.output_dir / "metrics.csv"
    start_time = time.perf_counter()
    with log_path.open("w", newline="", encoding="utf-8") as log_handle:
        writer = csv.DictWriter(
            log_handle,
            fieldnames=["epoch", "train_mse", "val_mse", "csi_mean", "csi_181", "csi_219"],
        )
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            train_batches = 0
            for batch_index, batch in enumerate(train_loader):
                if args.max_train_batches is not None and batch_index >= args.max_train_batches:
                    break
                inputs = batch["inputs"].to(device, non_blocking=True)
                targets = batch["targets"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    predictions = model(inputs)
                    loss = criterion(predictions, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += float(loss.detach())
                train_batches += 1

            validation = evaluate(
                model, val_loader, device, args.max_val_batches
            )
            row = {
                "epoch": epoch,
                "train_mse": total_loss / max(1, train_batches),
                "val_mse": validation["mse"],
                "csi_mean": validation["csi_mean"],
                "csi_181": validation["csi_181"],
                "csi_219": validation["csi_219"],
            }
            writer.writerow(row)
            log_handle.flush()
            print(json.dumps(row, sort_keys=True))

    summary = {
        "parameters": parameter_count(model),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "wall_seconds": time.perf_counter() - start_time,
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(device),
        "args": vars(args),
    }
    summary["args"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in summary["args"].items()
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    torch.save(
        {"model": model.state_dict(), "args": summary["args"]},
        args.output_dir / "last.pt",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
