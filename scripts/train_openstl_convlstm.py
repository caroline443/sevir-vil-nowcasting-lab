#!/usr/bin/env python3
"""Train pinned OpenSTL ConvLSTM under the paper-facing SEVIR protocol."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

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
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=4000,
        help="zero uses the full train loader; positive values are resource gates",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=200,
        help="zero uses the full validation loader",
    )
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--tail-area-weight", type=float, default=0.0)
    parser.add_argument("--tail-temperature-raw", type=float, default=10.0)
    parser.add_argument(
        "--tail-temporal-mode",
        choices=("per_lead", "sequence_mean"),
        default="per_lead",
    )
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
    parser.add_argument(
        "--sampling-schedule",
        choices=("upstream", "budget_linear"),
        default="upstream",
        help=(
            "upstream preserves OpenSTL's fixed decrement; budget_linear "
            "reaches sampling-end-probability on the final configured update"
        ),
    )
    parser.add_argument(
        "--sampling-end-probability",
        type=float,
        default=0.0,
        help="final probability for budget_linear scheduled sampling",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("mcsi_global", "mse"),
        default="mcsi_global",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="resume from a last.pt produced by this script",
    )
    parser.add_argument(
        "--stop-after-epoch",
        type=int,
        default=0,
        help="zero runs all epochs; positive values deliberately stop early",
    )
    args = parser.parse_args()
    if min(
        args.batch_size,
        args.epochs,
        args.patch_size,
    ) < 1:
        parser.error("batch size, epochs and patch size must be positive")
    if args.max_train_batches < 0 or args.max_val_batches < 0:
        parser.error("batch limits must be non-negative")
    if args.stop_after_epoch < 0 or args.stop_after_epoch > args.epochs:
        parser.error("stop-after-epoch must be zero or no greater than epochs")
    if args.resolution % args.patch_size:
        parser.error("resolution must be divisible by patch-size")
    if args.tail_area_weight < 0 or args.tail_temperature_raw <= 0:
        parser.error("tail weight must be non-negative and temperature positive")
    if args.sampling_stop_iter < 1 or args.sampling_changing_rate < 0:
        parser.error("scheduled-sampling arguments are invalid")
    if not 0 <= args.sampling_end_probability <= 1:
        parser.error("sampling-end-probability must lie in [0, 1]")
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(
    args: argparse.Namespace,
    split: str,
    *,
    shuffle: bool,
    drop_last: bool,
    generator: torch.Generator | None = None,
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
        generator=generator,
    )


def loader_batch_limit(requested: int, available: int) -> int:
    if available < 1:
        raise ValueError("loader must contain at least one batch")
    return available if requested == 0 else min(requested, available)


def is_better(
    selection_metric: str,
    candidate: float,
    incumbent: float | None,
) -> bool:
    if incumbent is None:
        return True
    if selection_metric == "mcsi_global":
        return candidate > incumbent
    if selection_metric == "mse":
        return candidate < incumbent
    raise ValueError(f"unsupported selection metric: {selection_metric}")


def configuration_signature(
    args: argparse.Namespace,
    train_batches_per_epoch: int,
) -> dict[str, object]:
    return {
        "manifest": str(args.manifest),
        "data_root": str(args.data_root),
        "resolution": args.resolution,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "train_batches_per_epoch": train_batches_per_epoch,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "patch_size": args.patch_size,
        "tail_area_weight": args.tail_area_weight,
        "tail_thresholds": list(args.tail_thresholds),
        "tail_temperature_raw": args.tail_temperature_raw,
        "tail_temporal_mode": args.tail_temporal_mode,
        "sampling_schedule": args.sampling_schedule,
        "sampling_end_probability": args.sampling_end_probability,
        "sampling_stop_iter": args.sampling_stop_iter,
        "sampling_changing_rate": args.sampling_changing_rate,
        "selection_metric": args.selection_metric,
    }


def capture_rng_state(train_generator: torch.Generator) -> dict[str, object]:
    return {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
        "train_generator_state": train_generator.get_state(),
    }


def restore_rng_state(
    checkpoint: dict[str, Any],
    train_generator: torch.Generator,
) -> None:
    random.setstate(checkpoint["python_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
    train_generator.set_state(checkpoint["train_generator_state"])


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


def next_teacher_forcing_probability(
    *,
    schedule: str,
    completed_updates: int,
    total_steps: int,
    current_probability: float,
    changing_rate: float,
    stop_iter: int,
    end_probability: float,
) -> float:
    """Return the probability used for the next optimizer update."""
    if total_steps < 1 or completed_updates < 0:
        raise ValueError("total steps and completed updates are invalid")
    if schedule == "budget_linear":
        progress = min(1.0, (completed_updates + 1) / total_steps)
        return 1.0 + (end_probability - 1.0) * progress
    if schedule == "upstream":
        if completed_updates < stop_iter:
            return max(0.0, current_probability - changing_rate)
        return 0.0
    raise ValueError(f"unknown sampling schedule: {schedule}")


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
    *,
    max_batches: int,
    patch_size: int,
) -> tuple[dict[str, object], int]:
    model.eval()
    metrics = LeadTimeVILMetrics(output_length=12)
    completed = 0
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        inputs = batch["inputs"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        mask = torch.zeros(inputs.shape[0], 11, 1, 1, 1, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predictions = predict_future(
                model, inputs, torch.zeros_like(targets), mask, patch_size
            )
        metrics.update(predictions, targets)
        completed += 1
    return metrics.compute(), completed


def checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    global_step: int,
    teacher_forcing_probability: float,
    best_metric: float | None,
    best_epoch: int | None,
    history: list[dict[str, object]],
    signature: dict[str, object],
    train_generator: torch.Generator,
) -> dict[str, object]:
    return {
        "format_version": 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "teacher_forcing_probability": teacher_forcing_probability,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "history": history,
        "configuration_signature": signature,
        **capture_rng_state(train_generator),
    }


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("this trainer requires CUDA with BF16 support")
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    train_loader = make_loader(
        args,
        "train",
        shuffle=True,
        drop_last=True,
        generator=train_generator,
    )
    val_loader = make_loader(args, "val", shuffle=False, drop_last=False)
    steps_per_epoch = loader_batch_limit(
        args.max_train_batches, len(train_loader)
    )
    validation_batches = loader_batch_limit(
        args.max_val_batches, len(val_loader)
    )
    total_steps = args.epochs * steps_per_epoch
    signature = configuration_signature(args, steps_per_epoch)
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
        temporal_mode=args.tail_temporal_mode,
    ).to(device)

    start_epoch = 1
    global_step = 0
    eta = 1.0
    best_metric: float | None = None
    best_epoch: int | None = None
    history: list[dict[str, object]] = []
    if args.resume is not None:
        checkpoint = torch.load(
            args.resume, map_location="cpu", weights_only=False
        )
        if checkpoint.get("configuration_signature") != signature:
            raise RuntimeError(
                "resume configuration differs from checkpoint signature"
            )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        eta = float(checkpoint["teacher_forcing_probability"])
        best_metric = checkpoint["best_metric"]
        best_epoch = checkpoint["best_epoch"]
        history = list(checkpoint["history"])
        restore_rng_state(checkpoint, train_generator)
    end_epoch = (
        args.epochs
        if args.stop_after_epoch == 0
        else args.stop_after_epoch
    )
    if start_epoch > args.epochs:
        raise RuntimeError("checkpoint already completed the requested epochs")
    if start_epoch > end_epoch:
        raise RuntimeError(
            "resume checkpoint is already beyond stop-after-epoch"
        )

    torch.cuda.reset_peak_memory_stats(device)
    started_at = time.perf_counter()
    for epoch in range(start_epoch, end_epoch + 1):
        model.train()
        epoch_training_started = time.perf_counter()
        objective_sum = 0.0
        mse_sum = 0.0
        tail_sum = 0.0
        completed_train_batches = 0
        for batch_index, batch in enumerate(train_loader):
            if batch_index >= steps_per_epoch:
                break
            inputs = batch["inputs"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            eta = next_teacher_forcing_probability(
                schedule=args.sampling_schedule,
                completed_updates=global_step,
                total_steps=total_steps,
                current_probability=eta,
                changing_rate=args.sampling_changing_rate,
                stop_iter=args.sampling_stop_iter,
                end_probability=args.sampling_end_probability,
            )
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
            completed_train_batches += 1
            objective_sum += float(objective.detach())
            mse_sum += float(mse_loss.detach())
            tail_sum += float(tail_loss.detach())
            if global_step == 1 or global_step % args.log_every == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "step": global_step,
                            "objective": float(objective.detach()),
                            "mse_loss": float(mse_loss.detach()),
                            "tail_area_loss": float(tail_loss.detach()),
                            "teacher_forcing_probability": eta,
                            "learning_rate": optimizer.param_groups[0]["lr"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        torch.cuda.synchronize(device)
        training_wall_seconds = (
            time.perf_counter() - epoch_training_started
        )
        validation_started = time.perf_counter()
        validation, val_batches = evaluate(
            model,
            val_loader,
            device,
            max_batches=validation_batches,
            patch_size=args.patch_size,
        )
        torch.cuda.synchronize(device)
        validation_wall_seconds = time.perf_counter() - validation_started
        candidate = float(validation[args.selection_metric])
        improved = is_better(args.selection_metric, candidate, best_metric)
        if improved:
            best_metric = candidate
            best_epoch = epoch
        epoch_record = {
            "epoch": epoch,
            "global_step": global_step,
            "train_batches": completed_train_batches,
            "mean_train_objective": objective_sum / completed_train_batches,
            "mean_train_mse": mse_sum / completed_train_batches,
            "mean_train_tail_area_loss": tail_sum / completed_train_batches,
            "teacher_forcing_probability": eta,
            "training_wall_seconds": training_wall_seconds,
            "training_seconds_per_batch": (
                training_wall_seconds / completed_train_batches
            ),
            "validation_batches": val_batches,
            "validation_wall_seconds": validation_wall_seconds,
            "validation_seconds_per_batch": (
                validation_wall_seconds / val_batches
            ),
            "validation_mse": validation["mse"],
            "validation_mae": validation["mae"],
            "validation_mcsi_global": validation["mcsi_global"],
            "validation_mcsi_lead_avg": validation["mcsi_lead_avg"],
            "selection_metric": args.selection_metric,
            "selection_value": candidate,
            "is_best": improved,
        }
        history.append(epoch_record)
        (args.output_dir / f"validation_epoch_{epoch:03d}.json").write_text(
            json.dumps(validation, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        payload = checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step,
            teacher_forcing_probability=eta,
            best_metric=best_metric,
            best_epoch=best_epoch,
            history=history,
            signature=signature,
            train_generator=train_generator,
        )
        torch.save(payload, args.output_dir / "last.pt")
        if improved:
            torch.save(payload, args.output_dir / "best.pt")
            (args.output_dir / "best_validation_metrics.json").write_text(
                json.dumps(validation, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        (args.output_dir / "history.json").write_text(
            json.dumps(history, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(epoch_record, sort_keys=True), flush=True)

    torch.cuda.synchronize(device)
    summary = {
        "ok": True,
        "purpose": (
            "paper_protocol_training"
            if args.max_train_batches == 0 and args.max_val_batches == 0
            else "paper_trainer_resource_gate"
        ),
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "amp_dtype": "bfloat16",
        "model": "OpenSTL ConvLSTM",
        "openstl_convlstm_sha256": OPENSTL_CONVLSTM_SHA256,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "wall_seconds_this_invocation": time.perf_counter() - started_at,
        "completed_epochs": int(history[-1]["epoch"]),
        "training_complete": int(history[-1]["epoch"]) == args.epochs,
        "global_step": global_step,
        "train_samples_available": len(train_loader.dataset),
        "validation_samples_available": len(val_loader.dataset),
        "train_batches_per_epoch": steps_per_epoch,
        "validation_batches_per_evaluation": validation_batches,
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "selection_metric": args.selection_metric,
        "configuration_signature": signature,
        "initial_teacher_forcing_probability": 1.0,
        "final_teacher_forcing_probability": eta,
        "scheduled_sampling_policy": (
            "OpenSTL fixed decrement"
            if args.sampling_schedule == "upstream"
            else "budget-aligned linear decrement"
        ),
        "optimizer": "Adam",
        "scheduler": "OneCycleLR",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
