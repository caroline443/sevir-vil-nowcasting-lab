#!/usr/bin/env python3
"""Run a bounded official-SimVP pilot with lead-time SEVIR diagnostics."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

# Make repository-root execution independent of an editable package install.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
from torch import nn
from torch.utils.data import DataLoader

from sevir_nowcasting.data import SevirVILWindowDataset
from sevir_nowcasting.losses import ProbabilityMatchingLoss, SoftExceedanceAreaLoss
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
    parser.add_argument(
        "--tail-area-weight",
        type=float,
        default=0.0,
        help="weight of soft severe-threshold area loss; zero reproduces baseline",
    )
    parser.add_argument(
        "--tail-thresholds",
        type=float,
        nargs="+",
        default=[160.0, 181.0, 219.0],
        help="severe thresholds in raw SEVIR VIL units",
    )
    parser.add_argument(
        "--tail-temperature-raw",
        type=float,
        default=2.0,
        help="soft-threshold sigmoid temperature in raw VIL units",
    )
    parser.add_argument(
        "--probability-matching-weight",
        type=float,
        default=0.0,
        help="weight of the Cao et al. (2025) sorted-field PM constraint",
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("float16", "bfloat16"),
        default="float16",
        help="CUDA autocast dtype; bfloat16 has FP32-like exponent range",
    )
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.max_train_batches < 1 or args.max_val_batches < 1:
        parser.error("epochs and maximum batch counts must be positive")
    if args.tail_area_weight < 0:
        parser.error("tail-area-weight must be non-negative")
    if args.tail_temperature_raw <= 0:
        parser.error("tail-temperature-raw must be positive")
    if args.probability_matching_weight < 0:
        parser.error("probability-matching-weight must be non-negative")
    if args.tail_area_weight > 0 and args.probability_matching_weight > 0:
        parser.error(
            "closest-loss controls must enable either tail area or probability matching, not both"
        )
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


def compute_training_objective(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mse_criterion: nn.Module,
    tail_criterion: nn.Module,
    tail_area_weight: float,
    probability_matching_criterion: nn.Module,
    probability_matching_weight: float,
    *,
    autocast_enabled: bool,
    autocast_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the objective, optionally using an AMP model forward."""
    with torch.autocast(
        "cuda", dtype=autocast_dtype, enabled=autocast_enabled
    ):
        predictions = predict_12(model, inputs)
    mse_loss = mse_criterion(predictions.float(), targets.float())
    if tail_area_weight > 0:
        tail_loss = tail_criterion(predictions, targets)
    else:
        # Preserve the baseline objective and memory behavior when disabled.
        tail_loss = mse_loss.new_zeros(())
    if probability_matching_weight > 0:
        probability_matching_loss = probability_matching_criterion(
            predictions, targets
        )
    else:
        probability_matching_loss = mse_loss.new_zeros(())
    loss = (
        mse_loss
        + tail_area_weight * tail_loss
        + probability_matching_weight * probability_matching_loss
    )
    return loss, mse_loss, tail_loss, probability_matching_loss


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> tuple[dict[str, object], int]:
    model.eval()
    metrics = LeadTimeVILMetrics(output_length=12)
    completed = 0
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        inputs = batch["inputs"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
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
    amp_dtype = (
        torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    )
    if (
        not args.no_amp
        and amp_dtype == torch.bfloat16
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("this CUDA device/runtime does not support bfloat16 AMP")
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
    mse_criterion = nn.MSELoss()
    tail_criterion = SoftExceedanceAreaLoss(
        thresholds_raw=args.tail_thresholds,
        temperature_raw=args.tail_temperature_raw,
    ).to(device)
    probability_matching_criterion = ProbabilityMatchingLoss().to(device)
    use_amp = not args.no_amp
    use_grad_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)

    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    global_step = 0
    train_objective_sum = 0.0
    train_mse_sum = 0.0
    train_tail_sum = 0.0
    train_probability_matching_sum = 0.0
    amp_fp32_fallbacks = 0
    skipped_optimizer_updates = 0
    optimizer_updates = 0
    train_log: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_index, batch in enumerate(train_loader):
            if batch_index >= steps_per_epoch:
                break
            inputs = batch["inputs"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            (
                loss,
                mse_loss,
                tail_loss,
                probability_matching_loss,
            ) = compute_training_objective(
                model,
                inputs,
                targets,
                mse_criterion,
                tail_criterion,
                args.tail_area_weight,
                probability_matching_criterion,
                args.probability_matching_weight,
                autocast_enabled=use_amp,
                autocast_dtype=amp_dtype,
            )
            used_fp32_fallback = False
            if use_amp and not torch.isfinite(loss):
                # A finite FP32 retry distinguishes batch-specific FP16
                # activation overflow from genuine model divergence.
                (
                    loss,
                    mse_loss,
                    tail_loss,
                    probability_matching_loss,
                ) = compute_training_objective(
                    model,
                    inputs,
                    targets,
                    mse_criterion,
                    tail_criterion,
                    args.tail_area_weight,
                    probability_matching_criterion,
                    args.probability_matching_weight,
                    autocast_enabled=False,
                    autocast_dtype=amp_dtype,
                )
                used_fp32_fallback = True
                amp_fp32_fallbacks += 1
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite loss after FP32 retry at epoch={epoch}, "
                    f"batch={batch_index}: "
                    f"{float(loss.detach())}"
                )
            scale_before = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()
            optimizer_step_skipped = use_grad_scaler and scale_after < scale_before
            if optimizer_step_skipped:
                # GradScaler found non-finite gradients and did not call
                # optimizer.step(); advancing OneCycleLR here is incorrect.
                skipped_optimizer_updates += 1
            else:
                scheduler.step()
                optimizer_updates += 1

            global_step += 1
            loss_value = float(loss.detach())
            mse_value = float(mse_loss.detach())
            tail_value = float(tail_loss.detach())
            probability_matching_value = float(probability_matching_loss.detach())
            train_objective_sum += loss_value
            train_mse_sum += mse_value
            train_tail_sum += tail_value
            train_probability_matching_sum += probability_matching_value
            if global_step == 1 or global_step % args.log_every == 0:
                entry = {
                    "step": global_step,
                    "epoch": epoch,
                    "loss": loss_value,
                    "mse_loss": mse_value,
                    "tail_area_loss": tail_value,
                    "probability_matching_loss": probability_matching_value,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "used_fp32_fallback": used_fp32_fallback,
                    "optimizer_step_skipped": optimizer_step_skipped,
                }
                train_log.append(entry)
                print(json.dumps(entry, sort_keys=True))

    validation, val_batches = evaluate(
        model, val_loader, device, args.max_val_batches, use_amp, amp_dtype
    )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    summary = {
        "ok": True,
        "purpose": "bounded_diagnostic_pilot_not_full_reproduction",
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "amp_dtype": args.amp_dtype if use_amp else "disabled",
        "grad_scaler_enabled": use_grad_scaler,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "wall_seconds": elapsed,
        "train_steps": global_step,
        "optimizer_updates": optimizer_updates,
        "skipped_optimizer_updates": skipped_optimizer_updates,
        "amp_fp32_fallbacks": amp_fp32_fallbacks,
        "train_samples": global_step * args.batch_size,
        "mean_train_objective": train_objective_sum / global_step,
        "mean_train_mse": train_mse_sum / global_step,
        "mean_train_tail_area_loss": train_tail_sum / global_step,
        "mean_train_probability_matching_loss": (
            train_probability_matching_sum / global_step
        ),
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
