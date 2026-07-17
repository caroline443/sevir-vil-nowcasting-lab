# EXP-025: paper-protocol SimVP seed-0 pair

Status: `planned`

## Purpose

Begin the first publishable native-resolution baseline/tail pair. Both runs are
configured for three epochs with identical data order and OneCycle schedule,
but deliberately stop after epoch 1 for a manual stability and budget gate.

Unlike earlier gates, accepted epochs belong to the final training trajectory
and can be resumed; their validation scores are not discarded. Test data remain
locked.

## Frozen settings

- manifest SHA-256
  `cd87c9df175cdf25c77d48da052e2650ffb78d722c34298c1a37e01a3a849630`;
- native 384×384, 13 input and 12 output frames;
- official pinned OpenSTL SimVP IncepU;
- BF16, batch 1 and seed 0;
- Adam, max LR 0.005 and OneCycle over three complete epochs;
- complete 35,718-window training and 9,060-window validation splits;
- checkpoint selection by validation `mcsi_global`;
- baseline MSE versus frozen tail-area method;
- no test evaluation.

## Baseline epoch-1 command

```bash
python scripts/train_paper_simvp.py \
  --data-root /home/amon/zyx/dataset/sevir_data \
  --manifest artifacts/local/sevir_official_manifest.csv \
  --output-dir artifacts/local/exp025_simvp_baseline_seed0_384 \
  --resolution 384 \
  --batch-size 1 \
  --epochs 3 \
  --learning-rate 0.005 \
  --selection-metric mcsi_global \
  --stop-after-epoch 1 \
  --log-every 500 \
  --seed 0 \
  --workers 2
```

## Tail epoch-1 command

```bash
python scripts/train_paper_simvp.py \
  --data-root /home/amon/zyx/dataset/sevir_data \
  --manifest artifacts/local/sevir_official_manifest.csv \
  --output-dir artifacts/local/exp025_simvp_tail_seed0_384 \
  --resolution 384 \
  --batch-size 1 \
  --epochs 3 \
  --learning-rate 0.005 \
  --tail-area-weight 0.0003 \
  --tail-temperature-raw 10 \
  --tail-thresholds 160 181 219 \
  --selection-metric mcsi_global \
  --stop-after-epoch 1 \
  --log-every 500 \
  --seed 0 \
  --workers 2
```

Each run is projected at approximately 5.0 hours. Run sequentially.

## Epoch-1 continuation gate

Continue both runs only if:

- 35,718 updates complete with finite losses and no OOM;
- full validation completes;
- MSE, MAE and all global threshold metrics are finite;
- neither model is degenerate at moderate thresholds;
- wall time remains close enough to the 5.01-hour projection;
- baseline and tail histories both identify epoch 1 as a valid checkpoint.

Do not select continuation based only on whether the tail model already wins
after epoch 1. Both models share a three-epoch OneCycle trajectory and may still
be in the rising part of the schedule.

## Deferred resume commands

Resume commands are intentionally withheld until the paired epoch-1 gate is
reviewed. No test command is authorized.
