# EXP-021: ConvLSTM hard/soft area surrogate audit

Status: `planned`

## Problem

The fixed tail-area loss improves ConvLSTM CSI and MSE in two paired seeds, but
both tail models overpredict hard-threshold severe area at long lead. Training
matches sigmoid-smoothed exceedance area at temperature 10, whereas evaluation
counts hard exceedances. The stable limitation may therefore be caused by a
surrogate-to-metric calibration gap rather than by the idea of area matching
itself.

## Question

At temperature 10, are soft predicted-to-observed area ratios close to one
while hard ratios are substantially above one? Does sharpening the diagnostic
temperature reduce that disagreement?

## Method

Read the two frozen tail checkpoints under the same 200-batch autoregressive
validation protocol. For thresholds 160/181/219 and temperatures 2/5/10,
record:

- hard predicted and observed pixel counts;
- soft predicted-to-observed count ratios;
- hard-minus-soft ratio gaps;
- per-sample absolute log-area error.

No parameters are updated.

## Command

```bash
python scripts/audit_convlstm_hard_soft_area.py \
  --data-root /home/amon/zyx/dataset/sevir_data \
  --manifest artifacts/local/sevir_official_manifest.csv \
  --checkpoint \
    artifacts/local/exp019_convlstm_tail_seed0_128/last.pt \
    artifacts/local/exp020_convlstm_tail_seed1_128/last.pt \
  --output artifacts/local/exp021_convlstm_hard_soft_area.json \
  --resolution 128 \
  --batch-size 8 \
  --max-batches 200 \
  --thresholds-raw 160 181 219 \
  --temperatures-raw 2 5 10 \
  --workers 2
```

## Decision rule

- If temperature-10 soft ratios are materially closer to one than hard ratios
  at long lead in both seeds, the surrogate-gap hypothesis is supported.
  Authorize a predeclared sharpness-continuation experiment.
- If soft and hard ratios overpredict similarly, reject the surrogate-gap
  hypothesis. Treat recurrent overpersistence as a backbone-specific limitation
  or investigate autoregressive exposure instead.
- Do not introduce a second loss term before this gate is resolved.
