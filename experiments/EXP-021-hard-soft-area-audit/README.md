# EXP-021: ConvLSTM hard/soft area surrogate audit

Status: `completed`

Current decision: smoothing contributes to the hard-area excess, but the pure
surrogate-gap hypothesis is rejected. Temperature annealing alone is not
authorized.

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

## Result

Both frozen tail checkpoints were evaluated on all 200 validation batches. At
60 minutes, temperature-10 soft forecast-to-observed area ratios for
160/181/219 are `1.267/1.658/3.068` at seed 0 and
`1.405/1.883/3.321` at seed 1. These are lower than the official hard ratios,
so smoothing contributes to the hard-threshold excess.

However, the temperature-10 soft ratios themselves substantially exceed one.
The model is therefore not calibrated even under its training surrogate during
fully autoregressive validation. Diagnostic temperatures 5 and 2 move the soft
ratios toward the hard ratios rather than revealing hidden calibration.

The pure surrogate-gap hypothesis is rejected. A temperature continuation may
still change a newly trained model, but this audit does not justify it as the
next experiment. The next gate tests teacher-forcing exposure mismatch.

The first audit used a broadcast threshold tensor for hard counts. This differed
from the official scalar threshold comparison at quantized boundary values,
most visibly at threshold 219. The script is corrected; all hard ratios quoted
in the interpretation use the already frozen official metrics. Soft-count
results are unaffected.

See `result-analysis.json` for exact values.
