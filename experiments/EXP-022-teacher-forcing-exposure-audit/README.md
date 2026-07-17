# EXP-022: ConvLSTM teacher-forcing exposure audit

Status: `planned`

## Problem

EXP-021 rejects a pure soft-versus-hard surrogate explanation for recurrent
long-lead overforecasting. The ConvLSTM tail models are trained with scheduled
sampling that still supplies the true previous frame with probability 0.92 at
the final update, but operational validation supplies no future truth. The tail
constraint may therefore be calibrated on mostly teacher-forced trajectories
and become overpersistent during a free autoregressive rollout.

## Question

For the two frozen tail checkpoints, does severe-area overforecasting emerge
monotonically as teacher-forcing probability decreases from 1.0 to 0.0?

## Method

Evaluate probabilities 1.0, 0.92, 0.5 and 0.0 on the same first 50 validation
batches for both checkpoints. Non-zero probabilities are counterfactual
diagnostics that use future truth and are not valid forecasts. Record MSE,
overall CSI, severe CSI, severe forecast-to-observed area and mean bias by lead.

The total number of checkpoint-probability batches equals the preceding
two-checkpoint 200-batch audit, so runtime should be of similar order.

## Command

```bash
python scripts/audit_convlstm_teacher_forcing.py \
  --data-root /home/amon/zyx/dataset/sevir_data \
  --manifest artifacts/local/sevir_official_manifest.csv \
  --checkpoint \
    artifacts/local/exp019_convlstm_tail_seed0_128/last.pt \
    artifacts/local/exp020_convlstm_tail_seed1_128/last.pt \
  --output artifacts/local/exp022_convlstm_teacher_forcing.json \
  --resolution 128 \
  --batch-size 8 \
  --max-batches 50 \
  --teacher-forcing-probabilities 1.0 0.92 0.5 0.0 \
  --mask-seed 2026 \
  --workers 2
```

## Decision rule

- Support the exposure hypothesis if both seeds are close to calibrated under
  high teacher forcing and severe-area ratios increase materially and
  consistently as probability approaches zero.
- If supported, authorize one small training-protocol experiment that aligns
  tail supervision with free rollouts. Do not add another loss term.
- Reject the hypothesis if overforecasting is already comparable at probability
  1.0 or if ratios do not respond consistently to teacher forcing.
- Regardless of outcome, probability-zero scores remain the only valid forecast
  metrics.
