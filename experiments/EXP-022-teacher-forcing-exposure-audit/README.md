# EXP-022: ConvLSTM teacher-forcing exposure audit

Status: `completed`

Current decision: free-rollout exposure mismatch is strongly supported, but it
is treated as a shortened-protocol defect rather than a new algorithmic
contribution. The core method is frozen.

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

## Result

Both frozen tail checkpoints were evaluated on the same 50 validation batches.
At teacher-forcing probabilities 1.0, 0.92 and 0.5, the models underpredict
60-minute severe area. At probability zero, severe area and mean bias jump
sharply:

- seed 0, threshold 219: area ratio `0.549` at probability 0.5 versus `5.867`
  at probability 0;
- seed 1, threshold 219: `0.611` versus `6.197`;
- seed 0 MSE: `0.001277` versus `0.003721`;
- seed 1 MSE: `0.001272` versus `0.003781`.

The transition is not smoothly monotonic across all four probabilities.
Instead, a distinct free-rollout failure appears when all future truth is
removed. This still strongly supports exposure mismatch: both models were
trained near probability 0.92 and are well behaved under teacher-forced and
partially teacher-forced trajectories, but not under the fully autoregressive
trajectory used in deployment.

## Final interpretation

The bounded 4000-update trainer retained the upstream decrement of 0.00002 per
update, a schedule designed to continue toward a 50000-update stop. It therefore
ended at probability 0.92. The resulting free-rollout failure should not be
promoted as a second method contribution without first running a
budget-aligned schedule.

The paper method is frozen at SoftExceedanceAreaLoss. Formal recurrent
experiments must use a declared training budget and scheduled-sampling schedule
that reaches a low or zero teacher-forcing probability before evaluation.
Probability-zero metrics remain the only valid nowcast results.

See `result-analysis.json` for exact values.
