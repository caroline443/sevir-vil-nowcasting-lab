# EXP-008: severe-tail area calibration loss

Status: `running — seed 0 passed; seed 1/2 replication pending`

## Question

Can a position-tolerant marginal tail constraint reduce SimVP's lead-time-dependent
severe-core extinction without sacrificing its low/moderate-threshold spatial skill?

## Hypothesis

Adding a soft exceedance-area loss at raw VIL thresholds 160, 181 and 219 will
increase forecast severe-pixel counts and delay the lead time at which persistence
overtakes SimVP. Because the loss aggregates area per sample and lead, it does not
double-penalize a displaced core as pixelwise F1 does.

## Fixed controls

- identical official SimVP architecture, manifest, seed, optimizer and OneCycleLR;
- 128×128 development resolution, batch 8, 4000 updates, validation first 200 batches;
- MSE baseline: EXP-007;
- only changed factor: training objective;
- thresholds and sigmoid temperature are fixed before looking at EXP-008 metrics.

## Stage A: choose loss weight from gradient scale

Run from the repository root after pulling `main`:

```bash
python scripts/probe_tail_loss_scale.py \
  --data-root /home/amon/zyx/dataset/sevir_data \
  --manifest artifacts/local/sevir_official_manifest.csv \
  --checkpoint artifacts/local/exp007_budget4000_128/last.pt \
  --output artifacts/local/exp008_tail_scale_probe.json \
  --resolution 128 \
  --batch-size 8 \
  --max-batches 8
```

The first probe at temperature 2 was rejected. Across eight batches, tail
gradient norms ranged from `2.62e-16` to `2.70e-1`, and implied weights ranged
from `4.53e-5` to `1.09e10`. The nominal median weight was therefore not a stable
scale estimate. This is sigmoid saturation: after a severe core has fallen more
than a few VIL units below a threshold, the proposed loss cannot restore it.

Temperature 10 is the next pre-registered gradient-health check. It gives a
useful gradient over intensity errors of several tens of VIL units, while a
zero-VIL background remains far below even threshold 160. This is a loss
implementation check, not selection by forecast validation score:

```bash
python scripts/probe_tail_loss_scale.py \
  --data-root /home/amon/zyx/dataset/sevir_data \
  --manifest artifacts/local/sevir_official_manifest.csv \
  --checkpoint artifacts/local/exp007_budget4000_128/last.pt \
  --output artifacts/local/exp008_tail_scale_probe_t10.json \
  --resolution 128 \
  --batch-size 8 \
  --max-batches 8 \
  --tail-temperature-raw 10
```

The temperature-10 probe passed. Tail gradient norms ranged from `5.44e-4` to
`7.77e-2`, replacing the temperature-2 span of roughly 15 orders of magnitude
with about two. The nominal median weight was `3.216e-4`; gradient cosine had a
median of `-0.00197`, showing no systematic conflict with MSE. The experiment
therefore freezes a rounded weight of `3e-4`. Neither temperature nor weight may
be changed after observing the training result.

## Stage B: controlled training

The pre-registered controlled run is:

```bash
python scripts/train_openstl_simvp.py \
  --data-root /home/amon/zyx/dataset/sevir_data \
  --manifest artifacts/local/sevir_official_manifest.csv \
  --output-dir artifacts/local/exp008_tail_area_t10_w3e-4_128 \
  --resolution 128 \
  --batch-size 8 \
  --epochs 1 \
  --max-train-batches 4000 \
  --max-val-batches 200 \
  --learning-rate 0.005 \
  --tail-area-weight 0.0003 \
  --tail-thresholds 160 181 219 \
  --tail-temperature-raw 10 \
  --seed 0 \
  --workers 2
```

## Decision rule

Advance the mechanism only if, relative to EXP-007, it satisfies all three:

1. improves CSI or persistence-crossover lead time at threshold 181 or 219;
2. increases high-threshold forecast area toward observation rather than through
   uncontrolled false alarms (inspect POD and SUCR together);
3. degrades CSI at thresholds 16 and 74 by no more than 2% relative.

Failure is informative: it would show that marginal tail preservation alone is
insufficient and motivate a representation or spatial-localization mechanism.

## Seed-0 result

The pre-registered seed-0 run passed all three decision rules. Relative to
EXP-007, overall lead-averaged mCSI rose from `0.30886` to `0.33824` (+9.51%)
while MSE increased from `0.002575` to `0.002592` (+0.65%). Lead-mean CSI gains
were +40.6%, +62.2% and +107.2% at thresholds 160, 181 and 219. EXP-008 beat
persistence at every lead for all three severe thresholds; EXP-007 had crossed
below persistence at 45, 30 and 25 minutes respectively.

The gain is not free. Mean SUCR decreases as POD rises, and at 60 minutes the
forecast retains only 13.2%, 8.15% and 9.78% of observed threshold-160/181/219
area. Mean-field positive bias also grows from 1.75% to 2.88%. The method
therefore mitigates rather than solves severe-tail extinction, with a measurable
false-alarm/calibration tradeoff.

No mechanism parameter will be changed for seeds 1 and 2. Advancement beyond
128×128 requires the severe-threshold gains to reproduce across both seeds.

## Seed-1 numerical interruption

The first paired replication attempt produced a non-finite FP16 baseline loss at
batch 2136. The tail loss was disabled, so this is not evidence against EXP-008.
The trainer also advanced OneCycleLR after GradScaler had skipped an optimizer
step, which PyTorch explicitly warned was an invalid ordering.

The numerical policy is now fixed for both arms: retry a non-finite FP16 forward
once in FP32; fail only if the FP32 objective is also non-finite; and advance the
scheduler only when GradScaler actually performs an optimizer update. Summaries
record fallback and skipped-update counts. Seed 1 must be rerun as a pair before
seed 2. If frequent fallbacks or true FP32 divergence occurs, the 5e-3 protocol
is considered numerically unstable and all seed-0 scores must later be rerun
under a newly frozen stable protocol.
