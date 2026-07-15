# EXP-008: severe-tail area calibration loss

Status: `running — objective frozen; controlled training pending`

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
