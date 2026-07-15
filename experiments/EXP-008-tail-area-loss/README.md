# EXP-008: severe-tail area calibration loss

Status: `planned — gradient-scale probe first`

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

The reported weight makes the tail-loss gradient norm approximately 10% of the
MSE gradient norm on the frozen EXP-007 model. Send the resulting JSON for the
weight to be frozen before Stage B.

## Stage B: controlled training

Do not run this until Stage A fixes `<WEIGHT>`:

```bash
python scripts/train_openstl_simvp.py \
  --data-root /home/amon/zyx/dataset/sevir_data \
  --manifest artifacts/local/sevir_official_manifest.csv \
  --output-dir artifacts/local/exp008_tail_area_128 \
  --resolution 128 \
  --batch-size 8 \
  --epochs 1 \
  --max-train-batches 4000 \
  --max-val-batches 200 \
  --learning-rate 0.005 \
  --tail-area-weight <WEIGHT> \
  --tail-thresholds 160 181 219 \
  --tail-temperature-raw 2 \
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
