# EXP-011: probability-matching closest-loss control

Status: `running` — published-weight arm completed; protocol-matched weight pending

## Question

Does the current severe-tail area loss add anything beyond the 2025 published
Probability-Matching (PM) loss under the same frozen development protocol?

## Hypothesis

PM should improve intensity-distribution calibration over MSE. The current loss
may retain an advantage specifically at thresholds 160/181/219 because it
allocates its auxiliary gradient only to the severe survival tail rather than
all quantiles. This is a falsifiable comparison, not an assumed advantage.

## Frozen controls

- Official OpenSTL SimVP source and hash already pinned by the training script.
- SEVIR manifest: `artifacts/local/sevir_official_manifest.csv`.
- 13 input and 12 output frames, 5-minute interval.
- 128 × 128 development resolution.
- seed 0, batch size 8, 4000 updates, first 200 validation batches.
- Adam + OneCycleLR, max LR 0.005, BF16 AMP.
- PM weight 10, matching Cao et al. (2025).
- Compare with the already completed seed-0 EXP-010 MSE and tail-area arms.

## Command

```bash
python scripts/train_openstl_simvp.py \
  --data-root /home/amon/zyx/dataset/sevir_data \
  --manifest artifacts/local/sevir_official_manifest.csv \
  --output-dir artifacts/local/exp011_pm_w10_seed0_128 \
  --resolution 128 \
  --batch-size 8 \
  --epochs 1 \
  --max-train-batches 4000 \
  --max-val-batches 200 \
  --learning-rate 0.005 \
  --probability-matching-weight 10 \
  --amp-dtype bfloat16 \
  --seed 0 \
  --workers 2
```

Expected A4000 runtime is roughly 40–60 minutes. Sorting adds training overhead,
but no inference parameters or latency.

## Numerical validity checks

- `train_steps == optimizer_updates == 4000`;
- `amp_fp32_fallbacks == 0`;
- `skipped_optimizer_updates == 0`;
- all reported losses and validation metrics are finite.

## Decision metrics

Use paired differences against the existing seed-0 MSE and tail-area runs:

- CSI by lead at 16, 74, 133, 160, 181 and 219;
- mean CSI and MSE;
- POD and SUCR at 160, 181 and 219;
- forecast/observed exceedance-area ratio by lead;
- mean-VIL bias by lead.

## Stop condition

If PM matches or beats SoftExceedanceAreaLoss at the severe thresholds with no
larger MSE or SUCR penalty, the standalone tail-loss innovation is rejected.
The project then moves to the unresolved spatial-reliability problem with PM as
a required baseline.

## Published-weight result

The numerically valid `weight=10` arm did not pass the closest-loss gate:

- mean CSI 0.30728 versus 0.30886 for MSE and 0.33824 for tail area;
- validation MSE +52.27% versus MSE, compared with +0.65% for tail area;
- PM improved CSI at 160/181/219 over MSE, but degraded CSI at 16/74 and had a
  much larger SUCR penalty;
- tail area exceeded PM lead-mean CSI at all six thresholds.

This arm alone is not a fair reason to dismiss PM because its published weight
was selected under a different 10-minute temporal protocol. Run the same 10%
gradient-scale procedure used to select the tail-area weight:

```bash
python scripts/probe_probability_matching_scale.py \
  --data-root /home/amon/zyx/dataset/sevir_data \
  --manifest artifacts/local/sevir_official_manifest.csv \
  --checkpoint artifacts/local/exp010_bf16_baseline_seed0/last.pt \
  --output artifacts/local/exp011_pm_scale_probe.json \
  --resolution 128 \
  --batch-size 8 \
  --max-batches 8 \
  --amp-dtype bfloat16 \
  --workers 2
```

The returned median weight determines one final seed-0 PM run. No broad metric
sweep is allowed: the purpose is fair gradient-scale matching, not tuning on the
validation score.
