# EXP-017: gSTA transfer gate

Status: `planned`

## Question

Does the fixed SoftExceedanceAreaLoss mechanism transfer from SimVP-v1 IncepU
to the different gSTA temporal translator used by SimVP-v2 without retuning?

This is a cross-translator gate, not yet a claim of cross-backbone
generalization. It tests whether the observed gain is tied to one Inception
translator implementation.

## Frozen settings

- official train/validation manifest and 13→12 windows;
- 128² bounded protocol, BF16, batch 8 and seed 0;
- 4000 updates and 200 validation batches after smoke-test acceptance;
- raw thresholds 160/181/219, temperature 10 and weight 0.0003;
- no weight or temperature probe on gSTA.

## First command: 100-update baseline smoke test

```bash
python scripts/train_openstl_simvp.py \
  --data-root /home/amon/zyx/dataset/sevir_data \
  --manifest artifacts/local/sevir_official_manifest.csv \
  --output-dir artifacts/local/exp017_gsta_smoke100_seed0_128 \
  --model-type gSTA \
  --resolution 128 \
  --batch-size 8 \
  --epochs 1 \
  --max-train-batches 100 \
  --max-val-batches 20 \
  --learning-rate 0.005 \
  --amp-dtype bfloat16 \
  --seed 0 \
  --workers 2
```

## Smoke decision rule

- Require 100 optimizer updates, no non-finite loss, no FP32 fallback and no
  skipped update.
- Require peak memory below the A4000 limit with useful margin.
- If stable, authorize one 4000-update MSE baseline and its paired tail-area
  run using identical seed and data order.

## Pair decision rule

Continue the applied-Q2 path only if the fixed tail component improves severe
CSI and long-lead area survival on gSTA without a materially larger MSE penalty.
Failure stops the generic-loss claim and returns the project to an IncepU-only
case study.
