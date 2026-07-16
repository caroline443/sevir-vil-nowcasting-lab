# EXP-017: gSTA transfer gate

Status: `running` (100-update smoke passed; 4000-update pair authorized)

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

## Smoke result

The gSTA smoke test passed all gates:

- 100/100 optimizer updates, zero skipped updates and zero FP32 fallbacks;
- 18,706,497 parameters;
- 9,917,962,240 bytes (9.92 GB decimal) peak allocated GPU memory;
- 38.68 seconds wall time;
- finite training and validation outputs.

The 100-update CSI values are not method evidence. They are intentionally
discarded as undertrained smoke-test metrics. The paired 4000-update gate is
authorized below.

## Authorized baseline command

```bash
python scripts/train_openstl_simvp.py \
  --data-root /home/amon/zyx/dataset/sevir_data \
  --manifest artifacts/local/sevir_official_manifest.csv \
  --output-dir artifacts/local/exp017_gsta_baseline_seed0_128 \
  --model-type gSTA \
  --resolution 128 \
  --batch-size 8 \
  --epochs 1 \
  --max-train-batches 4000 \
  --max-val-batches 200 \
  --learning-rate 0.005 \
  --amp-dtype bfloat16 \
  --seed 0 \
  --workers 2
```

## Authorized tail command

```bash
python scripts/train_openstl_simvp.py \
  --data-root /home/amon/zyx/dataset/sevir_data \
  --manifest artifacts/local/sevir_official_manifest.csv \
  --output-dir artifacts/local/exp017_gsta_tail_area_seed0_128 \
  --model-type gSTA \
  --resolution 128 \
  --batch-size 8 \
  --epochs 1 \
  --max-train-batches 4000 \
  --max-val-batches 200 \
  --learning-rate 0.005 \
  --tail-area-weight 0.0003 \
  --tail-temperature-raw 10 \
  --tail-thresholds 160 181 219 \
  --amp-dtype bfloat16 \
  --seed 0 \
  --workers 2
```

## Pair decision rule

Continue the applied-Q2 path only if the fixed tail component improves severe
CSI and long-lead area survival on gSTA without a materially larger MSE penalty.
Failure stops the generic-loss claim and returns the project to an IncepU-only
case study.
