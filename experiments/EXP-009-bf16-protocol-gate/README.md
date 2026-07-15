# EXP-009: BF16 numerical protocol gate

Status: `planned`

## Question

Does BF16 autocast provide a stable and efficient training protocol for the
official SimVP seed that required FP32 fallback on 41.05% of FP16 batches?

## Fixed controls

Architecture, manifest, data order, seed, optimizer, OneCycleLR, batch size,
training budget and validation subset match the failed seed-1 FP16 baseline.
Only the autocast dtype changes from FP16 to BF16. Tail-area loss is disabled.

## Command

```bash
python scripts/train_openstl_simvp.py \
  --data-root /home/amon/zyx/dataset/sevir_data \
  --manifest artifacts/local/sevir_official_manifest.csv \
  --output-dir artifacts/local/exp009_bf16_protocol_gate_seed1 \
  --resolution 128 \
  --batch-size 8 \
  --epochs 1 \
  --max-train-batches 4000 \
  --max-val-batches 200 \
  --learning-rate 0.005 \
  --tail-area-weight 0 \
  --amp-dtype bfloat16 \
  --seed 1 \
  --workers 2
```

## Decision rule

Accept BF16 only if all conditions hold:

1. `train_steps = optimizer_updates = 4000`;
2. `amp_fp32_fallbacks = 0`;
3. `skipped_optimizer_updates = 0`;
4. runtime and memory remain practical on the A4000;
5. validation metrics are finite.

If the gate passes, all seed-0/1/2 baseline and proposed runs used for statistical
comparison must be rerun under BF16. Existing FP16 scores remain development
evidence only and cannot be mixed into the final mean and standard deviation.
