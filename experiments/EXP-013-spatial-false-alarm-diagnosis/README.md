# EXP-013: spatial false-alarm diagnosis

Status: `planned`

## Question

When tail-area training lowers pixelwise SUCR, are the additional severe
forecasts mostly close to the observed storm (small displacement), or are they
remote hallucinations?

This distinction must be measured before designing a spatial loss. Pixelwise
SUCR treats a one-pixel shift and a distant false storm identically, but they
require different solutions.

## Method

For thresholds 160, 181 and 219, dilate each observed severe mask by radii
0, 1, 2, 4 and 8 pixels at the 128² working resolution. Measure the fraction of
forecast severe pixels falling inside each observed envelope. Also compute the
symmetric tolerant recall using dilated forecast masks.

- Rapid precision recovery at radius 1–2 indicates near-miss displacement.
- Recovery only at radius 4–8 indicates larger location error.
- Persistently low precision at radius 8 indicates remote hallucination.

## Command

```bash
python scripts/diagnose_spatial_false_alarms.py \
  --data-root /home/amon/zyx/dataset/sevir_data \
  --manifest artifacts/local/sevir_official_manifest.csv \
  --checkpoint artifacts/local/exp010_bf16_tail_area_seed0/last.pt \
  --output artifacts/local/exp013_tail_area_spatial_diagnosis.json \
  --resolution 128 \
  --batch-size 8 \
  --max-batches 200 \
  --thresholds-raw 160 181 219 \
  --radii 0 1 2 4 8 \
  --amp-dtype bfloat16 \
  --workers 2
```

## Decision rule

Do not implement a spatial regularizer until this result is available.

- If errors are mainly near misses, use a lead-adaptive asymmetric tolerance
  envelope rather than pixelwise F1/BCE.
- If errors are remote, a dilation loss is insufficient; the next component
  must condition severe mass on storm objects, motion corridors or calibrated
  predictability.

