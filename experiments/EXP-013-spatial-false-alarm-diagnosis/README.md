# EXP-013: spatial false-alarm diagnosis

Status: `completed`

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

## Result

The result rejects remote hallucination as the dominant explanation at raw VIL
thresholds 160 and 181. Mean tolerant precision rises from 0.468 to 0.868 at
radius 2 for threshold 160 and from 0.392 to 0.813 for threshold 181. At radius
4 it reaches 0.946 and 0.903, respectively. Most predicted severe pixels are
therefore close to an observed severe region even when pixelwise SUCR counts
them as false alarms.

Threshold 219 is harder: mean tolerant precision rises from 0.231 to 0.623 at
radius 2 and 0.819 at radius 8. The remaining extreme-core errors include both
larger displacement and some remote error, especially at long lead times.

The dominant unresolved failure is underprediction rather than excessive
severe area. At 60 minutes, forecast/observed severe-pixel ratios are only
13.7%, 7.7% and 7.5% for thresholds 160, 181 and 219. Even allowing an eight
pixel tolerance, 60-minute recalls are 0.288, 0.171 and 0.130.

## Decision

- Reject a pure false-alarm or outside-envelope penalty as the next training
  component. It risks suppressing an already sparse severe forecast.
- Do not present the tail-area SUCR reduction as evidence of widespread
  hallucination; much of it is a pixelwise displacement penalty.
- Add neighborhood-tolerant verification (for example FSS at declared scales)
  to the paper protocol, while retaining pixelwise CSI/SUCR/POD.
- The next method gate must jointly preserve severe mass and improve its
  localization. A plain FSS loss is only a control because spatial FSS training
  losses already exist; it is not sufficient novelty by itself.

Machine-readable interpretation is in `result-analysis.json`.
