# EXP-006: persistence control for intensity-collapse diagnosis

Status: ready for A4000 execution

## Question

Does last-observation persistence retain high-intensity forecast area better than the 1000-step SimVP pilot on the exact same validation subset?

Persistence repeats the final input frame at all 12 future lead times. Its CSI should decline as storms move and evolve, but its forecast intensity distribution cannot collapse with lead time because the predicted field is held fixed. This makes it a direct control for separating displacement error from model-induced extinction of severe cores.

## Run

This evaluation does not train a model and does not require the GPU:

```bash
git switch main
git pull --ff-only
source .venv-openstl/bin/activate

PYTHONPATH=src python scripts/evaluate_persistence.py \
  --manifest artifacts/local/sevir_official_manifest.csv \
  --data-root /home/amon/zyx/dataset/sevir_data \
  --output artifacts/local/exp006_persistence_128.json \
  --resolution 128 \
  --batch-size 8 \
  --workers 2 \
  --max-val-batches 200
```

Return `artifacts/local/exp006_persistence_128.json`.

## Interpretation gate

- If persistence maintains nonzero high-threshold forecast counts while SimVP drives them to zero, intensity extinction is model-induced rather than an unavoidable metric consequence of storm motion.
- If SimVP beats persistence at low thresholds but loses badly at high thresholds, the candidate problem is intensity-selective, not simply weak forecasting skill.
- If persistence also has essentially zero high-threshold CSI at the same leads, location/evolution error may dominate; intensity-preserving changes alone are unlikely to solve the task.
- No model or loss modification is authorized until this comparison is complete.
