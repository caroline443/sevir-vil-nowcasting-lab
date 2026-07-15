# EXP-006: persistence control for intensity-collapse diagnosis

Status: completed — intensity-selective collapse confirmed

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

The control completed on 2026-07-15 over the identical first 200 validation batches. Persistence evaluation took 32.3 seconds. See [`comparison.json`](comparison.json).

## Interpretation gate

- If persistence maintains nonzero high-threshold forecast counts while SimVP drives them to zero, intensity extinction is model-induced rather than an unavoidable metric consequence of storm motion.
- If SimVP beats persistence at low thresholds but loses badly at high thresholds, the candidate problem is intensity-selective, not simply weak forecasting skill.
- If persistence also has essentially zero high-threshold CSI at the same leads, location/evolution error may dominate; intensity-preserving changes alone are unlikely to solve the task.
- No model or loss modification is authorized until this comparison is complete.

## Outcome

SimVP is better overall but selectively worse on intense echoes:

- SimVP MSE is 0.00320 versus persistence's 0.00809, a 60.5% reduction;
- SimVP mean CSI is 0.2596 versus 0.2151, a 20.7% relative improvement;
- SimVP beats persistence at thresholds 16 and 74 at every lead time;
- persistence beats SimVP at threshold 181 and 219 at every lead time;
- persistence overtakes at threshold 160 from 15 minutes and at threshold 133 for 55–60 minutes;
- persistence retains 20,271 threshold-181 and 2,366 threshold-219 forecast pixels at all leads by construction, while SimVP predicts zero threshold-181 pixels from 30 minutes and zero threshold-219 pixels from 15 minutes.

This rules out the explanation that all high-threshold failure is merely a consequence of storm displacement. The confirmed candidate problem is **intensity-selective tail collapse under MSE optimization**: the learned model improves bulk-field accuracy while deleting rare severe cores. Because EXP-005 used only 1000 updates, EXP-007 must still rule out ordinary undertraining before testing a solution.
