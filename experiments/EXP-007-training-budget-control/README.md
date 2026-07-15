# EXP-007: training-budget control for tail collapse

Status: ready for A4000 execution

## Question

Does intensity-selective tail collapse remain after increasing the diagnostic SimVP training budget from 1000 updates and 8000 samples to 4000 updates and 32,000 samples, close to one full pass over the 35,718-window training split?

This is the final confound check before solution experiments. It changes only the OneCycle schedule length and number of training updates. Model, optimizer, data, seed, validation subset, resizing and metrics remain identical to EXP-005.

## Hypotheses

- **Undertraining explanation:** high-threshold forecast counts and CSI recover substantially at 4000 steps, especially thresholds 181 and 219.
- **Structural objective bias:** overall MSE/CSI improve, but severe-tail extinction and persistence crossovers remain. This outcome authorizes a narrowly targeted solution experiment.

## Run

```bash
git switch main
git pull --ff-only
source .venv-openstl/bin/activate

PYTHONPATH=src:scripts python scripts/train_openstl_simvp.py \
  --manifest artifacts/local/sevir_official_manifest.csv \
  --data-root /home/amon/zyx/dataset/sevir_data \
  --output-dir artifacts/local/exp007_budget4000_128 \
  --resolution 128 \
  --batch-size 8 \
  --workers 2 \
  --epochs 1 \
  --max-train-batches 4000 \
  --max-val-batches 200
```

Expected runtime is approximately 35–45 minutes. Return:

- `artifacts/local/exp007_budget4000_128/summary.json`
- `artifacts/local/exp007_budget4000_128/metrics.json`
- `artifacts/local/exp007_budget4000_128/train_log.json`

The checkpoint remains local.

## Decision gate

- Tail collapse is considered persistent if threshold-181 or threshold-219 forecast area still vanishes at later leads, or persistence still beats SimVP across most high-threshold leads despite improved bulk MSE.
- If high-threshold behavior recovers strongly, extend baseline convergence analysis instead of proposing a new method.
- Do not change loss, sampler, architecture, batch size, seed or resolution in this experiment.
