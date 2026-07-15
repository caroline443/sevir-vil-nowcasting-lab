# EXP-007: training-budget control for tail collapse

Status: completed — undertraining rejected as the sole explanation

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

The A4000 run completed on 2026-07-15: 4000 updates and 200 validation batches took 2192 seconds (36.5 minutes), with approximately 8.11 GiB peak allocation. See [`budget-comparison.json`](budget-comparison.json).

## Decision gate

- Tail collapse is considered persistent if threshold-181 or threshold-219 forecast area still vanishes at later leads, or persistence still beats SimVP across most high-threshold leads despite improved bulk MSE.
- If high-threshold behavior recovers strongly, extend baseline convergence analysis instead of proposing a new method.
- Do not change loss, sampler, architecture, batch size, seed or resolution in this experiment.

## Outcome

Additional training substantially improves the baseline but does not remove tail collapse:

- MSE improves from 0.003196 to 0.002575 (19.4% reduction);
- mean CSI improves from 0.2596 to 0.3089 (19.0% relative gain);
- threshold-219 skill recovers through 20 minutes, but forecast area falls to five pixels at 25 minutes and zero from 30 minutes, versus roughly 2000 observed pixels per lead;
- threshold-181 persistence crossover is delayed to 30 minutes, but SimVP still reaches zero CSI at 60 minutes;
- threshold-160 persistence crossover is delayed to 45 minutes;
- threshold-16 forecast area at 60 minutes grows to 6.95 million pixels versus 5.80 million observed pixels, a 19.8% excess;
- mean predicted VIL at 60 minutes is 1.75% above the target mean, so severe-tail loss cannot be described as global underprediction.

Training budget is therefore a modifier, not the root cause. The validated research problem is **lead-time-dependent severe-tail extinction coupled with weak-echo diffusion under deterministic MSE training**.
