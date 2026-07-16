# EXP-015: severe-trajectory distribution gate

Status: `planned`

## Question

Are long-lead persistent, initiating or growing severe sequences sufficiently
rare under uniform training sampling to explain the model's conservative
severe-event abstention?

This is a problem gate, not a method experiment. It must be completed before
implementing future-tail balanced sampling, a trajectory curriculum or a
specialized severe-evolution branch.

## Measurements

For raw VIL thresholds 160, 181 and 219, measure:

- fraction of samples active in the last input and at each future lead;
- sequence-level initiation, extinction and persistence to 60 minutes;
- growth, stability and decay among persistent endpoints;
- severe-area quantiles conditional on activity.

These labels describe sequence endpoints and do not track storm-object
identity. They are sufficient to gate sampling hypotheses but not to claim
physical storm initiation or survival.

## Command

Run the complete official training split; this reads data but does not train a
model or require GPU memory:

```bash
python scripts/diagnose_severe_trajectory_distribution.py \
  --data-root /home/amon/zyx/dataset/sevir_data \
  --manifest artifacts/local/sevir_official_manifest.csv \
  --output artifacts/local/exp015_train_severe_trajectory_distribution.json \
  --split train \
  --resolution 128 \
  --batch-size 16 \
  --max-batches 0 \
  --thresholds-raw 160 181 219 \
  --workers 2
```

## Decision rule

- Continue to a trajectory-balanced sampler only if persistent/growing or
  late-initiating severe sequences are rare enough that ordinary uniform
  minibatches frequently omit them.
- If these classes are common, reject sampling imbalance as the explanation;
  investigate representation/output calibration instead.
- Any sampler must preserve the official validation distribution and use
  importance-aware reporting so gains are not created by changing evaluation.
