# EXP-015: severe-trajectory distribution gate

Status: `completed`

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
  --log-every 100 \
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

## Result

The full 35,718-sample training split rejects sequence-level rarity as the
dominant explanation. At 60 minutes, 69.85%, 59.55% and 38.00% of samples
remain active at thresholds 160, 181 and 219. Persistence from the last input
to 60 minutes occurs in 66.29%, 55.57% and 33.53% of all samples. Persistent
growth is also not exceptionally rare: it occurs in 19.26%, 16.75% and 11.32%
of all samples.

The imbalance is instead spatial. Conditional on an active 60-minute target,
the median severe areas are 116, 58 and 14 pixels at the three thresholds,
equal to only 0.708%, 0.354% and 0.085% of a 128² frame.

## Decision

- Reject general trajectory-balanced sampling as component two. Uniform
  minibatches already see persistent and growing severe sequences frequently.
- Do not conflate storm-enriched SEVIR sequence frequency with severe-pixel
  balance inside each frame.
- Retain late initiation as a possible subgroup analysis, not the main method:
  only 3.56%--4.48% of samples are inactive in the last input but active at 60
  minutes.
- The next gate tests pixel-level tail-gradient saturation by lead time. A
  lead-adaptive soft-threshold continuation is authorized only if the fixed
  temperature loses useful upward gradient at long leads.

See `result-analysis.json` for the recorded statistics.
