# EXP-010: frozen BF16 paired replication

Status: `completed — development mechanism gate passed`

## Question

Does the seed-0 severe-threshold improvement from soft exceedance-area calibration
replicate across three paired seeds under the numerically stable BF16 protocol?

## Frozen design

- seeds 0, 1 and 2;
- official SimVP, 128×128, batch 8, 4,000 updates;
- Adam + OneCycleLR with max learning rate 0.005;
- baseline objective: MSE;
- proposed objective: MSE + `3e-4` tail-area loss;
- proposed thresholds 160/181/219 and raw temperature 10;
- first 200 validation batches;
- every accepted run must have zero fallback and zero skipped updates.

The accepted EXP-009 seed-1 baseline is reused. Five remaining jobs are run by a
fail-fast driver that validates each summary before continuing:

```bash
python scripts/run_bf16_paired_replications.py \
  --data-root /home/amon/zyx/dataset/sevir_data \
  --manifest artifacts/local/sevir_official_manifest.csv \
  --artifacts-root artifacts/local \
  --workers 2
```

Completed valid jobs are skipped, so the command can be safely restarted after
an external interruption. An invalid completed summary or failed subprocess stops
the sequence immediately.

After all jobs pass, generate the frozen paired summary:

```bash
python scripts/summarize_bf16_paired_replications.py \
  --artifacts-root artifacts/local \
  --output artifacts/local/exp010_bf16_paired_summary.json
```

The report contains per-seed paired changes, mean/sample standard deviation,
threshold-wise lead means, and the number of seeds that improve at every lead.

## Result and decision

The mechanism replicated. Mean mCSI increased from `0.30598±0.00283` to
`0.33577±0.00276`; every seed improved by 8.95%–10.37%. Lead-mean CSI gains
averaged +45.1%, +67.9% and +96.1% at thresholds 160, 181 and 219. For those
three thresholds, as well as threshold 133, the proposed objective improved all
144 seed-threshold-lead comparisons. Threshold 16 was effectively unchanged and
the worst seed changed by only -0.14%, within the pre-registered -2% guardrail.

The tradeoff is also reproducible. MSE worsened by 1.00% on average. Mean POD
rose by 58.7%, 84.4% and 125.4% at thresholds 160/181/219, while mean SUCR fell
by 22.3%, 29.9% and 16.8%. At 60 minutes, proposed severe forecast area still
represented only 12.3%, 7.1% and 7.2% of observed area. The loss therefore
mitigates severe-tail extinction but does not solve localization or calibration.

This is a passed 128×128 development gate, not a paper-level result. Native
resolution, complete evaluation, event-level uncertainty and another backbone
remain required. Before spending more GPU time, the method's novelty must be
audited against balanced losses, threshold-aware objectives, FACL, SimCast and
neighborhood/location-tolerant losses.
