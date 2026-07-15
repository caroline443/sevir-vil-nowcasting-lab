# EXP-010: frozen BF16 paired replication

Status: `planned`

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
