# EXP-024: manifest freeze and native-resolution runtime measurement

Status: `planned`

## Purpose

Freeze the exact paper manifest and obtain a stable A4000 throughput estimate
before choosing epoch count or renting a 5090. This experiment does not produce
paper scores.

## Step 1: freeze manifest identity and integrity

```bash
python scripts/summarize_sevir_manifest.py \
  --manifest artifacts/local/sevir_official_manifest.csv \
  --output artifacts/local/sevir_paper_manifest_summary.json \
  --expected-start-frames 0 12 24
```

Require:

- `ok == true`;
- train, validation and test splits are all present;
- zero cross-split events;
- zero duplicate windows;
- every event has starts 0, 12 and 24;
- retain the SHA-256 and exact row/event counts.

## Step 2: stable native-resolution throughput measurement

```bash
python scripts/train_paper_simvp.py \
  --data-root /home/amon/zyx/dataset/sevir_data \
  --manifest artifacts/local/sevir_official_manifest.csv \
  --output-dir artifacts/local/exp024_simvp_tail_384_runtime \
  --resolution 384 \
  --batch-size 1 \
  --epochs 1 \
  --max-train-batches 200 \
  --max-val-batches 50 \
  --learning-rate 0.005 \
  --tail-area-weight 0.0003 \
  --tail-temperature-raw 10 \
  --tail-thresholds 160 181 219 \
  --selection-metric mcsi_global \
  --log-every 50 \
  --seed 0 \
  --workers 2
```

The 200-update score is discarded. Use
`training_seconds_per_batch`, `validation_seconds_per_batch`, available sample
counts and peak memory to estimate:

- one full training epoch;
- one complete validation pass;
- baseline/tail paired runtime per seed;
- three-seed A4000 runtime;
- the maximum rational 5090 rental budget.

## Decision rule

- Do not authorize a long run if the manifest integrity check fails.
- Do not infer throughput from the 20-update resume gate; use this stabilized
  200/50 measurement.
- Decide epoch count from a declared convergence/early-stopping plan, not from
  an arbitrary round number.
- Rent a 5090 only if the projected A4000 wall time materially delays the paper
  and a short 5090 benchmark demonstrates a favorable cost ratio.
