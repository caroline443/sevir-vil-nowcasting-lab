# EXP-024: manifest freeze and native-resolution runtime measurement

Status: `completed`

Current decision: the manifest is frozen and the A4000 throughput supports a
three-epoch, validation-selected paired protocol with a manual gate after epoch
1. A 5090 is not yet authorized.

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

## Manifest result

The paper manifest passes every integrity check:

- SHA-256:
  `cd87c9df175cdf25c77d48da052e2650ffb78d722c34298c1a37e01a3a849630`;
- 56,937 windows from 18,979 unique events;
- train: 35,718 windows / 11,906 events;
- validation: 9,060 windows / 3,020 events;
- test: 12,159 windows / 4,053 events;
- zero cross-split events, duplicate windows or malformed start-frame sets.

The test split remains untouched.

## Throughput result

The 200-update tail run completed stably at 384 resolution:

- training: 0.443158 seconds per batch;
- validation: 0.244612 seconds per batch;
- peak allocated memory: 9,794,448,384 bytes;
- complete measured invocation: 101.92 seconds.

Projected A4000 costs are:

- full training epoch: 15,828.7 seconds / 4.397 hours;
- full validation: 2,216.2 seconds / 0.616 hours;
- one train-plus-validation epoch: 5.012 hours;
- one three-epoch model: 15.04 hours;
- one three-epoch baseline/tail pair: 30.07 hours;
- three paired seeds: 90.22 hours / 3.76 continuous days.

The 200-update metrics are discarded.

## Budget decision

Configure three epochs but stop each seed-0 run deliberately after epoch 1.
This makes the OneCycle schedule consistent across the partial and resumed
invocations. Continue to epoch 3 only if both baseline and tail are finite,
validation skill is non-degenerate and runtime matches the projection.

Three native-resolution epochs produce 107,154 optimizer updates at batch 1.
This is a compute-aware paired protocol, not a reproduction of a published
large-batch 50- or 100-epoch result. The limitation must be stated explicitly.

See `result-analysis.json` for exact values.
