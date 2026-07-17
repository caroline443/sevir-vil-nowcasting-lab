# EXP-020: ConvLSTM paired replication

Status: `planned`

## Question

Does the large seed-0 ConvLSTM transfer result in EXP-019 replicate under an
independent initialization and data order without changing the method
coefficient?

## Why replication comes before calibration

EXP-019 shows both a large severe-CSI gain and long-lead severe-area
overforecasting. Retuning the coefficient immediately would convert a
predeclared zero-retuning transfer test into post-hoc backbone-specific tuning.
This experiment first tests whether both observations are reproducible.

## Frozen protocol

Everything matches EXP-019 except seed 1:

- official pinned OpenSTL ConvLSTM;
- four 128-channel layers, filter size 5 and patch size 4;
- 13 input frames and 12 fully autoregressive validation frames;
- BF16, batch 8, 128 resolution;
- Adam, max LR 0.0005 and OneCycleLR;
- 4000 training updates and 200 validation batches;
- official linear scheduled sampling ending at probability 0.92;
- tail thresholds 160/181/219, temperature 10 and weight 0.0003;
- no coefficient retuning.

## Baseline command

```bash
python scripts/train_openstl_convlstm.py \
  --data-root /home/amon/zyx/dataset/sevir_data \
  --manifest artifacts/local/sevir_official_manifest.csv \
  --output-dir artifacts/local/exp020_convlstm_baseline_seed1_128 \
  --resolution 128 \
  --batch-size 8 \
  --epochs 1 \
  --max-train-batches 4000 \
  --max-val-batches 200 \
  --learning-rate 0.0005 \
  --seed 1 \
  --workers 2
```

## Tail command

```bash
python scripts/train_openstl_convlstm.py \
  --data-root /home/amon/zyx/dataset/sevir_data \
  --manifest artifacts/local/sevir_official_manifest.csv \
  --output-dir artifacts/local/exp020_convlstm_tail_seed1_128 \
  --resolution 128 \
  --batch-size 8 \
  --epochs 1 \
  --max-train-batches 4000 \
  --max-val-batches 200 \
  --learning-rate 0.0005 \
  --tail-area-weight 0.0003 \
  --tail-temperature-raw 10 \
  --tail-thresholds 160 181 219 \
  --seed 1 \
  --workers 2
```

## Decision rule

- Both runs must complete 4000 optimizer updates and end with identical
  teacher-forcing probability.
- Replication passes if overall CSI, severe-threshold CSI and severe POD improve
  in the tail run without worse aggregate MSE.
- Record SUCR and 60-minute forecast-to-observed area ratios whether or not the
  gate passes.
- If the gain replicates but long-lead area overforecast does not, treat the
  seed-0 overforecast magnitude as unstable.
- If both replicate, retain overforecasting as a stable limitation and consider
  a separately declared calibration study only after the core evidence is
  frozen.
