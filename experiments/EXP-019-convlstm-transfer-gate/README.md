# EXP-019: ConvLSTM cross-backbone transfer gate

Status: `running` (trainer smoke passed; seed-0 pair authorized)

## Question

Does the fixed tail-area component transfer from recurrent-free SimVP variants
to the official OpenSTL ConvLSTM recurrent backbone without retuning?

## Frozen protocol

- official ConvLSTM source hash pinned after EXP-018;
- four 128-channel layers, filter size 5, patch size 4;
- official OpenSTL linear scheduled sampling: teacher-forcing probability starts
  at 1, decreases by 0.00002 per update and reaches 0.92 after 4000 updates;
- validation is fully autoregressive with a zero future-input mask;
- Adam, max LR 0.0005 and OneCycleLR;
- BF16, batch 8, seed 0, 4000 updates and 200 validation batches;
- tail thresholds 160/181/219, temperature 10 and weight 0.0003 without retuning.

## First command: trainer smoke

The model-only smoke passed, but the new scheduled-sampling trainer must complete
100 updates, validation and checkpoint writing before the pair below is
authorized.

```bash
python scripts/train_openstl_convlstm.py \
  --data-root /home/amon/zyx/dataset/sevir_data \
  --manifest artifacts/local/sevir_official_manifest.csv \
  --output-dir artifacts/local/exp019_convlstm_trainer_smoke100_seed0_128 \
  --resolution 128 \
  --batch-size 8 \
  --epochs 1 \
  --max-train-batches 100 \
  --max-val-batches 20 \
  --learning-rate 0.0005 \
  --seed 0 \
  --workers 2
```

Require 100 updates, finite metrics, a final teacher-forcing probability of
0.998 and successful output files. The 100-step scores are discarded.

## Trainer-smoke result

The trainer gate passed:

- 100/100 optimizer updates and finite train/validation outputs;
- final teacher-forcing probability `0.997999999999998`;
- pinned source hash matched;
- 2,310,032,896 bytes (2.31 GB decimal) peak allocated memory;
- 38.89 seconds wall time;
- summary, metrics and checkpoint writing completed.

The 100-update CSI values are discarded as undertrained smoke metrics. The
4000-update pair below is now authorized.

## Baseline command

Authorized after trainer-smoke acceptance:

```bash
python scripts/train_openstl_convlstm.py \
  --data-root /home/amon/zyx/dataset/sevir_data \
  --manifest artifacts/local/sevir_official_manifest.csv \
  --output-dir artifacts/local/exp019_convlstm_baseline_seed0_128 \
  --resolution 128 \
  --batch-size 8 \
  --epochs 1 \
  --max-train-batches 4000 \
  --max-val-batches 200 \
  --learning-rate 0.0005 \
  --seed 0 \
  --workers 2
```

## Tail command

Authorized after trainer-smoke acceptance:

```bash
python scripts/train_openstl_convlstm.py \
  --data-root /home/amon/zyx/dataset/sevir_data \
  --manifest artifacts/local/sevir_official_manifest.csv \
  --output-dir artifacts/local/exp019_convlstm_tail_seed0_128 \
  --resolution 128 \
  --batch-size 8 \
  --epochs 1 \
  --max-train-batches 4000 \
  --max-val-batches 200 \
  --learning-rate 0.0005 \
  --tail-area-weight 0.0003 \
  --tail-temperature-raw 10 \
  --tail-thresholds 160 181 219 \
  --seed 0 \
  --workers 2
```

## Decision rule

- Require numerical stability and identical final teacher-forcing probability.
- Pass only if severe CSI and area survival improve across most leads with a
  bounded MSE cost and without destroying low-threshold skill.
- Failure rejects broad cross-backbone generality, even though cross-translator
  SimVP transfer has passed.
