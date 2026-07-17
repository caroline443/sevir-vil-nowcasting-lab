# EXP-023: paper SimVP 384 trainer gate

Status: `planned`

## Purpose

Validate the paper-facing native-resolution SimVP pipeline on the A4000 before
any long run. This gate checks worst-case tail-loss memory, global metrics,
validation selection, deliberate partial stopping, exact checkpoint resume and
standalone checkpoint evaluation. Its scores are discarded.

## Frozen gate configuration

- native 384×384;
- official OpenSTL SimVP IncepU;
- BF16, batch 1 and seed 0;
- frozen tail-area configuration;
- two configured epochs of ten updates each;
- five validation batches per epoch;
- epoch 1 stops deliberately, then epoch 2 resumes from `last.pt`;
- selection by `mcsi_global`.

## Step 1: deliberate partial run

```bash
python scripts/train_paper_simvp.py \
  --data-root /home/amon/zyx/dataset/sevir_data \
  --manifest artifacts/local/sevir_official_manifest.csv \
  --output-dir artifacts/local/exp023_paper_simvp_tail_384_gate \
  --resolution 384 \
  --batch-size 1 \
  --epochs 2 \
  --max-train-batches 10 \
  --max-val-batches 5 \
  --learning-rate 0.005 \
  --tail-area-weight 0.0003 \
  --tail-temperature-raw 10 \
  --tail-thresholds 160 181 219 \
  --selection-metric mcsi_global \
  --stop-after-epoch 1 \
  --log-every 5 \
  --seed 0 \
  --workers 2
```

Require `completed_epochs == 1`, `training_complete == false`, finite metrics,
and both `last.pt` and `best.pt`.

## Step 2: resume the frozen two-epoch configuration

```bash
python scripts/train_paper_simvp.py \
  --data-root /home/amon/zyx/dataset/sevir_data \
  --manifest artifacts/local/sevir_official_manifest.csv \
  --output-dir artifacts/local/exp023_paper_simvp_tail_384_gate \
  --resolution 384 \
  --batch-size 1 \
  --epochs 2 \
  --max-train-batches 10 \
  --max-val-batches 5 \
  --learning-rate 0.005 \
  --tail-area-weight 0.0003 \
  --tail-temperature-raw 10 \
  --tail-thresholds 160 181 219 \
  --selection-metric mcsi_global \
  --resume artifacts/local/exp023_paper_simvp_tail_384_gate/last.pt \
  --log-every 5 \
  --seed 0 \
  --workers 2
```

Require `completed_epochs == 2`, `training_complete == true`,
`global_step == 20`, two history records and finite global/lead metrics.

## Step 3: standalone validation evaluator

```bash
python scripts/evaluate_paper_simvp.py \
  --data-root /home/amon/zyx/dataset/sevir_data \
  --manifest artifacts/local/sevir_official_manifest.csv \
  --checkpoint artifacts/local/exp023_paper_simvp_tail_384_gate/best.pt \
  --output artifacts/local/exp023_paper_simvp_tail_384_gate/standalone_val.json \
  --split val \
  --batch-size 1 \
  --max-batches 5 \
  --log-every 5 \
  --workers 2
```

## Acceptance rule

- no non-finite loss or metric;
- peak allocated memory remains below the A4000 capacity with useful margin;
- partial and resumed summaries satisfy the requirements above;
- `best_validation_metrics.json` includes MSE, MAE, `mcsi_global`,
  `mcsi_lead_avg`, six global CSI/POD/SUCR values and lead-time arrays;
- standalone evaluation writes a checkpoint hash and the same metric schema;
- no test-split evaluation is performed.

Passing this gate authorizes runtime measurement and final-budget planning, not
an immediate multi-seed long run.
