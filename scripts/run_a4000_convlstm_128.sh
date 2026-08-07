#!/usr/bin/env bash
set -Eeuo pipefail

# Publication-grade 128x128 ConvLSTM transfer experiment.  Every run validates
# each epoch, selects its own best checkpoint by global mCSI, reaches zero
# teacher forcing on the final update, and resumes safely after interruption.

REPO_ROOT="${REPO_ROOT:-$PWD}"
DATA_ROOT="${DATA_ROOT:-/home/amon/zyx/dataset/sevir_data}"
PYTHON="${PYTHON:-python}"
MANIFEST="${MANIFEST:-artifacts/local/sevir_official_manifest.csv}"
ARTIFACTS="${ARTIFACTS:-artifacts/local}"

cd "$REPO_ROOT"
mkdir -p "$ARTIFACTS/exp034_convlstm_128_queue"
LOG="$ARTIFACTS/exp034_convlstm_128_queue/run.log"
exec > >(tee -a "$LOG") 2>&1

common_args=(
  --data-root "$DATA_ROOT"
  --manifest "$MANIFEST"
  --resolution 128
  --batch-size 8
  --epochs 10
  --max-train-batches 0
  --max-val-batches 0
  --learning-rate 0.0005
  --patch-size 4
  --sampling-schedule budget_linear
  --sampling-end-probability 0
  --selection-metric mcsi_global
  --log-every 500
  --workers 2
)

run_one() {
  local output_dir=$1
  shift
  if [[ -f "$output_dir/summary.json" ]] && \
      grep -q '"training_complete": true' "$output_dir/summary.json"; then
    echo "Skipping completed run: $output_dir"
    return
  fi
  local resume_args=()
  if [[ -f "$output_dir/last.pt" ]]; then
    resume_args=(--resume "$output_dir/last.pt")
    echo "Resuming: $output_dir"
  else
    echo "Starting: $output_dir"
  fi
  "$PYTHON" scripts/train_openstl_convlstm.py \
    "${common_args[@]}" \
    --output-dir "$output_dir" \
    "${resume_args[@]}" \
    "$@"
}

# Paired seed 0, followed by an independent paired replication.
run_one "$ARTIFACTS/exp034_convlstm_mse_seed0_128_e10" \
  --seed 0
run_one "$ARTIFACTS/exp034_convlstm_sea_seed0_128_e10" \
  --seed 0 \
  --tail-area-weight 0.0003 \
  --tail-temperature-raw 10 \
  --tail-thresholds 160 181 219
run_one "$ARTIFACTS/exp034_convlstm_mse_seed1_128_e10" \
  --seed 1
run_one "$ARTIFACTS/exp034_convlstm_sea_seed1_128_e10" \
  --seed 1 \
  --tail-area-weight 0.0003 \
  --tail-temperature-raw 10 \
  --tail-thresholds 160 181 219

date -Is > "$ARTIFACTS/exp034_convlstm_128_queue/SUCCESS"
echo "All four paper-facing ConvLSTM transfer runs completed."
