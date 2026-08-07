#!/usr/bin/env bash
set -Eeuo pipefail

# Publication-grade 128x128 ConvLSTM transfer experiment. The A4000 can fit a
# substantially larger batch than the original diagnostic batch of eight. A
# short SEA gate tries batch 48 first and falls back to 32 on failure.

REPO_ROOT="${REPO_ROOT:-$PWD}"
DATA_ROOT="${DATA_ROOT:-/home/amon/zyx/dataset/sevir_data}"
PYTHON="${PYTHON:-python}"
MANIFEST="${MANIFEST:-artifacts/local/sevir_official_manifest.csv}"
ARTIFACTS="${ARTIFACTS:-artifacts/local}"
WORKERS="${WORKERS:-4}"

cd "$REPO_ROOT"
mkdir -p "$ARTIFACTS/exp037_convlstm_128_fast_queue"
LOG="$ARTIFACTS/exp037_convlstm_128_fast_queue/run.log"
exec > >(tee -a "$LOG") 2>&1

probe_batch() {
  local batch_size=$1
  local learning_rate=$2
  local gate_dir="$ARTIFACTS/exp037_convlstm_gate_b${batch_size}_128"
  echo "Probing ConvLSTM batch=$batch_size: $(date -Is)"
  "$PYTHON" scripts/train_openstl_convlstm.py \
    --data-root "$DATA_ROOT" \
    --manifest "$MANIFEST" \
    --output-dir "$gate_dir" \
    --resolution 128 \
    --batch-size "$batch_size" \
    --epochs 1 \
    --max-train-batches 20 \
    --max-val-batches 5 \
    --learning-rate "$learning_rate" \
    --patch-size 4 \
    --sampling-schedule budget_linear \
    --sampling-end-probability 0 \
    --selection-metric mcsi_global \
    --tail-area-weight 0.0003 \
    --tail-temperature-raw 10 \
    --tail-thresholds 160 181 219 \
    --log-every 20 \
    --seed 2028 \
    --workers "$WORKERS"
}

if [[ -n "${CONVLSTM_BATCH_SIZE:-}" ]]; then
  BATCH_SIZE="$CONVLSTM_BATCH_SIZE"
  LEARNING_RATE="${CONVLSTM_LEARNING_RATE:-0.003}"
elif probe_batch 48 0.003; then
  BATCH_SIZE=48
  LEARNING_RATE=0.003
  echo "Selected ConvLSTM batch=48."
elif probe_batch 32 0.002; then
  BATCH_SIZE=32
  LEARNING_RATE=0.002
  echo "Batch 48 failed; selected ConvLSTM batch=32."
else
  echo "Both ConvLSTM batch probes failed." >&2
  exit 1
fi

common_args=(
  --data-root "$DATA_ROOT"
  --manifest "$MANIFEST"
  --resolution 128
  --batch-size "$BATCH_SIZE"
  --epochs 10
  --max-train-batches 0
  --max-val-batches 0
  --learning-rate "$LEARNING_RATE"
  --patch-size 4
  --sampling-schedule budget_linear
  --sampling-end-probability 0
  --selection-metric mcsi_global
  --log-every 500
  --workers "$WORKERS"
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
run_one "$ARTIFACTS/exp037_convlstm_mse_seed0_b${BATCH_SIZE}_128_e10" \
  --seed 0
run_one "$ARTIFACTS/exp037_convlstm_sea_seed0_b${BATCH_SIZE}_128_e10" \
  --seed 0 \
  --tail-area-weight 0.0003 \
  --tail-temperature-raw 10 \
  --tail-thresholds 160 181 219
run_one "$ARTIFACTS/exp037_convlstm_mse_seed1_b${BATCH_SIZE}_128_e10" \
  --seed 1
run_one "$ARTIFACTS/exp037_convlstm_sea_seed1_b${BATCH_SIZE}_128_e10" \
  --seed 1 \
  --tail-area-weight 0.0003 \
  --tail-temperature-raw 10 \
  --tail-thresholds 160 181 219

date -Is > "$ARTIFACTS/exp037_convlstm_128_fast_queue/SUCCESS"
echo "All four paper-facing ConvLSTM transfer runs completed."
