#!/usr/bin/env bash
set -Eeuo pipefail

# Complete one-factor-at-a-time SEA ablations at 128x128.  The frozen full SEA
# run (three thresholds, per-lead, weight 3e-4, temperature 10) already exists
# in EXP-031 and is not wastefully repeated here.

REPO_ROOT="${REPO_ROOT:-$PWD}"
DATA_ROOT="${DATA_ROOT:-/home/amon/zyx/dataset/sevir_data}"
PYTHON="${PYTHON:-python}"
MANIFEST="${MANIFEST:-artifacts/local/sevir_official_manifest.csv}"
ARTIFACTS="${ARTIFACTS:-artifacts/local}"

cd "$REPO_ROOT"
mkdir -p "$ARTIFACTS/exp035_sea_ablations_128_queue"
LOG="$ARTIFACTS/exp035_sea_ablations_128_queue/run.log"
exec > >(tee -a "$LOG") 2>&1

reference="$ARTIFACTS/exp031_sea_seed0_128_e10/summary.json"
if [[ ! -f "$reference" ]] || \
    ! grep -q '"training_complete": true' "$reference"; then
  echo "Missing completed frozen SEA reference: $reference" >&2
  exit 1
fi

common_args=(
  --data-root "$DATA_ROOT"
  --manifest "$MANIFEST"
  --resolution 128
  --batch-size 8
  --epochs 10
  --max-train-batches 0
  --max-val-batches 0
  --learning-rate 0.005
  --selection-metric mcsi_global
  --tail-area-weight 0.0003
  --tail-temperature-raw 10
  --tail-thresholds 160 181 219
  --tail-temporal-mode per_lead
  --log-every 500
  --seed 0
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
  "$PYTHON" scripts/train_paper_simvp.py \
    "${common_args[@]}" \
    --output-dir "$output_dir" \
    "${resume_args[@]}" \
    "$@"
}

# Component ablations: multilevel thresholds and explicit per-lead matching.
run_one "$ARTIFACTS/exp035_sea_threshold181_seed0_128_e10" \
  --tail-thresholds 181
run_one "$ARTIFACTS/exp035_sea_sequence_mean_seed0_128_e10" \
  --tail-temporal-mode sequence_mean

# Weight sensitivity around the frozen gradient-probed coefficient 3e-4.
run_one "$ARTIFACTS/exp035_sea_w1e-4_seed0_128_e10" \
  --tail-area-weight 0.0001
run_one "$ARTIFACTS/exp035_sea_w1e-3_seed0_128_e10" \
  --tail-area-weight 0.001

# Soft-threshold sensitivity around the frozen 10-raw-VIL temperature.
run_one "$ARTIFACTS/exp035_sea_t5_seed0_128_e10" \
  --tail-temperature-raw 5
run_one "$ARTIFACTS/exp035_sea_t20_seed0_128_e10" \
  --tail-temperature-raw 20

date -Is > "$ARTIFACTS/exp035_sea_ablations_128_queue/SUCCESS"
echo "All six SEA ablation runs completed."
