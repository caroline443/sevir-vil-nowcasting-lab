#!/usr/bin/env bash
set -Eeuo pipefail

# Six-run GRSL loss comparison at 128x128. Completed runs are skipped and
# interrupted runs resume from last.pt, so the command can be relaunched.

REPO_ROOT="${REPO_ROOT:-$PWD}"
DATA_ROOT="${DATA_ROOT:-/home/amon/zyx/dataset/sevir_data}"
PYTHON="${PYTHON:-python}"
MANIFEST="${MANIFEST:-artifacts/local/sevir_official_manifest.csv}"
ARTIFACTS="${ARTIFACTS:-artifacts/local}"

cd "$REPO_ROOT"
mkdir -p "$ARTIFACTS/exp031_complete_losses_128"
LOG="$ARTIFACTS/exp031_complete_losses_128/run.log"
exec > >(tee -a "$LOG") 2>&1

common_args=(
  --data-root "$DATA_ROOT"
  --manifest "$MANIFEST"
  --resolution 128
  --batch-size 8
  --epochs 10
  --learning-rate 0.005
  --selection-metric mcsi_global
  --log-every 500
  --workers 2
)

run_one() {
  output_dir=$1
  shift
  if [[ -f "$output_dir/summary.json" ]] && \
      grep -q '"training_complete": true' "$output_dir/summary.json"; then
    echo "Skipping completed run: $output_dir"
    return
  fi
  resume_args=()
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

# Main paired effect: two independent seeds.
run_one "$ARTIFACTS/exp031_mse_seed0_128_e10" \
  --seed 0
run_one "$ARTIFACTS/exp031_sea_seed0_128_e10" \
  --seed 0 \
  --tail-area-weight 0.0003 \
  --tail-temperature-raw 10 \
  --tail-thresholds 160 181 219
run_one "$ARTIFACTS/exp031_mse_seed1_128_e10" \
  --seed 1
run_one "$ARTIFACTS/exp031_sea_seed1_128_e10" \
  --seed 1 \
  --tail-area-weight 0.0003 \
  --tail-temperature-raw 10 \
  --tail-thresholds 160 181 219

# Closest published loss controls: one seed each.
run_one "$ARTIFACTS/exp031_pm_w10_seed0_128_e10" \
  --seed 0 \
  --probability-matching-weight 10
run_one "$ARTIFACTS/exp031_facl_seed0_128_e10" \
  --seed 0 \
  --training-loss facl \
  --facl-constant-ratio 0.1

date -Is > "$ARTIFACTS/exp031_complete_losses_128/SUCCESS"
echo "All six 128x128 loss-comparison runs completed."
