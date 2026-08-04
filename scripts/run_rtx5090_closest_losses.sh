#!/usr/bin/env bash
set -Eeuo pipefail

# Run publication-protocol closest-loss controls serially on the rented RTX 5090.
# The short resource gates must pass before either paid full run starts.

REPO_ROOT="${REPO_ROOT:-/root/autodl-tmp/sevir-vil-nowcasting-lab}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/sevir_data}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
MANIFEST="${MANIFEST:-artifacts/local/sevir_official_manifest.csv}"
ARTIFACTS="${ARTIFACTS:-artifacts/local}"
SHUTDOWN_AFTER="${SHUTDOWN_AFTER:-1}"

cd "$REPO_ROOT"
mkdir -p "$ARTIFACTS/rtx5090_closest_losses"
LOG="$ARTIFACTS/rtx5090_closest_losses/run.log"
exec > >(tee -a "$LOG") 2>&1

shutdown_on_exit() {
  status=$?
  date -Is
  echo "closest-loss queue exit status: $status"
  sync
  if [[ "$SHUTDOWN_AFTER" == "1" ]]; then
    /usr/bin/shutdown -h now
  fi
  exit "$status"
}
trap shutdown_on_exit EXIT

common_args=(
  --data-root "$DATA_ROOT"
  --manifest "$MANIFEST"
  --resolution 384
  --batch-size 2
  --learning-rate 0.005
  --selection-metric mcsi_global
  --log-every 500
  --seed 0
  --workers 4
)

echo "PM resource gate started: $(date -Is)"
"$PYTHON" scripts/train_paper_simvp.py \
  "${common_args[@]}" \
  --output-dir "$ARTIFACTS/exp029_pm_w10_gate_seed0_384_b2" \
  --epochs 1 \
  --max-train-batches 200 \
  --max-val-batches 50 \
  --probability-matching-weight 10

echo "FACL resource gate started: $(date -Is)"
"$PYTHON" scripts/train_paper_simvp.py \
  "${common_args[@]}" \
  --output-dir "$ARTIFACTS/exp030_facl_gate_seed0_384_b2" \
  --epochs 1 \
  --max-train-batches 200 \
  --max-val-batches 50 \
  --training-loss facl \
  --facl-constant-ratio 0.1

echo "PM formal run started: $(date -Is)"
"$PYTHON" scripts/train_paper_simvp.py \
  "${common_args[@]}" \
  --output-dir "$ARTIFACTS/exp029_pm_w10_seed0_384_b2_e10" \
  --epochs 10 \
  --probability-matching-weight 10

echo "FACL formal run started: $(date -Is)"
"$PYTHON" scripts/train_paper_simvp.py \
  "${common_args[@]}" \
  --output-dir "$ARTIFACTS/exp030_facl_seed0_384_b2_e10" \
  --epochs 10 \
  --training-loss facl \
  --facl-constant-ratio 0.1

date -Is > "$ARTIFACTS/rtx5090_closest_losses/SUCCESS"
