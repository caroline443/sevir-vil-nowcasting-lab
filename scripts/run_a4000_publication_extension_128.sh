#!/usr/bin/env bash
set -Eeuo pipefail

# One unattended A4000 queue: complete the recurrent cross-backbone transfer
# study first, then the full one-factor-at-a-time SEA ablation study.

REPO_ROOT="${REPO_ROOT:-$PWD}"
export REPO_ROOT

cd "$REPO_ROOT"
mkdir -p artifacts/local/exp036_a4000_publication_extension_128
LOG="artifacts/local/exp036_a4000_publication_extension_128/run.log"
exec > >(tee -a "$LOG") 2>&1

echo "ConvLSTM transfer queue started: $(date -Is)"
bash scripts/run_a4000_convlstm_128.sh

echo "SEA ablation queue started: $(date -Is)"
bash scripts/run_a4000_sea_ablations_128.sh

date -Is > artifacts/local/exp036_a4000_publication_extension_128/SUCCESS
echo "A4000 publication extension completed."
