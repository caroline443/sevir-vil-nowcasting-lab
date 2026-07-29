#!/usr/bin/env bash
set -euo pipefail

data_root="/home/amon/zyx/dataset/sevir_data"
manifest="artifacts/local/sevir_official_manifest.csv"
output_root="artifacts/local/grsl_final_test"

checkpoints=(
  "artifacts/local/exp025_simvp_baseline_seed0_384/best.pt"
  "artifacts/local/exp025_simvp_tail_seed0_384/best.pt"
  "artifacts/local/exp026_simvp_baseline_seed1_384/best.pt"
  "artifacts/local/exp026_simvp_tail_seed1_384/best.pt"
)

for checkpoint in "${checkpoints[@]}"; do
  if [[ ! -f "${checkpoint}" ]]; then
    echo "missing frozen checkpoint: ${checkpoint}" >&2
    exit 1
  fi
done

mkdir -p "${output_root}"

run_evaluation() {
  local result_path="$1"
  local stats_path="$2"
  shift 2
  if [[ -f "${result_path}" && -f "${stats_path}" ]]; then
    echo "skipping completed evaluation: ${result_path}"
    return
  fi
  if [[ -e "${result_path}" || -e "${stats_path}" ]]; then
    echo "partial evaluation output requires manual inspection: ${result_path} ${stats_path}" >&2
    exit 1
  fi
  "$@"
}

run_evaluation \
  "${output_root}/persistence_test.json" \
  "${output_root}/persistence_test_events.npz" \
  python scripts/evaluate_persistence.py \
  --data-root "${data_root}" \
  --manifest "${manifest}" \
  --output "${output_root}/persistence_test.json" \
  --event-stats-output "${output_root}/persistence_test_events.npz" \
  --split test \
  --resolution 384 \
  --batch-size 4 \
  --workers 2 \
  --confirm-final-test

run_evaluation \
  "${output_root}/baseline_seed0_test.json" \
  "${output_root}/baseline_seed0_test_events.npz" \
  python scripts/evaluate_paper_simvp.py \
  --data-root "${data_root}" \
  --manifest "${manifest}" \
  --checkpoint "${checkpoints[0]}" \
  --output "${output_root}/baseline_seed0_test.json" \
  --event-stats-output "${output_root}/baseline_seed0_test_events.npz" \
  --split test \
  --batch-size 1 \
  --workers 2 \
  --log-every 500 \
  --confirm-final-test

run_evaluation \
  "${output_root}/tail_seed0_test.json" \
  "${output_root}/tail_seed0_test_events.npz" \
  python scripts/evaluate_paper_simvp.py \
  --data-root "${data_root}" \
  --manifest "${manifest}" \
  --checkpoint "${checkpoints[1]}" \
  --output "${output_root}/tail_seed0_test.json" \
  --event-stats-output "${output_root}/tail_seed0_test_events.npz" \
  --split test \
  --batch-size 1 \
  --workers 2 \
  --log-every 500 \
  --confirm-final-test

run_evaluation \
  "${output_root}/baseline_seed1_test.json" \
  "${output_root}/baseline_seed1_test_events.npz" \
  python scripts/evaluate_paper_simvp.py \
  --data-root "${data_root}" \
  --manifest "${manifest}" \
  --checkpoint "${checkpoints[2]}" \
  --output "${output_root}/baseline_seed1_test.json" \
  --event-stats-output "${output_root}/baseline_seed1_test_events.npz" \
  --split test \
  --batch-size 1 \
  --workers 2 \
  --log-every 500 \
  --confirm-final-test

run_evaluation \
  "${output_root}/tail_seed1_test.json" \
  "${output_root}/tail_seed1_test_events.npz" \
  python scripts/evaluate_paper_simvp.py \
  --data-root "${data_root}" \
  --manifest "${manifest}" \
  --checkpoint "${checkpoints[3]}" \
  --output "${output_root}/tail_seed1_test.json" \
  --event-stats-output "${output_root}/tail_seed1_test_events.npz" \
  --split test \
  --batch-size 1 \
  --workers 2 \
  --log-every 500 \
  --confirm-final-test

if [[ -f "${output_root}/paired_event_bootstrap.json" ]]; then
  echo "skipping completed bootstrap"
else
  python scripts/bootstrap_paired_events.py \
    --baseline-stats \
      "${output_root}/baseline_seed0_test_events.npz" \
      "${output_root}/baseline_seed1_test_events.npz" \
    --proposed-stats \
      "${output_root}/tail_seed0_test_events.npz" \
      "${output_root}/tail_seed1_test_events.npz" \
    --output "${output_root}/paired_event_bootstrap.json" \
    --repetitions 10000 \
    --seed 2027
fi

echo "GRSL final test and paired event bootstrap completed."
