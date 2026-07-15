# EXP-003: official OpenSTL SimVP native-resolution gate

Status: completed — memory passed, full-training runtime rejected

## Question

Can the official OpenSTL SEVIR SimVP configuration complete one full mixed-precision optimization step at SEVIR's native 384×384 resolution on the 16 GiB A4000, and what does its measured step time imply for the feasibility of full training?

This experiment changes only spatial resolution from EXP-002. It is a compute-feasibility measurement, not training and not an architecture experiment.

## Hypothesis

EXP-002 used approximately 1.06 GiB at 128×128. Spatial activations grow roughly with pixel count, while parameters and optimizer state remain fixed. A 384×384 step is therefore expected to remain within 16 GiB, but only the measurement can determine whether enough safety margin and throughput remain for training.

## Run

After merging this experiment, run from the repository root in the existing compatibility environment:

```bash
git switch main
git pull --ff-only
source .venv-openstl/bin/activate

PYTHONPATH=src python scripts/smoke_openstl_simvp.py \
  --manifest artifacts/local/sevir_official_manifest.csv \
  --data-root /home/amon/zyx/dataset/sevir_data \
  --output artifacts/local/exp003_openstl_384.json \
  --resolution 384 \
  --batch-size 1
```

Return `artifacts/local/exp003_openstl_384.json`. If the process fails, return the complete traceback and the output of `nvidia-smi`; do not retry with a changed architecture or resolution.

The A4000 run completed successfully on 2026-07-15. Peak allocated memory was approximately 9.04 GiB and the optimization step took 1.28 seconds. With 35,718 training windows at batch size 1, the single-step extrapolation is 45,801 seconds, or 12.72 hours per epoch. See [`result.json`](result.json).

## Success condition

- input and output shapes are `[1,13,1,384,384]` and `[1,12,1,384,384]`;
- one AMP forward/backward/optimizer step finishes with finite loss;
- peak allocated memory is below 13 GiB, leaving at least 3 GiB nominal device margin;
- step time and manifest sample count are recorded to produce a conservative single-step epoch-cost estimate. This is a screening estimate, not a throughput benchmark.

## Stop condition

- OOM or peak allocation at or above 13 GiB rejects ordinary native-resolution training on the A4000.
- A step time that extrapolates to an impractical epoch duration rejects a long native-resolution baseline even if it fits memory.
- Do not rent a 5090 based only on this single result. A later cost calculation must first show that paid hardware changes the research decision.

## Outcome

Native-resolution training fits the A4000 but fails the practical-runtime gate: approximately 10.6 days for 20 epochs and 53.0 days for 100 epochs if the single-step rate persists. This estimate requires a short steady-state check, but it is already sufficient to reject an ordinary long run. EXP-004 will reuse one sample for 100 steps to verify learnability and obtain a steadier compute-only rate before choosing the development protocol.
