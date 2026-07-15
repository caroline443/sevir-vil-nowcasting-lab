# EXP-003: official OpenSTL SimVP native-resolution gate

Status: ready for A4000 execution

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

## Success condition

- input and output shapes are `[1,13,1,384,384]` and `[1,12,1,384,384]`;
- one AMP forward/backward/optimizer step finishes with finite loss;
- peak allocated memory is below 13 GiB, leaving at least 3 GiB nominal device margin;
- step time and manifest sample count are recorded to produce a conservative single-step epoch-cost estimate. This is a screening estimate, not a throughput benchmark.

## Stop condition

- OOM or peak allocation at or above 13 GiB rejects ordinary native-resolution training on the A4000.
- A step time that extrapolates to an impractical epoch duration rejects a long native-resolution baseline even if it fits memory.
- Do not rent a 5090 based only on this single result. A later cost calculation must first show that paid hardware changes the research decision.
