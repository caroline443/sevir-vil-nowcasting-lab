# EXP-012: FACL closest-loss control

Status: `planned`

## Question

Does the tail-selective area loss retain an advantage over the published
Fourier Amplitude and Correlation Loss (FACL) under the frozen development
dataset, model, update budget and evaluation protocol?

## Implementation fidelity

The local implementation follows the official MIT-licensed FACL code:

- spatial orthonormal FFT over each forecast frame;
- FCL is one minus global complex Fourier correlation;
- FAL is MSE between Fourier amplitudes;
- random per-update selection, with FAL probability increasing linearly;
- final constant ratio `0.1`, matching the official SEVIR command;
- loss multiplied by `sqrt(H * W)`;
- sigmoid applied to model outputs for both training and evaluation;
- FACL replaces MSE instead of being added to it.

This first gate retains the project's Adam + OneCycleLR schedule so that the
only method-level changes are the published FACL objective and required output
activation. It is a bounded comparison, not a reproduction of the paper's
50-epoch AdamW/cosine result.

Primary source: <https://github.com/argenycw/FACL>

## Numerical smoke test

```bash
python scripts/train_openstl_simvp.py \
  --data-root /home/amon/zyx/dataset/sevir_data \
  --manifest artifacts/local/sevir_official_manifest.csv \
  --output-dir artifacts/local/exp012_facl_smoke100_seed0_128 \
  --resolution 128 \
  --batch-size 8 \
  --epochs 1 \
  --max-train-batches 100 \
  --max-val-batches 20 \
  --learning-rate 0.005 \
  --training-loss facl \
  --facl-constant-ratio 0.1 \
  --amp-dtype bfloat16 \
  --seed 0 \
  --workers 2
```

The smoke test must have 100 optimizer updates, no fallback/skips, finite
metrics, and nonzero FCL and FAL term counts. Do not launch the 4000-update arm
until these checks pass.

## Planned full gate

If the smoke test passes, repeat with `--max-train-batches 4000`,
`--max-val-batches 200`, and output directory
`artifacts/local/exp012_facl4000_seed0_128`.

## Decision rule

- If FACL dominates tail area at severe thresholds with no worse MSE/SUCR
  trade-off, the current component is not competitive enough to scale.
- If tail area remains stronger specifically at 160/181/219, it survives as a
  severe-tail component, while FACL remains the sharpness/distribution baseline.
- A combination experiment is not automatic; it requires evidence that their
  gradient directions and error corrections are complementary.

