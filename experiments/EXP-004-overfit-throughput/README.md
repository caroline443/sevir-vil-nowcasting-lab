# EXP-004: native-resolution one-sample overfit and steady-state rate

Status: completed — learnability passed, fixed-LR transient exposed

## Question

Can the pinned official SimVP reduce loss substantially when optimized repeatedly on one native-resolution SEVIR window, and does a 100-step compute-only measurement confirm that long 384×384 training is impractical?

Failure to overfit one sample would indicate an optimization, numerical, target-alignment or implementation problem that must be fixed before any baseline comparison. Success is only a pipeline sanity check; it is not evidence of generalization.

## Run

After merging this experiment, run from the repository root in the existing compatibility environment:

```bash
git switch main
git pull --ff-only
source .venv-openstl/bin/activate

PYTHONPATH=src python scripts/smoke_openstl_simvp.py \
  --manifest artifacts/local/sevir_official_manifest.csv \
  --data-root /home/amon/zyx/dataset/sevir_data \
  --output artifacts/local/exp004_overfit100_384.json \
  --resolution 384 \
  --batch-size 1 \
  --steps 100
```

The same first training window is reused for all 100 updates. Expected runtime is a few minutes. Return `artifacts/local/exp004_overfit100_384.json` and do not start another run.

The A4000 run completed successfully on 2026-07-15. Loss fell from 0.05663 to 0.01127, a reduction of 80.10%, and steady average step time was 0.390 seconds. Peak allocated memory was approximately 9.12 GiB. See [`result-summary.json`](result-summary.json).

## Success condition

- all 100 AMP updates finish without OOM or non-finite loss;
- final MSE is at most 20% of initial MSE, or below 0.005;
- the loss trace shows an overall downward trend rather than divergence;
- average step time and the corresponding epoch estimate are recorded.

## Stop condition

- non-finite or strongly increasing loss blocks all longer experiments;
- weak loss reduction blocks baseline training until target alignment and optimizer settings are diagnosed;
- even after success, do not launch a native-resolution epoch. The result will instead determine a cheaper development protocol for failure-mode discovery.

## Outcome

The official architecture and 13→12 target path can learn a native-resolution example, so longer work is not blocked by a fundamental tensor-alignment error. However, loss jumped to 2.406 during steps 3–5 before recovering. EXP-004 intentionally reused the compatibility probe's fixed AdamW learning rate; OpenSTL's actual defaults are Adam with zero weight decay and OneCycle scheduling. Therefore the spike is treated as an optimizer-protocol warning, not a model failure. EXP-005 begins problem diagnosis using Adam and OneCycle while remaining explicitly shorter than a paper-reproduction run.
