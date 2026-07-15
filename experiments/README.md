# Experiment protocol

Every experiment receives an immutable identifier: `EXP-NNN-short-name`.

Each experiment directory must include:

- `README.md`: question, hypothesis, controls, commands and stop condition;
- `config.yaml` or an exact command line;
- `results.csv`: machine-readable metrics;
- optional plots containing no private paths or raw restricted data.

## Required metadata

- Git commit;
- dataset version and split manifest hash;
- input/output length and image resolution;
- seed;
- GPU model and peak allocated memory;
- total training time and inference latency;
- parameter count;
- package versions.

## Required forecast metrics

At minimum:

- MSE or MAE;
- CSI at the agreed SEVIR thresholds;
- CSI by lead time;
- POD and FAR at high thresholds;
- mean and standard deviation across three seeds for final comparisons.

Do not compare scores when the split, resolution, input/output length or threshold implementation differs.

## Experiment statuses

- `planned`: card exists, not executed;
- `running`: execution has begun;
- `completed`: outputs and interpretation recorded;
- `stopped`: stop condition met; negative result retained;
- `invalid`: execution or data error makes results unusable.

