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

EXP-008 is the first mechanism experiment. It tests soft severe-threshold area
calibration against the fixed EXP-007 diagnostic baseline and chooses its loss
weight from a frozen-checkpoint gradient-scale probe rather than a metric sweep.

EXP-011 is a mandatory novelty gate introduced after the literature audit. It
reproduces the 2025 Probability-Matching loss under the frozen 128² protocol
before any native-resolution scaling.

EXP-012 is the second mandatory closest-loss gate. It implements the official
NeurIPS 2024 FACL schedule and required sigmoid output, beginning with a
100-update numerical smoke test before any full diagnostic run.

EXP-013 diagnoses whether the remaining tail-area precision cost comes from
small spatial displacement or remote hallucination. It is read-only checkpoint
evaluation and gates the design of any second method component.

EXP-014 pairs the EXP-013 tail diagnosis with the seed-0 MSE baseline. It shows
that the baseline mostly abstains at severe thresholds and that most severe
pixels recovered by tail-area training are near observed storms. This cancels
the proposed pure false-alarm penalty and redirects the next problem gate to
long-lead severe-event rarity and temporal survival.

EXP-015 measures the train-split frequency of severe initiation, extinction,
persistence and area growth. It gates any trajectory-balanced sampling or
curriculum component before additional model training is authorized.

EXP-016 probes tail-area gradient support by lead and soft-threshold
temperature. It gates a possible lead-adaptive continuation component after
EXP-015 rejects general trajectory-balanced sampling.

## Experiment statuses

- `planned`: card exists, not executed;
- `running`: execution has begun;
- `completed`: outputs and interpretation recorded;
- `stopped`: stop condition met; negative result retained;
- `invalid`: execution or data error makes results unusable.
