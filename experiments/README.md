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

EXP-017 tests whether the fixed tail-area component transfers from SimVP-v1's
IncepU translator to SimVP-v2's gSTA translator without retuning. It begins
with a 100-update A4000 numerical and memory smoke test.

EXP-018 is a one-step official OpenSTL ConvLSTM resource gate. It decides
whether a genuinely recurrent cross-backbone baseline/tail pair is feasible on
the A4000 before a complete trainer is implemented.

EXP-019 runs the frozen seed-0 ConvLSTM baseline/tail pair with official linear
scheduled sampling and fully autoregressive validation. It is the first true
cross-backbone transfer gate. The gate passes: overall CSI improves by 23.54%
and MSE falls by 10.73%, with severe CSI gains at all leads. Long-lead severe
area overforecasting is retained as an explicit calibration limitation, and a
paired replication is required before the result becomes a central paper
claim.

EXP-020 repeats the frozen ConvLSTM pair at seed 1 before any coefficient
calibration. It determines whether both the large cross-backbone gain and the
long-lead overforecast limitation are reproducible. Both replicate: the
two-seed mean relative CSI gain is 25.17% and mean MSE change is -11.91%;
long-lead severe-area overforecasting is therefore retained as a stable
calibration boundary.

EXP-021 performs a read-only hard-versus-soft area audit on both frozen
ConvLSTM tail checkpoints. It tests whether the stable long-lead overforecast
comes from a mismatch between the temperature-10 training surrogate and hard
threshold evaluation before any second component is designed. Smoothing
contributes but does not explain the excess: temperature-10 soft areas also
overpredict, so temperature annealing alone is not authorized.

EXP-022 evaluates the two frozen ConvLSTM tail checkpoints under diagnostic
teacher-forcing probabilities from 1.0 to 0.0. It tests whether the remaining
overforecast is caused by calibrating tail supervision on mostly
teacher-forced trajectories and evaluating on free autoregressive rollouts.
The hypothesis is strongly supported by a sharp failure at probability zero.
Because the bounded trainer truncates a 50000-update schedule at 4000 updates,
this is treated as a protocol mismatch. The core loss method is frozen and the
project moves to a budget-aligned formal protocol rather than adding another
module.

EXP-023 is the first paper-protocol infrastructure gate. It runs the frozen
tail method with official SimVP at native 384 resolution, deliberately stops
after one short epoch, resumes to the second, and verifies global metrics,
validation selection and standalone checkpoint evaluation. Gate scores are
discarded. The gate passes at 9.79 GB peak training allocation; resumed and
standalone validation metrics match exactly.

EXP-024 freezes the exact manifest checksum and event/window counts, then runs
a 200-update/50-validation-batch native-resolution throughput measurement.
Those timings, not the very short EXP-023 gate, determine the A4000 versus
rented-5090 budget. The manifest passes; a three-epoch three-seed pair is
projected at 90.22 A4000 hours, so a manual seed-0 gate precedes continuation.

EXP-025 begins the publishable native-resolution seed-0 SimVP baseline/tail
pair. Both are configured for three epochs but stop after epoch 1. Accepted
checkpoints remain resumable parts of the final trajectory; test data stay
locked.

## Experiment statuses

- `planned`: card exists, not executed;
- `running`: execution has begun;
- `completed`: outputs and interpretation recorded;
- `stopped`: stop condition met; negative result retained;
- `invalid`: execution or data error makes results unusable.
