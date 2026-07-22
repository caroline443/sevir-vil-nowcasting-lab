# Method freeze and paper status

Date: 2026-07-22

## Working thesis

Deterministic radar nowcasting models trained primarily with pixelwise MSE
systematically attenuate rare severe VIL echoes and increasingly abstain from
predicting them at long lead. A displacement-tolerant, per-sample and per-lead
soft exceedance-area constraint can preserve severe-echo occurrence and extent
without requiring a new forecasting backbone.

## Frozen method

The method contribution is `SoftExceedanceAreaLoss`:

- raw VIL thresholds 160, 181 and 219;
- sigmoid temperature 10 raw VIL units;
- soft exceedance counts per sample, lead and threshold;
- `log1p` area compression;
- Smooth L1 matching to observed soft area;
- coefficient 0.0003, selected by a predeclared gradient-scale probe;
- combined with the backbone's standard MSE objective.

No attention block, Mamba replacement, SSIM mixture, temperature annealing or
second calibration loss is part of the frozen method.

## Evidence completed

- IncepU SimVP: three paired seeds; approximately 9.7% mean-CSI improvement
  with about 1% MSE cost.
- gSTA SimVP: one paired seed; 9.1% mean-CSI improvement with 1.2% MSE cost.
- OpenSTL ConvLSTM: two paired seeds; mean relative CSI improvement 25.17% and
  mean relative MSE change -11.91%.
- ConvLSTM severe CSI improves in all 72 seed-threshold-lead comparisons.
- Probability Matching and FACL controls are weaker under the same bounded
  development protocol.
- Spatial diagnostics show that most recovered severe pixels are near observed
  storms rather than remote hallucinations.
- Native-resolution full-coverage seed-0 epoch 1 passes the continuation gate:
  validation global mCSI rises from 0.31451 to 0.32463 (+3.22%), with the
  strongest relative CSI gains at thresholds 160/181/219
  (+18.55%/+32.93%/+63.43%). MSE and MAE worsen by 2.52% and 3.55%, and
  severe-threshold SUCR falls, so this is a recall/false-alarm tradeoff rather
  than uniform dominance.

## Diagnosed boundary

The bounded ConvLSTM runs end training at 92% teacher forcing but are evaluated
at zero. EXP-022 shows a sharp free-rollout failure when future truth is fully
removed. This is treated as a protocol mismatch caused by truncating a
50000-update scheduled-sampling schedule at 4000 updates, not as evidence that a
new model module is required.

## Claims currently allowed

- The method reduces severe-echo abstention under the frozen 128-resolution
  bounded protocol.
- The effect transfers across two SimVP translators and a recurrent ConvLSTM.
- The fixed loss coefficient transfers without backbone-specific retuning.
- The mechanism improves severe CSI much more strongly than low-threshold CSI.

## Claims not yet allowed

- SEVIR state of the art.
- Full-resolution superiority.
- Universal calibration across all architectures.
- Operational nowcasting readiness.
- Statistical superiority under a complete publication protocol.

## Native-resolution paper run in progress

The seed-0 baseline and method have each completed the first of three frozen
epochs over all 35,718 training windows and all 9,060 validation windows. Both
must resume to epoch 3 before their independently validation-selected `best.pt`
checkpoints can be compared. The test split remains untouched. See EXP-025.

## Remaining publication work

1. Finish the frozen native-resolution seed-0 pair and inspect convergence.
2. Decide the minimum replication count from the completed seed-0 effect and
   compute budget; do not tune on the test split.
3. Use training schedules consistent with each model and budget; recurrent
   schedules must reach low or zero teacher forcing.
4. Run the final selected cross-backbone control only after the principal pair.
5. Produce threshold-by-lead figures, qualitative cases, efficiency statistics
   and multi-seed summaries.
6. Write the paper around one coherent problem and one lightweight solution.

## Publication assessment

The current evidence supports continuing toward an applied Q2 journal, but the
native-resolution result is still one validation seed at one epoch. The
algorithmic change is lightweight, so acceptance depends on disciplined problem
diagnosis, fair controls, cross-backbone evidence and a credible full
publication protocol. A CCF-B claim is not currently supported.
