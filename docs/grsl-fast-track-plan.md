# GRSL fast-track plan

Target: IEEE Geoscience and Remote Sensing Letters (GRSL), five pages.

## Frozen contribution

The paper reports one focused contribution: deterministic pixelwise VIL
nowcasting progressively abstains from severe echoes, and a soft
exceedance-area constraint preserves severe-echo extent at long lead times
with a measurable recall/false-alarm tradeoff.

No attention, Mamba, SSIM, LLM component, second calibration objective or new
backbone is part of the GRSL contribution.

## Required work before submission

1. Complete the native 384 seed-1 baseline/method pair. **Completed.**
2. Confirm that the severe-threshold validation direction replicates.
   **Passed.**
3. Freeze both seeds' independently validation-selected checkpoints.
   **Frozen.**
4. Evaluate the test split exactly once for persistence and the two paired
   SimVP variants. **Completed.**
5. Save event-grouped sufficient statistics during the same test pass.
   **Completed.**
6. Compute paired event-bootstrap intervals conditional on the two trained
   seeds. **Completed, 10,000 replicates over 4,053 events.**
7. Produce one compact main table, one threshold/lead figure and one
   qualitative severe-event figure.
8. Write and audit a five-page IEEE manuscript.

Native seed 2, native cross-backbone training and large hyperparameter grids
are removed from the GRSL critical path. Existing bounded PM, FACL, gSTA and
ConvLSTM experiments may be reported compactly as development controls, with
their different protocol clearly labeled.

## Statistical claim boundary

The final paper may report the mean and sample standard deviation across the
two native seeds. Event bootstrap resamples complete storm events, retaining
all three windows and all leads for an event. Its confidence intervals
represent dataset-sampling uncertainty conditional on the trained seeds; two
seeds do not support a strong claim about training-randomness significance.

Primary endpoints are global mCSI, CSI at raw VIL 160/181/219, MSE and MAE.
POD and SUCR at severe thresholds must accompany CSI to disclose the
recall/false-alarm tradeoff.

## Test discipline

Seed 1 passed the frozen validation gate and the one-time test is authorized.
Test results must not change the loss weight, thresholds, temperature, epoch
count or selected checkpoints. Evaluation outputs and event-statistic sidecars
refuse overwrite.

## TGRS extension boundary

A later TGRS manuscript cannot be a longer version of the same experiment.
It must add a substantive new method and substantially expanded evidence, such
as spatial reliability control, multi-dataset validation or a new calibrated
forecast formulation. The GRSL paper and relationship to it must be disclosed.
