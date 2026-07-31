# GRSL manuscript outline

## Working title

**A Soft Exceedance-Area Constraint for Severe-Echo-Preserving Radar
Nowcasting**

Alternative: **Reducing Severe-Echo Abstention in Deterministic Radar
Nowcasting**

## Draft abstract

Pixelwise regression objectives can produce accurate average radar forecasts
while progressively suppressing rare severe echoes. We diagnose this behavior
in deterministic vertically integrated liquid (VIL) nowcasting and introduce a
soft exceedance-area constraint that matches the log-compressed spatial extent
above severe VIL thresholds for every sample and forecast lead. The constraint
is differentiable, displacement tolerant, and adds no inference-time
parameters. We evaluate it with an official SimVP implementation on the
event-disjoint SEVIR test set at native 384x384 resolution using two paired
training seeds. Under the literature-standard lead-averaged reduction, the
proposed objective increases mean critical success index from 0.3889 to
0.4116. The global-count reduction increases from 0.3943 to 0.4119. CSI at raw
VIL thresholds 160, 181, and 219
improves by 11.0%, 18.9%, and 19.1%, respectively, while MSE and MAE decrease
by 1.5% and 2.6%. Ten thousand paired storm-event bootstrap replicates place
the global mCSI gain in [0.0172, 0.0181]. The gain is driven by higher severe
echo detection and therefore reduces success ratio, exposing a clear
recall--false-alarm tradeoff. These results show that explicitly preserving
severe exceedance extent can reduce long-lead echo abstention without changing
the forecasting backbone.

This abstract is a working draft only. It must not be submitted until the
validation-only convergence continuation and strong-baseline audit described
in `sevir-result-benchmark-audit.md` are complete.

## Contributions

1. Diagnose severe-echo abstention as an intensity- and lead-dependent failure
   hidden by aggregate pixel error.
2. Introduce a lightweight per-sample, per-lead soft exceedance-area constraint
   with no inference-time cost.
3. Demonstrate replicated native-resolution gains with event-level uncertainty
   analysis and explicitly quantify the POD/SUCR tradeoff.

## Five-page allocation

- Page 1: motivation, problem definition, contributions.
- Page 2: related work and method equations.
- Page 3: protocol and main table.
- Page 4: threshold/lead figure, controls and qualitative case.
- Page 5: tradeoff analysis, limitations and conclusion.

## Main table

Rows: persistence, SimVP MSE, SimVP plus soft exceedance-area loss.

Columns: MSE, MAE, global mCSI, CSI@133, CSI@160, CSI@181, CSI@219. Report
two-seed mean and sample standard deviation for trained models. Put paired
event-bootstrap confidence intervals for method-minus-baseline in the caption
or a compact second row.

## Required figures

1. Mean mCSI versus 5--60 minute lead for baseline and method, with two-seed
   variation.
2. Global CSI versus VIL threshold, highlighting increasing relative benefit
   at 160/181/219.
3. One observed severe event with baseline and method forecasts at selected
   long leads, chosen by an observation-only rule.

## Claim language

Allowed after convergence confirmation: reduces severe-echo abstention;
improves severe CSI and POD; no added
inference parameters; replicated across two native-resolution seeds; positive
event-bootstrap intervals conditional on trained seeds.

Not allowed: universal SOTA; universally better calibration; training-seed
statistical significance; operational readiness; no false-alarm cost.

## TGRS extension

The future TGRS paper must disclose this Letter and add a substantive new
method, such as a localized reliability constraint or calibrated spatial
occurrence model, together with substantially broader experiments.
