# EXP-027: frozen GRSL final test

Status: `completed`

Revised decision after the 2026-07-31 literature result audit: the paired
method-effect gate passes, but the GRSL submission-readiness gate does not.
The configuration and existing test outputs remain permanently frozen. Further
test access is prohibited; validation-only checkpoint continuation is allowed
to determine whether the three-epoch models were under-converged.

## Protocol

- SEVIR VIL, native 384x384;
- event-disjoint temporal test split;
- 4,053 events and 12,159 windows;
- 13 input frames and 12 forecast frames;
- official OpenSTL SimVP IncepU;
- two independently trained seeds;
- each checkpoint selected only by validation global mCSI;
- baseline MSE versus MSE plus the frozen soft exceedance-area loss;
- 10,000 paired bootstrap replicates over complete storm events.

## Main result

| Metric | Baseline mean | Method mean | Mean difference | Event-bootstrap 95% CI |
|---|---:|---:|---:|---:|
| lead-average mCSI (literature primary) | 0.38893 | 0.41155 | +0.02262 | not yet computed |
| global mCSI | 0.39427 | 0.41194 | +0.01767 | [0.01722, 0.01815] |
| CSI@160 | 0.25004 | 0.27759 | +0.02756 | [0.02664, 0.02851] |
| CSI@181 | 0.20356 | 0.24207 | +0.03851 | [0.03738, 0.03966] |
| CSI@219 | 0.11811 | 0.14062 | +0.02251 | [0.02115, 0.02389] |
| MSE | 0.003721 | 0.003664 | -0.000057 | [-0.000061, -0.000054] |
| MAE | 0.026704 | 0.026003 | -0.000700 | [-0.000716, -0.000685] |

Relative to the baseline, the method improves literature-standard lead-average
mCSI by 5.82%, global mCSI by 4.48%, and
CSI@160/181/219 by 11.02%/18.92%/19.06%. MSE and MAE decrease by
1.54% and 2.62%. Every CSI threshold has a positive event-bootstrap interval.

Last-observation persistence obtains global mCSI 0.26152, MSE 0.01153 and MAE
0.04435. The proposed model improves mCSI by 57.52% relative to persistence
while reducing MSE by 68.22%.

## Tradeoff

The severe-event improvement is recall-driven. Averaged across seeds, POD
increases by 0.05397/0.07425/0.05016 at thresholds 160/181/219, while SUCR
decreases by 0.09004/0.14534/0.22600. Every corresponding bootstrap interval
excludes zero. The paper must therefore claim reduced severe-echo abstention,
not uniform calibration dominance.

## Statistical boundary

Event-bootstrap intervals quantify dataset-sampling uncertainty conditional on
the two trained seeds. They do not establish training-randomness significance.
The paper reports seed means and sample standard deviations alongside the
event-bootstrap intervals and explicitly states this limitation.

## Revised publication decision

The existing result is not yet sufficient for submission. Published
full-resolution work trains SimVP for substantially longer, while this pair
stopped after three epochs and obtains a weaker absolute baseline. No further
test evaluation or hyperparameter tuning is authorized. The frozen pairs may
be resumed using validation-only selection and a predeclared convergence rule.
See `docs/sevir-result-benchmark-audit.md`.
