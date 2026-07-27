# EXP-025: native-resolution paper SimVP seed-0 pair

Status: `completed`

Current decision: the frozen method passes the native-resolution seed-0 gate.
Both arms completed three epochs, both selected epoch 3, and the test split
remains untouched. Proceed to the frozen native seed-1 pair.

## Question

Under the frozen 13-to-12, event-disjoint SEVIR VIL protocol at native
384x384 resolution, does the soft exceedance-area term reduce severe-echo
abstention beyond the bounded 128x128 development experiments?

## Frozen pair

Both arms use official OpenSTL SimVP IncepU, BF16, batch size 1, seed 0,
35,718 training windows, 9,060 validation windows, three configured epochs and
the same OneCycle schedule. The only difference is:

- baseline: pixelwise MSE;
- method: MSE plus the frozen soft exceedance-area loss at raw thresholds
  160/181/219, temperature 10 and coefficient 0.0003.

The manifest SHA-256 is
`cd87c9df175cdf25c77d48da052e2650ffb78d722c34298c1a37e01a3a849630`.

## Final three-epoch result

| Metric | Baseline | Tail area | Relative change |
|---|---:|---:|---:|
| validation mCSI global | 0.385089 | 0.408221 | +6.01% |
| validation mCSI lead average | 0.381613 | 0.408331 | +7.00% |
| validation MSE | 0.003200 | 0.003239 | +1.22% |
| validation MAE | 0.026050 | 0.026022 | -0.11% |

The aggregate gain remains intensity-selective after convergence. Global CSI
changes by +0.03% at threshold 16, -0.07% at 74, +3.76% at 133, +12.47% at
160, +26.95% at 181 and +46.61% at 219. Mean CSI improves at every lead from
5 through 60 minutes; the relative gain grows from +0.64% at 5 minutes to
approximately +10.7% at 60 minutes.

The mechanism is not a free improvement. At thresholds 160/181/219, POD rises
by 22.66%/43.71%/71.61%, while SUCR falls by 15.52%/23.16%/37.58%. Thus the
method recovers severe echoes by accepting more false alarms. At 60 minutes,
the baseline forecast/observed area ratios are 6.05%, 0.62% and 0% for
160/181/219; the method raises them to 25.64%, 23.12% and 18.12%. Severe area
is still underforecast, so this is reduced abstention rather than complete
calibration.

## Interpretation boundary

This is the first converged positive result at native resolution and full
train/validation coverage. It is not yet a publishable superiority claim
because it is one seed and validation-only. The small MSE cost and
severe-threshold SUCR cost must be reported rather than hidden behind mean CSI.

## Runtime

The complete three-epoch wall time was approximately 27.0 hours for baseline
and 25.3 hours for the method across the initial and resumed invocations. Peak
allocated memory was 9.79 GB for both arms.

## Acceptance decision

- numerical completion: pass;
- independent validation selection: pass, epoch 3 for both arms;
- severe-threshold CSI and long-lead survival: pass;
- low-threshold guardrail: pass;
- calibration tradeoff: present and must be disclosed;
- native replication: required before test evaluation;
- test accessed: no.
