# EXP-014: paired baseline-tail spatial diagnosis

Status: `completed`

## Question

Does SoftExceedanceAreaLoss recover severe forecasts near observed storms, or
does it obtain its CSI gain by creating remote false alarms?

EXP-013 alone cannot answer this causal comparison because it evaluated only
the tail-area checkpoint. EXP-014 evaluates the seed-0 MSE baseline on the
identical 200 validation batches and pairs its counts with EXP-013.

## Result

The baseline largely abstains from severe prediction at long lead. At 60
minutes its forecast-to-observed area ratios are 1.00%, 0.03% and 0% at raw VIL
thresholds 160, 181 and 219. The corresponding tail-area ratios are 13.67%,
7.67% and 7.51%.

This makes raw precision alone misleading: a nearly empty forecast can have
high precision. We therefore reconstruct the location of the *incremental*
forecast pixels introduced by tail-area training. At thresholds 160 and 181,
85.1% and 84.4% of added pixels lie within two 128-resolution pixels of an
observed severe pixel; 94.1% and 92.3% lie within four pixels. At threshold 219
the corresponding fractions are 71.3% and 82.4%.

The tail-area checkpoint also increases mean tolerant recall at every tested
radius and threshold. At radius 8, the mean recall gains are +0.221, +0.250 and
+0.203 for thresholds 160, 181 and 219. At 60 minutes, radius-8 recall changes
from 2.61% to 28.81%, 0.20% to 17.13%, and 0% to 13.04%.

## Decision

- The hypothesis that the tail-area CSI gain is mainly purchased through
  remote hallucination is rejected under this bounded protocol.
- The lower pixelwise SUCR at thresholds 160 and 181 is partly the cost of
  replacing severe-event abstention with spatially near-miss forecasts.
- Cancel the proposed pure false-alarm/outside-envelope penalty. It targets the
  wrong dominant error and may restore the baseline's degenerate abstention.
- Do not add a spatial loss merely to make the method look multi-component.
  The next component must address the remaining long-lead severe-mass deficit
  or the rarity of persistent/growing severe sequences, and must pass a separate
  problem diagnostic before implementation.

See `result-analysis.json` for the paired numerical summary.
