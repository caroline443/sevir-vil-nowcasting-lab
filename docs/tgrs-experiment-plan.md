# TGRS experiment plan

Target: IEEE Transactions on Geoscience and Remote Sensing (TGRS).

The paper will not claim universal SEVIR SOTA. Its central claim will be a
remote-sensing-specific diagnosis and remedy: deterministic pixelwise VIL
nowcasting attenuates rare severe echoes at long lead times, and a soft
exceedance-area constraint improves severe-echo survival with an explicit
recall/false-alarm tradeoff.

TGRS requires novel and significant methodological research and complete
experimental evidence. The following gates are therefore ordered by necessity,
not by model novelty.

## P0: publication gates

### P0.1 Finish the current native main pair

Complete EXP-025 seed 0 for baseline and tail-area SimVP IncepU through the
configured three epochs. Select `best.pt` independently by validation
`mcsi_global`; do not inspect or use test results.

### P0.2 Native replication

Run the same frozen pair for seed 1 and, if the A4000 budget permits, seed 2.
The minimum TGRS-quality target is two native seeds plus the existing bounded
three-seed development evidence; three native seeds is preferred. Report mean,
sample standard deviation and paired event-bootstrap intervals.

### P0.3 Freeze and evaluate test once

After the configurations and checkpoints are frozen, evaluate on test exactly
once for:

1. last-observation persistence;
2. SimVP MSE baseline;
3. SimVP plus soft exceedance-area loss.

Report MSE, MAE, global and lead-averaged CSI at 16/74/133/160/181/219, POD,
SUCR, forecast/observed exceedance area and all 12 lead times. Test must never
be used for selecting epochs, thresholds, temperature or loss weight.

### P0.4 Statistical evidence

Use event-level paired bootstrap, not pixel-level bootstrap. Resample storm
events, preserve all windows and leads belonging to each event, and report the
confidence interval for tail-minus-baseline differences. Include the fraction
of event/lead cases improving at each threshold.

## P1: reviewer-risk reduction

### P1.1 Loss controls

Use the existing probability-matching and FACL implementations as controls
under the same 128 development protocol and budget. If compute permits, run the
strongest control at native resolution. Controls must use validation selection
and the same MSE/CSI/POD/SUCR schema.

### P1.2 Loss sensitivity

On 128 development data, run a small predeclared grid around the frozen choice:

- thresholds: 160; 181; 219; and 160+181+219;
- temperature: 5, 10, 20 raw VIL units;
- weight: 1e-4, 3e-4, 1e-3.

The purpose is not to find the best test score. It is to show whether the
chosen coefficient produces a stable severe-CSI versus SUCR/MSE tradeoff and to
explain why all three thresholds are used.

### P1.3 Cross-architecture confirmation

Use one additional backbone as confirmation, preferably gSTA or a recurrent
model with a declared low/zero scheduled-sampling endpoint. Existing bounded
128 gSTA and ConvLSTM results support transfer, but the ConvLSTM truncated at
92% teacher forcing must not be presented as a formal free-rollout result.

### P1.4 Paper figures and efficiency

Prepare 4–6 event case studies, threshold-by-lead curves, severe-area curves,
false-alarm maps, POD/SUCR tradeoff plots and a table of parameters, peak VRAM,
latency, training wall time, BF16 mode, manifest SHA and exact split counts.

## Stop rules

- If native seed 0 loses its validation advantage by the best epoch, stop the
  TGRS main claim and reassess the method before spending on replications.
- If the gain exists only at one threshold or only through unbounded false
  alarms, report it as a limitation rather than tuning another module.
- Do not add attention, Mamba, SSIM, LLM or a second calibration objective to
  the frozen main method.
- Do not rent a 5090 until a short benchmark demonstrates a material runtime
  advantage; the main protocol is already runnable on the A4000.

## Minimum paper-complete package

EXP-025 completed; native seed 1 pair; frozen test results for persistence,
MSE and tail-area SimVP; fair PM/FACL controls; event-bootstrap confidence
intervals; sensitivity table; one cross-architecture confirmation; qualitative
and efficiency figures; and a manuscript that explicitly states the
recall/false-alarm limitation and does not claim universal SOTA.

