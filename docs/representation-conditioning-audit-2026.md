# Representation and conditioning audit after EXP-016

Date: 2026-07-16

## Empirical requirement

EXP-013--016 narrow the unresolved failure to a specific case: at raw VIL 219
and long lead, a scalar severe-area objective retains gradient magnitude but
cannot identify where an extinguished core should be restored. Wider
soft-threshold temperatures mainly activate background. Any second model
component would therefore need to supply spatial or dynamical support, not
merely more tail weight.

## 2025--2026 collision matrix

| Candidate mechanism | Closest recent work | Audit decision |
|---|---|---|
| Advection plus growth/decay residual | NowcastNet; 2026 MvReNowcast explicitly separates advection from residual generation/dissipation | Reject: scientifically aligned but already a named framework family |
| Local severe-core refiner | 2026 exPreCast uses local spatiotemporal attention and a texture-preserving decoder; 2026 SDIR performs iterative fine residual refinement | Reject: a small SimVP refiner would look derivative |
| Frequency or wavelet severe branch | FACL; 2026 WADEPre uses a localized wavelet detail branch and multiscale curriculum | Reject: direct collision with extreme-precipitation motivation |
| Learned feature alignment | 2026 MFC-RFNet uses condition-guided spatial-transform fusion to correct displacement | Reject: direct collision and substantially larger generative system |
| Storm-object tracking/evolution | 2025--2026 storm-cell evolution and positional-nowcast work predicts cell motion, area and intensity | Defer: distinct representation, but requires object extraction/tracking labels and changes the task scope |
| Extra IR/lightning/environmental context | Multiple recent multimodal radar/satellite and physics-constrained methods | Reject for this project: VIL-only data and compute scope would no longer be preserved |
| Generic attention, Mamba or decoder replacement | Broad existing video-prediction and nowcasting literature | Reject: architecture substitution is not an answer to the diagnosed support failure |

## Publication decision

Do not invent a second architecture component merely to increase the module
count. The recommended target is an applied Q2 paper built around one generic
training mechanism plus unusually strong diagnosis and closest-method controls:

1. characterize severe-event abstention separately from remote hallucination;
2. introduce the threshold-survival area constraint as a lightweight,
   displacement-insensitive tail calibration component;
3. compare directly with MSE, Probability Matching and FACL under one protocol;
4. demonstrate transfer without retuning across at least two predictor variants;
5. report CSI/POD/SUCR, forecast-area ratio and neighborhood-tolerant skill.

This is not currently strong enough for a CCF-B claim. It becomes a realistic
applied-Q2 submission only if the effect transfers beyond SimVP-IncepU and later
survives a sufficiently converged, paper-facing protocol.

## Next gate

Use SimVP-v2's gSTA temporal translator as the first transfer test because it
changes the temporal mixing mechanism while preserving the same encoder,
decoder, data and compute envelope. Apply the exact threshold set, temperature
and weight selected on IncepU; do not re-tune on gSTA.

- First run a 100-update numerical/memory smoke test.
- If stable, run the frozen 4000-update seed-0 baseline/tail pair.
- Continue only if severe CSI and area survival improve without a materially
  larger MSE penalty.
- A positive result is cross-translator evidence, not yet full cross-backbone
  generalization. A recurrent predictor remains desirable for the final paper.

## Primary sources

- Liu et al., 2026, [WADEPre](https://arxiv.org/abs/2602.02096).
- Song et al., 2026, [exPreCast](https://arxiv.org/abs/2602.05204).
- Luo et al., 2026, [MFC-RFNet](https://arxiv.org/abs/2601.03633).
- Zhou et al., 2026, [SDIR](https://openreview.net/forum?id=zB4xF9tfdm).
- Zhang et al., 2023, [NowcastNet](https://doi.org/10.1038/s41586-023-06184-4).
- Zhu et al., 2026, [MvReNowcast](https://doi.org/10.1016/j.jhydrol.2026.135938).
- Song et al., 2026, [exPreCast ICLR publication page](https://openreview.net/pdf?id=fDknsQhSgm).
