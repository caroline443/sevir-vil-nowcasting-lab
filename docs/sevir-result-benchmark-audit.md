# SEVIR VIL result benchmark audit

Date: 2026-07-31

## Bottom line

The current experiment establishes a reproducible **within-training-protocol
effect**, but it is not yet a submission-ready state-of-the-art result.

Under the literature-standard reduction, `mCSI_lead_avg`, the two-seed result is
`0.41155 +/- 0.00038`, compared with `0.38893 +/- 0.00227` for the paired MSE
baseline. This is an absolute gain of `0.02262` and a relative gain of `5.82%`.
The effect is scientifically useful, but the absolute score remains below
several mature full-resolution systems.

The largest threat is training sufficiency. Our SimVP pair was trained for
three complete epochs. FACL trains full-resolution SEVIR SimVP for 50 epochs,
and FlowCast retrains comparison models with a 200-epoch budget. A reviewer can
reasonably argue that both our baseline and method are under-converged.

Therefore:

- the method-effect gate passes;
- the universal-SOTA claim fails;
- the GRSL submission-readiness gate remains open;
- the next experiment must test convergence, not add another module.

## Metric correction

Two reductions are present in the repository:

- `mCSI_lead_avg`: compute CSI independently at every lead and threshold, then
  average over the 12 leads and six thresholds. This is the number comparable
  with Earthformer-style `CSI-M6` and most later `CSI-M` tables.
- `mCSI_global`: aggregate contingency counts over all leads before computing
  CSI, then average over thresholds. This is useful internally and for the
  existing event bootstrap, but it is not the primary literature number.

The previous summary emphasized `mCSI_global=0.41194`. The paper-facing primary
number must instead be `mCSI_lead_avg=0.41155`. Both may be reported if they are
named explicitly.

## Protocol identity check

Our last-observation persistence result is:

| Metric | This repository | Earthformer paper |
|---|---:|---:|
| `mCSI_lead_avg` / `CSI-M6` | 0.269448 | 0.2695 |
| MSE | 0.01152859 | 0.0115283 |
| MAE | 0.04434994 | 0.044349 |

This near-exact three-metric match is strong evidence that our test data,
normalization, thresholding, lead reduction, and persistence implementation
match the Earthformer 13-to-12 native-resolution benchmark. The comparison to
that table is therefore substantially stronger than a loose cross-paper
comparison.

## Strict or near-strict 13-to-12, native-resolution comparison

The values below are copied from primary papers. They are not reruns in this
repository. `Comparable` means the task and metric are close enough to judge
absolute scale; it does not mean that every optimizer, duplicate filter, or
training seed is identical.

| Work | Venue/year | Resolution and horizon | CSI-M | Comparability and caveat |
|---|---|---:|---:|---|
| Persistence | Earthformer, NeurIPS 2022 | 384, 13-to-12 | 0.2695 | Strict anchor; reproduced almost exactly here |
| UNet | Earthformer, NeurIPS 2022 | 384, 13-to-12 | 0.3465 | Same published benchmark |
| Rainformer | Earthformer, NeurIPS 2022 | 384, 13-to-12 | 0.3619 | Same published benchmark |
| PhyDNet | Earthformer, NeurIPS 2022 | 384, 13-to-12 | 0.3884 | Same published benchmark |
| E3D-LSTM | Earthformer, NeurIPS 2022 | 384, 13-to-12 | 0.4015 | Same published benchmark |
| PredRNN | Earthformer, NeurIPS 2022 | 384, 13-to-12 | 0.4048 | Same published benchmark |
| **Our SimVP MSE** | this repository | 384, 13-to-12 | **0.38893** | Two seeds; only three epochs |
| **Our SimVP + SEA** | this repository | 384, 13-to-12 | **0.41155** | Two seeds; only three epochs |
| ConvLSTM | Earthformer, NeurIPS 2022 | 384, 13-to-12 | 0.4126 | Same published benchmark |
| SimVP MSE | FACL, NeurIPS 2024 | 384, 13-to-12 | 0.3989 | 50 epochs; split details less explicit |
| SimVP + FACL | FACL, NeurIPS 2024 | 384, 13-to-12 | 0.4100 | Closest published loss-based result |
| SimVP | CasCast, ICML 2024 | 384, 13-to-12 | 0.4153 | Paper's rerun; POOL1 |
| Earthformer | Earthformer, NeurIPS 2022 | 384, 13-to-12 | 0.4359 | Strong deterministic reference |
| Earthformer | CasCast, ICML 2024 | 384, 13-to-12 | 0.4310 | Official-checkpoint reevaluation |
| CasCast | ICML 2024 | 384, 13-to-12 | 0.4401 | Probabilistic cascade; much larger system |
| LLMDiff | Sensors 2024 | reported 384 | 0.4508 | Paper contains a 13/12 table but also 7/6 text; cite cautiously |
| SimCast | ICME 2025 | 384, 13-to-12 | 0.4521 | 35,718/9,060/12,159; weighted MSE plus distillation |
| FlowCast | ICLR 2026 | 384, 13-to-12 | 0.460 | Different extracted counts and 200-epoch, 4-H100 budget |
| FREUD + deterministic prior | CVPR 2026 | 384, 13-to-12 | up to 0.4455 | Ensemble rectified flow; not a lightweight deterministic peer |

Interpretation:

- SEA is slightly above the FACL paper's SimVP+FACL number (`0.41155` versus
  `0.4100`), but this is not a controlled head-to-head result.
- SEA is approximately tied with the old ConvLSTM reference and below the
  mature deterministic and probabilistic frontier.
- Relative to Earthformer `0.4359`, the absolute gap is `-0.02435` (`-5.59%`).
- Relative to SimCast `0.4521`, the gap is `-0.04055` (`-8.97%`).
- Relative to FlowCast `0.460`, the gap is `-0.04845` (`-10.53%`).

## Papers reviewed but excluded from the ranked table

These papers are relevant to novelty or method design, but their reported main
number is not a fair row in the table above.

| Work | Year | Reason for exclusion from direct ranking |
|---|---:|---|
| SEVIR dataset paper | 2020 | Establishes dataset and baseline tasks, not the later Earthformer protocol |
| Customized Multi-Scale Framework | 2023 | Non-standard temporal sampling and storm construction |
| PreDiff | 2023 | Uses 128x128 and a 7-to-6 temporally downsampled task |
| DiffCast | 2024 | Common SEVIR experiment uses 5-to-20 and downsampling |
| Feature Fusion Transformer | 2024 | Uses 6-to-18 rather than 13-to-12 |
| Probability-Matching Loss | 2025 | Uses 10-minute sampling and a 1 h-to-1 h task |
| AlphaPre | 2025 | Main SEVIR setup and metric table are not the Earthformer 13-to-12 benchmark |
| PercpCast | 2025 | Generative/perceptual protocol; not a deterministic POOL1 comparison |
| STLDM | 2025 | Downsampled 128x128 latent-diffusion benchmark |
| LMcast | 2025 | Different long-memory/generative setup |
| BlockGPT | 2025 | Autoregressive workshop protocol |
| Dual-Attention TrajGRU | 2025 | Multimodal VIL plus infrared inputs |
| ViT-Koop | 2025 | Earth-observation efficiency study; protocol details do not establish a strict row |
| exPreCast | 2026 | Efficient and relevant, but published SEVIR preprocessing differs/downscales |
| PixelFlowCast | 2026 | Recent preprint; inspectable protocol/result table not yet stable enough |
| HARECast | 2026 | Recent preprint and a larger diffusion/reconstruction framework |
| Spectral-Decoupled Iterative Refinement | 2026 | Iterative generative method with a distinct evaluation package |
| PEDNet | 2026 | Different reported error scale and precipitation interpretation |
| Multi-Scale Fourier Temporal Network | 2026 | Reports a single CSI-50 task rather than six-threshold CSI-M |
| Physics-guided extremity-aware VIL LDM | 2026 | Large latent-diffusion system and different contribution class |

This audit covers 29 result-bearing works or benchmark rows across the classic,
loss-based, deterministic, diffusion, and flow-matching families. It confirms
the earlier warning that “SEVIR result” alone does not define a protocol.

## Publication judgment

### Is the result good?

As a method ablation, yes. A `+5.82%` literature-standard mCSI improvement,
two-seed replication, stronger high-threshold gains, no inference overhead, and
an event-level uncertainty analysis are meaningful.

As a complete GRSL submission today, no. The absolute score is only around the
older ConvLSTM level, the baseline is weaker than published SimVP reruns, and
three epochs are not a defensible convergence budget beside 50- or 200-epoch
papers. The current cross-backbone evidence is also only a bounded
low-resolution diagnostic.

### Minimum experiment that can change the verdict

Do not add a new architecture. Continue the frozen baseline and SEA checkpoints
with identical schedules and evaluate validation after each epoch.

Predeclare:

1. resume both seed pairs from epoch 3;
2. train to at least epoch 10, preferably until validation `mCSI_lead_avg`
   fails to improve for three consecutive epochs;
3. select each checkpoint using validation only;
4. do not access test during continuation;
5. rerun the frozen test only once after the stopping rule fires;
6. report both `mCSI_lead_avg` and `mCSI_global`;
7. if the converged SEA result remains below approximately `0.43`, position the
   work as a loss/diagnostic Letter and include a strong reproduced reference;
8. if the effect collapses with convergence, do not submit the current method.

The fastest credible compute path is a rented RTX 5090 for checkpoint
continuation. Paying for convergence is more valuable than paying for another
backbone or another speculative module.

## Primary sources

- [SEVIR dataset, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/fa78a16157fed00d7a80515818432169-Abstract.html)
- [Earthformer, NeurIPS 2022](https://arxiv.org/abs/2207.05833)
- [PreDiff, NeurIPS 2023](https://papers.nips.cc/paper_files/paper/2023/file/f82ba6a6b981fbbecf5f2ee5de7db39c-Paper-Conference.pdf)
- [CasCast, ICML 2024](https://proceedings.mlr.press/v235/gong24a.html)
- [DiffCast, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Yu_DiffCast_A_Unified_Framework_via_Residual_Diffusion_for_Precipitation_Nowcasting_CVPR_2024_paper.html)
- [FACL, NeurIPS 2024](https://papers.neurips.cc/paper_files/paper/2024/file/b54532b0e57eb963b19e00583376cda3-Paper-Conference.pdf)
- [LLMDiff, Sensors 2024](https://www.mdpi.com/1424-8220/24/18/6049)
- [AlphaPre, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Lin_AlphaPre_Amplitude-Phase_Disentanglement_Model_for_Precipitation_Nowcasting_CVPR_2025_paper.html)
- [SimCast, ICME 2025](https://arxiv.org/abs/2510.07953)
- [Probability-Matching Loss, GRL 2025](https://doi.org/10.1029/2025GL119442)
- [FlowCast, ICLR 2026](https://arxiv.org/abs/2511.09731)
- [exPreCast, ICLR 2026](https://arxiv.org/abs/2602.05204)
- [FREUD, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Schusterbauer_Probabilistic_Precipitation_Nowcasting_with_Rectified_Flow_Transformers_CVPR_2026_paper.pdf)
- [PixelFlowCast preprint, 2026](https://arxiv.org/abs/2605.10046)
- [HARECast preprint, 2026](https://arxiv.org/abs/2605.13181)
