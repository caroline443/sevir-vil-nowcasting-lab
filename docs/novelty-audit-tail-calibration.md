# Novelty audit: severe-tail calibration for SEVIR VIL

Date of audit: 2026-07-16

## Decision

The replicated EXP-010 effect is real, but **SoftExceedanceAreaLoss is not yet a
paper-level standalone innovation**.

The current loss belongs to the same location-agnostic distribution-matching
family as the 2025 Probability-Matching (PM) loss. It is best described as a
soft, sparse, severe-tail specialization of distribution matching, not as a
method invented independently of prior work. NeurIPS 2024 FACL also already
shows on native-resolution SEVIR and SimVP that changing the objective can
recover sharp extreme echoes and improve threshold skill with some pixel-error
cost. A 2026 IEEE JSTARS paper additionally combines an extremity-aware loss
with physical consistency in a VIL-specific latent diffusion model.

Therefore:

- EXP-010 validates a useful mechanism and a reproducible failure diagnosis.
- It does not justify a novelty claim such as “the first loss that prevents
  strong-echo extinction” or “a new distribution-aware loss for SEVIR.”
- PM loss is now a mandatory closest-method control; FACL is a mandatory strong
  loss baseline for paper-level experiments.
- Native 384 training should wait until the cheap closest-loss gate is passed.

## What EXP-010 actually established

Across three paired BF16 seeds at 128 × 128 and 4000 updates:

- mean CSI increased from 0.30598 to 0.33577 (+9.74% relative);
- MSE worsened by 1.00% on average;
- CSI improved in all 144 seed × severe-threshold × lead cases for thresholds
  133, 160, 181 and 219;
- lead-averaged CSI gains were +45.1%, +67.9% and +96.1% at 160, 181 and 219;
- POD improved strongly, but SUCR fell by 22.3%, 29.9% and 16.8% at those three
  thresholds;
- at 60 minutes, forecast/observed severe-area ratios remained only 12.3%,
  7.1% and 7.2%.

This supports the narrow statement that matching severe exceedance area can
delay deterministic tail extinction. It also exposes the next problem: the
method buys recall partly with false alarms and still leaves most long-lead
severe area missing.

## Mathematical relationship to the closest paper

For one forecast field with `N` pixels, define its empirical survival function

`S_x(tau) = (1/N) sum_i 1[x_i > tau]`.

SoftExceedanceAreaLoss replaces the indicator with a sigmoid and compares
`log(1 + N S_x(tau))` at three severe thresholds, separately for each sample
and lead time. It is invariant to any spatial permutation of pixels.

The 2025 PM loss independently sorts every forecast and observation field and
computes MSE between the two ordered intensity vectors. For equal-weight
empirical distributions, this is the one-dimensional squared Wasserstein-2
distance between their intensity distributions. It constrains the full quantile
function and is also invariant to spatial permutation.

The objectives are not identical, but the relationship is close:

| Property | PM loss (2025) | Current loss |
|---|---|---|
| Spatial correspondence | ignored by PM term | ignored by tail term |
| Distribution coverage | all intensity quantiles | survival function at 160/181/219 |
| Tail emphasis | indirect, through all ranks | explicit severe-tail emphasis |
| Robust scaling | ordinary MSE on sorted values | log-count compression + SmoothL1 |
| Unit of matching | each sample field | each sample and lead field |
| Extra weight | tuned to 10 on validation skill | gradient-scale probe, fixed at 3e-4 |
| Published SEVIR protocol | 384², 10-minute sampling, 1 h→1 h | 128² diagnostic, 5-minute, 13→12 |

Thus the defensible claim, if it survives direct comparison, is only that a
tail-sampled survival constraint is a cheaper or better targeted alternative
for rare operational thresholds. That is a refinement claim, not a new loss
family.

## Closest-work matrix

| Work | Problem already claimed | Mechanism | Direct novelty risk to us |
|---|---|---|---|
| TrajGRU benchmark (NeurIPS 2017) | heavy-rain class imbalance | balanced MSE/MAE | rules out generic intensity reweighting as novelty |
| FACL (NeurIPS 2024) | MSE blur and loss of extremes | Fourier amplitude/correlation training loss | high: same backbone, native SEVIR and all six thresholds |
| PM loss (GRL 2025) | weak-rain overprediction and heavy-rain underprediction, worsening with lead | full-field location-agnostic distribution matching | critical: closest conceptual and experimental neighbor |
| SimCast (2025 preprint) | long-horizon degradation | short-to-long distillation + weighted MSE | rules out merely adding weighted MSE or horizon curriculum |
| Physics-guided extremity-aware LDM (JSTARS 2026) | VIL extreme intensity and long-lead skill loss | top-10% latent extreme loss + optical-flow/physics consistency | high for generic “extreme loss + physics” combinations |
| Differentiable FSS in precipitation downscaling (2024) | displacement double penalty | soft threshold + neighborhood fractions | makes a plain differentiable-FSS auxiliary loss incremental |

## Why a naive second component is also risky

The obvious repair is “make the global loss local,” because EXP-010 loses
precision and global distribution matching can put the right amount of severe
mass in the wrong place. However, the PM paper explicitly names localized PM as
future work. Localized probability-matched means also predate deep learning in
ensemble precipitation postprocessing, while patch-wise histograms and FSS are
already established.

Consequently, merely dividing the image into blocks and applying the same loss
would likely be judged as an implementation of an explicit prior-paper future
direction. Combining global tail loss with SSIM, FSS or a standard weighted MSE
would also resemble loss-term stacking rather than a coherent new method.

## Revised research question

The question should no longer be:

> Can a new loss prevent SimVP from smoothing severe echoes?

It should become:

> Can a deterministic, A4000-trainable nowcaster preserve the lead-dependent
> severe VIL survival distribution **without paying the false-alarm and spatial
> distortion costs of global distribution matching**?

This formulation has a measurable conflict rather than a desired metric gain:
tail recall versus spatial reliability. It also distinguishes the project from
papers that only report sharper fields or higher mean CSI.

## Next experimental gate: closest-loss control

EXP-011 reproduces the published PM constraint inside the frozen development
protocol. It must use the same seed-0 data order, model, optimizer, update count,
validation subset and BF16 arithmetic as the accepted baseline and current
method.

Decision rule:

1. If PM matches or exceeds the current method at 160/181/219 with a comparable
   MSE/SUCR cost, stop claiming the current loss as an innovation.
2. If the current method has a consistent tail-specific advantage, retain it as
   a component or specialized baseline, not yet as the entire paper.
3. In either case, do not scale to 384 until a localization mechanism addresses
   the observed POD–SUCR trade-off and is distinguished from PM, FSS and FACL.

After PM, reproduce FACL from its official implementation. A paper-level result
must beat or complement both closest objectives, not just MSE.

## Candidate paper structure if the gates succeed

A realistic paper needs three mutually supportive contributions rather than
three unrelated modules:

1. **Problem characterization:** lead-dependent severe-tail extinction,
   persistence crossover, and the distinction between mean-VIL bias and tail
   survival failure under a fixed SEVIR protocol.
2. **Tail component:** a survival-function objective that is demonstrably more
   selective for 160/181/219 than full-distribution PM and FACL.
3. **Spatial reliability component:** a genuinely new, displacement-tolerant
   constraint that reduces false alarms or spatial distortion without erasing
   the recovered tail. Its exact design remains open pending the PM/FACL
   controls; a naive blockwise PM is not sufficient.

The first two alone are borderline for a lower Q2 applied journal and unlikely
to be strong enough for a CCF-B conference. With a well-motivated third
component, native-resolution confirmation, a second backbone, three seeds and
proper nearest-loss baselines, a Q2 journal becomes realistic. Acceptance is
not guaranteed, and the present evidence is still below that bar.

## Verified primary sources

- Cao et al., 2025, [Probability-Matching Loss, Geophysical Research
  Letters](https://doi.org/10.1029/2025GL119442).
- Yan et al., 2024, [Fourier Amplitude and Correlation Loss, NeurIPS
  2024](https://papers.neurips.cc/paper_files/paper/2024/hash/b54532b0e57eb963b19e00583376cda3-Abstract-Conference.html).
- Shi et al., 2017, [Deep Learning for Precipitation Nowcasting: A Benchmark
  and a New Model, NeurIPS](https://proceedings.neurips.cc/paper/2017/hash/a6db4ed04f1621a119799fd3d7545d3d-Abstract.html).
- Chen et al., 2026, [Physics-Guided Latent Diffusion Model for Intensity- and
  Extremity-Aware VIL Nowcasting, IEEE
  JSTARS](https://doi.org/10.1109/JSTARS.2026.3694114).
- Ascenso et al., 2024, [Differentiable FSS used as a training
  loss](https://doi.org/10.1016/j.wace.2024.100724).
- Snook et al., 2020, [Localized Probability-Matched Mean precipitation
  postprocessing](https://doi.org/10.1029/2020GL087839).
- Veillette et al., 2020, [SEVIR dataset,
  NeurIPS](https://proceedings.neurips.cc/paper/2020/hash/fa78a16157fed00d7a80515818432169-Abstract.html).

