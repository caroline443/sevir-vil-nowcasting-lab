# Literature map: SEVIR VIL nowcasting

## Benchmark warning

There is no single directly comparable SEVIR leaderboard. Common protocols include:

- Earthformer-style: 13 input frames, 12 output frames, 5-minute interval, often 384 × 384.
- PreDiff-style: 7 input frames, 6 output frames, usually 128 × 128 after temporal and spatial downsampling.
- DiffCast-style: 5 input frames, 20 output frames, usually 128 × 128.
- Other papers use 6→18, different date splits, multimodal inputs or different temporal sampling.

Scores from different protocols must not be placed in one ranking without qualification.

## Core papers

| Year | Work | Main contribution | Important limitation for this project |
|---:|---|---|---|
| 2020 | SEVIR | Event-based aligned VIL, satellite and lightning benchmark | Event sampling and VIL are not equivalent to operational rainfall frequency or rain rate |
| 2022 | Earthformer | Cuboid space-time attention; established a widely used 13→12 protocol | Full-resolution training is expensive; deterministic forecasts blur |
| 2023 | Customized Multi-Scale Framework | Multi-scale module and loss for strong storms | Non-standard sampling complicates comparison |
| 2023 | PreDiff | Latent diffusion with knowledge alignment | 128² protocol and very expensive sampling |
| 2024 | CasCast | Deterministic mesoscale forecast followed by probabilistic local refinement | Two-stage error propagation and training cost |
| 2024 | DiffCast | Residual diffusion attached to deterministic backbones | Sampling remains slow; common protocol is downsampled |
| 2024 | FACL | Fourier amplitude/correlation loss recovers sharp extremes across SimVP, ConvLSTM and Earthformer | Native-resolution closest loss baseline; can trade pixel accuracy for skill and lacks temporal regularization |
| 2024 | LLMDiff | Frozen language-model transformer blocks in diffusion | Large compute and unclear benefit beyond capacity |
| 2024 | Feature Fusion Transformer | Explicit spatial-temporal feature crossing | Uses a 6→18 protocol; limited comparability |
| 2025 | AlphaPre | Frequency amplitude/phase disentanglement | Frequency interpretation is not a strict physical decomposition |
| 2025 | PercpCast | Posterior-mean estimator plus rectified-flow perceptual constraint | Distribution matching does not ensure event-level conditional correctness |
| 2025 | Probability-Matching Loss | Aligns predicted and observed intensity distributions | Marginal distribution matching does not ensure correct location |
| 2025 | Dual-Attention TrajGRU | VIL and infrared multimodal fusion | Requires interpolation and assumes aligned, available modalities |
| 2025 | SimCast | Short-to-long horizon distillation and weighted MSE | Additional teacher training; protocol-specific SOTA claim |
| 2025 | STLDM | End-to-end latent diffusion with 20-step sampling | Evaluated on downsampled 128² SEVIR |
| 2025 | LMcast | VQ-VAE and pretrained-language-model long-term memory | Heavy system; retrieval leakage must be checked |
| 2025 | BlockGPT | Frame-level autoregression for faster generation | Workshop evidence and autoregressive error accumulation |
| 2026 | FlowCast | Conditional flow matching for efficient probabilistic nowcasting | Full-resolution training remains heavy |
| 2026 | exPreCast | Local deterministic modeling and texture-preserving decoding | Locality may miss non-local precursors |
| 2026 | FREUD | Uncertainty-preserving compression and rectified-flow transformer | Reported gains partly rely on scaling and ensemble compute |
| 2026 | Spectral-Decoupled Iterative Refinement | Low-frequency backbone plus constrained high-frequency refinement | Spectral correctness does not guarantee correct phase or location |
| 2026 | StormDiT | Unified long-horizon generative model for the 2–6 h gray zone | Large external dataset and compute; preprint evidence |
| 2026 | HARECast | Attention-response energy stability regularization | Preprint; regularization may suppress legitimately rare responses |
| 2026 | MFC-RFNet | Multi-scale rectified flow with alignment and frequency fusion | Many interacting modules; unclear minimal cause of gains |
| 2026 | PW-FouCast | Frequency fusion of radar and Pangu-Weather priors | Additional foundation-model data and system cost |
| 2026 | Physics-Guided Extremity-Aware VIL LDM | Intensity-stratified optical-flow priors, top-tail latent loss and physical consistency | Large generative system; makes generic “extreme loss + physics” claims crowded |

## Novelty warning for loss-based work

The 2025 Probability-Matching loss sorts each predicted and observed field and
matches their full empirical intensity distributions. The current
SoftExceedanceAreaLoss matches a few points of the same empirical survival
distribution and is therefore a tail-specialized close relative, not a clearly
independent loss family. FACL already evaluates SimVP at native 384² SEVIR with
thresholds 16, 74, 133, 160, 181 and 219. Any paper based on the present loss
must directly compare with both methods. See
[the detailed novelty audit](novelty-audit-tail-calibration.md).

## Repeated unresolved tensions

1. Deterministic accuracy versus sharpness and uncertainty.
2. Motion extrapolation versus convective initiation, growth and decay.
3. Pixel scores versus storm-object usefulness.
4. Average performance versus rare high-impact failure.
5. Full resolution versus affordable training and inference.
6. Radar-only simplicity versus missing environmental precursors.
7. Claimed SOTA versus fragmented evaluation protocols.

## Reproduction priority under an A4000 budget

1. SimVP baseline with a fixed 128 × 128 development protocol.
2. One recent efficient deterministic baseline if code is stable.
3. STLDM or FlowCast inference from public weights, not training from scratch.
4. Full-resolution evaluation only after a low-resolution hypothesis succeeds.
