# Validated problem and feasible solution paths

## Validated problem

The working research problem is:

> **Lead-time-dependent severe-tail extinction coupled with weak-echo diffusion under deterministic MSE training.**

SimVP improves bulk-field MSE and low/moderate-threshold CSI, but increasingly reallocates forecast mass away from rare intense cores toward widespread weak echoes. The effect remains after a near-one-epoch 4000-step control and is not explained solely by storm displacement, global intensity bias or a broken training pipeline.

## Evidence chain

1. The pinned official SimVP architecture runs correctly and can overfit a native-resolution sample.
2. A 1000-step pilot shows threshold-181/219 forecasts disappearing while observed severe pixels remain present.
3. Persistence is worse overall but better at severe thresholds, proving that model-induced intensity extinction contributes beyond displacement error.
4. Increasing training to 4000 steps improves MSE and CSI substantially, but only delays rather than removes severe-tail extinction.
5. At 60 minutes the 4000-step model slightly overpredicts mean VIL and overpredicts weak-echo area by 19.8%, while almost eliminating thresholds 181/219. Global mean correction is therefore not a sufficient solution.

## Why the previous F1 + SSIM attempt was poorly matched

- Pixelwise F1 remains highly sensitive to small displacement: a shifted but structurally plausible core is counted as both a miss and a false alarm.
- SSIM is dominated by broad spatial structure and does not directly constrain rare intensity-tail mass.
- Combining them without first measuring the failure mixes location, structure and imbalance into one objective, making negative results hard to interpret.

The new problem definition calls for a component that preserves severe-tail statistics without requiring exact pixel alignment.

## Mutually distinct feasible solution paths

### 1. Soft exceedance-area calibration loss — recommended first

For thresholds 160, 181 and 219, compute differentiable exceedance areas using a temperature-controlled sigmoid. Penalize the log-ratio between predicted and target exceedance area separately for every sample and lead time, alongside MSE.

- Mechanism: objective-level marginal tail calibration; no architecture change.
- Difference from F1: position-tolerant and does not double-penalize displacement.
- Compute: negligible overhead; one 4000-step A4000 run is about 37 minutes.
- Falsifiable target: restore high-threshold forecast counts and delay persistence crossover without materially degrading threshold-16/74 CSI.
- Main risk: correct area but wrong location; CSI/POD/SUCR guard against this.

### 2. Ordinal exceedance auxiliary head

Add a small head that predicts nested exceedance probabilities for the six SEVIR thresholds, with monotonic consistency across thresholds. The original regression decoder remains unchanged.

- Mechanism: representation supervision for rare categorical tails.
- Compute: low; a few 1×1 convolutions.
- Main risk: auxiliary classification improves calibration but not the decoded VIL field.
- Mutually distinct feature: changes supervised representation/head, not the regression loss alone.

### 3. Severe-core residual branch

Retain the standard SimVP decoder for the bulk field and add a lightweight residual branch gated by high-intensity latent evidence to reconstruct severe cores.

- Mechanism: architectural capacity separation between common background and rare cores.
- Compute: moderate but A4000-compatible if limited to low-resolution latent maps.
- Main risk: gating collapses or hallucinates cores.
- Mutually distinct feature: explicit two-branch architecture rather than objective or sampling changes.

### 4. Rare-core-balanced event curriculum

Keep model and MSE unchanged, but construct batches with controlled proportions of windows containing thresholds 160/181/219, then anneal toward the natural distribution.

- Mechanism: optimization/data exposure.
- Compute: unchanged; requires only manifest statistics and a sampler.
- Main risk: probability distortion and false alarms under the natural validation distribution.
- Mutually distinct feature: data curriculum only, no new loss or model parameters.

### 5. Lead-time-conditioned monotone calibration

Learn a small monotone mapping from raw forecast VIL to calibrated VIL separately by lead time using held-out validation data.

- Mechanism: post-hoc output calibration.
- Compute: minimal and useful as a diagnostic lower bound.
- Main risk: cannot restore spatial cores that the model has completely erased.
- Mutually distinct feature: no retraining of SimVP; establishes how much is recoverable from output calibration alone.

## Recommended sequence

Start with path 1 because it directly matches the measured failure, is clearly different from pixelwise F1/SSIM, adds no material inference cost and can be rejected within one short run. Compare against the exact 4000-step baseline with the same seed and validation subset. Only if exceedance area improves without spatial skill should path 2 or 3 be considered.

No claim should be made from 128×128 diagnostics alone. A successful mechanism must later be confirmed at native resolution on a controlled subset and then evaluated under the official test split.
