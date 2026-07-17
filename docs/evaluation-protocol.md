# Evaluation protocol for paper-facing SEVIR VIL results

## Decision

Use one **Earthformer-compatible primary protocol** for all paper-facing model comparisons. Treat scores copied from papers with incompatible preprocessing, splits, resolutions or metric reductions as contextual references only, never as rows in the same ranked table.

OpenSTL remains the source of the official SimVP implementation and hyperparameters. It is not the primary data/evaluation protocol because its provided SEVIR preparation uses a train/test split around 2019-06-01 and its SimVP configuration highlights a single metric threshold of 74. Earthformer's public SEVIR setup provides the stronger train/validation/test temporal design and six-threshold evaluation used here.

## Primary protocol: `sevir-vil-earthformer-v1`

### Data

- modality: VIL only;
- raw event shape: 49 frames at 384×384;
- temporal interval: 5 minutes;
- final paper-facing spatial resolution: native 384×384;
- retain only catalog rows with `pct_missing == 0`;
- remove every duplicated VIL event ID rather than choosing an ambiguous row;
- normalize model tensors as `float32(raw_uint8) / 255`.

### Samples

- create 25-frame windows from each event at frame starts 0, 12 and 24;
- input: first 13 frames;
- target: following 12 frames;
- forecast leads: 5, 10, ..., 60 minutes;
- assign events to time splits before expanding windows so windows from one event cannot cross splits.

### Temporal split

- train: `time_utc <= 2019-01-01 00:00:00`;
- validation: `2019-01-01 00:00:00 < time_utc <= 2019-06-01 00:00:00`;
- test: `time_utc > 2019-06-01 00:00:00`.

The repository manifest builder implements these inclusive/exclusive boundaries to match Earthformer's loader semantics. The generated manifest, its row counts and a checksum must be frozen before final experiments.

## Required metrics

### Main table

- MSE and MAE on `[0,1]` VIL;
- global `CSI@16`, `CSI@74`, `CSI@133`, `CSI@160`, `CSI@181`, `CSI@219`;
- `mCSI_global`: arithmetic mean of the six global CSI values;
- POD and SUCR at the same six thresholds;
- parameter count, peak training memory and inference latency.

For a global threshold metric, sum hits, misses and false alarms over every test sample, pixel and lead time before computing the ratio.

### Lead-time table or figure

- `CSI@threshold(t)` for all 12 leads;
- `mCSI_by_lead(t)`: mean across the six thresholds at each lead;
- MSE by lead;
- forecast and observed exceedance-pixel counts by threshold and lead;
- persistence curves on the identical test samples.

### Naming rule

Do not label two different reductions simply as “mCSI.” Report both explicitly when needed:

- `mCSI_global`: average of threshold CSI after aggregating counts over time;
- `mCSI_lead_avg`: average of per-lead CSI over thresholds and leads.

This prevents accidental comparison between a ratio of globally aggregated counts and an average of time-specific ratios.

## Statistical reporting

- development/ablation stage: seed 0 is acceptable for rejection tests;
- final main comparison: at least three seeds for SimVP and the proposed method;
- report mean and standard deviation across seeds;
- compute confidence intervals or paired bootstrap differences by event, not by pixel, because pixels and overlapping windows are not independent;
- choose checkpoints using validation data only and evaluate the test split once per finalized configuration.

## Baseline policy

The minimum unified main table must rerun under this repository's protocol:

1. last-observation persistence;
2. official OpenSTL SimVP with MSE;
3. the proposed method with the same SimVP capacity and training budget;
4. at least one recurrent or alternative deterministic baseline if compute permits.

Published Earthformer, diffusion or recent SOTA numbers may be shown in a separate literature table only when their exact protocol is documented. Each such row must state resolution, split, input/output length, metric reduction and whether the score was copied or reproduced.

## Compute-aware two-stage workflow

### Development protocol

- 128×128 resize;
- fixed validation subset;
- one seed;
- 1000-step rejection tests and 4000-step confirmation tests.

These results diagnose mechanisms and select ablations. They are never used as final SOTA claims.

### Final protocol

- native 384×384;
- frozen full train/validation/test manifest;
- selected baseline and proposed configuration only;
- paid RTX 5090 use is allowed only after the 128×128 mechanism test succeeds and a measured runtime budget is approved.

## Comparability rule for the paper

The main claim should be phrased as improvement over rerun baselines **under `sevir-vil-earthformer-v1`**, not as universal SEVIR SOTA. A broader SOTA claim is justified only if the method is also evaluated under the exact protocol of the competing paper or the competing method is rerun under this protocol.

## Repository implementation

- `scripts/train_paper_simvp.py` implements native-resolution BF16 training,
  epoch validation, `mcsi_global` checkpoint selection, deliberate partial
  stopping and exact optimizer/scheduler/RNG resume.
- `scripts/evaluate_paper_simvp.py` evaluates a frozen checkpoint with the full
  metric schema. Test evaluation requires `--confirm-final-test` and refuses to
  overwrite an existing result.
- `LeadTimeVILMetrics` reports both `mcsi_global` and `mcsi_lead_avg`; the
  legacy `csi_mean` field remains an alias of `mcsi_lead_avg` for development
  result compatibility.
- ConvLSTM formal runs must use `--sampling-schedule budget_linear` with a
  declared end probability rather than truncating the upstream 50000-update
  schedule at a high teacher-forcing probability.
