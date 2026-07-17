# EXP-019: ConvLSTM cross-backbone transfer gate

Status: `completed`

Current decision: the frozen tail-area component transfers to the official
OpenSTL ConvLSTM under the bounded seed-0 protocol without retuning. The gate
passes, but long-lead severe-area overforecasting is now an explicit limitation.

## Question

Does the fixed tail-area component transfer from recurrent-free SimVP variants
to the official OpenSTL ConvLSTM recurrent backbone without retuning?

## Frozen protocol

- official ConvLSTM source hash pinned after EXP-018;
- four 128-channel layers, filter size 5, patch size 4;
- official OpenSTL linear scheduled sampling: teacher-forcing probability starts
  at 1, decreases by 0.00002 per update and reaches 0.92 after 4000 updates;
- validation is fully autoregressive with a zero future-input mask;
- Adam, max LR 0.0005 and OneCycleLR;
- BF16, batch 8, seed 0, 4000 updates and 200 validation batches;
- tail thresholds 160/181/219, temperature 10 and weight 0.0003 without retuning.

## First command: trainer smoke

The model-only smoke passed, but the new scheduled-sampling trainer must complete
100 updates, validation and checkpoint writing before the pair below is
authorized.

```bash
python scripts/train_openstl_convlstm.py \
  --data-root /home/amon/zyx/dataset/sevir_data \
  --manifest artifacts/local/sevir_official_manifest.csv \
  --output-dir artifacts/local/exp019_convlstm_trainer_smoke100_seed0_128 \
  --resolution 128 \
  --batch-size 8 \
  --epochs 1 \
  --max-train-batches 100 \
  --max-val-batches 20 \
  --learning-rate 0.0005 \
  --seed 0 \
  --workers 2
```

Require 100 updates, finite metrics, a final teacher-forcing probability of
0.998 and successful output files. The 100-step scores are discarded.

## Trainer-smoke result

The trainer gate passed:

- 100/100 optimizer updates and finite train/validation outputs;
- final teacher-forcing probability `0.997999999999998`;
- pinned source hash matched;
- 2,310,032,896 bytes (2.31 GB decimal) peak allocated memory;
- 38.89 seconds wall time;
- summary, metrics and checkpoint writing completed.

The 100-update CSI values are discarded as undertrained smoke metrics. The
4000-update pair below is now authorized.

## Baseline command

Authorized after trainer-smoke acceptance:

```bash
python scripts/train_openstl_convlstm.py \
  --data-root /home/amon/zyx/dataset/sevir_data \
  --manifest artifacts/local/sevir_official_manifest.csv \
  --output-dir artifacts/local/exp019_convlstm_baseline_seed0_128 \
  --resolution 128 \
  --batch-size 8 \
  --epochs 1 \
  --max-train-batches 4000 \
  --max-val-batches 200 \
  --learning-rate 0.0005 \
  --seed 0 \
  --workers 2
```

## Tail command

Authorized after trainer-smoke acceptance:

```bash
python scripts/train_openstl_convlstm.py \
  --data-root /home/amon/zyx/dataset/sevir_data \
  --manifest artifacts/local/sevir_official_manifest.csv \
  --output-dir artifacts/local/exp019_convlstm_tail_seed0_128 \
  --resolution 128 \
  --batch-size 8 \
  --epochs 1 \
  --max-train-batches 4000 \
  --max-val-batches 200 \
  --learning-rate 0.0005 \
  --tail-area-weight 0.0003 \
  --tail-temperature-raw 10 \
  --tail-thresholds 160 181 219 \
  --seed 0 \
  --workers 2
```

## Decision rule

- Require numerical stability and identical final teacher-forcing probability.
- Pass only if severe CSI and area survival improve across most leads with a
  bounded MSE cost and without destroying low-threshold skill.
- Failure rejects broad cross-backbone generality, even though cross-translator
  SimVP transfer has passed.

## Final paired result

Both runs completed 4000/4000 optimizer updates with identical final
teacher-forcing probability `0.91999999999992`, the same seed, data order,
validation subset and pinned ConvLSTM source. The baseline used 2.310 GB peak
allocated memory and took 2859.8 seconds; the tail run used 2.311 GB and took
2913.1 seconds.

The fixed tail component raises overall validation mean CSI from `0.287306` to
`0.354940` (`+0.067634`, `+23.54%` relative). Unlike the IncepU and gSTA
transfer results, MSE also improves, from `0.00331567` to `0.00295991`
(`-10.73%`).

Lead-mean CSI improves at all 12 leads for every reported threshold. Relative
CSI gains are:

- threshold 16: `+2.63%`;
- threshold 74: `+7.00%`;
- threshold 133: `+31.73%`;
- threshold 160: `+92.76%`;
- threshold 181: `+170.52%`;
- threshold 219: `+385.31%`.

At severe thresholds 160/181/219, mean POD increases by `155.05%`, `290.72%`
and `670.68%`. Mean SUCR changes by `-7.82%`, `+2.99%` and `-11.48%`.
Therefore the severe CSI gains are not produced by precision collapse alone.

## Important boundary

The recurrent baseline is strongly underpersistent at long lead, while the
tail model crosses into overpersistence. At 60 minutes, forecast-to-observed
area ratios change as follows:

- threshold 160: `0.153` to `1.458`;
- threshold 181: `0.168` to `2.079`;
- threshold 219: `0.096` to `4.464`.

The 60-minute domain-mean prediction bias changes from `-0.00482` to
`+0.00465` in normalized VIL units. This does not invalidate the CSI result:
60-minute CSI remains higher at every severe threshold, and aggregate MSE is
lower. It does show that the fixed coefficient is not perfectly calibrated for
ConvLSTM's recurrent dynamics.

## Final decision

- The true cross-backbone transfer gate passes.
- Evidence now covers two SimVP translators and one recurrent ConvLSTM
  backbone with no backbone-specific retuning.
- The result supports architecture portability under the bounded protocol, not
  a claim of universal calibration or full-resolution SOTA.
- One ConvLSTM paired replication is required before this result is used as a
  central paper claim.
- A new corrective module is not authorized from this result. The observed
  long-lead overforecast is first treated as a limitation and calibration
  question, so the paper remains focused on the diagnosed tail-abstention
  problem and its lightweight training remedy.

See `paired-result-analysis.json` for the exact summary.
