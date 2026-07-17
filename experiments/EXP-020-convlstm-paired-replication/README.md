# EXP-020: ConvLSTM paired replication

Status: `completed`

Current decision: the seed-1 pair replicates both the large cross-backbone
skill gain and the long-lead severe-area overforecasting observed at seed 0.

## Question

Does the large seed-0 ConvLSTM transfer result in EXP-019 replicate under an
independent initialization and data order without changing the method
coefficient?

## Why replication comes before calibration

EXP-019 shows both a large severe-CSI gain and long-lead severe-area
overforecasting. Retuning the coefficient immediately would convert a
predeclared zero-retuning transfer test into post-hoc backbone-specific tuning.
This experiment first tests whether both observations are reproducible.

## Frozen protocol

Everything matches EXP-019 except seed 1:

- official pinned OpenSTL ConvLSTM;
- four 128-channel layers, filter size 5 and patch size 4;
- 13 input frames and 12 fully autoregressive validation frames;
- BF16, batch 8, 128 resolution;
- Adam, max LR 0.0005 and OneCycleLR;
- 4000 training updates and 200 validation batches;
- official linear scheduled sampling ending at probability 0.92;
- tail thresholds 160/181/219, temperature 10 and weight 0.0003;
- no coefficient retuning.

## Baseline command

```bash
python scripts/train_openstl_convlstm.py \
  --data-root /home/amon/zyx/dataset/sevir_data \
  --manifest artifacts/local/sevir_official_manifest.csv \
  --output-dir artifacts/local/exp020_convlstm_baseline_seed1_128 \
  --resolution 128 \
  --batch-size 8 \
  --epochs 1 \
  --max-train-batches 4000 \
  --max-val-batches 200 \
  --learning-rate 0.0005 \
  --seed 1 \
  --workers 2
```

## Tail command

```bash
python scripts/train_openstl_convlstm.py \
  --data-root /home/amon/zyx/dataset/sevir_data \
  --manifest artifacts/local/sevir_official_manifest.csv \
  --output-dir artifacts/local/exp020_convlstm_tail_seed1_128 \
  --resolution 128 \
  --batch-size 8 \
  --epochs 1 \
  --max-train-batches 4000 \
  --max-val-batches 200 \
  --learning-rate 0.0005 \
  --tail-area-weight 0.0003 \
  --tail-temperature-raw 10 \
  --tail-thresholds 160 181 219 \
  --seed 1 \
  --workers 2
```

## Decision rule

- Both runs must complete 4000 optimizer updates and end with identical
  teacher-forcing probability.
- Replication passes if overall CSI, severe-threshold CSI and severe POD improve
  in the tail run without worse aggregate MSE.
- Record SUCR and 60-minute forecast-to-observed area ratios whether or not the
  gate passes.
- If the gain replicates but long-lead area overforecast does not, treat the
  seed-0 overforecast magnitude as unstable.
- If both replicate, retain overforecasting as a stable limitation and consider
  a separately declared calibration study only after the core evidence is
  frozen.

## Final paired result

Both runs completed 4000/4000 optimizer updates with identical final
teacher-forcing probability `0.91999999999992` and matching pinned source hash.
The baseline took 2840.4 seconds and the tail run 2910.9 seconds. Peak allocated
memory remained 2.31 GB.

Overall validation mean CSI rises from `0.281315` to `0.356728`
(`+26.81%` relative), while MSE falls from `0.00343712` to `0.00298727`
(`-13.09%`). CSI improves at all 12 leads for all six reported thresholds.

Relative lead-mean CSI gains are `+2.64%`, `+9.37%`, `+43.83%`, `+98.78%`,
`+171.26%` and `+494.98%` at thresholds 16, 74, 133, 160, 181 and 219.
Mean POD gains at 160/181/219 are `+163.19%`, `+293.24%` and `+889.76%`.
Mean SUCR changes by `+9.95%`, `+27.16%` and `-16.69%`, so the 160/181
gains improve both detection and precision.

The long-lead calibration boundary also replicates. At 60 minutes,
forecast-to-observed area ratios change from `0.221` to `1.650` at threshold
160, from `0.216` to `2.417` at 181, and from `0.014` to `4.598` at 219.
The domain-mean prediction bias changes from `-0.00270` to `+0.00740`.

## Two-seed conclusion

Across seeds 0 and 1:

- mean relative CSI improvement is `25.17%` (range `23.54–26.81%`);
- mean relative MSE change is `-11.91%` (range `-13.09–-10.73%`);
- every severe threshold improves at every lead in both seeds;
- recurrent long-lead severe-area overforecasting is reproducible.

The replication gate passes. The cross-backbone result can now be retained as
a central bounded-protocol paper claim. The next step is a read-only
hard-versus-soft area audit before designing any calibration component.

See `paired-result-analysis.json` and
`two-seed-convlstm-aggregate.json` for exact values.
