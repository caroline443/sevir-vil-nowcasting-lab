# EXP-025: native-resolution paper SimVP seed-0 pair

Status: `epoch 1 completed; epochs 2-3 pending`

Current decision: the frozen method passes the predeclared epoch-1 continuation
gate. Resume both arms without changing any hyperparameter, select each arm's
best checkpoint by validation `mcsi_global`, and keep the test split untouched.

## Question

Under the frozen 13-to-12, event-disjoint SEVIR VIL protocol at native
384x384 resolution, does the soft exceedance-area term reduce severe-echo
abstention beyond the bounded 128x128 development experiments?

## Frozen pair

Both arms use official OpenSTL SimVP IncepU, BF16, batch size 1, seed 0,
35,718 training windows, 9,060 validation windows, three configured epochs and
the same OneCycle schedule. The only difference is:

- baseline: pixelwise MSE;
- method: MSE plus the frozen soft exceedance-area loss at raw thresholds
  160/181/219, temperature 10 and coefficient 0.0003.

The manifest SHA-256 is
`cd87c9df175cdf25c77d48da052e2650ffb78d722c34298c1a37e01a3a849630`.

## Epoch-1 result

| Metric | Baseline | Tail area | Relative change |
|---|---:|---:|---:|
| validation mCSI global | 0.314512 | 0.324627 | +3.22% |
| validation mCSI lead average | 0.313179 | 0.324879 | +3.74% |
| validation MSE | 0.003979 | 0.004079 | +2.52% |
| validation MAE | 0.036015 | 0.037294 | +3.55% |

The aggregate gain is intensity-selective. Global CSI changes by +0.03% at
threshold 16, -2.11% at 74, -1.65% at 133, +18.55% at 160, +32.93% at 181
and +63.43% at 219. Mean CSI is lower at 5 minutes (-0.95%) but higher at
every lead from 10 through 60 minutes, with the largest relative gain at
25 minutes (+7.92%).

The mechanism is not a free improvement. At thresholds 160/181/219, POD rises
by 23.46%/39.13%/72.09%, while SUCR falls by 14.44%/20.21%/39.86%. Thus the
method recovers severe echoes by accepting more false alarms. At the 60-minute,
219 threshold the baseline predicts zero positive pixels, whereas the method
predicts 42,543 against 617,417 observed pixels. This is direct evidence of
reduced abstention, but also shows substantial remaining underprediction.

## Interpretation boundary

This is the first positive result at native resolution and full train/validation
coverage. It is not yet a publishable superiority claim because it is one seed,
one epoch and validation-only. The MSE/MAE regression and severe-threshold SUCR
cost must be reported rather than hidden behind mean CSI.

## Resume commands

Run the baseline first:

```bash
python scripts/train_paper_simvp.py \
  --data-root /home/amon/zyx/dataset/sevir_data \
  --manifest artifacts/local/sevir_official_manifest.csv \
  --output-dir artifacts/local/exp025_paper_simvp_baseline_seed0_384 \
  --resolution 384 \
  --batch-size 1 \
  --epochs 3 \
  --learning-rate 0.005 \
  --tail-area-weight 0 \
  --tail-temperature-raw 10 \
  --tail-thresholds 160 181 219 \
  --selection-metric mcsi_global \
  --resume artifacts/local/exp025_paper_simvp_baseline_seed0_384/last.pt \
  --log-every 500 \
  --seed 0 \
  --workers 2
```

Then run the frozen method:

```bash
python scripts/train_paper_simvp.py \
  --data-root /home/amon/zyx/dataset/sevir_data \
  --manifest artifacts/local/sevir_official_manifest.csv \
  --output-dir artifacts/local/exp025_paper_simvp_tail_seed0_384 \
  --resolution 384 \
  --batch-size 1 \
  --epochs 3 \
  --learning-rate 0.005 \
  --tail-area-weight 0.0003 \
  --tail-temperature-raw 10 \
  --tail-thresholds 160 181 219 \
  --selection-metric mcsi_global \
  --resume artifacts/local/exp025_paper_simvp_tail_seed0_384/last.pt \
  --log-every 500 \
  --seed 0 \
  --workers 2
```

The output-directory names above must match the directories used for epoch 1.
If the local names differ, substitute the existing names consistently in both
`--output-dir` and `--resume`; do not move or rename checkpoints mid-run.

## Stop and acceptance rules

- Stop on a non-finite objective, signature mismatch or corrupted history.
- Do not change learning rate, loss weight or thresholds between resume calls.
- Accept the seed-0 pair for test evaluation only after both reach epoch 3 and
  each `best.pt` is selected independently by validation `mcsi_global`.
- Do not evaluate test until the final seed/model set is frozen.

