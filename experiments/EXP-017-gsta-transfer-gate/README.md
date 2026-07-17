# EXP-017: gSTA transfer gate

Status: `completed`

Current decision: the fixed tail component transfers to gSTA under the bounded
seed-0 protocol without retuning.

## Question

Does the fixed SoftExceedanceAreaLoss mechanism transfer from SimVP-v1 IncepU
to the different gSTA temporal translator used by SimVP-v2 without retuning?

This is a cross-translator gate, not yet a claim of cross-backbone
generalization. It tests whether the observed gain is tied to one Inception
translator implementation.

## Frozen settings

- official train/validation manifest and 13→12 windows;
- 128² bounded protocol, BF16, batch 8 and seed 0;
- 4000 updates and 200 validation batches after smoke-test acceptance;
- raw thresholds 160/181/219, temperature 10 and weight 0.0003;
- no weight or temperature probe on gSTA.

## First command: 100-update baseline smoke test

```bash
python scripts/train_openstl_simvp.py \
  --data-root /home/amon/zyx/dataset/sevir_data \
  --manifest artifacts/local/sevir_official_manifest.csv \
  --output-dir artifacts/local/exp017_gsta_smoke100_seed0_128 \
  --model-type gSTA \
  --resolution 128 \
  --batch-size 8 \
  --epochs 1 \
  --max-train-batches 100 \
  --max-val-batches 20 \
  --learning-rate 0.005 \
  --amp-dtype bfloat16 \
  --seed 0 \
  --workers 2
```

## Smoke decision rule

- Require 100 optimizer updates, no non-finite loss, no FP32 fallback and no
  skipped update.
- Require peak memory below the A4000 limit with useful margin.
- If stable, authorize one 4000-update MSE baseline and its paired tail-area
  run using identical seed and data order.

## Smoke result

The gSTA smoke test passed all gates:

- 100/100 optimizer updates, zero skipped updates and zero FP32 fallbacks;
- 18,706,497 parameters;
- 9,917,962,240 bytes (9.92 GB decimal) peak allocated GPU memory;
- 38.68 seconds wall time;
- finite training and validation outputs.

The 100-update CSI values are not method evidence. They are intentionally
discarded as undertrained smoke-test metrics. The paired 4000-update gate is
authorized below.

## Authorized baseline command

```bash
python scripts/train_openstl_simvp.py \
  --data-root /home/amon/zyx/dataset/sevir_data \
  --manifest artifacts/local/sevir_official_manifest.csv \
  --output-dir artifacts/local/exp017_gsta_baseline_seed0_128 \
  --model-type gSTA \
  --resolution 128 \
  --batch-size 8 \
  --epochs 1 \
  --max-train-batches 4000 \
  --max-val-batches 200 \
  --learning-rate 0.005 \
  --amp-dtype bfloat16 \
  --seed 0 \
  --workers 2
```

## Authorized tail command

```bash
python scripts/train_openstl_simvp.py \
  --data-root /home/amon/zyx/dataset/sevir_data \
  --manifest artifacts/local/sevir_official_manifest.csv \
  --output-dir artifacts/local/exp017_gsta_tail_area_seed0_128 \
  --model-type gSTA \
  --resolution 128 \
  --batch-size 8 \
  --epochs 1 \
  --max-train-batches 4000 \
  --max-val-batches 200 \
  --learning-rate 0.005 \
  --tail-area-weight 0.0003 \
  --tail-temperature-raw 10 \
  --tail-thresholds 160 181 219 \
  --amp-dtype bfloat16 \
  --seed 0 \
  --workers 2
```

## Pair decision rule

Continue the applied-Q2 path only if the fixed tail component improves severe
CSI and long-lead area survival on gSTA without a materially larger MSE penalty.
Failure stops the generic-loss claim and returns the project to an IncepU-only
case study.

## Tail run received

The seed-0 gSTA tail run completed 4000/4000 updates with zero skipped updates
or FP32 fallbacks. It used 9.92 GB peak allocated memory and took 3359 seconds.
Validation mean CSI is 0.34891 and MSE is 0.002528. At 60 minutes, predicted to
observed area ratios are 16.28%, 11.80% and 15.45% for thresholds 160, 181 and
219. Unlike the undertrained smoke test, this is a valid bounded-protocol model
result.

Those absolute scores alone did not establish transfer because gSTA can differ
from IncepU even under MSE. They were held provisional until the paired baseline
below was received. See `tail-seed0-result.json` for the compact tail record.

## Final paired result

The paired baseline was received on the identical validation subset. Tail-area
training raises overall mean CSI from 0.31982 to 0.34891 (+9.09% relative) while
MSE changes from 0.0024975 to 0.0025279 (+1.21%).

CSI improves at all 12 leads for thresholds 133, 160, 181 and 219 (48/48
comparisons). Lead-mean CSI gains are +16.1%, +34.2%, +49.8% and +65.4%,
respectively. POD gains at 160/181/219 are +46.8%, +64.8% and +88.6%.

The mechanism and cost match the IncepU evidence: severe-event abstention is
reduced, while SUCR falls at 160 and 181. Lead-mean SUCR changes by -23.5%,
-27.0% and -4.1% at 160/181/219. At 60 minutes the baseline predicts no pixels
at thresholds 181 and 219; tail training restores forecast/observed area ratios
to 11.80% and 15.45% and obtains non-zero CSI.

Low thresholds remain essentially unchanged: mean CSI changes by -0.11% at 16
and +0.001% at 74. The effect is therefore tail selective rather than a broad
score shift.

## Final decision

- The cross-translator transfer gate passes.
- SoftExceedanceAreaLoss is retained as a generic SimVP training component, not
  an IncepU-only case study.
- gSTA is not a fully independent backbone. One recurrent-model gate is still
  desirable before claiming broad architecture generality.
- Multi-seed gSTA replication is deferred until a recurrent smoke test shows
  whether the broader generic-loss path is computationally feasible.

See `paired-result-analysis.json` for the exact summary.
