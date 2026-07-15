# EXP-005: short official-SimVP diagnostic baseline

Status: ready for A4000 execution

## Problem statement

Before proposing a new loss or architecture, determine whether a partially trained official SimVP exhibits a repeatable, lead-time-dependent failure on intense VIL echoes. The concrete question is:

> After a bounded but nontrivial training run, does forecast quality degrade disproportionately at high thresholds and long lead times, and is that degradation accompanied by systematic underprediction of VIL intensity?

A positive result would define a measurable problem for later experiments. A negative or inconclusive result means more baseline training is required; it does not authorize an innovation claim.

## Protocol

- pinned official OpenSTL SimVP architecture and SEVIR configuration;
- official optimizer family: Adam, zero weight decay;
- official scheduler family: OneCycle with maximum learning rate `5e-3`;
- official event/time/window protocol and all six SEVIR thresholds;
- 128×128 bilinear-resized development inputs, batch size 8;
- exactly 1000 training updates and at most 200 validation batches;
- lead-time metrics at 5, 10, ..., 60 minutes.

The OneCycle schedule is compressed into this 1000-step pilot, so the resulting scores are diagnostic only and must not be compared with published full-training numbers. The 128×128 resizing also makes this a development protocol, not the final native-resolution evaluation.

## Run

After merging this experiment, run from the repository root in the existing compatibility environment:

```bash
git switch main
git pull --ff-only
source .venv-openstl/bin/activate

PYTHONPATH=src:scripts python scripts/train_openstl_simvp.py \
  --manifest artifacts/local/sevir_official_manifest.csv \
  --data-root /home/amon/zyx/dataset/sevir_data \
  --output-dir artifacts/local/exp005_diagnostic_128 \
  --resolution 128 \
  --batch-size 8 \
  --workers 2 \
  --epochs 1 \
  --max-train-batches 1000 \
  --max-val-batches 200
```

Return these three small files; the checkpoint stays local:

- `artifacts/local/exp005_diagnostic_128/summary.json`
- `artifacts/local/exp005_diagnostic_128/metrics.json`
- `artifacts/local/exp005_diagnostic_128/train_log.json`

## Success condition

- 1000 finite updates complete without OOM;
- OneCycle removes the early fixed-LR explosion seen in EXP-004;
- validation metrics are finite at every lead time;
- high-threshold target counts are sufficient to interpret CSI/POD;
- the output supports a specific failure statement, rather than merely “CSI is low.”

## Stop condition

- If batch size 8 OOMs, return the traceback and do not retry; batch size 4 will be prescribed while preserving all other settings.
- If loss diverges, diagnose optimizer/AMP behavior before changing the model.
- If 200 validation batches contain too few threshold-219 pixels, expand validation only; do not train longer merely to manufacture a diagnosis.
- Do not add attention, Mamba, SSIM, F1 loss or any new module in this experiment.
