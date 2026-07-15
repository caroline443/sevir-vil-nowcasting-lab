# EXP-001: minimal SimVP baseline

Status: running — Stage A completed

## Question

Can a compact SimVP-style model complete a forward/backward step and a short SEVIR-VIL training run under the laboratory A4000's 16 GB memory limit using the 13→12, 128×128 development protocol?

This experiment validates infrastructure only. It is not an innovation experiment and its scores must not be presented as an OpenSTL reproduction.

## Implementation decision

The A4000 environment already provides Python 3.12, PyTorch 2.8.0 and CUDA 12.8. Historical Earthformer and PreDiff environments pin much older Python/PyTorch/CUDA stacks. EXP-001 therefore uses a compact local SimVP-style implementation rather than downgrading the working GPU environment.

## Stage A: synthetic CUDA test

No package installation or data is required:

```bash
PYTHONPATH=src python scripts/smoke_simvp.py \
  --batch-size 2 \
  --resolution 128 \
  --output artifacts/local/exp001_smoke.json
```

Return `artifacts/local/exp001_smoke.json` before starting real training.

Stage A completed successfully on 2026-07-15. The measured peak allocation was approximately 462 MiB for batch size 2, leaving substantial margin under the 12 GiB gate. See [`smoke-result.json`](smoke-result.json).

### Stage A success condition

- output shape is `[2, 12, 1, 128, 128]`;
- one mixed-precision backward/optimizer step succeeds;
- peak allocated memory is below 12 GiB, leaving margin for HDF5 loading and validation.

### Stage A stop condition

If the step fails or peak allocated memory exceeds 12 GiB, do not reduce scientific input/output lengths. First reduce batch size to 1, then reduce `hidden-temporal` from 192 to 128.

## Stage B: prepare a local manifest

Install only the data dependencies if they are absent:

```bash
python -m pip install -r requirements-data.txt
```

Before building a manifest, verify the downloaded dataset layout without exposing its absolute path:

```bash
python scripts/inspect_sevir_layout.py \
  --data-root /path/to/SEVIR \
  --output artifacts/local/sevir_layout.json
```

Return `artifacts/local/sevir_layout.json`. Continue only when the report confirms a `vil` HDF5 dataset with one event shaped as either `[384, 384, 49]` or `[49, 384, 384]`.

Create a time-based manifest from the raw SEVIR catalog. The dates below define a development split, not a claim of exact equivalence with another paper:

```bash
python scripts/build_sevir_manifest.py \
  --catalog /path/to/SEVIR/CATALOG.csv \
  --output artifacts/local/sevir_dev_manifest.csv \
  --train-end 2019-01-01 \
  --val-end 2019-06-01
```

The manifest splits events before expanding each event into windows, preventing windows from one event from crossing splits.

## Stage C: 100-batch pipeline test

```bash
PYTHONPATH=src python -m sevir_nowcasting.train \
  --manifest artifacts/local/sevir_dev_manifest.csv \
  --data-root /path/to/SEVIR \
  --output-dir artifacts/local/exp001_train100 \
  --resolution 128 \
  --batch-size 2 \
  --workers 2 \
  --epochs 1 \
  --max-train-batches 100 \
  --max-val-batches 20
```

## Stage C success condition

- 100 train batches and 20 validation batches finish without data errors or OOM;
- loss is finite;
- `metrics.csv`, `summary.json` and `last.pt` are written;
- peak memory and wall time are recorded.

## Stage C stop condition

Do not launch a full training run if:

- raw HDF5 orientation or catalog paths do not match the loader;
- loss becomes non-finite;
- data loading dominates wall time;
- the model cannot overfit a very small subset in a later sanity check.

## Required return files

- `artifacts/local/exp001_smoke.json` after Stage A;
- `artifacts/local/sevir_layout.json` before manifest generation;
- `artifacts/local/exp001_train100/summary.json` and `metrics.csv` after Stage C;
- the exact manifest command, but not private absolute data paths in a public commit.
