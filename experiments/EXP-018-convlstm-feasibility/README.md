# EXP-018: official ConvLSTM feasibility gate

Status: `completed`

## Question

Can the official OpenSTL ConvLSTM recurrent backbone run a real-data BF16
training step at batch 8 and 128² on the 17 GB A4000?

This is only a resource and interface smoke test. It is required before writing
a complete recurrent trainer or authorizing a baseline/tail pair.

## Configuration

The model uses OpenSTL's standard ConvLSTM configuration: four 128-channel
layers, 5×5 filters, patch size 4 and no layer normalization. Input and zero
future placeholders are patchified exactly for the official model interface.
The smoke uses a zero scheduled-sampling mask so it measures the fully
autoregressive memory path. A later trainer must state and freeze its scheduled
sampling policy.

## Command

```bash
python scripts/smoke_openstl_convlstm.py \
  --data-root /home/amon/zyx/dataset/sevir_data \
  --manifest artifacts/local/sevir_official_manifest.csv \
  --output artifacts/local/exp018_convlstm_b8_128_smoke.json \
  --resolution 128 \
  --batch-size 8 \
  --steps 1 \
  --patch-size 4 \
  --workers 2
```

## Decision rule

- Require exact output shape `[8,12,1,128,128]` and finite loss.
- Require peak allocated memory below 15 GB to retain a safety margin.
- Reject A4000 batch-8 training if a step exceeds five seconds; recurrent
  4000-update pairs would be disproportionate to the current evidence.
- The first output records the installed source hash. Pin that hash before a
  full recurrent trainer is committed.

## Result

The gate passed with substantial margin:

- exact output shape `[8,12,1,128,128]` and finite loss;
- 15,083,520 parameters;
- 2,228,940,800 bytes (2.23 GB decimal) peak allocated memory;
- 0.703 seconds for one BF16 forward/backward/update step;
- installed ConvLSTM source SHA-256
  `fefbdffed8ef3800a53eae41dfbfdd0718e962734e0f94b7b90dd441297b40ee`.

The observed source hash is now pinned. EXP-019 is authorized to implement the
official linear scheduled-sampling policy and run a bounded seed-0 pair.
