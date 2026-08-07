# A4000 publication extension at 128x128

This unattended queue adds evidence that is complementary to the native
384x384 SimVP main experiment. It does not replace or rank against the native
paper protocol.

## Experiment matrix

The queue first runs two paired ConvLSTM replications:

1. ConvLSTM + MSE, seed 0;
2. ConvLSTM + MSE + SEA, seed 0;
3. ConvLSTM + MSE, seed 1;
4. ConvLSTM + MSE + SEA, seed 1.

ConvLSTM uses budget-aligned scheduled sampling that reaches zero teacher
forcing on the final update. Validation always uses free rollout. Every epoch
is evaluated on the complete validation split and checkpoints are selected by
`mcsi_global`.

The queue then runs six one-factor-at-a-time SimVP SEA ablations:

- threshold 181 alone instead of 160+181+219;
- sequence-mean area matching instead of per-lead matching;
- loss weights 1e-4 and 1e-3 around the frozen 3e-4 reference;
- sigmoid temperatures 5 and 20 around the frozen 10 reference.

The completed EXP-031 seed-0 SEA run is the unrepeated frozen reference.

## Launch

From the repository root on the A4000 machine:

```bash
nohup bash scripts/run_a4000_publication_extension_128.sh \
  > artifacts/local/a4000_publication_extension_launcher.log 2>&1 &
echo $!
```

The scripts skip completed runs and resume interrupted runs from `last.pt`.
Follow the outer log with:

```bash
tail -f artifacts/local/a4000_publication_extension_launcher.log
```

Success markers are written to:

- `artifacts/local/exp034_convlstm_128_queue/SUCCESS`;
- `artifacts/local/exp035_sea_ablations_128_queue/SUCCESS`;
- `artifacts/local/exp036_a4000_publication_extension_128/SUCCESS`.

Do not evaluate the test split from this queue. These are validation-selected
cross-backbone and ablation experiments.
