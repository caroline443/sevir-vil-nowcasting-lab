# RTX 5090 migration and convergence plan

Date: 2026-07-31

## Important scheduler constraint

Do not resume the three-epoch `last.pt` files with a larger `--epochs` value.
`train_paper_simvp.py` intentionally stores the configured total epoch count in
the checkpoint signature, and its OneCycleLR schedule has already completed at
epoch 3. Changing the total steps after completion would not be an exact
continuation.

The convergence experiment must be a fresh paired training run with a longer
total budget declared before training. Existing checkpoints remain archived
evidence but are not required on the RTX 5090.

## Expected speed

The A4000 native runs measured approximately 8--9 hours per complete
train-plus-validation epoch in the normal seed-0 runs, with slower seed-1
invocations reaching roughly 13--14 hours per epoch. Peak allocated memory at
batch size 1 was about 9.8 GB.

The RTX 5090 has 32 GB memory and much higher compute and memory bandwidth. A
conservative planning estimate for this workload is:

- batch size 1: 2--3 hours per epoch, about 3--4x A4000 throughput;
- batch size 2: 1.3--2.2 hours per epoch if data loading keeps up, about 4--6x
  sample throughput;
- use a 200-train-batch plus 50-validation-batch resource gate to replace these
  estimates with a measured server-specific value before paying for a full run.

Do not assume the vendor peak-AI-TOPS ratio is the training speedup. HDF5 input,
small-batch convolution utilization, CPU speed, storage latency and PyTorch
kernels can materially reduce it.

## Minimum files to move

### Required

1. This Git repository.
2. `artifacts/local/sevir_official_manifest.csv`.
3. The raw VIL HDF5 tree referenced by the manifest:
   `/home/amon/zyx/dataset/sevir_data/vil/`.
4. Preferably `/home/amon/zyx/dataset/sevir_data/CATALOG.csv` for audit and
   manifest rebuilding, although the training loader does not read it.

The manifest stores HDF5 paths relative to `--data-root`, so the server may use
any absolute root as long as it contains the same `vil/...` layout.

### Optional archive

The following four directories are useful for provenance but are not required
for fresh convergence training:

- `artifacts/local/exp025_simvp_baseline_seed0_384/`;
- `artifacts/local/exp025_simvp_tail_seed0_384/`;
- `artifacts/local/exp026_simvp_baseline_seed1_384/`;
- `artifacts/local/exp026_simvp_tail_seed1_384/`.

Each directory should contain `best.pt`, `last.pt`, `history.json`,
`summary.json`, `best_validation_metrics.json` and per-epoch validation JSON.

### Not required

- ABI satellite channels, lightning data or non-VIL SEVIR modalities;
- prior 128x128 diagnostic artifacts;
- final-test event-statistic NPZ files;
- Python virtual-environment directories;
- pip caches or compiled `__pycache__` files.

## Transfer examples

Replace `USER`, `HOST` and server paths with the rental provider values.

```bash
rsync -aP --partial --info=progress2 \
  /home/amon/zyx/dataset/sevir_data/vil/ \
  USER@HOST:/data/sevir_data/vil/

rsync -aP \
  /home/amon/zyx/dataset/sevir_data/CATALOG.csv \
  USER@HOST:/data/sevir_data/CATALOG.csv

rsync -aP \
  artifacts/local/sevir_official_manifest.csv \
  USER@HOST:~/sevir-vil-nowcasting-lab/artifacts/local/
```

If the provider offers attached object storage or a preloaded SEVIR image,
prefer that over repeatedly uploading the VIL tree for each short-lived
instance.

## Environment gate

Use a provider image with PyTorch built for CUDA 12.8 or newer. RTX 5090 is
Blackwell (`sm_120`); older CUDA wheels may install successfully but cannot run
kernels for the GPU. The working A4000 versions, PyTorch 2.8.0 plus CUDA 12.8,
are suitable in principle.

After cloning the repository:

```bash
python -m pip install --no-deps -r requirements-openstl.txt
python -m pip install -r requirements-data.txt

python scripts/collect_environment.py \
  --cuda-smoke-test \
  --output artifacts/local/rtx5090_environment.json

python scripts/inspect_sevir_layout.py \
  --data-root /data/sevir_data \
  --output artifacts/local/rtx5090_sevir_layout.json
```

Verify that the environment report identifies an RTX 5090, reports compute
capability 12.0, and passes the CUDA smoke test before starting paid training.

## Cost-control order

1. Transfer VIL data and manifest.
2. Pass environment, layout and official-SimVP smoke tests.
3. Measure batch size 1 and 2 resource gates.
4. Freeze one batch size before formal training.
5. Run the seed-0 MSE and SEA pair from scratch.
6. Inspect validation convergence only.
7. Run seed 1 only if seed 0 retains a meaningful effect at the longer budget.
8. Keep the test split inaccessible until the longer-budget protocol is
   permanently frozen.
