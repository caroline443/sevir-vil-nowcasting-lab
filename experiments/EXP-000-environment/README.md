# EXP-000: A4000 environment audit

Status: planned

## Question

Which Python, PyTorch, CUDA and GPU-memory constraints must the reproducible baseline support on the laboratory A4000 machine?

## Why this comes first

The baseline implementation and dependency versions should be selected from observed machine constraints. Installing an arbitrary historical environment first may create avoidable CUDA and package conflicts.

## Run

From the repository root:

```bash
python scripts/collect_environment.py \
  --output artifacts/local/a4000_environment.json
```

To include a small CUDA allocation test:

```bash
python scripts/collect_environment.py \
  --cuda-smoke-test \
  --output artifacts/local/a4000_environment.json
```

The smoke test allocates only a small tensor and performs a matrix multiplication. It is not a stress test.

## Return

Share the JSON after checking the optional `platform` field and any paths shown by package tooling. The script does not intentionally collect usernames, hostnames, environment variables or GPU serial numbers. The local artifact is ignored by Git; a reviewed copy may be saved as `experiments/EXP-000-environment/result.json`.

## Success condition

- Python version recorded;
- PyTorch import status recorded;
- CUDA availability and CUDA runtime recorded;
- GPU model and total memory recorded;
- optional matrix multiplication succeeds.

## Stop condition

If importing PyTorch fails, do not install packages ad hoc. Return the error recorded in the JSON so the baseline environment can be specified once.
