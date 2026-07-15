# SEVIR VIL Nowcasting Lab

Research and reproducible experiments for severe convective weather nowcasting with the SEVIR VIL modality.

This repository is deliberately **problem-first**. A model change is added only after a diagnostic experiment establishes a repeatable failure mode. The initial goal is therefore not to build a large architecture, but to obtain a trustworthy SimVP baseline and determine which problem is both real and tractable on a single NVIDIA RTX A4000.

## Current status

- Literature review: initial 2020–2026 map completed.
- Candidate research problems: documented but not yet selected.
- Compute target: one RTX A4000; paid RTX 5090 use only after a small experiment establishes value.
- EXP-001: compact pipeline baseline completed on the A4000.
- EXP-002: pinned official OpenSTL SimVP compatibility gate passed.
- EXP-003: native 384×384 memory and throughput gate ready to run.

## Repository layout

```text
docs/                 Research reports and decisions
experiments/          Numbered experiment cards and result files
scripts/              Small reproducibility utilities
artifacts/local/      Local outputs ignored by Git
```

## Run the first task on the A4000 machine

```bash
python scripts/collect_environment.py \
  --output artifacts/local/a4000_environment.json
```

The generated file is ignored by Git. After checking it, either paste its contents into the experiment discussion or copy a sanitized version to `experiments/EXP-000-environment/result.json` for review.

## Research rules

1. Use event-disjoint and time-aware train/validation/test splits.
2. Keep the baseline protocol fixed before comparing a method.
3. Every experiment must state a hypothesis and a stop condition.
4. Report lead-time and high-threshold metrics, not only aggregate MSE or CSI.
5. Record peak GPU memory, wall-clock time, configuration and random seed.
6. Do not combine multiple unvalidated innovations in one run.

## Documents

- [Literature map](docs/literature-map.md)
- [Problem-first roadmap](docs/problem-first-roadmap.md)
- [Experiment protocol](experiments/README.md)
- [EXP-000: A4000 environment audit](experiments/EXP-000-environment/README.md)
- [EXP-001: minimal SimVP baseline](experiments/EXP-001-simvp-baseline/README.md)
- [EXP-002: official OpenSTL SimVP compatibility gate](experiments/EXP-002-openstl-simvp/README.md)
- [EXP-003: native-resolution OpenSTL SimVP gate](experiments/EXP-003-openstl-384-gate/README.md)

## Implementation references

The local baseline is a compact, dependency-light implementation of the SimVP design for the asymmetric SEVIR task (13 input frames to 12 output frames). It is intended for controlled experiments, not as a claim of bit-for-bit equivalence with OpenSTL. Method and dataset protocol decisions are checked against the official [OpenSTL](https://github.com/chengtan9907/OpenSTL) and [Earthformer](https://github.com/amazon-science/earth-forecasting-transformer) repositories.
