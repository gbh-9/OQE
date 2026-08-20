# OQE V23 reproduction repository

OQE is an ordinal temporal input representation for SNNs that retains the encoder-produced ordinal key for exact route grouping and uses nested temporal supports to enable suffix reconstruction.

This repository accompanies manuscript V23, *OQE: Ordered Temporal Coding for SNN Inputs with Exact Route Grouping and Suffix Reconstruction*. It is a clean reproduction tree: datasets, checkpoints, raw logs, development history, and superseded implementations are intentionally absent.

## Novelty boundary

- The cumulative/thermometer representation itself is not claimed as new.
- Exact grouping by a deterministic class key is not unique to OQE.
- The OQE-specific structure is the combination of ordinal temporal realization, retained route key, and nested-support suffix reconstruction.

The current prototype demonstrates an isolated reconstruction opportunity rather than an end-to-end acceleration. Its complexity reduction applies to **post-grouping class-time reconstruction, after the class aggregates `A_k` have been formed**. For `K=7`, generic reconstruction has 36 class-time coefficient incidences; suffix reconstruction uses one boundary initialization and five vector additions. The ordinal-key lower bound is 3 bits per scalar, while the current prototype may store keys as `int64`; 3 bits is not a measured memory footprint.

## Install

Python 3.11 is recommended.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The paper GPU timing environment was PyTorch `2.7.1+cu118`, CUDA runtime `11.8`, and an NVIDIA GeForce RTX 4060 Laptop GPU. Ordinary training can use another compatible PyTorch build; timing comparison should use the recorded environment.

## Prepare datasets

```bash
python scripts/prepare_datasets.py
```

This downloads Fashion-MNIST, KMNIST, CIFAR-10, and CIFAR-100 under `data/`. Training scripts deliberately use `download=False` after preparation.

## Reproduce

Main clean experiment, all four datasets and five arms:

```bash
python scripts/run_oqe_v6_main.py --dataset all --arm all --include-non-nested
python scripts/summarize_oqe_v6.py
```

Anchor-free and 0/1/2-anchor models:

```bash
python scripts/run_oqe_anchor_models.py
python scripts/run_oqe_v18_zero_anchor.py --phase all
python scripts/run_oqe_anchor_erasure_corrected.py
```

Independent and exactly-one-event deletion from trained checkpoints:

```bash
python scripts/run_oqe_v6_event_loss.py
python scripts/run_oqe_v12_followup.py
```

Generic versus suffix reconstruction and complete-forward benchmarks:

```bash
python scripts/benchmark_oqe_v6_suffix.py --mode microbench
python scripts/benchmark_oqe_v6_suffix.py --mode full
```

FP32/FP64 numerical-equivalence audit:

```bash
python scripts/audit_oqe_v6_suffix_executor.py
```

Lightweight structural tests (no training):

```bash
python -m pytest
```

See [reproduction details](docs/REPRODUCTION.md), the [paper-to-code map](docs/EXPERIMENT_MAP.md), and the [release audit](docs/AUDIT_REPORT.md). The YAML files under `configs/` are protocol manifests for readers; the validated CLI runners remain the executable source of truth.

## Known scope

- Clean and anchor studies use three fixed seeds and dataset-specific training budgets.
- The matched OQE/Single-Spike controls share thresholds, ordinal classes, tail length, anchors, total step budget, route key, backbone, optimizer, training budget, and seed protocol.
- Benchmark results isolate reconstruction after route grouping; complete-forward measurements do not show end-to-end speedup.
- Reference CSV/JSON files are small immutable summaries for sanity checking, not substitutes for checkpoints or raw datasets.
- No claim is made that OQE is universally more accurate, faster, or carries more ordinal information than Matched Single-Spike.
