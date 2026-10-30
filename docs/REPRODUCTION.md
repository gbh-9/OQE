# Reproduction guide

For the five-path deployment comparison and joint table, follow
[the execution benchmark guide](CAMERA_READY.md). The stage/scaling commands below
retain their original protocol; historical timing files are not combined with the
matched five-path cohort.

## Environment

Install the dependencies from the repository root. `requirements.txt` lists only direct runtime/test dependencies. The manuscript timing environment was:

| Item | Recorded value |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| PyTorch | 2.7.1+cu118 |
| CUDA runtime | 11.8 |
| Precision | FP32 |
| Mode | `model.eval()` and `torch.no_grad()` |

For the **GPU timing protocol**, use five warm-up runs, then 20 timed repetitions.
Timing uses `time.perf_counter()` with CUDA synchronization at measurement boundaries.
Summary CSVs report median, mean, and population standard deviation. The matched
five-path benchmark also retains raw repetitions, from which IQR, minimum, and
maximum can be calculated. Batch sizes are 1 and 128.

The stage/scaling runner records median, mean, population standard deviation, and
repetition count in its CSV output. The matched five-path runner additionally retains
all timing samples. CPU measurements use one warm-up and five repetitions; GPU
measurements use five warm-ups and 20 repetitions. Keep new runs separate from the
frozen paper references.

## Dataset layout

Run:

```bash
python scripts/prepare_datasets.py
```

Expected directories are `data/fashion_mnist`, `data/kmnist`, `data/cifar10`, and `data/cifar100`. No dataset is committed.

## Main matched comparison

```bash
python scripts/run_oqe_v6_main.py --dataset all --arm all --include-non-nested --batch-size 128
python scripts/summarize_oqe_v6.py
```

Seeds are `20261010`, `20261011`, and `20261012`. Fashion-MNIST/KMNIST train for 3 epochs; CIFAR-10/CIFAR-100 train for 20 epochs. The fixed `K=7` configuration has six tail steps and two analog anchors. Main output is written under `results/oqe_v6_main/`; checkpoints are required by downstream deletion and executor audits.

For a single run, for example CIFAR-10 OQE:

```bash
python scripts/run_oqe_v6_main.py --dataset cifar10 --arm oqe --seed 20261010 --rgb-epochs 20
```

## Anchor studies

Fashion-MNIST 0/1-anchor models are trained and two-anchor main records are indexed by:

```bash
python scripts/run_oqe_anchor_models.py --batch-size 128
```

KMNIST and CIFAR-10 zero-anchor extensions:

```bash
python scripts/run_oqe_v18_zero_anchor.py --phase all --batch-size 128
```

Corrected exactly-one active-event evaluation over 0/1/2 anchors:

```bash
python scripts/run_oqe_anchor_erasure_corrected.py --batch-size 128
```

The corrected implementation materializes each encoding's tail before deletion. OQE count-decodes to `max(k-1,0)`; Matched Single-Spike becomes the all-zero tail. Anchor construction is exactly `[tail]`, `[x, tail]`, or `[x, x, tail]`.

## Structural controls and deletion

Threshold policy, `K` sweep, and canonical/noncanonical order:

```bash
python scripts/run_oqe_v5_breadth.py --suite all
```

Activity-matched non-nested `Q_k` checkpoints are generated with `--include-non-nested` in the main runner. Their deletion study is:

```bash
python scripts/run_oqe_v12_followup.py
```

Four-dataset independent Bernoulli deletion and exactly-one-event deletion:

```bash
python scripts/run_oqe_v6_event_loss.py
```

All these evaluation paths require the matching trained checkpoints. The release does not include checkpoint binaries.

## Executor and benchmarks

```bash
python scripts/benchmark_oqe_v6_suffix.py --mode microbench
python scripts/benchmark_oqe_v6_suffix.py --mode full
python scripts/audit_oqe_v6_suffix_executor.py
```

Run `python scripts/benchmark_oqe_v6_suffix.py --help` and `python scripts/audit_oqe_v6_suffix_executor.py --help` for device/batch options. Generic grouped reconstruction includes every class-word coefficient required by the registered codebook. Suffix reconstruction targets the same currents through reverse cumulative additions. The scope begins only after `A_k` has been formed.

## Sanity references

The `reference/` directory contains frozen aggregate CSV/JSON files and the matched
deployment timing samples. It omits training logs and checkpoint tensors. These
files allow table regeneration and comparison with reported results; checkpoint
inference still requires trained weights. Original artifact identifiers are
retained for provenance.
