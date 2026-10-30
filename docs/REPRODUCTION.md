# Reproduction guide

For the camera-ready five-path deployment comparison and new table, follow
[CAMERA_READY.md](CAMERA_READY.md). The benchmark commands below retain the V23
protocol; historical timing files are not combined with the new cohort.

## Environment

Install the dependencies from the repository root. `requirements.txt` lists only direct runtime/test dependencies. The manuscript timing environment was:

| Item | Recorded value |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| PyTorch | 2.7.1+cu118 |
| CUDA runtime | 11.8 |
| Precision | FP32 |
| Mode | `model.eval()` and `torch.no_grad()` |

For the **GPU timing protocol**, use five warm-up runs, then 20 timed repetitions. Timing uses `time.perf_counter()`. CUDA is synchronized once after warm-up and after every timed call. The primary statistic is the median; retained secondary statistics are mean, population standard deviation, IQR, minimum, and maximum. Batch sizes are 1 and 128.

The checked-in validated runner records median, mean, population standard deviation, and repetition count in its CSV output. The V23 aggregate tables were produced from the frozen result cohort; do not replace those numbers with a later protocol-verification rerun. The runner's CPU branch historically uses one warm-up and five repetitions, whereas the GPU branch implements the V23 GPU protocol above.

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

The `reference/` directory contains small V23 aggregate CSV/JSON files. It intentionally omits per-run logs and checkpoint tensors. Their purpose is to confirm the expected cohort and reported summaries, not to bypass training.
