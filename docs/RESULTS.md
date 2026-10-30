# Results

The results below follow the paper's fixed protocols. Training accuracy summaries
use seeds 20261010, 20261011, and 20261012. Timing repetitions measure runtime
variation; they are not additional training seeds.

## Classification

The [README main table](../README.md#main-results) reports the four-dataset,
two-anchor comparison from
[`v23_main_paper_aggregate.json`](../reference/v23_main_paper_aggregate.json).
All methods share the registered backbone and dataset-specific training budget:
3 epochs on Fashion-MNIST/KMNIST and 20 on CIFAR-10/CIFAR-100. The reference file
retains its original name to preserve the experiment-to-artifact mapping.

With the analog anchors removed, clean accuracy (%) is:

| Dataset | Matched Single-Spike | OQE | OQE − Single (pp) |
|---|---:|---:|---:|
| Fashion-MNIST | 86.35 ± 0.33 | 88.53 ± 0.06 | +2.18 |
| KMNIST | 88.79 ± 0.08 | 91.07 ± 0.08 | +2.28 |
| CIFAR-10 | 73.09 ± 0.50 | 72.99 ± 0.29 | −0.10 |

Values are three-seed means ± population standard deviations, computed from
[`v23_zero_anchor_results.csv`](../reference/v23_zero_anchor_results.csv).
The two-anchor and anchor-free tables are separate experimental settings, not
interchangeable baselines. OQE's anchor-free grayscale gains come with more tail
activity; in the two-anchor Fashion-MNIST setting OQE emits 1.499 tail events per
scalar versus 0.431 for the matched single-spike control.

## Execution

Five paths were measured in one session using the frozen seed-20261010
Fashion-MNIST checkpoints, K = 7, two anchors, FP32, resident input, and a shared
measurement protocol. Full-forward medians in milliseconds:

| Path | CPU B=1 | CPU B=128 | GPU B=1 | GPU B=128 |
|---|---:|---:|---:|---:|
| OQE ordinary | 5.783 | 145.157 | 5.889 | 8.412 |
| OQE generic grouped | 8.281 | 282.083 | 7.467 | 20.809 |
| OQE suffix grouped | 9.628 | 232.347 | 6.908 | 14.892 |
| Single ordinary | 6.284 | 153.565 | 5.363 | 8.336 |
| Single grouped | 8.742 | 216.579 | 7.413 | 15.342 |

Suffix reconstruction alone is faster than generic reconstruction in all four
measured settings. Full grouped forwarding remains slower than ordinary forwarding
because it also prepares keys, forms groups, and moves intermediate data.
Reconstruction times, GPU incremental allocation, input-key bytes, and all raw
samples are provided in the [joint table and evidence](../reference/camera_ready_20261029/).
The theoretical three-bit key bound at K = 7 is not a measured packed implementation.

Run `python scripts/summarize_camera_ready.py` to rebuild the complete joint table.
The [benchmark guide](CAMERA_READY.md) specifies reset/synchronization, warm-up,
repetitions, memory accounting, and hardware details. Measurements exclude dataset
loading and host-to-device transfer; absolute times are hardware-dependent.

## Event deletion

The checked-in summaries cover
[independent deletion over the full probability grid](../reference/camera_ready_20261029/independent_deletion_all_p.csv)
and [exactly-one-event deletion across anchor counts](../reference/camera_ready_20261029/exactly_one_anchor_summary.csv).
The native temporal word is corrupted before encoding-specific decoding and
canonicalization. Analog anchors remain intact in these protocols.

Independent deletion and exactly-one-event stress answer different questions.
Results depend on encoding, receiver, and dataset; count recovery should not be
interpreted as a universal classification robustness advantage. The reproduction
commands are listed in the [experiment map](EXPERIMENT_MAP.md).
