# Frozen-checkpoint deployment benchmark (2026-10-29)

No training; all 5 paths measured together. CPU warmup 1/repeats 5; GPU warmup 5/repeats 20. FP32, eval/no_grad, reset inside timed call, synchronization at timing boundaries. Input and checkpoints are resident; data loading/H2D excluded. Timing repetitions are not independent training seeds. Desktop background load is not isolated; raw samples and dispersion are retained.

| Device | Batch | OQE ordinary | OQE generic | OQE suffix | Single ordinary | Single grouped | Suffix/ordinary OQE |
|---|---:|---:|---:|---:|---:|---:|---:|
| cpu | 1 | 5.783 | 8.281 | 9.628 | 6.284 | 8.742 | 1.665 |
| cpu | 128 | 145.157 | 282.083 | 232.347 | 153.565 | 216.579 | 1.601 |
| cuda:0 | 1 | 5.889 | 7.467 | 6.908 | 5.363 | 7.413 | 1.173 |
| cuda:0 | 128 | 8.412 | 20.809 | 14.892 | 8.336 | 15.342 | 1.770 |

## Reconstruction medians (ms)

| Device | Batch | OQE generic | OQE suffix | Single placement |
|---|---:|---:|---:|---:|
| cpu | 1 | 0.865 | 0.109 | 0.083 |
| cpu | 128 | 60.219 | 9.567 | 3.334 |
| cuda:0 | 1 | 0.390 | 0.196 | 0.114 |
| cuda:0 | 128 | 7.146 | 1.163 | 0.623 |

## Memory and interpretation

| Path | GPU extra peak MB, B1 | GPU extra peak MB, B128 |
|---|---:|---:|
| OQE ordinary | 1.537 | 199.836 |
| OQE generic grouped | 1.505 | 195.822 |
| OQE suffix grouped | 1.505 | 195.822 |
| Single ordinary | 1.537 | 199.836 |
| Single grouped | 1.505 | 195.822 |

1. Suffix reconstruction is faster than generic reconstruction in all 4 measured settings. This is a stage-level result.
2. Complete grouped forwarding remains slower than ordinary forwarding. Dense ordinary paths do not exploit the event-count difference as sparse kernels would; lower event activity alone does not establish proportional latency savings.
3. Extra peak allocation includes all live transient tensors/output above a reset baseline, not just metadata. Similar peaks across encodings are consistent with shared dense tensor shapes. No hardware energy claim follows.
4. All 12 matched-path batch checks had zero prediction and final-rate differences; these checks cover first B=1/128 batches only, not a new full-test accuracy evaluation.

## Instrumentation changelog

| File | Change |
|---|---|
| scripts/benchmark_oqe_camera_ready.py | New standalone wrapper; frozen load, randomized timing, separate memory pass, local Single grouped path, source/checkpoint hashes. Existing model/executor files unchanged. |
| camera-ready/build_deployment_20261029.py | New CSV-driven joint table/report generator. |
| camera-ready/manuscript/tables/deployment.tex | Generated updated deployment table. |

The preliminary run `oqe_camera_ready_20261029` is excluded: its Single wrapper retained grouping temporaries during downstream execution. The final helper releases them at the same boundary as OQE; all paths were then remeasured together in this final cohort. Historical v23 timings remain preserved and are not mixed into the new table.

No retraining or new architecture is planned. Remaining user input: vector PDF method overview; blank authors still pending.
