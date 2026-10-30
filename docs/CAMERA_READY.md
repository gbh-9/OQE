# Camera-ready reproduction update

Updated 2026-10-30. The original V23 code and reference summaries are preserved.
This update adds the frozen-model measurements used in camera-ready Table VI;
it does not add training or claim end-to-end acceleration.

## Rebuild the table without training

From the repository root, with any supported Python 3 installation:

```bash
python scripts/summarize_camera_ready.py
```

This reads `reference/camera_ready_20261029/` and creates
`results/camera_ready_tables/deployment.tex` and `REPORT.md`. The output directory
must not exist; use `--output-dir results/another_table_run` to preserve prior output.
The generated LaTeX table is the camera-ready Table VI; it requires booktabs and
textcomp in its containing manuscript. No PyTorch, datasets, or checkpoints are
needed to regenerate this table.

The generator is a path/CLI adaptation of the research workspace's
`camera-ready/build_deployment_20261029.py`. Table numbers and formatting are
unchanged. The representation block contains three-seed clean accuracy and tail
activity, whereas timing uses one frozen checkpoint per encoding. The original
mixed historical timing table was not imported; only its representation fields
were retained as `representation_summary.csv`.

## Rerun the inference benchmark

Prerequisites: installed requirements, Fashion-MNIST downloaded through
`scripts/prepare_datasets.py`, an NVIDIA CUDA GPU and `nvidia-smi` on PATH, and
the following two trusted checkpoints under `results/oqe_v6_main/checkpoints/`:

- `fashion_mnist__oqe__k7__canonical__s20261010__e3.pt`
- `fashion_mnist__single_hybrid__k7__canonical__s20261010__e3.pt`

The benchmark does not train. Checkpoints are not distributed in this repository.
If they are not already available, the existing training commands that produce
these files are:

```bash
python scripts/run_oqe_v6_main.py --dataset fashion_mnist --arm oqe --seed 20261010
python scripts/run_oqe_v6_main.py --dataset fashion_mnist --arm single_hybrid --seed 20261010
```

Only load trusted checkpoints: this runner uses PyTorch pickle-backed loading.
Regenerated checkpoints need not match the original hashes across environments.
With checkpoints in place, run:

```bash
python scripts/benchmark_oqe_camera_ready.py --output results/camera_ready_new_run
python scripts/summarize_camera_ready.py --run-dir results/camera_ready_new_run --output-dir results/camera_ready_new_table
```

Both output directories must be new. The benchmark is the measured research script
copied without changes; its source hash is recorded in the reference metadata.
It measures OQE ordinary/generic/suffix and Single ordinary/grouped in one session.
CPU uses 1 warm-up/5 repetitions; GPU uses 5/20. Each repetition shuffles the five
methods with a fixed seed. FP32 eval/no_grad, model reset inside the timed call,
CUDA synchronization at timing boundaries, and resident inputs/checkpoints are
used. Data loading and host-to-device transfer are excluded. The reconstruction
microbenchmark starts from synthetic, preformed class aggregates.

GPU incremental peak allocation is measured separately, above the post-reset live
allocation baseline. It is not total device memory, reserved memory, or solely
metadata. Logical key bytes and theoretical packed bits are separate columns.
Match the backend flags and CPU thread count (10 in the recorded run) in
`metadata.json` when comparing measurements. The original runtime used Python
3.13.5, PyTorch 2.7.1+cu118, CUDA 11.8, and an RTX 4060 Laptop GPU. Background
desktop load was not isolated. Absolute timing will vary on other systems.

## Evidence and interpretation

`reference/camera_ready_20261029/` contains:

- `timing_summary.csv`, `timing_samples.csv`: summary and every measured repetition.
- `metadata.json`: environment, timing boundary, source/input/checkpoint hashes.
- `equivalence_checks.json`: 12 first-batch B1/B128 comparisons, all with zero
  prediction mismatches and zero final-rate difference; not a full-test accuracy rerun.
- `deployment.tex`: frozen paper table, used to check regeneration.
- `representation_summary.csv`: accuracy, activity, and logical key storage.
- `independent_deletion_all_p.csv`, `exactly_one_anchor_summary.csv`: existing
  frozen-model deletion summaries used by the camera-ready revision.
- `REPORT.md`: unchanged historical benchmark report; its final pending-figure
  sentence reflects the 2026-10-29 snapshot, not the current manuscript status.

Suffix reconstruction is faster than generic reconstruction in the measured
settings, but full grouped forwarding remains slower than ordinary forwarding.
The camera-ready results do not support an end-to-end speedup. Existing deletion
results must retain their native-corruption/decoder/anchor boundaries.

No datasets, checkpoints, manuscript PDFs/sources, credentials, or private machine
paths are included in this update. Repository visibility is not changed by this
sync. The authors still need to choose a license before the intended public release.
