# OQE V23 release audit

Status in this file is updated at the end of repository preparation.

## Scope scanned

- Source development directories: `src/`, `scripts/`, and relevant `tests/`.
- Validated result summaries under `results/oqe_v6_main`, `results/oqe_v6_nestedness`, `results/oqe_v6_event_loss`, `results/oqe_anchor_erasure_corrected`, and `results/oqe_v18_followup`.
- The V23 manuscript PDF was inspected as the reporting source; it is not copied into the repository.

## Retained

- Final OQE V6 codebooks, matched SNN, route grouping, generic reconstruction, and suffix reconstruction.
- Main four-dataset training/evaluation runner and frozen breadth-study runner.
- Materialized-tail event deletion, corrected anchor erasure, non-nested control, anchor-count training, executor audits, and benchmark runner.
- Lightweight structural tests, protocol manifests, documentation, and small V23 aggregate reference tables.

## Intentionally excluded

- All datasets, model checkpoints, experiment log dumps, profiler traces, caches, generated figures, manuscript sources/PDFs, environment dumps, and the original Git history.
- Historical TA-OQE exploratory branches unrelated to V23.
- `scripts/run_oqe_v12_anchor_ablation.py` and its old anchor result artifacts. Its evaluator incorrectly reused an OQE `k→k-1` shortcut for Matched Single-Spike. Only the unaffected anchor model construction/training logic was extracted into clean release modules; the corrected evaluator materializes and decodes each corrupted representation.

## Protocol and result checks

- Main aggregate facts match V23: Direct is best on all four datasets; mean Matched Single-Spike accuracy exceeds mean OQE accuracy on all four.
- Checked anchor-free reference means match V23: Fashion-MNIST OQE/Single `88.530/86.347`, KMNIST `91.073/88.793`, CIFAR-10 `72.990/73.090` percent, with paired differences `+2.183`, `+2.280`, and `-0.100` pp and directions `3/0/0`, `3/0/0`, and `1/0/2`.
- The OQE and Matched Single-Spike release paths share thresholds, ordinal class, tail length, anchors, total step budget, route key, backbone, optimizer, training budget, and seeds in the matched main protocol.
- Generic and suffix reconstruction target identical tail currents. For `K=7`, the post-grouping accounting is 36 generic coefficient incidences versus one initialization plus five additions.
- No paper number was overwritten and no manuscript file was changed.

## Known inconsistencies and limits

- The benchmark runner's V23 GPU branch implements 5 warm-ups and 20 repetitions. Its CPU branch uses 1 warm-up and 5 repetitions; this is disclosed in `REPRODUCTION.md` and must not be described as the GPU protocol.
- The checked-in benchmark CSV records median, mean, population standard deviation, and repetition count, but not IQR/min/max fields. V23 protocol documentation states that those secondary statistics were retained; reproducing those fields requires retaining raw timing samples or extending reporting without changing the timed operation.
- Full result regeneration requires public datasets and long training. Checkpoints are deliberately excluded, so deletion/executor inference cannot run directly from a fresh clone before training.
- `LICENSE_PENDING.md` is deliberate: public release is blocked until the authors select a license.

## Privacy and security

- Release source/docs/configs contain no author name, institution, laboratory, email address, machine username, hostname, IP address, SSH/SLURM account, API key, access token, password, W&B identity, or hard-coded absolute dataset/checkpoint path.
- Paths in runnable code are derived from the repository root.
- The required private repository URL is operational metadata and will be recorded below after creation.

## Reproducibility classification

- Direct without retraining: structural/unit tests, codebook checks, reconstruction equivalence on synthetic tensors, protocol/reference-table inspection.
- Requires public dataset download and training: main clean, anchor-free, anchor-count, and non-nested model results.
- Requires generated checkpoints: event deletion, corrected anchor erasure, full executor equivalence, and checkpoint-backed benchmark/audit paths.
- Not reconstructable from checked-in files alone: the exact original per-run model weights and raw timing samples, intentionally omitted for size/privacy and clean-release requirements.

## Final verification

- Lightweight tests: `PASS` — 19 passed.
- Python imports/CLI syntax checks: `PASS` — core imports, Python compilation, nine CLI help paths, and five shell wrappers checked.
- Identity/absolute-path scan: `PASS` — no identity record or machine absolute path found. The GitHub URL field remains pending.
- Secret scan: `PASS` — no credential-shaped value found. Keyword-only hits were explanatory prose and a local variable named `token`.
- Files above reasonable binary size: `PASS` — none above 1 MiB and no committed model/data binary.
- Repository size before `.git`: approximately 0.46 MB.
- Git status: `CLEAN` after the final audit commit.
- Private GitHub repository: `https://github.com/gbh-9/oqe-v23-reproduction` (`PRIVATE`, verified with the GitHub API).
- Camera-ready public release candidate: **NO** until the license is selected and final verification items above pass.
