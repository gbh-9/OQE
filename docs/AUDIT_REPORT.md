# Validation and reproducibility

## Checked implementation

The lightweight test suite covers codebook geometry, matched non-nested controls,
grouped-current reconstruction, and encoding-specific anchor/deletion behavior.

```bash
python -m pytest
```

The latest execution completed **19 tests successfully**. The deployment benchmark
CLI also passed its import/help check. The table generator reproduced the frozen
paper table without a training or timing rerun.

## Evidence levels

| Artifact | What it supports |
|---|---|
| Structural tests | Codebook and executor invariants on specified test cases |
| Main reference summaries | Reported three-seed classification comparisons |
| Timing samples and metadata | Measured stage/full-forward performance under the recorded protocol |
| 12 matched-batch equivalence checks | Identical predictions and final rates for the tested batches, not a new full-test evaluation |

Reconstruction equivalence is distinct from an end-to-end speedup. Likewise,
symbol-level deletion behavior and downstream classification accuracy are reported
separately. See [Results](RESULTS.md) and the [execution protocol](CAMERA_READY.md).

## Reproducing results

Tests and reference-table generation run without model checkpoints. Full training,
deletion evaluation, and frozen-model inference use the public datasets and the
checkpoint paths documented in [Reproduction](REPRODUCTION.md). Original checkpoint
binaries are not bundled, so the exact original model weights cannot be recovered
from the summaries alone.

Core model and executor sources were unchanged during the deployment update.
The benchmark metadata records source, input, and checkpoint hashes. Recorded
measurements and historical reference files are preserved rather than replaced by
new runs.

## Packaging checks

The synchronization checks found no credential-pattern or private-machine-path
matches in release content and no included datasets or checkpoint binaries.
These are scoped packaging checks, not a security guarantee or an independent
raw-data audit. License selection remains an author decision.

Detailed dated records are retained in [release history](history/release_notes_20261030.md).
