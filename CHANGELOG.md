# Repository history

## 2026-10-30 — Paper-oriented documentation

- Organized the homepage around the accepted paper, method overview, results,
  installation, and reproduction tasks.
- Added the author-provided overview figure and a vector PDF companion.
- Added a result guide linking each comparison to its frozen numerical evidence.
- Retained original script/reference identifiers to keep commands, hashes, and
  checkpoint paths stable. Historical V23 labels identify the submitted-paper
  snapshot, not an unfinished public API.
- Preserved detailed packaging records under `docs/history/`.

## 2026-10-30 — Deployment evidence

- Added the same-session five-path CPU/GPU benchmark, all timing samples,
  incremental allocation, batch-level equivalence checks, and deletion summaries.
- Added a portable generator for the joint deployment table.
- Preserved the original training code and result references; no retraining.

## Initial reproduction snapshot

- Main four-dataset experiments, matched controls, anchor studies, event deletion,
  route grouping, suffix reconstruction, and structural tests.
- Original numerical artifacts retain `v23` in their filenames for traceability.
