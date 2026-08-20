# V23 paper-to-code map

All principal training results use seeds `20261010`, `20261011`, and `20261012`. YAML files are readable protocol manifests; CLI arguments shown below are the executable configuration interface.

| Paper result | Code and configuration | Evaluation/output |
|---|---|---|
| 1. Main clean comparison: Direct/Rate/Latency/Matched Single-Spike/OQE on Fashion-MNIST | `scripts/run_oqe_v6_main.py --dataset fashion_mnist --arm all`; `configs/fashion_mnist/main_k7.yaml` | `scripts/summarize_oqe_v6.py`; `results/oqe_v6_main/` |
| 1. Main clean comparison on KMNIST | `scripts/run_oqe_v6_main.py --dataset kmnist --arm all`; `configs/kmnist/main_k7.yaml` | same summarizer/output |
| 1. Main clean comparison on CIFAR-10 | `scripts/run_oqe_v6_main.py --dataset cifar10 --arm all --rgb-epochs 20`; `configs/cifar10/main_k7.yaml` | same summarizer/output |
| 1. Main clean comparison on CIFAR-100 | `scripts/run_oqe_v6_main.py --dataset cifar100 --arm all --rgb-epochs 20`; `configs/cifar100/main_k7.yaml` | same summarizer/output |
| 2. Anchor-free Fashion-MNIST | `scripts/run_oqe_anchor_models.py` (`anchor_count=0`) | run records under `results/oqe_anchor_models/` and corrected evaluator |
| 2. Anchor-free KMNIST/CIFAR-10 | `scripts/run_oqe_v18_zero_anchor.py --phase all` | `zero_anchor_extended_results.csv` |
| 3. 0/1/2 analog-anchor ablation | `scripts/run_oqe_anchor_models.py`, then `scripts/run_oqe_anchor_erasure_corrected.py` | corrected anchor result and invariant CSVs |
| 4. Threshold-policy study | `scripts/run_oqe_v5_breadth.py --suite main` | `results/oqe_v5_breadth/`; V5 tag denotes this frozen study cohort, not a superseded deletion path |
| 5. `K` sweep | `scripts/run_oqe_v5_breadth.py --suite sweep` | same output directory |
| 6. Canonical vs noncanonical temporal order | `scripts/run_oqe_v5_breadth.py --suite permutation` | same output directory |
| 7. Activity-matched non-nested `Q_k` | `scripts/run_oqe_v6_main.py --dataset fashion_mnist --arm non_nested --include-non-nested` | `scripts/run_oqe_v12_followup.py`; `Q_k` has weight `k` but not canonical nesting |
| 8. Independent event deletion | `scripts/run_oqe_v6_event_loss.py` | `results/oqe_v6_event_loss/`; materialized-tail corruption and encoding-specific decode |
| 9. Exactly-one-event deletion | `scripts/run_oqe_v6_event_loss.py`; anchor-specific version in `scripts/run_oqe_anchor_erasure_corrected.py` | OQE `k→k-1` for `k>0`; Single-Spike silent word |
| 10. Route contraction | `src/oqe_v6_executor.py` (`ordinal_class_ids`, `classwise_tail_aggregates`) | `scripts/audit_oqe_v6_suffix_executor.py`; scope is after `A_k` formation |
| 11. Generic vs suffix reconstruction | `src/oqe_v6_executor.py`; `scripts/benchmark_oqe_v6_suffix.py --mode microbench` | reconstruction benchmark CSV |
| 12. `K`-scaling benchmark | `scripts/benchmark_oqe_v6_suffix.py --mode microbench` (registered `K` values 3, 5, 7, 9, 17, 33 at batch 1) | reconstruction benchmark CSV |
| 13. Complete-forward benchmark | `scripts/benchmark_oqe_v6_suffix.py --mode full` | full-forward benchmark CSV; no end-to-end speedup claim |
| 14. FP32/FP64 numerical audit | `scripts/audit_oqe_v6_suffix_executor.py` | per-path numerical audit under `results/oqe_v6_nestedness/` |

## Key implementation locations

- Codebooks, fixed encoder, matched model, shared backbone helpers: `src/oqe_v6.py`
- Exact grouping, generic reconstruction, suffix reconstruction, and executor paths: `src/oqe_v6_executor.py`
- Strict anchor construction: `src/oqe_anchor.py`
- Corrected deletion semantics: `scripts/run_oqe_anchor_erasure_corrected.py` and `scripts/run_oqe_v6_event_loss.py`
- Small checked-in reference summaries: `reference/`

The repository does not contain checkpoints, so rows requiring model inference become directly runnable only after their corresponding training commands finish.
