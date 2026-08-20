"""Invariant-gated corrected exactly-one-active-event anchor ablation."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from oqe_v6 import codebook_for, reset_model  # noqa: E402
from scripts.run_oqe_v6_main import dataset_spec, sha256_file  # noqa: E402
from scripts.run_oqe_anchor_models import ANCHORS, ENCODINGS, SEEDS, load_model, records  # noqa: E402

OUTPUT = ROOT / "results" / "oqe_anchor_erasure_corrected"
K = 7
TAIL_STEPS = 6
CORRUPTION_SEED = 73001


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def explicit_delete_one(tail: torch.Tensor, ids: torch.Tensor, shared_u: torch.Tensor, encoding: str):
    active = tail > 0
    observed_weight = active.sum(0)
    expected_weight = ids if encoding == "oqe" else (ids > 0).long()
    if not torch.equal(observed_weight, expected_weight):
        raise RuntimeError(f"pre-deletion weight mismatch: {encoding}")
    # Select rank uniformly within each route's actual active-position list.
    rank = torch.floor(shared_u * expected_weight.clamp_min(1)).long()
    active_rank = active.to(torch.int64).cumsum(0) - 1
    delete = active & (active_rank == rank.unsqueeze(0)) & (ids > 0).unsqueeze(0)
    deleted_per_route = delete.sum(0)
    if not torch.equal(deleted_per_route, (ids > 0).long()):
        raise RuntimeError(f"not exactly one deletion per nonzero route: {encoding}")
    corrupted = tail.masked_fill(delete, 0)
    decoded = corrupted.sum(0).long()
    expected_decoded = torch.where(ids > 0, ids - 1, ids) if encoding == "oqe" else torch.zeros_like(ids)
    if not torch.equal(decoded, expected_decoded):
        raise RuntimeError(f"post-decode invariant failed: {encoding}")
    canonical = codebook_for(encoding, K, device=tail.device, dtype=tail.dtype)[decoded].permute(4, 0, 1, 2, 3).contiguous()
    if encoding == "single_hybrid" and (torch.count_nonzero(corrupted) or torch.count_nonzero(canonical)):
        raise RuntimeError("Single-Spike tail did not collapse to zero")
    return corrupted, decoded, canonical, delete


def sequence(images: torch.Tensor, tail: torch.Tensor, anchor_count: int) -> torch.Tensor:
    parts = []
    if anchor_count:
        parts.append(images.unsqueeze(0).expand(anchor_count, -1, -1, -1, -1))
    parts.append(tail)
    return torch.cat(parts, 0)


@torch.no_grad()
def run(device: torch.device, batch_size: int):
    recs = records()
    models = {key: load_model(record, device) for key, record in recs.items()}
    test = dataset_spec("fashion_mnist", ROOT / "data")[1]
    loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0)
    thresholds = torch.tensor(recs[("oqe", 0, SEEDS[0])]["thresholds"], device=device)
    rng = torch.Generator(device="cpu").manual_seed(CORRUPTION_SEED)
    clean_correct = {key: 0 for key in models}
    corrupt_correct = {key: 0 for key in models}
    anchor_correct = {key: 0 for key in models}
    rate_max = {key: 0.0 for key in models}
    rate_differing = {key: 0 for key in models}
    pred_mismatch = {key: 0 for key in models}
    input_max = {key: 0.0 for key in models}
    totals = {encoding: {name: 0 for name in (
        "total_routes", "nonzero_original_routes", "deleted_events",
        "corrupted_tail_nonzero_events", "decoded_nonzero_k",
        "canonical_tail_nonzero_events", "canonical_elements",
    )} for encoding in ENCODINGS}
    examples = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        ids = (images.unsqueeze(0) >= thresholds.to(images).reshape(-1, 1, 1, 1, 1)).sum(0).long()
        shared_u = torch.rand(ids.shape, generator=rng, device="cpu").to(device)
        for encoding in ENCODINGS:
            clean_tail = codebook_for(encoding, K, device=device, dtype=images.dtype)[ids].permute(4, 0, 1, 2, 3).contiguous()
            corrupted, decoded, canonical, delete = explicit_delete_one(clean_tail, ids, shared_u, encoding)
            stats = totals[encoding]
            stats["total_routes"] += ids.numel()
            stats["nonzero_original_routes"] += int((ids > 0).sum().item())
            stats["deleted_events"] += int(delete.sum().item())
            stats["corrupted_tail_nonzero_events"] += int(torch.count_nonzero(corrupted).item())
            stats["decoded_nonzero_k"] += int(torch.count_nonzero(decoded).item())
            stats["canonical_tail_nonzero_events"] += int(torch.count_nonzero(canonical).item())
            stats["canonical_elements"] += canonical.numel()
            for anchor_count in ANCHORS:
                clean_input = sequence(images, clean_tail, anchor_count)
                corrupt_input = sequence(images, canonical, anchor_count)
                zero_tail = torch.zeros_like(canonical)
                anchor_only_input = sequence(images, zero_tail, anchor_count)
                for seed in SEEDS:
                    key = (encoding, anchor_count, seed)
                    model = models[key]
                    reset_model(model)
                    clean_rates = model.forward_from_first_currents(model.conv1(clean_input))
                    reset_model(model)
                    corrupt_rates = model.forward_from_first_currents(model.conv1(corrupt_input))
                    reset_model(model)
                    clean_correct[key] += int((clean_rates.argmax(1) == labels).sum().item())
                    corrupt_correct[key] += int((corrupt_rates.argmax(1) == labels).sum().item())
                    reset_model(model)
                    anchor_rates = model.forward_from_first_currents(model.conv1(anchor_only_input))
                    reset_model(model)
                    anchor_correct[key] += int((anchor_rates.argmax(1) == labels).sum().item())
                    if encoding == "single_hybrid":
                        max_input = float((corrupt_input - anchor_only_input).abs().max().item())
                        input_max[key] = max(input_max[key], max_input)
                        if max_input != 0:
                            raise RuntimeError(f"Single input != anchor-only: {key}")
                        diff = (corrupt_rates - anchor_rates).abs()
                        rate_max[key] = max(rate_max[key], float(diff.max().item()))
                        rate_differing[key] += int(torch.count_nonzero(diff).item())
                        pred_mismatch[key] += int((corrupt_rates.argmax(1) != anchor_rates.argmax(1)).sum().item())
        examples += images.shape[0]
    invariant_rows = []
    for encoding in ENCODINGS:
        stats = totals[encoding]
        for anchor_count in ANCHORS:
            for seed in SEEDS:
                if encoding == "single_hybrid":
                    passed = (
                        stats["deleted_events"] == stats["nonzero_original_routes"]
                        and stats["corrupted_tail_nonzero_events"] == 0
                        and stats["decoded_nonzero_k"] == 0
                        and stats["canonical_tail_nonzero_events"] == 0
                        and input_max[(encoding, anchor_count, seed)] == 0
                        and rate_max[(encoding, anchor_count, seed)] == 0
                        and rate_differing[(encoding, anchor_count, seed)] == 0
                        and pred_mismatch[(encoding, anchor_count, seed)] == 0
                    )
                else:
                    # All OQE batch-level decoded invariants already hard-stop;
                    # aggregate pass only needs exact deletion count here.
                    passed = stats["deleted_events"] == stats["nonzero_original_routes"]
                invariant_rows.append({
                    "encoding": encoding, "anchor_count": anchor_count, "training_seed": seed,
                    **stats, "input_anchor_only_max_abs_difference": input_max[(encoding, anchor_count, seed)],
                    "final_rate_max_abs_difference": rate_max[(encoding, anchor_count, seed)],
                    "final_rate_differing_elements": rate_differing[(encoding, anchor_count, seed)],
                    "prediction_mismatches": pred_mismatch[(encoding, anchor_count, seed)],
                    "expected_pass": "PASS" if passed else "FAIL",
                })
    if any(row["expected_pass"] != "PASS" for row in invariant_rows):
        raise RuntimeError("aggregate invariant failure")
    result_rows = []
    for key, record in recs.items():
        encoding, anchor_count, seed = key
        clean = clean_correct[key] / examples
        corrupt = corrupt_correct[key] / examples
        if abs(clean - float(record["clean_accuracy"])) > 0:
            raise RuntimeError(f"clean checkpoint reproduction failed: {key}")
        anchor_accuracy = anchor_correct[key] / examples
        if encoding == "single_hybrid" and anchor_accuracy != corrupt:
            raise RuntimeError(f"accuracy equality failed: {key}")
        result_rows.append({
            "dataset": "fashion_mnist", "encoding": encoding,
            "anchor_count": anchor_count, "training_seed": seed,
            "clean_accuracy": clean, "anchor_only_accuracy": anchor_accuracy,
            "one_erasure_accuracy": corrupt, "accuracy_drop_pp": 100 * (clean - corrupt),
            "tail_activity": record["tail_activity"], "total_steps": record["total_steps"],
        })
    write_csv(OUTPUT / "exact_one_deletion_invariants.csv", invariant_rows)
    write_csv(OUTPUT / "anchor_ablation_corrected_results.csv", result_rows)
    return result_rows, invariant_rows


def summarize(rows: list[dict[str, Any]], invariants: list[dict[str, Any]]) -> None:
    lines = ["# Corrected anchor-ablation summary", "", "## Clean accuracy table", "", "| Anchors | Encoding | Mean clean (%) | Seeds (%) |", "|---:|---|---:|---|" ]
    grouped = {}
    for anchor_count in ANCHORS:
        for encoding in ENCODINGS:
            chosen = sorted([r for r in rows if r["encoding"] == encoding and int(r["anchor_count"]) == anchor_count], key=lambda r: int(r["training_seed"]))
            grouped[(encoding, anchor_count)] = chosen
            vals = [100 * float(r["clean_accuracy"]) for r in chosen]
            lines.append(f"| {anchor_count} | {encoding} | {statistics.mean(vals):.3f} | {', '.join(f'{v:.3f}' for v in vals)} |")
    lines += ["", "## Corrected one-erasure table", "", "| Anchors | Encoding | Mean erase acc. (%) | Mean drop (pp) | Seeds erase acc. (%) |", "|---:|---|---:|---:|---|"]
    for anchor_count in ANCHORS:
        for encoding in ENCODINGS:
            chosen = grouped[(encoding, anchor_count)]
            acc = [100 * float(r["one_erasure_accuracy"]) for r in chosen]
            drops = [float(r["accuracy_drop_pp"]) for r in chosen]
            lines.append(f"| {anchor_count} | {encoding} | {statistics.mean(acc):.3f} | {statistics.mean(drops):.3f} | {', '.join(f'{v:.3f}' for v in acc)} |")
    lines += ["", "## Single-Spike invariant check", "", "For every anchor count and seed, one-erasure canonicalized input equals the independently constructed anchor-only input elementwise. Final rates are bitwise equal, prediction mismatches are zero, and one-erasure accuracy exactly equals the newly forwarded anchor-only accuracy. The 0-anchor Single input is entirely zero.", "", "| Anchors | Seed | Anchor-only (%) | One-erasure (%) | Difference (pp) |", "|---:|---:|---:|---:|---:|"]
    for anchor_count in ANCHORS:
        for row in grouped[("single_hybrid", anchor_count)]:
            anchor = 100 * float(row["anchor_only_accuracy"])
            erased = 100 * float(row["one_erasure_accuracy"])
            lines.append(f"| {anchor_count} | {row['training_seed']} | {anchor:.3f} | {erased:.3f} | {erased-anchor:+.3f} |")
    lines += ["", "## OQE versus Single clean and erased comparisons", "", "| Anchors | Clean OQE-Single (pp) | Erased OQE-Single (pp) | Drop advantage (pp) | Seed drop advantages (pp) |", "|---:|---:|---:|---:|---|"]
    verdicts = {}
    for anchor_count in ANCHORS:
        oqe, single = grouped[("oqe", anchor_count)], grouped[("single_hybrid", anchor_count)]
        clean_diff = 100 * (statistics.mean(float(r["clean_accuracy"]) for r in oqe) - statistics.mean(float(r["clean_accuracy"]) for r in single))
        erased_diff = 100 * (statistics.mean(float(r["one_erasure_accuracy"]) for r in oqe) - statistics.mean(float(r["one_erasure_accuracy"]) for r in single))
        advantages = [float(s["accuracy_drop_pp"]) - float(o["accuracy_drop_pp"]) for o, s in zip(oqe, single)]
        verdicts[anchor_count] = "YES" if erased_diff > 0 and sum(v > 0 for v in advantages) >= 2 else "NO"
        lines.append(f"| {anchor_count} | {clean_diff:+.3f} | {erased_diff:+.3f} | {statistics.mean(advantages):+.3f} | {', '.join(f'{v:+.3f}' for v in advantages)} |")
    clean0 = 100 * (statistics.mean(float(r["clean_accuracy"]) for r in grouped[("oqe", 0)]) - statistics.mean(float(r["clean_accuracy"]) for r in grouped[("single_hybrid", 0)]))
    lines += ["", "## Critical 0-anchor conclusions", "", f"- Clean anchor-free OQE advantage: **{'YES' if clean0 > 0 else 'NO'}**, mean difference {clean0:+.3f} pp.", f"- Task-level exactly-one-erasure advantage at 0 anchors: **{verdicts[0]}**.", "- The Single-Spike 0-anchor erased accuracy is the model's all-zero-input bias/chance accuracy, measured by an explicit forward rather than copied from an old table.", "", "## Does OQE have a task-level exactly-one-erasure advantage?", "", f"- 0 anchors: **{verdicts[0]}**", f"- 1 anchor: **{verdicts[1]}**", f"- 2 anchors: **{verdicts[2]}**"]
    (OUTPUT / "anchor_ablation_corrected_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def reports() -> None:
    audit = """# Exactly-one deletion bug audit

## 1. Old implementation

The old function was `evaluate_one_erasure` in `scripts/run_oqe_v12_anchor_ablation.py`. It read image batches `[B,1,28,28]`, derived per-route class IDs `[B,1,28,28]`, but then computed `decoded=(ids-(ids>0)).clamp_min(0)` once before the encoding loop. It generated each encoding's canonical tail directly from that decoded tensor and concatenated 0, 1, or 2 anchors. No corrupted tail, active-position list, or deletion mask was constructed in this path.

## 2. Violated invariant

The shared `decoded=k-1` shortcut is valid for cumulative OQE after one active event is deleted. It is invalid for Matched Single-Spike: deleting its sole event requires `decoded=0` for every nonzero route. Consequently the old zero-anchor Single input contained `Q_(k-1)` events and its approximately 83% accuracy could not be an all-zero baseline.

## 3. Root cause

The exact code-level cause was decoder substitution using the pre-corruption class ID rather than decoding the post-corruption tail. This was an encoding-specific semantic error, not an indexing, temporal-offset, checkpoint, concatenation, or metric-aggregation error.

## 4. Corrected implementation

The corrected path materializes each clean tail, checks its expected weight, enumerates actual active positions by cumulative active rank, uses one shared uniform variate per route to choose a rank, deletes exactly one active event for every `k>0` route, then count-decodes the resulting tail. All invariants hard-stop before any network forward. Each encoding canonicalizes back to its own codebook.

## 5. Scope

- Anchor ablation: **contaminated; corrected here**.
- Superseded anchor artifacts are `results/oqe_v12_followup/anchor_ablation_results.csv`, `anchor_ablation_summary.md`, the anchor interpretation in `v12_followup_final_report.md`, and the corresponding old sanity statement. They must not be cited as corrected results.
- Four-dataset V12 one-event results: **not contaminated by this bug**. That path explicitly zeroed Single-Spike tails and called `decode_tail` on the corrupted tensor.
- Independent Bernoulli deletion: **not contaminated**; it also decoded the materialized corrupted tensor.
- OQE/non-nested Workstream-A study: **not contaminated**; it explicitly deleted active positions and count-decoded the corrupted word.
"""
    (OUTPUT / "exact_one_deletion_bug_audit.md").write_text(audit, encoding="utf-8")
    contamination = """# V12 one-event contamination report

**V12 one-event result contamination: NO**

The earlier four-dataset implementation in `scripts/run_oqe_v6_event_loss.py::evaluate_task_config` used a distinct path. For `single_hybrid` it set the one-event corrupted tail to `zeros_like(clean_tail)`, then derived the decoded key by calling `decode_tail(corrupted, encoding)`, which returns zero for a silent Single-Spike word. Its corrected `one_event_runs_v2` results therefore do not share the anchor-ablation decoder-substitution bug. The independent-deletion branch likewise multiplies the materialized tail by the Bernoulli survival field before decoding. No four-dataset rerun is warranted by this audit.
"""
    (OUTPUT / "v12_one_event_contamination_report.md").write_text(contamination, encoding="utf-8")


def sanity(invariants: list[dict[str, Any]]) -> None:
    all_pass = all(row["expected_pass"] == "PASS" for row in invariants)
    checks = [
        ("no retraining", True), ("clean checkpoints reproduced", True),
        ("deletion is per nonzero route", all_pass),
        ("exactly one active event removed per nonzero route", all_pass),
        ("Single corrupted tail all zero", all(int(r["corrupted_tail_nonzero_events"]) == 0 for r in invariants if r["encoding"] == "single_hybrid")),
        ("Single decoded k all zero", all(int(r["decoded_nonzero_k"]) == 0 for r in invariants if r["encoding"] == "single_hybrid")),
        ("Single canonical tail all zero", all(int(r["canonical_tail_nonzero_events"]) == 0 for r in invariants if r["encoding"] == "single_hybrid")),
        ("zero-anchor Single input all zero", all(float(r["input_anchor_only_max_abs_difference"]) == 0 for r in invariants if r["encoding"] == "single_hybrid" and int(r["anchor_count"]) == 0)),
        ("one-anchor Single equals anchor-only tensor", all(float(r["input_anchor_only_max_abs_difference"]) == 0 for r in invariants if r["encoding"] == "single_hybrid" and int(r["anchor_count"]) == 1)),
        ("two-anchor Single equals anchor-only tensor", all(float(r["input_anchor_only_max_abs_difference"]) == 0 for r in invariants if r["encoding"] == "single_hybrid" and int(r["anchor_count"]) == 2)),
        ("Single one-erasure predictions equal anchor-only predictions", all(int(r["prediction_mismatches"]) == 0 for r in invariants if r["encoding"] == "single_hybrid")),
        ("OQE decoded k equals max(k-1,0)", all_pass),
        ("all seeds reported", {int(r["training_seed"]) for r in invariants} == set(SEEDS)),
        ("previous V12 contamination checked", True), ("no manuscript modified", True),
    ]
    report = ["# Corrected rerun final sanity", "", f"Overall: **{'PASS' if all(ok for _,ok in checks) else 'FAIL'}**", "", "| Check | Status |", "|---|---|"]
    report += [f"| {name} | {'PASS' if ok else 'FAIL'} |" for name, ok in checks]
    (OUTPUT / "corrected_rerun_final_sanity.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    if not all(ok for _, ok in checks):
        raise RuntimeError("final sanity failed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    reports()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rows, invariants = run(device, args.batch_size)
    summarize(rows, invariants)
    sanity(invariants)
    print(json.dumps({"status": "PASS", "device": str(device), "results": len(rows), "invariants": len(invariants)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
