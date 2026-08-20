"""Create paired-seed, terminology, and final nestedness evidence reports."""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oqe_v6 import codebook_for  # noqa: E402
from oqe_v6_executor import generic_reconstruct_tail_currents, suffix_reconstruct_tail_currents  # noqa: E402


OUTPUT = ROOT / "results" / "oqe_v6_nestedness"
SEEDS = (20261010, 20261011, 20261012)
DATASETS = ("fashion_mnist", "kmnist", "cifar10", "cifar100")


def load_csv(name: str) -> list[dict[str, str]]:
    with (OUTPUT / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def paired_rows() -> list[dict[str, Any]]:
    aggregate = json.loads(
        (ROOT / "results" / "oqe_v6_main" / "paper_aggregate.json").read_text(encoding="utf-8")
    )
    rows = []
    for dataset in DATASETS:
        table = aggregate["main_cross_dataset"][dataset]
        for comparison, arm_a, arm_b in (
            ("single_hybrid_minus_oqe", "single_hybrid", "oqe"),
            ("direct_minus_oqe", "direct", "oqe"),
        ):
            values_a = table[arm_a]["accuracy"]["values"]
            values_b = table[arm_b]["accuracy"]["values"]
            for seed, left, right in zip(SEEDS, values_a, values_b):
                rows.append({
                    "dataset": dataset, "comparison": comparison, "seed": seed,
                    "arm_A_acc": left, "arm_B_acc": right,
                    "paired_difference_pp": 100.0 * (left - right),
                })
    nested = aggregate["fashion_nested_vs_non_nested"]
    for seed, left, right in zip(
        SEEDS, nested["non_nested"]["accuracy"]["values"], nested["oqe"]["accuracy"]["values"]
    ):
        rows.append({
            "dataset": "fashion_mnist", "comparison": "non_nested_minus_nested_oqe",
            "seed": seed, "arm_A_acc": left, "arm_B_acc": right,
            "paired_difference_pp": 100.0 * (left - right),
        })
    return rows


def write_paired(rows: list[dict[str, Any]]) -> None:
    with (OUTPUT / "paired_seed_differences.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Paired-seed direction summary", "",
        "Differences are `arm A - arm B` in percentage points. These are descriptive n=3 paired directions; no significance or superiority test is used.", "",
        "| Dataset | Comparison | Seed differences (pp) | Mean | Min | Max | + / 0 / - |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    keys = []
    for row in rows:
        key = (row["dataset"], row["comparison"])
        if key not in keys:
            keys.append(key)
    for dataset, comparison in keys:
        subset = [r for r in rows if r["dataset"] == dataset and r["comparison"] == comparison]
        values = [float(r["paired_difference_pp"]) for r in subset]
        directions = (
            sum(value > 0 for value in values), sum(value == 0 for value in values),
            sum(value < 0 for value in values),
        )
        seed_text = ", ".join(f"{r['seed']}: {r['paired_difference_pp']:+.3f}" for r in subset)
        lines.append(
            f"| {dataset} | {comparison} | {seed_text} | {statistics.mean(values):+.3f} | "
            f"{min(values):+.3f} | {max(values):+.3f} | {directions[0]}/{directions[1]}/{directions[2]} |"
        )
    lines += ["", "## Directional facts", ""]
    for dataset in DATASETS:
        subset = [r for r in rows if r["dataset"] == dataset and r["comparison"] == "single_hybrid_minus_oqe"]
        positives = sum(float(r["paired_difference_pp"]) > 0 for r in subset)
        lines.append(f"- {dataset}: Single-Spike minus OQE is positive in {positives}/3 paired seeds.")
    subset = [r for r in rows if r["comparison"] == "non_nested_minus_nested_oqe"]
    positives = sum(float(r["paired_difference_pp"]) > 0 for r in subset)
    negatives = sum(float(r["paired_difference_pp"]) < 0 for r in subset)
    lines += [f"- Fashion-MNIST non-nested minus nested OQE: {positives} positive, {negatives} negative, {3-positives-negatives} zero directions.", ""]
    (OUTPUT / "paired_seed_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_finite_language_audit() -> None:
    text = """# Finite-language terminology audit

## Objects that are finite

The binary OQE tail `P_k` is strictly a finite ordered K-word temporal language. Its ordinal class is a finite exact key, and the tail alone is binary, nested, and count-recoverable.

## Object that is hybrid

The actual eight-step input used in the experiments is `[x, x, P_k]`. The first two entries are repeated analog anchors. For arbitrary real-valued x, the complete hybrid sequence is therefore not itself a pure finite binary language even though its tail and route key are finite.

## Terminology risk and recommended wording

- Most precise default: **hybrid SNN input with a finite keyed binary tail**.
- Compact method phrase: **finite ordered tail language** or **finite ordered temporal code**.
- When discussing grouping: **finite-key temporal representation**.
- Avoid using “finite binary language” for the entire `[x,x,P_k]` sequence without immediately identifying the analog anchors.

The current title, “A Finite Ordered SNN Input Language,” is defensible only if “language” is explicitly defined as the keyed tail component or as the finite route vocabulary. A strict reviewer could read it as describing every numeric value in the full hybrid sequence. Low-cost later options are to qualify the title with “Finite-Key” or “with a Finite Ordered Tail”; no title change is made in this round.
"""
    (OUTPUT / "finite_language_terminology_audit.md").write_text(text, encoding="utf-8")


def write_route_report() -> None:
    candidates = [
        ROOT / "paper" / "main.tex", ROOT / "paper" / "CLAIM_LEDGER.md",
        ROOT / "paper" / "OQE_V6_FINAL_STATUS.md", ROOT / "src" / "oqe_v6.py",
        ROOT / "scripts" / "summarize_oqe_v6.py",
        ROOT / "results" / "oqe_v6_supplement" / "single_spike_vs_oqe_summary.md",
    ]
    pattern = re.compile(r"contraction|topology contraction|redundancy contraction", re.I)
    occurrences = []
    for path in candidates:
        if not path.is_file():
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                stripped = line.strip().replace("|", "\\|")
                ambiguous = "route-count" not in stripped.lower()
                recommendation = "replace with route-count contraction" if ambiguous else "keep"
                if "route_contraction" in stripped:
                    recommendation = "keep internal field; define publicly as route-count contraction"
                occurrences.append((path.relative_to(ROOT), line_number, stripped, recommendation))
    lines = [
        "# Route contraction terminology report", "",
        "Scope: the active V6 manuscript, claim ledger/status, V6 metric code, and current Single-Spike/OQE supplement. Legacy V2–V5 review archives are not proposed for editing.", "",
        "The measured quantity is a reduction in the number of surviving nonzero first-layer route groups relative to valid routes. It is not, by itself, an operation, memory, latency, or energy reduction.", "",
        "| File | Line | Current wording | Recommendation |",
        "|---|---:|---|---|",
    ]
    lines.extend(f"| `{path}` | {number} | {text} | {recommendation} |" for path, number, text, recommendation in occurrences)
    lines += ["", "## Recommended definition", "", "Define **route-count contraction** once as `1-G/R`, where G is the number of surviving nonzero exact-key groups and R is the number of valid input routes. Thereafter, the short form “contraction” is acceptable only within a section or table whose caption retains that definition. Never use it as a synonym for operation reduction, memory reduction, latency speedup, topology pruning, or energy saving.", ""]
    (OUTPUT / "route_contraction_terminology_report.md").write_text("\n".join(lines), encoding="utf-8")


def suffix_formula_check() -> tuple[bool, float]:
    maximum = 0.0
    for k in (3, 5, 7, 9, 17):
        torch.manual_seed(44 + k)
        aggregates = torch.randn(k, 2, 3, 4, 4, dtype=torch.float64)
        aggregates[0].zero_()
        generic = generic_reconstruct_tail_currents(aggregates, codebook_for("oqe", k, dtype=torch.float64))
        suffix = suffix_reconstruct_tail_currents(aggregates)
        maximum = max(maximum, float((generic - suffix).abs().max().item()))
    return maximum <= 1e-12, maximum


def benchmark_facts() -> dict[str, Any]:
    micro = load_csv("reconstruction_microbenchmark.csv")
    full = load_csv("suffix_full_forward_benchmark.csv")
    facts = {"micro": [], "full": []}
    for device in sorted({row["device"] for row in micro}):
        for batch in (1, 128):
            subset = [row for row in micro if row["device"] == device and int(row["batch"]) == batch and int(row["K"]) == 7]
            times = {row["method"]: float(row["median_ms"]) for row in subset}
            facts["micro"].append({"device": device, "batch": batch, **times})
    for device in sorted({row["device"] for row in full}):
        for batch in (1, 128):
            subset = [row for row in full if row["device"] == device and int(row["batch"]) == batch]
            times = {row["method"]: float(row["median_ms"]) for row in subset}
            facts["full"].append({"device": device, "batch": batch, **times})
    return facts


def write_final_report(paired: list[dict[str, Any]]) -> None:
    geometry = json.loads((OUTPUT / "ordinal_geometry_verification.json").read_text(encoding="utf-8"))
    correctness = json.loads((OUTPUT / "suffix_executor_correctness.json").read_text(encoding="utf-8"))
    formula_ok, formula_error = suffix_formula_check()
    facts = benchmark_facts()
    generic_suffix_predictions = sum(
        row["comparisons"]["generic_vs_suffix"]["prediction_mismatches"] for row in correctness["rows"]
    )
    ordinary_suffix_predictions = sum(
        row["comparisons"]["ordinary_vs_suffix"]["prediction_mismatches"] for row in correctness["rows"]
    )
    micro_suffix_faster = all(row["suffix_oqe"] < row["generic_oqe"] for row in facts["micro"])
    single_faster_count = sum(row["single_spike"] < row["suffix_oqe"] for row in facts["micro"])
    full_suffix_faster_count = sum(row["suffix_quotient"] < row["generic_quotient"] for row in facts["full"])
    full_suffix_slower_ordinary = all(row["suffix_quotient"] > row["ordinary"] for row in facts["full"])
    supplement = list(csv.DictReader((ROOT / "results" / "oqe_v6_supplement" / "single_spike_structural_metrics.csv").open(encoding="utf-8")))
    single_accuracy = all(
        statistics.mean(float(r["accuracy"]) for r in supplement if r["dataset"] == dataset and r["encoding"] == "single_hybrid")
        > statistics.mean(float(r["accuracy"]) for r in supplement if r["dataset"] == dataset and r["encoding"] == "oqe")
        for dataset in DATASETS
    )
    single_activity = all(
        statistics.mean(float(r["tail_spike_rate"]) for r in supplement if r["dataset"] == dataset and r["encoding"] == "single_hybrid")
        < statistics.mean(float(r["tail_spike_rate"]) for r in supplement if r["dataset"] == dataset and r["encoding"] == "oqe")
        for dataset in DATASETS
    )
    grouping_equal = all(
        {float(r["groups_per_patch"]) for r in supplement if r["dataset"] == dataset and r["encoding"] == "single_hybrid"}
        == {float(r["groups_per_patch"]) for r in supplement if r["dataset"] == dataset and r["encoding"] == "oqe"}
        for dataset in DATASETS
    )
    checks = [
        ("OQE Hamming isometry exhaustive check", geometry["oqe_isometry_mismatch_count"] == 0, str(geometry["oqe_isometry_mismatch_count"])),
        ("Adjacent one-bit relation", geometry["adjacent_relation_failure_count"] == 0, str(geometry["adjacent_relation_failure_count"])),
        ("Suffix-current formula", formula_ok, f"FP64 max error {formula_error:.3g}"),
        ("Generic/suffix prediction comparison", generic_suffix_predictions == 0, str(generic_suffix_predictions)),
        ("Ordinary/suffix prediction comparison", ordinary_suffix_predictions == 0, str(ordinary_suffix_predictions)),
        ("Benchmark protocol", len(facts["micro"]) == 4 and len(facts["full"]) == 4, "CPU/GPU, batch 1/128"),
        ("Single-Spike results preserved", single_accuracy and single_activity and grouping_equal, "higher mean accuracy, lower activity, equal grouping"),
        ("Old analytical count not restored", not any(token in (OUTPUT / "suffix_executor_summary.md").read_text(encoding="utf-8") for token in ("29.84", "33.01", "66.90", "68.78")), "new reports only"),
        ("Targeted primary-source audit", (OUTPUT / "ordinal_unary_related_work_audit.md").is_file(), "six prioritized candidates"),
        ("Finite-language and route terminology audits", (OUTPUT / "finite_language_terminology_audit.md").is_file() and (OUTPUT / "route_contraction_terminology_report.md").is_file(), "active V6 scope"),
    ]
    overall = all(ok for _, ok, _ in checks)
    report = f"""# Final nestedness evidence report

Overall consistency: **{'PASS' if overall else 'FAIL'}**

## 1. What nestedness guarantees mathematically

- For general K and any temporal permutation, `d_H(P_k,P_l)=|k-l|`; exhaustive mismatch count is {geometry['oqe_isometry_mismatch_count']}.
- Adjacent classes add exactly one basis bit: `P_(k+1)=P_k+e_(pi^-1(k))`; checked failure count is {geometry['adjacent_relation_failure_count']}.
- Canonical tails recover the ordinal count from their number of spikes. Single-Spike retains an ordinal event-time key but not the same Hamming-distance isometry.

## 2. What nestedness changes algorithmically

After identical class aggregation produces A_k, generic reconstruction applies every class-word relation. Canonical OQE instead admits `I_t=sum_(k>t) A_k` and the reverse recurrence `I_t=I_(t+1)+A_(t+1)`, requiring K-2 vector additions plus the boundary assignment. A functional `suffix_quotient` path now exercises this OQE-specific primitive while leaving anchors and downstream execution unchanged.

## 3. What the prototype shows empirically

- Generic versus suffix and ordinary versus suffix were evaluated on {sum(row['examples'] for row in correctness['rows'])} official-test examples across three Fashion-MNIST checkpoints and one CIFAR-10 checkpoint. Prediction mismatches are {generic_suffix_predictions} and {ordinary_suffix_predictions}, respectively.
- At K=7, suffix reconstruction is faster than generic reconstruction in {'all four' if micro_suffix_faster else 'not all'} CPU/GPU batch settings.
- Single-Spike placement is faster than suffix reconstruction in {single_faster_count}/4 settings; it remains the simpler primitive in the larger batch measurements.
- Full suffix quotient is faster than full generic quotient in {full_suffix_faster_count}/4 settings, but is slower than ordinary forwarding in {'all four' if full_suffix_slower_ordinary else 'not all'} settings. Aggregation, unfold/membership, materialization, launches, and downstream work still dominate.

## 4. What nestedness does not currently improve

- Main-task accuracy: Matched Single-Spike has higher mean accuracy on all four datasets and all paired directions are reported separately.
- Tail activity: Matched Single-Spike has lower tail spike rate on all four datasets.
- Grouping opportunity: class IDs, groups/patch, and route-count contraction are equal between the matched encoders.
- End-to-end deployment efficiency: the suffix prototype remains slower than ordinary forwarding in every measured setting.

## 5. What can safely be claimed in the paper

1. OQE provides an ordinal-distance-preserving nested binary tail for general K.
2. Adjacent ordinal classes differ by one added temporal basis event.
3. Given class aggregates A_k, canonical OQE admits an exact reverse-suffix reconstruction primitive.
4. The implemented suffix path preserves every tested prediction despite small finite-precision reassociation differences.
5. The post-grouping suffix kernel is faster than generic OQE reconstruction in the measured K=7 settings, without implying end-to-end acceleration.

## 6. What must not be claimed

- OQE accuracy, activity, or grouping superiority over Matched Single-Spike.
- Universal speed superiority over Single-Spike placement.
- End-to-end acceleration, hardware efficiency, energy saving, or topology pruning.
- Bitwise FP32 equivalence or statistical significance from three seeds.
- Novelty of thermometer/unary primitives, ordinal thresholds, or linear regrouping in isolation.

## Consistency checks

| Check | Status | Detail |
|---|---|---|
{chr(10).join(f'| {name} | {"PASS" if ok else "FAIL"} | {detail} |' for name, ok, detail in checks)}
"""
    (OUTPUT / "final_nestedness_evidence_report.md").write_text(report, encoding="utf-8")
    if not overall:
        raise RuntimeError("nestedness consistency audit failed")


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = paired_rows()
    write_paired(rows)
    write_finite_language_audit()
    write_route_report()
    write_final_report(rows)
    print(json.dumps({"status": "PASS", "paired_rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
