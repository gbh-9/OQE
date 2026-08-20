"""Create paper-facing aggregates for the OQE V6 final restructure."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (20261010, 20261011, 20261012)
DATASETS = ("fashion_mnist", "kmnist", "cifar10", "cifar100")
ARMS = ("direct", "rate", "latency", "single_hybrid", "oqe")


def load_records(directory: Path):
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(directory.glob("*.json"))]


def mean_std(values: list[float]):
    mean = sum(values) / len(values)
    std = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
    return {"mean": mean, "population_std": std, "values": values}


def main() -> int:
    v5 = load_records(ROOT / "results" / "oqe_v5_breadth" / "runs")
    v6 = load_records(ROOT / "results" / "oqe_v6_main" / "runs")
    selected: list[dict[str, Any]] = []
    for dataset in ("fashion_mnist", "kmnist"):
        for arm in ("direct", "rate", "latency"):
            selected.extend(
                row for row in v5
                if row["dataset"] == dataset and row["arm"] == arm and row["seed"] in SEEDS
            )
        selected.extend(
            row for row in v6
            if row["dataset"] == dataset and row["arm"] in {"single_hybrid", "oqe"} and row["seed"] in SEEDS
        )
    selected.extend(
        row for row in v6
        if row["dataset"] in {"cifar10", "cifar100"} and row["arm"] in ARMS and row["seed"] in SEEDS
    )

    main_table: dict[str, Any] = {}
    structure_table: dict[str, Any] = {}
    for dataset in DATASETS:
        main_table[dataset] = {}
        for arm in ARMS:
            rows = [row for row in selected if row["dataset"] == dataset and row["arm"] == arm]
            if len(rows) != 3:
                main_table[dataset][arm] = {"status": "INCOMPLETE", "count": len(rows)}
                continue
            rows.sort(key=lambda row: row["seed"])
            main_table[dataset][arm] = {
                "accuracy": mean_std([float(row["metrics"]["accuracy"]) for row in rows]),
                "seeds": [row["seed"] for row in rows],
                "records": [row["run_id"] for row in rows],
            }
        oqe_rows = sorted(
            [row for row in selected if row["dataset"] == dataset and row["arm"] == "oqe"],
            key=lambda row: row["seed"],
        )
        if len(oqe_rows) == 3:
            structure_table[dataset] = {
                "k": 7,
                "total_steps": 8,
                "tail_spike_rate": mean_std([row["metrics"]["tail_spike_rate"] for row in oqe_rows]),
                "tail_utility_pp": mean_std([row["metrics"]["tail_utility_pp"] for row in oqe_rows]),
                "groups_per_patch": mean_std([row["metrics"]["groups_per_patch"] for row in oqe_rows]),
                "route_contraction": mean_std([row["metrics"]["route_contraction"] for row in oqe_rows]),
                "cardinality_floor": mean_std([row["metrics"]["cardinality_floor"] for row in oqe_rows]),
            }
        else:
            structure_table[dataset] = {"status": "INCOMPLETE", "count": len(oqe_rows)}

    non_nested: dict[str, Any] = {}
    for arm in ("oqe", "non_nested"):
        rows = sorted(
            [row for row in v6 if row["dataset"] == "fashion_mnist" and row["arm"] == arm and row["seed"] in SEEDS],
            key=lambda row: row["seed"],
        )
        if len(rows) == 3:
            non_nested[arm] = {
                "accuracy": mean_std([row["metrics"]["accuracy"] for row in rows]),
                "tail_utility_pp": mean_std([row["metrics"]["tail_utility_pp"] for row in rows]),
                "tail_spike_rate": mean_std([row["metrics"]["tail_spike_rate"] for row in rows]),
                "route_contraction": mean_std([row["metrics"]["route_contraction"] for row in rows]),
            }
    output = {
        "status": "COMPLETE" if all(
            "status" not in main_table[d][a] for d in DATASETS for a in ARMS
        ) else "INCOMPLETE",
        "main_cross_dataset": main_table,
        "oqe_structural_profile": structure_table,
        "fashion_nested_vs_non_nested": non_nested,
        "formal_v6_record_count": len(v6),
        "selected_record_count": len(selected),
    }
    path = ROOT / "results" / "oqe_v6_main" / "paper_aggregate.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "formal_v6_record_count": len(v6), "selected_record_count": len(selected)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

