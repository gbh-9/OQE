import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_v23_main_and_anchor_free_reference_facts():
    main = json.loads((ROOT / "reference" / "v23_main_paper_aggregate.json").read_text())
    for arms in main["main_cross_dataset"].values():
        means = {name: item["accuracy"]["mean"] for name, item in arms.items()}
        assert max(means, key=means.get) == "direct"
        assert means["single_hybrid"] > means["oqe"]

    with (ROOT / "reference" / "v23_zero_anchor_results.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = {
        "fashion_mnist": (88.530, 86.347, 2.183, (3, 0, 0)),
        "kmnist": (91.073, 88.793, 2.280, (3, 0, 0)),
        "cifar10": (72.990, 73.090, -0.100, (1, 0, 2)),
    }
    for dataset, (oqe_mean, single_mean, delta, direction) in expected.items():
        values = {
            encoding: [100 * float(row["clean_accuracy"]) for row in rows
                       if row["dataset"] == dataset and row["encoding"] == encoding]
            for encoding in ("oqe", "single_hybrid")
        }
        paired = [a - b for a, b in zip(values["oqe"], values["single_hybrid"])]
        assert round(statistics.mean(values["oqe"]), 3) == oqe_mean
        assert round(statistics.mean(values["single_hybrid"]), 3) == single_mean
        assert round(statistics.mean(paired), 3) == delta
        assert (sum(x > 0 for x in paired), sum(x == 0 for x in paired), sum(x < 0 for x in paired)) == direction


def test_oqe_activity_is_approximately_3_5_times_single():
    with (ROOT / "reference" / "v23_matched_activity.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    for row in rows:
        ratio = float(row["oqe_tail_spike_rate"]) / float(row["single_tail_spike_rate"])
        assert 3.45 < ratio < 3.55
