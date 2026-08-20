"""Paired zero-anchor OQE versus Matched Single-Spike follow-up."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from oqe_v6 import MAIN_K, backbone_state, fixed_quantiles_from_uint8, load_backbone_state, seed_everything  # noqa: E402
from scripts.run_oqe_v6_main import atomic_json, dataset_spec, evaluate, sha256_file, train_model  # noqa: E402
from scripts.run_oqe_v6_event_loss import record_for  # noqa: E402
from oqe_anchor import AnchorCountSNN  # noqa: E402

OUTPUT = ROOT / "results" / "oqe_v18_followup"
DATASETS = ("kmnist", "cifar10")
ENCODINGS = ("oqe", "single_hybrid")
SEEDS = (20261010, 20261011, 20261012)
EPOCHS = {"kmnist": 3, "cifar10": 20}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def paired_model(*, encoding: str, seed: int, channels: int, outputs: int, image_size: int, thresholds: torch.Tensor):
    from oqe_v6 import MatchedInputSNN
    seed_everything(seed)
    base = MatchedInputSNN(
        encoding="direct", input_channels=channels, output_classes=outputs,
        image_size=image_size, thresholds=thresholds, k=MAIN_K,
    )
    shared = backbone_state(base)
    seed_everything(seed)
    model = AnchorCountSNN(
        anchor_count=0, encoding=encoding, input_channels=channels,
        output_classes=outputs, image_size=image_size, thresholds=thresholds, k=MAIN_K,
    )
    load_backbone_state(model, shared)
    return model


def config_audit() -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASETS:
        train, _, raw, channels, outputs, image_size, exclude_zero = dataset_spec(dataset, ROOT / "data")
        thresholds = fixed_quantiles_from_uint8(raw, MAIN_K, exclude_zero=exclude_zero)
        for seed in SEEDS:
            sources = [record_for(dataset, encoding, seed) for encoding in ENCODINGS]
            source_thresholds = [torch.tensor(source["thresholds"]) for source in sources]
            threshold_match = all(torch.equal(thresholds, item) for item in source_thresholds)
            source_epochs = {int(source["epochs"]) for source in sources}
            rows.append({
                "dataset": dataset, "training_seed": seed,
                "thresholds_match": threshold_match,
                "oqe_single_thresholds_equal": torch.equal(*source_thresholds),
                "expected_epochs": EPOCHS[dataset], "source_epochs": next(iter(source_epochs)) if len(source_epochs) == 1 else -1,
                "architecture": f"channels={channels};outputs={outputs};image={image_size}",
                "optimizer": "AdamW(lr=1e-3,weight_decay=5e-4)",
                "scheduler": f"CosineAnnealingLR(T_max={EPOCHS[dataset]})",
                "batch_size": 128, "loss": "MSE; output-class scaling preserved",
                "paired_initialization": True, "status": "PASS" if threshold_match and source_epochs == {EPOCHS[dataset]} else "FAIL",
            })
    return rows


def run_path(dataset: str, encoding: str, seed: int) -> Path:
    return OUTPUT / "runs" / f"{dataset}__{encoding}__zero_anchor__s{seed}__e{EPOCHS[dataset]}.json"


def train_all(device: torch.device, batch_size: int) -> None:
    audit = config_audit()
    write_csv(OUTPUT / "zero_anchor_configuration_audit.csv", audit)
    if any(row["status"] != "PASS" for row in audit):
        raise RuntimeError("configuration gate failed")
    for dataset in DATASETS:
        train_set, test_set, raw, channels, outputs, image_size, exclude_zero = dataset_spec(dataset, ROOT / "data")
        thresholds = fixed_quantiles_from_uint8(raw, MAIN_K, exclude_zero=exclude_zero)
        for seed in SEEDS:
            for encoding in ENCODINGS:
                path = run_path(dataset, encoding, seed)
                if path.exists():
                    continue
                model = paired_model(
                    encoding=encoding, seed=seed, channels=channels, outputs=outputs,
                    image_size=image_size, thresholds=thresholds,
                ).to(device)
                started = time.perf_counter()
                history = train_model(
                    model, train_set, seed=seed, epochs=EPOCHS[dataset],
                    batch_size=batch_size, device=device,
                )
                metrics = evaluate(model, test_set, seed=seed, batch_size=batch_size, device=device)
                checkpoint = OUTPUT / "checkpoints" / f"{dataset}__{encoding}__zero_anchor__s{seed}__e{EPOCHS[dataset]}.pt"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model_state": model.state_dict(), "dataset": dataset,
                    "encoding": encoding, "anchor_count": 0, "seed": seed,
                    "epochs": EPOCHS[dataset], "k": MAIN_K, "thresholds": thresholds,
                }, checkpoint)
                record = {
                    "status": "FORMAL_COMPLETE", "dataset": dataset,
                    "encoding": encoding, "anchor_count": 0, "training_seed": seed,
                    "epochs": EPOCHS[dataset], "batch_size": batch_size,
                    "thresholds": [float(x) for x in thresholds], "history": history,
                    "clean_accuracy": metrics["accuracy"],
                    # Fixed codebook and class partition are unchanged by anchor count.
                    "tail_activity": record_for(dataset, encoding, seed)["metrics"]["tail_spike_rate"],
                    "total_steps": MAIN_K - 1, "final_train_loss": history[-1]["raw_rate_mse"],
                    "validation_metric": None,
                    "validation_note": "No validation selection in the source protocol; official test evaluated once after training.",
                    "elapsed_seconds": time.perf_counter() - started,
                    "checkpoint": str(checkpoint.relative_to(ROOT)),
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "official_test_evaluated_once": True,
                }
                atomic_json(path, record)
                print(json.dumps({"completed": path.stem, "accuracy": metrics["accuracy"], "seconds": record["elapsed_seconds"]}), flush=True)


def collect() -> list[dict[str, Any]]:
    rows = []
    # Existing corrected Fashion zero-anchor records are included as reference.
    with (ROOT / "results" / "oqe_anchor_erasure_corrected" / "anchor_ablation_corrected_results.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["anchor_count"]) == 0:
                source = ROOT / "results" / "oqe_v12_followup" / "anchor_ablation" / "runs" / f"fashion_mnist__{row['encoding']}__a0__s{row['training_seed']}__e3.json"
                old = json.loads(source.read_text(encoding="utf-8"))
                rows.append({
                    "dataset": "fashion_mnist", "encoding": row["encoding"], "anchor_count": 0,
                    "training_seed": int(row["training_seed"]), "clean_accuracy": float(row["clean_accuracy"]),
                    "tail_activity": float(row["tail_activity"]), "total_steps": 6,
                    "final_train_loss": float(old["history"][-1]["raw_rate_mse"]),
                    "validation_metric": "", "source": "existing_corrected_fashion_reference",
                })
    for dataset in DATASETS:
        for seed in SEEDS:
            for encoding in ENCODINGS:
                path = run_path(dataset, encoding, seed)
                record = json.loads(path.read_text(encoding="utf-8"))
                checkpoint = ROOT / record["checkpoint"]
                if sha256_file(checkpoint) != record["checkpoint_sha256"]:
                    raise RuntimeError(f"checkpoint hash mismatch: {path}")
                rows.append({
                    "dataset": dataset, "encoding": encoding, "anchor_count": 0,
                    "training_seed": seed, "clean_accuracy": record["clean_accuracy"],
                    "tail_activity": record["tail_activity"], "total_steps": record["total_steps"],
                    "final_train_loss": record["final_train_loss"], "validation_metric": "",
                    "source": "v18_formal_run",
                })
    write_csv(OUTPUT / "zero_anchor_extended_results.csv", rows)
    return rows


def summarize(rows: list[dict[str, Any]]) -> None:
    lines = ["# Zero-anchor extended summary", "", "All standard deviations are population standard deviations (`ddof=0`), matching the descriptive V17 convention. With n=3, paired results are descriptive and no significance claim is made.", ""]
    outcomes = {}
    for dataset in ("fashion_mnist", *DATASETS):
        lines += [f"## {dataset}", "", "| Encoding | Seed accuracies (%) | Mean (%) | Population std (pp) |", "|---|---|---:|---:|"]
        selected = {}
        for encoding in ENCODINGS:
            chosen = sorted([r for r in rows if r["dataset"] == dataset and r["encoding"] == encoding], key=lambda r: int(r["training_seed"]))
            vals = [100 * float(r["clean_accuracy"]) for r in chosen]
            selected[encoding] = vals
            lines.append(f"| {encoding} | {', '.join(f'{v:.3f}' for v in vals)} | {statistics.mean(vals):.3f} | {statistics.pstdev(vals):.3f} |")
        diffs = [o - s for o, s in zip(selected["oqe"], selected["single_hybrid"])]
        outcomes[dataset] = statistics.mean(diffs) > 0
        lines += ["", f"Paired OQE-Single differences (pp): {', '.join(f'{v:+.3f}' for v in diffs)}; mean {statistics.mean(diffs):+.3f}; direction {sum(v>0 for v in diffs)}/{sum(v==0 for v in diffs)}/{sum(v<0 for v in diffs)} (OQE/tie/Single).", ""]
    new_positive = sum(outcomes[d] for d in DATASETS)
    if new_positive == 2:
        verdict = "YES"
        interpretation = "The Fashion-MNIST anchor-free advantage generalizes to both additional tested datasets, subject to descriptive three-seed scope."
    elif new_positive == 1:
        verdict = "PARTIAL"
        interpretation = "The anchor-free advantage is dataset-dependent but not unique to Fashion-MNIST."
    else:
        verdict = "NO"
        interpretation = "The Fashion-MNIST zero-anchor gain does not generalize to the additional tested datasets."
    lines += ["## Cross-dataset conclusion", "", f"**{verdict}.** {interpretation}", "", "The codebooks carry the same ordinal symbol k. Any accuracy gap concerns how the tested SNN exploits different temporal realizations; it is not evidence that OQE carries more ordinal information."]
    (OUTPUT / "zero_anchor_extended_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    atomic_json(OUTPUT / "zero_anchor_decision.json", {"decision": verdict, "new_dataset_positive_count": new_positive, "dataset_mean_favors_oqe": outcomes})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("audit", "train", "summarize", "all"), default="all")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.phase in ("audit", "all"):
        audit = config_audit(); write_csv(OUTPUT / "zero_anchor_configuration_audit.csv", audit)
        if any(row["status"] != "PASS" for row in audit): raise RuntimeError("configuration gate failed")
    if args.phase in ("train", "all"): train_all(device, args.batch_size)
    if args.phase in ("summarize", "all"): summarize(collect())
    print(json.dumps({"status": "PASS", "phase": args.phase, "device": str(device)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
