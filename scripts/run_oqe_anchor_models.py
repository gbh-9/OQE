"""Train the verified Fashion-MNIST 0/1-anchor models and index 2-anchor main runs.

This release runner intentionally contains no deletion evaluator. Corrected
exactly-one-event evaluation lives in run_oqe_anchor_erasure_corrected.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from oqe_anchor import AnchorCountSNN  # noqa: E402
from oqe_v6 import MAIN_K, backbone_state, fixed_quantiles_from_uint8, load_backbone_state, seed_everything  # noqa: E402
from scripts.run_oqe_v6_event_loss import record_for  # noqa: E402
from scripts.run_oqe_v6_main import atomic_json, dataset_spec, evaluate, sha256_file, train_model  # noqa: E402

OUTPUT = ROOT / "results" / "oqe_anchor_models"
DATASET = "fashion_mnist"
ENCODINGS = ("oqe", "single_hybrid")
ANCHORS = (0, 1, 2)
SEEDS = (20261010, 20261011, 20261012)
EPOCHS = 3


def run_path(encoding: str, anchor_count: int, seed: int) -> Path:
    return OUTPUT / "runs" / f"{DATASET}__{encoding}__a{anchor_count}__s{seed}__e{EPOCHS}.json"


def new_model(encoding: str, anchor_count: int, seed: int, thresholds: torch.Tensor) -> AnchorCountSNN:
    seed_everything(seed)
    from oqe_v6 import MatchedInputSNN
    base = MatchedInputSNN(
        encoding="direct", input_channels=1, output_classes=10, image_size=28,
        thresholds=thresholds, k=MAIN_K,
    )
    shared = backbone_state(base)
    seed_everything(seed)
    model = AnchorCountSNN(
        anchor_count=anchor_count, encoding=encoding, input_channels=1,
        output_classes=10, image_size=28, thresholds=thresholds, k=MAIN_K,
    )
    load_backbone_state(model, shared)
    return model


def existing_two_anchor_record(encoding: str, seed: int) -> dict[str, Any]:
    source = record_for(DATASET, encoding, seed)
    return {
        "status": "REUSED_FROZEN_TWO_ANCHOR", "dataset": DATASET,
        "encoding": encoding, "anchor_count": 2, "training_seed": seed,
        "epochs": source["epochs"], "thresholds": source["thresholds"],
        "history": source["history"], "clean_accuracy": source["metrics"]["accuracy"],
        "anchors_only_accuracy": source["metrics"].get("anchor_only_accuracy"),
        "tail_activity": source["metrics"]["tail_spike_rate"],
        "total_steps": source["total_steps"], "checkpoint": source["checkpoint"],
        "checkpoint_sha256": source["checkpoint_sha256"],
    }


def train_missing(device: torch.device, batch_size: int) -> None:
    train_set, test_set, raw, _, _, _, exclude_zero = dataset_spec(DATASET, ROOT / "data")
    thresholds = fixed_quantiles_from_uint8(raw, MAIN_K, exclude_zero=exclude_zero)
    for anchor_count in ANCHORS:
        for seed in SEEDS:
            for encoding in ENCODINGS:
                path = run_path(encoding, anchor_count, seed)
                if path.exists():
                    continue
                if anchor_count == 2:
                    atomic_json(path, existing_two_anchor_record(encoding, seed))
                    continue
                model = new_model(encoding, anchor_count, seed, thresholds).to(device)
                started = time.perf_counter()
                history = train_model(model, train_set, seed=seed, epochs=EPOCHS, batch_size=batch_size, device=device)
                metrics = evaluate(model, test_set, seed=seed, batch_size=batch_size, device=device)
                anchors = evaluate(model, test_set, seed=seed, batch_size=batch_size, device=device, anchors_only=True)
                checkpoint = OUTPUT / "checkpoints" / f"{DATASET}__{encoding}__a{anchor_count}__s{seed}__e{EPOCHS}.pt"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model_state": model.state_dict(), "dataset": DATASET, "encoding": encoding,
                    "anchor_count": anchor_count, "seed": seed, "epochs": EPOCHS,
                    "k": MAIN_K, "thresholds": thresholds,
                }, checkpoint)
                atomic_json(path, {
                    "status": "FORMAL_COMPLETE", "dataset": DATASET, "encoding": encoding,
                    "anchor_count": anchor_count, "training_seed": seed, "epochs": EPOCHS,
                    "thresholds": [float(x) for x in thresholds], "history": history,
                    "clean_accuracy": metrics["accuracy"], "anchors_only_accuracy": anchors["accuracy"],
                    "tail_activity": record_for(DATASET, encoding, seed)["metrics"]["tail_spike_rate"],
                    "total_steps": (MAIN_K - 1) + anchor_count,
                    "elapsed_seconds": time.perf_counter() - started,
                    "checkpoint": str(checkpoint.relative_to(ROOT)),
                    "checkpoint_sha256": sha256_file(checkpoint),
                })


def records() -> dict[tuple[str, int, int], dict[str, Any]]:
    result = {}
    for encoding in ENCODINGS:
        for anchor_count in ANCHORS:
            for seed in SEEDS:
                path = run_path(encoding, anchor_count, seed)
                record = json.loads(path.read_text(encoding="utf-8"))
                checkpoint = ROOT / record["checkpoint"]
                if sha256_file(checkpoint) != record["checkpoint_sha256"]:
                    raise RuntimeError(f"checkpoint mismatch: {path}")
                result[(encoding, anchor_count, seed)] = record
    return result


def load_model(record: dict[str, Any], device: torch.device) -> AnchorCountSNN:
    payload = torch.load(ROOT / record["checkpoint"], map_location="cpu", weights_only=False)
    model = AnchorCountSNN(
        anchor_count=int(record["anchor_count"]), encoding=record["encoding"],
        input_channels=1, output_classes=10, image_size=28,
        thresholds=payload["thresholds"], k=MAIN_K,
    )
    model.load_state_dict(payload["model_state"], strict=True)
    return model.to(device).eval()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    train_missing(torch.device(args.device), args.batch_size)
    print(json.dumps({"status": "PASS", "records": len(records())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
