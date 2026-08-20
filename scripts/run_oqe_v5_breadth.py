"""Run the lean OQE V5 breadth matrix with per-run durable outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torchvision


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oqe_v5 import (  # noqa: E402
    CANONICAL_PERMUTATION,
    MAIN_CLASSES,
    V4_PERMUTATION,
    GroupTotals,
    SmallInputSNN,
    foreground_quantiles,
    reset_model,
    seed_everything,
)


DATASETS = ("fashion_mnist", "kmnist")
SEEDS = (20261010, 20261011, 20261012)
MAIN_ARMS = (
    "direct",
    "rate",
    "latency",
    "oqe_uniform",
    "oqe_quantile",
    "oqe_learned",
)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_pair(name: str, data_root: Path):
    transform = torchvision.transforms.ToTensor()
    if name == "fashion_mnist":
        cls = torchvision.datasets.FashionMNIST
        root = data_root / "fashion_mnist"
    elif name == "kmnist":
        cls = torchvision.datasets.KMNIST
        root = data_root / "kmnist"
    else:
        raise ValueError(name)
    train = cls(root=root, train=True, transform=transform, download=False)
    test = cls(root=root, train=False, transform=transform, download=False)
    return train, test


def loaders(train, test, *, seed: int, batch_size: int, device: torch.device):
    generator = torch.Generator().manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    test_loader = torch.utils.data.DataLoader(
        test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    return train_loader, test_loader


def train_model(
    model: SmallInputSNN,
    loader,
    *,
    epochs: int,
    device: torch.device,
) -> list[dict[str, float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0
        loss_sum = 0.0
        started = time.perf_counter()
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            rates = model(images)
            targets = F.one_hot(labels, 10).to(rates.dtype)
            loss = F.mse_loss(rates, targets)
            loss.backward()
            optimizer.step()
            total += labels.numel()
            correct += int((rates.argmax(1) == labels).sum().item())
            loss_sum += float(loss.item()) * labels.numel()
            reset_model(model)
        scheduler.step()
        history.append(
            {
                "epoch": epoch + 1,
                "train_accuracy": correct / total,
                "train_mse": loss_sum / total,
                "seconds": time.perf_counter() - started,
            }
        )
    return history


@torch.no_grad()
def evaluate(
    model: SmallInputSNN,
    loader,
    *,
    device: torch.device,
    seed: int,
    anchors_only: bool = False,
) -> dict[str, Any]:
    model.eval()
    torch.manual_seed(seed + (200_000 if anchors_only else 100_000))
    total = 0
    correct = 0
    loss_sum = 0.0
    input_sum = 0.0
    input_elements = 0
    group_totals = GroupTotals()
    class_counts = torch.zeros(model.language_classes, dtype=torch.int64)
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        rates, state = model(
            images, anchors_only=anchors_only, return_input_state=True
        )
        targets = F.one_hot(labels, 10).to(rates.dtype)
        loss = F.mse_loss(rates, targets)
        total += labels.numel()
        correct += int((rates.argmax(1) == labels).sum().item())
        loss_sum += float(loss.item()) * labels.numel()
        sequence = state["sequence"]
        assert isinstance(sequence, torch.Tensor)
        input_sum += float(sequence.sum().item())
        input_elements += sequence.numel()
        class_ids = state["class_ids"]
        if isinstance(class_ids, torch.Tensor):
            class_counts += torch.bincount(
                class_ids.detach().cpu().reshape(-1), minlength=model.language_classes
            )
            if not anchors_only:
                group_totals.update(class_ids, model.language_classes)
        reset_model(model)
    result: dict[str, Any] = {
        "accuracy": correct / total,
        "rate_mse": loss_sum / total,
        "examples": total,
        "mean_input_activity": input_sum / input_elements,
    }
    if int(class_counts.sum().item()):
        probabilities = class_counts.float() / class_counts.sum()
        entropy = -(probabilities[probabilities > 0] * probabilities[probabilities > 0].log()).sum()
        result["class_counts"] = class_counts.tolist()
        result["class_perplexity"] = float(entropy.exp().item())
        result.update(group_totals.metrics())
    return result


def run_spec(
    *,
    dataset: str,
    arm: str,
    seed: int,
    classes: int,
    permutation_name: str,
    epochs: int,
    batch_size: int,
    data_root: Path,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    run_id = f"{dataset}__{arm}__k{classes}__{permutation_name}__s{seed}"
    record_path = output_dir / "runs" / f"{run_id}.json"
    if record_path.exists():
        return json.loads(record_path.read_text(encoding="utf-8"))

    seed_everything(seed)
    train_set, test_set = dataset_pair(dataset, data_root)
    quantiles = foreground_quantiles(train_set.data, classes)
    if classes == MAIN_CLASSES and permutation_name == "v4":
        permutation = V4_PERMUTATION
    else:
        permutation = tuple(range(classes - 1))
    model = SmallInputSNN(
        encoding=arm,
        classes=classes,
        quantiles=quantiles if arm == "oqe_quantile" else None,
        permutation=permutation,
    ).to(device)
    train_loader, test_loader = loaders(
        train_set, test_set, seed=seed, batch_size=batch_size, device=device
    )
    started = time.perf_counter()
    history = train_model(model, train_loader, epochs=epochs, device=device)
    metrics = evaluate(model, test_loader, device=device, seed=seed)
    if arm.startswith("oqe_"):
        anchor_metrics = evaluate(
            model, test_loader, device=device, seed=seed, anchors_only=True
        )
        metrics["anchor_only_accuracy"] = anchor_metrics["accuracy"]
        metrics["tail_utility_pp"] = 100.0 * (
            metrics["accuracy"] - anchor_metrics["accuracy"]
        )
    checkpoint_path = output_dir / "checkpoints" / f"{run_id}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "dataset": dataset,
            "arm": arm,
            "seed": seed,
            "classes": classes,
            "permutation": list(permutation),
            "epochs": epochs,
        },
        checkpoint_path,
    )
    thresholds = None
    if model.encoder is not None:
        thresholds = [float(x) for x in model.encoder.ordered_thresholds.detach().cpu()]
    record = {
        "run_id": run_id,
        "dataset": dataset,
        "split": "official test",
        "arm": arm,
        "seed": seed,
        "classes": classes,
        "tail_steps": classes - 1 if arm.startswith("oqe_") else None,
        "total_steps": model.t_steps,
        "permutation_name": permutation_name,
        "permutation": list(permutation),
        "epochs": epochs,
        "batch_size": batch_size,
        "foreground_quantiles": [float(x) for x in quantiles],
        "ordered_thresholds": thresholds,
        "history": history,
        "metrics": metrics,
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }
    atomic_json(record_path, record)
    print(json.dumps({"completed": run_id, "accuracy": metrics["accuracy"], "seconds": record["elapsed_seconds"]}))
    return record


def specifications(suite: str):
    specs: list[dict[str, Any]] = []
    if suite in {"main", "all"}:
        for dataset in DATASETS:
            for seed in SEEDS:
                for arm in MAIN_ARMS:
                    specs.append(
                        dict(
                            dataset=dataset,
                            arm=arm,
                            seed=seed,
                            classes=MAIN_CLASSES,
                            permutation_name="v4" if arm.startswith("oqe_") else "na",
                        )
                    )
    if suite in {"sweep", "all"}:
        for classes in (3, 5, 7, 9):
            specs.append(
                dict(
                    dataset="fashion_mnist",
                    arm="oqe_quantile",
                    seed=SEEDS[0],
                    classes=classes,
                    permutation_name="canonical",
                )
            )
    if suite in {"permutation", "all"}:
        specs.append(
            dict(
                dataset="fashion_mnist",
                arm="oqe_learned",
                seed=SEEDS[0],
                classes=MAIN_CLASSES,
                permutation_name="canonical",
            )
        )
    return specs


def write_summary(output_dir: Path) -> None:
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((output_dir / "runs").glob("*.json"))
    ]
    main: dict[str, dict[str, Any]] = {}
    for dataset in DATASETS:
        main[dataset] = {}
        for arm in MAIN_ARMS:
            selected = [
                item
                for item in records
                if item["dataset"] == dataset
                and item["arm"] == arm
                and item["classes"] == MAIN_CLASSES
                and item["permutation_name"] in {"v4", "na"}
            ]
            if not selected:
                continue
            accuracies = [item["metrics"]["accuracy"] for item in selected]
            main[dataset][arm] = {
                "seeds": [item["seed"] for item in selected],
                "accuracy_mean": sum(accuracies) / len(accuracies),
                "accuracy_std": math.sqrt(
                    sum((x - sum(accuracies) / len(accuracies)) ** 2 for x in accuracies)
                    / len(accuracies)
                ),
                "records": [item["run_id"] for item in selected],
            }
    atomic_json(
        output_dir / "oqe_v5_breadth_summary.json",
        {"status": "COMPLETE_RECORDS_AVAILABLE", "main": main, "records": records},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("main", "sweep", "permutation", "all"), default="all")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "oqe_v5_breadth")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.data_root = args.data_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    specs = specifications(args.suite)
    if args.limit is not None:
        specs = specs[: args.limit]
    device = torch.device(args.device)
    for spec in specs:
        run_spec(
            **spec,
            epochs=args.epochs,
            batch_size=args.batch_size,
            data_root=args.data_root,
            output_dir=args.output_dir,
            device=device,
        )
    write_summary(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
