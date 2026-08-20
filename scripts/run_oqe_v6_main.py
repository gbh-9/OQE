"""Run paired OQE V6 grayscale/RGB main comparisons with durable per-run outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchvision


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oqe_v6 import (  # noqa: E402
    MAIN_ARMS,
    MAIN_K,
    GroupTotals,
    MatchedInputSNN,
    backbone_state,
    fixed_quantiles_from_uint8,
    load_backbone_state,
    reset_model,
    seed_everything,
)


DATASETS = ("fashion_mnist", "kmnist", "cifar10", "cifar100")
SEEDS = (20261010, 20261011, 20261012)
GRAYSCALE_EPOCHS = 3


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


def dataset_spec(name: str, data_root: Path):
    to_tensor = torchvision.transforms.ToTensor()
    if name == "fashion_mnist":
        root = data_root / "fashion_mnist"
        train = torchvision.datasets.FashionMNIST(root=root, train=True, transform=to_tensor, download=False)
        test = torchvision.datasets.FashionMNIST(root=root, train=False, transform=to_tensor, download=False)
        raw = train.data
        return train, test, raw, 1, 10, 28, True
    if name == "kmnist":
        root = data_root / "kmnist"
        train = torchvision.datasets.KMNIST(root=root, train=True, transform=to_tensor, download=False)
        test = torchvision.datasets.KMNIST(root=root, train=False, transform=to_tensor, download=False)
        raw = train.data
        return train, test, raw, 1, 10, 28, True
    if name in {"cifar10", "cifar100"}:
        root = data_root / name
        train_transform = torchvision.transforms.Compose(
            [torchvision.transforms.RandomCrop(32, padding=4), torchvision.transforms.RandomHorizontalFlip(), to_tensor]
        )
        cls = torchvision.datasets.CIFAR10 if name == "cifar10" else torchvision.datasets.CIFAR100
        train = cls(root=root, train=True, transform=train_transform, download=False)
        test = cls(root=root, train=False, transform=to_tensor, download=False)
        raw = torch.from_numpy(np.asarray(train.data))
        return train, test, raw, 3, (10 if name == "cifar10" else 100), 32, False
    raise ValueError(name)


def make_loader(dataset, *, batch_size: int, shuffle: bool, seed: int, device: torch.device):
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=generator,
    )


def new_paired_model(
    *,
    arm: str,
    seed: int,
    input_channels: int,
    output_classes: int,
    image_size: int,
    thresholds: torch.Tensor,
) -> MatchedInputSNN:
    seed_everything(seed)
    base = MatchedInputSNN(
        encoding="direct",
        input_channels=input_channels,
        output_classes=output_classes,
        image_size=image_size,
        thresholds=thresholds,
    )
    shared = backbone_state(base)
    seed_everything(seed)
    model = MatchedInputSNN(
        encoding=arm,
        input_channels=input_channels,
        output_classes=output_classes,
        image_size=image_size,
        thresholds=thresholds,
    )
    load_backbone_state(model, shared)
    return model


def train_model(model, train_set, *, seed: int, epochs: int, batch_size: int, device: torch.device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history: list[dict[str, float]] = []
    persistent_loader = None
    if model.input_channels == 1:
        # Preserve the exact V5 grayscale sample-order protocol so unchanged
        # Direct/Rate/Latency arms remain valid paired records.
        persistent_loader = make_loader(
            train_set, batch_size=batch_size, shuffle=True, seed=seed, device=device
        )
    for epoch in range(epochs):
        if persistent_loader is None:
            seed_everything(seed + epoch)
            loader = make_loader(train_set, batch_size=batch_size, shuffle=True, seed=seed + epoch, device=device)
        else:
            loader = persistent_loader
        model.train()
        correct = total = 0
        loss_sum = 0.0
        started = time.perf_counter()
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            rates = model(images)
            target = F.one_hot(labels, model.output_classes).to(rates.dtype)
            raw_loss = F.mse_loss(rates, target)
            loss = raw_loss * (model.output_classes / 10.0)
            loss.backward()
            optimizer.step()
            total += labels.numel()
            correct += int((rates.argmax(1) == labels).sum().item())
            loss_sum += float(raw_loss.item()) * labels.numel()
            reset_model(model)
        scheduler.step()
        row = {
            "epoch": epoch + 1,
            "train_accuracy": correct / total,
            "raw_rate_mse": loss_sum / total,
            "seconds": time.perf_counter() - started,
        }
        history.append(row)
        print(json.dumps({"epoch_complete": epoch + 1, "train_accuracy": row["train_accuracy"], "seconds": row["seconds"]}), flush=True)
    return history


@torch.no_grad()
def evaluate(model, test_set, *, seed: int, batch_size: int, device: torch.device, anchors_only: bool = False):
    seed_everything(seed + (200_000 if anchors_only else 100_000))
    loader = make_loader(test_set, batch_size=batch_size, shuffle=False, seed=seed, device=device)
    model.eval()
    total = correct = 0
    loss_sum = input_sum = tail_sum = 0.0
    input_elements = tail_elements = 0
    group_totals = GroupTotals()
    class_counts = torch.zeros(model.k, dtype=torch.int64)
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        rates, state = model(images, anchors_only=anchors_only, return_input_state=True)
        target = F.one_hot(labels, model.output_classes).to(rates.dtype)
        total += labels.numel()
        correct += int((rates.argmax(1) == labels).sum().item())
        loss_sum += float(F.mse_loss(rates, target).item()) * labels.numel()
        sequence = state["sequence"]
        input_sum += float(sequence.sum().item())
        input_elements += sequence.numel()
        class_ids = state["class_ids"]
        if isinstance(class_ids, torch.Tensor):
            class_counts += torch.bincount(class_ids.detach().cpu().reshape(-1), minlength=model.k)
            tail = sequence[2:]
            tail_sum += float(tail.sum().item())
            tail_elements += tail.numel()
            if not anchors_only:
                group_totals.update(class_ids, model.k)
        reset_model(model)
    result: dict[str, Any] = {
        "accuracy": correct / total,
        "rate_mse": loss_sum / total,
        "examples": total,
        "mean_input_activity": input_sum / input_elements,
    }
    if int(class_counts.sum()):
        probabilities = class_counts.float() / class_counts.sum()
        entropy = -(probabilities[probabilities > 0] * probabilities[probabilities > 0].log()).sum()
        result.update(group_totals.metrics())
        result.update(
            {
                "class_counts": class_counts.tolist(),
                "class_perplexity": float(entropy.exp().item()),
                "tail_spike_rate": tail_sum / tail_elements,
            }
        )
    return result


def run_one(
    *, dataset: str, arm: str, seed: int, epochs: int, batch_size: int,
    data_root: Path, output_dir: Path, device: torch.device, pilot: bool,
):
    namespace = output_dir / ("pilot" if pilot else "runs")
    run_id = f"{dataset}__{arm}__k7__canonical__s{seed}__e{epochs}"
    record_path = namespace / f"{run_id}.json"
    if record_path.exists():
        return json.loads(record_path.read_text(encoding="utf-8"))
    train_set, test_set, raw, channels, output_classes, image_size, exclude_zero = dataset_spec(dataset, data_root)
    thresholds = fixed_quantiles_from_uint8(raw, MAIN_K, exclude_zero=exclude_zero)
    model = new_paired_model(
        arm=arm,
        seed=seed,
        input_channels=channels,
        output_classes=output_classes,
        image_size=image_size,
        thresholds=thresholds,
    ).to(device)
    started = time.perf_counter()
    history = train_model(model, train_set, seed=seed, epochs=epochs, batch_size=batch_size, device=device)
    if pilot:
        record = {
            "status": "PILOT_EXCLUDED_FROM_PAPER",
            "run_id": run_id,
            "dataset": dataset,
            "arm": arm,
            "seed": seed,
            "epochs": epochs,
            "history": history,
            "elapsed_seconds": time.perf_counter() - started,
            "official_test_evaluated": False,
        }
        atomic_json(record_path, record)
        return record
    metrics = evaluate(model, test_set, seed=seed, batch_size=batch_size, device=device)
    if arm in {"oqe", "single_hybrid", "non_nested"}:
        anchors = evaluate(model, test_set, seed=seed, batch_size=batch_size, device=device, anchors_only=True)
        metrics["anchor_only_accuracy"] = anchors["accuracy"]
        metrics["tail_utility_pp"] = 100.0 * (metrics["accuracy"] - anchors["accuracy"])
    checkpoint = output_dir / "checkpoints" / f"{run_id}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "dataset": dataset,
            "arm": arm,
            "seed": seed,
            "epochs": epochs,
            "k": MAIN_K,
            "thresholds": thresholds,
            "canonical_order": list(range(MAIN_K - 1)),
        },
        checkpoint,
    )
    record = {
        "status": "FORMAL_COMPLETE",
        "run_id": run_id,
        "dataset": dataset,
        "split": "official test",
        "arm": arm,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "k": MAIN_K,
        "tail_steps": MAIN_K - 1 if arm in {"oqe", "single_hybrid", "non_nested"} else None,
        "total_steps": MAIN_K + 1,
        "thresholds": [float(x) for x in thresholds],
        "history": history,
        "metrics": metrics,
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(checkpoint),
        "official_test_evaluated": True,
    }
    atomic_json(record_path, record)
    print(json.dumps({"completed": run_id, "accuracy": metrics["accuracy"], "seconds": record["elapsed_seconds"]}), flush=True)
    return record


def specifications(args):
    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    arms = MAIN_ARMS if args.arm == "all" else (args.arm,)
    seeds = SEEDS if args.seed is None else (args.seed,)
    specs = []
    for dataset in datasets:
        for seed in seeds:
            for arm in arms:
                epochs = args.rgb_epochs if dataset.startswith("cifar") else GRAYSCALE_EPOCHS
                specs.append((dataset, arm, seed, epochs))
        if dataset == "fashion_mnist" and args.include_non_nested:
            for seed in seeds:
                specs.append((dataset, "non_nested", seed, GRAYSCALE_EPOCHS))
    return specs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    parser.add_argument("--arm", choices=(*MAIN_ARMS, "non_nested", "all"), default="all")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--rgb-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "oqe_v6_main")
    parser.add_argument("--include-non-nested", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for dataset, arm, seed, epochs in specifications(args):
        run_one(
            dataset=dataset,
            arm=arm,
            seed=seed,
            epochs=(1 if args.pilot else epochs),
            batch_size=args.batch_size,
            data_root=args.data_root.resolve(),
            output_dir=args.output_dir.resolve(),
            device=device,
            pilot=args.pilot,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
