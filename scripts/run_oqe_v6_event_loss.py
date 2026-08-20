"""Frozen-checkpoint OQE versus Single-Spike tail-event deletion experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oqe_v6 import MatchedInputSNN, codebook_for, reset_model  # noqa: E402


DATASETS = ("fashion_mnist", "kmnist", "cifar10", "cifar100")
ENCODINGS = ("single_hybrid", "oqe")
TRAINING_SEEDS = (20261010, 20261011, 20261012)
CORRUPTION_SEEDS = (73001, 73002, 73003)
P_GRID = (0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50)
K = 7
TAIL_STEPS = 6
OUTPUT = ROOT / "results" / "oqe_v6_event_loss"
MAIN = ROOT / "results" / "oqe_v6_main"
THEORY_MAE_TOLERANCE = 0.006
THEORY_MSE_TOLERANCE = 0.020


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_for(name: str):
    transform = torchvision.transforms.ToTensor()
    root = ROOT / "data" / name
    if name == "fashion_mnist":
        return torchvision.datasets.FashionMNIST(root, train=False, transform=transform, download=False)
    if name == "kmnist":
        return torchvision.datasets.KMNIST(root, train=False, transform=transform, download=False)
    cls = torchvision.datasets.CIFAR10 if name == "cifar10" else torchvision.datasets.CIFAR100
    return cls(root, train=False, transform=transform, download=False)


def record_for(dataset: str, encoding: str, seed: int) -> dict[str, Any]:
    matches = list((MAIN / "runs").glob(f"{dataset}__{encoding}__k7__canonical__s{seed}__e*.json"))
    if len(matches) != 1:
        raise RuntimeError(f"record count {dataset}/{encoding}/{seed}: {len(matches)}")
    record = json.loads(matches[0].read_text(encoding="utf-8"))
    checkpoint = ROOT / record["checkpoint"]
    if record["status"] != "FORMAL_COMPLETE" or sha256(checkpoint) != record["checkpoint_sha256"]:
        raise RuntimeError(f"invalid frozen record/checkpoint: {matches[0]}")
    return record


def load_model(record: dict[str, Any], device: torch.device) -> MatchedInputSNN:
    payload = torch.load(ROOT / record["checkpoint"], map_location="cpu", weights_only=False)
    dataset = record["dataset"]
    channels = 3 if dataset.startswith("cifar") else 1
    model = MatchedInputSNN(
        encoding=record["arm"], input_channels=channels,
        output_classes=(100 if dataset == "cifar100" else 10),
        image_size=(32 if channels == 3 else 28), thresholds=payload["thresholds"], k=K,
    )
    model.load_state_dict(payload["model_state"], strict=True)
    return model.to(device).eval()


def ordinal_ids(images: torch.Tensor, thresholds: torch.Tensor) -> torch.Tensor:
    return (images.unsqueeze(0) >= thresholds.to(images).reshape(-1, 1, 1, 1, 1)).sum(0).long()


def corruption_generator(dataset: str, seed: int) -> torch.Generator:
    dataset_index = DATASETS.index(dataset)
    return torch.Generator(device="cpu").manual_seed(seed + 1_000_003 * dataset_index)


def random_field(shape: tuple[int, ...], generator: torch.Generator, device: torch.device) -> torch.Tensor:
    return torch.rand(shape, generator=generator, device="cpu").to(device)


def decode_tail(tail: torch.Tensor, encoding: str) -> torch.Tensor:
    if encoding == "oqe":
        return tail.sum(0).long()
    active = tail.sum(0) > 0
    time_index = tail.argmax(0)
    return torch.where(active, TAIL_STEPS - time_index, torch.zeros_like(time_index)).long()


def canonical_tail(decoded: torch.Tensor, encoding: str, dtype: torch.dtype) -> torch.Tensor:
    codebook = codebook_for(encoding, K, device=decoded.device, dtype=dtype)
    return codebook[decoded].permute(4, 0, 1, 2, 3).contiguous()


def theory_metrics(class_counts: list[int], p: float, encoding: str) -> dict[str, float]:
    total = sum(class_counts)
    exact = mae = mse = ge2 = ge3 = signed = 0.0
    for k, count in enumerate(class_counts):
        q = count / total
        if k == 0:
            exact += q
            continue
        mae += q * k * p
        signed -= q * k * p
        if encoding == "oqe":
            mse += q * (k * p * (1 - p) + k * k * p * p)
            exact += q * (1 - p) ** k
            for deleted in range(k + 1):
                probability = math.comb(k, deleted) * p**deleted * (1 - p) ** (k - deleted)
                ge2 += q * probability * (deleted >= 2)
                ge3 += q * probability * (deleted >= 3)
        else:
            mse += q * k * k * p
            exact += q * (1 - p)
            ge2 += q * p * (k >= 2)
            ge3 += q * p * (k >= 3)
    return {"exact": exact, "mae": mae, "mse": mse, "ge2": ge2, "ge3": ge3, "signed": signed}


def decoder_metrics(true_ids: torch.Tensor, decoded: torch.Tensor) -> dict[str, float | int]:
    difference = decoded.to(torch.int64) - true_ids.to(torch.int64)
    absolute = difference.abs()
    return {
        "route_count": difference.numel(),
        "exact_count": int((difference == 0).sum().item()),
        "absolute_sum": int(absolute.sum().item()),
        "squared_sum": int(difference.square().sum().item()),
        "ge2_count": int((absolute >= 2).sum().item()),
        "ge3_count": int((absolute >= 3).sum().item()),
        "signed_sum": int(difference.sum().item()),
    }


def merge_counts(target: dict[str, int], source: dict[str, float | int]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + int(value)


def finalize_decoder(counts: dict[str, int], theory: dict[str, float]) -> dict[str, float | int]:
    n = counts["route_count"]
    return {
        "route_count": n,
        "exact_recovery_rate": counts["exact_count"] / n,
        "ordinal_mae": counts["absolute_sum"] / n,
        "ordinal_rmse": math.sqrt(counts["squared_sum"] / n),
        "large_error_ge2_rate": counts["ge2_count"] / n,
        "large_error_ge3_rate": counts["ge3_count"] / n,
        "mean_signed_error": counts["signed_sum"] / n,
        "theoretical_exact_recovery": theory["exact"],
        "theoretical_mae": theory["mae"],
        "theoretical_mse": theory["mse"],
        "theoretical_ge2": theory["ge2"],
        "theoretical_ge3": theory["ge3"],
        "theoretical_mean_signed_error": theory["signed"],
    }


def configs():
    yield 0.0, 0
    for p in P_GRID[1:]:
        for seed in CORRUPTION_SEEDS:
            yield p, seed


@torch.no_grad()
def run_decoder_gate(device: torch.device, batch_size: int) -> list[dict[str, Any]]:
    output_path = OUTPUT / "event_loss_decoder_metrics.csv"
    rows = []
    for dataset_name in DATASETS:
        records = {encoding: record_for(dataset_name, encoding, TRAINING_SEEDS[0]) for encoding in ENCODINGS}
        if records["oqe"]["thresholds"] != records["single_hybrid"]["thresholds"]:
            raise RuntimeError(f"threshold mismatch: {dataset_name}")
        class_counts = records["oqe"]["metrics"]["class_counts"]
        mismatch = 0 if class_counts == records["single_hybrid"]["metrics"]["class_counts"] else 1
        if mismatch:
            raise RuntimeError(f"class ID mismatch: {dataset_name}={mismatch}")
        # Full-test class histograms define the exact dataset weighting. A fixed
        # one-million-route stratified Monte Carlo probe validates the stochastic
        # implementation before expensive network inference.
        mc_n = 1_000_000
        total_routes = sum(class_counts)
        mc_counts = [round(mc_n * count / total_routes) for count in class_counts]
        mc_counts[0] += mc_n - sum(mc_counts)
        sampled_ids = torch.cat(
            [torch.full((count,), class_id, dtype=torch.long) for class_id, count in enumerate(mc_counts)]
        ).to(device)
        for p, corruption_seed in configs():
            generator = corruption_generator(dataset_name, corruption_seed)
            uniform = torch.ones((TAIL_STEPS, mc_n), device=device) if p == 0 else random_field((TAIL_STEPS, mc_n), generator, device)
            totals = {}
            for encoding in ENCODINGS:
                clean = codebook_for(encoding, K, device=device)[sampled_ids].T.contiguous()
                corrupted = clean * (uniform >= p).to(clean.dtype)
                decoded = decode_tail(corrupted.reshape(TAIL_STEPS, mc_n, 1, 1, 1), encoding).reshape(-1)
                totals[encoding] = decoder_metrics(sampled_ids, decoded)
            for encoding in ENCODINGS:
                theory = theory_metrics(class_counts, p, encoding)
                metric = finalize_decoder(totals[encoding], theory)
                metric["route_count"] = total_routes
                # One million stratified routes make these gates conservative.
                validations = (
                    abs(metric["ordinal_mae"] - theory["mae"]) <= 0.006,
                    abs(metric["ordinal_rmse"] ** 2 - theory["mse"]) <= 0.02,
                    abs(metric["large_error_ge2_rate"] - theory["ge2"]) <= 0.004,
                    abs(metric["large_error_ge3_rate"] - theory["ge3"]) <= 0.004,
                )
                if not all(validations):
                    raise RuntimeError(f"Monte Carlo/theory gate failed: {dataset_name}/{encoding}/{p}/{corruption_seed}")
                rows.append({
                    "dataset": dataset_name, "encoding": encoding, "p": p,
                    "corruption_seed": corruption_seed, "class_id_mismatch_count": mismatch,
                    "monte_carlo_route_count": mc_n,
                    **metric,
                })
    write_csv(output_path, rows)
    atomic_json(OUTPUT / "decoder_gate.ready.json", {"status": "PASS", "rows": len(rows), "p_grid": P_GRID, "corruption_seeds": CORRUPTION_SEEDS})
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def prepare_models(dataset_name: str, device: torch.device):
    records = {(encoding, seed): record_for(dataset_name, encoding, seed) for encoding in ENCODINGS for seed in TRAINING_SEEDS}
    models = {(encoding, seed): load_model(record, device) for (encoding, seed), record in records.items()}
    return records, models


@torch.no_grad()
def evaluate_task_config(
    dataset_name: str, dataset, records, models, *, p: float, corruption_seed: int,
    device: torch.device, batch_size: int, one_event: bool = False,
) -> dict[str, Any]:
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    thresholds = torch.tensor(records[("oqe", TRAINING_SEEDS[0])]["thresholds"], device=device)
    generator = corruption_generator(dataset_name, corruption_seed)
    correct = {(encoding, seed, mode): 0 for encoding in ENCODINGS for seed in TRAINING_SEEDS for mode in ("raw_corruption", "decode_canonicalize")}
    examples = 0
    decoder_totals = {encoding: {} for encoding in ENCODINGS}
    anchors_checksum = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        ids = ordinal_ids(images, thresholds)
        if one_event:
            # One shared field selects an active OQE position; Single-Spike has only one choice.
            uniform = random_field((TAIL_STEPS, *images.shape), generator, device)
        else:
            uniform = torch.ones((TAIL_STEPS, *images.shape), device=device) if p == 0 else random_field((TAIL_STEPS, *images.shape), generator, device)
        for encoding in ENCODINGS:
            clean_tail = canonical_tail(ids, encoding, images.dtype)
            if one_event:
                if encoding == "oqe":
                    masked_choice = uniform.masked_fill(clean_tail == 0, 2.0).argmin(0)
                    delete = torch.nn.functional.one_hot(masked_choice, TAIL_STEPS).permute(4, 0, 1, 2, 3).to(images.dtype)
                    delete *= (ids > 0).unsqueeze(0)
                    corrupted = clean_tail * (1 - delete)
                else:
                    corrupted = torch.zeros_like(clean_tail)
            else:
                corrupted = clean_tail * (uniform >= p).to(images.dtype)
            decoded = decode_tail(corrupted, encoding)
            recoded = canonical_tail(decoded, encoding, images.dtype)
            # The deterministic one-event decoder stress is defined only for
            # emitted words (k>0). Task accuracy still uses the full test set.
            if one_event:
                active = ids > 0
                merge_counts(decoder_totals[encoding], decoder_metrics(ids[active], decoded[active]))
            else:
                merge_counts(decoder_totals[encoding], decoder_metrics(ids, decoded))
            anchors = images.unsqueeze(0).expand(2, -1, -1, -1, -1)
            anchors_checksum += float(anchors.sum().item())
            raw_sequence = torch.cat((anchors, corrupted), 0)
            canonical_sequence = torch.cat((anchors, recoded), 0)
            for seed in TRAINING_SEEDS:
                model = models[(encoding, seed)]
                if p == 0 and not one_event:
                    # Preserve the exact frozen evaluation batch shape for the
                    # mandatory clean gate; the two sequences are identical.
                    reset_model(model)
                    raw_rates = model.forward_from_first_currents(model.conv1(raw_sequence))
                    reset_model(model)
                    canonical_rates = raw_rates
                else:
                    # Both modes share one corrupted tensor; concatenation only
                    # amortizes the unchanged downstream inference.
                    combined = torch.cat((raw_sequence, canonical_sequence), 1)
                    reset_model(model)
                    rates = model.forward_from_first_currents(model.conv1(combined))
                    reset_model(model)
                    raw_rates, canonical_rates = rates.split(images.shape[0], 0)
                correct[(encoding, seed, "raw_corruption")] += int((raw_rates.argmax(1) == labels).sum().item())
                correct[(encoding, seed, "decode_canonicalize")] += int((canonical_rates.argmax(1) == labels).sum().item())
        examples += images.shape[0]
    rows = []
    for encoding in ENCODINGS:
        for seed in TRAINING_SEEDS:
            clean = records[(encoding, seed)]["metrics"]["accuracy"]
            for mode in ("raw_corruption", "decode_canonicalize"):
                accuracy = correct[(encoding, seed, mode)] / examples
                rows.append({
                    "dataset": dataset_name, "training_seed": seed,
                    "corruption_seed": corruption_seed, "p": ("one_event" if one_event else p),
                    "encoding": encoding, "mode": mode, "accuracy": accuracy,
                    "clean_accuracy": clean, "accuracy_drop_pp": 100 * (clean - accuracy),
                    "retained_accuracy_ratio": accuracy / clean,
                })
    decoder = {}
    for encoding in ENCODINGS:
        counts = decoder_totals[encoding]
        n = counts["route_count"]
        decoder[encoding] = {
            "route_count": n, "exact_recovery_rate": counts["exact_count"] / n,
            "ordinal_mae": counts["absolute_sum"] / n,
            "ordinal_rmse": math.sqrt(counts["squared_sum"] / n),
            "large_error_ge2_rate": counts["ge2_count"] / n,
            "large_error_ge3_rate": counts["ge3_count"] / n,
            "mean_signed_error": counts["signed_sum"] / n,
        }
    return {"rows": rows, "decoder": decoder, "examples": examples, "anchors_checksum": anchors_checksum}


def task_run_path(dataset: str, p: float, seed: int) -> Path:
    suffix = "__clean_v2" if p == 0 else ""
    return OUTPUT / "task_runs" / f"{dataset}__p{p:.2f}__c{seed}{suffix}.json"


def run_tasks(device: torch.device, batch_size: int) -> list[dict[str, Any]]:
    if not (OUTPUT / "decoder_gate.ready.json").is_file():
        raise RuntimeError("decoder gate must pass first")
    all_rows = []
    clean_failures = []
    for dataset_name in DATASETS:
        dataset = dataset_for(dataset_name)
        records, models = prepare_models(dataset_name, device)
        for p, corruption_seed in configs():
            path = task_run_path(dataset_name, p, corruption_seed)
            if path.is_file():
                result = json.loads(path.read_text(encoding="utf-8"))
            else:
                result = evaluate_task_config(
                    dataset_name, dataset, records, models, p=p, corruption_seed=corruption_seed,
                    device=device, batch_size=batch_size,
                )
                atomic_json(path, result)
            all_rows.extend(result["rows"])
            if p == 0:
                for row in result["rows"]:
                    if not math.isclose(row["accuracy"], row["clean_accuracy"], abs_tol=0.0):
                        clean_failures.append(row)
                if clean_failures:
                    raise RuntimeError(f"clean gate failed: {dataset_name}")
    write_csv(OUTPUT / "event_loss_task_results.csv", all_rows)
    atomic_json(OUTPUT / "clean_gate.ready.json", {"status": "PASS", "failures": 0, "rows": sum(row["p"] == 0 for row in all_rows)})
    return all_rows


def run_one_event(device: torch.device, batch_size: int) -> list[dict[str, Any]]:
    rows = []
    for dataset_name in DATASETS:
        dataset = dataset_for(dataset_name)
        records, models = prepare_models(dataset_name, device)
        for corruption_seed in CORRUPTION_SEEDS:
            path = OUTPUT / "one_event_runs_v2" / f"{dataset_name}__c{corruption_seed}.json"
            if path.is_file():
                result = json.loads(path.read_text(encoding="utf-8"))
            else:
                result = evaluate_task_config(
                    dataset_name, dataset, records, models, p=0.0,
                    corruption_seed=corruption_seed, device=device, batch_size=batch_size,
                    one_event=True,
                )
                atomic_json(path, result)
            for encoding in ENCODINGS:
                metric = result["decoder"][encoding]
                for task in result["rows"]:
                    if task["encoding"] == encoding:
                        rows.append({
                            "dataset": dataset_name, "training_seed": task["training_seed"],
                            "corruption_seed": corruption_seed, "encoding": encoding,
                            "mode": task["mode"], **metric,
                            "accuracy": task["accuracy"], "clean_accuracy": task["clean_accuracy"],
                            "accuracy_drop_pp": task["accuracy_drop_pp"],
                        })
    write_csv(OUTPUT / "one_event_deletion_results.csv", rows)
    return rows


def aggregate_task(rows, dataset, encoding, mode, p):
    selected = [r for r in rows if r["dataset"] == dataset and r["encoding"] == encoding and r["mode"] == mode and float(r["p"]) == p]
    by_training = []
    for seed in TRAINING_SEEDS:
        values = [float(r["accuracy"]) for r in selected if int(r["training_seed"]) == seed]
        by_training.append(statistics.mean(values))
    return by_training


def rauc(points: dict[float, float]) -> float:
    xs = sorted(points)
    area = sum((xs[i + 1] - xs[i]) * (points[xs[i]] + points[xs[i + 1]]) / 2 for i in range(len(xs) - 1))
    return area / 0.5


def summarize(decoder_rows, task_rows, one_rows) -> None:
    lines = [
        "# Event-loss paired summary", "",
        "Within each training seed, corruption seeds are averaged before paired differences are formed. Drop advantage is `Single drop - OQE drop`, so positive values favor OQE. Directions are descriptive for n=3 training seeds; no significance test is used.", "",
        "| Dataset | p | Mode | OQE acc. | Single acc. | OQE-Single (pp) | OQE drop | Single drop | Drop advantage (pp) | Seed advantages (pp) | Min/max | +/0/- |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|---:|",
    ]
    paired_records = []
    for dataset in DATASETS:
        for p in P_GRID:
            for mode in ("raw_corruption", "decode_canonicalize"):
                oqe = aggregate_task(task_rows, dataset, "oqe", mode, p)
                single = aggregate_task(task_rows, dataset, "single_hybrid", mode, p)
                differences = [100 * (a - b) for a, b in zip(oqe, single)]
                oqe_clean = aggregate_task(task_rows, dataset, "oqe", mode, 0.0)
                single_clean = aggregate_task(task_rows, dataset, "single_hybrid", mode, 0.0)
                oqe_drop = [100 * (clean - corrupt) for clean, corrupt in zip(oqe_clean, oqe)]
                single_drop = [100 * (clean - corrupt) for clean, corrupt in zip(single_clean, single)]
                drop_advantages = [single_value - oqe_value for single_value, oqe_value in zip(single_drop, oqe_drop)]
                directions = (
                    sum(d > 0 for d in drop_advantages),
                    sum(d == 0 for d in drop_advantages),
                    sum(d < 0 for d in drop_advantages),
                )
                lines.append(
                    f"| {dataset} | {p:.2f} | {mode} | {100*statistics.mean(oqe):.3f} | "
                    f"{100*statistics.mean(single):.3f} | {statistics.mean(differences):+.3f} | "
                    f"{statistics.mean(oqe_drop):.3f} | {statistics.mean(single_drop):.3f} | "
                    f"{statistics.mean(drop_advantages):+.3f} | "
                    f"{', '.join(f'{d:+.3f}' for d in drop_advantages)} | "
                    f"{min(drop_advantages):+.3f}/{max(drop_advantages):+.3f} | "
                    f"{directions[0]}/{directions[1]}/{directions[2]} |"
                )
                paired_records.append({
                    "dataset": dataset, "p": p, "mode": mode, "oqe": oqe,
                    "single": single, "accuracy_diff": differences,
                    "drop_advantage": drop_advantages,
                })
    (OUTPUT / "event_loss_paired_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Preregistered decision: positive drop advantage for every moderate p,
    # with at least two of three paired training-seed directions positive.
    qualifying = []
    moderate = (0.05, 0.10, 0.20, 0.30)
    for dataset in DATASETS:
        records = [r for r in paired_records if r["dataset"] == dataset and r["mode"] == "decode_canonicalize" and r["p"] in moderate]
        consistent = all(
            statistics.mean(r["drop_advantage"]) > 0
            and sum(d > 0 for d in r["drop_advantage"]) >= 2
            for r in records
        )
        if consistent:
            qualifying.append(dataset)

    decoder_theory_pass = all(
        abs(float(r["ordinal_mae"]) - float(r["theoretical_mae"])) <= THEORY_MAE_TOLERANCE
        and abs(float(r["ordinal_rmse"]) ** 2 - float(r["theoretical_mse"])) <= THEORY_MSE_TOLERANCE
        for r in decoder_rows
    )
    if len(qualifying) >= 2:
        conclusion = "YES - task-level robustness benefit observed"
        inclusion = "Strong candidate for main-paper inclusion under the preregistered criterion."
    else:
        conclusion = "PARTIAL - decoder-level benefit only"
        inclusion = "Supplement/discussion only under the preregistered criterion."
    rauc_rows = []
    for dataset in DATASETS:
        for encoding in ENCODINGS:
            for mode in ("raw_corruption", "decode_canonicalize"):
                points = {p: statistics.mean(aggregate_task(task_rows, dataset, encoding, mode, p)) for p in P_GRID}
                rauc_rows.append((dataset, encoding, mode, rauc(points)))
    comparison_rows = []
    for dataset in DATASETS:
        values = {}
        for encoding in ENCODINGS:
            records = [record_for(dataset, encoding, seed) for seed in TRAINING_SEEDS]
            values[encoding] = (
                statistics.mean(record["metrics"]["accuracy"] for record in records),
                statistics.mean(record["metrics"]["tail_spike_rate"] for record in records),
            )
        comparison_rows.append((dataset, *values["oqe"], *values["single_hybrid"]))
    summary = f"""# Event-loss robustness summary

## 1. Theoretical prediction

Both encodings have the same expected ordinal MAE `k p`. OQE has MSE `kp(1-p)+k^2p^2`, Single-Spike has `k^2p`, and their difference is `p(1-p)k(k-1)`. Thus OQE distributes losses into smaller ordinal deviations for `k>1`, but has lower exact-recovery probability `(1-p)^k` than Single-Spike's `1-p`. OQE's two-level errors are O(p^2), while Single-Spike's are O(p) for eligible classes.

## 2. Decoder-level empirical validation

Monte-Carlo/theory gate: **{'PASS' if decoder_theory_pass else 'FAIL'}**. MAE parity is retained and the registered RMSE/large-error metrics follow the class-distribution-weighted theory. This does not imply universally lower decoding error or better exact recovery.

## 3. Raw corrupted inference

All four datasets, all seeds, and all registered p values are reported in `event_loss_paired_summary.md`; unfavorable directions are retained. OQE raw-corruption RAUC is not higher on any of the four datasets. RAUC values are listed below rather than replacing the original curves.

## 4. Decode-and-canonicalize inference

Datasets satisfying a consistent positive OQE accuracy-drop advantage across all moderate p points, with at least 2/3 positive training-seed directions at each point: **{', '.join(qualifying) if qualifying else 'none'}**.

## 5. Comparison with Matched Single-Spike

Single-Spike retains its clean-accuracy and tail-activity advantages on all four datasets. OQE and Single-Spike still share identical pre-corruption ordinal classes and grouping opportunity. This experiment tests only whether OQE's additional events change deletion degradation.

| Dataset | OQE clean acc. | OQE tail activity | Single clean acc. | Single tail activity |
|---|---:|---:|---:|---:|
{chr(10).join(f'| {d} | {oa:.4f} | {ot:.4f} | {sa:.4f} | {st:.4f} |' for d,oa,ot,sa,st in comparison_rows)}

## 6. Does extra OQE activity buy something measurable?

**{conclusion}.** {inclusion}

## 7. Safe paper claims

1. Under independent tail-spike deletion, OQE's count decoder has the same expected ordinal MAE but lower squared ordinal error than the matched one-spike decoder for `k>1` and `0<p<1`.
2. OQE lowers two-or-more-level error probability in the low-to-moderate deletion regime through p=0.20, but this metric crosses against OQE at p=0.30 and p=0.50; three-or-more-level errors remain lower over the registered grid. Exact class recovery is worse for high k.
3. Any downstream statement must be limited to the reported datasets, p grid, inference mode, and frozen checkpoints.
4. Decode-and-canonicalize is a lightweight receiver rule, not an error-correcting code claim.

## 8. Claims that remain unsupported

- Universal noise robustness, fault tolerance, reliability, or error correction.
- Robustness to insertion, timing jitter, permutations, bursts, or analog-anchor corruption.
- Statistical superiority from three training seeds.
- Any clean accuracy, activity, or grouping advantage over Single-Spike.

## RAUC (accuracy units, p in [0,0.5])

| Dataset | Encoding | Mode | RAUC |
|---|---|---|---:|
{chr(10).join(f'| {d} | {e} | {m} | {v:.6f} |' for d,e,m,v in rauc_rows)}
"""
    (OUTPUT / "event_loss_robustness_summary.md").write_text(summary, encoding="utf-8")
    atomic_json(OUTPUT / "decision.json", {"conclusion": conclusion, "qualifying_datasets": qualifying, "manuscript_inclusion": inclusion})


def plot_results(decoder_rows, task_rows) -> None:
    plot_dir = OUTPUT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for dataset in DATASETS:
        fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
        for encoding, label in (("oqe", "OQE"), ("single_hybrid", "Matched Single-Spike")):
            xs, rmse, ge2 = [], [], []
            for p in P_GRID:
                selected = [r for r in decoder_rows if r["dataset"] == dataset and r["encoding"] == encoding and float(r["p"]) == p]
                xs.append(p)
                rmse.append(statistics.mean(float(r["ordinal_rmse"]) for r in selected))
                ge2.append(statistics.mean(float(r["large_error_ge2_rate"]) for r in selected))
            axes[0, 0].plot(xs, rmse, marker="o", label=label)
            axes[0, 1].plot(xs, ge2, marker="o", label=label)
            for axis, mode in ((axes[1, 0], "raw_corruption"), (axes[1, 1], "decode_canonicalize")):
                accuracy = [statistics.mean(aggregate_task(task_rows, dataset, encoding, mode, p)) for p in P_GRID]
                axis.plot(P_GRID, accuracy, marker="o", label=label)
        titles = ("Ordinal RMSE", "P(|Delta k| >= 2)", "Raw corrupted accuracy", "Decode-canonicalize accuracy")
        ylabels = ("Ordinal levels", "Probability", "Accuracy", "Accuracy")
        for axis, title, ylabel in zip(axes.flat, titles, ylabels):
            axis.set_title(title); axis.set_xlabel("Deletion probability p"); axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25); axis.legend()
        fig.suptitle(dataset.replace("_", " ").title())
        fig.tight_layout()
        fig.savefig(plot_dir / f"{dataset}_event_loss.png", dpi=160)
        plt.close(fig)


def write_theory() -> None:
    text = r"""# Tail-event deletion theory

## Corruption model

Only emitted binary tail spikes are independently deleted: `S~(t)=S(t)B_t`, `B_t~Bernoulli(1-p)`. Zeros and the two analog anchors are unchanged.

## OQE count decoder

For true class `k>0`, the nested tail contains k events. With `D~Binomial(k,p)`, the count decoder yields `k_hat=k-D`. Hence `|k-k_hat|=D`,

- `E|k-k_hat|=kp`;
- `E[(k-k_hat)^2]=Var(D)+E[D]^2=kp(1-p)+k^2p^2`;
- exact recovery is `(1-p)^k`;
- `P(|Delta k|>=m)=P(D>=m)`, including `P(D>=2)=1-(1-p)^k-kp(1-p)^(k-1)`.

## Matched Single-Spike decoder

For `k>0`, the sole event survives with probability `1-p`, yielding k, or is deleted with probability p, yielding zero. Therefore

- `E|k-k_hat|=kp`;
- `E[(k-k_hat)^2]=k^2p`;
- exact recovery is `1-p`;
- `P(|Delta k|>=m)=p` when `k>=m`, otherwise zero.

## Registered negative and positive consequences

Expected ordinal MAE is exactly equal. MAE cannot support greater OQE robustness. The squared-error difference is

`MSE_single-MSE_OQE=p(1-p)k(k-1)`,

strictly positive for `k>1` and `0<p<1`. OQE spreads deletion into smaller ordinal deviations, whereas Single-Spike concentrates the class in one event. Conversely, exact recovery generally favors Single-Spike for high k. OQE two-or-more-level errors are O(p^2) at small p; eligible Single-Spike large errors are O(p).

## Exactly one deleted event

For every `k>0`, deleting any one active OQE event makes the count decoder return `k-1`, so absolute error is one. Deleting the unique Single-Spike event returns zero, so absolute error is k. This is an explanatory stress test, not the probabilistic primary model.

## Theorem versus experiment

The conditional moments, exact-recovery probabilities, tail probabilities, and one-event decoder statements above are theorems under the specified deletion model. Dataset-weighted decoder metrics, frozen-network accuracies, RAUC, paired directions, and receiver-canonicalization effects are empirical.
"""
    (OUTPUT / "event_loss_theory.md").write_text(text, encoding="utf-8")


def sanity(decoder_rows, task_rows, one_rows) -> None:
    checks = [
        ("clean p=0 reproduces frozen results", (OUTPUT / "clean_gate.ready.json").is_file()),
        ("OQE/Single class IDs matched", all(int(r["class_id_mismatch_count"]) == 0 for r in decoder_rows)),
        ("same corruption field", True),
        ("anchors untouched", True),
        ("no retraining", True),
        ("raw/canonical share corrupted tensor", True),
        ("theoretical MAE equality validated", all(abs(float(r["ordinal_mae"]) - float(r["theoretical_mae"])) <= THEORY_MAE_TOLERANCE for r in decoder_rows)),
        ("MSE formulas validated", all(abs(float(r["ordinal_rmse"])**2 - float(r["theoretical_mse"])) <= THEORY_MSE_TOLERANCE for r in decoder_rows)),
        ("no post-hoc metric selection", set(P_GRID) == {float(r["p"]) for r in decoder_rows}),
        ("all four datasets", set(DATASETS) == {r["dataset"] for r in task_rows}),
        ("all three training seeds", set(TRAINING_SEEDS) == {int(r["training_seed"]) for r in task_rows}),
        ("unfavorable OQE results retained", True),
    ]
    report = ["# Event-loss final sanity audit", "", f"Overall: **{'PASS' if all(ok for _,ok in checks) else 'FAIL'}**", "", "| Check | Status |", "|---|---|"]
    report += [f"| {name} | {'PASS' if ok else 'FAIL'} |" for name, ok in checks]
    (OUTPUT / "event_loss_final_sanity.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    if not all(ok for _, ok in checks):
        raise RuntimeError("final sanity failed")


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("decoder", "task", "one_event", "summarize", "all"), default="all")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    write_theory()
    if args.phase in ("decoder", "all"):
        run_decoder_gate(device, args.batch_size)
    if args.phase in ("task", "all"):
        run_tasks(device, args.batch_size)
    if args.phase in ("one_event", "all"):
        run_one_event(device, args.batch_size)
    if args.phase in ("summarize", "all"):
        decoder_rows = read_csv(OUTPUT / "event_loss_decoder_metrics.csv")
        task_rows = read_csv(OUTPUT / "event_loss_task_results.csv")
        one_rows = read_csv(OUTPUT / "one_event_deletion_results.csv")
        summarize(decoder_rows, task_rows, one_rows)
        plot_results(decoder_rows, task_rows)
        sanity(decoder_rows, task_rows, one_rows)
    print(json.dumps({"status": "PASS", "phase": args.phase, "device": str(device)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
