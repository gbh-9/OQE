"""Microbenchmark OQE reconstruction geometry and full suffix quotient forwarding."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oqe_v6 import MatchedInputSNN, codebook_for, reset_model  # noqa: E402
from oqe_v6_executor import (  # noqa: E402
    generic_reconstruct_tail_currents,
    ordinal_class_ids,
    quotient_forward,
    single_spike_reconstruct_tail_currents,
    suffix_quotient_forward,
    suffix_reconstruct_tail_currents,
)


OUTPUT = ROOT / "results" / "oqe_v6_nestedness"


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(function, *, device: torch.device) -> dict[str, float | int]:
    warmup = 5 if device.type == "cuda" else 1
    repeats = 20 if device.type == "cuda" else 5
    for _ in range(warmup):
        function()
    synchronize(device)
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        synchronize(device)
        samples.append(1000.0 * (time.perf_counter() - started))
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "std_ms": statistics.pstdev(samples),
        "repeats": repeats,
    }


def device_name(device: torch.device) -> str:
    return torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"


@torch.no_grad()
def microbench(device: torch.device) -> list[dict[str, object]]:
    rows = []
    # K=7 covers both required batches; other K values are an optional batch-1 scaling probe.
    specs = sorted({(7, 1), (7, 128), *((k, 1) for k in (3, 5, 7, 9, 17, 33))})
    for k, batch in specs:
        generator = torch.Generator(device="cpu").manual_seed(17_000 + k + batch)
        aggregates = torch.randn(k, batch, 16, 28, 28, generator=generator).to(device)
        aggregates[0].zero_()
        codebook = codebook_for("oqe", k, device=device, dtype=aggregates.dtype)
        methods = {
            "generic_oqe": lambda: generic_reconstruct_tail_currents(aggregates, codebook),
            "suffix_oqe": lambda: suffix_reconstruct_tail_currents(aggregates),
            "single_spike": lambda: single_spike_reconstruct_tail_currents(aggregates),
        }
        for method, function in methods.items():
            measurement = timed(function, device=device)
            rows.append({
                "device": device_name(device), "batch": batch, "K": k, "method": method,
                **measurement,
            })
    return rows


def load_model(checkpoint: Path, device: torch.device) -> MatchedInputSNN:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = MatchedInputSNN(
        encoding="oqe", input_channels=1, output_classes=10, image_size=28,
        thresholds=payload["thresholds"], k=payload["k"],
    )
    model.load_state_dict(payload["model_state"], strict=True)
    return model.to(device).eval()


@torch.no_grad()
def full_benchmark(checkpoint: Path, device: torch.device) -> list[dict[str, object]]:
    model = load_model(checkpoint, device)
    dataset = torchvision.datasets.FashionMNIST(
        ROOT / "data" / "fashion_mnist", train=False,
        transform=torchvision.transforms.ToTensor(), download=False,
    )
    rows = []
    for batch in (1, 128):
        images = torch.stack([dataset[index][0] for index in range(batch)]).to(device)

        def ordinary():
            reset_model(model)
            return model(images)

        def generic():
            reset_model(model)
            return quotient_forward(model, images)

        def suffix():
            reset_model(model)
            return suffix_quotient_forward(model, images)

        for method, function in (("ordinary", ordinary), ("generic_quotient", generic), ("suffix_quotient", suffix)):
            rows.append({
                "device": device_name(device), "batch": batch, "method": method,
                **timed(function, device=device),
            })
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def cpu_stage_breakdown(checkpoint: Path) -> list[dict[str, object]]:
    device = torch.device("cpu")
    model = load_model(checkpoint, device)
    dataset = torchvision.datasets.FashionMNIST(
        ROOT / "data" / "fashion_mnist", train=False,
        transform=torchvision.transforms.ToTensor(), download=False,
    )
    rows = []
    stages = ("ordinal_class", "patch_unfold", "class_aggregation", "temporal_reconstruction", "anchors_and_downstream")
    for batch in (1, 128):
        images = torch.stack([dataset[index][0] for index in range(batch)])
        for mode in ("generic_quotient", "suffix_quotient"):
            samples = {stage: [] for stage in stages}
            for repeat in range(6):  # one warm-up followed by five recorded trials
                stamps = [time.perf_counter()]
                class_ids = ordinal_class_ids(model, images)
                stamps.append(time.perf_counter())
                memberships = [
                    F.unfold((class_ids == class_id).float(), kernel_size=(3, 3), padding=1)
                    for class_id in range(1, model.k)
                ]
                stamps.append(time.perf_counter())
                flat_weight = model.conv1.weight.reshape(model.conv1.out_channels, -1)
                aggregates = torch.zeros(
                    model.k, batch, model.conv1.out_channels, 28 * 28,
                    dtype=model.conv1.weight.dtype,
                )
                for class_id, membership in enumerate(memberships, 1):
                    aggregates[class_id] = torch.einsum("or,brl->bol", flat_weight, membership)
                aggregates = aggregates.reshape(model.k, batch, model.conv1.out_channels, 28, 28)
                stamps.append(time.perf_counter())
                if mode == "generic_quotient":
                    tail = generic_reconstruct_tail_currents(aggregates, model.encoder.codebook)
                else:
                    tail = suffix_reconstruct_tail_currents(aggregates)
                stamps.append(time.perf_counter())
                anchor = model.conv1(images.unsqueeze(0))
                currents = torch.cat((anchor.expand(2, -1, -1, -1, -1), tail), 0)
                reset_model(model)
                model.forward_from_first_currents(currents)
                reset_model(model)
                stamps.append(time.perf_counter())
                if repeat:
                    for index, stage in enumerate(stages):
                        samples[stage].append(1000.0 * (stamps[index + 1] - stamps[index]))
            total_mean = sum(statistics.mean(samples[stage]) for stage in stages)
            for stage in stages:
                rows.append({
                    "device": "CPU", "batch": batch, "mode": mode, "stage": stage,
                    "median_ms": statistics.median(samples[stage]),
                    "mean_ms": statistics.mean(samples[stage]),
                    "std_ms": statistics.pstdev(samples[stage]),
                    "mean_percent_total": 100.0 * statistics.mean(samples[stage]) / total_mean,
                    "repeats": 5,
                })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("microbench", "full", "all"), default="all")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda:0"))
    micro_rows = []
    full_rows = []
    stage_rows = []
    if args.mode in ("microbench", "all"):
        micro_rows = [row for device in devices for row in microbench(device)]
        write_csv(OUTPUT / "reconstruction_microbenchmark.csv", micro_rows)
    if args.mode in ("full", "all"):
        checkpoints = sorted(
            (ROOT / "results" / "oqe_v6_main" / "checkpoints").glob(
                "fashion_mnist__oqe__k7__canonical__s20261010__e3.pt"
            )
        )
        if len(checkpoints) != 1:
            raise RuntimeError("expected the Fashion-MNIST seed-20261010 checkpoint")
        full_rows = [row for device in devices for row in full_benchmark(checkpoints[0], device)]
        stage_rows = cpu_stage_breakdown(checkpoints[0])
        write_csv(OUTPUT / "suffix_full_forward_benchmark.csv", full_rows)
        write_csv(OUTPUT / "suffix_stage_timing_cpu.csv", stage_rows)
    if args.mode != "all":
        print(json.dumps({"mode": args.mode, "micro_rows": len(micro_rows), "full_rows": len(full_rows), "stage_rows": len(stage_rows)}))
        return 0

    def row_for(rows, device, batch, method, k=None):
        found = [r for r in rows if r["device"] == device and r["batch"] == batch and r["method"] == method and (k is None or r.get("K") == k)]
        if len(found) != 1:
            raise RuntimeError(f"ambiguous benchmark row: {device}/{batch}/{method}/{k}")
        return found[0]

    lines = [
        "# Suffix executor benchmark summary", "",
        "All measurements use the existing protocol: CPU warm-up 1/repeats 5; GPU warm-up 5/repeats 20; GPU synchronization before sampling and after each call. Values below are median milliseconds. The reconstruction benchmark starts with A_k already available and uses Fashion first-layer dimensions (16 outputs, 28x28 spatial map).", "",
        "| Device | Batch | Generic reconstruction | Suffix reconstruction | Single-Spike placement | Suffix/Generic | Single/Generic |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for device in [device_name(d) for d in devices]:
        for batch in (1, 128):
            generic = row_for(micro_rows, device, batch, "generic_oqe", 7)
            suffix = row_for(micro_rows, device, batch, "suffix_oqe", 7)
            single = row_for(micro_rows, device, batch, "single_spike", 7)
            lines.append(
                f"| {device} | {batch} | {generic['median_ms']:.6f} | {suffix['median_ms']:.6f} | "
                f"{single['median_ms']:.6f} | {suffix['median_ms']/generic['median_ms']:.3f} | "
                f"{single['median_ms']/generic['median_ms']:.3f} |"
            )
    lines += ["", "| Device | Batch | Ordinary full | Generic quotient full | Suffix quotient full | Suffix/Generic | Suffix/Ordinary |", "|---|---:|---:|---:|---:|---:|---:|"]
    for device in [device_name(d) for d in devices]:
        for batch in (1, 128):
            ordinary = row_for(full_rows, device, batch, "ordinary")
            generic = row_for(full_rows, device, batch, "generic_quotient")
            suffix = row_for(full_rows, device, batch, "suffix_quotient")
            lines.append(
                f"| {device} | {batch} | {ordinary['median_ms']:.6f} | {generic['median_ms']:.6f} | "
                f"{suffix['median_ms']:.6f} | {suffix['median_ms']/generic['median_ms']:.3f} | "
                f"{suffix['median_ms']/ordinary['median_ms']:.3f} |"
            )
    lines += ["", "## CPU stage timing", "", "A separate instrumented CPU pipeline records ordinal classification, unfold, class aggregation, reconstruction, and anchors/downstream. It is stored in `suffix_stage_timing_cpu.csv`; GPU stage instrumentation was intentionally omitted because per-stage synchronization would materially perturb these small kernels. The batch-128 instrumented rows have high variance and are diagnostic only; conclusions use the isolated reconstruction and end-to-end benchmarks instead.", "", "## Scope", "", "The microbenchmark measures only post-grouping temporal-current reconstruction. Full-forward timing additionally includes ordinal preparation, unfold/membership construction, class aggregation, materialization, anchors, and the downstream SNN. No end-to-end acceleration claim follows unless the full measurements support it. The K-scaling rows in the CSV use batch 1 and are descriptive; small GPU kernels can be dominated by launch overhead.", ""]
    (OUTPUT / "suffix_executor_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"micro_rows": len(micro_rows), "full_rows": len(full_rows), "stage_rows": len(stage_rows), "devices": [device_name(d) for d in devices]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
