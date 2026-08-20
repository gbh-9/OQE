"""Correctness audit for generic and suffix-scan OQE quotient execution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torchvision


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oqe_v6 import MatchedInputSNN, reset_model  # noqa: E402
from oqe_v6_executor import (  # noqa: E402
    ordinary_first_currents,
    quotient_first_currents,
    suffix_quotient_first_currents,
)


def load_model(path: Path, device: torch.device, *, dtype: torch.dtype = torch.float32):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    dataset = payload["dataset"]
    model = MatchedInputSNN(
        encoding="oqe",
        input_channels=(3 if dataset.startswith("cifar") else 1),
        output_classes=(100 if dataset == "cifar100" else 10),
        image_size=(32 if dataset.startswith("cifar") else 28),
        thresholds=payload["thresholds"],
        k=payload["k"],
    )
    model.load_state_dict(payload["model_state"], strict=True)
    return payload, model.to(device=device, dtype=dtype).eval()


def dataset_for(name: str):
    transform = torchvision.transforms.ToTensor()
    if name == "fashion_mnist":
        return torchvision.datasets.FashionMNIST(ROOT / "data" / name, train=False, transform=transform, download=False)
    if name == "cifar10":
        return torchvision.datasets.CIFAR10(ROOT / "data" / name, train=False, transform=transform, download=False)
    raise ValueError(name)


def empty_pair() -> dict[str, float | int]:
    return {
        "current_max_abs_error": 0.0,
        "current_peak_normalized_error": 0.0,
        "first_spike_element_mismatches": 0,
        "final_rate_element_mismatches": 0,
        "prediction_mismatches": 0,
    }


def update_pair(
    result: dict[str, float | int],
    left_current: torch.Tensor,
    right_current: torch.Tensor,
    left_spikes: torch.Tensor,
    right_spikes: torch.Tensor,
    left_rates: torch.Tensor,
    right_rates: torch.Tensor,
) -> None:
    difference = (left_current - right_current).abs()
    maximum = float(difference.max().item())
    peak = max(float(left_current.abs().max().item()), 1e-30)
    result["current_max_abs_error"] = max(float(result["current_max_abs_error"]), maximum)
    result["current_peak_normalized_error"] = max(
        float(result["current_peak_normalized_error"]), maximum / peak
    )
    result["first_spike_element_mismatches"] = int(result["first_spike_element_mismatches"]) + int(
        (left_spikes != right_spikes).sum().item()
    )
    result["final_rate_element_mismatches"] = int(result["final_rate_element_mismatches"]) + int(
        (left_rates != right_rates).sum().item()
    )
    result["prediction_mismatches"] = int(result["prediction_mismatches"]) + int(
        (left_rates.argmax(1) != right_rates.argmax(1)).sum().item()
    )


@torch.no_grad()
def audit_checkpoint(path: Path, dataset, *, device: torch.device, batch_size: int) -> dict[str, Any]:
    payload, model = load_model(path, device)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    comparisons = {"generic_vs_suffix": empty_pair(), "ordinary_vs_suffix": empty_pair()}
    examples = 0
    for images, _ in loader:
        images = images.to(device)
        ordinary = ordinary_first_currents(model, images)
        generic = quotient_first_currents(model, images)
        suffix = suffix_quotient_first_currents(model, images)

        spikes = {}
        rates = {}
        for name, currents in (("ordinary", ordinary), ("generic", generic), ("suffix", suffix)):
            reset_model(model)
            spikes[name] = model.if1(model.bn1(currents))
            reset_model(model)
            rates[name] = model.forward_from_first_currents(currents)
        update_pair(
            comparisons["generic_vs_suffix"], generic, suffix,
            spikes["generic"], spikes["suffix"], rates["generic"], rates["suffix"],
        )
        update_pair(
            comparisons["ordinary_vs_suffix"], ordinary, suffix,
            spikes["ordinary"], spikes["suffix"], rates["ordinary"], rates["suffix"],
        )
        examples += images.shape[0]
        reset_model(model)

    # A small FP64 reconstruction probe isolates reassociation from FP32 effects.
    _, model64 = load_model(path, device, dtype=torch.float64)
    probe_images = torch.stack([dataset[index][0] for index in range(min(32, len(dataset)))]).to(
        device=device, dtype=torch.float64
    )
    generic64 = quotient_first_currents(model64, probe_images)
    suffix64 = suffix_quotient_first_currents(model64, probe_images)
    fp64_error = float((generic64 - suffix64).abs().max().item())
    reset_model(model64)
    return {
        "dataset": payload["dataset"],
        "seed": payload["seed"],
        "checkpoint": str(path.relative_to(ROOT)),
        "examples": examples,
        "comparisons": comparisons,
        "fp64_generic_vs_suffix_current_max_abs_error_32_examples": fp64_error,
    }


def sum_mismatches(rows: list[dict[str, Any]], comparison: str, field: str) -> int:
    return sum(int(row["comparisons"][comparison][field]) for row in rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--include-cifar10", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "oqe_v6_nestedness")
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = ROOT / "results" / "oqe_v6_main" / "checkpoints"
    specs = [
        (path, "fashion_mnist")
        for path in sorted(checkpoint_dir.glob("fashion_mnist__oqe__k7__canonical__s*__e3.pt"))
    ]
    if len(specs) != 3:
        raise RuntimeError("expected three Fashion-MNIST OQE checkpoints")
    if args.include_cifar10:
        cifar = sorted(checkpoint_dir.glob("cifar10__oqe__k7__canonical__s20261010__e20.pt"))
        if len(cifar) != 1:
            raise RuntimeError("expected one CIFAR-10 seed-20261010 checkpoint")
        specs.append((cifar[0], "cifar10"))
    datasets = {name: dataset_for(name) for _, name in specs}
    rows = [
        audit_checkpoint(path, datasets[name], device=device, batch_size=args.batch_size)
        for path, name in specs
    ]
    result = {
        "status": "PASS" if all(
            row["comparisons"][comparison]["prediction_mismatches"] == 0
            for row in rows for comparison in ("generic_vs_suffix", "ordinary_vs_suffix")
        ) else "FAIL",
        "device": str(device),
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "suffix_executor_correctness.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Suffix executor correctness", "",
        "## Implementation and derivation", "",
        "Both quotient modes first build the same class-wise currents `A_k` from the same ordinal IDs. Generic reconstruction applies the full codebook relation `I_t=sum_k A_k P_k(t)`. For canonical OQE, `P_k(t)=1[t<k]`, hence `I_t=sum_{k=t+1}^{K-1} A_k`, with boundary `I_(K-2)=A_(K-1)` and recurrence `I_t=I_(t+1)+A_(t+1)`. The suffix implementation therefore uses exactly K-2 vector additions after aggregation. Anchors and the downstream BN/IF/SNN path are unchanged.", "",
        "For a general permutation, suffix sums are constructed by ordinal rank and reindexed as `I_t=sum_{k>pi_t} A_k`. The current evaluated checkpoints are canonical.", "",
        f"Device: `{device}`. Fashion-MNIST uses all three frozen fixed-quantile checkpoints; " + ("one current CIFAR-10 checkpoint is additionally included." if args.include_cifar10 else "CIFAR-10 was not requested in this run."), "",
        "| Dataset | Seed | Examples | Comparison | Max current error | Peak-normalized | Spike mismatches | Rate mismatches | Prediction mismatches | FP64 generic/suffix max (32) |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        for comparison in ("generic_vs_suffix", "ordinary_vs_suffix"):
            metric = row["comparisons"][comparison]
            lines.append(
                f"| {row['dataset']} | {row['seed']} | {row['examples']} | {comparison} | "
                f"{metric['current_max_abs_error']:.9g} | {metric['current_peak_normalized_error']:.9g} | "
                f"{metric['first_spike_element_mismatches']} | {metric['final_rate_element_mismatches']} | "
                f"{metric['prediction_mismatches']} | {row['fp64_generic_vs_suffix_current_max_abs_error_32_examples']:.9g} |"
            )
    lines += ["", "## Totals", ""]
    for comparison in ("generic_vs_suffix", "ordinary_vs_suffix"):
        lines.append(
            f"- {comparison}: spikes={sum_mismatches(rows, comparison, 'first_spike_element_mismatches')}, "
            f"rates={sum_mismatches(rows, comparison, 'final_rate_element_mismatches')}, "
            f"predictions={sum_mismatches(rows, comparison, 'prediction_mismatches')}."
        )
    lines += ["", "The comparison is numerical, not a bitwise-equivalence requirement.", ""]
    (args.output_dir / "suffix_executor_correctness.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": result["status"], "rows": len(rows), "device": str(device)}, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
