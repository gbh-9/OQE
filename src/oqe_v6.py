"""General finite-language input encoders and matched SNNs for OQE V6."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import functional, layer, neuron, surrogate


MAIN_K = 7
MAIN_STEPS = MAIN_K + 1
MAIN_ARMS = ("direct", "rate", "latency", "single_hybrid", "oqe")
CANONICAL_ORDER = tuple(range(MAIN_K - 1))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fixed_quantiles_from_uint8(data: torch.Tensor, k: int, *, exclude_zero: bool) -> torch.Tensor:
    """Return strict, deterministic empirical cuts from uint8 training pixels."""
    values = data.reshape(-1).to(torch.int64)
    histogram = torch.bincount(values, minlength=256)
    if exclude_zero:
        histogram[0] = 0
    cumulative = histogram.cumsum(0)
    total = int(cumulative[-1].item())
    if total <= 0:
        raise ValueError("quantile population is empty")
    cuts: list[int] = []
    previous = 0
    for rank in range(1, k):
        target = math.ceil(rank * total / k)
        value = int(torch.searchsorted(cumulative, torch.tensor(target)).item())
        value = max(value, previous + 1)
        value = min(value, 254 - ((k - 1) - rank))
        cuts.append(value)
        previous = value
    return torch.tensor(cuts, dtype=torch.float32) / 255.0


def nested_codebook(k: int, *, device=None, dtype=torch.float32) -> torch.Tensor:
    ranks = torch.arange(k, device=device).reshape(k, 1)
    steps = torch.arange(k - 1, device=device).reshape(1, k - 1)
    return (steps < ranks).to(dtype)


def single_spike_codebook(k: int, *, device=None, dtype=torch.float32) -> torch.Tensor:
    codebook = torch.zeros((k, k - 1), device=device, dtype=dtype)
    for class_id in range(1, k):
        codebook[class_id, (k - 1) - class_id] = 1
    return codebook


def non_nested_codebook(k: int, *, device=None, dtype=torch.float32) -> torch.Tensor:
    """Deterministic weight-matched cyclic rotations of canonical unary words."""
    base = nested_codebook(k, device=device, dtype=dtype)
    rows = [base[0]]
    for class_id in range(1, k):
        rows.append(torch.roll(base[class_id], shifts=class_id - 1, dims=0))
    return torch.stack(rows)


def codebook_for(encoding: str, k: int, *, device=None, dtype=torch.float32) -> torch.Tensor:
    if encoding == "oqe":
        return nested_codebook(k, device=device, dtype=dtype)
    if encoding == "single_hybrid":
        return single_spike_codebook(k, device=device, dtype=dtype)
    if encoding == "non_nested":
        return non_nested_codebook(k, device=device, dtype=dtype)
    raise ValueError(f"encoding has no finite codebook: {encoding}")


class FixedFiniteEncoder(nn.Module):
    def __init__(self, *, encoding: str, thresholds: torch.Tensor, k: int = MAIN_K) -> None:
        super().__init__()
        if encoding not in {"oqe", "single_hybrid", "non_nested"}:
            raise ValueError(encoding)
        if tuple(thresholds.shape) != (k - 1,):
            raise ValueError("threshold shape mismatch")
        if not bool(torch.all(thresholds[1:] > thresholds[:-1])):
            raise ValueError("thresholds must be strict")
        self.encoding = encoding
        self.k = int(k)
        self.tail_steps = self.k - 1
        self.register_buffer("thresholds", thresholds.detach().clone().float())
        self.register_buffer("codebook", codebook_for(encoding, self.k))

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        class_ids = (images.unsqueeze(0) >= self.thresholds.reshape(-1, 1, 1, 1, 1)).sum(0).long()
        tail = self.codebook.to(device=images.device, dtype=images.dtype)[class_ids]
        return tail.permute(4, 0, 1, 2, 3).contiguous(), class_ids


class MatchedInputSNN(nn.Module):
    def __init__(
        self,
        *,
        encoding: str,
        input_channels: int,
        output_classes: int,
        image_size: int,
        thresholds: torch.Tensor,
        k: int = MAIN_K,
        base_channels: int | None = None,
        hidden_width: int | None = None,
    ) -> None:
        super().__init__()
        if encoding not in {*MAIN_ARMS, "non_nested"}:
            raise ValueError(encoding)
        self.encoding = encoding
        self.k = int(k)
        self.t_steps = self.k + 1
        self.input_channels = int(input_channels)
        self.output_classes = int(output_classes)
        self.image_size = int(image_size)
        self.encoder: FixedFiniteEncoder | None = None
        if encoding in {"oqe", "single_hybrid", "non_nested"}:
            self.encoder = FixedFiniteEncoder(encoding=encoding, thresholds=thresholds, k=k)

        if input_channels == 1:
            c1, c2 = 16, 32
            hidden = 128 if hidden_width is None else hidden_width
            self.conv1 = layer.Conv2d(1, c1, 3, padding=1, bias=False)
            self.bn1 = layer.BatchNorm2d(c1)
            self.if1 = neuron.IFNode(surrogate_function=surrogate.ATan())
            self.pool1 = layer.MaxPool2d(2, 2)
            self.conv2 = layer.Conv2d(c1, c2, 3, padding=1, bias=False)
            self.bn2 = layer.BatchNorm2d(c2)
            self.if2 = neuron.IFNode(surrogate_function=surrogate.ATan())
            self.pool2 = layer.MaxPool2d(2, 2)
            self.conv3 = None
            self.bn3 = None
            self.if3 = None
            self.pool3 = None
            flattened = c2 * (image_size // 4) * (image_size // 4)
        else:
            c1 = 32 if base_channels is None else base_channels
            c2 = 2 * c1
            hidden = 512 if hidden_width is None else hidden_width
            self.conv1 = layer.Conv2d(input_channels, c1, 3, padding=1, bias=False)
            self.bn1 = layer.BatchNorm2d(c1)
            self.if1 = neuron.IFNode(surrogate_function=surrogate.ATan())
            self.pool1 = layer.MaxPool2d(2, 2)
            self.conv2 = layer.Conv2d(c1, c2, 3, padding=1, bias=False)
            self.bn2 = layer.BatchNorm2d(c2)
            self.if2 = neuron.IFNode(surrogate_function=surrogate.ATan())
            self.pool2 = layer.MaxPool2d(2, 2)
            self.conv3 = layer.Conv2d(c2, c2, 3, padding=1, bias=False)
            self.bn3 = layer.BatchNorm2d(c2)
            self.if3 = neuron.IFNode(surrogate_function=surrogate.ATan())
            self.pool3 = layer.MaxPool2d(2, 2)
            flattened = c2 * (image_size // 8) * (image_size // 8)

        self.flatten = layer.Flatten()
        self.fc1 = layer.Linear(flattened, hidden, bias=False)
        self.if_hidden = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.fc2 = layer.Linear(hidden, output_classes, bias=False)
        self.if_out = neuron.IFNode(surrogate_function=surrogate.ATan())
        functional.set_step_mode(self, step_mode="m")

    def input_sequence(self, images: torch.Tensor, *, anchors_only: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.encoding == "direct":
            return images.unsqueeze(0).expand(self.t_steps, -1, -1, -1, -1), None
        if self.encoding == "rate":
            expanded = images.unsqueeze(0).expand(self.t_steps, -1, -1, -1, -1)
            return (torch.rand_like(expanded) < expanded).to(images.dtype), None
        if self.encoding == "latency":
            bins = ((1.0 - images) * (self.t_steps - 1)).round().long()
            sequence = F.one_hot(bins, num_classes=self.t_steps).permute(4, 0, 1, 2, 3)
            return sequence.to(images.dtype), None
        assert self.encoder is not None
        tail, class_ids = self.encoder(images)
        if anchors_only:
            tail = torch.zeros_like(tail)
        anchors = images.unsqueeze(0).expand(2, -1, -1, -1, -1)
        return torch.cat((anchors, tail), 0), class_ids

    def forward_from_first_currents(self, currents: torch.Tensor) -> torch.Tensor:
        first = self.pool1(self.if1(self.bn1(currents)))
        second = self.pool2(self.if2(self.bn2(self.conv2(first))))
        if self.conv3 is not None:
            assert self.bn3 is not None and self.if3 is not None and self.pool3 is not None
            second = self.pool3(self.if3(self.bn3(self.conv3(second))))
        hidden = self.if_hidden(self.fc1(self.flatten(second)))
        return self.if_out(self.fc2(hidden)).mean(0)

    def forward(self, images: torch.Tensor, *, anchors_only: bool = False, return_input_state: bool = False):
        sequence, class_ids = self.input_sequence(images, anchors_only=anchors_only)
        rates = self.forward_from_first_currents(self.conv1(sequence))
        if return_input_state:
            return rates, {"sequence": sequence.detach(), "class_ids": class_ids}
        return rates


def reset_model(model: nn.Module) -> None:
    functional.reset_net(model)


def backbone_state(model: MatchedInputSNN) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items() if not name.startswith("encoder.")}


def load_backbone_state(model: MatchedInputSNN, state: dict[str, torch.Tensor]) -> None:
    current = model.state_dict()
    for name, value in state.items():
        if name not in current or current[name].shape != value.shape:
            raise ValueError(f"backbone mismatch: {name}")
        current[name] = value.clone()
    model.load_state_dict(current, strict=True)


@dataclass
class GroupTotals:
    output_patches: int = 0
    valid_routes: int = 0
    surviving_nonzero_groups: int = 0
    floor_groups: int = 0

    def update(self, class_ids: torch.Tensor, k: int) -> None:
        batch, channels, height, width = class_ids.shape
        ones = torch.ones((1, channels, height, width), device=class_ids.device)
        valid = F.unfold(ones, 3, padding=1)
        routes_per_patch = valid.sum(1).to(torch.int64)
        self.output_patches += batch * valid.shape[-1]
        self.valid_routes += batch * int(routes_per_patch.sum().item())
        self.floor_groups += batch * int(torch.minimum(routes_per_patch, torch.tensor(k - 1, device=routes_per_patch.device)).sum().item())
        for class_id in range(1, k):
            membership = F.unfold((class_ids == class_id).float(), 3, padding=1)
            self.surviving_nonzero_groups += int((membership.sum(1) > 0).sum().item())

    def metrics(self) -> dict[str, float | int]:
        denominator = max(self.valid_routes, 1)
        return {
            "groups_per_patch": self.surviving_nonzero_groups / max(self.output_patches, 1),
            "route_contraction": 1.0 - self.surviving_nonzero_groups / denominator,
            "cardinality_floor": 1.0 - self.floor_groups / denominator,
            "output_patches": self.output_patches,
            "valid_routes": self.valid_routes,
            "surviving_nonzero_groups": self.surviving_nonzero_groups,
        }

