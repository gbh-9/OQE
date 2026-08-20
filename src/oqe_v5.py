"""Lean multi-dataset OQE breadth models for the V5 paper study."""

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


MAIN_STEPS = 8
MAIN_CLASSES = 7
V4_PERMUTATION = (0, 4, 2, 5, 1, 3)
CANONICAL_PERMUTATION = tuple(range(6))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def foreground_quantiles(data: torch.Tensor, classes: int) -> torch.Tensor:
    """Strict empirical cuts from nonzero training pixels.

    Background zeros dominate MNIST-like images; conditioning fixed quantiles
    on foreground avoids duplicated zero-valued cuts.
    """
    histogram = torch.bincount(data.reshape(-1).to(torch.int64), minlength=256)
    histogram[0] = 0
    cumulative = histogram.cumsum(0)
    total = int(cumulative[-1].item())
    cuts = []
    for rank in range(1, classes):
        target = math.ceil(rank * total / classes)
        value = int(torch.searchsorted(cumulative, torch.tensor(target)).item())
        cuts.append(value / 255.0)
    cuts = torch.tensor(cuts, dtype=torch.float32)
    eps = 1.0 / 255.0
    for index in range(1, cuts.numel()):
        cuts[index] = torch.maximum(cuts[index], cuts[index - 1] + eps)
    return cuts.clamp(eps, 1.0 - eps)


def uniform_thresholds(classes: int) -> torch.Tensor:
    return torch.arange(1, classes, dtype=torch.float32) / classes


class OrderedLanguageEncoder(nn.Module):
    def __init__(
        self,
        *,
        policy: str,
        classes: int = MAIN_CLASSES,
        quantiles: torch.Tensor | None = None,
        permutation: Iterable[int] | None = None,
    ) -> None:
        super().__init__()
        if policy not in {"learned", "uniform", "quantile"}:
            raise ValueError(f"unknown OQE policy: {policy}")
        self.policy = policy
        self.classes = int(classes)
        self.tail_steps = self.classes - 1
        if policy == "learned":
            self.gap_logits = nn.Parameter(torch.zeros(self.classes))
            self.register_buffer("fixed_thresholds", torch.empty(0))
        else:
            self.register_parameter("gap_logits", None)
            cuts = uniform_thresholds(self.classes) if policy == "uniform" else quantiles
            if cuts is None or tuple(cuts.shape) != (self.tail_steps,):
                raise ValueError("invalid fixed thresholds")
            self.register_buffer("fixed_thresholds", cuts.detach().clone().float())
        self.alpha_logits = nn.Parameter(torch.full((self.tail_steps,), 2.0))
        ranks = tuple(range(self.tail_steps)) if permutation is None else tuple(permutation)
        if sorted(ranks) != list(range(self.tail_steps)):
            raise ValueError("permutation must contain every threshold rank once")
        self.register_buffer("rank_permutation", torch.tensor(ranks, dtype=torch.long))

    @property
    def ordered_thresholds(self) -> torch.Tensor:
        if self.policy != "learned":
            return self.fixed_thresholds
        gaps = self.gap_logits.softmax(0)
        return gaps.cumsum(0)[:-1]

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ordered = self.ordered_thresholds
        thresholds = ordered[self.rank_permutation].reshape(
            self.tail_steps, 1, 1, 1, 1
        )
        hard = (images.unsqueeze(0) >= thresholds).to(images.dtype)
        if self.training:
            alpha = F.softplus(self.alpha_logits).reshape(
                self.tail_steps, 1, 1, 1, 1
            )
            probability = (alpha * (images.unsqueeze(0) - thresholds)).sigmoid()
            spikes = hard + probability - probability.detach()
        else:
            spikes = hard
        return spikes, hard.sum(0).to(torch.long)


class SmallInputSNN(nn.Module):
    def __init__(
        self,
        *,
        encoding: str,
        classes: int = MAIN_CLASSES,
        quantiles: torch.Tensor | None = None,
        permutation: Iterable[int] | None = V4_PERMUTATION,
    ) -> None:
        super().__init__()
        self.encoding = encoding
        self.language_classes = int(classes)
        self.encoder: OrderedLanguageEncoder | None = None
        if encoding.startswith("oqe_"):
            policy = encoding.removeprefix("oqe_")
            self.encoder = OrderedLanguageEncoder(
                policy=policy,
                classes=classes,
                quantiles=quantiles,
                permutation=permutation,
            )
            self.t_steps = 2 + classes - 1
        elif encoding in {"direct", "rate", "latency"}:
            self.t_steps = MAIN_STEPS
        else:
            raise ValueError(f"unknown encoding: {encoding}")

        self.conv1 = layer.Conv2d(1, 16, 3, padding=1, bias=False)
        self.bn1 = layer.BatchNorm2d(16)
        self.if1 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.pool1 = layer.MaxPool2d(2, 2)
        self.conv2 = layer.Conv2d(16, 32, 3, padding=1, bias=False)
        self.bn2 = layer.BatchNorm2d(32)
        self.if2 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.pool2 = layer.MaxPool2d(2, 2)
        self.flatten = layer.Flatten()
        self.fc1 = layer.Linear(32 * 7 * 7, 128, bias=False)
        self.if3 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.fc2 = layer.Linear(128, 10, bias=False)
        self.if4 = neuron.IFNode(surrogate_function=surrogate.ATan())
        functional.set_step_mode(self, step_mode="m")

    def input_sequence(
        self, images: torch.Tensor, *, anchors_only: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
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
        return torch.cat((anchors, tail), dim=0), class_ids

    def forward(
        self,
        images: torch.Tensor,
        *,
        anchors_only: bool = False,
        return_input_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
        sequence, class_ids = self.input_sequence(images, anchors_only=anchors_only)
        first = self.pool1(self.if1(self.bn1(self.conv1(sequence))))
        second = self.pool2(self.if2(self.bn2(self.conv2(first))))
        hidden = self.if3(self.fc1(self.flatten(second)))
        rates = self.if4(self.fc2(hidden)).mean(0)
        if return_input_state:
            return rates, {"sequence": sequence.detach(), "class_ids": class_ids}
        return rates


def reset_model(model: nn.Module) -> None:
    functional.reset_net(model)


@dataclass
class GroupTotals:
    output_patches: int = 0
    valid_routes: int = 0
    surviving_nonzero_groups: int = 0

    def update(self, class_ids: torch.Tensor, classes: int) -> None:
        batch, channels, height, width = class_ids.shape
        ones = torch.ones((1, channels, height, width), device=class_ids.device)
        valid = F.unfold(ones, 3, padding=1)
        self.output_patches += batch * valid.shape[-1]
        self.valid_routes += batch * int(valid.sum().item())
        for index in range(1, classes):
            membership = F.unfold((class_ids == index).float(), 3, padding=1)
            self.surviving_nonzero_groups += int((membership.sum(1) > 0).sum().item())

    def metrics(self) -> dict[str, float | int]:
        return {
            "groups_per_patch": self.surviving_nonzero_groups / max(self.output_patches, 1),
            "route_contraction": 1.0
            - self.surviving_nonzero_groups / max(self.valid_routes, 1),
            "output_patches": self.output_patches,
            "valid_routes": self.valid_routes,
            "surviving_nonzero_groups": self.surviving_nonzero_groups,
        }
