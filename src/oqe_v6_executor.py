"""Functional unfused quotient-forward prototype for fixed-code OQE V6."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from oqe_v6 import MatchedInputSNN


@torch.no_grad()
def ordinal_class_ids(model: MatchedInputSNN, images: torch.Tensor) -> torch.Tensor:
    """Compute the exact frozen ordinal key without materializing temporal words."""
    if model.encoding != "oqe" or model.encoder is None:
        raise ValueError("ordinal keys require an OQE model")
    thresholds = model.encoder.thresholds.to(device=images.device, dtype=images.dtype)
    return (images.unsqueeze(0) >= thresholds.reshape(-1, 1, 1, 1, 1)).sum(0).long()


@torch.no_grad()
def classwise_tail_aggregates(
    class_ids: torch.Tensor,
    weight: torch.Tensor,
    *,
    k: int,
    padding: int = 1,
) -> torch.Tensor:
    """Build A_k class-wise first-layer currents, including a zero A_0 slot."""
    if class_ids.ndim != 4:
        raise ValueError("expected BCHW class IDs")
    batch, channels, height, width = class_ids.shape
    outputs, weight_channels, kernel_h, kernel_w = weight.shape
    if channels != weight_channels:
        raise ValueError("class IDs and convolution weights disagree on channels")
    flat_weight = weight.reshape(outputs, -1)
    output_h = height + 2 * padding - kernel_h + 1
    output_w = width + 2 * padding - kernel_w + 1
    locations = output_h * output_w
    aggregates = torch.zeros(
        (k, batch, outputs, locations), device=weight.device, dtype=weight.dtype
    )
    for class_id in range(1, k):
        membership = F.unfold(
            (class_ids == class_id).to(weight.dtype),
            kernel_size=(kernel_h, kernel_w),
            padding=padding,
        )
        aggregates[class_id] = torch.einsum("or,brl->bol", flat_weight, membership)
    return aggregates.reshape(k, batch, outputs, output_h, output_w)


@torch.no_grad()
def generic_reconstruct_tail_currents(
    aggregates: torch.Tensor, codebook: torch.Tensor
) -> torch.Tensor:
    """Generic class-by-word reconstruction from already-built A_k currents."""
    if aggregates.ndim != 5 or codebook.ndim != 2 or aggregates.shape[0] != codebook.shape[0]:
        raise ValueError("expected A[K,B,O,H,W] and codebook[K,T]")
    tail_steps = codebook.shape[1]
    currents = torch.zeros(
        (tail_steps, *aggregates.shape[1:]), device=aggregates.device, dtype=aggregates.dtype
    )
    for class_id in range(1, codebook.shape[0]):
        currents.add_(
            codebook[class_id].to(device=aggregates.device, dtype=aggregates.dtype).reshape(
                tail_steps, 1, 1, 1, 1
            )
            * aggregates[class_id].unsqueeze(0)
        )
    return currents


@torch.no_grad()
def suffix_reconstruct_tail_currents(
    aggregates: torch.Tensor, permutation: torch.Tensor | None = None
) -> torch.Tensor:
    """Reconstruct nested OQE currents using K-2 reverse cumulative additions."""
    if aggregates.ndim != 5 or aggregates.shape[0] < 2:
        raise ValueError("expected A[K,B,O,H,W] with K >= 2")
    k = aggregates.shape[0]
    suffix: list[torch.Tensor] = [aggregates[-1]]
    for rank in range(k - 2, 0, -1):
        suffix.append(suffix[-1] + aggregates[rank])
    by_rank = torch.stack(list(reversed(suffix)), 0)
    if permutation is None:
        return by_rank
    permutation = permutation.to(device=aggregates.device, dtype=torch.long)
    if tuple(permutation.shape) != (k - 1,) or sorted(permutation.cpu().tolist()) != list(range(k - 1)):
        raise ValueError("permutation must contain ordinal ranks 0..K-2")
    return by_rank.index_select(0, permutation)


@torch.no_grad()
def single_spike_reconstruct_tail_currents(aggregates: torch.Tensor) -> torch.Tensor:
    """Route A_k to the matching one-hot tail timestep used by the V6 control."""
    if aggregates.ndim != 5 or aggregates.shape[0] < 2:
        raise ValueError("expected A[K,B,O,H,W] with K >= 2")
    return aggregates[1:].flip(0).contiguous()


@torch.no_grad()
def quotient_tail_currents(
    class_ids: torch.Tensor,
    codebook: torch.Tensor,
    weight: torch.Tensor,
    *,
    padding: int = 1,
) -> torch.Tensor:
    """Existing generic quotient mode, now exposing A_k as a shared boundary."""
    if codebook.ndim != 2:
        raise ValueError("expected a [K,T] codebook")
    aggregates = classwise_tail_aggregates(
        class_ids, weight, k=codebook.shape[0], padding=padding
    )
    return generic_reconstruct_tail_currents(aggregates, codebook)


@torch.no_grad()
def suffix_quotient_tail_currents(
    class_ids: torch.Tensor,
    weight: torch.Tensor,
    *,
    k: int,
    permutation: torch.Tensor | None = None,
    padding: int = 1,
) -> torch.Tensor:
    aggregates = classwise_tail_aggregates(class_ids, weight, k=k, padding=padding)
    return suffix_reconstruct_tail_currents(aggregates, permutation=permutation)


@torch.no_grad()
def ordinary_first_currents(model: MatchedInputSNN, images: torch.Tensor) -> torch.Tensor:
    if model.encoding != "oqe" or model.encoder is None:
        raise ValueError("quotient prototype requires an OQE model")
    sequence, _ = model.input_sequence(images)
    return model.conv1(sequence)


@torch.no_grad()
def quotient_first_currents(model: MatchedInputSNN, images: torch.Tensor) -> torch.Tensor:
    if model.encoding != "oqe" or model.encoder is None:
        raise ValueError("quotient prototype requires an OQE model")
    class_ids = ordinal_class_ids(model, images)
    anchor = model.conv1(images.unsqueeze(0))
    tail_currents = quotient_tail_currents(
        class_ids,
        model.encoder.codebook,
        model.conv1.weight,
        padding=1,
    )
    return torch.cat((anchor.expand(2, -1, -1, -1, -1), tail_currents), 0)


@torch.no_grad()
def suffix_quotient_first_currents(model: MatchedInputSNN, images: torch.Tensor) -> torch.Tensor:
    if model.encoding != "oqe" or model.encoder is None:
        raise ValueError("suffix quotient requires an OQE model")
    class_ids = ordinal_class_ids(model, images)
    anchor = model.conv1(images.unsqueeze(0))
    tail_currents = suffix_quotient_tail_currents(
        class_ids,
        model.conv1.weight,
        k=model.k,
        padding=1,
    )
    return torch.cat((anchor.expand(2, -1, -1, -1, -1), tail_currents), 0)


@torch.no_grad()
def quotient_forward(model: MatchedInputSNN, images: torch.Tensor) -> torch.Tensor:
    return model.forward_from_first_currents(quotient_first_currents(model, images))


@torch.no_grad()
def suffix_quotient_forward(model: MatchedInputSNN, images: torch.Tensor) -> torch.Tensor:
    return model.forward_from_first_currents(suffix_quotient_first_currents(model, images))
