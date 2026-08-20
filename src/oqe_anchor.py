"""Verified anchor-count input construction shared by release experiments."""

from __future__ import annotations

import torch

from oqe_v6 import MatchedInputSNN


class AnchorCountSNN(MatchedInputSNN):
    """Matched SNN with exactly 0, 1, or 2 unchanged analog anchor steps."""

    def __init__(self, *, anchor_count: int, **kwargs) -> None:
        super().__init__(**kwargs)
        if anchor_count not in (0, 1, 2):
            raise ValueError("anchor_count must be 0, 1, or 2")
        self.anchor_count = anchor_count

    def input_sequence(self, images: torch.Tensor, *, anchors_only: bool = False):
        if self.encoder is None:
            raise RuntimeError("anchor-count construction requires a temporal encoder")
        tail, ids = self.encoder(images)
        if anchors_only:
            tail = torch.zeros_like(tail)
        pieces = []
        if self.anchor_count:
            pieces.append(images.unsqueeze(0).expand(self.anchor_count, -1, -1, -1, -1))
        pieces.append(tail)
        return torch.cat(pieces, 0), ids


def assemble_anchor_sequence(images: torch.Tensor, tail: torch.Tensor, anchor_count: int) -> torch.Tensor:
    """Construct [tail], [x, tail], or [x, x, tail] without hidden padding."""
    if anchor_count not in (0, 1, 2):
        raise ValueError("anchor_count must be 0, 1, or 2")
    pieces = []
    if anchor_count:
        pieces.append(images.unsqueeze(0).expand(anchor_count, -1, -1, -1, -1))
    pieces.append(tail)
    return torch.cat(pieces, 0)
