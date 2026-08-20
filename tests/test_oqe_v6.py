import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oqe_v6 import FixedFiniteEncoder, codebook_for, non_nested_codebook


def test_main_codebooks_are_finite_keyed_and_shape_matched():
    for name in ("oqe", "single_hybrid", "non_nested"):
        codebook = codebook_for(name, 7)
        assert codebook.shape == (7, 6)
        assert torch.unique(codebook, dim=0).shape[0] == 7


def test_single_spike_mapping_runs_late_to_early():
    codebook = codebook_for("single_hybrid", 7)
    assert codebook[0].sum() == 0
    for k in range(1, 7):
        assert codebook[k].sum() == 1
        assert codebook[k, 6 - k] == 1


def test_non_nested_matches_class_hamming_weight_but_breaks_containment():
    codebook = non_nested_codebook(7)
    assert torch.equal(codebook.sum(1), torch.arange(7, dtype=codebook.dtype))
    assert any(not torch.all(codebook[k] <= codebook[k + 1]) for k in range(1, 5))


def test_fixed_encoder_uses_quantile_class_as_direct_key():
    cuts = torch.arange(1, 7, dtype=torch.float32) / 7
    values = torch.arange(7, dtype=torch.float32).reshape(7, 1, 1, 1) / 7 + 1e-4
    encoder = FixedFiniteEncoder(encoding="oqe", thresholds=cuts)
    tail, class_ids = encoder(values)
    assert torch.equal(class_ids.reshape(-1), torch.arange(7))
    assert torch.equal(tail.sum(0).reshape(-1), torch.arange(7, dtype=tail.dtype))

