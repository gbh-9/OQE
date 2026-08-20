import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oqe_v6 import MatchedInputSNN, codebook_for, reset_model
from oqe_v6_executor import (
    generic_reconstruct_tail_currents,
    ordinary_first_currents,
    quotient_first_currents,
    single_spike_reconstruct_tail_currents,
    suffix_quotient_first_currents,
    suffix_reconstruct_tail_currents,
)


def test_general_quotient_matches_ordinary_in_fp64():
    torch.manual_seed(4)
    thresholds = torch.arange(1, 7, dtype=torch.float64) / 7
    model = MatchedInputSNN(
        encoding="oqe", input_channels=1, output_classes=10, image_size=28,
        thresholds=thresholds.float(),
    ).double().eval()
    images = torch.rand(2, 1, 28, 28, dtype=torch.float64)
    ordinary = ordinary_first_currents(model, images)
    quotient = quotient_first_currents(model, images)
    suffix = suffix_quotient_first_currents(model, images)
    assert torch.allclose(ordinary, quotient, atol=1e-12, rtol=0)
    assert torch.allclose(ordinary, suffix, atol=1e-12, rtol=0)
    reset_model(model)


def test_suffix_reconstruction_matches_generic_for_general_k():
    torch.manual_seed(7)
    for k in (3, 5, 7, 9):
        aggregates = torch.randn(k, 2, 4, 3, 3, dtype=torch.float64)
        aggregates[0].zero_()
        generic = generic_reconstruct_tail_currents(aggregates, codebook_for("oqe", k, dtype=torch.float64))
        suffix = suffix_reconstruct_tail_currents(aggregates)
        torch.testing.assert_close(generic, suffix, atol=1e-12, rtol=0)


def test_general_permutation_reindexes_suffix_currents():
    aggregates = torch.arange(5, dtype=torch.float64).reshape(5, 1, 1, 1, 1)
    permutation = torch.tensor([2, 0, 3, 1])
    canonical = suffix_reconstruct_tail_currents(aggregates)
    permuted = suffix_reconstruct_tail_currents(aggregates, permutation)
    assert torch.equal(permuted, canonical[permutation])


def test_single_spike_placement_matches_control_codebook():
    torch.manual_seed(8)
    aggregates = torch.randn(7, 2, 3, 4, 4)
    aggregates[0].zero_()
    generic = generic_reconstruct_tail_currents(aggregates, codebook_for("single_hybrid", 7))
    placed = single_spike_reconstruct_tail_currents(aggregates)
    assert torch.equal(generic, placed)
