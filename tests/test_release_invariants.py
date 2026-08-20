import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oqe_anchor import assemble_anchor_sequence
from oqe_v6 import codebook_for, non_nested_codebook
from oqe_v6_executor import generic_reconstruct_tail_currents, suffix_reconstruct_tail_currents


def test_exact_k7_codebooks_and_oqe_geometry():
    oqe = codebook_for("oqe", 7, dtype=torch.int64)
    single = codebook_for("single_hybrid", 7, dtype=torch.int64)
    assert oqe.tolist() == [
        [0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1],
    ]
    assert single.tolist() == [
        [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
    ]
    assert torch.equal(oqe.sum(1), torch.arange(7))
    for k in range(7):
        for ell in range(7):
            assert int(torch.count_nonzero(oqe[k] != oqe[ell])) == abs(k - ell)
    assert all(bool(torch.all(oqe[k] <= oqe[k + 1])) for k in range(6))


def test_nonnested_weight_match_without_canonical_nesting():
    q = non_nested_codebook(7, dtype=torch.int64)
    assert torch.equal(q.sum(1), torch.arange(7))
    assert any(not bool(torch.all(q[k] <= q[k + 1])) for k in range(6))


def test_generic_and_suffix_reconstruction_match():
    generator = torch.Generator().manual_seed(23)
    aggregates = torch.randn(7, 2, 3, 4, 4, generator=generator, dtype=torch.float64)
    aggregates[0].zero_()
    generic = generic_reconstruct_tail_currents(aggregates, codebook_for("oqe", 7, dtype=torch.float64))
    suffix = suffix_reconstruct_tail_currents(aggregates)
    torch.testing.assert_close(generic, suffix, atol=1e-12, rtol=0)


def test_zero_anchor_single_after_exact_deletion_is_all_zero():
    images = torch.rand(2, 1, 3, 3)
    silent_tail = torch.zeros(6, *images.shape)
    sequence = assemble_anchor_sequence(images, silent_tail, anchor_count=0)
    assert sequence.shape[0] == 6
    assert torch.count_nonzero(sequence) == 0
