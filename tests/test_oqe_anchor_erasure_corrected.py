import torch

from scripts.run_oqe_anchor_erasure_corrected import explicit_delete_one
from oqe_v6 import codebook_for


def _tail(ids, encoding):
    return codebook_for(encoding, 7)[ids].permute(4, 0, 1, 2, 3).contiguous()


def test_single_spike_every_nonzero_route_collapses_to_zero():
    ids = torch.arange(7).reshape(7, 1, 1, 1)
    tail = _tail(ids, "single_hybrid")
    corrupted, decoded, canonical, deleted = explicit_delete_one(
        tail, ids, torch.full(ids.shape, 0.73), "single_hybrid"
    )
    assert int(deleted.sum()) == 6
    assert torch.count_nonzero(corrupted) == 0
    assert torch.count_nonzero(decoded) == 0
    assert torch.count_nonzero(canonical) == 0


def test_oqe_every_nonzero_route_decodes_to_k_minus_one():
    ids = torch.arange(7).reshape(7, 1, 1, 1)
    tail = _tail(ids, "oqe")
    corrupted, decoded, canonical, deleted = explicit_delete_one(
        tail, ids, torch.full(ids.shape, 0.73), "oqe"
    )
    assert int(deleted.sum()) == 6
    assert torch.equal(decoded, torch.clamp(ids - 1, min=0))
    assert torch.equal(canonical.sum(0).long(), decoded)
