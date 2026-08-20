import torch

from scripts.run_oqe_v6_event_loss import canonical_tail, decode_tail, theory_metrics


def test_clean_codewords_decode_to_registered_ordinal_ids():
    ids = torch.arange(7).reshape(7, 1, 1, 1)
    for encoding in ("oqe", "single_hybrid"):
        tail = canonical_tail(ids, encoding, torch.float32)
        assert torch.equal(decode_tail(tail, encoding), ids)


def test_one_event_deletion_has_registered_conditional_geometry():
    ids = torch.arange(1, 7).reshape(6, 1, 1, 1)
    oqe = canonical_tail(ids, "oqe", torch.float32)
    single = canonical_tail(ids, "single_hybrid", torch.float32)
    for index, k in enumerate(range(1, 7)):
        oqe[k - 1, index] = 0
    single.zero_()
    assert torch.equal(decode_tail(oqe, "oqe"), ids - 1)
    assert torch.equal(decode_tail(single, "single_hybrid"), torch.zeros_like(ids))


def test_theoretical_mae_is_equal_and_single_mse_is_not_smaller():
    counts = [11, 13, 17, 19, 23, 29, 31]
    for p in (0.02, 0.10, 0.30, 0.50):
        oqe = theory_metrics(counts, p, "oqe")
        single = theory_metrics(counts, p, "single_hybrid")
        assert oqe["mae"] == single["mae"]
        assert oqe["mse"] <= single["mse"]
