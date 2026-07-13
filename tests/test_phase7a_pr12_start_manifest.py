"""Contracts for the PR-12 start manifest and held-out split."""

from scripts.start_phase7a_pr12 import (
    BASE_COMMIT,
    BASE_TAG,
    HELDOUT_SPLIT_ID,
    _heldout_split,
)


def test_pr12_base_is_the_frozen_pr11_tag() -> None:
    assert BASE_TAG == "pr11-validated-reduced"
    assert BASE_COMMIT.startswith("fee6cc0")


def test_final_heldout_is_disjoint_from_backend_gap_development_set() -> None:
    split = _heldout_split()
    assert split["split_id"] == HELDOUT_SPLIT_ID
    assert split["development"]["purpose"] == "motivation_and_debug_only"
    final_ids = [case["case_id"] for case in split["final_heldout"]]
    assert len(final_ids) == len(set(final_ids)) == 5
    assert {case["family"] for case in split["final_heldout"]} >= {
        "linear",
        "conv2d",
        "mini_resnet",
    }
