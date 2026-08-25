"""Fail-closed unit tests for the MR3 P-anchor production bridge."""

# pylint: disable=missing-function-docstring,protected-access,too-few-public-methods

from __future__ import annotations

import pytest
import torch

from boundflow.runtime.mr3_production_p_anchor_bridge import (
    MR3PAnchorBridgeReceiptV1,
)
from scripts.run_mr3_production_p_anchor_bridge_worker import _BridgeTracker


def _receipt(**changes):
    values = {
        "evaluation_count": 10,
        "forward_launch_count": 10,
        "backward_launch_count": 9,
        "empty_beta_tensor_count": 10,
        "empty_beta_numel": 0,
        "relu_conv_content_match_count": 10,
        "relu_conv_pointer_match_count": 0,
        "persistent_dense_a_count": 0,
        "fallback_count": 0,
        "eager_count": 0,
        "native_shadow_count": 0,
        "timing_recorded": False,
        "performance_claimed": False,
    }
    values.update(changes)
    return MR3PAnchorBridgeReceiptV1(**values)


def test_bridge_receipt_accepts_real_provider_empty_beta_and_pointer_copy() -> None:
    assert _receipt().to_dict()["backward_launch_count"] == 9


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("evaluation_count", 9),
        ("backward_launch_count", 8),
        ("empty_beta_tensor_count", 0),
        ("empty_beta_numel", 1),
        ("relu_conv_content_match_count", 9),
        ("persistent_dense_a_count", 1),
        ("fallback_count", 1),
        ("eager_count", 1),
        ("native_shadow_count", 1),
        ("timing_recorded", True),
        ("performance_claimed", True),
    ),
)
def test_bridge_receipt_rejects_contract_drift(field: str, value: object) -> None:
    with pytest.raises(ValueError, match="receipt differs"):
        _receipt(**{field: value}).validate()


def test_owner_rollback_restores_content_and_original_storage() -> None:
    owner = torch.tensor([0.25, 0.75], requires_grad=True)

    class _Node:
        alpha = {"/49": owner}
        sparse_betas = None
        beta = None
        split_beta = None

    class _Module:
        @staticmethod
        def nodes():
            return [_Node()]

    tracker = _BridgeTracker(torch, mode="bridge")
    snapshot = tracker._owner_snapshot(_Module())
    original_pointer = owner.data_ptr()
    owner.data = torch.tensor([9.0, 8.0]).data
    assert owner.data_ptr() != original_pointer

    receipt = tracker._restore_owner_snapshot(snapshot)

    assert torch.equal(owner, torch.tensor([0.25, 0.75]))
    assert owner.data_ptr() == original_pointer
    assert receipt["owner_content_hash_before"] == receipt["owner_content_hash_after"]
    assert receipt["owner_pointer_hash_before"] == receipt["owner_pointer_hash_after"]
