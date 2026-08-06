"""Tests for RVIR-v4 production alpha/beta/split ownership."""

# pylint: disable=missing-function-docstring,too-few-public-methods

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from boundflow.runtime.rvir_v4_production_state import (
    ProductionOptimizerPolicyV4,
    ProductionStateBuilderV4,
    ProductionStateSnapshotV4,
    ProductionTensorRole,
    capture_alpha_layout_state_v4,
    capture_alpha_state_v4,
    capture_module_alpha_beta_state_v4,
    capture_split_history_v4,
    diff_production_state_v4,
    production_snapshot_from_payload_v4,
    production_snapshot_to_payload_v4,
)


class _SparseBeta:
    def __init__(self, values: torch.Tensor) -> None:
        self.val = values
        self.loc = torch.tensor([[4, 7], [9, 0]], dtype=torch.long)
        self.sign = torch.tensor([[1.0, -1.0], [-1.0, 0.0]])
        self.bias = None


class _Node:
    def __init__(self, *, plural: bool = True) -> None:
        self.name = "relu0"
        self.alpha = {"output": torch.full((2, 1, 2, 3), 0.5)}
        self.inputs = [type("Input", (), {"output_shape": (1, 4)})()]
        self.alpha_indices = [torch.tensor([0, 2, 3], dtype=torch.long)]
        self.alpha_lookup_idx = {"output": None}
        beta = [_SparseBeta(torch.tensor([[0.1, 0.2], [0.3, 0.0]]))]
        if plural:
            self.sparse_betas = beta
        else:
            self.sparse_beta = beta


def _policy() -> ProductionOptimizerPolicyV4:
    return ProductionOptimizerPolicyV4(
        iteration=10,
        alpha_learning_rate=0.5,
        beta_learning_rate=0.05,
        bound_lower=True,
        bound_upper=False,
        fix_intermediate_bounds=True,
        deterministic=True,
        stop_criterion_id="batch-any",
    )


def _snapshot(node: _Node, *, snapshot_id: str) -> ProductionStateSnapshotV4:
    tensors = capture_module_alpha_beta_state_v4([node], require_beta=True)
    history = capture_split_history_v4(
        [
            {"relu0": ([4, 7], [1.0, -1.0])},
            {"relu0": ([9], [-1.0])},
        ]
    )
    snapshot = ProductionStateSnapshotV4(
        snapshot_id=snapshot_id,
        tensors=tensors,
        history=history,
        optimizer_policy=_policy(),
    )
    snapshot.validate()
    return snapshot


def test_plural_sparse_betas_capture_all_fields_and_semantic_axes() -> None:
    node = _Node()
    snapshot = _snapshot(node, snapshot_id="pre")
    tensors = snapshot.tensor_map()

    assert tensors["alpha/relu0/output"].axes == (
        "alpha_polarity",
        "start_spec",
        "domain",
        "feature_axis_0",
    )
    assert tensors["beta/relu0/0/value"].role == ProductionTensorRole.BETA_VALUE
    assert tensors["beta/relu0/0/location"].value.dtype == torch.long
    assert tensors["beta/relu0/0/sign"].axes == ("domain", "history_slot")
    assert snapshot.stable_hash() == snapshot.stable_hash()


def test_singular_sparse_beta_field_cannot_satisfy_beta_phase() -> None:
    with pytest.raises(ValueError, match="sparse_betas plural"):
        capture_module_alpha_beta_state_v4([_Node(plural=False)], require_beta=True)


def test_history_location_or_sign_mismatch_fails_closed() -> None:
    node = _Node()
    tensors = capture_module_alpha_beta_state_v4([node], require_beta=True)
    bad_history = capture_split_history_v4([{"relu0": ([5], [1.0])}])
    snapshot = ProductionStateSnapshotV4(
        snapshot_id="bad",
        tensors=tensors,
        history=bad_history,
        optimizer_policy=_policy(),
    )

    with pytest.raises(ValueError, match="SparseBeta/history content differs"):
        snapshot.validate()


def test_empty_provider_history_bias_does_not_require_beta_bias_tensor() -> None:
    node = _Node()
    tensors = capture_module_alpha_beta_state_v4([node], require_beta=True)
    history = capture_split_history_v4([{"relu0": ([], [], [])}])
    snapshot = ProductionStateSnapshotV4(
        snapshot_id="empty-history",
        tensors=tensors,
        history=history,
        optimizer_policy=_policy(),
    )

    snapshot.validate()


def test_zero_relu_history_bias_can_use_implicit_sparse_beta_zero() -> None:
    node = _Node()
    tensors = capture_module_alpha_beta_state_v4([node], require_beta=True)
    history = capture_split_history_v4([{"relu0": ([4], [1.0], [0.0])}])
    snapshot = ProductionStateSnapshotV4(
        snapshot_id="implicit-zero-bias",
        tensors=tensors,
        history=history,
        optimizer_policy=_policy(),
    )

    snapshot.validate()


def test_nonzero_history_bias_requires_sparse_beta_bias_tensor() -> None:
    node = _Node()
    tensors = capture_module_alpha_beta_state_v4([node], require_beta=True)
    history = capture_split_history_v4([{"relu0": ([4], [1.0], [0.25])}])
    snapshot = ProductionStateSnapshotV4(
        snapshot_id="missing-nonlinear-bias",
        tensors=tensors,
        history=history,
        optimizer_policy=_policy(),
    )

    with pytest.raises(ValueError, match="SparseBeta/history content differs"):
        snapshot.validate()


def test_mutation_receipts_close_alpha_and_beta_values() -> None:
    before_node = _Node()
    after_node = _Node()
    after_node.alpha["output"].add_(0.1)
    after_node.sparse_betas[0].val.add_(0.2)
    before = _snapshot(before_node, snapshot_id="pre")
    after = _snapshot(after_node, snapshot_id="post")

    receipts = diff_production_state_v4(before, after)
    changed = {receipt.semantic_path for receipt in receipts if receipt.changed}

    assert changed == {"alpha/relu0/output", "beta/relu0/0/value"}


def test_snapshot_tamper_is_rejected() -> None:
    snapshot = _snapshot(_Node(), snapshot_id="pre")
    tensor = snapshot.tensors[0]
    tensor.value.add_(1.0)

    with pytest.raises(ValueError, match="content differs"):
        snapshot.validate()


def test_mutation_schema_drift_is_rejected() -> None:
    before = _snapshot(_Node(), snapshot_id="pre")
    after = _snapshot(_Node(), snapshot_id="post")
    alpha_index = next(
        index
        for index, tensor in enumerate(after.tensors)
        if tensor.semantic_path == "alpha/relu0/output"
    )
    changed = list(after.tensors)
    changed[alpha_index] = replace(
        changed[alpha_index], axes=("domain",) * changed[alpha_index].value.ndim
    )
    drifted = replace(after, tensors=tuple(changed))

    with pytest.raises(ValueError, match="mutable tensor schema drift"):
        diff_production_state_v4(before, drifted)


def test_plain_payload_roundtrip_and_digest_tamper() -> None:
    snapshot = _snapshot(_Node(), snapshot_id="pre")
    payload = production_snapshot_to_payload_v4(snapshot)

    restored = production_snapshot_from_payload_v4(payload)

    assert restored.stable_hash() == snapshot.stable_hash()
    payload["snapshot_hash"] = "0" * 64
    with pytest.raises(ValueError, match="payload hash differs"):
        production_snapshot_from_payload_v4(payload)


def test_history_score_and_depth_are_digest_bound() -> None:
    entries = capture_split_history_v4([{"relu0": ([4], [1.0], [0.0], [0.75], [3.0])}])

    assert entries[0].scores == (0.75,)
    assert entries[0].depths == (3.0,)


def test_sparse_alpha_feature_layout_is_owned_and_validated() -> None:
    node = _Node()
    builder = ProductionStateBuilderV4()
    capture_alpha_state_v4({"relu0": node.alpha}, builder)
    capture_alpha_layout_state_v4([node], builder)
    snapshot = ProductionStateSnapshotV4(
        snapshot_id="alpha-layout",
        tensors=builder.finish(),
        history=(),
        optimizer_policy=_policy(),
    )

    snapshot.validate()
    roles = {tensor.role for tensor in snapshot.tensors}
    assert ProductionTensorRole.ALPHA_FEATURE_SHAPE in roles
    assert ProductionTensorRole.ALPHA_FEATURE_INDEX in roles
