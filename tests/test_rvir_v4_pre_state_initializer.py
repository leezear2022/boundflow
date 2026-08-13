"""Formal pre-state native initializer tests for RVIR-v4 V4-2C."""

# pylint: disable=missing-function-docstring

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from boundflow.frontends.plain_crown_bound_ir import relu_split_state_hash
from boundflow.runtime.native_alpha_beta_optimization_state import (
    NativeAlphaBetaStateScope,
)
from boundflow.runtime.rvir_v4_pre_state_initializer import (
    initialize_rvir_v4_native_pre_state,
    ProductionPreStateIdentityV4,
    ProductionReluTopologyV4,
)
from boundflow.runtime.rvir_v4_production_state import (
    production_snapshot_from_payload_v4,
    production_tensor_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
CAPTURE = (
    ROOT
    / "artifacts/rvir-v4-optimizer-step/resnet2b-core-step-trace-v1/production_capture.pt"
)
TOPOLOGY = tuple(
    ProductionReluTopologyV4(*values, provider_start_node="/49")
    for values in (
        ("/input-4", "/input", "17"),
        ("/input-12", "/input-8", "19"),
        ("/input-16", "/39", "23"),
        ("/input-24", "/input-20", "25"),
        ("/45", "/44", "28"),
        ("/48", "/input-28", "31"),
    )
)
IDENTITY = ProductionPreStateIdentityV4(
    snapshot_hash="2a775b66559c20ddfc0bec97ec026898ba5eccfc984e02b217fcb7472d03a256",
    topology_hash="9be361625e492b1401a402fd19ad5d80ac06a977c74f137c7563e96de06bca35",
    history_hash="8921a052baa3a1444c468851f9a8be6429b23830982a61ee285b2cb2b115a08a",
    intermediate_bounds_hash=(
        "f82523fb83031f5d0699dc5ff15078a7b6be1c0ca03511f2d53093721288cf06"
    ),
)


def _capture() -> dict[str, object]:
    value = torch.load(CAPTURE, map_location="cpu", weights_only=True)
    assert isinstance(value, dict)
    return value


def _pre_snapshot():  # type: ignore[no-untyped-def]
    value = _capture()
    cores = value["cores"]
    assert isinstance(cores, list)
    return production_snapshot_from_payload_v4(cores[0]["pre_snapshot"])


def test_formal_pre_state_restores_six_dense_relus_and_round_trips_exact() -> None:
    mapping = initialize_rvir_v4_native_pre_state(
        _pre_snapshot(), TOPOLOGY, expected_identity=IDENTITY
    )

    assert mapping.identity == IDENTITY
    assert mapping.stable_hash() == (
        "cfcebf92fc58c269899d98cd65cc9454d7caa6051e2c9da46d415eda1fecf8df"
    )
    assert set(mapping.alphas) == {"17", "19", "23", "25", "28", "31"}
    assert len(mapping.round_trip_receipts) == 12
    assert all(
        receipt.maximum_absolute_difference == 0.0
        and receipt.mapped_source_hash == receipt.mapped_round_trip_hash
        and receipt.full_source_hash == receipt.full_round_trip_hash
        for receipt in mapping.round_trip_receipts
    )
    alpha_receipts = [
        receipt
        for receipt in mapping.round_trip_receipts
        if receipt.role.value == "alpha"
    ]
    assert len(alpha_receipts) == 6
    assert all(receipt.copy_through_element_count > 0 for receipt in alpha_receipts)
    assert sum(int((value != 0).sum().item()) for value in mapping.splits.values()) == 6


def test_formal_snapshot_and_trace_step_zero_bind_same_twelve_mutable_paths() -> None:
    capture = _capture()
    cores = capture["cores"]
    traces = capture["optimizer_step_traces"]
    assert isinstance(cores, list) and isinstance(traces, list)
    snapshot = production_snapshot_from_payload_v4(cores[0]["pre_snapshot"])
    step_rows = traces[0]["steps"][0]["state_tensors"]
    step = {row["semantic_path"]: row["content_sha256"] for row in step_rows}
    mutable = {
        tensor.semantic_path: tensor.content_sha256
        for tensor in snapshot.tensors
        if tensor.role.value in {"alpha", "beta_value"}
    }

    assert mutable == {path: digest for path, digest in step.items() if path in mutable}
    assert len(mutable) == 12


def test_upper_alpha_plane_is_copy_through_and_frozen_identity_detects_drift() -> None:
    snapshot = _pre_snapshot()
    tensor_index = next(
        index
        for index, tensor in enumerate(snapshot.tensors)
        if tensor.semantic_path == "alpha/%2F45/%2F49"
    )
    source = snapshot.tensors[tensor_index]
    value = source.value.clone()
    value[1, 0].reshape(-1)[0] = 0.25
    tensors = list(snapshot.tensors)
    tensors[tensor_index] = replace(
        source,
        value=value,
        content_sha256=production_tensor_sha256(value),
    )
    changed = replace(snapshot, tensors=tuple(tensors))
    changed.validate()

    baseline = initialize_rvir_v4_native_pre_state(snapshot, TOPOLOGY)
    changed_mapping = initialize_rvir_v4_native_pre_state(changed, TOPOLOGY)
    receipt = next(
        item
        for item in changed_mapping.round_trip_receipts
        if item.semantic_path == source.semantic_path
    )
    assert torch.equal(baseline.alphas["28"], changed_mapping.alphas["28"])
    assert receipt.full_source_hash == receipt.full_round_trip_hash
    assert receipt.copy_through_element_count == receipt.mapped_element_count
    with pytest.raises(ValueError, match="frozen identity"):
        initialize_rvir_v4_native_pre_state(
            changed, TOPOLOGY, expected_identity=IDENTITY
        )


def test_wrong_start_node_and_repeated_topology_fail_closed() -> None:
    snapshot = _pre_snapshot()
    wrong_start = list(TOPOLOGY)
    wrong_start[0] = replace(wrong_start[0], provider_start_node="/wrong")
    with pytest.raises(ValueError, match="mutable path ownership"):
        initialize_rvir_v4_native_pre_state(snapshot, tuple(wrong_start))

    repeated = list(TOPOLOGY)
    repeated[1] = replace(
        repeated[1], native_preactivation=repeated[0].native_preactivation
    )
    with pytest.raises(ValueError, match="topology keys repeat"):
        initialize_rvir_v4_native_pre_state(snapshot, tuple(repeated))


def test_mapping_binds_to_independent_native_scope_without_provider_state() -> None:
    mapping = initialize_rvir_v4_native_pre_state(
        _pre_snapshot(), TOPOLOGY, expected_identity=IDENTITY
    )
    placeholder = "0" * 64
    scope = NativeAlphaBetaStateScope(
        primal_graph_hash=placeholder,
        input_region_hash=placeholder,
        objective_hash=placeholder,
        intermediate_bounds_hash=placeholder,
        split_state_hash=relu_split_state_hash(mapping.splits),
        optimizer_policy_hash=placeholder,
    )

    state = mapping.to_native_state(scope)

    assert state.splits.keys() == mapping.splits.keys()
    assert state.alphas.keys() == mapping.alphas.keys()
    assert state.betas.keys() == mapping.betas.keys()


def test_mapping_moves_live_tensors_without_changing_captured_identity() -> None:
    mapping = initialize_rvir_v4_native_pre_state(
        _pre_snapshot(), TOPOLOGY, expected_identity=IDENTITY
    )

    moved = mapping.to(device=torch.device("cpu"), dtype=torch.float64)

    assert moved.identity == mapping.identity
    assert moved.round_trip_receipts == mapping.round_trip_receipts
    assert all(value.dtype == torch.float64 for value in moved.alphas.values())
    assert all(value.dtype == torch.float64 for value in moved.betas.values())
    assert all(value.dtype == torch.int8 for value in moved.splits.values())
    assert all(
        interval.lower.dtype == torch.float64 and interval.upper.dtype == torch.float64
        for interval in moved.relu_pre.values()
    )

    with pytest.raises(ValueError, match="target dtype"):
        mapping.to(device=torch.device("cpu"), dtype=torch.int64)
