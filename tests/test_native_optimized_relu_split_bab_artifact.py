"""Frozen NRIR-12 optimized ReLU-split BaB artifact contracts."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from boundflow.runtime.abcrown_adapter import file_sha256
from scripts.run_native_optimized_relu_split_bab_artifact import (
    ARTIFACT_FILE,
    ARTIFACT_SCHEMA_VERSION,
    hashlib_sha256,
    validate_native_optimized_bab_evidence,
)

ARTIFACT_DIR = (
    Path("artifacts/native-optimized-relu-split-bab")
    / "vnncomp21-resnet2b-prop0-cpu-v1"
)


def _load(name: str) -> dict[str, object]:
    value = json.loads((ARTIFACT_DIR / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _evidence() -> dict[str, object]:
    artifact = _load(ARTIFACT_FILE)
    evidence = artifact["evidence"]
    assert isinstance(evidence, dict)
    return evidence


def test_frozen_optimized_bab_artifact_digest_and_contract() -> None:
    manifest = _load("manifest.json")
    artifact = _load(ARTIFACT_FILE)
    evidence = artifact["evidence"]
    assert isinstance(evidence, dict)
    assert manifest["schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert manifest["status"] == "ok"
    assert manifest["performance_claimed"] is False
    assert manifest["files"] == {
        ARTIFACT_FILE: file_sha256(ARTIFACT_DIR / ARTIFACT_FILE)
    }
    assert manifest["evidence_hash"] == hashlib_sha256(evidence)
    validate_native_optimized_bab_evidence(evidence)


@pytest.mark.parametrize(
    "gate",
    [
        "fixed_resnet_and_property_source_are_digest_bound",
        "every_node_executes_optimizer_and_selected_native_ir_stacks",
        "every_child_uses_monotonic_parent_initialization_only",
        "active_split_batches_execute_nonzero_beta_gradients",
        "selected_states_reexecute_through_native_compiler",
        "best_first_queue_forms_three_generations_and_four_frontier_nodes",
        "packed_three_stacks_replace_seven_serial_stacks",
        "packed_serial_queue_bounds_and_state_tensors_match",
        "bounded_run_remains_unknown_and_correctness_only",
    ],
)
def test_artifact_rejects_failed_gate(gate: str) -> None:
    evidence = deepcopy(_evidence())
    gates = evidence["gates"]
    assert isinstance(gates, dict)
    gates[gate] = False
    with pytest.raises(ValueError, match="gates"):
        validate_native_optimized_bab_evidence(evidence)


def test_artifact_rejects_rehashed_parent_warm_state_tampering() -> None:
    evidence = deepcopy(_evidence())
    packed = evidence["packed"]
    assert isinstance(packed, dict)
    trace = packed["trace"]
    assert isinstance(trace, dict)
    evaluations = trace["evaluations"]
    assert isinstance(evaluations, list) and isinstance(evaluations[1], dict)
    evaluations[1]["parent_selected_state_hash"] = "f" * 64
    packed["trace_hash"] = hashlib_sha256(trace)
    with pytest.raises(ValueError, match="parent warm-state link"):
        validate_native_optimized_bab_evidence(evidence)


def test_artifact_rejects_rehashed_optimizer_native_stack_tampering() -> None:
    evidence = deepcopy(_evidence())
    packed = evidence["packed"]
    assert isinstance(packed, dict)
    trace = packed["trace"]
    assert isinstance(trace, dict)
    stacks = trace["native_stacks"]
    assert isinstance(stacks, list) and isinstance(stacks[1], dict)
    stacks[1]["optimizer_action_count"] = 7
    packed["trace_hash"] = hashlib_sha256(trace)
    with pytest.raises(ValueError, match="optimizer/native stack"):
        validate_native_optimized_bab_evidence(evidence)

    evidence = deepcopy(_evidence())
    packed = evidence["packed"]
    assert isinstance(packed, dict)
    trace = packed["trace"]
    assert isinstance(trace, dict)
    stacks = trace["native_stacks"]
    assert isinstance(stacks, list) and isinstance(stacks[1], dict)
    stacks[1]["selected_native_lower_max_abs_diff"] = 1.0
    packed["trace_hash"] = hashlib_sha256(trace)
    with pytest.raises(ValueError, match="optimizer/native stack"):
        validate_native_optimized_bab_evidence(evidence)


def test_artifact_rejects_numeric_state_and_claim_inflation() -> None:
    evidence = deepcopy(_evidence())
    comparison = evidence["comparison"]
    assert isinstance(comparison, dict)
    state = comparison["state"]
    assert isinstance(state, dict)
    state["alpha_max_abs_diff"] = 1.0
    with pytest.raises(ValueError, match="numeric comparison"):
        validate_native_optimized_bab_evidence(evidence)

    evidence = deepcopy(_evidence())
    evidence["performance_claimed"] = True
    evidence["property_status"] = "verified"
    with pytest.raises(ValueError, match="header"):
        validate_native_optimized_bab_evidence(evidence)
