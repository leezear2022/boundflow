"""Frozen NRIR-9 first-class ReLU-split queue artifact contracts."""

# pylint: disable=missing-function-docstring,implicit-str-concat

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from boundflow.runtime.abcrown_adapter import file_sha256
from scripts.run_native_real_network_relu_split_bab_artifact import (
    RELU_SPLIT_BAB_ARTIFACT_SCHEMA_VERSION,
    hashlib_sha256,
    validate_relu_split_bab_evidence,
)

ARTIFACT_DIR = Path(
    "artifacts/native-real-network-relu-split-bab/" "vnncomp21-resnet2b-prop0-cpu-v1"
)


def _load(name: str) -> dict[str, object]:
    value = json.loads((ARTIFACT_DIR / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _evidence() -> dict[str, object]:
    artifact = _load("relu_split_bab.json")
    evidence = artifact["evidence"]
    assert isinstance(evidence, dict)
    return evidence


def _rehash_trace(evidence: dict[str, object], mode: str) -> None:
    section = evidence[mode]
    assert isinstance(section, dict)
    trace = section["trace"]
    assert isinstance(trace, dict)
    section["trace_hash"] = hashlib_sha256(trace)


def test_frozen_relu_split_bab_artifact_digest_and_contract() -> None:
    manifest = _load("manifest.json")
    artifact = _load("relu_split_bab.json")
    assert manifest["schema_version"] == RELU_SPLIT_BAB_ARTIFACT_SCHEMA_VERSION
    assert manifest["status"] == "ok"
    assert manifest["performance_claimed"] is False
    assert manifest["files"] == {
        "relu_split_bab.json": file_sha256(ARTIFACT_DIR / "relu_split_bab.json")
    }
    evidence = artifact["evidence"]
    assert isinstance(evidence, dict)
    assert manifest["evidence_hash"] == hashlib_sha256(evidence)
    validate_relu_split_bab_evidence(evidence)


@pytest.mark.parametrize(
    "gate",
    [
        "fixed_resnet_and_property_source_are_digest_bound",
        "all_six_relu_splits_are_first_class_bound_inputs",
        "split_state_reaches_plan_task_and_schedule_ir",
        "every_child_recomputes_exact_state_without_parent_reuse",
        "best_first_queue_forms_three_generations_and_four_frontier_nodes",
        "packed_three_stacks_replace_seven_serial_stacks",
        "packed_and_serial_bounds_and_queue_semantics_match",
        "bounded_run_reports_unknown_budget_exhausted_without_performance_claim",
    ],
)
def test_relu_split_bab_artifact_rejects_failed_gate(gate: str) -> None:
    evidence = deepcopy(_evidence())
    gates = evidence["gates"]
    assert isinstance(gates, dict)
    gates[gate] = False
    with pytest.raises(ValueError, match="gates"):
        validate_relu_split_bab_evidence(evidence)


def test_artifact_rejects_rehashed_parent_and_branch_tampering() -> None:
    evidence = deepcopy(_evidence())
    packed = evidence["packed"]
    assert isinstance(packed, dict)
    trace = packed["trace"]
    assert isinstance(trace, dict)
    evaluations = trace["evaluations"]
    assert isinstance(evaluations, list)
    child = evaluations[1]
    assert isinstance(child, dict)
    node = child["node"]
    assert isinstance(node, dict)
    node["parent_node_id"] = "missing-parent"
    _rehash_trace(evidence, "packed")
    with pytest.raises(ValueError, match="parent"):
        validate_relu_split_bab_evidence(evidence)

    evidence = deepcopy(_evidence())
    packed = evidence["packed"]
    assert isinstance(packed, dict)
    trace = packed["trace"]
    assert isinstance(trace, dict)
    decisions = trace["decisions"]
    assert isinstance(decisions, list)
    expansion = decisions[0]
    assert isinstance(expansion, dict)
    children = expansion["child_node_ids"]
    assert isinstance(children, list)
    expansion["child_node_ids"] = list(reversed(children))
    _rehash_trace(evidence, "packed")
    with pytest.raises(ValueError, match="branch"):
        validate_relu_split_bab_evidence(evidence)


def test_artifact_rejects_rehashed_ir_stack_and_bound_tampering() -> None:
    evidence = deepcopy(_evidence())
    packed = evidence["packed"]
    assert isinstance(packed, dict)
    trace = packed["trace"]
    assert isinstance(trace, dict)
    stacks = trace["native_stacks"]
    assert isinstance(stacks, list)
    stack = stacks[0]
    assert isinstance(stack, dict)
    stack["bound_split_input_count"] = 5
    _rehash_trace(evidence, "packed")
    with pytest.raises(ValueError, match="stack"):
        validate_relu_split_bab_evidence(evidence)

    evidence = deepcopy(_evidence())
    packed = evidence["packed"]
    assert isinstance(packed, dict)
    trace = packed["trace"]
    assert isinstance(trace, dict)
    evaluations = trace["evaluations"]
    assert isinstance(evaluations, list)
    evaluation = evaluations[1]
    assert isinstance(evaluation, dict)
    evaluation["lower"] = float(evaluation["lower"]) - 1.0
    _rehash_trace(evidence, "packed")
    with pytest.raises(ValueError, match="comparison"):
        validate_relu_split_bab_evidence(evidence)


def test_artifact_rejects_claim_inflation() -> None:
    evidence = deepcopy(_evidence())
    evidence["performance_claimed"] = True
    evidence["property_status"] = "proven"
    evidence["claim_boundary"] = "full BaB speedup"
    with pytest.raises(ValueError, match="header"):
        validate_relu_split_bab_evidence(evidence)
