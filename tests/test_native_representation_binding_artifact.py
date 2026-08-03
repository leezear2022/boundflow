"""Frozen NRIR-4 artifact validation and tamper rejection."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from boundflow.runtime.abcrown_adapter import file_sha256
from scripts.run_native_real_network_representation_binding_artifact import (
    REPRESENTATION_ARTIFACT_SCHEMA_VERSION,
    _canonical_json,
    validate_representation_binding_evidence,
)

ARTIFACT_DIR = Path(
    "artifacts/native-real-network-representation-binding/"
    "vnncomp21-resnet2b-prop0-cpu-v1"
)


def _load(name: str) -> dict[str, object]:
    value = json.loads((ARTIFACT_DIR / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _refresh_binding_hash(evidence: dict[str, object]) -> None:
    structured = evidence["structured_affine_policy"]
    assert isinstance(structured, dict)
    binding = structured["binding_trace"]
    hashes = structured["ir_hashes"]
    assert isinstance(binding, dict) and isinstance(hashes, dict)
    hashes["representation_binding_hash"] = hashlib.sha256(
        _canonical_json(binding).encode("utf-8")
    ).hexdigest()


def test_frozen_representation_artifact_digest_and_contract() -> None:
    manifest = _load("manifest.json")
    evidence = _load("representation.json")
    assert manifest["schema_version"] == REPRESENTATION_ARTIFACT_SCHEMA_VERSION
    assert manifest["status"] == "ok"
    assert manifest["performance_claimed"] is False
    files = manifest["files"]
    assert isinstance(files, dict)
    assert files == {
        "representation.json": file_sha256(ARTIFACT_DIR / "representation.json")
    }
    validate_representation_binding_evidence(evidence)


@pytest.mark.parametrize(
    "gate",
    [
        "budget_switches_global_policy",
        "transitions_bind_plan_schedule_bound_one_to_one",
        "structured_storage_is_dense_equivalent_not_compressed",
        "both_policies_match_semantics",
    ],
)
def test_artifact_rejects_failed_gate(gate: str) -> None:
    evidence = deepcopy(_load("representation.json"))
    gates = evidence["gates"]
    assert isinstance(gates, dict)
    gates[gate] = False
    with pytest.raises(ValueError, match="gates"):
        validate_representation_binding_evidence(evidence)


def test_artifact_rejects_transition_relink_even_with_new_trace_digest() -> None:
    evidence = deepcopy(_load("representation.json"))
    structured = evidence["structured_affine_policy"]
    assert isinstance(structured, dict)
    binding = structured["binding_trace"]
    assert isinstance(binding, dict)
    events = binding["events"]
    selected = binding["selected_transition_candidate_ids"]
    assert isinstance(events, list) and isinstance(selected, list)
    first = events[0]
    second = events[1]
    assert isinstance(first, dict) and isinstance(second, dict)
    second["transition_candidate_id"] = first["transition_candidate_id"]
    selected[1] = selected[0]
    _refresh_binding_hash(evidence)
    with pytest.raises(ValueError, match="binding trace linkage"):
        validate_representation_binding_evidence(evidence)


def test_artifact_rejects_mixed_policy_even_with_new_trace_digest() -> None:
    evidence = deepcopy(_load("representation.json"))
    structured = evidence["structured_affine_policy"]
    assert isinstance(structured, dict)
    binding = structured["binding_trace"]
    assert isinstance(binding, dict)
    selected = binding["selected_representation_candidate_ids"]
    assert isinstance(selected, list)
    selected[0] = str(selected[0]).replace("native-structured-affine-v1", "dense")
    _refresh_binding_hash(evidence)
    with pytest.raises(ValueError, match="binding trace linkage"):
        validate_representation_binding_evidence(evidence)


def test_artifact_rejects_materialize_direction_with_new_trace_digest() -> None:
    evidence = deepcopy(_load("representation.json"))
    structured = evidence["structured_affine_policy"]
    assert isinstance(structured, dict)
    binding = structured["binding_trace"]
    assert isinstance(binding, dict)
    events = binding["events"]
    assert isinstance(events, list)
    event = next(
        item
        for item in events
        if isinstance(item, dict) and item.get("transition_kind") == "materialize"
    )
    assert isinstance(event, dict)
    event["target_representation"] = "structured"
    _refresh_binding_hash(evidence)
    with pytest.raises(ValueError, match="binding event semantics"):
        validate_representation_binding_evidence(evidence)
