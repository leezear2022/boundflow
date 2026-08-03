"""Frozen NRIR-6 joint-policy artifact and tamper rejection."""

# pylint: disable=missing-function-docstring,implicit-str-concat

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from boundflow.runtime.abcrown_adapter import file_sha256
from scripts.run_native_real_network_joint_policy_artifact import (
    JOINT_POLICY_ARTIFACT_SCHEMA_VERSION,
    _canonical_json,
    validate_joint_policy_evidence,
)

ARTIFACT_DIR = Path(
    "artifacts/native-real-network-joint-policy/" "vnncomp21-resnet2b-prop0-cpu-v1"
)


def _load(name: str) -> dict[str, object]:
    value = json.loads((ARTIFACT_DIR / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _policy(evidence: dict[str, object], name: str) -> dict[str, object]:
    policies = evidence["policies"]
    assert isinstance(policies, dict)
    value = policies[name]
    assert isinstance(value, dict)
    return value


def _refresh_binding_hash(policy: dict[str, object]) -> None:
    binding = policy["binding_trace"]
    hashes = policy["source_ir_hashes"]
    execution = policy["execution_trace"]
    assert isinstance(binding, dict)
    assert isinstance(hashes, dict)
    assert isinstance(execution, dict)
    digest = hashlib.sha256(_canonical_json(binding).encode("utf-8")).hexdigest()
    hashes["joint_policy_binding_hash"] = digest
    execution["binding_hash"] = digest


def test_frozen_joint_policy_artifact_digest_and_contract() -> None:
    manifest = _load("manifest.json")
    evidence = _load("joint_policy.json")
    assert manifest["schema_version"] == JOINT_POLICY_ARTIFACT_SCHEMA_VERSION
    assert manifest["status"] == "ok"
    assert manifest["performance_claimed"] is False
    assert manifest["files"] == {
        "joint_policy.json": file_sha256(ARTIFACT_DIR / "joint_policy.json")
    }
    validate_joint_policy_evidence(evidence)


@pytest.mark.parametrize(
    "gate",
    [
        "budget_and_spec_limit_select_exact_cross_product",
        "children_inherit_source_representation_policy",
        "structured_transitions_remain_execution_owned",
        "four_paths_match_external_semantics",
        "no_joint_performance_or_memory_claim",
    ],
)
def test_joint_policy_artifact_rejects_failed_gate(gate: str) -> None:
    evidence = deepcopy(_load("joint_policy.json"))
    gates = evidence["gates"]
    assert isinstance(gates, dict)
    gates[gate] = False
    with pytest.raises(ValueError, match="gates"):
        validate_joint_policy_evidence(evidence)


def test_joint_policy_artifact_rejects_rehashed_range_tamper() -> None:
    evidence = deepcopy(_load("joint_policy.json"))
    policy = _policy(evidence, "structured_sliced")
    binding = policy["binding_trace"]
    assert isinstance(binding, dict)
    slices = binding["slices"]
    assert isinstance(slices, list)
    first = slices[0]
    assert isinstance(first, dict)
    first["stop_index"] = 2
    _refresh_binding_hash(policy)
    with pytest.raises(ValueError, match="child binding/range"):
        validate_joint_policy_evidence(evidence)


def test_joint_policy_artifact_rejects_policy_relink_with_new_digest() -> None:
    evidence = deepcopy(_load("joint_policy.json"))
    policy = _policy(evidence, "structured_sliced")
    binding = policy["binding_trace"]
    assert isinstance(binding, dict)
    binding["selected_representation_policy_id"] = "native-dense-v1"
    _refresh_binding_hash(policy)
    with pytest.raises(ValueError, match="joint binding linkage"):
        validate_joint_policy_evidence(evidence)


def test_joint_policy_artifact_rejects_execution_query_relink() -> None:
    evidence = deepcopy(_load("joint_policy.json"))
    policy = _policy(evidence, "dense_sliced")
    execution = policy["execution_trace"]
    assert isinstance(execution, dict)
    query_ids = execution["child_query_ids"]
    assert isinstance(query_ids, list)
    query_ids[1] = query_ids[0]
    with pytest.raises(ValueError, match="execution trace linkage"):
        validate_joint_policy_evidence(evidence)


def test_joint_policy_artifact_rejects_claim_inflation() -> None:
    evidence = deepcopy(_load("joint_policy.json"))
    evidence["performance_claimed"] = True
    evidence["claim_boundary"] = "joint Plan speedup"
    with pytest.raises(ValueError, match="contract"):
        validate_joint_policy_evidence(evidence)
