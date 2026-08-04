"""Frozen NRIR-13 sound property verdict artifact contracts."""

# pylint: disable=missing-function-docstring,duplicate-code

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from boundflow.runtime.abcrown_adapter import file_sha256
from scripts.run_native_property_verdict_artifact import (
    ARTIFACT_FILE,
    ARTIFACT_SCHEMA_VERSION,
    canonical_hash,
    validate_native_property_verdict_evidence,
)

ARTIFACT_DIR = (
    Path("artifacts/native-property-verdict") / "vnncomp21-resnet2b-prop0-cpu-v1"
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


def test_frozen_property_verdict_artifact_digest_and_contract() -> None:
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
    assert manifest["evidence_hash"] == canonical_hash(evidence)
    validate_native_property_verdict_evidence(evidence)


@pytest.mark.parametrize(
    "gate",
    [
        "verified_requires_closed_sound_prune_leaves",
        "unsafe_requires_reexecuted_concrete_witness",
        "open_budget_and_depth_never_inflate_to_verified",
        "fixed_resnet_center_reexecutes_primal_task_ir_without_counterexample",
        "fixed_resnet_frontier_remains_explicit_unknown",
        "fixed_source_and_property_are_digest_bound",
        "all_verdicts_are_correctness_only",
    ],
)
def test_artifact_rejects_failed_gate(gate: str) -> None:
    evidence = deepcopy(_evidence())
    gates = evidence["gates"]
    assert isinstance(gates, dict)
    gates[gate] = False
    with pytest.raises(ValueError, match="gates"):
        validate_native_property_verdict_evidence(evidence)


def test_artifact_rejects_rehashed_verdict_inflation() -> None:
    evidence = deepcopy(_evidence())
    fixed = evidence["fixed_resnet"]
    assert isinstance(fixed, dict)
    verdict = fixed["verdict_trace"]
    assert isinstance(verdict, dict)
    verdict["status"] = "verified"
    verdict["reason"] = "all_leaves_soundly_pruned"
    verdict["unresolved_leaf_node_ids"] = []
    fixed["verdict_trace_hash"] = canonical_hash(verdict)
    with pytest.raises(ValueError, match="case trace|unknown boundary"):
        validate_native_property_verdict_evidence(evidence)


def test_artifact_rejects_rehashed_witness_tampering() -> None:
    evidence = deepcopy(_evidence())
    unsafe = evidence["toy_unsafe"]
    assert isinstance(unsafe, dict)
    verdict = unsafe["verdict_trace"]
    assert isinstance(verdict, dict)
    witness = verdict["counterexample"]
    assert isinstance(witness, dict)
    witness["objective_value"] = 0.1
    witness["objective_margin"] = -0.4
    unsafe["verdict_trace_hash"] = canonical_hash(verdict)
    with pytest.raises(ValueError, match="unsafe witness"):
        validate_native_property_verdict_evidence(evidence)


def test_artifact_rejects_claim_and_performance_inflation() -> None:
    evidence = deepcopy(_evidence())
    evidence["performance_claimed"] = True
    evidence["property_status"] = "verified"
    with pytest.raises(ValueError, match="header"):
        validate_native_property_verdict_evidence(evidence)
