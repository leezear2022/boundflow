"""Frozen NRIR-14 complete verifier query artifact contracts."""

# pylint: disable=missing-function-docstring,duplicate-code

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from boundflow.runtime.abcrown_adapter import file_sha256
from boundflow.runtime.native_optimized_relu_split_bab_runtime import (
    NATIVE_REEXECUTION_TRACE_MAX_ABS_DIFF,
)
from scripts.run_complete_verifier_query_artifact import (
    ARTIFACT_FILE,
    ARTIFACT_SCHEMA_VERSION,
    canonical_hash,
    validate_complete_verifier_query_evidence,
)

ARTIFACT_DIR = (
    Path("artifacts/complete-verifier-query") / "vnncomp21-resnet2b-prop0-cpu-v1"
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


def test_frozen_complete_query_artifact_digest_and_contract() -> None:
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
    validate_complete_verifier_query_evidence(evidence)


@pytest.mark.parametrize(
    "gate",
    [
        "multi_clause_conjunction_requires_every_clause_verified",
        "pgd_candidate_is_replayed_before_unsafe_short_circuit",
        "attack_not_found_never_upgrades_unknown_to_verified",
        "cooperative_deadline_exposes_all_pending_clauses",
        "fixed_resnet_executes_nine_real_property_clauses",
        "fixed_resnet_pgd_search_finds_no_false_counterexample",
        "scale_aware_execution_accepts_only_runtime_allclose_traces",
        "fixed_source_and_property_are_digest_bound",
        "all_complete_queries_are_correctness_only",
    ],
)
def test_artifact_rejects_failed_gate(gate: str) -> None:
    evidence = deepcopy(_evidence())
    gates = evidence["gates"]
    assert isinstance(gates, dict)
    gates[gate] = False
    with pytest.raises(ValueError, match="gates"):
        validate_complete_verifier_query_evidence(evidence)


def test_artifact_rejects_rehashed_query_status_inflation() -> None:
    evidence = deepcopy(_evidence())
    fixed = evidence["fixed_resnet"]
    assert isinstance(fixed, dict)
    query = fixed["query_trace"]
    assert isinstance(query, dict)
    query["status"] = "verified"
    query["reason"] = "all_clauses_verified"
    query["unresolved_clause_indices"] = []
    fixed["query_trace_hash"] = canonical_hash(query)
    with pytest.raises(ValueError, match="header/hash|boundary"):
        validate_complete_verifier_query_evidence(evidence)


def test_artifact_rejects_rehashed_clause_pipeline_tampering() -> None:
    evidence = deepcopy(_evidence())
    unsafe = evidence["toy_unsafe"]
    assert isinstance(unsafe, dict)
    clauses = unsafe["clauses"]
    assert isinstance(clauses, list) and isinstance(clauses[-1], dict)
    clause = clauses[-1]
    search = clause["search_trace"]
    assert isinstance(search, dict)
    search["proof_claimed"] = True
    clause["search_trace_hash"] = canonical_hash(search)
    clause_trace = clause["clause_trace"]
    assert isinstance(clause_trace, dict)
    clause_trace["search_trace_hash"] = clause["search_trace_hash"]
    with pytest.raises(ValueError, match="pipeline identity"):
        validate_complete_verifier_query_evidence(evidence)


def test_artifact_rejects_scale_independent_diff_inflation() -> None:
    evidence = deepcopy(_evidence())
    fixed = evidence["fixed_resnet"]
    assert isinstance(fixed, dict)
    diffs = fixed["native_lower_max_abs_diffs"]
    assert isinstance(diffs, list)
    diffs[0] = NATIVE_REEXECUTION_TRACE_MAX_ABS_DIFF * 2.0
    with pytest.raises(ValueError, match="fixed ResNet"):
        validate_complete_verifier_query_evidence(evidence)


def test_artifact_rejects_claim_and_performance_inflation() -> None:
    evidence = deepcopy(_evidence())
    evidence["performance_claimed"] = True
    evidence["property_status"] = "verified"
    with pytest.raises(ValueError, match="header"):
        validate_complete_verifier_query_evidence(evidence)
