"""Frozen NRIR-16 prepared production fast-path artifact contracts."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from boundflow.runtime.abcrown_adapter import file_sha256
from scripts.run_prepared_production_fast_path_artifact import (
    ARTIFACT_FILE,
    ARTIFACT_SCHEMA_VERSION,
    canonical_hash,
    validate_evidence,
)

ARTIFACT_DIR = (
    Path("artifacts/prepared-production-fast-path") / "vnncomp21-resnet2b-prop0-cpu-v1"
)


def _load(name: str) -> dict[str, object]:
    value = json.loads((ARTIFACT_DIR / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _evidence() -> dict[str, object]:
    evidence = _load(ARTIFACT_FILE)["evidence"]
    assert isinstance(evidence, dict)
    return evidence


def test_frozen_prepared_fast_path_artifact_digest_and_contract() -> None:
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
    validate_evidence(evidence)


@pytest.mark.parametrize(
    "gate",
    [
        "source_model_property_and_intermediates_are_digest_bound",
        "prepared_and_audit_clause_statuses_match",
        "prepared_and_audit_bounds_are_allclose",
        "production_omits_audit_hash_chain_and_selected_native_reexecution",
        "three_group_warm_diagnostic_reduction_exceeds_ten_x",
        "cold_preparation_and_first_execution_are_exposed",
    ],
)
def test_prepared_fast_path_rejects_failed_gate(gate: str) -> None:
    evidence = deepcopy(_evidence())
    gates = evidence["gates"]
    assert isinstance(gates, dict)
    gates[gate] = False
    with pytest.raises(ValueError, match="gates"):
        validate_evidence(evidence)


def test_prepared_fast_path_rejects_semantic_inflation() -> None:
    evidence = deepcopy(_evidence())
    semantic = evidence["semantic"]
    assert isinstance(semantic, dict)
    production = semantic["prepared_production"]
    assert isinstance(production, dict)
    trace = production["trace"]
    assert isinstance(trace, dict)
    trace["status"] = "verified"
    trace["reason"] = "all_root_clauses_verified"
    trace["unresolved_clause_indices"] = []
    production["trace_hash"] = canonical_hash(trace)
    with pytest.raises(ValueError, match="semantic boundary"):
        validate_evidence(evidence)


def test_prepared_fast_path_rejects_timing_and_claim_inflation() -> None:
    evidence = deepcopy(_evidence())
    timing = evidence["timing"]
    assert isinstance(timing, dict)
    samples = timing["samples"]
    assert isinstance(samples, list) and isinstance(samples[0], dict)
    samples[0]["elapsed_ns"] = 0
    with pytest.raises(ValueError, match="timing sample"):
        validate_evidence(evidence)

    evidence = deepcopy(_evidence())
    evidence["performance_claimed"] = True
    evidence["property_status"] = "verified"
    with pytest.raises(ValueError, match="header"):
        validate_evidence(evidence)
