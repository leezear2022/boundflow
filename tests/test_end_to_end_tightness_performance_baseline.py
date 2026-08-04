"""Frozen NRIR-15 end-to-end tightness/performance diagnostic contracts."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from boundflow.runtime.abcrown_adapter import file_sha256
from scripts.run_end_to_end_tightness_performance_baseline import (
    ARTIFACT_FILE,
    ARTIFACT_SCHEMA_VERSION,
    canonical_hash,
    validate_evidence,
)

ARTIFACT_DIR = (
    Path("artifacts/end-to-end-tightness-performance")
    / "vnncomp21-resnet2b-prop0-cpu-v1"
)


def _load(name: str) -> dict[str, object]:
    value = json.loads((ARTIFACT_DIR / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _evidence() -> dict[str, object]:
    evidence = _load(ARTIFACT_FILE)["evidence"]
    assert isinstance(evidence, dict)
    return evidence


def test_frozen_baseline_artifact_digest_and_contract() -> None:
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
        "local_reference_remains_nine_of_nine_unknown",
        "external_semantics_restore_six_verified_three_unknown",
        "adaptive_optimizer_never_regresses_frozen_external_initial",
        "external_optimizer_and_selected_native_stacks_agree",
    ],
)
def test_baseline_rejects_failed_gate(gate: str) -> None:
    evidence = deepcopy(_evidence())
    gates = evidence["gates"]
    assert isinstance(gates, dict)
    gates[gate] = False
    with pytest.raises(ValueError, match="gates"):
        validate_evidence(evidence)


def test_baseline_rejects_rehashed_semantic_inflation() -> None:
    evidence = deepcopy(_evidence())
    semantic = evidence["semantic"]
    assert isinstance(semantic, dict)
    external = semantic["external_adaptive"]
    assert isinstance(external, dict)
    query = external["query_trace"]
    assert isinstance(query, dict)
    query["status"] = "verified"
    query["reason"] = "all_clauses_verified"
    query["unresolved_clause_indices"] = []
    external["query_trace_hash"] = canonical_hash(query)
    with pytest.raises(ValueError, match="semantic boundary"):
        validate_evidence(evidence)


def test_baseline_rejects_timing_and_claim_inflation() -> None:
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
