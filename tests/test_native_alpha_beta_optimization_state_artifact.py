"""Frozen NRIR-10 alpha/beta state artifact contracts."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from boundflow.runtime.abcrown_adapter import file_sha256
from scripts.run_native_alpha_beta_optimization_state_artifact import (
    ARTIFACT_FILE,
    ARTIFACT_SCHEMA_VERSION,
    hashlib_sha256,
    validate_native_alpha_beta_evidence,
)

ARTIFACT_DIR = Path(
    "artifacts/native-alpha-beta-optimization-state/" "vnncomp21-resnet2b-prop0-cpu-v1"
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


def test_frozen_alpha_beta_state_artifact_digest_and_contract() -> None:
    manifest = _load("manifest.json")
    artifact = _load(ARTIFACT_FILE)
    assert manifest["schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert manifest["status"] == "ok"
    assert manifest["performance_claimed"] is False
    assert manifest["files"] == {
        ARTIFACT_FILE: file_sha256(ARTIFACT_DIR / ARTIFACT_FILE)
    }
    evidence = artifact["evidence"]
    assert isinstance(evidence, dict)
    assert manifest["evidence_hash"] == hashlib_sha256(evidence)
    validate_native_alpha_beta_evidence(evidence)


@pytest.mark.parametrize(
    "gate",
    [
        "fixed_resnet_source_is_digest_bound",
        "six_relu_split_alpha_beta_inputs_are_first_class",
        "warm_start_is_monotonic_refinement_initialization_only",
        "native_bounds_match_legacy_alpha_beta_oracle",
        "nonzero_beta_changes_native_lower_dual_result",
        "plan_task_schedule_consume_optimized_state",
        "all_compiler_hashes_change_with_beta_payload",
        "claims_remain_correctness_only",
    ],
)
def test_artifact_rejects_failed_gate(gate: str) -> None:
    evidence = deepcopy(_evidence())
    gates = evidence["gates"]
    assert isinstance(gates, dict)
    gates[gate] = False
    with pytest.raises(ValueError, match="gates"):
        validate_native_alpha_beta_evidence(evidence)


def test_artifact_rejects_warm_start_and_beta_tampering() -> None:
    evidence = deepcopy(_evidence())
    warm = evidence["child_warm_start"]
    assert isinstance(warm, dict)
    warm["exact_state_reuse_allowed"] = True
    with pytest.raises(ValueError, match="warm-start"):
        validate_native_alpha_beta_evidence(evidence)

    evidence = deepcopy(_evidence())
    execution = evidence["execution"]
    assert isinstance(execution, dict)
    execution["beta_lower_improvement"] = 0.0
    with pytest.raises(ValueError, match="execution/beta"):
        validate_native_alpha_beta_evidence(evidence)


def test_artifact_rejects_ir_hash_and_claim_inflation() -> None:
    evidence = deepcopy(_evidence())
    hashes = evidence["ir_hashes"]
    zero_hashes = evidence["zero_beta_ir_hashes"]
    assert isinstance(hashes, dict) and isinstance(zero_hashes, dict)
    first = next(iter(hashes))
    zero_hashes[first] = hashes[first]
    with pytest.raises(ValueError, match="compiler hash"):
        validate_native_alpha_beta_evidence(evidence)

    evidence = deepcopy(_evidence())
    evidence["performance_claimed"] = True
    evidence["property_status"] = "proven"
    with pytest.raises(ValueError, match="header"):
        validate_native_alpha_beta_evidence(evidence)
