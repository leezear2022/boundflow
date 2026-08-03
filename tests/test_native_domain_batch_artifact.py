"""Frozen NRIR-8 domain-batch artifact and fail-closed contract tests."""

# pylint: disable=missing-function-docstring,implicit-str-concat

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from boundflow.runtime.abcrown_adapter import file_sha256
from scripts.run_native_real_network_domain_batch_artifact import (
    DOMAIN_BATCH_ARTIFACT_SCHEMA_VERSION,
    validate_domain_batch_evidence,
)

ARTIFACT_DIR = Path(
    "artifacts/native-real-network-domain-batch/" "vnncomp21-resnet2b-prop0-cpu-v1"
)


def _load(name: str) -> dict[str, object]:
    value = json.loads((ARTIFACT_DIR / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_frozen_domain_batch_artifact_digest_and_contract() -> None:
    manifest = _load("manifest.json")
    evidence = _load("domain_batch.json")
    assert manifest["schema_version"] == DOMAIN_BATCH_ARTIFACT_SCHEMA_VERSION
    assert manifest["status"] == "ok"
    assert manifest["performance_claimed"] is False
    assert manifest["files"] == {
        "domain_batch.json": file_sha256(ARTIFACT_DIR / "domain_batch.json")
    }
    validate_domain_batch_evidence(evidence)


@pytest.mark.parametrize(
    "gate",
    [
        "eight_distinct_leaf_boxes_have_exact_tree_lineage",
        "every_leaf_recomputes_distinct_exact_state",
        "parent_state_is_warm_start_only_and_never_exact",
        "source_plan_selects_full_or_packed_domain_candidate",
        "packed_two_children_replace_eight_serial_children",
        "packed_full_and_serial_restore_identical_results",
        "all_paths_preserve_one_representation_and_storage_policy",
        "mechanism_counts_do_not_claim_performance_or_full_bab",
    ],
)
def test_domain_batch_artifact_rejects_failed_gate(gate: str) -> None:
    evidence = deepcopy(_load("domain_batch.json"))
    gates = evidence["gates"]
    assert isinstance(gates, dict)
    gates[gate] = False
    with pytest.raises(ValueError, match="gates"):
        validate_domain_batch_evidence(evidence)


def test_domain_batch_artifact_rejects_parent_state_promotion() -> None:
    evidence = deepcopy(_load("domain_batch.json"))
    tree = evidence["domain_tree"]
    assert isinstance(tree, dict)
    states = tree["query_states"]
    assert isinstance(states, list)
    first = states[0]
    assert isinstance(first, dict)
    first["parent_state_consumed_as_exact"] = True
    with pytest.raises(ValueError, match="lineage/state"):
        validate_domain_batch_evidence(evidence)


def test_domain_batch_artifact_rejects_rehashed_domain_range_tamper() -> None:
    evidence = deepcopy(_load("domain_batch.json"))
    packed = evidence["packed"]
    assert isinstance(packed, dict)
    binding = packed["binding_trace"]
    assert isinstance(binding, dict)
    slices = binding["slices"]
    assert isinstance(slices, list)
    first = slices[0]
    second = slices[1]
    assert isinstance(first, dict)
    assert isinstance(second, dict)
    first["stop_index"] = 5
    second["start_index"] = 5
    with pytest.raises(ValueError, match="Schedule ranges"):
        validate_domain_batch_evidence(evidence)


def test_domain_batch_artifact_rejects_result_lineage_tamper() -> None:
    evidence = deepcopy(_load("domain_batch.json"))
    serial = evidence["serial"]
    assert isinstance(serial, dict)
    results = serial["results"]
    assert isinstance(results, list)
    second = results[1]
    assert isinstance(second, dict)
    second["query_id"] = "duplicate-domain"
    with pytest.raises(ValueError, match="restored result lineage"):
        validate_domain_batch_evidence(evidence)


def test_domain_batch_artifact_rejects_semantic_and_claim_inflation() -> None:
    evidence = deepcopy(_load("domain_batch.json"))
    semantics = evidence["semantics"]
    assert isinstance(semantics, dict)
    comparison = semantics["packed_vs_serial_lower"]
    assert isinstance(comparison, dict)
    comparison["allclose"] = False
    with pytest.raises(ValueError, match="semantic comparison"):
        validate_domain_batch_evidence(evidence)

    evidence = deepcopy(_load("domain_batch.json"))
    evidence["performance_claimed"] = True
    evidence["claim_boundary"] = "domain batching speedup"
    with pytest.raises(ValueError, match="contract"):
        validate_domain_batch_evidence(evidence)
