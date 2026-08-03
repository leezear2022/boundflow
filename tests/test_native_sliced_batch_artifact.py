"""Frozen NRIR-5 spec-sliced artifact validation and tamper rejection."""

# pylint: disable=missing-function-docstring,implicit-str-concat

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from boundflow.runtime.abcrown_adapter import file_sha256
from scripts.run_native_real_network_sliced_batch_artifact import (
    SLICED_BATCH_ARTIFACT_SCHEMA_VERSION,
    _canonical_json,
    validate_sliced_batch_evidence,
)

ARTIFACT_DIR = Path(
    "artifacts/native-real-network-sliced-batch/" "vnncomp21-resnet2b-prop0-cpu-v1"
)


def _load(name: str) -> dict[str, object]:
    value = json.loads((ARTIFACT_DIR / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sliced_policy(evidence: dict[str, object]) -> dict[str, object]:
    value = evidence["sliced_policy"]
    assert isinstance(value, dict)
    return value


def _refresh_binding_hash(evidence: dict[str, object]) -> None:
    sliced = _sliced_policy(evidence)
    binding = sliced["binding_trace"]
    hashes = sliced["source_ir_hashes"]
    trace = sliced["execution_trace"]
    assert isinstance(binding, dict)
    assert isinstance(hashes, dict)
    assert isinstance(trace, dict)
    digest = hashlib.sha256(_canonical_json(binding).encode("utf-8")).hexdigest()
    hashes["spec_batch_binding_hash"] = digest
    trace["binding_hash"] = digest


def test_frozen_sliced_batch_artifact_digest_and_contract() -> None:
    manifest = _load("manifest.json")
    evidence = _load("sliced_batch.json")
    assert manifest["schema_version"] == SLICED_BATCH_ARTIFACT_SCHEMA_VERSION
    assert manifest["status"] == "ok"
    assert manifest["performance_claimed"] is False
    assert manifest["files"] == {
        "sliced_batch.json": file_sha256(ARTIFACT_DIR / "sliced_batch.json")
    }
    validate_sliced_batch_evidence(evidence)


@pytest.mark.parametrize(
    "gate",
    [
        "source_schedule_owns_exact_spec_ranges",
        "all_child_bound_ops_are_tasked_and_launched",
        "full_and_sliced_match_external_semantics",
        "controller_storage_is_not_claimed_as_slice_memory",
    ],
)
def test_sliced_batch_artifact_rejects_failed_gate(gate: str) -> None:
    evidence = deepcopy(_load("sliced_batch.json"))
    gates = evidence["gates"]
    assert isinstance(gates, dict)
    gates[gate] = False
    with pytest.raises(ValueError, match="gates"):
        validate_sliced_batch_evidence(evidence)


def test_sliced_batch_artifact_rejects_rehashed_range_and_query_tamper() -> None:
    evidence = deepcopy(_load("sliced_batch.json"))
    sliced = _sliced_policy(evidence)
    binding = sliced["binding_trace"]
    trace = sliced["execution_trace"]
    assert isinstance(binding, dict)
    assert isinstance(trace, dict)
    slices = binding["slices"]
    child_query_ids = trace["child_query_ids"]
    assert isinstance(slices, list)
    assert isinstance(child_query_ids, list)
    first = slices[0]
    second = slices[1]
    assert isinstance(first, dict)
    assert isinstance(second, dict)
    first["stop_index"] = 2
    first["slice_id"] = "spec-slice:0000:0002"
    first["child_query_id"] = first_query = (
        "vnncomp21-resnet2b-prop0-native-ir5-spec-batch:spec:0000:0002"
    )
    second["start_index"] = 2
    second["slice_id"] = "spec-slice:0002:0006"
    second["child_query_id"] = second_query = (
        "vnncomp21-resnet2b-prop0-native-ir5-spec-batch:spec:0002:0006"
    )
    child_query_ids[:2] = [first_query, second_query]
    _refresh_binding_hash(evidence)
    with pytest.raises(ValueError, match="binding slice ranges/query ownership"):
        validate_sliced_batch_evidence(evidence)


def test_sliced_batch_artifact_rejects_execution_trace_relink() -> None:
    evidence = deepcopy(_load("sliced_batch.json"))
    sliced = _sliced_policy(evidence)
    trace = sliced["execution_trace"]
    assert isinstance(trace, dict)
    query_ids = trace["child_query_ids"]
    assert isinstance(query_ids, list)
    query_ids[1] = query_ids[0]
    with pytest.raises(ValueError, match="execution trace linkage"):
        validate_sliced_batch_evidence(evidence)


def test_sliced_batch_artifact_rejects_claim_inflation() -> None:
    evidence = deepcopy(_load("sliced_batch.json"))
    evidence["claim_boundary"] = "real-network spec-axis speedup"
    evidence["performance_claimed"] = True
    with pytest.raises(ValueError, match="contract"):
        validate_sliced_batch_evidence(evidence)
