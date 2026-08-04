"""Contract and tamper tests for the frozen NRIR-19 artifact."""

# pylint: disable=missing-function-docstring,duplicate-code

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.run_native_intermediate_refinement_artifact import (
    EVIDENCE_FILE,
    MANIFEST_FILE,
    canonical_hash,
    file_sha256,
    validate_evidence_structure,
)

ARTIFACT_DIR = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "native-intermediate-refinement"
    / "vnncomp21-three-topology-cpu-v1"
)


def _load(name: str) -> dict[str, object]:
    value = json.loads((ARTIFACT_DIR / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_refinement_artifact_has_exact_coverage_and_outcomes() -> None:
    manifest = _load(MANIFEST_FILE)
    evidence = _load(EVIDENCE_FILE)
    validate_evidence_structure(evidence)

    assert manifest["evidence_hash"] == canonical_hash(evidence)
    assert manifest["files"] == {
        str(path.relative_to(ARTIFACT_DIR)): file_sha256(path)
        for path in sorted(ARTIFACT_DIR.rglob("*"))
        if path.is_file() and path.name != MANIFEST_FILE
    }
    comparisons = {item["workload_id"]: item for item in evidence["comparisons"]}
    assert comparisons["mnistfc:000"]["closed_clause_indices"] == [3, 7]
    assert comparisons["mnistfc:000"]["refined_unresolved_clause_indices"] == [8]
    assert comparisons["cifar10_resnet:000"]["closed_clause_indices"] == []
    assert comparisons["cifar10_resnet:000"]["refined_status"] == "unknown"
    assert comparisons["oval21:000"]["closed_clause_indices"] == [8]
    assert comparisons["oval21:000"]["refined_status"] == "verified"
    assert all(item["performance_claimed"] is False for item in comparisons.values())


def test_refinement_artifact_binds_native_provenance_and_schedule_trace() -> None:
    evidence = _load(EVIDENCE_FILE)
    records = evidence["records"]
    assert isinstance(records, list)
    for record in records:
        result = record["result"]
        if record["mode"] == "baseline":
            assert result["intermediate_bound_source"] == "local_forward"
            assert result["refinement"] is None
            continue
        assert result["intermediate_bound_source"] == "native_refined"
        refinement = result["refinement"]
        hashes = refinement["hashes"]
        trace = refinement["execution_trace"]
        assert trace["plan_hash"] == hashes["refinement_plan_hash"]
        assert trace["task_module_hash"] == hashes["refinement_task_module_hash"]
        assert trace["schedule_hash"] == hashes["refinement_schedule_hash"]
        assert len(trace["action_traces"]) == len(refinement["schedule"]["actions"])
        assert trace["pass_traces"][0]["tightened_neuron_count"] > 0


def test_refinement_artifact_rejects_ir_and_trace_tamper() -> None:
    evidence = _load(EVIDENCE_FILE)
    changed_plan = deepcopy(evidence)
    refined = next(
        record
        for record in changed_plan["records"]
        if record["mode"] == "native_refined"
    )
    refined["result"]["refinement"]["plan"]["policy"]["max_neurons_per_relu"] += 1
    with pytest.raises(ValueError, match="IR/trace linkage"):
        validate_evidence_structure(changed_plan)

    changed_trace = deepcopy(evidence)
    refined = next(
        record
        for record in changed_trace["records"]
        if record["mode"] == "native_refined"
    )
    refined["result"]["refinement"]["execution_trace"]["action_traces"][0][
        "task_id"
    ] = "tampered.task"
    with pytest.raises(ValueError, match="IR/trace linkage"):
        validate_evidence_structure(changed_trace)


def test_refinement_artifact_log_digests_are_exact() -> None:
    evidence = _load(EVIDENCE_FILE)
    for record in evidence["records"]:
        log_path = ARTIFACT_DIR / record["log_path"]
        assert log_path.is_file()
        assert file_sha256(log_path) == record["log_sha256"]
        assert record["result"]["performance_claimed"] is False
