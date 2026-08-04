"""Contract and tamper tests for the frozen NRIR-22 artifact."""

# pylint: disable=protected-access,missing-function-docstring

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.run_ancestral_constraint_refinement_artifact import (
    ARTIFACT_SCHEMA_VERSION,
    _validate_mode_result,
    validate_evidence_structure,
)
from scripts.run_multiworkload_competitor_e2e_artifact import (
    canonical_hash,
    file_sha256,
)

ARTIFACT_DIR = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "ancestral-constraint-refinement"
    / "vnncomp21-resnet2b-two-clause-cpu-v1"
)


def _load(name: str):
    return json.loads((ARTIFACT_DIR / name).read_text(encoding="utf-8"))


def test_frozen_ancestral_artifact_is_digest_bound_and_reduced() -> None:
    manifest = _load("manifest.json")
    evidence = _load("evidence.json")
    actual_files = {
        str(path.relative_to(ARTIFACT_DIR)): file_sha256(path)
        for path in sorted(ARTIFACT_DIR.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }

    assert manifest["schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert manifest["status"] == evidence["status"] == "validated_reduced"
    assert manifest["performance_claimed"] is False
    assert manifest["files"] == actual_files
    assert manifest["evidence_hash"] == canonical_hash(evidence)
    validate_evidence_structure(evidence)

    assert [
        item["ancestral_vs_independent_delta"] for item in evidence["comparisons"]
    ] == [73.61517333984375, 75.0220947265625]
    assert [
        item["ancestral_vs_root_global_delta"] for item in evidence["comparisons"]
    ] == [72.7672119140625, 74.08544921875]
    assert all(
        item["strictly_improves_independent"] is True
        and item["not_weaker_than_root_global"] is True
        and item["root_bound_same"] is True
        for item in evidence["comparisons"]
    )


def test_ancestral_artifact_binds_parent_execution_and_task_input() -> None:
    evidence = _load("evidence.json")
    carry_results = [
        record["result"]
        for record in evidence["records"]
        if record["mode"] == "ancestral_carry"
    ]

    assert len(carry_results) == 2
    for result in carry_results:
        queue = result["queue_trace"]
        assert queue["per_child_refinement_strategy"] == (
            "ancestral_constraint_carry_v1"
        )
        assert (
            sum(
                "source_parent_node_id" in item
                for item in queue["per_child_refinements"]
            )
            == 6
        )
        _validate_mode_result("ancestral_carry", result)


def test_ancestral_artifact_source_lineage_tampering_fails_closed() -> None:
    evidence = _load("evidence.json")
    result = next(
        record["result"]
        for record in evidence["records"]
        if record["mode"] == "ancestral_carry"
    )
    tampered = deepcopy(result)
    queue = tampered["queue_trace"]
    child_index = next(
        index
        for index, evaluation in enumerate(queue["evaluations"])
        if evaluation["node"]["depth"] == 1
    )
    record = queue["per_child_refinements"][child_index]
    record["source_intermediate_constraints_hash"] = "f" * 64
    queue["evaluations"][child_index]["intermediate_refinement_trace_hash"] = (
        canonical_hash(record)
    )
    tampered["queue_trace_hash"] = canonical_hash(queue)

    with pytest.raises(ValueError, match="ancestral source lineage"):
        _validate_mode_result("ancestral_carry", tampered)


def test_ancestral_artifact_strategy_and_closure_tampering_fail_closed() -> None:
    evidence = _load("evidence.json")
    result = next(
        record["result"]
        for record in evidence["records"]
        if record["mode"] == "ancestral_carry"
    )
    tampered_result = deepcopy(result)
    tampered_result["queue_trace"][
        "per_child_refinement_strategy"
    ] = "independent_exact_split_v1"
    tampered_result["queue_trace_hash"] = canonical_hash(tampered_result["queue_trace"])
    with pytest.raises(ValueError, match="ancestral strategy identity"):
        _validate_mode_result("ancestral_carry", tampered_result)

    tampered_evidence = deepcopy(evidence)
    tampered_evidence["status"] = "validated_no_go"
    with pytest.raises(ValueError, match="closure status"):
        validate_evidence_structure(tampered_evidence)
