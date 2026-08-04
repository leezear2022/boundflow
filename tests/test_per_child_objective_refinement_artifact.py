"""Contract and tamper tests for the frozen NRIR-21 artifact."""

# pylint: disable=protected-access,missing-function-docstring

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.run_multiworkload_competitor_e2e_artifact import (
    canonical_hash,
    file_sha256,
)
from scripts.run_per_child_objective_refinement_artifact import (
    ARTIFACT_SCHEMA_VERSION,
    _validate_worker_result,
    validate_evidence_structure,
)

ARTIFACT_DIR = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "per-child-objective-refinement"
    / "vnncomp21-resnet2b-two-clause-cpu-v1"
)


def _load(name: str):
    return json.loads((ARTIFACT_DIR / name).read_text(encoding="utf-8"))


def test_frozen_per_child_artifact_is_digest_bound_and_no_go() -> None:
    manifest = _load("manifest.json")
    evidence = _load("evidence.json")
    actual_files = {
        str(path.relative_to(ARTIFACT_DIR)): file_sha256(path)
        for path in sorted(ARTIFACT_DIR.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }

    assert manifest["schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert manifest["status"] == evidence["status"] == "validated_no_go"
    assert manifest["performance_claimed"] is False
    assert manifest["files"] == actual_files
    assert manifest["evidence_hash"] == canonical_hash(evidence)
    validate_evidence_structure(evidence)

    assert [item["root_lower_delta"] for item in evidence["comparisons"]] == [
        0.0,
        0.0,
    ]
    assert [item["worst_leaf_lower_delta"] for item in evidence["comparisons"]] == [
        -0.84796142578125,
        -0.9366455078125,
    ]
    assert all(
        item["strict_worst_frontier_improvement"] is False
        for item in evidence["comparisons"]
    )


def test_per_child_artifact_binds_each_node_to_refinement_ir() -> None:
    evidence = _load("evidence.json")
    per_child = [
        record["result"]
        for record in evidence["records"]
        if record["mode"] == "per_child"
    ]

    assert len(per_child) == 2
    for result in per_child:
        queue = result["queue_trace"]
        assert len(queue["evaluations"]) == 7
        assert len(queue["per_child_refinements"]) == 7
        assert len(result["refinement_programs"]) == 7
        assert all(
            record["parent_refinement_consumed_as_exact"] is False
            for record in queue["per_child_refinements"]
        )
        _validate_worker_result(result)


def test_per_child_artifact_ir_and_lineage_tampering_fail_closed() -> None:
    evidence = _load("evidence.json")
    result = next(
        record["result"]
        for record in evidence["records"]
        if record["mode"] == "per_child"
    )

    tampered_ir = deepcopy(result)
    tampered_ir["refinement_programs"][0]["plan"]["semantics_owner"] = "tampered"
    with pytest.raises(ValueError, match="IR/trace linkage"):
        _validate_worker_result(tampered_ir)

    tampered_lineage = deepcopy(result)
    fake_hash = "f" * 64
    tampered_lineage["queue_trace"]["per_child_refinements"][1][
        "node_split_state_hash"
    ] = fake_hash
    tampered_lineage["queue_trace_hash"] = canonical_hash(
        tampered_lineage["queue_trace"]
    )
    with pytest.raises(ValueError, match="node/refinement lineage"):
        _validate_worker_result(tampered_lineage)


def test_per_child_artifact_closure_tampering_fails_closed() -> None:
    evidence = _load("evidence.json")
    tampered = deepcopy(evidence)
    tampered["status"] = "validated_reduced"
    with pytest.raises(ValueError, match="closure status"):
        validate_evidence_structure(tampered)
