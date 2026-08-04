"""Frozen NRIR-23 artifact, source-lineage, and claim-boundary tests."""

# pylint: disable=missing-function-docstring,duplicate-code

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from boundflow.runtime.abcrown_adapter import file_sha256
from scripts.run_external_seeded_ancestral_refinement_artifact import (
    ARTIFACT_FILE,
    ARTIFACT_SCHEMA_VERSION,
    EVIDENCE_SCHEMA_VERSION,
    MANIFEST_FILE,
    canonical_hash,
    validate_evidence,
)

ARTIFACT_DIR = Path(
    "artifacts/external-seeded-ancestral-refinement/"
    "vnncomp21-resnet2b-prop0-hard3-cpu-v1"
)


def _load() -> tuple[dict[str, object], dict[str, object]]:
    manifest = json.loads((ARTIFACT_DIR / MANIFEST_FILE).read_text(encoding="utf-8"))
    artifact = json.loads((ARTIFACT_DIR / ARTIFACT_FILE).read_text(encoding="utf-8"))
    assert isinstance(manifest, dict)
    assert isinstance(artifact, dict)
    return manifest, artifact


def test_frozen_external_seeded_artifact_is_digest_bound_and_reduced() -> None:
    manifest, artifact = _load()
    evidence = artifact["evidence"]

    assert manifest["schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert artifact["schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert evidence["schema_version"] == EVIDENCE_SCHEMA_VERSION
    assert manifest["status"] == artifact["status"] == evidence["status"]
    assert evidence["status"] == "validated_reduced"
    assert manifest["performance_claimed"] is False
    assert evidence["performance_claimed"] is False
    assert manifest["files"] == {
        ARTIFACT_FILE: file_sha256(ARTIFACT_DIR / ARTIFACT_FILE)
    }
    assert manifest["evidence_hash"] == canonical_hash(evidence)
    validate_evidence(evidence)


def test_external_seed_and_ancestry_improve_without_property_closure() -> None:
    _manifest, artifact = _load()
    evidence = artifact["evidence"]
    comparisons = evidence["comparisons"]

    assert [item["clause_index"] for item in comparisons] == [0, 2, 4]
    assert [item["root_refinement_delta"] for item in comparisons] == [
        0.0006889104843139648,
        0.00112837553024292,
        0.0005342960357666016,
    ]
    assert [item["ancestral_over_root_global_delta"] for item in comparisons] == [
        0.000822901725769043,
        4.172325134277344e-06,
        0.0,
    ]
    assert all(
        item["ancestral_not_weaker_than_root_global"] is True
        and item["ancestral_fixed_tree_status"] == "unknown"
        for item in comparisons
    )
    assert (
        evidence["gates"]["ancestral_strictly_improves_at_least_one_hard_clause"]
        is True
    )


def test_external_seed_ir_and_child_lineage_are_explicit() -> None:
    _manifest, artifact = _load()
    evidence = artifact["evidence"]
    for clause in ("0", "2", "4"):
        ancestral = evidence["clauses"][clause]["external_seeded_ancestral"]
        trace = ancestral["queue_trace"]
        refinements = ancestral["refinements"]
        records = trace["per_child_refinements"]
        root_plan = refinements[0]["plan"]

        assert trace["per_child_refinement_strategy"] == (
            "external_seeded_ancestral_carry_v1"
        )
        assert root_plan["external_constraint_seed"]["semantics_owner"] == (
            "external_verifier"
        )
        assert (
            "refine.external_constraint_seed"
            in refinements[0]["task"]["tasks"][0]["input_value_ids"]
        )
        assert "external_constraint_seed_hash" in records[0]
        assert sum("source_parent_node_id" in item for item in records) == 6


def test_external_seed_lineage_and_claim_tampering_fail_closed() -> None:
    _manifest, artifact = _load()
    evidence = artifact["evidence"]

    seed = deepcopy(evidence)
    seed_plan = seed["clauses"]["0"]["external_seeded_ancestral"]["refinements"][0][
        "plan"
    ]
    seed_plan["external_constraint_seed"]["semantics_owner"] = "boundflow_native"
    with pytest.raises(ValueError, match="seed binding|IR evidence"):
        validate_evidence(seed)

    lineage = deepcopy(evidence)
    ancestral = lineage["clauses"]["2"]["external_seeded_ancestral"]
    trace = ancestral["queue_trace"]
    child = trace["per_child_refinements"][1]
    child["source_intermediate_constraints_hash"] = "0" * 64
    trace["evaluations"][1]["intermediate_refinement_trace_hash"] = canonical_hash(
        child
    )
    ancestral["queue_trace_hash"] = canonical_hash(trace)
    with pytest.raises(ValueError, match="parent lineage"):
        validate_evidence(lineage)

    claim = deepcopy(evidence)
    claim["performance_claimed"] = True
    with pytest.raises(ValueError, match="header"):
        validate_evidence(claim)
