"""Frozen NRIR-24 convergence artifact, nesting, and tamper tests."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from boundflow.runtime.abcrown_adapter import file_sha256
from scripts.run_external_seeded_depth_node_convergence_artifact import (
    ARTIFACT_FILE,
    ARTIFACT_SCHEMA_VERSION,
    BUDGETS,
    EVIDENCE_SCHEMA_VERSION,
    HARD_CLAUSES,
    MANIFEST_FILE,
    SHARD_DIR,
    _load_shards,
    _shard_name,
    _validate_artifact,
    canonical_hash,
    shard_is_reusable,
    validate_evidence,
    validate_shard,
)

ARTIFACT_DIR = Path(
    "artifacts/external-seeded-depth-node-convergence/"
    "vnncomp21-resnet2b-prop0-hard3-cpu-v1"
)


def _load() -> tuple[dict[str, object], dict[str, object]]:
    manifest = json.loads((ARTIFACT_DIR / MANIFEST_FILE).read_text(encoding="utf-8"))
    artifact = json.loads((ARTIFACT_DIR / ARTIFACT_FILE).read_text(encoding="utf-8"))
    assert isinstance(manifest, dict)
    assert isinstance(artifact, dict)
    return manifest, artifact


def test_frozen_convergence_artifact_is_digest_bound_and_reduced() -> None:
    manifest, artifact = _load()
    checked_manifest, shards = _validate_artifact(ARTIFACT_DIR)
    evidence = artifact["evidence"]

    assert checked_manifest == manifest
    assert manifest["schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert artifact["schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert evidence["schema_version"] == EVIDENCE_SCHEMA_VERSION
    assert manifest["status"] == artifact["status"] == evidence["status"]
    assert evidence["status"] == "validated_reduced"
    assert evidence["performance_claimed"] is False
    assert manifest["performance_claimed"] is False
    assert manifest["evidence_hash"] == (
        "db0401bef0d938773fed04a173e49cae0ad0b4fdc4ffdd49450cc86fae7f0db6"
    )
    assert manifest["evidence_hash"] == canonical_hash(evidence)
    assert len(shards) == 9
    assert manifest["files"] == {
        ARTIFACT_FILE: file_sha256(ARTIFACT_DIR / ARTIFACT_FILE),
        **{
            f"{SHARD_DIR}/{name}": file_sha256(ARTIFACT_DIR / SHARD_DIR / name)
            for name in sorted(shards)
        },
    }


def test_frozen_curves_improve_but_do_not_close() -> None:
    _manifest, artifact = _load()
    curves = artifact["evidence"]["curves"]

    assert [curve["clause_index"] for curve in curves] == [0, 2, 4]
    assert [
        [point["worst_terminal_lower"] for point in curve["points"]] for curve in curves
    ] == [
        [-0.3182867765426636, -0.29950618743896484, -0.2823597192764282],
        [-0.425476610660553, -0.41345643997192383, -0.40184497833251953],
        [-0.5041420459747314, -0.47910404205322266, -0.45993947982788086],
    ]
    assert [
        [point["worst_lower_delta_from_previous"] for point in curve["points"]]
        for curve in curves
    ] == [
        [None, 0.01878058910369873, 0.01714646816253662],
        [None, 0.01202017068862915, 0.011611461639404297],
        [None, 0.02503800392150879, 0.019164562225341797],
    ]
    assert all(
        curve["monotonic_non_decreasing"] is True
        and curve["strict_improvement_after_depth_two"] is True
        and curve["depth_four_saturated"] is False
        and curve["deepest_bounded_tree_status"] == "unknown"
        for curve in curves
    )


def test_logical_domain_nesting_uses_split_identity_not_execution_order() -> None:
    _manifest, artifact = _load()
    evidence = artifact["evidence"]
    nesting = evidence["logical_domain_nesting"]

    assert evidence["gates"] == {
        "all_nine_units_present": True,
        "all_nested_logical_domains_match": True,
        "all_clauses_monotonic_non_decreasing": True,
        "strict_improvement_after_depth_two": True,
        "all_clauses_depth_four_saturated": False,
        "any_fixed_clause_bounded_tree_closed": False,
    }
    assert set(nesting) == {
        f"clause{clause}:n7_d2->n15_d3" for clause in HARD_CLAUSES
    } | {f"clause{clause}:n15_d3->n31_d4" for clause in HARD_CLAUSES}
    assert all(
        item["all_smaller_domains_present"] is True
        and item["logical_lineage_matches"] is True
        and item["selected_branch_matches"] is True
        and item["refinement_semantics_match"] is True
        and item["numerical_semantics_match"] is True
        and item["passed"] is True
        for item in nesting.values()
    )
    assert nesting["clause2:n15_d3->n31_d4"]["max_common_lower_abs_diff"] == (
        1.1324882507324219e-06
    )
    assert all(item["numeric_tolerance"] == 1e-5 for item in nesting.values())


def test_shard_source_lineage_numeric_and_claim_tampering_fail_closed() -> None:
    _manifest, artifact = _load()
    shards = _load_shards(ARTIFACT_DIR)
    name = _shard_name(0, *BUDGETS[0])
    original = shards[name]

    source = deepcopy(original)
    source["external_seed_ir"]["semantics_owner"] = "boundflow_native"
    source_body = dict(source)
    source_body.pop("semantic_hash")
    source["semantic_hash"] = canonical_hash(source_body)
    with pytest.raises(ValueError, match="seed identity"):
        validate_shard(
            source,
            expected_clause=0,
            expected_max_nodes=7,
            expected_max_depth=2,
        )

    lineage = deepcopy(original)
    refinement = lineage["queue"]["refinements"][1]
    refinement["source_intermediate_constraints_hash"] = "0" * 64
    lineage["queue"]["evaluations"][1]["intermediate_refinement_trace_hash"] = (
        canonical_hash(refinement)
    )
    lineage_body = dict(lineage)
    lineage_body.pop("semantic_hash")
    lineage["semantic_hash"] = canonical_hash(lineage_body)
    with pytest.raises(ValueError, match="ancestral refinement lineage"):
        validate_shard(
            lineage,
            expected_clause=0,
            expected_max_nodes=7,
            expected_max_depth=2,
        )

    numeric = deepcopy(original)
    numeric["queue"]["evaluations"][-1]["lower"] -= 0.1
    numeric_body = dict(numeric)
    numeric_body.pop("semantic_hash")
    numeric["semantic_hash"] = canonical_hash(numeric_body)
    with pytest.raises(ValueError, match="summary differs"):
        validate_shard(
            numeric,
            expected_clause=0,
            expected_max_nodes=7,
            expected_max_depth=2,
        )

    claim = deepcopy(artifact["evidence"])
    claim["performance_claimed"] = True
    with pytest.raises(ValueError, match="aggregate evidence"):
        validate_evidence(claim, shards)


def test_checkpoint_resume_reuses_only_strictly_valid_shards(tmp_path: Path) -> None:
    source = ARTIFACT_DIR / SHARD_DIR / _shard_name(4, *BUDGETS[-1])
    assert shard_is_reusable(source, clause=4, max_nodes=31, max_depth=4)

    shard = json.loads(source.read_text(encoding="utf-8"))
    shard["performance_claimed"] = True
    tampered = tmp_path / source.name
    tampered.write_text(json.dumps(shard), encoding="utf-8")
    assert not shard_is_reusable(tampered, clause=4, max_nodes=31, max_depth=4)
