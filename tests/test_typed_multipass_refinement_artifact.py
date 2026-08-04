"""Frozen NRIR-26 typed multi-pass artifact and tamper tests."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from boundflow.runtime.abcrown_adapter import file_sha256
from scripts.run_typed_multipass_refinement_artifact import (
    ARTIFACT_FILE,
    ARTIFACT_SCHEMA_VERSION,
    EVIDENCE_SCHEMA_VERSION,
    HARD_CLAUSES,
    MANIFEST_FILE,
    MODES,
    MULTI_PASS_POLICY,
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
    "artifacts/typed-multipass-refinement/vnncomp21-resnet2b-prop0-hard3-cpu-v1"
)


def _load() -> tuple[dict[str, object], dict[str, object]]:
    manifest = json.loads((ARTIFACT_DIR / MANIFEST_FILE).read_text(encoding="utf-8"))
    artifact = json.loads((ARTIFACT_DIR / ARTIFACT_FILE).read_text(encoding="utf-8"))
    assert isinstance(manifest, dict)
    assert isinstance(artifact, dict)
    return manifest, artifact


def test_frozen_multipass_artifact_is_digest_bound_and_no_go() -> None:
    manifest, artifact = _load()
    checked_manifest, shards = _validate_artifact(ARTIFACT_DIR)
    evidence = artifact["evidence"]

    assert checked_manifest == manifest
    assert manifest["schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert artifact["schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert evidence["schema_version"] == EVIDENCE_SCHEMA_VERSION
    assert manifest["status"] == artifact["status"] == evidence["status"]
    assert evidence["status"] == "validated_no_go"
    assert evidence["performance_claimed"] is False
    assert manifest["evidence_hash"] == (
        "38992cace70214ffcbd670f03dcfca182e0925bee31eb4df885dab4dab03494d"
    )
    assert manifest["evidence_hash"] == canonical_hash(evidence)
    assert len(shards) == 6
    assert manifest["files"] == {
        ARTIFACT_FILE: file_sha256(ARTIFACT_DIR / ARTIFACT_FILE),
        **{
            f"{SHARD_DIR}/{name}": file_sha256(ARTIFACT_DIR / SHARD_DIR / name)
            for name in sorted(shards)
        },
    }


def test_split_two_pass_conserves_caps_but_does_not_improve_worst_lower() -> None:
    _manifest, artifact = _load()
    evidence = artifact["evidence"]
    comparisons = evidence["comparisons"]

    assert [item["clause_index"] for item in comparisons] == [0, 2, 4]
    assert [
        item["single_pass_dynamic8_24"]["worst_terminal_lower"] for item in comparisons
    ] == [
        -0.2819737195968628,
        -0.40161198377609253,
        -0.4596676826477051,
    ]
    assert [
        item["split_two_pass_dynamic8_24"]["worst_terminal_lower"]
        for item in comparisons
    ] == [
        -0.2819737195968628,
        -0.40161198377609253,
        -0.4596676826477051,
    ]
    assert all(
        item["split_worst_lower_delta"] == 0.0
        and item["split_not_weaker"] is True
        and item["split_strictly_better"] is False
        and item["same_planned_total_target_cap"] is True
        and item["logical_domain_overlap"] == item["logical_domain_union"] == 31
        and item["single_pass_dynamic8_24"]["planned_total_target_cap_per_relu"]
        == item["split_two_pass_dynamic8_24"]["planned_total_target_cap_per_relu"]
        == 496
        and item["single_pass_dynamic8_24"]["actual_selected_target_count"]
        == item["split_two_pass_dynamic8_24"]["actual_selected_target_count"]
        == 2976
        for item in comparisons
    )
    assert evidence["gates"] == {
        "all_six_units_present": True,
        "all_tree_total_caps_conserved": True,
        "split_not_weaker_on_all_hard_clauses": True,
        "split_strictly_improves_at_least_one_hard_clause": False,
        "any_bounded_tree_closed": False,
    }


def test_two_pass_decisions_partition_caps_and_chain_target_ledgers() -> None:
    shards = _load_shards(ARTIFACT_DIR)
    for clause in HARD_CLAUSES:
        shard = shards[_shard_name(clause, MODES[1])]
        for record in shard["queue"]["refinements"]:
            assigned = record["budget_decision"]["assigned_max_neurons_per_relu"]
            first, second = record["multi_pass_decisions"]
            assert record["multi_pass_policy"] == MULTI_PASS_POLICY.to_dict()
            assert (
                first["pass_target_cap_per_relu"]
                == (second["pass_target_cap_per_relu"])
                == assigned // 2
            )
            assert (
                first["result_target_ledger_hash"] == second["prior_target_ledger_hash"]
            )
            assert (
                second["prior_selected_target_count"]
                == first["cumulative_selected_target_count"]
            )
            assert first["continuation"] is second["continuation"] is True


def test_program_pass_and_claim_tampering_fail_closed() -> None:
    _manifest, artifact = _load()
    shards = _load_shards(ARTIFACT_DIR)
    name = _shard_name(0, MODES[1])
    original = shards[name]

    program = deepcopy(original)
    program["queue"]["refinement_programs"][0]["actual_selected_target_count"] += 1
    program_body = dict(program)
    program_body.pop("semantic_hash")
    program["semantic_hash"] = canonical_hash(program_body)
    with pytest.raises(ValueError, match="actual target total differs"):
        validate_shard(program, expected_clause=0, expected_mode=MODES[1])

    decision = deepcopy(original)
    pass_decision = decision["queue"]["refinement_programs"][0]["pass_decisions"][1]
    pass_decision["prior_target_ledger_hash"] = "f" * 64
    decision_body = dict(decision)
    decision_body.pop("semantic_hash")
    decision["semantic_hash"] = canonical_hash(decision_body)
    with pytest.raises(ValueError, match="decision linkage differs"):
        validate_shard(decision, expected_clause=0, expected_mode=MODES[1])

    claim = deepcopy(artifact["evidence"])
    claim["comparisons"][0]["split_strictly_better"] = True
    with pytest.raises(ValueError, match="aggregate evidence"):
        validate_evidence(claim, shards)


def test_checkpoint_resume_reuses_only_strictly_valid_shards(tmp_path: Path) -> None:
    source = ARTIFACT_DIR / SHARD_DIR / _shard_name(4, MODES[1])
    assert shard_is_reusable(source, clause=4, mode=MODES[1])

    shard = json.loads(source.read_text(encoding="utf-8"))
    shard["performance_claimed"] = True
    tampered = tmp_path / source.name
    tampered.write_text(json.dumps(shard), encoding="utf-8")
    assert not shard_is_reusable(tampered, clause=4, mode=MODES[1])
