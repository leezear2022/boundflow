"""Frozen NRIR-25 dynamic-budget artifact and tamper tests."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from boundflow.runtime.abcrown_adapter import file_sha256
from scripts.run_dynamic_ancestral_refinement_budget_artifact import (
    ARTIFACT_FILE,
    ARTIFACT_SCHEMA_VERSION,
    EVIDENCE_SCHEMA_VERSION,
    HARD_CLAUSES,
    MANIFEST_FILE,
    MAX_NODES,
    MODES,
    REFINEMENT_POLICY,
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
    "artifacts/dynamic-ancestral-refinement-budget/"
    "vnncomp21-resnet2b-prop0-hard3-cpu-v1"
)


def _load() -> tuple[dict[str, object], dict[str, object]]:
    manifest = json.loads((ARTIFACT_DIR / MANIFEST_FILE).read_text(encoding="utf-8"))
    artifact = json.loads((ARTIFACT_DIR / ARTIFACT_FILE).read_text(encoding="utf-8"))
    assert isinstance(manifest, dict)
    assert isinstance(artifact, dict)
    return manifest, artifact


def test_frozen_dynamic_budget_artifact_is_digest_bound_and_reduced() -> None:
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
        "85d9f274c6e17614bcbf318bdbfea18219b03876024be16aea3329ee4d3c56bd"
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


def test_dynamic_budget_improves_all_three_clauses_at_equal_planned_cap() -> None:
    _manifest, artifact = _load()
    evidence = artifact["evidence"]
    comparisons = evidence["comparisons"]

    assert [item["clause_index"] for item in comparisons] == [0, 2, 4]
    assert [item["fixed16"]["worst_terminal_lower"] for item in comparisons] == [
        -0.2823597192764282,
        -0.40184497833251953,
        -0.45993947982788086,
    ]
    assert [item["dynamic8_24"]["worst_terminal_lower"] for item in comparisons] == [
        -0.2819737195968628,
        -0.40161198377609253,
        -0.4596676826477051,
    ]
    assert [item["dynamic_worst_lower_delta"] for item in comparisons] == [
        0.0003859996795654297,
        0.00023299455642700195,
        0.00027179718017578125,
    ]
    assert all(
        item["dynamic_not_weaker"] is True
        and item["dynamic_strictly_better"] is True
        and item["same_planned_target_cap"] is True
        and item["fixed16"]["planned_target_cap_per_relu"]
        == item["dynamic8_24"]["planned_target_cap_per_relu"]
        == MAX_NODES * REFINEMENT_POLICY.max_neurons_per_relu
        for item in comparisons
    )
    assert evidence["gates"] == {
        "all_six_units_present": True,
        "all_tree_planned_caps_conserved": True,
        "dynamic_not_weaker_on_all_hard_clauses": True,
        "dynamic_strictly_improves_at_least_one_hard_clause": True,
        "any_fixed_clause_bounded_tree_closed": False,
    }


def test_budget_decisions_are_group_conserved_and_lowered_into_plan_ir() -> None:
    shards = _load_shards(ARTIFACT_DIR)
    for clause in HARD_CLAUSES:
        shard = shards[_shard_name(clause, "dynamic8_24")]
        refinements = shard["queue"]["refinements"]
        programs = {
            item["node_id"]: item for item in shard["queue"]["refinement_programs"]
        }
        groups: dict[str, list[dict[str, object]]] = {}
        for refinement in refinements:
            decision = refinement["budget_decision"]
            groups.setdefault(decision["group_id"], []).append(decision)
            assert refinement["budget_decision_hash"] == canonical_hash(decision)
            assert (
                programs[refinement["node_id"]]["policy"]["max_neurons_per_relu"]
                == decision["assigned_max_neurons_per_relu"]
            )

        assert len(refinements) == len(programs) == MAX_NODES
        for decisions in groups.values():
            assert len(decisions) == decisions[0]["group_size"]
            assert (
                sum(int(item["assigned_max_neurons_per_relu"]) for item in decisions)
                == decisions[0]["group_assigned_cap_total"]
            )
            assert (
                decisions[0]["group_assigned_cap_total"]
                == decisions[0]["group_base_cap_total"]
            )


def test_budget_plan_and_claim_tampering_fail_closed() -> None:
    _manifest, artifact = _load()
    shards = _load_shards(ARTIFACT_DIR)
    name = _shard_name(0, "dynamic8_24")
    original = shards[name]

    budget = deepcopy(original)
    budget["queue"]["refinements"][2]["budget_decision"][
        "group_assigned_cap_total"
    ] += 1
    budget_refinement = budget["queue"]["refinements"][2]
    budget_refinement["budget_decision_hash"] = canonical_hash(
        budget_refinement["budget_decision"]
    )
    budget["queue"]["evaluations"][2]["intermediate_refinement_trace_hash"] = (
        canonical_hash(budget_refinement)
    )
    budget_body = dict(budget)
    budget_body.pop("semantic_hash")
    budget["semantic_hash"] = canonical_hash(budget_body)
    with pytest.raises(ValueError, match="budget"):
        validate_shard(budget, expected_clause=0, expected_mode="dynamic8_24")

    plan = deepcopy(original)
    high_index = next(
        index
        for index, refinement in enumerate(plan["queue"]["refinements"])
        if refinement["budget_decision"]["assigned_max_neurons_per_relu"] == 24
    )
    plan["queue"]["refinement_programs"][high_index]["policy"][
        "max_neurons_per_relu"
    ] = 16
    plan_body = dict(plan)
    plan_body.pop("semantic_hash")
    plan["semantic_hash"] = canonical_hash(plan_body)
    with pytest.raises(ValueError, match="Plan policy"):
        validate_shard(plan, expected_clause=0, expected_mode="dynamic8_24")

    claim = deepcopy(artifact["evidence"])
    claim["comparisons"][0]["dynamic_worst_lower_delta"] += 1.0
    with pytest.raises(ValueError, match="aggregate evidence"):
        validate_evidence(claim, shards)


def test_checkpoint_resume_reuses_only_strictly_valid_shards(tmp_path: Path) -> None:
    source = ARTIFACT_DIR / SHARD_DIR / _shard_name(4, MODES[-1])
    assert shard_is_reusable(source, clause=4, mode="dynamic8_24")

    shard = json.loads(source.read_text(encoding="utf-8"))
    shard["performance_claimed"] = True
    tampered = tmp_path / source.name
    tampered.write_text(json.dumps(shard), encoding="utf-8")
    assert not shard_is_reusable(tampered, clause=4, mode="dynamic8_24")
