"""Frozen NRIR-39 fixed-budget objective-branch artifact gates."""

# pylint: disable=missing-function-docstring

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.run_objective_branch_shared_evaluator_pilot import (
    MANIFEST_SCHEMA_VERSION,
    PILOT_SCHEMA_VERSION,
    _canonical_hash,
    _file_sha256,
    _semantic_clause,
    _semantic_pilot,
    validate_pilot,
)

ARTIFACT_DIR = Path(
    "artifacts/objective-branch-shared-evaluator/"
    "vnncomp21-resnet2b-property0-cpu-pilot-v1"
)


def _load() -> tuple[dict[str, object], dict[str, object]]:
    pilot = json.loads((ARTIFACT_DIR / "pilot.json").read_text(encoding="utf-8"))
    manifest = json.loads((ARTIFACT_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert isinstance(pilot, dict)
    assert isinstance(manifest, dict)
    return pilot, manifest


def _rehash(value: dict[str, object], *, clause_index: int) -> None:
    clauses = value["clauses"]
    assert isinstance(clauses, list)
    clause = clauses[clause_index]
    assert isinstance(clause, dict)
    clause["clause_hash"] = _canonical_hash(_semantic_clause(clause))
    value["pilot_hash"] = _canonical_hash(_semantic_pilot(value))


def test_frozen_objective_branch_shared_artifact_digest_and_replay() -> None:
    pilot, manifest = _load()

    assert pilot["schema_version"] == PILOT_SCHEMA_VERSION
    assert manifest["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert manifest["files"] == {
        "pilot.json": _file_sha256(ARTIFACT_DIR / "pilot.json")
    }
    assert manifest["pilot_hash"] == _canonical_hash(pilot)
    assert pilot["performance_claimed"] is False
    assert pilot["status"] == "validated-reduced"
    validate_pilot(pilot)


def test_both_preregistered_branch_gates_pass_without_claim_upgrade() -> None:
    pilot, _manifest = _load()
    clauses = {item["original_clause_index"]: item for item in pilot["clauses"]}

    assert set(clauses) == {2, 3}
    assert clauses[2]["control"]["worst_active_lower"] == -37.57428741455078
    assert clauses[3]["control"]["worst_active_lower"] == -35.90021514892578
    assert clauses[2]["decision"]["worst_active_lower_improvement"] > 2.04
    assert clauses[3]["decision"]["worst_active_lower_improvement"] > 5.64
    assert clauses[2]["decision"]["median_active_lower_delta"] > 2.53
    assert clauses[3]["decision"]["median_active_lower_delta"] > 5.88
    assert all(item["decision"]["go"] is True for item in clauses.values())
    assert all(len(item["branch_bindings"]) == 31 for item in clauses.values())
    assert pilot["claim_boundary"] == "fixed_budget_objective_branch_selection_only"


def test_policy_coverage_selection_and_task_tamper_fail_closed() -> None:
    pilot, _manifest = _load()

    policy = deepcopy(pilot)
    policy["clauses"][0]["plan"]["candidates_per_relu"] = 7
    _rehash(policy, clause_index=0)
    with pytest.raises(ValueError, match="frozen plan"):
        validate_pilot(policy)

    coverage = deepcopy(pilot)
    coverage["clauses"][0]["branch_bindings"].pop()
    coverage["clauses"][0]["branch_evidence"].pop()
    _rehash(coverage, clause_index=0)
    with pytest.raises(ValueError, match="branch coverage"):
        validate_pilot(coverage)

    selection = deepcopy(pilot)
    clause = selection["clauses"][0]
    trace = clause["branch_evidence"][0]["trace"]
    trace["selected_candidate_ordinal"] = (
        trace["selected_candidate_ordinal"] + 1
    ) % len(trace["scores"])
    clause["branch_evidence"][0]["trace_hash"] = _canonical_hash(trace)
    clause["branch_bindings"][0]["branch_trace_hash"] = clause["branch_evidence"][0][
        "trace_hash"
    ]
    _rehash(selection, clause_index=0)
    with pytest.raises(ValueError, match="branch semantic binding"):
        validate_pilot(selection)

    task = deepcopy(pilot)
    task["clauses"][1]["task_kinds"].pop()
    _rehash(task, clause_index=1)
    with pytest.raises(ValueError, match="Task/Schedule"):
        validate_pilot(task)


def test_claim_and_control_drift_fail_closed() -> None:
    pilot, _manifest = _load()

    claim = deepcopy(pilot)
    claim["performance_claimed"] = True
    claim["pilot_hash"] = _canonical_hash(_semantic_pilot(claim))
    with pytest.raises(ValueError, match="envelope"):
        validate_pilot(claim)

    control = deepcopy(pilot)
    control["clauses"][0]["control"]["worst_active_lower"] += 0.5
    _rehash(control, clause_index=0)
    with pytest.raises(ValueError, match="summary|NRIR-37"):
        validate_pilot(control)
