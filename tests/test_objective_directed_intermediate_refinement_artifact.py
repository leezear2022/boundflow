"""Contract and tamper tests for the frozen NRIR-20 artifact."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.run_objective_directed_intermediate_refinement_artifact import (
    EVIDENCE_FILE,
    MANIFEST_FILE,
    canonical_hash,
    file_sha256,
    validate_evidence_structure,
)

ARTIFACT_DIR = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "objective-directed-intermediate-refinement"
    / "vnncomp21-resnet2b-two-clause-cpu-v1"
)


def _load(name: str) -> dict[str, Any]:
    value = json.loads((ARTIFACT_DIR / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _relink_refinement(record: dict[str, Any]) -> None:
    result = record["result"]
    refinement = result["refinement"]
    plan = refinement["plan"]
    task = refinement["task"]
    schedule = refinement["schedule"]
    trace = refinement["execution_trace"]
    plan_hash = canonical_hash(plan)
    task["refinement_plan_hash"] = plan_hash
    task_hash = canonical_hash(task)
    schedule["refinement_plan_hash"] = plan_hash
    schedule["refinement_task_module_hash"] = task_hash
    schedule_hash = canonical_hash(schedule)
    refinement["hashes"] = {
        "refinement_plan_hash": plan_hash,
        "refinement_task_module_hash": task_hash,
        "refinement_schedule_hash": schedule_hash,
    }
    trace["plan_hash"] = plan_hash
    trace["task_module_hash"] = task_hash
    trace["schedule_hash"] = schedule_hash
    refinement["execution_trace_hash"] = canonical_hash(trace)


def test_objective_artifact_has_exact_same_budget_improvements() -> None:
    manifest = _load(MANIFEST_FILE)
    evidence = _load(EVIDENCE_FILE)
    validate_evidence_structure(evidence)

    assert manifest["evidence_hash"] == canonical_hash(evidence)
    assert manifest["files"] == {
        str(path.relative_to(ARTIFACT_DIR)): file_sha256(path)
        for path in sorted(ARTIFACT_DIR.rglob("*"))
        if path.is_file() and path.name != MANIFEST_FILE
    }
    comparisons = {item["clause_index"]: item for item in evidence["comparisons"]}
    assert comparisons[0]["width_target_count"] == 96
    assert comparisons[0]["objective_target_count"] == 96
    assert comparisons[0]["target_overlap_count"] == 16
    assert comparisons[0]["root_lower_delta"] == pytest.approx(55.928741455078125)
    assert comparisons[1]["target_overlap_count"] == 27
    assert comparisons[1]["root_lower_delta"] == pytest.approx(26.22894287109375)
    assert all(item["performance_claimed"] is False for item in comparisons.values())


def test_objective_artifact_binds_clause_objective_into_plan_task_schedule() -> None:
    evidence = _load(EVIDENCE_FILE)
    for record in evidence["records"]:
        result = record["result"]
        refinement = result["refinement"]
        plan = refinement["plan"]
        select_task = refinement["task"]["tasks"][2]
        if record["mode"] == "width":
            assert "objective_hash" not in plan
            assert select_task["input_value_ids"] == [
                "refine.candidates",
                "refine.policy",
            ]
        else:
            assert len(plan["objective_hash"]) == 64
            assert select_task["input_value_ids"] == [
                "refine.bounds.p0",
                "refine.candidates",
                "refine.policy",
                "refine.objective_influence",
            ]
            assert all(
                target["selection_score"]
                == pytest.approx(
                    target["objective_influence"] * target["initial_width"]
                )
                for target in plan["targets"]
            )
        trace = refinement["execution_trace"]
        assert trace["plan_hash"] == refinement["hashes"]["refinement_plan_hash"]
        assert (
            trace["task_module_hash"]
            == refinement["hashes"]["refinement_task_module_hash"]
        )
        assert (
            trace["schedule_hash"] == refinement["hashes"]["refinement_schedule_hash"]
        )


def test_objective_artifact_rejects_coherently_relinked_score_tamper() -> None:
    evidence = _load(EVIDENCE_FILE)
    changed = deepcopy(evidence)
    record = next(item for item in changed["records"] if item["mode"] == "objective")
    target = record["result"]["refinement"]["plan"]["targets"][0]
    target["selection_score"] += 1.0
    _relink_refinement(record)

    with pytest.raises(ValueError, match="target score differs"):
        validate_evidence_structure(changed)


def test_objective_artifact_rejects_coherently_relinked_dependency_tamper() -> None:
    evidence = _load(EVIDENCE_FILE)
    changed = deepcopy(evidence)
    record = next(item for item in changed["records"] if item["mode"] == "objective")
    refinement = record["result"]["refinement"]
    refinement["task"]["tasks"][2]["input_value_ids"].remove(
        "refine.objective_influence"
    )
    refinement["schedule"]["actions"][2]["input_value_ids"].remove(
        "refine.objective_influence"
    )
    _relink_refinement(record)

    with pytest.raises(ValueError, match="target-selection dependency differs"):
        validate_evidence_structure(changed)


def test_objective_artifact_log_digests_are_exact() -> None:
    evidence = _load(EVIDENCE_FILE)
    for record in evidence["records"]:
        path = ARTIFACT_DIR / record["log_path"]
        assert path.is_file()
        assert file_sha256(path) == record["log_sha256"]
