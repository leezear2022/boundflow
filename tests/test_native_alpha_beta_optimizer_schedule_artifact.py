"""Frozen NRIR-11 optimizer Schedule artifact contracts."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from boundflow.runtime.abcrown_adapter import file_sha256
from scripts.run_native_alpha_beta_optimizer_schedule_artifact import (
    ARTIFACT_FILE,
    ARTIFACT_SCHEMA_VERSION,
    hashlib_sha256,
    validate_native_optimizer_schedule_evidence,
)

ARTIFACT_DIR = (
    Path("artifacts/native-alpha-beta-optimizer-schedule")
    / "vnncomp21-resnet2b-prop0-cpu-v1"
)


def _load(name: str) -> dict[str, object]:
    value = json.loads((ARTIFACT_DIR / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _evidence() -> dict[str, object]:
    artifact = _load(ARTIFACT_FILE)
    evidence = artifact["evidence"]
    assert isinstance(evidence, dict)
    return evidence


def test_frozen_optimizer_schedule_artifact_digest_and_contract() -> None:
    manifest = _load("manifest.json")
    artifact = _load(ARTIFACT_FILE)
    evidence = artifact["evidence"]
    assert isinstance(evidence, dict)
    assert manifest["schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert manifest["status"] == "ok"
    assert manifest["performance_claimed"] is False
    assert manifest["files"] == {
        ARTIFACT_FILE: file_sha256(ARTIFACT_DIR / ARTIFACT_FILE)
    }
    assert manifest["evidence_hash"] == hashlib_sha256(evidence)
    validate_native_optimizer_schedule_evidence(evidence)


@pytest.mark.parametrize(
    "gate",
    [
        "fixed_resnet_source_is_digest_bound",
        "optimizer_plan_binds_nrir10_source_compiler_stack",
        "fixed_step_control_is_first_class_plan_task_schedule",
        "backward_update_projection_execute_in_schedule_order",
        "schedule_selected_state_matches_legacy_optimizer",
        "selected_state_reexecutes_through_native_compiler_stack",
        "best_state_is_selected_from_evaluated_iterations",
        "claims_remain_correctness_only",
    ],
)
def test_artifact_rejects_failed_gate(gate: str) -> None:
    evidence = deepcopy(_evidence())
    gates = evidence["gates"]
    assert isinstance(gates, dict)
    gates[gate] = False
    with pytest.raises(ValueError, match="gates"):
        validate_native_optimizer_schedule_evidence(evidence)


def test_artifact_rejects_plan_and_task_rehash_tampering() -> None:
    evidence = deepcopy(_evidence())
    initial = evidence["initial_state"]
    program = evidence["optimizer_program"]
    assert isinstance(initial, dict) and isinstance(program, dict)
    plan = program["plan"]
    hashes = program["hashes"]
    assert isinstance(plan, dict) and isinstance(hashes, dict)
    plan["initial_state_hash"] = "f" * 64
    hashes["optimizer_plan_hash"] = hashlib_sha256(plan)
    with pytest.raises(ValueError, match="Plan/source identity"):
        validate_native_optimizer_schedule_evidence(evidence)

    evidence = deepcopy(_evidence())
    program = evidence["optimizer_program"]
    trace = evidence["execution_trace"]
    assert isinstance(program, dict) and isinstance(trace, dict)
    task_module = program["task_module"]
    schedule = program["schedule"]
    hashes = program["hashes"]
    assert isinstance(task_module, dict)
    assert isinstance(schedule, dict) and isinstance(hashes, dict)
    tasks = task_module["tasks"]
    trace_actions = trace["actions"]
    assert isinstance(tasks, list) and isinstance(trace_actions, list)
    assert isinstance(tasks[2], dict) and isinstance(trace_actions[2], dict)
    tasks[2]["kind"] = "select_best"
    trace_actions[2]["kind"] = "select_best"
    task_hash = hashlib_sha256(task_module)
    hashes["optimizer_task_module_hash"] = task_hash
    schedule["optimizer_task_module_hash"] = task_hash
    schedule_hash = hashlib_sha256(schedule)
    hashes["optimizer_schedule_hash"] = schedule_hash
    trace["task_module_hash"] = task_hash
    trace["schedule_hash"] = schedule_hash
    program["execution_trace_hash"] = hashlib_sha256(trace)
    with pytest.raises(ValueError, match="Task/Schedule identity"):
        validate_native_optimizer_schedule_evidence(evidence)


def test_artifact_rejects_schedule_and_transition_tampering() -> None:
    evidence = deepcopy(_evidence())
    program = evidence["optimizer_program"]
    trace = evidence["execution_trace"]
    assert isinstance(program, dict) and isinstance(trace, dict)
    schedule = program["schedule"]
    hashes = program["hashes"]
    assert isinstance(schedule, dict) and isinstance(hashes, dict)
    actions = schedule["actions"]
    trace_actions = trace["actions"]
    assert isinstance(actions, list) and isinstance(trace_actions, list)
    assert isinstance(actions[0], dict) and isinstance(trace_actions[0], dict)
    actions[0]["task_id"] = "optimizer.reduce_metric.s000"
    trace_actions[0]["task_id"] = "optimizer.reduce_metric.s000"
    schedule_hash = hashlib_sha256(schedule)
    hashes["optimizer_schedule_hash"] = schedule_hash
    trace["schedule_hash"] = schedule_hash
    program["execution_trace_hash"] = hashlib_sha256(trace)
    with pytest.raises(ValueError, match="Schedule/Task linkage"):
        validate_native_optimizer_schedule_evidence(evidence)

    evidence = deepcopy(_evidence())
    program = evidence["optimizer_program"]
    trace = evidence["execution_trace"]
    assert isinstance(program, dict) and isinstance(trace, dict)
    trace_actions = trace["actions"]
    assert isinstance(trace_actions, list) and isinstance(trace_actions[0], dict)
    outputs = trace_actions[0]["output_hashes"]
    assert isinstance(outputs, dict)
    outputs["optimizer.bounds.s000"] = "f" * 64
    program["execution_trace_hash"] = hashlib_sha256(trace)
    with pytest.raises(ValueError, match="evaluated state/bound transition"):
        validate_native_optimizer_schedule_evidence(evidence)


def test_artifact_rejects_claim_and_gradient_inflation() -> None:
    evidence = deepcopy(_evidence())
    evidence["performance_claimed"] = True
    evidence["property_status"] = "proven"
    with pytest.raises(ValueError, match="header/claim boundary"):
        validate_native_optimizer_schedule_evidence(evidence)

    evidence = deepcopy(_evidence())
    trace = evidence["execution_trace"]
    program = evidence["optimizer_program"]
    execution = evidence["execution"]
    assert isinstance(trace, dict) and isinstance(program, dict)
    assert isinstance(execution, dict)
    actions = trace["actions"]
    assert isinstance(actions, list) and isinstance(actions[2], dict)
    actions[2]["beta_gradient_l1"] = 0.0
    execution["beta_gradient_l1"] = 0.0
    program["execution_trace_hash"] = hashlib_sha256(trace)
    with pytest.raises(ValueError, match="backward gradient"):
        validate_native_optimizer_schedule_evidence(evidence)
