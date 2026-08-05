"""Static IR tests for the NRIR44 ranking-floor root projection."""

# pylint: disable=missing-function-docstring

from dataclasses import replace

import pytest

from boundflow.ir.root_projection_floor import (
    NativeRootProjectionClauseOwnerIR,
    NativeRootProjectionFloorInstanceIR,
    NativeRootProjectionFloorPlanIR,
    lower_native_root_projection_floor_ir,
)


def _plan() -> NativeRootProjectionFloorPlanIR:
    return NativeRootProjectionFloorPlanIR(
        plan_id="root-projection-test",
        source_plan_hash="1" * 64,
        source_task_ir_hash="2" * 64,
        source_schedule_hash="3" * 64,
        clause_count=9,
        consumed_result_fields=(
            "root.lower",
            "root.upper",
            "root.branch_candidate",
            "query.status",
            "counterexample.evidence",
        ),
    )


def _instance(plan: NativeRootProjectionFloorPlanIR):
    return NativeRootProjectionFloorInstanceIR.create(
        plan=plan,
        objective_matrix_hash="4" * 64,
        thresholds_hash="5" * 64,
        clause_owners=tuple(
            NativeRootProjectionClauseOwnerIR(index, "6" * 64, "7" * 64)
            for index in range(9)
        ),
    )


def test_root_projection_lowers_first_class_plan_instance_task_schedule() -> None:
    plan = _plan()
    instance = _instance(plan)
    task_ir, schedule = lower_native_root_projection_floor_ir(plan, instance)

    assert len(task_ir.tasks) == 7
    assert [item.kind.value for item in task_ir.tasks] == [
        "admit_source",
        "analyze_consumers",
        "execute_baseline",
        "refine_objectives",
        "execute_root_projections",
        "rank_roots",
        "emit_floor",
    ]
    assert schedule.full_evaluation_budget == 279
    assert schedule.projected_evaluation_budget == 9
    schedule.validate(plan=plan, instance=instance, task_module=task_ir)


def test_root_projection_rejects_complete_verifier_budget_or_claim() -> None:
    with pytest.raises(ValueError, match="Plan IR is invalid"):
        replace(_plan(), projected_max_nodes=31).validate()
    with pytest.raises(ValueError, match="Plan IR is invalid"):
        replace(_plan(), performance_claimed=True).validate()


def test_root_projection_instance_tamper_fails_closed() -> None:
    plan = _plan()
    instance = _instance(plan)

    with pytest.raises(ValueError, match="Instance IR differs"):
        replace(instance, objective_matrix_hash="8" * 64).validate(plan=plan)
