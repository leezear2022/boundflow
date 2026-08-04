"""Typed production verifier Plan/Task/Schedule contracts."""

# pylint: disable=missing-function-docstring

from dataclasses import replace

import pytest

from boundflow.ir.production_verifier import (
    NativeProductionVerifierPlanIR,
    NativeProductionVerifierTaskKind,
    lower_native_production_verifier_ir,
)


def _plan() -> NativeProductionVerifierPlanIR:
    return NativeProductionVerifierPlanIR(
        plan_id="production-verifier-toy:eval:0000",
        node_ids=("node:0",),
        node_split_state_hashes=("1" * 64,),
        parent_selected_state_hashes=(None,),
        state_scope_hash="2" * 64,
        primal_graph_hash="3" * 64,
        input_region_hash="4" * 64,
        objective_hash="5" * 64,
        optimizer_policy_hash="6" * 64,
        intermediate_bounds_hash="7" * 64,
        intermediate_bound_source="local_forward",
        optimizer_ir_hashes=(
            ("optimizer_plan_hash", "8" * 64),
            ("optimizer_schedule_hash", "9" * 64),
            ("optimizer_task_module_hash", "a" * 64),
        ),
    )


def test_production_verifier_plan_lowers_exact_task_schedule_order() -> None:
    plan = _plan()
    task_ir, schedule = lower_native_production_verifier_ir(plan)

    assert tuple(task.kind for task in task_ir.tasks) == tuple(
        NativeProductionVerifierTaskKind
    )
    assert tuple(action.task_id for action in schedule.actions) == tuple(
        task.task_id for task in task_ir.tasks
    )
    assert plan.to_dict()["audit_hash_chain_constructed"] is False
    assert plan.to_dict()["selected_native_reexecution"] is False
    schedule.validate(plan=plan, task_ir=task_ir)


def test_production_verifier_ir_rejects_identity_and_order_tampering() -> None:
    plan = _plan()
    task_ir, schedule = lower_native_production_verifier_ir(plan)

    with pytest.raises(ValueError, match="Plan IR"):
        replace(plan, objective_hash="wrong").validate()
    with pytest.raises(ValueError, match="mixes root and child"):
        replace(
            plan,
            node_ids=("node:0", "node:1"),
            node_split_state_hashes=("1" * 64, "b" * 64),
            parent_selected_state_hashes=(None, "c" * 64),
        ).validate()
    actions = list(schedule.actions)
    actions[1] = replace(actions[1], sequence=3)
    with pytest.raises(ValueError, match="Schedule/Task order"):
        replace(schedule, actions=tuple(actions)).validate(plan=plan, task_ir=task_ir)
