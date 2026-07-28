"""IR-3A contracts for Schedule IR v1 schema, lowering, and verifier."""

from __future__ import annotations

from dataclasses import replace

import pytest

from boundflow.ir.schedule import (
    AllocateAction,
    EmitResultAction,
    LaunchAction,
    lower_plan_instance_to_reference_schedule,
)
from boundflow.planner.plan_ir_selector import select_plan_instance
from scripts.run_plan_ir_v1_reference_artifact import (
    build_reference_smoke_inputs,
)


def _schedule_fixture():
    module, template = build_reference_smoke_inputs()
    instance = select_plan_instance(
        template,
        bound_module=module,
        query_bucket_id="schedule-ir-v1",
        available_memory_bytes=1 << 30,
        memory_budget_bytes=1 << 30,
    )
    schedule = lower_plan_instance_to_reference_schedule(
        module,
        template=template,
        instance=instance,
        query_ids=("query:0", "query:1"),
    )
    return module, template, instance, schedule


def test_schedule_ir_v1_lowering_is_deterministic_and_fully_accounted() -> None:
    module, template, instance, schedule = _schedule_fixture()
    repeated = lower_plan_instance_to_reference_schedule(
        module,
        template=template,
        instance=instance,
        query_ids=("query:0", "query:1"),
    )
    assert schedule == repeated
    assert schedule.canonical_json(
        bound_module=module,
        template=template,
        instance=instance,
    ) == repeated.canonical_json(
        bound_module=module,
        template=template,
        instance=instance,
    )
    assert schedule.stable_hash(
        bound_module=module,
        template=template,
        instance=instance,
    ) == repeated.stable_hash(
        bound_module=module,
        template=template,
        instance=instance,
    )
    assert sum(isinstance(action, LaunchAction) for action in schedule.actions) == len(
        instance.region_decisions
    )
    emit = next(
        action for action in schedule.actions if isinstance(action, EmitResultAction)
    )
    assert emit.query_ids == schedule.query_ids
    assert set(emit.output_value_ids) == set(module.graph.outputs)


def test_schedule_ir_v1_rejects_use_before_def_and_wrong_arena_ledger() -> None:
    module, template, instance, schedule = _schedule_fixture()
    launch_index = next(
        index
        for index, action in enumerate(schedule.actions)
        if isinstance(action, LaunchAction)
        and any(
            value_id not in module.graph.inputs for value_id in action.input_value_ids
        )
    )
    broken_order = (
        schedule.actions[launch_index],
        *schedule.actions[:launch_index],
        *schedule.actions[launch_index + 1 :],
    )
    with pytest.raises(ValueError, match="use-before-def|allocation order"):
        replace(schedule, actions=broken_order).validate(
            bound_module=module,
            template=template,
            instance=instance,
        )

    allocate_index = next(
        index
        for index, action in enumerate(schedule.actions)
        if isinstance(action, AllocateAction)
    )
    allocate = schedule.actions[allocate_index]
    assert isinstance(allocate, AllocateAction)
    wrong_allocate = replace(allocate, size_bytes=allocate.size_bytes + 16)
    wrong_actions = (
        *schedule.actions[:allocate_index],
        wrong_allocate,
        *schedule.actions[allocate_index + 1 :],
    )
    with pytest.raises(ValueError, match="wrong arena size"):
        replace(schedule, actions=wrong_actions).validate(
            bound_module=module,
            template=template,
            instance=instance,
        )


def test_schedule_ir_v1_rejects_dropped_launch_and_query() -> None:
    module, template, instance, schedule = _schedule_fixture()
    missing_launch = tuple(
        action for action in schedule.actions if not isinstance(action, LaunchAction)
    )
    with pytest.raises(ValueError, match="unavailable outputs|launch every"):
        replace(schedule, actions=missing_launch).validate(
            bound_module=module,
            template=template,
            instance=instance,
        )
    emit_index = next(
        index
        for index, action in enumerate(schedule.actions)
        if isinstance(action, EmitResultAction)
    )
    emit = schedule.actions[emit_index]
    assert isinstance(emit, EmitResultAction)
    wrong_emit = replace(emit, query_ids=("query:0",))
    wrong_actions = (
        *schedule.actions[:emit_index],
        wrong_emit,
        *schedule.actions[emit_index + 1 :],
    )
    with pytest.raises(ValueError, match="query accounting"):
        replace(schedule, actions=wrong_actions).validate(
            bound_module=module,
            template=template,
            instance=instance,
        )
