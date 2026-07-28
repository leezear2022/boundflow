"""IR-3C contracts for typed Task IR v1 and Schedule linkage."""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest

from boundflow.ir.schedule import (
    LaunchAction,
    lower_plan_instance_to_reference_schedule,
)
from boundflow.ir.task_v1 import (
    TaskIRKind,
    TaskMemoryAccess,
    lower_plan_instance_to_task_ir,
)
from boundflow.planner.plan_ir_selector import select_plan_instance
from boundflow.runtime.task_ir_executor import execute_task_ir_reference
from scripts.run_plan_ir_v1_reference_artifact import build_reference_smoke_inputs


def _task_fixture():
    module, template = build_reference_smoke_inputs()
    instance = select_plan_instance(
        template,
        bound_module=module,
        query_bucket_id="task-ir-v1",
        available_memory_bytes=1 << 30,
        memory_budget_bytes=1 << 30,
    )
    task_module = lower_plan_instance_to_task_ir(
        module,
        template=template,
        instance=instance,
    )
    schedule = lower_plan_instance_to_reference_schedule(
        module,
        template=template,
        instance=instance,
        query_ids=("query:0", "query:1"),
    )
    return module, template, instance, task_module, schedule


def test_task_ir_v1_lowering_is_deterministic_typed_and_schedule_linked() -> None:
    module, template, instance, task_module, schedule = _task_fixture()
    repeated = lower_plan_instance_to_task_ir(
        module,
        template=template,
        instance=instance,
    )
    assert task_module == repeated
    assert task_module.stable_hash(
        bound_module=module,
        template=template,
        instance=instance,
    ) == repeated.stable_hash(
        bound_module=module,
        template=template,
        instance=instance,
    )
    assert {task.kind for task in task_module.tasks} >= {
        TaskIRKind.BOUND_BINDING,
        TaskIRKind.CONCRETIZATION,
    }
    assert all(
        effect.access in {TaskMemoryAccess.READ, TaskMemoryAccess.WRITE}
        for task in task_module.tasks
        for effect in task.memory_effects
    )
    assert any(task.parameter_value_ids for task in task_module.tasks)
    task_module.validate_schedule_linkage(
        schedule,
        bound_module=module,
        template=template,
        instance=instance,
    )


def test_task_ir_v1_rejects_missing_parameter_and_schedule_mismatch() -> None:
    module, template, instance, task_module, schedule = _task_fixture()
    parameter_index = next(
        index
        for index, task in enumerate(task_module.tasks)
        if task.parameter_value_ids
    )
    task = task_module.tasks[parameter_index]
    broken_task = replace(task, parameter_value_ids=task.parameter_value_ids[:-1])
    broken_tasks = (
        *task_module.tasks[:parameter_index],
        broken_task,
        *task_module.tasks[parameter_index + 1 :],
    )
    with pytest.raises(ValueError, match="parameter dependencies"):
        replace(task_module, tasks=broken_tasks).validate(
            bound_module=module,
            template=template,
            instance=instance,
        )

    launch_index = next(
        index
        for index, action in enumerate(schedule.actions)
        if isinstance(action, LaunchAction)
    )
    launch = schedule.actions[launch_index]
    assert isinstance(launch, LaunchAction)
    changed = replace(launch, task_id="task:wrong")
    actions = (
        *schedule.actions[:launch_index],
        changed,
        *schedule.actions[launch_index + 1 :],
    )
    broken_schedule = replace(schedule, actions=actions)
    with pytest.raises(ValueError, match="launch differs|launch sets"):
        task_module.validate_schedule_linkage(
            broken_schedule,
            bound_module=module,
            template=template,
            instance=instance,
        )


def test_task_ir_v1_core_has_no_any_dict_runtime_or_legacy_task_dependency() -> None:
    source = Path("boundflow/ir/task_v1.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "Any" not in names
    assert "Dict" not in names
    assert not any("runtime" in module for module in imported)
    assert not any(module.endswith(".task") for module in imported)


def test_task_ir_v1_reference_dispatch_trace_is_deterministic() -> None:
    module, template, instance, task_module, schedule = _task_fixture()
    first = execute_task_ir_reference(
        task_module,
        schedule,
        bound_module=module,
        template=template,
        instance=instance,
    )
    second = execute_task_ir_reference(
        task_module,
        schedule,
        bound_module=module,
        template=template,
        instance=instance,
    )
    assert first == second
    assert len(first.events) == len(task_module.tasks)
    assert tuple(event.task_id for event in first.events) == tuple(
        task.task_id for task in task_module.tasks
    )
    assert first.stable_hash() == second.stable_hash()
