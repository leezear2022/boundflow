"""IR-3C contracts for typed Task IR v1 and Schedule linkage."""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from boundflow.frontends.plain_crown_bound_ir import build_plain_crown_bound_ir
from boundflow.ir.bound import BoundOpKind
from boundflow.ir.bound_rewrite import rewrite_plain_crown_structured_regions
from boundflow.ir.schedule import (
    LaunchAction,
    lower_plan_instance_to_reference_schedule,
)
from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.ir.task_v1 import (
    TaskIRKind,
    TaskMemoryAccess,
    lower_plan_instance_to_task_ir,
)
from boundflow.planner.plan_ir_selector import select_plan_instance
from boundflow.runtime.bound_ir_interpreter import (
    PlainCrownBoundIRSession,
    execute_plain_crown_bound_ir,
)
from boundflow.runtime.crown_ibp import _forward_ibp_trace_mlp
from boundflow.runtime.task_ir_executor import (
    execute_task_ir_reference,
    execute_task_ir_semantics,
)
from boundflow.runtime.task_backend_dispatch import (
    PyTorchReferenceTaskBackend,
    build_backend_dispatch_key,
)
from boundflow.runtime.task_executor import InputSpec
from scripts.run_plan_ir_v1_reference_artifact import (
    build_reference_smoke_inputs,
    build_reference_smoke_template,
    build_reference_smoke_workload,
)


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
    assert all(
        tuple(item.value_id for item in task.input_constraints) == task.input_value_ids
        and tuple(item.value_id for item in task.output_constraints)
        == task.output_value_ids
        for task in task_module.tasks
    )
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
    constrained = task_module.tasks[0]
    input_constraint = constrained.input_constraints[0]
    wrong_constraint = replace(
        input_constraint,
        tensor_type=replace(
            input_constraint.tensor_type,
            shape=(999, *input_constraint.tensor_type.shape[1:]),
        ),
    )
    wrong_constraints = replace(
        constrained,
        input_constraints=(wrong_constraint, *constrained.input_constraints[1:]),
    )
    with pytest.raises(ValueError, match="input shape constraints"):
        replace(
            task_module,
            tasks=(wrong_constraints, *task_module.tasks[1:]),
        ).validate(
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


def test_task_ir_v1_semantic_execution_matches_whole_bound_interpreter() -> None:
    workload = build_reference_smoke_workload()
    module, template, instance, task_module, schedule = _task_fixture()
    assert module == workload.bound_module
    result, trace = execute_task_ir_semantics(
        task_module,
        schedule,
        bound_module=module,
        template=template,
        instance=instance,
        legacy_task_module=workload.task_module,
        input_spec=workload.input_spec,
        relu_pre=workload.relu_pre,
    )
    expected = execute_plain_crown_bound_ir(
        module,
        task_module=workload.task_module,
        input_spec=workload.input_spec,
        relu_pre=workload.relu_pre,
    )
    torch.testing.assert_close(result.lower, expected.lower)
    torch.testing.assert_close(result.upper, expected.upper)
    assert len(trace.events) == len(task_module.tasks)
    assert all(event.output_value_hashes for event in trace.events)
    assert (
        trace
        == execute_task_ir_semantics(
            task_module,
            schedule,
            bound_module=module,
            template=template,
            instance=instance,
            legacy_task_module=workload.task_module,
            input_spec=workload.input_spec,
            relu_pre=workload.relu_pre,
        )[1]
    )


def test_task_ir_v1_semantic_session_rejects_skip_and_early_result() -> None:
    workload = build_reference_smoke_workload()
    session = PlainCrownBoundIRSession(
        workload.bound_module,
        task_module=workload.task_module,
        input_spec=workload.input_spec,
        relu_pre=workload.relu_pre,
    )
    with pytest.raises(ValueError, match="before task completion"):
        session.result()
    op_ids = tuple(op.op_id for op in workload.bound_module.graph.ops)
    assert len(op_ids) > 1
    with pytest.raises(ValueError, match="non-contiguous"):
        session.execute_task(
            (op_ids[1],),
            output_value_ids=workload.bound_module.graph.outputs,
        )


def _semantic_case(case: str) -> tuple[BFTaskModule, InputSpec]:
    torch.manual_seed({"mlp": 101, "residual": 102, "concat": 103, "cnn": 104}[case])
    if case == "mlp":
        ops = [
            TaskOp("linear", "linear1", ["input", "W1", "b1"], ["h1"]),
            TaskOp("relu", "relu1", ["h1"], ["r1"]),
            TaskOp("linear", "linear2", ["r1", "W2", "b2"], ["out"]),
        ]
        params = {
            "W1": torch.randn(5, 4),
            "b1": torch.randn(5),
            "W2": torch.randn(3, 5),
            "b2": torch.randn(3),
        }
        spec = InputSpec.linf(value_name="input", center=torch.randn(2, 4), eps=0.2)
    elif case == "residual":
        ops = [
            TaskOp("linear", "linear1", ["input", "W1", "b1"], ["h1"]),
            TaskOp("relu", "relu1", ["h1"], ["r1"]),
            TaskOp("linear", "linear2", ["r1", "W2", "b2"], ["h2"]),
            TaskOp("add", "residual", ["input", "h2"], ["sum"]),
            TaskOp("relu", "relu2", ["sum"], ["r2"]),
            TaskOp("linear", "linear3", ["r2", "W3", "b3"], ["out"]),
        ]
        params = {
            "W1": torch.randn(5, 4),
            "b1": torch.randn(5),
            "W2": torch.randn(4, 5),
            "b2": torch.randn(4),
            "W3": torch.randn(3, 4),
            "b3": torch.randn(3),
        }
        spec = InputSpec.l2(value_name="input", center=torch.randn(2, 4), eps=0.15)
    elif case == "concat":
        ops = [
            TaskOp("linear", "left", ["input", "W1", "b1"], ["h1"]),
            TaskOp("relu", "left_relu", ["h1"], ["r1"]),
            TaskOp("linear", "right", ["input", "W2", "b2"], ["h2"]),
            TaskOp("relu", "right_relu", ["h2"], ["r2"]),
            TaskOp(
                "concat",
                "join",
                ["r1", "r2"],
                ["joined"],
                attrs={"axis": 1},
            ),
            TaskOp("linear", "output", ["joined", "W3", "b3"], ["out"]),
        ]
        params = {
            "W1": torch.randn(3, 4),
            "b1": torch.randn(3),
            "W2": torch.randn(2, 4),
            "b2": torch.randn(2),
            "W3": torch.randn(2, 5),
            "b3": torch.randn(2),
        }
        spec = InputSpec.linf(value_name="input", center=torch.randn(2, 4), eps=0.1)
    else:
        ops = [
            TaskOp(
                "conv2d",
                "conv",
                ["input", "Wc", "bc"],
                ["conv_out"],
                attrs={
                    "stride": 1,
                    "padding": 0,
                    "dilation": 1,
                    "groups": 1,
                },
            ),
            TaskOp("relu", "relu", ["conv_out"], ["relu_out"]),
            TaskOp(
                "flatten",
                "flatten",
                ["relu_out"],
                ["flat"],
                attrs={"start_dim": 1, "end_dim": -1},
            ),
            TaskOp("linear", "linear", ["flat", "Wl", "bl"], ["out"]),
        ]
        params = {
            "Wc": torch.randn(2, 1, 2, 2),
            "bc": torch.randn(2),
            "Wl": torch.randn(3, 8),
            "bl": torch.randn(3),
        }
        spec = InputSpec.box(
            value_name="input",
            lower=torch.full((1, 1, 3, 3), -0.4),
            upper=torch.full((1, 1, 3, 3), 0.6),
        )
    return (
        BFTaskModule(
            tasks=[
                BoundTask(
                    task_id=f"semantic-{case}",
                    kind=TaskKind.INTERVAL_IBP,
                    ops=ops,
                    input_values=["input"],
                    output_values=["out"],
                )
            ],
            entry_task_id=f"semantic-{case}",
            bindings={"params": params},
        ),
        spec,
    )


@pytest.mark.parametrize("case", ["mlp", "cnn", "residual", "concat"])
def test_task_ir_v1_semantic_partitions_match_graph_families(case: str) -> None:
    legacy_module, input_spec = _semantic_case(case)
    interval_env, relu_pre = _forward_ibp_trace_mlp(legacy_module, input_spec)
    bound_module = build_plain_crown_bound_ir(
        legacy_module,
        input_spec,
        interval_env=interval_env,
        relu_pre=relu_pre,
    ).module
    template = build_reference_smoke_template(bound_module)
    instance = select_plan_instance(
        template,
        bound_module=bound_module,
        query_bucket_id=f"semantic-{case}",
        available_memory_bytes=1 << 30,
        memory_budget_bytes=1 << 30,
    )
    task_module = lower_plan_instance_to_task_ir(
        bound_module,
        template=template,
        instance=instance,
    )
    schedule = lower_plan_instance_to_reference_schedule(
        bound_module,
        template=template,
        instance=instance,
        query_ids=("query:0", "query:1"),
    )
    actual, trace = execute_task_ir_semantics(
        task_module,
        schedule,
        bound_module=bound_module,
        template=template,
        instance=instance,
        legacy_task_module=legacy_module,
        input_spec=input_spec,
        relu_pre=relu_pre,
    )
    expected = execute_plain_crown_bound_ir(
        bound_module,
        task_module=legacy_module,
        input_spec=input_spec,
        relu_pre=relu_pre,
    )
    torch.testing.assert_close(actual.lower, expected.lower, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(actual.upper, expected.upper, atol=2e-6, rtol=2e-6)
    assert tuple(event.op_ids for event in trace.events) == tuple(
        tuple(op_ref.op_id for op_ref in task.op_refs) for task in task_module.tasks
    )
    assert all(event.output_value_hashes for event in trace.events)


def test_task_ir_v1_semantics_execute_explicit_materialization_ops() -> None:
    legacy_module, input_spec = _semantic_case("mlp")
    interval_env, relu_pre = _forward_ibp_trace_mlp(legacy_module, input_spec)
    dense_module = build_plain_crown_bound_ir(
        legacy_module,
        input_spec,
        interval_env=interval_env,
        relu_pre=relu_pre,
    ).module
    bound_module = rewrite_plain_crown_structured_regions(dense_module)
    template = build_reference_smoke_template(bound_module)
    instance = select_plan_instance(
        template,
        bound_module=bound_module,
        query_bucket_id="semantic-structured-mlp",
        available_memory_bytes=1 << 30,
        memory_budget_bytes=1 << 30,
    )
    task_module = lower_plan_instance_to_task_ir(
        bound_module,
        template=template,
        instance=instance,
    )
    schedule = lower_plan_instance_to_reference_schedule(
        bound_module,
        template=template,
        instance=instance,
        query_ids=("query:0",),
    )
    actual, trace = execute_task_ir_semantics(
        task_module,
        schedule,
        bound_module=bound_module,
        template=template,
        instance=instance,
        legacy_task_module=legacy_module,
        input_spec=input_spec,
        relu_pre=relu_pre,
    )
    expected = execute_plain_crown_bound_ir(
        bound_module,
        task_module=legacy_module,
        input_spec=input_spec,
        relu_pre=relu_pre,
    )
    torch.testing.assert_close(actual.lower, expected.lower, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(actual.upper, expected.upper, atol=2e-6, rtol=2e-6)
    materialize_ids = {
        op.op_id for op in bound_module.graph.ops if op.kind == BoundOpKind.MATERIALIZE
    }
    assert materialize_ids
    assert materialize_ids.issubset(
        {op_id for event in trace.events for op_id in event.op_ids}
    )


def test_task_schedule_reference_path_does_not_import_legacy_scheduler() -> None:
    for path in (
        Path("boundflow/runtime/task_ir_executor.py"),
        Path("boundflow/runtime/schedule_ir_executor.py"),
    ):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        assert not any(module.endswith("runtime.scheduler") for module in imported)


def test_typed_backend_dispatch_key_cache_and_capability_rejection() -> None:
    workload = build_reference_smoke_workload()
    module, template, instance, task_module, schedule = _task_fixture()
    backend = PyTorchReferenceTaskBackend()
    first_result, first_trace = execute_task_ir_semantics(
        task_module,
        schedule,
        bound_module=module,
        template=template,
        instance=instance,
        legacy_task_module=workload.task_module,
        input_spec=workload.input_spec,
        relu_pre=workload.relu_pre,
        backend=backend,
    )
    second_result, second_trace = execute_task_ir_semantics(
        task_module,
        schedule,
        bound_module=module,
        template=template,
        instance=instance,
        legacy_task_module=workload.task_module,
        input_spec=workload.input_spec,
        relu_pre=workload.relu_pre,
        backend=backend,
    )
    torch.testing.assert_close(first_result.lower, second_result.lower)
    torch.testing.assert_close(first_result.upper, second_result.upper)
    assert first_trace == second_trace
    assert backend.cache_misses == len(task_module.tasks)
    assert backend.cache_hits == len(task_module.tasks)
    assert len({event.backend_dispatch_key for event in first_trace.events}) == len(
        task_module.tasks
    )

    task = task_module.tasks[0]
    key = build_backend_dispatch_key(
        task,
        task_module,
        bound_module=module,
        template=template,
        instance=instance,
    )
    stale_key = replace(key, bound_module_hash="0" * 64)
    session = PlainCrownBoundIRSession(
        module,
        task_module=workload.task_module,
        input_spec=workload.input_spec,
        relu_pre=workload.relu_pre,
    )
    with pytest.raises(ValueError, match="does not match typed task"):
        backend.dispatch(task, stale_key, session=session, template=template)

    candidate_index = next(
        index
        for index, candidate in enumerate(template.backend_candidates)
        if candidate.candidate_id == task.backend.backend_candidate_id
    )
    wrong_candidate = replace(
        template.backend_candidates[candidate_index],
        static_legal=False,
        rejection_reasons=("injected_static_rejection",),
    )
    wrong_template = replace(
        template,
        backend_candidates=(
            *template.backend_candidates[:candidate_index],
            wrong_candidate,
            *template.backend_candidates[candidate_index + 1 :],
        ),
    )
    wrong_key = replace(
        key,
        plan_template_hash=wrong_template.stable_hash(bound_module=module),
    )
    with pytest.raises(ValueError, match="rejects selected capability"):
        backend.dispatch(task, wrong_key, session=session, template=wrong_template)
