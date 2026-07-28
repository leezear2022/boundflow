"""IR-3A contracts for Schedule IR v1 schema, lowering, and verifier."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from boundflow.frontends.plain_crown_bound_ir import build_plain_crown_bound_ir
from boundflow.ir.plan import StateAction, StateCandidate, StateValidity
from boundflow.ir.schedule import (
    AllocateAction,
    BatchLoopAction,
    EmitResultAction,
    FallbackAction,
    LaunchAction,
    RecordEventAction,
    RetryAction,
    StateLoadAction,
    WaitEventAction,
    lower_plan_instance_to_reference_schedule,
)
from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.planner.plan_ir_selector import select_plan_instance
from boundflow.runtime.bound_ir_interpreter import execute_plain_crown_bound_ir
from boundflow.runtime.crown_ibp import _forward_ibp_trace_mlp
from boundflow.runtime.schedule_ir_executor import (
    ScheduleOutOfMemoryError,
    ScheduleRetryExhausted,
    execute_schedule_reference,
    execute_schedule_with_bound_reference,
    replay_schedule_trace,
)
from boundflow.runtime.task_executor import InputSpec
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


def test_schedule_ir_v1_reference_trace_is_deterministic_and_replayable() -> None:
    module, template, instance, schedule = _schedule_fixture()
    first = execute_schedule_reference(
        schedule,
        bound_module=module,
        template=template,
        instance=instance,
    )
    second = execute_schedule_reference(
        schedule,
        bound_module=module,
        template=template,
        instance=instance,
    )
    assert first == second
    assert first.peak_memory_bytes == instance.cost_summary.predicted_peak_bytes
    assert first.emitted_query_ids == schedule.query_ids
    assert (
        replay_schedule_trace(
            first.canonical_json(),
            schedule,
            bound_module=module,
            template=template,
            instance=instance,
        )
        == first
    )
    with pytest.raises(ValueError, match="does not match"):
        replay_schedule_trace(
            first.canonical_json().replace("query:0", "query:tampered"),
            schedule,
            bound_module=module,
            template=template,
            instance=instance,
        )


def test_schedule_ir_v1_batch_loop_rejects_query_loss() -> None:
    module, template, instance, schedule = _schedule_fixture()
    batch_index = next(
        index
        for index, action in enumerate(schedule.actions)
        if isinstance(action, BatchLoopAction)
    )
    batch = schedule.actions[batch_index]
    assert isinstance(batch, BatchLoopAction)
    first_slice = batch.slices[0]
    broken = replace(
        batch,
        slices=(
            replace(first_slice, query_ids=first_slice.query_ids[:-1]),
            *batch.slices[1:],
        ),
    )
    actions = (
        *schedule.actions[:batch_index],
        broken,
        *schedule.actions[batch_index + 1 :],
    )
    with pytest.raises(ValueError, match="loses, duplicates, or reorders"):
        replace(schedule, actions=actions).validate(
            bound_module=module,
            template=template,
            instance=instance,
        )


class _OomSelectedBackend:
    def __init__(self, selected_backend_id: str) -> None:
        self.selected_backend_id = selected_backend_id

    def launch(
        self,
        action: LaunchAction,
        *,
        backend_candidate_id: str,
        attempt: int,
    ) -> None:
        del action, attempt
        if backend_candidate_id == self.selected_backend_id:
            raise ScheduleOutOfMemoryError("injected reference OOM")


def test_schedule_ir_v1_bounded_oom_retry_uses_declared_fallback() -> None:
    module, base_template = build_reference_smoke_inputs()
    selected_backend = base_template.backend_candidates[0]
    fallback_backend = replace(
        selected_backend,
        candidate_id="backend:reference-fallback",
        cost=replace(
            selected_backend.cost,
            predicted_latency_ms=selected_backend.cost.predicted_latency_ms + 10.0,
        ),
    )
    template = replace(
        base_template,
        backend_candidates=(*base_template.backend_candidates, fallback_backend),
    )
    instance = select_plan_instance(
        template,
        bound_module=module,
        query_bucket_id="schedule-retry",
        available_memory_bytes=1 << 30,
        memory_budget_bytes=1 << 30,
    )
    schedule = lower_plan_instance_to_reference_schedule(
        module,
        template=template,
        instance=instance,
        query_ids=("query:0",),
    )
    launch_index = next(
        index
        for index, action in enumerate(schedule.actions)
        if isinstance(action, LaunchAction)
        and action.backend_candidate_id == selected_backend.candidate_id
    )
    launch = schedule.actions[launch_index]
    assert isinstance(launch, LaunchAction)
    retry = RetryAction(
        action_id="retry:reference",
        launch_action_id=launch.action_id,
        fallback_action_ids=("fallback:reference",),
        max_attempts=2,
        retry_on=("oom",),
    )
    fallback = FallbackAction(
        action_id="fallback:reference",
        retry_action_id=retry.action_id,
        backend_candidate_id=fallback_backend.candidate_id,
        reason="selected_backend_oom",
    )
    actions = (
        *schedule.actions[:launch_index],
        retry,
        fallback,
        *schedule.actions[launch_index:],
    )
    retry_schedule = replace(schedule, actions=actions)
    trace = execute_schedule_reference(
        retry_schedule,
        bound_module=module,
        template=template,
        instance=instance,
        driver=_OomSelectedBackend(selected_backend.candidate_id),
    )
    attempts = [
        event
        for event in trace.events
        if event.action_id == launch.action_id and event.event_kind == "launch_attempt"
    ]
    assert [
        dict((field.key, field.value) for field in event.fields)["outcome"]
        for event in attempts
    ] == [
        "oom",
        "success",
    ]
    assert trace.emitted_query_ids == retry_schedule.query_ids

    with pytest.raises(ScheduleRetryExhausted) as exhausted:
        execute_schedule_reference(
            retry_schedule,
            bound_module=module,
            template=template,
            instance=instance,
            driver=_AlwaysOom(),
        )
    assert exhausted.value.attempts == 2


class _AlwaysOom:
    def launch(
        self,
        action: LaunchAction,
        *,
        backend_candidate_id: str,
        attempt: int,
    ) -> None:
        del action, backend_candidate_id, attempt
        raise ScheduleOutOfMemoryError("injected exhaustion")


def test_schedule_ir_v1_custom_stream_requires_record_wait_happens_before() -> None:
    module, template, instance, schedule = _schedule_fixture()
    launches = [
        (index, action)
        for index, action in enumerate(schedule.actions)
        if isinstance(action, LaunchAction)
    ]
    producer_position = next(
        position
        for position, (_index, action) in enumerate(launches[:-1])
        if set(action.input_value_ids).issubset(set(module.graph.inputs))
    )
    producer_index, producer = launches[producer_position]
    later = launches[producer_position + 1 :]
    consumer_index, _consumer = next(
        (index, action)
        for index, action in later
        if set(action.input_value_ids) & set(producer.output_value_ids)
    )
    changed = list(schedule.actions)
    changed[producer_index] = replace(producer, stream_id="producer")
    for index, action in later:
        changed[index] = replace(action, stream_id="consumer")
    with pytest.raises(ValueError, match="lacks event wait"):
        replace(schedule, actions=tuple(changed)).validate(
            bound_module=module,
            template=template,
            instance=instance,
        )

    changed.insert(
        producer_index + 1,
        RecordEventAction("event:record", "event:producer-done", "producer"),
    )
    changed.insert(
        consumer_index + 1,
        WaitEventAction("event:wait", "event:producer-done", "consumer"),
    )
    synchronized = replace(schedule, actions=tuple(changed))
    synchronized.validate(
        bound_module=module,
        template=template,
        instance=instance,
    )


def test_schedule_ir_v1_lowers_exact_state_reuse_to_state_load() -> None:
    module, template = build_reference_smoke_inputs()
    source = next(
        value for value in module.graph.values if value.state_version is not None
    )
    reuse = StateCandidate(
        candidate_id="state:reuse:schedule",
        state_id="schedule-state",
        source_value_id=source.value_id,
        action=StateAction.REUSE,
        state_version=source.state_version or "",
        size_bytes=32,
        static_legal=True,
        rejection_reasons=(),
        cost=replace(
            template.region_candidates[0].cost,
            predicted_latency_ms=0.0,
            predicted_peak_bytes=32,
        ),
    )
    stateful_template = replace(template, state_candidates=(reuse,))
    validity = StateValidity(
        state_id=reuse.state_id,
        source_value_id=reuse.source_value_id,
        state_version=reuse.state_version,
        valid=True,
    )
    instance = select_plan_instance(
        stateful_template,
        bound_module=module,
        query_bucket_id="schedule-state",
        available_memory_bytes=1 << 30,
        memory_budget_bytes=1 << 30,
        state_validities=(validity,),
    )
    schedule = lower_plan_instance_to_reference_schedule(
        module,
        template=stateful_template,
        instance=instance,
        query_ids=("query:0",),
    )
    loads = [
        action for action in schedule.actions if isinstance(action, StateLoadAction)
    ]
    assert len(loads) == 1
    assert loads[0].state_version == validity.state_version


def test_schedule_ir_v1_bound_reference_e2e_matches_direct_interpreter() -> None:
    module, template, instance, schedule = _schedule_fixture()
    task_module = BFTaskModule(
        tasks=[
            BoundTask(
                task_id="plan-ir-reference-smoke",
                kind=TaskKind.INTERVAL_IBP,
                ops=[
                    TaskOp(
                        "linear",
                        "linear1",
                        ["input", "weight", "bias"],
                        ["output"],
                    )
                ],
                input_values=["input"],
                output_values=["output"],
            )
        ],
        entry_task_id="plan-ir-reference-smoke",
        bindings={
            "params": {
                "weight": torch.tensor([[1.0, -0.5], [0.25, 0.75]]),
                "bias": torch.tensor([0.1, -0.2]),
            }
        },
    )
    input_spec = InputSpec.linf(value_name="input", center=torch.zeros(2, 2), eps=0.1)
    interval_env, relu_pre = _forward_ibp_trace_mlp(task_module, input_spec)
    rebuilt = build_plain_crown_bound_ir(
        task_module,
        input_spec,
        interval_env=interval_env,
        relu_pre=relu_pre,
    ).module
    assert rebuilt.stable_hash() == module.stable_hash()
    direct = execute_plain_crown_bound_ir(
        module,
        task_module=task_module,
        input_spec=input_spec,
        relu_pre=relu_pre,
    )
    scheduled, trace = execute_schedule_with_bound_reference(
        schedule,
        bound_module=module,
        template=template,
        instance=instance,
        task_module=task_module,
        input_spec=input_spec,
        relu_pre=relu_pre,
    )
    torch.testing.assert_close(scheduled.lower, direct.lower)
    torch.testing.assert_close(scheduled.upper, direct.upper)
    assert trace.emitted_query_ids == schedule.query_ids
