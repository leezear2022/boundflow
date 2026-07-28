"""Synchronous reference executor and deterministic trace for Schedule IR v1."""

# The executor deliberately mirrors the explicit action ledger.
# pylint: disable=too-few-public-methods,too-many-arguments,too-many-branches,too-many-locals,too-many-statements,missing-class-docstring,missing-function-docstring

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping, Optional, Protocol, Tuple

import torch

from ..domains.interval import IntervalState
from ..ir.bound import BFBoundModule
from ..ir.plan import PlanInstance, PlanTemplate
from ..ir.schedule import (
    AllocateAction,
    BatchLoopAction,
    CheckBudgetAction,
    EmitResultAction,
    FallbackAction,
    FreeAction,
    LaunchAction,
    MaterializeAction,
    RecordEventAction,
    RequestReplanAction,
    RetryAction,
    ScheduleAction,
    ScheduleModule,
    StateInvalidateAction,
    StateLoadAction,
    StateStoreAction,
    TransferAction,
    WaitEventAction,
)
from ..ir.task import BFTaskModule
from .bound_ir_interpreter import execute_plain_crown_bound_ir
from .task_executor import InputSpec


class ScheduleOutOfMemoryError(RuntimeError):
    """Reference backend signal allowed to trigger a declared retry."""


class ScheduleRetryExhausted(RuntimeError):
    """Raised after the exact bounded fallback ladder is exhausted."""

    def __init__(self, launch_action_id: str, attempts: int) -> None:
        self.launch_action_id = launch_action_id
        self.attempts = attempts
        super().__init__(
            f"Schedule IR retry exhausted: launch={launch_action_id} "
            f"attempts={attempts}"
        )


class ScheduleReferenceDriver(Protocol):
    """Backend-neutral callback used by the reference action executor."""

    def launch(
        self,
        action: LaunchAction,
        *,
        backend_candidate_id: str,
        attempt: int,
    ) -> None:
        """Execute or simulate one launch attempt."""


class NoOpScheduleReferenceDriver:
    """Deterministic success driver for contract/replay tests."""

    def launch(
        self,
        action: LaunchAction,
        *,
        backend_candidate_id: str,
        attempt: int,
    ) -> None:
        del action, backend_candidate_id, attempt


@dataclass(frozen=True)
class ScheduleTraceField:
    key: str
    value: str

    def validate(self) -> None:
        if not self.key or not self.value:
            raise ValueError("schedule trace field must be non-empty")

    def to_dict(self) -> dict[str, str]:
        self.validate()
        return {"key": self.key, "value": self.value}


@dataclass(frozen=True)
class ScheduleTraceEvent:
    sequence: int
    action_id: str
    event_kind: str
    live_bytes: int
    fields: Tuple[ScheduleTraceField, ...] = ()

    def validate(self) -> None:
        if self.sequence < 0 or not self.action_id or not self.event_kind:
            raise ValueError("schedule trace event identity is invalid")
        if self.live_bytes < 0:
            raise ValueError("schedule trace live bytes are negative")
        for field in self.fields:
            field.validate()
        if len({field.key for field in self.fields}) != len(self.fields):
            raise ValueError("schedule trace event duplicates field keys")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "sequence": self.sequence,
            "action_id": self.action_id,
            "event_kind": self.event_kind,
            "live_bytes": self.live_bytes,
            "fields": [field.to_dict() for field in self.fields],
        }


@dataclass(frozen=True)
class ScheduleExecutionTrace:
    """Canonical observable execution ledger."""

    schedule_hash: str
    events: Tuple[ScheduleTraceEvent, ...]
    peak_memory_bytes: int
    emitted_query_ids: Tuple[str, ...]
    emitted_output_value_ids: Tuple[str, ...]

    def validate(self) -> None:
        if len(self.schedule_hash) != 64 or not self.events:
            raise ValueError("schedule trace identity/events are incomplete")
        for index, event in enumerate(self.events):
            event.validate()
            if event.sequence != index:
                raise ValueError("schedule trace sequence is not contiguous")
        if self.peak_memory_bytes < 0:
            raise ValueError("schedule trace peak memory is negative")
        if len(self.emitted_query_ids) != len(set(self.emitted_query_ids)):
            raise ValueError("schedule trace duplicates emitted queries")
        if not self.emitted_query_ids or not self.emitted_output_value_ids:
            raise ValueError("schedule trace omits emitted results")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": "boundflow.schedule-trace/v1",
            "schedule_hash": self.schedule_hash,
            "events": [event.to_dict() for event in self.events],
            "peak_memory_bytes": self.peak_memory_bytes,
            "emitted_query_ids": list(self.emitted_query_ids),
            "emitted_output_value_ids": list(self.emitted_output_value_ids),
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    def stable_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def execute_schedule_reference(
    schedule: ScheduleModule,
    *,
    bound_module: BFBoundModule,
    template: PlanTemplate,
    instance: PlanInstance,
    driver: Optional[ScheduleReferenceDriver] = None,
) -> ScheduleExecutionTrace:
    """Execute explicit actions and return an auditable deterministic trace."""

    schedule.validate(bound_module=bound_module, template=template, instance=instance)
    if driver is None:
        driver = NoOpScheduleReferenceDriver()
    schedule_hash = schedule.stable_hash(
        bound_module=bound_module, template=template, instance=instance
    )
    fallback_by_id = {
        action.action_id: action
        for action in schedule.actions
        if isinstance(action, FallbackAction)
    }
    retry_by_launch = {
        action.launch_action_id: action
        for action in schedule.actions
        if isinstance(action, RetryAction)
    }
    arena_sizes = {
        action.arena_id: action.size_bytes
        for action in schedule.actions
        if isinstance(action, AllocateAction)
    }
    live_bytes = 0
    peak_bytes = 0
    emitted_queries: Tuple[str, ...] = ()
    emitted_outputs: Tuple[str, ...] = ()
    events: list[ScheduleTraceEvent] = []

    def append_event(
        action: ScheduleAction,
        event_kind: str,
        fields: Tuple[ScheduleTraceField, ...] = (),
    ) -> None:
        events.append(
            ScheduleTraceEvent(
                sequence=len(events),
                action_id=action.action_id,
                event_kind=event_kind,
                live_bytes=live_bytes,
                fields=fields,
            )
        )

    for action in schedule.actions:
        if isinstance(action, AllocateAction):
            live_bytes += action.size_bytes
            peak_bytes = max(peak_bytes, live_bytes)
            append_event(
                action,
                action.kind.value,
                (ScheduleTraceField("arena_id", action.arena_id),),
            )
        elif isinstance(action, FreeAction):
            live_bytes -= arena_sizes[action.arena_id]
            append_event(
                action,
                action.kind.value,
                (ScheduleTraceField("arena_id", action.arena_id),),
            )
        elif isinstance(action, LaunchAction):
            retry = retry_by_launch.get(action.action_id)
            ladder = [action.backend_candidate_id]
            if retry is not None:
                ladder.extend(
                    fallback_by_id[fallback_id].backend_candidate_id
                    for fallback_id in retry.fallback_action_ids
                )
            succeeded = False
            for attempt, backend_candidate_id in enumerate(ladder, start=1):
                try:
                    driver.launch(
                        action,
                        backend_candidate_id=backend_candidate_id,
                        attempt=attempt,
                    )
                except ScheduleOutOfMemoryError:
                    append_event(
                        action,
                        "launch_attempt",
                        (
                            ScheduleTraceField("attempt", str(attempt)),
                            ScheduleTraceField(
                                "backend_candidate_id", backend_candidate_id
                            ),
                            ScheduleTraceField("outcome", "oom"),
                        ),
                    )
                    continue
                append_event(
                    action,
                    "launch_attempt",
                    (
                        ScheduleTraceField("attempt", str(attempt)),
                        ScheduleTraceField(
                            "backend_candidate_id", backend_candidate_id
                        ),
                        ScheduleTraceField("outcome", "success"),
                    ),
                )
                succeeded = True
                break
            if not succeeded:
                raise ScheduleRetryExhausted(action.action_id, len(ladder))
        elif isinstance(action, EmitResultAction):
            emitted_queries = action.query_ids
            emitted_outputs = action.output_value_ids
            append_event(action, action.kind.value)
        elif isinstance(action, BatchLoopAction):
            append_event(
                action,
                action.kind.value,
                (
                    ScheduleTraceField("axis", action.axis),
                    ScheduleTraceField("slice_count", str(len(action.slices))),
                ),
            )
        elif isinstance(action, (FallbackAction, RetryAction)):
            append_event(action, f"declare_{action.kind.value}")
        elif isinstance(
            action,
            (
                CheckBudgetAction,
                MaterializeAction,
                RecordEventAction,
                WaitEventAction,
                StateLoadAction,
                StateStoreAction,
                StateInvalidateAction,
                TransferAction,
                RequestReplanAction,
            ),
        ):
            append_event(action, action.kind.value)
        else:
            raise AssertionError(f"unhandled Schedule IR action: {type(action)}")
    trace = ScheduleExecutionTrace(
        schedule_hash=schedule_hash,
        events=tuple(events),
        peak_memory_bytes=peak_bytes,
        emitted_query_ids=emitted_queries,
        emitted_output_value_ids=emitted_outputs,
    )
    trace.validate()
    if trace.peak_memory_bytes != instance.cost_summary.predicted_peak_bytes:
        raise ValueError("schedule runtime trace peak differs from PlanInstance")
    if trace.emitted_query_ids != schedule.query_ids:
        raise ValueError("schedule runtime trace loses or reorders queries")
    return trace


def replay_schedule_trace(
    encoded: str,
    schedule: ScheduleModule,
    *,
    bound_module: BFBoundModule,
    template: PlanTemplate,
    instance: PlanInstance,
    driver: Optional[ScheduleReferenceDriver] = None,
) -> ScheduleExecutionTrace:
    """Re-execute the schedule and reject any noncanonical/tampered trace."""

    expected = execute_schedule_reference(
        schedule,
        bound_module=bound_module,
        template=template,
        instance=instance,
        driver=driver,
    )
    try:
        parsed = json.loads(encoded)
    except json.JSONDecodeError as error:
        raise ValueError("invalid Schedule IR trace JSON") from error
    if not isinstance(parsed, dict):
        raise ValueError("Schedule IR trace must be a JSON object")
    if expected.canonical_json() != encoded:
        raise ValueError("Schedule IR trace does not match deterministic replay")
    return expected


def execute_schedule_with_bound_reference(
    schedule: ScheduleModule,
    *,
    bound_module: BFBoundModule,
    template: PlanTemplate,
    instance: PlanInstance,
    task_module: BFTaskModule,
    input_spec: InputSpec,
    relu_pre: Mapping[str, IntervalState],
    linear_spec_C: Optional[torch.Tensor] = None,
    driver: Optional[ScheduleReferenceDriver] = None,
) -> tuple[IntervalState, ScheduleExecutionTrace]:
    """Execute the schedule ledger and use Bound IR as the semantic oracle."""

    trace = execute_schedule_reference(
        schedule,
        bound_module=bound_module,
        template=template,
        instance=instance,
        driver=driver,
    )
    result = execute_plain_crown_bound_ir(
        bound_module,
        task_module=task_module,
        input_spec=input_spec,
        relu_pre=relu_pre,
        linear_spec_C=linear_spec_C,
    )
    return result, trace
