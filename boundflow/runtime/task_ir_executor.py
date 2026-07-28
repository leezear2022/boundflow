"""Reference task dispatch trace linked to typed Task and Schedule IR."""

# Compact trace dataclasses use self-describing method names.
# pylint: disable=missing-function-docstring,too-many-boolean-expressions
# pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-locals

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping, Optional, Tuple

import torch

from ..domains.interval import IntervalState
from ..ir.bound import BFBoundModule
from ..ir.plan import PlanInstance, PlanTemplate
from ..ir.schedule import LaunchAction, ScheduleModule
from ..ir.task import BFTaskModule
from ..ir.task_v1 import TaskIRModule
from .bound_ir_interpreter import PlainCrownBoundIRSession
from .task_backend_dispatch import (
    PyTorchReferenceTaskBackend,
    TypedTaskBackend,
    build_backend_dispatch_key,
)
from .task_executor import InputSpec


@dataclass(frozen=True)
class TaskExecutionEvent:
    """One typed task dispatch in dependency order."""

    sequence: int
    task_id: str
    region_id: str
    op_ids: Tuple[str, ...]
    dependency_task_ids: Tuple[str, ...]
    backend_candidate_id: str
    reference_implementation_id: str
    output_value_hashes: Tuple[Tuple[str, str], ...] = ()
    backend_dispatch_key: str = ""

    def validate(self) -> None:
        if (
            self.sequence < 0
            or not self.task_id
            or not self.region_id
            or not self.op_ids
            or not self.backend_candidate_id
            or not self.reference_implementation_id
        ):
            raise ValueError("Task IR execution event is incomplete")
        if any(
            not value_id or len(value_hash) != 64
            for value_id, value_hash in self.output_value_hashes
        ):
            raise ValueError("Task IR execution event has invalid output hashes")
        if len(self.backend_dispatch_key) != 64:
            raise ValueError("Task IR execution event backend key is not SHA-256")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "sequence": self.sequence,
            "task_id": self.task_id,
            "region_id": self.region_id,
            "op_ids": list(self.op_ids),
            "dependency_task_ids": list(self.dependency_task_ids),
            "backend_candidate_id": self.backend_candidate_id,
            "reference_implementation_id": self.reference_implementation_id,
            "output_value_hashes": [
                {"value_id": value_id, "sha256": value_hash}
                for value_id, value_hash in self.output_value_hashes
            ],
            "backend_dispatch_key": self.backend_dispatch_key,
        }


@dataclass(frozen=True)
class TaskExecutionTrace:
    """Canonical dispatch evidence for one Task IR module."""

    task_module_hash: str
    schedule_hash: str
    events: Tuple[TaskExecutionEvent, ...]

    def validate(self) -> None:
        if (
            len(self.task_module_hash) != 64
            or len(self.schedule_hash) != 64
            or not self.events
        ):
            raise ValueError("Task IR execution trace identity/events are incomplete")
        for index, event in enumerate(self.events):
            event.validate()
            if event.sequence != index:
                raise ValueError("Task IR execution trace sequence is not contiguous")
        task_ids = tuple(event.task_id for event in self.events)
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("Task IR execution trace dispatches a task twice")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": "boundflow.task-execution-trace/v1",
            "task_module_hash": self.task_module_hash,
            "schedule_hash": self.schedule_hash,
            "events": [event.to_dict() for event in self.events],
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    def stable_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def execute_task_ir_reference(
    task_module: TaskIRModule,
    schedule: ScheduleModule,
    *,
    bound_module: BFBoundModule,
    template: PlanTemplate,
    instance: PlanInstance,
) -> TaskExecutionTrace:
    """Dispatch typed tasks in Schedule launch order and verify dependencies."""

    task_module.validate_schedule_linkage(
        schedule,
        bound_module=bound_module,
        template=template,
        instance=instance,
    )
    task_by_id = {task.task_id: task for task in task_module.tasks}
    launches = tuple(
        action for action in schedule.actions if isinstance(action, LaunchAction)
    )
    completed: set[str] = set()
    events: list[TaskExecutionEvent] = []
    for launch in launches:
        task = task_by_id[launch.task_id]
        if any(dependency not in completed for dependency in task.dependency_task_ids):
            raise ValueError(
                "Task IR reference executor has dependency use-before-task"
            )
        events.append(
            TaskExecutionEvent(
                sequence=len(events),
                task_id=task.task_id,
                region_id=task.region_id,
                op_ids=tuple(op_ref.op_id for op_ref in task.op_refs),
                dependency_task_ids=task.dependency_task_ids,
                backend_candidate_id=task.backend.backend_candidate_id,
                reference_implementation_id=task.backend.reference_implementation_id,
                output_value_hashes=(),
                backend_dispatch_key=build_backend_dispatch_key(
                    task,
                    task_module,
                    bound_module=bound_module,
                    template=template,
                    instance=instance,
                ).stable_hash(),
            )
        )
        completed.add(task.task_id)
    trace = TaskExecutionTrace(
        task_module_hash=task_module.stable_hash(
            bound_module=bound_module,
            template=template,
            instance=instance,
        ),
        schedule_hash=schedule.stable_hash(
            bound_module=bound_module,
            template=template,
            instance=instance,
        ),
        events=tuple(events),
    )
    trace.validate()
    if completed != set(task_by_id):
        raise ValueError("Task IR reference executor did not dispatch every task")
    return trace


def execute_task_ir_semantics(
    task_module: TaskIRModule,
    schedule: ScheduleModule,
    *,
    bound_module: BFBoundModule,
    template: PlanTemplate,
    instance: PlanInstance,
    legacy_task_module: BFTaskModule,
    input_spec: InputSpec,
    relu_pre: Mapping[str, IntervalState],
    linear_spec_C: Optional[torch.Tensor] = None,
    backend: Optional[TypedTaskBackend] = None,
) -> tuple[IntervalState, TaskExecutionTrace]:
    """Execute each TaskIRUnit's exact Bound op partition in Schedule order."""

    task_module.validate_schedule_linkage(
        schedule,
        bound_module=bound_module,
        template=template,
        instance=instance,
    )
    session = PlainCrownBoundIRSession(
        bound_module,
        task_module=legacy_task_module,
        input_spec=input_spec,
        relu_pre=relu_pre,
        linear_spec_C=linear_spec_C,
    )
    if backend is None:
        backend = PyTorchReferenceTaskBackend()
    task_by_id = {task.task_id: task for task in task_module.tasks}
    launches = tuple(
        action for action in schedule.actions if isinstance(action, LaunchAction)
    )
    completed: set[str] = set()
    events: list[TaskExecutionEvent] = []
    for launch in launches:
        task = task_by_id[launch.task_id]
        if any(dependency not in completed for dependency in task.dependency_task_ids):
            raise ValueError("Task IR semantic executor has dependency use-before-task")
        dispatch_key = build_backend_dispatch_key(
            task,
            task_module,
            bound_module=bound_module,
            template=template,
            instance=instance,
        )
        step = backend.dispatch(
            task,
            dispatch_key,
            session=session,
            template=template,
        )
        events.append(
            TaskExecutionEvent(
                sequence=len(events),
                task_id=task.task_id,
                region_id=task.region_id,
                op_ids=step.op_ids,
                dependency_task_ids=task.dependency_task_ids,
                backend_candidate_id=task.backend.backend_candidate_id,
                reference_implementation_id=task.backend.reference_implementation_id,
                output_value_hashes=step.output_value_hashes,
                backend_dispatch_key=dispatch_key.stable_hash(),
            )
        )
        completed.add(task.task_id)
    if completed != set(task_by_id):
        raise ValueError("Task IR semantic executor did not execute every task")
    result = session.result()
    trace = TaskExecutionTrace(
        task_module_hash=task_module.stable_hash(
            bound_module=bound_module,
            template=template,
            instance=instance,
        ),
        schedule_hash=schedule.stable_hash(
            bound_module=bound_module,
            template=template,
            instance=instance,
        ),
        events=tuple(events),
    )
    trace.validate()
    return result, trace
