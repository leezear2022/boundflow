"""Reference task dispatch trace linked to typed Task and Schedule IR."""

# Compact trace dataclasses use self-describing method names.
# pylint: disable=missing-function-docstring,too-many-boolean-expressions
# pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-locals
# pylint: disable=too-many-branches,too-many-statements

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from typing import Mapping, Optional, Tuple

import torch

from ..domains.interval import IntervalState
from ..ir.bound import BFBoundModule
from ..ir.plan import PlanInstance, PlanTemplate
from ..ir.schedule import (
    FallbackAction,
    LaunchAction,
    RetryAction,
    ScheduleModule,
    StateInvalidateAction,
    StateLoadAction,
    StateStoreAction,
)
from ..ir.task import BFTaskModule
from ..ir.task_v1 import (
    TaskBackendBinding,
    TaskIRModule,
    task_backend_implementation_id,
)
from .bound_ir_interpreter import BoundIRTaskStepResult, PlainCrownBoundIRSession
from .bound_state_store import (
    BoundRuntimeStateStore,
    validate_state_value_capability,
)
from .task_backend_dispatch import (
    PyTorchReferenceTaskBackend,
    TypedTaskBackend,
    build_backend_dispatch_key,
)
from .task_executor import InputSpec
from .schedule_ir_executor import (
    ScheduleOutOfMemoryError,
    ScheduleRetryExhausted,
)


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
    attempted_backend_candidate_ids: Tuple[str, ...] = ()

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
        if (
            not self.attempted_backend_candidate_ids
            or len(self.attempted_backend_candidate_ids)
            != len(set(self.attempted_backend_candidate_ids))
            or self.backend_candidate_id != self.attempted_backend_candidate_ids[-1]
        ):
            raise ValueError("Task IR execution event backend attempts are invalid")

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
            "attempted_backend_candidate_ids": list(
                self.attempted_backend_candidate_ids
            ),
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
    launch_by_task = {
        action.task_id: action
        for action in schedule.actions
        if isinstance(action, LaunchAction)
    }
    loaded_value_ids = {
        action.source_value_id
        for action in schedule.actions
        if isinstance(action, StateLoadAction)
    }
    completed: set[str] = set()
    events: list[TaskExecutionEvent] = []
    for task in task_module.tasks:
        if any(dependency not in completed for dependency in task.dependency_task_ids):
            raise ValueError(
                "Task IR reference executor has dependency use-before-task"
            )
        launch = launch_by_task.get(task.task_id)
        state_reused = launch is None
        if state_reused and not set(task.output_value_ids).issubset(loaded_value_ids):
            raise ValueError("Task IR reference state reuse lacks task outputs")
        backend_candidate_id = (
            "state-reuse" if state_reused else task.backend.backend_candidate_id
        )
        implementation_id = (
            "boundflow.runtime.state-reuse/v1"
            if state_reused
            else task.backend.reference_implementation_id
        )
        dispatch_hash = (
            hashlib.sha256(
                (
                    "state-reuse|"
                    + task.task_id
                    + "|"
                    + schedule.schedule_id
                    + "|"
                    + "|".join(task.output_value_ids)
                ).encode("utf-8")
            ).hexdigest()
            if state_reused
            else build_backend_dispatch_key(
                task,
                task_module,
                bound_module=bound_module,
                template=template,
                instance=instance,
            ).stable_hash()
        )
        events.append(
            TaskExecutionEvent(
                sequence=len(events),
                task_id=task.task_id,
                region_id=task.region_id,
                op_ids=tuple(op_ref.op_id for op_ref in task.op_refs),
                dependency_task_ids=task.dependency_task_ids,
                backend_candidate_id=backend_candidate_id,
                reference_implementation_id=implementation_id,
                output_value_hashes=(),
                backend_dispatch_key=dispatch_hash,
                attempted_backend_candidate_ids=(backend_candidate_id,),
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
    state_store: Optional[BoundRuntimeStateStore] = None,
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
    if state_store is None:
        state_store = BoundRuntimeStateStore()
    task_by_id = {task.task_id: task for task in task_module.tasks}
    launch_by_task = {
        action.task_id: action
        for action in schedule.actions
        if isinstance(action, LaunchAction)
    }
    loaded_payload_hash_by_value: dict[str, str] = {}
    for action in schedule.actions:
        if isinstance(action, StateLoadAction):
            validate_state_value_capability(bound_module, action.source_value_id)
            payload = state_store.load(
                action,
                bound_module=bound_module,
                session=session,
            )
            loaded_payload_hash_by_value[action.source_value_id] = payload.stable_hash()
        elif isinstance(action, StateInvalidateAction):
            state_store.invalidate(action.state_id)
    retries = {
        action.launch_action_id: action
        for action in schedule.actions
        if isinstance(action, RetryAction)
    }
    fallbacks = {
        action.action_id: action
        for action in schedule.actions
        if isinstance(action, FallbackAction)
    }
    backend_candidates = {
        candidate.candidate_id: candidate for candidate in template.backend_candidates
    }
    completed: set[str] = set()
    events: list[TaskExecutionEvent] = []
    for task in task_module.tasks:
        if any(dependency not in completed for dependency in task.dependency_task_ids):
            raise ValueError("Task IR semantic executor has dependency use-before-task")
        launch = launch_by_task.get(task.task_id)
        if launch is None:
            if not set(task.output_value_ids).issubset(loaded_payload_hash_by_value):
                raise ValueError("Task IR state reuse lacks exact runtime outputs")
            reuse_step = session.skip_task_with_loaded_outputs(
                tuple(op_ref.op_id for op_ref in task.op_refs),
                output_value_ids=task.output_value_ids,
            )
            state_reuse_key = hashlib.sha256(
                "|".join(
                    (
                        "state-reuse",
                        task.task_id,
                        *(
                            loaded_payload_hash_by_value[value_id]
                            for value_id in task.output_value_ids
                        ),
                    )
                ).encode("utf-8")
            ).hexdigest()
            events.append(
                TaskExecutionEvent(
                    sequence=len(events),
                    task_id=task.task_id,
                    region_id=task.region_id,
                    op_ids=reuse_step.op_ids,
                    dependency_task_ids=task.dependency_task_ids,
                    backend_candidate_id="state-reuse",
                    reference_implementation_id=("boundflow.runtime.state-reuse/v1"),
                    output_value_hashes=reuse_step.output_value_hashes,
                    backend_dispatch_key=state_reuse_key,
                    attempted_backend_candidate_ids=("state-reuse",),
                )
            )
            completed.add(task.task_id)
            continue
        retry = retries.get(launch.action_id)
        candidate_ids = [task.backend.backend_candidate_id]
        if retry is not None:
            candidate_ids.extend(
                fallbacks[fallback_id].backend_candidate_id
                for fallback_id in retry.fallback_action_ids
            )
        attempted: list[str] = []
        step: Optional[BoundIRTaskStepResult] = None
        dispatch_key = None
        executed_task = task
        for candidate_id in candidate_ids:
            candidate = backend_candidates[candidate_id]
            executed_task = (
                task
                if candidate_id == task.backend.backend_candidate_id
                else replace(
                    task,
                    backend=TaskBackendBinding(
                        backend_candidate_id=candidate.candidate_id,
                        capability_id=candidate.capability_id,
                        compiled_artifact_key=candidate.compiled_artifact_key,
                        reference_implementation_id=task_backend_implementation_id(
                            candidate.backend
                        ),
                    ),
                )
            )
            dispatch_key = build_backend_dispatch_key(
                task,
                task_module,
                bound_module=bound_module,
                template=template,
                instance=instance,
                backend_candidate=candidate,
            )
            attempted.append(candidate_id)
            try:
                step = backend.dispatch(
                    executed_task,
                    dispatch_key,
                    session=session,
                    template=template,
                )
            except (ScheduleOutOfMemoryError, torch.OutOfMemoryError):
                continue
            break
        if step is None or dispatch_key is None:
            raise ScheduleRetryExhausted(launch.action_id, len(attempted))
        events.append(
            TaskExecutionEvent(
                sequence=len(events),
                task_id=task.task_id,
                region_id=task.region_id,
                op_ids=step.op_ids,
                dependency_task_ids=task.dependency_task_ids,
                backend_candidate_id=executed_task.backend.backend_candidate_id,
                reference_implementation_id=(
                    executed_task.backend.reference_implementation_id
                ),
                output_value_hashes=step.output_value_hashes,
                backend_dispatch_key=dispatch_key.stable_hash(),
                attempted_backend_candidate_ids=tuple(attempted),
            )
        )
        completed.add(task.task_id)
    for action in schedule.actions:
        if isinstance(action, StateStoreAction):
            validate_state_value_capability(bound_module, action.source_value_id)
            state_store.store(
                action,
                bound_module=bound_module,
                session=session,
            )
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
