"""Runtime enforcement and trace for selected Plan IR storage lifetimes."""

# Compact immutable trace types and the runtime ledger intentionally stay together.
# pylint: disable=missing-function-docstring,too-many-instance-attributes
# pylint: disable=too-many-boolean-expressions

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Mapping

from ..ir.bound import BFBoundModule, BoundRepresentation
from ..ir.plan import PlanInstance, PlanTemplate, StorageBinding, StorageCandidate
from ..ir.schedule import ScheduleModule, StateStoreAction
from ..ir.task_v1 import TaskIRUnit
from .bound_ir_interpreter import PlainCrownBoundIRSession

STORAGE_EXECUTION_TRACE_SCHEMA_VERSION = "boundflow.storage-execution-trace/v1"


@dataclass(frozen=True)
class StorageExecutionEvent:
    """Logical residency immediately after one Task IR region."""

    sequence: int
    task_id: str
    completed_op_id: str
    live_bytes_before_release: int
    live_bytes_after_release: int
    evicted_value_ids: tuple[str, ...]

    def validate(self) -> None:
        if self.sequence < 0 or not self.task_id or not self.completed_op_id:
            raise ValueError("storage execution event identity is incomplete")
        if (
            self.live_bytes_before_release < 0
            or self.live_bytes_after_release < 0
            or self.live_bytes_after_release > self.live_bytes_before_release
        ):
            raise ValueError("storage execution event residency is invalid")
        if len(self.evicted_value_ids) != len(set(self.evicted_value_ids)):
            raise ValueError("storage execution event repeats an eviction")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "sequence": self.sequence,
            "task_id": self.task_id,
            "completed_op_id": self.completed_op_id,
            "live_bytes_before_release": self.live_bytes_before_release,
            "live_bytes_after_release": self.live_bytes_after_release,
            "evicted_value_ids": list(self.evicted_value_ids),
        }


@dataclass(frozen=True)
class StorageExecutionTrace:
    """Replay-grade evidence that selected lifetimes changed runtime residency."""

    plan_template_hash: str
    plan_instance_hash: str
    storage_candidate_id: str
    planned_peak_bytes: int
    observed_peak_live_bytes: int
    released_value_count: int
    events: tuple[StorageExecutionEvent, ...]
    schema_version: str = STORAGE_EXECUTION_TRACE_SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != STORAGE_EXECUTION_TRACE_SCHEMA_VERSION:
            raise ValueError("unsupported storage execution trace schema")
        if (
            len(self.plan_template_hash) != 64
            or len(self.plan_instance_hash) != 64
            or not self.storage_candidate_id
            or self.planned_peak_bytes <= 0
            or self.observed_peak_live_bytes <= 0
            or self.observed_peak_live_bytes > self.planned_peak_bytes
            or self.released_value_count < 0
            or not self.events
        ):
            raise ValueError("storage execution trace summary is invalid")
        for index, event in enumerate(self.events):
            event.validate()
            if event.sequence != index:
                raise ValueError("storage execution trace sequence is not contiguous")
        if self.released_value_count != sum(
            len(event.evicted_value_ids) for event in self.events
        ):
            raise ValueError("storage execution trace release count differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "plan_template_hash": self.plan_template_hash,
            "plan_instance_hash": self.plan_instance_hash,
            "storage_candidate_id": self.storage_candidate_id,
            "planned_peak_bytes": self.planned_peak_bytes,
            "observed_peak_live_bytes": self.observed_peak_live_bytes,
            "released_value_count": self.released_value_count,
            "events": [event.to_dict() for event in self.events],
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    def stable_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PreparedStoragePlanRuntime:
    """Prevalidated immutable storage metadata reused across dynamic queries."""

    bound_module: BFBoundModule
    template: PlanTemplate
    instance: PlanInstance
    schedule: ScheduleModule
    selected: StorageCandidate
    bindings: Mapping[str, StorageBinding]
    op_index: Mapping[str, int]
    pinned_value_ids: frozenset[str]
    plan_template_hash: str
    plan_instance_hash: str

    @classmethod
    def prepare(
        cls,
        *,
        bound_module: BFBoundModule,
        template: PlanTemplate,
        instance: PlanInstance,
        schedule: ScheduleModule,
    ) -> "PreparedStoragePlanRuntime":
        """Validate cross-layer identity once and cache static lifetime lookups."""

        schedule.validate(
            bound_module=bound_module, template=template, instance=instance
        )
        candidates = {
            candidate.candidate_id: candidate
            for candidate in template.storage_candidates
        }
        selected = candidates.get(instance.storage_decision.candidate_id)
        if selected is None:
            raise ValueError("storage runtime cannot resolve selected candidate")
        if any(
            binding.representation != BoundRepresentation.DENSE
            for binding in selected.bindings
        ):
            raise NotImplementedError("storage runtime v1 requires dense bindings")
        return cls(
            bound_module=bound_module,
            template=template,
            instance=instance,
            schedule=schedule,
            selected=selected,
            bindings=MappingProxyType(
                {binding.value_id: binding for binding in selected.bindings}
            ),
            op_index=MappingProxyType(
                {op.op_id: index for index, op in enumerate(bound_module.graph.ops)}
            ),
            pinned_value_ids=frozenset(
                {
                    *bound_module.graph.outputs,
                    *(
                        action.source_value_id
                        for action in schedule.actions
                        if isinstance(action, StateStoreAction)
                    ),
                }
            ),
            plan_template_hash=template.stable_hash(bound_module=bound_module),
            plan_instance_hash=instance.stable_hash(
                template=template, bound_module=bound_module
            ),
        )

    def require_identity(
        self,
        *,
        bound_module: BFBoundModule,
        template: PlanTemplate,
        instance: PlanInstance,
        schedule: ScheduleModule,
    ) -> None:
        """Reject reuse with any different static compiler object."""

        if any(
            left is not right
            for left, right in zip(
                (self.bound_module, self.template, self.instance, self.schedule),
                (bound_module, template, instance, schedule),
            )
        ):
            raise ValueError("prepared storage runtime identity differs")


class StoragePlanRuntime:
    """Enforce one dense StorageCandidate's last-use release policy."""

    def __init__(
        self,
        *,
        bound_module: BFBoundModule,
        template: PlanTemplate,
        instance: PlanInstance,
        schedule: ScheduleModule,
        prepared: PreparedStoragePlanRuntime | None = None,
    ) -> None:
        if prepared is None:
            prepared = PreparedStoragePlanRuntime.prepare(
                bound_module=bound_module,
                template=template,
                instance=instance,
                schedule=schedule,
            )
        else:
            prepared.require_identity(
                bound_module=bound_module,
                template=template,
                instance=instance,
                schedule=schedule,
            )
        self.bound_module = bound_module
        self.template = template
        self.instance = instance
        self.schedule = schedule
        self.selected = prepared.selected
        self.bindings = prepared.bindings
        self.op_index = prepared.op_index
        self.pinned_value_ids = prepared.pinned_value_ids
        self.plan_template_hash = prepared.plan_template_hash
        self.plan_instance_hash = prepared.plan_instance_hash
        self.events: list[StorageExecutionEvent] = []
        self.observed_peak_live_bytes = 0
        self.released_value_count = 0
        self._bound = False
        self._finalized = False

    def bind_session(self, session: PlainCrownBoundIRSession) -> None:
        """Bind exactly once and account for graph inputs already resident."""

        if self._bound or session.bound_module is not self.bound_module:
            raise ValueError("storage runtime session identity differs or repeats")
        self._bound = True
        self._observe(session)

    def before_task(self, task: TaskIRUnit, session: PlainCrownBoundIRSession) -> None:
        """Reject a selected plan that released an input before its consumer."""

        self._require_active(session)
        missing = tuple(
            value_id for value_id in task.input_value_ids if value_id not in session.env
        )
        if missing:
            raise ValueError(f"storage runtime released task inputs early: {missing}")

    def after_task(self, task: TaskIRUnit, session: PlainCrownBoundIRSession) -> None:
        """Release values at selected last-use boundaries and record residency."""

        self._require_active(session)
        completed_op_id = task.op_refs[-1].op_id
        completed_index = self.op_index[completed_op_id]
        before = self._observe(session)
        evicted = tuple(
            sorted(
                value_id
                for value_id in tuple(session.env)
                if value_id not in self.pinned_value_ids
                and self.op_index[self.bindings[value_id].live_to_op_id]
                <= completed_index
            )
        )
        for value_id in evicted:
            del session.env[value_id]
        after = self._observe(session)
        self.events.append(
            StorageExecutionEvent(
                sequence=len(self.events),
                task_id=task.task_id,
                completed_op_id=completed_op_id,
                live_bytes_before_release=before,
                live_bytes_after_release=after,
                evicted_value_ids=evicted,
            )
        )
        self.released_value_count += len(evicted)

    def finalize(self, session: PlainCrownBoundIRSession) -> None:
        """Freeze the trace after result/state consumers have completed."""

        self._require_active(session)
        if self._finalized:
            raise ValueError("storage runtime finalized twice")
        if not set(self.bound_module.graph.outputs).issubset(session.env):
            raise ValueError("storage runtime lost final Bound outputs")
        self._finalized = True

    def trace(self) -> StorageExecutionTrace:
        """Return immutable evidence only after successful finalization."""

        if not self._finalized:
            raise ValueError("storage runtime trace requested before finalization")
        result = StorageExecutionTrace(
            plan_template_hash=self.plan_template_hash,
            plan_instance_hash=self.plan_instance_hash,
            storage_candidate_id=self.selected.candidate_id,
            planned_peak_bytes=self.selected.cost.predicted_peak_bytes,
            observed_peak_live_bytes=self.observed_peak_live_bytes,
            released_value_count=self.released_value_count,
            events=tuple(self.events),
        )
        result.validate()
        return result

    def _observe(self, session: PlainCrownBoundIRSession) -> int:
        unknown = set(session.env) - set(self.bindings)
        if unknown:
            raise ValueError(
                f"storage runtime found unplanned values: {sorted(unknown)}"
            )
        live_bytes = sum(self.bindings[value_id].size_bytes for value_id in session.env)
        self.observed_peak_live_bytes = max(self.observed_peak_live_bytes, live_bytes)
        if self.observed_peak_live_bytes > self.selected.cost.predicted_peak_bytes:
            raise ValueError("storage runtime residency exceeds selected arena")
        return live_bytes

    def _require_active(self, session: PlainCrownBoundIRSession) -> None:
        if (
            not self._bound
            or self._finalized
            or session.bound_module is not self.bound_module
        ):
            raise ValueError("storage runtime is not active for this session")


__all__ = [
    "PreparedStoragePlanRuntime",
    "STORAGE_EXECUTION_TRACE_SCHEMA_VERSION",
    "StorageExecutionEvent",
    "StorageExecutionTrace",
    "StoragePlanRuntime",
]
