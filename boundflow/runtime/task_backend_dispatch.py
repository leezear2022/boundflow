"""Typed backend dispatch and cache identity for Task IR v1."""

# Compact immutable key objects deliberately expose serialization helpers.
# pylint: disable=missing-function-docstring,too-many-instance-attributes
# pylint: disable=too-few-public-methods

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Optional, Protocol

from ..ir.bound import BFBoundModule
from ..ir.plan import BackendKind, PlanInstance, PlanTemplate
from ..ir.task_v1 import TaskIRModule, TaskIRUnit
from .bound_ir_interpreter import (
    BoundIRTaskStepResult,
    PlainCrownBoundIRSession,
)

BACKEND_DISPATCH_KEY_SCHEMA_VERSION = "boundflow.backend-dispatch-key/v1"


@dataclass(frozen=True)
class BackendDispatchKey:
    """Complete identity of one prepared backend task."""

    bound_module_hash: str
    plan_template_hash: str
    plan_instance_hash: str
    task_module_hash: str
    task_id: str
    backend_candidate_id: str
    capability_id: str
    compiled_artifact_key: Optional[str]
    reference_implementation_id: str
    schema_version: str = BACKEND_DISPATCH_KEY_SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != BACKEND_DISPATCH_KEY_SCHEMA_VERSION:
            raise ValueError("unsupported backend dispatch key schema")
        for name in (
            "bound_module_hash",
            "plan_template_hash",
            "plan_instance_hash",
            "task_module_hash",
        ):
            if len(getattr(self, name)) != 64:
                raise ValueError(f"backend dispatch {name} is not SHA-256")
        if any(
            not value
            for value in (
                self.task_id,
                self.backend_candidate_id,
                self.capability_id,
                self.reference_implementation_id,
            )
        ):
            raise ValueError("backend dispatch key identity is incomplete")
        if self.compiled_artifact_key is not None and not self.compiled_artifact_key:
            raise ValueError("backend dispatch compiled artifact key is empty")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "bound_module_hash": self.bound_module_hash,
            "plan_template_hash": self.plan_template_hash,
            "plan_instance_hash": self.plan_instance_hash,
            "task_module_hash": self.task_module_hash,
            "task_id": self.task_id,
            "backend_candidate_id": self.backend_candidate_id,
            "capability_id": self.capability_id,
            "compiled_artifact_key": self.compiled_artifact_key,
            "reference_implementation_id": self.reference_implementation_id,
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    def stable_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def build_backend_dispatch_key(
    task: TaskIRUnit,
    task_module: TaskIRModule,
    *,
    bound_module: BFBoundModule,
    template: PlanTemplate,
    instance: PlanInstance,
) -> BackendDispatchKey:
    """Build a key only after all cross-layer typed inputs validate."""

    task_module.validate(
        bound_module=bound_module,
        template=template,
        instance=instance,
    )
    if task not in task_module.tasks:
        raise ValueError("backend dispatch task is not owned by Task IR module")
    key = BackendDispatchKey(
        bound_module_hash=bound_module.stable_hash(),
        plan_template_hash=template.stable_hash(bound_module=bound_module),
        plan_instance_hash=instance.stable_hash(
            template=template, bound_module=bound_module
        ),
        task_module_hash=task_module.stable_hash(
            bound_module=bound_module,
            template=template,
            instance=instance,
        ),
        task_id=task.task_id,
        backend_candidate_id=task.backend.backend_candidate_id,
        capability_id=task.backend.capability_id,
        compiled_artifact_key=task.backend.compiled_artifact_key,
        reference_implementation_id=task.backend.reference_implementation_id,
    )
    key.validate()
    return key


class TypedTaskBackend(Protocol):
    """Backend entry point that consumes a typed Task IR unit."""

    def dispatch(
        self,
        task: TaskIRUnit,
        key: BackendDispatchKey,
        *,
        session: PlainCrownBoundIRSession,
        template: PlanTemplate,
    ) -> BoundIRTaskStepResult:
        """Execute one task or reject its typed capability."""
        ...


@dataclass(frozen=True)
class _PreparedReferenceTask:
    task_id: str
    op_ids: tuple[str, ...]
    output_value_ids: tuple[str, ...]
    backend_candidate_id: str
    capability_id: str


class PyTorchReferenceTaskBackend:
    """Hash-keyed PyTorch reference adapter for selected REFERENCE candidates."""

    def __init__(self) -> None:
        self._prepared: dict[str, _PreparedReferenceTask] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def dispatch(
        self,
        task: TaskIRUnit,
        key: BackendDispatchKey,
        *,
        session: PlainCrownBoundIRSession,
        template: PlanTemplate,
    ) -> BoundIRTaskStepResult:
        key.validate()
        if (
            key.task_id != task.task_id
            or key.backend_candidate_id != task.backend.backend_candidate_id
            or key.capability_id != task.backend.capability_id
            or key.bound_module_hash != session.bound_module.stable_hash()
            or key.plan_template_hash
            != template.stable_hash(bound_module=session.bound_module)
        ):
            raise ValueError("backend dispatch key does not match typed task")
        candidates = {
            candidate.candidate_id: candidate
            for candidate in template.backend_candidates
        }
        candidate = candidates.get(task.backend.backend_candidate_id)
        if (
            candidate is None
            or candidate.backend != BackendKind.REFERENCE
            or not candidate.static_legal
            or candidate.capability_id != task.backend.capability_id
        ):
            raise ValueError("PyTorch reference backend rejects selected capability")
        capabilities = {
            capability.capability_id: capability for capability in template.capabilities
        }
        capability = capabilities.get(task.backend.capability_id)
        if (
            capability is None
            or capability.backend != BackendKind.REFERENCE
            or any(
                op_ref.kind not in capability.supported_op_kinds
                for op_ref in task.op_refs
            )
        ):
            raise ValueError("PyTorch reference backend rejects Task IR op kinds")

        digest = key.stable_hash()
        prepared = _PreparedReferenceTask(
            task_id=task.task_id,
            op_ids=tuple(op_ref.op_id for op_ref in task.op_refs),
            output_value_ids=task.output_value_ids,
            backend_candidate_id=task.backend.backend_candidate_id,
            capability_id=task.backend.capability_id,
        )
        cached = self._prepared.get(digest)
        if cached is None:
            self._prepared[digest] = prepared
            self.cache_misses += 1
        elif cached != prepared:
            raise ValueError("backend dispatch cache key collision or stale task")
        else:
            self.cache_hits += 1
        return session.execute_task(
            prepared.op_ids,
            output_value_ids=prepared.output_value_ids,
        )
