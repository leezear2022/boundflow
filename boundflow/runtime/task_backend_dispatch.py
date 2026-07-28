"""Typed backend dispatch and cache identity for Task IR v1."""

# Compact immutable key objects deliberately expose serialization helpers.
# pylint: disable=missing-function-docstring,too-many-instance-attributes
# pylint: disable=too-few-public-methods,too-many-arguments

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Optional, Protocol
from pathlib import Path

from ..backends.tvm.fused_crown_cache import FusedCrownModuleCache
from ..ir.bound import BFBoundModule
from ..ir.plan import BackendCandidate, BackendKind, PlanInstance, PlanTemplate
from ..ir.task_v1 import (
    TaskIRModule,
    TaskIRUnit,
    task_backend_implementation_id,
)
from .bound_ir_interpreter import (
    BoundIRTaskStepResult,
    PlainCrownBoundIRSession,
)
from .fused_crown import (
    TorchChunkedFusedCrownExecutor,
    TorchDenseFusedCrownReference,
    TVMFusedCrownExecutor,
    TVMUnfusedCrownExecutor,
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
    backend_candidate: Optional[BackendCandidate] = None,
) -> BackendDispatchKey:
    """Build a key only after all cross-layer typed inputs validate."""

    task_module.validate(
        bound_module=bound_module,
        template=template,
        instance=instance,
    )
    if task not in task_module.tasks:
        raise ValueError("backend dispatch task is not owned by Task IR module")
    if backend_candidate is None:
        candidates = {
            candidate.candidate_id: candidate
            for candidate in template.backend_candidates
        }
        backend_candidate = candidates[task.backend.backend_candidate_id]
    elif (
        backend_candidate not in template.backend_candidates
        or backend_candidate.region_id != task.region_id
        or not backend_candidate.static_legal
    ):
        raise ValueError("backend fallback candidate is illegal or wrong-region")
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
        backend_candidate_id=backend_candidate.candidate_id,
        capability_id=backend_candidate.capability_id,
        compiled_artifact_key=backend_candidate.compiled_artifact_key,
        reference_implementation_id=task_backend_implementation_id(
            backend_candidate.backend
        ),
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


class _StaticTemplateHashCache:
    """Cache one validated template digest by exact immutable object identity."""

    def __init__(self) -> None:
        self._hashes: dict[int, tuple[PlanTemplate, BFBoundModule, str]] = {}

    def get(self, template: PlanTemplate, bound_module: BFBoundModule) -> str:
        """Return a digest only when both exact static objects still match."""

        cached = self._hashes.get(id(template))
        if cached is not None and cached[0] is template and cached[1] is bound_module:
            return cached[2]
        digest = template.stable_hash(bound_module=bound_module)
        self._hashes[id(template)] = (template, bound_module, digest)
        return digest


@dataclass(frozen=True)
class _PreparedReferenceTask:
    task_id: str
    op_ids: tuple[str, ...]
    output_value_ids: tuple[str, ...]
    backend_candidate_id: str
    capability_id: str


class PyTorchReferenceTaskBackend:
    """Hash-keyed PyTorch reference adapter for selected REFERENCE candidates."""

    def __init__(
        self, *, static_hash_cache: Optional[_StaticTemplateHashCache] = None
    ) -> None:
        self._prepared: dict[str, _PreparedReferenceTask] = {}
        self._static_hash_cache = static_hash_cache or _StaticTemplateHashCache()
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
            or key.bound_module_hash != session.bound_module_hash
            or key.plan_template_hash
            != self._static_hash_cache.get(template, session.bound_module)
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


@dataclass(frozen=True)
class _PreparedPyTorchTask:
    backend: BackendKind
    task_id: str
    op_ids: tuple[str, ...]
    output_value_ids: tuple[str, ...]
    capability_id: str


class PyTorchTaskBackendRegistry:
    """Typed registry for reference, dense, structured, and chunked PyTorch."""

    def __init__(self, *, chunk_rows: int = 128) -> None:
        self._static_hash_cache = _StaticTemplateHashCache()
        self._reference = PyTorchReferenceTaskBackend(
            static_hash_cache=self._static_hash_cache
        )
        self._dense_fused = TorchDenseFusedCrownReference()
        self._chunked = TorchChunkedFusedCrownExecutor(chunk_rows=chunk_rows)
        self._prepared: dict[str, _PreparedPyTorchTask] = {}
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
        candidate, backend = _validate_typed_dispatch(
            task,
            key,
            session=session,
            template=template,
            plan_template_hash=self._static_hash_cache.get(
                template, session.bound_module
            ),
        )
        if backend == BackendKind.REFERENCE:
            return self._reference.dispatch(
                task, key, session=session, template=template
            )
        if backend not in {
            BackendKind.PYTORCH_DENSE,
            BackendKind.PYTORCH_STRUCTURED,
            BackendKind.PYTORCH_CHUNKED,
        }:
            raise ValueError(f"PyTorch registry rejects backend: {backend.value}")
        prepared = _PreparedPyTorchTask(
            backend=backend,
            task_id=task.task_id,
            op_ids=tuple(op_ref.op_id for op_ref in task.op_refs),
            output_value_ids=task.output_value_ids,
            capability_id=candidate.capability_id,
        )
        digest = key.stable_hash()
        cached = self._prepared.get(digest)
        if cached is None:
            self._prepared[digest] = prepared
            self.cache_misses += 1
        elif cached != prepared:
            raise ValueError("PyTorch backend cache key collision or stale task")
        else:
            self.cache_hits += 1
        fused = (
            len(task.op_refs) == 2
            and task.op_refs[0].kind.value == "relu_relaxation"
            and task.op_refs[1].kind.value in {"linear_backward", "conv2d_backward"}
        )
        if backend == BackendKind.PYTORCH_CHUNKED:
            if not fused:
                raise ValueError(
                    "PyTorch chunked backend requires fused ReLU→Affine Task IR"
                )
            return session.execute_fused_relu_affine_task(
                prepared.op_ids,
                output_value_ids=prepared.output_value_ids,
                executor=self._chunked,
            )
        if backend == BackendKind.PYTORCH_DENSE and fused:
            return session.execute_fused_relu_affine_task(
                prepared.op_ids,
                output_value_ids=prepared.output_value_ids,
                executor=self._dense_fused,
            )
        return session.execute_task(
            prepared.op_ids,
            output_value_ids=prepared.output_value_ids,
        )


def _validate_typed_dispatch(
    task: TaskIRUnit,
    key: BackendDispatchKey,
    *,
    session: PlainCrownBoundIRSession,
    template: PlanTemplate,
    plan_template_hash: str,
) -> tuple[BackendCandidate, BackendKind]:
    key.validate()
    if (
        key.task_id != task.task_id
        or key.backend_candidate_id != task.backend.backend_candidate_id
        or key.capability_id != task.backend.capability_id
        or key.bound_module_hash != session.bound_module_hash
        or key.plan_template_hash != plan_template_hash
    ):
        raise ValueError("backend dispatch key does not match typed task")
    candidates = {
        candidate.candidate_id: candidate for candidate in template.backend_candidates
    }
    candidate = candidates.get(task.backend.backend_candidate_id)
    if (
        candidate is None
        or not candidate.static_legal
        or candidate.capability_id != task.backend.capability_id
    ):
        raise ValueError("typed backend registry rejects selected capability")
    capabilities = {
        capability.capability_id: capability for capability in template.capabilities
    }
    capability = capabilities.get(task.backend.capability_id)
    if (
        capability is None
        or capability.backend != candidate.backend
        or any(
            op_ref.kind not in capability.supported_op_kinds for op_ref in task.op_refs
        )
    ):
        raise ValueError("typed backend registry rejects Task IR op kinds")
    return candidate, candidate.backend


class _DispatchNamespacedFusedCache:
    """Bind the legacy cache interface to one typed backend dispatch key."""

    def __init__(self, cache: object, dispatch_key: str) -> None:
        self.cache = cache
        self.dispatch_key = dispatch_key

    def get(self, kind: str, signature: object) -> tuple[object, object]:
        return self.cache.get(  # type: ignore[attr-defined,no-any-return]
            kind,
            signature,
            backend_dispatch_key=self.dispatch_key,
        )


class TVMTaskBackendRegistry:
    """Typed Task IR registry for fused and explicit-workspace TVM CROWN."""

    def __init__(self, *, cache_dir: Optional[Path] = None) -> None:
        self._static_hash_cache = _StaticTemplateHashCache()
        self._reference = PyTorchReferenceTaskBackend(
            static_hash_cache=self._static_hash_cache
        )
        self.cache = None
        if cache_dir is not None:
            self.cache = FusedCrownModuleCache(cache_dir)
        self.cache_hits = 0
        self.cache_misses = 0
        self._prepared: dict[str, _PreparedPyTorchTask] = {}

    def dispatch(
        self,
        task: TaskIRUnit,
        key: BackendDispatchKey,
        *,
        session: PlainCrownBoundIRSession,
        template: PlanTemplate,
    ) -> BoundIRTaskStepResult:
        candidate, backend = _validate_typed_dispatch(
            task,
            key,
            session=session,
            template=template,
            plan_template_hash=self._static_hash_cache.get(
                template, session.bound_module
            ),
        )
        if backend == BackendKind.REFERENCE:
            return self._reference.dispatch(
                task, key, session=session, template=template
            )
        if backend not in {
            BackendKind.TVM_FUSED_TIR,
            BackendKind.TVM_TIR_UNFUSED,
        }:
            raise ValueError(f"TVM registry rejects backend: {backend.value}")
        prepared = _PreparedPyTorchTask(
            backend=backend,
            task_id=task.task_id,
            op_ids=tuple(op_ref.op_id for op_ref in task.op_refs),
            output_value_ids=task.output_value_ids,
            capability_id=candidate.capability_id,
        )
        if not (
            len(task.op_refs) == 2
            and task.op_refs[0].kind.value == "relu_relaxation"
            and task.op_refs[1].kind.value in {"linear_backward", "conv2d_backward"}
        ):
            raise ValueError("TVM CROWN backend requires fused ReLU→Affine Task IR")
        digest = key.stable_hash()
        cached = self._prepared.get(digest)
        if cached is None:
            self._prepared[digest] = prepared
            self.cache_misses += 1
        elif cached != prepared:
            raise ValueError("TVM backend cache key collision or stale task")
        else:
            self.cache_hits += 1
        if backend == BackendKind.TVM_FUSED_TIR:
            namespaced = (
                None
                if self.cache is None
                else _DispatchNamespacedFusedCache(self.cache, digest)
            )
            executor = TVMFusedCrownExecutor(compile_cache=namespaced)
        else:
            executor = TVMUnfusedCrownExecutor()
        return session.execute_fused_relu_affine_task(
            prepared.op_ids,
            output_value_ids=prepared.output_value_ids,
            executor=executor,
        )


class TypedTaskBackendRegistry:
    """Composite typed registry used by semantic Schedule fallback."""

    def __init__(
        self,
        *,
        chunk_rows: int = 128,
        tvm_cache_dir: Optional[Path] = None,
    ) -> None:
        self.pytorch = PyTorchTaskBackendRegistry(chunk_rows=chunk_rows)
        self.tvm = TVMTaskBackendRegistry(cache_dir=tvm_cache_dir)

    def dispatch(
        self,
        task: TaskIRUnit,
        key: BackendDispatchKey,
        *,
        session: PlainCrownBoundIRSession,
        template: PlanTemplate,
    ) -> BoundIRTaskStepResult:
        candidates = {
            candidate.candidate_id: candidate
            for candidate in template.backend_candidates
        }
        candidate = candidates.get(key.backend_candidate_id)
        if candidate is None:
            raise ValueError("typed registry dispatch references unknown backend")
        if candidate.backend in {
            BackendKind.REFERENCE,
            BackendKind.PYTORCH_DENSE,
            BackendKind.PYTORCH_STRUCTURED,
            BackendKind.PYTORCH_CHUNKED,
        }:
            return self.pytorch.dispatch(task, key, session=session, template=template)
        return self.tvm.dispatch(task, key, session=session, template=template)
