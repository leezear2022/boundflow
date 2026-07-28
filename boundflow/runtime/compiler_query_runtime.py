"""Capability-gated query entry for the typed BoundFlow compiler stack."""

# The payload preserves the established linear_spec_C spelling at its API boundary.
# pylint: disable=invalid-name

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping, Optional, Sequence, Tuple

import torch

from ..domains.interval import IntervalState
from ..frontends.plain_crown_bound_ir import tensor_content_hash
from ..ir.bound import BFBoundModule, BoundMethodKind
from ..ir.plan import PlanInstance, PlanTemplate, StateValidity
from ..ir.schedule import lower_plan_instance_to_reference_schedule
from ..ir.task import BFTaskModule
from ..ir.task_v1 import TaskIRModule, lower_plan_instance_to_task_ir
from ..planner.materialization import BoundMethod, OptimizationStage
from ..planner.plan_ir_selector import PlanSelectionContext, select_plan_instance
from .bab_query import BoundQueryRequest, model_versions
from .bound_state_store import BoundRuntimeStateStore
from .task_backend_dispatch import TypedTaskBackend, TypedTaskBackendRegistry
from .task_executor import InputSpec
from .task_ir_executor import TaskExecutionTrace, execute_task_ir_semantics


class CompilerQueryCapabilityError(ValueError):
    """The requested solver semantics are outside the proven compiler subset."""


@dataclass(frozen=True)
class TypedCompilerQueryPayload:
    """Complete semantic/compiler inputs omitted by the PR-13 BaB query schema."""

    legacy_task_module: BFTaskModule
    bound_module: BFBoundModule
    template: PlanTemplate
    input_spec: InputSpec
    relu_pre: Mapping[str, IntervalState]
    linear_spec_C: Optional[torch.Tensor] = None

    def validate(self) -> None:
        """Require the exact plain-CROWN subset closed by IR-1 through IR-4."""

        self.legacy_task_module.validate()
        self.bound_module.validate()
        self.template.validate(bound_module=self.bound_module)
        workload = self.template.workload
        unsupported_flags = (
            workload.requires_grad,
            workload.alpha_enabled,
            workload.beta_enabled,
            workload.split_state_present,
        )
        if (
            self.bound_module.domain.method != BoundMethodKind.CROWN
            or workload.method != BoundMethodKind.CROWN
            or any(unsupported_flags)
        ):
            raise CompilerQueryCapabilityError(
                "typed compiler query supports plain CROWN without "
                "grad/alpha/beta/split state only"
            )
        if str(self.input_spec.center.device) != workload.device:
            raise ValueError("compiler query input device differs from Plan workload")
        dtype = str(self.input_spec.center.dtype).removeprefix("torch.")
        if dtype != workload.dtype:
            raise ValueError("compiler query input dtype differs from Plan workload")
        if int(self.input_spec.center.shape[0]) > workload.domain_batch_size:
            raise ValueError("compiler query domain batch exceeds Plan workload")


@dataclass(frozen=True)
class CompilerRuntimeContext:
    """Per-query memory, deadline, cache, and distribution selection facts."""

    available_memory_bytes: int
    memory_budget_bytes: int
    deadline_us: Optional[int] = None
    plan_selection: PlanSelectionContext = PlanSelectionContext()

    def validate(self) -> None:
        """Reject invalid query-time resource or selection facts."""

        if self.available_memory_bytes <= 0 or self.memory_budget_bytes <= 0:
            raise ValueError("compiler runtime memory limits must be positive")
        if self.deadline_us is not None and self.deadline_us <= 0:
            raise ValueError("compiler runtime deadline must be positive")
        self.plan_selection.validate()


@dataclass(frozen=True)
class TypedCompilerQueryRequest:
    """One query with explicit order and fully typed compiler payload."""

    query_id: str
    sequence_number: int
    payload: TypedCompilerQueryPayload
    runtime_context: Optional[CompilerRuntimeContext] = None

    def validate(self) -> None:
        """Validate query identity and its complete typed payload."""

        if not self.query_id or self.sequence_number < 0:
            raise ValueError("typed compiler query identity/order is invalid")
        self.payload.validate()
        if self.runtime_context is not None:
            self.runtime_context.validate()

    def compatibility_key(self) -> str:
        """Hash only fields that permit exact Plan/Task IR reuse."""

        self.validate()
        payload = self.payload
        identity = "|".join(
            (
                payload.bound_module.stable_hash(),
                payload.template.stable_hash(bound_module=payload.bound_module),
                repr(tuple(int(dim) for dim in payload.input_spec.center.shape)),
                str(payload.input_spec.center.dtype),
                str(payload.input_spec.center.device),
                payload.input_spec.perturbation.perturbation_id,
            )
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CompilerBoundQueryRequest:
    """PR-13 query identity paired with its complete typed compiler payload."""

    query_request: BoundQueryRequest
    compiler_payload: TypedCompilerQueryPayload
    runtime_context: Optional[CompilerRuntimeContext] = None

    def validate(self) -> None:
        """Prove that PR-13 identity and compiler payload describe one query."""

        query = self.query_request.query
        query.validate()
        self.compiler_payload.validate()
        if self.runtime_context is not None:
            self.runtime_context.validate()
        if (
            query.bound_method != BoundMethod.CROWN
            or query.optimization_stage != OptimizationStage.FINAL_BOUND
            or query.requires_grad
            or query.compatibility_key.backend_capability_class
            != "plain_crown_typed_ir"
            or query.requested_outputs != ("bounds",)
        ):
            raise CompilerQueryCapabilityError(
                "PR-13 query is not the plain_crown_typed_ir compiler capability"
            )
        dynamic = self.query_request.payload
        compiler_input = self.compiler_payload.input_spec
        if (
            dynamic.input_spec.value_name != compiler_input.value_name
            or dynamic.input_spec.perturbation.perturbation_id
            != compiler_input.perturbation.perturbation_id
            or tensor_content_hash(dynamic.input_spec.center)
            != tensor_content_hash(compiler_input.center)
        ):
            raise ValueError("PR-13 input payload differs from compiler payload")
        if not _optional_tensor_equal(
            dynamic.linear_spec_c, self.compiler_payload.linear_spec_C
        ):
            raise ValueError("PR-13 objective payload differs from compiler payload")
        if (
            dynamic.split_by_relu_input
            or dynamic.warm_alpha_by_relu_input
            or dynamic.warm_beta_by_relu_input
        ):
            raise CompilerQueryCapabilityError(
                "plain-CROWN compiler query cannot carry alpha/beta/split state"
            )
        structure_hash, weight_version = model_versions(
            self.compiler_payload.legacy_task_module
        )
        if (
            query.model_structure_hash != structure_hash
            or query.weight_version != weight_version
        ):
            raise ValueError("PR-13 model identity differs from compiler payload")

    def to_typed_request(self) -> TypedCompilerQueryRequest:
        """Drop no identity while adapting into the typed compiler runtime."""

        self.validate()
        query = self.query_request.query
        return TypedCompilerQueryRequest(
            query_id=query.query_id,
            sequence_number=query.sequence_number,
            payload=self.compiler_payload,
            runtime_context=self.runtime_context,
        )


@dataclass(frozen=True)
class TypedCompilerQueryResult:
    """One ordered compiler result plus typed Task execution evidence."""

    query_id: str
    sequence_number: int
    bounds: IntervalState
    plan_instance_hash: str
    task_module_hash: str
    trace: TaskExecutionTrace
    plan_instance: PlanInstance


@dataclass(frozen=True)
class _CompiledPlan:
    instance: PlanInstance
    task_module: TaskIRModule


class TypedCompilerQueryRuntime:  # pylint: disable=too-many-instance-attributes
    """Select, cache, lower, and execute eligible typed compiler queries."""

    def __init__(
        self,
        *,
        available_memory_bytes: int,
        memory_budget_bytes: int,
        backend: Optional[TypedTaskBackend] = None,
        state_store: Optional[BoundRuntimeStateStore] = None,
    ) -> None:
        if available_memory_bytes <= 0 or memory_budget_bytes <= 0:
            raise ValueError("compiler query runtime memory limits must be positive")
        self.available_memory_bytes = available_memory_bytes
        self.memory_budget_bytes = memory_budget_bytes
        self.default_runtime_context = CompilerRuntimeContext(
            available_memory_bytes=available_memory_bytes,
            memory_budget_bytes=memory_budget_bytes,
        )
        self.backend = backend or TypedTaskBackendRegistry()
        self.state_store = state_store or BoundRuntimeStateStore()
        self._plan_cache: dict[
            tuple[str, CompilerRuntimeContext, Tuple[StateValidity, ...]],
            _CompiledPlan,
        ] = {}
        self.plan_cache_hits = 0
        self.plan_cache_misses = 0
        self.executed_queries = 0

    def execute(
        self, requests: Sequence[TypedCompilerQueryRequest]
    ) -> list[TypedCompilerQueryResult]:
        """Execute every request exactly once and preserve caller order."""

        if not requests:
            return []
        query_ids = tuple(request.query_id for request in requests)
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("typed compiler query batch contains duplicate IDs")
        results: list[TypedCompilerQueryResult] = []
        for request in requests:
            request.validate()
            compiled = self._resolve_compiled_plan(request)
            payload = request.payload
            schedule = lower_plan_instance_to_reference_schedule(
                payload.bound_module,
                template=payload.template,
                instance=compiled.instance,
                query_ids=(request.query_id,),
            )
            bounds, trace = execute_task_ir_semantics(
                compiled.task_module,
                schedule,
                bound_module=payload.bound_module,
                template=payload.template,
                instance=compiled.instance,
                legacy_task_module=payload.legacy_task_module,
                input_spec=payload.input_spec,
                relu_pre=payload.relu_pre,
                linear_spec_C=payload.linear_spec_C,
                backend=self.backend,
                state_store=self.state_store,
            )
            results.append(
                TypedCompilerQueryResult(
                    query_id=request.query_id,
                    sequence_number=request.sequence_number,
                    bounds=bounds,
                    plan_instance_hash=compiled.instance.stable_hash(
                        template=payload.template,
                        bound_module=payload.bound_module,
                    ),
                    task_module_hash=compiled.task_module.stable_hash(
                        bound_module=payload.bound_module,
                        template=payload.template,
                        instance=compiled.instance,
                    ),
                    trace=trace,
                    plan_instance=compiled.instance,
                )
            )
            self.executed_queries += 1
        if tuple(result.query_id for result in results) != query_ids:
            raise AssertionError("typed compiler runtime reordered query results")
        return results

    def execute_bound_queries(
        self, requests: Sequence[CompilerBoundQueryRequest]
    ) -> list[TypedCompilerQueryResult]:
        """Execute compiler-eligible PR-13 requests through typed IR only."""

        return self.execute(tuple(request.to_typed_request() for request in requests))

    def reject_legacy_bab_request(self, request: BoundQueryRequest) -> None:
        """Keep PR-14 external α/β/split No-Go explicit at the new entry."""

        request.query.validate()
        capability = request.query.compatibility_key.backend_capability_class
        raise CompilerQueryCapabilityError(
            "legacy PR-13 BaB query is not compiler-eligible: "
            f"capability={capability}, method={request.query.bound_method.value}; "
            "PR-14 whole-query replay remains NO-GO, so no plain-CROWN fallback "
            "is permitted"
        )

    def audit(self) -> dict[str, object]:
        """Return compiler-entry and state-runtime accounting."""

        return {
            "executed_queries": self.executed_queries,
            "plan_cache_entries": len(self._plan_cache),
            "plan_cache_hits": self.plan_cache_hits,
            "plan_cache_misses": self.plan_cache_misses,
            "physical_cross_query_batching_claimed": False,
            "state_store": self.state_store.audit(),
        }

    def _resolve_compiled_plan(
        self, request: TypedCompilerQueryRequest
    ) -> _CompiledPlan:
        payload = request.payload
        state_validities = _state_validities(
            payload.template,
            bound_module=payload.bound_module,
            state_store=self.state_store,
        )
        runtime_context = request.runtime_context or self.default_runtime_context
        runtime_context.validate()
        key = (request.compatibility_key(), runtime_context, state_validities)
        cached = self._plan_cache.get(key)
        if cached is not None:
            self.plan_cache_hits += 1
            return cached
        self.plan_cache_misses += 1
        instance = select_plan_instance(
            payload.template,
            bound_module=payload.bound_module,
            query_bucket_id=(
                request.compatibility_key()
                + ":"
                + runtime_context.plan_selection.query_distribution_id
            ),
            available_memory_bytes=runtime_context.available_memory_bytes,
            memory_budget_bytes=runtime_context.memory_budget_bytes,
            deadline_us=runtime_context.deadline_us,
            state_validities=state_validities,
            selection_context=runtime_context.plan_selection,
        )
        compiled = _CompiledPlan(
            instance=instance,
            task_module=lower_plan_instance_to_task_ir(
                payload.bound_module,
                template=payload.template,
                instance=instance,
            ),
        )
        self._plan_cache[key] = compiled
        return compiled


def _state_validities(
    template: PlanTemplate,
    *,
    bound_module: BFBoundModule,
    state_store: BoundRuntimeStateStore,
) -> Tuple[StateValidity, ...]:
    groups: dict[str, tuple[str, str]] = {}
    for candidate in template.state_candidates:
        identity = (candidate.source_value_id, candidate.state_version)
        previous = groups.setdefault(candidate.state_id, identity)
        if previous != identity:
            raise ValueError(
                "runtime state v1 requires one source/version per state candidate group"
            )
    bound_hash = bound_module.stable_hash()
    return tuple(
        state_store.validity(
            state_id=state_id,
            source_value_id=source_value_id,
            state_version=state_version,
            bound_module_hash=bound_hash,
        )
        for state_id, (source_value_id, state_version) in sorted(groups.items())
    )


def _optional_tensor_equal(
    left: Optional[torch.Tensor], right: Optional[torch.Tensor]
) -> bool:
    if left is None or right is None:
        return left is right
    return tensor_content_hash(left) == tensor_content_hash(right)


__all__ = [
    "CompilerBoundQueryRequest",
    "CompilerQueryCapabilityError",
    "CompilerRuntimeContext",
    "TypedCompilerQueryPayload",
    "TypedCompilerQueryRequest",
    "TypedCompilerQueryResult",
    "TypedCompilerQueryRuntime",
]
