"""Capability-gated query entry for the typed BoundFlow compiler stack."""

# The payload preserves the established linear_spec_C spelling at its API boundary.
# pylint: disable=invalid-name

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping, Optional, Sequence, Tuple

import torch

from ..domains.interval import IntervalState
from ..ir.bound import BFBoundModule, BoundMethodKind
from ..ir.plan import PlanInstance, PlanTemplate, StateValidity
from ..ir.schedule import lower_plan_instance_to_reference_schedule
from ..ir.task import BFTaskModule
from ..ir.task_v1 import TaskIRModule, lower_plan_instance_to_task_ir
from ..planner.plan_ir_selector import select_plan_instance
from .bab_query import BoundQueryRequest
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
class TypedCompilerQueryRequest:
    """One query with explicit order and fully typed compiler payload."""

    query_id: str
    sequence_number: int
    payload: TypedCompilerQueryPayload

    def validate(self) -> None:
        """Validate query identity and its complete typed payload."""

        if not self.query_id or self.sequence_number < 0:
            raise ValueError("typed compiler query identity/order is invalid")
        self.payload.validate()

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
class TypedCompilerQueryResult:
    """One ordered compiler result plus typed Task execution evidence."""

    query_id: str
    sequence_number: int
    bounds: IntervalState
    plan_instance_hash: str
    task_module_hash: str
    trace: TaskExecutionTrace


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
        self.backend = backend or TypedTaskBackendRegistry()
        self.state_store = state_store or BoundRuntimeStateStore()
        self._plan_cache: dict[tuple[str, Tuple[StateValidity, ...]], _CompiledPlan] = (
            {}
        )
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
                )
            )
            self.executed_queries += 1
        if tuple(result.query_id for result in results) != query_ids:
            raise AssertionError("typed compiler runtime reordered query results")
        return results

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
        key = (request.compatibility_key(), state_validities)
        cached = self._plan_cache.get(key)
        if cached is not None:
            self.plan_cache_hits += 1
            return cached
        self.plan_cache_misses += 1
        instance = select_plan_instance(
            payload.template,
            bound_module=payload.bound_module,
            query_bucket_id=request.compatibility_key(),
            available_memory_bytes=self.available_memory_bytes,
            memory_budget_bytes=self.memory_budget_bytes,
            state_validities=state_validities,
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


__all__ = [
    "CompilerQueryCapabilityError",
    "TypedCompilerQueryPayload",
    "TypedCompilerQueryRequest",
    "TypedCompilerQueryResult",
    "TypedCompilerQueryRuntime",
]
