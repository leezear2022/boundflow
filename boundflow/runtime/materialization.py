"""Opt-in tracing for logical LinearOperator materialization barriers."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator, Literal

import torch

if TYPE_CHECKING:
    from .linear_operator import LinearOperator


TRACE_SCHEMA_VERSION = "boundflow.materialization/v1"
Persistence = Literal["persistent", "ephemeral"]
STATE_BYTE_CATEGORIES = (
    "alpha_state_bytes",
    "beta_state_bytes",
    "intermediate_bound_bytes",
    "weight_bytes",
    "operator_state_bytes",
)


@dataclass(frozen=True)
class TraceQueryMetadata:
    """Identity and batch axes shared by all events in one traced query."""

    run_id: str = ""
    query_id: str = ""
    bound_method: str = "unknown"
    solver_phase: str = "backward"
    spec_batch: int | None = None
    domain_batch: int | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly metadata representation."""

        return {
            "run_id": self.run_id,
            "query_id": self.query_id,
            "bound_method": self.bound_method,
            "solver_phase": self.solver_phase,
            "spec_batch": self.spec_batch,
            "domain_batch": self.domain_batch,
        }


@dataclass(frozen=True)
class MaterializationEvent:  # pylint: disable=too-many-instance-attributes
    """Schema-v1 metadata for one logical dense-materialization boundary."""

    run_id: str
    query_id: str
    event_id: int
    bound_method: str
    solver_phase: str
    operator_site: str
    source_value: str
    source_primal_op: str
    reason: str
    operator_type: str
    operator_tree_depth: int
    operator_node_count: int
    shape: tuple[int, ...]
    dtype: str
    device: str
    spec_batch: int
    domain_batch: int
    logical_bytes: int
    observed_allocation_delta_bytes: int | None
    persistent_or_ephemeral: Persistence
    logical_lifetime_begin: str
    logical_lifetime_end: str
    consumer_count: int | None
    reuse_count_estimate: int | None
    requires_grad: bool
    autograd_saved: bool | None
    alpha_related: bool
    beta_related: bool

    def to_dict(self) -> dict[str, object]:
        """Return the stable JSON event representation."""

        return {
            "schema_version": TRACE_SCHEMA_VERSION,
            "run_id": self.run_id,
            "query_id": self.query_id,
            "event_id": self.event_id,
            "bound_method": self.bound_method,
            "solver_phase": self.solver_phase,
            "operator_site": self.operator_site,
            "source_value": self.source_value,
            "source_primal_op": self.source_primal_op,
            "reason": self.reason,
            "operator_type": self.operator_type,
            "operator_tree_depth": self.operator_tree_depth,
            "operator_node_count": self.operator_node_count,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "device": self.device,
            "spec_batch": self.spec_batch,
            "domain_batch": self.domain_batch,
            "logical_bytes": self.logical_bytes,
            "observed_allocation_delta_bytes": self.observed_allocation_delta_bytes,
            "persistent_or_ephemeral": self.persistent_or_ephemeral,
            "logical_lifetime_begin": self.logical_lifetime_begin,
            "logical_lifetime_end": self.logical_lifetime_end,
            "consumer_count": self.consumer_count,
            "reuse_count_estimate": self.reuse_count_estimate,
            "requires_grad": self.requires_grad,
            "autograd_saved": self.autograd_saved,
            "alpha_related": self.alpha_related,
            "beta_related": self.beta_related,
        }


@dataclass
class MaterializationTrace:
    """Collect schema-v1 events and distinct logical/allocator memory metrics."""

    query: TraceQueryMetadata = field(default_factory=TraceQueryMetadata)
    capture_cuda_memory: bool = False
    _events: list[MaterializationEvent] = field(default_factory=list)
    _state_bytes: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in STATE_BYTE_CATEGORIES}
    )
    _cuda_peak_allocated_bytes: int | None = None
    _cuda_peak_reserved_bytes: int | None = None
    _cuda_device: torch.device | None = None

    @property
    def events(self) -> tuple[MaterializationEvent, ...]:
        """Return an immutable view of events in execution order."""

        return tuple(self._events)

    def record(self, event: MaterializationEvent) -> None:
        """Append an event to this trace."""

        self._events.append(event)

    def record_state_bytes(self, category: str, value: int) -> None:
        """Record a compatible non-coefficient state-memory category."""

        if category not in STATE_BYTE_CATEGORIES:
            raise ValueError(f"unknown state byte category: {category}")
        if int(value) < 0:
            raise ValueError(f"state bytes must be non-negative, got {value}")
        self._state_bytes[category] = int(value)

    def _begin_cuda_memory(self) -> None:
        if not self.capture_cuda_memory or not torch.cuda.is_available():
            return
        self._cuda_device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.synchronize(self._cuda_device)
        torch.cuda.reset_peak_memory_stats(self._cuda_device)

    def _finish_cuda_memory(self) -> None:
        if self._cuda_device is None:
            return
        torch.cuda.synchronize(self._cuda_device)
        self._cuda_peak_allocated_bytes = int(
            torch.cuda.max_memory_allocated(self._cuda_device)
        )
        self._cuda_peak_reserved_bytes = int(
            torch.cuda.max_memory_reserved(self._cuda_device)
        )

    def summary(self) -> dict[str, object]:
        """Aggregate event counts and logical bytes without conflating allocator metrics."""

        by_reason: dict[str, dict[str, int]] = {}
        by_site: dict[str, dict[str, int]] = {}
        by_operator: dict[str, dict[str, int]] = {}
        by_lifetime_class: dict[str, dict[str, int]] = {}
        for event in self._events:
            _accumulate(by_reason, event.reason, event.logical_bytes)
            _accumulate(by_site, event.operator_site, event.logical_bytes)
            _accumulate(by_operator, event.operator_type, event.logical_bytes)
            _accumulate(
                by_lifetime_class,
                event.persistent_or_ephemeral,
                event.logical_bytes,
            )
        observed = [
            event.observed_allocation_delta_bytes
            for event in self._events
            if event.observed_allocation_delta_bytes is not None
        ]
        return {
            "event_count": len(self._events),
            "logical_materialized_bytes": sum(
                event.logical_bytes for event in self._events
            ),
            "observed_allocation_delta_bytes": sum(observed) if observed else None,
            "peak_cuda_allocated_bytes": self._cuda_peak_allocated_bytes,
            "peak_cuda_reserved_bytes": self._cuda_peak_reserved_bytes,
            "by_reason": by_reason,
            "by_site": by_site,
            "by_operator_type": by_operator,
            "by_lifetime_class": by_lifetime_class,
        }

    def to_record(self) -> dict[str, object]:
        """Return one JSONL-ready query record with schema, summary, state and events."""

        return {
            "schema_version": TRACE_SCHEMA_VERSION,
            **self.query.to_dict(),
            "materialization": self.summary(),
            "state_bytes": dict(self._state_bytes),
            "events": [event.to_dict() for event in self._events],
        }


def _accumulate(
    target: dict[str, dict[str, int]], key: str, logical_bytes: int
) -> None:
    item = target.setdefault(key, {"count": 0, "logical_bytes": 0})
    item["count"] += 1
    item["logical_bytes"] += int(logical_bytes)


def _is_operator(value: object) -> bool:
    return not torch.is_tensor(value) and all(
        hasattr(value, name) for name in ("to_dense", "shape", "input_shape")
    )


def _operator_children(operator: object) -> tuple[object, ...]:
    children: list[object] = []
    for name in ("base", "lhs", "rhs"):
        child = getattr(operator, name, None)
        if _is_operator(child):
            children.append(child)
    return tuple(children)


def operator_tree_stats(operator: object) -> tuple[int, int]:
    """Return deterministic maximum depth and unique object-node count."""

    seen: set[int] = set()

    def _visit(node: object, ancestors: frozenset[int]) -> int:
        node_id = id(node)
        if node_id in ancestors:
            raise ValueError("cycle detected in LinearOperator graph")
        seen.add(node_id)
        children = _operator_children(node)
        if not children:
            return 1
        next_ancestors = ancestors | {node_id}
        return 1 + max(_visit(child, next_ancestors) for child in children)

    return _visit(operator, frozenset()), len(seen)


_ACTIVE_TRACE: ContextVar[MaterializationTrace | None] = ContextVar(
    "boundflow_materialization_trace",
    default=None,
)


@contextmanager
def trace_materializations(  # pylint: disable=too-many-arguments
    trace: MaterializationTrace | None = None,
    *,
    run_id: str = "",
    query_id: str = "",
    bound_method: str = "unknown",
    solver_phase: str = "backward",
    spec_batch: int | None = None,
    domain_batch: int | None = None,
    capture_cuda_memory: bool = False,
) -> Iterator[MaterializationTrace]:
    """Activate a task-local trace; trace-off remains the normal timing path."""

    active = trace or MaterializationTrace(
        query=TraceQueryMetadata(
            run_id=str(run_id),
            query_id=str(query_id),
            bound_method=str(bound_method),
            solver_phase=str(solver_phase),
            spec_batch=spec_batch,
            domain_batch=domain_batch,
        ),
        capture_cuda_memory=bool(capture_cuda_memory),
    )
    token = _ACTIVE_TRACE.set(active)
    active._begin_cuda_memory()  # pylint: disable=protected-access
    try:
        yield active
    finally:
        active._finish_cuda_memory()  # pylint: disable=protected-access
        _ACTIVE_TRACE.reset(token)


def materialize_linear_operator(  # pylint: disable=too-many-arguments,too-many-locals
    operator: LinearOperator,
    *,
    reason: str,
    operator_site: str,
    source_value: str,
    source_primal_op: str,
    persistent_or_ephemeral: Persistence,
    logical_lifetime_begin: str,
    logical_lifetime_end: str,
    consumer_count: int | None = None,
    reuse_count_estimate: int | None = None,
    autograd_saved: bool | None = None,
    alpha_related: bool = False,
    beta_related: bool = False,
) -> torch.Tensor:
    """Materialize an operator and record the explicit boundary when tracing."""

    if not reason:
        raise ValueError("materialization reason must be non-empty")
    trace = _ACTIVE_TRACE.get()
    before_allocated: int | None = None
    if (
        trace is not None
        and trace.capture_cuda_memory
        and operator.device.type == "cuda"
    ):
        before_allocated = int(torch.cuda.memory_allocated(operator.device))

    dense = operator.to_dense()
    if trace is None:
        return dense

    observed_delta: int | None = None
    if before_allocated is not None:
        observed_delta = (
            int(torch.cuda.memory_allocated(dense.device)) - before_allocated
        )
    depth, node_count = operator_tree_stats(operator)
    spec_batch = trace.query.spec_batch or int(dense.shape[1])
    domain_batch = trace.query.domain_batch or int(dense.shape[0])
    trace.record(
        MaterializationEvent(
            run_id=trace.query.run_id,
            query_id=trace.query.query_id,
            event_id=len(trace.events),
            bound_method=trace.query.bound_method,
            solver_phase=trace.query.solver_phase,
            operator_site=str(operator_site),
            source_value=str(source_value),
            source_primal_op=str(source_primal_op),
            reason=str(reason),
            operator_type=type(operator).__name__,
            operator_tree_depth=depth,
            operator_node_count=node_count,
            shape=tuple(int(dim) for dim in dense.shape),
            dtype=str(dense.dtype),
            device=str(dense.device),
            spec_batch=int(spec_batch),
            domain_batch=int(domain_batch),
            logical_bytes=int(dense.numel()) * int(dense.element_size()),
            observed_allocation_delta_bytes=observed_delta,
            persistent_or_ephemeral=persistent_or_ephemeral,
            logical_lifetime_begin=str(logical_lifetime_begin),
            logical_lifetime_end=str(logical_lifetime_end),
            consumer_count=consumer_count,
            reuse_count_estimate=reuse_count_estimate,
            requires_grad=bool(dense.requires_grad),
            autograd_saved=autograd_saved,
            alpha_related=bool(alpha_related),
            beta_related=bool(beta_related),
        )
    )
    return dense
