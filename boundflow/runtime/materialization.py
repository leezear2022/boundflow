"""Opt-in tracing for logical LinearOperator materialization barriers."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator

import torch

if TYPE_CHECKING:
    from .linear_operator import LinearOperator


@dataclass(frozen=True)
class MaterializationEvent:  # pylint: disable=too-many-instance-attributes
    """Metadata for one explicit logical dense-materialization boundary."""

    reason: str
    site: str
    operator_type: str
    logical_shape: tuple[int, ...]
    dense_bytes: int
    dtype: str
    device: str
    lifetime: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly event representation."""

        return {
            "reason": self.reason,
            "site": self.site,
            "operator_type": self.operator_type,
            "logical_shape": list(self.logical_shape),
            "dense_bytes": self.dense_bytes,
            "dtype": self.dtype,
            "device": self.device,
            "lifetime": self.lifetime,
        }


@dataclass
class MaterializationTrace:
    """Collect materialization events inside a scoped bound computation."""

    _events: list[MaterializationEvent] = field(default_factory=list)

    @property
    def events(self) -> tuple[MaterializationEvent, ...]:
        """Return an immutable view of events in execution order."""

        return tuple(self._events)

    def record(self, event: MaterializationEvent) -> None:
        """Append an event to this trace."""

        self._events.append(event)

    def summary(self) -> dict[str, object]:
        """Aggregate count and logical dense bytes by reason and operator."""

        by_reason: dict[str, dict[str, int]] = {}
        by_operator: dict[str, dict[str, int]] = {}
        for event in self._events:
            _accumulate(by_reason, event.reason, event.dense_bytes)
            _accumulate(by_operator, event.operator_type, event.dense_bytes)
        return {
            "count": len(self._events),
            "dense_bytes": sum(event.dense_bytes for event in self._events),
            "by_reason": by_reason,
            "by_operator": by_operator,
            "events": [event.to_dict() for event in self._events],
        }


def _accumulate(target: dict[str, dict[str, int]], key: str, dense_bytes: int) -> None:
    item = target.setdefault(key, {"count": 0, "dense_bytes": 0})
    item["count"] += 1
    item["dense_bytes"] += int(dense_bytes)


_ACTIVE_TRACE: ContextVar[MaterializationTrace | None] = ContextVar(
    "boundflow_materialization_trace",
    default=None,
)


@contextmanager
def trace_materializations(
    trace: MaterializationTrace | None = None,
) -> Iterator[MaterializationTrace]:
    """Activate a task-local trace for the duration of the context."""

    active = trace if trace is not None else MaterializationTrace()
    token = _ACTIVE_TRACE.set(active)
    try:
        yield active
    finally:
        _ACTIVE_TRACE.reset(token)


def materialize_linear_operator(
    operator: LinearOperator,
    *,
    reason: str,
    site: str,
    lifetime: str,
) -> torch.Tensor:
    """Materialize an operator and record the boundary when tracing."""

    if not reason:
        raise ValueError("materialization reason must be non-empty")
    dense = operator.to_dense()
    trace = _ACTIVE_TRACE.get()
    if trace is not None:
        trace.record(
            MaterializationEvent(
                reason=str(reason),
                site=str(site),
                operator_type=type(operator).__name__,
                logical_shape=tuple(int(dim) for dim in dense.shape),
                dense_bytes=int(dense.numel()) * int(dense.element_size()),
                dtype=str(dense.dtype),
                device=str(dense.device),
                lifetime=str(lifetime),
            )
        )
    return dense
