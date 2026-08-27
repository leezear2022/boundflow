"""Low-perturbation host transaction markers for an external verifier."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import functools
import threading
import time
from typing import Any, Callable, Iterator, Mapping, Sequence

from boundflow.runtime.gpu_attribution import canonical_hash

SOLVER_TRANSACTION_SCHEMA_VERSION = "boundflow.solver-transaction-observer/v1"


class TransactionResolution(str, Enum):
    """Whether a marker proves a mechanism or only bounds a broad scope."""

    COARSE_SCOPE = "coarse_scope"
    EXACT_TRANSACTION = "exact_transaction"


class TransactionCategory(str, Enum):
    """Stable verification transaction categories used by S0 attribution."""

    FRONTEND_SETUP = "frontend_setup"
    CONSTRAINT_IMPORT = "constraint_import"
    ENVIRONMENT_SETUP = "environment_setup"
    MODEL_PREPARE = "model_prepare"
    SPEC_PREPARE = "spec_prepare"
    VERIFY_SCOPE = "verify_scope"
    INCOMPLETE_VERIFICATION = "incomplete_verification"
    COMPLETE_VERIFICATION_SCOPE = "complete_verification_scope"
    BAB_SCOPE = "bab_scope"
    BAB_BOOTSTRAP_SCOPE = "bab_bootstrap_scope"
    SPEC_HANDOFF = "spec_handoff"
    BAB_BOOTSTRAP = "bab_bootstrap"
    DOMAIN_PREPROCESS = "domain_preprocess"
    DOMAIN_SOLVE = "domain_solve"
    DOMAIN_POSTPROCESS = "domain_postprocess"
    BOUND_PREPARE = "bound_prepare"
    BOUND_CORE = "bound_core"
    BOUND_POSTPROCESS = "bound_postprocess"
    RESULT_PUBLISH = "result_publish"
    SOLVER_TERMINATION = "solver_termination"
    HOST_GARBAGE_COLLECTION = "host_garbage_collection"
    DEVICE_CACHE_RELEASE = "device_cache_release"


@dataclass(frozen=True)
class TransactionTarget:
    """One concrete class attribute or call-site module global to patch."""

    owner: Any
    attribute: str
    target_id: str
    category: TransactionCategory
    resolution: TransactionResolution

    def validate(self) -> None:
        if not self.attribute or not self.target_id:
            raise ValueError("solver transaction target identity must be non-empty")
        if not hasattr(self.owner, self.attribute):
            raise ValueError(f"solver transaction target is missing: {self.target_id}")
        if not callable(getattr(self.owner, self.attribute)):
            raise TypeError(
                f"solver transaction target is not callable: {self.target_id}"
            )


@dataclass(frozen=True)
class HostTransactionSpan:  # pylint: disable=too-many-instance-attributes
    """One nested host interval emitted by a patched solver transaction."""

    transaction_id: int
    parent_transaction_id: int | None
    target_id: str
    category: TransactionCategory
    resolution: TransactionResolution
    thread_id: int
    depth: int
    host_start_ns: int
    host_end_ns: int
    outcome: str

    def validate(self) -> None:
        if self.transaction_id < 0:
            raise ValueError("solver transaction ID is negative")
        if self.parent_transaction_id == self.transaction_id:
            raise ValueError("solver transaction cannot parent itself")
        if not self.target_id:
            raise ValueError("solver transaction target ID is empty")
        if self.thread_id < 0 or self.depth < 0:
            raise ValueError("solver transaction thread or depth is negative")
        if self.host_start_ns < 0 or self.host_end_ns < self.host_start_ns:
            raise ValueError("solver transaction interval differs")
        if self.outcome not in {"returned", "raised"}:
            raise ValueError("solver transaction outcome differs")

    @property
    def duration_ns(self) -> int:
        """Return inclusive wrapper wall duration."""

        return self.host_end_ns - self.host_start_ns

    def to_dict(self) -> dict[str, object]:
        """Return the stable artifact representation."""

        self.validate()
        return {
            "transaction_id": self.transaction_id,
            "parent_transaction_id": self.parent_transaction_id,
            "target_id": self.target_id,
            "category": self.category.value,
            "resolution": self.resolution.value,
            "thread_id": self.thread_id,
            "depth": self.depth,
            "host_start_ns": self.host_start_ns,
            "host_end_ns": self.host_end_ns,
            "outcome": self.outcome,
        }


@dataclass
class _PendingTransaction:
    transaction_id: int
    parent_transaction_id: int | None
    target: TransactionTarget
    thread_id: int
    depth: int
    host_start_ns: int


class SolverTransactionObserver:
    """Patch selected Python call sites and record nested host spans."""

    def __init__(
        self,
        *,
        scope_started_ns: int,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        if scope_started_ns < 0:
            raise ValueError("solver transaction scope start is negative")
        self._scope_started_ns = scope_started_ns
        self._clock_ns = clock_ns
        self._lock = threading.RLock()
        self._next_id = 0
        self._active: dict[int, list[_PendingTransaction]] = {}
        self._completed: list[HostTransactionSpan] = []
        self._installed = False

    def _now(self) -> int:
        observed = self._clock_ns() - self._scope_started_ns
        if observed < 0:
            raise ValueError("solver transaction clock precedes scope")
        return observed

    def _enter(self, target: TransactionTarget) -> _PendingTransaction:
        thread_id = threading.get_ident()
        with self._lock:
            stack = self._active.setdefault(thread_id, [])
            transaction_id = self._next_id
            self._next_id += 1
            pending = _PendingTransaction(
                transaction_id=transaction_id,
                parent_transaction_id=(stack[-1].transaction_id if stack else None),
                target=target,
                thread_id=thread_id,
                depth=len(stack),
                host_start_ns=self._now(),
            )
            stack.append(pending)
            return pending

    def _exit(self, pending: _PendingTransaction, *, outcome: str) -> None:
        host_end_ns = self._now()
        with self._lock:
            stack = self._active.get(pending.thread_id)
            if not stack or stack[-1].transaction_id != pending.transaction_id:
                raise RuntimeError("solver transaction observer stack differs")
            stack.pop()
            if not stack:
                del self._active[pending.thread_id]
            span = HostTransactionSpan(
                transaction_id=pending.transaction_id,
                parent_transaction_id=pending.parent_transaction_id,
                target_id=pending.target.target_id,
                category=pending.target.category,
                resolution=pending.target.resolution,
                thread_id=pending.thread_id,
                depth=pending.depth,
                host_start_ns=pending.host_start_ns,
                host_end_ns=host_end_ns,
                outcome=outcome,
            )
            span.validate()
            self._completed.append(span)

    def _make_wrapper(
        self, target: TransactionTarget, original: Callable[..., Any]
    ) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            pending = self._enter(target)
            try:
                result = original(*args, **kwargs)
            except BaseException:
                self._exit(pending, outcome="raised")
                raise
            self._exit(pending, outcome="returned")
            return result

        return functools.wraps(original)(wrapped)

    @contextmanager
    def instrument(self, targets: Sequence[TransactionTarget]) -> Iterator[None]:
        """Install wrappers transactionally and restore every target on exit."""

        if self._installed:
            raise RuntimeError("solver transaction observer is already installed")
        if not targets:
            raise ValueError("solver transaction observer targets are empty")
        identities = [(id(target.owner), target.attribute) for target in targets]
        if len(identities) != len(set(identities)):
            raise ValueError("solver transaction observer targets duplicate")
        for target in targets:
            target.validate()
        originals: list[tuple[TransactionTarget, Callable[..., Any]]] = []
        self._installed = True
        try:
            for target in targets:
                original = getattr(target.owner, target.attribute)
                wrapped = self._make_wrapper(target, original)
                setattr(target.owner, target.attribute, wrapped)
                originals.append((target, original))
            yield
        finally:
            for target, original in reversed(originals):
                setattr(target.owner, target.attribute, original)
            self._installed = False

    def finish(self, *, scope_ns: int) -> tuple[HostTransactionSpan, ...]:
        """Freeze completed spans after all patched calls and the outer scope end."""

        if scope_ns <= 0:
            raise ValueError("solver transaction scope must be positive")
        if self._installed:
            raise RuntimeError("solver transaction observer is still installed")
        with self._lock:
            if self._active:
                raise RuntimeError("solver transaction observer has active calls")
            spans = tuple(sorted(self._completed, key=lambda item: item.transaction_id))
        by_id = {span.transaction_id: span for span in spans}
        if len(by_id) != len(spans):
            raise ValueError("solver transaction IDs duplicate")
        for span in spans:
            span.validate()
            if span.host_end_ns > scope_ns:
                raise ValueError("solver transaction escapes outer scope")
            if span.parent_transaction_id is not None:
                if span.parent_transaction_id not in by_id:
                    raise ValueError("solver transaction parent is missing")
                parent = by_id[span.parent_transaction_id]
                if (
                    parent.thread_id != span.thread_id
                    or parent.depth + 1 != span.depth
                    or parent.host_start_ns > span.host_start_ns
                    or parent.host_end_ns < span.host_end_ns
                ):
                    raise ValueError("solver transaction nesting differs")
        return spans


def _strict_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    return value


def host_transaction_span_from_dict(value: Mapping[str, Any]) -> HostTransactionSpan:
    """Parse one exact artifact row and reject schema drift."""

    expected = {
        "transaction_id",
        "parent_transaction_id",
        "target_id",
        "category",
        "resolution",
        "thread_id",
        "depth",
        "host_start_ns",
        "host_end_ns",
        "outcome",
    }
    if set(value) != expected:
        raise ValueError("solver transaction span fields differ")
    parent = value["parent_transaction_id"]
    if parent is not None:
        parent = _strict_int(parent, "solver transaction parent")
    target_id = value["target_id"]
    outcome = value["outcome"]
    if not isinstance(target_id, str) or not isinstance(outcome, str):
        raise TypeError("solver transaction string field differs")
    span = HostTransactionSpan(
        transaction_id=_strict_int(value["transaction_id"], "transaction ID"),
        parent_transaction_id=parent,
        target_id=target_id,
        category=TransactionCategory(value["category"]),
        resolution=TransactionResolution(value["resolution"]),
        thread_id=_strict_int(value["thread_id"], "transaction thread"),
        depth=_strict_int(value["depth"], "transaction depth"),
        host_start_ns=_strict_int(value["host_start_ns"], "transaction start"),
        host_end_ns=_strict_int(value["host_end_ns"], "transaction end"),
        outcome=outcome,
    )
    span.validate()
    return span


def summarize_solver_transactions(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    spans: Sequence[HostTransactionSpan],
    *,
    compute_calls: Sequence[Mapping[str, Any]],
    scope_ns: int,
    minimum_mechanism_coverage: float = 0.97,
) -> dict[str, object]:
    """Build one exclusive timeline with compute calls deeper than host markers."""

    if scope_ns <= 0:
        raise ValueError("solver transaction summary scope must be positive")
    if not 0.0 <= minimum_mechanism_coverage <= 1.0:
        raise ValueError("solver transaction coverage gate differs")
    for span in spans:
        span.validate()
        if span.host_end_ns > scope_ns:
            raise ValueError("solver transaction span escapes summary scope")
    normalized_calls: list[dict[str, object]] = []
    for call in compute_calls:
        start = _strict_int(call.get("host_start_ns"), "compute call start")
        end = _strict_int(call.get("host_end_ns"), "compute call end")
        depth = _strict_int(call.get("depth"), "compute call depth")
        call_id = _strict_int(call.get("call_id"), "compute call ID")
        if start < 0 or end < start or end > scope_ns or depth < 0 or call_id < 0:
            raise ValueError("compute call interval differs")
        phase = call.get("phase")
        external_phase = call.get("external_phase")
        if not isinstance(phase, str) or not isinstance(external_phase, str):
            raise TypeError("compute call phase identity differs")
        normalized_calls.append(
            {
                "call_id": call_id,
                "start_ns": start,
                "end_ns": end,
                "depth": depth,
                "category": f"bound_compute:{phase}:{external_phase}",
            }
        )
    boundaries = sorted(
        {0, scope_ns}
        | {point for span in spans for point in (span.host_start_ns, span.host_end_ns)}
        | {
            point
            for call in normalized_calls
            for point in (
                _strict_int(call["start_ns"], "normalized compute start"),
                _strict_int(call["end_ns"], "normalized compute end"),
            )
        }
    )
    category_ns: dict[str, int] = {}
    target_call_counts: dict[str, int] = {}
    for span in spans:
        target_call_counts[span.target_id] = (
            target_call_counts.get(span.target_id, 0) + 1
        )
    unresolved_intervals: list[dict[str, object]] = []
    resolved_ns = 0
    for start, end in zip(boundaries, boundaries[1:]):
        if end <= start:
            continue
        active_calls = [
            call
            for call in normalized_calls
            if _strict_int(call["start_ns"], "normalized compute start") <= start
            and _strict_int(call["end_ns"], "normalized compute end") >= end
        ]
        if active_calls:
            owner_call = max(
                active_calls,
                key=lambda call: (
                    _strict_int(call["depth"], "normalized compute depth"),
                    _strict_int(call["call_id"], "normalized compute ID"),
                ),
            )
            category = str(owner_call["category"])
            source = f"compute-call-{owner_call['call_id']}"
            resolved = True
        else:
            active_spans = [
                span
                for span in spans
                if span.host_start_ns <= start and span.host_end_ns >= end
            ]
            if active_spans:
                owner_span = max(
                    active_spans,
                    key=lambda span: (
                        span.depth,
                        span.resolution == TransactionResolution.EXACT_TRANSACTION,
                        span.transaction_id,
                    ),
                )
                resolved = (
                    owner_span.resolution == TransactionResolution.EXACT_TRANSACTION
                )
                category = (
                    owner_span.category.value
                    if resolved
                    else f"mechanism_unresolved:{owner_span.category.value}"
                )
                source = owner_span.target_id
            else:
                resolved = False
                category = "mechanism_unresolved:no_marker"
                source = "outer-scope"
        duration = end - start
        category_ns[category] = category_ns.get(category, 0) + duration
        if resolved:
            resolved_ns += duration
        else:
            unresolved_intervals.append(
                {
                    "start_ns": start,
                    "end_ns": end,
                    "duration_ns": duration,
                    "category": category,
                    "source": source,
                }
            )
    if sum(category_ns.values()) != scope_ns:
        raise ValueError("solver transaction exclusive timeline does not close")
    unresolved_ns = scope_ns - resolved_ns
    coverage = resolved_ns / scope_ns
    unresolved_intervals.sort(
        key=lambda item: (
            -_strict_int(item["duration_ns"], "unresolved duration"),
            _strict_int(item["start_ns"], "unresolved start"),
        )
    )
    summary: dict[str, object] = {
        "schema_version": SOLVER_TRANSACTION_SCHEMA_VERSION,
        "scope_ns": scope_ns,
        "span_count": len(spans),
        "compute_call_count": len(normalized_calls),
        "target_call_counts": dict(sorted(target_call_counts.items())),
        "category_ns": dict(sorted(category_ns.items())),
        "category_share": {
            key: value / scope_ns for key, value in sorted(category_ns.items())
        },
        "mechanism_resolved_ns": resolved_ns,
        "mechanism_unresolved_ns": unresolved_ns,
        "mechanism_coverage_share": coverage,
        "mechanism_unresolved_share": unresolved_ns / scope_ns,
        "minimum_mechanism_coverage": minimum_mechanism_coverage,
        "mechanism_admitted": coverage >= minimum_mechanism_coverage,
        "largest_unresolved_intervals": unresolved_intervals[:20],
        "all_transactions_returned": all(span.outcome == "returned" for span in spans),
        "performance_claimed": False,
    }
    summary["summary_hash"] = canonical_hash(summary)
    return summary
