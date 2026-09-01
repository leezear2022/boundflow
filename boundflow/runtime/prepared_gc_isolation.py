"""Prepared-runtime generation isolation for query-local cyclic GC."""

# pylint: disable=unnecessary-ellipsis

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import time
from types import SimpleNamespace
from typing import Any, Iterator, Protocol, cast


class _GcModule(Protocol):  # pylint: disable=missing-function-docstring
    def collect(self, generation: int = -1) -> int:
        """Collect the requested generation."""
        ...

    def isenabled(self) -> bool:
        """Return whether automatic collection is enabled."""
        ...

    def enable(self) -> None:
        """Enable automatic collection."""
        ...

    def disable(self) -> None:
        """Disable automatic collection."""
        ...


@dataclass
class PreparedGcIsolationReceiptV1:  # pylint: disable=too-many-instance-attributes
    """Account for one reversible prepared-object GC isolation scope."""

    prepare_collect_ns: int
    prepare_collected_object_count: int
    query_collect_generation: int
    gc_enabled_before: bool
    query_collect_call_count: int = 0
    query_collected_object_count: int = 0
    query_collect_ns: int = 0
    restore_collect_ns: int = 0
    restore_collected_object_count: int = 0
    restored: bool = False
    performance_claimed: bool = False

    def to_dict(self) -> dict[str, object]:
        """Return a fail-closed JSON-safe projection after scope restoration."""

        nonnegative = (
            self.prepare_collected_object_count,
            self.query_collected_object_count,
            self.restore_collected_object_count,
        )
        positive = (
            self.prepare_collect_ns,
            self.query_collect_ns,
            self.restore_collect_ns,
        )
        if not (
            all(value >= 0 for value in nonnegative)
            and all(value > 0 for value in positive)
            and self.query_collect_generation == 1
            and self.query_collect_call_count == 1
            and self.restored
            and not self.performance_claimed
        ):
            raise ValueError("prepared GC isolation receipt differs")
        payload = asdict(self)
        payload.update(
            {
                "schema_version": "boundflow.prepared-gc-isolation/v1",
                "full_prepare_collection": True,
                "query_collection_preserved": True,
                "prepared_old_generation_scan_excluded": True,
                "query_timing_excluded": True,
            }
        )
        return payload


@contextmanager
def prepared_gc_isolation_v1(
    *, gc_module: Any, complete_verifier_module: Any
) -> Iterator[PreparedGcIsolationReceiptV1]:
    """Collect new query generations without rescanning prepared old objects."""

    typed_gc = cast(_GcModule, gc_module)
    original_owner = getattr(complete_verifier_module, "gc", None)
    if original_owner is not gc_module or not callable(
        getattr(typed_gc, "collect", None)
    ):
        raise TypeError("prepared GC isolation owner differs")
    enabled_before = bool(typed_gc.isenabled())
    started_ns = time.perf_counter_ns()
    collected = int(typed_gc.collect())
    collect_ns = time.perf_counter_ns() - started_ns
    if collect_ns <= 0 or collected < 0:
        raise RuntimeError("prepared GC isolation setup differs")
    receipt = PreparedGcIsolationReceiptV1(
        prepare_collect_ns=collect_ns,
        prepare_collected_object_count=collected,
        query_collect_generation=1,
        gc_enabled_before=enabled_before,
    )

    def collect_query_generation(*args: object, **kwargs: object) -> int:
        if args or kwargs or receipt.query_collect_call_count != 0:
            raise RuntimeError("prepared GC isolation query collection differs")
        query_started_ns = time.perf_counter_ns()
        query_collected = int(typed_gc.collect(receipt.query_collect_generation))
        receipt.query_collect_ns = time.perf_counter_ns() - query_started_ns
        receipt.query_collected_object_count = query_collected
        receipt.query_collect_call_count += 1
        return query_collected

    proxy = SimpleNamespace(collect=collect_query_generation)
    setattr(complete_verifier_module, "gc", proxy)
    try:
        yield receipt
    finally:
        setattr(complete_verifier_module, "gc", original_owner)
        restore_started_ns = time.perf_counter_ns()
        receipt.restore_collected_object_count = int(typed_gc.collect())
        receipt.restore_collect_ns = time.perf_counter_ns() - restore_started_ns
        if getattr(complete_verifier_module, "gc", None) is not original_owner:
            raise RuntimeError("prepared GC isolation did not restore module owner")
        if bool(typed_gc.isenabled()) != enabled_before:
            if enabled_before:
                typed_gc.enable()
            else:
                typed_gc.disable()
            raise RuntimeError("prepared GC isolation observed GC enable-state drift")
        receipt.restored = True
