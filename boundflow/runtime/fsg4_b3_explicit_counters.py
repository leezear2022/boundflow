"""Typed explicit-counter evidence for the FSG4/B3 B2 diagnostic."""

# pylint: disable=missing-function-docstring,too-many-instance-attributes
# pylint: disable=too-many-boolean-expressions

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Mapping, Sequence

FSG4_B3_COUNTER_SCHEMA = "boundflow.fsg4-b3-explicit-counters/v1"

COUNTER_NAMES = (
    "atomic_commit_call_count",
    "atomic_stage_call_count",
    "candidate_snapshot_materialization_count",
    "commit_copy_call_count",
    "committed_mutable_path_count",
    "device_rollback_backup_count",
    "forward_trace_build_count",
    "full_optimizer_step_snapshot_count",
    "gpu_tensor_content_hash_count",
    "kfsb_candidate_count",
    "kfsb_child_batch_count",
    "kfsb_evaluation_call_count",
    "live_tensor_copy_call_count",
    "module_binding_move_in_core_count",
    "optimizer_bound_evaluation_call_count",
    "optimizer_evaluation_count",
    "optimizer_trace_call_count",
    "optimizer_update_count",
    "rollback_copy_call_count",
    "scope_construction_count",
    "stable_hash_call_count",
    "template_compile_count",
    "template_hit_in_core_count",
    "tensor_content_hash_count",
    "timed_candidate_d2h_copy_count",
    "typed_validate_call_count",
)

EXPECTED_B2_FIXED_COUNTERS = {
    "atomic_commit_call_count": 1,
    "atomic_stage_call_count": 1,
    "candidate_snapshot_materialization_count": 12,
    "commit_copy_call_count": 12,
    "committed_mutable_path_count": 12,
    "device_rollback_backup_count": 12,
    "forward_trace_build_count": 5,
    "full_optimizer_step_snapshot_count": 10,
    "kfsb_candidate_count": 3,
    "kfsb_child_batch_count": 3,
    "kfsb_evaluation_call_count": 1,
    "live_tensor_copy_call_count": 12,
    "module_binding_move_in_core_count": 1,
    "optimizer_bound_evaluation_call_count": 10,
    "optimizer_evaluation_count": 10,
    "optimizer_trace_call_count": 1,
    "optimizer_update_count": 9,
    "rollback_copy_call_count": 0,
    "scope_construction_count": 2,
    "template_compile_count": 1,
    "template_hit_in_core_count": 0,
    "timed_candidate_d2h_copy_count": 12,
}

EXPECTED_B3A_FIXED_COUNTERS = {
    **EXPECTED_B2_FIXED_COUNTERS,
    "module_binding_move_in_core_count": 0,
    "scope_construction_count": 1,
    "template_hit_in_core_count": 1,
}

EXPECTED_B3B_FIXED_COUNTERS = {
    **EXPECTED_B3A_FIXED_COUNTERS,
    "full_optimizer_step_snapshot_count": 0,
    "forward_trace_build_count": 4,
}

EXPECTED_B3C_FIXED_COUNTERS = {
    **EXPECTED_B3B_FIXED_COUNTERS,
    "timed_candidate_d2h_copy_count": 0,
}

EXPECTED_FIXED_COUNTERS = {
    "B2": EXPECTED_B2_FIXED_COUNTERS,
    "B3-A": EXPECTED_B3A_FIXED_COUNTERS,
    "B3-B": EXPECTED_B3B_FIXED_COUNTERS,
    "B3-C": EXPECTED_B3C_FIXED_COUNTERS,
}


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"FSG4/B3 {label} must be an integer")
    return value


@dataclass(frozen=True)
class Fsg4B3CounterEvent:
    """One explicit event emitted at a named B2 execution seam."""

    ordinal: int
    counter: str
    amount: int
    detail: str

    def validate(self) -> None:
        if (
            self.ordinal < 0
            or self.counter not in COUNTER_NAMES
            or self.amount <= 0
            or not self.detail
        ):
            raise ValueError("FSG4/B3 counter event differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "ordinal": self.ordinal,
            "counter": self.counter,
            "amount": self.amount,
            "detail": self.detail,
        }


@dataclass
class Fsg4B3CounterRecorder:
    """Record only explicitly instrumented events; no profiler hooks are used."""

    events: list[Fsg4B3CounterEvent] = field(default_factory=list)
    retain_events: bool = True
    _direct_counts: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in COUNTER_NAMES}
    )

    def __post_init__(self) -> None:
        if set(self._direct_counts) != set(COUNTER_NAMES):
            raise ValueError("FSG4/B3 direct counter inventory differs")
        if self.events and all(value == 0 for value in self._direct_counts.values()):
            for event in self.events:
                event.validate()
                self._direct_counts[event.counter] += event.amount

    def add(self, counter: str, *, amount: int = 1, detail: str) -> None:
        if counter not in COUNTER_NAMES or amount <= 0 or not detail:
            raise ValueError("FSG4/B3 counter event differs")
        self._direct_counts[counter] += amount
        if not self.retain_events:
            return
        event = Fsg4B3CounterEvent(
            ordinal=len(self.events),
            counter=counter,
            amount=amount,
            detail=detail,
        )
        event.validate()
        self.events.append(event)

    def counts(self) -> dict[str, int]:
        if not self.retain_events:
            return dict(self._direct_counts)
        counts = {name: 0 for name in COUNTER_NAMES}
        for ordinal, event in enumerate(self.events):
            event.validate()
            if event.ordinal != ordinal:
                raise ValueError("FSG4/B3 counter event ordinal differs")
            counts[event.counter] += event.amount
        if counts != self._direct_counts:
            raise ValueError("FSG4/B3 direct counter projection differs")
        return counts


@dataclass(frozen=True)
class Fsg4B3CounterSnapshot:
    """Replayable B2 structural counter result, deliberately excluding speedup."""

    counts_by_name: tuple[tuple[str, int], ...]
    semantic_hash: str
    worker_result_sha256: str
    provider_core_call_count: int
    provider_compute_bounds_call_count: int
    provider_update_bounds_call_count: int
    fallback_dispatch_count: int
    environment_admitted: bool
    configuration: str = "B2"
    mode: str = "control"
    performance_claimed: bool = False
    schema_version: str = FSG4_B3_COUNTER_SCHEMA

    @property
    def counts(self) -> dict[str, int]:
        return dict(self.counts_by_name)

    def gate_failures(self) -> tuple[str, ...]:
        """Return deterministic failure details instead of hiding mismatches."""

        counts = self.counts
        failures: list[str] = []
        if self.schema_version != FSG4_B3_COUNTER_SCHEMA:
            failures.append("schema")
        if self.configuration not in EXPECTED_FIXED_COUNTERS or self.mode != "control":
            failures.append("configuration-mode")
        if self.performance_claimed is not False:
            failures.append("performance-claim")
        if self.environment_admitted is not True:
            failures.append("environment")
        if len(self.semantic_hash) != 64 or len(self.worker_result_sha256) != 64:
            failures.append("digest-length")
        if tuple(name for name, _value in self.counts_by_name) != tuple(
            sorted(COUNTER_NAMES)
        ) or set(counts) != set(COUNTER_NAMES):
            failures.append("counter-inventory")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in counts.values()
        ):
            failures.append("counter-type")
        for name, expected in EXPECTED_FIXED_COUNTERS.get(
            self.configuration, {}
        ).items():
            observed = counts.get(name)
            if observed != expected:
                failures.append(f"{name}:expected={expected}:observed={observed}")
        for name in (
            "tensor_content_hash_count",
            "gpu_tensor_content_hash_count",
            "typed_validate_call_count",
            "stable_hash_call_count",
        ):
            if counts.get(name, 0) <= 0:
                failures.append(f"{name}:not-positive")
        if any(
            value != 0
            for value in (
                self.provider_core_call_count,
                self.provider_compute_bounds_call_count,
                self.provider_update_bounds_call_count,
                self.fallback_dispatch_count,
            )
        ):
            failures.append("provider-or-fallback")
        return tuple(failures)

    def validate(self) -> None:
        failures = self.gate_failures()
        if failures:
            raise ValueError(
                f"FSG4/B3 {self.configuration} counter snapshot gate failed: "
                + ",".join(failures)
            )

    def to_dict(self) -> dict[str, object]:
        self.validate()
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "configuration": self.configuration,
            "mode": self.mode,
            "counts": self.counts,
            "semantic_hash": self.semantic_hash,
            "worker_result_sha256": self.worker_result_sha256,
            "provider_core_call_count": self.provider_core_call_count,
            "provider_compute_bounds_call_count": (
                self.provider_compute_bounds_call_count
            ),
            "provider_update_bounds_call_count": (
                self.provider_update_bounds_call_count
            ),
            "fallback_dispatch_count": self.fallback_dispatch_count,
            "environment_admitted": self.environment_admitted,
            "performance_claimed": self.performance_claimed,
        }
        payload["snapshot_hash"] = _canonical_hash(payload)
        return payload


def fsg4_b3_counter_snapshot_from_dict(
    value: Mapping[str, object],
) -> Fsg4B3CounterSnapshot:
    """Parse and verify a serialized explicit-counter snapshot."""

    expected = {
        "schema_version",
        "configuration",
        "mode",
        "counts",
        "semantic_hash",
        "worker_result_sha256",
        "provider_core_call_count",
        "provider_compute_bounds_call_count",
        "provider_update_bounds_call_count",
        "fallback_dispatch_count",
        "environment_admitted",
        "performance_claimed",
        "snapshot_hash",
    }
    if set(value) != expected:
        raise ValueError("FSG4/B3 counter snapshot fields differ")
    payload = dict(value)
    claimed_hash = payload.pop("snapshot_hash")
    if claimed_hash != _canonical_hash(payload):
        raise ValueError("FSG4/B3 counter snapshot hash differs")
    raw_counts = value["counts"]
    if not isinstance(raw_counts, Mapping):
        raise TypeError("FSG4/B3 counter snapshot counts differ")
    snapshot = Fsg4B3CounterSnapshot(
        counts_by_name=tuple(
            sorted(
                (str(name), _integer(count, f"counter {name}"))
                for name, count in raw_counts.items()
            )
        ),
        semantic_hash=str(value["semantic_hash"]),
        worker_result_sha256=str(value["worker_result_sha256"]),
        provider_core_call_count=_integer(
            value["provider_core_call_count"], "provider core count"
        ),
        provider_compute_bounds_call_count=_integer(
            value["provider_compute_bounds_call_count"], "provider compute count"
        ),
        provider_update_bounds_call_count=_integer(
            value["provider_update_bounds_call_count"], "provider update count"
        ),
        fallback_dispatch_count=_integer(
            value["fallback_dispatch_count"], "fallback count"
        ),
        environment_admitted=value["environment_admitted"] is True,
        configuration=str(value["configuration"]),
        mode=str(value["mode"]),
        performance_claimed=value["performance_claimed"] is True,
        schema_version=str(value["schema_version"]),
    )
    snapshot.validate()
    return snapshot


def events_from_rows(
    rows: Sequence[Mapping[str, object]],
) -> tuple[Fsg4B3CounterEvent, ...]:
    """Parse an ordered event journal used to independently rebuild counters."""

    events = tuple(
        Fsg4B3CounterEvent(
            ordinal=_integer(row["ordinal"], "event ordinal"),
            counter=str(row["counter"]),
            amount=_integer(row["amount"], "event amount"),
            detail=str(row["detail"]),
        )
        for row in rows
    )
    recorder = Fsg4B3CounterRecorder(events=list(events))
    recorder.counts()
    return events


__all__ = [
    "COUNTER_NAMES",
    "EXPECTED_B2_FIXED_COUNTERS",
    "EXPECTED_B3A_FIXED_COUNTERS",
    "EXPECTED_B3B_FIXED_COUNTERS",
    "EXPECTED_FIXED_COUNTERS",
    "events_from_rows",
    "FSG4_B3_COUNTER_SCHEMA",
    "Fsg4B3CounterEvent",
    "Fsg4B3CounterRecorder",
    "Fsg4B3CounterSnapshot",
    "fsg4_b3_counter_snapshot_from_dict",
]
