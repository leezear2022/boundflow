"""Deterministic contracts for NRIR49A G1 GPU attribution."""

# pylint: disable=missing-function-docstring,too-many-locals,too-many-arguments

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any
import pytest
import torch

from boundflow.domains.interval import IntervalState
from scripts.run_nrir49a_g1_gpu_attribution import (
    CHUNKS,
    CLAUSES,
    COMPLETE_SCHEMA_VERSION,
    DEFAULT_CHUNK,
    ENVIRONMENT_SCHEMA_VERSION,
    QUEUE_SCHEMA_VERSION,
    REPEAT_COUNT,
    REQUIRED_MAX_DEPTH,
    REQUIRED_NODES,
    REQUIRED_SIBLING_GROUPS,
    WORKER_SCHEMA_VERSION,
    _code_revision,
    _memory_decision,
    build_summary,
    canonical_hash,
    classify_compute_processes,
    latin_chunk_order,
    projected_scope_speedup,
    required_region_speedup,
    selected_call_geometry,
    validate_worker,
)


@dataclass(frozen=True)
class _Target:
    relu_input: str
    neuron_index: int


def _call(clause: int, chunk: int, *, device_ns: int = 400) -> dict[str, Any]:
    return {
        "scope": "chunk_queue",
        "clause": clause,
        "requested_production_chunk": DEFAULT_CHUNK,
        "effective_harness_chunk": chunk,
        "device_ns": device_ns,
        "host_wall_ns": 500,
        "output_schema": {},
    }


def _queue(
    repeat: int,
    clause: int,
    chunk: int,
    *,
    mode: str,
    stream_ns: int = 1000,
    wall_ns: int = 1000,
) -> dict[str, Any]:
    calls = [] if mode == "control" else [_call(clause, chunk)]
    row: dict[str, Any] = {
        "schema_version": QUEUE_SCHEMA_VERSION,
        "repeat_index": repeat,
        "clause": clause,
        "chunk": chunk,
        "order_position": 0,
        "mode": mode,
        "query_id": f"nrir49a:r{repeat}:c{clause}",
        "accepted_nodes": REQUIRED_NODES,
        "sibling_group_count": REQUIRED_SIBLING_GROUPS,
        "maximum_depth": REQUIRED_MAX_DEPTH,
        "worst_active_lower": -3.0 - clause,
        "queue_internal_elapsed_ns": stream_ns,
        "queue_stream_ns": stream_ns,
        "synchronized_wall_ns": wall_ns,
        "selected_device_ns": sum(int(call["device_ns"]) for call in calls),
        "selected_host_wall_ns": sum(int(call["host_wall_ns"]) for call in calls),
        "selected_call_count": len(calls),
        "baseline_allocated": 10_000_000,
        "baseline_reserved": 20_000_000,
        "peak_allocated": 100_000_000,
        "peak_reserved": 120_000_000,
        "semantics_hash": f"semantics-c{clause}-r{repeat}",
        "semantics": {"clause": clause},
        "calls": calls,
        "performance_claimed": False,
    }
    row["row_hash"] = canonical_hash(row)
    return row


def _worker(repeat: int) -> dict[str, Any]:
    profiles = [
        _queue(repeat, clause, chunk, mode="profile")
        for chunk in latin_chunk_order(repeat)
        for clause in CLAUSES
    ]
    controls = [
        _queue(repeat, clause, DEFAULT_CHUNK, mode="control") for clause in CLAUSES
    ]
    complete_queues = [
        _queue(repeat, clause, DEFAULT_CHUNK, mode="complete") for clause in CLAUSES
    ]
    complete: dict[str, Any] = {
        "schema_version": COMPLETE_SCHEMA_VERSION,
        "repeat_index": repeat,
        "chunk": DEFAULT_CHUNK,
        "stream_ns": 5000,
        "synchronized_wall_ns": 6000,
        "selected_all_device_ns": 2000,
        "selected_child_device_ns": 800,
        "baseline_allocated": 10_000_000,
        "baseline_reserved": 20_000_000,
        "peak_allocated": 150_000_000,
        "peak_reserved": 180_000_000,
        "calls": [],
        "queue_rows": complete_queues,
        "performance_claimed": False,
    }
    complete["row_hash"] = canonical_hash(complete)
    representative_profiler = None
    if repeat == 0:
        source_call = next(
            row["calls"][0]
            for row in profiles
            if row["clause"] == CLAUSES[0] and row["chunk"] == DEFAULT_CHUNK
        )
        representative_profiler = {
            "scope": (
                "representative-child-selected-crown-" "non-timing-clause2-default32"
            ),
            "excluded_from_timing_summary": True,
            "source_call_hash": canonical_hash(
                {key: value for key, value in source_call.items() if key != "device_ns"}
            ),
            "replayed_output_schema_hash": canonical_hash(source_call["output_schema"]),
            "kernel_count": 10,
            "runtime_launch_api_count": 10,
            "synchronization_api_count": 1,
            "memory_event_count": 1,
            "top_cuda_events": [],
            "top_cpu_events": [],
            "performance_claimed": False,
        }
        representative_profiler["profile_hash"] = canonical_hash(
            representative_profiler
        )
    worker: dict[str, Any] = {
        "schema_version": WORKER_SCHEMA_VERSION,
        "repeat_index": repeat,
        "chunk_order": list(latin_chunk_order(repeat)),
        "code_revision": _code_revision(),
        "environment": {
            "schema_version": ENVIRONMENT_SCHEMA_VERSION,
            "total_memory_bytes": 8_000_000_000,
        },
        "module_inventory": {},
        "profile_rows": profiles,
        "control_rows": controls,
        "complete": complete,
        "representative_profiler": representative_profiler,
        "performance_claimed": False,
    }
    worker["worker_hash"] = canonical_hash(worker)
    return worker


def test_latin_chunk_order_covers_every_chunk_at_every_position() -> None:
    rows = [latin_chunk_order(index) for index in range(REPEAT_COUNT)]
    assert all(set(row) == set(CHUNKS) for row in rows)
    for position in range(len(CHUNKS)):
        assert {row[position] for row in rows} == set(CHUNKS)
    with pytest.raises(ValueError, match="repeat index"):
        latin_chunk_order(REPEAT_COUNT)


def test_amdahl_inverse_and_infeasible_boundary() -> None:
    assert required_region_speedup(0.25, 1.20) == pytest.approx(3.0)
    assert required_region_speedup(0.20, 1.20) == pytest.approx(6.0)
    assert required_region_speedup(0.10, 1.20) is None
    assert projected_scope_speedup(0.25, 3.0) == pytest.approx(1.20)


def test_selected_geometry_closes_ragged_bytes_and_chunks() -> None:
    pre = {
        "a": IntervalState(torch.zeros(1, 10), torch.ones(1, 10)),
        "b": IntervalState(torch.zeros(1, 4), torch.ones(1, 4)),
    }
    targets = [_Target("a", index) for index in range(5)] + [
        _Target("b", index) for index in range(3)
    ]
    actual = selected_call_geometry(relu_pre=pre, targets=targets, chunk_size=2)
    assert actual["target_count"] == 8
    assert actual["relu_segment_count"] == 2
    assert actual["chunk_count"] == 5
    assert actual["one_hot_bytes"] == (5 * 10 + 3 * 4) * 4
    assert actual["index_bytes"] == 8 * 8
    assert actual["output_bytes"] == 2 * 8 * 4


def test_summary_recomputes_opportunity_amdahl_and_memory_n_a() -> None:
    workers = [_worker(repeat) for repeat in range(REPEAT_COUNT)]
    for worker in workers:
        validate_worker(worker)
    summary: Any = build_summary(workers)
    assert summary["s_queue_median"] == pytest.approx(0.4)
    assert summary["s_complete_median"] == pytest.approx(0.4)
    assert summary["r_latency_required"] < 10.0
    assert summary["decision"] == {
        "instrumentation_passed": True,
        "queue_opportunity_passed": True,
        "latency_feasible": True,
        "memory_physical_admitted": False,
        "next_route": "proceed-g2-qualification",
    }
    assert summary["memory"]["g8_memory_path"] == "n/a"


def test_worker_rejects_synchronized_outer_rehash_tamper() -> None:
    worker = _worker(0)
    tampered = deepcopy(worker)
    row = tampered["profile_rows"][0]
    row["selected_device_ns"] += 1
    row["row_hash"] = canonical_hash(
        {key: value for key, value in row.items() if key != "row_hash"}
    )
    tampered["worker_hash"] = canonical_hash(
        {key: value for key, value in tampered.items() if key != "worker_hash"}
    )
    with pytest.raises(ValueError, match="call coverage"):
        validate_worker(tampered)


def test_worker_cache_rejects_code_revision_mismatch() -> None:
    worker = _worker(0)
    worker["code_revision"]["scripts/run_nrir49a_g1_gpu_attribution.py"] = "0" * 64
    worker["worker_hash"] = canonical_hash(
        {key: value for key, value in worker.items() if key != "worker_hash"}
    )
    with pytest.raises(ValueError, match="worker envelope"):
        validate_worker(worker)


def test_worker_rejects_rehashed_profiler_source_tamper() -> None:
    worker = _worker(0)
    profiler = worker["representative_profiler"]
    profiler["source_call_hash"] = "0" * 64
    profiler["profile_hash"] = canonical_hash(
        {key: value for key, value in profiler.items() if key != "profile_hash"}
    )
    worker["worker_hash"] = canonical_hash(
        {key: value for key, value in worker.items() if key != "worker_hash"}
    )
    with pytest.raises(ValueError, match="profiler source"):
        validate_worker(worker)


def test_memory_admission_requires_real_80_percent_peak() -> None:
    rows = [{"peak_allocated": 6_000_000_000, "peak_reserved": 6_100_000_000}]
    rejected = _memory_decision(rows, 8_000_000_000)
    assert rejected["physical_memory_admitted"] is False
    assert rejected["b80_alloc"] is None
    rows[0]["peak_reserved"] = 6_500_000_000
    admitted = _memory_decision(rows, 8_000_000_000)
    assert admitted["physical_memory_admitted"] is True
    assert admitted["b80_reserved"] == 1


def test_compute_process_gate_only_allows_bounded_display_and_self() -> None:
    environment = {
        "compute_processes": {
            "returncode": 0,
            "stdout": (
                "10, /usr/bin/kwin_wayland, 7\n"
                "20, /usr/bin/python, 400\n"
                "30, /usr/bin/python, 1"
            ),
        }
    }
    inventory = classify_compute_processes(environment, own_pid=30)
    assert [row["pid"] for row in inventory["admitted"]] == [10, 30]
    assert [row["pid"] for row in inventory["rejected"]] == [20]
    environment["compute_processes"]["stdout"] = "10, /usr/bin/kwin_wayland, 65"
    assert (
        classify_compute_processes(environment, own_pid=30)["rejected"][0]["pid"] == 10
    )
