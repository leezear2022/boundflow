"""Contracts for FSG4/B4-0 raw profiler attribution."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from types import SimpleNamespace

import pytest

from boundflow.runtime.fsg4_b4_kernel_attribution import (
    B3_CORE_QUERY_SHARE,
    B3_CROWN14_QUERY_SHARE,
    B3_OPTIMIZER_QUERY_SHARE,
    B3_QUERY_RATIO_TO_B0,
    B4ProfilerEvent,
    b4_profiler_event_from_dict,
    derive_b4_attribution,
    extract_profiler_events,
    infinite_query_speedup,
    query_speedup,
    required_region_speedup,
)


def _event(
    *,
    event_id: int,
    name: str,
    device: str,
    parent: object | None = None,
    duration_us: float = 1.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=event_id,
        name=name,
        device_type=device,
        device_index=0 if device == "cuda" else -1,
        device_resource_id=7 if device == "cuda" else 1,
        thread=1,
        cpu_parent=parent,
        time_range=SimpleNamespace(start=2.0, end=2.0 + duration_us),
        cpu_time_total=duration_us,
        device_time_total=duration_us,
        input_shapes=[[6, 1, 64], []],
        cpu_memory_usage=16,
        device_memory_usage=32,
        is_user_annotation=False,
    )


def test_b4_amdahl_preregistered_values() -> None:
    target = 1.0 / B3_QUERY_RATIO_TO_B0
    assert infinite_query_speedup(share=B3_OPTIMIZER_QUERY_SHARE) == pytest.approx(
        1.0861667080859945
    )
    assert (
        required_region_speedup(share=B3_OPTIMIZER_QUERY_SHARE, target=target) is None
    )
    assert required_region_speedup(
        share=B3_CROWN14_QUERY_SHARE, target=target
    ) == pytest.approx(3.989702826086512)
    assert required_region_speedup(
        share=B3_CORE_QUERY_SHARE, target=target
    ) == pytest.approx(2.030218830784719)
    assert query_speedup(share=B3_CROWN14_QUERY_SHARE, region_speedup=40.0) == (
        pytest.approx(1.1326299460617053)
    )


def test_extract_profiler_events_uses_cuda_correlation_parent() -> None:
    marker = _event(
        event_id=1,
        name="boundflow::b4::optimizer.crown.00",
        device="cpu",
        duration_us=8.0,
    )
    aten = _event(
        event_id=2,
        name="aten::mul",
        device="cpu",
        parent=marker,
        duration_us=5.0,
    )
    launch = _event(
        event_id=17,
        name="cudaLaunchKernel",
        device="cpu",
        parent=aten,
        duration_us=1.0,
    )
    kernel = _event(
        event_id=17,
        name="vectorized_elementwise_kernel",
        device="cuda",
        duration_us=3.5,
    )
    rows = extract_profiler_events((marker, aten, launch, kernel))
    assert [row.event_kind for row in rows] == [
        "cpu_op",
        "cpu_op",
        "cpu_op",
        "cuda_kernel",
    ]
    assert {row.phase for row in rows} == {"optimizer.crown.00"}
    assert rows[-1].parent_name == "cudaLaunchKernel"
    assert rows[-1].duration_ns == 3500
    assert rows[-1].input_shapes == ((6, 1, 64), ())


def test_derive_b4_attribution_rebuilds_raw_kernel_summary() -> None:
    events = (
        B4ProfilerEvent(
            event_ordinal=0,
            correlation_id=1,
            event_kind="cuda_kernel",
            phase="optimizer.crown.00",
            name="kernel-a",
            parent_name="cudaLaunchKernel",
            duration_ns=3000,
            start_ns=0,
            end_ns=3000,
            device_index=0,
            stream_id=7,
            thread_id=1,
            input_shapes=((6, 1, 64),),
            cpu_memory_delta_bytes=0,
            device_memory_delta_bytes=64,
        ),
        B4ProfilerEvent(
            event_ordinal=1,
            correlation_id=2,
            event_kind="cuda_kernel",
            phase="optimizer.crown.01",
            name="kernel-a",
            parent_name="cudaLaunchKernel",
            duration_ns=4000,
            start_ns=3000,
            end_ns=7000,
            device_index=0,
            stream_id=7,
            thread_id=1,
            input_shapes=((6, 1, 64),),
            cpu_memory_delta_bytes=0,
            device_memory_delta_bytes=64,
        ),
    )
    summary = derive_b4_attribution(
        events,
        run_id="b4-0-profile-0",
        source_identity="a" * 64,
        protocol_identity="b" * 64,
        query_wall_ns=100_000,
        core_wall_ns=20_000,
    )
    assert summary["performance_claimed"] is False
    assert summary["cuda_kernel_count"] == 2
    assert summary["phase_closure"] == {
        "accounted_cuda_kernel_count": 2,
        "attributed_cuda_kernel_count": 2,
        "unattributed_cuda_kernel_count": 0,
        "attribution_method_counts": {
            "cpu_parent": 2,
            "device_marker": 0,
            "temporal_marker": 0,
            "unattributed": 0,
        },
    }
    assert summary["phase_attribution"] == {
        "optimizer.crown.00": {"kernel_count": 1, "cuda_kernel_sum_ns": 3000},
        "optimizer.crown.01": {"kernel_count": 1, "cuda_kernel_sum_ns": 4000},
    }
    assert summary["root_phase_attribution"] == {
        "optimizer": {"kernel_count": 2, "cuda_kernel_sum_ns": 7000}
    }
    assert summary["kernel_attribution"][0] == {
        "phase": "optimizer.crown.01",
        "kernel_name": "kernel-a",
        "kernel_count": 1,
        "cuda_kernel_sum_ns": 4000,
    }
    assert summary["opportunity"]["optimizer_only"]["required_region_speedup"] is None
    assert len(summary["summary_hash"]) == 64


def test_profiler_event_rejects_local_path() -> None:
    event = B4ProfilerEvent(
        event_ordinal=0,
        correlation_id=1,
        event_kind="cpu_op",
        phase="optimizer",
        name="/home/lee/private.py",
        parent_name=None,
        duration_ns=1,
        start_ns=0,
        end_ns=1,
        device_index=-1,
        stream_id=-1,
        thread_id=1,
        input_shapes=(),
        cpu_memory_delta_bytes=0,
        device_memory_delta_bytes=0,
    )
    with pytest.raises(ValueError, match="local path"):
        event.validate()


def test_profiler_event_canonical_roundtrip_rejects_field_drift() -> None:
    event = B4ProfilerEvent(
        event_ordinal=0,
        correlation_id=1,
        event_kind="cuda_kernel",
        phase="terminal_export.crown.00",
        name="kernel-a",
        parent_name="cudaLaunchKernel",
        duration_ns=3,
        start_ns=4,
        end_ns=7,
        device_index=0,
        stream_id=7,
        thread_id=1,
        input_shapes=((6, 1, 64),),
        cpu_memory_delta_bytes=0,
        device_memory_delta_bytes=32,
    )
    assert b4_profiler_event_from_dict(event.to_dict()) == event
    payload = event.to_dict()
    payload["stream_id"] = "7"
    with pytest.raises(ValueError, match="not canonical"):
        b4_profiler_event_from_dict(payload)


def test_extract_profiler_events_preserves_unattributed_cuda_kernel() -> None:
    kernel = _event(
        event_id=91,
        name="orphan_kernel",
        device="cuda",
        duration_us=2.0,
    )
    rows = extract_profiler_events((kernel,))
    assert len(rows) == 1
    assert rows[0].phase == "unattributed"
    assert rows[0].stream_id == 7


def test_extract_profiler_events_separates_cuda_annotation_from_kernel() -> None:
    marker = _event(
        event_id=1,
        name="boundflow::b4::optimizer",
        device="cpu",
        duration_us=8.0,
    )
    adam = _event(
        event_id=12,
        name="Optimizer.step#Adam.step",
        device="cpu",
        parent=marker,
        duration_us=5.0,
    )
    adam.is_user_annotation = True
    device_annotation = _event(
        event_id=12,
        name="Optimizer.step#Adam.step",
        device="cuda",
        duration_us=5.0,
    )
    device_annotation.is_user_annotation = True
    launch = _event(
        event_id=17,
        name="cudaLaunchKernel",
        device="cpu",
        parent=adam,
    )
    kernel = _event(event_id=17, name="adam_kernel", device="cuda")
    rows = extract_profiler_events((marker, adam, device_annotation, launch, kernel))
    by_name = {row.name: row for row in rows}
    assert by_name["Optimizer.step#Adam.step"].event_kind == "phase_device_total"
    assert by_name["Optimizer.step#Adam.step"].phase == "optimizer.adam"
    assert by_name["adam_kernel"].event_kind == "cuda_kernel"
    assert by_name["adam_kernel"].phase == "optimizer.adam"


def test_extract_profiler_events_temporal_fallback_is_explicit() -> None:
    marker = _event(
        event_id=1,
        name="boundflow::b4::worker",
        device="cpu",
        duration_us=8.0,
    )
    kernel = _event(
        event_id=91,
        name="async_kernel",
        device="cuda",
        duration_us=2.0,
    )
    rows = extract_profiler_events((marker, kernel))
    assert rows[-1].phase == "worker"
    assert rows[-1].attribution_method == "temporal_marker"
    assert rows[-1].parent_name == "boundflow::b4::worker"
