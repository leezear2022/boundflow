"""Deterministic gates for R3-3 isolated microphysics attribution."""

# pylint: disable=missing-function-docstring,too-many-arguments

from __future__ import annotations

from dataclasses import replace

import pytest

from boundflow.runtime.r3_3_isolated_attribution import (
    R33ProfilerEventV1,
    derive_ledger,
    derive_route,
    derive_route_or_stop,
    event_from_dict,
)


def _event(
    ordinal: int,
    kind: str,
    phase: str,
    start: int,
    end: int,
    *,
    stream: int = -1,
) -> R33ProfilerEventV1:
    return R33ProfilerEventV1(
        ordinal=ordinal,
        kind=kind,
        name=f"boundflow::r33attr::{phase}" if kind == "marker" else f"{phase}-k",
        phase=phase,
        start_ns=start,
        end_ns=end,
        correlation_id=ordinal + 1,
        stream_id=stream,
        attribution_method=(
            "explicit_marker" if kind == "marker" else "correlation_parent"
        ),
        marker_ordinal=ordinal if kind == "marker" else 0,
    )


def _admitted_ledger() -> dict[str, object]:
    rows = (
        _event(0, "cuda_kernel", "calibration", 100, 200, stream=7),
        _event(1, "marker", "wrapper", 1_000, 2_000),
        _event(2, "marker", "prepare-executor", 1_000, 1_050),
        _event(3, "marker", "autograd-apply", 1_050, 1_400),
        _event(4, "marker", "forward-ffi", 1_100, 1_300),
        _event(5, "cuda_kernel", "forward-ffi", 1_150, 1_200, stream=7),
        _event(6, "marker", "autograd-grad", 1_400, 2_000),
        _event(7, "marker", "backward-ffi", 1_500, 1_700),
        _event(8, "cuda_kernel", "backward-ffi", 1_550, 1_600, stream=7),
    )
    return derive_ledger(
        rows,
        unprofiled_median_ns=1_000,
        profiled_cuda_event_ns=1_000,
        calibration_cuda_event_ns=100,
    )


def test_event_round_trip_and_duration_tamper_fail_closed() -> None:
    event = _event(0, "marker", "wrapper", 10, 20)
    assert event_from_dict(event.to_dict()) == event
    payload = event.to_dict()
    payload["duration_ns"] = 11
    with pytest.raises(ValueError, match="derivation"):
        event_from_dict(payload)


def test_ledger_is_mutually_exclusive_conservative_and_admitted() -> None:
    ledger = _admitted_ledger()
    assert ledger["attribution_admitted"] is True
    assert not ledger["admission_failures"]
    assert sum(ledger["bucket_ns"].values()) == 1_000
    assert ledger["conservation_error_ns"] == 0
    assert ledger["stream_ids"] == [7]


def test_profiler_perturbation_fails_closed_with_explicit_reason() -> None:
    rows = [event_from_dict(row) for row in _events_from_ledger_fixture()]
    perturbed = derive_ledger(
        rows,
        unprofiled_median_ns=1_000,
        profiled_cuda_event_ns=1_201,
        calibration_cuda_event_ns=100,
    )
    assert perturbed["attribution_admitted"] is False
    assert perturbed["admission_failures"] == ["profiler-perturbation"]
    stopped = derive_route_or_stop([perturbed] * 5)
    assert stopped["route"] == "STOP"
    assert stopped["route_reason"] == "attribution-quality"
    assert stopped["failure_counts"] == {"profiler-perturbation": 5}
    assert stopped["diagnostic_shares_admitted"] is False


def _events_from_ledger_fixture() -> list[dict[str, object]]:
    # Rebuild the same independent event rows used by the admitted fixture.
    rows = (
        _event(0, "cuda_kernel", "calibration", 100, 200, stream=7),
        _event(1, "marker", "wrapper", 1_000, 2_000),
        _event(2, "marker", "prepare-executor", 1_000, 1_050),
        _event(3, "marker", "autograd-apply", 1_050, 1_400),
        _event(4, "marker", "forward-ffi", 1_100, 1_300),
        _event(5, "cuda_kernel", "forward-ffi", 1_150, 1_200, stream=7),
        _event(6, "marker", "autograd-grad", 1_400, 2_000),
        _event(7, "marker", "backward-ffi", 1_500, 1_700),
        _event(8, "cuda_kernel", "backward-ffi", 1_550, 1_600, stream=7),
    )
    return [row.to_dict() for row in rows]


def test_route_priority_and_quality_boundary() -> None:
    base = _admitted_ledger()
    ledgers = []
    for _ in range(5):
        row = dict(base)
        row["bucket_share"] = {
            "forward_kernel_union": 0.25,
            "backward_kernel_union": 0.25,
            "bridge_launch_idle": 0.15,
            "autograd_allocation": 0.25,
            "other_explained": 0.10,
            "unexplained": 0.0,
        }
        ledgers.append(row)
    assert derive_route(ledgers)["route"] == "KERNEL"
    rejected = dict(ledgers[0])
    rejected["attribution_admitted"] = False
    rejected["admission_failures"] = ["calibration-residual"]
    stopped = derive_route_or_stop([rejected, *ledgers[1:]])
    assert stopped["route"] == "STOP"
    assert stopped["failed_run_ordinals"] == [0]


def test_event_rejects_local_path_leak() -> None:
    event = _event(0, "marker", "wrapper", 10, 20)
    with pytest.raises(ValueError, match="local path"):
        replace(event, name="/home/lee/private").validate()
