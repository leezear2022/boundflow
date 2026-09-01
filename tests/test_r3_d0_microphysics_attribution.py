"""Unit gates for replayable R3-D0 event accounting and routing."""

from __future__ import annotations

from copy import deepcopy

import pytest

from boundflow.runtime.r3_d0_microphysics_attribution import (
    R3D0ProfilerEventV1,
    derive_pair_route,
    derive_worker_ledger,
    event_from_dict,
)


def _event(
    ordinal: int,
    kind: str,
    start: int,
    end: int,
    *,
    phase: str,
    family: str,
    method: str = "correlation_parent",
) -> R3D0ProfilerEventV1:
    return R3D0ProfilerEventV1(
        ordinal=ordinal,
        kind=kind,
        name=family,
        phase=phase,
        family=family,
        start_ns=start,
        end_ns=end,
        correlation_id=ordinal,
        stream_id=-1 if kind == "marker" else 7,
        attribution_method="explicit_marker" if kind == "marker" else method,
        marker_ordinal=0,
    )


def _ledger(mode: str, wall: int) -> dict[str, object]:
    events = (
        _event(0, "marker", 1_000, wall + 1_000, phase="wrapper", family="wrapper"),
        _event(1, "marker", 2_000, wall, phase="forward", family="symbol-a"),
        _event(
            2,
            "cuda_kernel",
            10_000,
            10_000 + wall // 4,
            phase="forward",
            family="symbol-a",
        ),
        _event(
            3,
            "cuda_kernel",
            10_000 + wall // 4,
            10_000 + wall // 2,
            phase="backward",
            family="symbol-b",
        ),
    )
    return derive_worker_ledger(
        events,
        mode=mode,
        unprofiled_median_ns=wall,
        profiled_host_wall_ns=wall,
        cuda_event_elapsed_ns=wall // 2,
    )


def test_r3d0_event_round_trip_and_union_accounting() -> None:
    ledger = _ledger("candidate", 100_000_000)
    assert ledger["kernel_union_ns"] == 50_000_000
    assert ledger["event_count"] == 4
    assert len(ledger["event_payload_hash"]) == 64
    assert ledger["kernel_overlap_ns"] == 0
    assert ledger["profiled_host_residual_ns"] == 50_000_000
    assert ledger["host_residual_ns"] == 50_000_000
    assert ledger["calibration_admitted"] is True
    event = _event(0, "marker", 1, 5, phase="wrapper", family="wrapper")
    assert event_from_dict(event.to_dict()) == event


def test_r3d0_pair_route_recomputes_target_and_does_not_admit_semantics() -> None:
    native = _ledger("native", 96_000_000)
    candidate = _ledger("candidate", 120_000_000)
    route = derive_pair_route(native, candidate)
    assert route["target_candidate_ns"] == 80_000_000.0
    assert route["required_saving_ns"] == 40_000_000.0
    assert route["graph_capture_admitted"] is False
    assert all(
        row["semantic_closure_admitted"] is False for row in route["family_routes"]
    )


@pytest.mark.parametrize(
    "field,replacement",
    (("duration_ns", 99), ("attribution_method", "unattributed"), ("phase", "")),
)
def test_r3d0_event_tamper_fails_closed(field: str, replacement: object) -> None:
    payload = _event(
        0, "cuda_kernel", 1, 5, phase="forward", family="symbol-a"
    ).to_dict()
    changed = deepcopy(payload)
    changed[field] = replacement
    with pytest.raises(ValueError, match="R3-D0"):
        event_from_dict(changed)


def test_r3d0_calibration_rejects_excessive_containment() -> None:
    events = [_event(0, "marker", 1_000, 10_001_000, phase="wrapper", family="wrapper")]
    events.extend(
        _event(
            ordinal,
            "cuda_kernel",
            10_000 + ordinal * 10_000,
            15_000 + ordinal * 10_000,
            phase="forward",
            family="symbol-a",
            method="marker_containment",
        )
        for ordinal in range(1, 21)
    )
    ledger = derive_worker_ledger(
        tuple(events),
        mode="candidate",
        unprofiled_median_ns=10_000_000,
        profiled_host_wall_ns=10_000_000,
        cuda_event_elapsed_ns=195_000,
    )
    assert ledger["fallback_count"] == 20
    assert ledger["calibration_admitted"] is False
