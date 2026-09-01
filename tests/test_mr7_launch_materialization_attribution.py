"""CPU contracts for MR7 host/device attribution."""

# pylint: disable=missing-function-docstring,too-few-public-methods

from __future__ import annotations

from types import SimpleNamespace

import pytest

from boundflow.runtime.mr7_launch_materialization_attribution import (
    MARKER_PREFIX,
    MR7HostLedger,
    extract_device_events,
    extract_device_marker_totals,
    required_region_speedup,
    validate_host_receipt,
)


class _DeviceType:
    def __init__(self, name: str) -> None:
        self.name = name


def test_mr7_nested_host_spans_close_against_outer() -> None:
    ledger = MR7HostLedger()
    with ledger.span("admission_handoff"):
        with ledger.span("layout_materialization"):
            sum(range(20))
    known = sum(ledger.category_ns.values())
    receipt = ledger.receipt(outer_ns=known + 1000)
    validate_host_receipt(receipt)
    assert receipt["closure_error_ratio"] == 0.0
    assert receipt["category_ns"]["optimizer_and_residual"] == 1000


def test_mr7_device_events_require_explicit_parent_marker() -> None:
    marker = SimpleNamespace(name=f"{MARKER_PREFIX}forward.C1.03", cpu_parent=None)
    cpu_launch = SimpleNamespace(name="tvm_ffi.launch", cpu_parent=marker)
    kernel = SimpleNamespace(
        device_type=_DeviceType("CUDA"),
        is_user_annotation=False,
        cpu_parent=cpu_launch,
        name="mr7_dense_kernel",
        device_time_total=7.5,
        id=11,
        device_resource_id=19,
    )
    marker_total = SimpleNamespace(
        device_type=_DeviceType("CUDA"),
        is_user_annotation=True,
        name=f"{MARKER_PREFIX}forward.C1.03",
        device_time_total=7.5,
    )
    rows = extract_device_events((kernel,))
    assert len(rows) == 1
    assert rows[0].to_dict()["duration_ns"] == 7500
    assert extract_device_marker_totals((marker_total,)) == {"forward.C1.03": 7500}


def test_mr7_device_events_reject_unattributed_input() -> None:
    kernel = SimpleNamespace(
        device_type=_DeviceType("CUDA"),
        is_user_annotation=False,
        cpu_parent=None,
        name="unowned_kernel",
        device_time_total=1.0,
        id=3,
        device_resource_id=1,
    )
    with pytest.raises(ValueError, match="no device event"):
        extract_device_events((kernel,))


def test_mr7_amdahl_gate_is_fail_closed() -> None:
    assert required_region_speedup(share=0.20, target=1.107412) == pytest.approx(
        1.9416290291
    )
    assert required_region_speedup(share=0.05, target=1.107412) is None
    with pytest.raises(ValueError, match="Amdahl"):
        required_region_speedup(share=1.0, target=1.1)
