"""Deterministic tests for MR0 explicit-event budget derivation."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import copy

import pytest

from boundflow.runtime.mr0_explicit_event_budget import (
    MR0_BUDGETS,
    MR0_GROUP_COUNT,
    MR0_ORDERS,
    canonical_hash,
    derive_budget_row,
    derive_summary,
    validate_budget_row,
)


def _row(budget: int, ratio: float) -> dict[str, object]:
    return derive_budget_row(
        budget=budget,
        control_ms=[1.0] * MR0_GROUP_COUNT,
        instrumented_ms=[ratio] * MR0_GROUP_COUNT,
    )


def _workers(ratio: float) -> list[dict[str, object]]:
    return [
        {
            "run_ordinal": ordinal,
            "order": MR0_ORDERS[ordinal],
            "semantic_admitted": True,
            "stream_admitted": True,
            "performance_claimed": False,
            "budget_rows": [_row(budget, ratio) for budget in MR0_BUDGETS],
        }
        for ordinal in range(5)
    ]


def test_budget_row_round_trip_and_record_count() -> None:
    row = _row(17, 1.04)
    assert validate_budget_row(row) == row
    assert row["logical_event_records_per_group"] == 3402
    assert row["event_objects"] == 36


def test_budget_row_rejects_resigned_ratio_tamper() -> None:
    row = _row(17, 1.04)
    row["overhead_ratio"] = 1.0
    row.pop("row_hash")
    row["row_hash"] = canonical_hash(row)
    with pytest.raises(ValueError, match="derivation"):
        validate_budget_row(row)


def test_summary_opens_only_mr1_correctness_when_all_gates_pass() -> None:
    summary = derive_summary(_workers(1.04))
    assert summary["mr0_admitted"] is True
    assert summary["mr1_internal_boundary_correctness_open"] is True
    assert summary["same_solver_open"] is False
    assert summary["r2_open"] is False
    assert summary["performance_claimed"] is False


def test_summary_fails_closed_on_geomean_or_worst_worker() -> None:
    workers = _workers(1.04)
    workers[0]["budget_rows"] = [_row(budget, 1.09) for budget in MR0_BUDGETS]
    summary = derive_summary(workers)
    assert summary["mr0_admitted"] is False
    assert summary["gates"]["worst_worker"] is False
    assert summary["mr1_internal_boundary_correctness_open"] is False
    assert summary["verdict"] == "VALIDATED-NO-GO-MR0-EXPLICIT-EVENT-BUDGET"


def test_summary_rejects_worker_order_and_semantic_tamper() -> None:
    workers = _workers(1.01)
    workers[0]["order"] = "IC"
    with pytest.raises(ValueError, match="worker admission"):
        derive_summary(workers)
    workers = _workers(1.01)
    workers[0]["semantic_admitted"] = False
    with pytest.raises(ValueError, match="worker admission"):
        derive_summary(workers)


def test_summary_hash_rejects_outer_resigning_by_recomputation() -> None:
    summary = derive_summary(_workers(1.01))
    tampered = copy.deepcopy(summary)
    tampered["same_solver_open"] = True
    tampered.pop("summary_hash")
    tampered["summary_hash"] = canonical_hash(tampered)
    assert tampered != derive_summary(_workers(1.01))
