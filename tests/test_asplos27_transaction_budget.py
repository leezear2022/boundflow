"""Tests for the explicit-transaction ASPLOS'27 research budget."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any, cast

import pytest

from boundflow.runtime.asplos27_transaction_budget import (
    DEFAULT_AXIS_POLICIES,
    derive_transaction_budgets,
)

WORKERS = Path("artifacts/asplos27-s0-transactions") / (
    "official-b0-five-pair-v1/worker_runs.jsonl"
)


def _records() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in WORKERS.read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_formal_transaction_budget_opens_research_implementation_only() -> None:
    report = derive_transaction_budgets(_records(), repeats=5)

    assert report["status"] == "s0-transaction-budget-research-route-open"
    assert report["profile_count"] == 10
    assert report["s1_implementation_open"] is True
    assert report["s1_performance_gate_open"] is False
    assert report["integration_overhead_share"] == 0.0
    assert report["performance_claimed"] is False
    workload_rows = cast(list[dict[str, Any]], report["workloads"])
    workloads = {row["workload_id"]: row for row in workload_rows}
    assert workloads["cifar10_resnet:000"][
        "projected_speedup_hypothesis"
    ] == pytest.approx(12.5622, rel=1e-4)
    assert workloads["mnistfc:2"]["projected_speedup_hypothesis"] == pytest.approx(
        11.6566, rel=1e-4
    )
    assert all(row["all_axis_targets_validated"] is False for row in workloads.values())


def test_transaction_budget_axes_and_unresolved_share_close() -> None:
    report = derive_transaction_budgets(_records(), repeats=5)

    for workload in cast(list[dict[str, Any]], report["workloads"]):
        resolved_share = sum(axis["baseline_share"] for axis in workload["axes"])
        assert resolved_share + workload["unresolved_share"] == pytest.approx(1.0)
        assert workload["unresolved_share"] < 0.01
        assert workload["required_uniform_resolved_speedup"] > 10.0


def test_transaction_budget_rejects_unvalidated_source_summary() -> None:
    records = _records()
    profile = next(record for record in records if record["mode"] == "profile")
    profile["transaction_summary"]["mechanism_coverage_share"] = 1.0

    with pytest.raises(ValueError, match="semantic summary differs"):
        derive_transaction_budgets(records, repeats=5)


def test_transaction_budget_rejects_duplicate_axis_owner() -> None:
    duplicate = replace(
        DEFAULT_AXIS_POLICIES[0],
        axis_id="duplicate",
        exact_categories=("bound_core",),
        category_prefixes=(),
    )

    with pytest.raises(ValueError, match="owners"):
        derive_transaction_budgets(
            _records(), repeats=5, policies=DEFAULT_AXIS_POLICIES + (duplicate,)
        )


def test_transaction_budget_charges_integration_overhead_explicitly() -> None:
    report = derive_transaction_budgets(
        _records(), repeats=5, integration_overhead_share=0.03
    )

    assert report["integration_overhead_share"] == 0.03
    assert report["s1_implementation_open"] is False
    assert all(
        row["projected_speedup_hypothesis"] < 10.0
        for row in cast(list[dict[str, Any]], report["workloads"])
    )
