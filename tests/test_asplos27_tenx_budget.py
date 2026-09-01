"""Tests for ASPLOS'27 attribution admission and 10x budgeting."""

# pylint: disable=missing-function-docstring

from dataclasses import replace
import json
import math
from pathlib import Path

import pytest

from boundflow.runtime.asplos27_tenx_budget import (
    BudgetBucket,
    ClaimMode,
    DirectCumulativeObservation,
    EvidenceScope,
    TenXBudget,
    derive_fsg1_diagnostic_budgets,
    derive_fsg1_transaction_inventory,
    summarize_tenx_budget,
    validate_direct_observation_ledger,
)


def _budget() -> TenXBudget:
    return TenXBudget(
        scope_id="official-b0:workload-0:complete-query",
        claim_mode=ClaimMode.SOLVED_QUERY_TTV,
        target_speedup=10.0,
        integration_overhead_share=0.01,
        semantic_coverage_share=0.98,
        semantic_unclassified_share=0.02,
        fixed_trajectory_complete=True,
        solved_query_complete=True,
        buckets=(
            BudgetBucket(
                "operator",
                0.62,
                20.0,
                "coarse_tir_regions",
                EvidenceScope.SAME_SOLVER_REGION,
            ),
            BudgetBucket(
                "solver_runtime",
                0.38,
                8.0,
                "compiled_state_machine",
                EvidenceScope.COMPLETE_QUERY,
            ),
        ),
    )


def test_tenx_budget_keeps_coverage_and_feasibility_separate() -> None:
    summary = summarize_tenx_budget(_budget())

    assert summary["attribution_admitted"] is True
    assert summary["projected_runtime_fraction"] == pytest.approx(0.0885)
    assert summary["projected_speedup"] == pytest.approx(11.299435028248588)
    assert summary["tenx_feasible_hypothesis"] is True
    assert summary["performance_claimed"] is False


def test_tenx_budget_rejects_incomplete_semantic_transactions() -> None:
    summary = summarize_tenx_budget(
        replace(
            _budget(),
            semantic_coverage_share=0.69,
            semantic_unclassified_share=0.31,
        )
    )

    assert summary["coverage_passed"] is False
    assert summary["unclassified_passed"] is False
    assert summary["attribution_admitted"] is False
    assert summary["tenx_feasible_hypothesis"] is True


def test_tenx_budget_rejects_nonexclusive_buckets() -> None:
    with pytest.raises(ValueError, match="buckets do not close"):
        summarize_tenx_budget(
            replace(
                _budget(),
                buckets=(replace(_budget().buckets[0], baseline_share=0.70),),
            )
        )


def test_tenx_budget_reports_impossible_immutable_share() -> None:
    budget = replace(
        _budget(),
        integration_overhead_share=0.0,
        buckets=(
            replace(_budget().buckets[0], target_speedup=100.0),
            replace(_budget().buckets[1], target_speedup=1.0),
        ),
    )
    summary = summarize_tenx_budget(budget)

    assert summary["tenx_feasible_hypothesis"] is False
    assert summary["required_uniform_speedup"] is None
    assert summary["projected_speedup"] < 3.0


def test_ttv_mode_requires_a_solved_complete_query() -> None:
    summary = summarize_tenx_budget(replace(_budget(), solved_query_complete=False))

    assert summary["scope_complete"] is False
    assert summary["attribution_admitted"] is False


def test_fsg1_diagnostic_exposes_phase_semantic_gap_and_amdahl_ceiling() -> None:
    artifact = Path(
        "artifacts/fsg1-official-control/"
        "resnet2b-mnistfc2-rtx4060-five-repeat-v1/closure.json"
    )
    closure = json.loads(artifact.read_text(encoding="utf-8"))

    report = derive_fsg1_diagnostic_budgets(
        closure, operator_target_speedup=12.795107698179335
    )

    assert report["run_count"] == 10
    assert report["status"] == "s0-attribution-not-admitted"
    resnet = [row for row in report["runs"] if "cifar10_resnet" in row["scope_id"]]
    assert len(resnet) == 5
    assert all(row["semantic_unclassified_share"] > 0.30 for row in resnet)
    assert all(row["attribution_admitted"] is False for row in resnet)
    assert all(row["operator_infinite_speedup_ceiling"] < 3.0 for row in resnet)
    assert report["performance_claimed"] is False


def test_direct_observation_ledger_forbids_implied_cross_scope_aggregate() -> None:
    observations = (
        DirectCumulativeObservation(
            "b3-vs-b0",
            "same-solver-fixed-prefix",
            EvidenceScope.FIXED_PREFIX,
            "B0",
            "B3",
            0.9100012637918488,
            "digest-a",
            True,
        ),
        DirectCumulativeObservation(
            "cibc-graph",
            "standalone-resnet2b-ibp",
            EvidenceScope.STANDALONE_GRAPH,
            "pytorch-cuda-graph",
            "cibc-cuda-graph",
            2.456310282102286,
            "digest-b",
            True,
        ),
    )

    ledger = validate_direct_observation_ledger(observations)

    assert ledger["observation_count"] == 2
    assert "geomean" not in ledger
    assert "product" not in ledger
    assert ledger["performance_claimed"] is False


def test_fsg1_transaction_inventory_closes_topology_but_not_mechanisms() -> None:
    artifact = Path(
        "artifacts/fsg1-official-control/"
        "resnet2b-mnistfc2-rtx4060-five-repeat-v1/worker_runs.jsonl"
    )
    records = [
        json.loads(line)
        for line in artifact.read_text(encoding="utf-8").splitlines()
        if line
    ]

    inventory = derive_fsg1_transaction_inventory(records)

    assert inventory["run_count"] == 10
    assert inventory["topology_context_closed_count"] == 10
    assert inventory["mechanism_admitted_count"] == 5
    assert inventory["status"] == "s0-transaction-mechanism-not-admitted"
    resnet = [
        row for row in inventory["runs"] if row["workload_id"] == "cifar10_resnet:000"
    ]
    assert all(row["mechanism_unresolved_share"] > 0.30 for row in resnet)
    assert all(row["topology_unclassified_share"] == 0.0 for row in resnet)
    assert all(
        any(
            "transition:initial_crown->beta_split" in key
            for key in row["transaction_ns"]
        )
        for row in resnet
    )


def test_budget_hash_is_finite_json_compatible() -> None:
    summary = summarize_tenx_budget(_budget())
    assert math.isfinite(summary["projected_speedup"])
    assert len(summary["budget_hash"]) == 64
