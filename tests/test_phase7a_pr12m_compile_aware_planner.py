"""Contracts for PR-12M compile/cache-aware backend selection."""

import pytest

from boundflow.planner.execution_candidate import BackendVariant, OperatorFamily
from boundflow.planner.fused_crown_backend import (
    CompileAwareBackendObservation,
    CompileAwareFusedCrownPlanner,
    CompileAwareQueryPolicy,
)


def _observation(
    backend: BackendVariant,
    *,
    warm: float,
    peak: int,
    fresh: float = 0.0,
    disk: float | None = 0.0,
) -> CompileAwareBackendObservation:
    return CompileAwareBackendObservation(
        case_id="cal-linear",
        family=OperatorFamily.LINEAR,
        backend=backend,
        boundary_bytes=4096,
        region_count=1,
        warm_latency_ms=warm,
        peak_allocated_bytes=peak,
        fresh_setup_ms=fresh,
        disk_setup_ms=disk,
        eligible=True,
        correct=True,
    )


def _planner() -> CompileAwareFusedCrownPlanner:
    return CompileAwareFusedCrownPlanner.fit(
        [
            _observation(BackendVariant.PYTORCH_EAGER, warm=2.0, peak=4000),
            _observation(BackendVariant.PYTORCH_CHUNKED, warm=1.5, peak=2500),
            _observation(
                BackendVariant.TVM_FUSED_TIR,
                warm=1.0,
                peak=1500,
                fresh=100.0,
                disk=10.0,
            ),
        ]
    )


def _decide(
    planner: CompileAwareFusedCrownPlanner,
    *,
    budget: int,
    reuse: int,
    memory_hit: float = 0.0,
    disk_hit: float = 0.0,
):
    return planner.decide(
        family=OperatorFamily.LINEAR,
        boundary_bytes=4096,
        region_count=1,
        budget_bytes=budget,
        eligible_backends=(
            BackendVariant.PYTORCH_EAGER,
            BackendVariant.PYTORCH_CHUNKED,
            BackendVariant.TVM_FUSED_TIR,
        ),
        policy=CompileAwareQueryPolicy(
            expected_reuse_queries=reuse,
            memory_cache_hit_probability=memory_hit,
            disk_cache_hit_probability=disk_hit,
        ),
    )


def test_compile_cost_changes_backend_across_reuse_regimes() -> None:
    planner = _planner()

    cold = _decide(planner, budget=10_000, reuse=1)
    repeated = _decide(planner, budget=10_000, reuse=1024)

    assert cold.backend == BackendVariant.PYTORCH_CHUNKED
    assert repeated.backend == BackendVariant.TVM_FUSED_TIR
    assert repeated.to_dict()["policy"]["expected_reuse_queries"] == 1024


def test_memory_budget_precedes_amortized_latency() -> None:
    decision = _decide(_planner(), budget=2000, reuse=1)

    assert decision.backend == BackendVariant.TVM_FUSED_TIR
    assert decision.selected_budget_feasible
    assert decision.reason == "lowest_risk_amortized_latency_within_budget"


def test_disk_cache_probability_uses_disk_setup_and_missing_disk_is_risk() -> None:
    planner = _planner()
    decision = _decide(planner, budget=10_000, reuse=8, disk_hit=1.0)
    prediction = decision.predictions[BackendVariant.TVM_FUSED_TIR.value]

    assert prediction["expected_setup_ms"] == 10.0
    assert not prediction["disk_cache_falls_back_to_fresh"]


def test_policy_rejects_invalid_probability_simplex() -> None:
    with pytest.raises(ValueError, match="sum to at most one"):
        _decide(_planner(), budget=10_000, reuse=1, memory_hit=0.7, disk_hit=0.4)
