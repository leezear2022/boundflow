"""Tests for nontrivial multi-barrier PR-11 placement."""

import json

import pytest
import torch

from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.planner.materialization import MaterializationAction
from boundflow.planner.materialization_placement import (
    BarrierCost,
    PLACEMENT_SCHEMA_VERSION,
    PlacementContext,
    PlacementPolicy,
    PlacementRetryCandidate,
    StaticPlacementQuery,
    generate_static_placement_candidates,
    plan_materialization_placement,
    rank_bounded_placement_retry_candidates,
)
from boundflow.planner.materialization_placement_cost_model import (
    LATENCY_FEATURE_NAMES,
    PEAK_FEATURE_NAMES,
    PLACEMENT_COST_MODEL_SCHEMA_VERSION,
    PlacementInteractionCostModel,
)
from boundflow.planner.materialization_static_features import summarize_static_barriers
from boundflow.runtime.alpha_crown import run_alpha_crown_mlp
from boundflow.runtime.crown_ibp import (
    run_crown_ibp_mlp,
    run_crown_ibp_mlp_with_placement_retry,
)
from boundflow.runtime.materialization import trace_materializations
from boundflow.runtime.scheduler import (
    PlacementRetryExhausted,
    execute_bounded_placement_candidates_with_retry,
    execute_placement_candidates_with_retry,
)
from boundflow.runtime.task_executor import InputSpec


def _barriers() -> tuple[BarrierCost, ...]:
    return tuple(
        BarrierCost(
            barrier_id=f"relu_{index}",
            dense_persistent_bytes=60,
            structured_persistent_bytes=10,
            structured_ephemeral_bytes=20,
            dense_latency_ms=1.0,
            structured_latency_ms=4.0,
        )
        for index in range(3)
    )


def _context(*, budget: int = 150, requires_grad: bool = False) -> PlacementContext:
    return PlacementContext(
        barriers=_barriers(),
        common_persistent_bytes=10,
        memory_budget_bytes=budget,
        available_memory_bytes=budget,
        safety_margin=1.0,
        requires_grad=requires_grad,
        alpha_enabled=requires_grad,
        domain_batch_size=8,
    )


def _two_relu_mlp() -> BFTaskModule:
    torch.manual_seed(73)
    task = BoundTask(
        task_id="t0",
        kind=TaskKind.INTERVAL_IBP,
        ops=[
            TaskOp("linear", "linear1", ["input", "W1", "b1"], ["h1"]),
            TaskOp("relu", "relu1", ["h1"], ["r1"]),
            TaskOp("linear", "linear2", ["r1", "W2", "b2"], ["h2"]),
            TaskOp("relu", "relu2", ["h2"], ["r2"]),
            TaskOp("linear", "linear3", ["r2", "W3", "b3"], ["out"]),
        ],
        input_values=["input"],
        output_values=["out"],
    )
    return BFTaskModule(
        tasks=[task],
        entry_task_id="t0",
        bindings={
            "params": {
                "W1": torch.randn(6, 4, dtype=torch.float64),
                "b1": torch.randn(6, dtype=torch.float64),
                "W2": torch.randn(6, 6, dtype=torch.float64),
                "b2": torch.randn(6, dtype=torch.float64),
                "W3": torch.randn(3, 6, dtype=torch.float64),
                "b3": torch.randn(3, dtype=torch.float64),
            }
        },
    )


def _runtime_placement_context(*, budget: int) -> PlacementContext:
    barriers = tuple(
        BarrierCost(
            barrier_id=name,
            dense_persistent_bytes=60,
            structured_persistent_bytes=10,
            structured_ephemeral_bytes=20,
            dense_latency_ms=1.0,
            structured_latency_ms=4.0,
        )
        for name in ("h1", "h2")
    )
    return PlacementContext(
        barriers=barriers,
        common_persistent_bytes=10,
        memory_budget_bytes=budget,
        available_memory_bytes=budget,
        safety_margin=1.0,
        domain_batch_size=2,
    )


def test_global_finds_mixed_plan_when_local_fastest_choices_exceed_budget() -> None:
    context = _context()
    local = plan_materialization_placement(context, policy=PlacementPolicy.LOCAL_GREEDY)
    global_plan = plan_materialization_placement(
        context, policy=PlacementPolicy.GLOBAL_EXHAUSTIVE
    )

    assert local.requires_replan is True
    assert global_plan.requires_replan is False
    assert global_plan.predicted_peak_bytes == 110
    assert global_plan.predicted_latency_ms == 9.0
    assert [placement.action for placement in global_plan.placements].count(
        MaterializationAction.DENSE
    ) == 1
    assert [placement.action for placement in global_plan.placements].count(
        MaterializationAction.STRUCTURED
    ) == 2


def test_global_uses_all_dense_when_budget_allows_fastest_combination() -> None:
    plan = plan_materialization_placement(
        _context(budget=200), policy=PlacementPolicy.GLOBAL_EXHAUSTIVE
    )

    assert plan.requires_replan is False
    assert plan.predicted_peak_bytes == 190
    assert plan.predicted_latency_ms == 3.0
    assert {placement.action for placement in plan.placements} == {
        MaterializationAction.DENSE
    }


def test_structured_is_filtered_for_unvalidated_autograd_placement() -> None:
    plan = plan_materialization_placement(
        _context(budget=200, requires_grad=True),
        policy=PlacementPolicy.GLOBAL_EXHAUSTIVE,
    )

    assert plan.requires_replan is False
    assert {placement.action for placement in plan.placements} == {
        MaterializationAction.DENSE
    }


def test_global_returns_deterministic_replan_when_no_combination_fits() -> None:
    plan = plan_materialization_placement(
        _context(budget=50), policy=PlacementPolicy.GLOBAL_EXHAUSTIVE
    )

    assert plan.requires_replan is True
    assert plan.placements == ()
    assert 1 <= plan.recommended_domain_batch_size < 8
    assert plan.reason == "no_barrier_placement_fits_safe_budget"


def test_bounded_retry_ladder_reserves_a_minimum_peak_fallback() -> None:
    candidates = (
        PlacementRetryCandidate("fast", 100, 1.0),
        PlacementRetryCandidate("middle", 94, 2.0),
        PlacementRetryCandidate("safe", 70, 4.0),
        PlacementRetryCandidate("slow", 90, 5.0),
    )

    ladder = rank_bounded_placement_retry_candidates(
        candidates,
        memory_budget_bytes=100,
        max_attempts=3,
    )

    assert ladder == ("fast", "safe", "slow")


def test_bounded_retry_ladder_validates_controls() -> None:
    candidates = (PlacementRetryCandidate("only", 10, 1.0),)

    with pytest.raises(ValueError, match="max_attempts"):
        rank_bounded_placement_retry_candidates(
            candidates, memory_budget_bytes=10, max_attempts=0
        )
    with pytest.raises(ValueError, match="candidate_id"):
        rank_bounded_placement_retry_candidates(
            (*candidates, candidates[0]), memory_budget_bytes=10
        )


def test_bounded_retry_ladder_fallback_does_not_repeat_fastest_candidate() -> None:
    candidates = (
        PlacementRetryCandidate("first", 0, 0.0),
        PlacementRetryCandidate("second", 0, 0.0),
    )

    ladder = rank_bounded_placement_retry_candidates(
        candidates, memory_budget_bytes=10, max_attempts=2
    )

    assert ladder == ("first", "second")


def test_bounded_retry_ladder_prefers_explicit_conservative_fallback() -> None:
    candidates = (
        PlacementRetryCandidate("fast", 10, 1.0),
        PlacementRetryCandidate("model_minimum", 1, 3.0),
        PlacementRetryCandidate("all_structured", 5, 4.0, conservative=True),
    )

    ladder = rank_bounded_placement_retry_candidates(
        candidates, memory_budget_bytes=10, max_attempts=2
    )

    assert ladder == ("fast", "all_structured")


def test_placement_plan_dump_is_stable_json() -> None:
    first = plan_materialization_placement(
        _context(), policy=PlacementPolicy.GLOBAL_EXHAUSTIVE
    )
    second = plan_materialization_placement(
        _context(), policy=PlacementPolicy.GLOBAL_EXHAUSTIVE
    )

    assert first == second
    payload = first.to_dict()
    assert payload["schema_version"] == PLACEMENT_SCHEMA_VERSION
    assert [item["barrier_id"] for item in payload["placements"]] == [
        "relu_0",
        "relu_1",
        "relu_2",
    ]
    json.dumps(payload, sort_keys=True)


def test_runtime_executes_mixed_relu_placement_and_matches_all_dense() -> None:
    module = _two_relu_mlp()
    spec = InputSpec.linf(
        value_name="input",
        center=torch.randn(2, 4, dtype=torch.float64),
        eps=0.1,
    )
    linear_spec = torch.randn(2, 5, 3, dtype=torch.float64)
    mixed_plan = plan_materialization_placement(
        _runtime_placement_context(budget=100),
        policy=PlacementPolicy.GLOBAL_EXHAUSTIVE,
    )
    dense_plan = plan_materialization_placement(
        _runtime_placement_context(budget=200),
        policy=PlacementPolicy.GLOBAL_EXHAUSTIVE,
    )

    with trace_materializations() as mixed_trace:
        mixed = run_crown_ibp_mlp(
            module,
            spec,
            linear_spec_C=linear_spec,
            materialization_placement_plan=mixed_plan,
        )
    dense = run_crown_ibp_mlp(
        module,
        spec,
        linear_spec_C=linear_spec,
        materialization_placement_plan=dense_plan,
    )

    assert torch.allclose(mixed.lower, dense.lower, atol=1e-10, rtol=1e-10)
    assert torch.allclose(mixed.upper, dense.upper, atol=1e-10, rtol=1e-10)
    persistent_sources = {
        event.source_value
        for event in mixed_trace.events
        if event.persistent_or_ephemeral == "persistent"
    }
    ephemeral_sources = {
        event.source_value
        for event in mixed_trace.events
        if event.persistent_or_ephemeral == "ephemeral"
    }
    assert persistent_sources == {"h1"}
    assert "h2" in ephemeral_sources


def test_optimized_bound_runtime_rejects_mixed_structured_placement() -> None:
    module = _two_relu_mlp()
    spec = InputSpec.linf(
        value_name="input",
        center=torch.randn(1, 4, dtype=torch.float64),
        eps=0.1,
    )
    mixed_plan = plan_materialization_placement(
        _runtime_placement_context(budget=100),
        policy=PlacementPolicy.GLOBAL_EXHAUSTIVE,
    )

    with pytest.raises(ValueError, match="structured optimized-bound placements"):
        run_alpha_crown_mlp(
            module,
            spec,
            steps=1,
            materialization_placement_plan=mixed_plan,
        )


def test_host_retry_blacklists_cuda_oom_candidates_until_success() -> None:
    dense = plan_materialization_placement(
        _runtime_placement_context(budget=200),
        policy=PlacementPolicy.GLOBAL_EXHAUSTIVE,
    )
    mixed = plan_materialization_placement(
        _runtime_placement_context(budget=100),
        policy=PlacementPolicy.GLOBAL_EXHAUSTIVE,
    )
    plans = (dense, mixed, dense)
    calls = []

    def execute(plan):
        calls.append(plan)
        if len(calls) < 3:
            raise torch.cuda.OutOfMemoryError("simulated candidate OOM")
        return "ok"

    result, stats = execute_placement_candidates_with_retry(
        plans, execute, clear_cuda_cache=False
    )

    assert result == "ok"
    assert stats.attempts == 3
    assert stats.oom_failures == 2
    assert stats.selected_index == 2
    assert len(stats.attempted_patterns) == 3


def test_host_retry_reports_exhausted_candidates() -> None:
    plan = plan_materialization_placement(
        _runtime_placement_context(budget=200),
        policy=PlacementPolicy.GLOBAL_EXHAUSTIVE,
    )

    with pytest.raises(PlacementRetryExhausted) as error:
        execute_placement_candidates_with_retry(
            (plan, plan),
            lambda _plan: (_ for _ in ()).throw(
                torch.cuda.OutOfMemoryError("simulated candidate OOM")
            ),
            clear_cuda_cache=False,
        )
    assert error.value.stats.attempts == 2
    assert error.value.stats.selected_index is None


def test_bounded_host_retry_ranks_plans_before_cuda_oom_feedback() -> None:
    dense = plan_materialization_placement(
        _runtime_placement_context(budget=200),
        policy=PlacementPolicy.GLOBAL_EXHAUSTIVE,
    )
    mixed = plan_materialization_placement(
        _runtime_placement_context(budget=100),
        policy=PlacementPolicy.GLOBAL_EXHAUSTIVE,
    )
    calls = []

    def execute(plan):
        calls.append(plan)
        if len(calls) == 1:
            raise torch.cuda.OutOfMemoryError("simulated candidate OOM")
        return "ok"

    result, stats = execute_bounded_placement_candidates_with_retry(
        (mixed, dense),
        execute,
        memory_budget_bytes=200,
        max_attempts=2,
        clear_cuda_cache=False,
    )

    assert result == "ok"
    assert calls == [dense, mixed]
    assert stats.attempts == 2
    assert stats.oom_failures == 1


def test_crown_retry_wrapper_executes_first_successful_plan() -> None:
    module = _two_relu_mlp()
    spec = InputSpec.linf(
        value_name="input",
        center=torch.randn(1, 4, dtype=torch.float64),
        eps=0.1,
    )
    dense_plan = plan_materialization_placement(
        _runtime_placement_context(budget=200),
        policy=PlacementPolicy.GLOBAL_EXHAUSTIVE,
    )

    bounds, stats = run_crown_ibp_mlp_with_placement_retry(
        module,
        spec,
        placement_plans=(dense_plan,),
    )

    assert torch.isfinite(bounds.lower).all()
    assert torch.isfinite(bounds.upper).all()
    assert stats.attempts == 1
    assert stats.oom_failures == 0
    assert stats.selected_index == 0


def test_static_candidate_generation_wires_into_plain_crown_retry() -> None:
    module = _two_relu_mlp()
    spec = InputSpec.linf(
        value_name="input",
        center=torch.randn(2, 4, dtype=torch.float64),
        eps=0.1,
    )
    barriers = summarize_static_barriers(
        module.get_entry_task(),
        {"h1": (2, 6), "h2": (2, 6)},
        spec_size=5,
        domain_batch_size=2,
        element_size_bytes=8,
    )
    model = PlacementInteractionCostModel(
        schema_version=PLACEMENT_COST_MODEL_SCHEMA_VERSION,
        ridge=0.0,
        peak_coefficients=tuple(0.0 for _ in PEAK_FEATURE_NAMES),
        latency_coefficients=tuple(0.0 for _ in LATENCY_FEATURE_NAMES),
        training_samples=1,
    )
    candidates = generate_static_placement_candidates(
        StaticPlacementQuery(
            barriers=barriers,
            cost_model=model,
            dense_baseline_peak_bytes=1,
            dense_baseline_latency_ms=1.0,
            memory_budget_bytes=1,
            available_memory_bytes=1,
            domain_batch_size=2,
            safety_margin=1.0,
        )
    )

    bounds, stats = run_crown_ibp_mlp_with_placement_retry(
        module,
        spec,
        placement_plans=candidates,
        linear_spec_C=torch.randn(2, 5, 3, dtype=torch.float64),
        memory_budget_bytes=1,
    )

    assert len(candidates) == 4
    assert stats.attempts == 1
    assert torch.isfinite(bounds.lower).all()
    assert torch.isfinite(bounds.upper).all()
