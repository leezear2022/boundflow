"""Tests for candidate-independent PR-11 topology/liveness summaries."""

from boundflow.ir.task import BoundTask, TaskKind, TaskOp
from boundflow.planner.materialization_static_features import (
    STATIC_BARRIER_SCHEMA_VERSION,
    StaticBarrierSummary,
    summarize_static_barriers,
)
from boundflow.planner.materialization_placement import (
    StaticPlacementQuery,
    generate_static_placement_candidates,
)
from boundflow.planner.materialization_placement_cost_model import (
    LATENCY_FEATURE_NAMES,
    PEAK_FEATURE_NAMES,
    PLACEMENT_COST_MODEL_SCHEMA_VERSION,
    PlacementInteractionCostModel,
)


def _branched_task() -> BoundTask:
    return BoundTask(
        task_id="branched",
        kind=TaskKind.INTERVAL_IBP,
        ops=[
            TaskOp("linear", "stem", ["input", "w0"], ["stem_pre"]),
            TaskOp("relu", "stem_relu", ["stem_pre"], ["stem"]),
            TaskOp("linear", "left", ["stem", "w1"], ["left"]),
            TaskOp("linear", "right", ["stem", "w2"], ["right"]),
            TaskOp("add", "merge", ["left", "right"], ["merge_pre"]),
            TaskOp("relu", "merge_relu", ["merge_pre"], ["merged"]),
            TaskOp("linear", "head", ["merged", "w3"], ["out"]),
        ],
        input_values=["input"],
        output_values=["out"],
    )


def test_static_barrier_summary_captures_branch_liveness_and_merge_type() -> None:
    summaries = summarize_static_barriers(
        _branched_task(),
        {"stem_pre": (2, 4), "merge_pre": (2, 4)},
        spec_size=3,
        domain_batch_size=2,
        element_size_bytes=4,
    )

    stem, merge = summaries
    assert stem.barrier_id == "stem_pre"
    assert stem.value_shape_per_domain == (4,)
    assert stem.spec_batch_size == 3
    assert stem.domain_batch_size == 2
    assert stem.element_size_bytes == 4
    assert stem.coefficient_elements == 2 * 3 * 4
    assert stem.coefficient_bytes == 2 * 3 * 4 * 4
    assert stem.estimated_dense_flops == stem.coefficient_elements
    assert stem.reuse_count == 2
    assert stem.direct_consumer_count == 2
    assert stem.direct_live_span == 2
    assert stem.is_branch_source is True
    assert stem.downstream_merge_count == 1
    assert stem.downstream_path_count == 2
    assert merge.producer_op_type == "add"
    assert merge.is_merge_output is True
    assert merge.direct_consumer_count == 1


def test_static_barrier_summary_round_trips_stable_schema() -> None:
    summary = summarize_static_barriers(
        _branched_task(),
        {"stem_pre": (1, 4), "merge_pre": (1, 4)},
        spec_size=1,
        domain_batch_size=1,
        element_size_bytes=4,
    )[0]

    payload = summary.to_dict()

    assert payload["schema_version"] == STATIC_BARRIER_SCHEMA_VERSION
    assert StaticBarrierSummary.from_dict(payload) == summary


def _zero_model() -> PlacementInteractionCostModel:
    return PlacementInteractionCostModel(
        schema_version=PLACEMENT_COST_MODEL_SCHEMA_VERSION,
        ridge=0.0,
        peak_coefficients=tuple(0.0 for _ in PEAK_FEATURE_NAMES),
        latency_coefficients=tuple(0.0 for _ in LATENCY_FEATURE_NAMES),
        training_samples=1,
    )


def test_placement_cost_model_round_trips_frozen_feature_contract() -> None:
    model = _zero_model()

    restored = PlacementInteractionCostModel.from_dict(model.to_dict())

    assert restored == model


def test_static_query_generates_every_legal_candidate_without_profile_trace() -> None:
    barriers = summarize_static_barriers(
        _branched_task(),
        {"stem_pre": (2, 4), "merge_pre": (2, 4)},
        spec_size=3,
        domain_batch_size=2,
        element_size_bytes=4,
    )
    query = StaticPlacementQuery(
        barriers=barriers,
        cost_model=_zero_model(),
        dense_baseline_peak_bytes=100,
        dense_baseline_latency_ms=1.0,
        memory_budget_bytes=100,
        available_memory_bytes=100,
        domain_batch_size=2,
        safety_margin=1.0,
    )

    candidates = generate_static_placement_candidates(query)

    assert len(candidates) == 4
    assert {
        tuple(placement.action.value for placement in plan.placements)
        for plan in candidates
    } == {
        ("dense", "dense"),
        ("dense", "structured"),
        ("structured", "dense"),
        ("structured", "structured"),
    }
    assert all(
        plan.reason == "static_topology_liveness_candidate_v2" for plan in candidates
    )
