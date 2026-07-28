"""Fair IR-5 policy/batching evaluator contracts."""

# pylint: disable=missing-function-docstring,too-many-arguments

from __future__ import annotations

import hashlib

from boundflow.planner.adaptive_plan_evaluator import (
    AdaptiveEvaluationContext,
    AdaptivePlanObservation,
)
from boundflow.planner.fair_adaptive_plan_evaluator import (
    FairAdaptivePlanPolicy,
    FairBaselinePlanIds,
    evaluate_fair_adaptive_plan_policies,
    summarize_fair_adaptive_outcomes,
)


def _observation(
    plan_id: str,
    predicted_latency: float,
    measured_latency: float,
    peak: int,
    *,
    predicted_compile: float = 0.0,
    measured_compile: float = 0.0,
    artifact: str | None = None,
) -> AdaptivePlanObservation:
    return AdaptivePlanObservation(
        plan_id=plan_id,
        plan_instance_hash=hashlib.sha256(plan_id.encode()).hexdigest(),
        predicted_latency_ms=predicted_latency,
        predicted_compile_ms=predicted_compile,
        local_score_ms=predicted_latency,
        measured_latency_ms=(
            measured_latency * 0.98,
            measured_latency,
            measured_latency * 1.02,
        ),
        measured_compile_ms=measured_compile,
        measured_peak_bytes=peak,
        compiled_artifact_key=artifact,
    )


def test_fair_evaluator_separates_baselines_compiler_pool_and_oracle() -> None:
    observations = (
        _observation("fixed-single", 5.0, 5.0, 50),
        _observation("ordinary-batching", 3.0, 3.0, 100),
        _observation("batched-original", 2.0, 2.0, 110),
        _observation("compiler-dense", 1.8, 2.2, 120),
        _observation(
            "compiler-tvm",
            1.0,
            1.0,
            80,
            predicted_compile=20.0,
            measured_compile=20.0,
            artifact="compiled:tvm",
        ),
    )
    contexts = (
        AdaptiveEvaluationContext("cold-single", 160, 1),
        AdaptiveEvaluationContext("cold-repeated", 160, 100),
        AdaptiveEvaluationContext("low-memory", 90, 10),
        AdaptiveEvaluationContext("warm", 160, 1, ("compiled:tvm",)),
    )
    outcomes = evaluate_fair_adaptive_plan_policies(
        contexts,
        observations,
        baselines=FairBaselinePlanIds(
            "fixed-single", "ordinary-batching", "batched-original"
        ),
        selectable_plan_ids=("compiler-dense", "compiler-tvm"),
    )
    selected = {
        (item.context_id, item.policy): item.selected_plan_id for item in outcomes
    }
    feasible = {(item.context_id, item.policy): item.feasible for item in outcomes}

    assert selected[("cold-single", FairAdaptivePlanPolicy.GLOBAL)] == "compiler-dense"
    assert selected[("cold-repeated", FairAdaptivePlanPolicy.GLOBAL)] == "compiler-tvm"
    assert selected[("warm", FairAdaptivePlanPolicy.ORACLE)] == "compiler-tvm"
    assert not feasible[("low-memory", FairAdaptivePlanPolicy.ORDINARY_BATCHING)]
    assert not feasible[("low-memory", FairAdaptivePlanPolicy.BATCHED_ORIGINAL)]
    assert selected[("low-memory", FairAdaptivePlanPolicy.GLOBAL)] == "compiler-tvm"
    summary = summarize_fair_adaptive_outcomes(outcomes)
    assert summary["global"]["feasible"] == 4
    assert summary["ordinary_batching"]["feasible"] == 3
