"""Fair IR-5 evaluator including single-query and physical-batching baselines."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import partial
import math
import statistics
from typing import Mapping, Sequence

from .adaptive_plan_evaluator import (
    AdaptiveEvaluationContext,
    AdaptivePlanObservation,
)


class FairAdaptivePlanPolicy(Enum):
    """Compiler policies and non-compiler baselines required by IR-5."""

    FIXED_SINGLE = "fixed_single"
    ORDINARY_BATCHING = "ordinary_batching"
    BATCHED_ORIGINAL = "batched_original"
    LOCAL_GREEDY = "local_greedy"
    GLOBAL = "global"
    ORACLE = "oracle"


@dataclass(frozen=True)
class FairBaselinePlanIds:
    """Exact observation identities reserved for the three fair baselines."""

    fixed_single: str
    ordinary_batching: str
    batched_original: str

    def validate(self) -> None:
        """Reject aliasing that would collapse distinct baseline mechanisms."""

        values = (self.fixed_single, self.ordinary_batching, self.batched_original)
        if any(not value for value in values) or len(set(values)) != len(values):
            raise ValueError("fair baseline plan IDs are empty or aliased")


@dataclass(frozen=True)
class FairAdaptivePolicyOutcome:  # pylint: disable=too-many-instance-attributes
    """One fair policy result using per-query-normalized latency."""

    context_id: str
    policy: FairAdaptivePlanPolicy
    feasible: bool
    selected_plan_id: str | None
    latency_p50_ms: float | None
    latency_p90_ms: float | None
    latency_p99_ms: float | None
    time_to_verify_ms: float | None
    peak_memory_bytes: int | None
    oracle_regret: float | None

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe fields shared with the v1 artifact vocabulary."""

        return {
            "context_id": self.context_id,
            "policy": self.policy.value,
            "feasible": self.feasible,
            "selected_plan_id": self.selected_plan_id,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p90_ms": self.latency_p90_ms,
            "latency_p99_ms": self.latency_p99_ms,
            "time_to_verify_ms": self.time_to_verify_ms,
            "peak_memory_bytes": self.peak_memory_bytes,
            "oracle_regret": self.oracle_regret,
        }


def evaluate_fair_adaptive_plan_policies(  # pylint: disable=too-many-locals
    contexts: Sequence[AdaptiveEvaluationContext],
    observations: Sequence[AdaptivePlanObservation],
    *,
    baselines: FairBaselinePlanIds,
    selectable_plan_ids: Sequence[str],
) -> tuple[FairAdaptivePolicyOutcome, ...]:
    """Evaluate compiler choices against fixed and physically batched baselines."""

    if not contexts or not observations or not selectable_plan_ids:
        raise ValueError("fair adaptive evaluation inputs must be non-empty")
    baselines.validate()
    for context in contexts:
        context.validate()
    for observation in observations:
        observation.validate()
    if len({item.context_id for item in contexts}) != len(contexts):
        raise ValueError("fair adaptive evaluation duplicates context IDs")
    by_id = {item.plan_id: item for item in observations}
    if len(by_id) != len(observations):
        raise ValueError("fair adaptive observation plan IDs are duplicate")
    baseline_ids = {
        baselines.fixed_single,
        baselines.ordinary_batching,
        baselines.batched_original,
    }
    selectable_ids = set(selectable_plan_ids)
    if (
        not baseline_ids.issubset(by_id)
        or not selectable_ids.issubset(by_id)
        or baseline_ids & selectable_ids
        or len(selectable_ids) != len(tuple(selectable_plan_ids))
    ):
        raise ValueError("fair baseline/compiler candidate pools are invalid")

    outcomes: list[FairAdaptivePolicyOutcome] = []
    for context in contexts:
        feasible = tuple(
            item
            for item in observations
            if item.legal and item.measured_peak_bytes <= context.memory_budget_bytes
        )
        compiler_feasible = tuple(
            item for item in feasible if item.plan_id in selectable_ids
        )
        if not feasible or not compiler_feasible:
            raise ValueError(
                f"fair context lacks oracle/compiler option: {context.context_id}"
            )
        oracle = min(
            feasible,
            key=partial(_ttv_key, context=context, predicted=False),
        )
        selected: dict[FairAdaptivePlanPolicy, AdaptivePlanObservation | None] = {
            FairAdaptivePlanPolicy.FIXED_SINGLE: _feasible_by_id(
                feasible, baselines.fixed_single
            ),
            FairAdaptivePlanPolicy.ORDINARY_BATCHING: _feasible_by_id(
                feasible, baselines.ordinary_batching
            ),
            FairAdaptivePlanPolicy.BATCHED_ORIGINAL: _feasible_by_id(
                feasible, baselines.batched_original
            ),
            FairAdaptivePlanPolicy.LOCAL_GREEDY: min(
                compiler_feasible,
                key=lambda item: (item.local_score_ms, item.plan_id),
            ),
            FairAdaptivePlanPolicy.GLOBAL: min(
                compiler_feasible,
                key=partial(_ttv_key, context=context, predicted=True),
            ),
            FairAdaptivePlanPolicy.ORACLE: oracle,
        }
        oracle_ttv = _measured_ttv(oracle, context)
        for policy in FairAdaptivePlanPolicy:
            choice = selected[policy]
            if choice is None:
                outcomes.append(_infeasible(context.context_id, policy))
                continue
            ttv = _measured_ttv(choice, context)
            outcomes.append(
                FairAdaptivePolicyOutcome(
                    context_id=context.context_id,
                    policy=policy,
                    feasible=True,
                    selected_plan_id=choice.plan_id,
                    latency_p50_ms=_percentile(choice.measured_latency_ms, 0.50),
                    latency_p90_ms=_percentile(choice.measured_latency_ms, 0.90),
                    latency_p99_ms=_percentile(choice.measured_latency_ms, 0.99),
                    time_to_verify_ms=ttv,
                    peak_memory_bytes=choice.measured_peak_bytes,
                    oracle_regret=ttv / oracle_ttv,
                )
            )
    return tuple(outcomes)


def summarize_fair_adaptive_outcomes(
    outcomes: Sequence[FairAdaptivePolicyOutcome],
) -> Mapping[str, object]:
    """Report feasibility/regret for every compiler policy and baseline."""

    summary: dict[str, object] = {}
    for policy in FairAdaptivePlanPolicy:
        selected = tuple(item for item in outcomes if item.policy == policy)
        regrets = tuple(
            float(item.oracle_regret)
            for item in selected
            if item.oracle_regret is not None
        )
        summary[policy.value] = {
            "contexts": len(selected),
            "feasible": sum(item.feasible for item in selected),
            "regret_p50": None if not regrets else _percentile(regrets, 0.50),
            "regret_p90": None if not regrets else _percentile(regrets, 0.90),
            "regret_max": None if not regrets else max(regrets),
        }
    return summary


def _feasible_by_id(
    feasible: Sequence[AdaptivePlanObservation], plan_id: str
) -> AdaptivePlanObservation | None:
    return next((item for item in feasible if item.plan_id == plan_id), None)


def _infeasible(
    context_id: str, policy: FairAdaptivePlanPolicy
) -> FairAdaptivePolicyOutcome:
    return FairAdaptivePolicyOutcome(
        context_id=context_id,
        policy=policy,
        feasible=False,
        selected_plan_id=None,
        latency_p50_ms=None,
        latency_p90_ms=None,
        latency_p99_ms=None,
        time_to_verify_ms=None,
        peak_memory_bytes=None,
        oracle_regret=None,
    )


def _compile_miss(
    observation: AdaptivePlanObservation,
    context: AdaptiveEvaluationContext,
    *,
    predicted: bool,
) -> float:
    if observation.compiled_artifact_key in set(context.cached_artifact_keys):
        return 0.0
    return (
        observation.predicted_compile_ms
        if predicted
        else observation.measured_compile_ms
    )


def _predicted_ttv(
    observation: AdaptivePlanObservation, context: AdaptiveEvaluationContext
) -> float:
    return _compile_miss(observation, context, predicted=True) + (
        observation.predicted_latency_ms * context.expected_query_count
    )


def _ttv_key(
    observation: AdaptivePlanObservation,
    *,
    context: AdaptiveEvaluationContext,
    predicted: bool,
) -> tuple[float, str]:
    score = (
        _predicted_ttv(observation, context)
        if predicted
        else _measured_ttv(observation, context)
    )
    return score, observation.plan_id


def _measured_ttv(
    observation: AdaptivePlanObservation, context: AdaptiveEvaluationContext
) -> float:
    return _compile_miss(observation, context, predicted=False) + (
        statistics.mean(observation.measured_latency_ms) * context.expected_query_count
    )


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values or not 0.0 <= quantile <= 1.0:
        raise ValueError("fair adaptive percentile inputs are invalid")
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


__all__ = [
    "FairAdaptivePlanPolicy",
    "FairAdaptivePolicyOutcome",
    "FairBaselinePlanIds",
    "evaluate_fair_adaptive_plan_policies",
    "summarize_fair_adaptive_outcomes",
]
