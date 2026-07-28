"""Fair policy evaluation over one frozen set of typed plan observations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import statistics
from typing import Mapping, Optional, Sequence, Tuple


class AdaptivePlanPolicy(Enum):
    """Baselines required by the IR-5 contract."""

    FIXED = "fixed"
    LOCAL_GREEDY = "local_greedy"
    GLOBAL = "global"
    ORACLE = "oracle"


@dataclass(frozen=True)
class AdaptiveEvaluationContext:
    """One held-out query-distribution/resource point."""

    context_id: str
    memory_budget_bytes: int
    expected_query_count: int
    cached_artifact_keys: Tuple[str, ...] = ()

    def validate(self) -> None:
        """Reject invalid or noncanonical held-out context facts."""

        if not self.context_id or self.memory_budget_bytes <= 0:
            raise ValueError("adaptive evaluation context identity/budget is invalid")
        if self.expected_query_count <= 0:
            raise ValueError("adaptive expected query count must be positive")
        if (
            self.cached_artifact_keys != tuple(sorted(self.cached_artifact_keys))
            or len(self.cached_artifact_keys) != len(set(self.cached_artifact_keys))
            or any(not key for key in self.cached_artifact_keys)
        ):
            raise ValueError("adaptive cached artifact keys are not canonical")


@dataclass(frozen=True)
class AdaptivePlanObservation:  # pylint: disable=too-many-instance-attributes
    """Measured outcome and prediction for one immutable PlanInstance."""

    plan_id: str
    plan_instance_hash: str
    predicted_latency_ms: float
    local_score_ms: float
    measured_latency_ms: Tuple[float, ...]
    measured_compile_ms: float
    measured_peak_bytes: int
    compiled_artifact_key: Optional[str] = None
    legal: bool = True

    def validate(self) -> None:
        """Reject incomplete, nonfinite, or physically invalid observations."""

        if not self.plan_id or len(self.plan_instance_hash) != 64:
            raise ValueError("adaptive plan observation identity is invalid")
        numeric = (
            self.predicted_latency_ms,
            self.local_score_ms,
            self.measured_compile_ms,
            *self.measured_latency_ms,
        )
        if (
            not self.measured_latency_ms
            or any(not math.isfinite(value) or value < 0.0 for value in numeric)
            or self.measured_peak_bytes <= 0
        ):
            raise ValueError("adaptive plan observation measurements are invalid")
        if self.compiled_artifact_key is not None and not self.compiled_artifact_key:
            raise ValueError("adaptive compiled artifact key is empty")


@dataclass(frozen=True)
class AdaptivePolicyOutcome:  # pylint: disable=too-many-instance-attributes
    """One policy selection and its measured metrics."""

    context_id: str
    policy: AdaptivePlanPolicy
    feasible: bool
    selected_plan_id: Optional[str]
    latency_p50_ms: Optional[float]
    latency_p90_ms: Optional[float]
    latency_p99_ms: Optional[float]
    time_to_verify_ms: Optional[float]
    peak_memory_bytes: Optional[int]
    oracle_regret: Optional[float]

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe evaluation fields."""

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


def evaluate_adaptive_plan_policies(
    contexts: Sequence[AdaptiveEvaluationContext],
    observations: Sequence[AdaptivePlanObservation],
    *,
    fixed_plan_id: str,
) -> Tuple[AdaptivePolicyOutcome, ...]:
    """Evaluate all policies on identical legal options without data leakage."""

    if not contexts or not observations or not fixed_plan_id:
        raise ValueError("adaptive evaluation inputs must be non-empty")
    for context in contexts:
        context.validate()
    for observation in observations:
        observation.validate()
    if len({item.context_id for item in contexts}) != len(contexts):
        raise ValueError("adaptive evaluation duplicates context IDs")
    by_id = {item.plan_id: item for item in observations}
    if len(by_id) != len(observations) or fixed_plan_id not in by_id:
        raise ValueError("adaptive plan IDs are duplicate or fixed plan is missing")

    outcomes: list[AdaptivePolicyOutcome] = []
    for context in contexts:
        feasible = tuple(
            item
            for item in observations
            if item.legal and item.measured_peak_bytes <= context.memory_budget_bytes
        )
        if not feasible:
            raise ValueError(
                f"adaptive context has no feasible oracle plan: {context.context_id}"
            )
        oracle = min(
            feasible,
            key=lambda item, current=context: (
                _measured_ttv(item, current),
                item.plan_id,
            ),
        )
        selected = {
            AdaptivePlanPolicy.FIXED: (
                by_id[fixed_plan_id] if by_id[fixed_plan_id] in feasible else None
            ),
            AdaptivePlanPolicy.LOCAL_GREEDY: min(
                feasible, key=lambda item: (item.local_score_ms, item.plan_id)
            ),
            AdaptivePlanPolicy.GLOBAL: min(
                feasible,
                key=lambda item, current=context: (
                    _predicted_ttv(item, current),
                    item.plan_id,
                ),
            ),
            AdaptivePlanPolicy.ORACLE: oracle,
        }
        oracle_ttv = _measured_ttv(oracle, context)
        for policy in AdaptivePlanPolicy:
            choice = selected[policy]
            if choice is None:
                outcomes.append(
                    AdaptivePolicyOutcome(
                        context_id=context.context_id,
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
                )
                continue
            ttv = _measured_ttv(choice, context)
            outcomes.append(
                AdaptivePolicyOutcome(
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


def summarize_adaptive_outcomes(
    outcomes: Sequence[AdaptivePolicyOutcome],
) -> Mapping[str, object]:
    """Aggregate feasibility and regret without hiding failed contexts."""

    summary: dict[str, object] = {}
    for policy in AdaptivePlanPolicy:
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


def _compile_miss_ms(
    observation: AdaptivePlanObservation, context: AdaptiveEvaluationContext
) -> float:
    if observation.compiled_artifact_key in set(context.cached_artifact_keys):
        return 0.0
    return observation.measured_compile_ms


def _predicted_ttv(
    observation: AdaptivePlanObservation, context: AdaptiveEvaluationContext
) -> float:
    return _compile_miss_ms(observation, context) + (
        observation.predicted_latency_ms * context.expected_query_count
    )


def _measured_ttv(
    observation: AdaptivePlanObservation, context: AdaptiveEvaluationContext
) -> float:
    return _compile_miss_ms(observation, context) + (
        statistics.mean(observation.measured_latency_ms) * context.expected_query_count
    )


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


__all__ = [
    "AdaptiveEvaluationContext",
    "AdaptivePlanObservation",
    "AdaptivePlanPolicy",
    "AdaptivePolicyOutcome",
    "evaluate_adaptive_plan_policies",
    "summarize_adaptive_outcomes",
]
