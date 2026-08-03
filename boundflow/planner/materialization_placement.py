"""Multi-barrier materialization placement for PR-11."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import itertools
import math
from typing import Iterable, Tuple

from .materialization import MaterializationAction, TargetProfile
from .materialization_placement_cost_model import (
    PlacementInteractionCostModel,
    placement_features_from_static,
)
from .materialization_static_features import StaticBarrierSummary

PLACEMENT_SCHEMA_VERSION = "boundflow.materialization_placement/v1"


class PlacementPolicy(Enum):
    """Multi-barrier policies evaluated independently from query-level baselines."""

    LOCAL_GREEDY = "local_greedy"
    GLOBAL_EXHAUSTIVE = "global_exhaustive"


@dataclass(frozen=True)
class PlacementRetryCandidate:
    """Predicted cost summary used to build a bounded host retry ladder."""

    candidate_id: str
    predicted_peak_bytes: int
    predicted_latency_ms: float
    conservative: bool = False
    structured_count: int = 0
    barrier_count: int = 0
    action_transition_count: int = 0

    def validate(self) -> None:
        """Validate stable identity and non-negative finite predictions."""

        if not self.candidate_id:
            raise ValueError("candidate_id must be non-empty")
        if int(self.predicted_peak_bytes) < 0:
            raise ValueError("predicted_peak_bytes must be >= 0")
        latency = float(self.predicted_latency_ms)
        if not math.isfinite(latency) or latency < 0.0:
            raise ValueError("predicted_latency_ms must be finite and >= 0")
        if not 0 <= int(self.structured_count) <= int(self.barrier_count):
            raise ValueError("structured_count must be in [0, barrier_count]")
        if int(self.action_transition_count) < 0 or (
            int(self.barrier_count) > 0
            and int(self.action_transition_count) >= int(self.barrier_count)
        ):
            raise ValueError("action_transition_count must be in [0, barrier_count)")


@dataclass(frozen=True)
class BarrierCost:
    """Representation costs for one named materialization barrier."""

    barrier_id: str
    dense_persistent_bytes: int
    structured_persistent_bytes: int
    structured_ephemeral_bytes: int
    dense_latency_ms: float
    structured_latency_ms: float

    def validate(self) -> None:
        """Validate stable identity and non-negative finite costs."""

        if not self.barrier_id:
            raise ValueError("barrier_id must be non-empty")
        for name in (
            "dense_persistent_bytes",
            "structured_persistent_bytes",
            "structured_ephemeral_bytes",
        ):
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        for name in ("dense_latency_ms", "structured_latency_ms"):
            float_value = float(getattr(self, name))
            if not math.isfinite(float_value) or float_value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0, got {float_value}")


@dataclass(frozen=True)
class PlacementContext:  # pylint: disable=too-many-instance-attributes
    """Global memory/capability context for a barrier sequence."""

    barriers: Tuple[BarrierCost, ...]
    common_persistent_bytes: int
    memory_budget_bytes: int
    available_memory_bytes: int
    safety_margin: float = 0.9
    requires_grad: bool = False
    alpha_enabled: bool = False
    beta_enabled: bool = False
    domain_batch_size: int = 1
    target: TargetProfile = TargetProfile()

    def validate(self) -> None:
        """Validate barriers, unique identities, and global limits."""

        if not self.barriers:
            raise ValueError("placement context requires at least one barrier")
        for barrier in self.barriers:
            barrier.validate()
        identities = [barrier.barrier_id for barrier in self.barriers]
        if len(set(identities)) != len(identities):
            raise ValueError("barrier_id values must be unique")
        for name in (
            "common_persistent_bytes",
            "memory_budget_bytes",
            "available_memory_bytes",
        ):
            value = int(getattr(self, name))
            minimum = 1 if name != "common_persistent_bytes" else 0
            if value < minimum:
                raise ValueError(f"{name} must be >= {minimum}, got {value}")
        if int(self.domain_batch_size) <= 0:
            raise ValueError("domain_batch_size must be > 0")
        if not 0.0 < float(self.safety_margin) <= 1.0:
            raise ValueError("safety_margin must be in (0, 1]")
        if self.beta_enabled and not self.alpha_enabled:
            raise ValueError("beta_enabled requires alpha_enabled")

    @property
    def safe_memory_budget_bytes(self) -> int:
        """Return the availability-capped budget after its safety margin."""

        return int(
            float(self.safety_margin)
            * min(int(self.memory_budget_bytes), int(self.available_memory_bytes))
        )


@dataclass(frozen=True)
class StaticPlacementQuery:  # pylint: disable=too-many-instance-attributes
    """Production inputs for generating static-model placement candidates."""

    barriers: Tuple[StaticBarrierSummary, ...]
    cost_model: PlacementInteractionCostModel
    dense_baseline_peak_bytes: int
    dense_baseline_latency_ms: float
    memory_budget_bytes: int
    available_memory_bytes: int
    domain_batch_size: int
    safety_margin: float = 0.9
    requires_grad: bool = False
    alpha_enabled: bool = False
    beta_enabled: bool = False
    target: TargetProfile = TargetProfile()

    def validate(self) -> None:
        """Validate static summaries, model, capabilities, and query limits."""

        if not self.barriers:
            raise ValueError("static placement query requires barriers")
        for barrier in self.barriers:
            barrier.validate()
        identities = [barrier.barrier_id for barrier in self.barriers]
        if len(set(identities)) != len(identities):
            raise ValueError("static barrier identities must be unique")
        self.cost_model.validate()
        for name in (
            "dense_baseline_peak_bytes",
            "memory_budget_bytes",
            "available_memory_bytes",
            "domain_batch_size",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be > 0")
        if (
            not math.isfinite(float(self.dense_baseline_latency_ms))
            or float(self.dense_baseline_latency_ms) < 0.0
        ):
            raise ValueError("dense_baseline_latency_ms must be finite and >= 0")
        if not 0.0 < float(self.safety_margin) <= 1.0:
            raise ValueError("safety_margin must be in (0, 1]")
        if self.beta_enabled and not self.alpha_enabled:
            raise ValueError("beta_enabled requires alpha_enabled")

    @property
    def safe_memory_budget_bytes(self) -> int:
        """Return the availability-capped budget after its safety margin."""

        return int(
            float(self.safety_margin)
            * min(int(self.memory_budget_bytes), int(self.available_memory_bytes))
        )


@dataclass(frozen=True)
class BarrierPlacement:
    """One barrier action and its local predicted costs."""

    barrier_id: str
    action: MaterializationAction
    persistent_bytes: int
    ephemeral_bytes: int
    latency_ms: float
    reason: str


@dataclass(frozen=True)
class MaterializationPlacementPlan:  # pylint: disable=too-many-instance-attributes
    """A mixed per-barrier plan or a deterministic host re-plan request."""

    schema_version: str
    policy: PlacementPolicy
    placements: Tuple[BarrierPlacement, ...]
    predicted_peak_bytes: int
    predicted_latency_ms: float
    safe_memory_budget_bytes: int
    requires_replan: bool
    recommended_domain_batch_size: int
    reason: str

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic JSON plan dump."""

        return {
            "schema_version": self.schema_version,
            "policy": self.policy.value,
            "placements": [
                {**asdict(placement), "action": placement.action.value}
                for placement in self.placements
            ],
            "predicted_peak_bytes": int(self.predicted_peak_bytes),
            "predicted_latency_ms": float(self.predicted_latency_ms),
            "safe_memory_budget_bytes": int(self.safe_memory_budget_bytes),
            "requires_replan": bool(self.requires_replan),
            "recommended_domain_batch_size": int(self.recommended_domain_batch_size),
            "reason": self.reason,
        }


def _structured_legal(context: PlacementContext) -> bool:
    if not context.target.supports_structured:
        return False
    if context.requires_grad and not context.target.supports_structured_autograd:
        return False
    if (context.alpha_enabled or context.beta_enabled) and not (
        context.target.supports_optimized_bound_structured
    ):
        return False
    return True


def _placement(barrier: BarrierCost, action: MaterializationAction) -> BarrierPlacement:
    if action == MaterializationAction.DENSE:
        return BarrierPlacement(
            barrier_id=barrier.barrier_id,
            action=action,
            persistent_bytes=int(barrier.dense_persistent_bytes),
            ephemeral_bytes=0,
            latency_ms=float(barrier.dense_latency_ms),
            reason="selected_dense",
        )
    if action == MaterializationAction.STRUCTURED:
        return BarrierPlacement(
            barrier_id=barrier.barrier_id,
            action=action,
            persistent_bytes=int(barrier.structured_persistent_bytes),
            ephemeral_bytes=int(barrier.structured_ephemeral_bytes),
            latency_ms=float(barrier.structured_latency_ms),
            reason="selected_structured",
        )
    raise ValueError("barrier placement cannot use reduce_batch")


def _plan_cost(
    context: PlacementContext, placements: Tuple[BarrierPlacement, ...]
) -> tuple[int, float]:
    persistent = int(context.common_persistent_bytes) + sum(
        int(placement.persistent_bytes) for placement in placements
    )
    max_ephemeral = max(
        (int(placement.ephemeral_bytes) for placement in placements), default=0
    )
    latency = sum(float(placement.latency_ms) for placement in placements)
    return persistent + max_ephemeral, latency


def _replan(
    context: PlacementContext,
    *,
    policy: PlacementPolicy,
    minimum_peak: int,
    reason: str,
) -> MaterializationPlacementPlan:
    ratio = float(context.safe_memory_budget_bytes) / float(max(1, minimum_peak))
    reduced = max(
        1,
        min(
            int(context.domain_batch_size) - 1,
            int(context.domain_batch_size * ratio),
        ),
    )
    return MaterializationPlacementPlan(
        schema_version=PLACEMENT_SCHEMA_VERSION,
        policy=policy,
        placements=(),
        predicted_peak_bytes=int(minimum_peak),
        predicted_latency_ms=0.0,
        safe_memory_budget_bytes=context.safe_memory_budget_bytes,
        requires_replan=True,
        recommended_domain_batch_size=reduced,
        reason=reason,
    )


def plan_materialization_placement(
    context: PlacementContext, *, policy: PlacementPolicy
) -> MaterializationPlacementPlan:
    """Plan mixed barrier actions with local or global feasibility semantics."""

    context.validate()
    structured_legal = _structured_legal(context)
    legal_actions = (
        (MaterializationAction.DENSE, MaterializationAction.STRUCTURED)
        if structured_legal
        else (MaterializationAction.DENSE,)
    )
    combinations: list[Tuple[BarrierPlacement, ...]] = []
    if policy == PlacementPolicy.LOCAL_GREEDY:
        placements = tuple(
            min(
                (_placement(barrier, action) for action in legal_actions),
                key=lambda item: (
                    item.latency_ms,
                    0 if item.action == MaterializationAction.DENSE else 1,
                ),
            )
            for barrier in context.barriers
        )
        combinations.append(placements)
    elif policy == PlacementPolicy.GLOBAL_EXHAUSTIVE:
        if len(context.barriers) > 20:
            raise ValueError("global exhaustive placement is limited to 20 barriers")
        for actions in itertools.product(legal_actions, repeat=len(context.barriers)):
            combinations.append(
                tuple(
                    _placement(barrier, action)
                    for barrier, action in zip(context.barriers, actions)
                )
            )
    else:  # pragma: no cover - exhaustive Enum guard
        raise AssertionError(f"unsupported placement policy: {policy}")

    evaluated = [
        (*_plan_cost(context, placements), placements) for placements in combinations
    ]
    feasible = [
        item for item in evaluated if int(item[0]) <= context.safe_memory_budget_bytes
    ]
    if feasible:
        peak, latency, placements = min(
            feasible,
            key=lambda item: (
                float(item[1]),
                int(item[0]),
                tuple(placement.action.value for placement in item[2]),
            ),
        )
        return MaterializationPlacementPlan(
            schema_version=PLACEMENT_SCHEMA_VERSION,
            policy=policy,
            placements=placements,
            predicted_peak_bytes=int(peak),
            predicted_latency_ms=float(latency),
            safe_memory_budget_bytes=context.safe_memory_budget_bytes,
            requires_replan=False,
            recommended_domain_batch_size=int(context.domain_batch_size),
            reason="selected_fastest_globally_feasible_placement",
        )

    minimum_peak = min(int(item[0]) for item in evaluated)
    return _replan(
        context,
        policy=policy,
        minimum_peak=minimum_peak,
        reason="no_barrier_placement_fits_safe_budget",
    )


def generate_static_placement_candidates(
    query: StaticPlacementQuery,
) -> Tuple[MaterializationPlacementPlan, ...]:
    """Enumerate legal plans and predict their aggregate peak/latency statically."""

    query.validate()
    if len(query.barriers) > 20:
        raise ValueError(
            "static placement candidate generation is limited to 20 barriers"
        )
    structured_legal = bool(
        query.target.supports_structured
        and (not query.requires_grad or query.target.supports_structured_autograd)
        and (
            not (query.alpha_enabled or query.beta_enabled)
            or query.target.supports_optimized_bound_structured
        )
    )
    legal_actions = (
        (MaterializationAction.DENSE, MaterializationAction.STRUCTURED)
        if structured_legal
        else (MaterializationAction.DENSE,)
    )
    candidates: list[MaterializationPlacementPlan] = []
    for actions in itertools.product(legal_actions, repeat=len(query.barriers)):
        features = placement_features_from_static(
            query.barriers,
            tuple(action.value for action in actions),
            dense_baseline_peak_bytes=query.dense_baseline_peak_bytes,
            dense_baseline_latency_ms=query.dense_baseline_latency_ms,
        )
        prediction = query.cost_model.predict(features)
        per_barrier_latency = float(prediction.latency_ms) / float(len(actions))
        placements = tuple(
            BarrierPlacement(
                barrier_id=barrier.barrier_id,
                action=action,
                persistent_bytes=(
                    barrier.coefficient_bytes
                    if action == MaterializationAction.DENSE
                    else 0
                ),
                ephemeral_bytes=(
                    barrier.coefficient_bytes
                    if action == MaterializationAction.STRUCTURED
                    else 0
                ),
                latency_ms=per_barrier_latency,
                reason="static_topology_liveness_cost_model_v2",
            )
            for barrier, action in zip(query.barriers, actions)
        )
        candidates.append(
            MaterializationPlacementPlan(
                schema_version=PLACEMENT_SCHEMA_VERSION,
                policy=PlacementPolicy.GLOBAL_EXHAUSTIVE,
                placements=placements,
                predicted_peak_bytes=int(prediction.peak_bytes),
                predicted_latency_ms=float(prediction.latency_ms),
                safe_memory_budget_bytes=query.safe_memory_budget_bytes,
                requires_replan=False,
                recommended_domain_batch_size=int(query.domain_batch_size),
                reason="static_topology_liveness_candidate_v2",
            )
        )
    return tuple(candidates)


def rank_bounded_placement_retry_candidates(  # pylint: disable=too-many-locals
    candidates: Iterable[PlacementRetryCandidate],
    *,
    memory_budget_bytes: int,
    max_attempts: int = 6,
) -> Tuple[str, ...]:
    """Build a finite latency-rank-stratified host retry ladder.

    The v3 ladder selects the fastest predicted-feasible candidate, a topology-
    diverse candidate from its fastest decile, the 80th and 90th latency-rank
    percentiles, and the fastest near-conservative candidate with exactly one
    dense barrier.  Its final slot is reserved for an explicit conservative
    candidate.  This bounds retry cost while covering latency, topology, and
    representation-density uncertainty.
    """

    materialized = tuple(candidates)
    if not materialized:
        raise ValueError("placement retry candidates must be non-empty")
    for candidate in materialized:
        candidate.validate()
    identities = [candidate.candidate_id for candidate in materialized]
    if len(set(identities)) != len(identities):
        raise ValueError("placement retry candidate_id values must be unique")
    if int(memory_budget_bytes) <= 0:
        raise ValueError("memory_budget_bytes must be > 0")
    if int(max_attempts) <= 0:
        raise ValueError("max_attempts must be > 0")

    by_latency = sorted(
        (
            candidate
            for candidate in materialized
            if int(candidate.predicted_peak_bytes) <= int(memory_budget_bytes)
        ),
        key=lambda candidate: (
            float(candidate.predicted_latency_ms),
            int(candidate.predicted_peak_bytes),
            candidate.candidate_id,
        ),
    )
    selected: list[str] = []

    def add(candidate: PlacementRetryCandidate) -> None:
        if (
            candidate.candidate_id not in selected
            and len(selected) < int(max_attempts) - 1
        ):
            selected.append(candidate.candidate_id)

    for candidate in by_latency[:1]:
        add(candidate)
    if by_latency:
        fastest_decile = by_latency[: max(1, math.ceil(len(by_latency) * 0.1))]
        diverse = max(
            fastest_decile,
            key=lambda candidate: (
                int(candidate.action_transition_count),
                -float(candidate.predicted_latency_ms),
                candidate.candidate_id,
            ),
        )
        add(diverse)
    for quantile in (0.8, 0.9):
        if by_latency:
            index = round((len(by_latency) - 1) * quantile)
            add(by_latency[index])
    near_conservative = [
        candidate
        for candidate in by_latency
        if int(candidate.barrier_count) > 0
        and int(candidate.structured_count) == int(candidate.barrier_count) - 1
    ]
    if near_conservative:
        add(near_conservative[0])

    if len(selected) < int(max_attempts):
        fallback_pool = tuple(
            candidate
            for candidate in materialized
            if candidate.candidate_id not in selected
        )
        if not fallback_pool:
            return tuple(selected)
        conservative_pool = tuple(
            candidate for candidate in fallback_pool if candidate.conservative
        )
        fallback = min(
            conservative_pool or fallback_pool,
            key=lambda candidate: (
                int(candidate.predicted_peak_bytes),
                float(candidate.predicted_latency_ms),
                candidate.candidate_id,
            ),
        )
        selected.append(fallback.candidate_id)
    return tuple(selected)


__all__ = [
    "BarrierCost",
    "BarrierPlacement",
    "MaterializationPlacementPlan",
    "PlacementRetryCandidate",
    "StaticPlacementQuery",
    "PLACEMENT_SCHEMA_VERSION",
    "PlacementContext",
    "PlacementPolicy",
    "plan_materialization_placement",
    "generate_static_placement_candidates",
    "rank_bounded_placement_retry_candidates",
]
