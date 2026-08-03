"""Method-, autograd-, and memory-aware materialization planning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Optional, Tuple

PLAN_SCHEMA_VERSION = "boundflow.materialization_plan/v1"


class BoundMethod(Enum):
    """Bound algorithms with materially different execution state."""

    IBP = "IBP"
    FORWARD = "forward"
    CROWN = "CROWN"
    ALPHA_FORWARD = "alpha-forward"
    ALPHA_CROWN = "alpha-CROWN"
    ALPHA_BETA_CROWN = "alpha-beta-CROWN"


class OptimizationStage(Enum):
    """Execution stage; deliberately independent of ``requires_grad``."""

    INFERENCE = "inference"
    ALPHA_INIT = "alpha_init"
    ALPHA_OPTIMIZE = "alpha_optimize"
    ALPHA_REUSE = "alpha_reuse"
    FINAL_BOUND = "final_bound"
    BAB_NODE_EVAL = "bab_node_eval"
    TRAINING = "training"


class MaterializationAction(Enum):
    """PR-11 v1 actions."""

    DENSE = "dense"
    STRUCTURED = "structured"
    REDUCE_BATCH = "reduce_batch"


class MaterializationPolicy(Enum):
    """Deterministic policies used by the planner and its baselines."""

    ALWAYS_DENSE = "always_dense"
    ALWAYS_STRUCTURED = "always_structured"
    METHOD_ONLY = "method_only"
    MEMORY_THRESHOLD = "memory_threshold"
    LOCAL_GREEDY = "local_greedy"
    GLOBAL = "global"
    ORACLE = "oracle"


@dataclass(frozen=True)
class TargetProfile:
    """Backend capabilities relevant to legal materialization choices."""

    name: str = "pytorch_eager"
    supports_structured: bool = True
    supports_structured_autograd: bool = False
    supports_optimized_bound_structured: bool = False
    supports_structured_latency_selection: bool = False


@dataclass(frozen=True)
class MaterializationPlannerOptions:
    """User/runtime options that enable automatic planning for a real query."""

    memory_budget_bytes: int
    available_memory_bytes: Optional[int] = None
    safety_margin: float = 0.9
    expected_query_reuse: int = 1
    policy: MaterializationPolicy = MaterializationPolicy.GLOBAL
    target: TargetProfile = TargetProfile()

    def validate(self) -> None:
        """Reject invalid or non-positive runtime limits."""

        if int(self.memory_budget_bytes) <= 0:
            raise ValueError(
                f"memory_budget_bytes must be > 0, got {self.memory_budget_bytes}"
            )
        if (
            self.available_memory_bytes is not None
            and int(self.available_memory_bytes) <= 0
        ):
            raise ValueError(
                "available_memory_bytes must be > 0 when provided, "
                f"got {self.available_memory_bytes}"
            )
        if not 0.0 < float(self.safety_margin) <= 1.0:
            raise ValueError(
                f"safety_margin must be in (0, 1], got {self.safety_margin}"
            )
        if int(self.expected_query_reuse) <= 0:
            raise ValueError(
                f"expected_query_reuse must be > 0, got {self.expected_query_reuse}"
            )


@dataclass(frozen=True)
class OperatorTreeSummary:  # pylint: disable=too-many-instance-attributes
    """Explainable cost features for one candidate operator region."""

    dense_a_bytes: int
    structured_base_bytes: int
    scale_bytes: int
    temporary_bytes: int = 0
    alpha_state_bytes: int = 0
    beta_state_bytes: int = 0
    operator_depth: int = 1
    operator_nodes: int = 1
    materialization_count: int = 0
    materialization_bytes: int = 0
    dense_latency_ms: Optional[float] = None
    structured_latency_ms: Optional[float] = None

    def validate(self) -> None:
        """Validate feature domains before using them in a cost model."""

        integer_fields = (
            "dense_a_bytes",
            "structured_base_bytes",
            "scale_bytes",
            "temporary_bytes",
            "alpha_state_bytes",
            "beta_state_bytes",
            "operator_depth",
            "operator_nodes",
            "materialization_count",
            "materialization_bytes",
        )
        for name in integer_fields:
            value = int(getattr(self, name))
            minimum = 1 if name in {"operator_depth", "operator_nodes"} else 0
            if value < minimum:
                raise ValueError(f"{name} must be >= {minimum}, got {value}")
        for name in ("dense_latency_ms", "structured_latency_ms"):
            value = getattr(self, name)
            if value is not None and (
                not math.isfinite(float(value)) or float(value) < 0.0
            ):
                raise ValueError(f"{name} must be finite and >= 0, got {value}")

    @property
    def common_state_bytes(self) -> int:
        """Return relaxation state shared by both representations."""

        return int(self.alpha_state_bytes + self.beta_state_bytes)

    def predicted_peak_bytes(self, action: MaterializationAction) -> int:
        """Return the explainable v1 peak estimate for an action."""

        if action == MaterializationAction.DENSE:
            return int(self.dense_a_bytes + self.common_state_bytes)
        if action == MaterializationAction.STRUCTURED:
            return int(
                self.structured_base_bytes
                + self.scale_bytes
                + self.temporary_bytes
                + self.common_state_bytes
            )
        raise ValueError(f"peak estimate is undefined for action={action.value}")

    def predicted_latency_ms(self, action: MaterializationAction) -> Optional[float]:
        """Return a calibrated latency when the caller supplied one."""

        if action == MaterializationAction.DENSE:
            return self.dense_latency_ms
        if action == MaterializationAction.STRUCTURED:
            return self.structured_latency_ms
        return None


@dataclass(frozen=True)
class MaterializationContext:  # pylint: disable=too-many-instance-attributes
    """All first-class state used to plan one bound query."""

    bound_method: BoundMethod
    requires_grad: bool
    optimization_stage: OptimizationStage
    alpha_enabled: bool
    beta_enabled: bool
    split_state_present: bool
    batch_size: int
    spec_size: int
    domain_batch_size: int
    operator_summary: OperatorTreeSummary
    memory_budget_bytes: int
    available_memory_bytes: int
    safety_margin: float = 0.9
    expected_query_reuse: int = 1
    target: TargetProfile = TargetProfile()

    def validate(self) -> None:
        """Validate query dimensions and method/state consistency."""

        for name in (
            "batch_size",
            "spec_size",
            "domain_batch_size",
            "memory_budget_bytes",
            "available_memory_bytes",
            "expected_query_reuse",
        ):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be > 0, got {value}")
        if not 0.0 < float(self.safety_margin) <= 1.0:
            raise ValueError(
                f"safety_margin must be in (0, 1], got {self.safety_margin}"
            )
        self.operator_summary.validate()
        if self.beta_enabled and not self.alpha_enabled:
            raise ValueError("beta_enabled requires alpha_enabled")
        if self.bound_method == BoundMethod.CROWN and (
            self.alpha_enabled or self.beta_enabled
        ):
            raise ValueError("plain CROWN context cannot enable alpha or beta")
        if self.bound_method == BoundMethod.ALPHA_CROWN and not self.alpha_enabled:
            raise ValueError("alpha-CROWN context must enable alpha")
        if self.bound_method == BoundMethod.ALPHA_BETA_CROWN and not (
            self.alpha_enabled and self.beta_enabled
        ):
            raise ValueError("alpha-beta-CROWN context must enable alpha and beta")

    @property
    def safe_memory_budget_bytes(self) -> int:
        """Return the budget after availability and safety-margin limits."""

        return int(
            float(self.safety_margin)
            * min(int(self.memory_budget_bytes), int(self.available_memory_bytes))
        )

    def to_dict(self) -> dict[str, object]:
        """Return the complete query context used to make a plan."""

        payload = asdict(self)
        payload["bound_method"] = self.bound_method.value
        payload["optimization_stage"] = self.optimization_stage.value
        return payload


@dataclass(frozen=True)
class MaterializationCandidate:
    """One candidate and the reasons it is legal/feasible or rejected."""

    action: MaterializationAction
    capability_legal: bool
    memory_feasible: bool
    predicted_peak_bytes: Optional[int]
    predicted_latency_ms: Optional[float]
    reasons: Tuple[str, ...]

    @property
    def feasible(self) -> bool:
        """Return whether both capability and memory constraints hold."""

        return bool(self.capability_legal and self.memory_feasible)


@dataclass(frozen=True)
class MaterializationObservation:
    """Measured result for one legal action in a per-case Oracle run."""

    action: MaterializationAction
    status: str
    peak_bytes: Optional[int]
    latency_ms: Optional[float]

    def validate(self) -> None:
        """Validate one measured Oracle result."""

        if self.action == MaterializationAction.REDUCE_BATCH:
            raise ValueError("Oracle observations only accept dense/structured actions")
        if not self.status:
            raise ValueError("Oracle observation status must be non-empty")
        if self.status == "ok":
            if self.peak_bytes is None or int(self.peak_bytes) < 0:
                raise ValueError(
                    "successful Oracle observation requires peak_bytes >= 0"
                )
            if self.latency_ms is None or not math.isfinite(float(self.latency_ms)):
                raise ValueError(
                    "successful Oracle observation requires finite latency_ms"
                )
            if float(self.latency_ms) < 0.0:
                raise ValueError(
                    "successful Oracle observation requires latency_ms >= 0"
                )


@dataclass(frozen=True)
class MaterializationPlan:
    """Deterministic plan plus an auditable candidate table."""

    schema_version: str
    policy: MaterializationPolicy
    action: MaterializationAction
    safe_memory_budget_bytes: int
    recommended_domain_batch_size: int
    reason: str
    candidates: Tuple[MaterializationCandidate, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-serializable plan dump."""

        payload = asdict(self)
        payload["policy"] = self.policy.value
        payload["action"] = self.action.value
        payload["candidates"] = [
            {
                **asdict(candidate),
                "action": candidate.action.value,
                "reasons": list(candidate.reasons),
            }
            for candidate in self.candidates
        ]
        return payload


@dataclass(frozen=True)
class MaterializationPlanRecord:
    """Frozen evidence record containing both planner input and output."""

    context: MaterializationContext
    plan: MaterializationPlan

    def to_dict(self) -> dict[str, object]:
        """Return the schema-v1 context/decision evidence record."""

        if self.plan.schema_version != PLAN_SCHEMA_VERSION:
            raise ValueError(
                "plan schema does not match record schema: "
                f"{self.plan.schema_version} != {PLAN_SCHEMA_VERSION}"
            )
        return {
            "schema_version": PLAN_SCHEMA_VERSION,
            "context": self.context.to_dict(),
            "plan": self.plan.to_dict(),
        }


def _candidate(
    context: MaterializationContext, action: MaterializationAction
) -> MaterializationCandidate:
    summary = context.operator_summary
    reasons: list[str] = []
    legal = True
    if action == MaterializationAction.STRUCTURED:
        if not context.target.supports_structured:
            legal = False
            reasons.append("target_lacks_structured")
        if context.requires_grad and not context.target.supports_structured_autograd:
            legal = False
            reasons.append("target_lacks_structured_autograd")
        if (context.alpha_enabled or context.beta_enabled) and not (
            context.target.supports_optimized_bound_structured
        ):
            legal = False
            reasons.append("target_lacks_optimized_bound_structured")
    peak = summary.predicted_peak_bytes(action)
    memory_feasible = peak <= context.safe_memory_budget_bytes
    if not memory_feasible:
        reasons.append("predicted_peak_exceeds_safe_budget")
    if legal and memory_feasible:
        reasons.append("feasible")
    return MaterializationCandidate(
        action=action,
        capability_legal=legal,
        memory_feasible=memory_feasible,
        predicted_peak_bytes=peak,
        predicted_latency_ms=summary.predicted_latency_ms(action),
        reasons=tuple(reasons),
    )


def estimate_operator_tree_summary(  # pylint: disable=too-many-arguments
    *,
    domain_batch_size: int,
    spec_size: int,
    output_numel: int,
    relu_numels: Tuple[int, ...],
    element_size: int,
    operator_nodes: int,
    alpha_state_bytes: int = 0,
    beta_state_bytes: int = 0,
) -> OperatorTreeSummary:
    """Build the shared shape-only v1 summary without profile measurements."""

    for name, value in (
        ("domain_batch_size", domain_batch_size),
        ("spec_size", spec_size),
        ("output_numel", output_numel),
        ("element_size", element_size),
        ("operator_nodes", operator_nodes),
    ):
        if int(value) <= 0:
            raise ValueError(f"{name} must be > 0, got {value}")
    if any(int(value) < 0 for value in relu_numels):
        raise ValueError("relu_numels must contain only non-negative values")
    total_relu_numel = sum(int(value) for value in relu_numels)
    max_relu_numel = max((int(value) for value in relu_numels), default=0)
    domain = int(domain_batch_size)
    specs = int(spec_size)
    item = int(element_size)
    return OperatorTreeSummary(
        dense_a_bytes=2 * domain * specs * total_relu_numel * item,
        structured_base_bytes=2 * domain * specs * int(output_numel) * item,
        scale_bytes=4 * domain * total_relu_numel * item,
        temporary_bytes=2 * domain * specs * max_relu_numel * item,
        alpha_state_bytes=int(alpha_state_bytes),
        beta_state_bytes=int(beta_state_bytes),
        operator_depth=max(1, int(operator_nodes)),
        operator_nodes=max(1, int(operator_nodes)),
    )


def _fastest(
    candidates: Tuple[MaterializationCandidate, ...],
    *,
    allow_structured_for_latency: bool,
) -> MaterializationCandidate:
    """Prefer measured latency; conservatively prefer dense when latency is unknown."""

    feasible = tuple(candidate for candidate in candidates if candidate.feasible)
    if not feasible:
        raise ValueError("_fastest requires at least one feasible candidate")
    dense = next(
        (
            candidate
            for candidate in feasible
            if candidate.action == MaterializationAction.DENSE
        ),
        None,
    )
    if dense is not None and not allow_structured_for_latency:
        return dense

    def key(candidate: MaterializationCandidate) -> tuple[float, int]:
        latency = candidate.predicted_latency_ms
        if latency is None:
            latency = math.inf
        dense_tie_break = 0 if candidate.action == MaterializationAction.DENSE else 1
        return float(latency), dense_tie_break

    measured = tuple(
        candidate
        for candidate in feasible
        if candidate.predicted_latency_ms is not None
    )
    if measured:
        return min(measured, key=key)
    return dense if dense is not None else feasible[0]


def _reduced_domain_batch(context: MaterializationContext, minimum_peak: int) -> int:
    if minimum_peak <= 0:
        return int(context.domain_batch_size)
    ratio = float(context.safe_memory_budget_bytes) / float(minimum_peak)
    return max(
        1,
        min(int(context.domain_batch_size) - 1, int(context.domain_batch_size * ratio)),
    )


def plan_materialization(  # pylint: disable=too-many-branches
    context: MaterializationContext,
    *,
    policy: MaterializationPolicy = MaterializationPolicy.GLOBAL,
) -> MaterializationPlan:
    """Choose a legal plan using feasibility-first, latency-second ordering."""

    context.validate()
    candidates = (
        _candidate(context, MaterializationAction.DENSE),
        _candidate(context, MaterializationAction.STRUCTURED),
    )
    by_action = {candidate.action: candidate for candidate in candidates}

    preferred: Optional[MaterializationCandidate] = None
    if policy == MaterializationPolicy.ALWAYS_DENSE:
        preferred = by_action[MaterializationAction.DENSE]
    elif policy == MaterializationPolicy.ALWAYS_STRUCTURED:
        preferred = by_action[MaterializationAction.STRUCTURED]
    elif policy == MaterializationPolicy.METHOD_ONLY:
        preferred = by_action[
            (
                MaterializationAction.STRUCTURED
                if context.bound_method == BoundMethod.CROWN
                and not context.requires_grad
                else MaterializationAction.DENSE
            )
        ]
    elif policy == MaterializationPolicy.MEMORY_THRESHOLD:
        dense = by_action[MaterializationAction.DENSE]
        preferred = (
            dense if dense.feasible else by_action[MaterializationAction.STRUCTURED]
        )
    elif policy == MaterializationPolicy.LOCAL_GREEDY:
        feasible = tuple(candidate for candidate in candidates if candidate.feasible)
        if feasible:
            preferred = min(
                feasible,
                key=lambda candidate: (
                    int(candidate.predicted_peak_bytes or 0),
                    0 if candidate.action == MaterializationAction.DENSE else 1,
                ),
            )
    elif policy == MaterializationPolicy.GLOBAL:
        feasible = tuple(candidate for candidate in candidates if candidate.feasible)
        if feasible:
            preferred = _fastest(
                feasible,
                allow_structured_for_latency=(
                    context.target.supports_structured_latency_selection
                ),
            )
    elif policy == MaterializationPolicy.ORACLE:
        raise ValueError(
            "ORACLE requires measured observations; use plan_materialization_oracle"
        )
    else:  # pragma: no cover - exhaustive Enum guard
        raise AssertionError(f"unsupported materialization policy: {policy}")

    if preferred is not None and preferred.feasible:
        return MaterializationPlan(
            schema_version=PLAN_SCHEMA_VERSION,
            policy=policy,
            action=preferred.action,
            safe_memory_budget_bytes=context.safe_memory_budget_bytes,
            recommended_domain_batch_size=int(context.domain_batch_size),
            reason=f"selected_{preferred.action.value}_feasibility_then_latency",
            candidates=candidates,
        )

    legal = tuple(candidate for candidate in candidates if candidate.capability_legal)
    if not legal:
        minimum_peak = min(
            candidate.predicted_peak_bytes or 0 for candidate in candidates
        )
        reason = "no_capability_legal_materialization_action"
    else:
        minimum_peak = min(candidate.predicted_peak_bytes or 0 for candidate in legal)
        reason = "no_materialization_action_fits_safe_budget"
    reduced = _reduced_domain_batch(context, minimum_peak)
    return MaterializationPlan(
        schema_version=PLAN_SCHEMA_VERSION,
        policy=policy,
        action=MaterializationAction.REDUCE_BATCH,
        safe_memory_budget_bytes=context.safe_memory_budget_bytes,
        recommended_domain_batch_size=reduced,
        reason=reason,
        candidates=candidates,
    )


def plan_materialization_oracle(
    context: MaterializationContext,
    observations: Tuple[MaterializationObservation, ...],
) -> MaterializationPlan:
    """Select the fastest measured legal action under the same safe budget."""

    context.validate()
    by_action: dict[MaterializationAction, MaterializationObservation] = {}
    for observation in observations:
        observation.validate()
        if observation.action in by_action:
            raise ValueError(
                f"duplicate Oracle observation for action={observation.action.value}"
            )
        by_action[observation.action] = observation

    candidates: list[MaterializationCandidate] = []
    for action in (MaterializationAction.DENSE, MaterializationAction.STRUCTURED):
        capability = _candidate(context, action)
        measured = by_action.get(action)
        reasons = [
            reason
            for reason in capability.reasons
            if reason not in {"feasible", "predicted_peak_exceeds_safe_budget"}
        ]
        if measured is None:
            candidates.append(
                MaterializationCandidate(
                    action=action,
                    capability_legal=capability.capability_legal,
                    memory_feasible=False,
                    predicted_peak_bytes=None,
                    predicted_latency_ms=None,
                    reasons=tuple(reasons + ["observation_missing"]),
                )
            )
            continue
        succeeded = measured.status == "ok"
        fits = bool(
            succeeded
            and measured.peak_bytes is not None
            and int(measured.peak_bytes) <= context.safe_memory_budget_bytes
        )
        if not succeeded:
            reasons.append(f"observed_status_{measured.status}")
        elif not fits:
            reasons.append("observed_peak_exceeds_safe_budget")
        elif capability.capability_legal:
            reasons.append("observed_feasible")
        candidates.append(
            MaterializationCandidate(
                action=action,
                capability_legal=capability.capability_legal,
                memory_feasible=fits,
                predicted_peak_bytes=measured.peak_bytes,
                predicted_latency_ms=measured.latency_ms,
                reasons=tuple(reasons),
            )
        )

    feasible = tuple(candidate for candidate in candidates if candidate.feasible)
    if feasible:
        selected = min(
            feasible,
            key=lambda candidate: (
                (
                    float(candidate.predicted_latency_ms)
                    if candidate.predicted_latency_ms is not None
                    else math.inf
                ),
                0 if candidate.action == MaterializationAction.DENSE else 1,
            ),
        )
        return MaterializationPlan(
            schema_version=PLAN_SCHEMA_VERSION,
            policy=MaterializationPolicy.ORACLE,
            action=selected.action,
            safe_memory_budget_bytes=context.safe_memory_budget_bytes,
            recommended_domain_batch_size=int(context.domain_batch_size),
            reason=f"oracle_fastest_observed_{selected.action.value}",
            candidates=tuple(candidates),
        )

    observed_peaks = tuple(
        int(candidate.predicted_peak_bytes)
        for candidate in candidates
        if candidate.capability_legal and candidate.predicted_peak_bytes is not None
    )
    minimum_peak = min(observed_peaks, default=context.safe_memory_budget_bytes + 1)
    return MaterializationPlan(
        schema_version=PLAN_SCHEMA_VERSION,
        policy=MaterializationPolicy.ORACLE,
        action=MaterializationAction.REDUCE_BATCH,
        safe_memory_budget_bytes=context.safe_memory_budget_bytes,
        recommended_domain_batch_size=_reduced_domain_batch(context, minimum_peak),
        reason="oracle_no_observed_action_fits_safe_budget",
        candidates=tuple(candidates),
    )


__all__ = [
    "BoundMethod",
    "MaterializationAction",
    "MaterializationCandidate",
    "MaterializationContext",
    "MaterializationObservation",
    "MaterializationPlan",
    "MaterializationPlanRecord",
    "MaterializationPlannerOptions",
    "MaterializationPolicy",
    "OperatorTreeSummary",
    "OptimizationStage",
    "PLAN_SCHEMA_VERSION",
    "TargetProfile",
    "estimate_operator_tree_summary",
    "plan_materialization",
    "plan_materialization_oracle",
]
