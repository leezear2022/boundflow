from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping


FinalConcretizationPolicy = Literal["structured", "dense_barrier"]
FinalConcretizationRequest = Literal["structured", "dense_barrier", "auto"]
CompareTarget = Literal["relu_barrier", "layout_only"]
Confidence = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class BoundOpCapability:
    op_type: str
    relu_pullback: str
    split_pos_neg: str
    dense_cache: str
    planner_action: str

    def to_jsonable(self) -> dict[str, str]:
        return {
            "op_type": self.op_type,
            "relu_pullback": self.relu_pullback,
            "split_pos_neg": self.split_pos_neg,
            "dense_cache": self.dense_cache,
            "planner_action": self.planner_action,
        }


PHASE7A_CAPABILITY_TABLE_SCHEMA_VERSION = 1
PHASE7A_CAPABILITY_TABLE: Mapping[str, BoundOpCapability] = {
    "DenseLinearOperator": BoundOpCapability(
        op_type="DenseLinearOperator",
        relu_pullback="exact_dense",
        split_pos_neg="exact_dense_view",
        dense_cache="not_needed",
        planner_action="baseline_or_terminal_dense",
    ),
    "RightMatmulLinearOperator": BoundOpCapability(
        op_type="RightMatmulLinearOperator",
        relu_pullback="exact_requires_dense_sign_split",
        split_pos_neg="exact_dense_fallback",
        dense_cache="eligible",
        planner_action="cached_dense_do_not_fake_structured_sign_split",
    ),
    "SliceInputLinearOperator": BoundOpCapability(
        op_type="SliceInputLinearOperator",
        relu_pullback="exact_embedding_materialization",
        split_pos_neg="exact_structured_delegation",
        dense_cache="eligible",
        planner_action="candidate_for_future_exact_fast_path",
    ),
    "AddLinearOperator": BoundOpCapability(
        op_type="AddLinearOperator",
        relu_pullback="delegates_to_children",
        split_pos_neg="exact_dense_fallback",
        dense_cache="eligible",
        planner_action="cache_and_consider_fusion_if_wrapper_cost_dominates",
    ),
    "ScaledInputLinearOperator": BoundOpCapability(
        op_type="ScaledInputLinearOperator",
        relu_pullback="delegates_to_scaled_child",
        split_pos_neg="exact_dense_fallback",
        dense_cache="eligible",
        planner_action="cache_and_fold_scale_when_possible",
    ),
}


@dataclass(frozen=True)
class Phase7APlannerDecision:
    schema_version: int
    planner: str
    requested_final_concretization_policy: FinalConcretizationRequest
    final_concretization_policy: FinalConcretizationPolicy
    use_dense_cache: bool
    reason: str
    selected_rules: tuple[str, ...]
    capability_table_schema_version: int
    confidence: Confidence | None = None
    evidence: Mapping[str, Any] | None = None

    def to_jsonable(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "schema_version": int(self.schema_version),
            "planner": self.planner,
            "requested_final_concretization_policy": self.requested_final_concretization_policy,
            "final_concretization_policy": self.final_concretization_policy,
            "use_dense_cache": bool(self.use_dense_cache),
            "reason": self.reason,
            "selected_rules": list(self.selected_rules),
            "capability_table_schema_version": int(self.capability_table_schema_version),
        }
        if self.confidence is not None:
            out["confidence"] = self.confidence
        if self.evidence is not None:
            out["evidence"] = dict(self.evidence)
        return out


@dataclass(frozen=True)
class Phase7BCostModelRule:
    device: str
    workload: str
    scale_id: str
    final_concretization_policy: FinalConcretizationPolicy
    confidence: Confidence
    source: str
    note: str

    def matches(self, *, device: str, workload: str, scale_id: str) -> bool:
        return self.device == device and self.workload == workload and self.scale_id == scale_id

    def to_jsonable(self) -> dict[str, str]:
        return {
            "device": self.device,
            "workload": self.workload,
            "scale_id": self.scale_id,
            "final_concretization_policy": self.final_concretization_policy,
            "confidence": self.confidence,
            "source": self.source,
            "note": self.note,
        }


PHASE7B_COST_MODEL_RULES_SCHEMA_VERSION = 1
PHASE7B_COST_MODEL_RULES: tuple[Phase7BCostModelRule, ...] = (
    Phase7BCostModelRule(
        device="cpu",
        workload="permute_reshape_linear",
        scale_id="small",
        final_concretization_policy="structured",
        confidence="high",
        source="phase7b_pr22_cpu_matrix",
        note="structured final concretization beat dense_barrier by a high-confidence margin on CPU small scale",
    ),
    Phase7BCostModelRule(
        device="cpu",
        workload="permute_reshape_linear",
        scale_id="bench",
        final_concretization_policy="structured",
        confidence="high",
        source="phase7b_pr22_cpu_matrix",
        note="structured final concretization beat dense_barrier by a high-confidence margin on CPU bench scale",
    ),
)


def phase7a_capability_table_jsonable() -> dict[str, Any]:
    return {
        "schema_version": PHASE7A_CAPABILITY_TABLE_SCHEMA_VERSION,
        "operators": {
            name: capability.to_jsonable()
            for name, capability in sorted(PHASE7A_CAPABILITY_TABLE.items())
        },
    }


def phase7b_cost_model_rules_jsonable() -> dict[str, Any]:
    return {
        "schema_version": PHASE7B_COST_MODEL_RULES_SCHEMA_VERSION,
        "rules": [rule.to_jsonable() for rule in PHASE7B_COST_MODEL_RULES],
    }


def _find_phase7b_rule(*, device: str, workload: str, scale_id: str) -> Phase7BCostModelRule | None:
    for rule in PHASE7B_COST_MODEL_RULES:
        if rule.matches(device=device, workload=workload, scale_id=scale_id):
            return rule
    return None


def plan_phase7a_shared_crown(
    *,
    compare_target: CompareTarget,
    requested_final_concretization_policy: FinalConcretizationRequest = "structured",
) -> Phase7APlannerDecision:
    if requested_final_concretization_policy in {"structured", "dense_barrier"}:
        return Phase7APlannerDecision(
            schema_version=1,
            planner="phase7a_hybrid_planner",
            requested_final_concretization_policy=requested_final_concretization_policy,
            final_concretization_policy=requested_final_concretization_policy,
            use_dense_cache=True,
            reason="manual_final_concretization_policy",
            selected_rules=("run_local_dense_cache", "manual_final_concretization_policy"),
            capability_table_schema_version=PHASE7A_CAPABILITY_TABLE_SCHEMA_VERSION,
        )

    if requested_final_concretization_policy != "auto":
        raise ValueError(f"unknown final concretization policy request: {requested_final_concretization_policy}")

    if compare_target == "layout_only":
        return Phase7APlannerDecision(
            schema_version=1,
            planner="phase7a_hybrid_planner",
            requested_final_concretization_policy="auto",
            final_concretization_policy="dense_barrier",
            use_dense_cache=True,
            reason="layout_only_final_concretization_prefers_dense_barrier",
            selected_rules=(
                "run_local_dense_cache",
                "layout_only_final_dense_barrier",
            ),
            capability_table_schema_version=PHASE7A_CAPABILITY_TABLE_SCHEMA_VERSION,
        )

    if compare_target == "relu_barrier":
        return Phase7APlannerDecision(
            schema_version=1,
            planner="phase7a_hybrid_planner",
            requested_final_concretization_policy="auto",
            final_concretization_policy="structured",
            use_dense_cache=True,
            reason="relu_workload_keeps_structured_final_path_until_right_matmul_cost_is_replanned",
            selected_rules=(
                "run_local_dense_cache",
                "right_matmul_cached_dense_exact_sign_split",
                "keep_structured_final_concretization",
            ),
            capability_table_schema_version=PHASE7A_CAPABILITY_TABLE_SCHEMA_VERSION,
        )

    raise ValueError(f"unknown compare target: {compare_target}")


def plan_phase7b_shared_crown(
    *,
    compare_target: CompareTarget,
    workload: str,
    scale_id: str,
    device: str,
    requested_final_concretization_policy: FinalConcretizationRequest = "structured",
) -> Phase7APlannerDecision:
    if requested_final_concretization_policy != "auto":
        return plan_phase7a_shared_crown(
            compare_target=compare_target,
            requested_final_concretization_policy=requested_final_concretization_policy,
        )

    normalized_device = "cuda" if str(device).startswith("cuda") else str(device)
    rule = _find_phase7b_rule(device=normalized_device, workload=str(workload), scale_id=str(scale_id))
    if rule is not None and rule.confidence == "high":
        return Phase7APlannerDecision(
            schema_version=1,
            planner="phase7b_cost_model_v1",
            requested_final_concretization_policy="auto",
            final_concretization_policy=rule.final_concretization_policy,
            use_dense_cache=True,
            reason="phase7b_high_confidence_cost_model_rule",
            selected_rules=(
                "run_local_dense_cache",
                f"cost_model:{rule.workload}:{rule.scale_id}:{rule.final_concretization_policy}",
            ),
            capability_table_schema_version=PHASE7A_CAPABILITY_TABLE_SCHEMA_VERSION,
            confidence=rule.confidence,
            evidence=rule.to_jsonable(),
        )

    fallback = plan_phase7a_shared_crown(
        compare_target=compare_target,
        requested_final_concretization_policy="auto",
    )
    return Phase7APlannerDecision(
        schema_version=fallback.schema_version,
        planner="phase7b_cost_model_v1_fallback",
        requested_final_concretization_policy=fallback.requested_final_concretization_policy,
        final_concretization_policy=fallback.final_concretization_policy,
        use_dense_cache=fallback.use_dense_cache,
        reason=fallback.reason,
        selected_rules=("no_high_confidence_cost_model_rule", *fallback.selected_rules),
        capability_table_schema_version=fallback.capability_table_schema_version,
        confidence="low",
        evidence={"fallback_planner": fallback.planner},
    )
