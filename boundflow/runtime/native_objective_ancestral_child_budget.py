"""Additive child-budget compiler over the validated NRIR-32 queue engine."""

# pylint: disable=too-many-arguments,protected-access,missing-function-docstring

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch

from ..ir.objective_ancestral_child_budget import (
    NativeObjectiveAncestralChildBudgetDecisionIR,
    NativeObjectiveAncestralChildBudgetPlanIR,
    NativeObjectiveAncestralChildBudgetPolicyIR,
)
from ..ir.objective_ancestral_queue import NativeObjectiveAncestralQueuePlanIR
from ..ir.refinement import NativeIntermediateRefinementPolicyIR
from ..ir.task import BFTaskModule
from .native_alpha_beta_optimization_state import NativeAlphaBetaOptimizerPolicy
from .native_intermediate_refinement import NativeIntermediateRefinementExecution
from .native_objective_ancestral_queue import (
    NativeObjectiveAncestralQueueExecution,
    compile_native_objective_ancestral_queue_plan,
    execute_native_objective_ancestral_queue,
    validate_native_objective_ancestral_queue_plan,
)
from .task_executor import InputSpec


@dataclass(frozen=True)
class NativeObjectiveAncestralChildBudgetExecution:
    """Evidence-bound budget Plan plus its committed NRIR-32 queue execution."""

    plan: NativeObjectiveAncestralChildBudgetPlanIR
    queue_execution: NativeObjectiveAncestralQueueExecution

    def validate_against(
        self,
        module: BFTaskModule,
        input_spec: InputSpec,
        *,
        linear_spec_C: torch.Tensor,
        threshold: torch.Tensor,
        root_refinement: NativeIntermediateRefinementExecution,
        optimizer_policy: NativeAlphaBetaOptimizerPolicy,
    ) -> None:
        self.plan.validate()
        if self.queue_execution.plan.stable_hash() != self.plan.stable_hash():
            raise ValueError("objective ancestral child-budget execution Plan differs")
        self.queue_execution.validate_against(
            module,
            input_spec,
            linear_spec_C=linear_spec_C,
            threshold=threshold,
            root_refinement=root_refinement,
            optimizer_policy=optimizer_policy,
        )

    def to_dict(self) -> dict[str, object]:
        value = self.queue_execution.to_dict()
        value["child_budget_execution"] = {
            "plan_hash": self.plan.stable_hash(),
            "policy_hash": self.plan.child_budget_policy.stable_hash(),
            "decision_hash": self.plan.child_budget_decision.stable_hash(
                self.plan.child_budget_policy
            ),
            "selected_cap": self.plan.child_budget_decision.selected_cap,
            "performance_claimed": False,
        }
        return value


def compile_native_objective_ancestral_child_budget_plan(
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    linear_spec_C: torch.Tensor,
    threshold: torch.Tensor,
    root_refinement: NativeIntermediateRefinementExecution,
    optimizer_policy: NativeAlphaBetaOptimizerPolicy,
    plan_id: str,
    selected_cap: int,
    budget_policy: NativeObjectiveAncestralChildBudgetPolicyIR | None = None,
    budget_decision: NativeObjectiveAncestralChildBudgetDecisionIR | None = None,
) -> NativeObjectiveAncestralChildBudgetPlanIR:
    policy = budget_policy or NativeObjectiveAncestralChildBudgetPolicyIR()
    policy.validate()
    decision = budget_decision or NativeObjectiveAncestralChildBudgetDecisionIR(
        policy_hash=policy.stable_hash(),
        selected_cap=selected_cap,
        selection_mode="calibration_candidate",
    )
    decision.validate_against(policy)
    if decision.selected_cap != selected_cap:
        raise ValueError("objective ancestral selected cap/decision differs")
    base = compile_native_objective_ancestral_queue_plan(
        module,
        input_spec,
        linear_spec_C=linear_spec_C,
        threshold=threshold,
        root_refinement=root_refinement,
        optimizer_policy=optimizer_policy,
        plan_id=plan_id,
    )
    plan = NativeObjectiveAncestralChildBudgetPlanIR(
        plan_id=base.plan_id,
        primal_graph_hash=base.primal_graph_hash,
        input_bounds_hash=base.input_bounds_hash,
        objective_hash=base.objective_hash,
        threshold_hash=base.threshold_hash,
        root_refinement_plan_hash=base.root_refinement_plan_hash,
        root_refinement_semantic_trace_hash=(base.root_refinement_semantic_trace_hash),
        root_intermediate_bounds_hash=base.root_intermediate_bounds_hash,
        optimizer_policy_hash=base.optimizer_policy_hash,
        search_budget=base.search_budget,
        child_refinement_policy=NativeIntermediateRefinementPolicyIR(
            passes=1,
            max_neurons_per_relu=selected_cap,
            backward_chunk_size=min(32, selected_cap),
            candidate_policy_id="objective_influence_width_per_relu_v1",
        ),
        child_budget_policy=policy,
        child_budget_decision=decision,
    )
    plan.validate()
    validate_native_objective_ancestral_queue_plan(
        cast(NativeObjectiveAncestralQueuePlanIR, plan),
        module,
        input_spec,
        linear_spec_C=linear_spec_C,
        threshold=threshold,
        root_refinement=root_refinement,
        optimizer_policy=optimizer_policy,
    )
    return plan


def execute_native_objective_ancestral_child_budget_queue(
    plan: NativeObjectiveAncestralChildBudgetPlanIR,
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    linear_spec_C: torch.Tensor,
    threshold: torch.Tensor,
    root_refinement: NativeIntermediateRefinementExecution,
    optimizer_policy: NativeAlphaBetaOptimizerPolicy,
    query_id: str,
    whole_query_started_ns: int | None = None,
) -> NativeObjectiveAncestralChildBudgetExecution:
    plan.validate()
    queue_execution = execute_native_objective_ancestral_queue(
        cast(NativeObjectiveAncestralQueuePlanIR, plan),
        module,
        input_spec,
        linear_spec_C=linear_spec_C,
        threshold=threshold,
        root_refinement=root_refinement,
        optimizer_policy=optimizer_policy,
        query_id=query_id,
        whole_query_started_ns=whole_query_started_ns,
    )
    execution = NativeObjectiveAncestralChildBudgetExecution(
        plan=plan,
        queue_execution=queue_execution,
    )
    execution.validate_against(
        module,
        input_spec,
        linear_spec_C=linear_spec_C,
        threshold=threshold,
        root_refinement=root_refinement,
        optimizer_policy=optimizer_policy,
    )
    return execution


__all__ = [
    "NativeObjectiveAncestralChildBudgetExecution",
    "compile_native_objective_ancestral_child_budget_plan",
    "execute_native_objective_ancestral_child_budget_queue",
]
