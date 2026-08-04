"""Exact-frontier attribution and a preregistered optimizer-step counterfactual."""

# pylint: disable=too-many-arguments,too-many-locals,too-many-statements
# pylint: disable=too-many-branches,too-many-instance-attributes
# pylint: disable=too-many-boolean-expressions,missing-function-docstring
# pylint: disable=protected-access

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
import statistics
from typing import Mapping, Tuple

import torch

from ..frontends.plain_crown_bound_ir import tensor_content_hash
from ..ir.frontier_tightness_attribution import (
    NativeFrontierCandidateNodeIR,
    NativeFrontierNodeAttributionIR,
    NativeFrontierTightnessAttributionPlanIR,
    NativeFrontierTightnessDecisionIR,
    NativeFrontierTightnessScheduleIR,
    NativeFrontierTightnessTaskIRModule,
    lower_native_frontier_tightness_attribution_schedule,
)
from ..ir.task import BFTaskModule
from .native_alpha_beta_optimization_state import NativeAlphaBetaOptimizerPolicy
from .native_intermediate_refinement import (
    intermediate_bounds_hash,
    intermediate_refinement_semantic_trace_hash,
)
from .native_objective_ancestral_sibling_pack import _project_objective
from .native_parametric_optimizer import (
    NativeParametricOptimizerTemplateCache,
)
from .native_parametric_production_verifier import NativeParametricCompilerBatchTrace
from .native_production_verifier import NativeProductionVerifierBatchTrace
from .native_relu_split_bab_runtime import _RuntimeNode
from .native_shared_parametric_ancestral import (
    NativeSharedParametricAncestralExecution,
    _SharedEvaluatedNode,
    _evaluate_shared_parametric_batch,
)
from .task_executor import InputSpec

BOUND_REPLAY_ATOL = 1e-5
BOUND_REPLAY_RTOL = 1e-5
ALPHA_BOUNDARY_TOLERANCE = 1e-4
BETA_POSITIVE_TOLERANCE = 1e-8


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _execution_hash(execution: NativeSharedParametricAncestralExecution) -> str:
    return _canonical_hash(execution.to_dict())


def _evaluation_hash(value: object) -> str:
    return _canonical_hash(value.to_dict())  # type: ignore[attr-defined]


def _active_node_ids(
    execution: NativeSharedParametricAncestralExecution,
) -> tuple[str, ...]:
    evaluations = {
        item.node.node_id: item for item in execution.queue.trace.evaluations
    }
    terminal = tuple(
        decision.node_id
        for decision in execution.queue.trace.decisions
        if decision.kind == "terminal"
    )
    active = tuple(execution.queue.trace.final_frontier_node_ids) + terminal
    if not active:
        active = tuple(
            item.node.node_id
            for item in execution.queue.trace.evaluations
            if item.lower < execution.queue.trace.config.threshold
        )
    if len(active) != len(set(active)) or any(
        node_id not in evaluations for node_id in active
    ):
        raise ValueError("frontier attribution active-node enumeration differs")
    return tuple(sorted(active))


def _source_values(
    execution: NativeSharedParametricAncestralExecution,
) -> dict[str, _SharedEvaluatedNode]:
    states = dict(execution.queue.selected_states)
    refinements = {item.node_id: item.execution for item in execution.node_refinements}
    result: dict[str, _SharedEvaluatedNode] = {}
    for evaluation in execution.queue.trace.evaluations:
        node_id = evaluation.node.node_id
        state = states.get(node_id)
        refinement = refinements.get(node_id)
        if state is None or refinement is None:
            raise ValueError("frontier attribution source node payload is absent")
        state.validate()
        split_state = tuple(
            (name, value[0].detach().contiguous().clone())
            for name, value in sorted(state.splits.items())
        )
        result[node_id] = _SharedEvaluatedNode(
            runtime_node=_RuntimeNode(node=evaluation.node, split_state=split_state),
            evaluation=evaluation,
            selected_state=state,
            relu_pre=refinement.relu_pre,
            refinement_execution=refinement,
        )
    return result


def _active_sibling_groups(
    execution: NativeSharedParametricAncestralExecution,
    active_node_ids: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    active = set(active_node_ids)
    groups: list[tuple[str, str]] = []
    covered: set[str] = set()
    for commit in execution.batch_commits:
        selected = tuple(node_id for node_id in commit.node_ids if node_id in active)
        if not selected:
            continue
        if (
            commit.commit_kind != "atomic_sibling_pair"
            or len(commit.node_ids) != 2
            or len(selected) != 2
            or selected != commit.node_ids
        ):
            raise ValueError(
                "frontier attribution requires complete active sibling groups"
            )
        groups.append((selected[0], selected[1]))
        covered.update(selected)
    if covered != active or len(groups) * 2 != len(active):
        raise ValueError("frontier attribution sibling coverage differs")
    return tuple(groups)


def _ambiguous_count(bounds: Mapping[str, object]) -> int:
    count = 0
    for value in bounds.values():
        lower = value.lower  # type: ignore[attr-defined]
        upper = value.upper  # type: ignore[attr-defined]
        count += int(((lower < 0.0) & (upper > 0.0)).sum().item())
    return count


def _state_counts(state: object) -> tuple[int, int, int, int, int]:
    alpha_count = 0
    alpha_boundary = 0
    beta_count = 0
    beta_positive = 0
    for value in state.alphas.values():  # type: ignore[attr-defined]
        alpha_count += value.numel()
        alpha_boundary += int(
            (
                (value <= ALPHA_BOUNDARY_TOLERANCE)
                | (value >= 1.0 - ALPHA_BOUNDARY_TOLERANCE)
            )
            .sum()
            .item()
        )
    for value in state.betas.values():  # type: ignore[attr-defined]
        beta_count += value.numel()
        beta_positive += int((value > BETA_POSITIVE_TOLERANCE).sum().item())
    return (
        alpha_count,
        alpha_boundary,
        alpha_count - alpha_boundary,
        beta_count,
        beta_positive,
    )


def _source_node_rows(
    execution: NativeSharedParametricAncestralExecution,
    *,
    threshold: float,
    active_node_ids: tuple[str, ...],
) -> tuple[NativeFrontierNodeAttributionIR, ...]:
    evaluations = {
        item.node.node_id: item for item in execution.queue.trace.evaluations
    }
    refinements = {item.node_id: item for item in execution.node_refinements}
    states = dict(execution.queue.selected_states)
    active = set(active_node_ids)
    rows: list[NativeFrontierNodeAttributionIR] = []
    for evaluation in execution.queue.trace.evaluations:
        node_id = evaluation.node.node_id
        refinement = refinements[node_id]
        state = states[node_id]
        parent = (
            None
            if evaluation.node.parent_node_id is None
            else evaluations[evaluation.node.parent_node_id]
        )
        pass_traces = refinement.execution.trace.pass_traces
        alpha_count, boundary, interior, beta_count, beta_positive = _state_counts(
            state
        )
        row = NativeFrontierNodeAttributionIR(
            node_id=node_id,
            parent_node_id=evaluation.node.parent_node_id,
            split_state_hash=evaluation.node.split_state_hash,
            evaluation_hash=_evaluation_hash(evaluation),
            depth=evaluation.node.depth,
            active=node_id in active,
            lower=evaluation.lower,
            upper=evaluation.upper,
            proof_deficit=threshold - evaluation.lower,
            parent_lower_gain=(
                None if parent is None else evaluation.lower - parent.lower
            ),
            refinement_plan_hash=refinement.program.plan.stable_hash(),
            refinement_semantic_trace_hash=(
                intermediate_refinement_semantic_trace_hash(refinement.execution)
            ),
            final_intermediate_bounds_hash=intermediate_bounds_hash(
                refinement.execution.relu_pre
            ),
            selected_target_count=sum(
                trace.selected_target_count for trace in pass_traces
            ),
            tightened_neuron_count=sum(
                trace.tightened_neuron_count for trace in pass_traces
            ),
            width_reduction_sum=sum(trace.width_reduction_sum for trace in pass_traces),
            initial_ambiguous_count=_ambiguous_count(
                refinement.execution.program.initial_relu_pre
            ),
            final_ambiguous_count=_ambiguous_count(refinement.execution.relu_pre),
            alpha_count=alpha_count,
            alpha_boundary_count=boundary,
            alpha_interior_count=interior,
            beta_count=beta_count,
            beta_positive_count=beta_positive,
        )
        row.validate()
        rows.append(row)
    return tuple(rows)


def _expected_plan(
    execution: NativeSharedParametricAncestralExecution,
    *,
    linear_spec_C: torch.Tensor,
    threshold: torch.Tensor,
    original_clause_index: int,
    baseline_optimizer_policy: NativeAlphaBetaOptimizerPolicy,
    candidate_optimizer_policy: NativeAlphaBetaOptimizerPolicy,
    plan_id: str,
    required_active_depth: int,
    required_active_nodes: int,
) -> NativeFrontierTightnessAttributionPlanIR:
    active_ids = _active_node_ids(execution)
    evaluations = {
        item.node.node_id: item for item in execution.queue.trace.evaluations
    }
    plan = NativeFrontierTightnessAttributionPlanIR(
        plan_id=plan_id,
        source_execution_hash=_execution_hash(execution),
        source_plan_hash=execution.plan.stable_hash(),
        source_queue_trace_hash=execution.queue.trace.stable_hash(),
        objective_hash=tensor_content_hash(linear_spec_C),
        threshold_hash=tensor_content_hash(threshold.reshape(1).contiguous()),
        original_clause_index=original_clause_index,
        active_node_split_hashes=tuple(
            (node_id, evaluations[node_id].node.split_state_hash)
            for node_id in active_ids
        ),
        baseline_optimizer_policy_hash=baseline_optimizer_policy.stable_hash(),
        candidate_optimizer_policy_hash=candidate_optimizer_policy.stable_hash(),
        baseline_optimizer_steps=baseline_optimizer_policy.steps,
        candidate_optimizer_steps=candidate_optimizer_policy.steps,
        required_active_depth=required_active_depth,
        required_active_nodes=required_active_nodes,
    )
    plan.validate()
    return plan


def compile_native_frontier_tightness_attribution_plan(
    execution: NativeSharedParametricAncestralExecution,
    *,
    linear_spec_C: torch.Tensor,
    threshold: torch.Tensor,
    original_clause_index: int,
    baseline_optimizer_policy: NativeAlphaBetaOptimizerPolicy,
    candidate_optimizer_policy: NativeAlphaBetaOptimizerPolicy,
    plan_id: str,
    required_active_depth: int = 4,
    required_active_nodes: int = 16,
) -> NativeFrontierTightnessAttributionPlanIR:
    if candidate_optimizer_policy != replace(baseline_optimizer_policy, steps=15):
        raise ValueError("frontier attribution candidate changes more than steps")
    plan = _expected_plan(
        execution,
        linear_spec_C=linear_spec_C,
        threshold=threshold,
        original_clause_index=original_clause_index,
        baseline_optimizer_policy=baseline_optimizer_policy,
        candidate_optimizer_policy=candidate_optimizer_policy,
        plan_id=plan_id,
        required_active_depth=required_active_depth,
        required_active_nodes=required_active_nodes,
    )
    depths = {
        item.node.depth
        for item in execution.queue.trace.evaluations
        if item.node.node_id in dict(plan.active_node_split_hashes)
    }
    if depths != {required_active_depth}:
        raise ValueError("frontier attribution source depth coverage differs")
    _active_sibling_groups(execution, tuple(dict(plan.active_node_split_hashes)))
    return plan


def _allclose(source: float, replay: float) -> bool:
    return abs(replay - source) <= BOUND_REPLAY_ATOL + BOUND_REPLAY_RTOL * abs(source)


def _decision(
    plan: NativeFrontierTightnessAttributionPlanIR,
    rows: tuple[NativeFrontierCandidateNodeIR, ...],
    *,
    source_coverage_passed: bool,
) -> NativeFrontierTightnessDecisionIR:
    lower_diffs = [abs(row.replay_lower_diff) for row in rows]
    upper_diffs = [abs(row.replay_upper_diff) for row in rows]
    deltas = [row.candidate_lower_delta for row in rows]
    replay_passed = all(
        _allclose(row.source_lower, row.replay_lower)
        and _allclose(row.source_upper, row.replay_upper)
        and row.source_refinement_hash == row.baseline_refinement_hash
        for row in rows
    )
    candidate_bounds_valid = all(
        math.isfinite(row.candidate_lower)
        and math.isfinite(row.candidate_upper)
        and row.candidate_lower <= row.candidate_upper
        and row.source_refinement_hash == row.candidate_refinement_hash
        for row in rows
    )
    improved = sum(delta > plan.lower_delta_tolerance for delta in deltas)
    regressed = sum(delta < -plan.lower_delta_tolerance for delta in deltas)
    source_worst = min(row.source_lower for row in rows)
    replay_worst = min(row.replay_lower for row in rows)
    candidate_worst = min(row.candidate_lower for row in rows)
    worst_improvement = candidate_worst - replay_worst
    go = bool(
        source_coverage_passed
        and replay_passed
        and candidate_bounds_valid
        and regressed == 0
        and worst_improvement >= plan.minimum_worst_lower_improvement
        and improved >= plan.minimum_improved_nodes
    )
    if not source_coverage_passed:
        reason = "source_frontier_coverage_failed"
    elif not replay_passed:
        reason = "baseline_frontier_replay_failed"
    elif not candidate_bounds_valid:
        reason = "candidate_bounds_invalid"
    elif regressed:
        reason = "candidate_lower_regression"
    elif worst_improvement < plan.minimum_worst_lower_improvement:
        reason = "candidate_worst_improvement_below_gate"
    elif improved < plan.minimum_improved_nodes:
        reason = "candidate_improved_node_coverage_below_gate"
    else:
        reason = "optimizer_steps_15_fixed_frontier_gate_passed"
    decision = NativeFrontierTightnessDecisionIR(
        plan_hash=plan.stable_hash(),
        source_coverage_passed=source_coverage_passed,
        baseline_replay_passed=replay_passed,
        candidate_bounds_valid=candidate_bounds_valid,
        active_node_count=len(rows),
        improved_node_count=improved,
        regressed_node_count=regressed,
        replay_lower_max_abs_diff=max(lower_diffs),
        replay_upper_max_abs_diff=max(upper_diffs),
        minimum_candidate_lower_delta=min(deltas),
        median_candidate_lower_delta=float(statistics.median(deltas)),
        source_worst_active_lower=source_worst,
        replay_worst_active_lower=replay_worst,
        candidate_worst_active_lower=candidate_worst,
        worst_active_lower_improvement=worst_improvement,
        go=go,
        reason=reason,
    )
    decision.validate()
    return decision


def _validate_cache_batches(
    batches: tuple[NativeProductionVerifierBatchTrace, ...],
    compilers: tuple[NativeParametricCompilerBatchTrace, ...],
    groups: tuple[tuple[str, str], ...],
    *,
    optimizer_policy_hash: str,
) -> None:
    if len(batches) != len(groups) or len(compilers) != len(groups):
        raise ValueError("frontier attribution batch coverage differs")
    templates: set[str] = set()
    for index, (batch, compiler, group) in enumerate(zip(batches, compilers, groups)):
        batch.validate()
        compiler.validate()
        templates.add(compiler.template_hash)
        if (
            batch.plan.node_ids != group
            or batch.plan.optimizer_policy_hash != optimizer_policy_hash
            or compiler.cache_event.event_index != index
            or compiler.cache_event.outcome
            != ("miss_compiled" if index == 0 else "hit_exact_contract")
        ):
            raise ValueError("frontier attribution batch/cache binding differs")
    if len(templates) != 1:
        raise ValueError("frontier attribution template ownership differs")


@dataclass(frozen=True)
class NativeFrontierTightnessAttributionExecution:
    """First-class source attribution, exact replay, and candidate decision."""

    plan: NativeFrontierTightnessAttributionPlanIR
    task_ir: NativeFrontierTightnessTaskIRModule
    schedule: NativeFrontierTightnessScheduleIR
    node_rows: Tuple[NativeFrontierNodeAttributionIR, ...]
    candidate_rows: Tuple[NativeFrontierCandidateNodeIR, ...]
    decision: NativeFrontierTightnessDecisionIR
    baseline_batches: Tuple[NativeProductionVerifierBatchTrace, ...]
    baseline_compilers: Tuple[NativeParametricCompilerBatchTrace, ...]
    candidate_batches: Tuple[NativeProductionVerifierBatchTrace, ...]
    candidate_compilers: Tuple[NativeParametricCompilerBatchTrace, ...]

    def validate_against(
        self,
        source: NativeSharedParametricAncestralExecution,
        module: BFTaskModule,
        input_spec: InputSpec,
        *,
        linear_spec_C: torch.Tensor,
        threshold: torch.Tensor,
        original_clause_index: int,
        baseline_optimizer_policy: NativeAlphaBetaOptimizerPolicy,
        candidate_optimizer_policy: NativeAlphaBetaOptimizerPolicy,
    ) -> None:
        root_refinement = source.node_refinements[0].execution
        source.validate_against(
            module,
            input_spec,
            linear_spec_C=linear_spec_C,
            threshold=threshold,
            root_refinement=root_refinement,
            optimizer_policy=baseline_optimizer_policy,
        )
        expected_plan = compile_native_frontier_tightness_attribution_plan(
            source,
            linear_spec_C=linear_spec_C,
            threshold=threshold,
            original_clause_index=original_clause_index,
            baseline_optimizer_policy=baseline_optimizer_policy,
            candidate_optimizer_policy=candidate_optimizer_policy,
            plan_id=self.plan.plan_id,
            required_active_depth=self.plan.required_active_depth,
            required_active_nodes=self.plan.required_active_nodes,
        )
        active_ids = tuple(dict(expected_plan.active_node_split_hashes))
        expected_rows = _source_node_rows(
            source,
            threshold=float(threshold.reshape(-1)[0].item()),
            active_node_ids=active_ids,
        )
        groups = _active_sibling_groups(source, active_ids)
        if (
            self.plan != expected_plan
            or self.node_rows != expected_rows
            or tuple(row.node_id for row in self.candidate_rows)
            != tuple(node_id for group in groups for node_id in group)
            or self.decision
            != _decision(
                self.plan,
                self.candidate_rows,
                source_coverage_passed=True,
            )
            or self.task_ir.node_attribution_hashes
            != tuple(row.stable_hash() for row in self.node_rows)
            or self.task_ir.candidate_node_hashes
            != tuple(row.stable_hash() for row in self.candidate_rows)
            or self.task_ir.decision_hash != self.decision.stable_hash()
        ):
            raise ValueError("frontier attribution execution evidence differs")
        expected_task, expected_schedule = (
            lower_native_frontier_tightness_attribution_schedule(
                self.plan, self.node_rows, self.candidate_rows, self.decision
            )
        )
        if self.task_ir != expected_task or self.schedule != expected_schedule:
            raise ValueError("frontier attribution Task/Schedule differs")
        self.schedule.validate_against(self.task_ir)
        _validate_cache_batches(
            self.baseline_batches,
            self.baseline_compilers,
            groups,
            optimizer_policy_hash=baseline_optimizer_policy.stable_hash(),
        )
        _validate_cache_batches(
            self.candidate_batches,
            self.candidate_compilers,
            groups,
            optimizer_policy_hash=candidate_optimizer_policy.stable_hash(),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "plan": self.plan.to_dict(),
            "plan_hash": self.plan.stable_hash(),
            "task_ir": self.task_ir.to_dict(),
            "task_ir_hash": self.task_ir.stable_hash(),
            "schedule": self.schedule.to_dict(),
            "schedule_hash": self.schedule.stable_hash(self.task_ir),
            "node_rows": [row.to_dict() for row in self.node_rows],
            "candidate_rows": [row.to_dict() for row in self.candidate_rows],
            "decision": self.decision.to_dict(),
            "baseline_batch_hashes": [
                batch.stable_hash() for batch in self.baseline_batches
            ],
            "baseline_compiler_hashes": [
                compiler.stable_hash() for compiler in self.baseline_compilers
            ],
            "candidate_batch_hashes": [
                batch.stable_hash() for batch in self.candidate_batches
            ],
            "candidate_compiler_hashes": [
                compiler.stable_hash() for compiler in self.candidate_compilers
            ],
            "performance_claimed": False,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


def execute_native_frontier_tightness_attribution(
    plan: NativeFrontierTightnessAttributionPlanIR,
    source: NativeSharedParametricAncestralExecution,
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    linear_spec_C: torch.Tensor,
    threshold: torch.Tensor,
    original_clause_index: int,
    baseline_optimizer_policy: NativeAlphaBetaOptimizerPolicy,
    candidate_optimizer_policy: NativeAlphaBetaOptimizerPolicy,
) -> NativeFrontierTightnessAttributionExecution:
    expected_plan = compile_native_frontier_tightness_attribution_plan(
        source,
        linear_spec_C=linear_spec_C,
        threshold=threshold,
        original_clause_index=original_clause_index,
        baseline_optimizer_policy=baseline_optimizer_policy,
        candidate_optimizer_policy=candidate_optimizer_policy,
        plan_id=plan.plan_id,
        required_active_depth=plan.required_active_depth,
        required_active_nodes=plan.required_active_nodes,
    )
    if plan != expected_plan:
        raise ValueError("frontier attribution Plan/source binding differs")
    active_ids = tuple(dict(plan.active_node_split_hashes))
    groups = _active_sibling_groups(source, active_ids)
    source_values = _source_values(source)
    evaluator_objective = _project_objective(linear_spec_C)
    baseline_cache = NativeParametricOptimizerTemplateCache()
    candidate_cache = NativeParametricOptimizerTemplateCache()
    baseline_batches: list[NativeProductionVerifierBatchTrace] = []
    baseline_compilers: list[NativeParametricCompilerBatchTrace] = []
    candidate_batches: list[NativeProductionVerifierBatchTrace] = []
    candidate_compilers: list[NativeParametricCompilerBatchTrace] = []
    candidate_rows: list[NativeFrontierCandidateNodeIR] = []
    source_evaluations = {
        item.node.node_id: item for item in source.queue.trace.evaluations
    }
    source_refinements = {
        item.node_id: item.execution for item in source.node_refinements
    }
    for batch_index, group in enumerate(groups):
        nodes = tuple(source_values[node_id].runtime_node for node_id in group)
        baseline_values, baseline_batch, baseline_compiler, baseline_refinements = (
            _evaluate_shared_parametric_batch(
                module,
                input_spec,
                objective=evaluator_objective,
                nodes=nodes,
                batch_id=f"{plan.plan_id}:baseline:{batch_index:04d}",
                policy=baseline_optimizer_policy,
                parent_by_id=source_values,
                root_refinement=None,
                child_refinement_policy=source.plan.sibling_pack_plan.child_refinement_policy,
                compiler_cache=baseline_cache,
            )
        )
        candidate_values, candidate_batch, candidate_compiler, candidate_refinements = (
            _evaluate_shared_parametric_batch(
                module,
                input_spec,
                objective=evaluator_objective,
                nodes=nodes,
                batch_id=f"{plan.plan_id}:candidate:{batch_index:04d}",
                policy=candidate_optimizer_policy,
                parent_by_id=source_values,
                root_refinement=None,
                child_refinement_policy=source.plan.sibling_pack_plan.child_refinement_policy,
                compiler_cache=candidate_cache,
            )
        )
        baseline_batches.append(baseline_batch)
        baseline_compilers.append(baseline_compiler)
        candidate_batches.append(candidate_batch)
        candidate_compilers.append(candidate_compiler)
        baseline_refinement_by_id = dict(baseline_refinements)
        candidate_refinement_by_id = dict(candidate_refinements)
        for node_id, baseline_value, candidate_value in zip(
            group, baseline_values, candidate_values
        ):
            source_evaluation = source_evaluations[node_id]
            source_refinement_hash = intermediate_bounds_hash(
                source_refinements[node_id].relu_pre
            )
            replay_lower = baseline_value.evaluation.lower
            replay_upper = baseline_value.evaluation.upper
            candidate_lower = candidate_value.evaluation.lower
            row = NativeFrontierCandidateNodeIR(
                node_id=node_id,
                sibling_batch_index=batch_index,
                split_state_hash=source_evaluation.node.split_state_hash,
                source_evaluation_hash=_evaluation_hash(source_evaluation),
                source_refinement_hash=source_refinement_hash,
                baseline_refinement_hash=intermediate_bounds_hash(
                    baseline_refinement_by_id[node_id].relu_pre
                ),
                candidate_refinement_hash=intermediate_bounds_hash(
                    candidate_refinement_by_id[node_id].relu_pre
                ),
                baseline_selected_state_hash=(
                    baseline_value.selected_state.stable_hash()
                ),
                candidate_selected_state_hash=(
                    candidate_value.selected_state.stable_hash()
                ),
                source_lower=source_evaluation.lower,
                source_upper=source_evaluation.upper,
                replay_lower=replay_lower,
                replay_upper=replay_upper,
                candidate_lower=candidate_lower,
                candidate_upper=candidate_value.evaluation.upper,
                replay_lower_diff=replay_lower - source_evaluation.lower,
                replay_upper_diff=replay_upper - source_evaluation.upper,
                candidate_lower_delta=candidate_lower - replay_lower,
            )
            row.validate()
            candidate_rows.append(row)
    node_rows = _source_node_rows(
        source,
        threshold=float(threshold.reshape(-1)[0].item()),
        active_node_ids=active_ids,
    )
    decision = _decision(plan, tuple(candidate_rows), source_coverage_passed=True)
    task_ir, schedule = lower_native_frontier_tightness_attribution_schedule(
        plan, node_rows, tuple(candidate_rows), decision
    )
    execution = NativeFrontierTightnessAttributionExecution(
        plan=plan,
        task_ir=task_ir,
        schedule=schedule,
        node_rows=node_rows,
        candidate_rows=tuple(candidate_rows),
        decision=decision,
        baseline_batches=tuple(baseline_batches),
        baseline_compilers=tuple(baseline_compilers),
        candidate_batches=tuple(candidate_batches),
        candidate_compilers=tuple(candidate_compilers),
    )
    execution.validate_against(
        source,
        module,
        input_spec,
        linear_spec_C=linear_spec_C,
        threshold=threshold,
        original_clause_index=original_clause_index,
        baseline_optimizer_policy=baseline_optimizer_policy,
        candidate_optimizer_policy=candidate_optimizer_policy,
    )
    return execution


__all__ = [
    "NativeFrontierTightnessAttributionExecution",
    "compile_native_frontier_tightness_attribution_plan",
    "execute_native_frontier_tightness_attribution",
]
