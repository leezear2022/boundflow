"""Objective-aware branching over the frozen shared ancestral evaluator."""

# pylint: disable=too-many-arguments,too-many-locals,too-many-statements
# pylint: disable=too-many-branches,too-many-instance-attributes
# pylint: disable=too-many-boolean-expressions,protected-access,duplicate-code
# pylint: disable=too-many-lines,missing-function-docstring

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import heapq
import json
import math
import statistics
import time
from typing import Callable, Optional, Tuple

import torch

from ..frontends.plain_crown_bound_ir import relu_split_state_hash, tensor_content_hash
from ..ir.bound import IntermediateBoundSource
from ..ir.objective_branch_shared_evaluator import (
    NativeObjectiveBranchBindingIR,
    NativeObjectiveBranchSharedDecisionIR,
    NativeObjectiveBranchSharedPlanIR,
    NativeObjectiveBranchSharedScheduleIR,
    NativeObjectiveBranchSharedTaskIRModule,
    lower_native_objective_branch_shared_schedule,
)
from ..ir.shared_parametric_ancestral import (
    NativeSharedParametricAncestralTaskIRUnit,
    NativeSharedParametricAncestralTaskKind,
    lower_native_shared_parametric_ancestral_schedule,
)
from ..ir.task import BFTaskModule
from .crown_ibp import _forward_ibp_trace_mlp
from .native_alpha_beta_optimization_state import NativeAlphaBetaOptimizerPolicy
from .native_intermediate_refinement import NativeIntermediateRefinementExecution
from .native_objective_ancestral_queue import NativeObjectiveAncestralNodeRefinement
from .native_objective_ancestral_sibling_pack import _project_objective
from .native_objective_branch_score import (
    NativeObjectiveBranchExecution,
    NativeObjectiveBranchPolicy,
    compile_native_objective_branch_program,
    execute_native_objective_branch_program,
)
from .native_parametric_optimizer import NativeParametricOptimizerTemplateCache
from .native_production_verifier import (
    NativeProductionBabEvaluation,
    NativeProductionReluSplitBabExecution,
    NativeProductionReluSplitBabTrace,
)
from .native_relu_split_bab_runtime import (
    BabQueueStatus,
    NativeReluSplitBabConfig,
    NativeReluSplitBabDecision,
    NativeReluSplitBabNode,
    _make_child_runtime_node,
    _QueueEntry,
    _repeat_box_input_spec,
    _root_box_bounds,
    _RuntimeNode,
)
from .native_shared_parametric_ancestral import (
    NativeSharedParametricAncestralExecution,
    NativeSharedParametricAncestralTrace,
    _append_batch_tasks,
    _evaluate_shared_parametric_batch,
    _make_batch_commit,
    _SharedEvaluatedNode,
    _task,
    compile_native_shared_parametric_ancestral_plan,
)
from .task_executor import InputSpec

ClockNs = Callable[[], int]


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _shared_execution_hash(execution: NativeSharedParametricAncestralExecution) -> str:
    return _canonical_hash(execution.to_dict())


def compile_native_objective_branch_shared_plan(
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    linear_spec_C: torch.Tensor,
    threshold: torch.Tensor,
    root_refinement: object,
    optimizer_policy: NativeAlphaBetaOptimizerPolicy,
    branch_policy: NativeObjectiveBranchPolicy,
    plan_id: str,
) -> NativeObjectiveBranchSharedPlanIR:
    """Compile the frozen shared plan plus the historical branch policy."""

    if not isinstance(root_refinement, NativeIntermediateRefinementExecution):
        raise TypeError("objective-branch shared root refinement is invalid")
    branch_policy.validate()
    shared = compile_native_shared_parametric_ancestral_plan(
        module,
        input_spec,
        linear_spec_C=linear_spec_C,
        threshold=threshold,
        root_refinement=root_refinement,
        optimizer_policy=optimizer_policy,
        plan_id=plan_id,
    )
    plan = NativeObjectiveBranchSharedPlanIR(
        plan_id=plan_id,
        shared_plan=shared,
        branch_policy_hash=branch_policy.stable_hash(),
        candidates_per_relu=branch_policy.candidates_per_relu,
        candidate_batch_size=branch_policy.candidate_batch_size,
        max_candidates=branch_policy.max_candidates,
        candidate_policy_id=branch_policy.candidate_policy_id,
        reduce_policy=branch_policy.reduce_policy,
    )
    plan.validate()
    return plan


def _bind_objective_branches(
    evaluated: tuple[_SharedEvaluatedNode, ...],
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    objective: torch.Tensor,
    optimizer_policy: NativeAlphaBetaOptimizerPolicy,
    branch_policy: NativeObjectiveBranchPolicy,
    batch_id: str,
) -> tuple[
    tuple[_SharedEvaluatedNode, ...],
    tuple[tuple[str, NativeObjectiveBranchExecution], ...],
]:
    rebound: list[_SharedEvaluatedNode] = []
    executions: list[tuple[str, NativeObjectiveBranchExecution]] = []
    scalar_input = _repeat_box_input_spec(input_spec, count=1)
    for position, item in enumerate(evaluated):
        if item.evaluation.branch_candidate is None:
            rebound.append(item)
            continue
        program = compile_native_objective_branch_program(
            module,
            scalar_input,
            linear_spec_C=objective,
            relu_pre=item.relu_pre,
            selected_state=item.selected_state,
            optimizer_policy=optimizer_policy,
            branch_policy=branch_policy,
            intermediate_bound_source=IntermediateBoundSource.NATIVE_REFINED,
            refine_external_constraints=False,
            plan_id=f"{batch_id}:node:{position}:objective-branch",
        )
        execution = execute_native_objective_branch_program(
            program, node_id=item.runtime_node.node.node_id
        )
        execution.validate()
        rebound.append(
            replace(
                item,
                evaluation=replace(item.evaluation, branch_candidate=execution.branch),
            )
        )
        executions.append((item.runtime_node.node.node_id, execution))
    return tuple(rebound), tuple(executions)


def _branch_bindings(
    execution: NativeSharedParametricAncestralExecution,
) -> Tuple[NativeObjectiveBranchBindingIR, ...]:
    evaluations = {
        item.node.node_id: item for item in execution.queue.trace.evaluations
    }
    states = execution.queue.state_map()
    bindings: list[NativeObjectiveBranchBindingIR] = []
    for node_id, branch_execution in execution.queue.objective_branch_executions:
        branch_execution.validate()
        evaluation = evaluations.get(node_id)
        state = states.get(node_id)
        if evaluation is None or state is None:
            raise ValueError("objective-branch execution lacks queue evaluation")
        program = branch_execution.program
        if (
            evaluation.branch_candidate != branch_execution.branch
            or program.selected_state != state
            or program.plan.split_state_hash != evaluation.node.split_state_hash
            or program.plan.policy_hash
            != execution.queue.objective_branch_policy.stable_hash()  # type: ignore[union-attr]
            or branch_execution.trace.node_id != node_id
        ):
            raise ValueError("objective-branch execution/queue binding differs")
        hashes = program.hashes()
        bindings.append(
            NativeObjectiveBranchBindingIR(
                node_id=node_id,
                evaluation_hash=_canonical_hash(evaluation.to_dict()),
                split_state_hash=evaluation.node.split_state_hash,
                selected_state_hash=evaluation.selected_state_hash,
                branch_plan_hash=hashes["branch_plan_hash"],
                branch_task_hash=hashes["branch_task_module_hash"],
                branch_schedule_hash=hashes["branch_schedule_hash"],
                branch_trace_hash=branch_execution.trace.stable_hash(program=program),
                selected_relu_input=branch_execution.branch.relu_input,
                selected_neuron_index=branch_execution.branch.neuron_index,
                selected_candidate_ordinal=(
                    branch_execution.trace.selected_candidate_ordinal
                ),
                candidate_count=len(program.plan.candidates),
            )
        )
    return tuple(bindings)


def _active_evaluations(
    execution: NativeSharedParametricAncestralExecution,
) -> tuple[NativeProductionBabEvaluation, ...]:
    decisions = {item.node_id: item for item in execution.queue.trace.decisions}
    result = tuple(
        item
        for item in execution.queue.trace.evaluations
        if (
            item.node.node_id in execution.queue.trace.final_frontier_node_ids
            or decisions.get(item.node.node_id) is not None
            and decisions[item.node.node_id].kind == "terminal"
        )
    )
    if not result:
        raise ValueError("objective-branch shared execution has no active frontier")
    return result


def _decision(
    plan: NativeObjectiveBranchSharedPlanIR,
    control: NativeSharedParametricAncestralExecution,
    candidate: NativeSharedParametricAncestralExecution,
    *,
    binding_count: int,
) -> NativeObjectiveBranchSharedDecisionIR:
    control_active_rows = _active_evaluations(control)
    candidate_active_rows = _active_evaluations(candidate)
    control_active = tuple(item.lower for item in control_active_rows)
    candidate_active = tuple(item.lower for item in candidate_active_rows)
    control_root = control.queue.trace.evaluations[0].lower
    candidate_root = candidate.queue.trace.evaluations[0].lower
    control_worst = min(control_active)
    candidate_worst = min(candidate_active)
    control_median = float(statistics.median(control_active))
    candidate_median = float(statistics.median(candidate_active))
    root_diff = abs(candidate_root - control_root)
    worst_delta = candidate_worst - control_worst
    median_delta = candidate_median - control_median
    structure = (
        len(control.queue.trace.evaluations) == 31
        and len(candidate.queue.trace.evaluations) == 31
        and len(control_active) == 16
        and len(candidate_active) == 16
        and all(item.node.depth == 4 for item in control_active_rows)
        and all(item.node.depth == 4 for item in candidate_active_rows)
        and binding_count
        == sum(
            item.branch_candidate is not None
            for item in candidate.queue.trace.evaluations
        )
        and candidate.queue.trace.config == control.queue.trace.config
        and candidate.queue.trace.optimizer_policy
        == control.queue.trace.optimizer_policy
        and candidate.queue.trace.objective_hash == control.queue.trace.objective_hash
        and candidate.queue.trace.root_input_lower_hash
        == control.queue.trace.root_input_lower_hash
        and candidate.queue.trace.root_input_upper_hash
        == control.queue.trace.root_input_upper_hash
    )
    root_parity = math.isclose(candidate_root, control_root, rel_tol=1e-5, abs_tol=1e-5)
    median_not_weaker = median_delta >= -plan.median_lower_tolerance
    go = (
        structure
        and root_parity
        and median_not_weaker
        and worst_delta >= plan.minimum_worst_active_lower_improvement
    )
    reason = (
        "candidate_passed_preregistered_gate"
        if go
        else (
            "candidate_structure_failed"
            if not structure
            else (
                "candidate_root_parity_failed"
                if not root_parity
                else (
                    "candidate_median_active_lower_weaker"
                    if not median_not_weaker
                    else "candidate_worst_improvement_below_gate"
                )
            )
        )
    )
    decision = NativeObjectiveBranchSharedDecisionIR(
        plan_hash=plan.stable_hash(),
        control_execution_hash=_shared_execution_hash(control),
        candidate_execution_hash=_shared_execution_hash(candidate),
        control_active_count=len(control_active),
        candidate_active_count=len(candidate_active),
        branch_execution_count=binding_count,
        control_root_lower=control_root,
        candidate_root_lower=candidate_root,
        root_lower_abs_diff=root_diff,
        control_worst_active_lower=control_worst,
        candidate_worst_active_lower=candidate_worst,
        worst_active_lower_improvement=worst_delta,
        control_median_active_lower=control_median,
        candidate_median_active_lower=candidate_median,
        median_active_lower_delta=median_delta,
        structure_passed=structure,
        root_parity_passed=root_parity,
        median_not_weaker=median_not_weaker,
        go=go,
        reason=reason,
    )
    decision.validate()
    return decision


@dataclass(frozen=True)
class NativeObjectiveBranchSharedExecution:
    """Composite proof joining shared queue and per-node branch programs."""

    plan: NativeObjectiveBranchSharedPlanIR
    shared_execution: NativeSharedParametricAncestralExecution
    branch_bindings: Tuple[NativeObjectiveBranchBindingIR, ...]
    decision: NativeObjectiveBranchSharedDecisionIR
    task_ir: NativeObjectiveBranchSharedTaskIRModule
    schedule: NativeObjectiveBranchSharedScheduleIR

    def validate_against(
        self,
        module: BFTaskModule,
        input_spec: InputSpec,
        *,
        linear_spec_C: torch.Tensor,
        threshold: torch.Tensor,
        root_refinement: object,
        optimizer_policy: NativeAlphaBetaOptimizerPolicy,
        branch_policy: NativeObjectiveBranchPolicy,
        control_execution: NativeSharedParametricAncestralExecution,
    ) -> None:
        if not isinstance(root_refinement, NativeIntermediateRefinementExecution):
            raise TypeError("objective-branch shared root refinement is invalid")
        branch_policy.validate()
        expected_plan = compile_native_objective_branch_shared_plan(
            module,
            input_spec,
            linear_spec_C=linear_spec_C,
            threshold=threshold,
            root_refinement=root_refinement,
            optimizer_policy=optimizer_policy,
            branch_policy=branch_policy,
            plan_id=self.plan.plan_id,
        )
        if self.plan != expected_plan:
            raise ValueError("objective-branch shared plan/runtime differs")
        self.shared_execution.validate_against(
            module,
            input_spec,
            linear_spec_C=linear_spec_C,
            threshold=threshold,
            root_refinement=root_refinement,
            optimizer_policy=optimizer_policy,
        )
        control_execution.validate_against(
            module,
            input_spec,
            linear_spec_C=linear_spec_C,
            threshold=threshold,
            root_refinement=root_refinement,
            optimizer_policy=optimizer_policy,
        )
        if (
            self.shared_execution.plan != self.plan.shared_plan
            or self.shared_execution.queue.objective_branch_policy != branch_policy
            or control_execution.plan != self.plan.shared_plan
            or control_execution.queue.objective_branch_policy is not None
        ):
            raise ValueError("objective-branch shared source ownership differs")
        expected_bindings = _branch_bindings(self.shared_execution)
        expected_decision = _decision(
            self.plan,
            control_execution,
            self.shared_execution,
            binding_count=len(expected_bindings),
        )
        if (
            self.branch_bindings != expected_bindings
            or self.decision != expected_decision
        ):
            raise ValueError("objective-branch shared evidence differs")
        expected_task, expected_schedule = (
            lower_native_objective_branch_shared_schedule(
                self.plan,
                shared_execution_hash=_shared_execution_hash(self.shared_execution),
                bindings=self.branch_bindings,
                decision=self.decision,
            )
        )
        if self.task_ir != expected_task or self.schedule != expected_schedule:
            raise ValueError("objective-branch shared Task/Schedule differs")

    def to_dict(self) -> dict[str, object]:
        self.schedule.validate_against(self.task_ir)
        branch_executions = dict(
            self.shared_execution.queue.objective_branch_executions
        )
        return {
            "plan": self.plan.to_dict(),
            "plan_hash": self.plan.stable_hash(),
            "shared_execution": self.shared_execution.to_dict(),
            "shared_execution_hash": _shared_execution_hash(self.shared_execution),
            "branch_bindings": [item.to_dict() for item in self.branch_bindings],
            "branch_executions": {
                node_id: branch_execution.trace.to_dict(
                    program=branch_execution.program
                )
                for node_id, branch_execution in branch_executions.items()
            },
            "decision": self.decision.to_dict(),
            "task_ir": self.task_ir.to_dict(),
            "task_ir_hash": self.task_ir.stable_hash(),
            "schedule": self.schedule.to_dict(task_ir=self.task_ir),
            "schedule_hash": self.schedule.stable_hash(self.task_ir),
            "performance_claimed": False,
        }


def execute_native_objective_branch_shared_queue(  # pylint: disable=too-many-statements
    plan: NativeObjectiveBranchSharedPlanIR,
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    linear_spec_C: torch.Tensor,
    threshold: torch.Tensor,
    root_refinement: object,
    optimizer_policy: NativeAlphaBetaOptimizerPolicy,
    branch_policy: NativeObjectiveBranchPolicy,
    compiler_cache: NativeParametricOptimizerTemplateCache,
    control_execution: NativeSharedParametricAncestralExecution,
    query_id: str,
    whole_query_started_ns: Optional[int] = None,
    clock_ns: ClockNs = time.monotonic_ns,
) -> NativeObjectiveBranchSharedExecution:
    """Run the shared queue while selecting each branch by exact bound impact."""

    if not query_id:
        raise ValueError("objective-branch shared query ID must be non-empty")
    if not isinstance(root_refinement, NativeIntermediateRefinementExecution):
        raise TypeError("objective-branch shared root refinement is invalid")
    if not isinstance(compiler_cache, NativeParametricOptimizerTemplateCache):
        raise TypeError("objective-branch shared compiler cache is invalid")
    branch_policy.validate()
    expected_plan = compile_native_objective_branch_shared_plan(
        module,
        input_spec,
        linear_spec_C=linear_spec_C,
        threshold=threshold,
        root_refinement=root_refinement,
        optimizer_policy=optimizer_policy,
        branch_policy=branch_policy,
        plan_id=plan.plan_id,
    )
    if plan != expected_plan:
        raise ValueError("objective-branch shared execution Plan differs")
    control_execution.validate_against(
        module,
        input_spec,
        linear_spec_C=linear_spec_C,
        threshold=threshold,
        root_refinement=root_refinement,
        optimizer_policy=optimizer_policy,
    )
    if control_execution.plan != plan.shared_plan:
        raise ValueError("objective-branch control Plan differs")

    sibling_plan = plan.shared_plan.sibling_pack_plan
    objective = _project_objective(linear_spec_C)
    threshold_value = float(threshold.reshape(-1)[0].item())
    started_ns = (
        clock_ns() if whole_query_started_ns is None else whole_query_started_ns
    )
    queue_started_ns = clock_ns()
    deadline_at_ns = started_ns + plan.shared_plan.whole_query_timeout_ns
    config = NativeReluSplitBabConfig(
        max_nodes=plan.shared_plan.max_nodes,
        max_depth=plan.shared_plan.max_depth,
        expansion_batch_size=sibling_plan.expansion_batch_size,
        max_eval_batch_size=sibling_plan.max_eval_batch_size,
        threshold=threshold_value,
    )
    lower, upper = _root_box_bounds(input_spec)
    _root_interval, root_pre = _forward_ibp_trace_mlp(module, input_spec)
    root_splits = tuple(
        (
            name,
            torch.zeros(
                tuple(int(dimension) for dimension in pre.lower.shape[1:]),
                dtype=torch.int8,
                device=pre.lower.device,
            ),
        )
        for name, pre in sorted(root_pre.items())
    )
    root_mapping = {name: value.unsqueeze(0) for name, value in root_splits}
    root = _RuntimeNode(
        node=NativeReluSplitBabNode(
            node_id=f"{query_id}:n000000",
            parent_node_id=None,
            depth=0,
            branch_relu_input=None,
            branch_neuron_index=None,
            branch_value=0,
            split_state_hash=relu_split_state_hash(root_mapping),
        ),
        split_state=root_splits,
    )
    tasks: list[NativeSharedParametricAncestralTaskIRUnit] = []
    admit = _task(
        tasks,
        plan.shared_plan,
        suffix="admit-query",
        kind=NativeSharedParametricAncestralTaskKind.ADMIT_QUERY,
        dependencies=(),
        batch_id=None,
        inputs={"sibling_pack_plan": sibling_plan.stable_hash()},
        output=plan.shared_plan.to_dict(),
    )
    raw_root, root_batch, root_compiler, root_refinements = (
        _evaluate_shared_parametric_batch(
            module,
            input_spec,
            objective=objective,
            nodes=(root,),
            batch_id=f"{query_id}:eval:0000",
            policy=optimizer_policy,
            parent_by_id={},
            root_refinement=root_refinement,
            child_refinement_policy=sibling_plan.child_refinement_policy,
            compiler_cache=compiler_cache,
        )
    )
    root_values, root_branches = _bind_objective_branches(
        raw_root,
        module,
        input_spec,
        objective=objective,
        optimizer_policy=optimizer_policy,
        branch_policy=branch_policy,
        batch_id=root_batch.plan.plan_id,
    )
    if clock_ns() > deadline_at_ns:
        raise RuntimeError("objective-branch shared deadline expired at root")
    root_commit = _make_batch_commit(
        batch_index=0,
        evaluated=root_values,
        batch=root_batch,
        compiler=root_compiler,
        refinements=root_refinements,
    )
    root_task = _append_batch_tasks(
        tasks,
        plan.shared_plan,
        batch=root_commit,
        compiler=root_compiler,
        dependencies=(admit,),
    )
    evaluations = [root_values[0].evaluation]
    decisions: list[NativeReluSplitBabDecision] = []
    batches = [root_batch]
    compilers = [root_compiler]
    commits = [root_commit]
    branch_executions = list(root_branches)
    runtime_by_id = {root.node.node_id: root_values[0]}
    node_refinements = [
        NativeObjectiveAncestralNodeRefinement(
            node_id=root.node.node_id,
            parent_node_id=None,
            node_split_state_hash=root.node.split_state_hash,
            program=root_refinement.program,
            execution=root_refinement,
        )
    ]
    evaluation_task_by_id = {root.node.node_id: root_task}
    heap = [_QueueEntry(root_values[0].evaluation.priority, 0, root.node.node_id)]
    next_node_serial = 1
    heap_serial = 1
    max_queue_size = 1
    budget_exhausted = False
    deadline_exhausted = False
    discarded_stage: Optional[str] = None
    discarded_compiler = None

    while heap and not budget_exhausted and not deadline_exhausted:
        if clock_ns() >= deadline_at_ns:
            deadline_exhausted = True
            break
        entry = heapq.heappop(heap)
        parent = runtime_by_id[entry.node_id]
        node = parent.runtime_node.node
        result = parent.evaluation
        if result.lower >= config.threshold:
            decision = NativeReluSplitBabDecision(
                decision_index=len(decisions),
                node_id=node.node_id,
                kind="prune",
                reason="lower_bound_meets_threshold",
            )
            decisions.append(decision)
            _task(
                tasks,
                plan.shared_plan,
                suffix=f"{node.node_id}:transition",
                kind=NativeSharedParametricAncestralTaskKind.TRANSITION_QUEUE,
                dependencies=(evaluation_task_by_id[node.node_id],),
                batch_id=result.eval_batch_id,
                inputs={"evaluation": _canonical_hash(result.to_dict())},
                output=decision.to_dict(),
            )
            continue
        if node.depth >= config.max_depth or result.branch_candidate is None:
            decision = NativeReluSplitBabDecision(
                decision_index=len(decisions),
                node_id=node.node_id,
                kind="terminal",
                reason=(
                    "configured_depth_limit"
                    if node.depth >= config.max_depth
                    else "no_unsplit_ambiguous_relu"
                ),
            )
            decisions.append(decision)
            _task(
                tasks,
                plan.shared_plan,
                suffix=f"{node.node_id}:transition",
                kind=NativeSharedParametricAncestralTaskKind.TRANSITION_QUEUE,
                dependencies=(evaluation_task_by_id[node.node_id],),
                batch_id=result.eval_batch_id,
                inputs={"evaluation": _canonical_hash(result.to_dict())},
                output=decision.to_dict(),
            )
            continue
        if len(evaluations) + 2 > config.max_nodes:
            heapq.heappush(heap, entry)
            budget_exhausted = True
            break
        branch = result.branch_candidate
        children = tuple(
            _make_child_runtime_node(
                parent.runtime_node,
                child_id=f"{query_id}:n{next_node_serial + index:06d}",
                branch=branch,
                branch_value=branch_value,
            )
            for index, branch_value in enumerate((-1, 1))
        )
        next_node_serial += 2
        decision = NativeReluSplitBabDecision(
            decision_index=len(decisions),
            node_id=node.node_id,
            kind="expand",
            reason="objective_bound_impact",
            child_node_ids=tuple(child.node.node_id for child in children),
            branch_candidate=branch,
        )
        if clock_ns() >= deadline_at_ns:
            heapq.heappush(heap, entry)
            deadline_exhausted = True
            discarded_stage = "before_sibling_pair"
            break
        raw_children, child_batch, child_compiler, child_refinement_values = (
            _evaluate_shared_parametric_batch(
                module,
                input_spec,
                objective=objective,
                nodes=children,
                batch_id=f"{query_id}:eval:{len(batches):04d}",
                policy=optimizer_policy,
                parent_by_id=runtime_by_id,
                root_refinement=None,
                child_refinement_policy=sibling_plan.child_refinement_policy,
                compiler_cache=compiler_cache,
            )
        )
        child_values, child_branches = _bind_objective_branches(
            raw_children,
            module,
            input_spec,
            objective=objective,
            optimizer_policy=optimizer_policy,
            branch_policy=branch_policy,
            batch_id=child_batch.plan.plan_id,
        )
        if len(child_values) != 2 or len(child_refinement_values) != 2:
            raise ValueError("objective-branch sibling evaluator coverage differs")
        if clock_ns() > deadline_at_ns:
            heapq.heappush(heap, entry)
            deadline_exhausted = True
            discarded_stage = "after_complete_sibling_pair"
            discarded_compiler = child_compiler
            break
        batch_index = len(batches)
        child_commit = _make_batch_commit(
            batch_index=batch_index,
            evaluated=child_values,
            batch=child_batch,
            compiler=child_compiler,
            refinements=child_refinement_values,
        )
        transition_task = _task(
            tasks,
            plan.shared_plan,
            suffix=f"{node.node_id}:transition",
            kind=NativeSharedParametricAncestralTaskKind.TRANSITION_QUEUE,
            dependencies=(evaluation_task_by_id[node.node_id],),
            batch_id=result.eval_batch_id,
            inputs={"evaluation": _canonical_hash(result.to_dict())},
            output=decision.to_dict(),
        )
        commit_task = _append_batch_tasks(
            tasks,
            plan.shared_plan,
            batch=child_commit,
            compiler=child_compiler,
            dependencies=(transition_task,),
        )
        decisions.append(decision)
        batches.append(child_batch)
        compilers.append(child_compiler)
        commits.append(child_commit)
        branch_executions.extend(child_branches)
        child_refinement_by_id = dict(child_refinement_values)
        for child, evaluated in zip(children, child_values):
            evaluations.append(evaluated.evaluation)
            runtime_by_id[child.node.node_id] = evaluated
            node_refinements.append(
                NativeObjectiveAncestralNodeRefinement(
                    node_id=child.node.node_id,
                    parent_node_id=node.node_id,
                    node_split_state_hash=child.node.split_state_hash,
                    program=child_refinement_by_id[child.node.node_id].program,
                    execution=child_refinement_by_id[child.node.node_id],
                )
            )
            evaluation_task_by_id[child.node.node_id] = commit_task
            heapq.heappush(
                heap,
                _QueueEntry(
                    evaluated.evaluation.priority,
                    heap_serial,
                    child.node.node_id,
                ),
            )
            heap_serial += 1
        max_queue_size = max(max_queue_size, len(heap))

    frontier = tuple(
        entry.node_id
        for entry in sorted(heap, key=lambda value: (value.priority, value.serial))
    )
    if deadline_exhausted:
        status: BabQueueStatus = "budget_exhausted"
        termination_reason = "whole_deadline_exhausted"
        fallback_reason = "deadline_preserve_atomic_sibling_frontier"
    elif budget_exhausted:
        status = "budget_exhausted"
        termination_reason = "node_budget_exhausted"
        fallback_reason = "none"
    else:
        status = "complete"
        termination_reason = "configured_bounded_tree_exhausted"
        fallback_reason = "none"
    queue_trace = NativeProductionReluSplitBabTrace(
        run_id=query_id,
        status=status,
        termination_reason=termination_reason,
        config=config,
        optimizer_policy=optimizer_policy,
        intermediate_bound_source=IntermediateBoundSource.NATIVE_REFINED,
        root_input_lower_hash=tensor_content_hash(lower),
        root_input_upper_hash=tensor_content_hash(upper),
        objective_hash=tensor_content_hash(objective),
        evaluations=tuple(evaluations),
        decisions=tuple(decisions),
        final_frontier_node_ids=frontier,
        batches=tuple(batches),
        max_queue_size=max_queue_size,
    )
    queue_trace.validate()
    queue = NativeProductionReluSplitBabExecution(
        trace=queue_trace,
        selected_states=tuple(
            (
                evaluation.node.node_id,
                runtime_by_id[evaluation.node.node_id].selected_state,
            )
            for evaluation in queue_trace.evaluations
        ),
        objective_branch_executions=tuple(branch_executions),
        objective_branch_policy=branch_policy,
    )
    queue.validate()
    emit_dependencies = tuple(
        task.task_id
        for task in tasks
        if task.kind
        in {
            NativeSharedParametricAncestralTaskKind.COMMIT_ROOT,
            NativeSharedParametricAncestralTaskKind.COMMIT_SIBLING_PAIR,
            NativeSharedParametricAncestralTaskKind.TRANSITION_QUEUE,
        }
    )
    _task(
        tasks,
        plan.shared_plan,
        suffix="emit",
        kind=NativeSharedParametricAncestralTaskKind.EMIT_RESULT,
        dependencies=emit_dependencies,
        batch_id=None,
        inputs={"queue_trace": queue.trace.stable_hash()},
        output=queue.trace.to_dict(),
    )
    task_ir, schedule = lower_native_shared_parametric_ancestral_schedule(
        plan.shared_plan, tuple(tasks), tuple(commits)
    )
    finished_ns = clock_ns()
    trace = NativeSharedParametricAncestralTrace(
        query_id=query_id,
        plan_hash=plan.shared_plan.stable_hash(),
        task_ir_hash=task_ir.stable_hash(),
        schedule_hash=schedule.stable_hash(task_ir),
        queue_trace_hash=queue.trace.stable_hash(),
        batch_commit_hashes=tuple(item.stable_hash() for item in commits),
        compiler_batch_hashes=tuple(item.stable_hash() for item in compilers),
        node_refinement_semantics=tuple(
            item.semantic_dict() for item in node_refinements
        ),
        cache_outcomes=tuple(item.cache_event.outcome for item in compilers),
        fallback_reason=fallback_reason,
        discarded_attempt_stage=discarded_stage,
        discarded_compiler_batch_hash=(
            None if discarded_compiler is None else discarded_compiler.stable_hash()
        ),
        source_elapsed_ns=max(0, queue_started_ns - started_ns),
        queue_elapsed_ns=max(0, finished_ns - queue_started_ns),
        whole_elapsed_ns=max(0, finished_ns - started_ns),
        deadline_ns=plan.shared_plan.whole_query_timeout_ns,
        semantic_signature_hash="",
    )
    trace = replace(
        trace, semantic_signature_hash=_canonical_hash(trace.semantic_dict())
    )
    shared_execution = NativeSharedParametricAncestralExecution(
        plan=plan.shared_plan,
        task_ir=task_ir,
        schedule=schedule,
        queue=queue,
        compiler_batches=tuple(compilers),
        batch_commits=tuple(commits),
        node_refinements=tuple(node_refinements),
        discarded_compiler_batch=discarded_compiler,
        trace=trace,
    )
    shared_execution.validate_against(
        module,
        input_spec,
        linear_spec_C=linear_spec_C,
        threshold=threshold,
        root_refinement=root_refinement,
        optimizer_policy=optimizer_policy,
    )
    bindings = _branch_bindings(shared_execution)
    composite_decision = _decision(
        plan,
        control_execution,
        shared_execution,
        binding_count=len(bindings),
    )
    composite_task, composite_schedule = lower_native_objective_branch_shared_schedule(
        plan,
        shared_execution_hash=_shared_execution_hash(shared_execution),
        bindings=bindings,
        decision=composite_decision,
    )
    execution = NativeObjectiveBranchSharedExecution(
        plan=plan,
        shared_execution=shared_execution,
        branch_bindings=bindings,
        decision=composite_decision,
        task_ir=composite_task,
        schedule=composite_schedule,
    )
    execution.validate_against(
        module,
        input_spec,
        linear_spec_C=linear_spec_C,
        threshold=threshold,
        root_refinement=root_refinement,
        optimizer_policy=optimizer_policy,
        branch_policy=branch_policy,
        control_execution=control_execution,
    )
    return execution


__all__ = [
    "NativeObjectiveBranchSharedExecution",
    "compile_native_objective_branch_shared_plan",
    "execute_native_objective_branch_shared_queue",
]
