"""Prepared-refinement production queue with compile-owned branch candidates."""

# pylint: disable=too-many-arguments,too-many-locals,too-many-statements
# pylint: disable=too-many-branches,too-many-boolean-expressions
# pylint: disable=protected-access,duplicate-code,too-many-lines
# pylint: disable=missing-function-docstring

from __future__ import annotations

from dataclasses import replace
import hashlib
import heapq
import json
import time
from typing import Callable, Optional

import torch

from ..frontends.plain_crown_bound_ir import relu_split_state_hash, tensor_content_hash
from ..ir.bound import IntermediateBoundSource
from ..ir.objective_branch_shared_evaluator import NativeObjectiveBranchSharedPlanIR
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
from .native_objective_branch_score import NativeObjectiveBranchPolicy
from .native_objective_branch_shared_evaluator import (
    compile_native_objective_branch_shared_plan,
)
from .native_prevalidated_objective_branch_shared_evaluator import (
    bind_prevalidated_objective_branches,
)
from .native_parametric_optimizer import NativeParametricOptimizerTemplateCache
from .native_parametric_production_verifier import NativeParametricCompilerBatchTrace
from .native_production_verifier import (
    NativeProductionVerifierBatchTrace,
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
    _root_box_bounds,
    _RuntimeNode,
)
from .native_shared_parametric_ancestral import (
    NativeSharedParametricAncestralExecution,
    NativeSharedParametricAncestralTrace,
    _SharedEvaluatedNode,
    _append_batch_tasks,
    _make_batch_commit,
    _task,
)
from .native_target_admission import (
    admit_native_intermediate_refinement_execution_targets,
)
from .native_prepared_shared_parametric_ancestral import (
    _evaluate_prepared_shared_parametric_batch,
    _evaluate_single_pass_prepared_shared_parametric_batch,
)
from .task_executor import InputSpec

ClockNs = Callable[[], int]
PreparedBatchEvaluator = Callable[
    ...,
    tuple[
        tuple[_SharedEvaluatedNode, ...],
        NativeProductionVerifierBatchTrace,
        NativeParametricCompilerBatchTrace,
        tuple[tuple[str, NativeIntermediateRefinementExecution], ...],
    ],
]


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _execute_native_prepared_objective_branch_shared_production_queue_with_evaluator(  # pylint: disable=too-many-statements
    plan: NativeObjectiveBranchSharedPlanIR,
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    batch_evaluator: PreparedBatchEvaluator,
    linear_spec_C: torch.Tensor,
    threshold: torch.Tensor,
    root_refinement: NativeIntermediateRefinementExecution,
    optimizer_policy: NativeAlphaBetaOptimizerPolicy,
    branch_policy: NativeObjectiveBranchPolicy,
    compiler_cache: NativeParametricOptimizerTemplateCache,
    query_id: str,
    whole_query_started_ns: Optional[int] = None,
    clock_ns: ClockNs = time.monotonic_ns,
) -> NativeSharedParametricAncestralExecution:
    """Execute compile-owned objective branch scores in the real queue."""

    if not query_id:
        raise ValueError("objective-branch production query ID must be non-empty")
    if not isinstance(root_refinement, NativeIntermediateRefinementExecution):
        raise TypeError("objective-branch production root refinement is invalid")
    if not isinstance(compiler_cache, NativeParametricOptimizerTemplateCache):
        raise TypeError("objective-branch production compiler cache is invalid")
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
        raise ValueError("objective-branch production Plan differs")

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
    raw_root, root_batch, root_compiler, root_refinements = batch_evaluator(
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
    root_values, root_branches = bind_prevalidated_objective_branches(
        raw_root,
        module,
        input_spec,
        objective=objective,
        optimizer_policy=optimizer_policy,
        branch_policy=branch_policy,
        batch_id=root_batch.plan.plan_id,
    )
    if clock_ns() > deadline_at_ns:
        raise RuntimeError("objective-branch production deadline expired at root")
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
            batch_evaluator(
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
        child_values, child_branches = bind_prevalidated_objective_branches(
            raw_children,
            module,
            input_spec,
            objective=objective,
            optimizer_policy=optimizer_policy,
            branch_policy=branch_policy,
            batch_id=child_batch.plan.plan_id,
        )
        if len(child_values) != 2 or len(child_refinement_values) != 2:
            raise ValueError("objective-branch production sibling coverage differs")
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
    execution = NativeSharedParametricAncestralExecution(
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
    execution.validate_against(
        module,
        input_spec,
        linear_spec_C=linear_spec_C,
        threshold=threshold,
        root_refinement=root_refinement,
        optimizer_policy=optimizer_policy,
    )
    if execution.queue.objective_branch_policy != branch_policy:
        raise ValueError("objective-branch production policy was erased")
    return execution


def execute_native_prepared_objective_branch_shared_production_queue(
    plan: NativeObjectiveBranchSharedPlanIR,
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    linear_spec_C: torch.Tensor,
    threshold: torch.Tensor,
    root_refinement: NativeIntermediateRefinementExecution,
    optimizer_policy: NativeAlphaBetaOptimizerPolicy,
    branch_policy: NativeObjectiveBranchPolicy,
    compiler_cache: NativeParametricOptimizerTemplateCache,
    query_id: str,
    whole_query_started_ns: Optional[int] = None,
    clock_ns: ClockNs = time.monotonic_ns,
) -> NativeSharedParametricAncestralExecution:
    """Execute the frozen NRIR45 queue with full compile-time reselection."""

    return _execute_native_prepared_objective_branch_shared_production_queue_with_evaluator(
        plan,
        module,
        input_spec,
        batch_evaluator=_evaluate_prepared_shared_parametric_batch,
        linear_spec_C=linear_spec_C,
        threshold=threshold,
        root_refinement=root_refinement,
        optimizer_policy=optimizer_policy,
        branch_policy=branch_policy,
        compiler_cache=compiler_cache,
        query_id=query_id,
        whole_query_started_ns=whole_query_started_ns,
        clock_ns=clock_ns,
    )


def execute_native_single_pass_prepared_objective_branch_shared_production_queue(
    plan: NativeObjectiveBranchSharedPlanIR,
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    linear_spec_C: torch.Tensor,
    threshold: torch.Tensor,
    root_refinement: NativeIntermediateRefinementExecution,
    optimizer_policy: NativeAlphaBetaOptimizerPolicy,
    branch_policy: NativeObjectiveBranchPolicy,
    compiler_cache: NativeParametricOptimizerTemplateCache,
    query_id: str,
    whole_query_started_ns: Optional[int] = None,
    clock_ns: ClockNs = time.monotonic_ns,
) -> NativeSharedParametricAncestralExecution:
    """Execute NRIR47 with receipt-admitted per-child target selection."""

    admitted_root_refinement = admit_native_intermediate_refinement_execution_targets(
        root_refinement, module, input_spec
    )
    return _execute_native_prepared_objective_branch_shared_production_queue_with_evaluator(
        plan,
        module,
        input_spec,
        batch_evaluator=_evaluate_single_pass_prepared_shared_parametric_batch,
        linear_spec_C=linear_spec_C,
        threshold=threshold,
        root_refinement=admitted_root_refinement,
        optimizer_policy=optimizer_policy,
        branch_policy=branch_policy,
        compiler_cache=compiler_cache,
        query_id=query_id,
        whole_query_started_ns=whole_query_started_ns,
        clock_ns=clock_ns,
    )


__all__ = [
    "execute_native_prepared_objective_branch_shared_production_queue",
    "execute_native_single_pass_prepared_objective_branch_shared_production_queue",
]
