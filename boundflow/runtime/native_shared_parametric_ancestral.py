"""Shared-parametric objective-ancestral queue with atomic sibling commits."""

# pylint: disable=too-many-arguments,too-many-locals,too-many-statements
# pylint: disable=too-many-branches,too-many-instance-attributes
# pylint: disable=too-many-boolean-expressions,missing-function-docstring
# pylint: disable=protected-access,duplicate-code,too-many-lines

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import heapq
import json
import time
from typing import Callable, Mapping, Optional, Sequence, Tuple

import torch

from ..domains.interval import IntervalState
from ..frontends.plain_crown_bound_ir import relu_split_state_hash, tensor_content_hash
from ..ir.bound import IntermediateBoundSource
from ..ir.production_verifier import (
    NativeProductionVerifierTaskKind,
    lower_native_production_verifier_ir,
)
from ..ir.refinement import NativeIntermediateRefinementPolicyIR
from ..ir.shared_parametric_ancestral import (
    NativeSharedParametricAncestralBatchIR,
    NativeSharedParametricAncestralPlanIR,
    NativeSharedParametricAncestralScheduleIR,
    NativeSharedParametricAncestralTaskIRModule,
    NativeSharedParametricAncestralTaskIRUnit,
    NativeSharedParametricAncestralTaskKind,
    lower_native_shared_parametric_ancestral_schedule,
)
from ..ir.task import BFTaskModule
from .crown_ibp import _forward_ibp_trace_mlp
from .native_alpha_beta_optimization_state import (
    NativeAlphaBetaOptimizationState,
    NativeAlphaBetaOptimizerPolicy,
)
from .native_intermediate_refinement import (
    NativeIntermediateRefinementExecution,
    intermediate_bounds_hash,
    intermediate_refinement_semantic_trace_hash,
)
from .native_objective_ancestral_queue import NativeObjectiveAncestralNodeRefinement
from .native_objective_ancestral_sibling_pack import (
    _project_objective,
    compile_native_objective_ancestral_sibling_pack_plan,
    validate_native_objective_ancestral_sibling_pack_plan,
)
from .native_optimized_relu_split_bab_runtime import (
    _batched_split_state,
    _build_batched_parent_warm_state,
    _execute_per_child_refinements,
    _repeat_relu_pre_override,
)
from .native_parametric_optimizer import (
    NativeParametricOptimizerTemplateCache,
    execute_native_parametric_optimizer,
    instantiate_native_parametric_optimizer,
)
from .native_parametric_production_verifier import (
    NativeParametricCompilerBatchTrace,
    NativeParametricProductionReluSplitBabExecution,
    _parametric_production_plan,
    _slice_parametric_state,
)
from .native_production_verifier import (
    NativeProductionBabEvaluation,
    NativeProductionReluSplitBabExecution,
    NativeProductionReluSplitBabTrace,
    NativeProductionVerifierActionTrace,
    NativeProductionVerifierBatchTrace,
)
from .native_relu_split_bab_runtime import (
    BabQueueStatus,
    NativeReluSplitBabConfig,
    NativeReluSplitBabDecision,
    NativeReluSplitBabNode,
    _make_child_runtime_node,
    _priority,
    _QueueEntry,
    _repeat_box_input_spec,
    _root_box_bounds,
    _RuntimeNode,
    _select_branch,
    _slice_interval,
)
from .task_executor import InputSpec

NATIVE_SHARED_PARAMETRIC_ANCESTRAL_TRACE_SCHEMA_VERSION = (
    "boundflow.native-shared-parametric-ancestral-trace/v1"
)
ClockNs = Callable[[], int]


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _SharedEvaluatedNode:
    runtime_node: _RuntimeNode
    evaluation: NativeProductionBabEvaluation
    selected_state: NativeAlphaBetaOptimizationState
    relu_pre: Mapping[str, IntervalState]
    refinement_execution: NativeIntermediateRefinementExecution


@dataclass(frozen=True)
class NativeSharedParametricAncestralTrace:
    """Replay identity for one deadline-bounded shared-template clause queue."""

    query_id: str
    plan_hash: str
    task_ir_hash: str
    schedule_hash: str
    queue_trace_hash: str
    batch_commit_hashes: Tuple[str, ...]
    compiler_batch_hashes: Tuple[str, ...]
    node_refinement_semantics: Tuple[dict[str, object], ...]
    cache_outcomes: Tuple[str, ...]
    fallback_reason: str
    discarded_attempt_stage: Optional[str]
    discarded_compiler_batch_hash: Optional[str]
    source_elapsed_ns: int
    queue_elapsed_ns: int
    whole_elapsed_ns: int
    deadline_ns: int
    semantic_signature_hash: str
    audit_hash_chain_constructed: bool = False
    selected_native_reexecution: bool = False
    performance_claimed: bool = False
    schema_version: str = NATIVE_SHARED_PARAMETRIC_ANCESTRAL_TRACE_SCHEMA_VERSION

    def semantic_dict(self) -> dict[str, object]:
        return {
            "plan_hash": self.plan_hash,
            "task_ir_hash": self.task_ir_hash,
            "schedule_hash": self.schedule_hash,
            "queue_trace_hash": self.queue_trace_hash,
            "batch_commit_hashes": list(self.batch_commit_hashes),
            "compiler_batch_hashes": list(self.compiler_batch_hashes),
            "node_refinement_semantics": list(self.node_refinement_semantics),
            "cache_outcomes": list(self.cache_outcomes),
            "fallback_reason": self.fallback_reason,
            "discarded_attempt_stage": self.discarded_attempt_stage,
            "discarded_compiler_batch_hash": self.discarded_compiler_batch_hash,
            "deadline_ns": self.deadline_ns,
            "audit_hash_chain_constructed": self.audit_hash_chain_constructed,
            "selected_native_reexecution": self.selected_native_reexecution,
            "performance_claimed": self.performance_claimed,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "query_id": self.query_id,
            **self.semantic_dict(),
            "source_elapsed_ns": self.source_elapsed_ns,
            "queue_elapsed_ns": self.queue_elapsed_ns,
            "whole_elapsed_ns": self.whole_elapsed_ns,
            "semantic_signature_hash": self.semantic_signature_hash,
        }


@dataclass(frozen=True)
class NativeSharedParametricAncestralExecution:
    """First-class shared compiler, dynamic batch, refinement, and queue proof."""

    plan: NativeSharedParametricAncestralPlanIR
    task_ir: NativeSharedParametricAncestralTaskIRModule
    schedule: NativeSharedParametricAncestralScheduleIR
    queue: NativeProductionReluSplitBabExecution
    compiler_batches: Tuple[NativeParametricCompilerBatchTrace, ...]
    batch_commits: Tuple[NativeSharedParametricAncestralBatchIR, ...]
    node_refinements: Tuple[NativeObjectiveAncestralNodeRefinement, ...]
    discarded_compiler_batch: Optional[NativeParametricCompilerBatchTrace]
    trace: NativeSharedParametricAncestralTrace

    def validate_against(
        self,
        module: BFTaskModule,
        input_spec: InputSpec,
        *,
        linear_spec_C: torch.Tensor,
        threshold: torch.Tensor,
        root_refinement: NativeIntermediateRefinementExecution,
        optimizer_policy: NativeAlphaBetaOptimizerPolicy,
    ) -> None:  # pylint: disable=too-many-statements
        validate_native_shared_parametric_ancestral_plan(
            self.plan,
            module,
            input_spec,
            linear_spec_C=linear_spec_C,
            threshold=threshold,
            root_refinement=root_refinement,
            optimizer_policy=optimizer_policy,
        )
        self.schedule.validate_against(self.task_ir)
        parametric_queue = NativeParametricProductionReluSplitBabExecution(
            queue=self.queue, compiler_batches=self.compiler_batches
        )
        parametric_queue.validate()
        if (
            len(self.batch_commits) != len(self.compiler_batches)
            or len(self.batch_commits) != len(self.queue.trace.batches)
            or not self.batch_commits
        ):
            raise ValueError("shared-parametric batch/compiler coverage differs")
        evaluations_by_id = {
            item.node.node_id: item for item in self.queue.trace.evaluations
        }
        refinements = {item.node_id: item for item in self.node_refinements}
        if (
            len(evaluations_by_id) != len(self.queue.trace.evaluations)
            or len(refinements) != len(self.node_refinements)
            or tuple(refinements) != tuple(evaluations_by_id)
            or len(evaluations_by_id) != 1 + 2 * (len(self.batch_commits) - 1)
        ):
            raise ValueError("shared-parametric node/refinement coverage differs")
        template_hashes: set[str] = set()
        event_indices: list[int] = []
        for index, (commit, compiler, batch) in enumerate(
            zip(self.batch_commits, self.compiler_batches, self.queue.trace.batches)
        ):
            commit.validate()
            compiler.validate()
            batch.validate()
            batch_evaluations = tuple(
                item
                for item in self.queue.trace.evaluations
                if item.eval_batch_id == batch.plan.plan_id
            )
            batch_refinements = tuple(
                refinements[item.node.node_id] for item in batch_evaluations
            )
            expected_parent = (
                None if index == 0 else batch_evaluations[0].node.parent_node_id
            )
            if (
                commit.batch_index != index
                or commit.batch_id != batch.plan.plan_id
                or commit.node_ids
                != tuple(item.node.node_id for item in batch_evaluations)
                or commit.parent_node_id != expected_parent
                or (
                    index > 0
                    and any(
                        item.node.parent_node_id != expected_parent
                        for item in batch_evaluations
                    )
                )
                or commit.node_split_state_hashes
                != tuple(item.node.split_state_hash for item in batch_evaluations)
                or commit.refinement_semantic_trace_hashes
                != tuple(
                    intermediate_refinement_semantic_trace_hash(item.execution)
                    for item in batch_refinements
                )
                or commit.production_batch_trace_hash != batch.stable_hash()
                or commit.compiler_batch_trace_hash != compiler.stable_hash()
                or commit.cache_event_hash != compiler.cache_event.stable_hash()
                or commit.instance_hash != compiler.instance_ir.stable_hash()
                or commit.template_hash != compiler.template_hash
                or commit.optimizer_task_hash != compiler.task_hash
                or commit.optimizer_schedule_hash != compiler.schedule_hash
                or commit.evaluation_hashes
                != tuple(_canonical_hash(item.to_dict()) for item in batch_evaluations)
            ):
                raise ValueError("shared-parametric Batch binding differs")
            template_hashes.add(compiler.template_hash)
            event_indices.append(compiler.cache_event.event_index)
        if len(template_hashes) != 1 or event_indices != list(
            range(event_indices[0], event_indices[0] + len(event_indices))
        ):
            raise ValueError("shared-parametric cache ownership differs")
        outcomes = tuple(item.cache_event.outcome for item in self.compiler_batches)
        if "miss_compiled" in outcomes[1:]:
            raise ValueError("shared-parametric cache recompiled after first batch")
        source = linear_spec_C
        evaluator = _project_objective(source)
        root_id = self.queue.trace.evaluations[0].node.node_id
        for node_id, refinement in refinements.items():
            evaluation = evaluations_by_id[node_id]
            refinement.program.validate(module, input_spec)
            refinement.execution.validate(module, input_spec)
            expected_objective_hash = tensor_content_hash(
                source if node_id == root_id else evaluator
            )
            if (
                refinement.execution.program != refinement.program
                or refinement.node_split_state_hash != evaluation.node.split_state_hash
                or refinement.parent_node_id != evaluation.node.parent_node_id
                or refinement.program.plan.objective_hash != expected_objective_hash
            ):
                raise ValueError("shared-parametric refinement binding differs")
            if node_id == root_id:
                if refinement.execution != root_refinement:
                    raise ValueError("shared-parametric root refinement differs")
            else:
                parent = refinements.get(refinement.parent_node_id or "")
                if (
                    parent is None
                    or refinement.program.plan.source_intermediate_constraints_hash
                    != intermediate_bounds_hash(parent.execution.relu_pre)
                    or refinement.program.plan.source_refinement_plan_hash
                    != parent.program.plan.stable_hash()
                    or refinement.program.plan.source_refinement_semantic_trace_hash
                    != intermediate_refinement_semantic_trace_hash(parent.execution)
                ):
                    raise ValueError("shared-parametric ancestral lineage differs")
        discarded_hash = (
            None
            if self.discarded_compiler_batch is None
            else self.discarded_compiler_batch.stable_hash()
        )
        if self.discarded_compiler_batch is not None:
            self.discarded_compiler_batch.validate()
        expected_semantics = tuple(
            item.semantic_dict() for item in self.node_refinements
        )
        if (
            self.trace.schema_version
            != NATIVE_SHARED_PARAMETRIC_ANCESTRAL_TRACE_SCHEMA_VERSION
            or not self.trace.query_id
            or self.trace.plan_hash != self.plan.stable_hash()
            or self.trace.task_ir_hash != self.task_ir.stable_hash()
            or self.trace.schedule_hash != self.schedule.stable_hash(self.task_ir)
            or self.trace.queue_trace_hash != self.queue.trace.stable_hash()
            or self.trace.batch_commit_hashes
            != tuple(item.stable_hash() for item in self.batch_commits)
            or self.trace.compiler_batch_hashes
            != tuple(item.stable_hash() for item in self.compiler_batches)
            or self.trace.node_refinement_semantics != expected_semantics
            or self.trace.cache_outcomes != outcomes
            or not self.trace.fallback_reason
            or self.trace.discarded_compiler_batch_hash != discarded_hash
            or min(
                self.trace.source_elapsed_ns,
                self.trace.queue_elapsed_ns,
                self.trace.whole_elapsed_ns,
            )
            < 0
            or self.trace.deadline_ns != self.plan.whole_query_timeout_ns
            or self.trace.audit_hash_chain_constructed is not False
            or self.trace.selected_native_reexecution is not False
            or self.trace.performance_claimed is not False
            or self.trace.semantic_signature_hash
            != _canonical_hash(self.trace.semantic_dict())
        ):
            raise ValueError("shared-parametric ancestral trace differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "plan": self.plan.to_dict(),
            "plan_hash": self.plan.stable_hash(),
            "task_ir": self.task_ir.to_dict(),
            "task_ir_hash": self.task_ir.stable_hash(),
            "schedule": self.schedule.to_dict(),
            "schedule_hash": self.schedule.stable_hash(self.task_ir),
            "queue_trace": self.queue.trace.to_dict(),
            "compiler_batches": [item.to_dict() for item in self.compiler_batches],
            "batch_commits": [item.to_dict() for item in self.batch_commits],
            "node_refinements": [
                item.semantic_dict() for item in self.node_refinements
            ],
            "discarded_compiler_batch": (
                None
                if self.discarded_compiler_batch is None
                else self.discarded_compiler_batch.to_dict()
            ),
            "trace": self.trace.to_dict(),
        }


def compile_native_shared_parametric_ancestral_plan(
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    linear_spec_C: torch.Tensor,
    threshold: torch.Tensor,
    root_refinement: NativeIntermediateRefinementExecution,
    optimizer_policy: NativeAlphaBetaOptimizerPolicy,
    plan_id: str,
) -> NativeSharedParametricAncestralPlanIR:
    sibling = compile_native_objective_ancestral_sibling_pack_plan(
        module,
        input_spec,
        linear_spec_C=linear_spec_C,
        threshold=threshold,
        root_refinement=root_refinement,
        optimizer_policy=optimizer_policy,
        plan_id=plan_id,
    )
    plan = NativeSharedParametricAncestralPlanIR(
        plan_id=sibling.plan_id,
        sibling_pack_plan=sibling,
        primal_graph_hash=sibling.primal_graph_hash,
        input_bounds_hash=sibling.input_bounds_hash,
        source_objective_hash=sibling.source_objective_hash,
        evaluator_objective_hash=sibling.evaluator_objective_hash,
        threshold_hash=sibling.threshold_hash,
        root_refinement_semantic_trace_hash=(
            sibling.root_refinement_semantic_trace_hash
        ),
        optimizer_policy_hash=sibling.optimizer_policy_hash,
        max_nodes=sibling.search_budget.max_nodes,
        max_depth=sibling.search_budget.max_depth,
        child_refinement_cap=(sibling.child_refinement_policy.max_neurons_per_relu),
        whole_query_timeout_ns=sibling.whole_query_timeout_ns,
    )
    plan.validate()
    return plan


def validate_native_shared_parametric_ancestral_plan(
    plan: NativeSharedParametricAncestralPlanIR,
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    linear_spec_C: torch.Tensor,
    threshold: torch.Tensor,
    root_refinement: NativeIntermediateRefinementExecution,
    optimizer_policy: NativeAlphaBetaOptimizerPolicy,
) -> None:
    plan.validate()
    validate_native_objective_ancestral_sibling_pack_plan(
        plan.sibling_pack_plan,
        module,
        input_spec,
        linear_spec_C=linear_spec_C,
        threshold=threshold,
        root_refinement=root_refinement,
        optimizer_policy=optimizer_policy,
    )


def _evaluate_shared_parametric_batch(  # pylint: disable=too-many-statements
    module: BFTaskModule,
    root_input_spec: InputSpec,
    *,
    objective: torch.Tensor,
    nodes: tuple[_RuntimeNode, ...],
    batch_id: str,
    policy: NativeAlphaBetaOptimizerPolicy,
    parent_by_id: Mapping[str, _SharedEvaluatedNode],
    root_refinement: Optional[NativeIntermediateRefinementExecution],
    child_refinement_policy: NativeIntermediateRefinementPolicyIR,
    compiler_cache: NativeParametricOptimizerTemplateCache,
) -> tuple[
    tuple[_SharedEvaluatedNode, ...],
    NativeProductionVerifierBatchTrace,
    NativeParametricCompilerBatchTrace,
    tuple[tuple[str, NativeIntermediateRefinementExecution], ...],
]:
    if not nodes:
        raise ValueError("shared-parametric batch cannot be empty")
    is_root = nodes[0].node.depth == 0
    if any((node.node.depth == 0) != is_root for node in nodes):
        raise ValueError("shared-parametric batch mixes root and child nodes")
    batch_input = _repeat_box_input_spec(root_input_spec, count=len(nodes))
    split_batch = _batched_split_state(nodes)
    refinement_executions: tuple[tuple[str, NativeIntermediateRefinementExecution], ...]
    if is_root:
        if root_refinement is None or len(nodes) != 1:
            raise ValueError("shared-parametric root refinement differs")
        refinement_executions = ((nodes[0].node.node_id, root_refinement),)
        batch_relu_pre = _repeat_relu_pre_override(
            root_refinement.relu_pre, count=len(nodes)
        )
        warm_state = None
    else:
        if root_refinement is not None:
            raise ValueError("shared-parametric child received root refinement")
        batch_relu_pre, refinement_executions, _records = (
            _execute_per_child_refinements(
                module,
                root_input_spec,
                objective=objective,
                nodes=nodes,
                policy=child_refinement_policy,
                budget_policy=None,
                multi_pass_policy=None,
                budget_group_id=batch_id,
                parent_by_id=parent_by_id,  # type: ignore[arg-type]
                strategy="ancestral_constraint_carry_v1",
                external_constraint_seed=None,
            )
        )
        warm_state = _build_batched_parent_warm_state(
            module,
            batch_input,
            objective=objective,
            nodes=nodes,
            policy=policy,
            parent_by_id=parent_by_id,  # type: ignore[arg-type]
            relu_pre_override=batch_relu_pre,
            intermediate_bound_source=IntermediateBoundSource.NATIVE_REFINED,
            refine_external_constraints=False,
            use_parent_runtime_bounds=True,
        )
    acquire_started_ns = time.perf_counter_ns()
    template, cache_event = compiler_cache.acquire(
        module,
        batch_input,
        linear_spec_C=objective,
        relu_pre=batch_relu_pre,
        policy=policy,
        intermediate_bound_source=IntermediateBoundSource.NATIVE_REFINED,
        refine_external_constraints=False,
        template_id=f"{batch_id}:optimizer-template",
        batch_id=batch_id,
    )
    acquire_elapsed_ns = time.perf_counter_ns() - acquire_started_ns
    instantiate_started_ns = time.perf_counter_ns()
    instance = instantiate_native_parametric_optimizer(
        template,
        module,
        batch_input,
        linear_spec_C=objective,
        relu_split_state=split_batch,
        instance_id=f"{batch_id}:optimizer-instance",
        warm_start=warm_state,
        relu_pre_override=batch_relu_pre,
        intermediate_bound_source=IntermediateBoundSource.NATIVE_REFINED,
        refine_external_constraints=False,
    )
    instantiate_elapsed_ns = time.perf_counter_ns() - instantiate_started_ns
    production_plan = _parametric_production_plan(
        batch_id=batch_id,
        nodes=nodes,
        parent_by_id=parent_by_id,  # type: ignore[arg-type]
        instance=instance,
    )
    task_ir, schedule = lower_native_production_verifier_ir(production_plan)
    selected = None
    materialized: Optional[tuple[_SharedEvaluatedNode, ...]] = None
    action_traces: list[NativeProductionVerifierActionTrace] = []
    execute_elapsed_ns = 0
    refinement_by_id = dict(refinement_executions)
    for action in schedule.actions:
        started_ns = time.perf_counter_ns()
        if action.kind == NativeProductionVerifierTaskKind.VALIDATE_PROGRAM:
            instance.require_exact_runtime(
                module,
                batch_input,
                linear_spec_C=objective,
                intermediate_bound_source=IntermediateBoundSource.NATIVE_REFINED,
            )
        elif action.kind == NativeProductionVerifierTaskKind.EXECUTE_OPTIMIZER:
            selected = execute_native_parametric_optimizer(
                instance,
                module,
                batch_input,
                linear_spec_C=objective,
                intermediate_bound_source=IntermediateBoundSource.NATIVE_REFINED,
            )
        elif action.kind == NativeProductionVerifierTaskKind.MATERIALIZE_NODE_RESULTS:
            if selected is None:
                raise ValueError("shared-parametric materializes before execution")
            if tuple(selected.bounds.lower.shape) != (len(nodes), 1):
                raise ValueError("shared-parametric batch must return scalar bounds")
            values: list[_SharedEvaluatedNode] = []
            for index, runtime_node in enumerate(nodes):
                node_pre = {
                    name: _slice_interval(value, index=index)
                    for name, value in instance.relu_pre.items()
                }
                node_split = {
                    name: value[index : index + 1].contiguous()
                    for name, value in split_batch.items()
                }
                state = _slice_parametric_state(
                    instance,
                    root_input_spec,
                    module=module,
                    objective=objective,
                    selected_state=selected.state,
                    index=index,
                )
                parent = (
                    None
                    if runtime_node.node.parent_node_id is None
                    else parent_by_id.get(runtime_node.node.parent_node_id)
                )
                if runtime_node.node.depth > 0 and parent is None:
                    raise ValueError("shared-parametric node lacks a parent")
                branch = _select_branch(node_pre, relu_split_state=node_split)
                lower = float(selected.bounds.lower[index, 0].item())
                upper = float(selected.bounds.upper[index, 0].item())
                values.append(
                    _SharedEvaluatedNode(
                        runtime_node=runtime_node,
                        evaluation=NativeProductionBabEvaluation(
                            node=runtime_node.node,
                            lower=lower,
                            upper=upper,
                            priority=_priority(lower),
                            selected_state_hash=state.stable_hash(),
                            parent_selected_state_hash=(
                                None
                                if parent is None
                                else parent.selected_state.stable_hash()
                            ),
                            warm_start_kind=instance.ir.warm_start_kind,
                            eval_batch_id=batch_id,
                            eval_batch_position=index,
                            batch_trace_hash="0" * 64,
                            branch_candidate=branch,
                        ),
                        selected_state=state,
                        relu_pre=node_pre,
                        refinement_execution=refinement_by_id[
                            runtime_node.node.node_id
                        ],
                    )
                )
            materialized = tuple(values)
        elif action.kind == NativeProductionVerifierTaskKind.COMMIT_QUEUE_RESULTS:
            if materialized is None:
                raise ValueError("shared-parametric commits before materialization")
        else:
            raise AssertionError("unreachable production verifier action")
        elapsed_ns = time.perf_counter_ns() - started_ns
        if action.kind == NativeProductionVerifierTaskKind.EXECUTE_OPTIMIZER:
            execute_elapsed_ns = elapsed_ns
        action_traces.append(
            NativeProductionVerifierActionTrace(
                sequence=action.sequence,
                action_id=action.action_id,
                task_id=action.task_id,
                kind=action.kind,
                elapsed_ns=elapsed_ns,
            )
        )
    if selected is None or materialized is None:
        raise ValueError("shared-parametric Schedule produced no result")
    batch_trace = NativeProductionVerifierBatchTrace(
        plan=production_plan,
        task_ir=task_ir,
        schedule=schedule,
        actions=tuple(action_traces),
        selected_batch_state_hash=selected.state.stable_hash(),
    )
    batch_trace.validate()
    trace_hash = batch_trace.stable_hash()
    rebound = tuple(
        replace(
            item,
            evaluation=replace(item.evaluation, batch_trace_hash=trace_hash),
        )
        for item in materialized
    )
    for item in rebound:
        item.evaluation.validate()
    compiler = NativeParametricCompilerBatchTrace(
        batch_id=batch_id,
        cache_event=cache_event,
        instance_ir=instance.ir,
        template_hash=template.template_hash,
        task_hash=template.task_hash,
        schedule_hash=template.schedule_hash,
        acquire_elapsed_ns=acquire_elapsed_ns,
        instantiate_elapsed_ns=instantiate_elapsed_ns,
        execute_elapsed_ns=execute_elapsed_ns,
    )
    compiler.validate()
    return rebound, batch_trace, compiler, refinement_executions


def _make_batch_commit(
    *,
    batch_index: int,
    evaluated: tuple[_SharedEvaluatedNode, ...],
    batch: NativeProductionVerifierBatchTrace,
    compiler: NativeParametricCompilerBatchTrace,
    refinements: tuple[tuple[str, NativeIntermediateRefinementExecution], ...],
) -> NativeSharedParametricAncestralBatchIR:
    parent = evaluated[0].runtime_node.node.parent_node_id
    return NativeSharedParametricAncestralBatchIR.committed(
        batch_index=batch_index,
        batch_id=batch.plan.plan_id,
        commit_kind="root" if batch_index == 0 else "atomic_sibling_pair",
        node_ids=tuple(item.runtime_node.node.node_id for item in evaluated),
        parent_node_id=parent,
        node_split_state_hashes=tuple(
            item.runtime_node.node.split_state_hash for item in evaluated
        ),
        refinement_semantic_trace_hashes=tuple(
            intermediate_refinement_semantic_trace_hash(execution)
            for _node_id, execution in refinements
        ),
        production_batch_trace_hash=batch.stable_hash(),
        compiler_batch_trace_hash=compiler.stable_hash(),
        cache_event_hash=compiler.cache_event.stable_hash(),
        instance_hash=compiler.instance_ir.stable_hash(),
        template_hash=compiler.template_hash,
        optimizer_task_hash=compiler.task_hash,
        optimizer_schedule_hash=compiler.schedule_hash,
        evaluation_hashes=tuple(
            _canonical_hash(item.evaluation.to_dict()) for item in evaluated
        ),
        selected_native_reexecution=False,
    )


def _task(
    tasks: list[NativeSharedParametricAncestralTaskIRUnit],
    plan: NativeSharedParametricAncestralPlanIR,
    *,
    suffix: str,
    kind: NativeSharedParametricAncestralTaskKind,
    dependencies: Sequence[str],
    batch_id: Optional[str],
    inputs: Mapping[str, str],
    output: object,
) -> str:
    task_id = f"{plan.plan_id}:{suffix}"
    task = NativeSharedParametricAncestralTaskIRUnit(
        sequence=len(tasks),
        task_id=task_id,
        kind=kind,
        dependency_task_ids=tuple(dependencies),
        batch_id=batch_id,
        input_hashes=tuple(sorted(inputs.items())),
        output_hash=_canonical_hash(output),
    )
    task.validate()
    tasks.append(task)
    return task_id


def _append_batch_tasks(
    tasks: list[NativeSharedParametricAncestralTaskIRUnit],
    plan: NativeSharedParametricAncestralPlanIR,
    *,
    batch: NativeSharedParametricAncestralBatchIR,
    compiler: NativeParametricCompilerBatchTrace,
    dependencies: Sequence[str],
) -> str:
    acquire = _task(
        tasks,
        plan,
        suffix=f"batch:{batch.batch_index:04d}:acquire-template",
        kind=NativeSharedParametricAncestralTaskKind.ACQUIRE_TEMPLATE,
        dependencies=dependencies,
        batch_id=batch.batch_id,
        inputs={"plan": plan.stable_hash()},
        output=compiler.cache_event.to_dict(),
    )
    instantiate = _task(
        tasks,
        plan,
        suffix=f"batch:{batch.batch_index:04d}:instantiate",
        kind=NativeSharedParametricAncestralTaskKind.INSTANTIATE_BATCH,
        dependencies=(acquire,),
        batch_id=batch.batch_id,
        inputs={
            "template": compiler.template_hash,
            "cache_event": compiler.cache_event.stable_hash(),
        },
        output=compiler.instance_ir.to_dict(),
    )
    execute = _task(
        tasks,
        plan,
        suffix=f"batch:{batch.batch_index:04d}:execute",
        kind=NativeSharedParametricAncestralTaskKind.EXECUTE_BATCH,
        dependencies=(instantiate,),
        batch_id=batch.batch_id,
        inputs={
            "instance": compiler.instance_ir.stable_hash(),
            "optimizer_task": compiler.task_hash,
            "optimizer_schedule": compiler.schedule_hash,
        },
        output=batch.production_batch_trace_hash,
    )
    return _task(
        tasks,
        plan,
        suffix=f"batch:{batch.batch_index:04d}:commit",
        kind=(
            NativeSharedParametricAncestralTaskKind.COMMIT_ROOT
            if batch.batch_index == 0
            else NativeSharedParametricAncestralTaskKind.COMMIT_SIBLING_PAIR
        ),
        dependencies=(execute,),
        batch_id=batch.batch_id,
        inputs={
            "production_batch": batch.production_batch_trace_hash,
            "compiler_batch": batch.compiler_batch_trace_hash,
        },
        output=batch.to_dict(),
    )


def execute_native_shared_parametric_ancestral_queue(  # pylint: disable=too-many-statements
    plan: NativeSharedParametricAncestralPlanIR,
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    linear_spec_C: torch.Tensor,
    threshold: torch.Tensor,
    root_refinement: NativeIntermediateRefinementExecution,
    optimizer_policy: NativeAlphaBetaOptimizerPolicy,
    compiler_cache: NativeParametricOptimizerTemplateCache,
    query_id: str,
    whole_query_started_ns: Optional[int] = None,
    clock_ns: ClockNs = time.monotonic_ns,
) -> NativeSharedParametricAncestralExecution:
    if not query_id:
        raise ValueError("shared-parametric query ID must be non-empty")
    if not isinstance(compiler_cache, NativeParametricOptimizerTemplateCache):
        raise TypeError("shared-parametric compiler cache is invalid")
    validate_native_shared_parametric_ancestral_plan(
        plan,
        module,
        input_spec,
        linear_spec_C=linear_spec_C,
        threshold=threshold,
        root_refinement=root_refinement,
        optimizer_policy=optimizer_policy,
    )
    sibling_plan = plan.sibling_pack_plan
    objective = _project_objective(linear_spec_C)
    threshold_value = float(threshold.reshape(-1)[0].item())
    started_ns = (
        clock_ns() if whole_query_started_ns is None else whole_query_started_ns
    )
    queue_started_ns = clock_ns()
    deadline_at_ns = started_ns + plan.whole_query_timeout_ns
    config = NativeReluSplitBabConfig(
        max_nodes=plan.max_nodes,
        max_depth=plan.max_depth,
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
        plan,
        suffix="admit-query",
        kind=NativeSharedParametricAncestralTaskKind.ADMIT_QUERY,
        dependencies=(),
        batch_id=None,
        inputs={"sibling_pack_plan": sibling_plan.stable_hash()},
        output=plan.to_dict(),
    )
    root_values, root_batch, root_compiler, root_refinements = (
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
    if clock_ns() > deadline_at_ns:
        raise RuntimeError("shared-parametric deadline expired at root")
    root_commit = _make_batch_commit(
        batch_index=0,
        evaluated=root_values,
        batch=root_batch,
        compiler=root_compiler,
        refinements=root_refinements,
    )
    root_task = _append_batch_tasks(
        tasks,
        plan,
        batch=root_commit,
        compiler=root_compiler,
        dependencies=(admit,),
    )
    evaluations = [root_values[0].evaluation]
    decisions: list[NativeReluSplitBabDecision] = []
    batches = [root_batch]
    compilers = [root_compiler]
    commits = [root_commit]
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
    discarded_compiler: Optional[NativeParametricCompilerBatchTrace] = None

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
                plan,
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
                plan,
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
            reason="widest_unsplit_ambiguous_relu",
            child_node_ids=tuple(child.node.node_id for child in children),
            branch_candidate=branch,
        )
        if clock_ns() >= deadline_at_ns:
            heapq.heappush(heap, entry)
            deadline_exhausted = True
            discarded_stage = "before_sibling_pair"
            break
        child_values, child_batch, child_compiler, child_refinement_values = (
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
        if len(child_values) != 2 or len(child_refinement_values) != 2:
            raise ValueError("shared-parametric sibling evaluator coverage differs")
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
            plan,
            suffix=f"{node.node_id}:transition",
            kind=NativeSharedParametricAncestralTaskKind.TRANSITION_QUEUE,
            dependencies=(evaluation_task_by_id[node.node_id],),
            batch_id=result.eval_batch_id,
            inputs={"evaluation": _canonical_hash(result.to_dict())},
            output=decision.to_dict(),
        )
        commit_task = _append_batch_tasks(
            tasks,
            plan,
            batch=child_commit,
            compiler=child_compiler,
            dependencies=(transition_task,),
        )
        decisions.append(decision)
        batches.append(child_batch)
        compilers.append(child_compiler)
        commits.append(child_commit)
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
    )
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
        plan,
        suffix="emit",
        kind=NativeSharedParametricAncestralTaskKind.EMIT_RESULT,
        dependencies=emit_dependencies,
        batch_id=None,
        inputs={"queue_trace": queue.trace.stable_hash()},
        output=queue.trace.to_dict(),
    )
    task_ir, schedule = lower_native_shared_parametric_ancestral_schedule(
        plan, tuple(tasks), tuple(commits)
    )
    finished_ns = clock_ns()
    trace = NativeSharedParametricAncestralTrace(
        query_id=query_id,
        plan_hash=plan.stable_hash(),
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
        deadline_ns=plan.whole_query_timeout_ns,
        semantic_signature_hash="",
    )
    trace = replace(
        trace, semantic_signature_hash=_canonical_hash(trace.semantic_dict())
    )
    execution = NativeSharedParametricAncestralExecution(
        plan=plan,
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
    return execution


__all__ = [
    "NativeSharedParametricAncestralExecution",
    "NativeSharedParametricAncestralTrace",
    "compile_native_shared_parametric_ancestral_plan",
    "execute_native_shared_parametric_ancestral_queue",
    "validate_native_shared_parametric_ancestral_plan",
]
