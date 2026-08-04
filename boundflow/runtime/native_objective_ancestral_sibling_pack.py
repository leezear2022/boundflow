"""Typed objective-ancestral queue with atomic sibling-packed evaluation."""

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

from ..frontends.plain_crown_bound_ir import (
    plain_crown_primal_graph_hash,
    relu_split_state_hash,
    tensor_content_hash,
)
from ..ir.bound import IntermediateBoundSource
from ..ir.objective_ancestral_queue import NativeObjectiveAncestralQueuePlanIR
from ..ir.objective_ancestral_sibling_pack import (
    NativeObjectiveAncestralSiblingGroupExecutionIR,
    NativeObjectiveAncestralSiblingPackPlanIR,
    NativeObjectiveAncestralSiblingPackScheduleIR,
    NativeObjectiveAncestralSiblingPackTaskIRModule,
    NativeObjectiveAncestralSiblingPackTaskIRUnit,
    NativeObjectiveAncestralSiblingPackTaskKind,
    lower_native_objective_ancestral_sibling_pack_schedule,
)
from ..ir.task import BFTaskModule
from .crown_ibp import _forward_ibp_trace_mlp
from .native_alpha_beta_optimization_state import NativeAlphaBetaOptimizerPolicy
from .native_intermediate_refinement import (
    NativeIntermediateRefinementExecution,
    _input_bounds_hash,
    intermediate_bounds_hash,
    intermediate_refinement_semantic_trace_hash,
)
from .native_objective_ancestral_queue import (
    NativeObjectiveAncestralNodeRefinement,
    _normalize_objective,
    _threshold_hash,
    compile_native_objective_ancestral_queue_plan,
    validate_native_objective_ancestral_queue_plan,
)
from .native_optimized_relu_split_bab_runtime import (
    NativeOptimizedReluSplitBabExecution,
    NativeOptimizedReluSplitBabTrace,
    _OptimizedEvaluatedNode,
    _QueueEntry,
    _RuntimeNode,
    _evaluate_optimized_node_batch,
    _make_child_runtime_node,
    _root_box_bounds,
)
from .native_relu_split_bab_runtime import (
    BabQueueStatus,
    NativeReluSplitBabConfig,
    NativeReluSplitBabDecision,
    NativeReluSplitBabNode,
)
from .task_executor import InputSpec

NATIVE_OBJECTIVE_ANCESTRAL_SIBLING_PACK_TRACE_SCHEMA_VERSION = (
    "boundflow.native-objective-ancestral-sibling-pack-trace/v1"
)
ClockNs = Callable[[], int]


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _project_objective(source: torch.Tensor) -> torch.Tensor:
    normalized = _normalize_objective(source)
    if normalized.dim() != 3 or tuple(normalized.shape[:2]) != (1, 1):
        raise ValueError("sibling-pack source objective requires one singleton domain")
    return normalized[0].detach().contiguous().clone()


def _as_queue_plan(
    plan: NativeObjectiveAncestralSiblingPackPlanIR,
) -> NativeObjectiveAncestralQueuePlanIR:
    return NativeObjectiveAncestralQueuePlanIR(
        plan_id=plan.plan_id,
        primal_graph_hash=plan.primal_graph_hash,
        input_bounds_hash=plan.input_bounds_hash,
        objective_hash=plan.source_objective_hash,
        threshold_hash=plan.threshold_hash,
        root_refinement_plan_hash=plan.root_refinement_plan_hash,
        root_refinement_semantic_trace_hash=(plan.root_refinement_semantic_trace_hash),
        root_intermediate_bounds_hash=plan.root_intermediate_bounds_hash,
        optimizer_policy_hash=plan.optimizer_policy_hash,
        search_budget=plan.search_budget,
        child_refinement_policy=plan.child_refinement_policy,
        whole_query_timeout_ns=plan.whole_query_timeout_ns,
    )


def _evaluation_hash(value: _OptimizedEvaluatedNode) -> str:
    return _canonical_hash(value.evaluation.to_dict())


def _normalized_evaluated(value: _OptimizedEvaluatedNode) -> _OptimizedEvaluatedNode:
    return replace(
        value,
        evaluation=replace(value.evaluation, intermediate_refinement_trace_hash=None),
    )


def _task(
    tasks: list[NativeObjectiveAncestralSiblingPackTaskIRUnit],
    plan: NativeObjectiveAncestralSiblingPackPlanIR,
    *,
    suffix: str,
    kind: NativeObjectiveAncestralSiblingPackTaskKind,
    dependencies: Sequence[str],
    group_id: Optional[str],
    node_ids: Sequence[str],
    inputs: Mapping[str, str],
    output: object,
) -> str:
    task_id = f"{plan.plan_id}:{suffix}"
    task = NativeObjectiveAncestralSiblingPackTaskIRUnit(
        sequence=len(tasks),
        task_id=task_id,
        kind=kind,
        dependency_task_ids=tuple(dependencies),
        group_id=group_id,
        node_ids=tuple(node_ids),
        input_hashes=tuple(sorted(inputs.items())),
        output_hash=_canonical_hash(output),
    )
    task.validate()
    tasks.append(task)
    return task_id


@dataclass(frozen=True)
class NativeObjectiveAncestralSiblingPackTrace:
    """Replay identity for objective projection and atomic sibling groups."""

    query_id: str
    plan_hash: str
    task_ir_hash: str
    schedule_hash: str
    queue_trace_hash: str
    sibling_group_hashes: Tuple[str, ...]
    node_refinement_semantics: Tuple[dict[str, object], ...]
    objective_projection: str
    fallback_reason: str
    discarded_attempt_stage: Optional[str]
    source_elapsed_ns: int
    queue_elapsed_ns: int
    whole_elapsed_ns: int
    deadline_ns: int
    semantic_signature_hash: str
    performance_claimed: bool = False
    schema_version: str = NATIVE_OBJECTIVE_ANCESTRAL_SIBLING_PACK_TRACE_SCHEMA_VERSION

    def semantic_dict(self) -> dict[str, object]:
        return {
            "plan_hash": self.plan_hash,
            "task_ir_hash": self.task_ir_hash,
            "schedule_hash": self.schedule_hash,
            "queue_trace_hash": self.queue_trace_hash,
            "sibling_group_hashes": list(self.sibling_group_hashes),
            "node_refinement_semantics": list(self.node_refinement_semantics),
            "objective_projection": self.objective_projection,
            "fallback_reason": self.fallback_reason,
            "discarded_attempt_stage": self.discarded_attempt_stage,
            "deadline_ns": self.deadline_ns,
        }

    def validate_against(
        self,
        plan: NativeObjectiveAncestralSiblingPackPlanIR,
        task_ir: NativeObjectiveAncestralSiblingPackTaskIRModule,
        schedule: NativeObjectiveAncestralSiblingPackScheduleIR,
        queue: NativeOptimizedReluSplitBabExecution,
        groups: Tuple[NativeObjectiveAncestralSiblingGroupExecutionIR, ...],
    ) -> None:
        plan.validate()
        schedule.validate_against(task_ir)
        if (
            self.schema_version
            != NATIVE_OBJECTIVE_ANCESTRAL_SIBLING_PACK_TRACE_SCHEMA_VERSION
            or not self.query_id
            or self.plan_hash != plan.stable_hash()
            or self.task_ir_hash != task_ir.stable_hash()
            or self.schedule_hash != schedule.stable_hash(task_ir)
            or self.queue_trace_hash != queue.trace.stable_hash()
            or self.sibling_group_hashes
            != tuple(group.atomic_commit_hash for group in groups)
            or not self.node_refinement_semantics
            or self.objective_projection != plan.objective_projection
            or not self.fallback_reason
            or min(
                self.source_elapsed_ns,
                self.queue_elapsed_ns,
                self.whole_elapsed_ns,
            )
            < 0
            or self.deadline_ns != plan.whole_query_timeout_ns
            or self.performance_claimed is not False
            or self.semantic_signature_hash != _canonical_hash(self.semantic_dict())
        ):
            raise ValueError("objective ancestral sibling-pack trace differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "query_id": self.query_id,
            **self.semantic_dict(),
            "source_elapsed_ns": self.source_elapsed_ns,
            "queue_elapsed_ns": self.queue_elapsed_ns,
            "whole_elapsed_ns": self.whole_elapsed_ns,
            "semantic_signature_hash": self.semantic_signature_hash,
            "performance_claimed": self.performance_claimed,
        }


@dataclass(frozen=True)
class NativeObjectiveAncestralSiblingPackExecution:
    """Plan, dynamic compiler IR, native queue, and sibling proof records."""

    plan: NativeObjectiveAncestralSiblingPackPlanIR
    task_ir: NativeObjectiveAncestralSiblingPackTaskIRModule
    schedule: NativeObjectiveAncestralSiblingPackScheduleIR
    queue: NativeOptimizedReluSplitBabExecution
    sibling_groups: Tuple[NativeObjectiveAncestralSiblingGroupExecutionIR, ...]
    node_refinements: Tuple[NativeObjectiveAncestralNodeRefinement, ...]
    trace: NativeObjectiveAncestralSiblingPackTrace

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
        validate_native_objective_ancestral_sibling_pack_plan(
            self.plan,
            module,
            input_spec,
            linear_spec_C=linear_spec_C,
            threshold=threshold,
            root_refinement=root_refinement,
            optimizer_policy=optimizer_policy,
        )
        self.queue.validate()
        self.schedule.validate_against(self.task_ir)
        evaluations = {item.node.node_id: item for item in self.queue.trace.evaluations}
        refinements = {item.node_id: item for item in self.node_refinements}
        if (
            len(evaluations) != len(self.queue.trace.evaluations)
            or len(refinements) != len(self.node_refinements)
            or tuple(refinements) != tuple(evaluations)
            or len(self.sibling_groups) * 2 + 1 != len(evaluations)
        ):
            raise ValueError("sibling-pack node/group coverage differs")
        source = _normalize_objective(linear_spec_C)
        evaluator = _project_objective(source)
        root_id = self.queue.trace.evaluations[0].node.node_id
        child_ids: list[str] = []
        for index, group in enumerate(self.sibling_groups):
            group.validate()
            if (
                group.group_index != index
                or group.parent_node_id not in evaluations
                or group.parent_evaluation_hash
                != _canonical_hash(evaluations[group.parent_node_id].to_dict())
            ):
                raise ValueError("sibling-pack group parent binding differs")
            child_ids.extend(group.child_node_ids)
        if tuple(child_ids) != tuple(evaluations)[1:]:
            raise ValueError("sibling-pack group child order differs")
        for node_id, refinement in refinements.items():
            evaluation = evaluations[node_id]
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
                raise ValueError("sibling-pack node refinement binding differs")
            if node_id == root_id:
                if refinement.execution != root_refinement:
                    raise ValueError("sibling-pack root refinement differs")
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
                    raise ValueError("sibling-pack parent refinement lineage differs")
        expected_semantics = tuple(
            item.semantic_dict() for item in self.node_refinements
        )
        if self.trace.node_refinement_semantics != expected_semantics:
            raise ValueError("sibling-pack refinement semantics differ")
        self.trace.validate_against(
            self.plan,
            self.task_ir,
            self.schedule,
            self.queue,
            self.sibling_groups,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "plan": self.plan.to_dict(),
            "plan_hash": self.plan.stable_hash(),
            "task_ir": self.task_ir.to_dict(),
            "task_ir_hash": self.task_ir.stable_hash(),
            "schedule": self.schedule.to_dict(),
            "schedule_hash": self.schedule.stable_hash(self.task_ir),
            "queue_trace": self.queue.trace.to_dict(),
            "sibling_groups": [group.to_dict() for group in self.sibling_groups],
            "node_refinements": [
                item.semantic_dict() for item in self.node_refinements
            ],
            "trace": self.trace.to_dict(),
        }


def compile_native_objective_ancestral_sibling_pack_plan(
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    linear_spec_C: torch.Tensor,
    threshold: torch.Tensor,
    root_refinement: NativeIntermediateRefinementExecution,
    optimizer_policy: NativeAlphaBetaOptimizerPolicy,
    plan_id: str,
) -> NativeObjectiveAncestralSiblingPackPlanIR:
    source = _normalize_objective(linear_spec_C)
    evaluator = _project_objective(source)
    base = compile_native_objective_ancestral_queue_plan(
        module,
        input_spec,
        linear_spec_C=source,
        threshold=threshold,
        root_refinement=root_refinement,
        optimizer_policy=optimizer_policy,
        plan_id=plan_id,
    )
    plan = NativeObjectiveAncestralSiblingPackPlanIR(
        plan_id=base.plan_id,
        primal_graph_hash=base.primal_graph_hash,
        input_bounds_hash=base.input_bounds_hash,
        source_objective_hash=tensor_content_hash(source),
        evaluator_objective_hash=tensor_content_hash(evaluator),
        threshold_hash=base.threshold_hash,
        root_refinement_plan_hash=base.root_refinement_plan_hash,
        root_refinement_semantic_trace_hash=(base.root_refinement_semantic_trace_hash),
        root_intermediate_bounds_hash=base.root_intermediate_bounds_hash,
        optimizer_policy_hash=base.optimizer_policy_hash,
        search_budget=base.search_budget,
        child_refinement_policy=base.child_refinement_policy,
    )
    validate_native_objective_ancestral_sibling_pack_plan(
        plan,
        module,
        input_spec,
        linear_spec_C=source,
        threshold=threshold,
        root_refinement=root_refinement,
        optimizer_policy=optimizer_policy,
    )
    return plan


def validate_native_objective_ancestral_sibling_pack_plan(
    plan: NativeObjectiveAncestralSiblingPackPlanIR,
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    linear_spec_C: torch.Tensor,
    threshold: torch.Tensor,
    root_refinement: NativeIntermediateRefinementExecution,
    optimizer_policy: NativeAlphaBetaOptimizerPolicy,
) -> None:
    plan.validate()
    source = _normalize_objective(linear_spec_C)
    evaluator = _project_objective(source)
    validate_native_objective_ancestral_queue_plan(
        _as_queue_plan(plan),
        module,
        input_spec,
        linear_spec_C=source,
        threshold=threshold,
        root_refinement=root_refinement,
        optimizer_policy=optimizer_policy,
    )
    if (
        plan.primal_graph_hash != plain_crown_primal_graph_hash(module)
        or plan.input_bounds_hash != _input_bounds_hash(input_spec)
        or plan.source_objective_hash != tensor_content_hash(source)
        or plan.evaluator_objective_hash != tensor_content_hash(evaluator)
        or plan.threshold_hash != _threshold_hash(threshold)
    ):
        raise ValueError("sibling-pack plan/query binding differs")


def _evaluate_root(
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    objective: torch.Tensor,
    root: _RuntimeNode,
    root_refinement: NativeIntermediateRefinementExecution,
    config: NativeReluSplitBabConfig,
    optimizer_policy: NativeAlphaBetaOptimizerPolicy,
    batch_id: str,
) -> tuple[_OptimizedEvaluatedNode, object]:
    evaluated, stack, branches, refinements, records = _evaluate_optimized_node_batch(
        module,
        input_spec,
        objective=objective,
        nodes=(root,),
        batch_id=batch_id,
        config=config,
        policy=optimizer_policy,
        parent_by_id={},
        relu_pre_override=root_refinement.relu_pre,
        intermediate_bound_source=IntermediateBoundSource.NATIVE_REFINED,
        objective_branch_policy=None,
        refine_external_constraints=False,
        per_child_refinement_policy=None,
        per_child_refinement_budget_policy=None,
        per_child_refinement_multi_pass_policy=None,
        per_child_refinement_strategy="independent_exact_split_v1",
        external_constraint_seed=None,
    )
    if len(evaluated) != 1 or branches or refinements or records:
        raise ValueError("sibling-pack root evaluator coverage differs")
    return replace(evaluated[0], refinement_execution=root_refinement), stack


def execute_native_objective_ancestral_sibling_pack_queue(
    plan: NativeObjectiveAncestralSiblingPackPlanIR,
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    linear_spec_C: torch.Tensor,
    threshold: torch.Tensor,
    root_refinement: NativeIntermediateRefinementExecution,
    optimizer_policy: NativeAlphaBetaOptimizerPolicy,
    query_id: str,
    whole_query_started_ns: Optional[int] = None,
    clock_ns: ClockNs = time.monotonic_ns,
) -> NativeObjectiveAncestralSiblingPackExecution:
    if not query_id:
        raise ValueError("sibling-pack query ID must be non-empty")
    validate_native_objective_ancestral_sibling_pack_plan(
        plan,
        module,
        input_spec,
        linear_spec_C=linear_spec_C,
        threshold=threshold,
        root_refinement=root_refinement,
        optimizer_policy=optimizer_policy,
    )
    source_objective = _normalize_objective(linear_spec_C)
    objective = _project_objective(source_objective)
    threshold_value = float(threshold.reshape(-1)[0].item())
    started_ns = (
        clock_ns() if whole_query_started_ns is None else whole_query_started_ns
    )
    queue_started_ns = clock_ns()
    deadline_at_ns = started_ns + plan.whole_query_timeout_ns
    config = NativeReluSplitBabConfig(
        max_nodes=plan.search_budget.max_nodes,
        max_depth=plan.search_budget.max_depth,
        expansion_batch_size=plan.expansion_batch_size,
        max_eval_batch_size=plan.max_eval_batch_size,
        threshold=threshold_value,
    )
    lower, upper = _root_box_bounds(input_spec)
    _root_interval, root_pre = _forward_ibp_trace_mlp(module, input_spec)
    root_splits = tuple(
        (
            name,
            torch.zeros(
                tuple(int(dim) for dim in pre.lower.shape[1:]),
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
    tasks: list[NativeObjectiveAncestralSiblingPackTaskIRUnit] = []
    root_source_task = _task(
        tasks,
        plan,
        suffix="root:admit-source",
        kind=NativeObjectiveAncestralSiblingPackTaskKind.ADMIT_ROOT_SOURCE,
        dependencies=(),
        group_id=None,
        node_ids=(root.node.node_id,),
        inputs={
            "root_refinement_plan": plan.root_refinement_plan_hash,
            "root_refinement_trace": plan.root_refinement_semantic_trace_hash,
            "root_intermediate_bounds": plan.root_intermediate_bounds_hash,
        },
        output=root_refinement.program.hashes(),
    )
    projection_task = _task(
        tasks,
        plan,
        suffix="root:project-objective",
        kind=NativeObjectiveAncestralSiblingPackTaskKind.PROJECT_OBJECTIVE,
        dependencies=(root_source_task,),
        group_id=None,
        node_ids=(root.node.node_id,),
        inputs={"source_objective": plan.source_objective_hash},
        output={
            "projection": plan.objective_projection,
            "evaluator_objective_hash": plan.evaluator_objective_hash,
        },
    )
    root_evaluated, root_stack = _evaluate_root(
        module,
        input_spec,
        objective=objective,
        root=root,
        root_refinement=root_refinement,
        config=config,
        optimizer_policy=optimizer_policy,
        batch_id=f"{query_id}:eval:0000",
    )
    if clock_ns() > deadline_at_ns:
        raise RuntimeError("sibling-pack deadline expired at root")
    root_eval_task = _task(
        tasks,
        plan,
        suffix="root:evaluate",
        kind=NativeObjectiveAncestralSiblingPackTaskKind.EVALUATE_ROOT,
        dependencies=(projection_task,),
        group_id=None,
        node_ids=(root.node.node_id,),
        inputs={
            "root_intermediate_bounds": plan.root_intermediate_bounds_hash,
            "optimizer_policy": plan.optimizer_policy_hash,
            "evaluator_objective": plan.evaluator_objective_hash,
        },
        output=root_evaluated.evaluation.to_dict(),
    )
    evaluations = [root_evaluated.evaluation]
    decisions: list[NativeReluSplitBabDecision] = []
    stacks = [root_stack]
    runtime_by_id = {root.node.node_id: root_evaluated}
    refinement_by_id = {root.node.node_id: root_refinement}
    node_refinements = [
        NativeObjectiveAncestralNodeRefinement(
            node_id=root.node.node_id,
            parent_node_id=None,
            node_split_state_hash=root.node.split_state_hash,
            program=root_refinement.program,
            execution=root_refinement,
        )
    ]
    evaluation_task_by_id = {root.node.node_id: root_eval_task}
    sibling_groups: list[NativeObjectiveAncestralSiblingGroupExecutionIR] = []
    heap = [_QueueEntry(root_evaluated.evaluation.priority, 0, root.node.node_id)]
    next_node_serial = 1
    heap_serial = 1
    batch_serial = 1
    max_queue_size = 1
    budget_exhausted = False
    deadline_exhausted = False
    discarded_stage: Optional[str] = None

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
                kind=NativeObjectiveAncestralSiblingPackTaskKind.TRANSITION_QUEUE,
                dependencies=(evaluation_task_by_id[node.node_id],),
                group_id=None,
                node_ids=(node.node_id,),
                inputs={"evaluation": _evaluation_hash(parent)},
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
                kind=NativeObjectiveAncestralSiblingPackTaskKind.TRANSITION_QUEUE,
                dependencies=(evaluation_task_by_id[node.node_id],),
                group_id=None,
                node_ids=(node.node_id,),
                inputs={"evaluation": _evaluation_hash(parent)},
                output=decision.to_dict(),
            )
            continue
        if len(evaluations) + plan.sibling_group_size > config.max_nodes:
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
        group_index = len(sibling_groups)
        group_id = f"{query_id}:sibling-group:{group_index:04d}"
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
            discarded_stage = "before_sibling_group"
            break
        packed, stack, branches, refinement_executions, refinement_records = (
            _evaluate_optimized_node_batch(
                module,
                input_spec,
                objective=objective,
                nodes=children,
                batch_id=f"{query_id}:eval:{batch_serial:04d}",
                config=config,
                policy=optimizer_policy,
                parent_by_id=runtime_by_id,
                relu_pre_override=None,
                intermediate_bound_source=IntermediateBoundSource.NATIVE_REFINED,
                objective_branch_policy=None,
                refine_external_constraints=False,
                per_child_refinement_policy=plan.child_refinement_policy,
                per_child_refinement_budget_policy=None,
                per_child_refinement_multi_pass_policy=None,
                per_child_refinement_strategy="ancestral_constraint_carry_v1",
                external_constraint_seed=None,
            )
        )
        batch_serial += 1
        if (
            len(packed) != 2
            or branches
            or len(refinement_executions) != 2
            or len(refinement_records) != 2
        ):
            raise ValueError("sibling-pack evaluator group coverage differs")
        if clock_ns() > deadline_at_ns:
            heapq.heappush(heap, entry)
            deadline_exhausted = True
            discarded_stage = "after_complete_sibling_group"
            break
        normalized = tuple(_normalized_evaluated(item) for item in packed)
        child_execution_by_id = dict(refinement_executions)
        child_refinements = tuple(
            NativeObjectiveAncestralNodeRefinement(
                node_id=child.node.node_id,
                parent_node_id=node.node_id,
                node_split_state_hash=child.node.split_state_hash,
                program=child_execution_by_id[child.node.node_id].program,
                execution=child_execution_by_id[child.node.node_id],
            )
            for child in children
        )
        group = NativeObjectiveAncestralSiblingGroupExecutionIR.committed(
            group_id=group_id,
            group_index=group_index,
            parent_node_id=node.node_id,
            parent_evaluation_hash=_evaluation_hash(parent),
            child_node_ids=tuple(child.node.node_id for child in children),
            child_branch_values=tuple(child.node.branch_value for child in children),
            child_split_state_hashes=tuple(
                child.node.split_state_hash for child in children
            ),
            parent_refinement_plan_hash=refinement_by_id[
                node.node_id
            ].program.plan.stable_hash(),
            parent_refinement_semantic_trace_hash=(
                intermediate_refinement_semantic_trace_hash(
                    refinement_by_id[node.node_id]
                )
            ),
            parent_final_intermediate_bounds_hash=intermediate_bounds_hash(
                refinement_by_id[node.node_id].relu_pre
            ),
            child_refinement_plan_hashes=tuple(
                item.program.plan.stable_hash() for item in child_refinements
            ),
            child_refinement_semantic_trace_hashes=tuple(
                intermediate_refinement_semantic_trace_hash(item.execution)
                for item in child_refinements
            ),
            child_final_intermediate_bounds_hashes=tuple(
                intermediate_bounds_hash(item.execution.relu_pre)
                for item in child_refinements
            ),
            optimizer_ir_hash=_canonical_hash(dict(stack.optimizer_ir_hashes)),
            optimizer_execution_trace_hash=stack.optimizer_execution_trace_hash,
            native_ir_hash=_canonical_hash(dict(stack.native_ir_hashes)),
            child_evaluation_hashes=tuple(
                _evaluation_hash(item) for item in normalized
            ),
        )
        transition_task = _task(
            tasks,
            plan,
            suffix=f"{node.node_id}:transition",
            kind=NativeObjectiveAncestralSiblingPackTaskKind.TRANSITION_QUEUE,
            dependencies=(evaluation_task_by_id[node.node_id],),
            group_id=None,
            node_ids=(node.node_id,),
            inputs={"evaluation": _evaluation_hash(parent)},
            output=decision.to_dict(),
        )
        refinement_execute_tasks: list[str] = []
        for child, refinement in zip(children, child_refinements):
            compile_task = _task(
                tasks,
                plan,
                suffix=f"{child.node.node_id}:compile-refinement",
                kind=(
                    NativeObjectiveAncestralSiblingPackTaskKind.COMPILE_CHILD_REFINEMENT
                ),
                dependencies=(transition_task,),
                group_id=group_id,
                node_ids=(child.node.node_id,),
                inputs={
                    "evaluator_objective": plan.evaluator_objective_hash,
                    "parent_refinement_plan": group.parent_refinement_plan_hash,
                    "parent_refinement_trace": (
                        group.parent_refinement_semantic_trace_hash
                    ),
                    "split_state": child.node.split_state_hash,
                },
                output=refinement.program.hashes(),
            )
            refinement_execute_tasks.append(
                _task(
                    tasks,
                    plan,
                    suffix=f"{child.node.node_id}:execute-refinement",
                    kind=(
                        NativeObjectiveAncestralSiblingPackTaskKind.EXECUTE_CHILD_REFINEMENT
                    ),
                    dependencies=(compile_task,),
                    group_id=group_id,
                    node_ids=(child.node.node_id,),
                    inputs={"refinement_plan": refinement.program.plan.stable_hash()},
                    output=intermediate_refinement_semantic_trace_hash(
                        refinement.execution
                    ),
                )
            )
        packed_compile_task = _task(
            tasks,
            plan,
            suffix=f"group:{group_index:04d}:compile-packed-evaluator",
            kind=NativeObjectiveAncestralSiblingPackTaskKind.COMPILE_PACKED_EVALUATOR,
            dependencies=tuple(refinement_execute_tasks),
            group_id=group_id,
            node_ids=group.child_node_ids,
            inputs={
                "optimizer_policy": plan.optimizer_policy_hash,
                "evaluator_objective": plan.evaluator_objective_hash,
                "child_refinement_pair": _canonical_hash(
                    group.child_refinement_semantic_trace_hashes
                ),
            },
            output={
                "optimizer_ir_hash": group.optimizer_ir_hash,
                "native_ir_hash": group.native_ir_hash,
            },
        )
        packed_execute_task = _task(
            tasks,
            plan,
            suffix=f"group:{group_index:04d}:execute-packed-evaluator",
            kind=NativeObjectiveAncestralSiblingPackTaskKind.EXECUTE_PACKED_EVALUATOR,
            dependencies=(packed_compile_task,),
            group_id=group_id,
            node_ids=group.child_node_ids,
            inputs={
                "optimizer_ir": group.optimizer_ir_hash,
                "native_ir": group.native_ir_hash,
                "optimizer_trace": group.optimizer_execution_trace_hash,
            },
            output=group.to_dict(),
        )
        decisions.append(decision)
        sibling_groups.append(group)
        stacks.append(stack)
        for child, evaluated, refinement in zip(
            children, normalized, child_refinements
        ):
            evaluations.append(evaluated.evaluation)
            runtime_by_id[child.node.node_id] = replace(
                evaluated, refinement_execution=refinement.execution
            )
            refinement_by_id[child.node.node_id] = refinement.execution
            node_refinements.append(refinement)
            evaluation_task_by_id[child.node.node_id] = packed_execute_task
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
        for entry in sorted(heap, key=lambda item: (item.priority, item.serial))
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
    queue_trace = NativeOptimizedReluSplitBabTrace(
        run_id=query_id,
        status=status,
        termination_reason=termination_reason,
        config=config,
        optimizer_policy=optimizer_policy,
        root_input_lower_hash=tensor_content_hash(lower),
        root_input_upper_hash=tensor_content_hash(upper),
        objective_hash=tensor_content_hash(objective),
        evaluations=tuple(evaluations),
        decisions=tuple(decisions),
        final_frontier_node_ids=frontier,
        native_stacks=tuple(stacks),  # type: ignore[arg-type]
        native_stack_count=len(stacks),
        max_queue_size=max_queue_size,
    )
    queue = NativeOptimizedReluSplitBabExecution(
        trace=queue_trace,
        selected_states=tuple(
            (
                evaluation.node.node_id,
                runtime_by_id[evaluation.node.node_id].selected_state,
            )
            for evaluation in queue_trace.evaluations
        ),
    )
    queue.validate()
    emit_dependencies = tuple(
        task.task_id
        for task in tasks
        if task.kind
        in {
            NativeObjectiveAncestralSiblingPackTaskKind.EVALUATE_ROOT,
            NativeObjectiveAncestralSiblingPackTaskKind.TRANSITION_QUEUE,
            NativeObjectiveAncestralSiblingPackTaskKind.EXECUTE_PACKED_EVALUATOR,
        }
    )
    _task(
        tasks,
        plan,
        suffix="emit",
        kind=NativeObjectiveAncestralSiblingPackTaskKind.EMIT_RESULT,
        dependencies=emit_dependencies,
        group_id=None,
        node_ids=tuple(item.node.node_id for item in evaluations),
        inputs={"queue_trace": queue.trace.stable_hash()},
        output=queue.trace.to_dict(),
    )
    task_ir, schedule = lower_native_objective_ancestral_sibling_pack_schedule(
        plan, tuple(tasks)
    )
    finished_ns = clock_ns()
    source_elapsed_ns = max(0, queue_started_ns - started_ns)
    queue_elapsed_ns = max(0, finished_ns - queue_started_ns)
    whole_elapsed_ns = max(0, finished_ns - started_ns)
    trace = NativeObjectiveAncestralSiblingPackTrace(
        query_id=query_id,
        plan_hash=plan.stable_hash(),
        task_ir_hash=task_ir.stable_hash(),
        schedule_hash=schedule.stable_hash(task_ir),
        queue_trace_hash=queue.trace.stable_hash(),
        sibling_group_hashes=tuple(
            group.atomic_commit_hash for group in sibling_groups
        ),
        node_refinement_semantics=tuple(
            item.semantic_dict() for item in node_refinements
        ),
        objective_projection=plan.objective_projection,
        fallback_reason=fallback_reason,
        discarded_attempt_stage=discarded_stage,
        source_elapsed_ns=source_elapsed_ns,
        queue_elapsed_ns=queue_elapsed_ns,
        whole_elapsed_ns=whole_elapsed_ns,
        deadline_ns=plan.whole_query_timeout_ns,
        semantic_signature_hash="",
    )
    trace = replace(
        trace, semantic_signature_hash=_canonical_hash(trace.semantic_dict())
    )
    execution = NativeObjectiveAncestralSiblingPackExecution(
        plan=plan,
        task_ir=task_ir,
        schedule=schedule,
        queue=queue,
        sibling_groups=tuple(sibling_groups),
        node_refinements=tuple(node_refinements),
        trace=trace,
    )
    execution.validate_against(
        module,
        input_spec,
        linear_spec_C=source_objective,
        threshold=threshold,
        root_refinement=root_refinement,
        optimizer_policy=optimizer_policy,
    )
    return execution


__all__ = [
    "NativeObjectiveAncestralSiblingPackExecution",
    "NativeObjectiveAncestralSiblingPackTrace",
    "compile_native_objective_ancestral_sibling_pack_plan",
    "execute_native_objective_ancestral_sibling_pack_queue",
    "validate_native_objective_ancestral_sibling_pack_plan",
]
