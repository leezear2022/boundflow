"""Whole-deadline objective-root ancestral BaB queue runtime."""

# pylint: disable=too-many-arguments,too-many-locals,too-many-statements
# pylint: disable=too-many-branches,too-many-instance-attributes
# pylint: disable=too-many-boolean-expressions,missing-function-docstring
# pylint: disable=protected-access,duplicate-code,too-many-lines

from __future__ import annotations

from dataclasses import dataclass
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
from ..ir.objective_ancestral_queue import (
    NativeObjectiveAncestralQueuePlanIR,
    NativeObjectiveAncestralQueueScheduleIR,
    NativeObjectiveAncestralQueueTaskIRModule,
    NativeObjectiveAncestralQueueTaskIRUnit,
    ObjectiveAncestralQueueTaskKind,
    lower_native_objective_ancestral_queue_schedule,
)
from ..ir.refinement import NativeIntermediateRefinementPolicyIR
from ..ir.search_scaling import NativeBabSearchBudgetIR
from ..ir.task import BFTaskModule
from .crown_ibp import _forward_ibp_trace_mlp
from .native_alpha_beta_optimization_state import NativeAlphaBetaOptimizerPolicy
from .native_intermediate_refinement import (
    NativeIntermediateRefinementExecution,
    NativeIntermediateRefinementProgram,
    _input_bounds_hash,
    compile_native_intermediate_refinement_program,
    execute_native_intermediate_refinement_program,
    intermediate_bounds_hash,
    intermediate_refinement_semantic_trace_hash,
)
from .native_optimized_relu_split_bab_runtime import (
    NativeOptimizedReluSplitBabExecution,
    NativeOptimizedReluSplitBabTrace,
    _OptimizedEvaluatedNode,
    _QueueEntry,
    _RuntimeNode,
    _evaluate_optimized_node_batch,
    _make_child_runtime_node,
    _node_split_mapping,
    _root_box_bounds,
)
from .native_relu_split_bab_runtime import (
    BabQueueStatus,
    NativeReluSplitBabConfig,
    NativeReluSplitBabDecision,
    NativeReluSplitBabNode,
)
from .task_executor import InputSpec

NATIVE_OBJECTIVE_ANCESTRAL_QUEUE_TRACE_SCHEMA_VERSION = (
    "boundflow.native-objective-ancestral-queue-trace/v1"
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


def _threshold_hash(value: torch.Tensor) -> str:
    if (
        not torch.is_tensor(value)
        or not torch.is_floating_point(value)
        or value.numel() != 1
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError("objective ancestral queue threshold is invalid")
    return tensor_content_hash(value.reshape(1).contiguous())


def _normalize_objective(value: torch.Tensor) -> torch.Tensor:
    if (
        not torch.is_tensor(value)
        or not torch.is_floating_point(value)
        or value.dim() not in {2, 3}
        or int(value.shape[-2]) != 1
        or (value.dim() == 3 and int(value.shape[0]) != 1)
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError("objective ancestral queue requires one scalar objective")
    return value.detach().contiguous().clone()


def _evaluation_hash(value: _OptimizedEvaluatedNode) -> str:
    return _canonical_hash(value.evaluation.to_dict())


@dataclass(frozen=True)
class NativeObjectiveAncestralNodeRefinement:
    """Exact refinement execution owned by one accepted queue node."""

    node_id: str
    parent_node_id: Optional[str]
    node_split_state_hash: str
    program: NativeIntermediateRefinementProgram
    execution: NativeIntermediateRefinementExecution

    def semantic_dict(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "parent_node_id": self.parent_node_id,
            "node_split_state_hash": self.node_split_state_hash,
            "program_hashes": self.program.hashes(),
            "semantic_trace_hash": intermediate_refinement_semantic_trace_hash(
                self.execution
            ),
            "initial_intermediate_bounds_hash": (
                self.program.plan.initial_intermediate_bounds_hash
            ),
            "final_intermediate_bounds_hash": intermediate_bounds_hash(
                self.execution.relu_pre
            ),
            "source_intermediate_constraints_hash": (
                self.program.plan.source_intermediate_constraints_hash
            ),
            "source_refinement_plan_hash": (
                self.program.plan.source_refinement_plan_hash
            ),
            "source_refinement_semantic_trace_hash": (
                self.program.plan.source_refinement_semantic_trace_hash
            ),
            "source_consumption": (
                "admitted_typed_root_execution"
                if self.parent_node_id is None
                else "sound_constraint_only"
            ),
        }


@dataclass(frozen=True)
class NativeObjectiveAncestralQueueTrace:
    """Typed queue/Task/Schedule identity plus deadline fallback."""

    query_id: str
    plan_hash: str
    task_ir_hash: str
    schedule_hash: str
    queue_trace_hash: str
    node_refinement_semantics: Tuple[dict[str, object], ...]
    fallback_reason: str
    discarded_attempt_stage: Optional[str]
    source_elapsed_ns: int
    queue_elapsed_ns: int
    whole_elapsed_ns: int
    deadline_ns: int
    semantic_signature_hash: str
    performance_claimed: bool = False
    schema_version: str = NATIVE_OBJECTIVE_ANCESTRAL_QUEUE_TRACE_SCHEMA_VERSION

    def semantic_dict(self) -> dict[str, object]:
        return {
            "plan_hash": self.plan_hash,
            "task_ir_hash": self.task_ir_hash,
            "schedule_hash": self.schedule_hash,
            "queue_trace_hash": self.queue_trace_hash,
            "node_refinement_semantics": list(self.node_refinement_semantics),
            "fallback_reason": self.fallback_reason,
            "discarded_attempt_stage": self.discarded_attempt_stage,
            "deadline_ns": self.deadline_ns,
        }

    def validate_against(
        self,
        plan: NativeObjectiveAncestralQueuePlanIR,
        task_ir: NativeObjectiveAncestralQueueTaskIRModule,
        schedule: NativeObjectiveAncestralQueueScheduleIR,
        queue: NativeOptimizedReluSplitBabExecution,
    ) -> None:
        plan.validate()
        schedule.validate_against(task_ir)
        if (
            self.schema_version != NATIVE_OBJECTIVE_ANCESTRAL_QUEUE_TRACE_SCHEMA_VERSION
            or not self.query_id
            or self.plan_hash != plan.stable_hash()
            or self.task_ir_hash != task_ir.stable_hash()
            or self.schedule_hash != schedule.stable_hash(task_ir)
            or self.queue_trace_hash != queue.trace.stable_hash()
            or not self.node_refinement_semantics
            or not self.fallback_reason
            or self.source_elapsed_ns < 0
            or self.queue_elapsed_ns < 0
            or self.whole_elapsed_ns < 0
            or self.deadline_ns != plan.whole_query_timeout_ns
            or self.performance_claimed is not False
            or self.semantic_signature_hash != _canonical_hash(self.semantic_dict())
        ):
            raise ValueError("objective ancestral queue aggregate trace differs")

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
class NativeObjectiveAncestralQueueExecution:
    """Plan plus committed dynamic IR, native queue, and refinement lineage."""

    plan: NativeObjectiveAncestralQueuePlanIR
    task_ir: NativeObjectiveAncestralQueueTaskIRModule
    schedule: NativeObjectiveAncestralQueueScheduleIR
    queue: NativeOptimizedReluSplitBabExecution
    node_refinements: Tuple[NativeObjectiveAncestralNodeRefinement, ...]
    trace: NativeObjectiveAncestralQueueTrace

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
        validate_native_objective_ancestral_queue_plan(
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
        ):
            raise ValueError("objective ancestral queue node coverage differs")
        objective = _normalize_objective(linear_spec_C)
        root_id = self.queue.trace.evaluations[0].node.node_id
        for node_id, refinement in refinements.items():
            evaluation = evaluations[node_id]
            refinement.program.validate(module, input_spec)
            refinement.execution.validate(module, input_spec)
            if (
                refinement.execution.program != refinement.program
                or refinement.node_split_state_hash != evaluation.node.split_state_hash
                or refinement.parent_node_id != evaluation.node.parent_node_id
                or refinement.program.plan.objective_hash
                != tensor_content_hash(objective)
            ):
                raise ValueError("objective ancestral node refinement binding differs")
            if node_id == root_id:
                if refinement.execution != root_refinement:
                    raise ValueError("objective ancestral root execution differs")
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
                    raise ValueError("objective ancestral parent lineage differs")
        expected_semantics = tuple(
            refinement.semantic_dict() for refinement in self.node_refinements
        )
        if self.trace.node_refinement_semantics != expected_semantics:
            raise ValueError("objective ancestral refinement semantics differ")
        self.trace.validate_against(self.plan, self.task_ir, self.schedule, self.queue)

    def to_dict(self) -> dict[str, object]:
        return {
            "plan": self.plan.to_dict(),
            "plan_hash": self.plan.stable_hash(),
            "task_ir": self.task_ir.to_dict(),
            "task_ir_hash": self.task_ir.stable_hash(),
            "schedule": self.schedule.to_dict(),
            "schedule_hash": self.schedule.stable_hash(self.task_ir),
            "queue_trace": self.queue.trace.to_dict(),
            "node_refinements": [
                item.semantic_dict() for item in self.node_refinements
            ],
            "trace": self.trace.to_dict(),
        }


def compile_native_objective_ancestral_queue_plan(
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    linear_spec_C: torch.Tensor,
    threshold: torch.Tensor,
    root_refinement: NativeIntermediateRefinementExecution,
    optimizer_policy: NativeAlphaBetaOptimizerPolicy,
    plan_id: str,
) -> NativeObjectiveAncestralQueuePlanIR:
    module.validate()
    optimizer_policy.validate()
    objective = _normalize_objective(linear_spec_C)
    root_refinement.validate(module, input_spec)
    if (
        root_refinement.program.plan.objective_hash != tensor_content_hash(objective)
        or root_refinement.program.plan.policy.candidate_policy_id
        != "objective_influence_width_per_relu_v1"
        or root_refinement.program.plan.source_refinement_plan_hash is None
        or root_refinement.program.plan.source_refinement_semantic_trace_hash is None
    ):
        raise ValueError("objective ancestral queue root refinement differs")
    plan = NativeObjectiveAncestralQueuePlanIR(
        plan_id=plan_id,
        primal_graph_hash=plain_crown_primal_graph_hash(module),
        input_bounds_hash=_input_bounds_hash(input_spec),
        objective_hash=tensor_content_hash(objective),
        threshold_hash=_threshold_hash(threshold),
        root_refinement_plan_hash=root_refinement.program.plan.stable_hash(),
        root_refinement_semantic_trace_hash=(
            intermediate_refinement_semantic_trace_hash(root_refinement)
        ),
        root_intermediate_bounds_hash=intermediate_bounds_hash(
            root_refinement.relu_pre
        ),
        optimizer_policy_hash=optimizer_policy.stable_hash(),
        search_budget=NativeBabSearchBudgetIR("objective-ancestral-n31d4", 31, 4),
        child_refinement_policy=NativeIntermediateRefinementPolicyIR(
            passes=1,
            max_neurons_per_relu=128,
            backward_chunk_size=32,
            candidate_policy_id="objective_influence_width_per_relu_v1",
        ),
    )
    validate_native_objective_ancestral_queue_plan(
        plan,
        module,
        input_spec,
        linear_spec_C=objective,
        threshold=threshold,
        root_refinement=root_refinement,
        optimizer_policy=optimizer_policy,
    )
    return plan


def validate_native_objective_ancestral_queue_plan(
    plan: NativeObjectiveAncestralQueuePlanIR,
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    linear_spec_C: torch.Tensor,
    threshold: torch.Tensor,
    root_refinement: NativeIntermediateRefinementExecution,
    optimizer_policy: NativeAlphaBetaOptimizerPolicy,
) -> None:
    plan.validate()
    objective = _normalize_objective(linear_spec_C)
    root_refinement.validate(module, input_spec)
    if (
        plan.primal_graph_hash != plain_crown_primal_graph_hash(module)
        or plan.input_bounds_hash != _input_bounds_hash(input_spec)
        or plan.objective_hash != tensor_content_hash(objective)
        or plan.threshold_hash != _threshold_hash(threshold)
        or plan.root_refinement_plan_hash != root_refinement.program.plan.stable_hash()
        or plan.root_refinement_semantic_trace_hash
        != intermediate_refinement_semantic_trace_hash(root_refinement)
        or plan.root_intermediate_bounds_hash
        != intermediate_bounds_hash(root_refinement.relu_pre)
        or plan.optimizer_policy_hash != optimizer_policy.stable_hash()
    ):
        raise ValueError("objective ancestral queue plan/query binding differs")


def _evaluate_serial(
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    objective: torch.Tensor,
    node: _RuntimeNode,
    batch_id: str,
    config: NativeReluSplitBabConfig,
    optimizer_policy: NativeAlphaBetaOptimizerPolicy,
    parent_by_id: Mapping[str, _OptimizedEvaluatedNode],
    relu_pre: Mapping[str, object],
) -> tuple[_OptimizedEvaluatedNode, object]:
    evaluated, stack, branches, refinements, records = _evaluate_optimized_node_batch(
        module,
        input_spec,
        objective=objective,
        nodes=(node,),
        batch_id=batch_id,
        config=config,
        policy=optimizer_policy,
        parent_by_id=parent_by_id,
        relu_pre_override=relu_pre,  # type: ignore[arg-type]
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
        raise ValueError("objective ancestral serial evaluator coverage differs")
    return evaluated[0], stack


def _task(
    tasks: list[NativeObjectiveAncestralQueueTaskIRUnit],
    plan: NativeObjectiveAncestralQueuePlanIR,
    *,
    suffix: str,
    kind: ObjectiveAncestralQueueTaskKind,
    dependencies: Sequence[str],
    node: Optional[NativeReluSplitBabNode],
    inputs: Mapping[str, str],
    output: object,
) -> str:
    task_id = f"{plan.plan_id}:{suffix}"
    task = NativeObjectiveAncestralQueueTaskIRUnit(
        sequence=len(tasks),
        task_id=task_id,
        kind=kind,
        dependency_task_ids=tuple(dependencies),
        node_id=None if node is None else node.node_id,
        parent_node_id=None if node is None else node.parent_node_id,
        node_split_state_hash=None if node is None else node.split_state_hash,
        input_hashes=tuple(sorted(inputs.items())),
        output_hash=_canonical_hash(output),
    )
    task.validate()
    tasks.append(task)
    return task_id


def execute_native_objective_ancestral_queue(
    plan: NativeObjectiveAncestralQueuePlanIR,
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
) -> NativeObjectiveAncestralQueueExecution:
    if not query_id:
        raise ValueError("objective ancestral queue query ID must be non-empty")
    validate_native_objective_ancestral_queue_plan(
        plan,
        module,
        input_spec,
        linear_spec_C=linear_spec_C,
        threshold=threshold,
        root_refinement=root_refinement,
        optimizer_policy=optimizer_policy,
    )
    objective = _normalize_objective(linear_spec_C)
    threshold_value = float(threshold.reshape(-1)[0].item())
    started_ns = (
        clock_ns() if whole_query_started_ns is None else whole_query_started_ns
    )
    queue_started_ns = clock_ns()
    deadline_at_ns = started_ns + plan.whole_query_timeout_ns
    config = NativeReluSplitBabConfig(
        max_nodes=plan.search_budget.max_nodes,
        max_depth=plan.search_budget.max_depth,
        expansion_batch_size=1,
        max_eval_batch_size=1,
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
    tasks: list[NativeObjectiveAncestralQueueTaskIRUnit] = []
    root_source_task = _task(
        tasks,
        plan,
        suffix="root:admit-source",
        kind=ObjectiveAncestralQueueTaskKind.ADMIT_ROOT_SOURCE,
        dependencies=(),
        node=root.node,
        inputs={
            "root_refinement_plan": plan.root_refinement_plan_hash,
            "root_refinement_trace": plan.root_refinement_semantic_trace_hash,
            "root_intermediate_bounds": plan.root_intermediate_bounds_hash,
        },
        output=root_refinement.program.hashes(),
    )
    root_evaluated, root_stack = _evaluate_serial(
        module,
        input_spec,
        objective=objective,
        node=root,
        batch_id=f"{query_id}:eval:0000",
        config=config,
        optimizer_policy=optimizer_policy,
        parent_by_id={},
        relu_pre=root_refinement.relu_pre,
    )
    if clock_ns() > deadline_at_ns:
        raise RuntimeError("objective ancestral queue deadline expired at root")
    root_eval_task = _task(
        tasks,
        plan,
        suffix="root:evaluate",
        kind=ObjectiveAncestralQueueTaskKind.EVALUATE_ROOT,
        dependencies=(root_source_task,),
        node=root.node,
        inputs={
            "root_intermediate_bounds": plan.root_intermediate_bounds_hash,
            "optimizer_policy": plan.optimizer_policy_hash,
            "objective": plan.objective_hash,
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
    source_task_by_id = {root.node.node_id: root_source_task}
    heap = [_QueueEntry(root_evaluated.evaluation.priority, 0, root.node.node_id)]
    next_node_serial = 1
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
                kind=ObjectiveAncestralQueueTaskKind.TRANSITION_QUEUE,
                dependencies=(evaluation_task_by_id[node.node_id],),
                node=node,
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
                kind=ObjectiveAncestralQueueTaskKind.TRANSITION_QUEUE,
                dependencies=(evaluation_task_by_id[node.node_id],),
                node=node,
                inputs={"evaluation": _evaluation_hash(parent)},
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
        transition_id = f"{plan.plan_id}:{node.node_id}:transition"
        temporary_tasks: list[NativeObjectiveAncestralQueueTaskIRUnit] = []
        temporary_evaluated: list[_OptimizedEvaluatedNode] = []
        temporary_stacks: list[object] = []
        temporary_refinements: list[NativeObjectiveAncestralNodeRefinement] = []
        failed = False
        for child in children:
            if clock_ns() >= deadline_at_ns:
                discarded_stage = "before_child_compile"
                failed = True
                break
            child_program = compile_native_intermediate_refinement_program(
                module,
                input_spec,
                policy=plan.child_refinement_policy,
                plan_id=f"{query_id}:refinement:{child.node.node_id}",
                relu_split_state=_node_split_mapping(child),
                linear_spec_C=objective,
                source_refinement_execution=refinement_by_id[node.node_id],
            )
            if clock_ns() >= deadline_at_ns:
                discarded_stage = "after_child_compile"
                failed = True
                break
            compile_task_id = f"{plan.plan_id}:{child.node.node_id}:compile"
            temporary_tasks.append(
                NativeObjectiveAncestralQueueTaskIRUnit(
                    sequence=-1,
                    task_id=compile_task_id,
                    kind=ObjectiveAncestralQueueTaskKind.COMPILE_CHILD_REFINEMENT,
                    dependency_task_ids=(
                        transition_id,
                        source_task_by_id[node.node_id],
                    ),
                    node_id=child.node.node_id,
                    parent_node_id=node.node_id,
                    node_split_state_hash=child.node.split_state_hash,
                    input_hashes=tuple(
                        sorted(
                            {
                                "objective": plan.objective_hash,
                                "parent_refinement_plan": refinement_by_id[
                                    node.node_id
                                ].program.plan.stable_hash(),
                                "parent_refinement_trace": (
                                    intermediate_refinement_semantic_trace_hash(
                                        refinement_by_id[node.node_id]
                                    )
                                ),
                                "split_state": child.node.split_state_hash,
                            }.items()
                        )
                    ),
                    output_hash=_canonical_hash(child_program.hashes()),
                )
            )
            child_refinement = execute_native_intermediate_refinement_program(
                child_program, module, input_spec
            )
            if clock_ns() >= deadline_at_ns:
                discarded_stage = "after_child_refinement"
                failed = True
                break
            execute_task_id = f"{plan.plan_id}:{child.node.node_id}:execute"
            temporary_tasks.append(
                NativeObjectiveAncestralQueueTaskIRUnit(
                    sequence=-1,
                    task_id=execute_task_id,
                    kind=ObjectiveAncestralQueueTaskKind.EXECUTE_CHILD_REFINEMENT,
                    dependency_task_ids=(compile_task_id,),
                    node_id=child.node.node_id,
                    parent_node_id=node.node_id,
                    node_split_state_hash=child.node.split_state_hash,
                    input_hashes=(
                        ("refinement_program", child_program.plan.stable_hash()),
                    ),
                    output_hash=_canonical_hash(
                        intermediate_refinement_semantic_trace_hash(child_refinement)
                    ),
                )
            )
            child_evaluated, child_stack = _evaluate_serial(
                module,
                input_spec,
                objective=objective,
                node=child,
                batch_id=f"{query_id}:eval:{batch_serial:04d}",
                config=config,
                optimizer_policy=optimizer_policy,
                parent_by_id=runtime_by_id,
                relu_pre=child_refinement.relu_pre,
            )
            batch_serial += 1
            if clock_ns() > deadline_at_ns:
                discarded_stage = "after_child_evaluation"
                failed = True
                break
            eval_task_id = f"{plan.plan_id}:{child.node.node_id}:evaluate"
            temporary_tasks.append(
                NativeObjectiveAncestralQueueTaskIRUnit(
                    sequence=-1,
                    task_id=eval_task_id,
                    kind=ObjectiveAncestralQueueTaskKind.EVALUATE_CHILD,
                    dependency_task_ids=(execute_task_id,),
                    node_id=child.node.node_id,
                    parent_node_id=node.node_id,
                    node_split_state_hash=child.node.split_state_hash,
                    input_hashes=tuple(
                        sorted(
                            {
                                "final_intermediate_bounds": intermediate_bounds_hash(
                                    child_refinement.relu_pre
                                ),
                                "objective": plan.objective_hash,
                                "optimizer_policy": plan.optimizer_policy_hash,
                            }.items()
                        )
                    ),
                    output_hash=_evaluation_hash(child_evaluated),
                )
            )
            temporary_evaluated.append(child_evaluated)
            temporary_stacks.append(child_stack)
            temporary_refinements.append(
                NativeObjectiveAncestralNodeRefinement(
                    node_id=child.node.node_id,
                    parent_node_id=node.node_id,
                    node_split_state_hash=child.node.split_state_hash,
                    program=child_program,
                    execution=child_refinement,
                )
            )
        if failed:
            heapq.heappush(heap, entry)
            deadline_exhausted = True
            break
        transition_task = NativeObjectiveAncestralQueueTaskIRUnit(
            sequence=len(tasks),
            task_id=transition_id,
            kind=ObjectiveAncestralQueueTaskKind.TRANSITION_QUEUE,
            dependency_task_ids=(evaluation_task_by_id[node.node_id],),
            node_id=node.node_id,
            parent_node_id=node.parent_node_id,
            node_split_state_hash=node.split_state_hash,
            input_hashes=(("evaluation", _evaluation_hash(parent)),),
            output_hash=_canonical_hash(decision.to_dict()),
        )
        transition_task.validate()
        tasks.append(transition_task)
        for task in temporary_tasks:
            committed = NativeObjectiveAncestralQueueTaskIRUnit(
                **{**task.__dict__, "sequence": len(tasks)}
            )
            committed.validate()
            tasks.append(committed)
        decisions.append(decision)
        for child, evaluated, stack, refinement in zip(
            children,
            temporary_evaluated,
            temporary_stacks,
            temporary_refinements,
        ):
            evaluations.append(evaluated.evaluation)
            stacks.append(stack)
            runtime_by_id[child.node.node_id] = evaluated
            refinement_by_id[child.node.node_id] = refinement.execution
            node_refinements.append(refinement)
            evaluation_task_by_id[child.node.node_id] = (
                f"{plan.plan_id}:{child.node.node_id}:evaluate"
            )
            source_task_by_id[child.node.node_id] = (
                f"{plan.plan_id}:{child.node.node_id}:execute"
            )
            heapq.heappush(
                heap,
                _QueueEntry(
                    evaluated.evaluation.priority,
                    next_node_serial,
                    child.node.node_id,
                ),
            )
        max_queue_size = max(max_queue_size, len(heap))

    frontier = tuple(
        entry.node_id
        for entry in sorted(heap, key=lambda item: (item.priority, item.serial))
    )
    if deadline_exhausted:
        status: BabQueueStatus = "budget_exhausted"
        termination_reason = "whole_deadline_exhausted"
        fallback_reason = "deadline_preserve_accepted_frontier"
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
            ObjectiveAncestralQueueTaskKind.EVALUATE_ROOT,
            ObjectiveAncestralQueueTaskKind.EVALUATE_CHILD,
            ObjectiveAncestralQueueTaskKind.TRANSITION_QUEUE,
        }
    )
    _task(
        tasks,
        plan,
        suffix="emit",
        kind=ObjectiveAncestralQueueTaskKind.EMIT_RESULT,
        dependencies=emit_dependencies,
        node=None,
        inputs={"queue_trace": queue.trace.stable_hash()},
        output=queue.trace.to_dict(),
    )
    task_ir, schedule = lower_native_objective_ancestral_queue_schedule(
        plan, tuple(tasks)
    )
    finished_ns = clock_ns()
    source_elapsed_ns = max(0, queue_started_ns - started_ns)
    queue_elapsed_ns = max(0, finished_ns - queue_started_ns)
    whole_elapsed_ns = max(0, finished_ns - started_ns)
    trace = NativeObjectiveAncestralQueueTrace(
        query_id=query_id,
        plan_hash=plan.stable_hash(),
        task_ir_hash=task_ir.stable_hash(),
        schedule_hash=schedule.stable_hash(task_ir),
        queue_trace_hash=queue.trace.stable_hash(),
        node_refinement_semantics=tuple(
            item.semantic_dict() for item in node_refinements
        ),
        fallback_reason=fallback_reason,
        discarded_attempt_stage=discarded_stage,
        source_elapsed_ns=source_elapsed_ns,
        queue_elapsed_ns=queue_elapsed_ns,
        whole_elapsed_ns=whole_elapsed_ns,
        deadline_ns=plan.whole_query_timeout_ns,
        semantic_signature_hash="",
    )
    trace = NativeObjectiveAncestralQueueTrace(
        **{
            **trace.__dict__,
            "semantic_signature_hash": _canonical_hash(trace.semantic_dict()),
        }
    )
    execution = NativeObjectiveAncestralQueueExecution(
        plan=plan,
        task_ir=task_ir,
        schedule=schedule,
        queue=queue,
        node_refinements=tuple(node_refinements),
        trace=trace,
    )
    execution.validate_against(
        module,
        input_spec,
        linear_spec_C=objective,
        threshold=threshold,
        root_refinement=root_refinement,
        optimizer_policy=optimizer_policy,
    )
    return execution


__all__ = [
    "NativeObjectiveAncestralNodeRefinement",
    "NativeObjectiveAncestralQueueExecution",
    "NativeObjectiveAncestralQueueTrace",
    "compile_native_objective_ancestral_queue_plan",
    "execute_native_objective_ancestral_queue",
    "validate_native_objective_ancestral_queue_plan",
]
