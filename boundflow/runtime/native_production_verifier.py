"""Production prepared ReLU-split verifier without audit double execution."""

# pylint: disable=too-many-arguments,too-many-locals,too-many-statements
# pylint: disable=too-many-branches,too-many-instance-attributes,protected-access
# pylint: disable=too-many-lines,too-many-boolean-expressions
# pylint: disable=missing-function-docstring,duplicate-code

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import heapq
import json
import time
from typing import Mapping, Optional, Sequence

import torch

from ..domains.interval import IntervalState
from ..frontends.plain_crown_bound_ir import (
    relu_split_state_hash,
    tensor_content_hash,
)
from ..ir.bound import IntermediateBoundSource
from ..ir.production_verifier import (
    NativeProductionVerifierPlanIR,
    NativeProductionVerifierScheduleIR,
    NativeProductionVerifierTaskIR,
    NativeProductionVerifierTaskKind,
    lower_native_production_verifier_ir,
)
from ..ir.task import BFTaskModule
from .crown_ibp import _forward_ibp_trace_mlp
from .native_alpha_beta_optimization_state import (
    NativeAlphaBetaOptimizationState,
    NativeAlphaBetaOptimizerPolicy,
    build_native_alpha_beta_scope,
)
from .native_alpha_beta_optimizer_schedule import (
    NativeOptimizerProgram,
    NativePreparedOptimizerProgram,
    NativeProductionOptimizerResult,
    _optimizer_intermediate_semantics,
    compile_native_alpha_beta_optimizer_program,
    execute_prepared_native_alpha_beta_optimizer_program,
)
from .native_objective_branch_score import (
    NativeObjectiveBranchExecution,
    NativeObjectiveBranchPolicy,
    compile_native_objective_branch_program,
    execute_native_objective_branch_program,
)
from .native_optimized_relu_split_bab_runtime import (
    PARENT_OPTIMIZER_STATE_VALIDITY,
    _batched_split_state,
    _repeat_relu_pre_override,
)
from .native_relu_split_bab_runtime import (
    BabQueueStatus,
    NativeReluSplitBabConfig,
    NativeReluSplitBabDecision,
    NativeReluSplitBabNode,
    ReluSplitBranch,
    _make_child_runtime_node,
    _normalize_scalar_objective,
    _priority,
    _QueueEntry,
    _repeat_box_input_spec,
    _root_box_bounds,
    _RuntimeNode,
    _select_branch,
    _slice_interval,
)
from .task_executor import InputSpec

NATIVE_PRODUCTION_VERIFIER_TRACE_SCHEMA_VERSION = (
    "boundflow.native-production-verifier-trace/v1"
)
NATIVE_PRODUCTION_VERIFIER_BATCH_TRACE_SCHEMA_VERSION = (
    "boundflow.native-production-verifier-batch-trace/v1"
)
NATIVE_PRODUCTION_VERIFIER_ACTION_TRACE_SCHEMA_VERSION = (
    "boundflow.native-production-verifier-action-trace/v1"
)


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class NativeProductionVerifierActionTrace:
    """One real production Schedule dispatch without tensor hash construction."""

    sequence: int
    action_id: str
    task_id: str
    kind: NativeProductionVerifierTaskKind
    elapsed_ns: int
    schema_version: str = NATIVE_PRODUCTION_VERIFIER_ACTION_TRACE_SCHEMA_VERSION

    def validate(self) -> None:
        if (
            self.schema_version
            != NATIVE_PRODUCTION_VERIFIER_ACTION_TRACE_SCHEMA_VERSION
            or self.sequence < 0
            or not self.action_id
            or not self.task_id
            or self.elapsed_ns < 0
        ):
            raise ValueError("production verifier action trace is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "action_id": self.action_id,
            "task_id": self.task_id,
            "kind": self.kind.value,
            "elapsed_ns": self.elapsed_ns,
        }


@dataclass(frozen=True)
class NativeProductionVerifierBatchTrace:
    """Plan/Task/Schedule and phase timing for one dynamic node batch."""

    plan: NativeProductionVerifierPlanIR
    task_ir: NativeProductionVerifierTaskIR
    schedule: NativeProductionVerifierScheduleIR
    actions: tuple[NativeProductionVerifierActionTrace, ...]
    selected_batch_state_hash: str
    audit_hash_chain_constructed: bool = False
    selected_native_reexecution: bool = False
    schema_version: str = NATIVE_PRODUCTION_VERIFIER_BATCH_TRACE_SCHEMA_VERSION

    def validate(self) -> None:
        self.schedule.validate(plan=self.plan, task_ir=self.task_ir)
        if (
            self.schema_version != NATIVE_PRODUCTION_VERIFIER_BATCH_TRACE_SCHEMA_VERSION
            or len(self.actions) != len(self.schedule.actions)
            or not _is_sha256(self.selected_batch_state_hash)
            or self.audit_hash_chain_constructed is not False
            or self.selected_native_reexecution is not False
        ):
            raise ValueError("production verifier batch trace is invalid")
        for expected, actual in zip(self.schedule.actions, self.actions):
            actual.validate()
            if (
                actual.sequence != expected.sequence
                or actual.action_id != expected.action_id
                or actual.task_id != expected.task_id
                or actual.kind != expected.kind
            ):
                raise ValueError("production verifier runtime/Schedule order differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "plan": self.plan.to_dict(),
            "plan_hash": self.plan.stable_hash(),
            "task_ir": self.task_ir.to_dict(),
            "task_ir_hash": self.task_ir.stable_hash(),
            "schedule": self.schedule.to_dict(),
            "schedule_hash": self.schedule.stable_hash(
                plan=self.plan, task_ir=self.task_ir
            ),
            "actions": [action.to_dict() for action in self.actions],
            "selected_batch_state_hash": self.selected_batch_state_hash,
            "audit_hash_chain_constructed": self.audit_hash_chain_constructed,
            "selected_native_reexecution": self.selected_native_reexecution,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeProductionBabEvaluation:
    """One production queue node result bound to its batch program."""

    node: NativeReluSplitBabNode
    lower: float
    upper: float
    priority: float
    selected_state_hash: str
    parent_selected_state_hash: Optional[str]
    warm_start_kind: str
    eval_batch_id: str
    eval_batch_position: int
    batch_trace_hash: str
    branch_candidate: Optional[ReluSplitBranch]
    parent_state_validity: str = PARENT_OPTIMIZER_STATE_VALIDITY
    parent_state_consumed_as_exact: bool = False

    def validate(self) -> None:
        self.node.validate()
        if (
            not self.eval_batch_id
            or self.eval_batch_position < 0
            or not _is_sha256(self.selected_state_hash)
            or not _is_sha256(self.batch_trace_hash)
            or not all(
                torch.isfinite(torch.tensor(value)).item()
                for value in (self.lower, self.upper, self.priority)
            )
            or self.lower > self.upper
            or self.parent_state_validity != PARENT_OPTIMIZER_STATE_VALIDITY
            or self.parent_state_consumed_as_exact is not False
        ):
            raise ValueError("production verifier node evaluation is invalid")
        if self.node.depth == 0:
            if (
                self.parent_selected_state_hash is not None
                or self.warm_start_kind != "none"
            ):
                raise ValueError("production verifier root warm state is invalid")
        elif (
            not _is_sha256(self.parent_selected_state_hash)
            or self.parent_selected_state_hash == self.selected_state_hash
            or self.warm_start_kind != "monotonic_split_refinement"
        ):
            raise ValueError("production verifier child warm-state link is invalid")
        if self.branch_candidate is not None:
            self.branch_candidate.validate()

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "node": self.node.to_dict(),
            "lower": self.lower,
            "upper": self.upper,
            "priority": self.priority,
            "selected_state_hash": self.selected_state_hash,
            "parent_selected_state_hash": self.parent_selected_state_hash,
            "warm_start_kind": self.warm_start_kind,
            "parent_state_validity": self.parent_state_validity,
            "parent_state_consumed_as_exact": self.parent_state_consumed_as_exact,
            "eval_batch_id": self.eval_batch_id,
            "eval_batch_position": self.eval_batch_position,
            "batch_trace_hash": self.batch_trace_hash,
            "branch_candidate": (
                None
                if self.branch_candidate is None
                else self.branch_candidate.to_dict()
            ),
        }


@dataclass(frozen=True)
class NativeProductionReluSplitBabTrace:
    """Typed production bounded queue with explicit non-audit disclosure."""

    run_id: str
    status: BabQueueStatus
    termination_reason: str
    config: NativeReluSplitBabConfig
    optimizer_policy: NativeAlphaBetaOptimizerPolicy
    intermediate_bound_source: IntermediateBoundSource
    root_input_lower_hash: str
    root_input_upper_hash: str
    objective_hash: str
    evaluations: tuple[NativeProductionBabEvaluation, ...]
    decisions: tuple[NativeReluSplitBabDecision, ...]
    final_frontier_node_ids: tuple[str, ...]
    batches: tuple[NativeProductionVerifierBatchTrace, ...]
    max_queue_size: int
    performance_claimed: bool = False
    property_status: str = "not_claimed"
    audit_hash_chain_constructed: bool = False
    selected_native_reexecution: bool = False
    schema_version: str = NATIVE_PRODUCTION_VERIFIER_TRACE_SCHEMA_VERSION

    def validate(self) -> None:  # pylint: disable=too-many-statements
        self.config.validate()
        self.optimizer_policy.validate()
        if (
            self.schema_version != NATIVE_PRODUCTION_VERIFIER_TRACE_SCHEMA_VERSION
            or not self.run_id
            or self.status not in {"complete", "budget_exhausted"}
            or not self.termination_reason
            or not isinstance(self.intermediate_bound_source, IntermediateBoundSource)
            or any(
                not _is_sha256(value)
                for value in (
                    self.root_input_lower_hash,
                    self.root_input_upper_hash,
                    self.objective_hash,
                )
            )
            or not self.evaluations
            or not self.batches
            or self.max_queue_size < 1
            or self.performance_claimed is not False
            or self.property_status != "not_claimed"
            or self.audit_hash_chain_constructed is not False
            or self.selected_native_reexecution is not False
            or len(self.evaluations) > self.config.max_nodes
        ):
            raise ValueError("production verifier queue trace header is invalid")

        evaluation_by_id: dict[str, NativeProductionBabEvaluation] = {}
        position: dict[str, int] = {}
        batch_positions: dict[str, list[int]] = {}
        for index, evaluation in enumerate(self.evaluations):
            evaluation.validate()
            node = evaluation.node
            if node.node_id in evaluation_by_id:
                raise ValueError("production verifier node was evaluated twice")
            if node.depth == 0 and index != 0:
                raise ValueError("production verifier root is not first")
            if node.depth > 0:
                parent = evaluation_by_id.get(node.parent_node_id or "")
                if (
                    parent is None
                    or node.depth != parent.node.depth + 1
                    or evaluation.parent_selected_state_hash
                    != parent.selected_state_hash
                ):
                    raise ValueError("production verifier parent link differs")
            evaluation_by_id[node.node_id] = evaluation
            position[node.node_id] = index
            batch_positions.setdefault(evaluation.eval_batch_id, []).append(
                evaluation.eval_batch_position
            )
        if len(batch_positions) != len(self.batches) or any(
            values != list(range(len(values))) for values in batch_positions.values()
        ):
            raise ValueError("production verifier batch accounting differs")

        batch_by_id: dict[str, NativeProductionVerifierBatchTrace] = {}
        for batch in self.batches:
            batch.validate()
            batch_id = batch.plan.plan_id
            if batch_id in batch_by_id:
                raise ValueError("production verifier batch ID repeats")
            batch_by_id[batch_id] = batch
        if set(batch_by_id) != set(batch_positions):
            raise ValueError("production verifier Plan/evaluation batches differ")
        for batch_id, batch in batch_by_id.items():
            evaluations = tuple(
                item for item in self.evaluations if item.eval_batch_id == batch_id
            )
            if (
                batch.plan.node_ids != tuple(item.node.node_id for item in evaluations)
                or batch.plan.node_split_state_hashes
                != tuple(item.node.split_state_hash for item in evaluations)
                or batch.plan.parent_selected_state_hashes
                != tuple(item.parent_selected_state_hash for item in evaluations)
                or batch.plan.objective_hash != self.objective_hash
                or any(
                    item.batch_trace_hash != batch.stable_hash() for item in evaluations
                )
            ):
                raise ValueError("production verifier batch/node binding differs")

        decision_nodes: set[str] = set()
        expanded_children: set[str] = set()
        for index, decision in enumerate(self.decisions):
            decision.validate()
            if decision.decision_index != index or decision.node_id in decision_nodes:
                raise ValueError("production verifier decisions repeat or reorder")
            if decision.node_id not in evaluation_by_id:
                raise ValueError("production verifier decision references unknown node")
            decision_nodes.add(decision.node_id)
            if decision.kind == "expand":
                parent_position = position[decision.node_id]
                for child_index, child_id in enumerate(decision.child_node_ids):
                    child = evaluation_by_id.get(child_id)
                    branch = decision.branch_candidate
                    if (
                        child is None
                        or child.node.parent_node_id != decision.node_id
                        or branch is None
                        or child.node.branch_relu_input != branch.relu_input
                        or child.node.branch_neuron_index != branch.neuron_index
                        or child.node.branch_value != (-1 if child_index == 0 else 1)
                        or position[child_id] <= parent_position
                    ):
                        raise ValueError("production verifier expansion branch differs")
                    expanded_children.add(child_id)
        frontier = set(self.final_frontier_node_ids)
        root_id = self.evaluations[0].node.node_id
        if (
            len(frontier) != len(self.final_frontier_node_ids)
            or not frontier <= set(evaluation_by_id)
            or frontier & decision_nodes
            or decision_nodes | frontier != set(evaluation_by_id)
            or expanded_children != set(evaluation_by_id) - {root_id}
            or (self.status == "complete" and frontier)
            or (self.status == "budget_exhausted" and not frontier)
        ):
            raise ValueError("production verifier queue accounting does not close")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "status": self.status,
            "termination_reason": self.termination_reason,
            "config": self.config.to_dict(),
            "optimizer_policy": self.optimizer_policy.to_dict(),
            "intermediate_bound_source": self.intermediate_bound_source.value,
            "root_input_lower_hash": self.root_input_lower_hash,
            "root_input_upper_hash": self.root_input_upper_hash,
            "objective_hash": self.objective_hash,
            "evaluations": [item.to_dict() for item in self.evaluations],
            "decisions": [item.to_dict() for item in self.decisions],
            "final_frontier_node_ids": list(self.final_frontier_node_ids),
            "batches": [item.to_dict() for item in self.batches],
            "max_queue_size": self.max_queue_size,
            "performance_claimed": self.performance_claimed,
            "property_status": self.property_status,
            "audit_hash_chain_constructed": self.audit_hash_chain_constructed,
            "selected_native_reexecution": self.selected_native_reexecution,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())

    def logical_queue_signature(self) -> tuple[tuple[object, ...], ...]:
        decisions = {decision.node_id: decision for decision in self.decisions}
        return tuple(
            (
                item.node.node_id,
                item.node.parent_node_id,
                item.node.depth,
                item.node.branch_relu_input,
                item.node.branch_neuron_index,
                item.node.branch_value,
                (
                    None
                    if item.branch_candidate is None
                    else (
                        item.branch_candidate.relu_input,
                        item.branch_candidate.neuron_index,
                    )
                ),
                (
                    "frontier"
                    if item.node.node_id not in decisions
                    else decisions[item.node.node_id].kind
                ),
                (
                    "frontier"
                    if item.node.node_id not in decisions
                    else decisions[item.node.node_id].reason
                ),
            )
            for item in self.evaluations
        )


@dataclass(frozen=True)
class NativeProductionReluSplitBabExecution:
    """Production queue trace plus selected tensors for parity validation."""

    trace: NativeProductionReluSplitBabTrace
    selected_states: tuple[tuple[str, NativeAlphaBetaOptimizationState], ...]
    objective_branch_executions: tuple[
        tuple[str, NativeObjectiveBranchExecution], ...
    ] = ()
    objective_branch_policy: Optional[NativeObjectiveBranchPolicy] = None

    def state_map(self) -> dict[str, NativeAlphaBetaOptimizationState]:
        """Return selected states in deterministic queue evaluation order."""

        self.validate()
        return dict(self.selected_states)

    def validate(self) -> None:
        self.trace.validate()
        states = dict(self.selected_states)
        if len(states) != len(self.selected_states) or tuple(states) != tuple(
            item.node.node_id for item in self.trace.evaluations
        ):
            raise ValueError("production verifier selected-state coverage differs")
        for evaluation in self.trace.evaluations:
            state = states[evaluation.node.node_id]
            state.validate()
            if (
                state.stable_hash() != evaluation.selected_state_hash
                or state.scope.split_state_hash != evaluation.node.split_state_hash
            ):
                raise ValueError("production verifier selected-state identity differs")
        branches = dict(self.objective_branch_executions)
        if len(branches) != len(self.objective_branch_executions):
            raise ValueError("production verifier objective branch IDs repeat")
        if self.objective_branch_policy is None:
            if branches:
                raise ValueError("production verifier branches lack a policy")
        else:
            self.objective_branch_policy.validate()
            expected = {
                item.node.node_id
                for item in self.trace.evaluations
                if item.branch_candidate is not None
            }
            if set(branches) != expected:
                raise ValueError(
                    "production verifier objective branch coverage differs"
                )


@dataclass(frozen=True)
class _ProductionEvaluatedNode:
    runtime_node: _RuntimeNode
    evaluation: NativeProductionBabEvaluation
    selected_state: NativeAlphaBetaOptimizationState
    relu_pre: Mapping[str, IntervalState]


def _build_parent_warm_state(
    module: BFTaskModule,
    batch_input: InputSpec,
    *,
    objective: torch.Tensor,
    nodes: tuple[_RuntimeNode, ...],
    policy: NativeAlphaBetaOptimizerPolicy,
    parent_by_id: Mapping[str, _ProductionEvaluatedNode],
    relu_pre_override: Optional[Mapping[str, IntervalState]],
    intermediate_bound_source: IntermediateBoundSource,
    refine_external_constraints: bool,
) -> NativeAlphaBetaOptimizationState:
    parents: list[_ProductionEvaluatedNode] = []
    for node in nodes:
        parent_id = node.node.parent_node_id
        parent = None if parent_id is None else parent_by_id.get(parent_id)
        if parent is None:
            raise ValueError("production verifier child lacks an evaluated parent")
        parents.append(parent)
    parent_splits = _batched_split_state(tuple(item.runtime_node for item in parents))
    _parent_env, parent_pre = _optimizer_intermediate_semantics(
        module,
        batch_input,
        relu_split_state=parent_splits,
        relu_pre_override=relu_pre_override,
        intermediate_bound_source=intermediate_bound_source,
        refine_external_constraints=refine_external_constraints,
    )
    scope = build_native_alpha_beta_scope(
        module,
        batch_input,
        linear_spec_C=objective,
        relu_pre=parent_pre,
        relu_split_state=parent_splits,
        policy=policy,
    )
    names = tuple(sorted(parent_splits))
    state = NativeAlphaBetaOptimizationState(
        scope=scope,
        split_by_relu_input=tuple(
            (name, parent_splits[name].detach().contiguous().clone()) for name in names
        ),
        alpha_by_relu_input=tuple(
            (
                name,
                torch.cat(
                    tuple(item.selected_state.alphas[name] for item in parents), dim=0
                )
                .detach()
                .contiguous()
                .clone(),
            )
            for name in names
        ),
        beta_by_relu_input=tuple(
            (
                name,
                torch.cat(
                    tuple(item.selected_state.betas[name] for item in parents), dim=0
                )
                .detach()
                .contiguous()
                .clone(),
            )
            for name in names
        ),
    )
    state.validate()
    return state


def _slice_production_state(
    program: NativeOptimizerProgram,
    root_input_spec: InputSpec,
    *,
    module: BFTaskModule,
    objective: torch.Tensor,
    selected: NativeProductionOptimizerResult,
    index: int,
) -> NativeAlphaBetaOptimizationState:
    single_input = _repeat_box_input_spec(root_input_spec, count=1)
    node_pre = {
        name: _slice_interval(value, index=index)
        for name, value in program.relu_pre.items()
    }
    node_split = {
        name: value[index : index + 1].contiguous()
        for name, value in selected.state.splits.items()
    }
    scope = build_native_alpha_beta_scope(
        module,
        single_input,
        linear_spec_C=objective,
        relu_pre=node_pre,
        relu_split_state=node_split,
        policy=program.policy,
    )
    state = NativeAlphaBetaOptimizationState(
        scope=scope,
        split_by_relu_input=tuple(sorted(node_split.items())),
        alpha_by_relu_input=tuple(
            (name, value[index : index + 1].detach().contiguous().clone())
            for name, value in sorted(selected.state.alphas.items())
        ),
        beta_by_relu_input=tuple(
            (name, value[index : index + 1].detach().contiguous().clone())
            for name, value in sorted(selected.state.betas.items())
        ),
    )
    state.validate()
    return state


def _production_plan(
    *,
    batch_id: str,
    nodes: tuple[_RuntimeNode, ...],
    parent_by_id: Mapping[str, _ProductionEvaluatedNode],
    program: NativeOptimizerProgram,
    intermediate_bound_source: IntermediateBoundSource,
) -> NativeProductionVerifierPlanIR:
    scope = program.initial_state.scope
    plan = NativeProductionVerifierPlanIR(
        plan_id=batch_id,
        node_ids=tuple(node.node.node_id for node in nodes),
        node_split_state_hashes=tuple(node.node.split_state_hash for node in nodes),
        parent_selected_state_hashes=tuple(
            (
                None
                if node.node.parent_node_id is None
                else parent_by_id[node.node.parent_node_id].selected_state.stable_hash()
            )
            for node in nodes
        ),
        state_scope_hash=scope.stable_hash(),
        primal_graph_hash=scope.primal_graph_hash,
        input_region_hash=scope.input_region_hash,
        objective_hash=scope.objective_hash,
        optimizer_policy_hash=scope.optimizer_policy_hash,
        intermediate_bounds_hash=scope.intermediate_bounds_hash,
        intermediate_bound_source=intermediate_bound_source.value,
        optimizer_ir_hashes=tuple(sorted(program.hashes().items())),
    )
    plan.validate()
    return plan


def _evaluate_production_node_batch(
    module: BFTaskModule,
    root_input_spec: InputSpec,
    *,
    objective: torch.Tensor,
    nodes: tuple[_RuntimeNode, ...],
    batch_id: str,
    policy: NativeAlphaBetaOptimizerPolicy,
    parent_by_id: Mapping[str, _ProductionEvaluatedNode],
    relu_pre_override: Optional[Mapping[str, IntervalState]],
    intermediate_bound_source: IntermediateBoundSource,
    objective_branch_policy: Optional[NativeObjectiveBranchPolicy],
    refine_external_constraints: bool,
) -> tuple[
    tuple[_ProductionEvaluatedNode, ...],
    NativeProductionVerifierBatchTrace,
    tuple[tuple[str, NativeObjectiveBranchExecution], ...],
]:
    if not nodes:
        raise ValueError("production verifier node batch cannot be empty")
    batch_input = _repeat_box_input_spec(root_input_spec, count=len(nodes))
    split_batch = _batched_split_state(nodes)
    batch_relu_pre_override = (
        None
        if relu_pre_override is None
        else _repeat_relu_pre_override(relu_pre_override, count=len(nodes))
    )
    warm_state = (
        None
        if nodes[0].node.depth == 0
        else _build_parent_warm_state(
            module,
            batch_input,
            objective=objective,
            nodes=nodes,
            policy=policy,
            parent_by_id=parent_by_id,
            relu_pre_override=batch_relu_pre_override,
            intermediate_bound_source=intermediate_bound_source,
            refine_external_constraints=refine_external_constraints,
        )
    )
    if any((node.node.depth == 0) != (warm_state is None) for node in nodes):
        raise ValueError("production verifier batch mixes root and child nodes")
    optimizer_program = compile_native_alpha_beta_optimizer_program(
        module,
        batch_input,
        linear_spec_C=objective,
        relu_split_state=split_batch,
        policy=policy,
        program_id=f"{batch_id}:optimizer",
        warm_start=warm_state,
        relu_pre_override=batch_relu_pre_override,
        intermediate_bound_source=intermediate_bound_source,
        refine_external_constraints=refine_external_constraints,
    )
    plan = _production_plan(
        batch_id=batch_id,
        nodes=nodes,
        parent_by_id=parent_by_id,
        program=optimizer_program,
        intermediate_bound_source=intermediate_bound_source,
    )
    task_ir, schedule = lower_native_production_verifier_ir(plan)

    prepared: Optional[NativePreparedOptimizerProgram] = None
    selected: Optional[NativeProductionOptimizerResult] = None
    materialized: Optional[tuple[_ProductionEvaluatedNode, ...]] = None
    branch_executions: list[tuple[str, NativeObjectiveBranchExecution]] = []
    action_traces: list[NativeProductionVerifierActionTrace] = []
    for action in schedule.actions:
        started_ns = time.perf_counter_ns()
        if action.kind == NativeProductionVerifierTaskKind.VALIDATE_PROGRAM:
            prepared = NativePreparedOptimizerProgram.prepare(
                optimizer_program,
                module,
                batch_input,
                linear_spec_C=objective,
                intermediate_bound_source=intermediate_bound_source,
            )
        elif action.kind == NativeProductionVerifierTaskKind.EXECUTE_OPTIMIZER:
            if prepared is None:
                raise ValueError("production verifier executes before validation")
            selected = execute_prepared_native_alpha_beta_optimizer_program(
                prepared,
                module,
                batch_input,
                linear_spec_C=objective,
                intermediate_bound_source=intermediate_bound_source,
            )
            selected.validate(prepared=prepared)
        elif action.kind == NativeProductionVerifierTaskKind.MATERIALIZE_NODE_RESULTS:
            if selected is None:
                raise ValueError("production verifier materializes before execution")
            if tuple(selected.bounds.lower.shape) != (len(nodes), 1):
                raise ValueError(
                    "production verifier node batch must return one scalar objective"
                )
            items: list[_ProductionEvaluatedNode] = []
            for index, runtime_node in enumerate(nodes):
                node_pre = {
                    name: _slice_interval(value, index=index)
                    for name, value in optimizer_program.relu_pre.items()
                }
                node_split = {
                    name: value[index : index + 1].contiguous()
                    for name, value in split_batch.items()
                }
                state = _slice_production_state(
                    optimizer_program,
                    root_input_spec,
                    module=module,
                    objective=objective,
                    selected=selected,
                    index=index,
                )
                parent = (
                    None
                    if runtime_node.node.parent_node_id is None
                    else parent_by_id.get(runtime_node.node.parent_node_id)
                )
                if runtime_node.node.depth > 0 and parent is None:
                    raise ValueError("production verifier node lacks a parent")
                branch = _select_branch(node_pre, relu_split_state=node_split)
                if objective_branch_policy is not None and branch is not None:
                    branch_program = compile_native_objective_branch_program(
                        module,
                        _repeat_box_input_spec(root_input_spec, count=1),
                        linear_spec_C=objective,
                        relu_pre=node_pre,
                        selected_state=state,
                        optimizer_policy=policy,
                        branch_policy=objective_branch_policy,
                        intermediate_bound_source=intermediate_bound_source,
                        refine_external_constraints=refine_external_constraints,
                        plan_id=f"{batch_id}:node:{index}:objective-branch",
                    )
                    branch_execution = execute_native_objective_branch_program(
                        branch_program, node_id=runtime_node.node.node_id
                    )
                    branch = branch_execution.branch
                    branch_executions.append(
                        (runtime_node.node.node_id, branch_execution)
                    )
                lower = float(selected.bounds.lower[index, 0].item())
                upper = float(selected.bounds.upper[index, 0].item())
                items.append(
                    _ProductionEvaluatedNode(
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
                            warm_start_kind=optimizer_program.plan.warm_start_kind,
                            eval_batch_id=batch_id,
                            eval_batch_position=index,
                            batch_trace_hash="0" * 64,
                            branch_candidate=branch,
                        ),
                        selected_state=state,
                        relu_pre=node_pre,
                    )
                )
            materialized = tuple(items)
        elif action.kind == NativeProductionVerifierTaskKind.COMMIT_QUEUE_RESULTS:
            if materialized is None:
                raise ValueError("production verifier commits before materialization")
        else:
            raise AssertionError("unreachable production verifier task kind")
        action_traces.append(
            NativeProductionVerifierActionTrace(
                sequence=action.sequence,
                action_id=action.action_id,
                task_id=action.task_id,
                kind=action.kind,
                elapsed_ns=time.perf_counter_ns() - started_ns,
            )
        )
    if selected is None or materialized is None:
        raise ValueError("production verifier Schedule did not produce node results")
    batch_trace = NativeProductionVerifierBatchTrace(
        plan=plan,
        task_ir=task_ir,
        schedule=schedule,
        actions=tuple(action_traces),
        selected_batch_state_hash=selected.state.stable_hash(),
    )
    batch_trace.validate()
    trace_hash = batch_trace.stable_hash()
    rebound = tuple(
        _ProductionEvaluatedNode(
            runtime_node=item.runtime_node,
            evaluation=NativeProductionBabEvaluation(
                **{
                    **item.evaluation.__dict__,
                    "batch_trace_hash": trace_hash,
                }
            ),
            selected_state=item.selected_state,
            relu_pre=item.relu_pre,
        )
        for item in materialized
    )
    for item in rebound:
        item.evaluation.validate()
    return rebound, batch_trace, tuple(branch_executions)


def execute_native_production_relu_split_bab(
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    linear_spec_C: torch.Tensor,
    run_id: str,
    config: NativeReluSplitBabConfig,
    optimizer_policy: NativeAlphaBetaOptimizerPolicy,
    relu_pre_override: Optional[Mapping[str, IntervalState]] = None,
    intermediate_bound_source: IntermediateBoundSource = (
        IntermediateBoundSource.LOCAL_FORWARD
    ),
    objective_branch_policy: Optional[NativeObjectiveBranchPolicy] = None,
    refine_external_constraints: bool = False,
) -> NativeProductionReluSplitBabExecution:
    """Run the typed production queue without audit hashes or oracle re-execution."""

    if not run_id:
        raise ValueError("production verifier run ID must be non-empty")
    config.validate()
    optimizer_policy.validate()
    module.validate()
    if objective_branch_policy is not None:
        objective_branch_policy.validate()
    if not isinstance(intermediate_bound_source, IntermediateBoundSource):
        raise TypeError("production verifier intermediate-bound source is invalid")
    if not isinstance(refine_external_constraints, bool):
        raise TypeError("production verifier external refinement flag is invalid")
    if refine_external_constraints and intermediate_bound_source != (
        IntermediateBoundSource.EXTERNAL_VERIFIER
    ):
        raise ValueError("external constraint refinement requires external provenance")
    if (relu_pre_override is None) != (
        intermediate_bound_source == IntermediateBoundSource.LOCAL_FORWARD
    ):
        raise ValueError("production verifier intermediate semantics/provenance differ")

    lower, upper = _root_box_bounds(input_spec)
    objective = _normalize_scalar_objective(linear_spec_C)
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
    if not root_splits:
        raise ValueError("production verifier requires at least one ReLU")
    root_mapping = {name: value.unsqueeze(0) for name, value in root_splits}
    root = _RuntimeNode(
        node=NativeReluSplitBabNode(
            node_id=f"{run_id}:n000000",
            parent_node_id=None,
            depth=0,
            branch_relu_input=None,
            branch_neuron_index=None,
            branch_value=0,
            split_state_hash=relu_split_state_hash(root_mapping),
        ),
        split_state=root_splits,
    )

    evaluations: list[NativeProductionBabEvaluation] = []
    decisions: list[NativeReluSplitBabDecision] = []
    batches: list[NativeProductionVerifierBatchTrace] = []
    runtime_by_id: dict[str, _ProductionEvaluatedNode] = {}
    objective_branch_executions: list[tuple[str, NativeObjectiveBranchExecution]] = []
    batch_serial = 0
    next_node_serial = 1

    def evaluate(nodes: Sequence[_RuntimeNode]) -> None:
        nonlocal batch_serial
        for start in range(0, len(nodes), config.max_eval_batch_size):
            chunk = tuple(nodes[start : start + config.max_eval_batch_size])
            batch_id = f"{run_id}:eval:{batch_serial:04d}"
            batch_serial += 1
            evaluated, batch, branches = _evaluate_production_node_batch(
                module,
                input_spec,
                objective=objective,
                nodes=chunk,
                batch_id=batch_id,
                policy=optimizer_policy,
                parent_by_id=runtime_by_id,
                relu_pre_override=relu_pre_override,
                intermediate_bound_source=intermediate_bound_source,
                objective_branch_policy=objective_branch_policy,
                refine_external_constraints=refine_external_constraints,
            )
            batches.append(batch)
            objective_branch_executions.extend(branches)
            evaluations.extend(item.evaluation for item in evaluated)
            runtime_by_id.update(
                {item.runtime_node.node.node_id: item for item in evaluated}
            )

    evaluate((root,))
    heap: list[_QueueEntry] = []
    root_evaluation = runtime_by_id[root.node.node_id].evaluation
    heapq.heappush(heap, _QueueEntry(root_evaluation.priority, 0, root.node.node_id))
    max_queue_size = 1
    budget_exhausted = False

    while heap and not budget_exhausted:
        selected_entries = [
            heapq.heappop(heap)
            for _unused in range(min(config.expansion_batch_size, len(heap)))
        ]
        generated: list[_RuntimeNode] = []
        for selected_index, entry in enumerate(selected_entries):
            evaluated = runtime_by_id[entry.node_id]
            node = evaluated.runtime_node.node
            result = evaluated.evaluation
            if result.lower >= config.threshold:
                decisions.append(
                    NativeReluSplitBabDecision(
                        decision_index=len(decisions),
                        node_id=node.node_id,
                        kind="prune",
                        reason="lower_bound_meets_threshold",
                    )
                )
                continue
            if node.depth >= config.max_depth:
                decisions.append(
                    NativeReluSplitBabDecision(
                        decision_index=len(decisions),
                        node_id=node.node_id,
                        kind="terminal",
                        reason="configured_depth_limit",
                    )
                )
                continue
            branch = result.branch_candidate
            if branch is None:
                decisions.append(
                    NativeReluSplitBabDecision(
                        decision_index=len(decisions),
                        node_id=node.node_id,
                        kind="terminal",
                        reason="no_unsplit_ambiguous_relu",
                    )
                )
                continue
            if len(evaluations) + len(generated) + 2 > config.max_nodes:
                for pending in selected_entries[selected_index:]:
                    heapq.heappush(heap, pending)
                budget_exhausted = True
                break
            children: list[_RuntimeNode] = []
            for branch_value in (-1, 1):
                child_id = f"{run_id}:n{next_node_serial:06d}"
                next_node_serial += 1
                children.append(
                    _make_child_runtime_node(
                        evaluated.runtime_node,
                        child_id=child_id,
                        branch=branch,
                        branch_value=branch_value,
                    )
                )
            generated.extend(children)
            decisions.append(
                NativeReluSplitBabDecision(
                    decision_index=len(decisions),
                    node_id=node.node_id,
                    kind="expand",
                    reason=(
                        "objective_bound_impact"
                        if objective_branch_policy is not None
                        else "widest_unsplit_ambiguous_relu"
                    ),
                    child_node_ids=tuple(item.node.node_id for item in children),
                    branch_candidate=branch,
                )
            )
        if generated:
            evaluate(generated)
            for child in generated:
                evaluation = runtime_by_id[child.node.node_id].evaluation
                heapq.heappush(
                    heap,
                    _QueueEntry(
                        evaluation.priority,
                        next_node_serial,
                        child.node.node_id,
                    ),
                )
            max_queue_size = max(max_queue_size, len(heap))

    frontier = tuple(
        entry.node_id
        for entry in sorted(heap, key=lambda item: (item.priority, item.serial))
    )
    status: BabQueueStatus = "budget_exhausted" if budget_exhausted else "complete"
    trace = NativeProductionReluSplitBabTrace(
        run_id=run_id,
        status=status,
        termination_reason=(
            "node_budget_exhausted"
            if budget_exhausted
            else "configured_bounded_tree_exhausted"
        ),
        config=config,
        optimizer_policy=optimizer_policy,
        intermediate_bound_source=intermediate_bound_source,
        root_input_lower_hash=tensor_content_hash(lower),
        root_input_upper_hash=tensor_content_hash(upper),
        objective_hash=tensor_content_hash(objective),
        evaluations=tuple(evaluations),
        decisions=tuple(decisions),
        final_frontier_node_ids=frontier,
        batches=tuple(batches),
        max_queue_size=max_queue_size,
    )
    trace.validate()
    execution = NativeProductionReluSplitBabExecution(
        trace=trace,
        selected_states=tuple(
            (
                evaluation.node.node_id,
                runtime_by_id[evaluation.node.node_id].selected_state,
            )
            for evaluation in trace.evaluations
        ),
        objective_branch_executions=tuple(objective_branch_executions),
        objective_branch_policy=objective_branch_policy,
    )
    execution.validate()
    return execution


__all__ = [
    "NATIVE_PRODUCTION_VERIFIER_ACTION_TRACE_SCHEMA_VERSION",
    "NATIVE_PRODUCTION_VERIFIER_BATCH_TRACE_SCHEMA_VERSION",
    "NATIVE_PRODUCTION_VERIFIER_TRACE_SCHEMA_VERSION",
    "NativeProductionBabEvaluation",
    "NativeProductionReluSplitBabExecution",
    "NativeProductionReluSplitBabTrace",
    "NativeProductionVerifierActionTrace",
    "NativeProductionVerifierBatchTrace",
    "execute_native_production_relu_split_bab",
]
