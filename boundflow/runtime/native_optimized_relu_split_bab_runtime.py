"""Native ReLU-split BaB with Schedule-driven alpha/beta node optimization."""

# pylint: disable=too-many-arguments,too-many-locals,too-many-statements
# pylint: disable=too-many-branches,too-many-instance-attributes,too-many-lines
# pylint: disable=too-many-boolean-expressions,missing-function-docstring
# pylint: disable=protected-access,invalid-name,duplicate-code

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import heapq
import json
from typing import Mapping, Optional, Sequence

import torch

from ..domains.interval import IntervalState
from ..frontends.plain_crown_bound_ir import (
    relu_split_state_hash,
    tensor_content_hash,
)
from ..ir.bound import IntermediateBoundSource
from ..ir.schedule import LaunchAction
from ..ir.task import BFTaskModule
from .crown_ibp import _apply_relu_split, _forward_ibp_trace_mlp
from .native_alpha_beta_optimization_state import (
    NativeAlphaBetaOptimizationResult,
    NativeAlphaBetaOptimizationState,
    NativeAlphaBetaOptimizerPolicy,
    build_native_alpha_beta_scope,
    compile_native_alpha_beta_state_query,
    execute_native_alpha_beta_state_query,
)
from .native_alpha_beta_optimizer_schedule import (
    NativeOptimizerProgram,
    NativeScheduledOptimizerResult,
    compile_native_alpha_beta_optimizer_program,
    execute_native_alpha_beta_optimizer_program,
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
from .native_verifier_ir_integration import (
    NativePlainCrownRepresentationCompilation,
)
from .task_executor import InputSpec

NATIVE_OPTIMIZED_RELU_SPLIT_BAB_COMPILER_VERSION = (
    "boundflow.native-optimized-relu-split-bab/v1"
)
NATIVE_OPTIMIZED_RELU_SPLIT_BAB_TRACE_SCHEMA_VERSION = (
    "boundflow.optimized-relu-split-bab-trace/v1"
)
PARENT_OPTIMIZER_STATE_VALIDITY = "monotonic_refinement_initialization_only"
NATIVE_REEXECUTION_ATOL = 2e-6
NATIVE_REEXECUTION_RTOL = 2e-6
# Execution is still guarded by torch.allclose(atol, rtol) before trace creation.
# This scale-independent ceiling only prevents serialized trace inflation when
# the reference tensor scale is unavailable inside the standalone stack record.
NATIVE_REEXECUTION_TRACE_MAX_ABS_DIFF = 2e-3
_IR_HASH_KEYS = {
    "source_bound_module_hash",
    "source_plan_template_hash",
    "source_plan_instance_hash",
    "source_schedule_hash",
    "representation_binding_hash",
    "execution_bound_module_hash",
    "execution_plan_template_hash",
    "execution_plan_instance_hash",
    "task_module_hash",
    "schedule_hash",
}
_OPTIMIZER_HASH_KEYS = {
    "optimizer_plan_hash",
    "optimizer_task_module_hash",
    "optimizer_schedule_hash",
}


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
class NativeOptimizedBabEvaluation:
    """One queue node selected by optimizer Schedule and native re-execution."""

    node: NativeReluSplitBabNode
    lower: float
    upper: float
    priority: float
    selected_state_hash: str
    parent_selected_state_hash: Optional[str]
    warm_start_kind: str
    eval_batch_id: str
    eval_batch_position: int
    optimizer_ir_hashes: tuple[tuple[str, str], ...]
    optimizer_execution_trace_hash: str
    native_ir_hashes: tuple[tuple[str, str], ...]
    branch_candidate: Optional[ReluSplitBranch]
    parent_state_validity: str = PARENT_OPTIMIZER_STATE_VALIDITY
    parent_state_consumed_as_exact: bool = False

    def validate(self) -> None:
        self.node.validate()
        optimizer_hashes = dict(self.optimizer_ir_hashes)
        native_hashes = dict(self.native_ir_hashes)
        if (
            not self.eval_batch_id
            or self.eval_batch_position < 0
            or not _is_sha256(self.selected_state_hash)
            or not _is_sha256(self.optimizer_execution_trace_hash)
            or set(optimizer_hashes) != _OPTIMIZER_HASH_KEYS
            or len(optimizer_hashes) != len(self.optimizer_ir_hashes)
            or any(not _is_sha256(value) for value in optimizer_hashes.values())
            or set(native_hashes) != _IR_HASH_KEYS
            or len(native_hashes) != len(self.native_ir_hashes)
            or any(not _is_sha256(value) for value in native_hashes.values())
            or not all(
                torch.isfinite(torch.tensor(value)).item()
                for value in (self.lower, self.upper, self.priority)
            )
            or self.lower > self.upper
            or self.parent_state_validity != PARENT_OPTIMIZER_STATE_VALIDITY
            or self.parent_state_consumed_as_exact is not False
        ):
            raise ValueError("native optimized BaB evaluation is invalid")
        if self.node.depth == 0:
            if (
                self.parent_selected_state_hash is not None
                or self.warm_start_kind != "none"
            ):
                raise ValueError("native optimized BaB root warm state is invalid")
        elif (
            not _is_sha256(self.parent_selected_state_hash)
            or self.parent_selected_state_hash == self.selected_state_hash
            or self.warm_start_kind != "monotonic_split_refinement"
        ):
            raise ValueError("native optimized BaB parent warm-state link is invalid")
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
            "optimizer_ir_hashes": dict(self.optimizer_ir_hashes),
            "optimizer_execution_trace_hash": self.optimizer_execution_trace_hash,
            "native_ir_hashes": dict(self.native_ir_hashes),
            "branch_candidate": (
                None
                if self.branch_candidate is None
                else self.branch_candidate.to_dict()
            ),
        }


@dataclass(frozen=True)
class NativeOptimizedBabStackTrace:
    """One node batch's optimizer and selected-state native compiler stacks."""

    stack_id: str
    node_ids: tuple[str, ...]
    parent_selected_state_hashes: tuple[Optional[str], ...]
    domain_batch_size: int
    warm_start_kind: str
    warm_source_state_hash: Optional[str]
    optimizer_ir_hashes: tuple[tuple[str, str], ...]
    optimizer_action_count: int
    optimizer_evaluation_count: int
    optimizer_backward_count: int
    optimizer_projection_count: int
    alpha_gradient_l1: float
    beta_gradient_l1: float
    active_split_count: int
    optimizer_execution_trace_hash: str
    optimizer_selected_batch_state_hash: str
    selected_native_lower_max_abs_diff: float
    selected_native_upper_max_abs_diff: float
    native_ir_hashes: tuple[tuple[str, str], ...]
    native_task_count: int
    native_schedule_launch_count: int
    native_task_trace_event_count: int

    def validate(self, *, policy: NativeAlphaBetaOptimizerPolicy) -> None:
        policy.validate()
        optimizer_hashes = dict(self.optimizer_ir_hashes)
        native_hashes = dict(self.native_ir_hashes)
        expected_actions = (policy.steps + 1) * 2 + policy.steps * 3 + 1
        if (
            not self.stack_id
            or not self.node_ids
            or len(self.node_ids) != len(set(self.node_ids))
            or self.domain_batch_size != len(self.node_ids)
            or len(self.parent_selected_state_hashes) != self.domain_batch_size
            or self.warm_start_kind not in {"none", "monotonic_split_refinement"}
            or set(optimizer_hashes) != _OPTIMIZER_HASH_KEYS
            or len(optimizer_hashes) != len(self.optimizer_ir_hashes)
            or any(not _is_sha256(value) for value in optimizer_hashes.values())
            or self.optimizer_action_count != expected_actions
            or self.optimizer_evaluation_count != policy.steps + 1
            or self.optimizer_backward_count != policy.steps
            or self.optimizer_projection_count != policy.steps
            or not all(
                torch.isfinite(torch.tensor(value)).item()
                for value in (
                    self.alpha_gradient_l1,
                    self.beta_gradient_l1,
                    self.selected_native_lower_max_abs_diff,
                    self.selected_native_upper_max_abs_diff,
                )
            )
            or self.alpha_gradient_l1 < 0.0
            or self.beta_gradient_l1 < 0.0
            or self.active_split_count < 0
            or not _is_sha256(self.optimizer_execution_trace_hash)
            or not _is_sha256(self.optimizer_selected_batch_state_hash)
            or self.selected_native_lower_max_abs_diff < 0.0
            or self.selected_native_upper_max_abs_diff < 0.0
            or self.selected_native_lower_max_abs_diff
            > NATIVE_REEXECUTION_TRACE_MAX_ABS_DIFF
            or self.selected_native_upper_max_abs_diff
            > NATIVE_REEXECUTION_TRACE_MAX_ABS_DIFF
            or set(native_hashes) != _IR_HASH_KEYS
            or len(native_hashes) != len(self.native_ir_hashes)
            or any(not _is_sha256(value) for value in native_hashes.values())
            or self.native_task_count < 1
            or self.native_schedule_launch_count != self.native_task_count
            or self.native_task_trace_event_count != self.native_task_count
        ):
            raise ValueError("native optimized BaB stack trace is invalid")
        if self.warm_start_kind == "none":
            if self.warm_source_state_hash is not None or any(
                value is not None for value in self.parent_selected_state_hashes
            ):
                raise ValueError("root optimized stack declares parent warm states")
        elif not _is_sha256(self.warm_source_state_hash) or any(
            not _is_sha256(value) for value in self.parent_selected_state_hashes
        ):
            raise ValueError("child optimized stack lacks parent warm states")

    def to_dict(self, *, policy: NativeAlphaBetaOptimizerPolicy) -> dict[str, object]:
        self.validate(policy=policy)
        return {
            "stack_id": self.stack_id,
            "node_ids": list(self.node_ids),
            "parent_selected_state_hashes": list(self.parent_selected_state_hashes),
            "domain_batch_size": self.domain_batch_size,
            "warm_start_kind": self.warm_start_kind,
            "warm_source_state_hash": self.warm_source_state_hash,
            "optimizer_ir_hashes": dict(self.optimizer_ir_hashes),
            "optimizer_action_count": self.optimizer_action_count,
            "optimizer_evaluation_count": self.optimizer_evaluation_count,
            "optimizer_backward_count": self.optimizer_backward_count,
            "optimizer_projection_count": self.optimizer_projection_count,
            "alpha_gradient_l1": self.alpha_gradient_l1,
            "beta_gradient_l1": self.beta_gradient_l1,
            "active_split_count": self.active_split_count,
            "optimizer_execution_trace_hash": self.optimizer_execution_trace_hash,
            "optimizer_selected_batch_state_hash": (
                self.optimizer_selected_batch_state_hash
            ),
            "selected_native_lower_max_abs_diff": (
                self.selected_native_lower_max_abs_diff
            ),
            "selected_native_upper_max_abs_diff": (
                self.selected_native_upper_max_abs_diff
            ),
            "native_ir_hashes": dict(self.native_ir_hashes),
            "native_task_count": self.native_task_count,
            "native_schedule_launch_count": self.native_schedule_launch_count,
            "native_task_trace_event_count": self.native_task_trace_event_count,
        }


@dataclass(frozen=True)
class NativeOptimizedReluSplitBabTrace:
    """Replayable optimized bounded queue without a complete property claim."""

    run_id: str
    status: BabQueueStatus
    termination_reason: str
    config: NativeReluSplitBabConfig
    optimizer_policy: NativeAlphaBetaOptimizerPolicy
    root_input_lower_hash: str
    root_input_upper_hash: str
    objective_hash: str
    evaluations: tuple[NativeOptimizedBabEvaluation, ...]
    decisions: tuple[NativeReluSplitBabDecision, ...]
    final_frontier_node_ids: tuple[str, ...]
    native_stacks: tuple[NativeOptimizedBabStackTrace, ...]
    native_stack_count: int
    max_queue_size: int
    performance_claimed: bool = False
    property_status: str = "not_claimed"
    schema_version: str = NATIVE_OPTIMIZED_RELU_SPLIT_BAB_TRACE_SCHEMA_VERSION

    def validate(self) -> None:  # pylint: disable=too-many-statements
        self.config.validate()
        self.optimizer_policy.validate()
        if (
            self.schema_version != NATIVE_OPTIMIZED_RELU_SPLIT_BAB_TRACE_SCHEMA_VERSION
            or not self.run_id
            or self.status not in {"complete", "budget_exhausted"}
            or not self.termination_reason
            or any(
                not _is_sha256(value)
                for value in (
                    self.root_input_lower_hash,
                    self.root_input_upper_hash,
                    self.objective_hash,
                )
            )
            or not self.evaluations
            or not self.native_stacks
            or self.native_stack_count != len(self.native_stacks)
            or self.max_queue_size < 1
            or self.performance_claimed is not False
            or self.property_status != "not_claimed"
            or len(self.evaluations) > self.config.max_nodes
        ):
            raise ValueError("native optimized BaB trace header is invalid")

        evaluation_by_id: dict[str, NativeOptimizedBabEvaluation] = {}
        position: dict[str, int] = {}
        batches: dict[str, list[int]] = {}
        for index, evaluation in enumerate(self.evaluations):
            evaluation.validate()
            node = evaluation.node
            if node.node_id in evaluation_by_id:
                raise ValueError("native optimized BaB node was evaluated twice")
            if node.depth == 0 and index != 0:
                raise ValueError("native optimized BaB root is not first")
            if node.depth > 0:
                parent = evaluation_by_id.get(node.parent_node_id or "")
                if parent is None or node.depth != parent.node.depth + 1:
                    raise ValueError("native optimized BaB parent is absent or late")
                if evaluation.parent_selected_state_hash != parent.selected_state_hash:
                    raise ValueError("native optimized BaB parent state link differs")
            evaluation_by_id[node.node_id] = evaluation
            position[node.node_id] = index
            batches.setdefault(evaluation.eval_batch_id, []).append(
                evaluation.eval_batch_position
            )
        if len(batches) != self.native_stack_count or any(
            values != list(range(len(values))) for values in batches.values()
        ):
            raise ValueError("native optimized BaB evaluation batch accounting differs")

        stack_by_id: dict[str, NativeOptimizedBabStackTrace] = {}
        for stack in self.native_stacks:
            stack.validate(policy=self.optimizer_policy)
            if stack.stack_id in stack_by_id:
                raise ValueError("native optimized BaB stack ID repeats")
            stack_by_id[stack.stack_id] = stack
        if set(stack_by_id) != set(batches):
            raise ValueError("native optimized BaB stack/evaluation batches differ")
        for batch_id, values in batches.items():
            batch_evaluations = tuple(
                item for item in self.evaluations if item.eval_batch_id == batch_id
            )
            stack = stack_by_id[batch_id]
            if (
                stack.node_ids != tuple(item.node.node_id for item in batch_evaluations)
                or stack.domain_batch_size != len(values)
                or stack.parent_selected_state_hashes
                != tuple(item.parent_selected_state_hash for item in batch_evaluations)
                or any(
                    item.optimizer_ir_hashes != stack.optimizer_ir_hashes
                    or item.optimizer_execution_trace_hash
                    != stack.optimizer_execution_trace_hash
                    or item.native_ir_hashes != stack.native_ir_hashes
                    or item.warm_start_kind != stack.warm_start_kind
                    for item in batch_evaluations
                )
            ):
                raise ValueError("native optimized BaB stack/node binding differs")

        decision_nodes: set[str] = set()
        expanded_children: set[str] = set()
        for index, decision in enumerate(self.decisions):
            decision.validate()
            if decision.decision_index != index or decision.node_id in decision_nodes:
                raise ValueError("native optimized BaB decisions repeat or reorder")
            if decision.node_id not in evaluation_by_id:
                raise ValueError(
                    "native optimized BaB decision references unknown node"
                )
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
                        raise ValueError(
                            "native optimized BaB expansion branch differs"
                        )
                    expanded_children.add(child_id)
        frontier = set(self.final_frontier_node_ids)
        root_id = self.evaluations[0].node.node_id
        if (
            len(frontier) != len(self.final_frontier_node_ids)
            or not frontier <= set(evaluation_by_id)
            or frontier & decision_nodes
            or decision_nodes | frontier != set(evaluation_by_id)
            or expanded_children != set(evaluation_by_id) - {root_id}
        ):
            raise ValueError("native optimized BaB queue accounting does not close")
        if self.status == "complete" and frontier:
            raise ValueError("complete native optimized BaB queue retains a frontier")
        if self.status == "budget_exhausted" and not frontier:
            raise ValueError("budget-exhausted optimized BaB queue lacks frontier")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "compiler_version": NATIVE_OPTIMIZED_RELU_SPLIT_BAB_COMPILER_VERSION,
            "run_id": self.run_id,
            "status": self.status,
            "termination_reason": self.termination_reason,
            "performance_claimed": self.performance_claimed,
            "property_status": self.property_status,
            "config": self.config.to_dict(),
            "optimizer_policy": self.optimizer_policy.to_dict(),
            "root_input_lower_hash": self.root_input_lower_hash,
            "root_input_upper_hash": self.root_input_upper_hash,
            "objective_hash": self.objective_hash,
            "evaluations": [item.to_dict() for item in self.evaluations],
            "decisions": [item.to_dict() for item in self.decisions],
            "final_frontier_node_ids": list(self.final_frontier_node_ids),
            "native_stacks": [
                item.to_dict(policy=self.optimizer_policy)
                for item in self.native_stacks
            ],
            "native_stack_count": self.native_stack_count,
            "max_queue_size": self.max_queue_size,
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
class NativeOptimizedReluSplitBabExecution:
    """Queue trace plus non-serialized selected tensors for numeric comparison."""

    trace: NativeOptimizedReluSplitBabTrace
    selected_states: tuple[tuple[str, NativeAlphaBetaOptimizationState], ...]

    def validate(self) -> None:
        self.trace.validate()
        states = dict(self.selected_states)
        if len(states) != len(self.selected_states) or tuple(states) != tuple(
            item.node.node_id for item in self.trace.evaluations
        ):
            raise ValueError("native optimized BaB execution state coverage differs")
        for evaluation in self.trace.evaluations:
            state = states[evaluation.node.node_id]
            state.validate()
            if (
                state.stable_hash() != evaluation.selected_state_hash
                or state.scope.split_state_hash != evaluation.node.split_state_hash
            ):
                raise ValueError(
                    "native optimized BaB execution state identity differs"
                )

    def state_map(self) -> dict[str, NativeAlphaBetaOptimizationState]:
        self.validate()
        return dict(self.selected_states)


def compare_native_optimized_bab_states(
    left: NativeOptimizedReluSplitBabExecution,
    right: NativeOptimizedReluSplitBabExecution,
) -> dict[str, object]:
    """Compare packed/serial per-node alpha/beta tensors without hiding layout drift."""

    left.validate()
    right.validate()
    left_states = left.state_map()
    right_states = right.state_map()
    if tuple(left_states) != tuple(right_states):
        raise ValueError("optimized BaB state comparison node order differs")
    alpha_max_diff = 0.0
    beta_max_diff = 0.0
    split_exact = True
    stable_scope_fields_equal = True
    intermediate_scope_hashes_equal = True
    exact_state_hashes_equal = True
    for node_id in left_states:
        left_state = left_states[node_id]
        right_state = right_states[node_id]
        if (
            set(left_state.splits) != set(right_state.splits)
            or set(left_state.alphas) != set(right_state.alphas)
            or set(left_state.betas) != set(right_state.betas)
        ):
            raise ValueError("optimized BaB state comparison ReLU keys differ")
        for name in left_state.splits:
            left_split = left_state.splits[name]
            right_split = right_state.splits[name]
            left_alpha = left_state.alphas[name]
            right_alpha = right_state.alphas[name]
            left_beta = left_state.betas[name]
            right_beta = right_state.betas[name]
            if (
                left_split.shape != right_split.shape
                or left_alpha.shape != right_alpha.shape
                or left_beta.shape != right_beta.shape
            ):
                raise ValueError("optimized BaB state comparison tensor shape differs")
            split_exact = split_exact and torch.equal(left_split, right_split)
            alpha_max_diff = max(
                alpha_max_diff,
                float((left_alpha - right_alpha).abs().max().item()),
            )
            beta_max_diff = max(
                beta_max_diff,
                float((left_beta - right_beta).abs().max().item()),
            )
        left_scope = left_state.scope
        right_scope = right_state.scope
        stable_scope_fields_equal = stable_scope_fields_equal and all(
            getattr(left_scope, name) == getattr(right_scope, name)
            for name in (
                "primal_graph_hash",
                "input_region_hash",
                "objective_hash",
                "split_state_hash",
                "optimizer_policy_hash",
            )
        )
        intermediate_scope_hashes_equal = (
            intermediate_scope_hashes_equal
            and left_scope.intermediate_bounds_hash
            == right_scope.intermediate_bounds_hash
        )
        exact_state_hashes_equal = (
            exact_state_hashes_equal
            and left_state.stable_hash() == right_state.stable_hash()
        )
    return {
        "node_ids_same": True,
        "split_tensors_exact": split_exact,
        "stable_scope_fields_equal": stable_scope_fields_equal,
        "intermediate_scope_hashes_equal": intermediate_scope_hashes_equal,
        "exact_state_hashes_equal": exact_state_hashes_equal,
        "alpha_max_abs_diff": alpha_max_diff,
        "beta_max_abs_diff": beta_max_diff,
    }


@dataclass(frozen=True)
class _OptimizedEvaluatedNode:
    runtime_node: _RuntimeNode
    evaluation: NativeOptimizedBabEvaluation
    selected_state: NativeAlphaBetaOptimizationState


def _batched_split_state(nodes: tuple[_RuntimeNode, ...]) -> dict[str, torch.Tensor]:
    names = tuple(name for name, _tensor in nodes[0].split_state)
    if any(
        tuple(name for name, _tensor in node.split_state) != names for node in nodes
    ):
        raise ValueError("native optimized node batch changes split schema")
    return {
        name: torch.stack(
            tuple(dict(node.split_state)[name] for node in nodes), dim=0
        ).contiguous()
        for name in names
    }


def _repeat_relu_pre_override(
    relu_pre: Mapping[str, IntervalState], *, count: int
) -> dict[str, IntervalState]:
    if count < 1:
        raise ValueError("external ReLU bound repeat count must be positive")
    repeated: dict[str, IntervalState] = {}
    for name, value in relu_pre.items():
        if (
            not name
            or not isinstance(value, IntervalState)
            or int(value.lower.shape[0]) != 1
            or int(value.upper.shape[0]) != 1
        ):
            raise ValueError("external ReLU bounds require one source domain")
        repeats = (count, *(1 for _unused in value.lower.shape[1:]))
        repeated[name] = IntervalState(
            lower=value.lower.repeat(repeats).contiguous(),
            upper=value.upper.repeat(repeats).contiguous(),
        )
    return repeated


def _build_batched_parent_warm_state(
    legacy_task_module: BFTaskModule,
    batch_input: InputSpec,
    *,
    objective: torch.Tensor,
    nodes: tuple[_RuntimeNode, ...],
    policy: NativeAlphaBetaOptimizerPolicy,
    parent_by_id: Mapping[str, _OptimizedEvaluatedNode],
    relu_pre_override: Optional[Mapping[str, IntervalState]],
) -> NativeAlphaBetaOptimizationState:
    parents: list[_OptimizedEvaluatedNode] = []
    for node in nodes:
        parent_id = node.node.parent_node_id
        parent = None if parent_id is None else parent_by_id.get(parent_id)
        if parent is None:
            raise ValueError("optimized child batch lacks evaluated parent state")
        parents.append(parent)
    parent_nodes = tuple(item.runtime_node for item in parents)
    parent_splits = _batched_split_state(parent_nodes)
    _parent_env, local_parent_pre = _forward_ibp_trace_mlp(
        legacy_task_module,
        batch_input,
        relu_split_state=parent_splits,
    )
    parent_pre = (
        local_parent_pre
        if relu_pre_override is None
        else {
            name: _apply_relu_split(
                relu_pre_override[name],
                parent_splits[name],
                relu_input_name=name,
            )
            for name in local_parent_pre
        }
    )
    scope = build_native_alpha_beta_scope(
        legacy_task_module,
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
                    tuple(item.selected_state.alphas[name] for item in parents),
                    dim=0,
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
                    tuple(item.selected_state.betas[name] for item in parents),
                    dim=0,
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


def _slice_selected_state(
    program: NativeOptimizerProgram,
    root_input_spec: InputSpec,
    *,
    legacy_task_module: BFTaskModule,
    objective: torch.Tensor,
    selected: NativeScheduledOptimizerResult,
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
        legacy_task_module,
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
            (
                name,
                value[index : index + 1].detach().contiguous().clone(),
            )
            for name, value in sorted(selected.state.alphas.items())
        ),
        beta_by_relu_input=tuple(
            (
                name,
                value[index : index + 1].detach().contiguous().clone(),
            )
            for name, value in sorted(selected.state.betas.items())
        ),
    )
    state.validate()
    return state


def _optimizer_stack_trace(
    *,
    stack_id: str,
    nodes: tuple[_RuntimeNode, ...],
    parent_by_id: Mapping[str, _OptimizedEvaluatedNode],
    warm_state: Optional[NativeAlphaBetaOptimizationState],
    program: NativeOptimizerProgram,
    scheduled: NativeScheduledOptimizerResult,
    native_compilation: NativePlainCrownRepresentationCompilation,
    native_task_trace_count: int,
    selected_native_lower_max_abs_diff: float,
    selected_native_upper_max_abs_diff: float,
) -> NativeOptimizedBabStackTrace:
    backward = [
        action for action in scheduled.trace.actions if action.kind.value == "backward"
    ]
    projections = [
        action
        for action in scheduled.trace.actions
        if action.kind.value == "project_state"
    ]
    alpha_gradient = sum(float(action.alpha_gradient_l1 or 0.0) for action in backward)
    beta_gradient = sum(float(action.beta_gradient_l1 or 0.0) for action in backward)
    active_splits = sum(
        int((value != 0).sum().item())
        for value in program.initial_state.splits.values()
    )
    launches = sum(
        isinstance(action, LaunchAction)
        for action in native_compilation.schedule.actions
    )
    parent_hashes = tuple(
        (
            None
            if node.node.parent_node_id is None
            else parent_by_id[node.node.parent_node_id].selected_state.stable_hash()
        )
        for node in nodes
    )
    trace = NativeOptimizedBabStackTrace(
        stack_id=stack_id,
        node_ids=tuple(node.node.node_id for node in nodes),
        parent_selected_state_hashes=parent_hashes,
        domain_batch_size=len(nodes),
        warm_start_kind=program.plan.warm_start_kind,
        warm_source_state_hash=(
            None if warm_state is None else warm_state.stable_hash()
        ),
        optimizer_ir_hashes=tuple(sorted(program.hashes().items())),
        optimizer_action_count=len(scheduled.trace.actions),
        optimizer_evaluation_count=len(scheduled.trace.evaluations),
        optimizer_backward_count=len(backward),
        optimizer_projection_count=len(projections),
        alpha_gradient_l1=alpha_gradient,
        beta_gradient_l1=beta_gradient,
        active_split_count=active_splits,
        optimizer_execution_trace_hash=scheduled.trace.stable_hash(program=program),
        optimizer_selected_batch_state_hash=scheduled.state.stable_hash(),
        selected_native_lower_max_abs_diff=selected_native_lower_max_abs_diff,
        selected_native_upper_max_abs_diff=selected_native_upper_max_abs_diff,
        native_ir_hashes=tuple(sorted(native_compilation.hashes().items())),
        native_task_count=len(native_compilation.task_module.tasks),
        native_schedule_launch_count=launches,
        native_task_trace_event_count=native_task_trace_count,
    )
    trace.validate(policy=program.policy)
    return trace


def _evaluate_optimized_node_batch(
    legacy_task_module: BFTaskModule,
    root_input_spec: InputSpec,
    *,
    objective: torch.Tensor,
    nodes: tuple[_RuntimeNode, ...],
    batch_id: str,
    config: NativeReluSplitBabConfig,
    policy: NativeAlphaBetaOptimizerPolicy,
    parent_by_id: Mapping[str, _OptimizedEvaluatedNode],
    relu_pre_override: Optional[Mapping[str, IntervalState]],
    intermediate_bound_source: IntermediateBoundSource,
) -> tuple[tuple[_OptimizedEvaluatedNode, ...], NativeOptimizedBabStackTrace]:
    if not nodes:
        raise ValueError("native optimized node batch cannot be empty")
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
        else _build_batched_parent_warm_state(
            legacy_task_module,
            batch_input,
            objective=objective,
            nodes=nodes,
            policy=policy,
            parent_by_id=parent_by_id,
            relu_pre_override=batch_relu_pre_override,
        )
    )
    if any((node.node.depth == 0) != (warm_state is None) for node in nodes):
        raise ValueError("native optimized batch mixes root and child nodes")
    optimizer_program = compile_native_alpha_beta_optimizer_program(
        legacy_task_module,
        batch_input,
        linear_spec_C=objective,
        relu_split_state=split_batch,
        policy=policy,
        program_id=f"{batch_id}:optimizer",
        warm_start=warm_state,
        relu_pre_override=batch_relu_pre_override,
        intermediate_bound_source=intermediate_bound_source,
    )
    scheduled = execute_native_alpha_beta_optimizer_program(
        optimizer_program,
        legacy_task_module,
        batch_input,
        linear_spec_C=objective,
    )
    selected = NativeAlphaBetaOptimizationResult(
        bounds=scheduled.bounds,
        state=scheduled.state,
        interval_env=optimizer_program.interval_env,
        relu_pre=optimizer_program.relu_pre,
        warm_start_decision=optimizer_program.warm_start_decision,
    )
    native_compilation = compile_native_alpha_beta_state_query(
        legacy_task_module,
        batch_input,
        linear_spec_C=objective,
        optimization=selected,
        query_id=f"{batch_id}:selected-native",
        available_memory_bytes=config.available_memory_bytes,
        memory_budget_bytes=config.memory_budget_bytes,
        intermediate_bound_source=intermediate_bound_source,
    )
    native_bounds, native_task_trace = execute_native_alpha_beta_state_query(
        native_compilation,
        legacy_task_module,
        batch_input,
        linear_spec_C=objective,
        optimization=selected,
    )
    lower_diff = float(
        (native_bounds.lower - scheduled.bounds.lower).abs().max().item()
    )
    upper_diff = float(
        (native_bounds.upper - scheduled.bounds.upper).abs().max().item()
    )
    if not torch.allclose(
        native_bounds.lower,
        scheduled.bounds.lower,
        atol=NATIVE_REEXECUTION_ATOL,
        rtol=NATIVE_REEXECUTION_RTOL,
    ) or not torch.allclose(
        native_bounds.upper,
        scheduled.bounds.upper,
        atol=NATIVE_REEXECUTION_ATOL,
        rtol=NATIVE_REEXECUTION_RTOL,
    ):
        raise ValueError(
            "optimized queue selected-state native re-execution differs: "
            f"lower={lower_diff}, upper={upper_diff}"
        )
    if tuple(native_bounds.lower.shape) != (len(nodes), 1):
        raise ValueError("native optimized node batch must return one scalar objective")

    optimizer_hashes = tuple(sorted(optimizer_program.hashes().items()))
    optimizer_trace_hash = scheduled.trace.stable_hash(program=optimizer_program)
    native_hashes = tuple(sorted(native_compilation.hashes().items()))
    evaluated: list[_OptimizedEvaluatedNode] = []
    for index, runtime_node in enumerate(nodes):
        node_pre = {
            name: _slice_interval(value, index=index)
            for name, value in optimizer_program.relu_pre.items()
        }
        node_split = {
            name: value[index : index + 1].contiguous()
            for name, value in split_batch.items()
        }
        state = _slice_selected_state(
            optimizer_program,
            root_input_spec,
            legacy_task_module=legacy_task_module,
            objective=objective,
            selected=scheduled,
            index=index,
        )
        parent = (
            None
            if runtime_node.node.parent_node_id is None
            else parent_by_id.get(runtime_node.node.parent_node_id)
        )
        if runtime_node.node.depth > 0 and parent is None:
            raise ValueError("optimized node lacks evaluated parent")
        branch = _select_branch(node_pre, relu_split_state=node_split)
        lower = float(native_bounds.lower[index, 0].item())
        upper = float(native_bounds.upper[index, 0].item())
        evaluation = NativeOptimizedBabEvaluation(
            node=runtime_node.node,
            lower=lower,
            upper=upper,
            priority=_priority(lower),
            selected_state_hash=state.stable_hash(),
            parent_selected_state_hash=(
                None if parent is None else parent.selected_state.stable_hash()
            ),
            warm_start_kind=optimizer_program.plan.warm_start_kind,
            eval_batch_id=batch_id,
            eval_batch_position=index,
            optimizer_ir_hashes=optimizer_hashes,
            optimizer_execution_trace_hash=optimizer_trace_hash,
            native_ir_hashes=native_hashes,
            branch_candidate=branch,
        )
        evaluation.validate()
        evaluated.append(
            _OptimizedEvaluatedNode(
                runtime_node=runtime_node,
                evaluation=evaluation,
                selected_state=state,
            )
        )
    stack = _optimizer_stack_trace(
        stack_id=batch_id,
        nodes=nodes,
        parent_by_id=parent_by_id,
        warm_state=warm_state,
        program=optimizer_program,
        scheduled=scheduled,
        native_compilation=native_compilation,
        native_task_trace_count=len(native_task_trace.events),
        selected_native_lower_max_abs_diff=lower_diff,
        selected_native_upper_max_abs_diff=upper_diff,
    )
    return tuple(evaluated), stack


def execute_native_optimized_relu_split_bab(
    legacy_task_module: BFTaskModule,
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
) -> NativeOptimizedReluSplitBabExecution:
    """Run best-first queue with optimizer Schedule-driven native node bounds."""

    if not run_id:
        raise ValueError("native optimized BaB run ID must be non-empty")
    config.validate()
    optimizer_policy.validate()
    if not isinstance(intermediate_bound_source, IntermediateBoundSource):
        raise TypeError("optimized queue intermediate-bound source is invalid")
    if (relu_pre_override is None) != (
        intermediate_bound_source == IntermediateBoundSource.LOCAL_FORWARD
    ):
        raise ValueError("optimized queue intermediate semantics/provenance differ")
    legacy_task_module.validate()
    lower, upper = _root_box_bounds(input_spec)
    objective = _normalize_scalar_objective(linear_spec_C)
    _root_interval, root_pre = _forward_ibp_trace_mlp(legacy_task_module, input_spec)
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
    if not root_splits:
        raise ValueError("native optimized BaB requires at least one ReLU")
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

    evaluations: list[NativeOptimizedBabEvaluation] = []
    decisions: list[NativeReluSplitBabDecision] = []
    native_stacks: list[NativeOptimizedBabStackTrace] = []
    runtime_by_id: dict[str, _OptimizedEvaluatedNode] = {}
    batch_serial = 0
    next_node_serial = 1

    def evaluate(nodes: Sequence[_RuntimeNode]) -> None:
        nonlocal batch_serial
        for start in range(0, len(nodes), config.max_eval_batch_size):
            chunk = tuple(nodes[start : start + config.max_eval_batch_size])
            batch_id = f"{run_id}:eval:{batch_serial:04d}"
            batch_serial += 1
            evaluated, stack = _evaluate_optimized_node_batch(
                legacy_task_module,
                input_spec,
                objective=objective,
                nodes=chunk,
                batch_id=batch_id,
                config=config,
                policy=optimizer_policy,
                parent_by_id=runtime_by_id,
                relu_pre_override=relu_pre_override,
                intermediate_bound_source=intermediate_bound_source,
            )
            native_stacks.append(stack)
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
                    reason="widest_unsplit_ambiguous_relu",
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
    trace = NativeOptimizedReluSplitBabTrace(
        run_id=run_id,
        status=status,
        termination_reason=(
            "node_budget_exhausted"
            if budget_exhausted
            else "configured_bounded_tree_exhausted"
        ),
        config=config,
        optimizer_policy=optimizer_policy,
        root_input_lower_hash=tensor_content_hash(lower),
        root_input_upper_hash=tensor_content_hash(upper),
        objective_hash=tensor_content_hash(objective),
        evaluations=tuple(evaluations),
        decisions=tuple(decisions),
        final_frontier_node_ids=frontier,
        native_stacks=tuple(native_stacks),
        native_stack_count=len(native_stacks),
        max_queue_size=max_queue_size,
    )
    trace.validate()
    execution = NativeOptimizedReluSplitBabExecution(
        trace=trace,
        selected_states=tuple(
            (
                evaluation.node.node_id,
                runtime_by_id[evaluation.node.node_id].selected_state,
            )
            for evaluation in trace.evaluations
        ),
    )
    execution.validate()
    return execution


def run_native_optimized_relu_split_bab(
    legacy_task_module: BFTaskModule,
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
) -> NativeOptimizedReluSplitBabTrace:
    """Return the serialized optimized queue trace."""

    return execute_native_optimized_relu_split_bab(
        legacy_task_module,
        input_spec,
        linear_spec_C=linear_spec_C,
        run_id=run_id,
        config=config,
        optimizer_policy=optimizer_policy,
        relu_pre_override=relu_pre_override,
        intermediate_bound_source=intermediate_bound_source,
    ).trace
