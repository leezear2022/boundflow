"""Native ReLU-split BaB with Schedule-driven alpha/beta node optimization."""

# pylint: disable=too-many-arguments,too-many-locals,too-many-statements
# pylint: disable=too-many-branches,too-many-instance-attributes,too-many-lines
# pylint: disable=too-many-boolean-expressions,missing-function-docstring
# pylint: disable=protected-access,invalid-name,duplicate-code

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import heapq
import json
from typing import Literal, Mapping, Optional, Sequence

import torch

from ..domains.interval import IntervalState
from ..frontends.plain_crown_bound_ir import (
    relu_split_state_hash,
    tensor_content_hash,
)
from ..ir.bound import IntermediateBoundSource
from ..ir.refinement import (
    NativeIntermediateRefinementBudgetDecisionIR,
    NativeIntermediateRefinementBudgetPolicyIR,
    NativeIntermediateRefinementPolicyIR,
)
from ..ir.schedule import LaunchAction
from ..ir.task import BFTaskModule
from .crown_ibp import _forward_ibp_trace_mlp
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
    _optimizer_intermediate_semantics,
    compile_native_alpha_beta_optimizer_program,
    execute_native_alpha_beta_optimizer_program,
)
from .native_objective_branch_score import (
    NativeObjectiveBranchExecution,
    NativeObjectiveBranchPolicy,
    compile_native_objective_branch_program,
    execute_native_objective_branch_program,
)
from .native_intermediate_refinement import (
    NativeExternalIntermediateConstraintSeed,
    NativeIntermediateRefinementExecution,
    compile_native_intermediate_refinement_program,
    execute_native_intermediate_refinement_program,
    intermediate_bounds_hash,
    intermediate_refinement_semantic_trace_hash,
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
NATIVE_PER_CHILD_REFINEMENT_TRACE_SCHEMA_VERSION = (
    "boundflow.per-child-refinement-trace/v1"
)
PARENT_OPTIMIZER_STATE_VALIDITY = "monotonic_refinement_initialization_only"
NATIVE_REEXECUTION_ATOL = 1e-5
NATIVE_REEXECUTION_RTOL = 1e-5
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
PerChildRefinementStrategy = Literal[
    "independent_exact_split_v1",
    "ancestral_constraint_carry_v1",
    "external_seeded_ancestral_carry_v1",
]


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
class NativePerChildRefinementTrace:
    """Semantic evidence for one node's exact-split refinement program."""

    node_id: str
    node_split_state_hash: str
    refinement_plan_hash: str
    refinement_task_module_hash: str
    refinement_schedule_hash: str
    refinement_semantic_trace_hash: str
    initial_intermediate_bounds_hash: str
    final_intermediate_bounds_hash: str
    selected_target_count: int
    source_parent_node_id: Optional[str] = None
    source_intermediate_constraints_hash: Optional[str] = None
    source_refinement_plan_hash: Optional[str] = None
    source_refinement_semantic_trace_hash: Optional[str] = None
    source_consumption: Optional[str] = None
    external_constraint_seed_hash: Optional[str] = None
    external_constraint_seed_constraints_hash: Optional[str] = None
    external_semantics_owner: Optional[str] = None
    external_seed_consumption: Optional[str] = None
    budget_decision: Optional[NativeIntermediateRefinementBudgetDecisionIR] = None
    budget_policy: Optional[NativeIntermediateRefinementBudgetPolicyIR] = None
    parent_refinement_consumed_as_exact: bool = False
    performance_claimed: bool = False
    schema_version: str = NATIVE_PER_CHILD_REFINEMENT_TRACE_SCHEMA_VERSION

    def validate(self) -> None:
        source_hashes = (
            self.source_intermediate_constraints_hash,
            self.source_refinement_plan_hash,
            self.source_refinement_semantic_trace_hash,
        )
        source_present = self.source_parent_node_id is not None
        external_values = (
            self.external_constraint_seed_hash,
            self.external_constraint_seed_constraints_hash,
            self.external_semantics_owner,
            self.external_seed_consumption,
        )
        external_present = self.external_constraint_seed_hash is not None
        budget_present = self.budget_decision is not None
        if budget_present != (self.budget_policy is not None):
            raise ValueError("native per-child refinement budget presence differs")
        if self.budget_decision is not None and self.budget_policy is not None:
            self.budget_decision.validate(policy=self.budget_policy)
            if (
                self.budget_decision.node_id != self.node_id
                or self.budget_decision.node_split_state_hash
                != self.node_split_state_hash
            ):
                raise ValueError("native per-child refinement budget node differs")
        if (
            self.schema_version != NATIVE_PER_CHILD_REFINEMENT_TRACE_SCHEMA_VERSION
            or not self.node_id
            or any(
                not _is_sha256(value)
                for value in (
                    self.node_split_state_hash,
                    self.refinement_plan_hash,
                    self.refinement_task_module_hash,
                    self.refinement_schedule_hash,
                    self.refinement_semantic_trace_hash,
                    self.initial_intermediate_bounds_hash,
                    self.final_intermediate_bounds_hash,
                )
            )
            or self.selected_target_count < 1
            or self.parent_refinement_consumed_as_exact is not False
            or self.performance_claimed is not False
            or source_present != any(value is not None for value in source_hashes)
            or external_present != any(value is not None for value in external_values)
            or (source_present and external_present)
            or (
                source_present
                and (
                    not self.source_parent_node_id
                    or any(not _is_sha256(value) for value in source_hashes)
                    or self.source_consumption != "sound_constraint_only"
                )
            )
            or (not source_present and self.source_consumption is not None)
            or (
                external_present
                and (
                    not _is_sha256(self.external_constraint_seed_hash)
                    or not _is_sha256(self.external_constraint_seed_constraints_hash)
                    or self.external_semantics_owner != "external_verifier"
                    or self.external_seed_consumption
                    != "sound_constraint_intersection_only"
                )
            )
        ):
            raise ValueError("native per-child refinement trace is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "node_id": self.node_id,
            "node_split_state_hash": self.node_split_state_hash,
            "refinement_plan_hash": self.refinement_plan_hash,
            "refinement_task_module_hash": self.refinement_task_module_hash,
            "refinement_schedule_hash": self.refinement_schedule_hash,
            "refinement_semantic_trace_hash": self.refinement_semantic_trace_hash,
            "initial_intermediate_bounds_hash": (self.initial_intermediate_bounds_hash),
            "final_intermediate_bounds_hash": self.final_intermediate_bounds_hash,
            "selected_target_count": self.selected_target_count,
            "parent_refinement_consumed_as_exact": (
                self.parent_refinement_consumed_as_exact
            ),
            "performance_claimed": self.performance_claimed,
        }
        if self.source_parent_node_id is not None:
            payload["source_parent_node_id"] = self.source_parent_node_id
            payload["source_intermediate_constraints_hash"] = (
                self.source_intermediate_constraints_hash
            )
            payload["source_refinement_plan_hash"] = self.source_refinement_plan_hash
            payload["source_refinement_semantic_trace_hash"] = (
                self.source_refinement_semantic_trace_hash
            )
            payload["source_consumption"] = self.source_consumption
        if self.external_constraint_seed_hash is not None:
            payload["external_constraint_seed_hash"] = (
                self.external_constraint_seed_hash
            )
            payload["external_constraint_seed_constraints_hash"] = (
                self.external_constraint_seed_constraints_hash
            )
            payload["external_semantics_owner"] = self.external_semantics_owner
            payload["external_seed_consumption"] = self.external_seed_consumption
        if self.budget_decision is not None and self.budget_policy is not None:
            payload["budget_decision"] = self.budget_decision.to_dict(
                policy=self.budget_policy
            )
            payload["budget_decision_hash"] = self.budget_decision.stable_hash(
                policy=self.budget_policy
            )
        return payload

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


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
    intermediate_refinement_trace_hash: Optional[str] = None
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
            or (
                self.intermediate_refinement_trace_hash is not None
                and not _is_sha256(self.intermediate_refinement_trace_hash)
            )
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
        payload: dict[str, object] = {
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
        if self.intermediate_refinement_trace_hash is not None:
            payload["intermediate_refinement_trace_hash"] = (
                self.intermediate_refinement_trace_hash
            )
        return payload


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
    per_child_refinement_policy: Optional[NativeIntermediateRefinementPolicyIR] = None
    per_child_refinement_budget_policy: Optional[
        NativeIntermediateRefinementBudgetPolicyIR
    ] = None
    per_child_refinements: tuple[NativePerChildRefinementTrace, ...] = ()
    per_child_refinement_strategy: PerChildRefinementStrategy = (
        "independent_exact_split_v1"
    )

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

        if self.per_child_refinement_policy is None:
            if (
                self.per_child_refinement_strategy != "independent_exact_split_v1"
                or self.per_child_refinement_budget_policy is not None
                or self.per_child_refinements
                or any(
                    item.intermediate_refinement_trace_hash is not None
                    for item in self.evaluations
                )
            ):
                raise ValueError("native optimized BaB refinements lack a queue policy")
        else:
            self.per_child_refinement_policy.validate()
            if self.per_child_refinement_budget_policy is not None:
                self.per_child_refinement_budget_policy.validate()
                if (
                    self.per_child_refinement_strategy
                    != "external_seeded_ancestral_carry_v1"
                    or self.per_child_refinement_policy.max_neurons_per_relu
                    != self.per_child_refinement_budget_policy.base_max_neurons_per_relu
                    or self.per_child_refinement_policy.backward_chunk_size
                    > self.per_child_refinement_budget_policy.low_max_neurons_per_relu
                ):
                    raise ValueError(
                        "native optimized BaB refinement budget policy differs"
                    )
            if (
                self.per_child_refinement_policy.candidate_policy_id
                != "objective_influence_width_per_relu_v1"
                or self.per_child_refinement_strategy
                not in {
                    "independent_exact_split_v1",
                    "ancestral_constraint_carry_v1",
                    "external_seeded_ancestral_carry_v1",
                }
                or len(self.per_child_refinements) != len(self.evaluations)
            ):
                raise ValueError(
                    "native optimized BaB per-child refinement coverage differs"
                )
            for evaluation, refinement in zip(
                self.evaluations, self.per_child_refinements
            ):
                refinement.validate()
                if (refinement.budget_decision is None) != (
                    self.per_child_refinement_budget_policy is None
                ) or refinement.budget_policy != self.per_child_refinement_budget_policy:
                    raise ValueError(
                        "native optimized BaB refinement budget coverage differs"
                    )
                parent = (
                    None
                    if evaluation.node.parent_node_id is None
                    else evaluation_by_id[evaluation.node.parent_node_id]
                )
                parent_refinement = (
                    None
                    if parent is None
                    else self.per_child_refinements[position[parent.node.node_id]]
                )
                if (
                    refinement.node_id != evaluation.node.node_id
                    or refinement.node_split_state_hash
                    != evaluation.node.split_state_hash
                    or evaluation.intermediate_refinement_trace_hash
                    != refinement.stable_hash()
                ):
                    raise ValueError(
                        "native optimized BaB node/refinement binding differs"
                    )
                if self.per_child_refinement_strategy == ("independent_exact_split_v1"):
                    if (
                        refinement.source_parent_node_id is not None
                        or refinement.external_constraint_seed_hash is not None
                    ):
                        raise ValueError(
                            "independent per-child refinement consumes a source"
                        )
                elif (
                    self.per_child_refinement_strategy
                    == ("external_seeded_ancestral_carry_v1")
                    and parent is None
                ):
                    if (
                        refinement.source_parent_node_id is not None
                        or refinement.external_constraint_seed_hash is None
                    ):
                        raise ValueError("external-seeded root source differs")
                elif parent is None:
                    if (
                        refinement.source_parent_node_id is not None
                        or refinement.external_constraint_seed_hash is not None
                    ):
                        raise ValueError("ancestral refinement root declares a source")
                elif (
                    parent_refinement is None
                    or refinement.external_constraint_seed_hash is not None
                    or refinement.source_parent_node_id != parent.node.node_id
                    or refinement.source_intermediate_constraints_hash
                    != parent_refinement.final_intermediate_bounds_hash
                    or refinement.source_refinement_plan_hash
                    != parent_refinement.refinement_plan_hash
                    or refinement.source_refinement_semantic_trace_hash
                    != parent_refinement.refinement_semantic_trace_hash
                ):
                    raise ValueError("ancestral refinement parent lineage differs")
            if self.per_child_refinement_budget_policy is not None:
                groups: dict[
                    str, list[NativeIntermediateRefinementBudgetDecisionIR]
                ] = {}
                for refinement in self.per_child_refinements:
                    budget_decision = refinement.budget_decision
                    if budget_decision is None:
                        raise ValueError(
                            "native optimized BaB refinement budget decision is absent"
                        )
                    groups.setdefault(budget_decision.group_id, []).append(
                        budget_decision
                    )
                for decisions in groups.values():
                    first = decisions[0]
                    group_semantics = {
                        "group_id": first.group_id,
                        "budget_policy_hash": first.budget_policy_hash,
                        "base_cap_total": first.group_base_cap_total,
                        "assigned_cap_total": first.group_assigned_cap_total,
                        "nodes": [
                            {
                                "node_id": item.node_id,
                                "node_split_state_hash": (item.node_split_state_hash),
                                "node_depth": item.node_depth,
                                "parent_node_id": item.parent_node_id,
                                "parent_lower": item.parent_lower,
                                "assigned_max_neurons_per_relu": (
                                    item.assigned_max_neurons_per_relu
                                ),
                                "allocation_rank": item.allocation_rank,
                            }
                            for item in decisions
                        ],
                    }
                    if (
                        len(decisions) != first.group_size
                        or _canonical_hash(group_semantics) != first.group_semantic_hash
                        or any(
                            item.group_semantic_hash != first.group_semantic_hash
                            or item.group_size != first.group_size
                            or item.group_base_cap_total != first.group_base_cap_total
                            or item.group_assigned_cap_total
                            != first.group_assigned_cap_total
                            for item in decisions
                        )
                        or sum(item.assigned_max_neurons_per_relu for item in decisions)
                        != first.group_assigned_cap_total
                    ):
                        raise ValueError(
                            "native optimized BaB refinement budget group differs"
                        )

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
        payload: dict[str, object] = {
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
        if self.per_child_refinement_policy is not None:
            payload["per_child_refinement_policy"] = (
                self.per_child_refinement_policy.to_dict()
            )
            if self.per_child_refinement_budget_policy is not None:
                payload["per_child_refinement_budget_policy"] = (
                    self.per_child_refinement_budget_policy.to_dict()
                )
            payload["per_child_refinements"] = [
                item.to_dict() for item in self.per_child_refinements
            ]
            if self.per_child_refinement_strategy != "independent_exact_split_v1":
                payload["per_child_refinement_strategy"] = (
                    self.per_child_refinement_strategy
                )
        return payload

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
    objective_branch_executions: tuple[
        tuple[str, NativeObjectiveBranchExecution], ...
    ] = ()
    objective_branch_policy: Optional[NativeObjectiveBranchPolicy] = None
    per_child_refinement_executions: tuple[
        tuple[str, NativeIntermediateRefinementExecution], ...
    ] = ()

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
        branch_executions = dict(self.objective_branch_executions)
        if len(branch_executions) != len(self.objective_branch_executions):
            raise ValueError("native optimized BaB objective branch IDs repeat")
        if self.objective_branch_policy is None:
            if branch_executions:
                raise ValueError("native optimized BaB objective branches lack policy")
        else:
            self.objective_branch_policy.validate()
            expected = {
                item.node.node_id
                for item in self.trace.evaluations
                if item.branch_candidate is not None
            }
            if set(branch_executions) != expected:
                raise ValueError(
                    "native optimized BaB objective branch coverage differs"
                )
            for evaluation in self.trace.evaluations:
                if evaluation.branch_candidate is None:
                    continue
                branch = branch_executions[evaluation.node.node_id]
                branch.validate()
                if branch.branch != evaluation.branch_candidate:
                    raise ValueError(
                        "native optimized BaB objective branch selection differs"
                    )
        refinements = dict(self.per_child_refinement_executions)
        if len(refinements) != len(self.per_child_refinement_executions):
            raise ValueError("native optimized BaB refinement node IDs repeat")
        if self.trace.per_child_refinement_policy is None:
            if refinements:
                raise ValueError(
                    "native optimized BaB refinement executions lack policy"
                )
        else:
            records = {item.node_id: item for item in self.trace.per_child_refinements}
            if tuple(refinements) != tuple(
                item.node.node_id for item in self.trace.evaluations
            ):
                raise ValueError(
                    "native optimized BaB refinement execution coverage differs"
                )
            for node_id, refinement in refinements.items():
                program = refinement.program
                program.plan.validate()
                program.task_module.validate(plan=program.plan)
                program.schedule.validate(
                    plan=program.plan, task_module=program.task_module
                )
                refinement.trace.validate(program=program)
                record = records[node_id]
                expected_refinement_policy = self.trace.per_child_refinement_policy
                if record.budget_decision is not None:
                    expected_refinement_policy = replace(
                        expected_refinement_policy,
                        max_neurons_per_relu=(
                            record.budget_decision.assigned_max_neurons_per_relu
                        ),
                    )
                hashes = program.hashes()
                external_seed_ir = program.plan.external_constraint_seed
                if program.external_constraint_seed is not None:
                    program.external_constraint_seed.validate_payload()
                if (
                    program.plan.policy != expected_refinement_policy
                    or program.plan.objective_hash != self.trace.objective_hash
                    or program.plan.split_state_hash != record.node_split_state_hash
                    or hashes["refinement_plan_hash"] != record.refinement_plan_hash
                    or hashes["refinement_task_module_hash"]
                    != record.refinement_task_module_hash
                    or hashes["refinement_schedule_hash"]
                    != record.refinement_schedule_hash
                    or program.plan.initial_intermediate_bounds_hash
                    != record.initial_intermediate_bounds_hash
                    or program.plan.source_intermediate_constraints_hash
                    != record.source_intermediate_constraints_hash
                    or program.plan.source_refinement_plan_hash
                    != record.source_refinement_plan_hash
                    or program.plan.source_refinement_semantic_trace_hash
                    != record.source_refinement_semantic_trace_hash
                    or (
                        None
                        if external_seed_ir is None
                        else external_seed_ir.stable_hash()
                    )
                    != record.external_constraint_seed_hash
                    or (
                        None
                        if external_seed_ir is None
                        else external_seed_ir.bound_intermediate_constraints_hash
                    )
                    != record.external_constraint_seed_constraints_hash
                    or (
                        None
                        if external_seed_ir is None
                        else external_seed_ir.semantics_owner
                    )
                    != record.external_semantics_owner
                    or (
                        None
                        if external_seed_ir is None
                        else external_seed_ir.consumption
                    )
                    != record.external_seed_consumption
                    or (external_seed_ir is None)
                    != (program.external_constraint_seed is None)
                    or (
                        program.external_constraint_seed is not None
                        and program.external_constraint_seed.ir != external_seed_ir
                    )
                    or refinement.trace.final_intermediate_bounds_hash
                    != record.final_intermediate_bounds_hash
                    or intermediate_bounds_hash(refinement.relu_pre)
                    != record.final_intermediate_bounds_hash
                    or _refinement_semantic_trace_hash(refinement)
                    != record.refinement_semantic_trace_hash
                    or len(program.plan.targets) != record.selected_target_count
                ):
                    raise ValueError(
                        "native optimized BaB refinement execution identity differs"
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
    relu_pre: Mapping[str, IntervalState]
    refinement_execution: Optional[NativeIntermediateRefinementExecution] = None


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


def _node_split_mapping(node: _RuntimeNode) -> dict[str, torch.Tensor]:
    mapping = {
        name: value.unsqueeze(0).detach().contiguous().clone()
        for name, value in node.split_state
    }
    if relu_split_state_hash(mapping) != node.node.split_state_hash:
        raise ValueError("native optimized node split payload/hash differs")
    return mapping


def _batch_intermediate_bounds(
    values: Sequence[Mapping[str, IntervalState]],
) -> dict[str, IntervalState]:
    if not values:
        raise ValueError("native optimized intermediate batch cannot be empty")
    names = tuple(values[0])
    if not names or any(tuple(item) != names for item in values):
        raise ValueError("native optimized intermediate batch schema differs")
    result: dict[str, IntervalState] = {}
    for name in names:
        intervals = tuple(item[name] for item in values)
        reference = intervals[0]
        if any(
            not isinstance(item, IntervalState)
            or int(item.lower.shape[0]) != 1
            or item.lower.shape != reference.lower.shape
            or item.upper.shape != reference.upper.shape
            or item.lower.dtype != reference.lower.dtype
            or item.upper.dtype != reference.upper.dtype
            or item.lower.device != reference.lower.device
            or item.upper.device != reference.upper.device
            for item in intervals
        ):
            raise ValueError("native optimized intermediate tensor schema differs")
        result[name] = IntervalState(
            lower=torch.cat(tuple(item.lower for item in intervals), dim=0)
            .detach()
            .contiguous()
            .clone(),
            upper=torch.cat(tuple(item.upper for item in intervals), dim=0)
            .detach()
            .contiguous()
            .clone(),
        )
    return result


def _refinement_semantic_trace_hash(
    execution: NativeIntermediateRefinementExecution,
) -> str:
    return intermediate_refinement_semantic_trace_hash(execution)


def _refinement_budget_decisions(
    nodes: tuple[_RuntimeNode, ...],
    *,
    base_policy: NativeIntermediateRefinementPolicyIR,
    budget_policy: Optional[NativeIntermediateRefinementBudgetPolicyIR],
    parent_by_id: Mapping[str, _OptimizedEvaluatedNode],
    group_id: str,
) -> tuple[Optional[NativeIntermediateRefinementBudgetDecisionIR], ...]:
    if budget_policy is None:
        return tuple(None for _node in nodes)
    budget_policy.validate()
    if (
        not nodes
        or base_policy.max_neurons_per_relu != budget_policy.base_max_neurons_per_relu
        or base_policy.backward_chunk_size > budget_policy.low_max_neurons_per_relu
    ):
        raise ValueError("native refinement budget base policy differs")
    parent_lowers: dict[str, float] = {}
    for node in nodes:
        parent_id = node.node.parent_node_id
        if parent_id is None:
            if len(nodes) != 1 or node.node.depth != 0:
                raise ValueError("native refinement budget root group differs")
            continue
        parent = parent_by_id.get(parent_id)
        if parent is None:
            raise ValueError("native refinement budget parent is absent")
        parent_lowers[parent_id] = parent.evaluation.lower
    assignments: list[tuple[int, str]] = []
    if not parent_lowers:
        assignments = [(budget_policy.base_max_neurons_per_relu, "root")]
    elif len(parent_lowers) == 2:
        ordered = sorted(parent_lowers.items(), key=lambda item: (item[1], item[0]))
        separated = (
            abs(ordered[0][1] - ordered[1][1])
            > budget_policy.parent_lower_tie_tolerance
        )
        for node in nodes:
            parent_id = str(node.node.parent_node_id)
            if not separated:
                assignments.append((budget_policy.base_max_neurons_per_relu, "base"))
            elif parent_id == ordered[0][0]:
                assignments.append(
                    (budget_policy.high_max_neurons_per_relu, "high_risk")
                )
            else:
                assignments.append((budget_policy.low_max_neurons_per_relu, "low_risk"))
    elif len(parent_lowers) == 1:
        assignments = [
            (budget_policy.base_max_neurons_per_relu, "base") for _node in nodes
        ]
    else:
        raise ValueError("native refinement budget group has too many parents")
    base_total = len(nodes) * budget_policy.base_max_neurons_per_relu
    assigned_total = sum(value for value, _rank in assignments)
    if assigned_total != base_total:
        raise ValueError("native refinement budget group is not conserved")
    group_semantics = {
        "group_id": group_id,
        "budget_policy_hash": budget_policy.stable_hash(),
        "base_cap_total": base_total,
        "assigned_cap_total": assigned_total,
        "nodes": [
            {
                "node_id": node.node.node_id,
                "node_split_state_hash": node.node.split_state_hash,
                "node_depth": node.node.depth,
                "parent_node_id": node.node.parent_node_id,
                "parent_lower": (
                    None
                    if node.node.parent_node_id is None
                    else parent_lowers[node.node.parent_node_id]
                ),
                "assigned_max_neurons_per_relu": assignment[0],
                "allocation_rank": assignment[1],
            }
            for node, assignment in zip(nodes, assignments)
        ],
    }
    group_hash = _canonical_hash(group_semantics)
    decisions: list[Optional[NativeIntermediateRefinementBudgetDecisionIR]] = []
    for index, (node, assignment) in enumerate(zip(nodes, assignments)):
        parent_id = node.node.parent_node_id
        decision = NativeIntermediateRefinementBudgetDecisionIR(
            decision_id=f"{group_id}:budget:{index:04d}",
            budget_policy_hash=budget_policy.stable_hash(),
            group_id=group_id,
            group_semantic_hash=group_hash,
            group_size=len(nodes),
            group_base_cap_total=base_total,
            group_assigned_cap_total=assigned_total,
            node_id=node.node.node_id,
            node_split_state_hash=node.node.split_state_hash,
            node_depth=node.node.depth,
            assigned_max_neurons_per_relu=assignment[0],
            allocation_rank=assignment[1],
            parent_node_id=parent_id,
            parent_lower=(None if parent_id is None else parent_lowers[parent_id]),
        )
        decision.validate(policy=budget_policy)
        decisions.append(decision)
    return tuple(decisions)


def _execute_per_child_refinements(
    legacy_task_module: BFTaskModule,
    root_input_spec: InputSpec,
    *,
    objective: torch.Tensor,
    nodes: tuple[_RuntimeNode, ...],
    policy: NativeIntermediateRefinementPolicyIR,
    budget_policy: Optional[NativeIntermediateRefinementBudgetPolicyIR],
    budget_group_id: str,
    parent_by_id: Mapping[str, _OptimizedEvaluatedNode],
    strategy: PerChildRefinementStrategy,
    external_constraint_seed: Optional[NativeExternalIntermediateConstraintSeed],
) -> tuple[
    dict[str, IntervalState],
    tuple[tuple[str, NativeIntermediateRefinementExecution], ...],
    tuple[NativePerChildRefinementTrace, ...],
]:
    single_input = _repeat_box_input_spec(root_input_spec, count=1)
    executions: list[tuple[str, NativeIntermediateRefinementExecution]] = []
    records: list[NativePerChildRefinementTrace] = []
    budget_decisions = _refinement_budget_decisions(
        nodes,
        base_policy=policy,
        budget_policy=budget_policy,
        parent_by_id=parent_by_id,
        group_id=budget_group_id,
    )
    for node, budget_decision in zip(nodes, budget_decisions):
        split = _node_split_mapping(node)
        parent_id = node.node.parent_node_id
        source_execution: Optional[NativeIntermediateRefinementExecution] = None
        node_external_seed: Optional[NativeExternalIntermediateConstraintSeed] = None
        if (
            strategy
            in {
                "ancestral_constraint_carry_v1",
                "external_seeded_ancestral_carry_v1",
            }
            and parent_id is not None
        ):
            parent = parent_by_id.get(parent_id)
            if parent is None or parent.refinement_execution is None:
                raise ValueError("ancestral refinement child lacks a parent execution")
            source_execution = parent.refinement_execution
        elif strategy == "external_seeded_ancestral_carry_v1":
            if external_constraint_seed is None:
                raise ValueError("external-seeded refinement root lacks a seed")
            node_external_seed = external_constraint_seed
        program = compile_native_intermediate_refinement_program(
            legacy_task_module,
            single_input,
            policy=(
                policy
                if budget_decision is None
                else replace(
                    policy,
                    max_neurons_per_relu=(
                        budget_decision.assigned_max_neurons_per_relu
                    ),
                )
            ),
            plan_id=f"per-child-refinement:{node.node.split_state_hash}",
            relu_split_state=split,
            linear_spec_C=objective,
            source_refinement_execution=source_execution,
            external_constraint_seed=node_external_seed,
        )
        execution = execute_native_intermediate_refinement_program(
            program, legacy_task_module, single_input
        )
        hashes = program.hashes()
        record = NativePerChildRefinementTrace(
            node_id=node.node.node_id,
            node_split_state_hash=node.node.split_state_hash,
            refinement_plan_hash=hashes["refinement_plan_hash"],
            refinement_task_module_hash=hashes["refinement_task_module_hash"],
            refinement_schedule_hash=hashes["refinement_schedule_hash"],
            refinement_semantic_trace_hash=_refinement_semantic_trace_hash(execution),
            initial_intermediate_bounds_hash=(
                program.plan.initial_intermediate_bounds_hash
            ),
            final_intermediate_bounds_hash=(
                execution.trace.final_intermediate_bounds_hash
            ),
            selected_target_count=len(program.plan.targets),
            source_parent_node_id=(None if source_execution is None else parent_id),
            source_intermediate_constraints_hash=(
                program.plan.source_intermediate_constraints_hash
            ),
            source_refinement_plan_hash=program.plan.source_refinement_plan_hash,
            source_refinement_semantic_trace_hash=(
                program.plan.source_refinement_semantic_trace_hash
            ),
            source_consumption=(
                None if source_execution is None else "sound_constraint_only"
            ),
            external_constraint_seed_hash=(
                None
                if program.plan.external_constraint_seed is None
                else program.plan.external_constraint_seed.stable_hash()
            ),
            external_constraint_seed_constraints_hash=(
                None
                if program.plan.external_constraint_seed is None
                else program.plan.external_constraint_seed.bound_intermediate_constraints_hash
            ),
            external_semantics_owner=(
                None
                if program.plan.external_constraint_seed is None
                else program.plan.external_constraint_seed.semantics_owner
            ),
            external_seed_consumption=(
                None
                if program.plan.external_constraint_seed is None
                else program.plan.external_constraint_seed.consumption
            ),
            budget_decision=budget_decision,
            budget_policy=budget_policy,
        )
        record.validate()
        executions.append((node.node.node_id, execution))
        records.append(record)
    return (
        _batch_intermediate_bounds(
            tuple(execution.relu_pre for _node_id, execution in executions)
        ),
        tuple(executions),
        tuple(records),
    )


def _build_batched_parent_warm_state(
    legacy_task_module: BFTaskModule,
    batch_input: InputSpec,
    *,
    objective: torch.Tensor,
    nodes: tuple[_RuntimeNode, ...],
    policy: NativeAlphaBetaOptimizerPolicy,
    parent_by_id: Mapping[str, _OptimizedEvaluatedNode],
    relu_pre_override: Optional[Mapping[str, IntervalState]],
    intermediate_bound_source: IntermediateBoundSource,
    refine_external_constraints: bool,
    use_parent_runtime_bounds: bool = False,
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
    parent_pre: Mapping[str, IntervalState]
    if use_parent_runtime_bounds:
        parent_pre = _batch_intermediate_bounds(
            tuple(item.relu_pre for item in parents)
        )
    else:
        _parent_env, parent_pre = _optimizer_intermediate_semantics(
            legacy_task_module,
            batch_input,
            relu_split_state=parent_splits,
            relu_pre_override=relu_pre_override,
            intermediate_bound_source=intermediate_bound_source,
            refine_external_constraints=refine_external_constraints,
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
    objective_branch_policy: Optional[NativeObjectiveBranchPolicy],
    refine_external_constraints: bool,
    per_child_refinement_policy: Optional[NativeIntermediateRefinementPolicyIR],
    per_child_refinement_budget_policy: Optional[
        NativeIntermediateRefinementBudgetPolicyIR
    ],
    per_child_refinement_strategy: PerChildRefinementStrategy,
    external_constraint_seed: Optional[NativeExternalIntermediateConstraintSeed],
) -> tuple[
    tuple[_OptimizedEvaluatedNode, ...],
    NativeOptimizedBabStackTrace,
    tuple[tuple[str, NativeObjectiveBranchExecution], ...],
    tuple[tuple[str, NativeIntermediateRefinementExecution], ...],
    tuple[NativePerChildRefinementTrace, ...],
]:
    if not nodes:
        raise ValueError("native optimized node batch cannot be empty")
    batch_input = _repeat_box_input_spec(root_input_spec, count=len(nodes))
    split_batch = _batched_split_state(nodes)
    refinement_executions: tuple[
        tuple[str, NativeIntermediateRefinementExecution], ...
    ] = ()
    refinement_records: tuple[NativePerChildRefinementTrace, ...] = ()
    if per_child_refinement_policy is None:
        batch_relu_pre_override = (
            None
            if relu_pre_override is None
            else _repeat_relu_pre_override(relu_pre_override, count=len(nodes))
        )
        effective_intermediate_bound_source = intermediate_bound_source
        effective_refine_external_constraints = refine_external_constraints
    else:
        (
            batch_relu_pre_override,
            refinement_executions,
            refinement_records,
        ) = _execute_per_child_refinements(
            legacy_task_module,
            root_input_spec,
            objective=objective,
            nodes=nodes,
            policy=per_child_refinement_policy,
            budget_policy=per_child_refinement_budget_policy,
            budget_group_id=batch_id,
            parent_by_id=parent_by_id,
            strategy=per_child_refinement_strategy,
            external_constraint_seed=external_constraint_seed,
        )
        effective_intermediate_bound_source = IntermediateBoundSource.NATIVE_REFINED
        effective_refine_external_constraints = False
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
            intermediate_bound_source=effective_intermediate_bound_source,
            refine_external_constraints=effective_refine_external_constraints,
            use_parent_runtime_bounds=per_child_refinement_policy is not None,
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
        intermediate_bound_source=effective_intermediate_bound_source,
        refine_external_constraints=effective_refine_external_constraints,
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
        intermediate_bound_source=effective_intermediate_bound_source,
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
    objective_branches: list[tuple[str, NativeObjectiveBranchExecution]] = []
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
        if objective_branch_policy is not None and branch is not None:
            branch_program = compile_native_objective_branch_program(
                legacy_task_module,
                _repeat_box_input_spec(root_input_spec, count=1),
                linear_spec_C=objective,
                relu_pre=node_pre,
                selected_state=state,
                optimizer_policy=policy,
                branch_policy=objective_branch_policy,
                intermediate_bound_source=effective_intermediate_bound_source,
                refine_external_constraints=effective_refine_external_constraints,
                plan_id=f"{batch_id}:node:{index}:objective-branch",
            )
            branch_execution = execute_native_objective_branch_program(
                branch_program,
                node_id=runtime_node.node.node_id,
            )
            branch = branch_execution.branch
            objective_branches.append((runtime_node.node.node_id, branch_execution))
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
            intermediate_refinement_trace_hash=(
                None
                if per_child_refinement_policy is None
                else refinement_records[index].stable_hash()
            ),
        )
        evaluation.validate()
        evaluated.append(
            _OptimizedEvaluatedNode(
                runtime_node=runtime_node,
                evaluation=evaluation,
                selected_state=state,
                relu_pre=node_pre,
                refinement_execution=(
                    None
                    if per_child_refinement_policy is None
                    else refinement_executions[index][1]
                ),
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
    return (
        tuple(evaluated),
        stack,
        tuple(objective_branches),
        refinement_executions,
        refinement_records,
    )


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
    objective_branch_policy: Optional[NativeObjectiveBranchPolicy] = None,
    refine_external_constraints: bool = False,
    per_child_refinement_policy: Optional[NativeIntermediateRefinementPolicyIR] = None,
    per_child_refinement_budget_policy: Optional[
        NativeIntermediateRefinementBudgetPolicyIR
    ] = None,
    per_child_refinement_strategy: PerChildRefinementStrategy = (
        "independent_exact_split_v1"
    ),
    external_constraint_seed: Optional[NativeExternalIntermediateConstraintSeed] = None,
) -> NativeOptimizedReluSplitBabExecution:
    """Run best-first queue with optimizer Schedule-driven native node bounds."""

    if not run_id:
        raise ValueError("native optimized BaB run ID must be non-empty")
    config.validate()
    optimizer_policy.validate()
    if objective_branch_policy is not None:
        objective_branch_policy.validate()
    if not isinstance(refine_external_constraints, bool):
        raise TypeError("optimized queue external refinement flag is invalid")
    if refine_external_constraints and intermediate_bound_source != (
        IntermediateBoundSource.EXTERNAL_VERIFIER
    ):
        raise ValueError("external constraint refinement requires external provenance")
    if not isinstance(intermediate_bound_source, IntermediateBoundSource):
        raise TypeError("optimized queue intermediate-bound source is invalid")
    if per_child_refinement_strategy not in {
        "independent_exact_split_v1",
        "ancestral_constraint_carry_v1",
        "external_seeded_ancestral_carry_v1",
    }:
        raise ValueError("optimized queue per-child refinement strategy is invalid")
    if per_child_refinement_policy is None:
        if per_child_refinement_strategy != "independent_exact_split_v1":
            raise ValueError("optimized queue refinement strategy requires a policy")
        if (relu_pre_override is None) != (
            intermediate_bound_source == IntermediateBoundSource.LOCAL_FORWARD
        ):
            raise ValueError("optimized queue intermediate semantics/provenance differ")
        if external_constraint_seed is not None:
            raise ValueError("optimized queue external seed requires refinement")
        if per_child_refinement_budget_policy is not None:
            raise ValueError("optimized queue refinement budget requires refinement")
    else:
        if not isinstance(
            per_child_refinement_policy, NativeIntermediateRefinementPolicyIR
        ):
            raise TypeError("optimized queue per-child refinement policy is invalid")
        per_child_refinement_policy.validate()
        if per_child_refinement_budget_policy is not None:
            if not isinstance(
                per_child_refinement_budget_policy,
                NativeIntermediateRefinementBudgetPolicyIR,
            ):
                raise TypeError("optimized queue refinement budget policy is invalid")
            per_child_refinement_budget_policy.validate()
            if (
                per_child_refinement_strategy != "external_seeded_ancestral_carry_v1"
                or per_child_refinement_policy.max_neurons_per_relu
                != per_child_refinement_budget_policy.base_max_neurons_per_relu
                or per_child_refinement_policy.backward_chunk_size
                > per_child_refinement_budget_policy.low_max_neurons_per_relu
            ):
                raise ValueError("optimized queue refinement budget policy differs")
        if (
            per_child_refinement_policy.candidate_policy_id
            != "objective_influence_width_per_relu_v1"
        ):
            raise ValueError(
                "optimized queue per-child refinement must be objective-directed"
            )
        if (
            relu_pre_override is not None
            or intermediate_bound_source != IntermediateBoundSource.NATIVE_REFINED
            or refine_external_constraints
        ):
            raise ValueError(
                "optimized queue per-child refinement semantics/provenance differ"
            )
        external_seeded = (
            per_child_refinement_strategy == "external_seeded_ancestral_carry_v1"
        )
        if external_seeded != (external_constraint_seed is not None):
            raise ValueError("optimized queue external seed strategy differs")
        if external_constraint_seed is not None:
            if not isinstance(
                external_constraint_seed, NativeExternalIntermediateConstraintSeed
            ):
                raise TypeError("optimized queue external seed is invalid")
            external_constraint_seed.validate(legacy_task_module, input_spec)
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
    objective_branch_executions: list[tuple[str, NativeObjectiveBranchExecution]] = []
    per_child_refinement_executions: list[
        tuple[str, NativeIntermediateRefinementExecution]
    ] = []
    per_child_refinement_records: list[NativePerChildRefinementTrace] = []
    batch_serial = 0
    next_node_serial = 1

    def evaluate(nodes: Sequence[_RuntimeNode]) -> None:
        nonlocal batch_serial
        for start in range(0, len(nodes), config.max_eval_batch_size):
            chunk = tuple(nodes[start : start + config.max_eval_batch_size])
            batch_id = f"{run_id}:eval:{batch_serial:04d}"
            batch_serial += 1
            (
                evaluated,
                stack,
                branch_executions,
                refinement_executions,
                refinement_records,
            ) = _evaluate_optimized_node_batch(
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
                objective_branch_policy=objective_branch_policy,
                refine_external_constraints=refine_external_constraints,
                per_child_refinement_policy=per_child_refinement_policy,
                per_child_refinement_budget_policy=(per_child_refinement_budget_policy),
                per_child_refinement_strategy=per_child_refinement_strategy,
                external_constraint_seed=external_constraint_seed,
            )
            native_stacks.append(stack)
            objective_branch_executions.extend(branch_executions)
            per_child_refinement_executions.extend(refinement_executions)
            per_child_refinement_records.extend(refinement_records)
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
        per_child_refinement_policy=per_child_refinement_policy,
        per_child_refinement_budget_policy=per_child_refinement_budget_policy,
        per_child_refinements=tuple(per_child_refinement_records),
        per_child_refinement_strategy=per_child_refinement_strategy,
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
        objective_branch_executions=tuple(objective_branch_executions),
        objective_branch_policy=objective_branch_policy,
        per_child_refinement_executions=tuple(per_child_refinement_executions),
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
    objective_branch_policy: Optional[NativeObjectiveBranchPolicy] = None,
    refine_external_constraints: bool = False,
    per_child_refinement_policy: Optional[NativeIntermediateRefinementPolicyIR] = None,
    per_child_refinement_budget_policy: Optional[
        NativeIntermediateRefinementBudgetPolicyIR
    ] = None,
    per_child_refinement_strategy: PerChildRefinementStrategy = (
        "independent_exact_split_v1"
    ),
    external_constraint_seed: Optional[NativeExternalIntermediateConstraintSeed] = None,
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
        objective_branch_policy=objective_branch_policy,
        refine_external_constraints=refine_external_constraints,
        per_child_refinement_policy=per_child_refinement_policy,
        per_child_refinement_budget_policy=per_child_refinement_budget_policy,
        per_child_refinement_strategy=per_child_refinement_strategy,
        external_constraint_seed=external_constraint_seed,
    ).trace
