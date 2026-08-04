"""Typed Plan, Task, and Schedule IR for native intermediate refinement."""

# pylint: disable=too-many-instance-attributes,too-many-boolean-expressions
# pylint: disable=too-many-locals
# pylint: disable=missing-function-docstring

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from typing import Optional, Tuple

INTERMEDIATE_REFINEMENT_PLAN_IR_SCHEMA_VERSION = (
    "boundflow.intermediate_refinement_plan_ir/v1"
)
INTERMEDIATE_REFINEMENT_TASK_IR_SCHEMA_VERSION = (
    "boundflow.intermediate_refinement_task_ir/v1"
)
INTERMEDIATE_REFINEMENT_SCHEDULE_IR_SCHEMA_VERSION = (
    "boundflow.intermediate_refinement_schedule_ir/v1"
)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class IntermediateRefinementTaskKind(Enum):
    """Closed native refinement pipeline."""

    MATERIALIZE_FORWARD = "materialize_forward"
    ENUMERATE_AMBIGUOUS = "enumerate_ambiguous"
    SELECT_TARGETS = "select_targets"
    BACKWARD_SELECTED = "backward_selected"
    INTERSECT_SELECTED = "intersect_selected"
    PROPAGATE_FORWARD = "propagate_forward"
    EMIT_REFINED = "emit_refined"


@dataclass(frozen=True)
class ExternalIntermediateConstraintSeedIR:
    """External-owned intermediate constraints admitted into native refinement."""

    seed_id: str
    provider: str
    primal_graph_hash: str
    input_bounds_hash: str
    external_intermediate_bounds_hash: str
    bound_intermediate_constraints_hash: str
    source_artifact_manifest_hash: str
    source_artifact_payload_hash: str
    source_model_hash: str
    source_property_hash: str
    source_objective_set_hash: str
    consumption: str = "sound_constraint_intersection_only"
    semantics_owner: str = "external_verifier"

    def validate(self) -> None:
        if (
            not self.seed_id
            or not self.provider
            or self.semantics_owner != "external_verifier"
            or self.consumption != "sound_constraint_intersection_only"
            or any(
                not _is_sha256(value)
                for value in (
                    self.primal_graph_hash,
                    self.input_bounds_hash,
                    self.external_intermediate_bounds_hash,
                    self.bound_intermediate_constraints_hash,
                    self.source_artifact_manifest_hash,
                    self.source_artifact_payload_hash,
                    self.source_model_hash,
                    self.source_property_hash,
                    self.source_objective_set_hash,
                )
            )
        ):
            raise ValueError("external intermediate constraint seed IR is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "seed_id": self.seed_id,
            "provider": self.provider,
            "primal_graph_hash": self.primal_graph_hash,
            "input_bounds_hash": self.input_bounds_hash,
            "external_intermediate_bounds_hash": (
                self.external_intermediate_bounds_hash
            ),
            "bound_intermediate_constraints_hash": (
                self.bound_intermediate_constraints_hash
            ),
            "source_artifact_manifest_hash": self.source_artifact_manifest_hash,
            "source_artifact_payload_hash": self.source_artifact_payload_hash,
            "source_model_hash": self.source_model_hash,
            "source_property_hash": self.source_property_hash,
            "source_objective_set_hash": self.source_objective_set_hash,
            "consumption": self.consumption,
            "semantics_owner": self.semantics_owner,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeIntermediateRefinementPolicyIR:
    """Bounded-cost policy for selecting and refining ReLU inputs."""

    passes: int
    max_neurons_per_relu: int
    backward_chunk_size: int
    minimum_width: float = 0.0
    candidate_policy_id: str = "top_ambiguous_width_per_relu_v1"
    refinement_method: str = "selected_plain_crown_v1"

    def validate(self) -> None:
        if (
            self.passes < 1
            or self.passes > 8
            or self.max_neurons_per_relu < 1
            or self.backward_chunk_size < 1
            or self.backward_chunk_size > self.max_neurons_per_relu
            or not math.isfinite(self.minimum_width)
            or self.minimum_width < 0.0
            or self.candidate_policy_id
            not in {
                "top_ambiguous_width_per_relu_v1",
                "objective_influence_width_per_relu_v1",
            }
            or self.refinement_method != "selected_plain_crown_v1"
        ):
            raise ValueError("native intermediate refinement policy IR is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "passes": self.passes,
            "max_neurons_per_relu": self.max_neurons_per_relu,
            "backward_chunk_size": self.backward_chunk_size,
            "minimum_width": self.minimum_width,
            "candidate_policy_id": self.candidate_policy_id,
            "refinement_method": self.refinement_method,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeIntermediateRefinementBudgetPolicyIR:
    """Conserved dynamic target-cap allocation across one evaluation group."""

    base_max_neurons_per_relu: int
    high_max_neurons_per_relu: int
    low_max_neurons_per_relu: int
    parent_lower_tie_tolerance: float = 1e-6
    allocation_mode: str = "parent_lower_generated_batch_v1"
    conservation: str = "exact_group_target_cap_sum"
    semantics_owner: str = "boundflow_native_refinement_budget"

    def validate(self) -> None:
        if (
            self.allocation_mode != "parent_lower_generated_batch_v1"
            or self.conservation != "exact_group_target_cap_sum"
            or self.semantics_owner != "boundflow_native_refinement_budget"
            or self.base_max_neurons_per_relu < 1
            or self.low_max_neurons_per_relu < 1
            or self.high_max_neurons_per_relu <= self.base_max_neurons_per_relu
            or self.low_max_neurons_per_relu >= self.base_max_neurons_per_relu
            or self.high_max_neurons_per_relu + self.low_max_neurons_per_relu
            != 2 * self.base_max_neurons_per_relu
            or not math.isfinite(self.parent_lower_tie_tolerance)
            or self.parent_lower_tie_tolerance < 0.0
        ):
            raise ValueError("native refinement budget policy IR is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "base_max_neurons_per_relu": self.base_max_neurons_per_relu,
            "high_max_neurons_per_relu": self.high_max_neurons_per_relu,
            "low_max_neurons_per_relu": self.low_max_neurons_per_relu,
            "parent_lower_tie_tolerance": self.parent_lower_tie_tolerance,
            "allocation_mode": self.allocation_mode,
            "conservation": self.conservation,
            "semantics_owner": self.semantics_owner,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeIntermediateRefinementBudgetDecisionIR:
    """One node's target cap bound to its conserved evaluation group."""

    decision_id: str
    budget_policy_hash: str
    group_id: str
    group_semantic_hash: str
    group_size: int
    group_base_cap_total: int
    group_assigned_cap_total: int
    node_id: str
    node_split_state_hash: str
    node_depth: int
    assigned_max_neurons_per_relu: int
    allocation_rank: str
    parent_node_id: Optional[str] = None
    parent_lower: Optional[float] = None
    semantics_owner: str = "boundflow_native_refinement_budget"

    def validate(self, *, policy: NativeIntermediateRefinementBudgetPolicyIR) -> None:
        policy.validate()
        parent_present = self.parent_node_id is not None
        if (
            not self.decision_id
            or self.budget_policy_hash != policy.stable_hash()
            or not self.group_id
            or not _is_sha256(self.group_semantic_hash)
            or self.group_size < 1
            or self.group_base_cap_total
            != self.group_size * policy.base_max_neurons_per_relu
            or self.group_assigned_cap_total != self.group_base_cap_total
            or not self.node_id
            or not _is_sha256(self.node_split_state_hash)
            or self.node_depth < 0
            or self.assigned_max_neurons_per_relu
            not in {
                policy.low_max_neurons_per_relu,
                policy.base_max_neurons_per_relu,
                policy.high_max_neurons_per_relu,
            }
            or self.allocation_rank not in {"root", "base", "high_risk", "low_risk"}
            or self.semantics_owner != policy.semantics_owner
            or parent_present != (self.parent_lower is not None)
            or (self.parent_lower is not None and not math.isfinite(self.parent_lower))
            or (self.node_depth == 0) != (not parent_present)
            or (self.node_depth == 0) != (self.allocation_rank == "root")
            or (
                self.allocation_rank in {"root", "base"}
                and self.assigned_max_neurons_per_relu
                != policy.base_max_neurons_per_relu
            )
            or (
                self.allocation_rank == "high_risk"
                and self.assigned_max_neurons_per_relu
                != policy.high_max_neurons_per_relu
            )
            or (
                self.allocation_rank == "low_risk"
                and self.assigned_max_neurons_per_relu
                != policy.low_max_neurons_per_relu
            )
        ):
            raise ValueError("native refinement budget decision IR is invalid")

    def to_dict(
        self, *, policy: NativeIntermediateRefinementBudgetPolicyIR
    ) -> dict[str, object]:
        self.validate(policy=policy)
        payload: dict[str, object] = {
            "decision_id": self.decision_id,
            "budget_policy_hash": self.budget_policy_hash,
            "group_id": self.group_id,
            "group_semantic_hash": self.group_semantic_hash,
            "group_size": self.group_size,
            "group_base_cap_total": self.group_base_cap_total,
            "group_assigned_cap_total": self.group_assigned_cap_total,
            "node_id": self.node_id,
            "node_split_state_hash": self.node_split_state_hash,
            "node_depth": self.node_depth,
            "assigned_max_neurons_per_relu": (self.assigned_max_neurons_per_relu),
            "allocation_rank": self.allocation_rank,
            "semantics_owner": self.semantics_owner,
        }
        if self.parent_node_id is not None:
            payload["parent_node_id"] = self.parent_node_id
            payload["parent_lower"] = self.parent_lower
        return payload

    def stable_hash(self, *, policy: NativeIntermediateRefinementBudgetPolicyIR) -> str:
        return _canonical_hash(self.to_dict(policy=policy))


@dataclass(frozen=True)
class NativeIntermediateRefinementTargetIR:
    """One selected unstable pre-activation neuron."""

    ordinal: int
    relu_input: str
    neuron_index: int
    initial_lower: float
    initial_upper: float
    initial_width: float
    objective_influence: Optional[float] = None
    selection_score: Optional[float] = None

    def validate(self) -> None:
        objective_influence = self.objective_influence
        selection_score = self.selection_score
        if (
            self.ordinal < 0
            or not self.relu_input
            or self.neuron_index < 0
            or not all(
                math.isfinite(value)
                for value in (
                    self.initial_lower,
                    self.initial_upper,
                    self.initial_width,
                )
            )
            or not self.initial_lower < 0.0 < self.initial_upper
            or self.initial_width <= 0.0
            or abs(self.initial_width - (self.initial_upper - self.initial_lower))
            > 1e-5
            or (objective_influence is None) != (selection_score is None)
            or (
                objective_influence is not None
                and selection_score is not None
                and (
                    not math.isfinite(objective_influence)
                    or objective_influence < 0.0
                    or not math.isfinite(selection_score)
                    or selection_score < 0.0
                    or abs(selection_score - objective_influence * self.initial_width)
                    > max(1e-7, 1e-6 * selection_score)
                )
            )
        ):
            raise ValueError("native intermediate refinement target IR is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        payload: dict[str, object] = {
            "ordinal": self.ordinal,
            "relu_input": self.relu_input,
            "neuron_index": self.neuron_index,
            "initial_lower": self.initial_lower,
            "initial_upper": self.initial_upper,
            "initial_width": self.initial_width,
        }
        if self.objective_influence is not None:
            payload["objective_influence"] = self.objective_influence
            payload["selection_score"] = self.selection_score
        return payload


@dataclass(frozen=True)
class NativeIntermediateRefinementPlanIR:
    """Exact source state, target set, and method for native refinement."""

    plan_id: str
    primal_graph_hash: str
    input_bounds_hash: str
    split_state_hash: str
    initial_intermediate_bounds_hash: str
    policy: NativeIntermediateRefinementPolicyIR
    targets: Tuple[NativeIntermediateRefinementTargetIR, ...]
    objective_hash: Optional[str] = None
    source_intermediate_constraints_hash: Optional[str] = None
    source_refinement_plan_hash: Optional[str] = None
    source_refinement_semantic_trace_hash: Optional[str] = None
    external_constraint_seed: Optional[ExternalIntermediateConstraintSeedIR] = None
    schema_version: str = INTERMEDIATE_REFINEMENT_PLAN_IR_SCHEMA_VERSION

    def validate(self) -> None:
        identities = tuple(
            (target.relu_input, target.neuron_index) for target in self.targets
        )
        if (
            self.schema_version != INTERMEDIATE_REFINEMENT_PLAN_IR_SCHEMA_VERSION
            or not self.plan_id
            or any(
                not _is_sha256(value)
                for value in (
                    self.primal_graph_hash,
                    self.input_bounds_hash,
                    self.split_state_hash,
                    self.initial_intermediate_bounds_hash,
                )
            )
            or not self.targets
            or len(identities) != len(set(identities))
        ):
            raise ValueError("native intermediate refinement Plan IR is invalid")
        source_hashes = (
            self.source_intermediate_constraints_hash,
            self.source_refinement_plan_hash,
            self.source_refinement_semantic_trace_hash,
        )
        if any(value is not None for value in source_hashes) and (
            any(not _is_sha256(value) for value in source_hashes)
        ):
            raise ValueError("native refinement source constraint identity differs")
        if self.external_constraint_seed is not None:
            self.external_constraint_seed.validate()
            if (
                any(value is not None for value in source_hashes)
                or self.external_constraint_seed.primal_graph_hash
                != self.primal_graph_hash
                or self.external_constraint_seed.input_bounds_hash
                != self.input_bounds_hash
            ):
                raise ValueError("native refinement external seed identity differs")
        self.policy.validate()
        objective_directed = (
            self.policy.candidate_policy_id == "objective_influence_width_per_relu_v1"
        )
        if objective_directed != (self.objective_hash is not None) or (
            self.objective_hash is not None and not _is_sha256(self.objective_hash)
        ):
            raise ValueError("native refinement objective identity differs")
        per_relu: dict[str, int] = {}
        for ordinal, target in enumerate(self.targets):
            target.validate()
            if target.ordinal != ordinal:
                raise ValueError("native refinement target ordinal differs")
            if objective_directed != (target.objective_influence is not None):
                raise ValueError("native refinement target scoring semantics differ")
            per_relu[target.relu_input] = per_relu.get(target.relu_input, 0) + 1
        if any(count > self.policy.max_neurons_per_relu for count in per_relu.values()):
            raise ValueError("native refinement target count exceeds policy")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "primal_graph_hash": self.primal_graph_hash,
            "input_bounds_hash": self.input_bounds_hash,
            "split_state_hash": self.split_state_hash,
            "initial_intermediate_bounds_hash": self.initial_intermediate_bounds_hash,
            "policy": self.policy.to_dict(),
            "targets": [target.to_dict() for target in self.targets],
            "semantics_owner": "boundflow_native_intermediate_refinement",
            "performance_claimed": False,
        }
        if self.objective_hash is not None:
            payload["objective_hash"] = self.objective_hash
        if self.source_intermediate_constraints_hash is not None:
            payload["source_intermediate_constraints_hash"] = (
                self.source_intermediate_constraints_hash
            )
            payload["source_refinement_plan_hash"] = self.source_refinement_plan_hash
            payload["source_refinement_semantic_trace_hash"] = (
                self.source_refinement_semantic_trace_hash
            )
        if self.external_constraint_seed is not None:
            payload["external_constraint_seed"] = (
                self.external_constraint_seed.to_dict()
            )
            payload["external_constraint_seed_hash"] = (
                self.external_constraint_seed.stable_hash()
            )
        return payload

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeIntermediateRefinementTaskIRUnit:
    """One typed refinement task and its exact data dependencies."""

    task_id: str
    kind: IntermediateRefinementTaskKind
    pass_index: Optional[int]
    input_value_ids: Tuple[str, ...]
    output_value_ids: Tuple[str, ...]
    dependency_task_ids: Tuple[str, ...]
    semantics_owner: str = "boundflow_native_intermediate_refinement"

    def validate(self) -> None:
        pass_kind = self.kind in {
            IntermediateRefinementTaskKind.BACKWARD_SELECTED,
            IntermediateRefinementTaskKind.INTERSECT_SELECTED,
            IntermediateRefinementTaskKind.PROPAGATE_FORWARD,
        }
        if (
            not self.task_id
            or pass_kind != (self.pass_index is not None)
            or (self.pass_index is not None and self.pass_index < 0)
            or not self.input_value_ids
            or not self.output_value_ids
            or len(self.input_value_ids) != len(set(self.input_value_ids))
            or len(self.output_value_ids) != len(set(self.output_value_ids))
            or len(self.dependency_task_ids) != len(set(self.dependency_task_ids))
            or self.semantics_owner != "boundflow_native_intermediate_refinement"
        ):
            raise ValueError("native intermediate refinement Task IR unit is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "task_id": self.task_id,
            "kind": self.kind.value,
            "pass_index": self.pass_index,
            "input_value_ids": list(self.input_value_ids),
            "output_value_ids": list(self.output_value_ids),
            "dependency_task_ids": list(self.dependency_task_ids),
            "semantics_owner": self.semantics_owner,
        }


@dataclass(frozen=True)
class NativeIntermediateRefinementTaskIRModule:
    """Task IR lowered from one refinement Plan."""

    module_id: str
    refinement_plan_hash: str
    tasks: Tuple[NativeIntermediateRefinementTaskIRUnit, ...]
    output_task_id: str
    schema_version: str = INTERMEDIATE_REFINEMENT_TASK_IR_SCHEMA_VERSION

    def validate(self, *, plan: NativeIntermediateRefinementPlanIR) -> None:
        plan.validate()
        expected_kinds = [
            IntermediateRefinementTaskKind.MATERIALIZE_FORWARD,
            IntermediateRefinementTaskKind.ENUMERATE_AMBIGUOUS,
            IntermediateRefinementTaskKind.SELECT_TARGETS,
        ]
        for _unused in range(plan.policy.passes):
            expected_kinds.extend(
                (
                    IntermediateRefinementTaskKind.BACKWARD_SELECTED,
                    IntermediateRefinementTaskKind.INTERSECT_SELECTED,
                    IntermediateRefinementTaskKind.PROPAGATE_FORWARD,
                )
            )
        expected_kinds.append(IntermediateRefinementTaskKind.EMIT_REFINED)
        if (
            self.schema_version != INTERMEDIATE_REFINEMENT_TASK_IR_SCHEMA_VERSION
            or not self.module_id
            or self.refinement_plan_hash != plan.stable_hash()
            or tuple(task.kind for task in self.tasks) != tuple(expected_kinds)
            or not self.output_task_id
            or self.tasks[-1].task_id != self.output_task_id
        ):
            raise ValueError("native intermediate refinement Task IR is invalid")
        completed: set[str] = set()
        available = {
            "refine.module",
            "refine.input",
            "refine.split_state",
            "refine.policy",
        }
        if plan.source_intermediate_constraints_hash is not None:
            available.add("refine.source_intermediate_constraints")
        if plan.external_constraint_seed is not None:
            available.add("refine.external_constraint_seed")
        if plan.objective_hash is not None:
            available.add("refine.objective_influence")
        for task in self.tasks:
            task.validate()
            if any(item not in completed for item in task.dependency_task_ids):
                raise ValueError("native refinement task dependency is absent or late")
            if any(item not in available for item in task.input_value_ids):
                raise ValueError("native refinement task input is absent or late")
            if any(item in available for item in task.output_value_ids):
                raise ValueError("native refinement task output redefines a value")
            completed.add(task.task_id)
            available.update(task.output_value_ids)

    def to_dict(self, *, plan: NativeIntermediateRefinementPlanIR) -> dict[str, object]:
        self.validate(plan=plan)
        return {
            "schema_version": self.schema_version,
            "module_id": self.module_id,
            "refinement_plan_hash": self.refinement_plan_hash,
            "tasks": [task.to_dict() for task in self.tasks],
            "output_task_id": self.output_task_id,
        }

    def stable_hash(self, *, plan: NativeIntermediateRefinementPlanIR) -> str:
        return _canonical_hash(self.to_dict(plan=plan))


@dataclass(frozen=True)
class NativeIntermediateRefinementScheduleAction:
    """One exact synchronous Task launch."""

    action_id: str
    sequence: int
    task_id: str
    input_value_ids: Tuple[str, ...]
    output_value_ids: Tuple[str, ...]

    def validate(self) -> None:
        if (
            not self.action_id
            or self.sequence < 0
            or not self.task_id
            or not self.input_value_ids
            or not self.output_value_ids
        ):
            raise ValueError(
                "native intermediate refinement Schedule action is invalid"
            )

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "action_id": self.action_id,
            "sequence": self.sequence,
            "task_id": self.task_id,
            "input_value_ids": list(self.input_value_ids),
            "output_value_ids": list(self.output_value_ids),
        }


@dataclass(frozen=True)
class NativeIntermediateRefinementScheduleIR:
    """Exact action order for native intermediate refinement."""

    schedule_id: str
    refinement_plan_hash: str
    refinement_task_module_hash: str
    actions: Tuple[NativeIntermediateRefinementScheduleAction, ...]
    refined_bounds_value_id: str
    schema_version: str = INTERMEDIATE_REFINEMENT_SCHEDULE_IR_SCHEMA_VERSION

    def validate(
        self,
        *,
        plan: NativeIntermediateRefinementPlanIR,
        task_module: NativeIntermediateRefinementTaskIRModule,
    ) -> None:
        task_module.validate(plan=plan)
        if (
            self.schema_version != INTERMEDIATE_REFINEMENT_SCHEDULE_IR_SCHEMA_VERSION
            or not self.schedule_id
            or self.refinement_plan_hash != plan.stable_hash()
            or self.refinement_task_module_hash != task_module.stable_hash(plan=plan)
            or len(self.actions) != len(task_module.tasks)
            or not self.refined_bounds_value_id
        ):
            raise ValueError("native intermediate refinement Schedule IR is invalid")
        for sequence, (action, task) in enumerate(zip(self.actions, task_module.tasks)):
            action.validate()
            if (
                action.sequence != sequence
                or action.task_id != task.task_id
                or action.input_value_ids != task.input_value_ids
                or action.output_value_ids != task.output_value_ids
            ):
                raise ValueError("native refinement Schedule/Task linkage differs")
        if self.refined_bounds_value_id not in self.actions[-1].output_value_ids:
            raise ValueError("native refinement Schedule does not emit bounds")

    def to_dict(
        self,
        *,
        plan: NativeIntermediateRefinementPlanIR,
        task_module: NativeIntermediateRefinementTaskIRModule,
    ) -> dict[str, object]:
        self.validate(plan=plan, task_module=task_module)
        return {
            "schema_version": self.schema_version,
            "schedule_id": self.schedule_id,
            "refinement_plan_hash": self.refinement_plan_hash,
            "refinement_task_module_hash": self.refinement_task_module_hash,
            "actions": [action.to_dict() for action in self.actions],
            "refined_bounds_value_id": self.refined_bounds_value_id,
        }

    def stable_hash(
        self,
        *,
        plan: NativeIntermediateRefinementPlanIR,
        task_module: NativeIntermediateRefinementTaskIRModule,
    ) -> str:
        return _canonical_hash(self.to_dict(plan=plan, task_module=task_module))


def lower_native_intermediate_refinement_ir(
    plan: NativeIntermediateRefinementPlanIR,
) -> tuple[
    NativeIntermediateRefinementTaskIRModule,
    NativeIntermediateRefinementScheduleIR,
]:
    """Deterministically lower a refinement Plan to Task and Schedule IR."""

    plan.validate()
    definitions: list[
        tuple[
            IntermediateRefinementTaskKind,
            Optional[int],
            Tuple[str, ...],
            Tuple[str, ...],
        ]
    ] = [
        (
            IntermediateRefinementTaskKind.MATERIALIZE_FORWARD,
            None,
            (
                "refine.module",
                "refine.input",
                "refine.split_state",
                *(
                    ("refine.source_intermediate_constraints",)
                    if plan.source_intermediate_constraints_hash is not None
                    else ()
                ),
                *(
                    ("refine.external_constraint_seed",)
                    if plan.external_constraint_seed is not None
                    else ()
                ),
            ),
            ("refine.forward_env.p0", "refine.bounds.p0"),
        ),
        (
            IntermediateRefinementTaskKind.ENUMERATE_AMBIGUOUS,
            None,
            ("refine.bounds.p0",),
            ("refine.candidates",),
        ),
        (
            IntermediateRefinementTaskKind.SELECT_TARGETS,
            None,
            (
                *(
                    (
                        "refine.bounds.p0",
                        "refine.candidates",
                        "refine.policy",
                        "refine.objective_influence",
                    )
                    if plan.objective_hash is not None
                    else ("refine.candidates", "refine.policy")
                ),
            ),
            ("refine.selected_targets",),
        ),
    ]
    for pass_index in range(plan.policy.passes):
        current = pass_index
        next_index = pass_index + 1
        definitions.extend(
            (
                (
                    IntermediateRefinementTaskKind.BACKWARD_SELECTED,
                    pass_index,
                    (
                        f"refine.forward_env.p{current}",
                        f"refine.bounds.p{current}",
                        "refine.selected_targets",
                    ),
                    (f"refine.crown_candidates.p{next_index}",),
                ),
                (
                    IntermediateRefinementTaskKind.INTERSECT_SELECTED,
                    pass_index,
                    (
                        f"refine.bounds.p{current}",
                        f"refine.crown_candidates.p{next_index}",
                    ),
                    (f"refine.intersected_bounds.p{next_index}",),
                ),
                (
                    IntermediateRefinementTaskKind.PROPAGATE_FORWARD,
                    pass_index,
                    (
                        "refine.module",
                        "refine.input",
                        "refine.split_state",
                        f"refine.intersected_bounds.p{next_index}",
                    ),
                    (
                        f"refine.forward_env.p{next_index}",
                        f"refine.bounds.p{next_index}",
                    ),
                ),
            )
        )
    definitions.append(
        (
            IntermediateRefinementTaskKind.EMIT_REFINED,
            None,
            (f"refine.bounds.p{plan.policy.passes}",),
            ("refine.refined_bounds",),
        )
    )
    tasks: list[NativeIntermediateRefinementTaskIRUnit] = []
    previous: Tuple[str, ...] = ()
    for sequence, (kind, definition_pass_index, inputs, outputs) in enumerate(
        definitions
    ):
        task_id = f"intermediate_refinement.{sequence:04d}.{kind.value}"
        tasks.append(
            NativeIntermediateRefinementTaskIRUnit(
                task_id=task_id,
                kind=kind,
                pass_index=definition_pass_index,
                input_value_ids=inputs,
                output_value_ids=outputs,
                dependency_task_ids=previous,
            )
        )
        previous = (task_id,)
    task_module = NativeIntermediateRefinementTaskIRModule(
        module_id=f"{plan.plan_id}.tasks",
        refinement_plan_hash=plan.stable_hash(),
        tasks=tuple(tasks),
        output_task_id=tasks[-1].task_id,
    )
    actions = tuple(
        NativeIntermediateRefinementScheduleAction(
            action_id=f"launch.{sequence:04d}.{task.kind.value}",
            sequence=sequence,
            task_id=task.task_id,
            input_value_ids=task.input_value_ids,
            output_value_ids=task.output_value_ids,
        )
        for sequence, task in enumerate(tasks)
    )
    schedule = NativeIntermediateRefinementScheduleIR(
        schedule_id=f"{plan.plan_id}.schedule",
        refinement_plan_hash=plan.stable_hash(),
        refinement_task_module_hash=task_module.stable_hash(plan=plan),
        actions=actions,
        refined_bounds_value_id="refine.refined_bounds",
    )
    schedule.validate(plan=plan, task_module=task_module)
    return task_module, schedule


__all__ = [
    "INTERMEDIATE_REFINEMENT_PLAN_IR_SCHEMA_VERSION",
    "INTERMEDIATE_REFINEMENT_SCHEDULE_IR_SCHEMA_VERSION",
    "INTERMEDIATE_REFINEMENT_TASK_IR_SCHEMA_VERSION",
    "ExternalIntermediateConstraintSeedIR",
    "IntermediateRefinementTaskKind",
    "NativeIntermediateRefinementPlanIR",
    "NativeIntermediateRefinementPolicyIR",
    "NativeIntermediateRefinementScheduleAction",
    "NativeIntermediateRefinementScheduleIR",
    "NativeIntermediateRefinementTargetIR",
    "NativeIntermediateRefinementTaskIRModule",
    "NativeIntermediateRefinementTaskIRUnit",
    "lower_native_intermediate_refinement_ir",
]
