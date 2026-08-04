"""Typed Plan, Task, and Schedule IR for production verifier node batches."""

# pylint: disable=missing-function-docstring,too-many-instance-attributes
# pylint: disable=too-many-boolean-expressions,duplicate-code

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import Optional, Tuple

PRODUCTION_VERIFIER_PLAN_IR_SCHEMA_VERSION = "boundflow.production-verifier-plan-ir/v1"
PRODUCTION_VERIFIER_TASK_IR_SCHEMA_VERSION = "boundflow.production-verifier-task-ir/v1"
PRODUCTION_VERIFIER_SCHEDULE_IR_SCHEMA_VERSION = (
    "boundflow.production-verifier-schedule-ir/v1"
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


class NativeProductionVerifierTaskKind(Enum):
    """Closed production node-batch execution phases."""

    VALIDATE_PROGRAM = "validate_program"
    EXECUTE_OPTIMIZER = "execute_optimizer"
    MATERIALIZE_NODE_RESULTS = "materialize_node_results"
    COMMIT_QUEUE_RESULTS = "commit_queue_results"


@dataclass(frozen=True)
class NativeProductionVerifierPlanIR:
    """One exact dynamic node batch bound to a prevalidated optimizer program."""

    plan_id: str
    node_ids: Tuple[str, ...]
    node_split_state_hashes: Tuple[str, ...]
    parent_selected_state_hashes: Tuple[Optional[str], ...]
    state_scope_hash: str
    primal_graph_hash: str
    input_region_hash: str
    objective_hash: str
    optimizer_policy_hash: str
    intermediate_bounds_hash: str
    intermediate_bound_source: str
    optimizer_ir_hashes: Tuple[Tuple[str, str], ...]
    execution_mode: str = "production_prepared_no_audit_reexecution"
    schema_version: str = PRODUCTION_VERIFIER_PLAN_IR_SCHEMA_VERSION

    def validate(self) -> None:
        optimizer_hashes = dict(self.optimizer_ir_hashes)
        expected_optimizer_keys = {
            "optimizer_plan_hash",
            "optimizer_task_module_hash",
            "optimizer_schedule_hash",
        }
        if (
            self.schema_version != PRODUCTION_VERIFIER_PLAN_IR_SCHEMA_VERSION
            or not self.plan_id
            or not self.node_ids
            or len(self.node_ids) != len(set(self.node_ids))
            or len(self.node_split_state_hashes) != len(self.node_ids)
            or len(self.parent_selected_state_hashes) != len(self.node_ids)
            or any(not _is_sha256(value) for value in self.node_split_state_hashes)
            or any(
                value is not None and not _is_sha256(value)
                for value in self.parent_selected_state_hashes
            )
            or any(
                not _is_sha256(value)
                for value in (
                    self.state_scope_hash,
                    self.primal_graph_hash,
                    self.input_region_hash,
                    self.objective_hash,
                    self.optimizer_policy_hash,
                    self.intermediate_bounds_hash,
                )
            )
            or self.intermediate_bound_source
            not in {"local_forward", "external_verifier", "native_refined"}
            or set(optimizer_hashes) != expected_optimizer_keys
            or len(optimizer_hashes) != len(self.optimizer_ir_hashes)
            or any(not _is_sha256(value) for value in optimizer_hashes.values())
            or self.execution_mode != "production_prepared_no_audit_reexecution"
        ):
            raise ValueError("production verifier Plan IR is invalid")
        root_batch = all(value is None for value in self.parent_selected_state_hashes)
        child_batch = all(
            value is not None for value in self.parent_selected_state_hashes
        )
        if not (root_batch or child_batch):
            raise ValueError("production verifier Plan IR mixes root and child nodes")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "node_ids": list(self.node_ids),
            "node_split_state_hashes": list(self.node_split_state_hashes),
            "parent_selected_state_hashes": list(self.parent_selected_state_hashes),
            "state_scope_hash": self.state_scope_hash,
            "primal_graph_hash": self.primal_graph_hash,
            "input_region_hash": self.input_region_hash,
            "objective_hash": self.objective_hash,
            "optimizer_policy_hash": self.optimizer_policy_hash,
            "intermediate_bounds_hash": self.intermediate_bounds_hash,
            "intermediate_bound_source": self.intermediate_bound_source,
            "optimizer_ir_hashes": dict(self.optimizer_ir_hashes),
            "execution_mode": self.execution_mode,
            "audit_hash_chain_constructed": False,
            "selected_native_reexecution": False,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeProductionVerifierTaskIRUnit:
    """One production phase and its exact SSA-like dependencies."""

    task_id: str
    kind: NativeProductionVerifierTaskKind
    dependency_task_ids: Tuple[str, ...]
    input_value_ids: Tuple[str, ...]
    output_value_ids: Tuple[str, ...]

    def validate(self) -> None:
        if (
            not self.task_id
            or len(self.dependency_task_ids) != len(set(self.dependency_task_ids))
            or not self.input_value_ids
            or len(self.input_value_ids) != len(set(self.input_value_ids))
            or not self.output_value_ids
            or len(self.output_value_ids) != len(set(self.output_value_ids))
        ):
            raise ValueError("production verifier Task IR unit is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "task_id": self.task_id,
            "kind": self.kind.value,
            "dependency_task_ids": list(self.dependency_task_ids),
            "input_value_ids": list(self.input_value_ids),
            "output_value_ids": list(self.output_value_ids),
        }


@dataclass(frozen=True)
class NativeProductionVerifierTaskIR:
    """Production tasks lowered from one node-batch Plan IR."""

    plan_hash: str
    tasks: Tuple[NativeProductionVerifierTaskIRUnit, ...]
    schema_version: str = PRODUCTION_VERIFIER_TASK_IR_SCHEMA_VERSION

    def validate(self) -> None:
        task_ids = tuple(task.task_id for task in self.tasks)
        if (
            self.schema_version != PRODUCTION_VERIFIER_TASK_IR_SCHEMA_VERSION
            or not _is_sha256(self.plan_hash)
            or len(self.tasks) != 4
            or len(task_ids) != len(set(task_ids))
            or tuple(task.kind for task in self.tasks)
            != tuple(NativeProductionVerifierTaskKind)
        ):
            raise ValueError("production verifier Task IR is invalid")
        available_tasks: set[str] = set()
        available_values = {"optimizer.program", "node.batch", "queue.state"}
        for task in self.tasks:
            task.validate()
            if any(
                dependency not in available_tasks
                for dependency in task.dependency_task_ids
            ) or any(value not in available_values for value in task.input_value_ids):
                raise ValueError("production verifier Task IR dependency differs")
            if any(value in available_values for value in task.output_value_ids):
                raise ValueError("production verifier Task IR redefines a value")
            available_tasks.add(task.task_id)
            available_values.update(task.output_value_ids)

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "plan_hash": self.plan_hash,
            "tasks": [task.to_dict() for task in self.tasks],
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeProductionVerifierScheduleAction:
    """One deterministic production runtime dispatch."""

    sequence: int
    action_id: str
    task_id: str
    kind: NativeProductionVerifierTaskKind

    def validate(self) -> None:
        if self.sequence < 0 or not self.action_id or not self.task_id:
            raise ValueError("production verifier Schedule action is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "sequence": self.sequence,
            "action_id": self.action_id,
            "task_id": self.task_id,
            "kind": self.kind.value,
        }


@dataclass(frozen=True)
class NativeProductionVerifierScheduleIR:
    """Exact production phase order for one node batch."""

    plan_hash: str
    task_ir_hash: str
    actions: Tuple[NativeProductionVerifierScheduleAction, ...]
    schema_version: str = PRODUCTION_VERIFIER_SCHEDULE_IR_SCHEMA_VERSION

    def validate(
        self,
        *,
        plan: NativeProductionVerifierPlanIR,
        task_ir: NativeProductionVerifierTaskIR,
    ) -> None:
        plan.validate()
        task_ir.validate()
        if (
            self.schema_version != PRODUCTION_VERIFIER_SCHEDULE_IR_SCHEMA_VERSION
            or self.plan_hash != plan.stable_hash()
            or task_ir.plan_hash != self.plan_hash
            or self.task_ir_hash != task_ir.stable_hash()
            or len(self.actions) != len(task_ir.tasks)
        ):
            raise ValueError("production verifier Schedule IR is invalid")
        for sequence, (action, task) in enumerate(zip(self.actions, task_ir.tasks)):
            action.validate()
            if (
                action.sequence != sequence
                or action.task_id != task.task_id
                or action.kind != task.kind
            ):
                raise ValueError("production verifier Schedule/Task order differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "plan_hash": self.plan_hash,
            "task_ir_hash": self.task_ir_hash,
            "actions": [action.to_dict() for action in self.actions],
            "dispatch": "sequential_fail_closed",
        }

    def stable_hash(
        self,
        *,
        plan: NativeProductionVerifierPlanIR,
        task_ir: NativeProductionVerifierTaskIR,
    ) -> str:
        self.validate(plan=plan, task_ir=task_ir)
        return _canonical_hash(self.to_dict())


def lower_native_production_verifier_ir(
    plan: NativeProductionVerifierPlanIR,
) -> tuple[NativeProductionVerifierTaskIR, NativeProductionVerifierScheduleIR]:
    """Lower one exact node-batch Plan into its production control program."""

    plan.validate()
    prefix = plan.plan_id
    task_specs = (
        (
            NativeProductionVerifierTaskKind.VALIDATE_PROGRAM,
            (),
            ("optimizer.program", "node.batch"),
            ("prepared.program",),
        ),
        (
            NativeProductionVerifierTaskKind.EXECUTE_OPTIMIZER,
            (f"{prefix}:validate-program",),
            ("prepared.program",),
            ("optimizer.result",),
        ),
        (
            NativeProductionVerifierTaskKind.MATERIALIZE_NODE_RESULTS,
            (f"{prefix}:execute-optimizer",),
            ("optimizer.result", "node.batch"),
            ("node.results",),
        ),
        (
            NativeProductionVerifierTaskKind.COMMIT_QUEUE_RESULTS,
            (f"{prefix}:materialize-node-results",),
            ("node.results", "queue.state"),
            ("queue.commit",),
        ),
    )
    tasks = tuple(
        NativeProductionVerifierTaskIRUnit(
            task_id=f"{prefix}:{kind.value.replace('_', '-')}",
            kind=kind,
            dependency_task_ids=dependencies,
            input_value_ids=inputs,
            output_value_ids=outputs,
        )
        for kind, dependencies, inputs, outputs in task_specs
    )
    task_ir = NativeProductionVerifierTaskIR(plan_hash=plan.stable_hash(), tasks=tasks)
    task_ir.validate()
    schedule = NativeProductionVerifierScheduleIR(
        plan_hash=plan.stable_hash(),
        task_ir_hash=task_ir.stable_hash(),
        actions=tuple(
            NativeProductionVerifierScheduleAction(
                sequence=index,
                action_id=f"{task.task_id}:launch",
                task_id=task.task_id,
                kind=task.kind,
            )
            for index, task in enumerate(task_ir.tasks)
        ),
    )
    schedule.validate(plan=plan, task_ir=task_ir)
    return task_ir, schedule


__all__ = [
    "NativeProductionVerifierPlanIR",
    "NativeProductionVerifierScheduleAction",
    "NativeProductionVerifierScheduleIR",
    "NativeProductionVerifierTaskIR",
    "NativeProductionVerifierTaskIRUnit",
    "NativeProductionVerifierTaskKind",
    "PRODUCTION_VERIFIER_PLAN_IR_SCHEMA_VERSION",
    "PRODUCTION_VERIFIER_SCHEDULE_IR_SCHEMA_VERSION",
    "PRODUCTION_VERIFIER_TASK_IR_SCHEMA_VERSION",
    "lower_native_production_verifier_ir",
]
