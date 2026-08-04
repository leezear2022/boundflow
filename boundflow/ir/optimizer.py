"""Typed optimizer Plan, Task, and Schedule IR for native alpha/beta state."""

# pylint: disable=too-many-instance-attributes,missing-function-docstring
# pylint: disable=too-many-boolean-expressions,too-many-branches,too-many-locals

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import Optional, Tuple

OPTIMIZER_PLAN_IR_SCHEMA_VERSION = "boundflow.optimizer_plan_ir/v1"
OPTIMIZER_TASK_IR_SCHEMA_VERSION = "boundflow.optimizer_task_ir/v1"
OPTIMIZER_SCHEDULE_IR_SCHEMA_VERSION = "boundflow.optimizer_schedule_ir/v1"

_SOURCE_IR_HASH_KEYS = {
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


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class OptimizerTaskKind(Enum):
    """Closed fixed-step alpha/beta optimizer task set."""

    EVALUATE_BOUND = "evaluate_bound"
    REDUCE_METRIC = "reduce_metric"
    BACKWARD = "backward"
    ADAM_UPDATE = "adam_update"
    PROJECT_STATE = "project_state"
    SELECT_BEST = "select_best"


@dataclass(frozen=True)
class NativeOptimizerPlanIR:
    """Static optimizer decision bound to one frozen native compiler stack."""

    plan_id: str
    source_ir_hashes: Tuple[Tuple[str, str], ...]
    initial_state_hash: str
    state_scope_hash: str
    optimizer_policy_hash: str
    steps: int
    relu_state_keys: Tuple[str, ...]
    warm_start_kind: str
    objective: str
    spec_reduce: str
    schema_version: str = OPTIMIZER_PLAN_IR_SCHEMA_VERSION

    def validate(self) -> None:
        hashes = dict(self.source_ir_hashes)
        if (
            self.schema_version != OPTIMIZER_PLAN_IR_SCHEMA_VERSION
            or not self.plan_id
            or set(hashes) != _SOURCE_IR_HASH_KEYS
            or len(hashes) != len(self.source_ir_hashes)
            or any(not _is_sha256(value) for value in hashes.values())
            or not _is_sha256(self.initial_state_hash)
            or not _is_sha256(self.state_scope_hash)
            or not _is_sha256(self.optimizer_policy_hash)
            or self.steps < 0
            or not self.relu_state_keys
            or len(self.relu_state_keys) != len(set(self.relu_state_keys))
            or tuple(sorted(self.relu_state_keys)) != self.relu_state_keys
            or self.warm_start_kind
            not in {"none", "exact", "monotonic_split_refinement"}
            or self.objective not in {"lower", "upper", "gap", "both"}
            or self.spec_reduce not in {"mean", "min", "softmin"}
        ):
            raise ValueError("native optimizer Plan IR is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "source_ir_hashes": dict(self.source_ir_hashes),
            "initial_state_hash": self.initial_state_hash,
            "state_scope_hash": self.state_scope_hash,
            "optimizer_policy_hash": self.optimizer_policy_hash,
            "steps": self.steps,
            "relu_state_keys": list(self.relu_state_keys),
            "warm_start_kind": self.warm_start_kind,
            "objective": self.objective,
            "spec_reduce": self.spec_reduce,
            "performance_claimed": False,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeOptimizerTaskIRUnit:
    """One explicit optimizer semantic task and its data dependencies."""

    task_id: str
    kind: OptimizerTaskKind
    iteration: Optional[int]
    input_value_ids: Tuple[str, ...]
    output_value_ids: Tuple[str, ...]
    dependency_task_ids: Tuple[str, ...]
    semantics_owner: str = "boundflow_native_alpha_beta_optimizer"

    def validate(self, *, steps: int) -> None:
        if (
            not self.task_id
            or not self.input_value_ids
            or not self.output_value_ids
            or len(self.input_value_ids) != len(set(self.input_value_ids))
            or len(self.output_value_ids) != len(set(self.output_value_ids))
            or len(self.dependency_task_ids) != len(set(self.dependency_task_ids))
            or self.semantics_owner != "boundflow_native_alpha_beta_optimizer"
        ):
            raise ValueError("native optimizer Task IR unit is invalid")
        if self.kind == OptimizerTaskKind.SELECT_BEST:
            if self.iteration is not None:
                raise ValueError("optimizer select-best cannot bind one iteration")
        elif self.iteration is None or self.iteration < 0 or self.iteration > steps:
            raise ValueError("optimizer task iteration is invalid")
        if (
            self.kind
            in {
                OptimizerTaskKind.BACKWARD,
                OptimizerTaskKind.ADAM_UPDATE,
                OptimizerTaskKind.PROJECT_STATE,
            }
            and self.iteration == steps
        ):
            raise ValueError("optimizer update task cannot follow final evaluation")

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "kind": self.kind.value,
            "iteration": self.iteration,
            "input_value_ids": list(self.input_value_ids),
            "output_value_ids": list(self.output_value_ids),
            "dependency_task_ids": list(self.dependency_task_ids),
            "semantics_owner": self.semantics_owner,
        }


@dataclass(frozen=True)
class NativeOptimizerTaskIRModule:
    """Unrolled fixed-step tasks derived from one optimizer Plan IR."""

    module_id: str
    optimizer_plan_hash: str
    tasks: Tuple[NativeOptimizerTaskIRUnit, ...]
    entry_task_ids: Tuple[str, ...]
    output_task_id: str
    schema_version: str = OPTIMIZER_TASK_IR_SCHEMA_VERSION

    def validate(self, *, plan: NativeOptimizerPlanIR) -> None:
        plan.validate()
        if (
            self.schema_version != OPTIMIZER_TASK_IR_SCHEMA_VERSION
            or not self.module_id
            or self.optimizer_plan_hash != plan.stable_hash()
            or not self.tasks
            or not self.entry_task_ids
            or not self.output_task_id
        ):
            raise ValueError("native optimizer Task IR module is invalid")
        task_by_id = {task.task_id: task for task in self.tasks}
        if len(task_by_id) != len(self.tasks):
            raise ValueError("native optimizer Task IR IDs repeat")
        if not set(self.entry_task_ids) <= set(task_by_id):
            raise ValueError("native optimizer Task IR entry is absent")
        if self.output_task_id not in task_by_id:
            raise ValueError("native optimizer Task IR output is absent")
        completed: set[str] = set()
        available = {"optimizer.state.s000"}
        for task in self.tasks:
            task.validate(steps=plan.steps)
            if any(
                dependency not in completed for dependency in task.dependency_task_ids
            ):
                raise ValueError("optimizer task dependency is absent or late")
            if any(value_id not in available for value_id in task.input_value_ids):
                raise ValueError("optimizer task input is absent or late")
            if any(value_id in available for value_id in task.output_value_ids):
                raise ValueError("optimizer task output redefines a value")
            completed.add(task.task_id)
            available.update(task.output_value_ids)
        if tuple(self.entry_task_ids) != (self.tasks[0].task_id,):
            raise ValueError("optimizer Task IR entry order differs")
        if self.tasks[-1].task_id != self.output_task_id:
            raise ValueError("optimizer Task IR output is not final")
        if tuple(task.kind for task in self.tasks) != _expected_task_kinds(plan.steps):
            raise ValueError("optimizer Task IR fixed-step sequence differs")

    def to_dict(self, *, plan: NativeOptimizerPlanIR) -> dict[str, object]:
        self.validate(plan=plan)
        return {
            "schema_version": self.schema_version,
            "module_id": self.module_id,
            "optimizer_plan_hash": self.optimizer_plan_hash,
            "tasks": [task.to_dict() for task in self.tasks],
            "entry_task_ids": list(self.entry_task_ids),
            "output_task_id": self.output_task_id,
        }

    def stable_hash(self, *, plan: NativeOptimizerPlanIR) -> str:
        return _canonical_hash(self.to_dict(plan=plan))


@dataclass(frozen=True)
class NativeOptimizerScheduleAction:
    """One synchronous launch of an exact optimizer Task IR unit."""

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
            raise ValueError("native optimizer Schedule action is invalid")

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
class NativeOptimizerScheduleIR:
    """Synchronous action order for one unrolled optimizer Task module."""

    schedule_id: str
    optimizer_plan_hash: str
    optimizer_task_module_hash: str
    actions: Tuple[NativeOptimizerScheduleAction, ...]
    selected_state_value_id: str
    schema_version: str = OPTIMIZER_SCHEDULE_IR_SCHEMA_VERSION

    def validate(
        self,
        *,
        plan: NativeOptimizerPlanIR,
        task_module: NativeOptimizerTaskIRModule,
    ) -> None:
        task_module.validate(plan=plan)
        if (
            self.schema_version != OPTIMIZER_SCHEDULE_IR_SCHEMA_VERSION
            or not self.schedule_id
            or self.optimizer_plan_hash != plan.stable_hash()
            or self.optimizer_task_module_hash != task_module.stable_hash(plan=plan)
            or len(self.actions) != len(task_module.tasks)
            or not self.selected_state_value_id
        ):
            raise ValueError("native optimizer Schedule IR is invalid")
        action_ids = {action.action_id for action in self.actions}
        if len(action_ids) != len(self.actions):
            raise ValueError("native optimizer Schedule action IDs repeat")
        for sequence, (action, task) in enumerate(zip(self.actions, task_module.tasks)):
            action.validate()
            if (
                action.sequence != sequence
                or action.task_id != task.task_id
                or action.input_value_ids != task.input_value_ids
                or action.output_value_ids != task.output_value_ids
            ):
                raise ValueError("optimizer Schedule/Task linkage differs")
        if self.selected_state_value_id not in self.actions[-1].output_value_ids:
            raise ValueError("optimizer Schedule selected state is not emitted")

    def to_dict(
        self,
        *,
        plan: NativeOptimizerPlanIR,
        task_module: NativeOptimizerTaskIRModule,
    ) -> dict[str, object]:
        self.validate(plan=plan, task_module=task_module)
        return {
            "schema_version": self.schema_version,
            "schedule_id": self.schedule_id,
            "optimizer_plan_hash": self.optimizer_plan_hash,
            "optimizer_task_module_hash": self.optimizer_task_module_hash,
            "actions": [action.to_dict() for action in self.actions],
            "selected_state_value_id": self.selected_state_value_id,
        }

    def stable_hash(
        self,
        *,
        plan: NativeOptimizerPlanIR,
        task_module: NativeOptimizerTaskIRModule,
    ) -> str:
        return _canonical_hash(self.to_dict(plan=plan, task_module=task_module))


def _expected_task_kinds(steps: int) -> Tuple[OptimizerTaskKind, ...]:
    kinds: list[OptimizerTaskKind] = []
    for iteration in range(steps + 1):
        kinds.extend(
            (OptimizerTaskKind.EVALUATE_BOUND, OptimizerTaskKind.REDUCE_METRIC)
        )
        if iteration < steps:
            kinds.extend(
                (
                    OptimizerTaskKind.BACKWARD,
                    OptimizerTaskKind.ADAM_UPDATE,
                    OptimizerTaskKind.PROJECT_STATE,
                )
            )
    kinds.append(OptimizerTaskKind.SELECT_BEST)
    return tuple(kinds)


def lower_native_optimizer_ir(
    plan: NativeOptimizerPlanIR,
) -> tuple[NativeOptimizerTaskIRModule, NativeOptimizerScheduleIR]:
    """Deterministically unroll one fixed-step Plan into Task and Schedule IR."""

    plan.validate()
    tasks: list[NativeOptimizerTaskIRUnit] = []
    state_ids: list[str] = []
    metric_ids: list[str] = []
    previous_task: Optional[str] = None

    def add(
        kind: OptimizerTaskKind,
        iteration: Optional[int],
        inputs: Tuple[str, ...],
        outputs: Tuple[str, ...],
    ) -> str:
        nonlocal previous_task
        suffix = "final" if iteration is None else f"s{iteration:03d}"
        task_id = f"optimizer.{kind.value}.{suffix}"
        tasks.append(
            NativeOptimizerTaskIRUnit(
                task_id=task_id,
                kind=kind,
                iteration=iteration,
                input_value_ids=inputs,
                output_value_ids=outputs,
                dependency_task_ids=(() if previous_task is None else (previous_task,)),
            )
        )
        previous_task = task_id
        return task_id

    current_state = "optimizer.state.s000"
    for iteration in range(plan.steps + 1):
        state_ids.append(current_state)
        bounds = f"optimizer.bounds.s{iteration:03d}"
        metric = f"optimizer.metric.s{iteration:03d}"
        add(OptimizerTaskKind.EVALUATE_BOUND, iteration, (current_state,), (bounds,))
        add(OptimizerTaskKind.REDUCE_METRIC, iteration, (bounds,), (metric,))
        metric_ids.append(metric)
        if iteration < plan.steps:
            gradient = f"optimizer.gradient.s{iteration:03d}"
            raw_state = f"optimizer.state.raw.s{iteration + 1:03d}"
            next_state = f"optimizer.state.s{iteration + 1:03d}"
            add(OptimizerTaskKind.BACKWARD, iteration, (metric,), (gradient,))
            add(
                OptimizerTaskKind.ADAM_UPDATE,
                iteration,
                (current_state, gradient),
                (raw_state,),
            )
            add(
                OptimizerTaskKind.PROJECT_STATE,
                iteration,
                (raw_state,),
                (next_state,),
            )
            current_state = next_state
    selected = "optimizer.state.selected"
    add(
        OptimizerTaskKind.SELECT_BEST,
        None,
        tuple((*state_ids, *metric_ids)),
        (selected,),
    )
    task_module = NativeOptimizerTaskIRModule(
        module_id=f"{plan.plan_id}.tasks",
        optimizer_plan_hash=plan.stable_hash(),
        tasks=tuple(tasks),
        entry_task_ids=(tasks[0].task_id,),
        output_task_id=tasks[-1].task_id,
    )
    actions = tuple(
        NativeOptimizerScheduleAction(
            action_id=f"launch.{index:04d}.{task.task_id}",
            sequence=index,
            task_id=task.task_id,
            input_value_ids=task.input_value_ids,
            output_value_ids=task.output_value_ids,
        )
        for index, task in enumerate(task_module.tasks)
    )
    schedule = NativeOptimizerScheduleIR(
        schedule_id=f"{plan.plan_id}.schedule",
        optimizer_plan_hash=plan.stable_hash(),
        optimizer_task_module_hash=task_module.stable_hash(plan=plan),
        actions=actions,
        selected_state_value_id=selected,
    )
    schedule.validate(plan=plan, task_module=task_module)
    return task_module, schedule
