"""Parametric optimizer PlanTemplate, PlanInstance, Task, and Schedule IR."""

# pylint: disable=too-many-instance-attributes,too-many-boolean-expressions
# pylint: disable=missing-function-docstring,too-many-locals

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Optional, Tuple

from .optimizer import (
    NativeOptimizerScheduleAction,
    NativeOptimizerTaskIRUnit,
    OptimizerTaskKind,
)

PARAMETRIC_OPTIMIZER_TEMPLATE_IR_SCHEMA_VERSION = (
    "boundflow.parametric-optimizer-template-ir/v1"
)
PARAMETRIC_OPTIMIZER_INSTANCE_IR_SCHEMA_VERSION = (
    "boundflow.parametric-optimizer-instance-ir/v1"
)
PARAMETRIC_OPTIMIZER_TASK_IR_SCHEMA_VERSION = (
    "boundflow.parametric-optimizer-task-ir/v1"
)
PARAMETRIC_OPTIMIZER_SCHEDULE_IR_SCHEMA_VERSION = (
    "boundflow.parametric-optimizer-schedule-ir/v1"
)
PARAMETRIC_OPTIMIZER_CACHE_EVENT_IR_SCHEMA_VERSION = (
    "boundflow.parametric-optimizer-cache-event-ir/v1"
)


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


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


@dataclass(frozen=True)
class NativeParametricOptimizerTemplateIR:
    """Static optimizer contract shared by exact dynamic batch instances."""

    template_id: str
    primal_graph_hash: str
    input_value_name: str
    input_nonbatch_shape: Tuple[int, ...]
    input_dtype: str
    input_device: str
    objective_shape: Tuple[int, ...]
    objective_dtype: str
    objective_device: str
    relu_state_layout: Tuple[Tuple[str, Tuple[int, ...], str, str], ...]
    optimizer_policy_hash: str
    steps: int
    objective: str
    spec_reduce: str
    intermediate_bound_source: str
    refine_external_constraints: bool
    semantics_owner: str = "boundflow_native_parametric_optimizer"
    schema_version: str = PARAMETRIC_OPTIMIZER_TEMPLATE_IR_SCHEMA_VERSION

    def validate(self) -> None:
        names = tuple(name for name, _shape, _dtype, _device in self.relu_state_layout)
        if (
            self.schema_version != PARAMETRIC_OPTIMIZER_TEMPLATE_IR_SCHEMA_VERSION
            or not self.template_id
            or not _is_sha256(self.primal_graph_hash)
            or not self.input_value_name
            or not self.input_nonbatch_shape
            or any(dimension < 1 for dimension in self.input_nonbatch_shape)
            or not self.input_dtype
            or not self.input_device
            or not self.objective_shape
            or any(dimension < 1 for dimension in self.objective_shape)
            or not self.objective_dtype
            or not self.objective_device
            or not self.relu_state_layout
            or names != tuple(sorted(names))
            or len(names) != len(set(names))
            or any(
                not name
                or not shape
                or any(dimension < 1 for dimension in shape)
                or not dtype
                or not device
                for name, shape, dtype, device in self.relu_state_layout
            )
            or not _is_sha256(self.optimizer_policy_hash)
            or self.steps < 0
            or self.objective not in {"lower", "upper", "gap", "both"}
            or self.spec_reduce not in {"mean", "min", "softmin"}
            or self.intermediate_bound_source
            not in {"local_forward", "external_verifier", "native_refined"}
            or not isinstance(self.refine_external_constraints, bool)
            or (
                self.refine_external_constraints
                and self.intermediate_bound_source != "external_verifier"
            )
            or self.semantics_owner != "boundflow_native_parametric_optimizer"
        ):
            raise ValueError("parametric optimizer template IR is invalid")

    def contract_dict(self) -> dict[str, object]:
        """Return the cache-key payload, excluding the diagnostic template ID."""

        self.validate()
        return {
            "schema_version": self.schema_version,
            "primal_graph_hash": self.primal_graph_hash,
            "input_value_name": self.input_value_name,
            "input_nonbatch_shape": list(self.input_nonbatch_shape),
            "input_dtype": self.input_dtype,
            "input_device": self.input_device,
            "objective_shape": list(self.objective_shape),
            "objective_dtype": self.objective_dtype,
            "objective_device": self.objective_device,
            "relu_state_layout": [
                {
                    "name": name,
                    "nonbatch_shape": list(shape),
                    "dtype": dtype,
                    "device": device,
                }
                for name, shape, dtype, device in self.relu_state_layout
            ],
            "optimizer_policy_hash": self.optimizer_policy_hash,
            "steps": self.steps,
            "objective": self.objective,
            "spec_reduce": self.spec_reduce,
            "intermediate_bound_source": self.intermediate_bound_source,
            "refine_external_constraints": self.refine_external_constraints,
            "semantics_owner": self.semantics_owner,
        }

    def cache_key(self) -> str:
        return _canonical_hash(self.contract_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            **self.contract_dict(),
            "template_id": self.template_id,
            "cache_key": self.cache_key(),
            "performance_claimed": False,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeParametricOptimizerInstanceIR:
    """Exact dynamic tensor binding for one optimizer template invocation."""

    instance_id: str
    template_hash: str
    cache_key: str
    batch_size: int
    input_region_hash: str
    objective_hash: str
    intermediate_bounds_hash: str
    split_state_hash: str
    state_scope_hash: str
    initial_state_hash: str
    warm_start_kind: str
    schema_version: str = PARAMETRIC_OPTIMIZER_INSTANCE_IR_SCHEMA_VERSION

    def validate(self) -> None:
        if (
            self.schema_version != PARAMETRIC_OPTIMIZER_INSTANCE_IR_SCHEMA_VERSION
            or not self.instance_id
            or self.batch_size < 1
            or any(
                not _is_sha256(value)
                for value in (
                    self.template_hash,
                    self.cache_key,
                    self.input_region_hash,
                    self.objective_hash,
                    self.intermediate_bounds_hash,
                    self.split_state_hash,
                    self.state_scope_hash,
                    self.initial_state_hash,
                )
            )
            or self.warm_start_kind
            not in {"none", "exact", "monotonic_split_refinement"}
        ):
            raise ValueError("parametric optimizer instance IR is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "instance_id": self.instance_id,
            "template_hash": self.template_hash,
            "cache_key": self.cache_key,
            "batch_size": self.batch_size,
            "input_region_hash": self.input_region_hash,
            "objective_hash": self.objective_hash,
            "intermediate_bounds_hash": self.intermediate_bounds_hash,
            "split_state_hash": self.split_state_hash,
            "state_scope_hash": self.state_scope_hash,
            "initial_state_hash": self.initial_state_hash,
            "warm_start_kind": self.warm_start_kind,
            "performance_claimed": False,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeParametricOptimizerCacheEventIR:
    """One fail-closed query-local template-cache decision."""

    event_index: int
    batch_id: str
    cache_key: str
    template_hash: str
    outcome: str
    compile_elapsed_ns: int
    schema_version: str = PARAMETRIC_OPTIMIZER_CACHE_EVENT_IR_SCHEMA_VERSION

    def validate(self) -> None:
        if (
            self.schema_version != PARAMETRIC_OPTIMIZER_CACHE_EVENT_IR_SCHEMA_VERSION
            or self.event_index < 0
            or not self.batch_id
            or not _is_sha256(self.cache_key)
            or not _is_sha256(self.template_hash)
            or self.outcome not in {"miss_compiled", "hit_exact_contract"}
            or self.compile_elapsed_ns < 0
            or (self.outcome == "hit_exact_contract" and self.compile_elapsed_ns != 0)
        ):
            raise ValueError("parametric optimizer cache event IR is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "event_index": self.event_index,
            "batch_id": self.batch_id,
            "cache_key": self.cache_key,
            "template_hash": self.template_hash,
            "outcome": self.outcome,
            "compile_elapsed_ns": self.compile_elapsed_ns,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeParametricOptimizerTaskIRModule:
    """Reusable optimizer tasks lowered once from a PlanTemplate."""

    module_id: str
    optimizer_template_hash: str
    tasks: Tuple[NativeOptimizerTaskIRUnit, ...]
    entry_task_ids: Tuple[str, ...]
    output_task_id: str
    schema_version: str = PARAMETRIC_OPTIMIZER_TASK_IR_SCHEMA_VERSION

    def validate(self, *, template: NativeParametricOptimizerTemplateIR) -> None:
        template.validate()
        if (
            self.schema_version != PARAMETRIC_OPTIMIZER_TASK_IR_SCHEMA_VERSION
            or not self.module_id
            or self.optimizer_template_hash != template.stable_hash()
            or not self.tasks
            or not self.entry_task_ids
            or not self.output_task_id
        ):
            raise ValueError("parametric optimizer Task IR module is invalid")
        task_by_id = {task.task_id: task for task in self.tasks}
        if len(task_by_id) != len(self.tasks):
            raise ValueError("parametric optimizer Task IR IDs repeat")
        if not set(self.entry_task_ids) <= set(task_by_id):
            raise ValueError("parametric optimizer Task IR entry is absent")
        if self.output_task_id not in task_by_id:
            raise ValueError("parametric optimizer Task IR output is absent")
        completed: set[str] = set()
        available = {"optimizer.state.s000"}
        for task in self.tasks:
            task.validate(steps=template.steps)
            if any(
                dependency not in completed for dependency in task.dependency_task_ids
            ):
                raise ValueError("parametric optimizer task dependency is late")
            if any(value_id not in available for value_id in task.input_value_ids):
                raise ValueError("parametric optimizer task input is absent")
            if any(value_id in available for value_id in task.output_value_ids):
                raise ValueError("parametric optimizer task output redefines a value")
            completed.add(task.task_id)
            available.update(task.output_value_ids)
        if tuple(self.entry_task_ids) != (self.tasks[0].task_id,):
            raise ValueError("parametric optimizer Task IR entry order differs")
        if self.tasks[-1].task_id != self.output_task_id:
            raise ValueError("parametric optimizer Task IR output is not final")
        if tuple(task.kind for task in self.tasks) != _expected_task_kinds(
            template.steps
        ):
            raise ValueError("parametric optimizer Task IR sequence differs")

    def to_dict(
        self, *, template: NativeParametricOptimizerTemplateIR
    ) -> dict[str, object]:
        self.validate(template=template)
        return {
            "schema_version": self.schema_version,
            "module_id": self.module_id,
            "optimizer_template_hash": self.optimizer_template_hash,
            "tasks": [task.to_dict() for task in self.tasks],
            "entry_task_ids": list(self.entry_task_ids),
            "output_task_id": self.output_task_id,
        }

    def stable_hash(self, *, template: NativeParametricOptimizerTemplateIR) -> str:
        return _canonical_hash(self.to_dict(template=template))


@dataclass(frozen=True)
class NativeParametricOptimizerScheduleIR:
    """Reusable optimizer launch order owned by one PlanTemplate."""

    schedule_id: str
    optimizer_template_hash: str
    optimizer_task_module_hash: str
    actions: Tuple[NativeOptimizerScheduleAction, ...]
    selected_state_value_id: str
    schema_version: str = PARAMETRIC_OPTIMIZER_SCHEDULE_IR_SCHEMA_VERSION

    def validate(
        self,
        *,
        template: NativeParametricOptimizerTemplateIR,
        task_module: NativeParametricOptimizerTaskIRModule,
    ) -> None:
        task_module.validate(template=template)
        if (
            self.schema_version != PARAMETRIC_OPTIMIZER_SCHEDULE_IR_SCHEMA_VERSION
            or not self.schedule_id
            or self.optimizer_template_hash != template.stable_hash()
            or self.optimizer_task_module_hash
            != task_module.stable_hash(template=template)
            or len(self.actions) != len(task_module.tasks)
            or not self.selected_state_value_id
        ):
            raise ValueError("parametric optimizer Schedule IR is invalid")
        if len({action.action_id for action in self.actions}) != len(self.actions):
            raise ValueError("parametric optimizer Schedule action IDs repeat")
        for sequence, (action, task) in enumerate(zip(self.actions, task_module.tasks)):
            action.validate()
            if (
                action.sequence != sequence
                or action.task_id != task.task_id
                or action.input_value_ids != task.input_value_ids
                or action.output_value_ids != task.output_value_ids
            ):
                raise ValueError("parametric optimizer Schedule/Task linkage differs")
        if self.selected_state_value_id not in self.actions[-1].output_value_ids:
            raise ValueError("parametric optimizer Schedule result is not emitted")

    def to_dict(
        self,
        *,
        template: NativeParametricOptimizerTemplateIR,
        task_module: NativeParametricOptimizerTaskIRModule,
    ) -> dict[str, object]:
        self.validate(template=template, task_module=task_module)
        return {
            "schema_version": self.schema_version,
            "schedule_id": self.schedule_id,
            "optimizer_template_hash": self.optimizer_template_hash,
            "optimizer_task_module_hash": self.optimizer_task_module_hash,
            "actions": [action.to_dict() for action in self.actions],
            "selected_state_value_id": self.selected_state_value_id,
        }

    def stable_hash(
        self,
        *,
        template: NativeParametricOptimizerTemplateIR,
        task_module: NativeParametricOptimizerTaskIRModule,
    ) -> str:
        return _canonical_hash(self.to_dict(template=template, task_module=task_module))


def lower_native_parametric_optimizer_template_ir(
    template: NativeParametricOptimizerTemplateIR,
) -> tuple[
    NativeParametricOptimizerTaskIRModule,
    NativeParametricOptimizerScheduleIR,
]:
    """Lower one static optimizer template into reusable Task/Schedule IR."""

    template.validate()
    tasks: list[NativeOptimizerTaskIRUnit] = []
    state_ids: list[str] = []
    metric_ids: list[str] = []
    previous_task: Optional[str] = None

    def add(
        kind: OptimizerTaskKind,
        iteration: Optional[int],
        inputs: Tuple[str, ...],
        outputs: Tuple[str, ...],
    ) -> None:
        nonlocal previous_task
        suffix = "final" if iteration is None else f"s{iteration:03d}"
        task_id = f"parametric_optimizer.{kind.value}.{suffix}"
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

    current_state = "optimizer.state.s000"
    for iteration in range(template.steps + 1):
        state_ids.append(current_state)
        bounds = f"optimizer.bounds.s{iteration:03d}"
        metric = f"optimizer.metric.s{iteration:03d}"
        add(OptimizerTaskKind.EVALUATE_BOUND, iteration, (current_state,), (bounds,))
        add(OptimizerTaskKind.REDUCE_METRIC, iteration, (bounds,), (metric,))
        metric_ids.append(metric)
        if iteration < template.steps:
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
    task_module = NativeParametricOptimizerTaskIRModule(
        module_id=f"{template.template_id}.tasks",
        optimizer_template_hash=template.stable_hash(),
        tasks=tuple(tasks),
        entry_task_ids=(tasks[0].task_id,),
        output_task_id=tasks[-1].task_id,
    )
    actions = tuple(
        NativeOptimizerScheduleAction(
            action_id=f"parametric.launch.{index:04d}.{task.task_id}",
            sequence=index,
            task_id=task.task_id,
            input_value_ids=task.input_value_ids,
            output_value_ids=task.output_value_ids,
        )
        for index, task in enumerate(task_module.tasks)
    )
    schedule = NativeParametricOptimizerScheduleIR(
        schedule_id=f"{template.template_id}.schedule",
        optimizer_template_hash=template.stable_hash(),
        optimizer_task_module_hash=task_module.stable_hash(template=template),
        actions=actions,
        selected_state_value_id=selected,
    )
    schedule.validate(template=template, task_module=task_module)
    return task_module, schedule


__all__ = [
    "PARAMETRIC_OPTIMIZER_CACHE_EVENT_IR_SCHEMA_VERSION",
    "PARAMETRIC_OPTIMIZER_INSTANCE_IR_SCHEMA_VERSION",
    "PARAMETRIC_OPTIMIZER_SCHEDULE_IR_SCHEMA_VERSION",
    "PARAMETRIC_OPTIMIZER_TASK_IR_SCHEMA_VERSION",
    "PARAMETRIC_OPTIMIZER_TEMPLATE_IR_SCHEMA_VERSION",
    "NativeParametricOptimizerCacheEventIR",
    "NativeParametricOptimizerInstanceIR",
    "NativeParametricOptimizerScheduleIR",
    "NativeParametricOptimizerTaskIRModule",
    "NativeParametricOptimizerTemplateIR",
    "lower_native_parametric_optimizer_template_ir",
]
