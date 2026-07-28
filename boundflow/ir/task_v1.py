"""First-class typed Task IR v1 derived from Bound and Plan IR."""

# Cross-layer task verification deliberately resolves every typed reference.
# pylint: disable=too-many-branches,too-many-instance-attributes,too-many-locals,too-many-statements,missing-function-docstring,duplicate-code

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import Optional, Tuple

from .bound import (
    AddBackwardAttrs,
    BFBoundModule,
    BoundOp,
    BoundOpKind,
    BoundTensorType,
    BoundValue,
    ConcretizeAttrs,
    Conv2dBackwardAttrs,
    InputBindAttrs,
    LinearBackwardAttrs,
    ReluRelaxationAttrs,
    SpecBindAttrs,
)
from .plan import (
    BackendCandidate,
    BackendKind,
    PlanInstance,
    PlanTemplate,
    RegionCandidate,
    RegionKind,
)
from .schedule import LaunchAction, ScheduleModule

TASK_IR_SCHEMA_VERSION = "boundflow.task_ir/v1.0"


class TaskIRKind(Enum):
    """Semantic task families required by the first compiler closure."""

    BOUND_BINDING = "bound_binding"
    PLAIN_CROWN_REGION = "plain_crown_region"
    CONCRETIZATION = "concretization"
    STATE_UPDATE = "state_update"


class TaskMemoryAccess(Enum):
    """Explicit task-level memory effect."""

    READ = "read"
    WRITE = "write"
    READ_WRITE = "read_write"


class TaskExternalDependencyKind(Enum):
    """Typed non-parameter runtime dependency."""

    OBJECTIVE = "objective"
    PERTURBATION = "perturbation"
    PREACTIVATION_BOUND = "preactivation_bound"


@dataclass(frozen=True)
class TaskOpRef:
    """Stable reference to one semantic Bound IR operation."""

    op_id: str
    kind: BoundOpKind

    def validate(self) -> None:
        if not self.op_id:
            raise ValueError("Task IR op reference ID is empty")

    def to_dict(self) -> dict[str, str]:
        self.validate()
        return {"op_id": self.op_id, "kind": self.kind.value}


@dataclass(frozen=True)
class TaskMemoryEffect:
    """One value read/write performed by a task."""

    value_id: str
    access: TaskMemoryAccess

    def validate(self) -> None:
        if not self.value_id:
            raise ValueError("Task IR memory effect value ID is empty")

    def to_dict(self) -> dict[str, str]:
        self.validate()
        return {"value_id": self.value_id, "access": self.access.value}


@dataclass(frozen=True)
class TaskStateDependency:
    """One versioned Bound value crossing a task boundary."""

    value_id: str
    state_version: str
    access: TaskMemoryAccess

    def validate(self) -> None:
        if not self.value_id or not self.state_version:
            raise ValueError("Task IR state dependency is incomplete")

    def to_dict(self) -> dict[str, str]:
        self.validate()
        return {
            "value_id": self.value_id,
            "state_version": self.state_version,
            "access": self.access.value,
        }


@dataclass(frozen=True)
class TaskExternalDependency:
    """One typed query/runtime dependency owned outside Task IR."""

    kind: TaskExternalDependencyKind
    dependency_id: str

    def validate(self) -> None:
        if not self.dependency_id:
            raise ValueError("Task IR external dependency ID is empty")

    def to_dict(self) -> dict[str, str]:
        self.validate()
        return {"kind": self.kind.value, "dependency_id": self.dependency_id}


@dataclass(frozen=True)
class TaskValueConstraint:
    """Exact tensor/shape contract for one task boundary value."""

    value_id: str
    tensor_type: BoundTensorType

    def validate(self) -> None:
        if not self.value_id:
            raise ValueError("Task IR value constraint ID is empty")
        self.tensor_type.validate()

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "value_id": self.value_id,
            "tensor_type": self.tensor_type.to_dict(),
        }


@dataclass(frozen=True)
class TaskBackendBinding:
    """Exact Plan backend selected for this task."""

    backend_candidate_id: str
    capability_id: str
    compiled_artifact_key: Optional[str]
    reference_implementation_id: str

    def validate(self) -> None:
        if (
            not self.backend_candidate_id
            or not self.capability_id
            or not self.reference_implementation_id
        ):
            raise ValueError("Task IR backend binding is incomplete")
        if self.compiled_artifact_key is not None and not self.compiled_artifact_key:
            raise ValueError("Task IR compiled artifact key is empty")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "backend_candidate_id": self.backend_candidate_id,
            "capability_id": self.capability_id,
            "compiled_artifact_key": self.compiled_artifact_key,
            "reference_implementation_id": self.reference_implementation_id,
        }


@dataclass(frozen=True)
class TaskIRUnit:  # pylint: disable=too-many-instance-attributes
    """One backend-callable task with no hidden semantic attrs."""

    task_id: str
    region_id: str
    kind: TaskIRKind
    op_refs: Tuple[TaskOpRef, ...]
    input_value_ids: Tuple[str, ...]
    output_value_ids: Tuple[str, ...]
    input_constraints: Tuple[TaskValueConstraint, ...]
    output_constraints: Tuple[TaskValueConstraint, ...]
    parameter_value_ids: Tuple[str, ...]
    external_dependencies: Tuple[TaskExternalDependency, ...]
    state_dependencies: Tuple[TaskStateDependency, ...]
    memory_effects: Tuple[TaskMemoryEffect, ...]
    dependency_task_ids: Tuple[str, ...]
    backend: TaskBackendBinding

    def validate(self) -> None:
        if not self.task_id or not self.region_id:
            raise ValueError("Task IR task/region identity is incomplete")
        if not self.op_refs or not self.input_value_ids or not self.output_value_ids:
            raise ValueError("Task IR unit requires ops and boundary values")
        for op_ref in self.op_refs:
            op_ref.validate()
        for constraint in (*self.input_constraints, *self.output_constraints):
            constraint.validate()
        for external_dependency in self.external_dependencies:
            external_dependency.validate()
        for state_dependency in self.state_dependencies:
            state_dependency.validate()
        for memory_effect in self.memory_effects:
            memory_effect.validate()
        self.backend.validate()
        for label, values in (
            ("op refs", tuple(item.op_id for item in self.op_refs)),
            ("inputs", self.input_value_ids),
            ("outputs", self.output_value_ids),
            (
                "input constraints",
                tuple(item.value_id for item in self.input_constraints),
            ),
            (
                "output constraints",
                tuple(item.value_id for item in self.output_constraints),
            ),
            ("parameters", self.parameter_value_ids),
            (
                "external dependencies",
                tuple(
                    (item.kind, item.dependency_id)
                    for item in self.external_dependencies
                ),
            ),
            (
                "state dependencies",
                tuple(
                    (item.value_id, item.state_version, item.access)
                    for item in self.state_dependencies
                ),
            ),
            (
                "memory effects",
                tuple((item.value_id, item.access) for item in self.memory_effects),
            ),
            ("task dependencies", self.dependency_task_ids),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"Task IR {label} must be unique")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "task_id": self.task_id,
            "region_id": self.region_id,
            "kind": self.kind.value,
            "op_refs": [item.to_dict() for item in self.op_refs],
            "input_value_ids": list(self.input_value_ids),
            "output_value_ids": list(self.output_value_ids),
            "input_constraints": [item.to_dict() for item in self.input_constraints],
            "output_constraints": [item.to_dict() for item in self.output_constraints],
            "parameter_value_ids": list(self.parameter_value_ids),
            "external_dependencies": [
                item.to_dict() for item in self.external_dependencies
            ],
            "state_dependencies": [item.to_dict() for item in self.state_dependencies],
            "memory_effects": [item.to_dict() for item in self.memory_effects],
            "dependency_task_ids": list(self.dependency_task_ids),
            "backend": self.backend.to_dict(),
        }


@dataclass(frozen=True)
class TaskIRModule:
    """Typed tasks selected for one exact PlanInstance."""

    module_id: str
    bound_module_hash: str
    plan_template_hash: str
    plan_instance_hash: str
    tasks: Tuple[TaskIRUnit, ...]
    entry_task_ids: Tuple[str, ...]
    output_task_ids: Tuple[str, ...]
    schema_version: str = TASK_IR_SCHEMA_VERSION

    def validate(
        self,
        *,
        bound_module: BFBoundModule,
        template: PlanTemplate,
        instance: PlanInstance,
    ) -> None:
        if self.schema_version != TASK_IR_SCHEMA_VERSION:
            raise ValueError(f"unsupported Task IR schema: {self.schema_version}")
        if not self.module_id or not self.tasks:
            raise ValueError("Task IR module identity/tasks are incomplete")
        instance.validate(template=template, bound_module=bound_module)
        expected_hashes = (
            bound_module.stable_hash(),
            template.stable_hash(bound_module=bound_module),
            instance.stable_hash(template=template, bound_module=bound_module),
        )
        if (
            self.bound_module_hash,
            self.plan_template_hash,
            self.plan_instance_hash,
        ) != expected_hashes:
            raise ValueError("Task IR input hashes do not match typed inputs")
        for task in self.tasks:
            task.validate()
        task_by_id = {task.task_id: task for task in self.tasks}
        if len(task_by_id) != len(self.tasks):
            raise ValueError("Task IR task IDs must be unique")
        if not self.entry_task_ids or not self.output_task_ids:
            raise ValueError("Task IR requires entry/output tasks")
        if not set(self.entry_task_ids).issubset(task_by_id):
            raise ValueError("Task IR entry references an unknown task")
        if not set(self.output_task_ids).issubset(task_by_id):
            raise ValueError("Task IR output references an unknown task")

        ops = {op.op_id: op for op in bound_module.graph.ops}
        values = {value.value_id: value for value in bound_module.graph.values}
        selected_regions = _selected_regions(template, instance)
        region_by_id = {region.region_id: region for region in selected_regions}
        backends = _selected_backends(template, instance)
        if {task.region_id for task in self.tasks} != set(region_by_id):
            raise ValueError("Task IR tasks do not cover selected Plan regions")
        producer_task_by_value = {
            value_id: task.task_id
            for task in self.tasks
            for value_id in task.output_value_ids
        }
        seen_tasks: set[str] = set()
        for task in self.tasks:
            region = region_by_id.get(task.region_id)
            if region is None:
                raise ValueError("Task IR references an unknown selected region")
            if task.task_id != f"task:{region.region_id}":
                raise ValueError("Task IR task ID is not derived from region ID")
            if tuple(item.op_id for item in task.op_refs) != region.op_ids:
                raise ValueError("Task IR op references differ from Plan region")
            if any(item.kind != ops[item.op_id].kind for item in task.op_refs):
                raise ValueError("Task IR op kind differs from Bound IR")
            if task.input_value_ids != region.input_value_ids:
                raise ValueError("Task IR inputs differ from Plan region boundary")
            if task.output_value_ids != region.output_value_ids:
                raise ValueError("Task IR outputs differ from Plan region boundary")
            expected_input_constraints = _value_constraints(
                region.input_value_ids, values=values
            )
            expected_output_constraints = _value_constraints(
                region.output_value_ids, values=values
            )
            if task.input_constraints != expected_input_constraints:
                raise ValueError("Task IR input shape constraints differ from Bound IR")
            if task.output_constraints != expected_output_constraints:
                raise ValueError(
                    "Task IR output shape constraints differ from Bound IR"
                )
            expected_parameters, expected_external = _region_dependencies(
                tuple(ops[op_id] for op_id in region.op_ids)
            )
            if task.parameter_value_ids != expected_parameters:
                raise ValueError("Task IR parameter dependencies are incomplete")
            if task.external_dependencies != expected_external:
                raise ValueError("Task IR external dependencies are incomplete")
            expected_states = _state_dependencies(
                region,
                values=values,
            )
            if task.state_dependencies != expected_states:
                raise ValueError("Task IR state dependencies are incomplete")
            expected_effects = (
                *(
                    TaskMemoryEffect(value_id, TaskMemoryAccess.READ)
                    for value_id in region.input_value_ids
                ),
                *(
                    TaskMemoryEffect(value_id, TaskMemoryAccess.WRITE)
                    for value_id in region.output_value_ids
                ),
            )
            if task.memory_effects != expected_effects:
                raise ValueError("Task IR memory effects differ from boundaries")
            expected_dependencies = tuple(
                dict.fromkeys(
                    producer_task_by_value[value_id]
                    for value_id in region.input_value_ids
                    if value_id in producer_task_by_value
                )
            )
            if task.dependency_task_ids != expected_dependencies:
                raise ValueError("Task IR dependency edges differ from use-def")
            if any(
                dependency not in seen_tasks for dependency in task.dependency_task_ids
            ):
                raise ValueError("Task IR dependency order has use-before-task")
            expected_backend = backends[region.region_id]
            if not _backend_matches(task.backend, expected_backend):
                raise ValueError("Task IR backend differs from PlanInstance")
            if task.kind != _task_kind(region.kind):
                raise ValueError("Task IR kind differs from Plan region kind")
            seen_tasks.add(task.task_id)
        expected_entry = tuple(
            task.task_id for task in self.tasks if not task.dependency_task_ids
        )
        if self.entry_task_ids != expected_entry:
            raise ValueError("Task IR entry task set is incorrect")
        expected_outputs = tuple(
            task.task_id
            for task in self.tasks
            if set(task.output_value_ids) & set(bound_module.graph.outputs)
        )
        if self.output_task_ids != expected_outputs:
            raise ValueError("Task IR output task set is incorrect")

    def validate_schedule_linkage(
        self,
        schedule: ScheduleModule,
        *,
        bound_module: BFBoundModule,
        template: PlanTemplate,
        instance: PlanInstance,
    ) -> None:
        """Require every typed task to have exactly one matching launch."""

        self.validate(bound_module=bound_module, template=template, instance=instance)
        schedule.validate(
            bound_module=bound_module, template=template, instance=instance
        )
        launches = tuple(
            action for action in schedule.actions if isinstance(action, LaunchAction)
        )
        launch_by_task = {launch.task_id: launch for launch in launches}
        if len(launch_by_task) != len(launches):
            raise ValueError("Schedule IR launches a Task IR task more than once")
        if set(launch_by_task) != {task.task_id for task in self.tasks}:
            raise ValueError("Task/Schedule IR launch sets differ")
        for task in self.tasks:
            launch = launch_by_task[task.task_id]
            if (
                launch.region_id != task.region_id
                or launch.backend_candidate_id != task.backend.backend_candidate_id
                or launch.backend_artifact_key != task.backend.compiled_artifact_key
                or launch.input_value_ids != task.input_value_ids
                or launch.output_value_ids != task.output_value_ids
            ):
                raise ValueError("Schedule launch differs from typed Task IR")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "module_id": self.module_id,
            "bound_module_hash": self.bound_module_hash,
            "plan_template_hash": self.plan_template_hash,
            "plan_instance_hash": self.plan_instance_hash,
            "tasks": [task.to_dict() for task in self.tasks],
            "entry_task_ids": list(self.entry_task_ids),
            "output_task_ids": list(self.output_task_ids),
        }

    def canonical_json(
        self,
        *,
        bound_module: BFBoundModule,
        template: PlanTemplate,
        instance: PlanInstance,
    ) -> str:
        self.validate(bound_module=bound_module, template=template, instance=instance)
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    def stable_hash(
        self,
        *,
        bound_module: BFBoundModule,
        template: PlanTemplate,
        instance: PlanInstance,
    ) -> str:
        return hashlib.sha256(
            self.canonical_json(
                bound_module=bound_module,
                template=template,
                instance=instance,
            ).encode("utf-8")
        ).hexdigest()


def lower_plan_instance_to_task_ir(
    bound_module: BFBoundModule,
    *,
    template: PlanTemplate,
    instance: PlanInstance,
) -> TaskIRModule:
    """Lower every selected Plan region into one typed Task IR unit."""

    instance.validate(template=template, bound_module=bound_module)
    ops = {op.op_id: op for op in bound_module.graph.ops}
    values = {value.value_id: value for value in bound_module.graph.values}
    regions = _selected_regions(template, instance)
    op_index = {op.op_id: index for index, op in enumerate(bound_module.graph.ops)}
    regions = tuple(
        sorted(regions, key=lambda item: min(op_index[op_id] for op_id in item.op_ids))
    )
    backends = _selected_backends(template, instance)
    producer_task_by_value: dict[str, str] = {}
    tasks: list[TaskIRUnit] = []
    for region in regions:
        region_ops = tuple(ops[op_id] for op_id in region.op_ids)
        parameters, external = _region_dependencies(region_ops)
        backend = backends[region.region_id]
        task = TaskIRUnit(
            task_id=f"task:{region.region_id}",
            region_id=region.region_id,
            kind=_task_kind(region.kind),
            op_refs=tuple(TaskOpRef(op.op_id, op.kind) for op in region_ops),
            input_value_ids=region.input_value_ids,
            output_value_ids=region.output_value_ids,
            input_constraints=_value_constraints(region.input_value_ids, values=values),
            output_constraints=_value_constraints(
                region.output_value_ids, values=values
            ),
            parameter_value_ids=parameters,
            external_dependencies=external,
            state_dependencies=_state_dependencies(region, values=values),
            memory_effects=(
                *(
                    TaskMemoryEffect(value_id, TaskMemoryAccess.READ)
                    for value_id in region.input_value_ids
                ),
                *(
                    TaskMemoryEffect(value_id, TaskMemoryAccess.WRITE)
                    for value_id in region.output_value_ids
                ),
            ),
            dependency_task_ids=tuple(
                dict.fromkeys(
                    producer_task_by_value[value_id]
                    for value_id in region.input_value_ids
                    if value_id in producer_task_by_value
                )
            ),
            backend=TaskBackendBinding(
                backend_candidate_id=backend.candidate_id,
                capability_id=backend.capability_id,
                compiled_artifact_key=backend.compiled_artifact_key,
                reference_implementation_id=_implementation_id(backend.backend),
            ),
        )
        tasks.append(task)
        for value_id in region.output_value_ids:
            producer_task_by_value[value_id] = task.task_id
    entry_task_ids = tuple(
        task.task_id for task in tasks if not task.dependency_task_ids
    )
    output_task_ids = tuple(
        task.task_id
        for task in tasks
        if set(task.output_value_ids) & set(bound_module.graph.outputs)
    )
    instance_hash = instance.stable_hash(template=template, bound_module=bound_module)
    identity = "|".join((bound_module.stable_hash(), instance_hash, *entry_task_ids))
    module = TaskIRModule(
        module_id="task-module:"
        + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24],
        bound_module_hash=bound_module.stable_hash(),
        plan_template_hash=template.stable_hash(bound_module=bound_module),
        plan_instance_hash=instance_hash,
        tasks=tuple(tasks),
        entry_task_ids=entry_task_ids,
        output_task_ids=output_task_ids,
    )
    module.validate(bound_module=bound_module, template=template, instance=instance)
    return module


def _selected_regions(
    template: PlanTemplate, instance: PlanInstance
) -> Tuple[RegionCandidate, ...]:
    candidates = {
        candidate.candidate_id: candidate for candidate in template.region_candidates
    }
    return tuple(candidates[item.candidate_id] for item in instance.region_decisions)


def _selected_backends(
    template: PlanTemplate, instance: PlanInstance
) -> dict[str, BackendCandidate]:
    candidates = {
        candidate.candidate_id: candidate for candidate in template.backend_candidates
    }
    return {
        item.region_id: candidates[item.candidate_id]
        for item in instance.backend_decisions
    }


def _task_kind(region_kind: RegionKind) -> TaskIRKind:
    if region_kind == RegionKind.BINDING:
        return TaskIRKind.BOUND_BINDING
    if region_kind == RegionKind.CONCRETIZATION:
        return TaskIRKind.CONCRETIZATION
    return TaskIRKind.PLAIN_CROWN_REGION


def _region_dependencies(
    ops: Tuple[BoundOp, ...],
) -> tuple[Tuple[str, ...], Tuple[TaskExternalDependency, ...]]:
    parameters: list[str] = []
    external: list[TaskExternalDependency] = []
    for op in ops:
        attrs = op.attrs
        if isinstance(attrs, (LinearBackwardAttrs, Conv2dBackwardAttrs)):
            parameters.append(attrs.weight_primal_value_id)
            if attrs.bias_primal_value_id is not None:
                parameters.append(attrs.bias_primal_value_id)
        elif isinstance(attrs, AddBackwardAttrs):
            parameters.extend(attrs.constant_input_primal_value_ids)
        elif isinstance(attrs, SpecBindAttrs):
            external.append(
                TaskExternalDependency(
                    TaskExternalDependencyKind.OBJECTIVE,
                    attrs.objective_id,
                )
            )
        elif isinstance(attrs, InputBindAttrs):
            external.append(
                TaskExternalDependency(
                    TaskExternalDependencyKind.PERTURBATION,
                    attrs.perturbation_id,
                )
            )
        elif isinstance(attrs, ConcretizeAttrs):
            external.append(
                TaskExternalDependency(
                    TaskExternalDependencyKind.PERTURBATION,
                    attrs.perturbation_id,
                )
            )
        elif isinstance(attrs, ReluRelaxationAttrs):
            if attrs.preactivation_primal_value_id is not None:
                external.append(
                    TaskExternalDependency(
                        TaskExternalDependencyKind.PREACTIVATION_BOUND,
                        attrs.preactivation_primal_value_id,
                    )
                )
    return tuple(dict.fromkeys(parameters)), tuple(dict.fromkeys(external))


def _state_dependencies(
    region: RegionCandidate, *, values: dict[str, BoundValue]
) -> Tuple[TaskStateDependency, ...]:
    result: list[TaskStateDependency] = []
    for value_id, access in (
        *((value_id, TaskMemoryAccess.READ) for value_id in region.input_value_ids),
        *((value_id, TaskMemoryAccess.WRITE) for value_id in region.output_value_ids),
    ):
        value = values[value_id]
        state_version = value.state_version
        if state_version is not None:
            result.append(TaskStateDependency(value_id, state_version, access))
    return tuple(result)


def _value_constraints(
    value_ids: Tuple[str, ...], *, values: dict[str, BoundValue]
) -> Tuple[TaskValueConstraint, ...]:
    return tuple(
        TaskValueConstraint(value_id, values[value_id].tensor_type)
        for value_id in value_ids
    )


def _backend_matches(binding: TaskBackendBinding, candidate: BackendCandidate) -> bool:
    return (
        binding.backend_candidate_id == candidate.candidate_id
        and binding.capability_id == candidate.capability_id
        and binding.compiled_artifact_key == candidate.compiled_artifact_key
        and binding.reference_implementation_id == _implementation_id(candidate.backend)
    )


def _implementation_id(backend: BackendKind) -> str:
    return {
        BackendKind.REFERENCE: "bound_ir_region_reference/v1",
        BackendKind.PYTORCH_DENSE: "pytorch_dense_bound_region/v1",
        BackendKind.PYTORCH_STRUCTURED: "pytorch_structured_bound_region/v1",
        BackendKind.PYTORCH_CHUNKED: "pytorch_chunked_fused_relu_affine/v1",
        BackendKind.TORCH_COMPILE: "torch_compile_bound_region/v1",
        BackendKind.TVM_RELAX_UNFUSED: "tvm_relax_unfused_bound_region/v1",
        BackendKind.TVM_TIR_UNFUSED: "tvm_tir_unfused_bound_region/v1",
        BackendKind.TVM_FUSED_TIR: "tvm_fused_tir_bound_region/v1",
    }[backend]
