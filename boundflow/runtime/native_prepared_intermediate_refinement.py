"""Prepare-once validation ownership for native intermediate refinement."""

# pylint: disable=too-many-arguments,too-many-instance-attributes
# pylint: disable=missing-function-docstring,missing-class-docstring
# pylint: disable=protected-access,no-member,too-many-boolean-expressions
# pylint: disable=too-many-locals

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
import hashlib
import json
from typing import Mapping, Optional, cast

import torch

from ..ir.prepared_intermediate_refinement import (
    NativePreparedIntermediateRefinementCapsuleIR,
    NativePreparedRefinementExecutionTraceIR,
    NativePreparedRefinementScheduleIR,
    NativePreparedRefinementTaskIRModule,
    lower_native_prepared_refinement_ir,
)
from ..ir.refinement import (
    NativeIntermediateRefinementMultiPassPolicyIR,
    NativeIntermediateRefinementPolicyIR,
)
from ..ir.target_admission import (
    NativeTargetAdmissionReceiptIR,
    NativeTargetAdmissionScheduleIR,
    NativeTargetAdmissionTaskIRModule,
)
from ..ir.task import BFTaskModule
from .native_intermediate_refinement import (
    NativeExternalIntermediateConstraintSeed,
    NativeIntermediateRefinementExecution,
    NativeIntermediateRefinementProgram,
    compile_native_intermediate_refinement_program,
    execute_native_intermediate_refinement_program,
    intermediate_refinement_semantic_trace_hash,
)
from .native_target_admission import (
    compile_native_single_pass_target_admission_program,
    validate_native_target_admission_structure,
)
from .task_executor import InputSpec


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _input_spec_matches(admitted: InputSpec, candidate: InputSpec) -> bool:
    if admitted.value_name != candidate.value_name:
        return False
    admitted_lower, admitted_upper = admitted.perturbation.bounding_box(admitted.center)
    candidate_lower, candidate_upper = candidate.perturbation.bounding_box(
        candidate.center
    )
    return bool(
        type(admitted.perturbation) is type(candidate.perturbation)
        and admitted.center.shape == candidate.center.shape
        and admitted.center.dtype == candidate.center.dtype
        and admitted.center.device == candidate.center.device
        and torch.equal(admitted.center, candidate.center)
        and torch.equal(admitted_lower, candidate_lower)
        and torch.equal(admitted_upper, candidate_upper)
    )


@dataclass(frozen=True)
class _TensorVersionWitness:
    tensor: torch.Tensor
    version: int
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device

    def valid(self) -> bool:
        return bool(
            int(self.tensor._version)
            == self.version  # pylint: disable=protected-access
            and tuple(self.tensor.shape) == self.shape
            and self.tensor.dtype == self.dtype
            and self.tensor.device == self.device
        )


@dataclass(frozen=True)
class _ContainerIdentityWitness:
    owner: object
    members: tuple[tuple[object, int], ...]
    kind: str

    def valid(self) -> bool:
        if self.kind == "mapping":
            if not isinstance(self.owner, Mapping):
                return False
            current = tuple((key, id(value)) for key, value in self.owner.items())
        elif self.kind == "list":
            if not isinstance(self.owner, list):
                return False
            current = tuple(
                (index, id(value)) for index, value in enumerate(self.owner)
            )
        elif self.kind == "dataclass":
            if not is_dataclass(self.owner) or isinstance(self.owner, type):
                return False
            current = tuple(
                (field.name, id(getattr(self.owner, field.name)))
                for field in fields(self.owner)
            )
        else:
            raise AssertionError("unknown prepared receipt witness")
        return current == self.members


@dataclass(frozen=True)
class _PreparedRuntimeReceipt:
    shallow_roots: tuple[object, ...]
    shallow_root_ids: tuple[int, ...]
    containers: tuple[_ContainerIdentityWitness, ...]
    tensors: tuple[_TensorVersionWitness, ...]

    def validate(self, shallow_roots: tuple[object, ...]) -> None:
        if (
            tuple(id(value) for value in shallow_roots) != self.shallow_root_ids
            or any(not witness.valid() for witness in self.containers)
            or any(not witness.valid() for witness in self.tensors)
        ):
            raise ValueError("prepared refinement runtime identity differs")


def _build_runtime_receipt(
    *, shallow_roots: tuple[object, ...], deep_roots: tuple[object, ...]
) -> _PreparedRuntimeReceipt:
    containers: list[_ContainerIdentityWitness] = []
    tensors: list[_TensorVersionWitness] = []
    seen: set[int] = set()

    def collect(value: object) -> None:
        if value is None or isinstance(value, (bool, int, float, str)):
            return
        if torch.is_tensor(value):
            tensor = cast(torch.Tensor, value)
            tensors.append(
                _TensorVersionWitness(
                    tensor=tensor,
                    version=int(tensor._version),  # pylint: disable=protected-access
                    shape=tuple(tensor.shape),
                    dtype=tensor.dtype,
                    device=tensor.device,
                )
            )
            return
        object_id = id(value)
        if object_id in seen:
            return
        seen.add(object_id)
        if isinstance(value, Mapping):
            containers.append(
                _ContainerIdentityWitness(
                    owner=value,
                    members=tuple((key, id(item)) for key, item in value.items()),
                    kind="mapping",
                )
            )
            for item in value.values():
                collect(item)
        elif isinstance(value, tuple):
            for item in value:
                collect(item)
        elif isinstance(value, list):
            containers.append(
                _ContainerIdentityWitness(
                    owner=value,
                    members=tuple(
                        (index, id(item)) for index, item in enumerate(value)
                    ),
                    kind="list",
                )
            )
            for item in value:
                collect(item)
        elif is_dataclass(value) and not isinstance(value, type):
            containers.append(
                _ContainerIdentityWitness(
                    owner=value,
                    members=tuple(
                        (field.name, id(getattr(value, field.name)))
                        for field in fields(value)
                    ),
                    kind="dataclass",
                )
            )
            for field in fields(value):
                collect(getattr(value, field.name))

    for root in deep_roots:
        collect(root)
    return _PreparedRuntimeReceipt(
        shallow_roots=shallow_roots,
        shallow_root_ids=tuple(id(value) for value in shallow_roots),
        containers=tuple(containers),
        tensors=tuple(tensors),
    )


def _program_runtime_roots(
    program: NativeIntermediateRefinementProgram,
    module: BFTaskModule,
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    immutable_roots: tuple[object, ...] = (
        module,
        program.plan,
        program.task_module,
        program.schedule,
        program.initial_interval_env,
        program.initial_relu_pre,
        program.split_state,
        program.objective,
        program.objective_influence,
        program.source_intermediate_constraints,
        program.external_constraint_seed,
        getattr(program, "capsule", None),
        getattr(program, "prepared_task_module", None),
        getattr(program, "prepared_schedule", None),
        getattr(program, "admitted_input_spec", None),
        getattr(program, "target_admission_receipt", None),
        getattr(program, "target_admission_task_module", None),
        getattr(program, "target_admission_schedule", None),
    )
    mutable_roots = (
        program.initial_interval_env,
        program.initial_relu_pre,
        program.split_state,
        program.objective,
        program.objective_influence,
        program.source_intermediate_constraints,
        program.external_constraint_seed,
        getattr(program, "admitted_input_spec", None),
    )
    return immutable_roots, mutable_roots


def _execution_runtime_roots(
    execution: NativeIntermediateRefinementExecution,
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    return (
        (
            execution.program,
            execution.interval_env,
            execution.relu_pre,
            execution.trace,
        ),
        (execution.interval_env, execution.relu_pre),
    )


@dataclass(frozen=True)
class NativePreparedIntermediateRefinementProgram(NativeIntermediateRefinementProgram):
    capsule: NativePreparedIntermediateRefinementCapsuleIR = None  # type: ignore[assignment]
    prepared_task_module: NativePreparedRefinementTaskIRModule = None  # type: ignore[assignment]
    prepared_schedule: NativePreparedRefinementScheduleIR = None  # type: ignore[assignment]
    admitted_input_spec: Optional[InputSpec] = None
    runtime_receipt: Optional[_PreparedRuntimeReceipt] = None

    def validate(self, module: BFTaskModule, input_spec: InputSpec) -> None:
        if (
            not isinstance(self.capsule, NativePreparedIntermediateRefinementCapsuleIR)
            or not isinstance(
                self.prepared_task_module, NativePreparedRefinementTaskIRModule
            )
            or not isinstance(
                self.prepared_schedule, NativePreparedRefinementScheduleIR
            )
        ):
            raise ValueError("prepared refinement program ownership is absent")
        if (
            not isinstance(self.runtime_receipt, _PreparedRuntimeReceipt)
            or self.capsule.full_validation_count != 1
            or self.prepared_schedule.full_validation_launches != 1
            or self.prepared_schedule.runtime_target_selection_launches != 1
            or not isinstance(self.admitted_input_spec, InputSpec)
            or not _input_spec_matches(self.admitted_input_spec, input_spec)
        ):
            raise ValueError("prepared refinement runtime identity differs")
        shallow_roots, _deep_roots = _program_runtime_roots(self, module)
        self.runtime_receipt.validate(shallow_roots)

    def validate_full(self, module: BFTaskModule, input_spec: InputSpec) -> None:
        NativeIntermediateRefinementProgram.validate(self, module, input_spec)
        if NativeIntermediateRefinementProgram.hashes(self) != self.hashes():
            raise ValueError("prepared refinement full hash replay differs")

    def hashes(self) -> dict[str, str]:
        if not isinstance(self.capsule, NativePreparedIntermediateRefinementCapsuleIR):
            raise ValueError("prepared refinement capsule is absent")
        return {
            "refinement_plan_hash": self.capsule.source_plan_hash,
            "refinement_task_module_hash": self.capsule.source_task_module_hash,
            "refinement_schedule_hash": self.capsule.source_schedule_hash,
        }


@dataclass(frozen=True)
class NativeSinglePassPreparedIntermediateRefinementProgram(
    NativePreparedIntermediateRefinementProgram
):
    """Prepared Program whose exact targets are admitted once by typed receipt."""

    target_admission_receipt: NativeTargetAdmissionReceiptIR = None  # type: ignore[assignment]
    target_admission_task_module: NativeTargetAdmissionTaskIRModule = None  # type: ignore[assignment]
    target_admission_schedule: NativeTargetAdmissionScheduleIR = None  # type: ignore[assignment]

    def _validate_single_pass_binding(self) -> None:
        if (
            not isinstance(
                self.target_admission_receipt, NativeTargetAdmissionReceiptIR
            )
            or not isinstance(
                self.target_admission_task_module, NativeTargetAdmissionTaskIRModule
            )
            or not isinstance(
                self.target_admission_schedule, NativeTargetAdmissionScheduleIR
            )
            or self.capsule.target_admission_receipt_hash
            != self.target_admission_receipt.stable_hash()
        ):
            raise ValueError("single-pass prepared target admission differs")
        validate_native_target_admission_structure(
            self,
            receipt=self.target_admission_receipt,
            task_module=self.target_admission_task_module,
            schedule=self.target_admission_schedule,
        )

    def validate(self, module: BFTaskModule, input_spec: InputSpec) -> None:
        NativePreparedIntermediateRefinementProgram.validate(self, module, input_spec)
        self._validate_single_pass_binding()

    def validate_full(self, module: BFTaskModule, input_spec: InputSpec) -> None:
        NativePreparedIntermediateRefinementProgram.validate_full(
            self, module, input_spec
        )
        self._validate_single_pass_binding()


@dataclass(frozen=True)
class NativePreparedIntermediateRefinementExecution(
    NativeIntermediateRefinementExecution
):
    prepared_trace: NativePreparedRefinementExecutionTraceIR = None  # type: ignore[assignment]
    runtime_receipt: Optional[_PreparedRuntimeReceipt] = None

    def validate(self, module: BFTaskModule, input_spec: InputSpec) -> None:
        if not isinstance(
            self.program, NativePreparedIntermediateRefinementProgram
        ) or not isinstance(
            self.prepared_trace, NativePreparedRefinementExecutionTraceIR
        ):
            raise ValueError("prepared refinement execution ownership is absent")
        self.program.validate(module, input_spec)
        if (
            not isinstance(self.runtime_receipt, _PreparedRuntimeReceipt)
            or self.prepared_trace.full_validation_count != 1
            or self.prepared_trace.runtime_target_selection_count != 1
            or self.prepared_trace.final_intermediate_bounds_hash
            != self.trace.final_intermediate_bounds_hash
        ):
            raise ValueError("prepared refinement execution identity differs")
        shallow_roots, _deep_roots = _execution_runtime_roots(self)
        self.runtime_receipt.validate(shallow_roots)

    def validate_full(self, module: BFTaskModule, input_spec: InputSpec) -> None:
        NativeIntermediateRefinementExecution.validate(self, module, input_spec)


def _prepared_capsule(
    source: NativeIntermediateRefinementProgram,
    *,
    target_admission_receipt_hash: Optional[str] = None,
) -> NativePreparedIntermediateRefinementCapsuleIR:
    hashes = source.hashes()
    capsule = NativePreparedIntermediateRefinementCapsuleIR(
        capsule_id=f"{source.plan.plan_id}:prepared-v1",
        source_plan_hash=hashes["refinement_plan_hash"],
        source_task_module_hash=hashes["refinement_task_module_hash"],
        source_schedule_hash=hashes["refinement_schedule_hash"],
        primal_graph_hash=source.plan.primal_graph_hash,
        input_bounds_hash=source.plan.input_bounds_hash,
        split_state_hash=source.plan.split_state_hash,
        initial_intermediate_bounds_hash=(source.plan.initial_intermediate_bounds_hash),
        objective_hash=source.plan.objective_hash,
        source_refinement_plan_hash=source.plan.source_refinement_plan_hash,
        source_refinement_semantic_trace_hash=(
            source.plan.source_refinement_semantic_trace_hash
        ),
        target_table_hash=_canonical_hash(
            [target.to_dict() for target in source.plan.targets]
        ),
        target_count=len(source.plan.targets),
        full_validation_receipt="0" * 64,
        target_admission_receipt_hash=target_admission_receipt_hash,
        schema_version=(
            "boundflow.prepared-intermediate-refinement-capsule/v2"
            if target_admission_receipt_hash is not None
            else "boundflow.prepared-intermediate-refinement-capsule/v1"
        ),
    )
    capsule = replace(
        capsule, full_validation_receipt=_canonical_hash(capsule.receipt_dict())
    )
    capsule.validate()
    return capsule


def compile_native_prepared_intermediate_refinement_program(
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    policy: NativeIntermediateRefinementPolicyIR,
    plan_id: str,
    multi_pass_policy: Optional[NativeIntermediateRefinementMultiPassPolicyIR] = None,
    relu_split_state: Optional[Mapping[str, torch.Tensor]] = None,
    linear_spec_C: Optional[torch.Tensor] = None,
    source_refinement_execution: Optional[NativeIntermediateRefinementExecution] = None,
    external_constraint_seed: Optional[NativeExternalIntermediateConstraintSeed] = None,
) -> NativePreparedIntermediateRefinementProgram:
    """Compile and fully validate once, then publish prepared ownership."""

    source = compile_native_intermediate_refinement_program(
        module,
        input_spec,
        policy=policy,
        plan_id=plan_id,
        multi_pass_policy=multi_pass_policy,
        relu_split_state=relu_split_state,
        linear_spec_C=linear_spec_C,
        source_refinement_execution=source_refinement_execution,
        external_constraint_seed=external_constraint_seed,
    )
    capsule = _prepared_capsule(source)
    prepared_task_module, prepared_schedule = lower_native_prepared_refinement_ir(
        capsule
    )
    program = NativePreparedIntermediateRefinementProgram(
        plan=source.plan,
        task_module=source.task_module,
        schedule=source.schedule,
        initial_interval_env=source.initial_interval_env,
        initial_relu_pre=source.initial_relu_pre,
        split_state=source.split_state,
        objective=source.objective,
        objective_influence=source.objective_influence,
        source_intermediate_constraints=source.source_intermediate_constraints,
        external_constraint_seed=source.external_constraint_seed,
        capsule=capsule,
        prepared_task_module=prepared_task_module,
        prepared_schedule=prepared_schedule,
        admitted_input_spec=input_spec,
    )
    shallow_roots, deep_roots = _program_runtime_roots(program, module)
    program = replace(
        program,
        runtime_receipt=_build_runtime_receipt(
            shallow_roots=shallow_roots, deep_roots=deep_roots
        ),
    )
    program.validate(module, input_spec)
    return program


def compile_native_single_pass_prepared_intermediate_refinement_program(
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    policy: NativeIntermediateRefinementPolicyIR,
    plan_id: str,
    multi_pass_policy: Optional[NativeIntermediateRefinementMultiPassPolicyIR] = None,
    relu_split_state: Optional[Mapping[str, torch.Tensor]] = None,
    linear_spec_C: Optional[torch.Tensor] = None,
    source_refinement_execution: Optional[NativeIntermediateRefinementExecution] = None,
    external_constraint_seed: Optional[NativeExternalIntermediateConstraintSeed] = None,
) -> NativeSinglePassPreparedIntermediateRefinementProgram:
    """Compile prepared refinement with one production target-selection launch."""

    source = compile_native_single_pass_target_admission_program(
        module,
        input_spec,
        policy=policy,
        plan_id=plan_id,
        multi_pass_policy=multi_pass_policy,
        relu_split_state=relu_split_state,
        linear_spec_C=linear_spec_C,
        source_refinement_execution=source_refinement_execution,
        external_constraint_seed=external_constraint_seed,
    )
    receipt = source.target_admission_receipt
    capsule = _prepared_capsule(
        source, target_admission_receipt_hash=receipt.stable_hash()
    )
    prepared_task_module, prepared_schedule = lower_native_prepared_refinement_ir(
        capsule
    )
    program = NativeSinglePassPreparedIntermediateRefinementProgram(
        plan=source.plan,
        task_module=source.task_module,
        schedule=source.schedule,
        initial_interval_env=source.initial_interval_env,
        initial_relu_pre=source.initial_relu_pre,
        split_state=source.split_state,
        objective=source.objective,
        objective_influence=source.objective_influence,
        source_intermediate_constraints=source.source_intermediate_constraints,
        external_constraint_seed=source.external_constraint_seed,
        capsule=capsule,
        prepared_task_module=prepared_task_module,
        prepared_schedule=prepared_schedule,
        admitted_input_spec=input_spec,
        target_admission_receipt=receipt,
        target_admission_task_module=source.target_admission_task_module,
        target_admission_schedule=source.target_admission_schedule,
    )
    shallow_roots, deep_roots = _program_runtime_roots(program, module)
    program = replace(
        program,
        runtime_receipt=_build_runtime_receipt(
            shallow_roots=shallow_roots, deep_roots=deep_roots
        ),
    )
    program.validate(module, input_spec)
    return program


def execute_native_prepared_intermediate_refinement_program(
    program: NativePreparedIntermediateRefinementProgram,
    module: BFTaskModule,
    input_spec: InputSpec,
) -> NativePreparedIntermediateRefinementExecution:
    """Execute the frozen semantics and publish a cheap validated result receipt."""

    source = execute_native_intermediate_refinement_program(program, module, input_spec)
    prepared_trace = NativePreparedRefinementExecutionTraceIR(
        capsule_hash=program.capsule.stable_hash(),
        task_module_hash=program.prepared_task_module.stable_hash(
            capsule=program.capsule
        ),
        schedule_hash=program.prepared_schedule.stable_hash(
            capsule=program.capsule, task_module=program.prepared_task_module
        ),
        source_execution_semantic_trace_hash=(
            intermediate_refinement_semantic_trace_hash(source)
        ),
        final_intermediate_bounds_hash=source.trace.final_intermediate_bounds_hash,
        full_validation_count=1,
        runtime_target_selection_count=1,
    )
    prepared_trace.validate(
        capsule=program.capsule,
        task_module=program.prepared_task_module,
        schedule=program.prepared_schedule,
    )
    execution = NativePreparedIntermediateRefinementExecution(
        program=program,
        interval_env=source.interval_env,
        relu_pre=source.relu_pre,
        trace=source.trace,
        prepared_trace=prepared_trace,
    )
    shallow_roots, deep_roots = _execution_runtime_roots(execution)
    execution = replace(
        execution,
        runtime_receipt=_build_runtime_receipt(
            shallow_roots=shallow_roots, deep_roots=deep_roots
        ),
    )
    execution.validate(module, input_spec)
    return execution


def validate_native_prepared_intermediate_refinement_full(
    execution: NativePreparedIntermediateRefinementExecution,
    module: BFTaskModule,
    input_spec: InputSpec,
) -> None:
    """Replay-grade validation that explicitly bypasses prepared fast paths."""

    if not isinstance(execution, NativePreparedIntermediateRefinementExecution):
        raise TypeError("prepared refinement execution is required")
    if not isinstance(execution.program, NativePreparedIntermediateRefinementProgram):
        raise TypeError("prepared refinement program is required")
    execution.program.validate_full(module, input_spec)
    execution.validate_full(module, input_spec)


__all__ = [
    "NativePreparedIntermediateRefinementExecution",
    "NativePreparedIntermediateRefinementProgram",
    "NativeSinglePassPreparedIntermediateRefinementProgram",
    "compile_native_prepared_intermediate_refinement_program",
    "compile_native_single_pass_prepared_intermediate_refinement_program",
    "execute_native_prepared_intermediate_refinement_program",
    "validate_native_prepared_intermediate_refinement_full",
]
