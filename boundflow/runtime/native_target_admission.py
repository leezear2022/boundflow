"""Single-pass exact target selection with replay-grade admission ownership."""

# pylint: disable=protected-access,too-many-arguments,too-many-boolean-expressions
# pylint: disable=too-many-locals,too-many-branches,too-many-statements
# pylint: disable=line-too-long

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from typing import Mapping, Optional

import torch

from ..domains.interval import IntervalState
from ..frontends.plain_crown_bound_ir import (
    plain_crown_primal_graph_hash,
    relu_split_state_hash,
    tensor_content_hash,
)
from ..ir.refinement import (
    NativeIntermediateRefinementPlanIR,
    NativeIntermediateRefinementMultiPassPolicyIR,
    NativeIntermediateRefinementPolicyIR,
    lower_native_intermediate_refinement_ir,
)
from ..ir.target_admission import (
    NativeTargetAdmissionReceiptIR,
    NativeTargetAdmissionScheduleIR,
    NativeTargetAdmissionTaskIRModule,
    lower_native_target_admission_ir,
)
from ..ir.task import BFTaskModule
from .crown_ibp import (
    _forward_ibp_trace_mlp,
    run_crown_ibp_mlp_with_relu_influence_from_forward_trace,
)
from .native_intermediate_refinement import (
    NativeExternalIntermediateConstraintSeed,
    NativeIntermediateRefinementExecution,
    NativeIntermediateRefinementProgram,
    _clone_bounds,
    _input_bounds_hash,
    _multi_pass_selection_policy,
    _select_targets,
    _targets_hash,
    _validate_monotonic_bounds,
    _validate_split_state,
    _zero_split_state,
    intermediate_bounds_hash,
    intermediate_refinement_semantic_trace_hash,
)
from .task_executor import InputSpec


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validate_program_identity(
    program: NativeIntermediateRefinementProgram,
    module: BFTaskModule,
    input_spec: InputSpec,
) -> None:
    """Validate materialized program identity without replaying target selection."""

    program.schedule.validate(plan=program.plan, task_module=program.task_module)
    if (
        program.plan.primal_graph_hash != plain_crown_primal_graph_hash(module)
        or program.plan.input_bounds_hash != _input_bounds_hash(input_spec)
        or program.plan.split_state_hash != relu_split_state_hash(program.split_state)
        or program.plan.initial_intermediate_bounds_hash
        != intermediate_bounds_hash(program.initial_relu_pre)
    ):
        raise ValueError("native intermediate refinement program identity differs")
    source_present = program.plan.source_intermediate_constraints_hash is not None
    if source_present != (program.source_intermediate_constraints is not None) or (
        program.source_intermediate_constraints is not None
        and program.plan.source_intermediate_constraints_hash
        != intermediate_bounds_hash(program.source_intermediate_constraints)
    ):
        raise ValueError("native refinement source constraints differ")
    seed_present = program.plan.external_constraint_seed is not None
    if seed_present != (program.external_constraint_seed is not None):
        raise ValueError("native refinement external seed presence differs")
    if program.external_constraint_seed is not None:
        program.external_constraint_seed.validate(module, input_spec)
        if (
            program.plan.external_constraint_seed != program.external_constraint_seed.ir
            or program.source_intermediate_constraints is not None
        ):
            raise ValueError("native refinement external seed identity differs")
    materialization_constraints = (
        program.external_constraint_seed.constraints
        if program.external_constraint_seed is not None
        else program.source_intermediate_constraints
    )
    expected_env, expected_pre = _forward_ibp_trace_mlp(
        module,
        input_spec,
        relu_split_state=dict(program.split_state),
        relu_pre_constraints=materialization_constraints,
    )
    if intermediate_bounds_hash(expected_env) != intermediate_bounds_hash(
        program.initial_interval_env
    ) or intermediate_bounds_hash(expected_pre) != intermediate_bounds_hash(
        program.initial_relu_pre
    ):
        raise ValueError("native refinement materialized forward state differs")
    if materialization_constraints is not None:
        _local_env, local_pre = _forward_ibp_trace_mlp(
            module,
            input_spec,
            relu_split_state=dict(program.split_state),
        )
        _validate_monotonic_bounds(
            local_pre,
            program.initial_relu_pre,
            caller="native refinement initial-constraint intersection",
        )
    _validate_split_state(program.split_state, program.initial_relu_pre)
    objective_directed = program.plan.objective_hash is not None
    if objective_directed != (program.objective is not None) or objective_directed != (
        program.objective_influence is not None
    ):
        raise ValueError("native refinement program objective semantics differ")
    if (
        program.objective is not None
        and program.plan.objective_hash != tensor_content_hash(program.objective)
    ):
        raise ValueError("native refinement program objective hash differs")


def _compile_target_admission_source_unvalidated(
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
) -> NativeIntermediateRefinementProgram:
    """Build one exact selector result without invoking the legacy replay validator."""

    if not plan_id:
        raise ValueError("native intermediate refinement plan ID must be non-empty")
    module.validate()
    policy.validate()
    selection_policy = policy
    if multi_pass_policy is not None:
        if not isinstance(
            multi_pass_policy, NativeIntermediateRefinementMultiPassPolicyIR
        ):
            raise TypeError("native refinement multi-pass policy is invalid")
        multi_pass_policy.validate()
        if policy.passes != multi_pass_policy.maximum_passes:
            raise ValueError("native refinement multi-pass count differs")
        selection_policy = _multi_pass_selection_policy(
            policy, multi_pass_policy, pass_index=0
        )
    source_constraints: Optional[Mapping[str, IntervalState]] = None
    source_refinement_plan_hash: Optional[str] = None
    source_refinement_semantic_trace_hash: Optional[str] = None
    admitted_external_seed: Optional[NativeExternalIntermediateConstraintSeed] = None
    if source_refinement_execution is not None and external_constraint_seed is not None:
        raise ValueError("native refinement external and ancestral sources conflict")
    if external_constraint_seed is not None:
        if not isinstance(
            external_constraint_seed, NativeExternalIntermediateConstraintSeed
        ):
            raise TypeError("native refinement external constraint seed is invalid")
        external_constraint_seed.validate(module, input_spec)
        admitted_external_seed = NativeExternalIntermediateConstraintSeed(
            ir=external_constraint_seed.ir,
            constraints=_clone_bounds(external_constraint_seed.constraints),
        )
    if source_refinement_execution is not None:
        if not isinstance(
            source_refinement_execution, NativeIntermediateRefinementExecution
        ):
            raise TypeError("native refinement source execution is invalid")
        source_refinement_execution.validate(module, input_spec)
        source_constraints = _clone_bounds(source_refinement_execution.relu_pre)
        source_refinement_plan_hash = (
            source_refinement_execution.program.plan.stable_hash()
        )
        source_refinement_semantic_trace_hash = (
            intermediate_refinement_semantic_trace_hash(source_refinement_execution)
        )
    initial_env, unsplit_pre = _forward_ibp_trace_mlp(module, input_spec)
    splits = (
        _zero_split_state(unsplit_pre)
        if relu_split_state is None
        else {
            name: value.detach().contiguous().clone()
            for name, value in relu_split_state.items()
        }
    )
    _validate_split_state(splits, unsplit_pre)
    materialization_constraints = (
        admitted_external_seed.constraints
        if admitted_external_seed is not None
        else source_constraints
    )
    if relu_split_state is not None or materialization_constraints is not None:
        initial_env, initial_pre = _forward_ibp_trace_mlp(
            module,
            input_spec,
            relu_split_state=dict(splits),
            relu_pre_constraints=materialization_constraints,
        )
    else:
        initial_pre = unsplit_pre
    objective_directed = (
        policy.candidate_policy_id == "objective_influence_width_per_relu_v1"
    )
    if objective_directed != (linear_spec_C is not None):
        raise ValueError("native refinement policy/objective admission differs")
    objective = None
    objective_influence = None
    objective_hash = None
    if linear_spec_C is not None:
        if (
            not torch.is_tensor(linear_spec_C)
            or not torch.is_floating_point(linear_spec_C)
            or linear_spec_C.dim() not in {2, 3}
            or int(linear_spec_C.shape[-2]) != 1
            or (linear_spec_C.dim() == 3 and int(linear_spec_C.shape[0]) != 1)
            or not bool(torch.isfinite(linear_spec_C).all())
        ):
            raise ValueError(
                "native objective-directed refinement requires one finite scalar clause"
            )
        objective = linear_spec_C.detach().contiguous().clone()
        _objective_bounds, objective_influence = (
            run_crown_ibp_mlp_with_relu_influence_from_forward_trace(
                module,
                input_spec,
                interval_env=dict(initial_env),
                relu_pre=dict(initial_pre),
                linear_spec_C=objective,
            )
        )
        objective_hash = tensor_content_hash(objective)
    targets = _select_targets(
        initial_pre, selection_policy, objective_influence=objective_influence
    )
    plan = NativeIntermediateRefinementPlanIR(
        plan_id=plan_id,
        primal_graph_hash=plain_crown_primal_graph_hash(module),
        input_bounds_hash=_input_bounds_hash(input_spec),
        split_state_hash=relu_split_state_hash(splits),
        initial_intermediate_bounds_hash=intermediate_bounds_hash(initial_pre),
        policy=policy,
        targets=targets,
        multi_pass_policy=multi_pass_policy,
        objective_hash=objective_hash,
        source_intermediate_constraints_hash=(
            None
            if source_constraints is None
            else intermediate_bounds_hash(source_constraints)
        ),
        source_refinement_plan_hash=source_refinement_plan_hash,
        source_refinement_semantic_trace_hash=source_refinement_semantic_trace_hash,
        external_constraint_seed=(
            None if admitted_external_seed is None else admitted_external_seed.ir
        ),
    )
    task_module, schedule = lower_native_intermediate_refinement_ir(plan)
    return NativeIntermediateRefinementProgram(
        plan=plan,
        task_module=task_module,
        schedule=schedule,
        initial_interval_env=initial_env,
        initial_relu_pre=initial_pre,
        split_state=splits,
        objective=objective,
        objective_influence=objective_influence,
        source_intermediate_constraints=source_constraints,
        external_constraint_seed=admitted_external_seed,
    )


def _objective_influence_hash(
    program: NativeIntermediateRefinementProgram,
) -> Optional[str]:
    values = program.objective_influence
    if values is None:
        return None
    return _canonical_hash(
        {
            "derivation": "plain-crown-objective-influence/v1",
            "primal_graph_hash": program.plan.primal_graph_hash,
            "initial_intermediate_bounds_hash": (
                program.plan.initial_intermediate_bounds_hash
            ),
            "objective_hash": program.plan.objective_hash,
            "values": [
                {
                    "value_id": name,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "device": str(value.device),
                }
                for name, value in values.items()
            ],
        }
    )


def _effective_policy_hash(program: NativeIntermediateRefinementProgram) -> str:
    selection_policy = program.plan.policy
    if program.plan.multi_pass_policy is not None:
        selection_policy = _multi_pass_selection_policy(
            program.plan.policy, program.plan.multi_pass_policy, pass_index=0
        )
    return selection_policy.stable_hash()


def _objective_influence_versions(
    values: Optional[Mapping[str, torch.Tensor]],
) -> tuple[tuple[str, torch.Tensor, int], ...]:
    if values is None:
        return ()
    return tuple(
        (name, value, int(value._version))  # pylint: disable=protected-access
        for name, value in values.items()
    )


def _validate_objective_influence_versions(
    program: NativeIntermediateRefinementProgram,
    versions: tuple[tuple[str, torch.Tensor, int], ...],
) -> None:
    values = program.objective_influence
    if values is None:
        if versions:
            raise ValueError("single-pass target influence witness differs")
        return
    if tuple(values) != tuple(name for name, _tensor, _version in versions):
        raise ValueError("single-pass target influence witness differs")
    for name, tensor, version in versions:
        if values[name] is not tensor or int(tensor._version) != version:
            raise ValueError("single-pass target influence witness differs")


def _build_target_admission_receipt(
    program: NativeIntermediateRefinementProgram,
) -> NativeTargetAdmissionReceiptIR:
    receipt = NativeTargetAdmissionReceiptIR(
        receipt_id=f"{program.plan.plan_id}:target-admission-v1",
        plan_id=program.plan.plan_id,
        primal_graph_hash=program.plan.primal_graph_hash,
        input_bounds_hash=program.plan.input_bounds_hash,
        split_state_hash=program.plan.split_state_hash,
        initial_intermediate_bounds_hash=(
            program.plan.initial_intermediate_bounds_hash
        ),
        effective_policy_hash=_effective_policy_hash(program),
        objective_hash=program.plan.objective_hash,
        objective_influence_hash=_objective_influence_hash(program),
        target_table_hash=_targets_hash(program.plan.targets),
        target_count=len(program.plan.targets),
        admission_receipt_hash="0" * 64,
    )
    receipt = replace(receipt, admission_receipt_hash=receipt.expected_receipt_hash())
    receipt.validate()
    return receipt


def validate_native_target_admission_binding(
    program: NativeIntermediateRefinementProgram,
    *,
    receipt: NativeTargetAdmissionReceiptIR,
    task_module: NativeTargetAdmissionTaskIRModule,
    schedule: NativeTargetAdmissionScheduleIR,
) -> None:
    """Validate receipt and admission stack against an exact source Program."""

    receipt.validate()
    if (
        receipt.plan_id != program.plan.plan_id
        or receipt.primal_graph_hash != program.plan.primal_graph_hash
        or receipt.input_bounds_hash != program.plan.input_bounds_hash
        or receipt.split_state_hash != program.plan.split_state_hash
        or receipt.initial_intermediate_bounds_hash
        != program.plan.initial_intermediate_bounds_hash
        or receipt.effective_policy_hash != _effective_policy_hash(program)
        or receipt.objective_hash != program.plan.objective_hash
        or receipt.objective_influence_hash != _objective_influence_hash(program)
        or receipt.target_table_hash != _targets_hash(program.plan.targets)
        or receipt.target_count != len(program.plan.targets)
        or task_module.source_plan_hash != program.plan.stable_hash()
    ):
        raise ValueError("single-pass target admission receipt differs")
    task_module.validate(receipt=receipt)
    schedule.validate(receipt=receipt, task_module=task_module)


def validate_native_target_admission_structure(
    program: NativeIntermediateRefinementProgram,
    *,
    receipt: NativeTargetAdmissionReceiptIR,
    task_module: NativeTargetAdmissionTaskIRModule,
    schedule: NativeTargetAdmissionScheduleIR,
) -> None:
    """Validate immutable receipt linkage after content was admitted once."""

    receipt.validate()
    if (
        receipt.plan_id != program.plan.plan_id
        or receipt.primal_graph_hash != program.plan.primal_graph_hash
        or receipt.input_bounds_hash != program.plan.input_bounds_hash
        or receipt.split_state_hash != program.plan.split_state_hash
        or receipt.initial_intermediate_bounds_hash
        != program.plan.initial_intermediate_bounds_hash
        or receipt.effective_policy_hash != _effective_policy_hash(program)
        or receipt.objective_hash != program.plan.objective_hash
        or receipt.target_count != len(program.plan.targets)
        or task_module.source_plan_hash != program.plan.stable_hash()
    ):
        raise ValueError("single-pass target admission structure differs")
    task_module.validate(receipt=receipt)
    schedule.validate(receipt=receipt, task_module=task_module)


@dataclass(frozen=True)
class NativeSinglePassTargetAdmissionProgram(NativeIntermediateRefinementProgram):
    """Exact refinement Program whose production validator consumes a receipt."""

    target_admission_receipt: NativeTargetAdmissionReceiptIR = None  # type: ignore[assignment]
    target_admission_task_module: NativeTargetAdmissionTaskIRModule = None  # type: ignore[assignment]
    target_admission_schedule: NativeTargetAdmissionScheduleIR = None  # type: ignore[assignment]
    objective_influence_versions: tuple[tuple[str, torch.Tensor, int], ...] = ()

    def _validate_target_admission(self) -> None:
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
        ):
            raise ValueError("single-pass target admission ownership is absent")
        validate_native_target_admission_binding(
            self,
            receipt=self.target_admission_receipt,
            task_module=self.target_admission_task_module,
            schedule=self.target_admission_schedule,
        )

    def validate(self, module: BFTaskModule, input_spec: InputSpec) -> None:
        _validate_program_identity(self, module, input_spec)
        validate_native_target_admission_structure(
            self,
            receipt=self.target_admission_receipt,
            task_module=self.target_admission_task_module,
            schedule=self.target_admission_schedule,
        )
        _validate_objective_influence_versions(self, self.objective_influence_versions)

    def validate_full(self, module: BFTaskModule, input_spec: InputSpec) -> None:
        """Recompute exact target selection and compare it with the receipt."""

        NativeIntermediateRefinementProgram.validate(self, module, input_spec)
        self._validate_target_admission()


def compile_native_single_pass_target_admission_program(
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
) -> NativeSinglePassTargetAdmissionProgram:
    """Compile once, then admit the exact selector result without reselecting."""

    source = _compile_target_admission_source_unvalidated(
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
    receipt = _build_target_admission_receipt(source)
    task_module, schedule = lower_native_target_admission_ir(
        source_plan_hash=source.plan.stable_hash(), receipt=receipt
    )
    program = NativeSinglePassTargetAdmissionProgram(
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
        target_admission_receipt=receipt,
        target_admission_task_module=task_module,
        target_admission_schedule=schedule,
        objective_influence_versions=_objective_influence_versions(
            source.objective_influence
        ),
    )
    _validate_program_identity(program, module, input_spec)
    validate_native_target_admission_structure(
        program,
        receipt=receipt,
        task_module=task_module,
        schedule=schedule,
    )
    return program


def admit_native_intermediate_refinement_program_targets(
    source: NativeIntermediateRefinementProgram,
    module: BFTaskModule,
    input_spec: InputSpec,
) -> NativeSinglePassTargetAdmissionProgram:
    """Admit an existing exact Program once without replaying its selector."""

    if not isinstance(source, NativeIntermediateRefinementProgram):
        raise TypeError("native intermediate refinement Program is required")
    _validate_program_identity(source, module, input_spec)
    receipt = _build_target_admission_receipt(source)
    task_module, schedule = lower_native_target_admission_ir(
        source_plan_hash=source.plan.stable_hash(), receipt=receipt
    )
    program = NativeSinglePassTargetAdmissionProgram(
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
        target_admission_receipt=receipt,
        target_admission_task_module=task_module,
        target_admission_schedule=schedule,
        objective_influence_versions=_objective_influence_versions(
            source.objective_influence
        ),
    )
    validate_native_target_admission_structure(
        program,
        receipt=receipt,
        task_module=task_module,
        schedule=schedule,
    )
    return program


def admit_native_intermediate_refinement_execution_targets(
    source: NativeIntermediateRefinementExecution,
    module: BFTaskModule,
    input_spec: InputSpec,
) -> NativeIntermediateRefinementExecution:
    """Bind one existing execution to a target receipt for child consumption."""

    if not isinstance(source, NativeIntermediateRefinementExecution):
        raise TypeError("native intermediate refinement execution is required")
    program = admit_native_intermediate_refinement_program_targets(
        source.program, module, input_spec
    )
    execution = NativeIntermediateRefinementExecution(
        program=program,
        interval_env=source.interval_env,
        relu_pre=source.relu_pre,
        trace=source.trace,
    )
    execution.validate(module, input_spec)
    return execution


def validate_native_single_pass_target_admission_full(
    program: NativeSinglePassTargetAdmissionProgram,
    module: BFTaskModule,
    input_spec: InputSpec,
) -> None:
    """Replay selector semantics rather than trusting the admission receipt."""

    if not isinstance(program, NativeSinglePassTargetAdmissionProgram):
        raise TypeError("single-pass target admission Program is required")
    program.validate_full(module, input_spec)


__all__ = [
    "NativeSinglePassTargetAdmissionProgram",
    "admit_native_intermediate_refinement_execution_targets",
    "admit_native_intermediate_refinement_program_targets",
    "compile_native_single_pass_target_admission_program",
    "validate_native_target_admission_binding",
    "validate_native_target_admission_structure",
    "validate_native_single_pass_target_admission_full",
]
