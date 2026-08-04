"""Schedule-driven native intermediate pre-activation refinement."""

# pylint: disable=too-many-instance-attributes,too-many-locals,too-many-statements
# pylint: disable=too-many-arguments,too-many-branches,missing-function-docstring
# pylint: disable=too-many-boolean-expressions,too-many-return-statements
# pylint: disable=too-many-lines

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
import time
from typing import Mapping, Optional

import torch

from ..domains.interval import IntervalState
from ..frontends.plain_crown_bound_ir import (
    plain_crown_primal_graph_hash,
    relu_split_state_hash,
    tensor_content_hash,
)
from ..ir.refinement import (
    ExternalIntermediateConstraintSeedIR,
    IntermediateRefinementTaskKind,
    NativeIntermediateRefinementPlanIR,
    NativeIntermediateRefinementMultiPassPolicyIR,
    NativeIntermediateRefinementPassDecisionIR,
    NativeIntermediateRefinementPolicyIR,
    NativeIntermediateRefinementScheduleIR,
    NativeIntermediateRefinementTargetIR,
    NativeIntermediateRefinementTaskIRModule,
    lower_native_intermediate_refinement_ir,
)
from ..ir.task import BFTaskModule
from .crown_ibp import (
    _forward_ibp_trace_mlp,
    run_crown_ibp_mlp_from_forward_trace,
    run_crown_ibp_mlp_with_relu_influence_from_forward_trace,
)
from .task_executor import InputSpec

NATIVE_INTERMEDIATE_REFINEMENT_EXECUTION_SCHEMA_VERSION = (
    "boundflow.native-intermediate-refinement-execution/v1"
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


def _tensor_schema(value: torch.Tensor) -> dict[str, object]:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "content_hash": tensor_content_hash(value),
    }


def intermediate_bounds_hash(values: Mapping[str, IntervalState]) -> str:
    """Stable identity for a complete ordered ReLU pre-activation mapping."""

    if not values:
        raise ValueError("intermediate bounds mapping must be non-empty")
    payload: list[dict[str, object]] = []
    for name, value in values.items():
        if (
            not name
            or not isinstance(value, IntervalState)
            or value.lower.shape != value.upper.shape
            or not torch.is_floating_point(value.lower)
            or value.lower.dtype != value.upper.dtype
            or value.lower.device != value.upper.device
            or not bool(torch.isfinite(value.lower).all())
            or not bool(torch.isfinite(value.upper).all())
            or not bool((value.lower <= value.upper).all())
        ):
            raise ValueError("intermediate bounds mapping schema differs")
        payload.append(
            {
                "relu_input": name,
                "lower": _tensor_schema(value.lower),
                "upper": _tensor_schema(value.upper),
            }
        )
    return _canonical_hash(payload)


def _input_bounds_hash(input_spec: InputSpec) -> str:
    lower, upper = input_spec.perturbation.bounding_box(input_spec.center)
    return _canonical_hash(
        {
            "value_name": input_spec.value_name,
            "center": _tensor_schema(input_spec.center),
            "lower": _tensor_schema(lower),
            "upper": _tensor_schema(upper),
        }
    )


def _clone_bounds(values: Mapping[str, IntervalState]) -> dict[str, IntervalState]:
    return {
        name: IntervalState(
            lower=value.lower.detach().contiguous().clone(),
            upper=value.upper.detach().contiguous().clone(),
        )
        for name, value in values.items()
    }


@dataclass(frozen=True)
class NativeExternalIntermediateConstraintSeed:
    """Typed external provenance plus constraints bound to the local graph."""

    ir: ExternalIntermediateConstraintSeedIR
    constraints: Mapping[str, IntervalState]

    def validate_payload(self) -> None:
        self.ir.validate()
        if self.ir.bound_intermediate_constraints_hash != intermediate_bounds_hash(
            self.constraints
        ):
            raise ValueError("external intermediate constraint seed content differs")

    def validate(self, module: BFTaskModule, input_spec: InputSpec) -> None:
        self.validate_payload()
        if self.ir.primal_graph_hash != plain_crown_primal_graph_hash(
            module
        ) or self.ir.input_bounds_hash != _input_bounds_hash(input_spec):
            raise ValueError("external intermediate constraint seed scope differs")
        _local_env, local_pre = _forward_ibp_trace_mlp(module, input_spec)
        _validate_monotonic_bounds(
            local_pre,
            self.constraints,
            caller="external intermediate constraint seed",
        )

    def stable_hash(self) -> str:
        self.validate_payload()
        return self.ir.stable_hash()


def build_native_external_intermediate_constraint_seed(
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    seed_id: str,
    provider: str,
    constraints: Mapping[str, IntervalState],
    external_intermediate_bounds_hash: str,
    source_artifact_manifest_hash: str,
    source_artifact_payload_hash: str,
    source_model_hash: str,
    source_property_hash: str,
    source_objective_set_hash: str,
) -> NativeExternalIntermediateConstraintSeed:
    """Build and validate one external-owned seed for an exact local scope."""

    external = _clone_bounds(constraints)
    _local_env, local = _forward_ibp_trace_mlp(module, input_spec)
    if tuple(external) != tuple(local):
        raise ValueError("external intermediate constraint identities differ")
    effective: dict[str, IntervalState] = {}
    for name, local_value in local.items():
        external_value = external[name]
        if (
            external_value.lower.shape != local_value.lower.shape
            or external_value.upper.shape != local_value.upper.shape
            or external_value.lower.dtype != local_value.lower.dtype
            or external_value.upper.dtype != local_value.upper.dtype
            or external_value.lower.device != local_value.lower.device
            or external_value.upper.device != local_value.upper.device
        ):
            raise ValueError("external intermediate constraint schema differs")
        lower = torch.maximum(local_value.lower, external_value.lower)
        upper = torch.minimum(local_value.upper, external_value.upper)
        if bool((lower > upper).any()):
            raise ValueError(
                "external intermediate constraint intersection is infeasible"
            )
        effective[name] = IntervalState(
            lower=lower.detach().contiguous().clone(),
            upper=upper.detach().contiguous().clone(),
        )
    seed = NativeExternalIntermediateConstraintSeed(
        ir=ExternalIntermediateConstraintSeedIR(
            seed_id=seed_id,
            provider=provider,
            primal_graph_hash=plain_crown_primal_graph_hash(module),
            input_bounds_hash=_input_bounds_hash(input_spec),
            external_intermediate_bounds_hash=external_intermediate_bounds_hash,
            bound_intermediate_constraints_hash=intermediate_bounds_hash(effective),
            source_artifact_manifest_hash=source_artifact_manifest_hash,
            source_artifact_payload_hash=source_artifact_payload_hash,
            source_model_hash=source_model_hash,
            source_property_hash=source_property_hash,
            source_objective_set_hash=source_objective_set_hash,
        ),
        constraints=effective,
    )
    seed.validate(module, input_spec)
    return seed


def _zero_split_state(
    values: Mapping[str, IntervalState],
) -> dict[str, torch.Tensor]:
    return {
        name: torch.zeros_like(value.lower, dtype=torch.int8)
        for name, value in values.items()
    }


def _validate_split_state(
    split_state: Mapping[str, torch.Tensor],
    relu_pre: Mapping[str, IntervalState],
) -> None:
    if tuple(split_state) != tuple(relu_pre):
        raise ValueError("native refinement split/ReLU identities differ")
    for name, value in split_state.items():
        if (
            not torch.is_tensor(value)
            or value.dtype != torch.int8
            or value.device != relu_pre[name].lower.device
            or value.shape != relu_pre[name].lower.shape
            or not bool(((value >= -1) & (value <= 1)).all())
        ):
            raise ValueError("native refinement split state schema differs")


def _enumerate_ambiguous(
    relu_pre: Mapping[str, IntervalState],
) -> dict[str, tuple[int, ...]]:
    candidates: dict[str, tuple[int, ...]] = {}
    for name, value in relu_pre.items():
        if int(value.lower.shape[0]) != 1:
            raise ValueError("native refinement v1 requires one source domain")
        lower = value.lower.reshape(-1)
        upper = value.upper.reshape(-1)
        candidates[name] = tuple(
            int(index)
            for index in torch.nonzero((lower < 0.0) & (upper > 0.0), as_tuple=False)
            .reshape(-1)
            .tolist()
        )
    return candidates


def _select_targets(
    relu_pre: Mapping[str, IntervalState],
    policy: NativeIntermediateRefinementPolicyIR,
    *,
    objective_influence: Optional[Mapping[str, torch.Tensor]] = None,
    excluded_target_identities: frozenset[tuple[str, int]] = frozenset(),
    allow_empty: bool = False,
) -> tuple[NativeIntermediateRefinementTargetIR, ...]:
    policy.validate()
    objective_directed = (
        policy.candidate_policy_id == "objective_influence_width_per_relu_v1"
    )
    if objective_directed != (objective_influence is not None):
        raise ValueError("native refinement target policy/influence differs")
    if objective_influence is not None and tuple(objective_influence) != tuple(
        relu_pre
    ):
        raise ValueError("native refinement influence identities differ")
    targets: list[NativeIntermediateRefinementTargetIR] = []
    for name, value in relu_pre.items():
        if int(value.lower.shape[0]) != 1:
            raise ValueError("native refinement v1 requires one source domain")
        lower = value.lower.reshape(-1)
        upper = value.upper.reshape(-1)
        influence = (
            None
            if objective_influence is None
            else objective_influence[name].reshape(-1)
        )
        if influence is not None and (
            influence.shape != lower.shape
            or influence.dtype != lower.dtype
            or influence.device != lower.device
            or not bool(torch.isfinite(influence).all())
            or bool((influence < 0.0).any())
        ):
            raise ValueError("native refinement influence tensor schema differs")
        indices = _enumerate_ambiguous({name: value})[name]
        ranked = sorted(
            (
                (
                    -float(
                        (
                            (upper[index] - lower[index])
                            if influence is None
                            else (upper[index] - lower[index]) * influence[index]
                        ).item()
                    ),
                    int(index),
                )
                for index in indices
                if (name, int(index)) not in excluded_target_identities
                if float((upper[index] - lower[index]).item()) >= policy.minimum_width
            )
        )[: policy.max_neurons_per_relu]
        for _negative_width, index in ranked:
            initial_lower = float(lower[index].item())
            initial_upper = float(upper[index].item())
            objective_influence_value = (
                None if influence is None else float(influence[index].item())
            )
            targets.append(
                NativeIntermediateRefinementTargetIR(
                    ordinal=len(targets),
                    relu_input=name,
                    neuron_index=index,
                    initial_lower=initial_lower,
                    initial_upper=initial_upper,
                    initial_width=initial_upper - initial_lower,
                    objective_influence=objective_influence_value,
                    selection_score=(
                        None
                        if objective_influence_value is None
                        else objective_influence_value * (initial_upper - initial_lower)
                    ),
                )
            )
    if not targets and not allow_empty:
        raise ValueError("native refinement found no eligible ambiguous ReLU")
    result = tuple(targets)
    for target in result:
        target.validate()
    return result


def _multi_pass_selection_policy(
    policy: NativeIntermediateRefinementPolicyIR,
    multi_pass_policy: NativeIntermediateRefinementMultiPassPolicyIR,
    *,
    pass_index: int,
) -> NativeIntermediateRefinementPolicyIR:
    pass_cap = multi_pass_policy.pass_target_cap(
        total_target_cap=policy.max_neurons_per_relu,
        pass_index=pass_index,
    )
    selected = replace(policy, max_neurons_per_relu=pass_cap)
    selected.validate()
    return selected


def _target_identities(
    targets: tuple[NativeIntermediateRefinementTargetIR, ...],
) -> tuple[tuple[str, int], ...]:
    return tuple((target.relu_input, target.neuron_index) for target in targets)


def _target_ledger_hash(ledger: tuple[tuple[str, int], ...]) -> str:
    return _canonical_hash(
        [
            {"relu_input": relu_input, "neuron_index": neuron_index}
            for relu_input, neuron_index in ledger
        ]
    )


def _targets_hash(
    targets: tuple[NativeIntermediateRefinementTargetIR, ...],
) -> str:
    return _canonical_hash([target.to_dict() for target in targets])


@dataclass(frozen=True)
class NativeIntermediateRefinementProgram:
    """Cross-linked refinement IR and exact initial runtime state."""

    plan: NativeIntermediateRefinementPlanIR
    task_module: NativeIntermediateRefinementTaskIRModule
    schedule: NativeIntermediateRefinementScheduleIR
    initial_interval_env: Mapping[str, IntervalState]
    initial_relu_pre: Mapping[str, IntervalState]
    split_state: Mapping[str, torch.Tensor]
    objective: Optional[torch.Tensor] = None
    objective_influence: Optional[Mapping[str, torch.Tensor]] = None
    source_intermediate_constraints: Optional[Mapping[str, IntervalState]] = None
    external_constraint_seed: Optional[NativeExternalIntermediateConstraintSeed] = None

    def validate(self, module: BFTaskModule, input_spec: InputSpec) -> None:
        self.schedule.validate(plan=self.plan, task_module=self.task_module)
        if (
            self.plan.primal_graph_hash != plain_crown_primal_graph_hash(module)
            or self.plan.input_bounds_hash != _input_bounds_hash(input_spec)
            or self.plan.split_state_hash != relu_split_state_hash(self.split_state)
            or self.plan.initial_intermediate_bounds_hash
            != intermediate_bounds_hash(self.initial_relu_pre)
        ):
            raise ValueError("native intermediate refinement program identity differs")
        source_hashes_present = (
            self.plan.source_intermediate_constraints_hash is not None
        )
        if source_hashes_present != (
            self.source_intermediate_constraints is not None
        ) or (
            self.source_intermediate_constraints is not None
            and self.plan.source_intermediate_constraints_hash
            != intermediate_bounds_hash(self.source_intermediate_constraints)
        ):
            raise ValueError("native refinement source constraints differ")
        seed_present = self.plan.external_constraint_seed is not None
        if seed_present != (self.external_constraint_seed is not None):
            raise ValueError("native refinement external seed presence differs")
        if self.external_constraint_seed is not None:
            self.external_constraint_seed.validate(module, input_spec)
            if (
                self.plan.external_constraint_seed != self.external_constraint_seed.ir
                or self.source_intermediate_constraints is not None
            ):
                raise ValueError("native refinement external seed identity differs")
        materialization_constraints = (
            self.external_constraint_seed.constraints
            if self.external_constraint_seed is not None
            else self.source_intermediate_constraints
        )
        expected_env, expected_pre = _forward_ibp_trace_mlp(
            module,
            input_spec,
            relu_split_state=dict(self.split_state),
            relu_pre_constraints=materialization_constraints,
        )
        if intermediate_bounds_hash(expected_env) != intermediate_bounds_hash(
            self.initial_interval_env
        ) or intermediate_bounds_hash(expected_pre) != intermediate_bounds_hash(
            self.initial_relu_pre
        ):
            raise ValueError("native refinement materialized forward state differs")
        if materialization_constraints is not None:
            _local_env, local_pre = _forward_ibp_trace_mlp(
                module,
                input_spec,
                relu_split_state=dict(self.split_state),
            )
            _validate_monotonic_bounds(
                local_pre,
                self.initial_relu_pre,
                caller="native refinement initial-constraint intersection",
            )
        _validate_split_state(self.split_state, self.initial_relu_pre)
        objective_directed = self.plan.objective_hash is not None
        if objective_directed != (self.objective is not None) or objective_directed != (
            self.objective_influence is not None
        ):
            raise ValueError("native refinement program objective semantics differ")
        if (
            self.objective is not None
            and self.plan.objective_hash != tensor_content_hash(self.objective)
        ):
            raise ValueError("native refinement program objective hash differs")
        selection_policy = self.plan.policy
        if self.plan.multi_pass_policy is not None:
            selection_policy = _multi_pass_selection_policy(
                self.plan.policy, self.plan.multi_pass_policy, pass_index=0
            )
        if self.plan.targets != _select_targets(
            self.initial_relu_pre,
            selection_policy,
            objective_influence=self.objective_influence,
        ):
            raise ValueError("native intermediate refinement target selection differs")

    def hashes(self) -> dict[str, str]:
        return {
            "refinement_plan_hash": self.plan.stable_hash(),
            "refinement_task_module_hash": self.task_module.stable_hash(plan=self.plan),
            "refinement_schedule_hash": self.schedule.stable_hash(
                plan=self.plan, task_module=self.task_module
            ),
        }


@dataclass(frozen=True)
class NativeIntermediateRefinementPassTrace:
    """Tightening evidence for one backward/intersection/propagation pass."""

    pass_index: int
    selected_target_count: int
    tightened_neuron_count: int
    lower_improvement_max: float
    upper_improvement_max: float
    width_reduction_sum: float
    input_bounds_hash: str
    selected_crown_hash: str
    output_bounds_hash: str
    selection_decision: Optional[NativeIntermediateRefinementPassDecisionIR] = None

    def validate(self) -> None:
        if self.selection_decision is not None:
            self.selection_decision.validate()
        if (
            self.pass_index < 0
            or self.selected_target_count < 0
            or (self.selection_decision is None and self.selected_target_count < 1)
            or (
                self.selection_decision is not None
                and (
                    self.selection_decision.pass_index != self.pass_index
                    or self.selection_decision.selected_target_count
                    != self.selected_target_count
                    or self.selection_decision.input_bounds_hash
                    != self.input_bounds_hash
                )
            )
            or self.tightened_neuron_count < 0
            or not all(
                math.isfinite(value) and value >= 0.0
                for value in (
                    self.lower_improvement_max,
                    self.upper_improvement_max,
                    self.width_reduction_sum,
                )
            )
            or any(
                not _is_sha256(value)
                for value in (
                    self.input_bounds_hash,
                    self.selected_crown_hash,
                    self.output_bounds_hash,
                )
            )
            or (
                self.selected_target_count == 0
                and (
                    self.tightened_neuron_count != 0
                    or self.lower_improvement_max != 0.0
                    or self.upper_improvement_max != 0.0
                    or self.width_reduction_sum != 0.0
                    or self.input_bounds_hash != self.output_bounds_hash
                )
            )
        ):
            raise ValueError("native intermediate refinement pass trace is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        payload: dict[str, object] = {
            "pass_index": self.pass_index,
            "selected_target_count": self.selected_target_count,
            "tightened_neuron_count": self.tightened_neuron_count,
            "lower_improvement_max": self.lower_improvement_max,
            "upper_improvement_max": self.upper_improvement_max,
            "width_reduction_sum": self.width_reduction_sum,
            "input_bounds_hash": self.input_bounds_hash,
            "selected_crown_hash": self.selected_crown_hash,
            "output_bounds_hash": self.output_bounds_hash,
        }
        if self.selection_decision is not None:
            payload["selection_decision"] = self.selection_decision.to_dict()
            payload["selection_decision_hash"] = self.selection_decision.stable_hash()
        return payload


@dataclass(frozen=True)
class NativeIntermediateRefinementActionTrace:
    """Proof that one Schedule action executed with exact inputs/outputs."""

    sequence: int
    action_id: str
    task_id: str
    kind: IntermediateRefinementTaskKind
    pass_index: Optional[int]
    input_hashes: tuple[tuple[str, str], ...]
    output_hashes: tuple[tuple[str, str], ...]

    def validate(self) -> None:
        if (
            self.sequence < 0
            or not self.action_id
            or not self.task_id
            or not self.input_hashes
            or not self.output_hashes
            or len(dict(self.input_hashes)) != len(self.input_hashes)
            or len(dict(self.output_hashes)) != len(self.output_hashes)
            or any(
                not _is_sha256(value)
                for _name, value in (*self.input_hashes, *self.output_hashes)
            )
        ):
            raise ValueError("native intermediate refinement action trace is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "sequence": self.sequence,
            "action_id": self.action_id,
            "task_id": self.task_id,
            "kind": self.kind.value,
            "pass_index": self.pass_index,
            "input_hashes": dict(self.input_hashes),
            "output_hashes": dict(self.output_hashes),
        }


@dataclass(frozen=True)
class NativeIntermediateRefinementExecutionTrace:
    """Replay-grade native refinement execution evidence."""

    plan_hash: str
    task_module_hash: str
    schedule_hash: str
    action_traces: tuple[NativeIntermediateRefinementActionTrace, ...]
    pass_traces: tuple[NativeIntermediateRefinementPassTrace, ...]
    final_intermediate_bounds_hash: str
    elapsed_ns: int
    schema_version: str = NATIVE_INTERMEDIATE_REFINEMENT_EXECUTION_SCHEMA_VERSION

    def validate(self, *, program: NativeIntermediateRefinementProgram) -> None:
        hashes = program.hashes()
        if (
            self.schema_version
            != NATIVE_INTERMEDIATE_REFINEMENT_EXECUTION_SCHEMA_VERSION
            or self.plan_hash != hashes["refinement_plan_hash"]
            or self.task_module_hash != hashes["refinement_task_module_hash"]
            or self.schedule_hash != hashes["refinement_schedule_hash"]
            or len(self.action_traces) != len(program.schedule.actions)
            or len(self.pass_traces) != program.plan.policy.passes
            or not _is_sha256(self.final_intermediate_bounds_hash)
            or self.elapsed_ns < 0
        ):
            raise ValueError("native intermediate refinement execution trace differs")
        for action, task, trace in zip(
            program.schedule.actions, program.task_module.tasks, self.action_traces
        ):
            trace.validate()
            if (
                trace.sequence != action.sequence
                or trace.action_id != action.action_id
                or trace.task_id != task.task_id
                or trace.kind != task.kind
                or trace.pass_index != task.pass_index
            ):
                raise ValueError("native refinement action trace linkage differs")
        for pass_index, pass_trace in enumerate(self.pass_traces):
            pass_trace.validate()
            if pass_trace.pass_index != pass_index:
                raise ValueError("native refinement pass trace order differs")
        multi_pass_policy = program.plan.multi_pass_policy
        if multi_pass_policy is None:
            if any(trace.selection_decision is not None for trace in self.pass_traces):
                raise ValueError("native refinement legacy pass has a decision")
        else:
            select_traces = {
                trace.pass_index: trace
                for trace in self.action_traces
                if trace.kind == IntermediateRefinementTaskKind.SELECT_TARGETS
            }
            if set(select_traces) != set(range(multi_pass_policy.maximum_passes)):
                raise ValueError("native multi-pass selection trace coverage differs")
            previous_ledger_hash = _target_ledger_hash(())
            for pass_index, pass_trace in enumerate(self.pass_traces):
                decision = pass_trace.selection_decision
                select_trace = select_traces[pass_index]
                if decision is None:
                    raise ValueError("native multi-pass selection decision is absent")
                decision.validate()
                input_hashes = dict(select_trace.input_hashes)
                output_hashes = dict(select_trace.output_hashes)
                if (
                    decision.plan_hash != program.plan.stable_hash()
                    or decision.multi_pass_policy_hash
                    != multi_pass_policy.stable_hash()
                    or decision.pass_index != pass_index
                    or decision.total_target_cap_per_relu
                    != program.plan.policy.max_neurons_per_relu
                    or decision.pass_target_cap_per_relu
                    != multi_pass_policy.pass_target_cap(
                        total_target_cap=program.plan.policy.max_neurons_per_relu,
                        pass_index=pass_index,
                    )
                    or decision.prior_target_ledger_hash != previous_ledger_hash
                    or input_hashes.get(f"refine.bounds.p{pass_index}")
                    != decision.input_bounds_hash
                    or input_hashes.get(f"refine.target_ledger.p{pass_index}")
                    != decision.prior_target_ledger_hash
                    or output_hashes.get(f"refine.selected_targets.p{pass_index}")
                    != decision.selected_targets_hash
                    or output_hashes.get(f"refine.target_ledger.p{pass_index + 1}")
                    != decision.result_target_ledger_hash
                    or output_hashes.get(f"refine.pass_decision.p{pass_index}")
                    != decision.stable_hash()
                ):
                    raise ValueError("native multi-pass selection lineage differs")
                previous_ledger_hash = decision.result_target_ledger_hash

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "plan_hash": self.plan_hash,
            "task_module_hash": self.task_module_hash,
            "schedule_hash": self.schedule_hash,
            "action_traces": [trace.to_dict() for trace in self.action_traces],
            "pass_traces": [trace.to_dict() for trace in self.pass_traces],
            "final_intermediate_bounds_hash": self.final_intermediate_bounds_hash,
            "elapsed_ns": self.elapsed_ns,
            "performance_claimed": False,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeIntermediateRefinementExecution:
    """Refined native bounds plus the IR and trace that produced them."""

    program: NativeIntermediateRefinementProgram
    interval_env: Mapping[str, IntervalState]
    relu_pre: Mapping[str, IntervalState]
    trace: NativeIntermediateRefinementExecutionTrace

    def validate(self, module: BFTaskModule, input_spec: InputSpec) -> None:
        self.program.validate(module, input_spec)
        self.trace.validate(program=self.program)
        if self.trace.final_intermediate_bounds_hash != intermediate_bounds_hash(
            self.relu_pre
        ):
            raise ValueError("native refinement result hash differs")
        _validate_monotonic_bounds(
            self.program.initial_relu_pre,
            self.relu_pre,
            caller="native refinement result",
        )


def intermediate_refinement_semantic_trace_hash(
    execution: NativeIntermediateRefinementExecution,
) -> str:
    """Hash replay semantics while excluding diagnostic wall-clock timing."""

    payload = execution.trace.to_dict()
    payload.pop("elapsed_ns")
    return _canonical_hash(payload)


def compile_native_intermediate_refinement_program(
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
    """Compile an exact target set and unrolled refinement schedule."""

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
        source_refinement_semantic_trace_hash=(source_refinement_semantic_trace_hash),
        external_constraint_seed=(
            None if admitted_external_seed is None else admitted_external_seed.ir
        ),
    )
    task_module, schedule = lower_native_intermediate_refinement_ir(plan)
    program = NativeIntermediateRefinementProgram(
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
    program.validate(module, input_spec)
    return program


def _selected_crown_hash(
    values: Mapping[str, tuple[tuple[int, ...], torch.Tensor, torch.Tensor]],
) -> str:
    return _canonical_hash(
        [
            {
                "relu_input": name,
                "indices": list(indices),
                "lower": _tensor_schema(lower),
                "upper": _tensor_schema(upper),
            }
            for name, (indices, lower, upper) in values.items()
        ]
    )


def _run_selected_crown(
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    interval_env: Mapping[str, IntervalState],
    relu_pre: Mapping[str, IntervalState],
    targets: tuple[NativeIntermediateRefinementTargetIR, ...],
    chunk_size: int,
) -> dict[str, tuple[tuple[int, ...], torch.Tensor, torch.Tensor]]:
    grouped: dict[str, list[int]] = {}
    for target in targets:
        grouped.setdefault(target.relu_input, []).append(target.neuron_index)
    result: dict[str, tuple[tuple[int, ...], torch.Tensor, torch.Tensor]] = {}
    for name in relu_pre:
        if name not in grouped:
            continue
        indices = tuple(grouped[name])
        lower_parts: list[torch.Tensor] = []
        upper_parts: list[torch.Tensor] = []
        numel = int(relu_pre[name].lower[0].numel())
        for start in range(0, len(indices), chunk_size):
            chunk = indices[start : start + chunk_size]
            objective = torch.zeros(
                len(chunk),
                numel,
                device=relu_pre[name].lower.device,
                dtype=relu_pre[name].lower.dtype,
            )
            objective[
                torch.arange(len(chunk), device=objective.device),
                torch.tensor(chunk, device=objective.device),
            ] = 1.0
            bounds = run_crown_ibp_mlp_from_forward_trace(
                module,
                input_spec,
                interval_env=dict(interval_env),
                relu_pre=dict(relu_pre),
                linear_spec_C=objective,
                output_value=name,
            )
            lower_parts.append(bounds.lower.detach().contiguous())
            upper_parts.append(bounds.upper.detach().contiguous())
        result[name] = (
            indices,
            torch.cat(lower_parts, dim=1),
            torch.cat(upper_parts, dim=1),
        )
    if sum(len(indices) for indices, _lower, _upper in result.values()) != len(targets):
        raise ValueError("native selected CROWN target coverage differs")
    return result


def _intersect_selected(
    current: Mapping[str, IntervalState],
    selected: Mapping[str, tuple[tuple[int, ...], torch.Tensor, torch.Tensor]],
) -> dict[str, IntervalState]:
    refined = _clone_bounds(current)
    for name, (indices, selected_lower, selected_upper) in selected.items():
        lower = refined[name].lower.reshape(1, -1)
        upper = refined[name].upper.reshape(1, -1)
        index_tensor = torch.tensor(indices, device=lower.device)
        lower[:, index_tensor] = torch.maximum(lower[:, index_tensor], selected_lower)
        upper[:, index_tensor] = torch.minimum(upper[:, index_tensor], selected_upper)
        if bool((lower[:, index_tensor] > upper[:, index_tensor]).any()):
            raise ValueError("native refinement selected intersection is infeasible")
    return refined


def _validate_monotonic_bounds(
    before: Mapping[str, IntervalState],
    after: Mapping[str, IntervalState],
    *,
    caller: str,
) -> None:
    if tuple(before) != tuple(after):
        raise ValueError(f"{caller} ReLU identities differ")
    for name, previous in before.items():
        current = after[name]
        if (
            previous.lower.shape != current.lower.shape
            or previous.lower.dtype != current.lower.dtype
            or previous.lower.device != current.lower.device
            or bool((current.lower < previous.lower).any())
            or bool((current.upper > previous.upper).any())
            or bool((current.lower > current.upper).any())
        ):
            raise ValueError(f"{caller} is not a valid monotonic refinement")


def _pass_trace(
    pass_index: int,
    before: Mapping[str, IntervalState],
    selected: Mapping[str, tuple[tuple[int, ...], torch.Tensor, torch.Tensor]],
    after: Mapping[str, IntervalState],
    *,
    selected_target_count: int,
    selection_decision: Optional[NativeIntermediateRefinementPassDecisionIR] = None,
) -> NativeIntermediateRefinementPassTrace:
    _validate_monotonic_bounds(before, after, caller="native refinement pass")
    lower_max = 0.0
    upper_max = 0.0
    width_sum = 0.0
    tightened = 0
    for name, previous in before.items():
        current = after[name]
        lower_gain = (current.lower - previous.lower).clamp_min(0.0)
        upper_gain = (previous.upper - current.upper).clamp_min(0.0)
        changed = (lower_gain > 0.0) | (upper_gain > 0.0)
        tightened += int(changed.sum().item())
        lower_max = max(lower_max, float(lower_gain.max().item()))
        upper_max = max(upper_max, float(upper_gain.max().item()))
        width_sum += float((lower_gain + upper_gain).sum().item())
    trace = NativeIntermediateRefinementPassTrace(
        pass_index=pass_index,
        selected_target_count=selected_target_count,
        tightened_neuron_count=tightened,
        lower_improvement_max=lower_max,
        upper_improvement_max=upper_max,
        width_reduction_sum=width_sum,
        input_bounds_hash=intermediate_bounds_hash(before),
        selected_crown_hash=_selected_crown_hash(selected),
        output_bounds_hash=intermediate_bounds_hash(after),
        selection_decision=selection_decision,
    )
    trace.validate()
    return trace


def _runtime_value_hash(value: object) -> str:
    if isinstance(value, NativeExternalIntermediateConstraintSeed):
        return value.stable_hash()
    if isinstance(value, NativeIntermediateRefinementPolicyIR):
        return value.stable_hash()
    if isinstance(value, NativeIntermediateRefinementMultiPassPolicyIR):
        return value.stable_hash()
    if isinstance(value, NativeIntermediateRefinementPassDecisionIR):
        return value.stable_hash()
    if isinstance(value, BFTaskModule):
        return plain_crown_primal_graph_hash(value)
    if isinstance(value, InputSpec):
        return _input_bounds_hash(value)
    if isinstance(value, Mapping):
        if not value:
            return _canonical_hash([])
        if value and all(isinstance(item, IntervalState) for item in value.values()):
            return intermediate_bounds_hash(value)  # type: ignore[arg-type]
        if value and all(torch.is_tensor(item) for item in value.values()):
            tensors = value  # type: ignore[assignment]
            if all(item.dtype == torch.int8 for item in tensors.values()):
                return relu_split_state_hash(tensors)  # type: ignore[arg-type]
            return _canonical_hash(
                [
                    {"relu_input": name, "influence": _tensor_schema(item)}
                    for name, item in tensors.items()
                ]
            )
        if value and all(
            isinstance(item, tuple) and all(isinstance(index, int) for index in item)
            for item in value.values()
        ):
            return _canonical_hash(
                [
                    {"relu_input": name, "ambiguous_indices": list(indices)}
                    for name, indices in value.items()
                ]
            )
        if value and all(
            isinstance(item, tuple) and len(item) == 3 for item in value.values()
        ):
            return _selected_crown_hash(value)  # type: ignore[arg-type]
    if (
        isinstance(value, tuple)
        and value
        and all(
            isinstance(item, NativeIntermediateRefinementTargetIR) for item in value
        )
    ):
        return _canonical_hash([item.to_dict() for item in value])
    if isinstance(value, tuple) and all(
        isinstance(item, tuple)
        and len(item) == 2
        and isinstance(item[0], str)
        and isinstance(item[1], int)
        for item in value
    ):
        return _target_ledger_hash(value)  # type: ignore[arg-type]
    if isinstance(value, tuple) and not value:
        return _canonical_hash([])
    raise TypeError(f"unsupported native refinement runtime value: {type(value)}")


def execute_native_intermediate_refinement_program(
    program: NativeIntermediateRefinementProgram,
    module: BFTaskModule,
    input_spec: InputSpec,
) -> NativeIntermediateRefinementExecution:
    """Execute the exact Task/Schedule pipeline and return refined bounds."""

    program.validate(module, input_spec)
    started_ns = time.perf_counter_ns()
    values: dict[str, object] = {
        "refine.module": module,
        "refine.input": input_spec,
        "refine.split_state": program.split_state,
        "refine.policy": program.plan.policy,
    }
    if program.source_intermediate_constraints is not None:
        values["refine.source_intermediate_constraints"] = (
            program.source_intermediate_constraints
        )
    if program.external_constraint_seed is not None:
        values["refine.external_constraint_seed"] = program.external_constraint_seed
    if program.objective_influence is not None:
        values["refine.objective_influence"] = program.objective_influence
    if program.plan.multi_pass_policy is not None:
        values["refine.multi_pass_policy"] = program.plan.multi_pass_policy
    action_traces: list[NativeIntermediateRefinementActionTrace] = []
    pass_traces: list[NativeIntermediateRefinementPassTrace] = []
    for action, task in zip(program.schedule.actions, program.task_module.tasks):
        input_hashes = tuple(
            (name, _runtime_value_hash(values[name])) for name in action.input_value_ids
        )
        if task.kind == IntermediateRefinementTaskKind.MATERIALIZE_FORWARD:
            values[action.output_value_ids[0]] = program.initial_interval_env
            values[action.output_value_ids[1]] = program.initial_relu_pre
            if program.plan.multi_pass_policy is not None:
                values[action.output_value_ids[2]] = ()
        elif task.kind == IntermediateRefinementTaskKind.ENUMERATE_AMBIGUOUS:
            source = values[action.input_value_ids[0]]
            if not isinstance(source, Mapping):
                raise TypeError("native refinement enumeration input differs")
            if task.pass_index is not None or program.plan.objective_hash is not None:
                values[action.output_value_ids[0]] = _enumerate_ambiguous(
                    source  # type: ignore[arg-type]
                )
            else:
                values[action.output_value_ids[0]] = _select_targets(
                    source, program.plan.policy  # type: ignore[arg-type]
                )
        elif task.kind == IntermediateRefinementTaskKind.SELECT_TARGETS:
            if task.pass_index is not None:
                source = values[action.input_value_ids[0]]
                candidates = values[action.input_value_ids[1]]
                policy = values[action.input_value_ids[2]]
                multi_pass_policy = values[action.input_value_ids[3]]
                prior_ledger = values[action.input_value_ids[4]]
                influence = (
                    None
                    if program.plan.objective_hash is None
                    else values[action.input_value_ids[5]]
                )
                if (
                    not isinstance(source, Mapping)
                    or not isinstance(candidates, Mapping)
                    or not isinstance(policy, NativeIntermediateRefinementPolicyIR)
                    or not isinstance(
                        multi_pass_policy,
                        NativeIntermediateRefinementMultiPassPolicyIR,
                    )
                    or not isinstance(prior_ledger, tuple)
                    or any(
                        not isinstance(item, tuple)
                        or len(item) != 2
                        or not isinstance(item[0], str)
                        or not isinstance(item[1], int)
                        for item in prior_ledger
                    )
                    or candidates
                    != _enumerate_ambiguous(source)  # type: ignore[arg-type]
                    or (influence is not None and not isinstance(influence, Mapping))
                ):
                    raise ValueError("native multi-pass target inputs differ")
                pass_policy = _multi_pass_selection_policy(
                    policy, multi_pass_policy, pass_index=task.pass_index
                )
                selected = _select_targets(
                    source,  # type: ignore[arg-type]
                    pass_policy,
                    objective_influence=influence,  # type: ignore[arg-type]
                    excluded_target_identities=frozenset(prior_ledger),
                    allow_empty=True,
                )
                if task.pass_index == 0 and selected != program.plan.targets:
                    raise ValueError(
                        "native multi-pass initial targets differ from Plan"
                    )
                selected_identities = _target_identities(selected)
                result_ledger = (*prior_ledger, *selected_identities)
                if len(result_ledger) != len(set(result_ledger)):
                    raise ValueError("native multi-pass target ledger repeats")
                decision = NativeIntermediateRefinementPassDecisionIR(
                    plan_hash=program.plan.stable_hash(),
                    multi_pass_policy_hash=multi_pass_policy.stable_hash(),
                    pass_index=task.pass_index,
                    total_target_cap_per_relu=policy.max_neurons_per_relu,
                    pass_target_cap_per_relu=pass_policy.max_neurons_per_relu,
                    input_bounds_hash=intermediate_bounds_hash(
                        source  # type: ignore[arg-type]
                    ),
                    prior_target_ledger_hash=_target_ledger_hash(prior_ledger),
                    selected_targets_hash=_targets_hash(selected),
                    result_target_ledger_hash=_target_ledger_hash(result_ledger),
                    prior_selected_target_count=len(prior_ledger),
                    selected_target_count=len(selected),
                    cumulative_selected_target_count=len(result_ledger),
                    continuation=bool(selected),
                    termination_reason=(
                        "selected_unseen_targets"
                        if selected
                        else "no_unseen_eligible_targets"
                    ),
                )
                decision.validate()
                values[action.output_value_ids[0]] = selected
                values[action.output_value_ids[1]] = result_ledger
                values[action.output_value_ids[2]] = decision
            elif program.plan.objective_hash is not None:
                source = values[action.input_value_ids[0]]
                candidates = values[action.input_value_ids[1]]
                policy = values[action.input_value_ids[2]]
                influence = values[action.input_value_ids[3]]
                if (
                    not isinstance(source, Mapping)
                    or not isinstance(candidates, Mapping)
                    or not isinstance(policy, NativeIntermediateRefinementPolicyIR)
                    or not isinstance(influence, Mapping)
                    or candidates
                    != _enumerate_ambiguous(source)  # type: ignore[arg-type]
                ):
                    raise ValueError("native refinement runtime candidates differ")
                selected = _select_targets(
                    source,  # type: ignore[arg-type]
                    policy,
                    objective_influence=influence,  # type: ignore[arg-type]
                )
                if selected != program.plan.targets:
                    raise ValueError(
                        "native refinement runtime targets differ from Plan"
                    )
                values[action.output_value_ids[0]] = selected
            else:
                candidates = values[action.input_value_ids[0]]
                if candidates != program.plan.targets:
                    raise ValueError(
                        "native refinement runtime targets differ from Plan"
                    )
                values[action.output_value_ids[0]] = program.plan.targets
        elif task.kind == IntermediateRefinementTaskKind.BACKWARD_SELECTED:
            interval_env = values[action.input_value_ids[0]]
            relu_pre = values[action.input_value_ids[1]]
            targets = values[action.input_value_ids[2]]
            if (
                not isinstance(interval_env, Mapping)
                or not isinstance(relu_pre, Mapping)
                or not isinstance(targets, tuple)
            ):
                raise TypeError("native refinement backward inputs differ")
            values[action.output_value_ids[0]] = _run_selected_crown(
                module,
                input_spec,
                interval_env=interval_env,  # type: ignore[arg-type]
                relu_pre=relu_pre,  # type: ignore[arg-type]
                targets=targets,  # type: ignore[arg-type]
                chunk_size=program.plan.policy.backward_chunk_size,
            )
        elif task.kind == IntermediateRefinementTaskKind.INTERSECT_SELECTED:
            current = values[action.input_value_ids[0]]
            selected_crown = values[action.input_value_ids[1]]
            if not isinstance(current, Mapping) or not isinstance(
                selected_crown, Mapping
            ):
                raise TypeError("native refinement intersection inputs differ")
            values[action.output_value_ids[0]] = _intersect_selected(
                current, selected_crown  # type: ignore[arg-type]
            )
        elif task.kind == IntermediateRefinementTaskKind.PROPAGATE_FORWARD:
            constraints = values[action.input_value_ids[3]]
            if task.pass_index is None:
                raise ValueError("native refinement propagation lacks pass index")
            selected_key = f"refine.crown_candidates.p{task.pass_index + 1}"
            before_key = f"refine.bounds.p{task.pass_index}"
            if not isinstance(constraints, Mapping):
                raise TypeError("native refinement forward constraints differ")
            next_env, next_pre = _forward_ibp_trace_mlp(
                module,
                input_spec,
                relu_split_state=dict(program.split_state),
                relu_pre_constraints=constraints,  # type: ignore[arg-type]
            )
            before = values[before_key]
            selected_crown = values[selected_key]
            if not isinstance(before, Mapping) or not isinstance(
                selected_crown, Mapping
            ):
                raise TypeError("native refinement pass trace inputs differ")
            selection_decision = (
                None
                if program.plan.multi_pass_policy is None
                else values[action.input_value_ids[4]]
            )
            if selection_decision is not None and not isinstance(
                selection_decision, NativeIntermediateRefinementPassDecisionIR
            ):
                raise TypeError("native refinement pass decision differs")
            selected_targets_key = (
                "refine.selected_targets"
                if program.plan.multi_pass_policy is None
                else f"refine.selected_targets.p{task.pass_index}"
            )
            selected_targets = values[selected_targets_key]
            if not isinstance(selected_targets, tuple):
                raise TypeError("native refinement selected targets differ")
            pass_traces.append(
                _pass_trace(
                    task.pass_index,
                    before,  # type: ignore[arg-type]
                    selected_crown,  # type: ignore[arg-type]
                    next_pre,
                    selected_target_count=len(selected_targets),
                    selection_decision=selection_decision,
                )
            )
            values[action.output_value_ids[0]] = next_env
            values[action.output_value_ids[1]] = next_pre
        elif task.kind == IntermediateRefinementTaskKind.EMIT_REFINED:
            values[action.output_value_ids[0]] = values[action.input_value_ids[0]]
        else:
            raise AssertionError(f"unhandled native refinement task: {task.kind}")
        output_hashes = tuple(
            (name, _runtime_value_hash(values[name]))
            for name in action.output_value_ids
        )
        action_trace = NativeIntermediateRefinementActionTrace(
            sequence=action.sequence,
            action_id=action.action_id,
            task_id=task.task_id,
            kind=task.kind,
            pass_index=task.pass_index,
            input_hashes=input_hashes,
            output_hashes=output_hashes,
        )
        action_trace.validate()
        action_traces.append(action_trace)
    refined = values[program.schedule.refined_bounds_value_id]
    final_env = values[f"refine.forward_env.p{program.plan.policy.passes}"]
    if not isinstance(refined, Mapping) or not isinstance(final_env, Mapping):
        raise TypeError("native refinement final values differ")
    hashes = program.hashes()
    trace = NativeIntermediateRefinementExecutionTrace(
        plan_hash=hashes["refinement_plan_hash"],
        task_module_hash=hashes["refinement_task_module_hash"],
        schedule_hash=hashes["refinement_schedule_hash"],
        action_traces=tuple(action_traces),
        pass_traces=tuple(pass_traces),
        final_intermediate_bounds_hash=intermediate_bounds_hash(
            refined  # type: ignore[arg-type]
        ),
        elapsed_ns=time.perf_counter_ns() - started_ns,
    )
    execution = NativeIntermediateRefinementExecution(
        program=program,
        interval_env=final_env,  # type: ignore[arg-type]
        relu_pre=refined,  # type: ignore[arg-type]
        trace=trace,
    )
    execution.validate(module, input_spec)
    return execution


__all__ = [
    "NATIVE_INTERMEDIATE_REFINEMENT_EXECUTION_SCHEMA_VERSION",
    "NativeIntermediateRefinementActionTrace",
    "NativeExternalIntermediateConstraintSeed",
    "NativeIntermediateRefinementExecution",
    "NativeIntermediateRefinementExecutionTrace",
    "NativeIntermediateRefinementPassTrace",
    "NativeIntermediateRefinementProgram",
    "build_native_external_intermediate_constraint_seed",
    "compile_native_intermediate_refinement_program",
    "execute_native_intermediate_refinement_program",
    "intermediate_bounds_hash",
    "intermediate_refinement_semantic_trace_hash",
]
