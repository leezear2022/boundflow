"""Joint native representation and spec-batch Plan execution."""

# pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-locals
# pylint: disable=too-many-statements,too-many-branches,invalid-name
# pylint: disable=missing-function-docstring,too-many-boolean-expressions

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping

import torch

from ..domains.interval import IntervalState
from ..frontends.plain_crown_bound_ir import (
    PlainCrownBoundIRBuild,
    tensor_content_hash,
)
from ..ir.plan import PlanInstance, PlanTemplate
from ..ir.schedule import (
    BatchLoopAction,
    ScheduleModule,
    lower_plan_instance_to_reference_schedule,
)
from ..ir.task import BFTaskModule
from ..planner.plan_ir_selector import PlanSelectionContext, select_plan_instance
from ..planner.representation_plan_binding import (
    BoundRepresentationBinding,
    bind_native_representation_plan,
    build_native_representation_plan_variants,
)
from ..planner.spec_batch_plan_variants import (
    build_native_spec_batch_plan_variants,
)
from .native_verifier_ir_integration import (
    NativePlainCrownRepresentationCompilation,
    compile_native_plain_crown_query,
    compile_native_plain_crown_representation_query,
    execute_native_plain_crown_representation_query,
)
from .task_executor import InputSpec
from .task_ir_executor import TaskExecutionTrace

NATIVE_JOINT_POLICY_COMPILER_VERSION = "boundflow.native-joint-policy-execution/v1"
JOINT_POLICY_BINDING_SCHEMA_VERSION = "boundflow.native-joint-policy-binding/v1"
JOINT_POLICY_EXECUTION_TRACE_SCHEMA_VERSION = (
    "boundflow.native-joint-policy-execution-trace/v1"
)
_CHILD_HASH_KEYS = {
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


@dataclass(frozen=True)
class JointPolicySliceBinding:
    """One source batch range linked to one representation-bound child stack."""

    slice_id: str
    start_index: int
    stop_index: int
    child_query_id: str
    representation_policy_id: str
    child_ir_hashes: tuple[tuple[str, str], ...]

    def validate(self) -> None:
        if (
            not self.slice_id
            or self.start_index < 0
            or self.stop_index <= self.start_index
            or not self.child_query_id
            or not self.representation_policy_id
        ):
            raise ValueError("joint policy slice identity/range is invalid")
        hashes = dict(self.child_ir_hashes)
        if (
            len(hashes) != len(self.child_ir_hashes)
            or set(hashes) != _CHILD_HASH_KEYS
            or any(len(value) != 64 for value in hashes.values())
        ):
            raise ValueError("joint policy child IR hash set differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "slice_id": self.slice_id,
            "start_index": self.start_index,
            "stop_index": self.stop_index,
            "child_query_id": self.child_query_id,
            "representation_policy_id": self.representation_policy_id,
            "child_ir_hashes": dict(self.child_ir_hashes),
        }


@dataclass(frozen=True)
class JointPolicyBindingTrace:
    """Replay-grade source representation/batch decisions and child ownership."""

    source_bound_module_hash: str
    source_plan_template_hash: str
    source_plan_instance_hash: str
    source_schedule_hash: str
    source_representation_binding_hash: str
    source_execution_bound_module_hash: str
    selected_representation_policy_id: str
    selected_storage_candidate_id: str
    selected_batch_candidate_id: str
    total_spec_count: int
    selected_spec_batch_size: int
    source_linear_spec_hash: str
    slices: tuple[JointPolicySliceBinding, ...]
    schema_version: str = JOINT_POLICY_BINDING_SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != JOINT_POLICY_BINDING_SCHEMA_VERSION:
            raise ValueError("unsupported joint policy binding schema")
        for name in (
            "source_bound_module_hash",
            "source_plan_template_hash",
            "source_plan_instance_hash",
            "source_schedule_hash",
            "source_representation_binding_hash",
            "source_execution_bound_module_hash",
            "source_linear_spec_hash",
        ):
            if len(getattr(self, name)) != 64:
                raise ValueError(f"joint policy binding {name} is not SHA-256")
        if (
            not self.selected_representation_policy_id
            or not self.selected_storage_candidate_id
            or not self.selected_batch_candidate_id
            or self.total_spec_count <= 0
            or self.selected_spec_batch_size <= 0
            or self.selected_spec_batch_size > self.total_spec_count
            or not self.slices
        ):
            raise ValueError("joint policy selection is invalid")
        expected_start = 0
        query_ids: list[str] = []
        for item in self.slices:
            item.validate()
            if (
                item.start_index != expected_start
                or item.stop_index - item.start_index > self.selected_spec_batch_size
                or item.representation_policy_id
                != self.selected_representation_policy_id
            ):
                raise ValueError("joint policy slices overlap or change policy")
            expected_start = item.stop_index
            query_ids.append(item.child_query_id)
        if expected_start != self.total_spec_count:
            raise ValueError("joint policy slices do not cover every objective")
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("joint policy repeats a child query")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "source_bound_module_hash": self.source_bound_module_hash,
            "source_plan_template_hash": self.source_plan_template_hash,
            "source_plan_instance_hash": self.source_plan_instance_hash,
            "source_schedule_hash": self.source_schedule_hash,
            "source_representation_binding_hash": (
                self.source_representation_binding_hash
            ),
            "source_execution_bound_module_hash": (
                self.source_execution_bound_module_hash
            ),
            "selected_representation_policy_id": (
                self.selected_representation_policy_id
            ),
            "selected_storage_candidate_id": self.selected_storage_candidate_id,
            "selected_batch_candidate_id": self.selected_batch_candidate_id,
            "total_spec_count": self.total_spec_count,
            "selected_spec_batch_size": self.selected_spec_batch_size,
            "source_linear_spec_hash": self.source_linear_spec_hash,
            "slices": [item.to_dict() for item in self.slices],
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    def stable_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class NativePlainCrownJointPolicyCompilation:
    """One joint source Plan plus exact representation-bound child stacks."""

    query_id: str
    intermediate_bounds_hash: str
    source_build: PlainCrownBoundIRBuild
    source_template: PlanTemplate
    source_instance: PlanInstance
    source_schedule: ScheduleModule
    source_representation_binding: BoundRepresentationBinding
    binding_trace: JointPolicyBindingTrace
    child_compilations: tuple[NativePlainCrownRepresentationCompilation, ...]

    def validate(self) -> None:  # pylint: disable=too-many-statements
        if not self.query_id or len(self.intermediate_bounds_hash) != 64:
            raise ValueError("native joint policy query identity is invalid")
        source = self.source_build.module
        self.source_instance.validate(
            template=self.source_template, bound_module=source
        )
        self.source_schedule.validate(
            bound_module=source,
            template=self.source_template,
            instance=self.source_instance,
        )
        self.source_representation_binding.validate()
        self.binding_trace.validate()
        trace = self.binding_trace
        representation_trace = self.source_representation_binding.trace
        if (
            trace.source_bound_module_hash != source.stable_hash()
            or trace.source_plan_template_hash
            != self.source_template.stable_hash(bound_module=source)
            or trace.source_plan_instance_hash
            != self.source_instance.stable_hash(
                template=self.source_template, bound_module=source
            )
            or trace.source_schedule_hash
            != self.source_schedule.stable_hash(
                bound_module=source,
                template=self.source_template,
                instance=self.source_instance,
            )
            or trace.source_representation_binding_hash
            != representation_trace.stable_hash()
            or trace.source_execution_bound_module_hash
            != self.source_representation_binding.execution_bound_module.stable_hash()
            or trace.selected_representation_policy_id != representation_trace.policy_id
            or trace.selected_storage_candidate_id
            != self.source_instance.storage_decision.candidate_id
            or trace.selected_batch_candidate_id
            != self.source_instance.batch_decision.candidate_id
        ):
            raise ValueError("native joint policy source linkage differs")
        loop = _batch_loop(self.source_schedule)
        expected_ranges = (
            tuple(
                (item.slice_id, item.start_index, item.stop_index)
                for item in loop.slices
            )
            if loop.axis == "spec"
            else (("full-spec", 0, trace.total_spec_count),)
        )
        actual_ranges = tuple(
            (item.slice_id, item.start_index, item.stop_index) for item in trace.slices
        )
        if expected_ranges != actual_ranges:
            raise ValueError("native joint policy Schedule/binding ranges differ")
        if len(self.child_compilations) != len(trace.slices):
            raise ValueError("native joint policy child count differs")
        for item, child in zip(trace.slices, self.child_compilations):
            child.validate()
            if (
                tuple(sorted(child.hashes().items())) != item.child_ir_hashes
                or child.query_id != item.child_query_id
                or child.binding.trace.policy_id != item.representation_policy_id
                or child.source_instance.storage_decision.candidate_id
                != trace.selected_storage_candidate_id
            ):
                raise ValueError("native joint policy child linkage differs")
            objective = child.source_bound_module.spec.objectives[0]
            if objective.num_objectives != item.stop_index - item.start_index:
                raise ValueError("native joint policy child objective width differs")

    def hashes(self) -> dict[str, str]:
        self.validate()
        source = self.source_build.module
        return {
            "source_bound_module_hash": source.stable_hash(),
            "source_plan_template_hash": self.source_template.stable_hash(
                bound_module=source
            ),
            "source_plan_instance_hash": self.source_instance.stable_hash(
                template=self.source_template, bound_module=source
            ),
            "source_schedule_hash": self.source_schedule.stable_hash(
                bound_module=source,
                template=self.source_template,
                instance=self.source_instance,
            ),
            "source_representation_binding_hash": (
                self.source_representation_binding.trace.stable_hash()
            ),
            "joint_policy_binding_hash": self.binding_trace.stable_hash(),
        }


@dataclass(frozen=True)
class JointPolicyExecutionTrace:
    """Exact representation policy, child execution, and aggregation record."""

    binding_hash: str
    representation_policy_id: str
    child_query_ids: tuple[str, ...]
    child_representation_binding_hashes: tuple[str, ...]
    child_task_trace_hashes: tuple[str, ...]
    result_lower_hash: str
    result_upper_hash: str
    schema_version: str = JOINT_POLICY_EXECUTION_TRACE_SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != JOINT_POLICY_EXECUTION_TRACE_SCHEMA_VERSION:
            raise ValueError("unsupported joint policy execution trace schema")
        if len(self.binding_hash) != 64 or not self.representation_policy_id:
            raise ValueError("joint policy execution identity is invalid")
        count = len(self.child_query_ids)
        if (
            count == 0
            or count != len(set(self.child_query_ids))
            or count != len(self.child_representation_binding_hashes)
            or count != len(self.child_task_trace_hashes)
        ):
            raise ValueError("joint policy execution child accounting differs")
        if any(
            len(value) != 64
            for value in (
                *self.child_representation_binding_hashes,
                *self.child_task_trace_hashes,
                self.result_lower_hash,
                self.result_upper_hash,
            )
        ):
            raise ValueError("joint policy execution digest differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "binding_hash": self.binding_hash,
            "representation_policy_id": self.representation_policy_id,
            "child_query_ids": list(self.child_query_ids),
            "child_representation_binding_hashes": list(
                self.child_representation_binding_hashes
            ),
            "child_task_trace_hashes": list(self.child_task_trace_hashes),
            "result_lower_hash": self.result_lower_hash,
            "result_upper_hash": self.result_upper_hash,
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    def stable_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


# pylint: disable-next=too-many-arguments
def compile_native_plain_crown_joint_policy_query(
    legacy_task_module: BFTaskModule,
    input_spec: InputSpec,
    *,
    interval_env: Mapping[str, IntervalState],
    relu_pre: Mapping[str, IntervalState],
    linear_spec_C: torch.Tensor,
    intermediate_bounds_hash: str,
    query_id: str,
    available_memory_bytes: int,
    memory_budget_bytes: int,
    spec_slice_candidate_size: int,
    max_spec_batch_size: int,
) -> NativePlainCrownJointPolicyCompilation:
    """Jointly select representation/storage and spec-batch execution policies."""

    total_specs = _spec_count(linear_spec_C)
    if spec_slice_candidate_size <= 0 or spec_slice_candidate_size >= total_specs:
        raise ValueError("joint policy spec slice must be smaller than objective")
    if max_spec_batch_size <= 0 or max_spec_batch_size > total_specs:
        raise ValueError("joint policy max spec batch is outside objective range")
    base = compile_native_plain_crown_query(
        legacy_task_module,
        input_spec,
        interval_env=interval_env,
        relu_pre=relu_pre,
        linear_spec_C=linear_spec_C,
        intermediate_bounds_hash=intermediate_bounds_hash,
        query_id=query_id,
        available_memory_bytes=available_memory_bytes,
    )
    source = base.bound_module
    source_template = build_native_representation_plan_variants(
        base.template, bound_module=source
    )
    source_template = build_native_spec_batch_plan_variants(
        source_template,
        bound_module=source,
        spec_batch_size=spec_slice_candidate_size,
    )
    source_instance = select_plan_instance(
        source_template,
        bound_module=source,
        query_bucket_id=f"native-joint-policy:{query_id}",
        available_memory_bytes=available_memory_bytes,
        memory_budget_bytes=memory_budget_bytes,
        selection_context=PlanSelectionContext(
            query_distribution_id=f"native-joint-policy:{query_id}",
            max_spec_batch_size=max_spec_batch_size,
        ),
    )
    source_schedule = lower_plan_instance_to_reference_schedule(
        source,
        template=source_template,
        instance=source_instance,
        query_ids=(query_id,),
    )
    representation_binding = bind_native_representation_plan(
        source,
        template=source_template,
        instance=source_instance,
        schedule=source_schedule,
    )
    policy_id = representation_binding.trace.policy_id
    storage_id = source_instance.storage_decision.candidate_id
    selected_batch = next(
        candidate
        for candidate in source_template.batch_candidates
        if candidate.candidate_id == source_instance.batch_decision.candidate_id
    )
    loop = _batch_loop(source_schedule)
    ranges = (
        tuple(
            (item.slice_id, item.start_index, item.stop_index) for item in loop.slices
        )
        if loop.axis == "spec"
        else (("full-spec", 0, total_specs),)
    )
    children: list[NativePlainCrownRepresentationCompilation] = []
    bindings: list[JointPolicySliceBinding] = []
    for slice_id, start, stop in ranges:
        if start is None or stop is None:
            raise ValueError("joint policy Schedule omitted objective range")
        child_query_id = f"{query_id}:spec:{start:04d}:{stop:04d}"
        child = compile_native_plain_crown_representation_query(
            legacy_task_module,
            input_spec,
            interval_env=interval_env,
            relu_pre=relu_pre,
            linear_spec_C=_slice_objective(linear_spec_C, start, stop),
            intermediate_bounds_hash=intermediate_bounds_hash,
            query_id=child_query_id,
            available_memory_bytes=available_memory_bytes,
            memory_budget_bytes=available_memory_bytes,
            selection_context=PlanSelectionContext(
                query_distribution_id=(
                    f"native-joint-child:{query_id}:{start:04d}:{stop:04d}"
                ),
                required_storage_candidate_id=storage_id,
            ),
        )
        if child.binding.trace.policy_id != policy_id:
            raise ValueError("joint policy child changed representation policy")
        children.append(child)
        bindings.append(
            JointPolicySliceBinding(
                slice_id=slice_id,
                start_index=start,
                stop_index=stop,
                child_query_id=child_query_id,
                representation_policy_id=policy_id,
                child_ir_hashes=tuple(sorted(child.hashes().items())),
            )
        )
    trace = JointPolicyBindingTrace(
        source_bound_module_hash=source.stable_hash(),
        source_plan_template_hash=source_template.stable_hash(bound_module=source),
        source_plan_instance_hash=source_instance.stable_hash(
            template=source_template, bound_module=source
        ),
        source_schedule_hash=source_schedule.stable_hash(
            bound_module=source,
            template=source_template,
            instance=source_instance,
        ),
        source_representation_binding_hash=(representation_binding.trace.stable_hash()),
        source_execution_bound_module_hash=(
            representation_binding.execution_bound_module.stable_hash()
        ),
        selected_representation_policy_id=policy_id,
        selected_storage_candidate_id=storage_id,
        selected_batch_candidate_id=source_instance.batch_decision.candidate_id,
        total_spec_count=total_specs,
        selected_spec_batch_size=selected_batch.spec_batch_size,
        source_linear_spec_hash=tensor_content_hash(linear_spec_C),
        slices=tuple(bindings),
    )
    compilation = NativePlainCrownJointPolicyCompilation(
        query_id=query_id,
        intermediate_bounds_hash=intermediate_bounds_hash,
        source_build=base.build,
        source_template=source_template,
        source_instance=source_instance,
        source_schedule=source_schedule,
        source_representation_binding=representation_binding,
        binding_trace=trace,
        child_compilations=tuple(children),
    )
    compilation.validate()
    return compilation


def execute_native_plain_crown_joint_policy_query(
    compilation: NativePlainCrownJointPolicyCompilation,
    *,
    legacy_task_module: BFTaskModule,
    input_spec: InputSpec,
    relu_pre: Mapping[str, IntervalState],
    linear_spec_C: torch.Tensor,
) -> tuple[IntervalState, JointPolicyExecutionTrace]:
    """Execute exact representation-bound slices and aggregate in spec order."""

    compilation.validate()
    if tensor_content_hash(linear_spec_C) != (
        compilation.binding_trace.source_linear_spec_hash
    ):
        raise ValueError("native joint policy runtime objective hash differs")
    results: list[IntervalState] = []
    task_traces: list[TaskExecutionTrace] = []
    for item, child in zip(
        compilation.binding_trace.slices, compilation.child_compilations
    ):
        result, task_trace = execute_native_plain_crown_representation_query(
            child,
            legacy_task_module=legacy_task_module,
            input_spec=input_spec,
            relu_pre=relu_pre,
            linear_spec_C=_slice_objective(
                linear_spec_C, item.start_index, item.stop_index
            ),
        )
        results.append(result)
        task_traces.append(task_trace)
    aggregated = IntervalState(
        lower=torch.cat(tuple(item.lower for item in results), dim=1),
        upper=torch.cat(tuple(item.upper for item in results), dim=1),
    )
    aggregated.validate()
    expected_shape = (
        compilation.source_template.workload.domain_batch_size,
        compilation.binding_trace.total_spec_count,
    )
    if tuple(aggregated.lower.shape) != expected_shape:
        raise ValueError("native joint policy aggregate shape differs")
    trace = JointPolicyExecutionTrace(
        binding_hash=compilation.binding_trace.stable_hash(),
        representation_policy_id=(
            compilation.binding_trace.selected_representation_policy_id
        ),
        child_query_ids=tuple(
            item.child_query_id for item in compilation.binding_trace.slices
        ),
        child_representation_binding_hashes=tuple(
            child.binding.trace.stable_hash()
            for child in compilation.child_compilations
        ),
        child_task_trace_hashes=tuple(item.stable_hash() for item in task_traces),
        result_lower_hash=tensor_content_hash(aggregated.lower),
        result_upper_hash=tensor_content_hash(aggregated.upper),
    )
    trace.validate()
    return aggregated, trace


def _batch_loop(schedule: ScheduleModule) -> BatchLoopAction:
    loops = tuple(
        action for action in schedule.actions if isinstance(action, BatchLoopAction)
    )
    if len(loops) != 1:
        raise ValueError("joint policy source Schedule loop count differs")
    return loops[0]


def _spec_count(linear_spec_C: torch.Tensor) -> int:
    if not torch.is_tensor(linear_spec_C):
        raise TypeError("joint policy objective must be a tensor")
    if linear_spec_C.dim() == 2:
        return int(linear_spec_C.shape[0])
    if linear_spec_C.dim() == 3:
        return int(linear_spec_C.shape[1])
    raise ValueError("joint policy objective must have rank two or three")


def _slice_objective(
    linear_spec_C: torch.Tensor, start: int, stop: int
) -> torch.Tensor:
    if start < 0 or stop <= start or stop > _spec_count(linear_spec_C):
        raise ValueError("joint policy objective slice range is invalid")
    sliced = (
        linear_spec_C[start:stop]
        if linear_spec_C.dim() == 2
        else linear_spec_C[:, start:stop, :]
    )
    return sliced.contiguous()


__all__ = [
    "JOINT_POLICY_BINDING_SCHEMA_VERSION",
    "JOINT_POLICY_EXECUTION_TRACE_SCHEMA_VERSION",
    "NATIVE_JOINT_POLICY_COMPILER_VERSION",
    "JointPolicyBindingTrace",
    "JointPolicyExecutionTrace",
    "JointPolicySliceBinding",
    "NativePlainCrownJointPolicyCompilation",
    "compile_native_plain_crown_joint_policy_query",
    "execute_native_plain_crown_joint_policy_query",
]
