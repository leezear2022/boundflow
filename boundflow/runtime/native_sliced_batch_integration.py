"""Native spec-batch Plan binding and real sliced execution."""

# pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-locals
# pylint: disable=too-many-statements,missing-function-docstring,invalid-name

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
from ..planner.spec_batch_plan_variants import (
    build_native_spec_batch_plan_variants,
)
from .native_verifier_ir_integration import (
    NativePlainCrownIRCompilation,
    compile_native_plain_crown_query,
    execute_native_plain_crown_query,
)
from .task_executor import InputSpec
from .task_ir_executor import TaskExecutionTrace

NATIVE_SLICED_BATCH_COMPILER_VERSION = "boundflow.native-spec-batch-execution/v1"
SPEC_BATCH_BINDING_SCHEMA_VERSION = "boundflow.native-spec-batch-binding/v1"
SPEC_BATCH_EXECUTION_TRACE_SCHEMA_VERSION = (
    "boundflow.native-spec-batch-execution-trace/v1"
)


@dataclass(frozen=True)
class SpecBatchSliceBinding:
    """One source Schedule objective range bound to one child compiler stack."""

    slice_id: str
    start_index: int
    stop_index: int
    child_query_id: str
    child_bound_module_hash: str
    child_plan_template_hash: str
    child_plan_instance_hash: str
    child_task_module_hash: str
    child_schedule_hash: str

    def validate(self) -> None:
        if (
            not self.slice_id
            or self.start_index < 0
            or self.stop_index <= self.start_index
            or not self.child_query_id
        ):
            raise ValueError("spec batch slice binding identity/range is invalid")
        for name in (
            "child_bound_module_hash",
            "child_plan_template_hash",
            "child_plan_instance_hash",
            "child_task_module_hash",
            "child_schedule_hash",
        ):
            if len(getattr(self, name)) != 64:
                raise ValueError(f"spec batch binding {name} is not SHA-256")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "slice_id": self.slice_id,
            "start_index": self.start_index,
            "stop_index": self.stop_index,
            "child_query_id": self.child_query_id,
            "child_bound_module_hash": self.child_bound_module_hash,
            "child_plan_template_hash": self.child_plan_template_hash,
            "child_plan_instance_hash": self.child_plan_instance_hash,
            "child_task_module_hash": self.child_task_module_hash,
            "child_schedule_hash": self.child_schedule_hash,
        }


@dataclass(frozen=True)
class SpecBatchBindingTrace:
    """Replay-grade linkage from one source BatchDecision to child IR stacks."""

    source_bound_module_hash: str
    source_plan_template_hash: str
    source_plan_instance_hash: str
    source_schedule_hash: str
    selected_batch_candidate_id: str
    total_spec_count: int
    selected_spec_batch_size: int
    source_linear_spec_hash: str
    slices: tuple[SpecBatchSliceBinding, ...]
    schema_version: str = SPEC_BATCH_BINDING_SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SPEC_BATCH_BINDING_SCHEMA_VERSION:
            raise ValueError("unsupported spec batch binding schema")
        for name in (
            "source_bound_module_hash",
            "source_plan_template_hash",
            "source_plan_instance_hash",
            "source_schedule_hash",
            "source_linear_spec_hash",
        ):
            if len(getattr(self, name)) != 64:
                raise ValueError(f"spec batch binding {name} is not SHA-256")
        if (
            not self.selected_batch_candidate_id
            or self.total_spec_count <= 0
            or self.selected_spec_batch_size <= 0
            or self.selected_spec_batch_size > self.total_spec_count
            or not self.slices
        ):
            raise ValueError("spec batch binding selection is invalid")
        expected_start = 0
        query_ids: list[str] = []
        for item in self.slices:
            item.validate()
            if (
                item.start_index != expected_start
                or item.stop_index - item.start_index > self.selected_spec_batch_size
            ):
                raise ValueError(
                    "spec batch binding slices overlap or exceed selection"
                )
            expected_start = item.stop_index
            query_ids.append(item.child_query_id)
        if expected_start != self.total_spec_count:
            raise ValueError("spec batch binding does not cover every objective")
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("spec batch binding repeats a child query")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "source_bound_module_hash": self.source_bound_module_hash,
            "source_plan_template_hash": self.source_plan_template_hash,
            "source_plan_instance_hash": self.source_plan_instance_hash,
            "source_schedule_hash": self.source_schedule_hash,
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
class NativePlainCrownSlicedBatchCompilation:
    """Source BatchDecision plus every statically specialized child stack."""

    query_id: str
    intermediate_bounds_hash: str
    source_build: PlainCrownBoundIRBuild
    source_template: PlanTemplate
    source_instance: PlanInstance
    source_schedule: ScheduleModule
    binding_trace: SpecBatchBindingTrace
    child_compilations: tuple[NativePlainCrownIRCompilation, ...]

    def validate(self) -> None:  # pylint: disable=too-many-branches
        if not self.query_id or len(self.intermediate_bounds_hash) != 64:
            raise ValueError("native sliced batch query identity is invalid")
        source_module = self.source_build.module
        self.source_instance.validate(
            template=self.source_template, bound_module=source_module
        )
        self.source_schedule.validate(
            bound_module=source_module,
            template=self.source_template,
            instance=self.source_instance,
        )
        self.binding_trace.validate()
        trace = self.binding_trace
        if (
            trace.source_bound_module_hash != source_module.stable_hash()
            or trace.source_plan_template_hash
            != self.source_template.stable_hash(bound_module=source_module)
            or trace.source_plan_instance_hash
            != self.source_instance.stable_hash(
                template=self.source_template, bound_module=source_module
            )
            or trace.source_schedule_hash
            != self.source_schedule.stable_hash(
                bound_module=source_module,
                template=self.source_template,
                instance=self.source_instance,
            )
            or trace.selected_batch_candidate_id
            != self.source_instance.batch_decision.candidate_id
        ):
            raise ValueError("native sliced batch source linkage differs")
        batch = next(
            candidate
            for candidate in self.source_template.batch_candidates
            if candidate.candidate_id
            == self.source_instance.batch_decision.candidate_id
        )
        if batch.spec_batch_size != trace.selected_spec_batch_size:
            raise ValueError("native sliced batch selected size differs")
        loops = tuple(
            action
            for action in self.source_schedule.actions
            if isinstance(action, BatchLoopAction)
        )
        if len(loops) != 1:
            raise ValueError("native sliced batch source Schedule loop count differs")
        loop = loops[0]
        expected_schedule_slices = (
            tuple(
                (
                    item.slice_id,
                    item.start_index,
                    item.stop_index,
                )
                for item in loop.slices
            )
            if loop.axis == "spec"
            else (("full-spec", 0, trace.total_spec_count),)
        )
        actual_slices = tuple(
            (item.slice_id, item.start_index, item.stop_index) for item in trace.slices
        )
        if expected_schedule_slices != actual_slices:
            raise ValueError("native sliced batch Schedule/binding ranges differ")
        if len(self.child_compilations) != len(trace.slices):
            raise ValueError("native sliced batch child count differs")
        for item, child in zip(trace.slices, self.child_compilations):
            child.validate()
            hashes = child.hashes()
            expected_hashes = (
                item.child_bound_module_hash,
                item.child_plan_template_hash,
                item.child_plan_instance_hash,
                item.child_task_module_hash,
                item.child_schedule_hash,
            )
            actual_hashes = (
                hashes["bound_module_hash"],
                hashes["plan_template_hash"],
                hashes["plan_instance_hash"],
                hashes["task_module_hash"],
                hashes["schedule_hash"],
            )
            if (
                actual_hashes != expected_hashes
                or child.query_id != item.child_query_id
            ):
                raise ValueError("native sliced batch child IR linkage differs")
            objective = child.bound_module.spec.objectives[0]
            if objective.num_objectives != item.stop_index - item.start_index:
                raise ValueError("native sliced batch child objective width differs")

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
            "spec_batch_binding_hash": self.binding_trace.stable_hash(),
        }


@dataclass(frozen=True)
class SlicedBatchExecutionTrace:
    """Exact child execution and aggregation record."""

    binding_hash: str
    child_query_ids: tuple[str, ...]
    child_task_trace_hashes: tuple[str, ...]
    result_lower_hash: str
    result_upper_hash: str
    schema_version: str = SPEC_BATCH_EXECUTION_TRACE_SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SPEC_BATCH_EXECUTION_TRACE_SCHEMA_VERSION:
            raise ValueError("unsupported sliced batch execution trace schema")
        if len(self.binding_hash) != 64:
            raise ValueError("sliced batch execution binding hash is invalid")
        if (
            not self.child_query_ids
            or len(self.child_query_ids) != len(set(self.child_query_ids))
            or len(self.child_query_ids) != len(self.child_task_trace_hashes)
        ):
            raise ValueError("sliced batch execution child accounting differs")
        if any(len(value) != 64 for value in self.child_task_trace_hashes):
            raise ValueError("sliced batch execution child trace hash is invalid")
        if len(self.result_lower_hash) != 64 or len(self.result_upper_hash) != 64:
            raise ValueError("sliced batch execution result hash is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "binding_hash": self.binding_hash,
            "child_query_ids": list(self.child_query_ids),
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
def compile_native_plain_crown_sliced_batch_query(
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
) -> NativePlainCrownSlicedBatchCompilation:
    """Select full/spec-sliced source Plan and compile every selected range."""

    total_specs = _spec_count(linear_spec_C)
    if spec_slice_candidate_size <= 0 or spec_slice_candidate_size >= total_specs:
        raise ValueError("native spec slice candidate must be smaller than objective")
    if max_spec_batch_size <= 0 or max_spec_batch_size > total_specs:
        raise ValueError("native max spec batch size is outside objective range")
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
    source_module = base.bound_module
    source_template = build_native_spec_batch_plan_variants(
        base.template,
        bound_module=source_module,
        spec_batch_size=spec_slice_candidate_size,
    )
    source_instance = select_plan_instance(
        source_template,
        bound_module=source_module,
        query_bucket_id=f"native-spec-batch:{query_id}",
        available_memory_bytes=available_memory_bytes,
        memory_budget_bytes=memory_budget_bytes,
        selection_context=PlanSelectionContext(
            query_distribution_id=f"native-spec-batch:{query_id}",
            max_spec_batch_size=max_spec_batch_size,
        ),
    )
    source_schedule = lower_plan_instance_to_reference_schedule(
        source_module,
        template=source_template,
        instance=source_instance,
        query_ids=(query_id,),
    )
    selected_batch = next(
        candidate
        for candidate in source_template.batch_candidates
        if candidate.candidate_id == source_instance.batch_decision.candidate_id
    )
    loop = next(
        action
        for action in source_schedule.actions
        if isinstance(action, BatchLoopAction)
    )
    ranges = (
        tuple(
            (item.slice_id, item.start_index, item.stop_index) for item in loop.slices
        )
        if loop.axis == "spec"
        else (("full-spec", 0, total_specs),)
    )
    children: list[NativePlainCrownIRCompilation] = []
    bindings: list[SpecBatchSliceBinding] = []
    for slice_id, start, stop in ranges:
        if start is None or stop is None:
            raise ValueError("native spec batch Schedule omitted objective range")
        child_query_id = f"{query_id}:spec:{start:04d}:{stop:04d}"
        child = compile_native_plain_crown_query(
            legacy_task_module,
            input_spec,
            interval_env=interval_env,
            relu_pre=relu_pre,
            linear_spec_C=_slice_objective(linear_spec_C, start, stop),
            intermediate_bounds_hash=intermediate_bounds_hash,
            query_id=child_query_id,
            available_memory_bytes=available_memory_bytes,
        )
        child_hashes = child.hashes()
        children.append(child)
        bindings.append(
            SpecBatchSliceBinding(
                slice_id=slice_id,
                start_index=start,
                stop_index=stop,
                child_query_id=child_query_id,
                child_bound_module_hash=child_hashes["bound_module_hash"],
                child_plan_template_hash=child_hashes["plan_template_hash"],
                child_plan_instance_hash=child_hashes["plan_instance_hash"],
                child_task_module_hash=child_hashes["task_module_hash"],
                child_schedule_hash=child_hashes["schedule_hash"],
            )
        )
    trace = SpecBatchBindingTrace(
        source_bound_module_hash=source_module.stable_hash(),
        source_plan_template_hash=source_template.stable_hash(
            bound_module=source_module
        ),
        source_plan_instance_hash=source_instance.stable_hash(
            template=source_template, bound_module=source_module
        ),
        source_schedule_hash=source_schedule.stable_hash(
            bound_module=source_module,
            template=source_template,
            instance=source_instance,
        ),
        selected_batch_candidate_id=source_instance.batch_decision.candidate_id,
        total_spec_count=total_specs,
        selected_spec_batch_size=selected_batch.spec_batch_size,
        source_linear_spec_hash=tensor_content_hash(linear_spec_C),
        slices=tuple(bindings),
    )
    compilation = NativePlainCrownSlicedBatchCompilation(
        query_id=query_id,
        intermediate_bounds_hash=intermediate_bounds_hash,
        source_build=base.build,
        source_template=source_template,
        source_instance=source_instance,
        source_schedule=source_schedule,
        binding_trace=trace,
        child_compilations=tuple(children),
    )
    compilation.validate()
    return compilation


def execute_native_plain_crown_sliced_batch_query(
    compilation: NativePlainCrownSlicedBatchCompilation,
    *,
    legacy_task_module: BFTaskModule,
    input_spec: InputSpec,
    relu_pre: Mapping[str, IntervalState],
    linear_spec_C: torch.Tensor,
) -> tuple[IntervalState, SlicedBatchExecutionTrace]:
    """Execute every specialized child and concatenate exact spec-order results."""

    compilation.validate()
    if tensor_content_hash(linear_spec_C) != (
        compilation.binding_trace.source_linear_spec_hash
    ):
        raise ValueError("native sliced batch runtime objective hash differs")
    results: list[IntervalState] = []
    task_traces: list[TaskExecutionTrace] = []
    for item, child in zip(
        compilation.binding_trace.slices, compilation.child_compilations
    ):
        result, trace = execute_native_plain_crown_query(
            child,
            legacy_task_module=legacy_task_module,
            input_spec=input_spec,
            relu_pre=relu_pre,
            linear_spec_C=_slice_objective(
                linear_spec_C, item.start_index, item.stop_index
            ),
        )
        results.append(result)
        task_traces.append(trace)
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
        raise ValueError("native sliced batch aggregate shape differs")
    aggregate_trace = SlicedBatchExecutionTrace(
        binding_hash=compilation.binding_trace.stable_hash(),
        child_query_ids=tuple(
            item.child_query_id for item in compilation.binding_trace.slices
        ),
        child_task_trace_hashes=tuple(item.stable_hash() for item in task_traces),
        result_lower_hash=tensor_content_hash(aggregated.lower),
        result_upper_hash=tensor_content_hash(aggregated.upper),
    )
    aggregate_trace.validate()
    return aggregated, aggregate_trace


def _spec_count(linear_spec_C: torch.Tensor) -> int:
    if not torch.is_tensor(linear_spec_C):
        raise TypeError("native spec batching requires a tensor objective")
    if linear_spec_C.dim() == 2:
        return int(linear_spec_C.shape[0])
    if linear_spec_C.dim() == 3:
        return int(linear_spec_C.shape[1])
    raise ValueError("native spec batching requires rank-2 or rank-3 objective")


def _slice_objective(
    linear_spec_C: torch.Tensor, start: int, stop: int
) -> torch.Tensor:
    if start < 0 or stop <= start or stop > _spec_count(linear_spec_C):
        raise ValueError("native objective slice range is invalid")
    sliced = (
        linear_spec_C[start:stop]
        if linear_spec_C.dim() == 2
        else linear_spec_C[:, start:stop, :]
    )
    return sliced.contiguous()


__all__ = [
    "NATIVE_SLICED_BATCH_COMPILER_VERSION",
    "SPEC_BATCH_BINDING_SCHEMA_VERSION",
    "SPEC_BATCH_EXECUTION_TRACE_SCHEMA_VERSION",
    "NativePlainCrownSlicedBatchCompilation",
    "SlicedBatchExecutionTrace",
    "SpecBatchBindingTrace",
    "SpecBatchSliceBinding",
    "compile_native_plain_crown_sliced_batch_query",
    "execute_native_plain_crown_sliced_batch_query",
]
