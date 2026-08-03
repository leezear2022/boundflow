"""Native repeated-query packing, exact cache, and result restoration."""

# pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-locals
# pylint: disable=too-many-statements,too-many-branches,invalid-name
# pylint: disable=missing-function-docstring,too-many-boolean-expressions
# pylint: disable=missing-class-docstring

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Mapping

import torch

from ..domains.interval import IntervalState
from ..frontends.plain_crown_bound_ir import tensor_content_hash
from ..ir.task import BFTaskModule
from ..planner.plan_ir_selector import PlanSelectionContext
from .native_joint_policy_integration import (
    NativePlainCrownJointPolicyCompilation,
    compile_native_plain_crown_joint_policy_query,
    execute_native_plain_crown_joint_policy_query,
)
from .native_verifier_ir_integration import (
    compile_native_plain_crown_representation_query,
    execute_native_plain_crown_representation_query,
)
from .task_executor import InputSpec

NATIVE_REPEATED_QUERY_COMPILER_VERSION = "boundflow.native-repeated-query-runtime/v1"
REPEATED_QUERY_LAYOUT_SCHEMA_VERSION = "boundflow.repeated-query-layout/v1"
REPEATED_QUERY_EXECUTION_TRACE_SCHEMA_VERSION = (
    "boundflow.repeated-query-execution-trace/v1"
)
SERIAL_QUERY_TRACE_SCHEMA_VERSION = "boundflow.serial-query-reference-trace/v1"


@dataclass(frozen=True)
class NativeRepeatedQuerySpec:
    """One independently named query with one or more linear objectives."""

    query_id: str
    linear_spec_C: torch.Tensor

    def validate(self) -> None:
        if not self.query_id:
            raise ValueError("repeated query ID must be non-empty")
        if not torch.is_tensor(self.linear_spec_C):
            raise TypeError("repeated query objective must be a tensor")
        if self.linear_spec_C.dim() == 2:
            specs = int(self.linear_spec_C.shape[0])
        elif self.linear_spec_C.dim() == 3:
            if int(self.linear_spec_C.shape[0]) != 1:
                raise ValueError("repeated query v1 requires one input domain")
            specs = int(self.linear_spec_C.shape[1])
        else:
            raise ValueError("repeated query objective must have rank two or three")
        if specs <= 0 or not torch.is_floating_point(self.linear_spec_C):
            raise ValueError("repeated query objective is empty or non-floating")
        if not bool(torch.isfinite(self.linear_spec_C).all()):
            raise ValueError("repeated query objective must be finite")

    @property
    def spec_count(self) -> int:
        self.validate()
        return int(
            self.linear_spec_C.shape[0]
            if self.linear_spec_C.dim() == 2
            else self.linear_spec_C.shape[1]
        )

    @property
    def objective_hash(self) -> str:
        self.validate()
        return tensor_content_hash(self.normalized_objective())

    def normalized_objective(self) -> torch.Tensor:
        self.validate()
        value = (
            self.linear_spec_C.unsqueeze(0)
            if self.linear_spec_C.dim() == 2
            else self.linear_spec_C
        )
        return value.contiguous()

    def to_dict(self) -> dict[str, object]:
        objective = self.normalized_objective()
        return {
            "query_id": self.query_id,
            "spec_count": self.spec_count,
            "shape": list(objective.shape),
            "dtype": str(objective.dtype),
            "device": str(objective.device),
            "objective_hash": self.objective_hash,
        }


@dataclass(frozen=True)
class RepeatedQueryRange:
    query_id: str
    start_index: int
    stop_index: int
    objective_hash: str

    def validate(self) -> None:
        if (
            not self.query_id
            or self.start_index < 0
            or self.stop_index <= self.start_index
            or len(self.objective_hash) != 64
        ):
            raise ValueError("repeated query range is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "query_id": self.query_id,
            "start_index": self.start_index,
            "stop_index": self.stop_index,
            "objective_hash": self.objective_hash,
        }


@dataclass(frozen=True)
class RepeatedQueryLayoutTrace:
    """Exact workload/state/query layout and selected joint policy."""

    stream_id: str
    workload_identity_hash: str
    state_identity_hash: str
    input_identity_hash: str
    intermediate_bounds_hash: str
    packed_objective_hash: str
    available_memory_bytes: int
    memory_budget_bytes: int
    spec_slice_candidate_size: int
    max_spec_batch_size: int
    representation_policy_id: str
    storage_candidate_id: str
    batch_candidate_id: str
    joint_policy_binding_hash: str
    query_ranges: tuple[RepeatedQueryRange, ...]
    schema_version: str = REPEATED_QUERY_LAYOUT_SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != REPEATED_QUERY_LAYOUT_SCHEMA_VERSION:
            raise ValueError("unsupported repeated query layout schema")
        if not self.stream_id:
            raise ValueError("repeated query stream ID is empty")
        for name in (
            "workload_identity_hash",
            "state_identity_hash",
            "input_identity_hash",
            "intermediate_bounds_hash",
            "packed_objective_hash",
            "joint_policy_binding_hash",
        ):
            if len(getattr(self, name)) != 64:
                raise ValueError(f"repeated query layout {name} is not SHA-256")
        if (
            self.available_memory_bytes <= 0
            or self.memory_budget_bytes <= 0
            or self.spec_slice_candidate_size <= 0
            or self.max_spec_batch_size <= 0
            or not self.representation_policy_id
            or not self.storage_candidate_id
            or not self.batch_candidate_id
            or not self.query_ranges
        ):
            raise ValueError("repeated query layout policy/configuration is invalid")
        expected_start = 0
        query_ids: list[str] = []
        for item in self.query_ranges:
            item.validate()
            if item.start_index != expected_start:
                raise ValueError("repeated query layout ranges overlap or reorder")
            expected_start = item.stop_index
            query_ids.append(item.query_id)
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("repeated query layout duplicates query IDs")

    @property
    def total_spec_count(self) -> int:
        self.validate()
        return self.query_ranges[-1].stop_index

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "stream_id": self.stream_id,
            "workload_identity_hash": self.workload_identity_hash,
            "state_identity_hash": self.state_identity_hash,
            "input_identity_hash": self.input_identity_hash,
            "intermediate_bounds_hash": self.intermediate_bounds_hash,
            "packed_objective_hash": self.packed_objective_hash,
            "available_memory_bytes": self.available_memory_bytes,
            "memory_budget_bytes": self.memory_budget_bytes,
            "spec_slice_candidate_size": self.spec_slice_candidate_size,
            "max_spec_batch_size": self.max_spec_batch_size,
            "representation_policy_id": self.representation_policy_id,
            "storage_candidate_id": self.storage_candidate_id,
            "batch_candidate_id": self.batch_candidate_id,
            "joint_policy_binding_hash": self.joint_policy_binding_hash,
            "query_ranges": [item.to_dict() for item in self.query_ranges],
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    def stable_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class NativeRepeatedQueryCompilation:
    """Packed joint compilation plus per-query layout and exact cache key."""

    query_specs: tuple[NativeRepeatedQuerySpec, ...]
    packed_objective: torch.Tensor
    joint_compilation: NativePlainCrownJointPolicyCompilation
    layout_trace: RepeatedQueryLayoutTrace
    cache_key: str

    def validate(self) -> None:
        if len(self.cache_key) != 64 or not self.query_specs:
            raise ValueError("repeated query compilation identity is invalid")
        for item in self.query_specs:
            item.validate()
        if len({item.query_id for item in self.query_specs}) != len(self.query_specs):
            raise ValueError("repeated query compilation duplicates query IDs")
        self.joint_compilation.validate()
        self.layout_trace.validate()
        if tensor_content_hash(self.packed_objective) != (
            self.layout_trace.packed_objective_hash
        ):
            raise ValueError("repeated query packed objective hash differs")
        ranges = _query_ranges(self.query_specs)
        if ranges != self.layout_trace.query_ranges:
            raise ValueError("repeated query spec/layout ranges differ")
        joint = self.joint_compilation
        if (
            joint.binding_trace.source_linear_spec_hash
            != self.layout_trace.packed_objective_hash
            or joint.binding_trace.total_spec_count
            != self.layout_trace.total_spec_count
            or joint.binding_trace.selected_representation_policy_id
            != self.layout_trace.representation_policy_id
            or joint.binding_trace.selected_storage_candidate_id
            != self.layout_trace.storage_candidate_id
            or joint.binding_trace.selected_batch_candidate_id
            != self.layout_trace.batch_candidate_id
            or joint.binding_trace.stable_hash()
            != self.layout_trace.joint_policy_binding_hash
        ):
            raise ValueError("repeated query layout/joint compilation linkage differs")
        expected_key = _cache_key(
            stream_id=self.layout_trace.stream_id,
            workload_identity_hash=self.layout_trace.workload_identity_hash,
            state_identity_hash=self.layout_trace.state_identity_hash,
            input_identity_hash=self.layout_trace.input_identity_hash,
            intermediate_bounds_hash=self.layout_trace.intermediate_bounds_hash,
            query_specs=self.query_specs,
            available_memory_bytes=self.layout_trace.available_memory_bytes,
            memory_budget_bytes=self.layout_trace.memory_budget_bytes,
            spec_slice_candidate_size=self.layout_trace.spec_slice_candidate_size,
            max_spec_batch_size=self.layout_trace.max_spec_batch_size,
        )
        if expected_key != self.cache_key:
            raise ValueError("repeated query compilation cache key differs")


@dataclass(frozen=True)
class RepeatedQueryCompileResult:
    compilation: NativeRepeatedQueryCompilation
    cache_hit: bool

    def validate(self) -> None:
        self.compilation.validate()
        if not isinstance(self.cache_hit, bool):
            raise TypeError("repeated query cache-hit flag must be boolean")


@dataclass
class NativeRepeatedQueryCompilationCache:
    """Exact in-process compilation cache with observable hit/miss counts."""

    entries: dict[str, NativeRepeatedQueryCompilation] = field(default_factory=dict)
    hit_count: int = 0
    miss_count: int = 0

    def lookup(self, key: str) -> NativeRepeatedQueryCompilation | None:
        if len(key) != 64:
            raise ValueError("repeated query cache key is not SHA-256")
        value = self.entries.get(key)
        if value is None:
            self.miss_count += 1
            return None
        value.validate()
        self.hit_count += 1
        return value

    def store(self, compilation: NativeRepeatedQueryCompilation) -> None:
        compilation.validate()
        existing = self.entries.get(compilation.cache_key)
        if existing is not None and existing.layout_trace.canonical_json() != (
            compilation.layout_trace.canonical_json()
        ):
            raise ValueError("repeated query cache key collision")
        self.entries[compilation.cache_key] = compilation


@dataclass(frozen=True)
class NativeRepeatedQueryResult:
    query_id: str
    result: IntervalState

    def validate(self) -> None:
        if not self.query_id:
            raise ValueError("repeated query result ID is empty")
        self.result.validate()

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "query_id": self.query_id,
            "lower_hash": tensor_content_hash(self.result.lower),
            "upper_hash": tensor_content_hash(self.result.upper),
            "shape": list(self.result.lower.shape),
        }


@dataclass(frozen=True)
class RepeatedQueryExecutionTrace:
    layout_hash: str
    cache_key: str
    cache_hit: bool
    joint_execution_trace_hash: str
    packed_child_stack_count: int
    query_results: tuple[dict[str, object], ...]
    schema_version: str = REPEATED_QUERY_EXECUTION_TRACE_SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != REPEATED_QUERY_EXECUTION_TRACE_SCHEMA_VERSION:
            raise ValueError("unsupported repeated query execution trace schema")
        if any(
            len(value) != 64
            for value in (
                self.layout_hash,
                self.cache_key,
                self.joint_execution_trace_hash,
            )
        ):
            raise ValueError("repeated query execution identity differs")
        if (
            not isinstance(self.cache_hit, bool)
            or self.packed_child_stack_count <= 0
            or not self.query_results
        ):
            raise ValueError("repeated query execution accounting is invalid")
        query_ids = [str(item.get("query_id", "")) for item in self.query_results]
        if any(not item for item in query_ids) or len(query_ids) != len(set(query_ids)):
            raise ValueError("repeated query execution duplicates result IDs")
        for item in self.query_results:
            if (
                len(str(item.get("lower_hash", ""))) != 64
                or len(str(item.get("upper_hash", ""))) != 64
            ):
                raise ValueError("repeated query result digest differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "layout_hash": self.layout_hash,
            "cache_key": self.cache_key,
            "cache_hit": self.cache_hit,
            "joint_execution_trace_hash": self.joint_execution_trace_hash,
            "packed_child_stack_count": self.packed_child_stack_count,
            "query_results": list(self.query_results),
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    def stable_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SerialQueryReferenceTrace:
    representation_policy_id: str
    storage_candidate_id: str
    serial_child_stack_count: int
    query_results: tuple[dict[str, object], ...]
    schema_version: str = SERIAL_QUERY_TRACE_SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SERIAL_QUERY_TRACE_SCHEMA_VERSION:
            raise ValueError("unsupported serial query trace schema")
        if (
            not self.representation_policy_id
            or not self.storage_candidate_id
            or self.serial_child_stack_count <= 0
            or len(self.query_results) != self.serial_child_stack_count
        ):
            raise ValueError("serial query reference accounting differs")
        query_ids = [str(item.get("query_id", "")) for item in self.query_results]
        if any(not item for item in query_ids) or len(query_ids) != len(set(query_ids)):
            raise ValueError("serial query reference duplicates result IDs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "representation_policy_id": self.representation_policy_id,
            "storage_candidate_id": self.storage_candidate_id,
            "serial_child_stack_count": self.serial_child_stack_count,
            "query_results": list(self.query_results),
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    def stable_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


# pylint: disable-next=too-many-arguments
def compile_native_repeated_query_stream(
    cache: NativeRepeatedQueryCompilationCache,
    legacy_task_module: BFTaskModule,
    input_spec: InputSpec,
    query_specs: tuple[NativeRepeatedQuerySpec, ...],
    *,
    interval_env: Mapping[str, IntervalState],
    relu_pre: Mapping[str, IntervalState],
    intermediate_bounds_hash: str,
    stream_id: str,
    workload_identity_hash: str,
    state_identity_hash: str,
    available_memory_bytes: int,
    memory_budget_bytes: int,
    spec_slice_candidate_size: int,
    max_spec_batch_size: int,
) -> RepeatedQueryCompileResult:
    """Compile or exactly reuse one compatible repeated-query stream."""

    if not stream_id or not query_specs:
        raise ValueError("repeated query stream identity/specs are empty")
    if len(workload_identity_hash) != 64 or len(state_identity_hash) != 64:
        raise ValueError("repeated query workload/state identity is invalid")
    if len(intermediate_bounds_hash) != 64:
        raise ValueError("repeated query intermediate-bound identity is invalid")
    for item in query_specs:
        item.validate()
    if len({item.query_id for item in query_specs}) != len(query_specs):
        raise ValueError("repeated query stream duplicates query IDs")
    input_hash = _input_identity_hash(input_spec)
    key = _cache_key(
        stream_id=stream_id,
        workload_identity_hash=workload_identity_hash,
        state_identity_hash=state_identity_hash,
        input_identity_hash=input_hash,
        intermediate_bounds_hash=intermediate_bounds_hash,
        query_specs=query_specs,
        available_memory_bytes=available_memory_bytes,
        memory_budget_bytes=memory_budget_bytes,
        spec_slice_candidate_size=spec_slice_candidate_size,
        max_spec_batch_size=max_spec_batch_size,
    )
    cached = cache.lookup(key)
    if cached is not None:
        result = RepeatedQueryCompileResult(cached, cache_hit=True)
        result.validate()
        return result
    packed = torch.cat(
        tuple(item.normalized_objective() for item in query_specs), dim=1
    ).contiguous()
    joint = compile_native_plain_crown_joint_policy_query(
        legacy_task_module,
        input_spec,
        interval_env=interval_env,
        relu_pre=relu_pre,
        linear_spec_C=packed,
        intermediate_bounds_hash=intermediate_bounds_hash,
        query_id=stream_id,
        available_memory_bytes=available_memory_bytes,
        memory_budget_bytes=memory_budget_bytes,
        spec_slice_candidate_size=spec_slice_candidate_size,
        max_spec_batch_size=max_spec_batch_size,
    )
    layout = RepeatedQueryLayoutTrace(
        stream_id=stream_id,
        workload_identity_hash=workload_identity_hash,
        state_identity_hash=state_identity_hash,
        input_identity_hash=input_hash,
        intermediate_bounds_hash=intermediate_bounds_hash,
        packed_objective_hash=tensor_content_hash(packed),
        available_memory_bytes=available_memory_bytes,
        memory_budget_bytes=memory_budget_bytes,
        spec_slice_candidate_size=spec_slice_candidate_size,
        max_spec_batch_size=max_spec_batch_size,
        representation_policy_id=(
            joint.binding_trace.selected_representation_policy_id
        ),
        storage_candidate_id=joint.binding_trace.selected_storage_candidate_id,
        batch_candidate_id=joint.binding_trace.selected_batch_candidate_id,
        joint_policy_binding_hash=joint.binding_trace.stable_hash(),
        query_ranges=_query_ranges(query_specs),
    )
    compilation = NativeRepeatedQueryCompilation(
        query_specs=query_specs,
        packed_objective=packed,
        joint_compilation=joint,
        layout_trace=layout,
        cache_key=key,
    )
    compilation.validate()
    cache.store(compilation)
    result = RepeatedQueryCompileResult(compilation, cache_hit=False)
    result.validate()
    return result


def execute_native_repeated_query_stream(
    compiled: RepeatedQueryCompileResult,
    *,
    legacy_task_module: BFTaskModule,
    input_spec: InputSpec,
    relu_pre: Mapping[str, IntervalState],
) -> tuple[tuple[NativeRepeatedQueryResult, ...], RepeatedQueryExecutionTrace]:
    """Execute packed child stacks and restore exact per-query result ranges."""

    compiled.validate()
    compilation = compiled.compilation
    packed_result, joint_trace = execute_native_plain_crown_joint_policy_query(
        compilation.joint_compilation,
        legacy_task_module=legacy_task_module,
        input_spec=input_spec,
        relu_pre=relu_pre,
        linear_spec_C=compilation.packed_objective,
    )
    results = tuple(
        NativeRepeatedQueryResult(
            query_id=item.query_id,
            result=IntervalState(
                lower=packed_result.lower[:, item.start_index : item.stop_index],
                upper=packed_result.upper[:, item.start_index : item.stop_index],
            ),
        )
        for item in compilation.layout_trace.query_ranges
    )
    for item in results:
        item.validate()
    trace = RepeatedQueryExecutionTrace(
        layout_hash=compilation.layout_trace.stable_hash(),
        cache_key=compilation.cache_key,
        cache_hit=compiled.cache_hit,
        joint_execution_trace_hash=joint_trace.stable_hash(),
        packed_child_stack_count=len(compilation.joint_compilation.child_compilations),
        query_results=tuple(item.to_dict() for item in results),
    )
    trace.validate()
    return results, trace


def execute_native_repeated_query_serial_reference(
    compilation: NativeRepeatedQueryCompilation,
    *,
    legacy_task_module: BFTaskModule,
    input_spec: InputSpec,
    interval_env: Mapping[str, IntervalState],
    relu_pre: Mapping[str, IntervalState],
) -> tuple[tuple[NativeRepeatedQueryResult, ...], SerialQueryReferenceTrace]:
    """Compile/execute every query separately under the packed source policy."""

    compilation.validate()
    policy_id = compilation.layout_trace.representation_policy_id
    storage_id = compilation.layout_trace.storage_candidate_id
    results: list[NativeRepeatedQueryResult] = []
    for item in compilation.query_specs:
        child = compile_native_plain_crown_representation_query(
            legacy_task_module,
            input_spec,
            interval_env=interval_env,
            relu_pre=relu_pre,
            linear_spec_C=item.normalized_objective(),
            intermediate_bounds_hash=compilation.layout_trace.intermediate_bounds_hash,
            query_id=f"{compilation.layout_trace.stream_id}:serial:{item.query_id}",
            available_memory_bytes=(compilation.layout_trace.available_memory_bytes),
            memory_budget_bytes=(compilation.layout_trace.available_memory_bytes),
            selection_context=PlanSelectionContext(
                query_distribution_id=(
                    f"native-repeated-serial:{compilation.layout_trace.stream_id}:"
                    f"{item.query_id}"
                ),
                required_storage_candidate_id=storage_id,
            ),
        )
        if child.binding.trace.policy_id != policy_id:
            raise ValueError("serial query reference changed representation policy")
        result, _task_trace = execute_native_plain_crown_representation_query(
            child,
            legacy_task_module=legacy_task_module,
            input_spec=input_spec,
            relu_pre=relu_pre,
            linear_spec_C=item.normalized_objective(),
        )
        results.append(NativeRepeatedQueryResult(item.query_id, result))
    result_tuple = tuple(results)
    for result_item in result_tuple:
        result_item.validate()
    trace = SerialQueryReferenceTrace(
        representation_policy_id=policy_id,
        storage_candidate_id=storage_id,
        serial_child_stack_count=len(result_tuple),
        query_results=tuple(item.to_dict() for item in result_tuple),
    )
    trace.validate()
    return result_tuple, trace


def _query_ranges(
    query_specs: tuple[NativeRepeatedQuerySpec, ...],
) -> tuple[RepeatedQueryRange, ...]:
    ranges: list[RepeatedQueryRange] = []
    start = 0
    for item in query_specs:
        stop = start + item.spec_count
        ranges.append(
            RepeatedQueryRange(
                query_id=item.query_id,
                start_index=start,
                stop_index=stop,
                objective_hash=item.objective_hash,
            )
        )
        start = stop
    return tuple(ranges)


def _input_identity_hash(input_spec: InputSpec) -> str:
    if not input_spec.value_name:
        raise ValueError("repeated query input value name is empty")
    lower, upper = input_spec.perturbation.bounding_box(input_spec.center)
    payload = json.dumps(
        {
            "value_name": input_spec.value_name,
            "center_hash": tensor_content_hash(input_spec.center),
            "lower_hash": tensor_content_hash(lower),
            "upper_hash": tensor_content_hash(upper),
            "perturbation_id": input_spec.perturbation.perturbation_id,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_key(
    *,
    stream_id: str,
    workload_identity_hash: str,
    state_identity_hash: str,
    input_identity_hash: str,
    intermediate_bounds_hash: str,
    query_specs: tuple[NativeRepeatedQuerySpec, ...],
    available_memory_bytes: int,
    memory_budget_bytes: int,
    spec_slice_candidate_size: int,
    max_spec_batch_size: int,
) -> str:
    payload = json.dumps(
        {
            "compiler": NATIVE_REPEATED_QUERY_COMPILER_VERSION,
            "stream_id": stream_id,
            "workload_identity_hash": workload_identity_hash,
            "state_identity_hash": state_identity_hash,
            "input_identity_hash": input_identity_hash,
            "intermediate_bounds_hash": intermediate_bounds_hash,
            "queries": [item.to_dict() for item in query_specs],
            "available_memory_bytes": available_memory_bytes,
            "memory_budget_bytes": memory_budget_bytes,
            "spec_slice_candidate_size": spec_slice_candidate_size,
            "max_spec_batch_size": max_spec_batch_size,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "NATIVE_REPEATED_QUERY_COMPILER_VERSION",
    "REPEATED_QUERY_EXECUTION_TRACE_SCHEMA_VERSION",
    "REPEATED_QUERY_LAYOUT_SCHEMA_VERSION",
    "SERIAL_QUERY_TRACE_SCHEMA_VERSION",
    "NativeRepeatedQueryCompilation",
    "NativeRepeatedQueryCompilationCache",
    "NativeRepeatedQueryResult",
    "NativeRepeatedQuerySpec",
    "RepeatedQueryCompileResult",
    "RepeatedQueryExecutionTrace",
    "RepeatedQueryLayoutTrace",
    "RepeatedQueryRange",
    "SerialQueryReferenceTrace",
    "compile_native_repeated_query_stream",
    "execute_native_repeated_query_serial_reference",
    "execute_native_repeated_query_stream",
]
