"""Native parent/child input-domain batching with exact state recomputation."""

# pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-locals
# pylint: disable=too-many-statements,too-many-branches,invalid-name
# pylint: disable=missing-function-docstring,missing-class-docstring
# pylint: disable=too-many-lines,too-many-boolean-expressions

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
from ..planner.domain_batch_plan_variants import (
    build_native_domain_batch_plan_variants,
)
from ..planner.plan_ir_selector import PlanSelectionContext, select_plan_instance
from ..planner.representation_plan_binding import (
    BoundRepresentationBinding,
    bind_native_representation_plan,
    build_native_representation_plan_variants,
)
from .crown_ibp import _forward_ibp_trace_mlp
from .native_verifier_ir_integration import (
    NativePlainCrownRepresentationCompilation,
    compile_native_plain_crown_query,
    compile_native_plain_crown_representation_query,
    execute_native_plain_crown_representation_query,
)
from .perturbation import BoxPerturbation
from .task_executor import InputSpec
from .task_ir_executor import TaskExecutionTrace

NATIVE_DOMAIN_BATCH_COMPILER_VERSION = "boundflow.native-domain-batch-runtime/v1"
DOMAIN_QUERY_STATE_SCHEMA_VERSION = "boundflow.domain-query-state/v1"
DOMAIN_BATCH_BINDING_SCHEMA_VERSION = "boundflow.domain-batch-binding/v1"
DOMAIN_BATCH_EXECUTION_TRACE_SCHEMA_VERSION = (
    "boundflow.domain-batch-execution-trace/v1"
)
SERIAL_DOMAIN_TRACE_SCHEMA_VERSION = "boundflow.serial-domain-reference-trace/v1"
PARENT_STATE_VALIDITY = "warm_start_only"
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
class NativeDomainQuerySpec:
    """One leaf box plus its immediate parent box and deterministic tree lineage."""

    query_id: str
    parent_query_id: str
    input_spec: InputSpec
    parent_input_spec: InputSpec
    depth: int
    branch_ordinal: int

    def validate(self) -> None:
        if (
            not self.query_id
            or not self.parent_query_id
            or self.query_id == self.parent_query_id
            or self.depth <= 0
            or self.branch_ordinal not in (0, 1)
        ):
            raise ValueError("domain query tree identity is invalid")
        child_lower, child_upper = _box_bounds(self.input_spec, expected_batch=1)
        parent_lower, parent_upper = _box_bounds(
            self.parent_input_spec, expected_batch=1
        )
        if self.input_spec.value_name != self.parent_input_spec.value_name:
            raise ValueError("domain query parent/child input value differs")
        if (
            child_lower.shape != parent_lower.shape
            or not bool((child_lower >= parent_lower).all())
            or not bool((child_upper <= parent_upper).all())
            or (
                torch.equal(child_lower, parent_lower)
                and torch.equal(child_upper, parent_upper)
            )
        ):
            raise ValueError("domain query box is not a strict parent subset")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        child_lower, child_upper = _box_bounds(self.input_spec, expected_batch=1)
        parent_lower, parent_upper = _box_bounds(
            self.parent_input_spec, expected_batch=1
        )
        return {
            "query_id": self.query_id,
            "parent_query_id": self.parent_query_id,
            "depth": self.depth,
            "branch_ordinal": self.branch_ordinal,
            "input_lower_hash": tensor_content_hash(child_lower),
            "input_upper_hash": tensor_content_hash(child_upper),
            "parent_input_lower_hash": tensor_content_hash(parent_lower),
            "parent_input_upper_hash": tensor_content_hash(parent_upper),
        }


@dataclass(frozen=True)
class DomainQueryStateTrace:
    query_id: str
    parent_query_id: str
    input_lower_hash: str
    input_upper_hash: str
    exact_state_hash: str
    parent_state_hash: str
    parent_state_validity: str = PARENT_STATE_VALIDITY
    parent_state_consumed_as_exact: bool = False
    schema_version: str = DOMAIN_QUERY_STATE_SCHEMA_VERSION

    def validate(self) -> None:
        if (
            self.schema_version != DOMAIN_QUERY_STATE_SCHEMA_VERSION
            or not self.query_id
            or not self.parent_query_id
            or self.query_id == self.parent_query_id
        ):
            raise ValueError("domain query state identity differs")
        if any(
            len(value) != 64
            for value in (
                self.input_lower_hash,
                self.input_upper_hash,
                self.exact_state_hash,
                self.parent_state_hash,
            )
        ):
            raise ValueError("domain query state digest is not SHA-256")
        if (
            self.parent_state_validity != PARENT_STATE_VALIDITY
            or self.parent_state_consumed_as_exact is not False
            or self.parent_state_hash == self.exact_state_hash
        ):
            raise ValueError("parent state was promoted to exact child state")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "query_id": self.query_id,
            "parent_query_id": self.parent_query_id,
            "input_lower_hash": self.input_lower_hash,
            "input_upper_hash": self.input_upper_hash,
            "exact_state_hash": self.exact_state_hash,
            "parent_state_hash": self.parent_state_hash,
            "parent_state_validity": self.parent_state_validity,
            "parent_state_consumed_as_exact": self.parent_state_consumed_as_exact,
        }


@dataclass(frozen=True)
class NativeDomainExecutionPayload:
    query_specs: tuple[NativeDomainQuerySpec, ...]
    input_spec: InputSpec
    interval_env: Mapping[str, IntervalState]
    relu_pre: Mapping[str, IntervalState]
    linear_spec_C: torch.Tensor
    exact_state_hash: str

    def validate(self) -> None:
        if not self.query_specs or len(self.exact_state_hash) != 64:
            raise ValueError("domain execution payload identity is invalid")
        for item in self.query_specs:
            item.validate()
        query_ids = tuple(item.query_id for item in self.query_specs)
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("domain execution payload duplicates query IDs")
        lower, _upper = _box_bounds(
            self.input_spec, expected_batch=len(self.query_specs)
        )
        objective = _normalize_objective(
            self.linear_spec_C, domain_count=len(self.query_specs)
        )
        if int(lower.shape[0]) != int(objective.shape[0]):
            raise ValueError("domain execution payload batch axes differ")
        expected = _exact_state_hash(self.input_spec, self.interval_env, self.relu_pre)
        if expected != self.exact_state_hash:
            raise ValueError("domain execution payload exact state differs")


@dataclass(frozen=True)
class DomainBatchSliceBinding:
    slice_id: str
    start_index: int
    stop_index: int
    query_ids: tuple[str, ...]
    child_query_id: str
    child_exact_state_hash: str
    representation_policy_id: str
    child_ir_hashes: tuple[tuple[str, str], ...]

    def validate(self) -> None:
        hashes = dict(self.child_ir_hashes)
        if (
            not self.slice_id
            or self.start_index < 0
            or self.stop_index <= self.start_index
            or len(self.query_ids) != self.stop_index - self.start_index
            or len(self.query_ids) != len(set(self.query_ids))
            or not self.child_query_id
            or len(self.child_exact_state_hash) != 64
            or not self.representation_policy_id
            or len(hashes) != len(self.child_ir_hashes)
            or set(hashes) != _CHILD_HASH_KEYS
            or any(len(value) != 64 for value in hashes.values())
        ):
            raise ValueError("domain batch child slice binding differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "slice_id": self.slice_id,
            "start_index": self.start_index,
            "stop_index": self.stop_index,
            "query_ids": list(self.query_ids),
            "child_query_id": self.child_query_id,
            "child_exact_state_hash": self.child_exact_state_hash,
            "representation_policy_id": self.representation_policy_id,
            "child_ir_hashes": dict(self.child_ir_hashes),
        }


@dataclass(frozen=True)
class DomainBatchBindingTrace:
    source_bound_module_hash: str
    source_plan_template_hash: str
    source_plan_instance_hash: str
    source_schedule_hash: str
    source_representation_binding_hash: str
    source_execution_bound_module_hash: str
    source_exact_state_hash: str
    source_linear_spec_hash: str
    selected_representation_policy_id: str
    selected_storage_candidate_id: str
    selected_batch_candidate_id: str
    total_domain_count: int
    selected_domain_batch_size: int
    query_states: tuple[DomainQueryStateTrace, ...]
    slices: tuple[DomainBatchSliceBinding, ...]
    schema_version: str = DOMAIN_BATCH_BINDING_SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != DOMAIN_BATCH_BINDING_SCHEMA_VERSION:
            raise ValueError("unsupported domain batch binding schema")
        for name in (
            "source_bound_module_hash",
            "source_plan_template_hash",
            "source_plan_instance_hash",
            "source_schedule_hash",
            "source_representation_binding_hash",
            "source_execution_bound_module_hash",
            "source_exact_state_hash",
            "source_linear_spec_hash",
        ):
            if len(getattr(self, name)) != 64:
                raise ValueError(f"domain batch binding {name} is not SHA-256")
        if (
            not self.selected_representation_policy_id
            or not self.selected_storage_candidate_id
            or not self.selected_batch_candidate_id
            or self.total_domain_count <= 1
            or self.selected_domain_batch_size <= 0
            or self.selected_domain_batch_size > self.total_domain_count
            or len(self.query_states) != self.total_domain_count
            or not self.slices
        ):
            raise ValueError("domain batch source selection is invalid")
        for query_state in self.query_states:
            query_state.validate()
        query_ids = tuple(query_state.query_id for query_state in self.query_states)
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("domain batch query state IDs repeat")
        expected_start = 0
        sliced_ids: list[str] = []
        for batch_slice in self.slices:
            batch_slice.validate()
            if (
                batch_slice.start_index != expected_start
                or batch_slice.stop_index - batch_slice.start_index
                > self.selected_domain_batch_size
                or batch_slice.query_ids
                != query_ids[batch_slice.start_index : batch_slice.stop_index]
                or batch_slice.representation_policy_id
                != self.selected_representation_policy_id
            ):
                raise ValueError(
                    "domain batch slices overlap, reorder, or change policy"
                )
            expected_start = batch_slice.stop_index
            sliced_ids.extend(batch_slice.query_ids)
        if expected_start != self.total_domain_count or tuple(sliced_ids) != query_ids:
            raise ValueError("domain batch slices do not cover exact query order")

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
            "source_exact_state_hash": self.source_exact_state_hash,
            "source_linear_spec_hash": self.source_linear_spec_hash,
            "selected_representation_policy_id": (
                self.selected_representation_policy_id
            ),
            "selected_storage_candidate_id": self.selected_storage_candidate_id,
            "selected_batch_candidate_id": self.selected_batch_candidate_id,
            "total_domain_count": self.total_domain_count,
            "selected_domain_batch_size": self.selected_domain_batch_size,
            "query_states": [item.to_dict() for item in self.query_states],
            "slices": [item.to_dict() for item in self.slices],
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    def stable_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class NativeDomainBatchCompilation:
    query_id: str
    source_payload: NativeDomainExecutionPayload
    source_build: PlainCrownBoundIRBuild
    source_template: PlanTemplate
    source_instance: PlanInstance
    source_schedule: ScheduleModule
    source_representation_binding: BoundRepresentationBinding
    binding_trace: DomainBatchBindingTrace
    child_payloads: tuple[NativeDomainExecutionPayload, ...]
    child_compilations: tuple[NativePlainCrownRepresentationCompilation, ...]

    def validate(self) -> None:  # pylint: disable=too-many-statements
        if not self.query_id:
            raise ValueError("native domain batch query ID is empty")
        self.source_payload.validate()
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
        representation = self.source_representation_binding.trace
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
            or trace.source_representation_binding_hash != representation.stable_hash()
            or trace.source_execution_bound_module_hash
            != self.source_representation_binding.execution_bound_module.stable_hash()
            or trace.source_exact_state_hash != self.source_payload.exact_state_hash
            or trace.source_linear_spec_hash
            != tensor_content_hash(self.source_payload.linear_spec_C)
            or trace.selected_representation_policy_id != representation.policy_id
            or trace.selected_storage_candidate_id
            != self.source_instance.storage_decision.candidate_id
            or trace.selected_batch_candidate_id
            != self.source_instance.batch_decision.candidate_id
        ):
            raise ValueError("native domain batch source linkage differs")
        query_ids = tuple(item.query_id for item in self.source_payload.query_specs)
        if self.source_schedule.query_ids != query_ids:
            raise ValueError("native domain batch source query order differs")
        batch = next(
            item
            for item in self.source_template.batch_candidates
            if item.candidate_id == self.source_instance.batch_decision.candidate_id
        )
        if batch.domain_batch_size != trace.selected_domain_batch_size:
            raise ValueError("native domain batch selected width differs")
        loop = _batch_loop(self.source_schedule)
        if loop.axis != "domain":
            raise ValueError("native domain batch source Schedule axis differs")
        expected_schedule = tuple(
            (item.slice_id, item.query_ids) for item in loop.slices
        )
        actual_schedule = tuple(
            (item.slice_id, item.query_ids) for item in trace.slices
        )
        if expected_schedule != actual_schedule:
            raise ValueError("native domain batch Schedule/binding slices differ")
        if not (
            len(self.child_payloads)
            == len(self.child_compilations)
            == len(trace.slices)
        ):
            raise ValueError("native domain batch child count differs")
        for item, payload, child in zip(
            trace.slices, self.child_payloads, self.child_compilations
        ):
            payload.validate()
            child.validate()
            if (
                tuple(spec.query_id for spec in payload.query_specs) != item.query_ids
                or payload.exact_state_hash != item.child_exact_state_hash
                or child.intermediate_bounds_hash != item.child_exact_state_hash
                or tuple(sorted(child.hashes().items())) != item.child_ir_hashes
                or child.query_id != item.child_query_id
                or child.binding.trace.policy_id != item.representation_policy_id
                or child.source_instance.storage_decision.candidate_id
                != trace.selected_storage_candidate_id
                or child.source_template.workload.domain_batch_size
                != len(item.query_ids)
            ):
                raise ValueError("native domain batch child IR/state linkage differs")

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
            "domain_batch_binding_hash": self.binding_trace.stable_hash(),
        }


@dataclass(frozen=True)
class NativeDomainQueryResult:
    query_id: str
    parent_query_id: str
    result: IntervalState

    def validate(self) -> None:
        if not self.query_id or not self.parent_query_id:
            raise ValueError("domain result lineage is empty")
        self.result.validate()
        if int(self.result.lower.shape[0]) != 1:
            raise ValueError("domain result must own exactly one domain")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "query_id": self.query_id,
            "parent_query_id": self.parent_query_id,
            "shape": list(self.result.lower.shape),
            "lower_hash": tensor_content_hash(self.result.lower),
            "upper_hash": tensor_content_hash(self.result.upper),
        }


@dataclass(frozen=True)
class DomainBatchExecutionTrace:
    binding_hash: str
    parent_state_consumed_as_exact: bool
    packed_child_stack_count: int
    child_task_trace_hashes: tuple[str, ...]
    query_results: tuple[dict[str, object], ...]
    schema_version: str = DOMAIN_BATCH_EXECUTION_TRACE_SCHEMA_VERSION

    def validate(self) -> None:
        if (
            self.schema_version != DOMAIN_BATCH_EXECUTION_TRACE_SCHEMA_VERSION
            or len(self.binding_hash) != 64
            or self.parent_state_consumed_as_exact is not False
            or self.packed_child_stack_count <= 0
            or len(self.child_task_trace_hashes) != self.packed_child_stack_count
            or any(len(value) != 64 for value in self.child_task_trace_hashes)
            or not self.query_results
        ):
            raise ValueError("domain batch execution trace differs")
        ids = tuple(str(item.get("query_id", "")) for item in self.query_results)
        if any(not item for item in ids) or len(ids) != len(set(ids)):
            raise ValueError("domain batch execution result IDs differ")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "binding_hash": self.binding_hash,
            "parent_state_consumed_as_exact": self.parent_state_consumed_as_exact,
            "packed_child_stack_count": self.packed_child_stack_count,
            "child_task_trace_hashes": list(self.child_task_trace_hashes),
            "query_results": list(self.query_results),
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    def stable_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SerialDomainReferenceTrace:
    representation_policy_id: str
    storage_candidate_id: str
    parent_state_consumed_as_exact: bool
    serial_child_stack_count: int
    child_task_trace_hashes: tuple[str, ...]
    query_results: tuple[dict[str, object], ...]
    schema_version: str = SERIAL_DOMAIN_TRACE_SCHEMA_VERSION

    def validate(self) -> None:
        if (
            self.schema_version != SERIAL_DOMAIN_TRACE_SCHEMA_VERSION
            or not self.representation_policy_id
            or not self.storage_candidate_id
            or self.parent_state_consumed_as_exact is not False
            or self.serial_child_stack_count <= 0
            or len(self.child_task_trace_hashes) != self.serial_child_stack_count
            or len(self.query_results) != self.serial_child_stack_count
        ):
            raise ValueError("serial domain reference trace differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "representation_policy_id": self.representation_policy_id,
            "storage_candidate_id": self.storage_candidate_id,
            "parent_state_consumed_as_exact": self.parent_state_consumed_as_exact,
            "serial_child_stack_count": self.serial_child_stack_count,
            "child_task_trace_hashes": list(self.child_task_trace_hashes),
            "query_results": list(self.query_results),
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    def stable_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def build_deterministic_box_domain_queries(
    root_input_spec: InputSpec,
    *,
    root_query_id: str,
    split_depth: int,
) -> tuple[NativeDomainQuerySpec, ...]:
    """Bisect the first positive-width coordinates into ordered leaf boxes."""

    if not root_query_id or split_depth <= 0:
        raise ValueError("domain split root/depth is invalid")
    root_lower, root_upper = _box_bounds(root_input_spec, expected_batch=1)
    widths = (root_upper - root_lower).reshape(-1)
    split_indices = tuple(
        int(index)
        for index in torch.nonzero(widths > 0, as_tuple=False).flatten().tolist()
    )
    if len(split_indices) < split_depth:
        raise ValueError("domain split lacks enough positive-width coordinates")
    level: list[tuple[str, torch.Tensor, torch.Tensor]] = [
        (root_query_id, root_lower, root_upper)
    ]
    leaves: list[NativeDomainQuerySpec] = []
    for depth, flat_index in enumerate(split_indices[:split_depth], start=1):
        next_level: list[tuple[str, torch.Tensor, torch.Tensor]] = []
        for parent_index, (parent_id, lower, upper) in enumerate(level):
            midpoint = (
                lower.reshape(-1)[flat_index] + upper.reshape(-1)[flat_index]
            ) / 2
            for branch in (0, 1):
                child_lower = lower.clone()
                child_upper = upper.clone()
                if branch == 0:
                    child_upper.reshape(-1)[flat_index] = midpoint
                else:
                    child_lower.reshape(-1)[flat_index] = midpoint
                child_id = (
                    f"{root_query_id}:d{depth:02d}:n{parent_index * 2 + branch:04d}"
                )
                child_spec = InputSpec.box(
                    value_name=root_input_spec.value_name,
                    lower=child_lower,
                    upper=child_upper,
                )
                parent_spec = InputSpec.box(
                    value_name=root_input_spec.value_name,
                    lower=lower,
                    upper=upper,
                )
                if depth == split_depth:
                    leaves.append(
                        NativeDomainQuerySpec(
                            query_id=child_id,
                            parent_query_id=parent_id,
                            input_spec=child_spec,
                            parent_input_spec=parent_spec,
                            depth=depth,
                            branch_ordinal=branch,
                        )
                    )
                next_level.append((child_id, child_lower, child_upper))
        level = next_level
    result = tuple(leaves)
    if len(result) != 2**split_depth:
        raise ValueError("domain split leaf count differs")
    for item in result:
        item.validate()
    return result


# pylint: disable-next=too-many-arguments,too-many-locals,too-many-statements
def compile_native_domain_batch_query(
    legacy_task_module: BFTaskModule,
    query_specs: tuple[NativeDomainQuerySpec, ...],
    *,
    linear_spec_C: torch.Tensor,
    query_id: str,
    available_memory_bytes: int,
    memory_budget_bytes: int,
    domain_batch_candidate_size: int,
    max_domain_batch_size: int,
) -> NativeDomainBatchCompilation:
    """Recompute leaf states, select a domain batch, and compile each slice."""

    if not query_id or len(query_specs) <= 1:
        raise ValueError("native domain batch query identity/workload is invalid")
    if len({item.query_id for item in query_specs}) != len(query_specs):
        raise ValueError("native domain batch duplicates leaf query IDs")
    for item in query_specs:
        item.validate()
    if (
        domain_batch_candidate_size <= 0
        or domain_batch_candidate_size >= len(query_specs)
        or max_domain_batch_size <= 0
        or max_domain_batch_size > len(query_specs)
    ):
        raise ValueError("native domain batch candidate/runtime width is invalid")
    leaf_payloads: list[NativeDomainExecutionPayload] = []
    query_states: list[DomainQueryStateTrace] = []
    normalized_single = _normalize_objective(linear_spec_C, domain_count=1)
    for item in query_specs:
        interval_env, relu_pre = _forward_ibp_trace_mlp(
            legacy_task_module, item.input_spec
        )
        exact_hash = _exact_state_hash(item.input_spec, interval_env, relu_pre)
        parent_interval, parent_relu = _forward_ibp_trace_mlp(
            legacy_task_module, item.parent_input_spec
        )
        parent_hash = _exact_state_hash(
            item.parent_input_spec, parent_interval, parent_relu
        )
        child_lower, child_upper = _box_bounds(item.input_spec, expected_batch=1)
        payload = NativeDomainExecutionPayload(
            query_specs=(item,),
            input_spec=item.input_spec,
            interval_env=interval_env,
            relu_pre=relu_pre,
            linear_spec_C=normalized_single.clone(),
            exact_state_hash=exact_hash,
        )
        payload.validate()
        leaf_payloads.append(payload)
        query_states.append(
            DomainQueryStateTrace(
                query_id=item.query_id,
                parent_query_id=item.parent_query_id,
                input_lower_hash=tensor_content_hash(child_lower),
                input_upper_hash=tensor_content_hash(child_upper),
                exact_state_hash=exact_hash,
                parent_state_hash=parent_hash,
            )
        )
    source_payload = _combine_payloads(tuple(leaf_payloads))
    base = compile_native_plain_crown_query(
        legacy_task_module,
        source_payload.input_spec,
        interval_env=source_payload.interval_env,
        relu_pre=source_payload.relu_pre,
        linear_spec_C=source_payload.linear_spec_C,
        intermediate_bounds_hash=source_payload.exact_state_hash,
        query_id=query_id,
        available_memory_bytes=available_memory_bytes,
    )
    source = base.bound_module
    source_template = build_native_representation_plan_variants(
        base.template, bound_module=source
    )
    source_template = build_native_domain_batch_plan_variants(
        source_template,
        bound_module=source,
        domain_batch_size=domain_batch_candidate_size,
    )
    source_instance = select_plan_instance(
        source_template,
        bound_module=source,
        query_bucket_id=f"native-domain-batch:{query_id}",
        available_memory_bytes=available_memory_bytes,
        memory_budget_bytes=memory_budget_bytes,
        selection_context=PlanSelectionContext(
            query_distribution_id=f"native-domain-batch:{query_id}",
            max_domain_batch_size=max_domain_batch_size,
        ),
    )
    source_schedule = lower_plan_instance_to_reference_schedule(
        source,
        template=source_template,
        instance=source_instance,
        query_ids=tuple(item.query_id for item in query_specs),
    )
    source_binding = bind_native_representation_plan(
        source,
        template=source_template,
        instance=source_instance,
        schedule=source_schedule,
    )
    policy_id = source_binding.trace.policy_id
    storage_id = source_instance.storage_decision.candidate_id
    selected_batch = next(
        item
        for item in source_template.batch_candidates
        if item.candidate_id == source_instance.batch_decision.candidate_id
    )
    loop = _batch_loop(source_schedule)
    children: list[NativePlainCrownRepresentationCompilation] = []
    child_payloads: list[NativeDomainExecutionPayload] = []
    slice_bindings: list[DomainBatchSliceBinding] = []
    query_index = {item.query_id: index for index, item in enumerate(query_specs)}
    expected_start = 0
    for schedule_slice in loop.slices:
        indices = tuple(query_index[item] for item in schedule_slice.query_ids)
        start = indices[0]
        stop = indices[-1] + 1
        if indices != tuple(range(start, stop)) or start != expected_start:
            raise ValueError("native domain Schedule reordered leaf queries")
        expected_start = stop
        payload = _combine_payloads(tuple(leaf_payloads[start:stop]))
        child_query_id = f"{query_id}:domain:{start:04d}:{stop:04d}"
        child = compile_native_plain_crown_representation_query(
            legacy_task_module,
            payload.input_spec,
            interval_env=payload.interval_env,
            relu_pre=payload.relu_pre,
            linear_spec_C=payload.linear_spec_C,
            intermediate_bounds_hash=payload.exact_state_hash,
            query_id=child_query_id,
            available_memory_bytes=available_memory_bytes,
            memory_budget_bytes=available_memory_bytes,
            selection_context=PlanSelectionContext(
                query_distribution_id=f"native-domain-child:{child_query_id}",
                required_storage_candidate_id=storage_id,
            ),
        )
        if child.binding.trace.policy_id != policy_id:
            raise ValueError("native domain child changed representation policy")
        child_payloads.append(payload)
        children.append(child)
        slice_bindings.append(
            DomainBatchSliceBinding(
                slice_id=schedule_slice.slice_id,
                start_index=start,
                stop_index=stop,
                query_ids=schedule_slice.query_ids,
                child_query_id=child_query_id,
                child_exact_state_hash=payload.exact_state_hash,
                representation_policy_id=policy_id,
                child_ir_hashes=tuple(sorted(child.hashes().items())),
            )
        )
    trace = DomainBatchBindingTrace(
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
        source_representation_binding_hash=source_binding.trace.stable_hash(),
        source_execution_bound_module_hash=(
            source_binding.execution_bound_module.stable_hash()
        ),
        source_exact_state_hash=source_payload.exact_state_hash,
        source_linear_spec_hash=tensor_content_hash(source_payload.linear_spec_C),
        selected_representation_policy_id=policy_id,
        selected_storage_candidate_id=storage_id,
        selected_batch_candidate_id=source_instance.batch_decision.candidate_id,
        total_domain_count=len(query_specs),
        selected_domain_batch_size=selected_batch.domain_batch_size,
        query_states=tuple(query_states),
        slices=tuple(slice_bindings),
    )
    compilation = NativeDomainBatchCompilation(
        query_id=query_id,
        source_payload=source_payload,
        source_build=base.build,
        source_template=source_template,
        source_instance=source_instance,
        source_schedule=source_schedule,
        source_representation_binding=source_binding,
        binding_trace=trace,
        child_payloads=tuple(child_payloads),
        child_compilations=tuple(children),
    )
    compilation.validate()
    return compilation


def execute_native_domain_batch_query(
    compilation: NativeDomainBatchCompilation,
    *,
    legacy_task_module: BFTaskModule,
) -> tuple[tuple[NativeDomainQueryResult, ...], DomainBatchExecutionTrace]:
    """Execute domain-sliced children and restore exact leaf result order."""

    compilation.validate()
    results: list[IntervalState] = []
    task_traces: list[TaskExecutionTrace] = []
    for payload, child in zip(
        compilation.child_payloads, compilation.child_compilations
    ):
        result, task_trace = execute_native_plain_crown_representation_query(
            child,
            legacy_task_module=legacy_task_module,
            input_spec=payload.input_spec,
            relu_pre=payload.relu_pre,
            linear_spec_C=payload.linear_spec_C,
        )
        results.append(result)
        task_traces.append(task_trace)
    lower = torch.cat(tuple(item.lower for item in results), dim=0)
    upper = torch.cat(tuple(item.upper for item in results), dim=0)
    if int(lower.shape[0]) != compilation.binding_trace.total_domain_count:
        raise ValueError("native domain batch aggregate domain count differs")
    restored = tuple(
        NativeDomainQueryResult(
            query_id=spec.query_id,
            parent_query_id=spec.parent_query_id,
            result=IntervalState(
                lower=lower[index : index + 1].contiguous(),
                upper=upper[index : index + 1].contiguous(),
            ),
        )
        for index, spec in enumerate(compilation.source_payload.query_specs)
    )
    for item in restored:
        item.validate()
    trace = DomainBatchExecutionTrace(
        binding_hash=compilation.binding_trace.stable_hash(),
        parent_state_consumed_as_exact=False,
        packed_child_stack_count=len(compilation.child_compilations),
        child_task_trace_hashes=tuple(item.stable_hash() for item in task_traces),
        query_results=tuple(item.to_dict() for item in restored),
    )
    trace.validate()
    return restored, trace


def execute_native_domain_serial_reference(
    compilation: NativeDomainBatchCompilation,
    *,
    legacy_task_module: BFTaskModule,
    available_memory_bytes: int,
) -> tuple[tuple[NativeDomainQueryResult, ...], SerialDomainReferenceTrace]:
    """Compile and execute one same-policy IR stack per exact leaf domain."""

    compilation.validate()
    trace = compilation.binding_trace
    results: list[NativeDomainQueryResult] = []
    task_traces: list[TaskExecutionTrace] = []
    for payload in _leaf_payloads(compilation.source_payload):
        spec = payload.query_specs[0]
        child = compile_native_plain_crown_representation_query(
            legacy_task_module,
            payload.input_spec,
            interval_env=payload.interval_env,
            relu_pre=payload.relu_pre,
            linear_spec_C=payload.linear_spec_C,
            intermediate_bounds_hash=payload.exact_state_hash,
            query_id=f"{spec.query_id}:serial",
            available_memory_bytes=available_memory_bytes,
            memory_budget_bytes=available_memory_bytes,
            selection_context=PlanSelectionContext(
                query_distribution_id=f"native-domain-serial:{spec.query_id}",
                required_storage_candidate_id=trace.selected_storage_candidate_id,
            ),
        )
        if child.binding.trace.policy_id != trace.selected_representation_policy_id:
            raise ValueError("serial domain reference changed representation policy")
        result, task_trace = execute_native_plain_crown_representation_query(
            child,
            legacy_task_module=legacy_task_module,
            input_spec=payload.input_spec,
            relu_pre=payload.relu_pre,
            linear_spec_C=payload.linear_spec_C,
        )
        item = NativeDomainQueryResult(spec.query_id, spec.parent_query_id, result)
        item.validate()
        results.append(item)
        task_traces.append(task_trace)
    serial_trace = SerialDomainReferenceTrace(
        representation_policy_id=trace.selected_representation_policy_id,
        storage_candidate_id=trace.selected_storage_candidate_id,
        parent_state_consumed_as_exact=False,
        serial_child_stack_count=len(results),
        child_task_trace_hashes=tuple(item.stable_hash() for item in task_traces),
        query_results=tuple(item.to_dict() for item in results),
    )
    serial_trace.validate()
    return tuple(results), serial_trace


def _leaf_payloads(
    source: NativeDomainExecutionPayload,
) -> tuple[NativeDomainExecutionPayload, ...]:
    result: list[NativeDomainExecutionPayload] = []
    for index, spec in enumerate(source.query_specs):
        input_spec = _slice_input_spec(source.input_spec, index, index + 1)
        interval_env = _slice_interval_mapping(source.interval_env, index, index + 1)
        relu_pre = _slice_interval_mapping(source.relu_pre, index, index + 1)
        payload = NativeDomainExecutionPayload(
            query_specs=(spec,),
            input_spec=input_spec,
            interval_env=interval_env,
            relu_pre=relu_pre,
            linear_spec_C=source.linear_spec_C[index : index + 1].contiguous(),
            exact_state_hash=_exact_state_hash(input_spec, interval_env, relu_pre),
        )
        payload.validate()
        result.append(payload)
    return tuple(result)


def _combine_payloads(
    payloads: tuple[NativeDomainExecutionPayload, ...],
) -> NativeDomainExecutionPayload:
    if not payloads:
        raise ValueError("cannot combine empty domain payloads")
    for item in payloads:
        item.validate()
    lower = torch.cat(
        tuple(_box_bounds(item.input_spec)[0] for item in payloads), dim=0
    ).contiguous()
    upper = torch.cat(
        tuple(_box_bounds(item.input_spec)[1] for item in payloads), dim=0
    ).contiguous()
    input_spec = InputSpec.box(
        value_name=payloads[0].input_spec.value_name,
        lower=lower,
        upper=upper,
    )
    interval_env = _concat_interval_mappings(
        tuple(item.interval_env for item in payloads)
    )
    relu_pre = _concat_interval_mappings(tuple(item.relu_pre for item in payloads))
    objective = torch.cat(
        tuple(item.linear_spec_C for item in payloads), dim=0
    ).contiguous()
    result = NativeDomainExecutionPayload(
        query_specs=tuple(spec for item in payloads for spec in item.query_specs),
        input_spec=input_spec,
        interval_env=interval_env,
        relu_pre=relu_pre,
        linear_spec_C=objective,
        exact_state_hash=_exact_state_hash(input_spec, interval_env, relu_pre),
    )
    result.validate()
    return result


def _concat_interval_mappings(
    mappings: tuple[Mapping[str, IntervalState], ...],
) -> dict[str, IntervalState]:
    if not mappings:
        raise ValueError("cannot concatenate empty interval mappings")
    keys = tuple(sorted(mappings[0]))
    if any(tuple(sorted(item)) != keys for item in mappings):
        raise ValueError("domain interval mappings have different keys")
    result: dict[str, IntervalState] = {}
    for key in keys:
        result[key] = IntervalState(
            lower=torch.cat(tuple(item[key].lower for item in mappings), dim=0),
            upper=torch.cat(tuple(item[key].upper for item in mappings), dim=0),
        )
        result[key].validate()
    return result


def _slice_interval_mapping(
    mapping: Mapping[str, IntervalState], start: int, stop: int
) -> dict[str, IntervalState]:
    result = {
        key: IntervalState(
            lower=value.lower[start:stop].contiguous(),
            upper=value.upper[start:stop].contiguous(),
        )
        for key, value in mapping.items()
    }
    for item in result.values():
        item.validate()
    return result


def _slice_input_spec(input_spec: InputSpec, start: int, stop: int) -> InputSpec:
    lower, upper = _box_bounds(input_spec)
    return InputSpec.box(
        value_name=input_spec.value_name,
        lower=lower[start:stop].contiguous(),
        upper=upper[start:stop].contiguous(),
    )


def _box_bounds(
    input_spec: InputSpec, *, expected_batch: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(input_spec.perturbation, BoxPerturbation):
        raise TypeError("native domain batching requires BoxPerturbation")
    lower, upper = input_spec.perturbation.bounding_box(input_spec.center)
    if lower.dim() < 2:
        raise ValueError("native domain boxes require an explicit batch axis")
    if expected_batch is not None and int(lower.shape[0]) != expected_batch:
        raise ValueError("native domain box batch size differs")
    return lower, upper


def _normalize_objective(
    linear_spec_C: torch.Tensor, *, domain_count: int
) -> torch.Tensor:
    if not torch.is_tensor(linear_spec_C) or not torch.is_floating_point(linear_spec_C):
        raise TypeError("native domain objective must be a floating tensor")
    if not bool(torch.isfinite(linear_spec_C).all()):
        raise ValueError("native domain objective must be finite")
    if linear_spec_C.dim() == 2:
        objective = linear_spec_C.unsqueeze(0)
    elif linear_spec_C.dim() == 3:
        objective = linear_spec_C
    else:
        raise ValueError("native domain objective must have rank two or three")
    if int(objective.shape[0]) == 1 and domain_count > 1:
        objective = objective.expand(domain_count, -1, -1)
    if int(objective.shape[0]) != domain_count:
        raise ValueError("native domain objective/domain batch differs")
    if int(objective.shape[1]) <= 0 or int(objective.shape[2]) <= 0:
        raise ValueError("native domain objective shape is empty")
    return objective.contiguous()


def _interval_mapping_hash(mapping: Mapping[str, IntervalState]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(mapping.items()):
        value.validate()
        digest.update(key.encode("utf-8"))
        digest.update(tensor_content_hash(value.lower).encode("utf-8"))
        digest.update(tensor_content_hash(value.upper).encode("utf-8"))
    return digest.hexdigest()


def _exact_state_hash(
    input_spec: InputSpec,
    interval_env: Mapping[str, IntervalState],
    relu_pre: Mapping[str, IntervalState],
) -> str:
    lower, upper = _box_bounds(input_spec)
    payload = "|".join(
        (
            tensor_content_hash(lower),
            tensor_content_hash(upper),
            _interval_mapping_hash(interval_env),
            _interval_mapping_hash(relu_pre),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _batch_loop(schedule: ScheduleModule) -> BatchLoopAction:
    loops = tuple(
        item for item in schedule.actions if isinstance(item, BatchLoopAction)
    )
    if len(loops) != 1:
        raise ValueError("native domain Schedule batch loop count differs")
    return loops[0]


__all__ = [
    "DOMAIN_BATCH_BINDING_SCHEMA_VERSION",
    "DOMAIN_BATCH_EXECUTION_TRACE_SCHEMA_VERSION",
    "DOMAIN_QUERY_STATE_SCHEMA_VERSION",
    "NATIVE_DOMAIN_BATCH_COMPILER_VERSION",
    "PARENT_STATE_VALIDITY",
    "SERIAL_DOMAIN_TRACE_SCHEMA_VERSION",
    "DomainBatchBindingTrace",
    "DomainBatchExecutionTrace",
    "DomainBatchSliceBinding",
    "DomainQueryStateTrace",
    "NativeDomainBatchCompilation",
    "NativeDomainQueryResult",
    "NativeDomainQuerySpec",
    "SerialDomainReferenceTrace",
    "build_deterministic_box_domain_queries",
    "compile_native_domain_batch_query",
    "execute_native_domain_batch_query",
    "execute_native_domain_serial_reference",
]
