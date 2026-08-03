"""Native plain-CROWN compilation through Bound, Plan, Task, and Schedule IR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from ..domains.interval import IntervalState
from ..frontends.plain_crown_bound_ir import (
    PlainCrownBoundIRBuild,
    build_plain_crown_bound_ir,
)
from ..ir.bound import (
    BFBoundModule,
    BoundOpKind,
    BoundRepresentation,
    IntermediateBoundSource,
    ReluLowerSlopePolicy,
)
from ..ir.plan import (
    BackendCapabilitySpec,
    BackendKind,
    HardwareProfile,
    PlanCost,
    PlanInstance,
    PlanProvenance,
    PlanTemplate,
    WorkloadProfile,
)
from ..ir.schedule import (
    EmitResultAction,
    LaunchAction,
    ScheduleModule,
    lower_plan_instance_to_reference_schedule,
)
from ..ir.task import BFTaskModule
from ..ir.task_v1 import TaskIRKind, TaskIRModule, lower_plan_instance_to_task_ir
from ..planner.plan_ir_builder import (
    BackendEvidence,
    BatchEvidence,
    ReferencePlanEvidence,
    RegionEvidence,
    RepresentationEvidence,
    StorageEvidence,
    build_reference_plan_template,
)
from ..planner.plan_ir_selector import select_plan_instance
from ..planner.storage_plan_variants import build_native_storage_plan_variants
from .task_executor import InputSpec
from .task_ir_executor import (
    PreparedTaskIRExecution,
    TaskExecutionTrace,
    TaskTraceMode,
    execute_task_ir_semantics,
)
from .storage_plan_runtime import (
    PreparedStoragePlanRuntime,
    StorageExecutionTrace,
    StoragePlanRuntime,
)
from .task_backend_dispatch import TypedTaskBackend

NATIVE_PLAIN_CROWN_COMPILER_VERSION = "boundflow.native-plain-crown-ir/v1"
NATIVE_PLAIN_CROWN_MEMORY_COMPILER_VERSION = "boundflow.native-plain-crown-memory-ir/v1"


@dataclass(frozen=True)
class NativePlainCrownIRCompilation:
    """Fully validated native compilation for one plain-CROWN query."""

    query_id: str
    intermediate_bounds_hash: str
    build: PlainCrownBoundIRBuild
    template: PlanTemplate
    instance: PlanInstance
    task_module: TaskIRModule
    schedule: ScheduleModule

    @property
    def bound_module(self) -> BFBoundModule:
        """Return the native Bound IR module owned by this compilation."""

        return self.build.module

    def validate(self) -> None:
        """Prove native ownership and exact cross-layer linkage."""

        if not self.query_id:
            raise ValueError("native plain-CROWN query ID must be non-empty")
        if len(self.intermediate_bounds_hash) != 64:
            raise ValueError("native plain-CROWN intermediate-bound hash is invalid")
        module = self.bound_module
        module.validate()
        self.task_module.validate_schedule_linkage(
            self.schedule,
            bound_module=module,
            template=self.template,
            instance=self.instance,
        )
        if any(
            op.kind == BoundOpKind.EXTERNAL_VERIFIER_CALL for op in module.graph.ops
        ):
            raise ValueError("native plain-CROWN Bound IR contains an external call")
        expected_state_version = (
            f"external-intermediate-bounds:{self.intermediate_bounds_hash}"
        )
        values = {value.value_id: value for value in module.graph.values}
        for op in module.graph.ops:
            if op.kind == BoundOpKind.RELU_RELAXATION and any(
                values[value_id].state_version != expected_state_version
                for value_id in op.outputs
            ):
                raise ValueError(
                    "native plain-CROWN ReLU state does not bind its external payload"
                )
        if len(module.graph.ops) <= 1:
            raise ValueError(
                "native plain-CROWN compilation requires multiple Bound ops"
            )
        if len(self.task_module.tasks) != len(module.graph.ops):
            raise ValueError(
                "native plain-CROWN requires one Task IR unit per Bound op"
            )
        if any(
            task.kind == TaskIRKind.EXTERNAL_VERIFIER_CALL
            for task in self.task_module.tasks
        ):
            raise ValueError("native plain-CROWN Task IR contains an external call")
        launches = tuple(
            action
            for action in self.schedule.actions
            if isinstance(action, LaunchAction)
        )
        emits = tuple(
            action
            for action in self.schedule.actions
            if isinstance(action, EmitResultAction)
        )
        if len(launches) != len(self.task_module.tasks) or len(launches) <= 1:
            raise ValueError(
                "native plain-CROWN schedule does not own every task launch"
            )
        if len(emits) != 1 or emits[0].query_ids != (self.query_id,):
            raise ValueError(
                "native plain-CROWN schedule must emit its exact query once"
            )
        if any(
            task.backend.reference_implementation_id != "bound_ir_region_reference/v1"
            for task in self.task_module.tasks
        ):
            raise ValueError(
                "native correctness slice selected a non-reference backend"
            )

    def hashes(self) -> dict[str, str]:
        """Return deterministic identities for all compiler layers."""

        self.validate()
        module = self.bound_module
        return {
            "bound_module_hash": module.stable_hash(),
            "plan_template_hash": self.template.stable_hash(bound_module=module),
            "plan_instance_hash": self.instance.stable_hash(
                template=self.template, bound_module=module
            ),
            "task_module_hash": self.task_module.stable_hash(
                bound_module=module,
                template=self.template,
                instance=self.instance,
            ),
            "schedule_hash": self.schedule.stable_hash(
                bound_module=module,
                template=self.template,
                instance=self.instance,
            ),
        }


def compile_native_plain_crown_query(  # pylint: disable=too-many-arguments
    legacy_task_module: BFTaskModule,
    input_spec: InputSpec,
    *,
    interval_env: Mapping[str, IntervalState],
    relu_pre: Mapping[str, IntervalState],
    linear_spec_C: torch.Tensor,
    intermediate_bounds_hash: str,
    query_id: str,
    available_memory_bytes: int = 1 << 40,
) -> NativePlainCrownIRCompilation:
    """Compile one externally calibrated query into native first-class IR."""

    return _compile_native_plain_crown_query(
        legacy_task_module,
        input_spec,
        interval_env=interval_env,
        relu_pre=relu_pre,
        linear_spec_C=linear_spec_C,
        intermediate_bounds_hash=intermediate_bounds_hash,
        query_id=query_id,
        available_memory_bytes=available_memory_bytes,
        memory_budget_bytes=available_memory_bytes,
        enable_memory_plans=False,
    )


def compile_native_plain_crown_memory_query(  # pylint: disable=too-many-arguments
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
) -> NativePlainCrownIRCompilation:
    """Compile one query with budget-selectable, runtime-enforced storage plans."""

    compilation = _compile_native_plain_crown_query(
        legacy_task_module,
        input_spec,
        interval_env=interval_env,
        relu_pre=relu_pre,
        linear_spec_C=linear_spec_C,
        intermediate_bounds_hash=intermediate_bounds_hash,
        query_id=query_id,
        available_memory_bytes=available_memory_bytes,
        memory_budget_bytes=memory_budget_bytes,
        enable_memory_plans=True,
    )
    if len(compilation.template.storage_candidates) != 2:
        raise ValueError("native memory compilation requires two storage plans")
    return compilation


# pylint: disable-next=too-many-arguments,too-many-locals
def _compile_native_plain_crown_query(
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
    enable_memory_plans: bool,
) -> NativePlainCrownIRCompilation:
    """Shared native compilation without changing the frozen v1 entry point."""

    if not query_id:
        raise ValueError("native plain-CROWN query ID must be non-empty")
    if available_memory_bytes <= 0 or memory_budget_bytes <= 0:
        raise ValueError("native plain-CROWN memory limits must be positive")
    build = build_plain_crown_bound_ir(
        legacy_task_module,
        input_spec,
        interval_env=interval_env,
        relu_pre=relu_pre,
        linear_spec_C=linear_spec_C,
        intermediate_bound_source=IntermediateBoundSource.EXTERNAL_VERIFIER,
        intermediate_bounds_hash=intermediate_bounds_hash,
        relu_lower_slope_policy=ReluLowerSlopePolicy.ADAPTIVE,
    )
    template = _build_native_reference_template(
        build.module,
        query_id=query_id,
        intermediate_bounds_hash=intermediate_bounds_hash,
        available_memory_bytes=available_memory_bytes,
    )
    if enable_memory_plans:
        template = build_native_storage_plan_variants(
            template, bound_module=build.module
        )
    instance = select_plan_instance(
        template,
        bound_module=build.module,
        query_bucket_id=f"native-plain-crown:{query_id}",
        available_memory_bytes=available_memory_bytes,
        memory_budget_bytes=memory_budget_bytes,
    )
    task_module = lower_plan_instance_to_task_ir(
        build.module, template=template, instance=instance
    )
    schedule = lower_plan_instance_to_reference_schedule(
        build.module,
        template=template,
        instance=instance,
        query_ids=(query_id,),
    )
    compilation = NativePlainCrownIRCompilation(
        query_id=query_id,
        intermediate_bounds_hash=intermediate_bounds_hash,
        build=build,
        template=template,
        instance=instance,
        task_module=task_module,
        schedule=schedule,
    )
    compilation.validate()
    return compilation


def execute_native_plain_crown_query(
    compilation: NativePlainCrownIRCompilation,
    *,
    legacy_task_module: BFTaskModule,
    input_spec: InputSpec,
    relu_pre: Mapping[str, IntervalState],
    linear_spec_C: torch.Tensor,
) -> tuple[IntervalState, TaskExecutionTrace]:
    """Execute every native region through the selected Schedule IR."""

    compilation.validate()
    return execute_task_ir_semantics(
        compilation.task_module,
        compilation.schedule,
        bound_module=compilation.bound_module,
        template=compilation.template,
        instance=compilation.instance,
        legacy_task_module=legacy_task_module,
        input_spec=input_spec,
        relu_pre=relu_pre,
        linear_spec_C=linear_spec_C,
    )


# pylint: disable-next=too-many-arguments
def execute_native_plain_crown_memory_query(
    compilation: NativePlainCrownIRCompilation,
    *,
    legacy_task_module: BFTaskModule,
    input_spec: InputSpec,
    relu_pre: Mapping[str, IntervalState],
    linear_spec_C: torch.Tensor,
    prepared: PreparedTaskIRExecution | None = None,
    prepared_storage: PreparedStoragePlanRuntime | None = None,
    trace_mode: TaskTraceMode = TaskTraceMode.AUDIT,
    backend: TypedTaskBackend | None = None,
) -> tuple[IntervalState, TaskExecutionTrace, StorageExecutionTrace]:
    """Execute the selected storage lifetime policy and return its exact trace."""

    compilation.validate()
    if len(compilation.template.storage_candidates) != 2:
        raise ValueError("native memory execution requires two storage candidates")
    storage_runtime = StoragePlanRuntime(
        bound_module=compilation.bound_module,
        template=compilation.template,
        instance=compilation.instance,
        schedule=compilation.schedule,
        prepared=prepared_storage,
    )
    result, task_trace = execute_task_ir_semantics(
        compilation.task_module,
        compilation.schedule,
        bound_module=compilation.bound_module,
        template=compilation.template,
        instance=compilation.instance,
        legacy_task_module=legacy_task_module,
        input_spec=input_spec,
        relu_pre=relu_pre,
        linear_spec_C=linear_spec_C,
        prepared=prepared,
        trace_mode=trace_mode,
        backend=backend,
        storage_runtime=storage_runtime,
    )
    return result, task_trace, storage_runtime.trace()


# pylint: disable-next=too-many-locals
def _build_native_reference_template(
    module: BFBoundModule,
    *,
    query_id: str,
    intermediate_bounds_hash: str,
    available_memory_bytes: int,
) -> PlanTemplate:
    values = module.graph.values
    dtypes = tuple(dict.fromkeys(value.tensor_type.dtype for value in values))
    devices = tuple(
        dict.fromkeys(
            value.tensor_type.device
            for value in values
            if value.tensor_type.device is not None
        )
    )
    if len(dtypes) != 1 or len(devices) != 1:
        raise ValueError("native plain-CROWN reference plan requires one dtype/device")
    dtype = dtypes[0]
    device = devices[0]
    objective = next(value for value in values if value.value_id == "query.objective")
    domain_batch = int(objective.tensor_type.shape[0] or 0)
    spec_batch = int(objective.tensor_type.shape[1] or 0)
    if domain_batch <= 0 or spec_batch <= 0:
        raise ValueError("native plain-CROWN query batches must be static and positive")

    cost = PlanCost(
        predicted_latency_ms=0.0,
        predicted_peak_bytes=0,
        compile_cost_ms=0.0,
        setup_cost_ms=0.0,
        confidence=1.0,
        risk_tags=("correctness_reference", "no_performance_claim"),
    )
    regions = tuple(
        RegionEvidence(
            evidence_id=f"native-op:{index:04d}",
            op_ids=(op.op_id,),
            cost=cost,
        )
        for index, op in enumerate(module.graph.ops)
    )
    representations = tuple(
        RepresentationEvidence(
            evidence_id=f"dense:{region.evidence_id}",
            region_evidence_id=region.evidence_id,
            representation=BoundRepresentation.DENSE,
            required_transition_evidence_ids=(),
            cost=cost,
        )
        for region in regions
    )
    capability = BackendCapabilitySpec(
        capability_id="native-plain-crown-reference-v1",
        backend=BackendKind.REFERENCE,
        supported_methods=(module.domain.method,),
        supported_op_kinds=tuple(dict.fromkeys(op.kind for op in module.graph.ops)),
        supported_representations=(BoundRepresentation.DENSE,),
        supported_dtypes=(dtype,),
        supported_devices=(device,),
        supports_grad=False,
        supports_alpha=False,
        supports_beta=False,
        supports_split_state=False,
        static_shape_only=True,
    )
    backends = tuple(
        BackendEvidence(
            evidence_id=f"reference:{region.evidence_id}",
            region_evidence_id=region.evidence_id,
            representation_evidence_id=f"dense:{region.evidence_id}",
            capability_id=capability.capability_id,
            cost=cost,
        )
        for region in regions
    )
    hardware = HardwareProfile(
        profile_id=f"native-plain-crown:{device}",
        device=device,
        total_memory_bytes=available_memory_bytes,
        supported_dtypes=(dtype,),
        backend_capability_ids=(capability.capability_id,),
        alignment_bytes=16,
    )
    workload = WorkloadProfile(
        profile_id=f"native-plain-crown:{query_id}",
        method=module.domain.method,
        requires_grad=False,
        alpha_enabled=False,
        beta_enabled=False,
        split_state_present=False,
        static_shapes=True,
        domain_batch_size=domain_batch,
        spec_batch_size=spec_batch,
        sample_batch_size=1,
        dtype=dtype,
        device=device,
        numeric_policy="external_intermediate_adaptive_float32_reference",
    )
    evidence = ReferencePlanEvidence(
        evidence_set_id=f"native-plain-crown:{query_id}",
        regions=regions,
        transitions=(),
        representations=representations,
        backends=backends,
        batches=(
            BatchEvidence(
                evidence_id="full-query",
                domain_batch_size=domain_batch,
                spec_batch_size=spec_batch,
                sample_batch_size=1,
                estimated_payload_bytes=0,
                cost=cost,
            ),
        ),
        storage=(
            StorageEvidence(
                evidence_id="dense-arena",
                compatible_batch_evidence_ids=("full-query",),
                compatible_representation_evidence_ids=tuple(
                    item.evidence_id for item in representations
                ),
                value_layout_overrides=(),
                arena_id=f"{device}-native-plain-crown",
                cost=cost,
            ),
        ),
        provenance=(
            PlanProvenance("compiler", NATIVE_PLAIN_CROWN_COMPILER_VERSION),
            PlanProvenance("semantics_owner", "boundflow_native_plain_crown"),
            PlanProvenance("intermediate_bound_source", "external_verifier"),
            PlanProvenance("intermediate_bounds_hash", intermediate_bounds_hash),
            PlanProvenance("relu_lower_slope_policy", "adaptive"),
            PlanProvenance("performance_claim", "forbidden"),
        ),
    )
    return build_reference_plan_template(
        module,
        hardware=hardware,
        workload=workload,
        capabilities=(capability,),
        evidence=evidence,
    )


__all__ = [
    "NATIVE_PLAIN_CROWN_COMPILER_VERSION",
    "NATIVE_PLAIN_CROWN_MEMORY_COMPILER_VERSION",
    "NativePlainCrownIRCompilation",
    "compile_native_plain_crown_memory_query",
    "compile_native_plain_crown_query",
    "execute_native_plain_crown_memory_query",
    "execute_native_plain_crown_query",
]
