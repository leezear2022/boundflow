"""Typed Bound/Plan/Task/Schedule admission for exact external verifier calls."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Callable, Mapping, Optional, Tuple

import torch

from ..ir.bound import (
    BOUND_IR_SCHEMA_VERSION,
    BFBoundGraph,
    BFBoundModule,
    BatchAxisKind,
    BoundBatchAxis,
    BoundDomainConfig,
    BoundMethodKind,
    BoundOp,
    BoundOpKind,
    BoundPolarity,
    BoundRepresentation,
    BoundTensorType,
    BoundValue,
    BoundValueRole,
    ExternalVerifierCallAttrs,
    ObjectiveKind,
    ObjectiveSpec,
    PerturbationKind,
    PerturbationSpec,
    VerificationSpec,
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
from ..ir.task_v1 import (
    TaskIRKind,
    TaskIRModule,
    lower_plan_instance_to_task_ir,
)
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


@dataclass(frozen=True)
class ExternalVerifierCallSpec:  # pylint: disable=too-many-instance-attributes
    """Serializable identity needed to compile one provider-owned call."""

    query_id: str
    sequence_number: int
    parent_query_id: Optional[str]
    model_structure_hash: str
    weight_version: str
    input_region_hash: str
    objective_hash: str
    solver_phase: str
    observed_method: str
    effective_method: BoundMethodKind
    requires_grad: bool
    alpha_state_version: Optional[str]
    beta_state_version: Optional[str]
    split_state_version: Optional[str]
    cuts_version: Optional[str]
    input_shape: Tuple[int, ...]
    spec_shape: Tuple[int, ...]
    dtype: str
    device: str
    numeric_policy: str
    requested_bounds: Tuple[BoundPolarity, ...] = (BoundPolarity.LOWER,)

    @classmethod
    def from_query_dict(cls, query: Mapping[str, Any]) -> "ExternalVerifierCallSpec":
        """Upgrade one PR-14 JSONL identity into an explicit external-call spec."""

        compatibility = _mapping(query.get("compatibility_key"), "compatibility_key")
        options = _mapping(query.get("execution_options"), "execution_options")
        observed_method = str(query.get("bound_method", ""))
        split_signature = str(query.get("split_signature", ""))
        split_present = bool(options.get("split_state_present", False))
        effective_method = _effective_method(
            observed_method, split_present=split_present
        )
        alpha_version = _optional_text(query.get("alpha_state_version"))
        beta_version = _optional_text(query.get("beta_state_version"))
        split_version = None
        if split_present:
            split_version = split_signature or "external-live:split-state-unresolved"
            if alpha_version is None:
                alpha_version = "external-live:alpha-state-unresolved"
            if beta_version is None:
                beta_version = f"external-live:beta-state:{split_version}"
        elif effective_method == BoundMethodKind.CROWN:
            alpha_version = None
            beta_version = None
        requested = _requested_bound_polarities(query, options)
        spec = cls(
            query_id=str(query.get("query_id", "")),
            sequence_number=int(query.get("sequence_number", -1)),
            parent_query_id=_optional_text(query.get("parent_query_id")),
            model_structure_hash=str(query.get("model_structure_hash", "")),
            weight_version=str(query.get("weight_version", "")),
            input_region_hash=str(query.get("input_region_hash", "")),
            objective_hash=str(query.get("output_spec_hash", "")),
            solver_phase=str(options.get("solver_phase", "")),
            observed_method=observed_method,
            effective_method=effective_method,
            requires_grad=bool(query.get("requires_grad", False)),
            alpha_state_version=alpha_version,
            beta_state_version=beta_version,
            split_state_version=split_version,
            cuts_version=_optional_text(query.get("cuts_version")),
            input_shape=_positive_shape(
                compatibility.get("input_shape"), "input_shape"
            ),
            spec_shape=_shape(compatibility.get("spec_shape"), "spec_shape"),
            dtype=str(query.get("dtype", compatibility.get("dtype", ""))).removeprefix(
                "torch."
            ),
            device=str(query.get("device", compatibility.get("device", ""))),
            numeric_policy=str(query.get("numeric_policy", "")),
            requested_bounds=requested,
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        """Validate identity, state ownership, shapes, and method/state consistency."""

        for name in (
            "query_id",
            "model_structure_hash",
            "weight_version",
            "input_region_hash",
            "objective_hash",
            "solver_phase",
            "observed_method",
            "dtype",
            "device",
            "numeric_policy",
        ):
            if not getattr(self, name):
                raise ValueError(f"external verifier spec {name} must be non-empty")
        if self.sequence_number < 0:
            raise ValueError("external verifier sequence number must be non-negative")
        if self.parent_query_id is not None and not self.parent_query_id:
            raise ValueError("external verifier parent query ID is empty")
        if not self.input_shape or any(dim <= 0 for dim in self.input_shape):
            raise ValueError(
                "external verifier input shape must be statically positive"
            )
        if any(dim <= 0 for dim in self.spec_shape):
            raise ValueError("external verifier spec shape must be statically positive")
        if self.effective_method == BoundMethodKind.ALPHA_CROWN:
            if self.alpha_state_version is None or self.beta_state_version is not None:
                raise ValueError("external alpha-CROWN state identity is incomplete")
        if self.effective_method == BoundMethodKind.ALPHA_BETA_CROWN:
            if (
                self.alpha_state_version is None
                or self.beta_state_version is None
                or self.split_state_version is None
            ):
                raise ValueError(
                    "external activation-BaB requires alpha/beta/split identity"
                )
        VerificationSpec(
            perturbations=(
                PerturbationSpec(
                    "query.input",
                    "external.input",
                    PerturbationKind.BOX,
                    payload_hash=self.input_region_hash,
                ),
            ),
            objectives=(
                ObjectiveSpec(
                    "query.objective",
                    "external.output",
                    ObjectiveKind.LINEAR,
                    num_objectives=self.spec_count,
                    payload_hash=self.objective_hash,
                ),
            ),
            requested_bounds=self.requested_bounds,
            numeric_policy=self.numeric_policy,
        ).validate()

    @property
    def domain_count(self) -> int:
        return self.input_shape[0]

    @property
    def spec_count(self) -> int:
        if len(self.spec_shape) >= 2:
            return self.spec_shape[1]
        return 1

    @property
    def split_present(self) -> bool:
        return self.split_state_version is not None


@dataclass(frozen=True)
class ExternalVerifierIRCompilation:
    """Fully validated Bound→Plan→Task→Schedule compilation result."""

    call_spec: ExternalVerifierCallSpec
    bound_module: BFBoundModule
    template: PlanTemplate
    instance: PlanInstance
    task_module: TaskIRModule
    schedule: ScheduleModule

    def validate(self) -> None:
        self.call_spec.validate()
        self.task_module.validate_schedule_linkage(
            self.schedule,
            bound_module=self.bound_module,
            template=self.template,
            instance=self.instance,
        )
        if len(self.task_module.tasks) != 1:
            raise ValueError("external verifier compilation requires exactly one task")
        task = self.task_module.tasks[0]
        if task.kind != TaskIRKind.EXTERNAL_VERIFIER_CALL:
            raise ValueError("external verifier compilation has the wrong Task IR kind")
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
        if len(launches) != 1 or len(emits) != 1:
            raise ValueError(
                "external verifier schedule requires one launch and one emit"
            )
        if launches[0].backend_candidate_id != task.backend.backend_candidate_id:
            raise ValueError("external verifier schedule/backend identity differs")

    def hashes(self) -> dict[str, str]:
        """Return the exact cross-IR identities used by artifacts."""

        self.validate()
        return {
            "bound_module_hash": self.bound_module.stable_hash(),
            "plan_template_hash": self.template.stable_hash(
                bound_module=self.bound_module
            ),
            "plan_instance_hash": self.instance.stable_hash(
                template=self.template, bound_module=self.bound_module
            ),
            "task_module_hash": self.task_module.stable_hash(
                bound_module=self.bound_module,
                template=self.template,
                instance=self.instance,
            ),
            "schedule_hash": self.schedule.stable_hash(
                bound_module=self.bound_module,
                template=self.template,
                instance=self.instance,
            ),
        }


@dataclass(frozen=True)
class ExternalVerifierExecution:
    """One exact provider call plus typed schedule evidence."""

    query_id: str
    sequence_number: int
    result: Any
    result_hash: str
    ir_hashes: Mapping[str, str]


def compile_external_verifier_call(
    call_spec: ExternalVerifierCallSpec,
) -> ExternalVerifierIRCompilation:
    """Compile one provider-owned query through all four first-class IR layers."""

    call_spec.validate()
    bound_module = _build_bound_module(call_spec)
    template = _build_plan_template(bound_module, call_spec)
    memory = template.hardware.total_memory_bytes
    instance = select_plan_instance(
        template,
        bound_module=bound_module,
        query_bucket_id=f"external:{call_spec.query_id}",
        available_memory_bytes=memory,
        memory_budget_bytes=memory,
    )
    task_module = lower_plan_instance_to_task_ir(
        bound_module, template=template, instance=instance
    )
    schedule = lower_plan_instance_to_reference_schedule(
        bound_module,
        template=template,
        instance=instance,
        query_ids=(call_spec.query_id,),
    )
    compiled = ExternalVerifierIRCompilation(
        call_spec=call_spec,
        bound_module=bound_module,
        template=template,
        instance=instance,
        task_module=task_module,
        schedule=schedule,
    )
    compiled.validate()
    return compiled


def execute_external_verifier_call(
    compilation: ExternalVerifierIRCompilation,
    exact_call: Callable[[], Any],
) -> ExternalVerifierExecution:
    """Execute exactly once through the external backend declared by the schedule."""

    compilation.validate()
    task = compilation.task_module.tasks[0]
    if task.backend.reference_implementation_id != "external_abcrown_exact_call/v1":
        raise ValueError("external verifier execution refuses an undeclared backend")
    result = exact_call()
    return ExternalVerifierExecution(
        query_id=compilation.call_spec.query_id,
        sequence_number=compilation.call_spec.sequence_number,
        result=result,
        result_hash=_result_hash(result),
        ir_hashes=compilation.hashes(),
    )


def _build_bound_module(spec: ExternalVerifierCallSpec) -> BFBoundModule:
    dtype = spec.dtype
    device = spec.device
    domain_axis = (BoundBatchAxis(BatchAxisKind.DOMAIN, 0),)
    input_type = BoundTensorType(
        shape=spec.input_shape,
        dtype=dtype,
        layout="contiguous",
        device=device,
        batch_axes=domain_axis,
    )
    objective_shape = spec.spec_shape or (spec.domain_count, spec.spec_count, 1)
    objective_axes: Tuple[BoundBatchAxis, ...] = (
        BoundBatchAxis(BatchAxisKind.DOMAIN, 0),
    )
    if len(objective_shape) >= 2:
        objective_axes += (BoundBatchAxis(BatchAxisKind.SPEC, 1),)
    objective_type = BoundTensorType(
        shape=objective_shape,
        dtype=dtype,
        layout="contiguous",
        device=device,
        batch_axes=objective_axes,
    )
    output_type = BoundTensorType(
        shape=(spec.domain_count, spec.spec_count),
        dtype=dtype,
        layout="contiguous",
        device=device,
        batch_axes=(
            BoundBatchAxis(BatchAxisKind.DOMAIN, 0),
            BoundBatchAxis(BatchAxisKind.SPEC, 1),
        ),
    )
    token_type = BoundTensorType(
        shape=(1,), dtype="uint8", layout="opaque", device=device
    )
    values = [
        BoundValue(
            "query.input",
            input_type,
            BoundValueRole.PERTURBATION,
            BoundPolarity.BOTH,
            BoundRepresentation.DENSE,
        ),
        BoundValue(
            "query.objective",
            objective_type,
            BoundValueRole.OBJECTIVE,
            BoundPolarity.BOTH,
            BoundRepresentation.DENSE,
        ),
    ]
    state_inputs: list[str] = []
    for value_id, role, version in (
        ("query.alpha", BoundValueRole.RELAXATION, spec.alpha_state_version),
        ("query.beta", BoundValueRole.RELAXATION, spec.beta_state_version),
        ("query.split", BoundValueRole.SPLIT, spec.split_state_version),
    ):
        if version is not None:
            values.append(
                BoundValue(
                    value_id,
                    token_type,
                    role,
                    BoundPolarity.BOTH,
                    BoundRepresentation.SCALAR,
                    state_version=version,
                )
            )
            state_inputs.append(value_id)
    output_polarities = (
        (BoundPolarity.LOWER, BoundPolarity.UPPER)
        if spec.requested_bounds == (BoundPolarity.BOTH,)
        else spec.requested_bounds
    )
    output_ids: list[str] = []
    for polarity in output_polarities:
        value_id = f"result.{polarity.value}"
        output_ids.append(value_id)
        values.append(
            BoundValue(
                value_id,
                output_type,
                BoundValueRole.OBJECTIVE,
                polarity,
                BoundRepresentation.DENSE,
            )
        )
    op = BoundOp(
        op_id="external-verifier-call",
        kind=BoundOpKind.EXTERNAL_VERIFIER_CALL,
        inputs=("query.input", "query.objective", *state_inputs),
        outputs=tuple(output_ids),
        attrs=ExternalVerifierCallAttrs(
            provider="alpha-beta-CROWN",
            solver_phase=spec.solver_phase,
            method=spec.effective_method,
            requested_bounds=spec.requested_bounds,
            input_region_hash=spec.input_region_hash,
            objective_hash=spec.objective_hash,
            alpha_state_version=spec.alpha_state_version,
            beta_state_version=spec.beta_state_version,
            split_state_version=spec.split_state_version,
            cuts_version=spec.cuts_version,
        ),
    )
    module = BFBoundModule(
        module_id=f"external-verifier:{spec.query_id}",
        primal_graph_hash=_identity_hash(
            spec.model_structure_hash, spec.weight_version
        ),
        spec=VerificationSpec(
            perturbations=(
                PerturbationSpec(
                    "query.input",
                    "external.input",
                    PerturbationKind.BOX,
                    payload_hash=spec.input_region_hash,
                ),
            ),
            objectives=(
                ObjectiveSpec(
                    "query.objective",
                    "external.output",
                    ObjectiveKind.LINEAR,
                    num_objectives=spec.spec_count,
                    payload_hash=spec.objective_hash,
                ),
            ),
            requested_bounds=spec.requested_bounds,
            numeric_policy=spec.numeric_policy,
        ),
        domain=BoundDomainConfig(
            method=spec.effective_method,
            requires_grad=spec.requires_grad,
            alpha_enabled=spec.alpha_state_version is not None,
            beta_enabled=spec.beta_state_version is not None,
            split_state_present=spec.split_present,
        ),
        graph=BFBoundGraph(
            values=tuple(values),
            ops=(op,),
            inputs=("query.input", "query.objective", *state_inputs),
            outputs=tuple(output_ids),
        ),
        schema_version=BOUND_IR_SCHEMA_VERSION,
    )
    module.validate()
    return module


def _build_plan_template(
    module: BFBoundModule, spec: ExternalVerifierCallSpec
) -> PlanTemplate:
    capability = BackendCapabilitySpec(
        capability_id="external-abcrown-exact-v1",
        backend=BackendKind.EXTERNAL_ABCROWN,
        supported_methods=(spec.effective_method,),
        supported_op_kinds=(BoundOpKind.EXTERNAL_VERIFIER_CALL,),
        supported_representations=(BoundRepresentation.DENSE,),
        supported_dtypes=(spec.dtype,),
        supported_devices=(spec.device,),
        supports_grad=True,
        supports_alpha=True,
        supports_beta=True,
        supports_split_state=True,
        static_shape_only=True,
    )
    tensor_bytes = sum(
        _tensor_bytes(value.tensor_type.shape, value.tensor_type.dtype)
        for value in module.graph.values
    )
    total_memory = max(1 << 20, tensor_bytes * 2)
    hardware = HardwareProfile(
        profile_id=f"external-abcrown:{spec.device}",
        device=spec.device,
        total_memory_bytes=total_memory,
        supported_dtypes=(spec.dtype,),
        backend_capability_ids=(capability.capability_id,),
        alignment_bytes=16,
    )
    workload = WorkloadProfile(
        profile_id=f"external-query:{spec.query_id}",
        method=spec.effective_method,
        requires_grad=spec.requires_grad,
        alpha_enabled=spec.alpha_state_version is not None,
        beta_enabled=spec.beta_state_version is not None,
        split_state_present=spec.split_present,
        static_shapes=True,
        domain_batch_size=spec.domain_count,
        spec_batch_size=spec.spec_count,
        sample_batch_size=1,
        dtype=spec.dtype,
        device=spec.device,
        numeric_policy=spec.numeric_policy,
    )
    cost = PlanCost(
        0.0, 0, 0.0, 0.0, 1.0, ("external_semantics_owner", "no_performance_claim")
    )
    evidence = ReferencePlanEvidence(
        evidence_set_id=f"external-call:{spec.query_id}",
        regions=(RegionEvidence("external-call", ("external-verifier-call",), cost),),
        transitions=(),
        representations=(
            RepresentationEvidence(
                "external-dense", "external-call", BoundRepresentation.DENSE, (), cost
            ),
        ),
        backends=(
            BackendEvidence(
                "external-abcrown",
                "external-call",
                "external-dense",
                capability.capability_id,
                cost,
            ),
        ),
        batches=(
            BatchEvidence(
                "exact-query",
                spec.domain_count,
                spec.spec_count,
                1,
                max(1, tensor_bytes),
                cost,
            ),
        ),
        storage=(
            StorageEvidence(
                "external-storage",
                ("exact-query",),
                ("external-dense",),
                (),
                "external-provider",
                cost,
            ),
        ),
        provenance=(
            PlanProvenance("semantics_owner", "external_verifier"),
            PlanProvenance("performance_claim", "forbidden"),
            PlanProvenance("solver_phase", spec.solver_phase),
        ),
    )
    return build_reference_plan_template(
        module,
        hardware=hardware,
        workload=workload,
        capabilities=(capability,),
        evidence=evidence,
    )


def _effective_method(observed: str, *, split_present: bool) -> BoundMethodKind:
    normalized = observed.lower().replace("_", "-")
    if split_present or "beta" in normalized:
        return BoundMethodKind.ALPHA_BETA_CROWN
    if "alpha" in normalized:
        return BoundMethodKind.ALPHA_CROWN
    if "crown" in normalized:
        return BoundMethodKind.CROWN
    if "ibp" in normalized or "interval" in normalized:
        return BoundMethodKind.INTERVAL
    raise ValueError(f"unsupported external bound method: {observed}")


def _requested_bound_polarities(
    query: Mapping[str, Any], options: Mapping[str, Any]
) -> Tuple[BoundPolarity, ...]:
    if "bound_lower_requested" in options or "bound_upper_requested" in options:
        lower = bool(options.get("bound_lower_requested", True))
        upper = bool(options.get("bound_upper_requested", True))
        if lower and upper:
            return (BoundPolarity.BOTH,)
        if lower:
            return (BoundPolarity.LOWER,)
        if upper:
            return (BoundPolarity.UPPER,)
        raise ValueError("external verifier query requests neither lower nor upper")
    raw = query.get("requested_outputs")
    if not isinstance(raw, (tuple, list)) or not raw:
        return (BoundPolarity.BOTH,)
    values = {str(item).lower() for item in raw}
    if values <= {"bounds", "lower", "upper"}:
        if "bounds" in values or {"lower", "upper"} <= values:
            return (BoundPolarity.BOTH,)
        if "upper" in values and "lower" not in values:
            return (BoundPolarity.UPPER,)
        return (BoundPolarity.LOWER,)
    raise ValueError(f"unsupported external requested outputs: {sorted(values)}")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"external verifier {label} must be a mapping")
    return value


def _optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    return text or None


def _shape(value: Any, label: str) -> Tuple[int, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"external verifier {label} must be a sequence")
    return tuple(int(dim) for dim in value)


def _positive_shape(value: Any, label: str) -> Tuple[int, ...]:
    shape = _shape(value, label)
    if not shape or any(dim <= 0 for dim in shape):
        raise ValueError(f"external verifier {label} must be statically positive")
    return shape


def _tensor_bytes(shape: Tuple[Optional[int], ...], dtype: str) -> int:
    size = {
        "uint8": 1,
        "int8": 1,
        "float16": 2,
        "bfloat16": 2,
        "float32": 4,
        "float64": 8,
    }.get(dtype)
    if size is None or any(dim is None for dim in shape):
        raise ValueError(f"unsupported external tensor type: {dtype}{shape}")
    for dim in shape:
        assert dim is not None
        size *= dim
    return size


def _identity_hash(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _result_hash(result: Any) -> str:
    digest = hashlib.sha256()

    def update(value: Any) -> None:
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("utf-8"))
            digest.update(str(tuple(tensor.shape)).encode("utf-8"))
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value, (tuple, list)):
            digest.update(type(value).__name__.encode("utf-8"))
            for item in value:
                update(item)
        elif isinstance(value, Mapping):
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
                digest.update(str(key).encode("utf-8"))
                update(item)
        else:
            digest.update(
                json.dumps(value, sort_keys=True, default=str).encode("utf-8")
            )

    update(result)
    return digest.hexdigest()


__all__ = [
    "ExternalVerifierCallSpec",
    "ExternalVerifierExecution",
    "ExternalVerifierIRCompilation",
    "compile_external_verifier_call",
    "execute_external_verifier_call",
]
