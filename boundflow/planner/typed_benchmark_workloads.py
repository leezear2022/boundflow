"""Deterministic typed workloads used by IR-5 measured evaluation."""

# Workload construction deliberately assembles all compiler layers in one place.
# pylint: disable=too-many-arguments,too-many-locals

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import torch

from ..domains.interval import IntervalState
from ..frontends.plain_crown_bound_ir import build_plain_crown_bound_ir
from ..ir.bound import BFBoundModule, BoundRepresentation
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
from ..ir.schedule import ScheduleModule, lower_plan_instance_to_reference_schedule
from ..ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from ..ir.task_v1 import TaskIRModule, lower_plan_instance_to_task_ir
from ..runtime.crown_ibp import _forward_ibp_trace_mlp
from ..runtime.task_executor import InputSpec
from .plan_ir_builder import (
    BackendEvidence,
    BatchEvidence,
    ReferencePlanEvidence,
    RegionEvidence,
    RepresentationEvidence,
    StorageEvidence,
    build_reference_plan_template,
)
from .plan_ir_selector import select_plan_instance


@dataclass(frozen=True)
class PreparedTypedBenchmark:  # pylint: disable=too-many-instance-attributes
    """All immutable compiler inputs for one backend candidate measurement."""

    workload_id: str
    backend: BackendKind
    legacy_module: BFTaskModule
    input_spec: InputSpec
    relu_pre: Mapping[str, IntervalState]
    bound_module: BFBoundModule
    template: PlanTemplate
    instance: PlanInstance
    task_module: TaskIRModule
    schedule: ScheduleModule


def build_mlp_candidate(
    *,
    workload_id: str,
    backend: BackendKind,
    device: str,
    batch: int,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    seed: int,
) -> PreparedTypedBenchmark:
    """Build the same MLP semantics under one typed backend candidate."""

    if min(batch, input_dim, hidden_dim, output_dim) <= 0:
        raise ValueError("typed benchmark dimensions must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    params = {
        "W1": torch.randn(hidden_dim, input_dim, generator=generator),
        "b1": torch.randn(hidden_dim, generator=generator),
        "W2": torch.randn(output_dim, hidden_dim, generator=generator),
        "b2": torch.randn(output_dim, generator=generator),
    }
    center = torch.randn(batch, input_dim, generator=generator)
    if device != "cpu":
        params = {name: value.to(device) for name, value in params.items()}
        center = center.to(device)
    legacy = BFTaskModule(
        tasks=[
            BoundTask(
                task_id=f"ir5:{workload_id}",
                kind=TaskKind.INTERVAL_IBP,
                ops=[
                    TaskOp("linear", "linear1", ["input", "W1", "b1"], ["h1"]),
                    TaskOp("relu", "relu1", ["h1"], ["r1"]),
                    TaskOp("linear", "linear2", ["r1", "W2", "b2"], ["out"]),
                ],
                input_values=["input"],
                output_values=["out"],
            )
        ],
        entry_task_id=f"ir5:{workload_id}",
        bindings={"params": params},
    )
    input_spec = InputSpec.linf(
        value_name="input",
        center=center,
        eps=0.2,
    )
    return _prepare_candidate(
        workload_id=workload_id,
        backend=backend,
        device=device,
        legacy=legacy,
        input_spec=input_spec,
    )


def build_cnn_candidate(  # pylint: disable=too-many-arguments
    *,
    workload_id: str,
    backend: BackendKind,
    device: str,
    batch: int,
    input_channels: int,
    image_size: int,
    conv1_channels: int,
    conv2_channels: int,
    output_dim: int,
    seed: int,
    input_center: torch.Tensor | None = None,
) -> PreparedTypedBenchmark:
    """Build a deterministic two-convolution family under one typed backend."""

    dimensions = (
        batch,
        input_channels,
        image_size,
        conv1_channels,
        conv2_channels,
        output_dim,
    )
    if min(dimensions) <= 0 or image_size % 2:
        raise ValueError("typed CNN dimensions must be positive with even image size")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    half = image_size // 2
    params = {
        "W1": torch.randn(
            conv1_channels,
            input_channels,
            3,
            3,
            generator=generator,
        ),
        "b1": torch.randn(conv1_channels, generator=generator),
        "W2": torch.randn(
            conv2_channels,
            conv1_channels,
            3,
            3,
            generator=generator,
        ),
        "b2": torch.randn(conv2_channels, generator=generator),
        "W3": torch.randn(
            output_dim,
            conv2_channels * half * half,
            generator=generator,
        ),
        "b3": torch.randn(output_dim, generator=generator),
    }
    center = _input_center_or_random(
        (batch, input_channels, image_size, image_size),
        generator=generator,
        input_center=input_center,
        device=device,
    )
    if device != "cpu":
        params = {name: value.to(device) for name, value in params.items()}
    conv = {
        "stride": (1, 1),
        "padding": (1, 1),
        "dilation": (1, 1),
        "groups": 1,
    }
    downsample = {**conv, "stride": (2, 2)}
    legacy = BFTaskModule(
        tasks=[
            BoundTask(
                task_id=f"ir5:{workload_id}",
                kind=TaskKind.INTERVAL_IBP,
                ops=[
                    TaskOp(
                        "conv2d",
                        "conv1",
                        ["input", "W1", "b1"],
                        ["h1"],
                        conv,
                    ),
                    TaskOp("relu", "relu1", ["h1"], ["r1"]),
                    TaskOp(
                        "conv2d",
                        "conv2",
                        ["r1", "W2", "b2"],
                        ["h2"],
                        downsample,
                    ),
                    TaskOp("relu", "relu2", ["h2"], ["r2"]),
                    TaskOp(
                        "flatten",
                        "flatten",
                        ["r2"],
                        ["flat"],
                        {"start_dim": 1, "end_dim": -1},
                    ),
                    TaskOp("linear", "head", ["flat", "W3", "b3"], ["out"]),
                ],
                input_values=["input"],
                output_values=["out"],
            )
        ],
        entry_task_id=f"ir5:{workload_id}",
        bindings={"params": params},
    )
    input_spec = InputSpec.linf(
        value_name="input",
        center=center,
        eps=0.1,
    )
    return _prepare_candidate(
        workload_id=workload_id,
        backend=backend,
        device=device,
        legacy=legacy,
        input_spec=input_spec,
    )


def build_residual_cnn_candidate(  # pylint: disable=too-many-arguments
    *,
    workload_id: str,
    backend: BackendKind,
    device: str,
    batch: int,
    input_channels: int,
    image_size: int,
    block_channels: int,
    output_dim: int,
    seed: int,
    input_center: torch.Tensor | None = None,
) -> PreparedTypedBenchmark:
    """Build a deterministic residual-CNN family under one typed backend."""

    dimensions = (
        batch,
        input_channels,
        image_size,
        block_channels,
        output_dim,
    )
    if min(dimensions) <= 0:
        raise ValueError("typed residual-CNN dimensions must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    params = {
        "W1": torch.randn(
            block_channels,
            input_channels,
            3,
            3,
            generator=generator,
        ),
        "b1": torch.randn(block_channels, generator=generator),
        "W2": torch.randn(
            block_channels,
            block_channels,
            3,
            3,
            generator=generator,
        ),
        "b2": torch.randn(block_channels, generator=generator),
        "W3": torch.randn(
            output_dim,
            block_channels * image_size * image_size,
            generator=generator,
        ),
        "b3": torch.randn(output_dim, generator=generator),
    }
    center = _input_center_or_random(
        (batch, input_channels, image_size, image_size),
        generator=generator,
        input_center=input_center,
        device=device,
    )
    if device != "cpu":
        params = {name: value.to(device) for name, value in params.items()}
    conv = {
        "stride": (1, 1),
        "padding": (1, 1),
        "dilation": (1, 1),
        "groups": 1,
    }
    legacy = BFTaskModule(
        tasks=[
            BoundTask(
                task_id=f"ir5:{workload_id}",
                kind=TaskKind.INTERVAL_IBP,
                ops=[
                    TaskOp(
                        "conv2d",
                        "stem",
                        ["input", "W1", "b1"],
                        ["stem_pre"],
                        conv,
                    ),
                    TaskOp("relu", "stem_relu", ["stem_pre"], ["skip"]),
                    TaskOp(
                        "conv2d",
                        "residual_conv",
                        ["skip", "W2", "b2"],
                        ["residual"],
                        conv,
                    ),
                    TaskOp(
                        "add",
                        "residual_add",
                        ["skip", "residual"],
                        ["merged"],
                    ),
                    TaskOp("relu", "merge_relu", ["merged"], ["features"]),
                    TaskOp(
                        "flatten",
                        "flatten",
                        ["features"],
                        ["flat"],
                        {"start_dim": 1, "end_dim": -1},
                    ),
                    TaskOp("linear", "head", ["flat", "W3", "b3"], ["out"]),
                ],
                input_values=["input"],
                output_values=["out"],
            )
        ],
        entry_task_id=f"ir5:{workload_id}",
        bindings={"params": params},
    )
    input_spec = InputSpec.linf(
        value_name="input",
        center=center,
        eps=0.1,
    )
    return _prepare_candidate(
        workload_id=workload_id,
        backend=backend,
        device=device,
        legacy=legacy,
        input_spec=input_spec,
    )


def _prepare_candidate(
    *,
    workload_id: str,
    backend: BackendKind,
    device: str,
    legacy: BFTaskModule,
    input_spec: InputSpec,
) -> PreparedTypedBenchmark:
    """Lower one deterministic legacy workload through every compiler layer."""

    interval_env, relu_pre = _forward_ibp_trace_mlp(legacy, input_spec)
    bound_module = build_plain_crown_bound_ir(
        legacy,
        input_spec,
        interval_env=interval_env,
        relu_pre=relu_pre,
    ).module
    fused_pair = _relu_affine_pair(bound_module)
    template = _backend_template(
        bound_module,
        workload_id=workload_id,
        backend=backend,
        device=device,
        fused_pair=None if backend == BackendKind.REFERENCE else fused_pair,
    )
    instance = select_plan_instance(
        template,
        bound_module=bound_module,
        query_bucket_id=f"ir5:{workload_id}:{backend.value}",
        available_memory_bytes=1 << 40,
        memory_budget_bytes=1 << 40,
    )
    task_module = lower_plan_instance_to_task_ir(
        bound_module,
        template=template,
        instance=instance,
    )
    schedule = lower_plan_instance_to_reference_schedule(
        bound_module,
        template=template,
        instance=instance,
        query_ids=(f"query:{workload_id}",),
    )
    return PreparedTypedBenchmark(
        workload_id=workload_id,
        backend=backend,
        legacy_module=legacy,
        input_spec=input_spec,
        relu_pre=relu_pre,
        bound_module=bound_module,
        template=template,
        instance=instance,
        task_module=task_module,
        schedule=schedule,
    )


def _input_center_or_random(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    input_center: torch.Tensor | None,
    device: str,
) -> torch.Tensor:
    """Bind an exact input sample or generate the frozen workload input."""

    center = (
        torch.randn(*shape, generator=generator)
        if input_center is None
        else input_center.detach().clone()
    )
    if tuple(int(dim) for dim in center.shape) != shape:
        raise ValueError("typed benchmark input center shape mismatch")
    if center.dtype != torch.float32:
        raise ValueError("typed benchmark input center must use float32")
    return center.to(device)


def _cost(
    latency: float, *, compile_ms: float = 0.0, risk: str = "ir5_measured"
) -> PlanCost:
    return PlanCost(
        predicted_latency_ms=latency,
        predicted_peak_bytes=0,
        compile_cost_ms=compile_ms,
        setup_cost_ms=0.0,
        confidence=1.0,
        risk_tags=(risk,),
    )


def _relu_affine_pair(bound_module: BFBoundModule) -> tuple[str, str]:
    for first, second in zip(bound_module.graph.ops[:-1], bound_module.graph.ops[1:]):
        if first.kind.value == "relu_relaxation" and second.kind.value in {
            "linear_backward",
            "conv2d_backward",
        }:
            return first.op_id, second.op_id
    raise ValueError("typed benchmark workload has no ReLU→Affine pair")


def _backend_template(
    bound_module: BFBoundModule,
    *,
    workload_id: str,
    backend: BackendKind,
    device: str,
    fused_pair: tuple[str, str] | None,
) -> PlanTemplate:
    single_regions = tuple(
        RegionEvidence(f"single:{op.op_id}", (op.op_id,), _cost(1.0))
        for op in bound_module.graph.ops
    )
    fused_regions = (
        ()
        if fused_pair is None
        else (RegionEvidence("fused:relu-affine", fused_pair, _cost(0.1)),)
    )
    regions = (*single_regions, *fused_regions)
    representations = tuple(
        RepresentationEvidence(
            evidence_id=f"dense:{region.evidence_id}",
            region_evidence_id=region.evidence_id,
            representation=BoundRepresentation.DENSE,
            required_transition_evidence_ids=(),
            cost=_cost(0.0),
        )
        for region in regions
    )
    reference = BackendCapabilitySpec(
        capability_id=f"ir5-reference:{workload_id}",
        backend=BackendKind.REFERENCE,
        supported_methods=(bound_module.domain.method,),
        supported_op_kinds=tuple(
            dict.fromkeys(op.kind for op in bound_module.graph.ops)
        ),
        supported_representations=(BoundRepresentation.DENSE,),
        supported_dtypes=("float32",),
        supported_devices=(device,),
        supports_grad=False,
        supports_alpha=False,
        supports_beta=False,
        supports_split_state=False,
        static_shape_only=True,
    )
    selected = replace(
        reference,
        capability_id=f"ir5-{backend.value}:{workload_id}",
        backend=backend,
    )
    capabilities = (
        (reference,)
        if backend == BackendKind.REFERENCE
        else (
            reference,
            selected,
        )
    )
    backends: list[BackendEvidence] = []
    for region in single_regions:
        capability = selected if fused_pair is None else reference
        backends.append(
            BackendEvidence(
                evidence_id=f"{capability.backend.value}:{region.evidence_id}",
                region_evidence_id=region.evidence_id,
                representation_evidence_id=f"dense:{region.evidence_id}",
                capability_id=capability.capability_id,
                compiled_artifact_key=(
                    None
                    if capability.backend == BackendKind.REFERENCE
                    else f"ir5:{workload_id}:{backend.value}:{region.evidence_id}"
                ),
                cost=_cost(1.0),
            )
        )
    if fused_pair is not None:
        backends.append(
            BackendEvidence(
                evidence_id=f"{backend.value}:fused",
                region_evidence_id="fused:relu-affine",
                representation_evidence_id="dense:fused:relu-affine",
                capability_id=selected.capability_id,
                compiled_artifact_key=f"ir5:{workload_id}:{backend.value}:fused",
                cost=_cost(
                    0.1,
                    compile_ms=(
                        1.0
                        if backend
                        in {BackendKind.TVM_FUSED_TIR, BackendKind.TVM_TIR_UNFUSED}
                        else 0.0
                    ),
                ),
            )
        )
    objective_shape = bound_module.graph.values[0].tensor_type.shape
    domain_batch = int(objective_shape[0] or 1)
    spec_batch = int(objective_shape[1] or 1)
    hardware = HardwareProfile(
        profile_id=f"ir5:{workload_id}:{device}",
        device=device,
        total_memory_bytes=1 << 40,
        supported_dtypes=("float32",),
        backend_capability_ids=tuple(item.capability_id for item in capabilities),
        alignment_bytes=256 if device == "cuda" else 16,
    )
    workload = WorkloadProfile(
        profile_id=f"ir5:{workload_id}",
        method=bound_module.domain.method,
        requires_grad=False,
        alpha_enabled=False,
        beta_enabled=False,
        split_state_present=False,
        static_shapes=True,
        domain_batch_size=domain_batch,
        spec_batch_size=spec_batch,
        sample_batch_size=1,
        dtype="float32",
        device=device,
        numeric_policy="float32_ir5_measured",
    )
    evidence = ReferencePlanEvidence(
        evidence_set_id=f"ir5:{workload_id}:{backend.value}",
        regions=regions,
        transitions=(),
        representations=representations,
        backends=tuple(backends),
        batches=(
            BatchEvidence(
                "full",
                domain_batch,
                spec_batch,
                1,
                0,
                _cost(0.0),
            ),
        ),
        storage=(
            StorageEvidence(
                "all",
                ("full",),
                tuple(item.evidence_id for item in representations),
                (),
                f"{device}-main",
                _cost(0.0),
            ),
        ),
        provenance=(PlanProvenance("benchmark_scope", "ir5_measured"),),
    )
    return build_reference_plan_template(
        bound_module,
        hardware=hardware,
        workload=workload,
        capabilities=capabilities,
        evidence=evidence,
    )


__all__ = [
    "PreparedTypedBenchmark",
    "build_cnn_candidate",
    "build_mlp_candidate",
    "build_residual_cnn_candidate",
]
