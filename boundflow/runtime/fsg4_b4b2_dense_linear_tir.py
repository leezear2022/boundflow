"""B4-B2 B2-1 dense S-anchor Linear TIR correctness runtime."""

# mypy: disable-error-code=import-untyped
# pylint: disable=import-error,import-outside-toplevel,missing-function-docstring
# pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-locals
# pylint: disable=too-many-statements,too-many-boolean-expressions
# pylint: disable=abstract-method,arguments-differ,too-few-public-methods
# pylint: disable=too-many-positional-arguments,duplicate-code

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, cast

import torch

from boundflow.backends.tvm.differentiable_lower_dense_linear import (
    CompiledDifferentiableLowerDenseLinearTIR,
    compile_dense_linear_tir,
)
from boundflow.ir.differentiable_lower_dense_linear_tir import (
    DENSE_LINEAR_INPUT_NAMES,
    DENSE_LINEAR_OUTPUT_NAMES,
    DifferentiableLowerDenseLinearTIRInstanceV1,
    DifferentiableLowerDenseLinearTIRLaunchReceiptV1,
    DifferentiableLowerDenseLinearTIRModuleReceiptV1,
    DifferentiableLowerDenseLinearTIRScheduleV1,
    DifferentiableLowerDenseLinearTIRTemplateV1,
)
from boundflow.ir.differentiable_lower_region import (
    DifferentiableLowerRegionIRV1,
    DifferentiableLowerRegionInstanceV1,
)
from boundflow.ir.differentiable_lower_tir import canonical_tir_hash

from .fsg4_b4b1_pytorch_reference import (
    B4B1_REFERENCE_ATOL,
    B4B1_REFERENCE_RTOL,
    DifferentiableLowerReferenceResultV1,
    build_b4b1_differentiable_lower_instance_v1,
    build_b4b1_differentiable_lower_ir_v1,
    run_b4b1_pytorch_reference_v1,
)
from .fsg4_b4b1_reference_capture import ProductionDifferentiableReferenceCaptureV1
from .rvir_v4_production_state import production_tensor_sha256


@dataclass(frozen=True)
class DenseLinearTIRTensorsV1:
    """Complete dense ABI values and output adjoints for one S-anchor run."""

    incoming_lower_a: torch.Tensor
    preactivation_lower: torch.Tensor
    preactivation_upper: torch.Tensor
    native_alpha: torch.Tensor
    native_beta: torch.Tensor
    dense_split_sign: torch.Tensor
    incoming_lower_bias: torch.Tensor
    operator_weight: torch.Tensor
    operator_bias: torch.Tensor
    output_lower_a_gradient: torch.Tensor
    output_bias_gradient: torch.Tensor

    @property
    def tensor_map(self) -> dict[str, torch.Tensor]:
        return {
            name: cast(torch.Tensor, getattr(self, name))
            for name in DENSE_LINEAR_INPUT_NAMES
        }


@dataclass(frozen=True)
class DenseLinearTIRParityMetricV1:
    """One direct B4-B1 reference versus dense TIR tensor comparison."""

    name: str
    element_count: int
    maximum_absolute_difference: float
    allclose: bool
    sign_exact: bool
    reference_hash: str
    candidate_hash: str

    def validate(self) -> None:
        if (
            self.name not in DENSE_LINEAR_OUTPUT_NAMES
            or self.element_count < 1
            or not math.isfinite(self.maximum_absolute_difference)
            or self.maximum_absolute_difference < 0.0
            or len(self.reference_hash) != 64
            or len(self.candidate_hash) != 64
        ):
            raise ValueError("dense Linear TIR parity metric differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "name": self.name,
            "element_count": self.element_count,
            "maximum_absolute_difference": self.maximum_absolute_difference,
            "allclose": self.allclose,
            "sign_exact": self.sign_exact,
            "reference_hash": self.reference_hash,
            "candidate_hash": self.candidate_hash,
        }


@dataclass(frozen=True)
class DenseLinearTIRCandidateResultV1:
    """Candidate outputs, gradients and complete compiler/execution evidence."""

    output_lower_a: torch.Tensor
    output_bias: torch.Tensor
    native_alpha_gradient: torch.Tensor
    native_beta_gradient: torch.Tensor
    metrics: tuple[DenseLinearTIRParityMetricV1, ...]
    module_receipt: DifferentiableLowerDenseLinearTIRModuleReceiptV1
    launch_receipt: DifferentiableLowerDenseLinearTIRLaunchReceiptV1


def build_b4b2_dense_linear_template_v1(
    lower_ir: DifferentiableLowerRegionIRV1,
    *,
    compute_capability: str,
) -> DifferentiableLowerDenseLinearTIRTemplateV1:
    """Bind the dense ABI to the exact approved S-anchor static IR."""

    lower_ir.validate()
    if (
        lower_ir.anchor_id != "semantic-active-beta-gemm-14"
        or lower_ir.operator_kind != "linear"
        or lower_ir.coefficient_shape != (6, 1, 100)
        or lower_ir.result_coefficient_shape != (6, 1, 1024)
        or lower_ir.beta_active is not True
        or "amendment/operator_bias" not in lower_ir.tensor_contract_map
    ):
        raise ValueError("dense Linear TIR S-anchor differs")
    mapping_contracts = [
        contract.to_dict()
        for contract in lower_ir.tensor_contracts
        if contract.name.startswith("mapping/")
    ]
    template = DifferentiableLowerDenseLinearTIRTemplateV1(
        lower_region_ir_hash=lower_ir.stable_hash(),
        mapping_layout_hash=canonical_tir_hash(mapping_contracts),
        operator_attributes_hash=canonical_tir_hash(dict(lower_ir.operator_attributes)),
        compute_capability=compute_capability,
    )
    template.validate()
    return template


def build_b4b2_dense_linear_schedule_v1(
    template: DifferentiableLowerDenseLinearTIRTemplateV1,
) -> DifferentiableLowerDenseLinearTIRScheduleV1:
    """Return the only B2-1 correctness schedule."""

    schedule = DifferentiableLowerDenseLinearTIRScheduleV1(
        template_hash=template.stable_hash()
    )
    schedule.validate_against(template)
    return schedule


def _mapping_by_suffix(
    capture: ProductionDifferentiableReferenceCaptureV1, suffix: str
) -> torch.Tensor:
    matches = [
        snapshot.value
        for name, snapshot in capture.mapping_tensors
        if name.endswith(suffix)
    ]
    if len(matches) != 1:
        raise ValueError(f"dense Linear TIR mapping differs: {suffix}")
    return matches[0]


def _dense_split_sign(
    capture: ProductionDifferentiableReferenceCaptureV1,
    template: DifferentiableLowerDenseLinearTIRTemplateV1,
) -> torch.Tensor:
    locations = _mapping_by_suffix(capture, "/location")
    signs = _mapping_by_suffix(capture, "/sign")
    if (
        locations.dtype != torch.int64
        or locations.shape != (template.domain_count, 1)
        or signs.dtype != torch.float32
        or signs.shape != locations.shape
        or bool(((signs != -1) & (signs != 1)).any().item())
    ):
        raise ValueError("dense Linear TIR beta mapping differs")
    dense = torch.zeros(
        (template.domain_count, template.current_features), dtype=torch.float32
    )
    for domain in range(template.domain_count):
        location = int(locations[domain, 0].item())
        if location not in range(template.current_features):
            raise ValueError("dense Linear TIR beta location differs")
        dense[domain, location] = signs[domain, 0]
    return dense


def build_b4b2_dense_linear_tensors_v1(
    capture: ProductionDifferentiableReferenceCaptureV1,
    template: DifferentiableLowerDenseLinearTIRTemplateV1,
    *,
    device: torch.device,
) -> DenseLinearTIRTensorsV1:
    """Materialize the explicitly non-timed dense ABI from frozen raw values."""

    capture.validate()
    template.validate()
    values = capture.base.value_map

    def cuda_clone(name: str, *, requires_grad: bool = False) -> torch.Tensor:
        tensor = values[name].value.detach().to(device).contiguous().clone()
        tensor.requires_grad_(requires_grad)
        return tensor

    if capture.operator_bias is None:
        raise ValueError("dense Linear TIR operator bias is absent")
    tensors = DenseLinearTIRTensorsV1(
        incoming_lower_a=cuda_clone("incoming_lower_a"),
        preactivation_lower=cuda_clone("preactivation_lower"),
        preactivation_upper=cuda_clone("preactivation_upper"),
        native_alpha=cuda_clone("native_alpha", requires_grad=True),
        native_beta=cuda_clone("native_beta", requires_grad=True),
        dense_split_sign=_dense_split_sign(capture, template).to(device),
        incoming_lower_bias=capture.incoming_lower_bias.value.to(device).contiguous(),
        operator_weight=cuda_clone("operator_weight"),
        operator_bias=capture.operator_bias.value.to(device).contiguous(),
        output_lower_a_gradient=(
            capture.output_lower_a_gradient.value.to(device).contiguous()
        ),
        output_bias_gradient=capture.output_bias_gradient.value.to(device).contiguous(),
    )
    _validate_dense_linear_tensors(tensors, template)
    return tensors


def _validate_dense_linear_tensors(
    tensors: DenseLinearTIRTensorsV1,
    template: DifferentiableLowerDenseLinearTIRTemplateV1,
) -> None:
    expected_shapes = {
        "incoming_lower_a": (6, 1, 100),
        "preactivation_lower": (6, 100),
        "preactivation_upper": (6, 100),
        "native_alpha": (6, 100),
        "native_beta": (6, 100),
        "dense_split_sign": (6, 100),
        "incoming_lower_bias": (6, 1),
        "operator_weight": (100, 1024),
        "operator_bias": (100,),
        "output_lower_a_gradient": (6, 1, 1024),
        "output_bias_gradient": (6, 1),
    }
    values = tensors.tensor_map
    if tuple(sorted(values)) != DENSE_LINEAR_INPUT_NAMES:
        raise ValueError("dense Linear TIR tensor inventory differs")
    for name, tensor in values.items():
        expected_requires_grad = name in {"native_alpha", "native_beta"}
        if (
            tuple(tensor.shape) != expected_shapes[name]
            or tensor.device.type != "cuda"
            or tensor.dtype != torch.float32
            or not tensor.is_contiguous()
            or tensor.requires_grad is not expected_requires_grad
            or not bool(torch.isfinite(tensor).all().item())
        ):
            raise ValueError(f"dense Linear TIR tensor differs: {name}")
    if bool((values["preactivation_lower"] > values["preactivation_upper"]).any()):
        raise ValueError("dense Linear TIR interval differs")
    template.validate()


def build_b4b2_dense_linear_instance_v1(
    template: DifferentiableLowerDenseLinearTIRTemplateV1,
    lower_ir: DifferentiableLowerRegionIRV1,
    lower_instance: DifferentiableLowerRegionInstanceV1,
    capture: ProductionDifferentiableReferenceCaptureV1,
    tensors: DenseLinearTIRTensorsV1,
    *,
    fresh_run_ordinal: int,
) -> DifferentiableLowerDenseLinearTIRInstanceV1:
    """Hash-bind all dense values and adjoints without polluting the compile key."""

    lower_instance.validate_against(lower_ir)
    _validate_dense_linear_tensors(tensors, template)
    ordinal = tensors.incoming_lower_a.device.index
    if ordinal is None:
        ordinal = torch.cuda.current_device()
    instance = DifferentiableLowerDenseLinearTIRInstanceV1(
        template_hash=template.stable_hash(),
        lower_region_instance_hash=lower_instance.stable_hash(lower_ir),
        reference_capture_hash=cast(str, capture.metadata()["reference_capture_hash"]),
        tensor_hashes=tuple(
            (name, production_tensor_sha256(tensor))
            for name, tensor in sorted(tensors.tensor_map.items())
        ),
        fresh_run_ordinal=fresh_run_ordinal,
        device_ordinal=int(ordinal),
    )
    instance.validate_against(template)
    return instance


class DifferentiableLowerDenseLinearModuleCache:
    """Explicit cache whose identity excludes all dynamic tensor values."""

    def __init__(self) -> None:
        self._entries: dict[
            str,
            tuple[
                CompiledDifferentiableLowerDenseLinearTIR,
                DifferentiableLowerDenseLinearTIRModuleReceiptV1,
            ],
        ] = {}

    def get(
        self,
        template: DifferentiableLowerDenseLinearTIRTemplateV1,
        schedule: DifferentiableLowerDenseLinearTIRScheduleV1,
    ) -> tuple[
        CompiledDifferentiableLowerDenseLinearTIR,
        DifferentiableLowerDenseLinearTIRModuleReceiptV1,
        str,
    ]:
        cache_key = DifferentiableLowerDenseLinearTIRModuleReceiptV1.expected_cache_key(
            template, schedule
        )
        existing = self._entries.get(cache_key)
        if existing is not None:
            return (*existing, "hit")
        compiled = compile_dense_linear_tir(template, schedule)
        receipt = DifferentiableLowerDenseLinearTIRModuleReceiptV1(
            template_hash=template.stable_hash(),
            schedule_hash=schedule.stable_hash(template),
            unscheduled_tir_hash=compiled.unscheduled_tir_hash,
            scheduled_tir_hash=compiled.scheduled_tir_hash,
            device_source_hash=compiled.device_source_hash,
            cache_key=cache_key,
            tvm_version=compiled.tvm_version,
            torch_version=str(torch.__version__),
            exported_symbols=(template.forward_symbol, template.backward_symbol),
        )
        receipt.validate_against(template, schedule)
        self._entries[cache_key] = (compiled, receipt)
        return compiled, receipt, "miss"


@dataclass(frozen=True)
class _LaunchObservation:
    stream_id: int
    tvm_ffi_stream_id: int
    pointer_count: int
    pointer_exact_count: int


class _DenseLinearTIRExecutor:
    def __init__(
        self,
        template: DifferentiableLowerDenseLinearTIRTemplateV1,
        schedule: DifferentiableLowerDenseLinearTIRScheduleV1,
        cache: DifferentiableLowerDenseLinearModuleCache,
    ) -> None:
        self.template = template
        self.schedule = schedule
        self.compiled, self.module_receipt, self.cache_event = cache.get(
            template, schedule
        )
        self.forward_launch_count = 0
        self.backward_launch_count = 0
        self.fallback_count = 0
        self.eager_backward_count = 0
        self.forward_observation: Optional[_LaunchObservation] = None
        self.backward_observation: Optional[_LaunchObservation] = None
        self._tensors: Optional[DenseLinearTIRTensorsV1] = None

    @property
    def primed_tensors(self) -> Optional[DenseLinearTIRTensorsV1]:
        return self._tensors

    def prime(self, tensors: DenseLinearTIRTensorsV1) -> None:
        """Bind the exact forward values and backward adjoints once."""

        if self._tensors is not None:
            raise RuntimeError("dense Linear TIR executor is already primed")
        _validate_dense_linear_tensors(tensors, self.template)
        self._tensors = tensors

    def reject_fallback(self, *, eager_backward: bool) -> None:
        """Count and reject every unsupported execution instead of falling back."""

        self.fallback_count += 1
        self.eager_backward_count += int(eager_backward)
        raise RuntimeError("dense Linear TIR fallback is forbidden")

    def _launch(
        self,
        symbol: str,
        sources: tuple[torch.Tensor, ...],
        outputs: tuple[torch.Tensor, ...],
    ) -> _LaunchObservation:
        import tvm
        import tvm_ffi

        device = sources[0].device
        current = torch.cuda.current_stream(device)
        entry_device = torch.cuda.current_device()
        entry_stream_id = int(current.cuda_stream)
        entry_policy = (
            torch.are_deterministic_algorithms_enabled(),
            torch.get_deterministic_debug_mode(),
        )
        ordinal = device.index if device.index is not None else entry_device
        pointer_count = len(sources) + len(outputs)
        exact_count = 0
        ffi_stream_id = -1
        try:
            with tvm_ffi.use_torch_stream(torch.cuda.stream(current)):
                ffi_stream_id = int(
                    tvm_ffi.get_raw_stream(tvm_ffi.device(f"cuda:{ordinal}"))
                )
                if ffi_stream_id != entry_stream_id:
                    raise RuntimeError("dense Linear TIR current stream differs")
                source_views = []
                output_views = []
                for tensor in sources:
                    view = tvm.runtime.from_dlpack(tensor)
                    exact_count += int(
                        torch.from_dlpack(view).data_ptr() == tensor.data_ptr()
                    )
                    source_views.append(view)
                for tensor in outputs:
                    view = tvm.runtime.from_dlpack(tensor)
                    exact_count += int(
                        torch.from_dlpack(view).data_ptr() == tensor.data_ptr()
                    )
                    output_views.append(view)
                if exact_count != pointer_count:
                    raise RuntimeError("dense Linear TIR DLPack pointer differs")
                self.compiled.executable[symbol](*source_views, *output_views)
        finally:
            current_policy = (
                torch.are_deterministic_algorithms_enabled(),
                torch.get_deterministic_debug_mode(),
            )
            if (
                torch.cuda.current_device() != entry_device
                or int(torch.cuda.current_stream(device).cuda_stream) != entry_stream_id
                or current_policy != entry_policy
            ):
                raise RuntimeError("dense Linear TIR global execution state drifted")
        return _LaunchObservation(
            stream_id=entry_stream_id,
            tvm_ffi_stream_id=ffi_stream_id,
            pointer_count=pointer_count,
            pointer_exact_count=exact_count,
        )

    def forward(
        self, tensors: DenseLinearTIRTensorsV1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_dense_linear_tensors(tensors, self.template)
        if self.forward_launch_count != 0:
            raise RuntimeError("dense Linear TIR forward launched more than once")
        output_a = torch.empty(
            (6, 1, 1024), device=tensors.incoming_lower_a.device, dtype=torch.float32
        )
        output_bias = torch.empty(
            (6, 1), device=tensors.incoming_lower_a.device, dtype=torch.float32
        )
        sources = (
            tensors.incoming_lower_a,
            tensors.preactivation_lower,
            tensors.preactivation_upper,
            tensors.native_alpha,
            tensors.native_beta,
            tensors.dense_split_sign,
            tensors.incoming_lower_bias,
            tensors.operator_weight,
            tensors.operator_bias,
        )
        self.forward_observation = self._launch(
            self.template.forward_symbol, sources, (output_a, output_bias)
        )
        self._tensors = tensors
        self.forward_launch_count += 1
        return output_a, output_bias

    def backward(
        self, output_a_gradient: torch.Tensor, output_bias_gradient: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tensors = self._tensors
        if tensors is None or self.forward_launch_count != 1:
            raise RuntimeError("dense Linear TIR backward precedes forward")
        if self.backward_launch_count != 0:
            raise RuntimeError("dense Linear TIR backward launched more than once")
        if (
            output_a_gradient.data_ptr() != tensors.output_lower_a_gradient.data_ptr()
            or output_bias_gradient.data_ptr()
            != tensors.output_bias_gradient.data_ptr()
        ):
            raise ValueError("dense Linear TIR output adjoint differs")
        alpha_gradient = torch.empty_like(tensors.native_alpha)
        beta_gradient = torch.empty_like(tensors.native_beta)
        sources = (
            tensors.incoming_lower_a,
            tensors.preactivation_lower,
            tensors.preactivation_upper,
            tensors.native_alpha,
            tensors.native_beta,
            tensors.dense_split_sign,
            tensors.operator_weight,
            tensors.operator_bias,
            output_a_gradient,
            output_bias_gradient,
        )
        self.backward_observation = self._launch(
            self.template.backward_symbol, sources, (alpha_gradient, beta_gradient)
        )
        self.backward_launch_count += 1
        return alpha_gradient, beta_gradient


class _DenseLinearTIRFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        incoming_lower_a: torch.Tensor,
        preactivation_lower: torch.Tensor,
        preactivation_upper: torch.Tensor,
        native_alpha: torch.Tensor,
        native_beta: torch.Tensor,
        dense_split_sign: torch.Tensor,
        incoming_lower_bias: torch.Tensor,
        operator_weight: torch.Tensor,
        operator_bias: torch.Tensor,
        executor: _DenseLinearTIRExecutor,
    ):
        tensors = executor.primed_tensors
        if tensors is None:
            raise RuntimeError("dense Linear TIR executor is not primed")
        passed = (
            incoming_lower_a,
            preactivation_lower,
            preactivation_upper,
            native_alpha,
            native_beta,
            dense_split_sign,
            incoming_lower_bias,
            operator_weight,
            operator_bias,
        )
        expected = (
            tensors.incoming_lower_a,
            tensors.preactivation_lower,
            tensors.preactivation_upper,
            tensors.native_alpha,
            tensors.native_beta,
            tensors.dense_split_sign,
            tensors.incoming_lower_bias,
            tensors.operator_weight,
            tensors.operator_bias,
        )
        if any(
            left.data_ptr() != right.data_ptr() for left, right in zip(passed, expected)
        ):
            raise ValueError("dense Linear TIR autograd input differs")
        ctx.executor = executor
        ctx.set_materialize_grads(False)
        return executor.forward(tensors)

    @staticmethod
    def backward(
        ctx, output_a_gradient: torch.Tensor, output_bias_gradient: torch.Tensor
    ):
        if torch.is_grad_enabled():
            raise RuntimeError(
                "dense Linear TIR higher-order gradients are unsupported"
            )
        alpha_gradient, beta_gradient = ctx.executor.backward(
            output_a_gradient, output_bias_gradient
        )
        return (
            None,
            None,
            None,
            alpha_gradient,
            beta_gradient,
            None,
            None,
            None,
            None,
            None,
        )


def _metric(
    name: str, candidate: torch.Tensor, reference: torch.Tensor
) -> DenseLinearTIRParityMetricV1:
    candidate_cpu = candidate.detach().cpu().contiguous()
    reference_cpu = reference.detach().cpu().contiguous()
    difference = (candidate_cpu - reference_cpu).abs()
    metric = DenseLinearTIRParityMetricV1(
        name=name,
        element_count=candidate_cpu.numel(),
        maximum_absolute_difference=float(difference.max().item()),
        allclose=bool(
            torch.allclose(
                candidate_cpu,
                reference_cpu,
                atol=B4B1_REFERENCE_ATOL,
                rtol=B4B1_REFERENCE_RTOL,
            )
        ),
        sign_exact=bool(
            torch.equal(torch.sign(candidate_cpu), torch.sign(reference_cpu))
        ),
        reference_hash=production_tensor_sha256(reference_cpu),
        candidate_hash=production_tensor_sha256(candidate_cpu),
    )
    metric.validate()
    return metric


def run_b4b2_dense_linear_tir_v1(
    capture: ProductionDifferentiableReferenceCaptureV1,
    *,
    fresh_run_ordinal: int,
    cache: Optional[DifferentiableLowerDenseLinearModuleCache] = None,
) -> DenseLinearTIRCandidateResultV1:
    """Compile one S raw instance, execute dense TIR, and compare B4-B1 directly."""

    lower_ir = build_b4b1_differentiable_lower_ir_v1(capture)
    lower_instance = build_b4b1_differentiable_lower_instance_v1(capture, lower_ir)
    major, minor = torch.cuda.get_device_capability()
    template = build_b4b2_dense_linear_template_v1(
        lower_ir, compute_capability=f"sm_{major}{minor}"
    )
    schedule = build_b4b2_dense_linear_schedule_v1(template)
    tensors = build_b4b2_dense_linear_tensors_v1(
        capture, template, device=torch.device("cuda:0")
    )
    instance = build_b4b2_dense_linear_instance_v1(
        template,
        lower_ir,
        lower_instance,
        capture,
        tensors,
        fresh_run_ordinal=fresh_run_ordinal,
    )
    reference: DifferentiableLowerReferenceResultV1 = run_b4b1_pytorch_reference_v1(
        capture, lower_ir, lower_instance
    )
    executor = _DenseLinearTIRExecutor(
        template,
        schedule,
        cache or DifferentiableLowerDenseLinearModuleCache(),
    )
    executor.prime(tensors)
    incoming_hash_before = production_tensor_sha256(tensors.incoming_lower_a)
    output_a, output_bias = _DenseLinearTIRFunction.apply(
        tensors.incoming_lower_a,
        tensors.preactivation_lower,
        tensors.preactivation_upper,
        tensors.native_alpha,
        tensors.native_beta,
        tensors.dense_split_sign,
        tensors.incoming_lower_bias,
        tensors.operator_weight,
        tensors.operator_bias,
        executor,
    )
    alpha_gradient, beta_gradient = torch.autograd.grad(
        (output_a, output_bias),
        (tensors.native_alpha, tensors.native_beta),
        grad_outputs=(
            tensors.output_lower_a_gradient,
            tensors.output_bias_gradient,
        ),
        create_graph=False,
        retain_graph=False,
    )
    if production_tensor_sha256(tensors.incoming_lower_a) != incoming_hash_before:
        raise RuntimeError("dense Linear TIR mutated incoming A")
    candidates = {
        "output_lower_a": output_a,
        "output_bias": output_bias,
        "native_alpha_gradient": alpha_gradient,
        "native_beta_gradient": beta_gradient,
    }
    references = {
        "output_lower_a": reference.output_lower_a,
        "output_bias": reference.output_bias,
        "native_alpha_gradient": reference.native_alpha_gradient,
        "native_beta_gradient": reference.native_beta_gradient,
    }
    if references["native_beta_gradient"] is None:
        raise RuntimeError("dense Linear TIR S-anchor beta gradient is absent")
    metrics = tuple(
        _metric(name, candidates[name], cast(torch.Tensor, references[name]))
        for name in DENSE_LINEAR_OUTPUT_NAMES
    )
    semantic_passed = all(metric.allclose and metric.sign_exact for metric in metrics)
    forward = executor.forward_observation
    backward = executor.backward_observation
    if forward is None or backward is None:
        raise RuntimeError("dense Linear TIR launch inventory is incomplete")
    if (
        forward.stream_id != backward.stream_id
        or forward.tvm_ffi_stream_id != backward.tvm_ffi_stream_id
    ):
        raise RuntimeError("dense Linear TIR forward/backward stream differs")
    launch = DifferentiableLowerDenseLinearTIRLaunchReceiptV1(
        template_hash=template.stable_hash(),
        instance_hash=instance.stable_hash(template),
        schedule_hash=schedule.stable_hash(template),
        module_receipt_hash=executor.module_receipt.stable_hash(template, schedule),
        stream_id=forward.stream_id,
        tvm_ffi_stream_id=forward.tvm_ffi_stream_id,
        input_data_ptrs=tuple(
            (name, tensor.data_ptr())
            for name, tensor in sorted(tensors.tensor_map.items())
        ),
        output_data_ptrs=tuple(
            (name, tensor.data_ptr()) for name, tensor in sorted(candidates.items())
        ),
        output_tensor_hashes=tuple(
            (name, production_tensor_sha256(tensor))
            for name, tensor in sorted(candidates.items())
        ),
        dlpack_pointer_exact_count=(
            forward.pointer_exact_count + backward.pointer_exact_count
        ),
        dlpack_pointer_count=forward.pointer_count + backward.pointer_count,
        cache_event=executor.cache_event,
        forward_launch_count=executor.forward_launch_count,
        backward_launch_count=executor.backward_launch_count,
        fallback_count=executor.fallback_count,
        eager_backward_count=executor.eager_backward_count,
        semantic_passed=semantic_passed,
    )
    launch.validate_against(template, instance, schedule, executor.module_receipt)
    return DenseLinearTIRCandidateResultV1(
        output_lower_a=output_a,
        output_bias=output_bias,
        native_alpha_gradient=alpha_gradient,
        native_beta_gradient=beta_gradient,
        metrics=metrics,
        module_receipt=executor.module_receipt,
        launch_receipt=launch,
    )


__all__ = [
    "DenseLinearTIRCandidateResultV1",
    "DenseLinearTIRParityMetricV1",
    "DenseLinearTIRTensorsV1",
    "DifferentiableLowerDenseLinearModuleCache",
    "build_b4b2_dense_linear_instance_v1",
    "build_b4b2_dense_linear_schedule_v1",
    "build_b4b2_dense_linear_template_v1",
    "build_b4b2_dense_linear_tensors_v1",
    "run_b4b2_dense_linear_tir_v1",
]
