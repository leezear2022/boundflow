"""B4-B2 B2-3 dense P-anchor ConvTranspose TIR correctness runtime."""

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

from boundflow.backends.tvm.differentiable_lower_dense_conv import (
    CompiledDifferentiableLowerDenseConvTIR,
    compile_dense_conv_tir,
)
from boundflow.ir.differentiable_lower_dense_conv_tir import (
    DENSE_CONV_INPUT_NAMES,
    DENSE_CONV_OUTPUT_NAMES,
    DifferentiableLowerDenseConvTIRInstanceV1,
    DifferentiableLowerDenseConvTIRLaunchReceiptV1,
    DifferentiableLowerDenseConvTIRModuleReceiptV1,
    DifferentiableLowerDenseConvTIRScheduleV1,
    DifferentiableLowerDenseConvTIRTemplateV1,
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
class DenseConvTIRTensorsV1:
    """Complete beta-free P-anchor ABI and output adjoints."""

    incoming_lower_a: torch.Tensor
    preactivation_lower: torch.Tensor
    preactivation_upper: torch.Tensor
    native_alpha: torch.Tensor
    incoming_lower_bias: torch.Tensor
    operator_weight: torch.Tensor
    operator_bias: torch.Tensor
    output_lower_a_gradient: torch.Tensor
    output_bias_gradient: torch.Tensor

    @property
    def tensor_map(self) -> dict[str, torch.Tensor]:
        return {
            name: cast(torch.Tensor, getattr(self, name))
            for name in DENSE_CONV_INPUT_NAMES
        }


@dataclass(frozen=True)
class DenseConvTIRParityMetricV1:
    """One direct B4-B1 oracle versus P-anchor TIR comparison."""

    name: str
    element_count: int
    maximum_absolute_difference: float
    allclose: bool
    sign_exact: bool
    reference_hash: str
    candidate_hash: str

    def validate(self) -> None:
        if (
            self.name not in DENSE_CONV_OUTPUT_NAMES
            or self.element_count < 1
            or not math.isfinite(self.maximum_absolute_difference)
            or self.maximum_absolute_difference < 0.0
            or len(self.reference_hash) != 64
            or len(self.candidate_hash) != 64
        ):
            raise ValueError("dense Conv TIR parity metric differs")

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
class DenseConvTIRCandidateResultV1:
    """P-anchor outputs, gradients and compiler/execution evidence."""

    output_lower_a: torch.Tensor
    output_bias: torch.Tensor
    native_alpha_gradient: torch.Tensor
    incoming_lower_a_gradient: torch.Tensor
    metrics: tuple[DenseConvTIRParityMetricV1, ...]
    module_receipt: DifferentiableLowerDenseConvTIRModuleReceiptV1
    launch_receipt: DifferentiableLowerDenseConvTIRLaunchReceiptV1


def build_b4b2_dense_conv_template_v1(
    lower_ir: DifferentiableLowerRegionIRV1,
    *,
    compute_capability: str,
) -> DifferentiableLowerDenseConvTIRTemplateV1:
    """Bind the beta-free dense ABI to the exact approved P-anchor IR."""

    lower_ir.validate()
    attributes = dict(lower_ir.operator_attributes)
    if (
        lower_ir.anchor_id != "performance-conv-8-candidate"
        or lower_ir.operator_kind != "conv2d"
        or lower_ir.coefficient_shape != (6, 1, 16, 8, 8)
        or lower_ir.result_coefficient_shape != (6, 1, 16, 8, 8)
        or lower_ir.relu_logical_shape != (16, 8, 8)
        or lower_ir.beta_active is not False
        or "amendment/operator_bias" not in lower_ir.tensor_contract_map
        or attributes.get("stride") != (1, 1)
        or attributes.get("padding") != (1, 1)
        or attributes.get("dilation") != (1, 1)
        or attributes.get("output_padding") != (0, 0)
        or attributes.get("groups") != 1
    ):
        raise ValueError("dense Conv TIR P-anchor differs")
    mapping_contracts = [
        contract.to_dict()
        for contract in lower_ir.tensor_contracts
        if contract.name.startswith("mapping/")
    ]
    template = DifferentiableLowerDenseConvTIRTemplateV1(
        lower_region_ir_hash=lower_ir.stable_hash(),
        mapping_layout_hash=canonical_tir_hash(mapping_contracts),
        operator_attributes_hash=canonical_tir_hash(attributes),
        compute_capability=compute_capability,
    )
    template.validate()
    return template


def build_b4b2_dense_conv_schedule_v1(
    template: DifferentiableLowerDenseConvTIRTemplateV1,
) -> DifferentiableLowerDenseConvTIRScheduleV1:
    """Return the only B2-3 correctness schedule."""

    schedule = DifferentiableLowerDenseConvTIRScheduleV1(
        template_hash=template.stable_hash()
    )
    schedule.validate_against(template)
    return schedule


def build_b4b2_dense_conv_tensors_v1(
    capture: ProductionDifferentiableReferenceCaptureV1,
    template: DifferentiableLowerDenseConvTIRTemplateV1,
    *,
    device: torch.device,
) -> DenseConvTIRTensorsV1:
    """Materialize the beta-free, explicitly non-timed ABI from frozen raw."""

    capture.validate()
    template.validate()
    values = capture.base.value_map

    def cuda_clone(name: str, *, requires_grad: bool = False) -> torch.Tensor:
        tensor = values[name].value.detach().to(device).contiguous().clone()
        tensor.requires_grad_(requires_grad)
        return tensor

    if capture.operator_bias is None:
        raise ValueError("dense Conv TIR operator bias is absent")
    tensors = DenseConvTIRTensorsV1(
        incoming_lower_a=cuda_clone("incoming_lower_a", requires_grad=True),
        preactivation_lower=cuda_clone("preactivation_lower"),
        preactivation_upper=cuda_clone("preactivation_upper"),
        native_alpha=cuda_clone("native_alpha", requires_grad=True),
        incoming_lower_bias=capture.incoming_lower_bias.value.to(device).contiguous(),
        operator_weight=cuda_clone("operator_weight"),
        operator_bias=capture.operator_bias.value.to(device).contiguous(),
        output_lower_a_gradient=capture.output_lower_a_gradient.value.to(
            device
        ).contiguous(),
        output_bias_gradient=capture.output_bias_gradient.value.to(device).contiguous(),
    )
    _validate_dense_conv_tensors(tensors, template)
    return tensors


def _validate_dense_conv_tensors(
    tensors: DenseConvTIRTensorsV1,
    template: DifferentiableLowerDenseConvTIRTemplateV1,
) -> None:
    expected_shapes = {
        "incoming_lower_a": (6, 1, 16, 8, 8),
        "preactivation_lower": (6, 16, 8, 8),
        "preactivation_upper": (6, 16, 8, 8),
        "native_alpha": (6, 16, 8, 8),
        "incoming_lower_bias": (6, 1),
        "operator_weight": (16, 16, 3, 3),
        "operator_bias": (16,),
        "output_lower_a_gradient": (6, 1, 16, 8, 8),
        "output_bias_gradient": (6, 1),
    }
    values = tensors.tensor_map
    if tuple(sorted(values)) != DENSE_CONV_INPUT_NAMES:
        raise ValueError("dense Conv TIR tensor inventory differs")
    for name, tensor in values.items():
        expected_requires_grad = name in {"incoming_lower_a", "native_alpha"}
        if (
            tuple(tensor.shape) != expected_shapes[name]
            or tensor.device.type != "cuda"
            or tensor.dtype != torch.float32
            or not tensor.is_contiguous()
            or tensor.requires_grad is not expected_requires_grad
            or not bool(torch.isfinite(tensor).all().item())
        ):
            raise ValueError(f"dense Conv TIR tensor differs: {name}")
    if bool((values["preactivation_lower"] > values["preactivation_upper"]).any()):
        raise ValueError("dense Conv TIR interval differs")
    if bool(((values["native_alpha"] < 0) | (values["native_alpha"] > 1)).any()):
        raise ValueError("dense Conv TIR alpha range differs")
    template.validate()


def build_b4b2_dense_conv_instance_v1(
    template: DifferentiableLowerDenseConvTIRTemplateV1,
    lower_ir: DifferentiableLowerRegionIRV1,
    lower_instance: DifferentiableLowerRegionInstanceV1,
    capture: ProductionDifferentiableReferenceCaptureV1,
    tensors: DenseConvTIRTensorsV1,
    *,
    fresh_run_ordinal: int,
) -> DifferentiableLowerDenseConvTIRInstanceV1:
    """Hash-bind every dense value and adjoint without polluting compile keys."""

    lower_instance.validate_against(lower_ir)
    _validate_dense_conv_tensors(tensors, template)
    ordinal = tensors.incoming_lower_a.device.index
    if ordinal is None:
        ordinal = torch.cuda.current_device()
    instance = DifferentiableLowerDenseConvTIRInstanceV1(
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


class DifferentiableLowerDenseConvModuleCache:
    """Explicit compile cache whose identity excludes dynamic tensor values."""

    def __init__(self) -> None:
        self._entries: dict[
            str,
            tuple[
                CompiledDifferentiableLowerDenseConvTIR,
                DifferentiableLowerDenseConvTIRModuleReceiptV1,
            ],
        ] = {}

    def get(
        self,
        template: DifferentiableLowerDenseConvTIRTemplateV1,
        schedule: DifferentiableLowerDenseConvTIRScheduleV1,
    ) -> tuple[
        CompiledDifferentiableLowerDenseConvTIR,
        DifferentiableLowerDenseConvTIRModuleReceiptV1,
        str,
    ]:
        cache_key = DifferentiableLowerDenseConvTIRModuleReceiptV1.expected_cache_key(
            template, schedule
        )
        existing = self._entries.get(cache_key)
        if existing is not None:
            return (*existing, "hit")
        compiled = compile_dense_conv_tir(template, schedule)
        receipt = DifferentiableLowerDenseConvTIRModuleReceiptV1(
            template_hash=template.stable_hash(),
            schedule_hash=schedule.stable_hash(template),
            unscheduled_tir_hash=compiled.unscheduled_tir_hash,
            scheduled_tir_hash=compiled.scheduled_tir_hash,
            device_source_hash=compiled.device_source_hash,
            cache_key=cache_key,
            tvm_version=compiled.tvm_version,
            torch_version=str(torch.__version__),
            exported_symbols=(template.forward_symbol, template.backward_symbol),
            observed_workspace_inventory=compiled.observed_workspace_inventory,
            structural_workspace_check=True,
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


class _DenseConvTIRExecutor:
    def __init__(
        self,
        template: DifferentiableLowerDenseConvTIRTemplateV1,
        schedule: DifferentiableLowerDenseConvTIRScheduleV1,
        cache: DifferentiableLowerDenseConvModuleCache,
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
        self._tensors: Optional[DenseConvTIRTensorsV1] = None

    @property
    def primed_tensors(self) -> Optional[DenseConvTIRTensorsV1]:
        return self._tensors

    def prime(self, tensors: DenseConvTIRTensorsV1) -> None:
        if self._tensors is not None:
            raise RuntimeError("dense Conv TIR executor is already primed")
        _validate_dense_conv_tensors(tensors, self.template)
        self._tensors = tensors

    def reject_fallback(self, *, eager_backward: bool) -> None:
        self.fallback_count += 1
        self.eager_backward_count += int(eager_backward)
        raise RuntimeError("dense Conv TIR fallback is forbidden")

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
                    raise RuntimeError("dense Conv TIR current stream differs")
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
                    raise RuntimeError("dense Conv TIR DLPack pointer differs")
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
                raise RuntimeError("dense Conv TIR global execution state drifted")
        return _LaunchObservation(
            stream_id=entry_stream_id,
            tvm_ffi_stream_id=ffi_stream_id,
            pointer_count=pointer_count,
            pointer_exact_count=exact_count,
        )

    def forward(
        self, tensors: DenseConvTIRTensorsV1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_dense_conv_tensors(tensors, self.template)
        if self.forward_launch_count != 0:
            raise RuntimeError("dense Conv TIR forward launched more than once")
        output_a = torch.empty_like(tensors.incoming_lower_a)
        output_bias = torch.empty_like(tensors.incoming_lower_bias)
        sources = (
            tensors.incoming_lower_a,
            tensors.preactivation_lower,
            tensors.preactivation_upper,
            tensors.native_alpha,
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
            raise RuntimeError("dense Conv TIR backward precedes forward")
        if self.backward_launch_count != 0:
            raise RuntimeError("dense Conv TIR backward launched more than once")
        if (
            output_a_gradient.data_ptr() != tensors.output_lower_a_gradient.data_ptr()
            or output_bias_gradient.data_ptr()
            != tensors.output_bias_gradient.data_ptr()
        ):
            raise ValueError("dense Conv TIR output adjoint differs")
        alpha_gradient = torch.empty_like(tensors.native_alpha)
        incoming_gradient = torch.empty_like(tensors.incoming_lower_a)
        sources = (
            tensors.incoming_lower_a,
            tensors.preactivation_lower,
            tensors.preactivation_upper,
            tensors.native_alpha,
            tensors.operator_weight,
            tensors.operator_bias,
            output_a_gradient,
            output_bias_gradient,
        )
        self.backward_observation = self._launch(
            self.template.backward_symbol,
            sources,
            (alpha_gradient, incoming_gradient),
        )
        self.backward_launch_count += 1
        return alpha_gradient, incoming_gradient


class _DenseConvTIRFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        incoming_lower_a: torch.Tensor,
        preactivation_lower: torch.Tensor,
        preactivation_upper: torch.Tensor,
        native_alpha: torch.Tensor,
        incoming_lower_bias: torch.Tensor,
        operator_weight: torch.Tensor,
        operator_bias: torch.Tensor,
        executor: _DenseConvTIRExecutor,
    ):
        tensors = executor.primed_tensors
        if tensors is None:
            raise RuntimeError("dense Conv TIR executor is not primed")
        passed = (
            incoming_lower_a,
            preactivation_lower,
            preactivation_upper,
            native_alpha,
            incoming_lower_bias,
            operator_weight,
            operator_bias,
        )
        expected = (
            tensors.incoming_lower_a,
            tensors.preactivation_lower,
            tensors.preactivation_upper,
            tensors.native_alpha,
            tensors.incoming_lower_bias,
            tensors.operator_weight,
            tensors.operator_bias,
        )
        if any(
            left.data_ptr() != right.data_ptr() for left, right in zip(passed, expected)
        ):
            raise ValueError("dense Conv TIR autograd input differs")
        ctx.executor = executor
        ctx.set_materialize_grads(False)
        return executor.forward(tensors)

    @staticmethod
    def backward(
        ctx, output_a_gradient: torch.Tensor, output_bias_gradient: torch.Tensor
    ):
        if torch.is_grad_enabled():
            raise RuntimeError("dense Conv TIR higher-order gradients are unsupported")
        alpha_gradient, incoming_gradient = ctx.executor.backward(
            output_a_gradient, output_bias_gradient
        )
        return (
            incoming_gradient,
            None,
            None,
            alpha_gradient,
            None,
            None,
            None,
            None,
        )


def _metric(
    name: str, candidate: torch.Tensor, reference: torch.Tensor
) -> DenseConvTIRParityMetricV1:
    candidate_cpu = candidate.detach().cpu().contiguous()
    reference_cpu = reference.detach().cpu().contiguous()
    difference = (candidate_cpu - reference_cpu).abs()
    metric = DenseConvTIRParityMetricV1(
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


def run_b4b2_dense_conv_tir_v1(
    capture: ProductionDifferentiableReferenceCaptureV1,
    *,
    fresh_run_ordinal: int,
    cache: Optional[DifferentiableLowerDenseConvModuleCache] = None,
) -> DenseConvTIRCandidateResultV1:
    """Execute one P raw instance and compare all outputs to B4-B1 directly."""

    lower_ir = build_b4b1_differentiable_lower_ir_v1(capture)
    lower_instance = build_b4b1_differentiable_lower_instance_v1(capture, lower_ir)
    major, minor = torch.cuda.get_device_capability()
    template = build_b4b2_dense_conv_template_v1(
        lower_ir, compute_capability=f"sm_{major}{minor}"
    )
    schedule = build_b4b2_dense_conv_schedule_v1(template)
    tensors = build_b4b2_dense_conv_tensors_v1(
        capture, template, device=torch.device("cuda:0")
    )
    instance = build_b4b2_dense_conv_instance_v1(
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
    executor = _DenseConvTIRExecutor(
        template, schedule, cache or DifferentiableLowerDenseConvModuleCache()
    )
    executor.prime(tensors)
    incoming_hash_before = production_tensor_sha256(tensors.incoming_lower_a)
    output_a, output_bias = _DenseConvTIRFunction.apply(
        tensors.incoming_lower_a,
        tensors.preactivation_lower,
        tensors.preactivation_upper,
        tensors.native_alpha,
        tensors.incoming_lower_bias,
        tensors.operator_weight,
        tensors.operator_bias,
        executor,
    )
    incoming_gradient, alpha_gradient = torch.autograd.grad(
        (output_a, output_bias),
        (tensors.incoming_lower_a, tensors.native_alpha),
        grad_outputs=(
            tensors.output_lower_a_gradient,
            tensors.output_bias_gradient,
        ),
        create_graph=False,
        retain_graph=False,
    )
    if production_tensor_sha256(tensors.incoming_lower_a) != incoming_hash_before:
        raise RuntimeError("dense Conv TIR mutated incoming A")
    if reference.native_beta_gradient is not None:
        raise RuntimeError("dense Conv TIR P-anchor beta gradient is present")
    if reference.incoming_lower_a_gradient is None:
        raise RuntimeError("dense Conv TIR P-anchor incoming gradient is absent")
    candidates = {
        "output_lower_a": output_a,
        "output_bias": output_bias,
        "native_alpha_gradient": alpha_gradient,
        "incoming_lower_a_gradient": incoming_gradient,
    }
    references = {
        "output_lower_a": reference.output_lower_a,
        "output_bias": reference.output_bias,
        "native_alpha_gradient": reference.native_alpha_gradient,
        "incoming_lower_a_gradient": reference.incoming_lower_a_gradient,
    }
    metrics = tuple(
        _metric(name, candidates[name], cast(torch.Tensor, references[name]))
        for name in DENSE_CONV_OUTPUT_NAMES
    )
    semantic_passed = all(metric.allclose and metric.sign_exact for metric in metrics)
    forward = executor.forward_observation
    backward = executor.backward_observation
    if forward is None or backward is None:
        raise RuntimeError("dense Conv TIR launch inventory is incomplete")
    if (
        forward.stream_id != backward.stream_id
        or forward.tvm_ffi_stream_id != backward.tvm_ffi_stream_id
    ):
        raise RuntimeError("dense Conv TIR forward/backward stream differs")
    launch = DifferentiableLowerDenseConvTIRLaunchReceiptV1(
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
        incoming_a_gradient_present=True,
        native_alpha_gradient_present=True,
        beta_gradient_present=False,
        semantic_passed=semantic_passed,
    )
    launch.validate_against(template, instance, schedule, executor.module_receipt)
    return DenseConvTIRCandidateResultV1(
        output_lower_a=output_a,
        output_bias=output_bias,
        native_alpha_gradient=alpha_gradient,
        incoming_lower_a_gradient=incoming_gradient,
        metrics=metrics,
        module_receipt=executor.module_receipt,
        launch_receipt=launch,
    )


__all__ = [
    "DenseConvTIRCandidateResultV1",
    "DenseConvTIRParityMetricV1",
    "DenseConvTIRTensorsV1",
    "DifferentiableLowerDenseConvModuleCache",
    "build_b4b2_dense_conv_instance_v1",
    "build_b4b2_dense_conv_schedule_v1",
    "build_b4b2_dense_conv_template_v1",
    "build_b4b2_dense_conv_tensors_v1",
    "run_b4b2_dense_conv_tir_v1",
]
