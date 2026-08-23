"""B4-B2 B2-4 sparse-source P-anchor Conv TIR correctness runtime."""

# mypy: disable-error-code=import-untyped
# pylint: disable=import-error,import-outside-toplevel,missing-function-docstring
# pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-locals
# pylint: disable=too-many-statements,too-many-boolean-expressions
# pylint: disable=abstract-method,arguments-differ,too-few-public-methods
# pylint: disable=too-many-positional-arguments,duplicate-code,protected-access
# pylint: disable=missing-class-docstring,line-too-long

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, cast

import torch

from boundflow.backends.tvm.differentiable_lower_sparse_conv import (
    CompiledDifferentiableLowerSparseConvTIR,
    compile_sparse_conv_tir,
)
from boundflow.ir.differentiable_lower_sparse_conv_tir import (
    SPARSE_CONV_CANDIDATE_KNOBS,
    SPARSE_CONV_INPUT_NAMES,
    SPARSE_CONV_OUTPUT_NAMES,
    DifferentiableLowerSparseConvCandidateLedgerV1,
    DifferentiableLowerSparseConvGradientProjectionReceiptV1,
    DifferentiableLowerSparseConvTIRInstanceV1,
    DifferentiableLowerSparseConvTIRLaunchReceiptV1,
    DifferentiableLowerSparseConvTIRModuleReceiptV1,
    DifferentiableLowerSparseConvTIRScheduleV1,
    DifferentiableLowerSparseConvTIRTemplateV1,
)
from boundflow.ir.differentiable_lower_region import (
    DifferentiableLowerRegionIRV1,
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
from .fsg4_b4b2_dense_conv_tir import build_b4b2_dense_conv_template_v1
from .rvir_v4_production_state import production_tensor_sha256


@dataclass(frozen=True)
class SparseConvTIRTensorsV1:
    incoming_lower_a: torch.Tensor
    preactivation_lower: torch.Tensor
    preactivation_upper: torch.Tensor
    compressed_alpha: torch.Tensor
    incoming_lower_bias: torch.Tensor
    operator_weight: torch.Tensor
    operator_bias: torch.Tensor
    output_lower_a_gradient: torch.Tensor
    output_bias_gradient: torch.Tensor

    @property
    def tensor_map(self) -> dict[str, torch.Tensor]:
        return {
            name: cast(torch.Tensor, getattr(self, name))
            for name in SPARSE_CONV_INPUT_NAMES
        }


@dataclass(frozen=True)
class SparseConvTIRParityMetricV1:
    name: str
    element_count: int
    maximum_absolute_difference: float
    allclose: bool
    sign_exact: bool
    reference_hash: str
    candidate_hash: str

    def validate(self) -> None:
        if (
            self.name not in SPARSE_CONV_OUTPUT_NAMES
            or self.element_count < 1
            or not math.isfinite(self.maximum_absolute_difference)
            or self.maximum_absolute_difference < 0.0
            or len(self.reference_hash) != 64
            or len(self.candidate_hash) != 64
        ):
            raise ValueError("sparse-source Conv parity metric differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return dict(self.__dict__)


@dataclass(frozen=True)
class SparseConvTIRCandidateResultV1:
    output_lower_a: torch.Tensor
    output_bias: torch.Tensor
    compressed_alpha_gradient: torch.Tensor
    incoming_lower_a_gradient: torch.Tensor
    metrics: tuple[SparseConvTIRParityMetricV1, ...]
    projection_receipt: DifferentiableLowerSparseConvGradientProjectionReceiptV1
    module_receipt: DifferentiableLowerSparseConvTIRModuleReceiptV1
    launch_receipt: DifferentiableLowerSparseConvTIRLaunchReceiptV1


def _mapping_coordinates(
    capture: ProductionDifferentiableReferenceCaptureV1,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    indices = [
        snapshot.value
        for name, snapshot in sorted(capture.mapping_tensors)
        if "/feature_index/" in name
    ]
    feature_shapes = [
        snapshot.value
        for name, snapshot in capture.mapping_tensors
        if name.endswith("/feature_shape")
    ]
    beta_locations = [
        snapshot.value
        for name, snapshot in capture.mapping_tensors
        if name.endswith("/location")
    ]
    beta_signs = [
        snapshot.value
        for name, snapshot in capture.mapping_tensors
        if name.endswith("/sign")
    ]
    if (
        len(indices) != 3
        or any(index.dtype != torch.int64 or index.shape != (86,) for index in indices)
        or len(feature_shapes) != 1
        or feature_shapes[0].dtype != torch.int64
        or tuple(feature_shapes[0].tolist()) != (16, 8, 8)
        or len(beta_locations) != 1
        or beta_locations[0].dtype != torch.int64
        or beta_locations[0].shape != (6, 0)
        or len(beta_signs) != 1
        or beta_signs[0].dtype != torch.float32
        or beta_signs[0].shape != (6, 0)
    ):
        raise ValueError("sparse-source Conv mapping differs")
    coordinates = tuple(
        zip(*(tuple(int(value) for value in index.tolist()) for index in indices))
    )
    if len(set(coordinates)) != 86:
        raise ValueError("sparse-source Conv coordinates differ")
    return tuple(tuple(int(value) for value in index.tolist()) for index in indices)  # type: ignore[return-value]


def build_b4b2_sparse_conv_template_v1(
    lower_ir: DifferentiableLowerRegionIRV1,
    capture: ProductionDifferentiableReferenceCaptureV1,
    *,
    compute_capability: str,
) -> DifferentiableLowerSparseConvTIRTemplateV1:
    lower_ir.validate()
    capture.validate()
    dense_template = build_b4b2_dense_conv_template_v1(
        lower_ir, compute_capability=compute_capability
    )
    channels, heights, widths = _mapping_coordinates(capture)
    coordinates = tuple(zip(channels, heights, widths))
    template = DifferentiableLowerSparseConvTIRTemplateV1(
        lower_region_ir_hash=lower_ir.stable_hash(),
        dense_template_hash=dense_template.stable_hash(),
        operator_attributes_hash=canonical_tir_hash(dict(lower_ir.operator_attributes)),
        alpha_coordinate_hash=canonical_tir_hash(
            [list(coordinate) for coordinate in coordinates]
        ),
        alpha_channels=channels,
        alpha_heights=heights,
        alpha_widths=widths,
        compute_capability=compute_capability,
    )
    template.validate()
    return template


def build_b4b2_sparse_conv_schedules_v1(
    template: DifferentiableLowerSparseConvTIRTemplateV1,
) -> tuple[DifferentiableLowerSparseConvTIRScheduleV1, ...]:
    schedules = tuple(
        DifferentiableLowerSparseConvTIRScheduleV1(
            template_hash=template.stable_hash(),
            candidate_ordinal=ordinal,
            thread_extent=knobs[0],
            output_channel_tile=knobs[1],
            spatial_tile=knobs[2],
            reduction_unroll=knobs[3],
        )
        for ordinal, knobs in enumerate(SPARSE_CONV_CANDIDATE_KNOBS)
    )
    for schedule in schedules:
        schedule.validate_against(template)
    return schedules


def build_b4b2_sparse_conv_ledger_v1(
    template: DifferentiableLowerSparseConvTIRTemplateV1,
    schedules: tuple[DifferentiableLowerSparseConvTIRScheduleV1, ...],
) -> DifferentiableLowerSparseConvCandidateLedgerV1:
    ledger = DifferentiableLowerSparseConvCandidateLedgerV1(
        template_hash=template.stable_hash(),
        schedule_hashes=tuple(schedule.stable_hash(template) for schedule in schedules),
    )
    ledger.validate_against(template, schedules)
    return ledger


def build_b4b2_sparse_conv_tensors_v1(
    capture: ProductionDifferentiableReferenceCaptureV1,
    template: DifferentiableLowerSparseConvTIRTemplateV1,
    *,
    device: torch.device,
) -> SparseConvTIRTensorsV1:
    capture.validate()
    template.validate()
    coordinates = _mapping_coordinates(capture)
    if coordinates != (
        template.alpha_channels,
        template.alpha_heights,
        template.alpha_widths,
    ):
        raise ValueError("sparse-source Conv template mapping differs")
    values = capture.base.value_map

    def cuda_clone(name: str, *, requires_grad: bool = False) -> torch.Tensor:
        tensor = values[name].value.detach().to(device).contiguous().clone()
        tensor.requires_grad_(requires_grad)
        return tensor

    production_alpha = values["production_alpha"].value
    production_beta = values["production_beta"].value
    if production_alpha.shape != (2, 1, 6, 86) or production_beta.shape != (6, 0):
        raise ValueError("sparse-source Conv production state differs")
    compressed_alpha = production_alpha[0, 0].detach().to(device).contiguous().clone()
    compressed_alpha.requires_grad_(True)
    if capture.operator_bias is None:
        raise ValueError("sparse-source Conv operator bias is absent")
    tensors = SparseConvTIRTensorsV1(
        incoming_lower_a=cuda_clone("incoming_lower_a", requires_grad=True),
        preactivation_lower=cuda_clone("preactivation_lower"),
        preactivation_upper=cuda_clone("preactivation_upper"),
        compressed_alpha=compressed_alpha,
        incoming_lower_bias=capture.incoming_lower_bias.value.to(device).contiguous(),
        operator_weight=cuda_clone("operator_weight"),
        operator_bias=capture.operator_bias.value.to(device).contiguous(),
        output_lower_a_gradient=capture.output_lower_a_gradient.value.to(
            device
        ).contiguous(),
        output_bias_gradient=capture.output_bias_gradient.value.to(device).contiguous(),
    )
    _validate_sparse_conv_tensors(tensors, template)
    return tensors


def _validate_sparse_conv_tensors(tensors, template) -> None:
    expected_shapes = {
        "incoming_lower_a": (6, 1, 16, 8, 8),
        "preactivation_lower": (6, 16, 8, 8),
        "preactivation_upper": (6, 16, 8, 8),
        "compressed_alpha": (6, 86),
        "incoming_lower_bias": (6, 1),
        "operator_weight": (16, 16, 3, 3),
        "operator_bias": (16,),
        "output_lower_a_gradient": (6, 1, 16, 8, 8),
        "output_bias_gradient": (6, 1),
    }
    values = tensors.tensor_map
    if tuple(sorted(values)) != SPARSE_CONV_INPUT_NAMES:
        raise ValueError("sparse-source Conv tensor inventory differs")
    for name, tensor in values.items():
        requires_grad = name in {"incoming_lower_a", "compressed_alpha"}
        if (
            tuple(tensor.shape) != expected_shapes[name]
            or tensor.device.type != "cuda"
            or tensor.dtype != torch.float32
            or not tensor.is_contiguous()
            or tensor.requires_grad is not requires_grad
            or not bool(torch.isfinite(tensor).all().item())
        ):
            raise ValueError(f"sparse-source Conv tensor differs: {name}")
    if bool((values["preactivation_lower"] > values["preactivation_upper"]).any()):
        raise ValueError("sparse-source Conv interval differs")
    if bool(
        ((values["compressed_alpha"] < 0) | (values["compressed_alpha"] > 1)).any()
    ):
        raise ValueError("sparse-source Conv alpha range differs")
    template.validate()


def build_b4b2_sparse_conv_instance_v1(
    template,
    lower_ir,
    lower_instance,
    capture,
    tensors,
    *,
    fresh_run_ordinal: int,
) -> DifferentiableLowerSparseConvTIRInstanceV1:
    lower_instance.validate_against(lower_ir)
    _validate_sparse_conv_tensors(tensors, template)
    ordinal = tensors.incoming_lower_a.device.index
    if ordinal is None:
        ordinal = torch.cuda.current_device()
    instance = DifferentiableLowerSparseConvTIRInstanceV1(
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


class DifferentiableLowerSparseConvModuleCache:
    def __init__(self) -> None:
        self._entries: dict[
            str,
            tuple[
                CompiledDifferentiableLowerSparseConvTIR,
                DifferentiableLowerSparseConvTIRModuleReceiptV1,
            ],
        ] = {}

    def get(self, template, schedule):
        cache_key = DifferentiableLowerSparseConvTIRModuleReceiptV1.expected_cache_key(
            template, schedule
        )
        existing = self._entries.get(cache_key)
        if existing is not None:
            return (*existing, "hit")
        compiled = compile_sparse_conv_tir(template, schedule)
        receipt = DifferentiableLowerSparseConvTIRModuleReceiptV1(
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


class _SparseConvTIRExecutor:
    def __init__(self, template, schedule, cache) -> None:
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
        self._tensors: Optional[SparseConvTIRTensorsV1] = None

    @property
    def primed_tensors(self):
        return self._tensors

    def prime(self, tensors) -> None:
        if self._tensors is not None:
            raise RuntimeError("sparse-source Conv executor is already primed")
        _validate_sparse_conv_tensors(tensors, self.template)
        self._tensors = tensors

    def reject_fallback(self, *, eager_backward: bool) -> None:
        self.fallback_count += 1
        self.eager_backward_count += int(eager_backward)
        raise RuntimeError("sparse-source Conv fallback is forbidden")

    def _launch(self, symbol, sources, outputs):
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
                    raise RuntimeError("sparse-source Conv current stream differs")
                source_views, output_views = [], []
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
                    raise RuntimeError("sparse-source Conv DLPack pointer differs")
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
                raise RuntimeError("sparse-source Conv global state drifted")
        return _LaunchObservation(
            entry_stream_id, ffi_stream_id, pointer_count, exact_count
        )

    def forward(self, tensors):
        _validate_sparse_conv_tensors(tensors, self.template)
        if self.forward_launch_count:
            raise RuntimeError("sparse-source Conv forward repeats")
        output_a = torch.empty_like(tensors.incoming_lower_a)
        output_bias = torch.empty_like(tensors.incoming_lower_bias)
        sources = (
            tensors.incoming_lower_a,
            tensors.preactivation_lower,
            tensors.preactivation_upper,
            tensors.compressed_alpha,
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

    def backward(self, output_a_gradient, output_bias_gradient):
        tensors = self._tensors
        if tensors is None or self.forward_launch_count != 1:
            raise RuntimeError("sparse-source Conv backward precedes forward")
        if self.backward_launch_count:
            raise RuntimeError("sparse-source Conv backward repeats")
        if (
            output_a_gradient.data_ptr() != tensors.output_lower_a_gradient.data_ptr()
            or output_bias_gradient.data_ptr()
            != tensors.output_bias_gradient.data_ptr()
        ):
            raise ValueError("sparse-source Conv output adjoint differs")
        alpha_gradient = torch.empty_like(tensors.compressed_alpha)
        incoming_gradient = torch.empty_like(tensors.incoming_lower_a)
        sources = (
            tensors.incoming_lower_a,
            tensors.preactivation_lower,
            tensors.preactivation_upper,
            tensors.compressed_alpha,
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


class _SparseConvTIRFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        incoming,
        lower,
        upper,
        alpha,
        incoming_bias,
        weight,
        operator_bias,
        executor,
    ):
        tensors = executor.primed_tensors
        if tensors is None:
            raise RuntimeError("sparse-source Conv executor is not primed")
        passed = (incoming, lower, upper, alpha, incoming_bias, weight, operator_bias)
        expected = (
            tensors.incoming_lower_a,
            tensors.preactivation_lower,
            tensors.preactivation_upper,
            tensors.compressed_alpha,
            tensors.incoming_lower_bias,
            tensors.operator_weight,
            tensors.operator_bias,
        )
        if any(
            left.data_ptr() != right.data_ptr() for left, right in zip(passed, expected)
        ):
            raise ValueError("sparse-source Conv autograd input differs")
        ctx.executor = executor
        ctx.set_materialize_grads(False)
        return executor.forward(tensors)

    @staticmethod
    def backward(ctx, output_a_gradient, output_bias_gradient):
        if torch.is_grad_enabled():
            raise RuntimeError("sparse-source Conv higher-order gradients unsupported")
        alpha_gradient, incoming_gradient = ctx.executor.backward(
            output_a_gradient, output_bias_gradient
        )
        return incoming_gradient, None, None, alpha_gradient, None, None, None, None


def _metric(name, candidate, reference):
    candidate_cpu = candidate.detach().cpu().contiguous()
    reference_cpu = reference.detach().cpu().contiguous()
    difference = (candidate_cpu - reference_cpu).abs()
    metric = SparseConvTIRParityMetricV1(
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


def _reference_compressed_alpha(reference, template):
    native = reference.native_alpha_gradient
    channels = torch.tensor(template.alpha_channels)
    heights = torch.tensor(template.alpha_heights)
    widths = torch.tensor(template.alpha_widths)
    compressed = native[:, channels, heights, widths].contiguous()
    mask = torch.ones_like(native, dtype=torch.bool)
    mask[:, channels, heights, widths] = False
    return compressed, bool(torch.count_nonzero(native[mask]).item() == 0)


def _projection_receipt(template, instance, reference, candidate_alpha):
    if reference.native_beta_gradient is not None:
        raise RuntimeError("sparse-source Conv beta gradient is present")
    native = reference.native_alpha_gradient.contiguous()
    reference_alpha, unowned_zero = _reference_compressed_alpha(reference, template)
    candidate = candidate_alpha.detach().cpu().contiguous()
    projected = torch.zeros_like(native)
    channels = torch.tensor(template.alpha_channels)
    heights = torch.tensor(template.alpha_heights)
    widths = torch.tensor(template.alpha_widths)
    projected[:, channels, heights, widths] = candidate
    receipt = DifferentiableLowerSparseConvGradientProjectionReceiptV1(
        template_hash=template.stable_hash(),
        instance_hash=instance.stable_hash(template),
        reference_native_alpha_gradient_hash=production_tensor_sha256(native),
        reference_compressed_alpha_gradient_hash=production_tensor_sha256(
            reference_alpha
        ),
        candidate_compressed_alpha_gradient_hash=production_tensor_sha256(candidate),
        projected_native_alpha_gradient_hash=production_tensor_sha256(projected),
        alpha_owned_element_count=6 * 86,
        coordinate_mapping_exact=bool(
            torch.equal(projected[:, channels, heights, widths], candidate)
        ),
        alpha_numerical_passed=bool(
            torch.allclose(
                candidate,
                reference_alpha,
                atol=B4B1_REFERENCE_ATOL,
                rtol=B4B1_REFERENCE_RTOL,
            )
        ),
        nonzero_sign_exact=bool(
            torch.equal(torch.sign(candidate), torch.sign(reference_alpha))
        ),
        unowned_native_zero_exact=unowned_zero,
        beta_gradient_absent=True,
    )
    receipt.validate_against(template, instance)
    return receipt, reference_alpha


def run_b4b2_sparse_conv_tir_v1(
    capture: ProductionDifferentiableReferenceCaptureV1,
    *,
    fresh_run_ordinal: int,
    candidate_ordinal: int = 0,
    cache: Optional[DifferentiableLowerSparseConvModuleCache] = None,
) -> SparseConvTIRCandidateResultV1:
    lower_ir = build_b4b1_differentiable_lower_ir_v1(capture)
    lower_instance = build_b4b1_differentiable_lower_instance_v1(capture, lower_ir)
    major, minor = torch.cuda.get_device_capability()
    template = build_b4b2_sparse_conv_template_v1(
        lower_ir, capture, compute_capability=f"sm_{major}{minor}"
    )
    schedules = build_b4b2_sparse_conv_schedules_v1(template)
    build_b4b2_sparse_conv_ledger_v1(template, schedules)
    schedule = schedules[candidate_ordinal]
    tensors = build_b4b2_sparse_conv_tensors_v1(
        capture, template, device=torch.device("cuda:0")
    )
    instance = build_b4b2_sparse_conv_instance_v1(
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
    executor = _SparseConvTIRExecutor(
        template, schedule, cache or DifferentiableLowerSparseConvModuleCache()
    )
    executor.prime(tensors)
    source_hashes = tuple(
        production_tensor_sha256(tensor)
        for tensor in (tensors.incoming_lower_a, tensors.compressed_alpha)
    )
    output_a, output_bias = _SparseConvTIRFunction.apply(
        tensors.incoming_lower_a,
        tensors.preactivation_lower,
        tensors.preactivation_upper,
        tensors.compressed_alpha,
        tensors.incoming_lower_bias,
        tensors.operator_weight,
        tensors.operator_bias,
        executor,
    )
    incoming_gradient, alpha_gradient = torch.autograd.grad(
        (output_a, output_bias),
        (tensors.incoming_lower_a, tensors.compressed_alpha),
        grad_outputs=(tensors.output_lower_a_gradient, tensors.output_bias_gradient),
    )
    if source_hashes != tuple(
        production_tensor_sha256(tensor)
        for tensor in (tensors.incoming_lower_a, tensors.compressed_alpha)
    ):
        raise RuntimeError("sparse-source Conv mutated an input")
    if reference.incoming_lower_a_gradient is None:
        raise RuntimeError("sparse-source Conv incoming gradient is absent")
    projection, reference_alpha = _projection_receipt(
        template, instance, reference, alpha_gradient
    )
    candidates = {
        "output_lower_a": output_a,
        "output_bias": output_bias,
        "compressed_alpha_gradient": alpha_gradient,
        "incoming_lower_a_gradient": incoming_gradient,
    }
    references = {
        "output_lower_a": reference.output_lower_a,
        "output_bias": reference.output_bias,
        "compressed_alpha_gradient": reference_alpha,
        "incoming_lower_a_gradient": reference.incoming_lower_a_gradient,
    }
    metrics = tuple(
        _metric(name, candidates[name], references[name])
        for name in SPARSE_CONV_OUTPUT_NAMES
    )
    semantic_passed = all(metric.allclose and metric.sign_exact for metric in metrics)
    forward, backward = executor.forward_observation, executor.backward_observation
    if forward is None or backward is None:
        raise RuntimeError("sparse-source Conv launch inventory incomplete")
    if (
        forward.stream_id != backward.stream_id
        or forward.tvm_ffi_stream_id != backward.tvm_ffi_stream_id
    ):
        raise RuntimeError("sparse-source Conv stream differs")
    launch = DifferentiableLowerSparseConvTIRLaunchReceiptV1(
        template_hash=template.stable_hash(),
        instance_hash=instance.stable_hash(template),
        schedule_hash=schedule.stable_hash(template),
        module_receipt_hash=executor.module_receipt.stable_hash(template, schedule),
        projection_receipt_hash=projection.stable_hash(template, instance),
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
        dlpack_pointer_exact_count=forward.pointer_exact_count
        + backward.pointer_exact_count,
        dlpack_pointer_count=forward.pointer_count + backward.pointer_count,
        cache_event=executor.cache_event,
        forward_launch_count=executor.forward_launch_count,
        backward_launch_count=executor.backward_launch_count,
        fallback_count=executor.fallback_count,
        eager_backward_count=executor.eager_backward_count,
        semantic_passed=semantic_passed,
    )
    launch.validate_against(
        template, instance, schedule, executor.module_receipt, projection
    )
    return SparseConvTIRCandidateResultV1(
        output_lower_a=output_a,
        output_bias=output_bias,
        compressed_alpha_gradient=alpha_gradient,
        incoming_lower_a_gradient=incoming_gradient,
        metrics=metrics,
        projection_receipt=projection,
        module_receipt=executor.module_receipt,
        launch_receipt=launch,
    )


__all__ = [
    "DifferentiableLowerSparseConvModuleCache",
    "SparseConvTIRCandidateResultV1",
    "SparseConvTIRParityMetricV1",
    "SparseConvTIRTensorsV1",
    "build_b4b2_sparse_conv_instance_v1",
    "build_b4b2_sparse_conv_ledger_v1",
    "build_b4b2_sparse_conv_schedules_v1",
    "build_b4b2_sparse_conv_template_v1",
    "build_b4b2_sparse_conv_tensors_v1",
    "run_b4b2_sparse_conv_tir_v1",
]
