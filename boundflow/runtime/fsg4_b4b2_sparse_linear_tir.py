"""B4-B2 B2-2 S-anchor sparse-source Linear TIR correctness runtime."""

# mypy: disable-error-code=import-untyped
# pylint: disable=import-error,import-outside-toplevel,missing-function-docstring
# pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-locals
# pylint: disable=too-many-statements,too-many-boolean-expressions
# pylint: disable=abstract-method,arguments-differ,too-few-public-methods
# pylint: disable=too-many-positional-arguments,duplicate-code,protected-access

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, cast

import torch

from boundflow.backends.tvm.differentiable_lower_sparse_linear import (
    CompiledDifferentiableLowerSparseLinearTIR,
    compile_sparse_linear_tir,
)
from boundflow.ir.differentiable_lower_dense_linear_tir import (
    DifferentiableLowerDenseLinearTIRTemplateV1,
)
from boundflow.ir.differentiable_lower_sparse_linear_tir import (
    SPARSE_LINEAR_INPUT_NAMES,
    SPARSE_LINEAR_OUTPUT_NAMES,
    DifferentiableLowerSparseLinearGradientProjectionReceiptV1,
    DifferentiableLowerSparseLinearTIRInstanceV1,
    DifferentiableLowerSparseLinearTIRLaunchReceiptV1,
    DifferentiableLowerSparseLinearTIRModuleReceiptV1,
    DifferentiableLowerSparseLinearTIRScheduleV1,
    DifferentiableLowerSparseLinearTIRTemplateV1,
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
from .fsg4_b4b2_dense_linear_tir import build_b4b2_dense_linear_template_v1
from .rvir_v4_production_state import production_tensor_sha256


@dataclass(frozen=True)
class SparseLinearTIRTensorsV1:
    """Compressed production values and exact output adjoints for one S run."""

    incoming_lower_a: torch.Tensor
    preactivation_lower: torch.Tensor
    preactivation_upper: torch.Tensor
    compressed_alpha: torch.Tensor
    compressed_beta: torch.Tensor
    incoming_lower_bias: torch.Tensor
    operator_weight: torch.Tensor
    operator_bias: torch.Tensor
    output_lower_a_gradient: torch.Tensor
    output_bias_gradient: torch.Tensor

    @property
    def tensor_map(self) -> dict[str, torch.Tensor]:
        return {
            name: cast(torch.Tensor, getattr(self, name))
            for name in SPARSE_LINEAR_INPUT_NAMES
        }


@dataclass(frozen=True)
class SparseLinearTIRParityMetricV1:
    """One direct B4-B1 oracle versus sparse-source TIR comparison."""

    name: str
    element_count: int
    maximum_absolute_difference: float
    allclose: bool
    sign_exact: bool
    reference_hash: str
    candidate_hash: str

    def validate(self) -> None:
        if (
            self.name not in SPARSE_LINEAR_OUTPUT_NAMES
            or self.element_count < 1
            or not math.isfinite(self.maximum_absolute_difference)
            or self.maximum_absolute_difference < 0.0
            or len(self.reference_hash) != 64
            or len(self.candidate_hash) != 64
        ):
            raise ValueError("sparse-source Linear TIR parity metric differs")

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
class SparseLinearTIRCandidateResultV1:
    """Sparse outputs, compressed gradients and complete compiler evidence."""

    output_lower_a: torch.Tensor
    output_bias: torch.Tensor
    compressed_alpha_gradient: torch.Tensor
    compressed_beta_gradient: torch.Tensor
    metrics: tuple[SparseLinearTIRParityMetricV1, ...]
    projection_receipt: DifferentiableLowerSparseLinearGradientProjectionReceiptV1
    module_receipt: DifferentiableLowerSparseLinearTIRModuleReceiptV1
    launch_receipt: DifferentiableLowerSparseLinearTIRLaunchReceiptV1


def _mapping_by_suffix(
    capture: ProductionDifferentiableReferenceCaptureV1, suffix: str
) -> torch.Tensor:
    matches = [
        snapshot.value
        for name, snapshot in capture.mapping_tensors
        if name.endswith(suffix)
    ]
    if len(matches) != 1:
        raise ValueError(f"sparse-source Linear mapping differs: {suffix}")
    return matches[0]


def _mapping_constants(
    capture: ProductionDifferentiableReferenceCaptureV1,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    alpha_index = _mapping_by_suffix(capture, "/feature_index/0")
    beta_location = _mapping_by_suffix(capture, "/location")
    beta_sign = _mapping_by_suffix(capture, "/sign")
    if (
        alpha_index.dtype != torch.int64
        or alpha_index.shape != (27,)
        or beta_location.dtype != torch.int64
        or beta_location.shape != (6, 1)
        or beta_sign.dtype != torch.float32
        or beta_sign.shape != (6, 1)
        or bool(((beta_sign != -1) & (beta_sign != 1)).any().item())
    ):
        raise ValueError("sparse-source Linear mapping tensor differs")
    alpha = tuple(int(value) for value in alpha_index.tolist())
    locations = tuple(int(value) for value in beta_location[:, 0].tolist())
    signs = tuple(int(value) for value in beta_sign[:, 0].tolist())
    if tuple(sorted(alpha)) != alpha or len(set(alpha)) != len(alpha):
        raise ValueError("sparse-source Linear alpha mapping differs")
    return alpha, locations, signs


def build_b4b2_sparse_linear_template_v1(
    lower_ir: DifferentiableLowerRegionIRV1,
    capture: ProductionDifferentiableReferenceCaptureV1,
    *,
    compute_capability: str,
) -> DifferentiableLowerSparseLinearTIRTemplateV1:
    """Bind compressed mapping constants to the externally approved dense stage."""

    lower_ir.validate()
    capture.validate()
    dense_template: DifferentiableLowerDenseLinearTIRTemplateV1 = (
        build_b4b2_dense_linear_template_v1(
            lower_ir, compute_capability=compute_capability
        )
    )
    alpha, locations, signs = _mapping_constants(capture)
    template = DifferentiableLowerSparseLinearTIRTemplateV1(
        lower_region_ir_hash=lower_ir.stable_hash(),
        dense_template_hash=dense_template.stable_hash(),
        operator_attributes_hash=canonical_tir_hash(dict(lower_ir.operator_attributes)),
        alpha_feature_index_hash=canonical_tir_hash(list(alpha)),
        beta_location_hash=canonical_tir_hash(list(locations)),
        beta_sign_hash=canonical_tir_hash(list(signs)),
        alpha_feature_indices=alpha,
        beta_locations=locations,
        beta_signs=signs,
        compute_capability=compute_capability,
    )
    template.validate()
    return template


def build_b4b2_sparse_linear_schedule_v1(
    template: DifferentiableLowerSparseLinearTIRTemplateV1,
) -> DifferentiableLowerSparseLinearTIRScheduleV1:
    """Return the single B2-2 correctness schedule; it is not timed."""

    schedule = DifferentiableLowerSparseLinearTIRScheduleV1(
        template_hash=template.stable_hash()
    )
    schedule.validate_against(template)
    return schedule


def build_b4b2_sparse_linear_tensors_v1(
    capture: ProductionDifferentiableReferenceCaptureV1,
    template: DifferentiableLowerSparseLinearTIRTemplateV1,
    *,
    device: torch.device,
) -> SparseLinearTIRTensorsV1:
    """Move compressed production values directly to CUDA without dense scatter."""

    capture.validate()
    template.validate()
    alpha, locations, signs = _mapping_constants(capture)
    if (
        alpha != template.alpha_feature_indices
        or locations != template.beta_locations
        or signs != template.beta_signs
    ):
        raise ValueError("sparse-source Linear template mapping differs")
    values = capture.base.value_map

    def cuda_clone(name: str, *, requires_grad: bool = False) -> torch.Tensor:
        tensor = values[name].value.detach().to(device).contiguous().clone()
        tensor.requires_grad_(requires_grad)
        return tensor

    production_alpha = values["production_alpha"].value[0, 0]
    production_beta = values["production_beta"].value
    compressed_alpha = production_alpha.detach().to(device).contiguous().clone()
    compressed_beta = production_beta.detach().to(device).contiguous().clone()
    compressed_alpha.requires_grad_(True)
    compressed_beta.requires_grad_(True)
    if capture.operator_bias is None:
        raise ValueError("sparse-source Linear operator bias is absent")
    tensors = SparseLinearTIRTensorsV1(
        incoming_lower_a=cuda_clone("incoming_lower_a"),
        preactivation_lower=cuda_clone("preactivation_lower"),
        preactivation_upper=cuda_clone("preactivation_upper"),
        compressed_alpha=compressed_alpha,
        compressed_beta=compressed_beta,
        incoming_lower_bias=capture.incoming_lower_bias.value.to(device).contiguous(),
        operator_weight=cuda_clone("operator_weight"),
        operator_bias=capture.operator_bias.value.to(device).contiguous(),
        output_lower_a_gradient=(
            capture.output_lower_a_gradient.value.to(device).contiguous()
        ),
        output_bias_gradient=capture.output_bias_gradient.value.to(device).contiguous(),
    )
    _validate_sparse_linear_tensors(tensors, template)
    return tensors


def _validate_sparse_linear_tensors(
    tensors: SparseLinearTIRTensorsV1,
    template: DifferentiableLowerSparseLinearTIRTemplateV1,
) -> None:
    expected_shapes = {
        "incoming_lower_a": (6, 1, 100),
        "preactivation_lower": (6, 100),
        "preactivation_upper": (6, 100),
        "compressed_alpha": (6, 27),
        "compressed_beta": (6, 1),
        "incoming_lower_bias": (6, 1),
        "operator_weight": (100, 1024),
        "operator_bias": (100,),
        "output_lower_a_gradient": (6, 1, 1024),
        "output_bias_gradient": (6, 1),
    }
    values = tensors.tensor_map
    if tuple(sorted(values)) != SPARSE_LINEAR_INPUT_NAMES:
        raise ValueError("sparse-source Linear tensor inventory differs")
    for name, tensor in values.items():
        expected_requires_grad = name in {"compressed_alpha", "compressed_beta"}
        if (
            tuple(tensor.shape) != expected_shapes[name]
            or tensor.device.type != "cuda"
            or tensor.dtype != torch.float32
            or not tensor.is_contiguous()
            or tensor.requires_grad is not expected_requires_grad
            or not bool(torch.isfinite(tensor).all().item())
        ):
            raise ValueError(f"sparse-source Linear tensor differs: {name}")
    if bool((values["preactivation_lower"] > values["preactivation_upper"]).any()):
        raise ValueError("sparse-source Linear interval differs")
    if bool(
        ((values["compressed_alpha"] < 0) | (values["compressed_alpha"] > 1)).any()
    ):
        raise ValueError("sparse-source Linear alpha range differs")
    if bool((values["compressed_beta"] < 0).any()):
        raise ValueError("sparse-source Linear beta range differs")
    template.validate()


def build_b4b2_sparse_linear_instance_v1(
    template: DifferentiableLowerSparseLinearTIRTemplateV1,
    lower_ir: DifferentiableLowerRegionIRV1,
    lower_instance: DifferentiableLowerRegionInstanceV1,
    capture: ProductionDifferentiableReferenceCaptureV1,
    tensors: SparseLinearTIRTensorsV1,
    *,
    fresh_run_ordinal: int,
) -> DifferentiableLowerSparseLinearTIRInstanceV1:
    """Bind every dynamic tensor while keeping mapping constants in the template."""

    lower_instance.validate_against(lower_ir)
    _validate_sparse_linear_tensors(tensors, template)
    ordinal = tensors.incoming_lower_a.device.index
    if ordinal is None:
        ordinal = torch.cuda.current_device()
    instance = DifferentiableLowerSparseLinearTIRInstanceV1(
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


class DifferentiableLowerSparseLinearModuleCache:
    """Compile cache bound to mapping constants but independent of dynamic values."""

    def __init__(self) -> None:
        self._entries: dict[
            str,
            tuple[
                CompiledDifferentiableLowerSparseLinearTIR,
                DifferentiableLowerSparseLinearTIRModuleReceiptV1,
            ],
        ] = {}

    def get(
        self,
        template: DifferentiableLowerSparseLinearTIRTemplateV1,
        schedule: DifferentiableLowerSparseLinearTIRScheduleV1,
    ) -> tuple[
        CompiledDifferentiableLowerSparseLinearTIR,
        DifferentiableLowerSparseLinearTIRModuleReceiptV1,
        str,
    ]:
        cache_key = (
            DifferentiableLowerSparseLinearTIRModuleReceiptV1.expected_cache_key(
                template, schedule
            )
        )
        existing = self._entries.get(cache_key)
        if existing is not None:
            return (*existing, "hit")
        compiled = compile_sparse_linear_tir(template, schedule)
        receipt = DifferentiableLowerSparseLinearTIRModuleReceiptV1(
            template_hash=template.stable_hash(),
            schedule_hash=schedule.stable_hash(template),
            unscheduled_tir_hash=compiled.unscheduled_tir_hash,
            scheduled_tir_hash=compiled.scheduled_tir_hash,
            device_source_hash=compiled.device_source_hash,
            cache_key=cache_key,
            tvm_version=compiled.tvm_version,
            torch_version=str(torch.__version__),
            exported_symbols=(template.forward_symbol, template.backward_symbol),
            observed_workspace_names=compiled.observed_workspace_names,
            forbidden_workspace_count=compiled.forbidden_workspace_count,
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


class _SparseLinearTIRExecutor:
    def __init__(
        self,
        template: DifferentiableLowerSparseLinearTIRTemplateV1,
        schedule: DifferentiableLowerSparseLinearTIRScheduleV1,
        cache: DifferentiableLowerSparseLinearModuleCache,
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
        self._tensors: Optional[SparseLinearTIRTensorsV1] = None

    @property
    def primed_tensors(self) -> Optional[SparseLinearTIRTensorsV1]:
        return self._tensors

    def prime(self, tensors: SparseLinearTIRTensorsV1) -> None:
        if self._tensors is not None:
            raise RuntimeError("sparse-source Linear executor is already primed")
        _validate_sparse_linear_tensors(tensors, self.template)
        self._tensors = tensors

    def reject_fallback(self, *, eager_backward: bool) -> None:
        self.fallback_count += 1
        self.eager_backward_count += int(eager_backward)
        raise RuntimeError("sparse-source Linear TIR fallback is forbidden")

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
                    raise RuntimeError("sparse-source Linear current stream differs")
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
                    raise RuntimeError("sparse-source Linear DLPack pointer differs")
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
                raise RuntimeError("sparse-source Linear global state drifted")
        return _LaunchObservation(
            stream_id=entry_stream_id,
            tvm_ffi_stream_id=ffi_stream_id,
            pointer_count=pointer_count,
            pointer_exact_count=exact_count,
        )

    def forward(
        self, tensors: SparseLinearTIRTensorsV1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_sparse_linear_tensors(tensors, self.template)
        if self.forward_launch_count != 0:
            raise RuntimeError("sparse-source Linear forward launched more than once")
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
            tensors.compressed_alpha,
            tensors.compressed_beta,
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
            raise RuntimeError("sparse-source Linear backward precedes forward")
        if self.backward_launch_count != 0:
            raise RuntimeError("sparse-source Linear backward launched more than once")
        if (
            output_a_gradient.data_ptr() != tensors.output_lower_a_gradient.data_ptr()
            or output_bias_gradient.data_ptr()
            != tensors.output_bias_gradient.data_ptr()
        ):
            raise ValueError("sparse-source Linear output adjoint differs")
        alpha_gradient = torch.empty_like(tensors.compressed_alpha)
        beta_gradient = torch.empty_like(tensors.compressed_beta)
        sources = (
            tensors.incoming_lower_a,
            tensors.preactivation_lower,
            tensors.preactivation_upper,
            tensors.compressed_alpha,
            tensors.compressed_beta,
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


class _SparseLinearTIRFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        incoming_lower_a: torch.Tensor,
        preactivation_lower: torch.Tensor,
        preactivation_upper: torch.Tensor,
        compressed_alpha: torch.Tensor,
        compressed_beta: torch.Tensor,
        incoming_lower_bias: torch.Tensor,
        operator_weight: torch.Tensor,
        operator_bias: torch.Tensor,
        executor: _SparseLinearTIRExecutor,
    ):
        tensors = executor.primed_tensors
        if tensors is None:
            raise RuntimeError("sparse-source Linear executor is not primed")
        passed = (
            incoming_lower_a,
            preactivation_lower,
            preactivation_upper,
            compressed_alpha,
            compressed_beta,
            incoming_lower_bias,
            operator_weight,
            operator_bias,
        )
        expected = (
            tensors.incoming_lower_a,
            tensors.preactivation_lower,
            tensors.preactivation_upper,
            tensors.compressed_alpha,
            tensors.compressed_beta,
            tensors.incoming_lower_bias,
            tensors.operator_weight,
            tensors.operator_bias,
        )
        if any(
            left.data_ptr() != right.data_ptr() for left, right in zip(passed, expected)
        ):
            raise ValueError("sparse-source Linear autograd input differs")
        ctx.executor = executor
        ctx.set_materialize_grads(False)
        return executor.forward(tensors)

    @staticmethod
    def backward(
        ctx, output_a_gradient: torch.Tensor, output_bias_gradient: torch.Tensor
    ):
        if torch.is_grad_enabled():
            raise RuntimeError(
                "sparse-source Linear higher-order gradients unsupported"
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
        )


def _metric(
    name: str, candidate: torch.Tensor, reference: torch.Tensor
) -> SparseLinearTIRParityMetricV1:
    candidate_cpu = candidate.detach().cpu().contiguous()
    reference_cpu = reference.detach().cpu().contiguous()
    difference = (candidate_cpu - reference_cpu).abs()
    metric = SparseLinearTIRParityMetricV1(
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


def _reference_compressed_gradients(
    reference: DifferentiableLowerReferenceResultV1,
    template: DifferentiableLowerSparseLinearTIRTemplateV1,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    native_alpha = reference.native_alpha_gradient
    native_beta = reference.native_beta_gradient
    if native_beta is None:
        raise RuntimeError("sparse-source Linear S beta gradient is absent")
    alpha_indices = torch.tensor(template.alpha_feature_indices, dtype=torch.int64)
    alpha = native_alpha[:, alpha_indices].contiguous()
    beta = torch.stack(
        [
            native_beta[domain, location]
            for domain, location in enumerate(template.beta_locations)
        ]
    ).reshape(6, 1)
    alpha_mask = torch.ones_like(native_alpha, dtype=torch.bool)
    alpha_mask[:, alpha_indices] = False
    beta_mask = torch.ones_like(native_beta, dtype=torch.bool)
    for domain, location in enumerate(template.beta_locations):
        beta_mask[domain, location] = False
    unowned_zero = bool(
        torch.count_nonzero(native_alpha[alpha_mask]).item() == 0
        and torch.count_nonzero(native_beta[beta_mask]).item() == 0
    )
    return alpha, beta, unowned_zero


def _projection_receipt(
    template: DifferentiableLowerSparseLinearTIRTemplateV1,
    instance: DifferentiableLowerSparseLinearTIRInstanceV1,
    reference: DifferentiableLowerReferenceResultV1,
    candidate_alpha: torch.Tensor,
    candidate_beta: torch.Tensor,
) -> DifferentiableLowerSparseLinearGradientProjectionReceiptV1:
    native_alpha = reference.native_alpha_gradient.contiguous()
    native_beta = cast(torch.Tensor, reference.native_beta_gradient).contiguous()
    reference_alpha, reference_beta, unowned_zero = _reference_compressed_gradients(
        reference, template
    )
    candidate_alpha_cpu = candidate_alpha.detach().cpu().contiguous()
    candidate_beta_cpu = candidate_beta.detach().cpu().contiguous()
    projected_alpha = torch.zeros_like(native_alpha)
    projected_alpha[:, torch.tensor(template.alpha_feature_indices)] = (
        candidate_alpha_cpu
    )
    projected_beta = torch.zeros_like(native_beta)
    for domain, location in enumerate(template.beta_locations):
        projected_beta[domain, location] = candidate_beta_cpu[domain, 0]
    alpha_passed = bool(
        torch.allclose(
            candidate_alpha_cpu,
            reference_alpha,
            atol=B4B1_REFERENCE_ATOL,
            rtol=B4B1_REFERENCE_RTOL,
        )
    )
    beta_passed = bool(
        torch.allclose(
            candidate_beta_cpu,
            reference_beta,
            atol=B4B1_REFERENCE_ATOL,
            rtol=B4B1_REFERENCE_RTOL,
        )
    )
    sign_exact = bool(
        torch.equal(torch.sign(candidate_alpha_cpu), torch.sign(reference_alpha))
        and torch.equal(torch.sign(candidate_beta_cpu), torch.sign(reference_beta))
    )
    receipt = DifferentiableLowerSparseLinearGradientProjectionReceiptV1(
        template_hash=template.stable_hash(),
        instance_hash=instance.stable_hash(template),
        reference_native_alpha_gradient_hash=production_tensor_sha256(native_alpha),
        reference_native_beta_gradient_hash=production_tensor_sha256(native_beta),
        reference_compressed_alpha_gradient_hash=production_tensor_sha256(
            reference_alpha
        ),
        reference_compressed_beta_gradient_hash=production_tensor_sha256(
            reference_beta
        ),
        candidate_compressed_alpha_gradient_hash=production_tensor_sha256(
            candidate_alpha_cpu
        ),
        candidate_compressed_beta_gradient_hash=production_tensor_sha256(
            candidate_beta_cpu
        ),
        projected_native_alpha_gradient_hash=production_tensor_sha256(projected_alpha),
        projected_native_beta_gradient_hash=production_tensor_sha256(projected_beta),
        alpha_owned_element_count=6 * 27,
        beta_owned_element_count=6,
        alpha_mapping_exact=bool(
            torch.equal(
                projected_alpha[:, torch.tensor(template.alpha_feature_indices)],
                candidate_alpha_cpu,
            )
        ),
        beta_mapping_exact=all(
            projected_beta[domain, location].item()
            == candidate_beta_cpu[domain, 0].item()
            for domain, location in enumerate(template.beta_locations)
        ),
        alpha_numerical_passed=alpha_passed,
        beta_numerical_passed=beta_passed,
        nonzero_sign_exact=sign_exact,
        unowned_native_zero_exact=unowned_zero,
    )
    receipt.validate_against(template, instance)
    return receipt


def run_b4b2_sparse_linear_tir_v1(
    capture: ProductionDifferentiableReferenceCaptureV1,
    *,
    fresh_run_ordinal: int,
    cache: Optional[DifferentiableLowerSparseLinearModuleCache] = None,
) -> SparseLinearTIRCandidateResultV1:
    """Execute compressed S values directly and compare the B4-B1 oracle."""

    lower_ir = build_b4b1_differentiable_lower_ir_v1(capture)
    lower_instance = build_b4b1_differentiable_lower_instance_v1(capture, lower_ir)
    major, minor = torch.cuda.get_device_capability()
    template = build_b4b2_sparse_linear_template_v1(
        lower_ir, capture, compute_capability=f"sm_{major}{minor}"
    )
    schedule = build_b4b2_sparse_linear_schedule_v1(template)
    tensors = build_b4b2_sparse_linear_tensors_v1(
        capture, template, device=torch.device("cuda:0")
    )
    instance = build_b4b2_sparse_linear_instance_v1(
        template,
        lower_ir,
        lower_instance,
        capture,
        tensors,
        fresh_run_ordinal=fresh_run_ordinal,
    )
    reference = run_b4b1_pytorch_reference_v1(capture, lower_ir, lower_instance)
    executor = _SparseLinearTIRExecutor(
        template,
        schedule,
        cache or DifferentiableLowerSparseLinearModuleCache(),
    )
    executor.prime(tensors)
    source_hashes_before = (
        production_tensor_sha256(tensors.compressed_alpha),
        production_tensor_sha256(tensors.compressed_beta),
        production_tensor_sha256(tensors.incoming_lower_a),
    )
    output_a, output_bias = _SparseLinearTIRFunction.apply(
        tensors.incoming_lower_a,
        tensors.preactivation_lower,
        tensors.preactivation_upper,
        tensors.compressed_alpha,
        tensors.compressed_beta,
        tensors.incoming_lower_bias,
        tensors.operator_weight,
        tensors.operator_bias,
        executor,
    )
    alpha_gradient, beta_gradient = torch.autograd.grad(
        (output_a, output_bias),
        (tensors.compressed_alpha, tensors.compressed_beta),
        grad_outputs=(tensors.output_lower_a_gradient, tensors.output_bias_gradient),
        create_graph=False,
        retain_graph=False,
    )
    source_hashes_after = (
        production_tensor_sha256(tensors.compressed_alpha),
        production_tensor_sha256(tensors.compressed_beta),
        production_tensor_sha256(tensors.incoming_lower_a),
    )
    if source_hashes_before != source_hashes_after:
        raise RuntimeError("sparse-source Linear TIR mutated an input")
    reference_alpha, reference_beta, _unowned_zero = _reference_compressed_gradients(
        reference, template
    )
    candidates = {
        "output_lower_a": output_a,
        "output_bias": output_bias,
        "compressed_alpha_gradient": alpha_gradient,
        "compressed_beta_gradient": beta_gradient,
    }
    references = {
        "output_lower_a": reference.output_lower_a,
        "output_bias": reference.output_bias,
        "compressed_alpha_gradient": reference_alpha,
        "compressed_beta_gradient": reference_beta,
    }
    metrics = tuple(
        _metric(name, candidates[name], references[name])
        for name in SPARSE_LINEAR_OUTPUT_NAMES
    )
    semantic_passed = all(metric.allclose and metric.sign_exact for metric in metrics)
    projection = _projection_receipt(
        template, instance, reference, alpha_gradient, beta_gradient
    )
    forward = executor.forward_observation
    backward = executor.backward_observation
    if forward is None or backward is None:
        raise RuntimeError("sparse-source Linear launch inventory is incomplete")
    if (
        forward.stream_id != backward.stream_id
        or forward.tvm_ffi_stream_id != backward.tvm_ffi_stream_id
    ):
        raise RuntimeError("sparse-source Linear forward/backward stream differs")
    launch = DifferentiableLowerSparseLinearTIRLaunchReceiptV1(
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
    return SparseLinearTIRCandidateResultV1(
        output_lower_a=output_a,
        output_bias=output_bias,
        compressed_alpha_gradient=alpha_gradient,
        compressed_beta_gradient=beta_gradient,
        metrics=metrics,
        projection_receipt=projection,
        module_receipt=executor.module_receipt,
        launch_receipt=launch,
    )


__all__ = [
    "DifferentiableLowerSparseLinearModuleCache",
    "SparseLinearTIRCandidateResultV1",
    "SparseLinearTIRParityMetricV1",
    "SparseLinearTIRTensorsV1",
    "build_b4b2_sparse_linear_instance_v1",
    "build_b4b2_sparse_linear_schedule_v1",
    "build_b4b2_sparse_linear_template_v1",
    "build_b4b2_sparse_linear_tensors_v1",
    "run_b4b2_sparse_linear_tir_v1",
]
