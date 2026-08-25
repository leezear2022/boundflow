"""R3-3 active-beta S-anchor isolated wrapper timing primitives."""

# pylint: disable=protected-access,too-many-locals,too-many-instance-attributes
# pylint: disable=missing-function-docstring,not-callable

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from boundflow.ir.differentiable_lower_sparse_linear_tir import (
    DifferentiableLowerSparseLinearTIRTemplateV1,
)

from .fsg4_b4b1_pytorch_reference import (
    B4B1_REFERENCE_ATOL,
    B4B1_REFERENCE_RTOL,
    build_b4b1_differentiable_lower_instance_v1,
    build_b4b1_differentiable_lower_ir_v1,
)
from .fsg4_b4b1_reference_capture import ProductionDifferentiableReferenceCaptureV1
from .fsg4_b4b2_sparse_linear_tir import (
    DifferentiableLowerSparseLinearModuleCache,
    SparseLinearTIRTensorsV1,
    _SparseLinearTIRExecutor,
    _SparseLinearTIRFunction,
    build_b4b2_sparse_linear_instance_v1,
    build_b4b2_sparse_linear_schedule_v1,
    build_b4b2_sparse_linear_template_v1,
    build_b4b2_sparse_linear_tensors_v1,
)

R3_3_TIMING_WARMUP_COUNT = 10
R3_3_TIMING_PAIR_COUNT = 30
R3_3_TIMING_SPEEDUP_GATE = 1.05
R3_3_TIMING_BOOTSTRAP_LOWER_GATE = 1.0
R3_3_TIMING_WORST_GATE = 0.98
R3_3_TIMING_MEMORY_RATIO_GATE = 1.05
R3_3_TIMING_BOOTSTRAP_SAMPLE_COUNT = 10_000
R3_3_TIMING_BOOTSTRAP_SEED = 20260826


@dataclass(frozen=True)
class R3ActiveBetaExecutionV1:
    """Four observable values from one complete active-beta local wrapper."""

    output_lower_a: torch.Tensor
    output_bias: torch.Tensor
    compressed_alpha_gradient: torch.Tensor
    compressed_beta_gradient: torch.Tensor

    @property
    def tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.output_lower_a,
            self.output_bias,
            self.compressed_alpha_gradient,
            self.compressed_beta_gradient,
        )


@dataclass(frozen=True)
class R3ActiveBetaTimingParityV1:
    """Direct public-PyTorch versus TIR wrapper parity."""

    maximum_absolute_difference: float
    allclose: bool
    sign_exact: bool
    element_count: int

    def to_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class R3ActiveBetaMemoryV1:
    """Absolute and incremental CUDA allocation observation for one call."""

    base_allocated_bytes: int
    peak_allocated_bytes: int
    incremental_allocated_bytes: int
    peak_reserved_bytes: int

    def to_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


def _dense_sources(
    tensors: SparseLinearTIRTensorsV1,
    template: DifferentiableLowerSparseLinearTIRTemplateV1,
) -> tuple[torch.Tensor, torch.Tensor]:
    dense_alpha = torch.zeros(
        (template.domain_count, template.current_features),
        dtype=tensors.compressed_alpha.dtype,
        device=tensors.compressed_alpha.device,
    )
    alpha_indices = torch.tensor(
        template.alpha_feature_indices,
        dtype=torch.int64,
        device=dense_alpha.device,
    )
    dense_alpha[:, alpha_indices] = tensors.compressed_alpha
    dense_beta = torch.zeros_like(dense_alpha)
    split = torch.zeros_like(dense_alpha)
    for domain, (location, sign) in enumerate(
        zip(template.beta_locations, template.beta_signs)
    ):
        dense_beta[domain, location] = tensors.compressed_beta[domain, 0]
        split[domain, location] = sign
    return dense_alpha, -dense_beta * split


def execute_r3_active_beta_pytorch_baseline_v1(
    tensors: SparseLinearTIRTensorsV1,
    template: DifferentiableLowerSparseLinearTIRTemplateV1,
) -> R3ActiveBetaExecutionV1:
    """Execute dense reconstruction, lower Linear and first-order VJP on CUDA."""

    dense_alpha, beta_pre_add = _dense_sources(tensors, template)
    incoming = tensors.incoming_lower_a
    lower = tensors.preactivation_lower
    upper = tensors.preactivation_upper
    zero = torch.zeros((), dtype=lower.dtype, device=lower.device)
    positive = lower >= zero
    negative = upper <= zero
    ambiguous = (~positive) & (~negative)
    denominator = (upper - lower).clamp_min(torch.finfo(lower.dtype).eps)
    upper_slope = torch.where(
        positive,
        torch.ones_like(lower),
        torch.where(negative, torch.zeros_like(lower), upper / denominator),
    )
    lower_slope = torch.where(
        ambiguous,
        dense_alpha.clamp(0.0, 1.0),
        torch.where(positive, torch.ones_like(lower), torch.zeros_like(lower)),
    )
    upper_intercept = torch.where(
        ambiguous, -lower * upper_slope, torch.zeros_like(lower)
    )
    selected_slope = torch.where(
        incoming >= zero,
        lower_slope.unsqueeze(1),
        upper_slope.unsqueeze(1),
    )
    selected_intercept = torch.where(
        incoming >= zero,
        torch.zeros_like(upper_intercept).unsqueeze(1),
        upper_intercept.unsqueeze(1),
    )
    relu_lower_a = incoming * selected_slope + beta_pre_add.unsqueeze(1)
    relu_bias = tensors.incoming_lower_bias + (incoming * selected_intercept).sum(dim=2)
    output_lower_a = torch.matmul(relu_lower_a, tensors.operator_weight)
    output_bias = relu_bias + (
        relu_lower_a * tensors.operator_bias.reshape(1, 1, -1)
    ).sum(dim=2)
    alpha_gradient, beta_gradient = torch.autograd.grad(
        (output_lower_a, output_bias),
        (tensors.compressed_alpha, tensors.compressed_beta),
        grad_outputs=(
            tensors.output_lower_a_gradient,
            tensors.output_bias_gradient,
        ),
        create_graph=False,
        retain_graph=False,
    )
    return R3ActiveBetaExecutionV1(
        output_lower_a=output_lower_a,
        output_bias=output_bias,
        compressed_alpha_gradient=alpha_gradient,
        compressed_beta_gradient=beta_gradient,
    )


class PreparedR3ActiveBetaTimingV1:
    """One immutable S-anchor with a precompiled cache-hit candidate module."""

    def __init__(self, capture: ProductionDifferentiableReferenceCaptureV1) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("R3-3 active-beta timing requires CUDA")
        self.capture = capture
        self.lower_ir = build_b4b1_differentiable_lower_ir_v1(capture)
        self.lower_instance = build_b4b1_differentiable_lower_instance_v1(
            capture, self.lower_ir
        )
        major, minor = torch.cuda.get_device_capability()
        self.template = build_b4b2_sparse_linear_template_v1(
            self.lower_ir, capture, compute_capability=f"sm_{major}{minor}"
        )
        self.schedule = build_b4b2_sparse_linear_schedule_v1(self.template)
        self.tensors = build_b4b2_sparse_linear_tensors_v1(
            capture, self.template, device=torch.device("cuda:0")
        )
        self.instance = build_b4b2_sparse_linear_instance_v1(
            self.template,
            self.lower_ir,
            self.lower_instance,
            capture,
            self.tensors,
            fresh_run_ordinal=0,
        )
        self.cache = DifferentiableLowerSparseLinearModuleCache()
        _compiled, receipt, event = self.cache.get(self.template, self.schedule)
        if event != "miss":
            raise RuntimeError("R3-3 initial module cache event differs")
        self.module_receipt = receipt
        _compiled_again, receipt_again, event_again = self.cache.get(
            self.template, self.schedule
        )
        if event_again != "hit" or receipt_again.stable_hash(
            self.template, self.schedule
        ) != receipt.stable_hash(self.template, self.schedule):
            raise RuntimeError("R3-3 warm module cache differs")

    def baseline_once(self) -> R3ActiveBetaExecutionV1:
        return execute_r3_active_beta_pytorch_baseline_v1(self.tensors, self.template)

    def candidate_once(self) -> R3ActiveBetaExecutionV1:
        executor = _SparseLinearTIRExecutor(self.template, self.schedule, self.cache)
        if executor.cache_event != "hit":
            raise RuntimeError("R3-3 candidate timing includes cache miss")
        executor.prime(self.tensors)
        output_a, output_bias = _SparseLinearTIRFunction.apply(
            self.tensors.incoming_lower_a,
            self.tensors.preactivation_lower,
            self.tensors.preactivation_upper,
            self.tensors.compressed_alpha,
            self.tensors.compressed_beta,
            self.tensors.incoming_lower_bias,
            self.tensors.operator_weight,
            self.tensors.operator_bias,
            executor,
        )
        alpha_gradient, beta_gradient = torch.autograd.grad(
            (output_a, output_bias),
            (self.tensors.compressed_alpha, self.tensors.compressed_beta),
            grad_outputs=(
                self.tensors.output_lower_a_gradient,
                self.tensors.output_bias_gradient,
            ),
            create_graph=False,
            retain_graph=False,
        )
        if (
            executor.forward_launch_count != 1
            or executor.backward_launch_count != 1
            or executor.fallback_count != 0
            or executor.eager_backward_count != 0
        ):
            raise RuntimeError("R3-3 candidate module-call receipt differs")
        return R3ActiveBetaExecutionV1(
            output_lower_a=output_a,
            output_bias=output_bias,
            compressed_alpha_gradient=alpha_gradient,
            compressed_beta_gradient=beta_gradient,
        )


def compare_r3_active_beta_executions_v1(
    baseline: R3ActiveBetaExecutionV1, candidate: R3ActiveBetaExecutionV1
) -> R3ActiveBetaTimingParityV1:
    maximum = 0.0
    allclose = True
    sign_exact = True
    elements = 0
    for reference, observed in zip(baseline.tensors, candidate.tensors):
        maximum = max(maximum, float((reference - observed).abs().max().item()))
        allclose = allclose and bool(
            torch.allclose(
                reference,
                observed,
                atol=B4B1_REFERENCE_ATOL,
                rtol=B4B1_REFERENCE_RTOL,
            )
        )
        sign_exact = sign_exact and bool(
            torch.equal(torch.sign(reference), torch.sign(observed))
        )
        elements += reference.numel()
    return R3ActiveBetaTimingParityV1(
        maximum_absolute_difference=maximum,
        allclose=allclose,
        sign_exact=sign_exact,
        element_count=elements,
    )


def cuda_event_wrapper_ms_v1(call: Callable[[], object]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = call()
    end.record()
    end.synchronize()
    del result
    value = float(start.elapsed_time(end))
    if value <= 0.0:
        raise RuntimeError("R3-3 CUDA event timing differs")
    return value


def measure_r3_active_beta_memory_v1(
    call: Callable[[], object],
) -> R3ActiveBetaMemoryV1:
    torch.cuda.synchronize()
    base = int(torch.cuda.memory_allocated())
    torch.cuda.reset_peak_memory_stats()
    result = call()
    torch.cuda.synchronize()
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    del result
    torch.cuda.synchronize()
    return R3ActiveBetaMemoryV1(
        base_allocated_bytes=base,
        peak_allocated_bytes=peak_allocated,
        incremental_allocated_bytes=max(0, peak_allocated - base),
        peak_reserved_bytes=peak_reserved,
    )


__all__ = [
    "PreparedR3ActiveBetaTimingV1",
    "R3ActiveBetaExecutionV1",
    "R3ActiveBetaMemoryV1",
    "R3ActiveBetaTimingParityV1",
    "compare_r3_active_beta_executions_v1",
    "cuda_event_wrapper_ms_v1",
    "execute_r3_active_beta_pytorch_baseline_v1",
    "measure_r3_active_beta_memory_v1",
]
