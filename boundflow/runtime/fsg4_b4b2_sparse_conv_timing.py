"""B4-B2 B2-5 wrapper-inclusive sparse Conv microbenchmark primitives."""

# pylint: disable=protected-access,too-many-locals,too-many-instance-attributes
# pylint: disable=missing-function-docstring,missing-class-docstring,not-callable

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Callable

import torch
import torch.nn.functional as torch_functional

from boundflow.ir.differentiable_lower_sparse_conv_tir import (
    DifferentiableLowerSparseConvTIRTemplateV1,
)

from .fsg4_b4b1_reference_capture import ProductionDifferentiableReferenceCaptureV1
from .fsg4_b4b2_sparse_conv_tir import (
    DifferentiableLowerSparseConvModuleCache,
    SparseConvTIRTensorsV1,
    _SparseConvTIRExecutor,
    _SparseConvTIRFunction,
    build_b4b2_sparse_conv_instance_v1,
    build_b4b2_sparse_conv_ledger_v1,
    build_b4b2_sparse_conv_schedules_v1,
    build_b4b2_sparse_conv_template_v1,
    build_b4b2_sparse_conv_tensors_v1,
)
from .fsg4_b4b1_pytorch_reference import (
    B4B1_REFERENCE_ATOL,
    B4B1_REFERENCE_RTOL,
    build_b4b1_differentiable_lower_instance_v1,
    build_b4b1_differentiable_lower_ir_v1,
)

B2_5_WARMUP_COUNT = 10
B2_5_PAIR_COUNT = 30
B2_5_SPEEDUP_GATE = 1.05
B2_5_WORST_WORKER_GATE = 0.98
B2_5_MEMORY_RATIO_GATE = 1.05
B2_5_BOOTSTRAP_SAMPLE_COUNT = 10_000
B2_5_BOOTSTRAP_SEED = 20260824


@dataclass(frozen=True)
class SparseConvExecutionV1:
    output_lower_a: torch.Tensor
    output_bias: torch.Tensor
    compressed_alpha_gradient: torch.Tensor
    incoming_lower_a_gradient: torch.Tensor

    @property
    def tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.output_lower_a,
            self.output_bias,
            self.compressed_alpha_gradient,
            self.incoming_lower_a_gradient,
        )


@dataclass(frozen=True)
class SparseConvTimingParityV1:
    maximum_absolute_difference: float
    allclose: bool
    sign_exact: bool
    element_count: int

    def to_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class SparseConvKernelInventoryV1:
    kernel_names: tuple[str, ...]
    forward_kernel_count: int
    backward_kernel_count: int
    shared_memory_token_count: int
    vector_token_count: int
    half_token_count: int

    @property
    def total_kernel_count(self) -> int:
        return len(self.kernel_names)

    def to_dict(self) -> dict[str, object]:
        return {
            "kernel_names": list(self.kernel_names),
            "forward_kernel_count": self.forward_kernel_count,
            "backward_kernel_count": self.backward_kernel_count,
            "total_kernel_count": self.total_kernel_count,
            "shared_memory_token_count": self.shared_memory_token_count,
            "vector_token_count": self.vector_token_count,
            "half_token_count": self.half_token_count,
        }


class PreparedSparseConvTimingV1:
    """One immutable production P-anchor prepared for repeated fair calls."""

    def __init__(
        self,
        capture: ProductionDifferentiableReferenceCaptureV1,
        *,
        candidate_ordinal: int,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("B4-B2 B2-5 timing requires CUDA")
        self.capture = capture
        self.lower_ir = build_b4b1_differentiable_lower_ir_v1(capture)
        self.lower_instance = build_b4b1_differentiable_lower_instance_v1(
            capture, self.lower_ir
        )
        major, minor = torch.cuda.get_device_capability()
        self.template = build_b4b2_sparse_conv_template_v1(
            self.lower_ir, capture, compute_capability=f"sm_{major}{minor}"
        )
        self.schedules = build_b4b2_sparse_conv_schedules_v1(self.template)
        self.ledger = build_b4b2_sparse_conv_ledger_v1(self.template, self.schedules)
        if candidate_ordinal < 0 or candidate_ordinal >= len(self.schedules):
            raise ValueError("B4-B2 B2-5 candidate ordinal differs")
        self.schedule = self.schedules[candidate_ordinal]
        self.tensors = build_b4b2_sparse_conv_tensors_v1(
            capture, self.template, device=torch.device("cuda:0")
        )
        self.instance = build_b4b2_sparse_conv_instance_v1(
            self.template,
            self.lower_ir,
            self.lower_instance,
            capture,
            self.tensors,
            fresh_run_ordinal=0,
        )
        self.cache = DifferentiableLowerSparseConvModuleCache()
        compiled, receipt, cache_event = self.cache.get(self.template, self.schedule)
        if cache_event != "miss":
            raise RuntimeError("B4-B2 B2-5 initial module cache event differs")
        self.compiled = compiled
        self.module_receipt = receipt
        _compiled_again, receipt_again, cache_event_again = self.cache.get(
            self.template, self.schedule
        )
        if cache_event_again != "hit" or receipt_again.stable_hash(
            self.template, self.schedule
        ) != receipt.stable_hash(self.template, self.schedule):
            raise RuntimeError("B4-B2 B2-5 warm module cache differs")
        self.kernel_inventory = kernel_inventory_from_compiled_v1(compiled)

    @property
    def candidate_ordinal(self) -> int:
        return self.schedule.candidate_ordinal

    def baseline_once(self) -> SparseConvExecutionV1:
        return execute_sparse_conv_pytorch_baseline_v1(self.tensors, self.template)

    def candidate_once(self) -> SparseConvExecutionV1:
        executor = _SparseConvTIRExecutor(self.template, self.schedule, self.cache)
        if executor.cache_event != "hit":
            raise RuntimeError(
                "B4-B2 B2-5 candidate timing includes compile/cache miss"
            )
        executor.prime(self.tensors)
        output_a, output_bias = _SparseConvTIRFunction.apply(
            self.tensors.incoming_lower_a,
            self.tensors.preactivation_lower,
            self.tensors.preactivation_upper,
            self.tensors.compressed_alpha,
            self.tensors.incoming_lower_bias,
            self.tensors.operator_weight,
            self.tensors.operator_bias,
            executor,
        )
        incoming_gradient, alpha_gradient = torch.autograd.grad(
            (output_a, output_bias),
            (self.tensors.incoming_lower_a, self.tensors.compressed_alpha),
            grad_outputs=(
                self.tensors.output_lower_a_gradient,
                self.tensors.output_bias_gradient,
            ),
        )
        if (
            executor.forward_launch_count != 1
            or executor.backward_launch_count != 1
            or executor.fallback_count != 0
            or executor.eager_backward_count != 0
        ):
            raise RuntimeError("B4-B2 B2-5 candidate module-call receipt differs")
        return SparseConvExecutionV1(
            output_lower_a=output_a,
            output_bias=output_bias,
            compressed_alpha_gradient=alpha_gradient,
            incoming_lower_a_gradient=incoming_gradient,
        )


def _dense_alpha(
    tensors: SparseConvTIRTensorsV1,
    template: DifferentiableLowerSparseConvTIRTemplateV1,
) -> torch.Tensor:
    dense = torch.zeros(
        (template.domain_count, template.channels, template.height, template.width),
        dtype=tensors.compressed_alpha.dtype,
        device=tensors.compressed_alpha.device,
    )
    coordinates = template.alpha_coordinates
    channel = torch.tensor(
        [row[0] for row in coordinates], dtype=torch.int64, device=dense.device
    )
    height = torch.tensor(
        [row[1] for row in coordinates], dtype=torch.int64, device=dense.device
    )
    width = torch.tensor(
        [row[2] for row in coordinates], dtype=torch.int64, device=dense.device
    )
    dense[:, channel, height, width] = tensors.compressed_alpha
    return dense


def execute_sparse_conv_pytorch_baseline_v1(
    tensors: SparseConvTIRTensorsV1,
    template: DifferentiableLowerSparseConvTIRTemplateV1,
) -> SparseConvExecutionV1:
    """Public-PyTorch sparse reconstruction + lower region + first-order VJP."""

    incoming = tensors.incoming_lower_a
    lower = tensors.preactivation_lower
    upper = tensors.preactivation_upper
    dense_alpha = _dense_alpha(tensors, template)
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
    selected_slope = torch.where(
        incoming >= zero,
        lower_slope.unsqueeze(1),
        upper_slope.unsqueeze(1),
    )
    upper_intercept = torch.where(
        ambiguous, -lower * upper_slope, torch.zeros_like(lower)
    )
    selected_intercept = torch.where(
        incoming >= zero,
        torch.zeros_like(incoming),
        upper_intercept.unsqueeze(1),
    )
    relu_lower_a = incoming * selected_slope
    batch_spec = template.domain_count * template.spec_count
    output_a = torch_functional.conv_transpose2d(
        relu_lower_a.reshape(
            batch_spec, template.channels, template.height, template.width
        ),
        tensors.operator_weight,
        bias=None,
        stride=template.stride,
        padding=template.padding,
        output_padding=template.output_padding,
        groups=template.groups,
        dilation=template.dilation,
    ).reshape_as(incoming)
    output_bias = tensors.incoming_lower_bias + (
        incoming * selected_intercept
        + relu_lower_a * tensors.operator_bias.reshape(1, 1, -1, 1, 1)
    ).flatten(2).sum(2)
    local_vjp = (output_a * tensors.output_lower_a_gradient).sum() + (
        output_bias * tensors.output_bias_gradient
    ).sum()
    incoming_gradient, alpha_gradient = torch.autograd.grad(
        local_vjp, (incoming, tensors.compressed_alpha)
    )
    return SparseConvExecutionV1(
        output_lower_a=output_a,
        output_bias=output_bias,
        compressed_alpha_gradient=alpha_gradient,
        incoming_lower_a_gradient=incoming_gradient,
    )


def compare_sparse_conv_executions_v1(
    baseline: SparseConvExecutionV1, candidate: SparseConvExecutionV1
) -> SparseConvTimingParityV1:
    maximum = 0.0
    count = 0
    allclose = True
    sign_exact = True
    for reference, observed in zip(baseline.tensors, candidate.tensors):
        if reference.shape != observed.shape or reference.dtype != observed.dtype:
            raise ValueError("B4-B2 B2-5 parity tensor schema differs")
        maximum = max(maximum, float((reference - observed).abs().max().item()))
        count += reference.numel()
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
    return SparseConvTimingParityV1(maximum, allclose, sign_exact, count)


def _unique_kernel_names(source: str) -> tuple[str, ...]:
    pattern = re.compile(
        r'extern "C" __global__ void(?: __launch_bounds__\([^)]*\))? ([A-Za-z0-9_]+)\('
    )
    return tuple(sorted(set(pattern.findall(source))))


def kernel_inventory_from_compiled_v1(compiled) -> SparseConvKernelInventoryV1:
    source = "\n".join(
        module.inspect_source() for module in compiled.executable.mod.imports
    )
    names = _unique_kernel_names(source)
    forward = sum("_forward_kernel" in name for name in names)
    backward = sum("_backward_kernel" in name for name in names)
    if not names or forward + backward != len(names):
        raise RuntimeError("B4-B2 B2-5 CUDA kernel inventory differs")
    return SparseConvKernelInventoryV1(
        kernel_names=names,
        forward_kernel_count=forward,
        backward_kernel_count=backward,
        shared_memory_token_count=source.count("__shared__"),
        vector_token_count=sum(source.count(token) for token in ("float2", "float4")),
        half_token_count=source.count("half"),
    )


def cuda_event_call_ms_v1(call: Callable[[], SparseConvExecutionV1]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    finish = torch.cuda.Event(enable_timing=True)
    start.record()
    result = call()
    finish.record()
    finish.synchronize()
    if not all(bool(torch.isfinite(tensor).all().item()) for tensor in result.tensors):
        raise RuntimeError("B4-B2 B2-5 timed output is nonfinite")
    elapsed = float(start.elapsed_time(finish))
    if not math.isfinite(elapsed) or elapsed <= 0.0:
        raise RuntimeError("B4-B2 B2-5 CUDA event duration differs")
    return elapsed


def measure_peak_memory_v1(
    call: Callable[[], SparseConvExecutionV1],
) -> tuple[int, int]:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    result = call()
    torch.cuda.synchronize()
    if not result.tensors:
        raise RuntimeError("B4-B2 B2-5 memory call produced no tensors")
    return torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()


__all__ = [
    "B2_5_BOOTSTRAP_SAMPLE_COUNT",
    "B2_5_BOOTSTRAP_SEED",
    "B2_5_MEMORY_RATIO_GATE",
    "B2_5_PAIR_COUNT",
    "B2_5_SPEEDUP_GATE",
    "B2_5_WARMUP_COUNT",
    "B2_5_WORST_WORKER_GATE",
    "PreparedSparseConvTimingV1",
    "SparseConvExecutionV1",
    "SparseConvKernelInventoryV1",
    "SparseConvTimingParityV1",
    "compare_sparse_conv_executions_v1",
    "cuda_event_call_ms_v1",
    "execute_sparse_conv_pytorch_baseline_v1",
    "kernel_inventory_from_compiled_v1",
    "measure_peak_memory_v1",
]
