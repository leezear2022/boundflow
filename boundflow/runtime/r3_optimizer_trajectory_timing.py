"""R3-2B capture-free wrapper-inclusive P-anchor timing runtime."""

# pylint: disable=protected-access,too-many-locals,too-many-statements
# pylint: disable=too-many-instance-attributes,missing-function-docstring
# pylint: disable=import-outside-toplevel,too-many-boolean-expressions

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import torch

from boundflow.backends.tvm.r3_full_lower_forward import (
    R31B1_CONCRETIZE_SYMBOL,
    R31B1_CONV0_SYMBOL,
    R31B1_LINEAR14_SYMBOL,
    R31B1_LINEAR16_SYMBOL,
    R31B1_RESIDUAL11_SYMBOL,
    R31B1_RESIDUAL6_SYMBOL,
    R31B1_SEED_SYMBOL,
)
from boundflow.backends.tvm.r3_p_alpha_vjp import (
    R31B2_CLEAR_SYMBOL,
    R31B2_COMPRESSED_GRADIENT_SYMBOL,
)

from .fsg4_b3_terminal_optimizer_schedule import (
    NativeTerminalOptimizerScheduleV1,
)
from .r3_compiled_p_alpha_vjp import (
    _EXECUTOR_REGISTRY,
    _R31B2CompiledFunction,
    PreparedR31B2CompiledCustomBackwardV1,
)
from .r3_structured_owner_custom_backward import (
    _evaluate_full_region,
    R31FullRegionPlanV1,
)

R32B_TIMING_SCHEMA = "boundflow.r3-2b-wrapper-timing/v1"


@dataclass(frozen=True)
class R32BWrapperResultV1:
    """Terminal state and scalar-only execution counters for one wrapper."""

    terminal_lower: torch.Tensor
    terminal_alpha: torch.Tensor
    evaluation_count: int
    optimizer_mutation_count: int
    scheduler_mutation_count: int
    custom_forward_count: int
    custom_backward_count: int
    fallback_count: int = 0
    eager_candidate_count: int = 0
    native_shadow_count: int = 0
    timing_capture_count: int = 0
    schema_version: str = R32B_TIMING_SCHEMA

    def validate(self, *, candidate: bool) -> None:
        if (
            self.schema_version != R32B_TIMING_SCHEMA
            or tuple(self.terminal_lower.shape) != (6, 1)
            or tuple(self.terminal_alpha.shape) != (2, 1, 6, 86)
            or not bool(torch.isfinite(self.terminal_lower).all().item())
            or not bool(torch.isfinite(self.terminal_alpha).all().item())
            or self.evaluation_count != 10
            or self.optimizer_mutation_count != 9
            or self.scheduler_mutation_count != 9
            or self.custom_forward_count != (10 if candidate else 0)
            or self.custom_backward_count != (10 if candidate else 0)
            or self.fallback_count
            or self.eager_candidate_count
            or self.native_shadow_count
            or self.timing_capture_count
        ):
            raise ValueError("R3-2B wrapper result differs")


class PreparedR32BTimingCandidateV1(PreparedR31B2CompiledCustomBackwardV1):
    """Persistent compiled owner with capture instrumentation removed."""

    def __init__(
        self,
        plan: R31FullRegionPlanV1,
        trace,  # type: ignore[no-untyped-def]
        tensors: tuple[torch.Tensor, ...],
    ) -> None:
        super().__init__(plan, trace, tensors)
        self._tensor_pointers = tuple(value.data_ptr() for value in tensors)
        self._tensor_versions = tuple(value._version for value in tensors)
        self._expected_evaluation = 0
        self._sample_ordinal = 0

    def begin_sample(self) -> None:
        self._expected_evaluation = 0
        self._sample_ordinal += 1

    def begin_evaluation(self, ordinal: int) -> None:
        if ordinal != self._expected_evaluation:
            raise RuntimeError("R3-2B candidate evaluation order differs")
        if any(
            value.data_ptr() != pointer
            or tuple(value.shape) != spec.shape
            or str(value.dtype) != spec.dtype
            or value.device != self.device
            for value, pointer, spec in zip(
                self.tensors, self._tensor_pointers, self.plan.tensor_specs
            )
        ):
            raise ValueError("R3-2B candidate runtime identity differs")
        immutable_versions = tuple(
            value._version
            for index, value in enumerate(self.tensors)
            if index != self.plan.p_alpha_input_ordinal
        )
        expected_immutable = tuple(
            version
            for index, version in enumerate(self._tensor_versions)
            if index != self.plan.p_alpha_input_ordinal
        )
        if immutable_versions != expected_immutable:
            raise ValueError("R3-2B candidate immutable version drifted")
        self.custom_forward_count = 0
        self.custom_backward_count = 0
        self.forward_executor.launch_count = 0
        self.b1_backward_launch_count = 0
        self.b2_launch_count = 0
        self.runtime_dlpack_pointer_count = 0
        self.runtime_dlpack_pointer_exact_count = 0
        self._expected_evaluation += 1

    def _run_forward_fast(self) -> torch.Tensor:
        import tvm_ffi

        current = torch.cuda.current_stream(self.device)
        if int(current.cuda_stream) == int(
            torch.cuda.default_stream(self.device).cuda_stream
        ):
            raise RuntimeError("R3-2B candidate non-default stream is required")
        executor = self.forward_executor
        s0 = executor.scratch_0
        s1 = executor.scratch_1
        bias = executor.bias_accumulator
        with tvm_ffi.use_torch_stream(torch.cuda.stream(current)):
            executor._launch(
                R31B1_SEED_SYMBOL, executor._tensor("objective"), s0[:60], bias
            )
            executor._launch(
                R31B1_LINEAR16_SYMBOL,
                s0[:60],
                executor._tensor("param/linear2.weight"),
                executor._tensor("param/linear2.bias"),
                bias,
                s1[:600],
                bias,
            )
            executor._relu("31", s1[:600], bias, active_beta=True)
            executor._launch(
                R31B1_LINEAR14_SYMBOL,
                s1[:600],
                executor._tensor("param/linear1.weight"),
                executor._tensor("param/linear1.bias"),
                bias,
                s0[:6144],
                bias,
            )
            executor._relu("28", s0[:6144], bias)
            executor._launch(
                R31B1_RESIDUAL11_SYMBOL,
                s0,
                executor._tensor("param/layer1.1.conv2.weight"),
                executor._tensor("param/layer1.1.conv2.bias"),
                executor._tensor("relu/25/lower").reshape(6, 1024),
                executor._tensor("relu/25/upper").reshape(6, 1024),
                executor._tensor("relu/25/alpha"),
                executor.alpha_maps["25"],
                executor._tensor("param/layer1.1.conv1.weight"),
                executor._tensor("param/layer1.1.conv1.bias"),
                bias,
                s1,
            )
            executor._relu("23", s1[:6144], bias)
            executor._launch(
                R31B1_RESIDUAL6_SYMBOL,
                s1,
                executor._tensor("param/layer1.0.conv2.weight"),
                executor._tensor("param/layer1.0.conv2.bias"),
                executor._tensor("relu/19/lower").reshape(6, 1024),
                executor._tensor("relu/19/upper").reshape(6, 1024),
                executor._tensor("relu/19/alpha"),
                executor.alpha_maps["19"],
                executor._tensor("param/layer1.0.conv1.weight"),
                executor._tensor("param/layer1.0.conv1.bias"),
                executor._tensor("param/layer1.0.shortcut.0.weight"),
                executor._tensor("param/layer1.0.shortcut.0.bias"),
                bias,
                s0,
            )
            executor._relu("17", s0[:12288], bias)
            executor._launch(
                R31B1_CONV0_SYMBOL,
                s0[:12288],
                executor._tensor("param/conv1.weight"),
                executor._tensor("param/conv1.bias"),
                bias,
                s1,
                bias,
            )
            executor._launch(
                R31B1_CONCRETIZE_SYMBOL,
                s1,
                executor._tensor("input/lower").reshape(6, 3072),
                executor._tensor("input/upper").reshape(6, 3072),
                bias,
                executor.output,
            )
        return executor.output.reshape(6, 1)

    def forward(self) -> torch.Tensor:
        import tvm_ffi

        if self.custom_forward_count:
            raise RuntimeError("R3-2B candidate forward count differs")
        self.custom_forward_count = 1
        lower = self._run_forward_fast()
        current = torch.cuda.current_stream(self.device)
        with tvm_ffi.use_torch_stream(torch.cuda.stream(current)):
            self._launch_b2(
                R31B2_CLEAR_SYMBOL,
                self.forward_executor.scratch_0,
                self.forward_executor.scratch_1,
            )
        return lower

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        import tvm_ffi

        if self.custom_backward_count:
            raise RuntimeError("R3-2B candidate backward count differs")
        if (
            tuple(grad_output.shape) != (6, 1)
            or grad_output.dtype != torch.float32
            or grad_output.device != self.device
            or not grad_output.is_contiguous()
        ):
            raise ValueError("R3-2B candidate upstream gradient differs")
        self.custom_backward_count = 1
        current = torch.cuda.current_stream(self.device)
        if int(current.cuda_stream) == int(
            torch.cuda.default_stream(self.device).cuda_stream
        ):
            raise RuntimeError("R3-2B candidate non-default stream is required")
        runtime_view = self._view(grad_output.reshape(6))
        with tvm_ffi.use_torch_stream(torch.cuda.stream(current)):
            s0 = self.forward_executor.scratch_0
            s1 = self.forward_executor.scratch_1
            bias = self.forward_executor.bias_accumulator
            self._coefficient_sign_pass(s0, s1, bias)
            self._effective_value_pass(s0, s1)
            self._recompute_a26(s0, s1, bias)
            self.compiled.executable[R31B2_COMPRESSED_GRADIENT_SYMBOL](
                self._view(s1[:6144]),
                self._view(self.pre25_value),
                self._view(self._tensor("relu/25/lower").reshape(6, 1024)),
                self._view(self._tensor("relu/25/upper").reshape(6, 1024)),
                self._view(self.p_indices),
                runtime_view,
                self._view(self.gradient),
            )
            self.b2_launch_count += 1
        return self.gradient.reshape(2, 1, 6, 86)


def _candidate_evaluate(
    prepared: PreparedR32BTimingCandidateV1,
    ordinal: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    prepared.begin_evaluation(ordinal)
    key = hashlib.sha256(
        f"{id(prepared)}:{prepared._sample_ordinal}:{ordinal}".encode()
    ).hexdigest()
    if key in _EXECUTOR_REGISTRY:
        raise RuntimeError("R3-2B execution key repeats")
    _EXECUTOR_REGISTRY[key] = prepared
    try:
        lower = _R31B2CompiledFunction.apply(key, *prepared.tensors)
        gradient = torch.autograd.grad(
            lower,
            prepared.tensors[prepared.plan.p_alpha_input_ordinal],
            grad_outputs=prepared.upstream_gradient,
        )[0]
    finally:
        _EXECUTOR_REGISTRY.pop(key, None)
    return lower, gradient


def execute_r32b_wrapper_v1(
    plan: R31FullRegionPlanV1,
    tensors: tuple[torch.Tensor, ...],
    schedule: NativeTerminalOptimizerScheduleV1,
    *,
    candidate: PreparedR32BTimingCandidateV1 | None,
) -> R32BWrapperResultV1:
    """Run one complete 10/9 wrapper without correctness-capture synchronization."""

    plan.validate()
    schedule.validate()
    p_ordinal = plan.p_alpha_input_ordinal
    alpha = tensors[p_ordinal]
    if not alpha.is_leaf or not alpha.requires_grad:
        raise ValueError("R3-2B P-alpha owner differs")
    if candidate is not None:
        if (
            candidate.tensors is not tensors
            or candidate.plan.stable_hash() != plan.stable_hash()
        ):
            raise ValueError("R3-2B candidate binding differs")
        candidate.begin_sample()
    optimizer = torch.optim.Adam([alpha], lr=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    terminal: torch.Tensor | None = None
    custom_forward_count = 0
    custom_backward_count = 0
    for action in schedule.actions:
        if not math.isclose(
            float(optimizer.param_groups[0]["lr"]),
            action.alpha_learning_rate,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("R3-2B runtime learning rate differs")
        if candidate is None:
            lower = _evaluate_full_region(plan, tensors)
            gradient = torch.autograd.grad(-lower.sum(), alpha)[0]
        else:
            lower, gradient = _candidate_evaluate(candidate, action.evaluation_ordinal)
            custom_forward_count += candidate.custom_forward_count
            custom_backward_count += candidate.custom_backward_count
        if action.update_after:
            optimizer.zero_grad(set_to_none=True)
            alpha.grad = gradient
            optimizer.step()
            with torch.no_grad():
                alpha.clamp_(0.0, 1.0)
            scheduler.step()
        else:
            terminal = lower
    if terminal is None:
        raise RuntimeError("R3-2B terminal lower is absent")
    result = R32BWrapperResultV1(
        terminal_lower=terminal.detach(),
        terminal_alpha=alpha.detach(),
        evaluation_count=10,
        optimizer_mutation_count=9,
        scheduler_mutation_count=9,
        custom_forward_count=custom_forward_count,
        custom_backward_count=custom_backward_count,
    )
    result.validate(candidate=candidate is not None)
    return result


__all__ = [
    "execute_r32b_wrapper_v1",
    "PreparedR32BTimingCandidateV1",
    "R32BWrapperResultV1",
]
