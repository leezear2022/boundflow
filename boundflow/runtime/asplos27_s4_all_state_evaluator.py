"""Prepared all-state S4 evaluator with reusable compact alpha/beta storage."""

# mypy: disable-error-code=import-untyped
# pylint: disable=import-error,import-outside-toplevel,protected-access
# pylint: disable=too-many-instance-attributes,too-many-locals,too-many-arguments
# pylint: disable=missing-function-docstring,too-few-public-methods

from __future__ import annotations

from dataclasses import dataclass

import torch

from boundflow.backends.tvm.asplos27_s4_six_site_value import (
    CompiledS4SixSiteValueV1,
    compile_s4_six_site_value_v1,
)
from boundflow.runtime.asplos27_s4_coefficient_selector_pass import (
    PreparedS4CoefficientSelectorPassV1,
)
from boundflow.runtime.asplos27_s4_compact_coefficient import (
    PreparedS4CompactCoefficientV1,
)
from boundflow.runtime.asplos27_s4_gradient_emitters import (
    NativeTerminalLowerAdjointLeaseS4V1,
    PreparedS4GradientEmittersV1,
    S4GradientResultV1,
)
from boundflow.runtime.asplos27_s4_ordered_buffer_abi import (
    PreparedS4MutableBuffersV1,
)
from boundflow.runtime.asplos27_s4_six_site_value import (
    PreparedS4SixSiteValueV1,
)
from boundflow.runtime.r3_d2b_staged_backward import (
    PreparedR3D2BStagedBackwardCandidateV1,
)


class S4AllStateEvaluatorError(RuntimeError):
    """Fail-closed all-state evaluator error."""


@dataclass(frozen=True)
class S4AllStateEvaluationV1:
    """Internal policy-consumer payload for one evaluation."""

    ordinal: int
    state_version: int
    lower: torch.Tensor
    gradient_result: S4GradientResultV1
    terminal_lease: NativeTerminalLowerAdjointLeaseS4V1 | None
    compact_parameter_count: int
    selector_launch_count: int
    value_graph_submission_count: int
    compact_coefficient_launch_count: int
    fallback_count: int = 0

    @property
    def gradients(self) -> tuple[torch.Tensor, ...]:
        return self.gradient_result.gradients


def _active_value_read_arguments(
    executor: PreparedR3D2BStagedBackwardCandidateV1,
    buffers: PreparedS4MutableBuffersV1,
    selector: PreparedS4CoefficientSelectorPassV1,
) -> tuple[torch.Tensor, ...]:
    resources = buffers._resources
    if resources is None or len(resources._parameters) != 7:
        raise S4AllStateEvaluatorError("S4_EVALUATOR_PARAMETER_OWNER_MISMATCH")
    tensor = executor._tensor
    alpha = resources._parameters
    return (
        tensor("input/lower"),
        tensor("input/upper"),
        selector.selector("endpoint_ainput_v2"),
        tensor("param/conv1.weight"),
        tensor("param/conv1.bias"),
        tensor("relu/17/lower"),
        tensor("relu/17/upper"),
        alpha[0],
        executor.forward_executor.alpha_maps["17"],
        selector.selector("sign_a18"),
        tensor("param/layer1.0.conv1.weight"),
        tensor("param/layer1.0.conv1.bias"),
        tensor("relu/19/lower"),
        tensor("relu/19/upper"),
        alpha[1],
        executor.forward_executor.alpha_maps["19"],
        selector.selector("sign_a20"),
        tensor("param/layer1.0.conv2.weight"),
        tensor("param/layer1.0.conv2.bias"),
        tensor("param/layer1.0.shortcut.0.weight"),
        tensor("param/layer1.0.shortcut.0.bias"),
        tensor("relu/23/lower"),
        tensor("relu/23/upper"),
        alpha[2],
        executor.forward_executor.alpha_maps["23"],
        selector.selector("sign_a24"),
        tensor("param/layer1.1.conv1.weight"),
        tensor("param/layer1.1.conv1.bias"),
        tensor("relu/25/lower"),
        tensor("relu/25/upper"),
        alpha[3],
        executor.forward_executor.alpha_maps["25"],
        selector.selector("sign_a26"),
        tensor("param/layer1.1.conv2.weight"),
        tensor("param/layer1.1.conv2.bias"),
        tensor("relu/28/lower"),
        tensor("relu/28/upper"),
        alpha[4],
        executor.forward_executor.alpha_maps["28"],
        selector.selector("sign_a29"),
        tensor("param/linear1.weight"),
        tensor("param/linear1.bias"),
    )


class PreparedS4AllStateEvaluatorV1:
    """One prepared GPU evaluator reusable across a 10/9 optimizer run."""

    def __init__(
        self,
        executor: PreparedR3D2BStagedBackwardCandidateV1,
        buffers: PreparedS4MutableBuffersV1,
        *,
        exact_call_id: str,
        stream: torch.cuda.Stream,
        compiled_value: CompiledS4SixSiteValueV1 | None = None,
    ) -> None:
        import tvm_ffi

        resources = buffers._resources
        if resources is None or resources._lower is None:
            raise S4AllStateEvaluatorError("S4_EVALUATOR_BUFFER_OWNER_MISMATCH")
        device = executor.device
        if int(stream.cuda_stream) == int(
            torch.cuda.default_stream(device).cuda_stream
        ):
            raise S4AllStateEvaluatorError("S4_EVALUATOR_NONDEFAULT_STREAM_REQUIRED")
        # Admission/buffer cloning may have been issued on the caller's current
        # stream.  Establish the one-time prepare boundary before persistent
        # execution moves to the dedicated stream; warm runs add no sync.
        torch.cuda.synchronize(device)
        self.executor = executor
        self.buffers = buffers
        self.resources = resources
        self.device = device
        self.stream = stream
        self.exact_call_id = exact_call_id
        self.compact = PreparedS4CompactCoefficientV1(executor, buffers)
        self.selector = PreparedS4CoefficientSelectorPassV1(
            device=device,
            exact_call_id=exact_call_id,
            evaluation_generation=1,
            parameter_generation=2,
            coefficient_generation=3,
            selector_generation=4,
        )
        with (
            torch.cuda.stream(stream),
            tvm_ffi.use_torch_stream(torch.cuda.stream(stream)),
        ):
            self.compact.capture_selectors(self.selector)
        stream.synchronize()
        read_arguments = _active_value_read_arguments(executor, buffers, self.selector)
        coefficient_arena = executor.forward_executor.scratch_1
        ordinal = device.index if device.index is not None else 0
        self.value = PreparedS4SixSiteValueV1(
            compiled_value or compile_s4_six_site_value_v1(device_index=ordinal),
            read_arguments,
            coefficient_arena=coefficient_arena,
            selected_input_alias=coefficient_arena[:18432].view(6, 3, 32, 32),
            device=device,
        )
        with torch.cuda.stream(stream):
            self.value.begin_pass_a()
            self.value.adopt_selectors(self.selector)
            self.value.run_pass_b()
            warm_value = self.value.handoff_to_coefficient_recompute()
        stream.synchronize()
        self.gradient = PreparedS4GradientEmittersV1(
            executor,
            warm_value,
            buffers,
            evaluation_generation=0,
            state_version=0,
            compact_coefficient=self.compact,
        )
        with torch.cuda.stream(stream):
            self.gradient.run(terminal=False)
        stream.synchronize()
        self._next_ordinal = 0
        self._closed = False
        self._terminal_complete = False
        self._compact_launch_base = self.compact.launch_count

    @staticmethod
    def _generations(ordinal: int) -> tuple[int, int, int, int]:
        base = 101 + ordinal * 4
        return base, base + 1, base + 2, base + 3

    def evaluate(self, ordinal: int, *, terminal: bool) -> S4AllStateEvaluationV1:
        import tvm_ffi

        if (
            self._closed
            or self._terminal_complete
            or ordinal != self._next_ordinal
            or terminal != (ordinal == 9)
        ):
            raise S4AllStateEvaluatorError("S4_EVALUATOR_ORDINAL_OR_MODE_MISMATCH")
        self.executor.begin_evaluation(ordinal)
        generations = self._generations(ordinal)
        self.selector.rearm(
            evaluation_generation=generations[0],
            parameter_generation=generations[1],
            coefficient_generation=generations[2],
            selector_generation=generations[3],
        )
        self.value.rearm()
        launch_before = self.compact.launch_count
        with (
            torch.cuda.stream(self.stream),
            tvm_ffi.use_torch_stream(torch.cuda.stream(self.stream)),
        ):
            self.value.begin_pass_a()
            self.compact.capture_selectors(self.selector)
            self.value.adopt_selectors(self.selector)
            self.value.run_pass_b()
            value_result = self.value.handoff_to_coefficient_recompute()
        with torch.cuda.stream(self.stream):
            self.gradient.rearm(
                value_result,
                evaluation_generation=ordinal + 1,
                state_version=ordinal + 1,
            )
            gradient_result = self.gradient.run(terminal=terminal)
        terminal_lease = self.gradient.take_terminal_lease() if terminal else None
        lower = self.resources._lower
        if lower is None:
            raise S4AllStateEvaluatorError("S4_EVALUATOR_BUFFER_OWNER_MISMATCH")
        result = S4AllStateEvaluationV1(
            ordinal=ordinal,
            state_version=ordinal,
            lower=lower,
            gradient_result=gradient_result,
            terminal_lease=terminal_lease,
            compact_parameter_count=7,
            selector_launch_count=6,
            value_graph_submission_count=1,
            compact_coefficient_launch_count=self.compact.launch_count - launch_before,
        )
        self._next_ordinal += 1
        self._terminal_complete = terminal
        return result

    def close(self) -> None:
        self.gradient.close()
        self.value.close()
        self.selector.close()
        self._closed = True


__all__ = [
    "PreparedS4AllStateEvaluatorV1",
    "S4AllStateEvaluationV1",
    "S4AllStateEvaluatorError",
]
