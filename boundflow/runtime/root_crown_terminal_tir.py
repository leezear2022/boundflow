"""Runtime and custom autograd bridge for the root terminal CROWN TIR."""

# mypy: disable-error-code=import-untyped
# pylint: disable=import-error,import-outside-toplevel,too-many-arguments
# pylint: disable=too-many-positional-arguments,too-many-locals
# pylint: disable=too-many-instance-attributes,missing-function-docstring
# pylint: disable=abstract-method,arguments-differ

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from boundflow.backends.tvm.root_crown_terminal_linear import (
    CompiledRootCrownTerminalLinearTIRV1,
    RootCrownTerminalLinearTemplateV1,
    compile_root_crown_terminal_tir_v1,
)


@dataclass(frozen=True)
class RootCrownTerminalTensorsV1:
    """Native-layout inputs and VJP sources for one root evaluation."""

    incoming_lower_a: torch.Tensor
    preactivation_lower: torch.Tensor
    preactivation_upper: torch.Tensor
    raw_alpha: torch.Tensor
    operator_weight: torch.Tensor
    operator_bias: torch.Tensor


def validate_root_crown_terminal_tensors_v1(
    tensors: RootCrownTerminalTensorsV1,
    template: RootCrownTerminalLinearTemplateV1,
) -> None:
    """Fail closed on shape, layout, device, dtype and numerical validity."""

    template.validate()
    expected = {
        "incoming_lower_a": (
            template.spec_count,
            template.domain_count,
            template.current_features,
        ),
        "preactivation_lower": (
            template.domain_count,
            template.current_features,
        ),
        "preactivation_upper": (
            template.domain_count,
            template.current_features,
        ),
        "raw_alpha": (
            2,
            template.spec_count,
            template.domain_count,
            template.alpha_feature_count,
        ),
        "operator_weight": (
            template.current_features,
            template.previous_features,
        ),
        "operator_bias": (template.current_features,),
    }
    for name, shape in expected.items():
        value = getattr(tensors, name)
        if (
            tuple(value.shape) != shape
            or value.device.type != "cuda"
            or value.dtype != torch.float32
            or not value.is_contiguous()
            or not bool(torch.isfinite(value).all().item())
        ):
            raise ValueError(f"root CROWN terminal tensor differs: {name}")
    if bool((tensors.preactivation_lower > tensors.preactivation_upper).any().item()):
        raise ValueError("root CROWN terminal interval differs")
    if bool(((tensors.raw_alpha < 0) | (tensors.raw_alpha > 1)).any().item()):
        raise ValueError("root CROWN terminal alpha range differs")


def _validate_runtime_structure(
    tensors: RootCrownTerminalTensorsV1,
    template: RootCrownTerminalLinearTemplateV1,
) -> None:
    """Check O(1) launch invariants without synchronizing device values."""

    expected = {
        "incoming_lower_a": (
            template.spec_count,
            template.domain_count,
            template.current_features,
        ),
        "preactivation_lower": (
            template.domain_count,
            template.current_features,
        ),
        "preactivation_upper": (
            template.domain_count,
            template.current_features,
        ),
        "raw_alpha": (
            2,
            template.spec_count,
            template.domain_count,
            template.alpha_feature_count,
        ),
        "operator_weight": (
            template.current_features,
            template.previous_features,
        ),
        "operator_bias": (template.current_features,),
    }
    devices = set()
    for name, shape in expected.items():
        value = getattr(tensors, name)
        devices.add(value.device)
        if (
            tuple(value.shape) != shape
            or value.device.type != "cuda"
            or value.dtype != torch.float32
            or not value.is_contiguous()
        ):
            raise ValueError(f"root CROWN terminal runtime tensor differs: {name}")
    if len(devices) != 1:
        raise ValueError("root CROWN terminal runtime device differs")


class RootCrownTerminalTIRExecutorV1:
    """One compiled module with exact current-stream DLPack launches."""

    def __init__(self, template: RootCrownTerminalLinearTemplateV1) -> None:
        self.template = template
        self.compiled: CompiledRootCrownTerminalLinearTIRV1 = (
            compile_root_crown_terminal_tir_v1(template)
        )
        self.forward_launch_count = 0
        self.backward_launch_count = 0
        self.fallback_count = 0
        self.pointer_count = 0
        self.pointer_exact_count = 0
        self._view_cache: dict[tuple[int, tuple[int, ...], str, str], Any] = {}
        self._output_a: torch.Tensor | None = None
        self._output_bias: torch.Tensor | None = None
        self._alpha_gradient: torch.Tensor | None = None
        self._bound_gradient: torch.Tensor | None = None
        self._feature_indices = torch.tensor(
            template.alpha_feature_indices,
            dtype=torch.int32,
            device="cuda",
        )
        feature_to_ordinal = torch.full(
            (template.current_features,), -1, dtype=torch.int32, device="cuda"
        )
        feature_to_ordinal[self._feature_indices.to(torch.int64)] = torch.arange(
            template.alpha_feature_count, dtype=torch.int32, device="cuda"
        )
        self._feature_to_ordinal = feature_to_ordinal

    def _view(self, tensor: torch.Tensor) -> tuple[Any, bool]:
        import tvm

        key = (
            tensor.data_ptr(),
            tuple(tensor.shape),
            str(tensor.dtype),
            str(tensor.device),
        )
        existing = self._view_cache.get(key)
        if existing is not None:
            return existing, True
        view = tvm.runtime.from_dlpack(tensor)
        exact = torch.from_dlpack(view).data_ptr() == tensor.data_ptr()
        if not exact:
            raise RuntimeError("root CROWN terminal DLPack pointer differs")
        self._view_cache[key] = view
        return view, True

    def _launch(
        self,
        symbol: str,
        sources: tuple[torch.Tensor, ...],
        outputs: tuple[torch.Tensor, ...],
    ) -> None:
        import tvm_ffi

        device = sources[0].device
        current = torch.cuda.current_stream(device)
        stream_id = int(current.cuda_stream)
        ordinal = device.index
        if ordinal is None:
            ordinal = torch.cuda.current_device()
        source_views = []
        output_views = []
        exact = 0
        with tvm_ffi.use_torch_stream(torch.cuda.stream(current)):
            ffi_stream = int(tvm_ffi.get_raw_stream(tvm_ffi.device(f"cuda:{ordinal}")))
            if ffi_stream != stream_id:
                raise RuntimeError("root CROWN terminal current stream differs")
            for tensor in sources:
                view, pointer_exact = self._view(tensor)
                exact += int(pointer_exact)
                source_views.append(view)
            for tensor in outputs:
                view, pointer_exact = self._view(tensor)
                exact += int(pointer_exact)
                output_views.append(view)
            self.compiled.executable[symbol](*source_views, *output_views)
        count = len(sources) + len(outputs)
        self.pointer_count += count
        self.pointer_exact_count += exact
        if exact != count:
            raise RuntimeError("root CROWN terminal DLPack pointer differs")

    def forward(
        self, tensors: RootCrownTerminalTensorsV1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_runtime_structure(tensors, self.template)
        if self._output_a is None:
            self._output_a = torch.empty(
                (
                    self.template.spec_count,
                    self.template.domain_count,
                    self.template.previous_features,
                ),
                device=tensors.incoming_lower_a.device,
                dtype=torch.float32,
            )
            self._output_bias = torch.empty(
                (self.template.spec_count, self.template.domain_count),
                device=tensors.incoming_lower_a.device,
                dtype=torch.float32,
            )
        output_a = self._output_a
        output_bias = self._output_bias
        if output_bias is None:
            raise RuntimeError("root CROWN terminal output arena differs")
        self._launch(
            self.template.forward_symbol,
            (
                tensors.incoming_lower_a,
                tensors.preactivation_lower,
                tensors.preactivation_upper,
                tensors.raw_alpha,
                self._feature_to_ordinal,
                tensors.operator_weight,
                tensors.operator_bias,
            ),
            (output_a, output_bias),
        )
        self.forward_launch_count += 1
        return output_a, output_bias

    def backward(
        self,
        tensors: RootCrownTerminalTensorsV1,
        output_a_gradient: torch.Tensor,
        output_bias_gradient: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        expected_a = (
            self.template.spec_count,
            self.template.domain_count,
            self.template.previous_features,
        )
        expected_bias = (self.template.spec_count, self.template.domain_count)
        for name, value, shape in (
            ("output A gradient", output_a_gradient, expected_a),
            ("output bias gradient", output_bias_gradient, expected_bias),
        ):
            if (
                tuple(value.shape) != shape
                or value.device != tensors.incoming_lower_a.device
                or value.dtype != torch.float32
                or not value.is_contiguous()
            ):
                raise ValueError(f"root CROWN terminal {name} differs")
        if self._alpha_gradient is None:
            self._alpha_gradient = torch.zeros_like(tensors.raw_alpha)
            self._bound_gradient = torch.empty(
                (
                    2,
                    self.template.domain_count,
                    self.template.current_features,
                ),
                device=tensors.raw_alpha.device,
                dtype=torch.float32,
            )
        alpha_gradient = self._alpha_gradient
        bound_gradient = self._bound_gradient
        if bound_gradient is None:
            raise RuntimeError("root CROWN terminal backward arena differs")
        self._launch(
            self.template.backward_symbol,
            (
                tensors.incoming_lower_a,
                tensors.preactivation_lower,
                tensors.preactivation_upper,
                tensors.raw_alpha,
                self._feature_indices,
                tensors.operator_weight,
                tensors.operator_bias,
                output_a_gradient,
                output_bias_gradient,
            ),
            (alpha_gradient[0], bound_gradient),
        )
        self.backward_launch_count += 1
        return alpha_gradient, bound_gradient[0], bound_gradient[1]


class _RootCrownTerminalTIRFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        incoming_lower_a: torch.Tensor,
        preactivation_lower: torch.Tensor,
        preactivation_upper: torch.Tensor,
        raw_alpha: torch.Tensor,
        operator_weight: torch.Tensor,
        operator_bias: torch.Tensor,
        executor: RootCrownTerminalTIRExecutorV1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tensors = RootCrownTerminalTensorsV1(
            incoming_lower_a=incoming_lower_a,
            preactivation_lower=preactivation_lower,
            preactivation_upper=preactivation_upper,
            raw_alpha=raw_alpha,
            operator_weight=operator_weight,
            operator_bias=operator_bias,
        )
        _validate_runtime_structure(tensors, executor.template)
        ctx.tensors = tensors
        ctx.executor = executor
        ctx.set_materialize_grads(False)
        return executor.forward(tensors)

    @staticmethod
    def backward(
        ctx: Any,
        output_a_gradient: torch.Tensor,
        output_bias_gradient: torch.Tensor,
    ) -> tuple[None, None, None, torch.Tensor, None, None, None]:
        if torch.is_grad_enabled():
            raise RuntimeError("root CROWN terminal higher-order gradient unsupported")
        alpha_gradient, lower_gradient, upper_gradient = ctx.executor.backward(
            ctx.tensors,
            output_a_gradient.contiguous(),
            output_bias_gradient.contiguous(),
        )
        return (
            None,
            lower_gradient,
            upper_gradient,
            alpha_gradient,
            None,
            None,
            None,
        )


def execute_root_crown_terminal_tir_v1(
    tensors: RootCrownTerminalTensorsV1,
    executor: RootCrownTerminalTIRExecutorV1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute the fused transaction through a custom-autograd owner."""

    return _RootCrownTerminalTIRFunction.apply(
        tensors.incoming_lower_a,
        tensors.preactivation_lower,
        tensors.preactivation_upper,
        tensors.raw_alpha,
        tensors.operator_weight,
        tensors.operator_bias,
        executor,
    )


__all__ = [
    "RootCrownTerminalTIRExecutorV1",
    "RootCrownTerminalTensorsV1",
    "execute_root_crown_terminal_tir_v1",
    "validate_root_crown_terminal_tensors_v1",
]
