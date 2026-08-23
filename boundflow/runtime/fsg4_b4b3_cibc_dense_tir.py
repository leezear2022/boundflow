"""Live custom-autograd runtime for production-native dense-alpha CIBC TIR."""

# mypy: disable-error-code=import-untyped
# pylint: disable=import-error,too-many-instance-attributes,too-many-arguments
# pylint: disable=too-many-locals,abstract-method,arguments-differ
# pylint: disable=missing-function-docstring,import-outside-toplevel
# pylint: disable=too-many-boolean-expressions,too-many-positional-arguments

from __future__ import annotations

import torch

from boundflow.backends.tvm.cibc_dense_exact_conv import (
    CIBC_DENSE_BACKWARD_SYMBOL,
    CIBC_DENSE_FORWARD_SYMBOL,
    CompiledCIBCDenseExactConvTIRV3,
)


class CIBCDenseExactTIRExecutorV3:
    """One exact production evaluation using full native alpha state."""

    def __init__(
        self,
        *,
        incoming_lower_a: torch.Tensor,
        preactivation_lower: torch.Tensor,
        preactivation_upper: torch.Tensor,
        native_alpha: torch.Tensor,
        incoming_lower_bias: torch.Tensor,
        operator_weight: torch.Tensor,
        operator_bias: torch.Tensor,
        compiled: CompiledCIBCDenseExactConvTIRV3,
    ) -> None:
        import tvm
        import tvm_ffi

        self.forward_function = compiled.executable[CIBC_DENSE_FORWARD_SYMBOL]
        self.backward_function = compiled.executable[CIBC_DENSE_BACKWARD_SYMBOL]
        self.forward_launch_count = 0
        self.backward_launch_count = 0
        self.fallback_count = 0
        self.eager_count = 0
        self.adjoint_materialization_count = 0
        self.device = incoming_lower_a.device
        self.stream_id = int(torch.cuda.current_stream(self.device).cuda_stream)
        ordinal = self.device.index
        if ordinal is None:
            ordinal = torch.cuda.current_device()
        ffi_device = tvm_ffi.device(f"cuda:{ordinal}")
        if int(tvm_ffi.get_raw_stream(ffi_device)) != self.stream_id:
            raise RuntimeError("CIBC dense exact initial stream differs")
        self.inputs = (
            incoming_lower_a,
            preactivation_lower,
            preactivation_upper,
            native_alpha,
            incoming_lower_bias,
            operator_weight,
            operator_bias,
        )
        expected_shapes = (
            (6, 1, 16, 8, 8),
            (6, 16, 8, 8),
            (6, 16, 8, 8),
            (6, 16, 8, 8),
            (6, 1),
            (16, 16, 3, 3),
            (16,),
        )
        if any(
            tuple(tensor.shape) != shape
            or tensor.dtype != torch.float32
            or tensor.device != self.device
            or not tensor.is_contiguous()
            or not bool(torch.isfinite(tensor).all().item())
            for tensor, shape in zip(self.inputs, expected_shapes)
        ):
            raise ValueError("CIBC dense exact input contract differs")
        self.combined_output = torch.empty(
            6150, dtype=torch.float32, device=self.device
        )
        self.combined_gradient = torch.empty(
            12288, dtype=torch.float32, device=self.device
        )
        forward_tensors = (*self.inputs, self.combined_output)
        static_backward_tensors = (
            incoming_lower_a,
            preactivation_lower,
            preactivation_upper,
            native_alpha,
            operator_weight,
            operator_bias,
        )
        all_static = tuple(
            {
                tensor.data_ptr(): tensor
                for tensor in (
                    *forward_tensors,
                    *static_backward_tensors,
                    self.combined_gradient,
                )
            }.values()
        )
        self._static_views = {
            tensor.data_ptr(): tvm.runtime.from_dlpack(tensor) for tensor in all_static
        }
        if any(
            torch.from_dlpack(self._static_views[tensor.data_ptr()]).data_ptr()
            != tensor.data_ptr()
            for tensor in all_static
        ):
            raise RuntimeError("CIBC dense exact static DLPack pointer differs")
        self.forward_views = tuple(
            self._static_views[tensor.data_ptr()] for tensor in forward_tensors
        )
        self.static_backward_views = tuple(
            self._static_views[tensor.data_ptr()] for tensor in static_backward_tensors
        )
        self.combined_gradient_view = self._static_views[
            self.combined_gradient.data_ptr()
        ]
        self._dynamic_adjoint_views: tuple[object, object] | None = None

    def _validate_stream(self) -> None:
        if int(torch.cuda.current_stream(self.device).cuda_stream) != self.stream_id:
            self.fallback_count += 1
            raise RuntimeError("CIBC dense exact current stream differs")

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_stream()
        self.forward_function(*self.forward_views)
        self.forward_launch_count += 1
        return (
            self.combined_output[:6144].view_as(self.inputs[0]),
            self.combined_output[6144:].view_as(self.inputs[4]),
        )

    def backward(
        self, output_a_gradient: torch.Tensor, output_bias_gradient: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        import tvm

        self._validate_stream()
        if (
            tuple(output_a_gradient.shape) != (6, 1, 16, 8, 8)
            or output_a_gradient.dtype != torch.float32
            or output_a_gradient.device != self.device
            or not output_a_gradient.is_contiguous()
            or tuple(output_bias_gradient.shape) != (6, 1)
            or output_bias_gradient.dtype != torch.float32
            or output_bias_gradient.device != self.device
            or tuple(output_bias_gradient.stride()) != (0, 0)
        ):
            self.fallback_count += 1
            raise ValueError("CIBC dense exact output adjoint differs")
        bias_seed = output_bias_gradient[0, 0].reshape(1)
        if not bias_seed.is_contiguous():
            self.fallback_count += 1
            raise ValueError("CIBC dense exact scalar bias seed differs")
        adjoints = (output_a_gradient, bias_seed)
        dynamic_views = tuple(tvm.runtime.from_dlpack(tensor) for tensor in adjoints)
        if any(
            torch.from_dlpack(view).data_ptr() != tensor.data_ptr()
            for view, tensor in zip(dynamic_views, adjoints)
        ):
            self.fallback_count += 1
            raise RuntimeError("CIBC dense exact adjoint DLPack pointer differs")
        self._dynamic_adjoint_views = dynamic_views
        self.backward_function(
            *self.static_backward_views,
            *dynamic_views,
            self.combined_gradient_view,
        )
        self.backward_launch_count += 1
        incoming_gradient = self.combined_gradient[:6144].view_as(self.inputs[0])
        alpha_gradient = self.combined_gradient[6144:].view_as(self.inputs[3])
        return incoming_gradient, alpha_gradient, output_bias_gradient


class _CIBCDenseExactTIRFunctionV3(torch.autograd.Function):
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
        passed = (incoming, lower, upper, alpha, incoming_bias, weight, operator_bias)
        if any(
            left.data_ptr() != right.data_ptr()
            for left, right in zip(passed, executor.inputs)
        ):
            executor.fallback_count += 1
            raise ValueError("CIBC dense exact autograd input differs")
        ctx.executor = executor
        return executor.forward()

    @staticmethod
    def backward(ctx, output_a_gradient, output_bias_gradient):
        if torch.is_grad_enabled():
            ctx.executor.fallback_count += 1
            raise RuntimeError("CIBC dense exact higher-order gradient is not admitted")
        incoming_gradient, alpha_gradient, incoming_bias_gradient = (
            ctx.executor.backward(output_a_gradient, output_bias_gradient)
        )
        return (
            incoming_gradient,
            None,
            None,
            alpha_gradient,
            incoming_bias_gradient,
            None,
            None,
            None,
        )


def execute_cibc_dense_exact_tir_v3(
    *,
    incoming_lower_a: torch.Tensor,
    preactivation_lower: torch.Tensor,
    preactivation_upper: torch.Tensor,
    native_alpha: torch.Tensor,
    incoming_lower_bias: torch.Tensor,
    operator_weight: torch.Tensor,
    operator_bias: torch.Tensor,
    compiled: CompiledCIBCDenseExactConvTIRV3,
) -> tuple[torch.Tensor, torch.Tensor, CIBCDenseExactTIRExecutorV3]:
    executor = CIBCDenseExactTIRExecutorV3(
        incoming_lower_a=incoming_lower_a,
        preactivation_lower=preactivation_lower,
        preactivation_upper=preactivation_upper,
        native_alpha=native_alpha,
        incoming_lower_bias=incoming_lower_bias,
        operator_weight=operator_weight,
        operator_bias=operator_bias,
        compiled=compiled,
    )
    output_a, output_bias = _CIBCDenseExactTIRFunctionV3.apply(
        incoming_lower_a,
        preactivation_lower,
        preactivation_upper,
        native_alpha,
        incoming_lower_bias,
        operator_weight,
        operator_bias,
        executor,
    )
    return output_a, output_bias, executor


__all__ = ["CIBCDenseExactTIRExecutorV3", "execute_cibc_dense_exact_tir_v3"]
