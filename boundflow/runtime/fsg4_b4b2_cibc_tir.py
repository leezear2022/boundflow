"""Runtime for the manual TVM TIR CIBC-parity horizontal Conv port."""

# mypy: disable-error-code=import-untyped
# pylint: disable=import-error,too-many-instance-attributes,too-many-locals
# pylint: disable=too-many-arguments,too-many-positional-arguments,protected-access
# pylint: disable=abstract-method,arguments-differ,missing-function-docstring
# pylint: disable=missing-class-docstring,import-outside-toplevel

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import torch

from boundflow.backends.tvm.cibc_horizontal_fused_conv import (
    CIBC_TIR_BACKWARD_SYMBOL,
    CIBC_TIR_FORWARD_SYMBOL,
    CompiledCIBCHorizontalConvTIRV2,
    compile_cibc_horizontal_conv_tir_v2,
)

from .fsg4_b4b1_pytorch_reference import build_b4b1_differentiable_lower_ir_v1
from .fsg4_b4b1_reference_capture import ProductionDifferentiableReferenceCaptureV1
from .fsg4_b4b2_sparse_conv_timing import (
    SparseConvExecutionV1,
    execute_sparse_conv_pytorch_baseline_v1,
)
from .fsg4_b4b2_sparse_conv_tir import (
    SparseConvTIRTensorsV1,
    build_b4b2_sparse_conv_template_v1,
    build_b4b2_sparse_conv_tensors_v1,
)


@dataclass(frozen=True)
class CIBCTIRCompilationReceiptV2:
    module_hash: str
    device_source_hash: str
    device_source_sha256: str
    exported_symbols: tuple[str, ...]
    global_workspace_bytes: int
    local_buffer_count: int

    def validate(self) -> None:
        if (
            any(
                len(value) != 64
                for value in (self.module_hash, self.device_source_hash)
            )
            or self.device_source_hash != self.device_source_sha256
            or self.exported_symbols
            != (CIBC_TIR_FORWARD_SYMBOL, CIBC_TIR_BACKWARD_SYMBOL)
            or self.global_workspace_bytes != 0
            or self.local_buffer_count < 0
        ):
            raise ValueError("CIBC horizontal TIR compilation receipt differs")


class CIBCHorizontalTIRExecutorV2:
    """Plan-instance-owned tensors, buffers, DLPack views, and compiled module."""

    def __init__(
        self,
        tensors: SparseConvTIRTensorsV1,
        template,
        compiled: CompiledCIBCHorizontalConvTIRV2,
    ) -> None:
        import tvm
        import tvm_ffi

        self.tensors = tensors
        self.template = template
        self.compiled = compiled
        self.forward_function = compiled.executable[CIBC_TIR_FORWARD_SYMBOL]
        self.backward_function = compiled.executable[CIBC_TIR_BACKWARD_SYMBOL]
        self.forward_launch_count = 0
        self.backward_launch_count = 0
        self.fallback_count = 0
        self.eager_count = 0
        self.device = tensors.incoming_lower_a.device
        self.stream_id = int(torch.cuda.current_stream(self.device).cuda_stream)
        ordinal = self.device.index
        if ordinal is None:
            ordinal = torch.cuda.current_device()
        self.ffi_device = tvm_ffi.device(f"cuda:{ordinal}")
        self.ffi_stream_id = int(tvm_ffi.get_raw_stream(self.ffi_device))
        if self.ffi_stream_id != self.stream_id:
            raise RuntimeError("CIBC horizontal TIR initial stream differs")
        alpha_map = torch.full(
            (template.channels, template.height, template.width),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        coordinates = template.alpha_coordinates
        index = tuple(
            torch.tensor(
                [coordinate[axis] for coordinate in coordinates],
                dtype=torch.int64,
                device=self.device,
            )
            for axis in range(3)
        )
        alpha_map[index] = torch.arange(
            len(coordinates), dtype=torch.int32, device=self.device
        )
        self.alpha_map = alpha_map.contiguous()
        self.alpha_coordinates = tuple(
            torch.tensor(
                [coordinate[axis] for coordinate in coordinates],
                dtype=torch.int32,
                device=self.device,
            ).contiguous()
            for axis in range(3)
        )
        self.combined_output = torch.empty(
            6150, dtype=torch.float32, device=self.device
        )
        self.combined_gradient = torch.empty(
            6660, dtype=torch.float32, device=self.device
        )
        forward_tensors = (
            tensors.incoming_lower_a,
            tensors.preactivation_lower,
            tensors.preactivation_upper,
            tensors.compressed_alpha,
            tensors.incoming_lower_bias,
            tensors.operator_weight,
            tensors.operator_bias,
            self.alpha_map,
            self.combined_output,
        )
        backward_tensors = (
            tensors.incoming_lower_a,
            tensors.preactivation_lower,
            tensors.preactivation_upper,
            tensors.compressed_alpha,
            tensors.operator_weight,
            tensors.operator_bias,
            tensors.output_lower_a_gradient,
            tensors.output_bias_gradient,
            self.alpha_map,
            *self.alpha_coordinates,
            self.combined_gradient,
        )
        all_tensors = tuple(
            {
                tensor.data_ptr(): tensor
                for tensor in (*forward_tensors, *backward_tensors)
            }.values()
        )
        if len({tensor.data_ptr() for tensor in all_tensors}) != len(all_tensors):
            raise ValueError("CIBC horizontal TIR tensor aliases differ")
        self._tvm_views = {
            tensor.data_ptr(): tvm.runtime.from_dlpack(tensor) for tensor in all_tensors
        }
        if any(
            torch.from_dlpack(self._tvm_views[tensor.data_ptr()]).data_ptr()
            != tensor.data_ptr()
            for tensor in all_tensors
        ):
            raise RuntimeError("CIBC horizontal TIR DLPack pointer differs")
        self.forward_views = tuple(
            self._tvm_views[tensor.data_ptr()] for tensor in forward_tensors
        )
        self.backward_views = tuple(
            self._tvm_views[tensor.data_ptr()] for tensor in backward_tensors
        )
        receipt = CIBCTIRCompilationReceiptV2(
            module_hash=compiled.module_hash,
            device_source_hash=compiled.device_source_hash,
            device_source_sha256=hashlib.sha256(
                compiled.device_source.encode("utf-8")
            ).hexdigest(),
            exported_symbols=compiled.exported_symbols,
            global_workspace_bytes=compiled.global_workspace_bytes,
            local_buffer_count=compiled.local_buffer_count,
        )
        receipt.validate()
        self.compilation_receipt = receipt

    def _validate_stream(self) -> None:
        if int(torch.cuda.current_stream(self.device).cuda_stream) != self.stream_id:
            self.fallback_count += 1
            raise RuntimeError("CIBC horizontal TIR current stream differs")

    def forward(self):
        self._validate_stream()
        self.forward_function(*self.forward_views)
        self.forward_launch_count += 1
        return (
            self.combined_output[:6144].view_as(self.tensors.incoming_lower_a),
            self.combined_output[6144:].view_as(self.tensors.incoming_lower_bias),
        )

    def backward(self, output_a_gradient, output_bias_gradient):
        if (
            output_a_gradient.data_ptr()
            != self.tensors.output_lower_a_gradient.data_ptr()
            or output_bias_gradient.data_ptr()
            != self.tensors.output_bias_gradient.data_ptr()
        ):
            self.fallback_count += 1
            raise ValueError("CIBC horizontal TIR output adjoint differs")
        self.backward_function(*self.backward_views)
        self.backward_launch_count += 1
        return (
            self.combined_gradient[:6144].view_as(self.tensors.incoming_lower_a),
            self.combined_gradient[6144:].view_as(self.tensors.compressed_alpha),
        )


class _CIBCHorizontalTIRFunctionV2(torch.autograd.Function):
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
        expected = (
            executor.tensors.incoming_lower_a,
            executor.tensors.preactivation_lower,
            executor.tensors.preactivation_upper,
            executor.tensors.compressed_alpha,
            executor.tensors.incoming_lower_bias,
            executor.tensors.operator_weight,
            executor.tensors.operator_bias,
        )
        passed = (incoming, lower, upper, alpha, incoming_bias, weight, operator_bias)
        if any(
            left.data_ptr() != right.data_ptr() for left, right in zip(passed, expected)
        ):
            executor.fallback_count += 1
            raise ValueError("CIBC horizontal TIR autograd input differs")
        ctx.executor = executor
        return executor.forward()

    @staticmethod
    def backward(ctx, output_a_gradient, output_bias_gradient):
        if torch.is_grad_enabled():
            ctx.executor.fallback_count += 1
            raise RuntimeError(
                "CIBC horizontal TIR higher-order gradient is not admitted"
            )
        incoming_gradient, alpha_gradient = ctx.executor.backward(
            output_a_gradient, output_bias_gradient
        )
        return incoming_gradient, None, None, alpha_gradient, None, None, None, None


class PreparedCIBCHorizontalTIRV2:
    """Compile once, admit once, and reuse plan-instance runtime state."""

    def __init__(
        self,
        capture: ProductionDifferentiableReferenceCaptureV1,
        *,
        compiled: CompiledCIBCHorizontalConvTIRV2 | None = None,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CIBC horizontal TIR requires CUDA")
        lower_ir = build_b4b1_differentiable_lower_ir_v1(capture)
        major, minor = torch.cuda.get_device_capability()
        self.template = build_b4b2_sparse_conv_template_v1(
            lower_ir, capture, compute_capability=f"sm_{major}{minor}"
        )
        self.tensors = build_b4b2_sparse_conv_tensors_v1(
            capture, self.template, device=torch.device("cuda:0")
        )
        self.compiled = compiled or compile_cibc_horizontal_conv_tir_v2(
            compute_capability=f"sm_{major}{minor}"
        )
        self.executor = CIBCHorizontalTIRExecutorV2(
            self.tensors, self.template, self.compiled
        )
        self.candidate_once()
        torch.cuda.synchronize()

    def baseline_once(self) -> SparseConvExecutionV1:
        return execute_sparse_conv_pytorch_baseline_v1(self.tensors, self.template)

    def candidate_once(self) -> SparseConvExecutionV1:
        tensors = self.tensors
        output_a, output_bias = _CIBCHorizontalTIRFunctionV2.apply(
            tensors.incoming_lower_a,
            tensors.preactivation_lower,
            tensors.preactivation_upper,
            tensors.compressed_alpha,
            tensors.incoming_lower_bias,
            tensors.operator_weight,
            tensors.operator_bias,
            self.executor,
        )
        incoming_gradient, alpha_gradient = torch.autograd.grad(
            (output_a, output_bias),
            (tensors.incoming_lower_a, tensors.compressed_alpha),
            grad_outputs=(
                tensors.output_lower_a_gradient,
                tensors.output_bias_gradient,
            ),
        )
        return SparseConvExecutionV1(
            output_lower_a=output_a,
            output_bias=output_bias,
            compressed_alpha_gradient=alpha_gradient,
            incoming_lower_a_gradient=incoming_gradient,
        )


__all__ = [
    "CIBCHorizontalTIRExecutorV2",
    "CIBCTIRCompilationReceiptV2",
    "PreparedCIBCHorizontalTIRV2",
]
