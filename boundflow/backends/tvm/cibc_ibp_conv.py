"""Horizontally fused IBP Conv2d forward kernels for fixed CUDA signatures."""

# mypy: disable-error-code=import-untyped
# mypy: disable-error-code=valid-type
# pylint: disable=import-error,missing-function-docstring,import-outside-toplevel
# pylint: disable=too-many-arguments,too-many-locals,too-many-positional-arguments
# pylint: disable=invalid-name,too-many-statements,missing-class-docstring
# pylint: disable=too-many-boolean-expressions,chained-comparison

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import cast

from boundflow.backends.tvm.differentiable_lower_identity import (
    DifferentiableLowerTIRExecutable,
)

CIBC_IBP_CONV_SYMBOL = "boundflow_cibc_ibp_conv_horizontal_v1"
CIBC_IBP_CONV_RELAX_SYMBOL = "boundflow_cibc_ibp_conv_horizontal_relax_v1"


@dataclass(frozen=True)
class CIBCIBPConvSignatureV1:
    input_shape: tuple[int, int, int, int]
    weight_shape: tuple[int, int, int, int]
    stride: tuple[int, int]
    padding: tuple[int, int]
    dilation: tuple[int, int]
    groups: int

    @property
    def output_shape(self) -> tuple[int, int, int, int]:
        batch, _input_channels, input_height, input_width = self.input_shape
        output_channels, _group_channels, kernel_height, kernel_width = (
            self.weight_shape
        )
        output_height = (
            input_height
            + 2 * self.padding[0]
            - self.dilation[0] * (kernel_height - 1)
            - 1
        ) // self.stride[0] + 1
        output_width = (
            input_width
            + 2 * self.padding[1]
            - self.dilation[1] * (kernel_width - 1)
            - 1
        ) // self.stride[1] + 1
        return batch, output_channels, output_height, output_width

    def validate(self) -> None:
        if (
            self.groups != 1
            or any(value <= 0 for value in (*self.input_shape, *self.weight_shape))
            or self.input_shape[1] != self.weight_shape[1]
            or any(value <= 0 for value in (*self.stride, *self.dilation))
            or any(value < 0 for value in self.padding)
            or any(value <= 0 for value in self.output_shape)
        ):
            raise ValueError("CIBC IBP Conv signature differs")


@dataclass(frozen=True)
class CompiledCIBCIBPConvV1:
    executable: DifferentiableLowerTIRExecutable
    signature: CIBCIBPConvSignatureV1
    threads_per_block: int
    module_hash: str
    device_source_hash: str
    device_source: str
    tvm_version: str


def build_cibc_ibp_conv_tir_v1(
    signature: CIBCIBPConvSignatureV1, *, threads_per_block: int
):
    import tvm
    from tvm.script import tir as T

    signature.validate()
    if threads_per_block not in {64, 128, 256}:
        raise ValueError("CIBC IBP Conv schedule differs")
    input_shape = signature.input_shape
    weight_shape = signature.weight_shape
    output_shape = signature.output_shape
    output_count = math.prod(output_shape)
    reduction_extent = math.prod(weight_shape[1:])
    block_count = math.ceil(output_count / threads_per_block)

    @T.prim_func
    def horizontal_conv(
        lower: T.Buffer(input_shape, "float32"),
        upper: T.Buffer(input_shape, "float32"),
        weight: T.Buffer(weight_shape, "float32"),
        bias: T.Buffer((weight_shape[0],), "float32"),
        combined_output: T.Buffer((2 * output_count,), "float32"),
    ):
        T.func_attr(
            {
                "global_symbol": CIBC_IBP_CONV_SYMBOL,
                "tir.noalias": True,
                "boundflow.schema_version": "cibc-ibp-conv-horizontal/v1",
            }
        )
        center = T.alloc_buffer((1,), "float32", scope="local")
        deviation = T.alloc_buffer((1,), "float32", scope="local")
        for block_x in T.thread_binding(block_count, thread="blockIdx.x"):
            for thread_x in T.thread_binding(threads_per_block, thread="threadIdx.x"):
                flat = block_x * threads_per_block + thread_x
                if flat < output_count:
                    output_w = flat % output_shape[3]
                    output_h = flat // output_shape[3] % output_shape[2]
                    output_channel = (
                        flat // (output_shape[2] * output_shape[3]) % output_shape[1]
                    )
                    output_batch = flat // (
                        output_shape[1] * output_shape[2] * output_shape[3]
                    )
                    center[0] = T.float32(0)
                    deviation[0] = T.float32(0)
                    for reduction in range(reduction_extent):
                        kernel_w = reduction % weight_shape[3]
                        kernel_h = reduction // weight_shape[3] % weight_shape[2]
                        input_channel = reduction // (weight_shape[2] * weight_shape[3])
                        input_h = (
                            output_h * signature.stride[0]
                            - signature.padding[0]
                            + kernel_h * signature.dilation[0]
                        )
                        input_w = (
                            output_w * signature.stride[1]
                            - signature.padding[1]
                            + kernel_w * signature.dilation[1]
                        )
                        if (
                            0 <= input_h
                            and input_h < input_shape[2]
                            and 0 <= input_w
                            and input_w < input_shape[3]
                        ):
                            lower_value = lower[
                                output_batch, input_channel, input_h, input_w
                            ]
                            upper_value = upper[
                                output_batch, input_channel, input_h, input_w
                            ]
                            weight_value = weight[
                                output_channel, input_channel, kernel_h, kernel_w
                            ]
                            center[0] = (
                                center[0]
                                + (lower_value + upper_value)
                                * T.float32(0.5)
                                * weight_value
                            )
                            deviation[0] = deviation[0] + (
                                upper_value - lower_value
                            ) * T.float32(0.5) * T.abs(weight_value)
                    center[0] = center[0] + bias[output_channel]
                    combined_output[flat] = center[0] - deviation[0]
                    combined_output[output_count + flat] = center[0] + deviation[0]

    return tvm.IRModule({CIBC_IBP_CONV_SYMBOL: horizontal_conv})


def build_cibc_ibp_conv_relax_tir_v1(
    signature: CIBCIBPConvSignatureV1, *, threads_per_block: int
):
    """Build the same fused Conv with two DPS outputs for Relax ``call_tir``."""

    import tvm
    from tvm.script import tir as T

    signature.validate()
    if threads_per_block not in {64, 128, 256}:
        raise ValueError("CIBC IBP Conv schedule differs")
    input_shape = signature.input_shape
    weight_shape = signature.weight_shape
    output_shape = signature.output_shape
    output_count = math.prod(output_shape)
    reduction_extent = math.prod(weight_shape[1:])
    block_count = math.ceil(output_count / threads_per_block)

    @T.prim_func
    def horizontal_conv_relax(
        lower: T.Buffer(input_shape, "float32"),
        upper: T.Buffer(input_shape, "float32"),
        weight: T.Buffer(weight_shape, "float32"),
        bias: T.Buffer((weight_shape[0],), "float32"),
        lower_output: T.Buffer(output_shape, "float32"),
        upper_output: T.Buffer(output_shape, "float32"),
    ):
        T.func_attr(
            {
                "global_symbol": CIBC_IBP_CONV_RELAX_SYMBOL,
                "tir.noalias": True,
                "boundflow.schema_version": "cibc-ibp-conv-horizontal-relax/v1",
            }
        )
        center = T.alloc_buffer((1,), "float32", scope="local")
        deviation = T.alloc_buffer((1,), "float32", scope="local")
        for block_x in T.thread_binding(block_count, thread="blockIdx.x"):
            for thread_x in T.thread_binding(threads_per_block, thread="threadIdx.x"):
                flat = block_x * threads_per_block + thread_x
                if flat < output_count:
                    output_w = flat % output_shape[3]
                    output_h = flat // output_shape[3] % output_shape[2]
                    output_channel = (
                        flat // (output_shape[2] * output_shape[3]) % output_shape[1]
                    )
                    output_batch = flat // (
                        output_shape[1] * output_shape[2] * output_shape[3]
                    )
                    center[0] = T.float32(0)
                    deviation[0] = T.float32(0)
                    for reduction in range(reduction_extent):
                        kernel_w = reduction % weight_shape[3]
                        kernel_h = reduction // weight_shape[3] % weight_shape[2]
                        input_channel = reduction // (weight_shape[2] * weight_shape[3])
                        input_h = (
                            output_h * signature.stride[0]
                            - signature.padding[0]
                            + kernel_h * signature.dilation[0]
                        )
                        input_w = (
                            output_w * signature.stride[1]
                            - signature.padding[1]
                            + kernel_w * signature.dilation[1]
                        )
                        if (
                            0 <= input_h
                            and input_h < input_shape[2]
                            and 0 <= input_w
                            and input_w < input_shape[3]
                        ):
                            lower_value = lower[
                                output_batch,
                                input_channel,
                                input_h,
                                input_w,
                            ]
                            upper_value = upper[
                                output_batch,
                                input_channel,
                                input_h,
                                input_w,
                            ]
                            weight_value = weight[
                                output_channel,
                                input_channel,
                                kernel_h,
                                kernel_w,
                            ]
                            center[0] = (
                                center[0]
                                + (lower_value + upper_value)
                                * T.float32(0.5)
                                * weight_value
                            )
                            deviation[0] = deviation[0] + (
                                upper_value - lower_value
                            ) * T.float32(0.5) * T.abs(weight_value)
                    center[0] = center[0] + bias[output_channel]
                    lower_output[output_batch, output_channel, output_h, output_w] = (
                        center[0] - deviation[0]
                    )
                    upper_output[output_batch, output_channel, output_h, output_w] = (
                        center[0] + deviation[0]
                    )

    return tvm.IRModule({CIBC_IBP_CONV_RELAX_SYMBOL: horizontal_conv_relax})


def compile_cibc_ibp_conv_tir_v1(
    signature: CIBCIBPConvSignatureV1,
    *,
    threads_per_block: int,
    compute_capability: str,
) -> CompiledCIBCIBPConvV1:
    import tvm

    module = build_cibc_ibp_conv_tir_v1(signature, threads_per_block=threads_per_block)
    executable = tvm.compile(module, target=f"cuda -arch={compute_capability}")
    sources = tuple(imported.inspect_source() for imported in executable.mod.imports)
    if len(sources) != 1 or CIBC_IBP_CONV_SYMBOL not in sources[0]:
        raise RuntimeError("CIBC IBP Conv device module differs")
    source = sources[0]
    return CompiledCIBCIBPConvV1(
        executable=cast(DifferentiableLowerTIRExecutable, executable),
        signature=signature,
        threads_per_block=threads_per_block,
        module_hash=hashlib.sha256(
            tvm.ir.save_json(module).encode("utf-8")
        ).hexdigest(),
        device_source_hash=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        device_source=source,
        tvm_version=str(tvm.__version__),
    )


__all__ = [
    "CIBC_IBP_CONV_RELAX_SYMBOL",
    "CIBC_IBP_CONV_SYMBOL",
    "CIBCIBPConvSignatureV1",
    "CompiledCIBCIBPConvV1",
    "build_cibc_ibp_conv_tir_v1",
    "build_cibc_ibp_conv_relax_tir_v1",
    "compile_cibc_ibp_conv_tir_v1",
]
