"""Zero-copy runtime for horizontally fused CIBC IBP Conv2d."""

# mypy: disable-error-code=import-untyped
# pylint: disable=import-error,missing-function-docstring,import-outside-toplevel
# pylint: disable=too-many-arguments,too-many-positional-arguments
# pylint: disable=too-many-instance-attributes

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache
import math
from typing import cast, Iterator

import torch

from boundflow.backends.tvm.cibc_ibp_conv import (
    CIBC_IBP_CONV_SYMBOL,
    CIBCIBPConvSignatureV1,
    CompiledCIBCIBPConvV1,
    compile_cibc_ibp_conv_tir_v1,
)


@dataclass
class CIBCIBPConvExecutionContextV1:
    """Typed activation and launch receipt for horizontal Conv fusion."""

    threads_per_block: int = 128
    launch_count: int = 0
    fallback_count: int = 0


_ACTIVE_CONTEXT: ContextVar[CIBCIBPConvExecutionContextV1 | None] = ContextVar(
    "boundflow_cibc_ibp_conv_context", default=None
)


@contextmanager
def use_cibc_ibp_conv_v1(
    *, threads_per_block: int = 128
) -> Iterator[CIBCIBPConvExecutionContextV1]:
    context = CIBCIBPConvExecutionContextV1(threads_per_block=threads_per_block)
    if threads_per_block not in {64, 128, 256}:
        raise ValueError("CIBC IBP Conv context schedule differs")
    token = _ACTIVE_CONTEXT.set(context)
    try:
        yield context
    finally:
        _ACTIVE_CONTEXT.reset(token)


def active_cibc_ibp_conv_context_v1() -> CIBCIBPConvExecutionContextV1 | None:
    return _ACTIVE_CONTEXT.get()


@lru_cache(maxsize=64)
def _compile_cached(
    signature: CIBCIBPConvSignatureV1,
    threads_per_block: int,
    compute_capability: str,
) -> CompiledCIBCIBPConvV1:
    return compile_cibc_ibp_conv_tir_v1(
        signature,
        threads_per_block=threads_per_block,
        compute_capability=compute_capability,
    )


def execute_active_cibc_ibp_conv_v1(
    lower: torch.Tensor,
    upper: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    groups: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    context = _ACTIVE_CONTEXT.get()
    if context is None:
        return None
    signature = CIBCIBPConvSignatureV1(
        input_shape=cast(
            tuple[int, int, int, int], tuple(int(value) for value in lower.shape)
        ),
        weight_shape=cast(
            tuple[int, int, int, int], tuple(int(value) for value in weight.shape)
        ),
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    major, minor = torch.cuda.get_device_capability(lower.device)
    compiled = _compile_cached(
        signature, context.threads_per_block, f"sm_{major}{minor}"
    )
    result = execute_cibc_ibp_conv_v1(
        lower.contiguous(),
        upper.contiguous(),
        weight.contiguous(),
        bias.contiguous(),
        compiled=compiled,
    )
    context.launch_count += 1
    return result


class CIBCIBPConvExecutorV1:
    """Plan-owned zero-copy views and output buffer for one fixed call site."""

    def __init__(
        self,
        lower: torch.Tensor,
        upper: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        *,
        compiled: CompiledCIBCIBPConvV1,
        output: torch.Tensor | None = None,
    ) -> None:
        import tvm
        import tvm_ffi

        self.compiled = compiled
        self.function = compiled.executable[CIBC_IBP_CONV_SYMBOL]
        self.inputs = (lower, upper, weight, bias)
        signature = compiled.signature
        expected_shapes = (
            signature.input_shape,
            signature.input_shape,
            signature.weight_shape,
            (signature.weight_shape[0],),
        )
        if (
            any(
                tuple(tensor.shape) != shape
                or tensor.dtype != torch.float32
                or tensor.device != lower.device
                or not tensor.is_contiguous()
                or tensor.requires_grad
                for tensor, shape in zip(self.inputs, expected_shapes)
            )
            or lower.device.type != "cuda"
        ):
            raise ValueError("CIBC IBP Conv runtime input differs")
        self.output_count = math.prod(signature.output_shape)
        self.combined = (
            torch.empty(2 * self.output_count, dtype=torch.float32, device=lower.device)
            if output is None
            else output
        )
        if (
            tuple(self.combined.shape) != (2 * self.output_count,)
            or self.combined.dtype != torch.float32
            or self.combined.device != lower.device
            or not self.combined.is_contiguous()
        ):
            raise ValueError("CIBC IBP Conv output buffer differs")
        ordinal = lower.device.index
        if ordinal is None:
            ordinal = torch.cuda.current_device()
        self.device = lower.device
        self.stream = int(torch.cuda.current_stream(lower.device).cuda_stream)
        if (
            int(tvm_ffi.get_raw_stream(tvm_ffi.device(f"cuda:{ordinal}")))
            != self.stream
        ):
            raise RuntimeError("CIBC IBP Conv stream differs")
        tensors = (*self.inputs, self.combined)
        self.views = tuple(tvm.runtime.from_dlpack(tensor) for tensor in tensors)
        if any(
            torch.from_dlpack(view).data_ptr() != tensor.data_ptr()
            for view, tensor in zip(self.views, tensors)
        ):
            raise RuntimeError("CIBC IBP Conv zero-copy pointer differs")
        self.lower_output = self.combined[: self.output_count].view(
            self.compiled.signature.output_shape
        )
        self.upper_output = self.combined[self.output_count :].view(
            self.compiled.signature.output_shape
        )
        self.launch_count = 0

    def validate_stream(self) -> None:
        """Fail closed once at the outer execution-plan boundary."""

        if int(torch.cuda.current_stream(self.device).cuda_stream) != self.stream:
            raise RuntimeError("CIBC IBP Conv current stream differs")

    def run(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.function(*self.views)
        self.launch_count += 1
        return self.lower_output, self.upper_output


def execute_cibc_ibp_conv_v1(
    lower: torch.Tensor,
    upper: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    compiled: CompiledCIBCIBPConvV1,
    output: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return CIBCIBPConvExecutorV1(
        lower, upper, weight, bias, compiled=compiled, output=output
    ).run()


__all__ = [
    "CIBCIBPConvExecutionContextV1",
    "CIBCIBPConvExecutorV1",
    "active_cibc_ibp_conv_context_v1",
    "execute_active_cibc_ibp_conv_v1",
    "execute_cibc_ibp_conv_v1",
    "use_cibc_ibp_conv_v1",
]
