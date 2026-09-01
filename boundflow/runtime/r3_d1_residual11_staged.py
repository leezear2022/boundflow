"""Zero-copy current-stream runtime for D1-A residual11 qualification."""

# mypy: disable-error-code=import-untyped
# pylint: disable=import-error,import-outside-toplevel,too-many-arguments
# pylint: disable=too-many-positional-arguments,too-many-locals,missing-function-docstring
# pylint: disable=too-few-public-methods,too-many-boolean-expressions
# pylint: disable=too-many-instance-attributes

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from boundflow.backends.tvm.r3_d1_residual11_staged import (
    CompiledR3D1Residual11StagedV1,
    compile_r3d1_residual11_staged_v1,
    R3D1_RESIDUAL11_STAGE1_SYMBOL,
    R3D1_RESIDUAL11_STAGE2_SYMBOL,
    R3D1_RESIDUAL11_SYMBOLS,
)


@dataclass(frozen=True)
class R3D1Residual11StagedReceiptV1:
    """Correctness-only ABI and ownership receipt."""

    unscheduled_tir_hash: str
    scheduled_tir_hash: str
    device_source_hash: str
    exported_symbols: tuple[str, ...]
    stream_id: int
    tvm_ffi_stream_id: int
    launch_count: int
    dlpack_pointer_count: int
    dlpack_pointer_exact_count: int
    scratch_count: int
    persistent_dense_a: bool
    fallback_count: int
    timing_recorded: bool
    performance_claimed: bool

    def validate(self) -> None:
        hashes = (
            self.unscheduled_tir_hash,
            self.scheduled_tir_hash,
            self.device_source_hash,
        )
        if (
            any(
                len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in hashes
            )
            or self.exported_symbols != R3D1_RESIDUAL11_SYMBOLS
            or self.stream_id <= 0
            or self.tvm_ffi_stream_id != self.stream_id
            or self.launch_count != 2
            or self.dlpack_pointer_count != 13
            or self.dlpack_pointer_exact_count != self.dlpack_pointer_count
            or self.scratch_count != 1
            or self.persistent_dense_a
            or self.fallback_count
            or self.timing_recorded
            or self.performance_claimed
        ):
            raise ValueError("R3-D1 residual11 receipt differs")


class R3D1Residual11ModuleCacheV1:
    """Tensor-free compile cache for the isolated D1-A module."""

    def __init__(self) -> None:
        self._compiled: CompiledR3D1Residual11StagedV1 | None = None

    def get(self) -> CompiledR3D1Residual11StagedV1:
        if self._compiled is None:
            self._compiled = compile_r3d1_residual11_staged_v1()
        return self._compiled


def _validate_tensor(
    value: torch.Tensor, shape: tuple[int, ...], dtype: torch.dtype
) -> None:
    if (
        tuple(value.shape) != shape
        or value.dtype != dtype
        or value.device.type != "cuda"
        or not value.is_contiguous()
        or not bool(torch.isfinite(value).all().item())
    ):
        raise ValueError("R3-D1 residual11 tensor contract differs")


def execute_r3d1_residual11_staged_v1(
    incoming: torch.Tensor,
    weight10: torch.Tensor,
    lower25: torch.Tensor,
    upper25: torch.Tensor,
    alpha25: torch.Tensor,
    alpha_map25: torch.Tensor,
    weight8: torch.Tensor,
    bias10: torch.Tensor,
    bias8: torch.Tensor,
    bias_in: torch.Tensor,
    scratch: torch.Tensor,
    output: torch.Tensor,
    bias_out: torch.Tensor,
    *,
    cache: R3D1Residual11ModuleCacheV1,
) -> R3D1Residual11StagedReceiptV1:
    """Launch exactly two staged symbols; all storage is caller-owned."""

    import tvm
    import tvm_ffi

    float_contracts = (
        (incoming, (18_432,)),
        (weight10, (16, 16, 3, 3)),
        (lower25, (6, 1024)),
        (upper25, (6, 1024)),
        (alpha25, (2, 1, 6, 86)),
        (weight8, (16, 16, 3, 3)),
        (bias10, (16,)),
        (bias8, (16,)),
        (bias_in, (6,)),
        (scratch, (6144,)),
        (output, (6144,)),
        (bias_out, (6,)),
    )
    for tensor, shape in float_contracts:
        _validate_tensor(tensor, shape, torch.float32)
    _validate_tensor(alpha_map25, (1024,), torch.int32)
    tensors = (
        incoming,
        weight10,
        lower25,
        upper25,
        alpha25,
        alpha_map25,
        weight8,
        bias10,
        bias8,
        bias_in,
        scratch,
        output,
        bias_out,
    )
    if len({tensor.data_ptr() for tensor in (incoming, scratch, output)}) != 3:
        raise ValueError("R3-D1 residual11 arena alias differs")
    device = incoming.device
    if any(tensor.device != device for tensor in tensors):
        raise ValueError("R3-D1 residual11 device differs")
    current = torch.cuda.current_stream(device)
    if int(current.cuda_stream) == int(torch.cuda.default_stream(device).cuda_stream):
        raise RuntimeError("R3-D1 residual11 non-default stream is required")
    ordinal = device.index if device.index is not None else torch.cuda.current_device()
    compiled = cache.get()
    views: dict[int, Any] = {
        tensor.data_ptr(): tvm.runtime.from_dlpack(tensor) for tensor in tensors
    }
    if len(views) != len(tensors):
        raise ValueError("R3-D1 residual11 DLPack pointer identity differs")
    with tvm_ffi.use_torch_stream(torch.cuda.stream(current)):
        ffi_stream = int(tvm_ffi.get_raw_stream(tvm_ffi.device(f"cuda:{ordinal}")))
        if ffi_stream != int(current.cuda_stream):
            raise RuntimeError("R3-D1 residual11 current stream differs")
        compiled.executable[R3D1_RESIDUAL11_STAGE1_SYMBOL](
            views[incoming.data_ptr()],
            views[weight10.data_ptr()],
            views[scratch.data_ptr()],
        )
        compiled.executable[R3D1_RESIDUAL11_STAGE2_SYMBOL](
            views[incoming.data_ptr()],
            views[scratch.data_ptr()],
            views[lower25.data_ptr()],
            views[upper25.data_ptr()],
            views[alpha25.data_ptr()],
            views[alpha_map25.data_ptr()],
            views[weight8.data_ptr()],
            views[bias10.data_ptr()],
            views[bias8.data_ptr()],
            views[bias_in.data_ptr()],
            views[output.data_ptr()],
            views[bias_out.data_ptr()],
        )
    receipt = R3D1Residual11StagedReceiptV1(
        unscheduled_tir_hash=compiled.unscheduled_tir_hash,
        scheduled_tir_hash=compiled.scheduled_tir_hash,
        device_source_hash=compiled.device_source_hash,
        exported_symbols=compiled.exported_symbols,
        stream_id=int(current.cuda_stream),
        tvm_ffi_stream_id=ffi_stream,
        launch_count=2,
        dlpack_pointer_count=len(tensors),
        dlpack_pointer_exact_count=len(views),
        scratch_count=1,
        persistent_dense_a=False,
        fallback_count=0,
        timing_recorded=False,
        performance_claimed=False,
    )
    receipt.validate()
    return receipt


__all__ = [
    "R3D1Residual11ModuleCacheV1",
    "R3D1Residual11StagedReceiptV1",
    "execute_r3d1_residual11_staged_v1",
]
