"""Zero-copy current-stream runtime for D1-A residual6 qualification."""

# mypy: disable-error-code=import-untyped
# pylint: disable=import-error,import-outside-toplevel,too-many-instance-attributes
# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
# pylint: disable=missing-function-docstring,too-few-public-methods
# pylint: disable=too-many-boolean-expressions
# pylint: disable=duplicate-code

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from boundflow.backends.tvm.r3_d1_residual6_staged import (
    CompiledR3D1Residual6StagedV1,
    compile_r3d1_residual6_staged_v1,
    R3D1_RESIDUAL6_STAGE1_SYMBOL,
    R3D1_RESIDUAL6_STAGE2_SYMBOL,
    R3D1_RESIDUAL6_SYMBOLS,
)


@dataclass(frozen=True)
class R3D1Residual6StagedReceiptV1:
    """Correctness-only residual6 ABI and ownership receipt."""

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
            or self.exported_symbols != R3D1_RESIDUAL6_SYMBOLS
            or self.stream_id <= 0
            or self.tvm_ffi_stream_id != self.stream_id
            or self.launch_count != 2
            or self.dlpack_pointer_count != 15
            or self.dlpack_pointer_exact_count != self.dlpack_pointer_count
            or self.scratch_count != 1
            or self.persistent_dense_a
            or self.fallback_count
            or self.timing_recorded
            or self.performance_claimed
        ):
            raise ValueError("R3-D1 residual6 receipt differs")


class R3D1Residual6ModuleCacheV1:
    """Tensor-free compile cache for isolated residual6."""

    def __init__(self) -> None:
        self._compiled: CompiledR3D1Residual6StagedV1 | None = None

    def get(self) -> CompiledR3D1Residual6StagedV1:
        if self._compiled is None:
            self._compiled = compile_r3d1_residual6_staged_v1()
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
        raise ValueError("R3-D1 residual6 tensor contract differs")


def execute_r3d1_residual6_staged_v1(
    incoming: torch.Tensor,
    weight4: torch.Tensor,
    lower19: torch.Tensor,
    upper19: torch.Tensor,
    alpha19: torch.Tensor,
    alpha_map19: torch.Tensor,
    weight2: torch.Tensor,
    weight5: torch.Tensor,
    bias4: torch.Tensor,
    bias2: torch.Tensor,
    bias5: torch.Tensor,
    bias_in: torch.Tensor,
    scratch: torch.Tensor,
    output: torch.Tensor,
    bias_out: torch.Tensor,
    *,
    cache: R3D1Residual6ModuleCacheV1,
) -> R3D1Residual6StagedReceiptV1:
    """Launch exactly two staged residual6 symbols with caller-owned storage."""

    import tvm
    import tvm_ffi

    float_contracts = (
        (incoming, (18_432,)),
        (weight4, (16, 16, 3, 3)),
        (lower19, (6, 1024)),
        (upper19, (6, 1024)),
        (alpha19, (2, 1, 6, 132)),
        (weight2, (16, 8, 3, 3)),
        (weight5, (16, 8, 1, 1)),
        (bias4, (16,)),
        (bias2, (16,)),
        (bias5, (16,)),
        (bias_in, (6,)),
        (scratch, (6144,)),
        (output, (12_288,)),
        (bias_out, (6,)),
    )
    for tensor, shape in float_contracts:
        _validate_tensor(tensor, shape, torch.float32)
    _validate_tensor(alpha_map19, (1024,), torch.int32)
    tensors = (
        incoming,
        weight4,
        lower19,
        upper19,
        alpha19,
        alpha_map19,
        weight2,
        weight5,
        bias4,
        bias2,
        bias5,
        bias_in,
        scratch,
        output,
        bias_out,
    )
    if len({tensor.data_ptr() for tensor in (incoming, scratch, output)}) != 3:
        raise ValueError("R3-D1 residual6 arena alias differs")
    device = incoming.device
    if any(tensor.device != device for tensor in tensors):
        raise ValueError("R3-D1 residual6 device differs")
    current = torch.cuda.current_stream(device)
    if int(current.cuda_stream) == int(torch.cuda.default_stream(device).cuda_stream):
        raise RuntimeError("R3-D1 residual6 non-default stream is required")
    ordinal = device.index if device.index is not None else torch.cuda.current_device()
    compiled = cache.get()
    views: dict[int, Any] = {
        tensor.data_ptr(): tvm.runtime.from_dlpack(tensor) for tensor in tensors
    }
    if len(views) != len(tensors):
        raise ValueError("R3-D1 residual6 DLPack pointer identity differs")
    with tvm_ffi.use_torch_stream(torch.cuda.stream(current)):
        ffi_stream = int(tvm_ffi.get_raw_stream(tvm_ffi.device(f"cuda:{ordinal}")))
        if ffi_stream != int(current.cuda_stream):
            raise RuntimeError("R3-D1 residual6 current stream differs")
        compiled.executable[R3D1_RESIDUAL6_STAGE1_SYMBOL](
            views[incoming.data_ptr()],
            views[weight4.data_ptr()],
            views[scratch.data_ptr()],
        )
        compiled.executable[R3D1_RESIDUAL6_STAGE2_SYMBOL](
            views[incoming.data_ptr()],
            views[scratch.data_ptr()],
            views[lower19.data_ptr()],
            views[upper19.data_ptr()],
            views[alpha19.data_ptr()],
            views[alpha_map19.data_ptr()],
            views[weight2.data_ptr()],
            views[weight5.data_ptr()],
            views[bias4.data_ptr()],
            views[bias2.data_ptr()],
            views[bias5.data_ptr()],
            views[bias_in.data_ptr()],
            views[output.data_ptr()],
            views[bias_out.data_ptr()],
        )
    receipt = R3D1Residual6StagedReceiptV1(
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
    "R3D1Residual6ModuleCacheV1",
    "R3D1Residual6StagedReceiptV1",
    "execute_r3d1_residual6_staged_v1",
]
