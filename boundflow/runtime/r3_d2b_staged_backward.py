"""D2-B coefficient-sign candidate with staged residual reuse."""

# mypy: disable-error-code=import-untyped
# pylint: disable=import-error,protected-access,too-many-instance-attributes
# pylint: disable=too-many-arguments,too-many-positional-arguments
# pylint: disable=missing-function-docstring,too-many-boolean-expressions
# pylint: disable=line-too-long

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from boundflow.backends.tvm.r3_d1c_wrapper_schedule import (
    R3D1C_RESIDUAL11_STAGE1,
    R3D1C_RESIDUAL11_STAGE2,
    R3D1C_RESIDUAL6_STAGE1,
    R3D1C_RESIDUAL6_STAGE2,
    R3D1C_SYMBOLS,
)
from boundflow.backends.tvm.r3_full_lower_forward import (
    R31B1_RESIDUAL11_SYMBOL,
    R31B1_RESIDUAL6_SYMBOL,
)
from boundflow.runtime.r3_d1c_cumulative_wrapper import (
    PreparedR3D1CCumulativeCandidateV1,
)


@dataclass(frozen=True)
class R3D2BStagedBackwardReceiptV1:
    """Correctness-only ownership receipt for one final evaluation."""

    scheduled_tir_hash: str
    device_source_hash: str
    exported_symbols: tuple[str, ...]
    forward_staged_launch_count: int
    backward_staged_launch_count: int
    raw_b1_backward_launch_count: int
    backward_bias_inplace_alias_count: int
    existing_arena_count: int
    scratch_region_count: int
    scratch_region_pointers: tuple[int, int]
    persistent_dense_a: bool
    saved_autograd_history: bool
    global_workspace_bytes: int
    fallback_count: int
    eager_candidate_count: int
    native_shadow_count: int
    timing_recorded: bool
    performance_claimed: bool

    def validate(self) -> None:
        hashes = (self.scheduled_tir_hash, self.device_source_hash)
        if (
            any(
                len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in hashes
            )
            or self.exported_symbols != R3D1C_SYMBOLS
            or self.forward_staged_launch_count != 4
            or self.backward_staged_launch_count != 4
            or self.raw_b1_backward_launch_count != 13
            or self.backward_bias_inplace_alias_count != 2
            or self.existing_arena_count != 2
            or self.scratch_region_count != 2
            or len(set(self.scratch_region_pointers)) != 2
            or any(pointer <= 0 for pointer in self.scratch_region_pointers)
            or self.persistent_dense_a
            or self.saved_autograd_history
            or self.global_workspace_bytes
            or self.fallback_count
            or self.eager_candidate_count
            or self.native_shadow_count
            or self.timing_recorded
            or self.performance_claimed
        ):
            raise ValueError("R3-D2B staged backward receipt differs")


class PreparedR3D2BStagedBackwardCandidateV1(PreparedR3D1CCumulativeCandidateV1):
    """D1-C candidate with only two coefficient-sign residuals replaced."""

    def __init__(self, plan, trace, tensors: tuple[torch.Tensor, ...]) -> None:  # type: ignore[no-untyped-def]
        super().__init__(plan, trace, tensors)
        self.d2b_backward_staged_launch_count = 0
        self.d2b_backward_bias_inplace_alias_count = 0

    def begin_evaluation(self, ordinal: int) -> None:
        super().begin_evaluation(ordinal)
        self.d2b_backward_staged_launch_count = 0
        self.d2b_backward_bias_inplace_alias_count = 0

    def _launch_d2b(self, symbol: str, *tensors: torch.Tensor) -> None:
        self.d1c_compiled.executable[symbol](
            *(self._view(tensor) for tensor in tensors)
        )
        self.d2b_backward_staged_launch_count += 1

    def _dispatch_d2b_residual11(self, tensors: tuple[torch.Tensor, ...]) -> None:
        if len(tensors) != 11:
            raise ValueError("R3-D2B residual11 ABI differs")
        (
            incoming,
            weight10,
            bias10,
            lower25,
            upper25,
            alpha25,
            alpha_map25,
            weight8,
            bias8,
            bias_acc,
            output,
        ) = tensors
        self._launch_d2b(
            R3D1C_RESIDUAL11_STAGE1,
            incoming,
            weight10,
            self._residual11_scratch,
        )
        self._launch_d2b(
            R3D1C_RESIDUAL11_STAGE2,
            incoming,
            self._residual11_scratch,
            lower25,
            upper25,
            alpha25,
            alpha_map25,
            weight8,
            bias10,
            bias8,
            bias_acc,
            output[:6144],
            bias_acc,
        )
        self.d2b_backward_bias_inplace_alias_count += 1

    def _dispatch_d2b_residual6(self, tensors: tuple[torch.Tensor, ...]) -> None:
        if len(tensors) != 13:
            raise ValueError("R3-D2B residual6 ABI differs")
        (
            incoming,
            weight4,
            bias4,
            lower19,
            upper19,
            alpha19,
            alpha_map19,
            weight2,
            bias2,
            weight5,
            bias5,
            bias_acc,
            output,
        ) = tensors
        self._launch_d2b(
            R3D1C_RESIDUAL6_STAGE1,
            incoming,
            weight4,
            self._residual6_scratch,
        )
        self._launch_d2b(
            R3D1C_RESIDUAL6_STAGE2,
            incoming,
            self._residual6_scratch,
            lower19,
            upper19,
            alpha19,
            alpha_map19,
            weight2,
            weight5,
            bias4,
            bias2,
            bias5,
            bias_acc,
            output[:12288],
            bias_acc,
        )
        self.d2b_backward_bias_inplace_alias_count += 1

    def _coefficient_sign_pass(
        self, s0: torch.Tensor, s1: torch.Tensor, bias: torch.Tensor
    ) -> None:
        original: Callable[..., None] = self._launch_b1

        def dispatch(symbol: str, *tensors: torch.Tensor) -> None:
            if symbol == R31B1_RESIDUAL11_SYMBOL:
                self._dispatch_d2b_residual11(tensors)
            elif symbol == R31B1_RESIDUAL6_SYMBOL:
                self._dispatch_d2b_residual6(tensors)
            else:
                original(symbol, *tensors)

        self._launch_b1 = dispatch  # type: ignore[method-assign]
        try:
            super()._coefficient_sign_pass(s0, s1, bias)
        finally:
            self._launch_b1 = original  # type: ignore[method-assign]
        if (
            self.d2b_backward_staged_launch_count != 4
            or self.d2b_backward_bias_inplace_alias_count != 2
            or self.b1_backward_launch_count != 8
        ):
            raise RuntimeError("R3-D2B coefficient-sign execution count differs")

    def d2b_receipt(self) -> R3D2BStagedBackwardReceiptV1:
        receipt = R3D2BStagedBackwardReceiptV1(
            scheduled_tir_hash=self.d1c_compiled.scheduled_tir_hash,
            device_source_hash=self.d1c_compiled.device_source_hash,
            exported_symbols=self.d1c_compiled.exported_symbols,
            forward_staged_launch_count=self.d1c_launch_count,
            backward_staged_launch_count=self.d2b_backward_staged_launch_count,
            raw_b1_backward_launch_count=self.b1_backward_launch_count,
            backward_bias_inplace_alias_count=(
                self.d2b_backward_bias_inplace_alias_count
            ),
            existing_arena_count=2,
            scratch_region_count=2,
            scratch_region_pointers=(
                self._residual11_scratch.data_ptr(),
                self._residual6_scratch.data_ptr(),
            ),
            persistent_dense_a=False,
            saved_autograd_history=False,
            global_workspace_bytes=self.d1c_compiled.global_workspace_bytes,
            fallback_count=0,
            eager_candidate_count=0,
            native_shadow_count=0,
            timing_recorded=False,
            performance_claimed=False,
        )
        receipt.validate()
        return receipt


__all__ = [
    "PreparedR3D2BStagedBackwardCandidateV1",
    "R3D2BStagedBackwardReceiptV1",
]
