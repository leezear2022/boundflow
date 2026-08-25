"""D1-C cumulative 10/9 wrapper candidate with staged residual schedules."""

# mypy: disable-error-code=import-untyped
# pylint: disable=import-error,import-outside-toplevel,protected-access
# pylint: disable=too-many-instance-attributes,too-many-locals
# pylint: disable=too-many-arguments,too-many-positional-arguments
# pylint: disable=missing-function-docstring,too-many-boolean-expressions,line-too-long
# pylint: disable=duplicate-code

from __future__ import annotations

from dataclasses import dataclass
import torch

from boundflow.backends.tvm.r3_d1c_wrapper_schedule import (
    CompiledR3D1CWrapperScheduleV1,
    compile_r3d1c_wrapper_schedule_v1,
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
from boundflow.runtime.r3_optimizer_trajectory_timing import (
    PreparedR32BTimingCandidateV1,
)


@dataclass(frozen=True)
class R3D1CCumulativeReceiptV1:
    """Static schedule and warm ownership receipt for one evaluation."""

    scheduled_tir_hash: str
    device_source_hash: str
    exported_symbols: tuple[str, ...]
    threads_per_block: int
    reduction_kind: str
    vector_width: int
    launch_count: int
    existing_arena_count: int
    scratch_region_count: int
    scratch_region_pointers: tuple[int, int]
    bias_inplace_alias_count: int
    persistent_dense_a: bool
    global_workspace_bytes: int
    fallback_count: int
    eager_candidate_count: int
    native_shadow_count: int
    wrapper_performance_claimed: bool

    def validate(self) -> None:
        hashes = (self.scheduled_tir_hash, self.device_source_hash)
        if (
            any(
                len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in hashes
            )
            or self.exported_symbols != R3D1C_SYMBOLS
            or self.threads_per_block != 256
            or self.reduction_kind != "serial-reference"
            or self.vector_width != 1
            or self.launch_count != 4
            or self.existing_arena_count != 2
            or self.scratch_region_count != 2
            or len(set(self.scratch_region_pointers)) != 2
            or any(pointer <= 0 for pointer in self.scratch_region_pointers)
            or self.bias_inplace_alias_count != 2
            or self.persistent_dense_a
            or self.global_workspace_bytes
            or self.fallback_count
            or self.eager_candidate_count
            or self.native_shadow_count
            or self.wrapper_performance_claimed
        ):
            raise ValueError("R3-D1C cumulative receipt differs")


class PreparedR3D1CCumulativeCandidateV1(PreparedR32BTimingCandidateV1):
    """R3-2B candidate with only residual11/residual6 forward symbols replaced."""

    def __init__(self, plan, trace, tensors: tuple[torch.Tensor, ...]) -> None:  # type: ignore[no-untyped-def]
        super().__init__(plan, trace, tensors)
        import tvm

        self.d1c_compiled: CompiledR3D1CWrapperScheduleV1 = (
            compile_r3d1c_wrapper_schedule_v1()
        )
        self._residual11_scratch = self.forward_executor.scratch_1[6144:12288]
        self._residual6_scratch = self.forward_executor.scratch_0[12288:18432]
        self._register_view(tvm, self._residual11_scratch)
        self._register_view(tvm, self._residual6_scratch)
        self.d1c_launch_count = 0
        self.d1c_bias_inplace_alias_count = 0

    def begin_evaluation(self, ordinal: int) -> None:
        super().begin_evaluation(ordinal)
        self.d1c_launch_count = 0
        self.d1c_bias_inplace_alias_count = 0

    def _launch_d1c(self, symbol: str, *tensors: torch.Tensor) -> None:
        self.d1c_compiled.executable[symbol](
            *(self._view(tensor) for tensor in tensors)
        )
        self.d1c_launch_count += 1
        self.forward_executor.launch_count += 1

    def _dispatch_residual11(self, tensors: tuple[torch.Tensor, ...]) -> None:
        if len(tensors) != 11:
            raise ValueError("R3-D1C residual11 ABI differs")
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
        self._launch_d1c(
            R3D1C_RESIDUAL11_STAGE1,
            incoming,
            weight10,
            self._residual11_scratch,
        )
        self._launch_d1c(
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
        self.d1c_bias_inplace_alias_count += 1

    def _dispatch_residual6(self, tensors: tuple[torch.Tensor, ...]) -> None:
        if len(tensors) != 13:
            raise ValueError("R3-D1C residual6 ABI differs")
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
        self._launch_d1c(
            R3D1C_RESIDUAL6_STAGE1,
            incoming,
            weight4,
            self._residual6_scratch,
        )
        self._launch_d1c(
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
        self.d1c_bias_inplace_alias_count += 1

    def _run_forward_fast(self) -> torch.Tensor:
        executor = self.forward_executor
        original = executor._launch

        def dispatch(symbol: str, *tensors: torch.Tensor) -> None:
            if symbol == R31B1_RESIDUAL11_SYMBOL:
                self._dispatch_residual11(tensors)
            elif symbol == R31B1_RESIDUAL6_SYMBOL:
                self._dispatch_residual6(tensors)
            else:
                original(symbol, *tensors)

        executor._launch = dispatch  # type: ignore[method-assign]
        try:
            result = super()._run_forward_fast()
        finally:
            executor._launch = original  # type: ignore[method-assign]
        if (
            self.d1c_launch_count != 4
            or self.d1c_bias_inplace_alias_count != 2
            or executor.launch_count != 17
        ):
            raise RuntimeError("R3-D1C forward execution count differs")
        return result

    def d1c_receipt(self) -> R3D1CCumulativeReceiptV1:
        receipt = R3D1CCumulativeReceiptV1(
            scheduled_tir_hash=self.d1c_compiled.scheduled_tir_hash,
            device_source_hash=self.d1c_compiled.device_source_hash,
            exported_symbols=self.d1c_compiled.exported_symbols,
            threads_per_block=self.d1c_compiled.threads_per_block,
            reduction_kind=self.d1c_compiled.reduction_kind,
            vector_width=self.d1c_compiled.vector_width,
            launch_count=self.d1c_launch_count,
            existing_arena_count=2,
            scratch_region_count=2,
            scratch_region_pointers=(
                self._residual11_scratch.data_ptr(),
                self._residual6_scratch.data_ptr(),
            ),
            bias_inplace_alias_count=self.d1c_bias_inplace_alias_count,
            persistent_dense_a=False,
            global_workspace_bytes=self.d1c_compiled.global_workspace_bytes,
            fallback_count=0,
            eager_candidate_count=0,
            native_shadow_count=0,
            wrapper_performance_claimed=False,
        )
        receipt.validate()
        return receipt


__all__ = ["PreparedR3D1CCumulativeCandidateV1", "R3D1CCumulativeReceiptV1"]
