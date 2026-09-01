"""Persistent compact-state coefficient recurrence for S4 evaluations."""

# mypy: disable-error-code=import-untyped
# pylint: disable=import-error,import-outside-toplevel,protected-access
# pylint: disable=too-many-instance-attributes,too-many-locals,too-many-arguments
# pylint: disable=too-many-positional-arguments,missing-function-docstring
# pylint: disable=too-few-public-methods
# pylint: disable=duplicate-code

from __future__ import annotations

from typing import Any

import torch

from boundflow.backends.tvm.asplos27_s4_compact_coefficient import (
    CompiledS4CompactCoefficientV1,
    S4_COMPACT_RELU17,
    S4_COMPACT_RELU17_BIAS,
    S4_COMPACT_RELU23,
    S4_COMPACT_RELU23_BIAS,
    S4_COMPACT_RELU28,
    S4_COMPACT_RELU28_BIAS,
    S4_COMPACT_RELU31,
    S4_COMPACT_RELU31_BIAS,
    S4_COMPACT_RESIDUAL11_STAGE2,
    S4_COMPACT_RESIDUAL6_STAGE2,
    compile_s4_compact_coefficient_v1,
)
from boundflow.backends.tvm.asplos27_s4_six_site_value import (
    CompiledS4SelectorPackV1,
)
from boundflow.backends.tvm.r3_d1c_wrapper_schedule import (
    R3D1C_RESIDUAL11_STAGE1,
    R3D1C_RESIDUAL6_STAGE1,
)
from boundflow.backends.tvm.r3_full_lower_forward import (
    R31B1_CONCRETIZE_SYMBOL,
    R31B1_CONV0_SYMBOL,
    R31B1_LINEAR14_SYMBOL,
    R31B1_LINEAR16_SYMBOL,
    R31B1_SEED_SYMBOL,
)
from boundflow.backends.tvm.r3_p_alpha_vjp import R31B2_CLEAR_SYMBOL
from boundflow.runtime.asplos27_s4_coefficient_selector_pass import (
    PreparedS4CoefficientSelectorPassV1,
)
from boundflow.runtime.asplos27_s4_ordered_buffer_abi import (
    PreparedS4MutableBuffersV1,
)
from boundflow.runtime.r3_d2b_staged_backward import (
    PreparedR3D2BStagedBackwardCandidateV1,
)

S4_COMPACT_SITE_ORDER = (17, 19, 23, 25, 28, 31)


class S4CompactCoefficientError(RuntimeError):
    """Fail-closed compact recurrence error."""


class _S4CompactCoefficientCacheV1:
    def __init__(self) -> None:
        self._entries: dict[str, CompiledS4CompactCoefficientV1] = {}

    def get(self, capability: str) -> CompiledS4CompactCoefficientV1:
        compiled = self._entries.get(capability)
        if compiled is None:
            compiled = compile_s4_compact_coefficient_v1(compute_capability=capability)
            self._entries[capability] = compiled
        return compiled


_S4_COMPACT_CACHE = _S4CompactCoefficientCacheV1()


class PreparedS4CompactCoefficientV1:
    """Directly bind six compact alpha buffers and one compact beta buffer."""

    def __init__(
        self,
        executor: PreparedR3D2BStagedBackwardCandidateV1,
        mutable_buffers: PreparedS4MutableBuffersV1,
        *,
        compiled: CompiledS4CompactCoefficientV1 | None = None,
    ) -> None:
        import tvm

        resources = mutable_buffers._resources
        if resources is None or len(resources._parameters) != 7:
            raise S4CompactCoefficientError("S4_COMPACT_PARAMETER_OWNER_MISMATCH")
        device = executor.device
        if device.type != "cuda" or any(
            value.device != device for value in resources._parameters
        ):
            raise S4CompactCoefficientError("S4_COMPACT_DEVICE_MISMATCH")
        capability = torch.cuda.get_device_capability(device)
        capability_name = f"sm_{capability[0]}{capability[1]}"
        self.compiled = compiled or _S4_COMPACT_CACHE.get(capability_name)
        self.compiled.validate()
        self.executor = executor
        self.mutable_buffers = mutable_buffers
        self.active_alpha = tuple(resources._parameters[:6])
        self.active_beta = resources._parameters[6]
        self.lower_output = resources._lower
        if self.lower_output is None:
            raise S4CompactCoefficientError("S4_COMPACT_LOWER_OWNER_MISMATCH")
        self.device = device
        self._site_to_ordinal = {
            site: ordinal for ordinal, site in enumerate(S4_COMPACT_SITE_ORDER)
        }
        self._views: dict[tuple[int, tuple[int, ...], str], Any] = {}
        tensors: list[torch.Tensor] = [*self.active_alpha, self.active_beta]
        for site in S4_COMPACT_SITE_ORDER:
            tensors.extend(
                (
                    executor._tensor(f"relu/{site}/lower").reshape(6, -1),
                    executor._tensor(f"relu/{site}/upper").reshape(6, -1),
                    executor.forward_executor.alpha_maps[str(site)],
                )
            )
        tensors.extend(
            (
                executor.forward_executor.beta_maps["31"],
                executor.forward_executor.split_maps["31"],
                executor._tensor("param/layer1.1.conv1.weight"),
                executor._tensor("param/layer1.1.conv2.bias"),
                executor._tensor("param/layer1.1.conv1.bias"),
                executor._tensor("param/layer1.0.conv1.weight"),
                executor._tensor("param/layer1.0.shortcut.0.weight"),
                executor._tensor("param/layer1.0.conv2.bias"),
                executor._tensor("param/layer1.0.conv1.bias"),
                executor._tensor("param/layer1.0.shortcut.0.bias"),
                executor.forward_executor.scratch_0,
                executor.forward_executor.scratch_1,
                executor.forward_executor.bias_accumulator,
                executor._residual11_scratch,
                executor._residual6_scratch,
            )
        )
        for tensor in tensors:
            for candidate in (tensor,):
                key = self._key(candidate)
                if key not in self._views:
                    self._views[key] = tvm.runtime.from_dlpack(candidate)
        executor._register_view(tvm, self.lower_output)
        for scratch in (
            executor.forward_executor.scratch_0,
            executor.forward_executor.scratch_1,
        ):
            for length in (60, 600, 6144, 12288, 18432):
                candidate = scratch[:length]
                key = self._key(candidate)
                if key not in self._views:
                    self._views[key] = tvm.runtime.from_dlpack(candidate)
        self.launch_count = 0

    @staticmethod
    def _key(tensor: torch.Tensor) -> tuple[int, tuple[int, ...], str]:
        return tensor.data_ptr(), tuple(tensor.shape), str(tensor.dtype)

    def _view(self, tensor: torch.Tensor) -> Any:
        value = self._views.get(self._key(tensor))
        if value is None:
            raise S4CompactCoefficientError("S4_COMPACT_WARM_VIEW_ABSENT")
        return value

    def _launch(self, symbol: str, *tensors: torch.Tensor) -> None:
        self.compiled.executable[symbol](*(self._view(tensor) for tensor in tensors))
        self.launch_count += 1

    def relu(self, site: int, arena: torch.Tensor) -> None:
        symbols = {
            31: S4_COMPACT_RELU31,
            28: S4_COMPACT_RELU28,
            23: S4_COMPACT_RELU23,
            17: S4_COMPACT_RELU17,
        }
        bias_symbols = {
            31: S4_COMPACT_RELU31_BIAS,
            28: S4_COMPACT_RELU28_BIAS,
            23: S4_COMPACT_RELU23_BIAS,
            17: S4_COMPACT_RELU17_BIAS,
        }
        symbol = symbols.get(site)
        bias_symbol = bias_symbols.get(site)
        ordinal = self._site_to_ordinal.get(site)
        if symbol is None or bias_symbol is None or ordinal is None:
            raise S4CompactCoefficientError("S4_COMPACT_RELU_SITE_MISMATCH")
        executor = self.executor
        lower = executor._tensor(f"relu/{site}/lower").reshape(6, -1)
        upper = executor._tensor(f"relu/{site}/upper").reshape(6, -1)
        alpha = self.active_alpha[ordinal]
        alpha_map = executor.forward_executor.alpha_maps[str(site)]
        bias = executor.forward_executor.bias_accumulator
        self._launch(
            bias_symbol,
            arena,
            lower,
            upper,
            alpha,
            alpha_map,
            bias,
            bias,
        )
        arguments = [
            arena,
            lower,
            upper,
            alpha,
            alpha_map,
        ]
        if site == 31:
            arguments.extend(
                (
                    self.active_beta,
                    executor.forward_executor.beta_maps["31"],
                    executor.forward_executor.split_maps["31"],
                )
            )
        arguments.append(arena)
        self._launch(symbol, *arguments)

    def residual11_stage2(
        self,
        incoming: torch.Tensor,
        staged: torch.Tensor,
        output: torch.Tensor,
        bias: torch.Tensor,
    ) -> None:
        executor = self.executor
        self._launch(
            S4_COMPACT_RESIDUAL11_STAGE2,
            incoming,
            staged,
            executor._tensor("relu/25/lower").reshape(6, 1024),
            executor._tensor("relu/25/upper").reshape(6, 1024),
            self.active_alpha[self._site_to_ordinal[25]],
            executor.forward_executor.alpha_maps["25"],
            executor._tensor("param/layer1.1.conv1.weight"),
            executor._tensor("param/layer1.1.conv2.bias"),
            executor._tensor("param/layer1.1.conv1.bias"),
            bias,
            output,
            bias,
        )

    def residual6_stage2(
        self,
        incoming: torch.Tensor,
        staged: torch.Tensor,
        output: torch.Tensor,
        bias: torch.Tensor,
    ) -> None:
        executor = self.executor
        self._launch(
            S4_COMPACT_RESIDUAL6_STAGE2,
            incoming,
            staged,
            executor._tensor("relu/19/lower").reshape(6, 1024),
            executor._tensor("relu/19/upper").reshape(6, 1024),
            self.active_alpha[self._site_to_ordinal[19]],
            executor.forward_executor.alpha_maps["19"],
            executor._tensor("param/layer1.0.conv1.weight"),
            executor._tensor("param/layer1.0.shortcut.0.weight"),
            executor._tensor("param/layer1.0.conv2.bias"),
            executor._tensor("param/layer1.0.conv1.bias"),
            executor._tensor("param/layer1.0.shortcut.0.bias"),
            bias,
            output,
            bias,
        )

    def capture_selectors(
        self,
        owner: PreparedS4CoefficientSelectorPassV1,
        *,
        compiled_selector: CompiledS4SelectorPackV1 | None = None,
    ) -> None:
        """Run Pass A from the active compact parameters."""

        executor = self.executor
        forward = executor.forward_executor
        s0, s1, bias = forward.scratch_0, forward.scratch_1, forward.bias_accumulator
        if not owner.has_compiled_binding:
            owner.bind_compiled_sources(
                {
                    "pack_a29": s0[:6144],
                    "pack_a26": executor._residual11_scratch,
                    "pack_a24": s1[:6144],
                    "pack_a20": executor._residual6_scratch,
                    "pack_a18": s0[:12288],
                    "pack_ainput": s1[:18432],
                },
                compiled=compiled_selector,
            )
        executor._launch_b2(R31B2_CLEAR_SYMBOL, s0, s1)
        owner.begin()
        executor._launch_b1(
            R31B1_SEED_SYMBOL, executor._tensor("objective"), s0[:60], bias
        )
        owner.record("seed")
        executor._launch_b1(
            R31B1_LINEAR16_SYMBOL,
            s0[:60],
            executor._tensor("param/linear2.weight"),
            executor._tensor("param/linear2.bias"),
            bias,
            s1[:600],
            bias,
        )
        owner.record("linear16_right")
        self.relu(31, s1[:600])
        owner.record("relu31_coefficient")
        executor._launch_b1(
            R31B1_LINEAR14_SYMBOL,
            s1[:600],
            executor._tensor("param/linear1.weight"),
            executor._tensor("param/linear1.bias"),
            bias,
            s0[:6144],
            bias,
        )
        owner.record("linear14_right")
        owner.record("pack_a29", s0[:6144])
        self.relu(28, s0[:6144])
        owner.record("relu28_coefficient")
        executor._launch_d2b(
            R3D1C_RESIDUAL11_STAGE1,
            s0,
            executor._tensor("param/layer1.1.conv2.weight"),
            executor._residual11_scratch,
        )
        owner.record("residual11_stage1")
        owner.record("pack_a26", executor._residual11_scratch)
        self.residual11_stage2(s0, executor._residual11_scratch, s1[:6144], bias)
        owner.record("residual11_stage2")
        owner.record("pack_a24", s1[:6144])
        self.relu(23, s1[:6144])
        owner.record("relu23_coefficient")
        executor._launch_d2b(
            R3D1C_RESIDUAL6_STAGE1,
            s1,
            executor._tensor("param/layer1.0.conv2.weight"),
            executor._residual6_scratch,
        )
        owner.record("residual6_stage1")
        owner.record("pack_a20", executor._residual6_scratch)
        self.residual6_stage2(s1, executor._residual6_scratch, s0[:12288], bias)
        owner.record("residual6_stage2")
        owner.record("pack_a18", s0[:12288])
        self.relu(17, s0[:12288])
        owner.record("relu17_coefficient")
        executor._launch_b1(
            R31B1_CONV0_SYMBOL,
            s0[:12288],
            executor._tensor("param/conv1.weight"),
            executor._tensor("param/conv1.bias"),
            bias,
            s1,
            bias,
        )
        owner.record("conv0_right")
        owner.record("pack_ainput", s1[:18432])
        if self.lower_output is None:
            raise S4CompactCoefficientError("S4_COMPACT_LOWER_OWNER_MISMATCH")
        executor._launch_b1(
            R31B1_CONCRETIZE_SYMBOL,
            s1,
            executor._tensor("input/lower").reshape(6, 3072),
            executor._tensor("input/upper").reshape(6, 3072),
            bias,
            self.lower_output,
        )
        owner.record("box_concretize")


__all__ = ["PreparedS4CompactCoefficientV1", "S4CompactCoefficientError"]
