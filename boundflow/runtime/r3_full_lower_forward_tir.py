"""Zero-copy current-stream launcher for the frozen R3-1b1 full-lower module."""

# mypy: disable-error-code=import-untyped
# pylint: disable=import-error,import-outside-toplevel,too-many-instance-attributes
# pylint: disable=too-many-locals,too-many-statements,missing-function-docstring
# pylint: disable=too-many-arguments,too-many-positional-arguments
# pylint: disable=missing-class-docstring,too-many-boolean-expressions
# pylint: disable=too-few-public-methods,too-many-branches

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import torch

from boundflow.backends.tvm.r3_full_lower_forward import (
    CompiledR31B1FullLowerForwardV1,
    R31B1_ARENA_CAPACITY,
    R31B1_CONCRETIZE_SYMBOL,
    R31B1_CONV0_SYMBOL,
    R31B1_EXPORTED_SYMBOLS,
    R31B1_LINEAR14_SYMBOL,
    R31B1_LINEAR16_SYMBOL,
    R31B1_RELU17_BIAS_SYMBOL,
    R31B1_RELU17_COEFF_SYMBOL,
    R31B1_RELU23_BIAS_SYMBOL,
    R31B1_RELU23_COEFF_SYMBOL,
    R31B1_RELU28_BIAS_SYMBOL,
    R31B1_RELU28_COEFF_SYMBOL,
    R31B1_RELU31_BIAS_SYMBOL,
    R31B1_RELU31_COEFF_SYMBOL,
    R31B1_RESIDUAL11_SYMBOL,
    R31B1_RESIDUAL6_SYMBOL,
    R31B1_SEED_SYMBOL,
    compile_r31b1_full_lower_forward_tir_v1,
)
from boundflow.ir.r3_bounded_arena import R31BBoundedArenaTraceV1

from .r3_structured_owner_custom_backward import R31FullRegionPlanV1
from .rvir_v4_production_state import production_tensor_sha256

R31B1_LAUNCH_SCHEMA = "boundflow.r3-1b1-full-lower-launch/v1"


def _is_hash(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


@dataclass(frozen=True)
class R31B1CompilationReceiptV1:
    module_hash: str
    device_source_hash: str
    trace_hash: str
    production_plan_hash: str
    exported_symbols: tuple[str, ...]
    global_workspace_bytes: int
    tensor_free_module_cache: bool

    def validate(self) -> None:
        if (
            not all(
                _is_hash(value)
                for value in (
                    self.module_hash,
                    self.device_source_hash,
                    self.trace_hash,
                    self.production_plan_hash,
                )
            )
            or self.exported_symbols != R31B1_EXPORTED_SYMBOLS
            or self.global_workspace_bytes != 0
            or self.tensor_free_module_cache is not True
        ):
            raise ValueError("R3-1b1 compilation receipt differs")


@dataclass(frozen=True)
class R31B1LaunchReceiptV1:
    schema_version: str
    stream_id: int
    tvm_ffi_stream_id: int
    scratch_pointers: tuple[int, int]
    scratch_capacity_elements: tuple[int, int]
    scratch_high_water_elements: tuple[int, int]
    coefficient_scratch_count: int
    launch_count: int
    dlpack_pointer_count: int
    dlpack_pointer_exact_count: int
    warm_dynamic_allocated_bytes: int
    python_visible_intermediate_coefficient_count: int
    fallback_count: int
    eager_count: int
    native_shadow_count: int
    compiled_region: bool
    timing_recorded: bool
    performance_claimed: bool

    def validate(self) -> None:
        if (
            self.schema_version != R31B1_LAUNCH_SCHEMA
            or self.stream_id <= 0
            or self.tvm_ffi_stream_id != self.stream_id
            or len(set(self.scratch_pointers)) != 2
            or any(pointer <= 0 for pointer in self.scratch_pointers)
            or self.scratch_capacity_elements
            != (R31B1_ARENA_CAPACITY, R31B1_ARENA_CAPACITY)
            or self.scratch_high_water_elements
            != (R31B1_ARENA_CAPACITY, R31B1_ARENA_CAPACITY)
            or self.coefficient_scratch_count != 2
            or self.launch_count != len(R31B1_EXPORTED_SYMBOLS)
            or self.dlpack_pointer_count <= 0
            or self.dlpack_pointer_exact_count != self.dlpack_pointer_count
            or self.warm_dynamic_allocated_bytes != 0
            or self.python_visible_intermediate_coefficient_count != 0
            or self.fallback_count != 0
            or self.eager_count != 0
            or self.native_shadow_count != 0
            or self.compiled_region is not True
            or self.timing_recorded
            or self.performance_claimed
        ):
            raise ValueError("R3-1b1 launch receipt differs")


@dataclass(frozen=True)
class R31B1ForwardResultV1:
    lower: torch.Tensor
    compilation_receipt: R31B1CompilationReceiptV1
    launch_receipt: R31B1LaunchReceiptV1

    def validate(self) -> None:
        self.compilation_receipt.validate()
        self.launch_receipt.validate()
        if (
            tuple(self.lower.shape) != (6, 1)
            or self.lower.dtype != torch.float32
            or self.lower.device.type != "cuda"
            or not bool(torch.isfinite(self.lower).all().item())
        ):
            raise ValueError("R3-1b1 forward result differs")


class R31B1ModuleCacheV1:
    """Compiled-code-only cache; instance tensors never enter this object."""

    def __init__(self) -> None:
        self._entries: dict[str, CompiledR31B1FullLowerForwardV1] = {}

    def get(
        self, compute_capability: str
    ) -> tuple[CompiledR31B1FullLowerForwardV1, str]:
        compiled = self._entries.get(compute_capability)
        if compiled is not None:
            return compiled, "hit"
        compiled = compile_r31b1_full_lower_forward_tir_v1(
            compute_capability=compute_capability
        )
        self._entries[compute_capability] = compiled
        return compiled, "miss"


R31B1_MODULE_CACHE = R31B1ModuleCacheV1()


class PreparedR31B1FullLowerForwardV1:
    """Plan-instance tensors, metadata views and exactly two coefficient arenas."""

    def __init__(
        self,
        plan: R31FullRegionPlanV1,
        trace: R31BBoundedArenaTraceV1,
        tensors: tuple[torch.Tensor, ...],
        *,
        compute_capability: str = "sm_89",
        cache: R31B1ModuleCacheV1 = R31B1_MODULE_CACHE,
    ) -> None:
        import tvm

        plan.validate()
        trace.validate()
        if (
            trace.production_plan_hash != plan.stable_hash()
            or len(tensors) != len(plan.tensor_specs)
            or any(tensor.device.type != "cuda" for tensor in tensors)
            or any(
                tuple(tensor.shape) != spec.shape
                or str(tensor.dtype) != spec.dtype
                or production_tensor_sha256(tensor) != spec.content_sha256
                for tensor, spec in zip(tensors, plan.tensor_specs)
            )
        ):
            raise ValueError("R3-1b1 runtime admission differs")
        self.plan = plan
        self.trace = trace
        self.tensors = tensors
        self.device = tensors[0].device
        self.compiled, self.cache_event = cache.get(compute_capability)
        self.scratch_0 = torch.empty(
            R31B1_ARENA_CAPACITY, device=self.device, dtype=torch.float32
        )
        self.scratch_1 = torch.empty_like(self.scratch_0)
        self.bias_accumulator = torch.empty(6, device=self.device, dtype=torch.float32)
        self.output = torch.empty(6, device=self.device, dtype=torch.float32)
        self._by_name = {
            spec.name: tensor for spec, tensor in zip(plan.tensor_specs, tensors)
        }
        self.alpha_maps: dict[str, torch.Tensor] = {}
        self.beta_maps: dict[str, torch.Tensor] = {}
        self.split_maps: dict[str, torch.Tensor] = {}
        for layout in plan.relu_layouts:
            feature_count = 1
            for dimension in layout.feature_shape:
                feature_count *= dimension
            alpha_map = torch.full(
                (feature_count,), -1, dtype=torch.int32, device=self.device
            )
            alpha_map[
                torch.tensor(
                    layout.alpha_flat_indices, dtype=torch.int64, device=self.device
                )
            ] = torch.arange(
                len(layout.alpha_flat_indices), dtype=torch.int32, device=self.device
            )
            self.alpha_maps[layout.native_preactivation] = alpha_map.contiguous()
            if any(layout.beta_locations):
                beta_map = torch.full(
                    (6, feature_count), -1, dtype=torch.int32, device=self.device
                )
                for domain, locations in enumerate(layout.beta_locations):
                    for ordinal, location in enumerate(locations):
                        beta_map[domain, location] = ordinal
                self.beta_maps[layout.native_preactivation] = beta_map.contiguous()
                self.split_maps[layout.native_preactivation] = torch.tensor(
                    layout.split_values, dtype=torch.int8, device=self.device
                ).contiguous()
        if set(self.beta_maps) != {"31"} or set(self.split_maps) != {"31"}:
            raise ValueError("R3-1b1 active beta ownership differs")
        self._metadata_hashes = {
            f"alpha/{name}": production_tensor_sha256(tensor)
            for name, tensor in self.alpha_maps.items()
        }
        self._metadata_hashes.update(
            {
                f"beta/{name}": production_tensor_sha256(tensor)
                for name, tensor in self.beta_maps.items()
            }
        )
        self._metadata_hashes.update(
            {
                f"split/{name}": production_tensor_sha256(tensor)
                for name, tensor in self.split_maps.items()
            }
        )
        self._views: dict[tuple[int, tuple[int, ...]], Any] = {}
        view_tensors = list(tensors) + [
            self.scratch_0,
            self.scratch_1,
            self.bias_accumulator,
            self.output,
            *self.alpha_maps.values(),
            *self.beta_maps.values(),
            *self.split_maps.values(),
        ]
        for tensor in view_tensors:
            if tensor.numel():
                self._register_view(tvm, tensor)
        for name, tensor in self._by_name.items():
            if name in {"input/lower", "input/upper"} or (
                name.startswith("relu/")
                and (name.endswith("/lower") or name.endswith("/upper"))
            ):
                self._register_view(tvm, tensor.reshape(6, -1))
        for scratch in (self.scratch_0, self.scratch_1):
            for length in (60, 600, 6144, 12288, R31B1_ARENA_CAPACITY):
                self._register_view(tvm, scratch[:length])
        self._pointer_count = len(self._views)
        self._pointer_exact_count = 0
        for key, view in self._views.items():
            self._pointer_exact_count += int(
                torch.from_dlpack(view).data_ptr() == key[0]
            )
        if self._pointer_exact_count != self._pointer_count:
            raise RuntimeError("R3-1b1 DLPack pointer differs")
        self.compilation_receipt = R31B1CompilationReceiptV1(
            module_hash=self.compiled.module_hash,
            device_source_hash=self.compiled.device_source_hash,
            trace_hash=trace.stable_hash(),
            production_plan_hash=plan.stable_hash(),
            exported_symbols=self.compiled.exported_symbols,
            global_workspace_bytes=self.compiled.global_workspace_bytes,
            tensor_free_module_cache=True,
        )
        self.compilation_receipt.validate()
        self.launch_count = 0

    def _register_view(self, tvm: Any, tensor: torch.Tensor) -> None:
        key = (tensor.data_ptr(), tuple(int(dimension) for dimension in tensor.shape))
        if key not in self._views:
            self._views[key] = tvm.runtime.from_dlpack(tensor)

    def _view(self, tensor: torch.Tensor) -> Any:
        key = (tensor.data_ptr(), tuple(int(dimension) for dimension in tensor.shape))
        view = self._views.get(key)
        if view is None:
            raise RuntimeError("R3-1b1 warm DLPack view is absent")
        return view

    def _tensor(self, name: str) -> torch.Tensor:
        tensor = self._by_name.get(name)
        if tensor is None:
            raise KeyError(f"R3-1b1 tensor is absent: {name}")
        return tensor

    def _launch(self, symbol: str, *tensors: torch.Tensor) -> None:
        self.compiled.executable[symbol](*(self._view(tensor) for tensor in tensors))
        self.launch_count += 1

    def run(self) -> R31B1ForwardResultV1:  # pylint: disable=too-many-statements
        import tvm_ffi

        if self.launch_count != 0:
            raise RuntimeError("R3-1b1 prepared forward is single-use")
        current_metadata = {
            f"alpha/{name}": production_tensor_sha256(tensor)
            for name, tensor in self.alpha_maps.items()
        }
        current_metadata.update(
            {
                f"beta/{name}": production_tensor_sha256(tensor)
                for name, tensor in self.beta_maps.items()
            }
        )
        current_metadata.update(
            {
                f"split/{name}": production_tensor_sha256(tensor)
                for name, tensor in self.split_maps.items()
            }
        )
        if current_metadata != self._metadata_hashes:
            raise ValueError("R3-1b1 runtime metadata identity differs")
        current = torch.cuda.current_stream(self.device)
        entry_stream_id = int(current.cuda_stream)
        if entry_stream_id == int(torch.cuda.default_stream(self.device).cuda_stream):
            raise RuntimeError("R3-1b1 non-default stream is required")
        entry_device = torch.cuda.current_device()
        ordinal = self.device.index if self.device.index is not None else entry_device
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        baseline_allocated = torch.cuda.memory_allocated(self.device)
        ffi_stream_id = -1
        try:
            with tvm_ffi.use_torch_stream(torch.cuda.stream(current)):
                ffi_stream_id = int(
                    tvm_ffi.get_raw_stream(tvm_ffi.device(f"cuda:{ordinal}"))
                )
                if ffi_stream_id != entry_stream_id:
                    raise RuntimeError("R3-1b1 current stream differs")
                s0 = self.scratch_0
                s1 = self.scratch_1
                bias = self.bias_accumulator
                self._launch(
                    R31B1_SEED_SYMBOL, self._tensor("objective"), s0[:60], bias
                )
                self._launch(
                    R31B1_LINEAR16_SYMBOL,
                    s0[:60],
                    self._tensor("param/linear2.weight"),
                    self._tensor("param/linear2.bias"),
                    bias,
                    s1[:600],
                    bias,
                )
                self._relu("31", s1[:600], bias, active_beta=True)
                self._launch(
                    R31B1_LINEAR14_SYMBOL,
                    s1[:600],
                    self._tensor("param/linear1.weight"),
                    self._tensor("param/linear1.bias"),
                    bias,
                    s0[:6144],
                    bias,
                )
                self._relu("28", s0[:6144], bias)
                self._launch(
                    R31B1_RESIDUAL11_SYMBOL,
                    s0,
                    self._tensor("param/layer1.1.conv2.weight"),
                    self._tensor("param/layer1.1.conv2.bias"),
                    self._tensor("relu/25/lower").reshape(6, 1024),
                    self._tensor("relu/25/upper").reshape(6, 1024),
                    self._tensor("relu/25/alpha"),
                    self.alpha_maps["25"],
                    self._tensor("param/layer1.1.conv1.weight"),
                    self._tensor("param/layer1.1.conv1.bias"),
                    bias,
                    s1,
                )
                self._relu("23", s1[:6144], bias)
                self._launch(
                    R31B1_RESIDUAL6_SYMBOL,
                    s1,
                    self._tensor("param/layer1.0.conv2.weight"),
                    self._tensor("param/layer1.0.conv2.bias"),
                    self._tensor("relu/19/lower").reshape(6, 1024),
                    self._tensor("relu/19/upper").reshape(6, 1024),
                    self._tensor("relu/19/alpha"),
                    self.alpha_maps["19"],
                    self._tensor("param/layer1.0.conv1.weight"),
                    self._tensor("param/layer1.0.conv1.bias"),
                    self._tensor("param/layer1.0.shortcut.0.weight"),
                    self._tensor("param/layer1.0.shortcut.0.bias"),
                    bias,
                    s0,
                )
                self._relu("17", s0[:12288], bias)
                self._launch(
                    R31B1_CONV0_SYMBOL,
                    s0[:12288],
                    self._tensor("param/conv1.weight"),
                    self._tensor("param/conv1.bias"),
                    bias,
                    s1,
                    bias,
                )
                self._launch(
                    R31B1_CONCRETIZE_SYMBOL,
                    s1,
                    self._tensor("input/lower").reshape(6, 3072),
                    self._tensor("input/upper").reshape(6, 3072),
                    bias,
                    self.output,
                )
            torch.cuda.synchronize(self.device)
        finally:
            if (
                torch.cuda.current_device() != entry_device
                or int(torch.cuda.current_stream(self.device).cuda_stream)
                != entry_stream_id
            ):
                raise RuntimeError("R3-1b1 global execution state drifted")
        peak_allocated = torch.cuda.max_memory_allocated(self.device)
        dynamic_bytes = max(0, peak_allocated - baseline_allocated)
        receipt = R31B1LaunchReceiptV1(
            schema_version=R31B1_LAUNCH_SCHEMA,
            stream_id=entry_stream_id,
            tvm_ffi_stream_id=ffi_stream_id,
            scratch_pointers=(self.scratch_0.data_ptr(), self.scratch_1.data_ptr()),
            scratch_capacity_elements=(R31B1_ARENA_CAPACITY, R31B1_ARENA_CAPACITY),
            scratch_high_water_elements=(R31B1_ARENA_CAPACITY, R31B1_ARENA_CAPACITY),
            coefficient_scratch_count=2,
            launch_count=self.launch_count,
            dlpack_pointer_count=self._pointer_count,
            dlpack_pointer_exact_count=self._pointer_exact_count,
            warm_dynamic_allocated_bytes=dynamic_bytes,
            python_visible_intermediate_coefficient_count=0,
            fallback_count=0,
            eager_count=0,
            native_shadow_count=0,
            compiled_region=True,
            timing_recorded=False,
            performance_claimed=False,
        )
        result = R31B1ForwardResultV1(
            lower=self.output.reshape(6, 1),
            compilation_receipt=self.compilation_receipt,
            launch_receipt=receipt,
        )
        result.validate()
        return result

    def _relu(
        self,
        name: str,
        arena: torch.Tensor,
        bias: torch.Tensor,
        *,
        active_beta: bool = False,
    ) -> None:
        symbols = {
            "31": (R31B1_RELU31_BIAS_SYMBOL, R31B1_RELU31_COEFF_SYMBOL),
            "28": (R31B1_RELU28_BIAS_SYMBOL, R31B1_RELU28_COEFF_SYMBOL),
            "23": (R31B1_RELU23_BIAS_SYMBOL, R31B1_RELU23_COEFF_SYMBOL),
            "17": (R31B1_RELU17_BIAS_SYMBOL, R31B1_RELU17_COEFF_SYMBOL),
        }
        bias_symbol, coefficient_symbol = symbols[name]
        lower = self._tensor(f"relu/{name}/lower").reshape(6, -1)
        upper = self._tensor(f"relu/{name}/upper").reshape(6, -1)
        alpha = self._tensor(f"relu/{name}/alpha")
        alpha_map = self.alpha_maps[name]
        self._launch(bias_symbol, arena, lower, upper, alpha, alpha_map, bias, bias)
        coefficient_arguments = [arena, lower, upper, alpha, alpha_map]
        if active_beta:
            coefficient_arguments.extend(
                (
                    self._tensor(f"relu/{name}/beta"),
                    self.beta_maps[name],
                    self.split_maps[name],
                )
            )
        coefficient_arguments.append(arena)
        self._launch(coefficient_symbol, *coefficient_arguments)


def source_receipt_hash(receipt: R31B1CompilationReceiptV1) -> str:
    receipt.validate()
    payload = "|".join(
        (
            receipt.module_hash,
            receipt.device_source_hash,
            receipt.trace_hash,
            receipt.production_plan_hash,
            *receipt.exported_symbols,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "PreparedR31B1FullLowerForwardV1",
    "R31B1CompilationReceiptV1",
    "R31B1ForwardResultV1",
    "R31B1LaunchReceiptV1",
    "R31B1ModuleCacheV1",
    "R31B1_MODULE_CACHE",
    "source_receipt_hash",
]
