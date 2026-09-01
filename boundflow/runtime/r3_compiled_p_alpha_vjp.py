"""R3-1b2 compiled full-lower custom Function and P-alpha VJP launcher."""

# mypy: disable-error-code=import-untyped
# pylint: disable=import-error,import-outside-toplevel,protected-access
# pylint: disable=too-many-instance-attributes,too-many-locals,too-many-statements
# pylint: disable=too-many-arguments,too-many-positional-arguments
# pylint: disable=missing-function-docstring,missing-class-docstring
# pylint: disable=too-few-public-methods,too-many-boolean-expressions
# pylint: disable=abstract-method,arguments-differ

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import torch
from torch.autograd.function import once_differentiable

from boundflow.backends.tvm.r3_full_lower_forward import (
    R31B1_CONV0_SYMBOL,
    R31B1_LINEAR14_SYMBOL,
    R31B1_LINEAR16_SYMBOL,
    R31B1_RELU17_COEFF_SYMBOL,
    R31B1_RELU23_COEFF_SYMBOL,
    R31B1_RELU28_COEFF_SYMBOL,
    R31B1_RELU31_COEFF_SYMBOL,
    R31B1_RESIDUAL11_SYMBOL,
    R31B1_RESIDUAL6_SYMBOL,
    R31B1_SEED_SYMBOL,
)
from boundflow.backends.tvm.r3_p_alpha_vjp import (
    CompiledR31B2PAlphaVJPV1,
    R31B2_CLEAR_SYMBOL,
    R31B2_COMPRESSED_GRADIENT_SYMBOL,
    R31B2_CONV10_RIGHT_SYMBOL,
    R31B2_EFFECTIVE_PRE17_SYMBOL,
    R31B2_EFFECTIVE_PRE23_SYMBOL,
    R31B2_EFFECTIVE_PRE25_SYMBOL,
    R31B2_EXPORTED_SYMBOLS,
    R31B2_PACK_A18_SYMBOL,
    R31B2_PACK_A20_SYMBOL,
    R31B2_PACK_A24_SYMBOL,
    R31B2_PACK_AINPUT_SYMBOL,
    compile_r31b2_p_alpha_vjp_tir_v1,
)
from boundflow.ir.r3_bounded_arena import R31BBoundedArenaTraceV1

from .r3_full_lower_forward_tir import PreparedR31B1FullLowerForwardV1
from .r3_structured_owner_custom_backward import R31FullRegionPlanV1
from .rvir_v4_production_state import production_tensor_sha256

R31B2_CONTEXT_SCHEMA = "boundflow.r3-1b2-compiled-context/v1"
R31B2_RECEIPT_SCHEMA = "boundflow.r3-1b2-compiled-receipt/v1"

_EXECUTOR_REGISTRY: dict[str, "PreparedR31B2CompiledCustomBackwardV1"] = {}


def _is_hash(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


@dataclass(frozen=True)
class R31B2CompiledReceiptV1:
    production_plan_hash: str
    trace_hash: str
    b1_module_hash: str
    b2_module_hash: str
    b2_device_source_hash: str
    b2_exported_symbols: tuple[str, ...]
    custom_forward_count: int
    custom_backward_count: int
    b1_forward_launch_count: int
    b1_backward_launch_count: int
    b2_launch_count: int
    coefficient_scratch_count: int
    sign_bitmap_count: int
    sign_bitmap_bytes: int
    saved_dense_a_count: int
    python_visible_intermediate_coefficient_count: int
    warm_dynamic_allocated_bytes: int
    fallback_count: int
    eager_candidate_count: int
    native_shadow_count: int
    dlpack_pointer_count: int
    dlpack_pointer_exact_count: int
    runtime_dlpack_pointer_count: int
    runtime_dlpack_pointer_exact_count: int
    compiled_vjp: bool
    custom_vjp: bool
    timing_recorded: bool = False
    performance_claimed: bool = False
    schema_version: str = R31B2_RECEIPT_SCHEMA

    def validate(self) -> None:
        if (
            self.schema_version != R31B2_RECEIPT_SCHEMA
            or not all(
                _is_hash(value)
                for value in (
                    self.production_plan_hash,
                    self.trace_hash,
                    self.b1_module_hash,
                    self.b2_module_hash,
                    self.b2_device_source_hash,
                )
            )
            or self.b2_exported_symbols != R31B2_EXPORTED_SYMBOLS
            or self.custom_forward_count != 1
            or self.custom_backward_count != 1
            or self.b1_forward_launch_count != 15
            or self.b1_backward_launch_count != 15
            or self.b2_launch_count != len(R31B2_EXPORTED_SYMBOLS)
            or self.coefficient_scratch_count != 2
            or self.sign_bitmap_count != 4
            or self.sign_bitmap_bytes != 43_008
            or self.saved_dense_a_count != 0
            or self.python_visible_intermediate_coefficient_count != 0
            or self.warm_dynamic_allocated_bytes != 0
            or self.fallback_count != 0
            or self.eager_candidate_count != 0
            or self.native_shadow_count != 0
            or self.dlpack_pointer_count <= 0
            or self.dlpack_pointer_exact_count != self.dlpack_pointer_count
            or self.runtime_dlpack_pointer_count != 1
            or self.runtime_dlpack_pointer_exact_count != 1
            or self.compiled_vjp is not True
            or self.custom_vjp is not True
            or self.timing_recorded
            or self.performance_claimed
        ):
            raise ValueError("R3-1b2 compiled receipt differs")


@dataclass(frozen=True)
class R31B2CompiledResultV1:
    final_lower: torch.Tensor
    compressed_alpha_gradient: torch.Tensor
    receipt: R31B2CompiledReceiptV1

    def validate(self) -> None:
        self.receipt.validate()
        if (
            tuple(self.final_lower.shape) != (6, 1)
            or tuple(self.compressed_alpha_gradient.shape) != (2, 1, 6, 86)
            or not bool(torch.isfinite(self.final_lower).all().item())
            or not bool(torch.isfinite(self.compressed_alpha_gradient).all().item())
        ):
            raise ValueError("R3-1b2 compiled result differs")


class R31B2ModuleCacheV1:
    def __init__(self) -> None:
        self._entries: dict[str, CompiledR31B2PAlphaVJPV1] = {}

    def get(self, compute_capability: str) -> CompiledR31B2PAlphaVJPV1:
        compiled = self._entries.get(compute_capability)
        if compiled is None:
            compiled = compile_r31b2_p_alpha_vjp_tir_v1(
                compute_capability=compute_capability
            )
            self._entries[compute_capability] = compiled
        return compiled


R31B2_MODULE_CACHE = R31B2ModuleCacheV1()


class PreparedR31B2CompiledCustomBackwardV1:
    def __init__(
        self,
        plan: R31FullRegionPlanV1,
        trace: R31BBoundedArenaTraceV1,
        tensors: tuple[torch.Tensor, ...],
        *,
        compute_capability: str = "sm_89",
        cache: R31B2ModuleCacheV1 = R31B2_MODULE_CACHE,
    ) -> None:
        import tvm

        plan.validate()
        trace.validate()
        self.plan = plan
        self.trace = trace
        self.tensors = tensors
        self.forward_executor = PreparedR31B1FullLowerForwardV1(
            plan, trace, tensors, compute_capability=compute_capability
        )
        self.compiled = cache.get(compute_capability)
        self.device = tensors[0].device
        self.sign_a24 = torch.empty(6144, dtype=torch.int8, device=self.device)
        self.sign_a20 = torch.empty(6144, dtype=torch.int8, device=self.device)
        self.sign_a18 = torch.empty(12288, dtype=torch.int8, device=self.device)
        self.sign_ainput = torch.empty(18432, dtype=torch.int8, device=self.device)
        self.pre25_value = torch.empty(6144, dtype=torch.float32, device=self.device)
        self.gradient = torch.empty(1032, dtype=torch.float32, device=self.device)
        self.upstream_gradient = torch.full(
            (6, 1), -1.0, dtype=torch.float32, device=self.device
        )
        p_layout = plan.relu_layouts[plan.p_layout_ordinal]
        self.p_indices = torch.tensor(
            p_layout.alpha_flat_indices, dtype=torch.int32, device=self.device
        ).contiguous()
        self._by_name = {
            spec.name: tensor for spec, tensor in zip(plan.tensor_specs, tensors)
        }
        self._views: dict[tuple[int, tuple[int, ...]], Any] = {}
        view_tensors = list(tensors) + [
            self.forward_executor.scratch_0,
            self.forward_executor.scratch_1,
            self.forward_executor.bias_accumulator,
            self.forward_executor.output,
            *self.forward_executor.alpha_maps.values(),
            *self.forward_executor.beta_maps.values(),
            *self.forward_executor.split_maps.values(),
            self.sign_a24,
            self.sign_a20,
            self.sign_a18,
            self.sign_ainput,
            self.pre25_value,
            self.gradient,
            self.upstream_gradient,
            self.p_indices,
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
        for scratch in (
            self.forward_executor.scratch_0,
            self.forward_executor.scratch_1,
        ):
            for length in (60, 600, 6144, 12288, 18432):
                self._register_view(tvm, scratch[:length])
        self._register_view(tvm, self.upstream_gradient.reshape(6))
        self._pointer_count = len(self._views)
        self._pointer_exact_count = sum(
            int(torch.from_dlpack(view).data_ptr() == key[0])
            for key, view in self._views.items()
        )
        if self._pointer_exact_count != self._pointer_count:
            raise RuntimeError("R3-1b2 static DLPack pointer differs")
        self._metadata_hashes = {
            f"alpha/{name}": production_tensor_sha256(tensor)
            for name, tensor in self.forward_executor.alpha_maps.items()
        }
        self._metadata_hashes.update(
            {
                f"beta/{name}": production_tensor_sha256(tensor)
                for name, tensor in self.forward_executor.beta_maps.items()
            }
        )
        self._metadata_hashes.update(
            {
                f"split/{name}": production_tensor_sha256(tensor)
                for name, tensor in self.forward_executor.split_maps.items()
            }
        )
        self.custom_forward_count = 0
        self.custom_backward_count = 0
        self.b1_backward_launch_count = 0
        self.b2_launch_count = 0
        self.runtime_dlpack_pointer_count = 0
        self.runtime_dlpack_pointer_exact_count = 0
        self.warm_dynamic_allocated_bytes = -1

    def _register_view(self, tvm: Any, tensor: torch.Tensor) -> None:
        key = (tensor.data_ptr(), tuple(int(value) for value in tensor.shape))
        if key not in self._views:
            self._views[key] = tvm.runtime.from_dlpack(tensor)

    def _view(self, tensor: torch.Tensor) -> Any:
        key = (tensor.data_ptr(), tuple(int(value) for value in tensor.shape))
        view = self._views.get(key)
        if view is None:
            raise RuntimeError("R3-1b2 warm DLPack view is absent")
        return view

    def _tensor(self, name: str) -> torch.Tensor:
        value = self._by_name.get(name)
        if value is None:
            raise KeyError(f"R3-1b2 tensor is absent: {name}")
        return value

    def _launch_b1(self, symbol: str, *tensors: torch.Tensor) -> None:
        self.forward_executor.compiled.executable[symbol](
            *(self._view(tensor) for tensor in tensors)
        )
        self.b1_backward_launch_count += 1

    def _launch_b2(self, symbol: str, *tensors: torch.Tensor) -> None:
        self.compiled.executable[symbol](*(self._view(tensor) for tensor in tensors))
        self.b2_launch_count += 1

    def _relu_coefficient(self, name: str, arena: torch.Tensor) -> None:
        symbols = {
            "31": R31B1_RELU31_COEFF_SYMBOL,
            "28": R31B1_RELU28_COEFF_SYMBOL,
            "23": R31B1_RELU23_COEFF_SYMBOL,
            "17": R31B1_RELU17_COEFF_SYMBOL,
        }
        arguments = [
            arena,
            self._tensor(f"relu/{name}/lower").reshape(6, -1),
            self._tensor(f"relu/{name}/upper").reshape(6, -1),
            self._tensor(f"relu/{name}/alpha"),
            self.forward_executor.alpha_maps[name],
        ]
        if name == "31":
            arguments.extend(
                (
                    self._tensor("relu/31/beta"),
                    self.forward_executor.beta_maps["31"],
                    self.forward_executor.split_maps["31"],
                )
            )
        arguments.append(arena)
        self._launch_b1(symbols[name], *arguments)

    def forward(self) -> torch.Tensor:
        import tvm_ffi

        if self.custom_forward_count != 0:
            raise RuntimeError("R3-1b2 custom forward count differs")
        self.custom_forward_count = 1
        result = self.forward_executor.run()
        current = torch.cuda.current_stream(self.device)
        with tvm_ffi.use_torch_stream(torch.cuda.stream(current)):
            self._launch_b2(
                R31B2_CLEAR_SYMBOL,
                self.forward_executor.scratch_0,
                self.forward_executor.scratch_1,
            )
        return result.lower

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        import tvm_ffi

        if self.custom_backward_count != 0:
            raise RuntimeError("R3-1b2 custom backward count differs")
        if (
            tuple(grad_output.shape) != (6, 1)
            or grad_output.dtype != torch.float32
            or grad_output.device != self.device
            or not grad_output.is_contiguous()
        ):
            raise ValueError("R3-1b2 upstream gradient differs")
        self.custom_backward_count = 1
        runtime_view = self._view(grad_output.reshape(6))
        self.runtime_dlpack_pointer_count = 1
        self.runtime_dlpack_pointer_exact_count = int(
            torch.from_dlpack(runtime_view).data_ptr() == grad_output.data_ptr()
        )
        current_metadata = {
            f"alpha/{name}": production_tensor_sha256(tensor)
            for name, tensor in self.forward_executor.alpha_maps.items()
        }
        current_metadata.update(
            {
                f"beta/{name}": production_tensor_sha256(tensor)
                for name, tensor in self.forward_executor.beta_maps.items()
            }
        )
        current_metadata.update(
            {
                f"split/{name}": production_tensor_sha256(tensor)
                for name, tensor in self.forward_executor.split_maps.items()
            }
        )
        if current_metadata != self._metadata_hashes:
            raise ValueError("R3-1b2 runtime metadata identity differs")
        current = torch.cuda.current_stream(self.device)
        stream_id = int(current.cuda_stream)
        if stream_id == int(torch.cuda.default_stream(self.device).cuda_stream):
            raise RuntimeError("R3-1b2 non-default stream is required")
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        baseline_allocated = torch.cuda.memory_allocated(self.device)
        s0 = self.forward_executor.scratch_0
        s1 = self.forward_executor.scratch_1
        bias = self.forward_executor.bias_accumulator
        with tvm_ffi.use_torch_stream(torch.cuda.stream(current)):
            self._coefficient_sign_pass(s0, s1, bias)
            self._effective_value_pass(s0, s1)
            self._recompute_a26(s0, s1, bias)
            self.compiled.executable[R31B2_COMPRESSED_GRADIENT_SYMBOL](
                self._view(s1[:6144]),
                self._view(self.pre25_value),
                self._view(self._tensor("relu/25/lower").reshape(6, 1024)),
                self._view(self._tensor("relu/25/upper").reshape(6, 1024)),
                self._view(self.p_indices),
                runtime_view,
                self._view(self.gradient),
            )
            self.b2_launch_count += 1
        torch.cuda.synchronize(self.device)
        self.warm_dynamic_allocated_bytes = max(
            0, torch.cuda.max_memory_allocated(self.device) - baseline_allocated
        )
        return self.gradient.reshape(2, 1, 6, 86)

    def _coefficient_sign_pass(
        self, s0: torch.Tensor, s1: torch.Tensor, bias: torch.Tensor
    ) -> None:
        self._launch_b1(R31B1_SEED_SYMBOL, self._tensor("objective"), s0[:60], bias)
        self._launch_b1(
            R31B1_LINEAR16_SYMBOL,
            s0[:60],
            self._tensor("param/linear2.weight"),
            self._tensor("param/linear2.bias"),
            bias,
            s1[:600],
            bias,
        )
        self._relu_coefficient("31", s1[:600])
        self._launch_b1(
            R31B1_LINEAR14_SYMBOL,
            s1[:600],
            self._tensor("param/linear1.weight"),
            self._tensor("param/linear1.bias"),
            bias,
            s0[:6144],
            bias,
        )
        self._relu_coefficient("28", s0[:6144])
        self._launch_b1(
            R31B1_RESIDUAL11_SYMBOL,
            s0,
            self._tensor("param/layer1.1.conv2.weight"),
            self._tensor("param/layer1.1.conv2.bias"),
            self._tensor("relu/25/lower").reshape(6, 1024),
            self._tensor("relu/25/upper").reshape(6, 1024),
            self._tensor("relu/25/alpha"),
            self.forward_executor.alpha_maps["25"],
            self._tensor("param/layer1.1.conv1.weight"),
            self._tensor("param/layer1.1.conv1.bias"),
            bias,
            s1,
        )
        self._launch_b2(R31B2_PACK_A24_SYMBOL, s1[:6144], self.sign_a24)
        self._relu_coefficient("23", s1[:6144])
        self._launch_b2(
            R31B2_PACK_A20_SYMBOL,
            s1[:6144],
            self._tensor("param/layer1.0.conv2.weight"),
            self.sign_a20,
        )
        self._launch_b1(
            R31B1_RESIDUAL6_SYMBOL,
            s1,
            self._tensor("param/layer1.0.conv2.weight"),
            self._tensor("param/layer1.0.conv2.bias"),
            self._tensor("relu/19/lower").reshape(6, 1024),
            self._tensor("relu/19/upper").reshape(6, 1024),
            self._tensor("relu/19/alpha"),
            self.forward_executor.alpha_maps["19"],
            self._tensor("param/layer1.0.conv1.weight"),
            self._tensor("param/layer1.0.conv1.bias"),
            self._tensor("param/layer1.0.shortcut.0.weight"),
            self._tensor("param/layer1.0.shortcut.0.bias"),
            bias,
            s0,
        )
        self._launch_b2(R31B2_PACK_A18_SYMBOL, s0[:12288], self.sign_a18)
        self._relu_coefficient("17", s0[:12288])
        self._launch_b1(
            R31B1_CONV0_SYMBOL,
            s0[:12288],
            self._tensor("param/conv1.weight"),
            self._tensor("param/conv1.bias"),
            bias,
            s1,
            bias,
        )
        self._launch_b2(R31B2_PACK_AINPUT_SYMBOL, s1, self.sign_ainput)

    def _effective_value_pass(self, s0: torch.Tensor, s1: torch.Tensor) -> None:
        self._launch_b2(
            R31B2_EFFECTIVE_PRE17_SYMBOL,
            self._tensor("input/lower").reshape(6, 3072),
            self._tensor("input/upper").reshape(6, 3072),
            self.sign_ainput,
            self._tensor("param/conv1.weight"),
            self._tensor("param/conv1.bias"),
            s0[:12288],
        )
        self._launch_b2(
            R31B2_EFFECTIVE_PRE23_SYMBOL,
            s0[:12288],
            self.sign_a18,
            self._tensor("relu/17/lower").reshape(6, 2048),
            self._tensor("relu/17/upper").reshape(6, 2048),
            self._tensor("relu/17/alpha"),
            self.forward_executor.alpha_maps["17"],
            self._tensor("param/layer1.0.conv1.weight"),
            self._tensor("param/layer1.0.conv1.bias"),
            self.sign_a20,
            self._tensor("relu/19/lower").reshape(6, 1024),
            self._tensor("relu/19/upper").reshape(6, 1024),
            self._tensor("relu/19/alpha"),
            self.forward_executor.alpha_maps["19"],
            self._tensor("param/layer1.0.conv2.weight"),
            self._tensor("param/layer1.0.conv2.bias"),
            self._tensor("param/layer1.0.shortcut.0.weight"),
            self._tensor("param/layer1.0.shortcut.0.bias"),
            s1[:6144],
        )
        self._launch_b2(
            R31B2_EFFECTIVE_PRE25_SYMBOL,
            s1[:6144],
            self.sign_a24,
            self._tensor("relu/23/lower").reshape(6, 1024),
            self._tensor("relu/23/upper").reshape(6, 1024),
            self._tensor("relu/23/alpha"),
            self.forward_executor.alpha_maps["23"],
            self._tensor("param/layer1.1.conv1.weight"),
            self._tensor("param/layer1.1.conv1.bias"),
            self.pre25_value,
        )

    def _recompute_a26(
        self, s0: torch.Tensor, s1: torch.Tensor, bias: torch.Tensor
    ) -> None:
        self._launch_b1(R31B1_SEED_SYMBOL, self._tensor("objective"), s0[:60], bias)
        self._launch_b1(
            R31B1_LINEAR16_SYMBOL,
            s0[:60],
            self._tensor("param/linear2.weight"),
            self._tensor("param/linear2.bias"),
            bias,
            s1[:600],
            bias,
        )
        self._relu_coefficient("31", s1[:600])
        self._launch_b1(
            R31B1_LINEAR14_SYMBOL,
            s1[:600],
            self._tensor("param/linear1.weight"),
            self._tensor("param/linear1.bias"),
            bias,
            s0[:6144],
            bias,
        )
        self._relu_coefficient("28", s0[:6144])
        self._launch_b2(
            R31B2_CONV10_RIGHT_SYMBOL,
            s0[:6144],
            self._tensor("param/layer1.1.conv2.weight"),
            self._tensor("param/layer1.1.conv2.bias"),
            bias,
            s1[:6144],
            bias,
        )

    def receipt(self, *, saved_dense_a_count: int) -> R31B2CompiledReceiptV1:
        receipt = R31B2CompiledReceiptV1(
            production_plan_hash=self.plan.stable_hash(),
            trace_hash=self.trace.stable_hash(),
            b1_module_hash=self.forward_executor.compiled.module_hash,
            b2_module_hash=self.compiled.module_hash,
            b2_device_source_hash=self.compiled.device_source_hash,
            b2_exported_symbols=self.compiled.exported_symbols,
            custom_forward_count=self.custom_forward_count,
            custom_backward_count=self.custom_backward_count,
            b1_forward_launch_count=self.forward_executor.launch_count,
            b1_backward_launch_count=self.b1_backward_launch_count,
            b2_launch_count=self.b2_launch_count,
            coefficient_scratch_count=2,
            sign_bitmap_count=4,
            sign_bitmap_bytes=sum(
                value.numel()
                for value in (
                    self.sign_a24,
                    self.sign_a20,
                    self.sign_a18,
                    self.sign_ainput,
                )
            ),
            saved_dense_a_count=saved_dense_a_count,
            python_visible_intermediate_coefficient_count=0,
            warm_dynamic_allocated_bytes=self.warm_dynamic_allocated_bytes,
            fallback_count=0,
            eager_candidate_count=0,
            native_shadow_count=0,
            dlpack_pointer_count=self._pointer_count,
            dlpack_pointer_exact_count=self._pointer_exact_count,
            runtime_dlpack_pointer_count=self.runtime_dlpack_pointer_count,
            runtime_dlpack_pointer_exact_count=self.runtime_dlpack_pointer_exact_count,
            compiled_vjp=True,
            custom_vjp=True,
        )
        receipt.validate()
        return receipt


class _R31B2CompiledFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: object, execution_key: str, *tensors: torch.Tensor
    ) -> torch.Tensor:
        executor = _EXECUTOR_REGISTRY.get(execution_key)
        if not isinstance(executor, PreparedR31B2CompiledCustomBackwardV1):
            raise RuntimeError("R3-1b2 executor registry differs")
        ctx.set_materialize_grads(False)  # type: ignore[attr-defined]
        setattr(ctx, "execution_key", execution_key)
        setattr(ctx, "schema_version", R31B2_CONTEXT_SCHEMA)
        setattr(ctx, "p_alpha_input_ordinal", executor.plan.p_alpha_input_ordinal)
        ctx.save_for_backward(*tensors)  # type: ignore[attr-defined]
        return executor.forward()

    @staticmethod
    @once_differentiable
    def backward(  # type: ignore[override]
        ctx: object, grad_output: torch.Tensor
    ) -> tuple[object, ...]:
        execution_key = getattr(ctx, "execution_key", None)
        executor = (
            _EXECUTOR_REGISTRY.get(execution_key)
            if isinstance(execution_key, str)
            else None
        )
        if (
            not isinstance(executor, PreparedR31B2CompiledCustomBackwardV1)
            or getattr(ctx, "schema_version", None) != R31B2_CONTEXT_SCHEMA
        ):
            raise TypeError("R3-1b2 custom backward context differs")
        gradient = executor.backward(grad_output)
        saved = tuple(ctx.saved_tensors)  # type: ignore[attr-defined]
        tensor_gradients: list[torch.Tensor | None] = [None] * len(saved)
        tensor_gradients[executor.plan.p_alpha_input_ordinal] = gradient
        return (None, *tensor_gradients)


def execute_r31b2_compiled_custom_backward_v1(
    plan: R31FullRegionPlanV1,
    trace: R31BBoundedArenaTraceV1,
    tensors: tuple[torch.Tensor, ...],
) -> R31B2CompiledResultV1:
    plan.validate()
    executor = PreparedR31B2CompiledCustomBackwardV1(plan, trace, tensors)
    execution_key = hashlib.sha256(
        f"{plan.stable_hash()}:{id(executor)}".encode("utf-8")
    ).hexdigest()
    if execution_key in _EXECUTOR_REGISTRY:
        raise RuntimeError("R3-1b2 execution key repeats")
    saved: list[torch.Tensor] = []

    def pack(value: torch.Tensor) -> torch.Tensor:
        saved.append(value)
        return value

    _EXECUTOR_REGISTRY[execution_key] = executor
    try:
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
            lower = _R31B2CompiledFunction.apply(execution_key, *tensors)
            p_alpha = tensors[plan.p_alpha_input_ordinal]
            gradient = torch.autograd.grad(
                lower,
                p_alpha,
                grad_outputs=executor.upstream_gradient,
            )[0]
    finally:
        _EXECUTOR_REGISTRY.pop(execution_key, None)
    saved_dense_a_count = sum(
        value.ndim == 5 and tuple(value.shape[:2]) == (6, 1) for value in saved
    )
    result = R31B2CompiledResultV1(
        final_lower=lower.detach(),
        compressed_alpha_gradient=gradient.detach(),
        receipt=executor.receipt(saved_dense_a_count=saved_dense_a_count),
    )
    result.validate()
    return result


__all__ = [
    "execute_r31b2_compiled_custom_backward_v1",
    "PreparedR31B2CompiledCustomBackwardV1",
    "R31B2CompiledReceiptV1",
    "R31B2CompiledResultV1",
]
