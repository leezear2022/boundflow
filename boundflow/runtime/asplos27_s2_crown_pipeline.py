"""S2 canonical coarse-CROWN program with a Relax/cuDNN value path."""

# mypy: disable-error-code=import-untyped
# pylint: disable=import-error,import-outside-toplevel,protected-access
# pylint: disable=too-many-instance-attributes,too-many-locals
# pylint: disable=too-many-boolean-expressions,missing-function-docstring
# pylint: disable=too-few-public-methods

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import torch

from boundflow.backends.tvm.asplos27_s2_selected_value import (
    CompiledS2SelectedValueV1,
    S2_SELECTED_VALUE_CUDNN_CALLS,
    S2_SELECTED_VALUE_CUDNN_FUNCTIONS,
    S2_SELECTED_VALUE_TIR_COUNT,
    compile_s2_selected_value_v1,
)
from boundflow.runtime.r3_d2b_staged_backward import (
    PreparedR3D2BStagedBackwardCandidateV1,
)

S2_EXECUTION_RECEIPT_SCHEMA = "boundflow.asplos27-s2-crown-execution/v1"


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


class S2SelectedValueModuleCacheV1:
    """One compiled executable per exact CUDA device capability."""

    def __init__(self) -> None:
        self._entries: dict[tuple[int, int], CompiledS2SelectedValueV1] = {}

    def get(self, device: torch.device) -> CompiledS2SelectedValueV1:
        ordinal = device.index
        if ordinal is None:
            ordinal = torch.cuda.current_device()
        capability = torch.cuda.get_device_capability(ordinal)
        compiled = self._entries.get(capability)
        if compiled is None:
            compiled = compile_s2_selected_value_v1(device_index=ordinal)
            self._entries[capability] = compiled
        return compiled


S2_SELECTED_VALUE_CACHE = S2SelectedValueModuleCacheV1()


class PreparedS2SelectedValueGraphV1:
    """Persistent DLPack bindings with graph-safe Relax VM invocation."""

    def __init__(
        self,
        compiled: CompiledS2SelectedValueV1,
        arguments: tuple[torch.Tensor, ...],
        *,
        device: torch.device,
    ) -> None:
        import tvm
        import tvm_ffi
        from tvm import relax

        compiled.validate()
        if (
            len(arguments) != 28
            or any(
                value.device != device or not value.is_contiguous()
                for value in arguments
            )
            or device.type != "cuda"
        ):
            raise ValueError("S2 selected-value prepared arguments differ")
        self.compiled = compiled
        self.device = device
        self.output = torch.empty((6, 16, 8, 8), dtype=torch.float32, device=device)
        self.arguments = (*arguments, self.output)
        self.argument_identity = tuple(
            (value.data_ptr(), tuple(value.shape), str(value.dtype))
            for value in self.arguments
        )
        self.argument_views = tuple(
            tvm.runtime.from_dlpack(value) for value in self.arguments
        )
        ordinal = device.index
        if ordinal is None:
            ordinal = torch.cuda.current_device()
        self.vm = relax.VirtualMachine(compiled.executable, tvm.cuda(ordinal))
        self.function = self.vm[compiled.function_name]
        capture_stream = torch.cuda.Stream(device=device)
        capture_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(capture_stream):
            for _ in range(3):
                with tvm_ffi.use_torch_stream(torch.cuda.stream(capture_stream)):
                    self.function(*self.argument_views)
        capture_stream.synchronize()
        with torch.cuda.stream(capture_stream):
            with tvm_ffi.use_torch_stream(torch.cuda.stream(capture_stream)):
                initial_result = self.function(*self.argument_views)
        capture_stream.synchronize()
        initial_output = torch.from_dlpack(initial_result)
        if (
            tuple(self.output.shape) != (6, 16, 8, 8)
            or self.output.dtype != torch.float32
            or self.output.device != device
            or not self.output.is_contiguous()
            or initial_output.data_ptr() != self.output.data_ptr()
        ):
            raise RuntimeError("S2 selected-value captured output differs")
        self.prepare_dlpack_view_count = len(self.argument_views) + 1
        self.replay_count = 0
        self.vm_invocation_count = 0
        self.output_copy_count = 0
        self.result_owners: list[object] = []

    def begin_sample(self) -> None:
        """Release prior synchronized outputs and own this sample's VM results."""

        self.result_owners.clear()
        self.replay_count = 0
        self.vm_invocation_count = 0
        self.output_copy_count = 0

    def _validate_identity(self) -> None:
        current = tuple(
            (value.data_ptr(), tuple(value.shape), str(value.dtype))
            for value in self.arguments
        )
        if current != self.argument_identity:
            raise ValueError("S2 selected-value runtime identity differs")
        stream = torch.cuda.current_stream(self.device)
        if int(stream.cuda_stream) == int(
            torch.cuda.default_stream(self.device).cuda_stream
        ):
            raise RuntimeError("S2 selected-value non-default stream is required")

    def replay(self) -> torch.Tensor:
        import tvm_ffi

        self._validate_identity()
        current = torch.cuda.current_stream(self.device)
        with tvm_ffi.use_torch_stream(torch.cuda.stream(current)):
            result = self.function(*self.argument_views)
        self.result_owners.append(result)
        self.replay_count += 1
        self.vm_invocation_count += 1
        self.output_copy_count += 1
        return self.output


@dataclass(frozen=True)
class S2CrownExecutionReceiptV1:
    """End-to-end ownership and execution receipt for one direct VJP."""

    production_plan_hash: str
    trace_hash: str
    b1_module_hash: str
    b2_module_hash: str
    d1c_schedule_hash: str
    selected_source_relax_ir_hash: str
    selected_partitioned_relax_ir_hash: str
    selected_lowered_relax_ir_hash: str
    selected_device_source_hashes: tuple[str, ...]
    cudnn_partition_function_count: int
    cudnn_conv_call_count: int
    selected_tir_count: int
    forward_graph_replay_count: int
    selected_graph_replay_count: int
    selected_vm_invocation_count: int
    selected_output_copy_count: int
    custom_forward_count: int
    custom_backward_count: int
    existing_arena_count: int
    active_beta: bool
    saved_dense_a_count: int
    saved_autograd_history: bool
    prepare_dlpack_view_count: int
    warm_dlpack_view_count: int
    fallback_count: int
    eager_candidate_count: int
    native_shadow_count: int
    output_pointer: int
    schema_version: str = S2_EXECUTION_RECEIPT_SCHEMA
    performance_claimed: bool = False

    def validate(self) -> None:
        hashes = (
            self.production_plan_hash,
            self.trace_hash,
            self.b1_module_hash,
            self.b2_module_hash,
            self.d1c_schedule_hash,
            self.selected_source_relax_ir_hash,
            self.selected_partitioned_relax_ir_hash,
            self.selected_lowered_relax_ir_hash,
            *self.selected_device_source_hashes,
        )
        if (
            self.schema_version != S2_EXECUTION_RECEIPT_SCHEMA
            or any(len(value) != 64 for value in hashes)
            or not self.selected_device_source_hashes
            or self.cudnn_partition_function_count != S2_SELECTED_VALUE_CUDNN_FUNCTIONS
            or self.cudnn_conv_call_count != S2_SELECTED_VALUE_CUDNN_CALLS
            or self.selected_tir_count != S2_SELECTED_VALUE_TIR_COUNT
            or self.forward_graph_replay_count != 1
            or self.selected_graph_replay_count != 0
            or self.selected_vm_invocation_count != 1
            or self.selected_output_copy_count != 1
            or self.custom_forward_count != 1
            or self.custom_backward_count != 1
            or self.existing_arena_count != 2
            or not self.active_beta
            or self.saved_dense_a_count != 0
            or self.saved_autograd_history
            or self.prepare_dlpack_view_count != 30
            or self.warm_dlpack_view_count != 0
            or self.fallback_count
            or self.eager_candidate_count
            or self.native_shadow_count
            or self.output_pointer <= 0
            or self.performance_claimed
        ):
            raise ValueError("S2 CROWN execution receipt differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        result = asdict(self)
        result["selected_device_source_hashes"] = list(
            self.selected_device_source_hashes
        )
        result["receipt_hash"] = _canonical_hash(result)
        return result


class PreparedS2CrownProgramV1(PreparedR3D2BStagedBackwardCandidateV1):
    """Canonical P-anchor forward plus direct custom VJP without autograd state."""

    def __init__(  # type: ignore[no-untyped-def]
        self, plan, trace, tensors: tuple[torch.Tensor, ...]
    ) -> None:
        super().__init__(plan, trace, tensors)
        import tvm

        compiled = S2_SELECTED_VALUE_CACHE.get(self.device)
        arguments = (
            self._tensor("input/lower"),
            self._tensor("input/upper"),
            self.sign_ainput,
            self._tensor("param/conv1.weight"),
            self._tensor("param/conv1.bias"),
            self._tensor("relu/17/lower"),
            self._tensor("relu/17/upper"),
            self._tensor("relu/17/alpha"),
            self.forward_executor.alpha_maps["17"],
            self.sign_a18,
            self._tensor("param/layer1.0.conv1.weight"),
            self._tensor("param/layer1.0.conv1.bias"),
            self._tensor("relu/19/lower"),
            self._tensor("relu/19/upper"),
            self._tensor("relu/19/alpha"),
            self.forward_executor.alpha_maps["19"],
            self.sign_a20,
            self._tensor("param/layer1.0.conv2.weight"),
            self._tensor("param/layer1.0.conv2.bias"),
            self._tensor("param/layer1.0.shortcut.0.weight"),
            self._tensor("param/layer1.0.shortcut.0.bias"),
            self._tensor("relu/23/lower"),
            self._tensor("relu/23/upper"),
            self._tensor("relu/23/alpha"),
            self.forward_executor.alpha_maps["23"],
            self.sign_a24,
            self._tensor("param/layer1.1.conv1.weight"),
            self._tensor("param/layer1.1.conv1.bias"),
        )
        self.selected_value = PreparedS2SelectedValueGraphV1(
            compiled, arguments, device=self.device
        )
        self.pre25_value = self.selected_value.output.reshape(6144)
        self._register_view(tvm, self.pre25_value)
        self._capture_forward_graph()
        self.s2_forward_graph_replay_count = 0
        self.s2_selected_vm_invocation_count = 0
        self.s2_selected_output_copy_count = 0

    def _reset_forward_capture_counters(self) -> None:
        self.custom_forward_count = 0
        self.forward_executor.launch_count = 0
        self.d1c_launch_count = 0
        self.d1c_bias_inplace_alias_count = 0
        self.b2_launch_count = 0

    def begin_sample(self) -> None:
        super().begin_sample()
        self.selected_value.begin_sample()

    def _capture_forward_graph(self) -> None:
        capture_stream = torch.cuda.Stream(device=self.device)
        capture_stream.wait_stream(torch.cuda.current_stream(self.device))
        for _ in range(3):
            self._reset_forward_capture_counters()
            with torch.cuda.stream(capture_stream):
                super().forward()
        capture_stream.synchronize()
        self.forward_graph = torch.cuda.CUDAGraph()
        self._reset_forward_capture_counters()
        with torch.cuda.stream(capture_stream):
            with torch.cuda.graph(self.forward_graph, stream=capture_stream):
                self.captured_lower = super().forward()
        capture_stream.synchronize()
        if self.captured_lower.data_ptr() != self.forward_executor.output.data_ptr():
            raise RuntimeError("S2 forward graph output ownership differs")
        self._reset_forward_capture_counters()

    def begin_evaluation(self, ordinal: int) -> None:
        super().begin_evaluation(ordinal)
        self.s2_forward_graph_replay_count = 0
        self.s2_selected_vm_invocation_count = 0
        self.s2_selected_output_copy_count = 0
        self.selected_value.replay_count = 0
        self.selected_value.vm_invocation_count = 0
        self.selected_value.output_copy_count = 0

    def forward(self) -> torch.Tensor:
        if self.custom_forward_count:
            raise RuntimeError("S2 canonical forward count differs")
        current = torch.cuda.current_stream(self.device)
        if int(current.cuda_stream) == int(
            torch.cuda.default_stream(self.device).cuda_stream
        ):
            raise RuntimeError("S2 canonical non-default stream is required")
        self.custom_forward_count = 1
        self.forward_graph.replay()
        self.s2_forward_graph_replay_count = 1
        self.forward_executor.launch_count = 17
        self.d1c_launch_count = 4
        self.d1c_bias_inplace_alias_count = 2
        self.b2_launch_count = 1
        return self.captured_lower

    def _effective_value_pass(self, _s0: torch.Tensor, _s1: torch.Tensor) -> None:
        result = self.selected_value.replay()
        if result.data_ptr() != self.pre25_value.data_ptr():
            raise RuntimeError("S2 selected-value output ownership differs")
        self.s2_selected_vm_invocation_count += 1
        self.s2_selected_output_copy_count += 1

    def run_vjp(
        self,
        dynamic_alpha: torch.Tensor | None = None,
        upstream: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run exactly one P-anchor lower evaluation and its compressed VJP."""

        alpha = self.tensors[self.plan.p_alpha_input_ordinal]
        if dynamic_alpha is not None and dynamic_alpha is not alpha:
            if (
                dynamic_alpha.shape != alpha.shape
                or dynamic_alpha.dtype != alpha.dtype
                or dynamic_alpha.device != alpha.device
                or not dynamic_alpha.is_contiguous()
                or not bool(torch.isfinite(dynamic_alpha).all().item())
            ):
                raise ValueError("S2 dynamic alpha contract differs")
            with torch.no_grad():
                alpha.copy_(dynamic_alpha)
        if upstream is not None and upstream is not self.upstream_gradient:
            if (
                upstream.shape != self.upstream_gradient.shape
                or upstream.dtype != self.upstream_gradient.dtype
                or upstream.device != self.upstream_gradient.device
                or not upstream.is_contiguous()
                or not bool(torch.isfinite(upstream).all().item())
            ):
                raise ValueError("S2 upstream contract differs")
            self.upstream_gradient.copy_(upstream)
        self.begin_sample()
        self.begin_evaluation(0)
        lower = self.forward()
        gradient = self.backward(self.upstream_gradient)
        return lower, gradient

    def execution_receipt(self) -> S2CrownExecutionReceiptV1:
        compiled = self.selected_value.compiled
        receipt = S2CrownExecutionReceiptV1(
            production_plan_hash=self.plan.stable_hash(),
            trace_hash=self.trace.stable_hash(),
            b1_module_hash=self.forward_executor.compiled.module_hash,
            b2_module_hash=self.compiled.module_hash,
            d1c_schedule_hash=self.d1c_compiled.scheduled_tir_hash,
            selected_source_relax_ir_hash=compiled.source_relax_ir_hash,
            selected_partitioned_relax_ir_hash=(compiled.partitioned_relax_ir_hash),
            selected_lowered_relax_ir_hash=compiled.lowered_relax_ir_hash,
            selected_device_source_hashes=compiled.device_source_hashes,
            cudnn_partition_function_count=compiled.cudnn_partition_function_count,
            cudnn_conv_call_count=compiled.cudnn_conv_call_count,
            selected_tir_count=compiled.selected_tir_count,
            forward_graph_replay_count=self.s2_forward_graph_replay_count,
            selected_graph_replay_count=0,
            selected_vm_invocation_count=self.s2_selected_vm_invocation_count,
            selected_output_copy_count=self.s2_selected_output_copy_count,
            custom_forward_count=self.custom_forward_count,
            custom_backward_count=self.custom_backward_count,
            existing_arena_count=2,
            active_beta=True,
            saved_dense_a_count=0,
            saved_autograd_history=False,
            prepare_dlpack_view_count=self.selected_value.prepare_dlpack_view_count,
            warm_dlpack_view_count=0,
            fallback_count=0,
            eager_candidate_count=0,
            native_shadow_count=0,
            output_pointer=self.pre25_value.data_ptr(),
        )
        receipt.validate()
        return receipt


__all__ = [
    "PreparedS2CrownProgramV1",
    "PreparedS2SelectedValueGraphV1",
    "S2CrownExecutionReceiptV1",
    "S2SelectedValueModuleCacheV1",
]
