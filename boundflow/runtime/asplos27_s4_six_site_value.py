"""Prepared runtime owner for the S4-1B six-site selected-value graph."""

# mypy: disable-error-code=import-untyped
# pylint: disable=import-error,import-outside-toplevel,too-many-instance-attributes
# pylint: disable=too-many-arguments,too-many-locals,too-many-boolean-expressions
# pylint: disable=too-few-public-methods,protected-access,too-many-lines
# pylint: disable=missing-class-docstring,missing-function-docstring
# pylint: disable=unidiomatic-typecheck
# pylint: disable=duplicate-code

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
from typing import NoReturn

import torch

from boundflow.backends.tvm.asplos27_s4_six_site_value import (
    CompiledS4SixSiteValueV1,
    S4_SIX_SITE_ARGUMENTS,
    S4_SIX_SITE_CUDNN_CALLS,
    S4_SIX_SITE_CUDNN_FUNCTIONS,
    S4_SIX_SITE_READ_ARGUMENTS,
    S4_SIX_SITE_TIR_COUNT,
    S4_SIX_SITE_WRITE_TARGETS,
    S4_VALUE_ARENA_BYTES,
    S4_VALUE_ARENA_ELEMENTS,
    S4_VALUE_SLOTS_V1,
)
from boundflow.runtime.asplos27_s4_coefficient_selector_pass import (
    PreparedS4CoefficientSelectorPassV1,
)

S4_SIX_SITE_RUNTIME_SCHEMA = "boundflow.asplos27-s4-six-site-runtime/v1"

S4_SIX_SITE_READ_SPECS = (
    ("input_lower", (6, 3, 32, 32), torch.float32),
    ("input_upper", (6, 3, 32, 32), torch.float32),
    ("endpoint_selector", (18432,), torch.int8),
    ("weight0", (8, 3, 3, 3), torch.float32),
    ("bias0", (8,), torch.float32),
    ("lower17", (6, 8, 16, 16), torch.float32),
    ("upper17", (6, 8, 16, 16), torch.float32),
    ("alpha17", (6, 164), torch.float32),
    ("alpha_map17", (2048,), torch.int32),
    ("sign_a18", (12288,), torch.int8),
    ("weight2", (16, 8, 3, 3), torch.float32),
    ("bias2", (16,), torch.float32),
    ("lower19", (6, 16, 8, 8), torch.float32),
    ("upper19", (6, 16, 8, 8), torch.float32),
    ("alpha19", (6, 132), torch.float32),
    ("alpha_map19", (1024,), torch.int32),
    ("sign_a20", (6144,), torch.int8),
    ("weight4", (16, 16, 3, 3), torch.float32),
    ("bias4", (16,), torch.float32),
    ("weight5", (16, 8, 1, 1), torch.float32),
    ("bias5", (16,), torch.float32),
    ("lower23", (6, 16, 8, 8), torch.float32),
    ("upper23", (6, 16, 8, 8), torch.float32),
    ("alpha23", (6, 121), torch.float32),
    ("alpha_map23", (1024,), torch.int32),
    ("sign_a24", (6144,), torch.int8),
    ("weight8", (16, 16, 3, 3), torch.float32),
    ("bias8", (16,), torch.float32),
    ("lower25", (6, 16, 8, 8), torch.float32),
    ("upper25", (6, 16, 8, 8), torch.float32),
    ("alpha25", (6, 86), torch.float32),
    ("alpha_map25", (1024,), torch.int32),
    ("sign_a26", (6144,), torch.int8),
    ("weight10", (16, 16, 3, 3), torch.float32),
    ("bias10", (16,), torch.float32),
    ("lower28", (6, 16, 8, 8), torch.float32),
    ("upper28", (6, 16, 8, 8), torch.float32),
    ("alpha28", (6, 178), torch.float32),
    ("alpha_map28", (1024,), torch.int32),
    ("sign_a29", (6144,), torch.int8),
    ("weight14", (100, 1024), torch.float32),
    ("bias14", (100,), torch.float32),
)

_SELECTOR_ARGUMENTS = {
    "endpoint_ainput_v2": 2,
    "sign_a18": 9,
    "sign_a20": 16,
    "sign_a24": 25,
    "sign_a26": 32,
    "sign_a29": 39,
}


class S4SixSitePhase(str, Enum):
    PREPARED = "PREPARED"
    PASS_A_RUNNING = "PASS_A_RUNNING"
    SELECTORS_READY = "SELECTORS_READY"
    ARENA_REBOUND_FOR_SELECTED_INPUT = "ARENA_REBOUND_FOR_SELECTED_INPUT"
    PASS_B_RUNNING = "PASS_B_RUNNING"
    VALUES_READY = "VALUES_READY"
    COEFFICIENT_RECOMPUTE_READY = "COEFFICIENT_RECOMPUTE_READY"
    POISONED = "POISONED"
    CLOSED = "CLOSED"


class S4SixSiteRuntimeError(RuntimeError):
    """Stable fail-closed S4-1B runtime error."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def _reject(reason: str) -> NoReturn:
    raise S4SixSiteRuntimeError(reason)


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _storage_identity(tensor: torch.Tensor) -> tuple[str, int, int, int]:
    storage = tensor.untyped_storage()
    return (
        str(tensor.device),
        int(storage._cdata),
        int(storage.data_ptr()),
        int(storage.nbytes()),
    )


@dataclass(frozen=True)
class S4SixSiteValueReceiptV1:
    selector_receipt_hash: str
    source_relax_ir_hash: str
    partitioned_relax_ir_hash: str
    lowered_relax_ir_hash: str
    device_source_hashes: tuple[str, ...]
    phase_order: tuple[str, ...]
    read_argument_count: int
    write_target_count: int
    total_argument_count: int
    s4_1b_union_descriptor_count: int
    s4_1abc_union_descriptor_count: int
    base_overlap_count: int
    prepare_dlpack_view_count: int
    warm_dlpack_view_count: int
    value_arena_elements: int
    value_arena_bytes: int
    value_slot_count: int
    selected_input_alias_exact: bool
    selected_input_live_reader_count: int
    cudnn_partition_function_count: int
    cudnn_conv_call_count: int
    selected_tir_count: int
    logical_stage_count: int
    persistent_output_copy_count: int
    vm_invocation_count: int
    graph_submission_count: int
    dynamic_output_allocation_count: int
    result_owner_capacity: int
    output_pointer_exact_count: int
    fallback_count: int
    eager_candidate_count: int
    native_shadow_count: int
    saved_dense_a_count: int
    timing_recorded: bool
    performance_claimed: bool
    schema_version: str = S4_SIX_SITE_RUNTIME_SCHEMA

    def validate(self) -> None:
        hashes = (
            self.selector_receipt_hash,
            self.source_relax_ir_hash,
            self.partitioned_relax_ir_hash,
            self.lowered_relax_ir_hash,
            *self.device_source_hashes,
        )
        expected_phases = tuple(
            phase.value
            for phase in (
                S4SixSitePhase.PREPARED,
                S4SixSitePhase.PASS_A_RUNNING,
                S4SixSitePhase.SELECTORS_READY,
                S4SixSitePhase.ARENA_REBOUND_FOR_SELECTED_INPUT,
                S4SixSitePhase.PASS_B_RUNNING,
                S4SixSitePhase.VALUES_READY,
                S4SixSitePhase.COEFFICIENT_RECOMPUTE_READY,
            )
        )
        if (
            self.schema_version != S4_SIX_SITE_RUNTIME_SCHEMA
            or any(len(value) != 64 for value in hashes)
            or not self.device_source_hashes
            or self.phase_order != expected_phases
            or self.read_argument_count != S4_SIX_SITE_READ_ARGUMENTS
            or self.write_target_count != S4_SIX_SITE_WRITE_TARGETS
            or self.total_argument_count != S4_SIX_SITE_ARGUMENTS
            or self.s4_1b_union_descriptor_count != 90
            or self.s4_1abc_union_descriptor_count != 110
            or self.base_overlap_count != 5
            or self.prepare_dlpack_view_count != S4_SIX_SITE_ARGUMENTS + 6
            or self.warm_dlpack_view_count != 0
            or self.value_arena_elements != S4_VALUE_ARENA_ELEMENTS
            or self.value_arena_bytes != S4_VALUE_ARENA_BYTES
            or self.value_slot_count != 6
            or not self.selected_input_alias_exact
            or self.selected_input_live_reader_count != 0
            or self.cudnn_partition_function_count != S4_SIX_SITE_CUDNN_FUNCTIONS
            or self.cudnn_conv_call_count != S4_SIX_SITE_CUDNN_CALLS
            or self.selected_tir_count != S4_SIX_SITE_TIR_COUNT
            or self.logical_stage_count != 6
            or self.persistent_output_copy_count != 6
            or self.vm_invocation_count != 1
            or self.graph_submission_count != 1
            or self.dynamic_output_allocation_count != 0
            or self.result_owner_capacity != 1
            or self.output_pointer_exact_count != 6
            or self.fallback_count
            or self.eager_candidate_count
            or self.native_shadow_count
            or self.saved_dense_a_count
            or self.timing_recorded
            or self.performance_claimed
        ):
            _reject("SIX_SITE_RECEIPT_MISMATCH")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        result = asdict(self)
        result["device_source_hashes"] = list(self.device_source_hashes)
        result["phase_order"] = list(self.phase_order)
        result["receipt_hash"] = _canonical_hash(result)
        return result


@dataclass(frozen=True)
class S4SixSiteValueResultV1:
    values: tuple[torch.Tensor, ...]
    receipt: S4SixSiteValueReceiptV1

    def validate(self) -> None:
        self.receipt.validate()
        if len(self.values) != len(S4_VALUE_SLOTS_V1):
            _reject("SIX_SITE_RESULT_COUNT_MISMATCH")
        for value, slot in zip(self.values, S4_VALUE_SLOTS_V1):
            if (
                tuple(value.shape) != slot.shape
                or value.dtype != torch.float32
                or not value.is_contiguous()
            ):
                _reject("SIX_SITE_RESULT_LAYOUT_MISMATCH")


class PreparedS4SixSiteValueV1:
    """Persistent 49-view VM owner with a one-attempt phase machine."""

    def __init__(
        self,
        compiled: CompiledS4SixSiteValueV1,
        read_arguments: tuple[torch.Tensor, ...],
        *,
        coefficient_arena: torch.Tensor,
        selected_input_alias: torch.Tensor,
        device: torch.device,
    ) -> None:
        import tvm
        from tvm import relax

        compiled.validate()
        if device.type != "cuda" or len(read_arguments) != len(S4_SIX_SITE_READ_SPECS):
            _reject("SIX_SITE_READ_ARGUMENT_COUNT_MISMATCH")
        for tensor, (name, shape, dtype) in zip(read_arguments, S4_SIX_SITE_READ_SPECS):
            if (
                type(tensor) is not torch.Tensor
                or tensor.device != device
                or tuple(tensor.shape) != shape
                or tensor.dtype != dtype
                or not tensor.is_contiguous()
            ):
                _reject(f"SIX_SITE_READ_LAYOUT_MISMATCH:{name}")
        if (
            coefficient_arena.device != device
            or coefficient_arena.dtype != torch.float32
            or not coefficient_arena.is_contiguous()
            or selected_input_alias.device != device
            or selected_input_alias.dtype != torch.float32
            or tuple(selected_input_alias.shape) != (6, 3, 32, 32)
            or not selected_input_alias.is_contiguous()
            or _storage_identity(coefficient_arena)
            != _storage_identity(selected_input_alias)
        ):
            _reject("SIX_SITE_SELECTED_INPUT_ALIAS_MISMATCH")
        alias_end = selected_input_alias.storage_offset() + selected_input_alias.numel()
        if (
            alias_end
            > coefficient_arena.untyped_storage().nbytes()
            // coefficient_arena.element_size()
        ):
            _reject("SIX_SITE_SELECTED_INPUT_ALIAS_MISMATCH")

        self.compiled = compiled
        self.device = device
        self.read_arguments = read_arguments
        self.coefficient_arena = coefficient_arena
        self.selected_input_alias = selected_input_alias
        self.value_arena = torch.empty(
            S4_VALUE_ARENA_ELEMENTS, dtype=torch.float32, device=device
        )
        self.values = tuple(
            self.value_arena.narrow(0, slot.offset_elements, slot.length_elements).view(
                slot.shape
            )
            for slot in S4_VALUE_SLOTS_V1
        )
        self.arguments = (*read_arguments, selected_input_alias, *self.values)
        if len(self.arguments) != S4_SIX_SITE_ARGUMENTS:
            _reject("SIX_SITE_ARGUMENT_COUNT_MISMATCH")
        self.argument_identity = tuple(
            (value.data_ptr(), tuple(value.shape), str(value.dtype), str(value.device))
            for value in self.arguments
        )
        self.argument_views = tuple(
            tvm.runtime.from_dlpack(value) for value in self.arguments
        )
        ordinal = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        self.vm = relax.VirtualMachine(compiled.executable, tvm.cuda(ordinal))
        self.function = self.vm[compiled.function_name]
        self.phase = S4SixSitePhase.PREPARED
        self._phase_order = [self.phase.value]
        self._expected_stream: int | None = None
        self._selector_receipt_hash = ""
        self._result_owner: object | None = None
        self._vm_invocation_count = 0
        self._graph_submission_count = 0
        self._output_pointer_exact_count = 0
        self._warm_dlpack_view_count = 0

    def _poison(self, reason: str) -> NoReturn:
        self.phase = S4SixSitePhase.POISONED
        _reject(reason)

    def _advance(self, expected: S4SixSitePhase, target: S4SixSitePhase) -> None:
        if self.phase != expected:
            self._poison("SIX_SITE_PHASE_MISMATCH")
        self.phase = target
        self._phase_order.append(target.value)

    def begin_pass_a(self) -> None:
        self._advance(S4SixSitePhase.PREPARED, S4SixSitePhase.PASS_A_RUNNING)
        current = torch.cuda.current_stream(self.device)
        if int(current.cuda_stream) == int(
            torch.cuda.default_stream(self.device).cuda_stream
        ):
            self._poison("SIX_SITE_NONDEFAULT_STREAM_REQUIRED")
        self._expected_stream = int(current.cuda_stream)

    def adopt_selectors(self, owner: PreparedS4CoefficientSelectorPassV1) -> None:
        if self.phase != S4SixSitePhase.PASS_A_RUNNING:
            self._poison("SIX_SITE_PHASE_MISMATCH")
        selector_receipt = owner.receipt()
        if (
            selector_receipt.compiled_pack_launch_count != 6
            or selector_receipt.eager_pack_count != 0
        ):
            self._poison("SIX_SITE_COMPILED_SELECTOR_REQUIRED")
        for name, argument_index in _SELECTOR_ARGUMENTS.items():
            if (
                owner.selector(name).data_ptr()
                != self.read_arguments[argument_index].data_ptr()
            ):
                self._poison("SIX_SITE_SELECTOR_POINTER_MISMATCH")
        self._selector_receipt_hash = selector_receipt.stable_hash()
        self.phase = S4SixSitePhase.SELECTORS_READY
        self._phase_order.append(self.phase.value)
        if _storage_identity(self.coefficient_arena) != _storage_identity(
            self.selected_input_alias
        ):
            self._poison("SIX_SITE_SELECTED_INPUT_ALIAS_MISMATCH")
        self.phase = S4SixSitePhase.ARENA_REBOUND_FOR_SELECTED_INPUT
        self._phase_order.append(self.phase.value)

    def _validate_warm_identity(self) -> None:
        import tvm_ffi

        current_identity = tuple(
            (value.data_ptr(), tuple(value.shape), str(value.dtype), str(value.device))
            for value in self.arguments
        )
        if current_identity != self.argument_identity:
            self._poison("SIX_SITE_ARGUMENT_IDENTITY_MISMATCH")
        current = int(torch.cuda.current_stream(self.device).cuda_stream)
        if self._expected_stream is None or current != self._expected_stream:
            self._poison("SIX_SITE_STREAM_IDENTITY_MISMATCH")
        ordinal = (
            self.device.index
            if self.device.index is not None
            else torch.cuda.current_device()
        )
        ffi_stream = int(tvm_ffi.get_raw_stream(tvm_ffi.device(f"cuda:{ordinal}")))
        if ffi_stream not in (0, current):
            self._poison("SIX_SITE_FFI_STREAM_IDENTITY_MISMATCH")

    def run_pass_b(self) -> tuple[torch.Tensor, ...]:
        import tvm_ffi

        self._advance(
            S4SixSitePhase.ARENA_REBOUND_FOR_SELECTED_INPUT,
            S4SixSitePhase.PASS_B_RUNNING,
        )
        self._validate_warm_identity()
        current = torch.cuda.current_stream(self.device)
        try:
            with tvm_ffi.use_torch_stream(torch.cuda.stream(current)):
                ordinal = (
                    self.device.index
                    if self.device.index is not None
                    else torch.cuda.current_device()
                )
                if int(
                    tvm_ffi.get_raw_stream(tvm_ffi.device(f"cuda:{ordinal}"))
                ) != int(current.cuda_stream):
                    self._poison("SIX_SITE_FFI_STREAM_IDENTITY_MISMATCH")
                result = self.function(*self.argument_views)
        except Exception as error:  # pylint: disable=broad-exception-caught
            self._result_owner = None
            self._poison(f"SIX_SITE_VM_EXECUTION_FAILED:{type(error).__name__}")
        self._result_owner = result
        self._vm_invocation_count = 1
        self._graph_submission_count = 1
        pointer_exact = 0
        for ordinal, target in enumerate(self.values):
            output = torch.from_dlpack(result[ordinal])
            pointer_exact += int(output.data_ptr() == target.data_ptr())
        self._output_pointer_exact_count = pointer_exact
        if pointer_exact != 6:
            self._poison("SIX_SITE_OUTPUT_POINTER_MISMATCH")
        self.phase = S4SixSitePhase.VALUES_READY
        self._phase_order.append(self.phase.value)
        return self.values

    def handoff_to_coefficient_recompute(self) -> S4SixSiteValueResultV1:
        self._advance(
            S4SixSitePhase.VALUES_READY,
            S4SixSitePhase.COEFFICIENT_RECOMPUTE_READY,
        )
        receipt = S4SixSiteValueReceiptV1(
            selector_receipt_hash=self._selector_receipt_hash,
            source_relax_ir_hash=self.compiled.source_relax_ir_hash,
            partitioned_relax_ir_hash=self.compiled.partitioned_relax_ir_hash,
            lowered_relax_ir_hash=self.compiled.lowered_relax_ir_hash,
            device_source_hashes=self.compiled.device_source_hashes,
            phase_order=tuple(self._phase_order),
            read_argument_count=len(self.read_arguments),
            write_target_count=7,
            total_argument_count=len(self.arguments),
            s4_1b_union_descriptor_count=90,
            s4_1abc_union_descriptor_count=110,
            base_overlap_count=5,
            prepare_dlpack_view_count=len(self.argument_views) + 6,
            warm_dlpack_view_count=self._warm_dlpack_view_count,
            value_arena_elements=self.value_arena.numel(),
            value_arena_bytes=self.value_arena.numel()
            * self.value_arena.element_size(),
            value_slot_count=len(self.values),
            selected_input_alias_exact=True,
            selected_input_live_reader_count=0,
            cudnn_partition_function_count=self.compiled.cudnn_partition_function_count,
            cudnn_conv_call_count=self.compiled.cudnn_conv_call_count,
            selected_tir_count=self.compiled.selected_tir_count,
            logical_stage_count=6,
            persistent_output_copy_count=6,
            vm_invocation_count=self._vm_invocation_count,
            graph_submission_count=self._graph_submission_count,
            dynamic_output_allocation_count=0,
            result_owner_capacity=1,
            output_pointer_exact_count=self._output_pointer_exact_count,
            fallback_count=0,
            eager_candidate_count=0,
            native_shadow_count=0,
            saved_dense_a_count=0,
            timing_recorded=False,
            performance_claimed=False,
        )
        result = S4SixSiteValueResultV1(values=self.values, receipt=receipt)
        result.validate()
        return result

    def close(self) -> None:
        self._result_owner = None
        self.argument_views = ()
        self.phase = S4SixSitePhase.CLOSED


__all__ = [
    "PreparedS4SixSiteValueV1",
    "S4_SIX_SITE_READ_SPECS",
    "S4_SIX_SITE_RUNTIME_SCHEMA",
    "S4SixSitePhase",
    "S4SixSiteRuntimeError",
    "S4SixSiteValueReceiptV1",
    "S4SixSiteValueResultV1",
]
