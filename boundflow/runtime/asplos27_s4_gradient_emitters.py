"""S4-1C production-order compressed-gradient and terminal-lA runtime."""

# mypy: disable-error-code=import-untyped
# pylint: disable=import-error,import-outside-toplevel,protected-access
# pylint: disable=too-many-instance-attributes,too-many-locals,too-many-statements
# pylint: disable=too-many-arguments,too-many-positional-arguments
# pylint: disable=missing-function-docstring,too-many-boolean-expressions
# pylint: disable=missing-class-docstring,too-few-public-methods
# pylint: disable=too-many-branches

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
from typing import Any, NoReturn

import torch

from boundflow.backends.tvm.asplos27_s4_compressed_gradient import (
    CompiledS4CompressedGradientV1,
    S4_COMPRESSED_GRADIENT_CONSTRUCTION_HASH_V1,
    S4_DBETA31_SYMBOL_V1,
    S4_GRADIENT_EXPORTED_SYMBOLS_V1,
    S4_GRADIENT_SITE_SPECS_V1,
    compile_s4_compressed_gradient_v1,
)
from boundflow.backends.tvm.r3_d1c_wrapper_schedule import (
    R3D1C_RESIDUAL11_STAGE1,
    R3D1C_RESIDUAL11_STAGE2,
    R3D1C_RESIDUAL6_STAGE1,
    R3D1C_RESIDUAL6_STAGE2,
)
from boundflow.backends.tvm.r3_full_lower_forward import (
    R31B1_LINEAR14_SYMBOL,
    R31B1_LINEAR16_SYMBOL,
    R31B1_SEED_SYMBOL,
)
from boundflow.runtime.asplos27_s4_ordered_buffer_abi import (
    PreparedS4MutableBuffersV1,
)
from boundflow.runtime.asplos27_s4_compact_coefficient import (
    PreparedS4CompactCoefficientV1,
)
from boundflow.runtime.asplos27_s4_six_site_value import S4SixSiteValueResultV1
from boundflow.runtime.r3_d2b_staged_backward import (
    PreparedR3D2BStagedBackwardCandidateV1,
)
from boundflow.runtime.rvir_v4_production_state import production_tensor_sha256

S4_GRADIENT_RUNTIME_SCHEMA_V1 = "boundflow.asplos27-s4-gradient-runtime/v1"
S4_SITE_ORDER_V1 = (17, 19, 23, 25, 28, 31)
S4_PASS_C_REVERSE_SITE_ORDER_V1 = (31, 28, 25, 23, 19, 17)
S4_BETA_LOCATION_V1 = (17, 17, 31, 17, 17, 31)
S4_BETA_SIGN_V1 = (1, 1, 1, -1, -1, -1)

S4_NONTERMINAL_ACTIONS_V1 = (
    "seed",
    "linear16_right",
    "emit_dalpha31",
    "emit_dbeta31",
    "relu31_coefficient",
    "linear14_right",
    "emit_dalpha28",
    "relu28_coefficient",
    "residual11_stage1",
    "emit_dalpha25",
    "residual11_stage2",
    "emit_dalpha23",
    "relu23_coefficient",
    "residual6_stage1",
    "emit_dalpha19",
    "residual6_stage2",
    "emit_dalpha17",
)


def _terminal_actions() -> tuple[str, ...]:
    actions: list[str] = []
    for action in S4_NONTERMINAL_ACTIONS_V1:
        actions.append(action)
        if action == "emit_dbeta31":
            actions.append("copy_terminal_la31")
        elif action.startswith("emit_dalpha"):
            site = int(action.removeprefix("emit_dalpha"))
            if site != 31:
                actions.append(f"copy_terminal_la{site}")
    return tuple(actions)


S4_TERMINAL_ACTIONS_V1 = _terminal_actions()


class S4GradientRuntimeError(RuntimeError):
    """Stable fail-closed S4-1C runtime error."""


def _reject(reason: str) -> NoReturn:
    raise S4GradientRuntimeError(reason)


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


class S4GradientPhase(str, Enum):
    PREPARED = "prepared"
    RUNNING = "running"
    COMPLETE = "complete"
    TERMINAL_LEASED = "terminal_leased"
    CLOSED = "closed"
    POISONED = "poisoned"


@dataclass(frozen=True)
class S4GradientRuntimeReceiptV1:
    construction_hash: str
    prepared_id: str
    value_receipt_hash: str
    mutable_buffer_receipt_hash: str
    metadata_identity_hash: str
    template_hash: str
    scheduled_tir_hash: str
    device_source_hash: str
    exported_symbols: tuple[str, ...]
    evaluation_generation: int
    state_version: int
    mode: str
    action_inventory: tuple[str, ...]
    action_count: int
    coefficient_action_count: int
    dalpha_launch_count: int
    dbeta_launch_count: int
    terminal_copy_count: int
    stream_id: int
    device_ordinal: int
    emitter_argument_occurrences: int
    emitter_unique_view_count: int
    full_prepared_descriptor_union_count: int
    prepare_dlpack_view_count: int
    warm_dlpack_view_count: int
    dynamic_output_allocation_count: int
    value_arena_elements: int
    value_arena_physical_storage_count: int
    saved_dense_a_count: int
    dense_gradient_escape_count: int
    fallback_count: int
    eager_candidate_count: int
    native_shadow_count: int
    timing_recorded: bool
    performance_claimed: bool
    schema_version: str = S4_GRADIENT_RUNTIME_SCHEMA_V1

    def validate(self) -> None:
        expected = (
            S4_TERMINAL_ACTIONS_V1
            if self.mode == "terminal"
            else S4_NONTERMINAL_ACTIONS_V1
        )
        expected_copies = 6 if self.mode == "terminal" else 0
        if (
            self.schema_version != S4_GRADIENT_RUNTIME_SCHEMA_V1
            or self.construction_hash != S4_COMPRESSED_GRADIENT_CONSTRUCTION_HASH_V1
            or any(
                len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in (
                    self.prepared_id,
                    self.value_receipt_hash,
                    self.mutable_buffer_receipt_hash,
                    self.metadata_identity_hash,
                    self.template_hash,
                    self.scheduled_tir_hash,
                    self.device_source_hash,
                )
            )
            or self.exported_symbols != S4_GRADIENT_EXPORTED_SYMBOLS_V1
            or self.evaluation_generation < 0
            or self.state_version < 0
            or self.mode not in {"nonterminal", "terminal"}
            or self.action_inventory != expected
            or self.action_count != len(expected)
            or self.coefficient_action_count != 10
            or self.dalpha_launch_count != 6
            or self.dbeta_launch_count != 1
            or self.terminal_copy_count != expected_copies
            or self.stream_id <= 0
            or self.device_ordinal < 0
            or self.emitter_argument_occurrences != 53
            or self.emitter_unique_view_count != 46
            or self.full_prepared_descriptor_union_count != 110
            or self.prepare_dlpack_view_count != 46
            or self.warm_dlpack_view_count
            or self.dynamic_output_allocation_count
            or self.value_arena_elements != 37_464
            or self.value_arena_physical_storage_count != 1
            or self.saved_dense_a_count
            or self.dense_gradient_escape_count
            or self.fallback_count
            or self.eager_candidate_count
            or self.native_shadow_count
            or self.timing_recorded
            or self.performance_claimed
        ):
            _reject("S4_GRADIENT_RECEIPT_MISMATCH")

    def stable_hash(self) -> str:
        self.validate()
        return _canonical_hash(asdict(self))


@dataclass(frozen=True)
class S4GradientResultV1:
    gradients: tuple[torch.Tensor, ...]
    receipt: S4GradientRuntimeReceiptV1

    def validate(self) -> None:
        self.receipt.validate()
        widths = tuple(spec.alpha_width for spec in S4_GRADIENT_SITE_SPECS_V1) + (1,)
        if len(self.gradients) != 7:
            _reject("S4_GRADIENT_OUTPUT_INVENTORY_MISMATCH")
        for tensor, width in zip(self.gradients, widths):
            if (
                tuple(tensor.shape) != (6, width)
                or tensor.dtype != torch.float32
                or not tensor.is_contiguous()
            ):
                _reject("S4_GRADIENT_OUTPUT_LAYOUT_MISMATCH")


class NativeTerminalLowerAdjointLeaseS4V1:
    """One-shot lease over six shaped views of the overwritten V/lA arena."""

    __slots__ = ("_views", "_generation", "_consumed")

    def __init__(self, views: tuple[torch.Tensor, ...], generation: int) -> None:
        self._views = views
        self._generation = generation
        self._consumed = False

    def consume(self, *, evaluation_generation: int) -> tuple[torch.Tensor, ...]:
        if self._consumed:
            _reject("S4_TERMINAL_LA_LEASE_REUSED")
        if evaluation_generation != self._generation:
            _reject("S4_GRADIENT_STATE_VERSION_MISMATCH")
        self._consumed = True
        return self._views


class S4GradientModuleCacheV1:
    def __init__(self) -> None:
        self._entries: dict[str, CompiledS4CompressedGradientV1] = {}

    def get(self, compute_capability: str) -> CompiledS4CompressedGradientV1:
        compiled = self._entries.get(compute_capability)
        if compiled is None:
            compiled = compile_s4_compressed_gradient_v1(
                compute_capability=compute_capability
            )
            self._entries[compute_capability] = compiled
        return compiled


S4_GRADIENT_MODULE_CACHE_V1 = S4GradientModuleCacheV1()


class PreparedS4GradientEmittersV1:
    """Prepared S4-1C owner with exact 17/23-action execution."""

    def __init__(
        self,
        executor: PreparedR3D2BStagedBackwardCandidateV1,
        value_result: S4SixSiteValueResultV1,
        mutable_buffers: PreparedS4MutableBuffersV1,
        *,
        evaluation_generation: int,
        state_version: int,
        compiled: CompiledS4CompressedGradientV1 | None = None,
        compact_coefficient: PreparedS4CompactCoefficientV1 | None = None,
    ) -> None:
        import tvm

        value_result.validate()
        mutable_buffers.receipt.validate()
        resources = mutable_buffers._resources
        if resources is None or resources._lower is None or resources._upstream is None:
            _reject("S4_GRADIENT_STATE_VERSION_MISMATCH")
        device = executor.device
        if device.type != "cuda" or any(
            value.device != device for value in value_result.values
        ):
            _reject("S4_GRADIENT_DEVICE_MISMATCH")
        ordinal = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        capability = torch.cuda.get_device_capability(ordinal)
        capability_name = f"sm_{capability[0]}{capability[1]}"
        self.compiled = compiled or S4_GRADIENT_MODULE_CACHE_V1.get(capability_name)
        self.compiled.validate()
        self.executor = executor
        self.compact_coefficient = compact_coefficient
        self.value_result = value_result
        self.mutable_buffers = mutable_buffers
        self.resources = resources
        self.device = device
        self.device_ordinal = ordinal
        self.evaluation_generation = evaluation_generation
        self.state_version = state_version
        self.phase = S4GradientPhase.PREPARED
        self._actions: list[str] = []
        self._terminal = False
        self._lease: NativeTerminalLowerAdjointLeaseS4V1 | None = None

        self.values = tuple(
            value.reshape(6, 1, spec.feature_count)
            for value, spec in zip(value_result.values, S4_GRADIENT_SITE_SPECS_V1)
        )
        if len({int(value.untyped_storage()._cdata) for value in self.values}) != 1:
            _reject("S4_VALUE_SITE_NOT_READY")
        self.gradients = tuple(resources._gradients)
        if len(self.gradients) != 7 or len(resources._parameters) != 7:
            _reject("S4_GRADIENT_OUTPUT_INVENTORY_MISMATCH")
        self.active_alpha = tuple(resources._parameters[:6])
        self.active_beta = resources._parameters[6]
        self.upstream = resources._upstream
        self.alpha_indices = tuple(
            torch.tensor(
                layout.alpha_flat_indices,
                dtype=torch.int32,
                device=device,
            ).contiguous()
            for layout in executor.plan.relu_layouts
        )
        if tuple(
            layout.native_preactivation for layout in executor.plan.relu_layouts
        ) != tuple(str(site) for site in S4_SITE_ORDER_V1):
            _reject("S4_GRADIENT_SITE_INVENTORY_MISMATCH")
        self.beta_location = torch.tensor(
            S4_BETA_LOCATION_V1, dtype=torch.int32, device=device
        ).view(6, 1)
        self.beta_sign = torch.tensor(
            S4_BETA_SIGN_V1, dtype=torch.int8, device=device
        ).view(6, 1)
        self.incoming = (
            executor.forward_executor.scratch_0[:12288].view(6, 1, 2048),
            executor._residual6_scratch.view(6, 1, 1024),
            executor.forward_executor.scratch_1[:6144].view(6, 1, 1024),
            executor._residual11_scratch.view(6, 1, 1024),
            executor.forward_executor.scratch_0[:6144].view(6, 1, 1024),
            executor.forward_executor.scratch_1[:600].view(6, 1, 100),
        )
        by_site = {
            spec.site_id: ordinal
            for ordinal, spec in enumerate(S4_GRADIENT_SITE_SPECS_V1)
        }
        self._by_site = by_site
        self.bounds = tuple(
            (
                executor._tensor(f"relu/{spec.site_id}/lower").reshape(
                    6, spec.feature_count
                ),
                executor._tensor(f"relu/{spec.site_id}/upper").reshape(
                    6, spec.feature_count
                ),
            )
            for spec in S4_GRADIENT_SITE_SPECS_V1
        )
        self.terminal_views = tuple(
            value.reshape(spec.terminal_shape)
            for value, spec in zip(value_result.values, S4_GRADIENT_SITE_SPECS_V1)
        )
        self._validate_static_legality()

        if compact_coefficient is None:
            for spec, active in zip(S4_GRADIENT_SITE_SPECS_V1, self.active_alpha):
                source = executor._tensor(f"relu/{spec.site_id}/alpha")[0, 0]
                if production_tensor_sha256(active) != production_tensor_sha256(source):
                    _reject("S4_GRADIENT_STATE_VERSION_MISMATCH")
            if production_tensor_sha256(self.active_beta) != production_tensor_sha256(
                executor._tensor("relu/31/beta")
            ):
                _reject("S4_GRADIENT_STATE_VERSION_MISMATCH")
        elif (
            compact_coefficient.executor is not executor
            or compact_coefficient.mutable_buffers is not mutable_buffers
            or any(
                left.data_ptr() != right.data_ptr()
                for left, right in zip(
                    compact_coefficient.active_alpha, self.active_alpha
                )
            )
            or compact_coefficient.active_beta.data_ptr() != self.active_beta.data_ptr()
        ):
            _reject("S4_GRADIENT_STATE_VERSION_MISMATCH")

        unique_tensors = (
            *self.incoming,
            *self.values,
            *(item for pair in self.bounds for item in pair),
            *self.active_alpha,
            *self.alpha_indices,
            self.upstream,
            *self.gradients[:6],
            self.beta_location,
            self.beta_sign,
            self.gradients[6],
        )
        unique: dict[tuple[int, tuple[int, ...], str], torch.Tensor] = {}
        for tensor in unique_tensors:
            key = (tensor.data_ptr(), tuple(tensor.shape), str(tensor.dtype))
            unique[key] = tensor
        if len(unique) != 46:
            _reject("S4_GRADIENT_VIEW_INVENTORY_MISMATCH")
        self._views = {
            key: tvm.runtime.from_dlpack(tensor) for key, tensor in unique.items()
        }
        self._identity = tuple(
            (key, str(tensor.device), tuple(tensor.stride()))
            for key, tensor in unique.items()
        )
        self._metadata_hashes = self._current_metadata_hashes()
        self._value_receipt_hash = _canonical_hash(value_result.receipt.to_dict())
        self._mutable_buffer_receipt_hash = mutable_buffers.receipt.stable_hash()
        self._metadata_identity_hash = _canonical_hash(self._metadata_hashes)
        self._prepared_id = _canonical_hash(
            {
                "construction": S4_COMPRESSED_GRADIENT_CONSTRUCTION_HASH_V1,
                "value": self._value_receipt_hash,
                "buffers": self._mutable_buffer_receipt_hash,
                "metadata": self._metadata_identity_hash,
                "template": self.compiled.template_hash,
                "schedule": self.compiled.scheduled_tir_hash,
                "device": self.compiled.device_source_hash,
                "evaluation_generation": evaluation_generation,
                "state_version": state_version,
            }
        )

    def rearm(
        self,
        value_result: S4SixSiteValueResultV1,
        *,
        evaluation_generation: int,
        state_version: int,
    ) -> None:
        """Reuse prepared emitter views after a completed nonterminal generation."""

        value_result.validate()
        if (
            self.phase != S4GradientPhase.COMPLETE
            or self._terminal
            or evaluation_generation <= self.evaluation_generation
            or state_version <= self.state_version
            or any(
                left.data_ptr() != right.data_ptr()
                for left, right in zip(value_result.values, self.value_result.values)
            )
        ):
            self._poison("S4_GRADIENT_STATE_VERSION_MISMATCH")
        self.value_result = value_result
        self.evaluation_generation = evaluation_generation
        self.state_version = state_version
        self._actions.clear()
        self._terminal = False
        self._lease = None
        self._value_receipt_hash = _canonical_hash(value_result.receipt.to_dict())
        self._prepared_id = _canonical_hash(
            {
                "construction": S4_COMPRESSED_GRADIENT_CONSTRUCTION_HASH_V1,
                "value": self._value_receipt_hash,
                "buffers": self._mutable_buffer_receipt_hash,
                "metadata": self._metadata_identity_hash,
                "template": self.compiled.template_hash,
                "schedule": self.compiled.scheduled_tir_hash,
                "device": self.compiled.device_source_hash,
                "evaluation_generation": evaluation_generation,
                "state_version": state_version,
            }
        )
        self.phase = S4GradientPhase.PREPARED

    def _poison(self, reason: str) -> NoReturn:
        self.phase = S4GradientPhase.POISONED
        _reject(reason)

    def _current_metadata_hashes(self) -> tuple[str, ...]:
        return tuple(
            production_tensor_sha256(value)
            for value in (*self.alpha_indices, self.beta_location, self.beta_sign)
        )

    def _validate_static_legality(self) -> None:
        for spec, alpha, indices in zip(
            S4_GRADIENT_SITE_SPECS_V1, self.active_alpha, self.alpha_indices
        ):
            host = indices.detach().cpu().tolist()
            if (
                tuple(alpha.shape) != (6, spec.alpha_width)
                or len(host) != spec.alpha_width
                or any(index < 0 or index >= spec.feature_count for index in host)
                or any(left >= right for left, right in zip(host, host[1:]))
            ):
                _reject("S4_ALPHA_INDEX_INVALID")
        if (
            tuple(self.active_beta.shape) != (6, 1)
            or tuple(self.beta_location.shape) != (6, 1)
            or tuple(self.beta_sign.shape) != (6, 1)
        ):
            _reject("S4_DBETA_SITE_OR_INVENTORY_MISMATCH")

    def _view(self, tensor: torch.Tensor) -> Any:
        key = (tensor.data_ptr(), tuple(tensor.shape), str(tensor.dtype))
        view = self._views.get(key)
        if view is None:
            self._poison("S4_GRADIENT_OUTPUT_POINTER_DRIFT")
        return view

    def _record(self, action: str, expected: tuple[str, ...]) -> None:
        ordinal = len(self._actions)
        if ordinal >= len(expected) or expected[ordinal] != action:
            self._poison("S4_PASS_C_ACTION_ORDER_MISMATCH")
        self._actions.append(action)

    def _emit_alpha(self, site: int, expected: tuple[str, ...]) -> None:
        ordinal = self._by_site[site]
        spec = S4_GRADIENT_SITE_SPECS_V1[ordinal]
        lower, upper = self.bounds[ordinal]
        self.compiled.executable[spec.dalpha_symbol](
            self._view(self.incoming[ordinal]),
            self._view(self.values[ordinal]),
            self._view(lower),
            self._view(upper),
            self._view(self.active_alpha[ordinal]),
            self._view(self.alpha_indices[ordinal]),
            self._view(self.upstream),
            self._view(self.gradients[ordinal]),
        )
        self._record(f"emit_dalpha{site}", expected)

    def _emit_beta(self, expected: tuple[str, ...]) -> None:
        ordinal = self._by_site[31]
        self.compiled.executable[S4_DBETA31_SYMBOL_V1](
            self._view(self.values[ordinal]),
            self._view(self.beta_location),
            self._view(self.beta_sign),
            self._view(self.upstream),
            self._view(self.gradients[6]),
        )
        self._record("emit_dbeta31", expected)

    def _copy_terminal(self, site: int, expected: tuple[str, ...]) -> None:
        if not self._terminal:
            self._poison("S4_TERMINAL_COPY_IN_NONTERMINAL")
        ordinal = self._by_site[site]
        spec = S4_GRADIENT_SITE_SPECS_V1[ordinal]
        self.compiled.executable[spec.copy_symbol](
            self._view(self.incoming[ordinal]), self._view(self.values[ordinal])
        )
        self._record(f"copy_terminal_la{site}", expected)

    def _validate_warm(self) -> int:
        import tvm_ffi

        current_metadata = self._current_metadata_hashes()
        if current_metadata != self._metadata_hashes:
            self._poison("S4_GRADIENT_STATE_VERSION_MISMATCH")
        current = torch.cuda.current_stream(self.device)
        stream_id = int(current.cuda_stream)
        if stream_id == int(torch.cuda.default_stream(self.device).cuda_stream):
            self._poison("S4_GRADIENT_CROSS_STREAM_USE")
        ffi_stream = int(
            tvm_ffi.get_raw_stream(tvm_ffi.device(f"cuda:{self.device_ordinal}"))
        )
        if ffi_stream not in (0, stream_id):
            self._poison("S4_GRADIENT_CROSS_STREAM_USE")
        return stream_id

    def run(self, *, terminal: bool) -> S4GradientResultV1:
        """Execute exactly one single-evaluation Pass C."""

        import tvm_ffi

        if self.phase != S4GradientPhase.PREPARED:
            self._poison("S4_GRADIENT_STATE_VERSION_MISMATCH")
        self.phase = S4GradientPhase.RUNNING
        self._terminal = terminal
        expected = S4_TERMINAL_ACTIONS_V1 if terminal else S4_NONTERMINAL_ACTIONS_V1
        stream_id = self._validate_warm()
        executor = self.executor
        s0 = executor.forward_executor.scratch_0
        s1 = executor.forward_executor.scratch_1
        bias = executor.forward_executor.bias_accumulator
        current = torch.cuda.current_stream(self.device)
        compact = self.compact_coefficient
        try:
            with tvm_ffi.use_torch_stream(torch.cuda.stream(current)):
                executor._launch_b1(
                    R31B1_SEED_SYMBOL, executor._tensor("objective"), s0[:60], bias
                )
                self._record("seed", expected)
                executor._launch_b1(
                    R31B1_LINEAR16_SYMBOL,
                    s0[:60],
                    executor._tensor("param/linear2.weight"),
                    executor._tensor("param/linear2.bias"),
                    bias,
                    s1[:600],
                    bias,
                )
                self._record("linear16_right", expected)
                self._emit_alpha(31, expected)
                self._emit_beta(expected)
                if terminal:
                    self._copy_terminal(31, expected)
                if compact is None:
                    executor._relu_coefficient("31", s1[:600])
                else:
                    compact.relu(31, s1[:600])
                self._record("relu31_coefficient", expected)
                executor._launch_b1(
                    R31B1_LINEAR14_SYMBOL,
                    s1[:600],
                    executor._tensor("param/linear1.weight"),
                    executor._tensor("param/linear1.bias"),
                    bias,
                    s0[:6144],
                    bias,
                )
                self._record("linear14_right", expected)
                self._emit_alpha(28, expected)
                if terminal:
                    self._copy_terminal(28, expected)
                if compact is None:
                    executor._relu_coefficient("28", s0[:6144])
                else:
                    compact.relu(28, s0[:6144])
                self._record("relu28_coefficient", expected)
                executor._launch_d2b(
                    R3D1C_RESIDUAL11_STAGE1,
                    s0,
                    executor._tensor("param/layer1.1.conv2.weight"),
                    executor._residual11_scratch,
                )
                self._record("residual11_stage1", expected)
                self._emit_alpha(25, expected)
                if terminal:
                    self._copy_terminal(25, expected)
                if compact is None:
                    executor._launch_d2b(
                        R3D1C_RESIDUAL11_STAGE2,
                        s0,
                        executor._residual11_scratch,
                        executor._tensor("relu/25/lower").reshape(6, 1024),
                        executor._tensor("relu/25/upper").reshape(6, 1024),
                        executor._tensor("relu/25/alpha"),
                        executor.forward_executor.alpha_maps["25"],
                        executor._tensor("param/layer1.1.conv1.weight"),
                        executor._tensor("param/layer1.1.conv2.bias"),
                        executor._tensor("param/layer1.1.conv1.bias"),
                        bias,
                        s1[:6144],
                        bias,
                    )
                else:
                    compact.residual11_stage2(
                        s0, executor._residual11_scratch, s1[:6144], bias
                    )
                self._record("residual11_stage2", expected)
                self._emit_alpha(23, expected)
                if terminal:
                    self._copy_terminal(23, expected)
                if compact is None:
                    executor._relu_coefficient("23", s1[:6144])
                else:
                    compact.relu(23, s1[:6144])
                self._record("relu23_coefficient", expected)
                executor._launch_d2b(
                    R3D1C_RESIDUAL6_STAGE1,
                    s1,
                    executor._tensor("param/layer1.0.conv2.weight"),
                    executor._residual6_scratch,
                )
                self._record("residual6_stage1", expected)
                self._emit_alpha(19, expected)
                if terminal:
                    self._copy_terminal(19, expected)
                if compact is None:
                    executor._launch_d2b(
                        R3D1C_RESIDUAL6_STAGE2,
                        s1,
                        executor._residual6_scratch,
                        executor._tensor("relu/19/lower").reshape(6, 1024),
                        executor._tensor("relu/19/upper").reshape(6, 1024),
                        executor._tensor("relu/19/alpha"),
                        executor.forward_executor.alpha_maps["19"],
                        executor._tensor("param/layer1.0.conv1.weight"),
                        executor._tensor("param/layer1.0.shortcut.0.weight"),
                        executor._tensor("param/layer1.0.conv2.bias"),
                        executor._tensor("param/layer1.0.conv1.bias"),
                        executor._tensor("param/layer1.0.shortcut.0.bias"),
                        bias,
                        s0[:12288],
                        bias,
                    )
                else:
                    compact.residual6_stage2(
                        s1, executor._residual6_scratch, s0[:12288], bias
                    )
                self._record("residual6_stage2", expected)
                self._emit_alpha(17, expected)
                if terminal:
                    self._copy_terminal(17, expected)
        except S4GradientRuntimeError:
            raise
        except Exception as error:  # pylint: disable=broad-exception-caught
            self._poison(f"S4_GRADIENT_EXECUTION_FAILED:{type(error).__name__}")
        if tuple(self._actions) != expected:
            self._poison("S4_PASS_C_ACTION_COUNT_MISMATCH")
        self.phase = S4GradientPhase.COMPLETE
        if terminal:
            self._lease = NativeTerminalLowerAdjointLeaseS4V1(
                self.terminal_views, self.evaluation_generation
            )
        receipt = S4GradientRuntimeReceiptV1(
            construction_hash=S4_COMPRESSED_GRADIENT_CONSTRUCTION_HASH_V1,
            prepared_id=self._prepared_id,
            value_receipt_hash=self._value_receipt_hash,
            mutable_buffer_receipt_hash=self._mutable_buffer_receipt_hash,
            metadata_identity_hash=self._metadata_identity_hash,
            template_hash=self.compiled.template_hash,
            scheduled_tir_hash=self.compiled.scheduled_tir_hash,
            device_source_hash=self.compiled.device_source_hash,
            exported_symbols=self.compiled.exported_symbols,
            evaluation_generation=self.evaluation_generation,
            state_version=self.state_version,
            mode="terminal" if terminal else "nonterminal",
            action_inventory=tuple(self._actions),
            action_count=len(self._actions),
            coefficient_action_count=10,
            dalpha_launch_count=6,
            dbeta_launch_count=1,
            terminal_copy_count=6 if terminal else 0,
            stream_id=stream_id,
            device_ordinal=self.device_ordinal,
            emitter_argument_occurrences=53,
            emitter_unique_view_count=46,
            full_prepared_descriptor_union_count=110,
            prepare_dlpack_view_count=len(self._views),
            warm_dlpack_view_count=0,
            dynamic_output_allocation_count=0,
            value_arena_elements=sum(value.numel() for value in self.values),
            value_arena_physical_storage_count=len(
                {int(value.untyped_storage()._cdata) for value in self.values}
            ),
            saved_dense_a_count=0,
            dense_gradient_escape_count=0,
            fallback_count=0,
            eager_candidate_count=0,
            native_shadow_count=0,
            timing_recorded=False,
            performance_claimed=False,
        )
        result = S4GradientResultV1(gradients=self.gradients, receipt=receipt)
        result.validate()
        return result

    def take_terminal_lease(self) -> NativeTerminalLowerAdjointLeaseS4V1:
        if self.phase != S4GradientPhase.COMPLETE or self._lease is None:
            self._poison("S4_TERMINAL_LA_INVENTORY_INCOMPLETE")
        self.phase = S4GradientPhase.TERMINAL_LEASED
        return self._lease

    def close(self) -> None:
        self._views.clear()
        self.phase = S4GradientPhase.CLOSED


__all__ = [
    "NativeTerminalLowerAdjointLeaseS4V1",
    "PreparedS4GradientEmittersV1",
    "S4_BETA_LOCATION_V1",
    "S4_BETA_SIGN_V1",
    "S4_GRADIENT_RUNTIME_SCHEMA_V1",
    "S4_NONTERMINAL_ACTIONS_V1",
    "S4_TERMINAL_ACTIONS_V1",
    "S4GradientPhase",
    "S4GradientResultV1",
    "S4GradientRuntimeError",
    "S4GradientRuntimeReceiptV1",
]
