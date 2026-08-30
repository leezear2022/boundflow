"""S4-1A ordered persistent mutable-buffer preparation.

The implementation consumes the one-shot S4-0 live-state lease, creates the
six compressed alpha parameters and the only active beta parameter, and owns
their persistent gradient/output storage plus prepare-time DLPack views.  It
deliberately contains no CROWN evaluator, optimizer, timing, or solver commit.
"""

# pylint: disable=too-many-lines,too-many-instance-attributes,too-many-arguments
# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=protected-access,too-few-public-methods,unidiomatic-typecheck
# pylint: disable=line-too-long,broad-exception-caught,missing-function-docstring
# pylint: disable=too-many-boolean-expressions,too-many-positional-arguments
# pylint: disable=use-dict-literal

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
from typing import NoReturn

import torch

from boundflow.ir.verification_graph import VerificationRejectionReason
from boundflow.runtime import asplos27_s4_mutable_state_admission as _admission
from boundflow.runtime.asplos27_s4_mutable_state_admission import (
    PreparedS4MutableStateAdmissionV1,
    S4MutableStateAdmissionError,
)
from boundflow.runtime.rvir_v4_production_state import production_tensor_sha256

S4_MUTABLE_BUFFER_SCHEMA_V1 = "boundflow.asplos27-s4-ordered-buffer-abi/v1"

_DETAIL_CODES = (
    "BUFFER_PREPARE_EXACT_CALL_MISMATCH",
    "BUFFER_PREPARE_ALREADY_ATTEMPTED",
    "BUFFER_PREPARE_OWNER_CONTEXT_MISMATCH",
    "BUFFER_PREPARE_SOURCE_IDENTITY_MISMATCH",
    "BUFFER_PREPARE_SOURCE_READ_RACE",
    "BUFFER_PREPARE_MANIFEST_MISMATCH",
    "PARAMETER_SOURCE_STORAGE_ALIAS",
    "PARAMETER_GRADIENT_STORAGE_ALIAS",
    "CANDIDATE_STORAGE_ALIAS",
    "BUFFER_INITIAL_CONTENT_MISMATCH",
    "BASE_DLPACK_VIEW_KEY_MISMATCH",
    "BASE_DLPACK_VIEW_COUNT_MISMATCH",
    "BUFFER_PREPARE_VALIDATION_COPY_ACCOUNTING_MISMATCH",
    "BUFFER_PREPARE_RESOURCE_CONTEXT_RETAINED",
    "BUFFER_PREPARE_ERROR_CONTEXT_RETAINED",
    "BUFFER_PREPARE_CLEANUP_INCOMPLETE",
    "BUFFER_PREPARE_ADOPTION_OWNER_MISMATCH",
    "BUFFER_PREPARE_SCOPE_ESCAPE",
    "BUFFER_PREPARE_FALLBACK_OR_RETRY_FORBIDDEN",
    "BUFFER_PREPARE_SERIALIZATION_FORBIDDEN",
)

_REASON_BY_DETAIL = {
    "BUFFER_PREPARE_EXACT_CALL_MISMATCH": VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
    "BUFFER_PREPARE_ALREADY_ATTEMPTED": VerificationRejectionReason.UNSAFE_ALIAS_OR_LIFETIME,
    "BUFFER_PREPARE_OWNER_CONTEXT_MISMATCH": VerificationRejectionReason.UNSAFE_ALIAS_OR_LIFETIME,
    "BUFFER_PREPARE_SOURCE_IDENTITY_MISMATCH": VerificationRejectionReason.STATE_VERSION_MISMATCH,
    "BUFFER_PREPARE_SOURCE_READ_RACE": VerificationRejectionReason.STATE_VERSION_MISMATCH,
    "BUFFER_PREPARE_MANIFEST_MISMATCH": VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
    "PARAMETER_SOURCE_STORAGE_ALIAS": VerificationRejectionReason.UNSAFE_ALIAS_OR_LIFETIME,
    "PARAMETER_GRADIENT_STORAGE_ALIAS": VerificationRejectionReason.UNSAFE_ALIAS_OR_LIFETIME,
    "CANDIDATE_STORAGE_ALIAS": VerificationRejectionReason.UNSAFE_ALIAS_OR_LIFETIME,
    "BUFFER_INITIAL_CONTENT_MISMATCH": VerificationRejectionReason.VJP_OWNER_OR_SAVED_STATE_MISMATCH,
    "BASE_DLPACK_VIEW_KEY_MISMATCH": VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
    "BASE_DLPACK_VIEW_COUNT_MISMATCH": VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
    "BUFFER_PREPARE_VALIDATION_COPY_ACCOUNTING_MISMATCH": VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
    "BUFFER_PREPARE_RESOURCE_CONTEXT_RETAINED": VerificationRejectionReason.UNSAFE_ALIAS_OR_LIFETIME,
    "BUFFER_PREPARE_ERROR_CONTEXT_RETAINED": VerificationRejectionReason.UNSAFE_ALIAS_OR_LIFETIME,
    "BUFFER_PREPARE_CLEANUP_INCOMPLETE": VerificationRejectionReason.UNSAFE_ALIAS_OR_LIFETIME,
    "BUFFER_PREPARE_ADOPTION_OWNER_MISMATCH": VerificationRejectionReason.UNSAFE_ALIAS_OR_LIFETIME,
    "BUFFER_PREPARE_SCOPE_ESCAPE": VerificationRejectionReason.VJP_OWNER_OR_SAVED_STATE_MISMATCH,
    "BUFFER_PREPARE_FALLBACK_OR_RETRY_FORBIDDEN": VerificationRejectionReason.RUNTIME_FALLBACK_REQUIRED,
    "BUFFER_PREPARE_SERIALIZATION_FORBIDDEN": VerificationRejectionReason.UNSAFE_ALIAS_OR_LIFETIME,
}

S4_MUTABLE_BUFFER_CONSTRUCTION_MODEL_V1: dict[str, object] = {
    "claims": {
        "buffer_ownership_validated": True,
        "crown_numeric_semantics": False,
        "local_single_owner_transfer": True,
        "optimizer_trajectory": False,
        "performance": False,
        "process_global_exclusivity": False,
        "provider_mapping_stability": False,
    },
    "cleanup_order": [
        "roundtrip_locals",
        "base_dlpack_views",
        "lower_and_upstream",
        "gradients",
        "parameters",
        "staging_containers",
        "lease_ticket",
        "device_stream_check",
    ],
    "counts": {
        "active_beta_parameter": 1,
        "alpha_parameter": 6,
        "base_dlpack_view": 16,
        "candidate_logical_bytes": 34080,
        "candidate_storage": 16,
        "empty_beta_token": 5,
        "gradient_bytes": 17016,
        "gradient_elements": 4254,
        "leased_source_bytes": 34008,
        "leased_source_elements": 8502,
        "leased_source_tensor": 12,
        "parameter_bytes": 17016,
        "parameter_elements": 4254,
    },
    "detail_codes": list(_DETAIL_CODES),
    "formal_processes": {"isolated_fault": 7, "positive": 5, "total": 12},
    "negative_minimum": 68,
    "phase_order": [
        "begin_single_attempt_ticket",
        "validate_owner_and_exact_call",
        "validate_live_source_envelope",
        "entry_source_capture",
        "derive_ordered_manifest",
        "allocate_alpha_parameters",
        "allocate_active_beta_parameter",
        "allocate_gradients",
        "allocate_lower",
        "allocate_upstream",
        "validate_storage_and_leaf",
        "create_base_dlpack_views",
        "validate_dlpack_roundtrip",
        "exit_source_capture",
        "validate_initialized_content",
        "build_and_validate_receipt",
        "single_owner_adoption",
    ],
    "resource_order": [
        "alpha_parameter_0_5",
        "active_beta_parameter",
        "alpha_gradient_0_5",
        "active_beta_gradient",
        "lower_output",
        "fixed_upstream",
        "base_dlpack_view_0_15",
    ],
    "scope": {
        "buffer_prepare": True,
        "crown_math": False,
        "evaluator": False,
        "optimizer": False,
        "terminal_handoff": False,
        "timing": False,
    },
    "signature": ["prepared_admission", "current_live_sources", "exact_call_id"],
    "validation_accounting": {
        "cumulative_d2h_bytes": 153072,
        "cumulative_d2h_copies": 56,
        "initialized_candidate_d2h_bytes": 17040,
        "initialized_candidate_d2h_copies": 8,
        "parameter_d2d_bytes": 17016,
        "parameter_d2d_copies": 7,
        "prior_s4_0_d2h_bytes": 68016,
        "prior_s4_0_d2h_copies": 24,
        "s4_1a_d2h_bytes": 85056,
        "s4_1a_d2h_copies": 32,
        "source_d2h_bytes": 68016,
        "source_d2h_copies": 24,
        "source_passes": 2,
    },
}


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


S4_MUTABLE_BUFFER_CONSTRUCTION_HASH_V1 = _canonical_hash(
    S4_MUTABLE_BUFFER_CONSTRUCTION_MODEL_V1
)
_EXPECTED_CONSTRUCTION_HASH = (
    "8ad25c2abf1eb98c3b1097bf7acb46aba227f7e94f0c7c03169f39e8da409a9d"
)


class S4MutableBufferPreparationError(RuntimeError):
    """Stable S4-1A rejection without retaining a lower exception."""

    def __init__(
        self,
        detail_code: str,
        *,
        slot_ordinal: int | None = None,
        semantic_role: str | None = None,
    ) -> None:
        self.detail_code = detail_code
        self.verification_reason = _REASON_BY_DETAIL.get(
            detail_code, VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH
        )
        self.slot_ordinal = slot_ordinal
        self.semantic_role = semantic_role
        suffix = "" if semantic_role is None else f":{semantic_role}"
        super().__init__(f"{detail_code}{suffix}")


def _reject(
    detail_code: str,
    *,
    slot_ordinal: int | None = None,
    semantic_role: str | None = None,
) -> NoReturn:
    raise S4MutableBufferPreparationError(
        detail_code, slot_ordinal=slot_ordinal, semantic_role=semantic_role
    )


def _canonical_field(value: object) -> object:
    if isinstance(value, tuple):
        return [_canonical_field(item) for item in value]
    if isinstance(value, VerificationRejectionReason):
        return value.value
    return value


def _tensor_free_walk(value: object, ancestors: set[int] | None = None) -> None:
    if isinstance(value, torch.Tensor):
        _reject("BUFFER_PREPARE_MANIFEST_MISMATCH")
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    active = set() if ancestors is None else ancestors
    identity = id(value)
    if identity in active:
        _reject("BUFFER_PREPARE_MANIFEST_MISMATCH")
    active.add(identity)
    try:
        if isinstance(value, tuple):
            for item in value:
                _tensor_free_walk(item, active)
            return
        if is_dataclass(value) and getattr(type(value), "__dataclass_params__").frozen:
            for item in fields(value):
                _tensor_free_walk(object.__getattribute__(value, item.name), active)
            return
        _reject("BUFFER_PREPARE_MANIFEST_MISMATCH")
    finally:
        active.remove(identity)


def _shape_product(shape: tuple[int, ...]) -> int:
    product = 1
    for dimension in shape:
        product *= dimension
    return product


def _dtype_bytes(dtype: str) -> int:
    if dtype != "torch.float32":
        _reject("BUFFER_PREPARE_MANIFEST_MISMATCH")
    return 4


@dataclass(frozen=True)
class S4EmptyBetaSlotTokenV1:
    """Typed evidence for a beta slot that owns no physical candidate buffer."""

    slot_ordinal: int
    semantic_path: str
    shape: tuple[int, ...]
    source_content_hash: str
    physical_buffer_present: bool = False
    physical_view_present: bool = False
    optimizer_ordinal: int = -1

    def validate(self) -> None:
        if (
            self.slot_ordinal < 0
            or not self.semantic_path
            or len(self.shape) != 2
            or self.shape[0] <= 0
            or self.shape[1] != 0
            or len(self.source_content_hash) != 64
            or self.physical_buffer_present
            or self.physical_view_present
            or self.optimizer_ordinal != -1
        ):
            _reject("BUFFER_PREPARE_MANIFEST_MISMATCH")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            item.name: _canonical_field(object.__getattribute__(self, item.name))
            for item in fields(self)
        }


@dataclass(frozen=True)
class S4MutableBufferDescriptorV1:
    """Pointer-free canonical description of one ordered physical buffer."""

    buffer_ordinal: int
    semantic_role: str
    slot_ordinal_or_minus_one: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int
    dtype: str
    device: str
    element_count: int
    logical_bytes: int
    requires_grad: bool
    is_leaf: bool
    contiguous: bool
    initialized_at_prepare: bool
    initial_content_hash_or_none: str | None
    view_ordinal: int

    def validate(self) -> None:
        if (
            self.buffer_ordinal < 0
            or self.view_ordinal != self.buffer_ordinal
            or not self.semantic_role
            or len(self.shape) != len(self.stride)
            or self.storage_offset != 0
            or self.dtype != "torch.float32"
            or not self.device.startswith("cuda:")
            or self.element_count != _shape_product(self.shape)
            or self.logical_bytes != 4 * self.element_count
            or not self.is_leaf
            or not self.contiguous
            or (
                self.initialized_at_prepare
                != (self.initial_content_hash_or_none is not None)
            )
            or (
                self.initial_content_hash_or_none is not None
                and len(self.initial_content_hash_or_none) != 64
            )
        ):
            _reject("BUFFER_PREPARE_MANIFEST_MISMATCH")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            item.name: _canonical_field(object.__getattribute__(self, item.name))
            for item in fields(self)
        }


@dataclass(frozen=True)
class S4MutableBufferPreparationReceiptV1:
    """Tensor-free canonical S4-1A buffer-preparation receipt."""

    construction_model_hash: str
    admission_hash: str
    snapshot_hash: str
    plan_hash: str
    exact_call_identity_hash: str
    device: str
    dtype: str
    buffer_descriptors: tuple[S4MutableBufferDescriptorV1, ...]
    empty_beta_tokens: tuple[S4EmptyBetaSlotTokenV1, ...]
    parameter_count: int
    gradient_count: int
    empty_beta_token_count: int
    candidate_storage_count: int
    base_dlpack_view_count: int
    parameter_elements: int
    parameter_bytes: int
    gradient_elements: int
    gradient_bytes: int
    candidate_logical_bytes: int
    leased_source_tensor_count: int
    leased_source_elements: int
    leased_source_bytes: int
    source_entry_projection_hash: str
    source_exit_projection_hash: str
    initialized_candidate_projection_hash: str
    private_view_descriptor_hash: str
    source_d2h_copy_count: int
    source_d2h_bytes: int
    initialized_candidate_d2h_copy_count: int
    initialized_candidate_d2h_bytes: int
    s4_1a_d2h_copy_count: int
    s4_1a_d2h_bytes: int
    prior_s4_0_d2h_copy_count: int
    prior_s4_0_d2h_bytes: int
    cumulative_d2h_copy_count: int
    cumulative_d2h_bytes: int
    parameter_d2d_copy_count: int
    parameter_d2d_bytes: int
    warm_dlpack_view_count: int
    full_alpha_device_copy_count: int
    dense_alpha_materialization_count: int
    dense_beta_materialization_count: int
    prepare_retry_count: int
    prepare_fallback_count: int
    empty_cache_call_count: int
    provider_mapping_stability_validated: bool
    process_global_exclusivity_validated: bool
    crown_numeric_semantics_validated: bool
    optimizer_trajectory_validated: bool
    timing_recorded: bool
    performance_claimed: bool
    receipt_hash: str
    schema_version: str = S4_MUTABLE_BUFFER_SCHEMA_V1

    def _payload_without_hash(self) -> dict[str, object]:
        return {
            item.name: (
                [entry.to_dict() for entry in object.__getattribute__(self, item.name)]
                if item.name in {"buffer_descriptors", "empty_beta_tokens"}
                else _canonical_field(object.__getattribute__(self, item.name))
            )
            for item in fields(self)
            if item.name != "receipt_hash"
        }

    def validate(self) -> None:
        for descriptor in self.buffer_descriptors:
            descriptor.validate()
        for token in self.empty_beta_tokens:
            token.validate()
        descriptors = self.buffer_descriptors
        roles = tuple(item.semantic_role for item in descriptors)
        expected_roles = (
            *("alpha_parameter" for _ in range(6)),
            "active_beta_parameter",
            *("alpha_gradient" for _ in range(6)),
            "active_beta_gradient",
            "lower_output",
            "fixed_upstream",
        )
        parameter = descriptors[:7]
        gradient = descriptors[7:14]
        source_bytes = self.leased_source_bytes * 2
        initialized = tuple(item for item in descriptors if item.initialized_at_prepare)
        claim_flags = (
            self.provider_mapping_stability_validated,
            self.process_global_exclusivity_validated,
            self.crown_numeric_semantics_validated,
            self.optimizer_trajectory_validated,
            self.timing_recorded,
            self.performance_claimed,
        )
        if (
            self.schema_version != S4_MUTABLE_BUFFER_SCHEMA_V1
            or self.construction_model_hash != S4_MUTABLE_BUFFER_CONSTRUCTION_HASH_V1
            or self.construction_model_hash != _EXPECTED_CONSTRUCTION_HASH
            or tuple(item.buffer_ordinal for item in descriptors) != tuple(range(16))
            or roles != expected_roles
            or any(
                item.requires_grad != item.semantic_role.endswith("parameter")
                for item in descriptors
            )
            or tuple(item.slot_ordinal_or_minus_one for item in descriptors[:6])
            != tuple(range(6))
            or descriptors[6].slot_ordinal_or_minus_one != 5
            or tuple(item.slot_ordinal_or_minus_one for item in descriptors[7:13])
            != tuple(range(6))
            or descriptors[13].slot_ordinal_or_minus_one != 5
            or self.parameter_count != 7
            or self.gradient_count != 7
            or self.empty_beta_token_count != len(self.empty_beta_tokens)
            or len(self.empty_beta_tokens) != 5
            or tuple(item.slot_ordinal for item in self.empty_beta_tokens)
            != tuple(range(5))
            or self.candidate_storage_count != len(descriptors)
            or len(descriptors) != 16
            or self.base_dlpack_view_count != 16
            or self.parameter_elements != sum(item.element_count for item in parameter)
            or self.parameter_bytes != sum(item.logical_bytes for item in parameter)
            or self.gradient_elements != sum(item.element_count for item in gradient)
            or self.gradient_bytes != sum(item.logical_bytes for item in gradient)
            or self.candidate_logical_bytes
            != sum(item.logical_bytes for item in descriptors)
            or (
                self.parameter_elements,
                self.parameter_bytes,
                self.gradient_elements,
                self.gradient_bytes,
                self.candidate_logical_bytes,
            )
            != (4254, 17016, 4254, 17016, 34080)
            or (
                self.leased_source_tensor_count,
                self.leased_source_elements,
                self.leased_source_bytes,
            )
            != (12, 8502, 34008)
            or any(
                len(value) != 64
                for value in (
                    self.admission_hash,
                    self.snapshot_hash,
                    self.plan_hash,
                    self.exact_call_identity_hash,
                    self.source_entry_projection_hash,
                    self.source_exit_projection_hash,
                    self.initialized_candidate_projection_hash,
                    self.private_view_descriptor_hash,
                    self.receipt_hash,
                )
            )
            or self.source_entry_projection_hash != self.source_exit_projection_hash
            or self.source_d2h_copy_count != 24
            or self.source_d2h_bytes != source_bytes
            or source_bytes != 68016
            or self.initialized_candidate_d2h_copy_count != len(initialized)
            or len(initialized) != 8
            or self.initialized_candidate_d2h_bytes
            != sum(item.logical_bytes for item in initialized)
            or self.initialized_candidate_d2h_bytes != 17040
            or (self.s4_1a_d2h_copy_count, self.s4_1a_d2h_bytes) != (32, 85056)
            or (self.prior_s4_0_d2h_copy_count, self.prior_s4_0_d2h_bytes)
            != (24, 68016)
            or (self.cumulative_d2h_copy_count, self.cumulative_d2h_bytes)
            != (56, 153072)
            or (self.parameter_d2d_copy_count, self.parameter_d2d_bytes) != (7, 17016)
            or any(
                (
                    self.warm_dlpack_view_count,
                    self.full_alpha_device_copy_count,
                    self.dense_alpha_materialization_count,
                    self.dense_beta_materialization_count,
                    self.prepare_retry_count,
                    self.prepare_fallback_count,
                    self.empty_cache_call_count,
                )
            )
            or any(claim_flags)
            or self.dtype != "torch.float32"
            or not self.device.startswith("cuda:")
        ):
            _reject("BUFFER_PREPARE_VALIDATION_COPY_ACCOUNTING_MISMATCH")
        if self.receipt_hash != _canonical_hash(self._payload_without_hash()):
            _reject("BUFFER_PREPARE_MANIFEST_MISMATCH")
        _tensor_free_walk(self)

    def to_dict(self) -> dict[str, object]:
        self.validate()
        payload = self._payload_without_hash()
        payload["receipt_hash"] = self.receipt_hash
        return payload

    def stable_hash(self) -> str:
        self.validate()
        return self.receipt_hash


def _serialization_forbidden(*_args: object, **_kwargs: object) -> NoReturn:
    _reject("BUFFER_PREPARE_SERIALIZATION_FORBIDDEN")


class _S4BufferPrepareTicketV1:
    """S4-1A refinement of the consumed S4-0 adoption ticket."""

    __slots__ = ("receipt", "_adoption", "_state")

    def __init__(self, adoption: object) -> None:
        self.receipt = object.__getattribute__(adoption, "receipt")
        self._adoption: object | None = adoption
        self._state = "OPEN"

    __copy__ = _serialization_forbidden
    __deepcopy__ = _serialization_forbidden
    __getstate__ = _serialization_forbidden
    __reduce__ = _serialization_forbidden
    __reduce_ex__ = _serialization_forbidden

    def lease(self) -> _admission.S4LiveMutableLeaseV1:
        if self._adoption is None or self._state != "OPEN":
            _reject("BUFFER_PREPARE_ADOPTION_OWNER_MISMATCH")
        return object.__getattribute__(self._adoption, "_lease")

    def close(self) -> None:
        if self._adoption is not None:
            object.__getattribute__(self._adoption, "_lease").close()
            self._adoption = None
        self._state = "CLOSED"


class _S4BufferResourceOwnerV1:
    """The sole owner of ticket, tensors, and process-local DLPack views."""

    __slots__ = (
        "_ticket",
        "_parameters",
        "_gradients",
        "_lower",
        "_upstream",
        "_views",
        "_private_view_keys",
        "_initialized",
        "_state",
    )

    def __init__(self, ticket: _S4BufferPrepareTicketV1) -> None:
        self._ticket: _S4BufferPrepareTicketV1 | None = ticket
        self._parameters: list[torch.Tensor] = []
        self._gradients: list[torch.Tensor] = []
        self._lower: torch.Tensor | None = None
        self._upstream: torch.Tensor | None = None
        self._views: list[object] = []
        self._private_view_keys: list[tuple[object, ...]] = []
        self._initialized: list[bool] = []
        self._state = "STAGING"

    __copy__ = _serialization_forbidden
    __deepcopy__ = _serialization_forbidden
    __getstate__ = _serialization_forbidden
    __reduce__ = _serialization_forbidden
    __reduce_ex__ = _serialization_forbidden

    def buffers(self) -> tuple[torch.Tensor, ...]:
        if self._lower is None or self._upstream is None:
            _reject("BUFFER_PREPARE_ADOPTION_OWNER_MISMATCH")
        return (
            *self._parameters,
            *self._gradients,
            self._lower,
            self._upstream,
        )

    def close(self) -> None:
        self._views.clear()
        self._private_view_keys.clear()
        self._upstream = None
        self._lower = None
        self._gradients.clear()
        self._parameters.clear()
        self._initialized.clear()
        ticket = self._ticket
        self._ticket = None
        if ticket is not None:
            ticket.close()
        self._state = "CLOSED"


class PreparedS4MutableBuffersV1:
    """Prepared S4-1A resource owner; only its receipt is public."""

    __slots__ = ("receipt", "_resources", "_state")

    def __init__(
        self,
        receipt: S4MutableBufferPreparationReceiptV1,
        resources: _S4BufferResourceOwnerV1,
    ) -> None:
        self.receipt = receipt
        self._resources: _S4BufferResourceOwnerV1 | None = resources
        self._state = "PREPARED"

    __copy__ = _serialization_forbidden
    __deepcopy__ = _serialization_forbidden
    __getstate__ = _serialization_forbidden
    __reduce__ = _serialization_forbidden
    __reduce_ex__ = _serialization_forbidden

    def close(self) -> None:
        if self._resources is not None:
            self._resources.close()
            self._resources = None
        self._state = "CLOSED"


def _clone_parameter(source: torch.Tensor) -> torch.Tensor:
    return (
        source.detach()
        .clone(memory_format=torch.contiguous_format)
        .requires_grad_(True)
    )


def _empty_buffer(shape: tuple[int, ...], source: torch.Tensor) -> torch.Tensor:
    return torch.empty(shape, dtype=torch.float32, device=source.device)


def _full_upstream(domain_count: int, source: torch.Tensor) -> torch.Tensor:
    return torch.full(
        (domain_count, 1), -1.0, dtype=torch.float32, device=source.device
    )


def _create_dlpack_view(tensor: torch.Tensor) -> object:
    import tvm  # type: ignore[import-untyped]  # pylint: disable=import-outside-toplevel

    return tvm.runtime.from_dlpack(tensor)


def _roundtrip_dlpack(view: object) -> torch.Tensor:
    return torch.from_dlpack(view)


def _storage_token(tensor: torch.Tensor) -> tuple[str, int, int, int]:
    storage = tensor.untyped_storage()
    return (
        str(tensor.device),
        int(storage._cdata),  # pylint: disable=protected-access
        int(storage.data_ptr()),
        int(storage.nbytes()),
    )


def _view_key(ordinal: int, tensor: torch.Tensor) -> tuple[object, ...]:
    if type(tensor) is not torch.Tensor or not tensor.is_contiguous():
        _reject("BASE_DLPACK_VIEW_KEY_MISMATCH")
    storage = tensor.untyped_storage()
    return (
        ordinal,
        int(storage._cdata),  # pylint: disable=protected-access
        int(storage.data_ptr()),
        int(storage.nbytes()),
        int(tensor.data_ptr()),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        int(tensor.storage_offset()),
        str(tensor.dtype),
        str(tensor.device),
    )


def _descriptor(
    ordinal: int,
    role: str,
    slot: int,
    tensor: torch.Tensor,
    *,
    initialized: bool,
    initial_hash: str | None,
) -> S4MutableBufferDescriptorV1:
    return S4MutableBufferDescriptorV1(
        buffer_ordinal=ordinal,
        semantic_role=role,
        slot_ordinal_or_minus_one=slot,
        shape=tuple(tensor.shape),
        stride=tuple(tensor.stride()),
        storage_offset=int(tensor.storage_offset()),
        dtype=str(tensor.dtype),
        device=str(tensor.device),
        element_count=tensor.numel(),
        logical_bytes=tensor.numel() * tensor.element_size(),
        requires_grad=bool(tensor.requires_grad),
        is_leaf=bool(tensor.is_leaf),
        contiguous=bool(tensor.is_contiguous()),
        initialized_at_prepare=initialized,
        initial_content_hash_or_none=initial_hash if initialized else None,
        view_ordinal=ordinal,
    )


def _row_projection(
    rows: tuple[_admission._LiveTensorRow, ...],
) -> list[dict[str, object]]:
    return [
        {
            "semantic_path": row.semantic_path,
            "shape": list(row.shape),
            "stride": list(row.stride),
            "storage_offset": row.storage_offset,
            "dtype": row.dtype,
            "device": row.device,
            "requires_grad": row.requires_grad,
            "is_leaf": row.is_leaf,
            "version": row.version,
            "content_hash": row.content_hash,
        }
        for row in rows
    ]


def _translate_admission_error(error: S4MutableStateAdmissionError) -> str:
    detail = error.detail_code
    if detail in {"EXACT_CALL_IDENTITY_INVALID", "EXACT_CALL_IDENTITY_MISMATCH"}:
        return "BUFFER_PREPARE_EXACT_CALL_MISMATCH"
    if detail in {
        "LIVE_LEASE_ALREADY_TRANSFERRED",
        "LIVE_LEASE_ALREADY_CLOSED",
        "LIVE_LEASE_STATE_MISMATCH",
    }:
        return "BUFFER_PREPARE_ALREADY_ATTEMPTED"
    if detail in {
        "LIVE_SOURCE_OWNER_PROCESS_MISMATCH",
        "LIVE_SOURCE_OWNER_THREAD_MISMATCH",
        "LIVE_SOURCE_STREAM_MISMATCH",
    }:
        return "BUFFER_PREPARE_OWNER_CONTEXT_MISMATCH"
    if detail in {
        "LIVE_SOURCE_CONTAINER_TYPE_MISMATCH",
        "LIVE_LEASE_ADMISSION_MISMATCH",
    }:
        return "BUFFER_PREPARE_MANIFEST_MISMATCH"
    return "BUFFER_PREPARE_SOURCE_IDENTITY_MISMATCH"


def _validate_tensor(
    tensor: torch.Tensor,
    *,
    requires_grad: bool,
    role: str,
) -> None:
    if (
        type(tensor) is not torch.Tensor
        or tensor.dtype != torch.float32
        or tensor.device.type != "cuda"
        or not tensor.is_contiguous()
        or tensor.storage_offset() != 0
        or bool(tensor.requires_grad) != requires_grad
        or not tensor.is_leaf
    ):
        _reject("BUFFER_PREPARE_MANIFEST_MISMATCH", semantic_role=role)


def _build_receipt(
    admission_receipt: _admission.S4MutableStateAdmissionV1,
    descriptors: tuple[S4MutableBufferDescriptorV1, ...],
    tokens: tuple[S4EmptyBetaSlotTokenV1, ...],
    entry_hash: str,
    exit_hash: str,
    initialized_hash: str,
    view_hash: str,
) -> S4MutableBufferPreparationReceiptV1:
    payload = dict(
        construction_model_hash=S4_MUTABLE_BUFFER_CONSTRUCTION_HASH_V1,
        admission_hash=admission_receipt.admission_hash,
        snapshot_hash=admission_receipt.snapshot_hash,
        plan_hash=admission_receipt.production_plan_hash,
        exact_call_identity_hash=admission_receipt.exact_call_identity_hash,
        device=descriptors[0].device,
        dtype="torch.float32",
        buffer_descriptors=descriptors,
        empty_beta_tokens=tokens,
        parameter_count=7,
        gradient_count=7,
        empty_beta_token_count=5,
        candidate_storage_count=16,
        base_dlpack_view_count=16,
        parameter_elements=4254,
        parameter_bytes=17016,
        gradient_elements=4254,
        gradient_bytes=17016,
        candidate_logical_bytes=34080,
        leased_source_tensor_count=admission_receipt.live_tensor_count,
        leased_source_elements=admission_receipt.live_element_count_per_pass,
        leased_source_bytes=admission_receipt.live_bytes_per_pass,
        source_entry_projection_hash=entry_hash,
        source_exit_projection_hash=exit_hash,
        initialized_candidate_projection_hash=initialized_hash,
        private_view_descriptor_hash=view_hash,
        source_d2h_copy_count=24,
        source_d2h_bytes=68016,
        initialized_candidate_d2h_copy_count=8,
        initialized_candidate_d2h_bytes=17040,
        s4_1a_d2h_copy_count=32,
        s4_1a_d2h_bytes=85056,
        prior_s4_0_d2h_copy_count=24,
        prior_s4_0_d2h_bytes=68016,
        cumulative_d2h_copy_count=56,
        cumulative_d2h_bytes=153072,
        parameter_d2d_copy_count=7,
        parameter_d2d_bytes=17016,
        warm_dlpack_view_count=0,
        full_alpha_device_copy_count=0,
        dense_alpha_materialization_count=0,
        dense_beta_materialization_count=0,
        prepare_retry_count=0,
        prepare_fallback_count=0,
        empty_cache_call_count=0,
        provider_mapping_stability_validated=False,
        process_global_exclusivity_validated=False,
        crown_numeric_semantics_validated=False,
        optimizer_trajectory_validated=False,
        timing_recorded=False,
        performance_claimed=False,
        receipt_hash="",
    )
    draft = S4MutableBufferPreparationReceiptV1(**payload)  # type: ignore[arg-type]
    payload["receipt_hash"] = _canonical_hash(draft._payload_without_hash())
    return S4MutableBufferPreparationReceiptV1(**payload)  # type: ignore[arg-type]


def _adopt_prepared(
    receipt: S4MutableBufferPreparationReceiptV1,
    owner: _S4BufferResourceOwnerV1,
) -> PreparedS4MutableBuffersV1:
    prepared = PreparedS4MutableBuffersV1(receipt, owner)
    owner._state = "PREPARED"
    return prepared


def prepare_s4_mutable_buffers_v1(
    prepared_admission: PreparedS4MutableStateAdmissionV1,
    current_live_sources: dict[str, torch.Tensor],
    *,
    exact_call_id: str,
) -> PreparedS4MutableBuffersV1:
    """Consume one S4-0 lease and prepare the fixed ordered buffer ABI."""

    if type(prepared_admission) is not PreparedS4MutableStateAdmissionV1:
        _reject("BUFFER_PREPARE_MANIFEST_MISMATCH")

    adoption = None
    ticket = None
    owner = None
    failure_detail: str | None = None
    failure_role: str | None = None
    failure_phase: str | None = None
    parameter = None
    gradient = None
    candidate = None
    tensor = None
    output = None
    view = None
    roundtrip = None
    buffers: tuple[torch.Tensor, ...] = ()
    entry_rows: tuple[_admission._LiveTensorRow, ...] = ()
    exit_rows: tuple[_admission._LiveTensorRow, ...] = ()
    descriptors: tuple[S4MutableBufferDescriptorV1, ...] = ()
    tokens: tuple[S4EmptyBetaSlotTokenV1, ...] = ()
    try:
        try:
            adoption = prepared_admission.begin_buffer_prepare(
                current_live_sources, exact_call_id=exact_call_id
            )
        except S4MutableStateAdmissionError as error:
            failure_detail = _translate_admission_error(error)
        if failure_detail is not None:
            raise RuntimeError("translated-admission-failure")
        assert adoption is not None

        ticket = _S4BufferPrepareTicketV1(adoption)
        adoption = None
        owner = _S4BufferResourceOwnerV1(ticket)
        ticket = None
        assert owner._ticket is not None
        admission_receipt = owner._ticket.receipt
        admission_receipt.validate()
        if (
            S4_MUTABLE_BUFFER_CONSTRUCTION_HASH_V1 != _EXPECTED_CONSTRUCTION_HASH
            or type(current_live_sources) is not dict
        ):
            _reject("BUFFER_PREPARE_MANIFEST_MISMATCH")
        lease = owner._ticket.lease()
        entry_rows = lease._source_rows
        if len(entry_rows) != 12 or len(admission_receipt.slots) != 6:
            _reject("BUFFER_PREPARE_MANIFEST_MISMATCH")

        source_by_path = {row.semantic_path: row.tensor for row in entry_rows}
        active_sources: list[tuple[int, str, torch.Tensor, str]] = []
        empty_tokens: list[S4EmptyBetaSlotTokenV1] = []
        for slot in admission_receipt.slots:
            alpha_source = source_by_path[slot.alpha_semantic_path]
            active_sources.append(
                (
                    slot.slot_ordinal,
                    "alpha_parameter",
                    alpha_source[0, 0],
                    slot.alpha_active_hash,
                )
            )
            beta_source = source_by_path[slot.beta_semantic_path]
            if slot.beta_active:
                active_sources.append(
                    (
                        slot.slot_ordinal,
                        "active_beta_parameter",
                        beta_source,
                        slot.beta_live_content_hash,
                    )
                )
            else:
                empty_tokens.append(
                    S4EmptyBetaSlotTokenV1(
                        slot_ordinal=slot.slot_ordinal,
                        semantic_path=slot.beta_semantic_path,
                        shape=tuple(beta_source.shape),
                        source_content_hash=slot.beta_live_content_hash,
                    )
                )
        if (
            tuple((slot, role) for slot, role, _, _ in active_sources)
            != (
                (0, "alpha_parameter"),
                (1, "alpha_parameter"),
                (2, "alpha_parameter"),
                (3, "alpha_parameter"),
                (4, "alpha_parameter"),
                (5, "alpha_parameter"),
                (5, "active_beta_parameter"),
            )
            or len(empty_tokens) != 5
        ):
            _reject("BUFFER_PREPARE_MANIFEST_MISMATCH")
        tokens = tuple(empty_tokens)
        active_source_rows = active_sources

        for _slot_ordinal, role, source, _ in active_source_rows:
            failure_role = role
            failure_phase = "parameter_allocation"
            parameter = _clone_parameter(source)
            owner._parameters.append(parameter)
            parameter = None
        for _slot_ordinal, role, source, _ in active_source_rows:
            failure_role = role.replace("parameter", "gradient")
            failure_phase = "gradient_allocation"
            gradient = _empty_buffer(tuple(source.shape), source)
            owner._gradients.append(gradient)
            gradient = None
        domain_count = admission_receipt.slots[0].alpha_active_shape[0]
        first_source = active_source_rows[0][2]
        failure_role = "lower_output"
        failure_phase = "output_allocation"
        output = _empty_buffer((domain_count,), first_source)
        owner._lower = output
        output = None
        failure_role = "fixed_upstream"
        failure_phase = "output_allocation"
        output = _full_upstream(domain_count, first_source)
        owner._upstream = output
        output = None
        owner._initialized = [True] * 7 + [False] * 8 + [True]

        buffers = owner.buffers()
        roles = (
            *("alpha_parameter" for _ in range(6)),
            "active_beta_parameter",
            *("alpha_gradient" for _ in range(6)),
            "active_beta_gradient",
            "lower_output",
            "fixed_upstream",
        )
        slots = (*range(6), 5, *range(6), 5, -1, -1)
        for tensor, role, initialized in zip(buffers, roles, owner._initialized):
            _validate_tensor(
                tensor,
                requires_grad=role.endswith("parameter"),
                role=role,
            )
        candidate_tokens = [_storage_token(tensor) for tensor in buffers]
        if len(set(candidate_tokens)) != 16:
            _reject("CANDIDATE_STORAGE_ALIAS")
        parameter_tokens = set(candidate_tokens[:7])
        gradient_tokens = set(candidate_tokens[7:14])
        if parameter_tokens & gradient_tokens:
            _reject("PARAMETER_GRADIENT_STORAGE_ALIAS")
        source_tokens = {
            row.storage_token for row in entry_rows if _shape_product(row.shape) > 0
        }
        if set(candidate_tokens) & source_tokens:
            _reject("PARAMETER_SOURCE_STORAGE_ALIAS")
        initialized_hashes: list[str | None] = []
        for candidate, (_, _, _, expected_hash) in zip(
            owner._parameters, active_source_rows
        ):
            candidate_hash = production_tensor_sha256(candidate)
            initialized_hashes.append(candidate_hash)
            if candidate_hash != expected_hash:
                _reject("BUFFER_INITIAL_CONTENT_MISMATCH")
        candidate = None
        initialized_hashes.extend([None] * 8)
        upstream_hash = production_tensor_sha256(owner._upstream)
        expected_upstream_hash = production_tensor_sha256(
            torch.full(tuple(owner._upstream.shape), -1.0, dtype=torch.float32)
        )
        initialized_hashes.append(upstream_hash)
        if upstream_hash != expected_upstream_hash:
            _reject("BUFFER_INITIAL_CONTENT_MISMATCH")

        for ordinal, tensor in enumerate(buffers):
            failure_role = roles[ordinal]
            failure_phase = "view_creation"
            key = _view_key(ordinal, tensor)
            view = _create_dlpack_view(tensor)
            failure_phase = "roundtrip"
            roundtrip = _roundtrip_dlpack(view)
            if (
                type(roundtrip) is not torch.Tensor
                or roundtrip.data_ptr() != tensor.data_ptr()
                or tuple(roundtrip.shape) != tuple(tensor.shape)
                or tuple(roundtrip.stride()) != tuple(tensor.stride())
                or roundtrip.storage_offset() != tensor.storage_offset()
                or roundtrip.dtype != tensor.dtype
                or roundtrip.device != tensor.device
            ):
                _reject("BASE_DLPACK_VIEW_KEY_MISMATCH")
            owner._private_view_keys.append(key)
            owner._views.append(view)
            roundtrip = None
            view = None
        tensor = None
        if len(owner._views) != 16 or len(owner._private_view_keys) != 16:
            _reject("BASE_DLPACK_VIEW_COUNT_MISMATCH")

        paths = tuple(row.semantic_path for row in entry_rows)
        failure_phase = "source_exit"
        try:
            exit_rows = _admission._capture_live_rows(
                current_live_sources, paths, check_aliases=False
            )
            _admission._rows_equal(entry_rows, exit_rows, read_race=True)
        except S4MutableStateAdmissionError:
            _reject("BUFFER_PREPARE_SOURCE_READ_RACE")

        descriptors = tuple(
            _descriptor(
                ordinal,
                role,
                slot,
                tensor,
                initialized=initialized,
                initial_hash=initialized_hashes[ordinal],
            )
            for ordinal, (role, slot, tensor, initialized) in enumerate(
                zip(roles, slots, buffers, owner._initialized)
            )
        )
        entry_hash = _canonical_hash(_row_projection(entry_rows))
        exit_hash = _canonical_hash(_row_projection(exit_rows))
        initialized_hash = _canonical_hash(
            [
                {
                    "buffer_ordinal": item.buffer_ordinal,
                    "content_hash": item.initial_content_hash_or_none,
                }
                for item in descriptors
                if item.initialized_at_prepare
            ]
        )
        view_hash = _canonical_hash([item.to_dict() for item in descriptors])
        failure_phase = "receipt"
        receipt = _build_receipt(
            admission_receipt,
            descriptors,
            tokens,
            entry_hash,
            exit_hash,
            initialized_hash,
            view_hash,
        )
        receipt.validate()
        failure_phase = "adoption"
        prepared = _adopt_prepared(receipt, owner)
        if prepared._resources is not owner or owner._state != "PREPARED":
            _reject("BUFFER_PREPARE_ADOPTION_OWNER_MISMATCH")
        return prepared
    except BaseException as error:
        if failure_detail is None:
            if isinstance(error, S4MutableBufferPreparationError):
                failure_detail = error.detail_code
                failure_role = error.semantic_role or failure_role
            elif failure_phase == "view_creation":
                failure_detail = "BASE_DLPACK_VIEW_COUNT_MISMATCH"
            elif failure_phase == "roundtrip":
                failure_detail = "BASE_DLPACK_VIEW_KEY_MISMATCH"
            elif failure_phase == "receipt":
                failure_detail = "BUFFER_PREPARE_VALIDATION_COPY_ACCOUNTING_MISMATCH"
            elif failure_phase == "adoption":
                failure_detail = "BUFFER_PREPARE_ADOPTION_OWNER_MISMATCH"
            elif failure_role in {"alpha_parameter", "active_beta_parameter"}:
                failure_detail = "BUFFER_PREPARE_MANIFEST_MISMATCH"
            elif failure_role in {"alpha_gradient", "active_beta_gradient"}:
                failure_detail = "BUFFER_PREPARE_MANIFEST_MISMATCH"
            elif failure_role in {"lower_output", "fixed_upstream"}:
                failure_detail = "BUFFER_PREPARE_RESOURCE_CONTEXT_RETAINED"
            elif failure_role is not None and len(getattr(owner, "_views", ())) < 16:
                failure_detail = "BASE_DLPACK_VIEW_COUNT_MISMATCH"
            else:
                failure_detail = "BUFFER_PREPARE_MANIFEST_MISMATCH"
        if owner is not None:
            owner.close()
        elif ticket is not None:
            ticket.close()
        elif adoption is not None:
            adoption._lease.close()
        parameter = None
        gradient = None
        candidate = None
        tensor = None
        output = None
        view = None
        roundtrip = None
        buffers = ()
        entry_rows = ()
        exit_rows = ()
        descriptors = ()
        tokens = ()
        adoption = None
        ticket = None
        owner = None
    # This error is intentionally constructed after leaving ``except`` so it
    # has no implicit lower exception context or retained candidate frame.
    raise S4MutableBufferPreparationError(
        failure_detail or "BUFFER_PREPARE_MANIFEST_MISMATCH",
        semantic_role=failure_role,
    )


__all__ = [
    "PreparedS4MutableBuffersV1",
    "S4EmptyBetaSlotTokenV1",
    "S4MutableBufferDescriptorV1",
    "S4MutableBufferPreparationError",
    "S4MutableBufferPreparationReceiptV1",
    "S4_MUTABLE_BUFFER_CONSTRUCTION_HASH_V1",
    "S4_MUTABLE_BUFFER_CONSTRUCTION_MODEL_V1",
    "S4_MUTABLE_BUFFER_SCHEMA_V1",
    "prepare_s4_mutable_buffers_v1",
]
