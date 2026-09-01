"""S4-0 production mutable-state admission and ephemeral live ownership.

This module deliberately stops before buffer packing, TVM compilation, solver
execution, optimizer mutation, timing, and commit.  Its public receipt is
canonical and tensor-free; live Tensor ownership stays in a non-serializable
process-local lease.
"""

# pylint: disable=too-many-lines,too-many-instance-attributes,too-many-arguments
# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=too-many-boolean-expressions,missing-function-docstring
# Exact built-in container/Tensor checks are a security and ownership boundary.
# pylint: disable=unidiomatic-typecheck,too-few-public-methods,protected-access
# pylint: disable=line-too-long

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, fields, is_dataclass, replace
import hashlib
import json
import os
import re
import threading
from typing import Callable, NoReturn

import torch

from boundflow.ir.verification_graph import VerificationRejectionReason
from boundflow.runtime.r3_structured_owner_custom_backward import (
    R31_PLAN_SCHEMA,
    R31FullRegionPlanV1,
    R31ReluLayoutV1,
)
from boundflow.runtime.rvir_v4_pre_state_initializer import ProductionReluTopologyV4
from boundflow.runtime.rvir_v4_production_state import (
    OwnedProductionTensorV4,
    ProductionStateSnapshotV4,
    ProductionTensorOwnership,
    ProductionTensorRole,
    RVIR_V4_STATE_SCHEMA_VERSION,
    production_tensor_sha256,
)

S4_MUTABLE_STATE_ADMISSION_SCHEMA_V1 = (
    "boundflow.asplos27-s4-mutable-state-admission/v1"
)
S4_MUTABLE_STATE_CONSTRUCTION_HASH_V4 = (
    "471424594fb4b6d017feac936a6005eb9d0451fd5579d026204ec952d0995239"
)
_EXACT_CALL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@+\-]{0,255}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


_REASON_BY_DETAIL: dict[str, VerificationRejectionReason] = {
    "EXACT_CALL_IDENTITY_INVALID": VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
    "EXACT_CALL_IDENTITY_MISMATCH": VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
    "LIVE_SOURCE_CONTAINER_TYPE_MISMATCH": VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
    "LIVE_SOURCE_NESTED_CONTAINER_TYPE_MISMATCH": VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
    "LIVE_SOURCE_TENSOR_SUBCLASS_UNSUPPORTED": VerificationRejectionReason.DTYPE_OR_DEVICE_MISMATCH,
    "LIVE_SOURCE_OWNER_PROCESS_MISMATCH": VerificationRejectionReason.UNSAFE_ALIAS_OR_LIFETIME,
    "LIVE_SOURCE_OWNER_THREAD_MISMATCH": VerificationRejectionReason.UNSAFE_ALIAS_OR_LIFETIME,
    "LIVE_SOURCE_STREAM_MISMATCH": VerificationRejectionReason.UNSAFE_ALIAS_OR_LIFETIME,
    "LIVE_SOURCE_READ_RACE": VerificationRejectionReason.STATE_VERSION_MISMATCH,
    "SNAPSHOT_SCHEMA_VERSION_MISMATCH": VerificationRejectionReason.STATE_VERSION_MISMATCH,
    "SNAPSHOT_SEMANTIC_INVALID": VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
    "PLAN_SCHEMA_VERSION_MISMATCH": VerificationRejectionReason.STATE_VERSION_MISMATCH,
    "PLAN_SEMANTIC_INVALID": VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
    "OPTIMIZER_POLICY_MISMATCH": VerificationRejectionReason.BOUND_POLARITY_MISMATCH,
    "TOPOLOGY_IDENTITY_MISMATCH": VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
    "MUTABLE_STATE_COVERAGE_INCOMPLETE": VerificationRejectionReason.VJP_OWNER_OR_SAVED_STATE_MISMATCH,
    "ACTIVE_BETA_COVERAGE_INCOMPLETE": VerificationRejectionReason.BETA_ACTIVE_EMPTY_MISMATCH,
    "ALPHA_MUTABLE_DIRECTION_MISMATCH": VerificationRejectionReason.ALPHA_INDEX_OR_DIRECTION_MISMATCH,
    "ALPHA_PRESERVED_DIRECTION_DRIFT": VerificationRejectionReason.ALPHA_INDEX_OR_DIRECTION_MISMATCH,
    "ALPHA_LAYOUT_IDENTITY_MISMATCH": VerificationRejectionReason.ALPHA_INDEX_OR_DIRECTION_MISMATCH,
    "BETA_LOCATION_SIGN_HISTORY_MISMATCH": VerificationRejectionReason.BETA_LOCATION_SIGN_HISTORY_MISMATCH,
    "BETA_HISTORY_WIDTH_MISMATCH": VerificationRejectionReason.BETA_LOCATION_SIGN_HISTORY_MISMATCH,
    "LIVE_SOURCE_COVERAGE_MISMATCH": VerificationRejectionReason.VJP_OWNER_OR_SAVED_STATE_MISMATCH,
    "LIVE_SOURCE_OBJECT_ALIAS_CONFLICT": VerificationRejectionReason.UNSAFE_ALIAS_OR_LIFETIME,
    "LIVE_SOURCE_STORAGE_ALIAS_CONFLICT": VerificationRejectionReason.UNSAFE_ALIAS_OR_LIFETIME,
    "LIVE_SOURCE_OBJECT_REPLACED": VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
    "LIVE_SOURCE_STORAGE_REPLACED": VerificationRejectionReason.UNSAFE_ALIAS_OR_LIFETIME,
    "LIVE_SOURCE_SHAPE_DTYPE_DEVICE_MISMATCH": VerificationRejectionReason.DTYPE_OR_DEVICE_MISMATCH,
    "LIVE_SOURCE_STRIDE_OFFSET_MISMATCH": VerificationRejectionReason.LAYOUT_NOT_NORMALIZABLE,
    "LIVE_TENSOR_VERSION_MISMATCH": VerificationRejectionReason.STATE_VERSION_MISMATCH,
    "LIVE_SOURCE_CONTENT_MISMATCH": VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
    "LIVE_SOURCE_READINESS_MISMATCH": VerificationRejectionReason.DTYPE_OR_DEVICE_MISMATCH,
    "NONFINITE_MUTABLE_STATE": VerificationRejectionReason.DTYPE_OR_DEVICE_MISMATCH,
    "PLAN_SNAPSHOT_PROJECTION_MISMATCH": VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
    "MUTABLE_SLOT_ORDER_MISMATCH": VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
    "RECEIPT_IDENTITY_MISMATCH": VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
    "RECEIPT_LIVE_COPY_ACCOUNTING_MISMATCH": VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
    "CLAIM_FLAG_TRUE_BEFORE_FORMAL": VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
    "LIVE_LEASE_ADMISSION_MISMATCH": VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
    "LIVE_LEASE_ALREADY_TRANSFERRED": VerificationRejectionReason.UNSAFE_ALIAS_OR_LIFETIME,
    "LIVE_LEASE_ALREADY_CLOSED": VerificationRejectionReason.UNSAFE_ALIAS_OR_LIFETIME,
    "LIVE_LEASE_STATE_MISMATCH": VerificationRejectionReason.UNSAFE_ALIAS_OR_LIFETIME,
    "LIVE_LEASE_SERIALIZATION_FORBIDDEN": VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
    "S4_0_CROSS_QUERY_EXCLUSIVITY_UNPROVEN": VerificationRejectionReason.QUEUE_OR_TERMINATION_EFFECT_CROSSED,
}


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _encode_component(value: str) -> str:
    return value.replace("%", "%25").replace("/", "%2F")


class S4MutableStateAdmissionError(RuntimeError):
    """Stable fail-closed error at the S4-0 runtime binding boundary."""

    def __init__(
        self,
        detail_code: str,
        *,
        verification_reason: VerificationRejectionReason | None = None,
        slot_ordinal: int | None = None,
        semantic_path: str | None = None,
    ) -> None:
        reason = verification_reason or _REASON_BY_DETAIL.get(detail_code)
        if reason is None:
            reason = VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH
        self.detail_code = detail_code
        self.verification_reason = reason
        self.slot_ordinal = slot_ordinal
        self.semantic_path = semantic_path
        suffix = "" if semantic_path is None else f":{semantic_path}"
        super().__init__(f"{detail_code}{suffix}")


def _reject(
    detail_code: str,
    *,
    slot_ordinal: int | None = None,
    semantic_path: str | None = None,
    cause: BaseException | None = None,
) -> NoReturn:
    error = S4MutableStateAdmissionError(
        detail_code,
        slot_ordinal=slot_ordinal,
        semantic_path=semantic_path,
    )
    if cause is None:
        raise error
    raise error from cause


def _validate_exact_call_id(exact_call_id: object) -> tuple[str, str]:
    if (
        not isinstance(exact_call_id, str)
        or _EXACT_CALL_ID.fullmatch(exact_call_id) is None
    ):
        _reject("EXACT_CALL_IDENTITY_INVALID")
    return exact_call_id, _canonical_hash({"exact_call_id": exact_call_id})


def _snapshot_hash(snapshot: ProductionStateSnapshotV4) -> str:
    return _canonical_hash(
        {
            "schema_version": snapshot.schema_version,
            "snapshot_id": snapshot.snapshot_id,
            "tensors": [
                item.metadata()
                for item in sorted(
                    snapshot.tensors, key=lambda value: value.semantic_path
                )
            ],
            "history": [
                item.to_dict()
                for item in sorted(
                    snapshot.history,
                    key=lambda value: (value.domain_ordinal, value.layer_name),
                )
            ],
            "optimizer_policy": snapshot.optimizer_policy.to_dict(),
        }
    )


def _tensor_free_walk(value: object, *, _ancestors: set[int] | None = None) -> None:
    if isinstance(value, torch.Tensor):
        _reject("RECEIPT_IDENTITY_MISMATCH")
    if value is None or isinstance(
        value, (str, int, float, bool, VerificationRejectionReason)
    ):
        return
    ancestors = _ancestors if _ancestors is not None else set()
    identity = id(value)
    if identity in ancestors:
        _reject("RECEIPT_IDENTITY_MISMATCH")
    ancestors.add(identity)
    try:
        if isinstance(value, tuple):
            for item in value:
                _tensor_free_walk(item, _ancestors=ancestors)
            return
        if isinstance(value, list):
            for item in value:
                _tensor_free_walk(item, _ancestors=ancestors)
            return
        if type(value) is dict:
            for key, item in value.items():
                if not isinstance(key, str):
                    _reject("RECEIPT_IDENTITY_MISMATCH")
                _tensor_free_walk(item, _ancestors=ancestors)
            return
        if is_dataclass(value) and getattr(type(value), "__dataclass_params__").frozen:
            for item in fields(value):
                _tensor_free_walk(
                    object.__getattribute__(value, item.name),
                    _ancestors=ancestors,
                )
            return
        _reject("RECEIPT_IDENTITY_MISMATCH")
    finally:
        ancestors.remove(identity)


@dataclass(frozen=True)
class S4MutableSlotV1:
    """Canonical projection for one production ReLU's α and β owner."""

    slot_ordinal: int
    native_preactivation: str
    provider_activation: str
    provider_preactivation: str
    provider_start_node: str
    alpha_semantic_path: str
    alpha_source_axes: tuple[str, ...]
    alpha_source_shape: tuple[int, ...]
    alpha_source_dtype: str
    alpha_source_device: str
    alpha_source_hash: str
    alpha_live_object_group: str
    alpha_live_storage_group: str
    alpha_live_version: int
    alpha_live_stride: tuple[int, ...]
    alpha_live_storage_offset: int
    alpha_live_contiguous: bool
    alpha_live_requires_grad: bool
    alpha_live_is_leaf: bool
    alpha_live_content_hash: str
    alpha_active_shape: tuple[int, ...]
    alpha_active_hash: str
    alpha_active_element_count: int
    alpha_preserved_shape: tuple[int, ...]
    alpha_preserved_hash: str
    alpha_preserved_element_count: int
    feature_shape: tuple[int, ...]
    alpha_flat_indices: tuple[int, ...]
    alpha_layout_hash: str
    beta_semantic_path: str
    beta_source_axes: tuple[str, ...]
    beta_source_shape: tuple[int, ...]
    beta_source_dtype: str
    beta_source_device: str
    beta_source_hash: str
    beta_live_object_group: str
    beta_live_storage_group: str
    beta_live_version: int
    beta_live_stride: tuple[int, ...]
    beta_live_storage_offset: int
    beta_live_contiguous: bool
    beta_live_requires_grad: bool
    beta_live_is_leaf: bool
    beta_live_content_hash: str
    beta_location_hash: str
    beta_sign_hash: str
    beta_history_hash: str
    beta_active: bool
    beta_element_count: int
    entry_content_capture_ordinal: int = 1
    exit_content_capture_ordinal: int = 2

    def validate(self) -> None:
        hashes = (
            self.alpha_source_hash,
            self.alpha_live_content_hash,
            self.alpha_active_hash,
            self.alpha_preserved_hash,
            self.alpha_layout_hash,
            self.beta_source_hash,
            self.beta_live_content_hash,
            self.beta_location_hash,
            self.beta_sign_hash,
            self.beta_history_hash,
        )
        if (
            self.slot_ordinal < 0
            or any(
                not value
                for value in (
                    self.native_preactivation,
                    self.provider_activation,
                    self.provider_preactivation,
                    self.provider_start_node,
                    self.alpha_semantic_path,
                    self.beta_semantic_path,
                    self.alpha_live_object_group,
                    self.alpha_live_storage_group,
                    self.beta_live_object_group,
                    self.beta_live_storage_group,
                )
            )
            or not all(_is_sha256(value) for value in hashes)
            or len(self.alpha_source_axes) != len(self.alpha_source_shape)
            or len(self.beta_source_axes) != len(self.beta_source_shape)
            or self.alpha_source_shape[:3] != (2, 1, self.alpha_active_shape[0])
            or self.alpha_active_shape != self.alpha_preserved_shape
            or self.alpha_active_element_count
            != _shape_product(self.alpha_active_shape)
            or self.alpha_preserved_element_count
            != _shape_product(self.alpha_preserved_shape)
            or self.beta_element_count != _shape_product(self.beta_source_shape)
            or self.beta_active != (self.beta_element_count > 0)
            or self.entry_content_capture_ordinal != 1
            or self.exit_content_capture_ordinal != 2
        ):
            _reject(
                "RECEIPT_IDENTITY_MISMATCH",
                slot_ordinal=self.slot_ordinal,
                semantic_path=self.alpha_semantic_path,
            )

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            item.name: _canonical_field(object.__getattribute__(self, item.name))
            for item in fields(self)
        }


def _canonical_field(value: object) -> object:
    if isinstance(value, tuple):
        return [_canonical_field(item) for item in value]
    if isinstance(value, VerificationRejectionReason):
        return value.value
    return value


def _shape_product(shape: tuple[int, ...]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


@dataclass(frozen=True)
class S4MutableStateAdmissionV1:
    """Tensor-free, independently replayable S4-0 admission receipt."""

    snapshot_hash: str
    production_plan_hash: str
    plan_binding_projection_hash: str
    oracle_mapping_provenance_hash: str
    topology_hash: str
    optimizer_policy_hash: str
    exact_call_identity_hash: str
    slots: tuple[S4MutableSlotV1, ...]
    mutable_path_set_hash: str
    alpha_source_count: int
    alpha_stored_element_count: int
    alpha_active_element_count: int
    alpha_preserved_element_count: int
    beta_slot_count: int
    active_beta_slot_count: int
    active_beta_element_count: int
    live_tensor_count: int
    live_element_count_per_pass: int
    live_bytes_per_pass: int
    live_content_capture_pass_count: int
    device_to_host_validation_copy_count: int
    device_to_host_validation_bytes: int
    candidate_kernel_launch_count: int
    candidate_cuda_allocation_count: int
    dense_materialization_observed: bool
    timing_recorded: bool
    performance_claimed: bool
    process_global_query_exclusivity_validated: bool
    construction_model_hash: str
    admission_hash: str
    schema_version: str = S4_MUTABLE_STATE_ADMISSION_SCHEMA_V1

    def _payload_without_hash(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "snapshot_hash": self.snapshot_hash,
            "production_plan_hash": self.production_plan_hash,
            "plan_binding_projection_hash": self.plan_binding_projection_hash,
            "oracle_mapping_provenance_hash": self.oracle_mapping_provenance_hash,
            "topology_hash": self.topology_hash,
            "optimizer_policy_hash": self.optimizer_policy_hash,
            "exact_call_identity_hash": self.exact_call_identity_hash,
            "slots": [slot.to_dict() for slot in self.slots],
            "mutable_path_set_hash": self.mutable_path_set_hash,
            "alpha_source_count": self.alpha_source_count,
            "alpha_stored_element_count": self.alpha_stored_element_count,
            "alpha_active_element_count": self.alpha_active_element_count,
            "alpha_preserved_element_count": self.alpha_preserved_element_count,
            "beta_slot_count": self.beta_slot_count,
            "active_beta_slot_count": self.active_beta_slot_count,
            "active_beta_element_count": self.active_beta_element_count,
            "live_tensor_count": self.live_tensor_count,
            "live_element_count_per_pass": self.live_element_count_per_pass,
            "live_bytes_per_pass": self.live_bytes_per_pass,
            "live_content_capture_pass_count": self.live_content_capture_pass_count,
            "device_to_host_validation_copy_count": self.device_to_host_validation_copy_count,
            "device_to_host_validation_bytes": self.device_to_host_validation_bytes,
            "candidate_kernel_launch_count": self.candidate_kernel_launch_count,
            "candidate_cuda_allocation_count": self.candidate_cuda_allocation_count,
            "dense_materialization_observed": self.dense_materialization_observed,
            "timing_recorded": self.timing_recorded,
            "performance_claimed": self.performance_claimed,
            "process_global_query_exclusivity_validated": self.process_global_query_exclusivity_validated,
            "construction_model_hash": self.construction_model_hash,
        }

    def validate(self) -> None:
        for slot in self.slots:
            slot.validate()
        if tuple(slot.slot_ordinal for slot in self.slots) != tuple(
            range(len(self.slots))
        ):
            _reject("MUTABLE_SLOT_ORDER_MISMATCH")
        paths = tuple(
            path
            for slot in self.slots
            for path in (slot.alpha_semantic_path, slot.beta_semantic_path)
        )
        stored = sum(_shape_product(slot.alpha_source_shape) for slot in self.slots)
        active = sum(slot.alpha_active_element_count for slot in self.slots)
        preserved = sum(slot.alpha_preserved_element_count for slot in self.slots)
        active_beta = sum(slot.beta_active for slot in self.slots)
        beta_elements = sum(slot.beta_element_count for slot in self.slots)
        live_elements = stored + beta_elements
        live_bytes = sum(
            _dtype_bytes(slot.alpha_source_dtype)
            * _shape_product(slot.alpha_source_shape)
            + _dtype_bytes(slot.beta_source_dtype) * slot.beta_element_count
            for slot in self.slots
        )
        if (
            self.schema_version != S4_MUTABLE_STATE_ADMISSION_SCHEMA_V1
            or not all(
                _is_sha256(value)
                for value in (
                    self.snapshot_hash,
                    self.production_plan_hash,
                    self.plan_binding_projection_hash,
                    self.oracle_mapping_provenance_hash,
                    self.topology_hash,
                    self.optimizer_policy_hash,
                    self.exact_call_identity_hash,
                    self.mutable_path_set_hash,
                    self.construction_model_hash,
                    self.admission_hash,
                )
            )
            or self.construction_model_hash != S4_MUTABLE_STATE_CONSTRUCTION_HASH_V4
            or len(set(paths)) != len(paths)
            or self.mutable_path_set_hash != _canonical_hash(sorted(paths))
            or self.alpha_source_count != len(self.slots)
            or self.alpha_stored_element_count != stored
            or self.alpha_active_element_count != active
            or self.alpha_preserved_element_count != preserved
            or self.beta_slot_count != len(self.slots)
            or self.active_beta_slot_count != active_beta
            or self.active_beta_element_count != beta_elements
            or self.live_tensor_count != len(paths)
            or self.live_element_count_per_pass != live_elements
            or self.live_bytes_per_pass != live_bytes
            or self.live_content_capture_pass_count != 2
            or self.device_to_host_validation_copy_count != 2 * len(paths)
            or self.device_to_host_validation_bytes != 2 * live_bytes
        ):
            _reject("RECEIPT_LIVE_COPY_ACCOUNTING_MISMATCH")
        if self.process_global_query_exclusivity_validated:
            _reject("S4_0_CROSS_QUERY_EXCLUSIVITY_UNPROVEN")
        if (
            self.candidate_kernel_launch_count != 0
            or self.candidate_cuda_allocation_count != 0
            or self.dense_materialization_observed
            or self.timing_recorded
            or self.performance_claimed
        ):
            _reject("CLAIM_FLAG_TRUE_BEFORE_FORMAL")
        if self.admission_hash != _canonical_hash(self._payload_without_hash()):
            _reject("RECEIPT_IDENTITY_MISMATCH")
        _tensor_free_walk(self)

    def to_dict(self) -> dict[str, object]:
        self.validate()
        payload = self._payload_without_hash()
        payload["admission_hash"] = self.admission_hash
        return payload

    def stable_hash(self) -> str:
        self.validate()
        return self.admission_hash


def _dtype_bytes(dtype: str) -> int:
    widths = {
        "torch.float16": 2,
        "torch.bfloat16": 2,
        "torch.float32": 4,
        "torch.float64": 8,
        "torch.int32": 4,
        "torch.int64": 8,
        "torch.bool": 1,
    }
    if dtype not in widths:
        _reject("RECEIPT_IDENTITY_MISMATCH")
    return widths[dtype]


@dataclass(frozen=True)
class _LiveTensorRow:
    semantic_path: str
    tensor: torch.Tensor
    object_identity: int
    storage_token: tuple[str, int, int, int]
    shape: tuple[int, ...]
    dtype: str
    device: str
    stride: tuple[int, ...]
    storage_offset: int
    contiguous: bool
    requires_grad: bool
    is_leaf: bool
    version: int
    content_hash: str


def _raw_storage_token(value: torch.Tensor) -> tuple[str, int, int, int]:
    try:
        storage = value.untyped_storage()
        cdata = int(storage._cdata)  # pylint: disable=protected-access
        return str(value.device), cdata, int(storage.data_ptr()), int(storage.nbytes())
    except (AttributeError, RuntimeError, TypeError) as error:
        _reject(
            "LIVE_SOURCE_STORAGE_REPLACED",
            semantic_path="<storage-token>",
            cause=error,
        )


def _live_tensor_row(path: str, value: torch.Tensor) -> _LiveTensorRow:
    if type(value) is not torch.Tensor:
        _reject("LIVE_SOURCE_TENSOR_SUBCLASS_UNSUPPORTED", semantic_path=path)
    return _LiveTensorRow(
        semantic_path=path,
        tensor=value,
        object_identity=id(value),
        storage_token=_raw_storage_token(value),
        shape=tuple(value.shape),
        dtype=str(value.dtype),
        device=str(value.device),
        stride=tuple(value.stride()),
        storage_offset=int(value.storage_offset()),
        contiguous=bool(value.is_contiguous()),
        requires_grad=bool(value.requires_grad),
        is_leaf=bool(value.is_leaf),
        version=int(value._version),  # pylint: disable=protected-access
        content_hash=production_tensor_sha256(value),
    )


def _capture_live_rows(
    live_sources: dict[str, torch.Tensor],
    ordered_paths: tuple[str, ...],
    *,
    check_aliases: bool = True,
) -> tuple[_LiveTensorRow, ...]:
    if type(live_sources) is not dict:
        _reject("LIVE_SOURCE_CONTAINER_TYPE_MISMATCH")
    if any(type(key) is not str for key in live_sources):
        _reject("LIVE_SOURCE_CONTAINER_TYPE_MISMATCH")
    if set(live_sources) != set(ordered_paths):
        _reject("LIVE_SOURCE_COVERAGE_MISMATCH")
    rows = tuple(_live_tensor_row(path, live_sources[path]) for path in ordered_paths)
    if check_aliases:
        _validate_no_live_aliases(rows)
    return rows


def _validate_no_live_aliases(rows: tuple[_LiveTensorRow, ...]) -> None:
    object_ids: set[int] = set()
    storage_tokens: set[tuple[str, int, int, int]] = set()
    for row in rows:
        if row.object_identity in object_ids:
            _reject(
                "LIVE_SOURCE_OBJECT_ALIAS_CONFLICT", semantic_path=row.semantic_path
            )
        object_ids.add(row.object_identity)
        if _shape_product(row.shape) > 0:
            if row.storage_token in storage_tokens:
                _reject(
                    "LIVE_SOURCE_STORAGE_ALIAS_CONFLICT",
                    semantic_path=row.semantic_path,
                )
            storage_tokens.add(row.storage_token)


def _current_stream_token(device: str) -> tuple[str, int]:
    parsed = torch.device(device)
    if parsed.type != "cuda":
        return str(parsed), 0
    stream = torch.cuda.current_stream(parsed)
    return str(parsed), int(stream.cuda_stream)


def _run_failure_hook(failure_hook: Callable[[str], None] | None, phase: str) -> None:
    if failure_hook is not None:
        failure_hook(phase)


def _rows_equal(
    before: tuple[_LiveTensorRow, ...],
    after: tuple[_LiveTensorRow, ...],
    *,
    read_race: bool,
) -> None:
    if len(before) != len(after):
        _reject(
            "LIVE_SOURCE_READ_RACE" if read_race else "LIVE_SOURCE_COVERAGE_MISMATCH"
        )
    for left, right in zip(before, after):
        code = "LIVE_SOURCE_READ_RACE" if read_race else "LIVE_SOURCE_OBJECT_REPLACED"
        if left.semantic_path != right.semantic_path or left.tensor is not right.tensor:
            _reject(code, semantic_path=left.semantic_path)
        if left.storage_token != right.storage_token:
            _reject(
                (
                    "LIVE_SOURCE_READ_RACE"
                    if read_race
                    else "LIVE_SOURCE_STORAGE_REPLACED"
                ),
                semantic_path=left.semantic_path,
            )
        if (left.shape, left.dtype, left.device) != (
            right.shape,
            right.dtype,
            right.device,
        ):
            _reject(
                (
                    "LIVE_SOURCE_READ_RACE"
                    if read_race
                    else "LIVE_SOURCE_SHAPE_DTYPE_DEVICE_MISMATCH"
                ),
                semantic_path=left.semantic_path,
            )
        if (left.stride, left.storage_offset, left.contiguous) != (
            right.stride,
            right.storage_offset,
            right.contiguous,
        ):
            _reject(
                (
                    "LIVE_SOURCE_READ_RACE"
                    if read_race
                    else "LIVE_SOURCE_STRIDE_OFFSET_MISMATCH"
                ),
                semantic_path=left.semantic_path,
            )
        if (left.requires_grad, left.is_leaf) != (
            right.requires_grad,
            right.is_leaf,
        ):
            _reject(
                (
                    "LIVE_SOURCE_READ_RACE"
                    if read_race
                    else "LIVE_SOURCE_READINESS_MISMATCH"
                ),
                semantic_path=left.semantic_path,
            )
        if left.version != right.version:
            _reject(
                (
                    "LIVE_SOURCE_READ_RACE"
                    if read_race
                    else "LIVE_TENSOR_VERSION_MISMATCH"
                ),
                semantic_path=left.semantic_path,
            )
        if left.content_hash != right.content_hash:
            _reject(
                (
                    "LIVE_SOURCE_READ_RACE"
                    if read_race
                    else "LIVE_SOURCE_CONTENT_MISMATCH"
                ),
                semantic_path=left.semantic_path,
            )


class S4LiveMutableLeaseV1:
    """Process-local strong ownership of the admitted provider Tensors."""

    __slots__ = (
        "_admission_hash",
        "_exact_call_id",
        "_owner_process_id",
        "_owner_thread_id",
        "_entry_device",
        "_entry_stream_token",
        "_state",
        "_source_rows",
    )

    def __init__(
        self,
        admission_hash: str,
        exact_call_id: str,
        rows: tuple[_LiveTensorRow, ...],
    ) -> None:
        devices = {row.device for row in rows}
        if len(devices) != 1:
            _reject("LIVE_SOURCE_SHAPE_DTYPE_DEVICE_MISMATCH")
        self._admission_hash = admission_hash
        self._exact_call_id = exact_call_id
        self._owner_process_id = os.getpid()
        self._owner_thread_id = threading.get_ident()
        self._entry_device = next(iter(devices))
        self._entry_stream_token = _current_stream_token(self._entry_device)
        self._state = "OPEN"
        self._source_rows = rows

    def _serialization_forbidden(self, *_args: object, **_kwargs: object) -> NoReturn:
        _reject("LIVE_LEASE_SERIALIZATION_FORBIDDEN")

    __copy__ = _serialization_forbidden
    __deepcopy__ = _serialization_forbidden
    __getstate__ = _serialization_forbidden
    __reduce__ = _serialization_forbidden
    __reduce_ex__ = _serialization_forbidden

    def _check_owner(self, exact_call_id: str) -> None:
        if self._state == "CLOSED":
            _reject("LIVE_LEASE_ALREADY_CLOSED")
        if os.getpid() != self._owner_process_id:
            _reject("LIVE_SOURCE_OWNER_PROCESS_MISMATCH")
        if threading.get_ident() != self._owner_thread_id:
            _reject("LIVE_SOURCE_OWNER_THREAD_MISMATCH")
        if exact_call_id != self._exact_call_id:
            _reject("EXACT_CALL_IDENTITY_MISMATCH")
        if _current_stream_token(self._entry_device) != self._entry_stream_token:
            _reject("LIVE_SOURCE_STREAM_MISMATCH")

    def revalidate_current_mapping(
        self,
        current_sources: dict[str, torch.Tensor],
        *,
        expected_admission_hash: str,
        exact_call_id: str,
        phase: str,
        require_content_unchanged: bool,
    ) -> None:
        self._check_owner(exact_call_id)
        if expected_admission_hash != self._admission_hash:
            _reject("LIVE_LEASE_ADMISSION_MISMATCH")
        if not phase or self._state not in {"OPEN", "TRANSFERRED_TO_PREPARED_RUNTIME"}:
            _reject("LIVE_LEASE_STATE_MISMATCH")
        paths = tuple(row.semantic_path for row in self._source_rows)
        current = _capture_live_rows(current_sources, paths, check_aliases=False)
        for before, after in zip(self._source_rows, current):
            if before.tensor is not after.tensor:
                _reject(
                    "LIVE_SOURCE_OBJECT_REPLACED", semantic_path=before.semantic_path
                )
            if before.storage_token != after.storage_token:
                _reject(
                    "LIVE_SOURCE_STORAGE_REPLACED", semantic_path=before.semantic_path
                )
            if (before.shape, before.dtype, before.device) != (
                after.shape,
                after.dtype,
                after.device,
            ):
                _reject(
                    "LIVE_SOURCE_SHAPE_DTYPE_DEVICE_MISMATCH",
                    semantic_path=before.semantic_path,
                )
            if (before.stride, before.storage_offset, before.contiguous) != (
                after.stride,
                after.storage_offset,
                after.contiguous,
            ):
                _reject(
                    "LIVE_SOURCE_STRIDE_OFFSET_MISMATCH",
                    semantic_path=before.semantic_path,
                )
            if (before.requires_grad, before.is_leaf) != (
                after.requires_grad,
                after.is_leaf,
            ):
                _reject(
                    "LIVE_SOURCE_READINESS_MISMATCH",
                    semantic_path=before.semantic_path,
                )
            if before.version != after.version:
                _reject(
                    "LIVE_TENSOR_VERSION_MISMATCH", semantic_path=before.semantic_path
                )
            if require_content_unchanged and before.content_hash != after.content_hash:
                _reject(
                    "LIVE_SOURCE_CONTENT_MISMATCH", semantic_path=before.semantic_path
                )
        _validate_no_live_aliases(current)

    def transfer_to_prepared_runtime(
        self, *, expected_admission_hash: str, exact_call_id: str
    ) -> None:
        self._check_owner(exact_call_id)
        if self._state != "OPEN":
            _reject("LIVE_LEASE_ALREADY_TRANSFERRED")
        if expected_admission_hash != self._admission_hash:
            _reject("LIVE_LEASE_ADMISSION_MISMATCH")
        self._state = "TRANSFERRED_TO_PREPARED_RUNTIME"

    def mark_commit_started(self, *, exact_call_id: str) -> None:
        self._check_owner(exact_call_id)
        if self._state != "TRANSFERRED_TO_PREPARED_RUNTIME":
            _reject("LIVE_LEASE_STATE_MISMATCH")
        self._state = "COMMITTING"

    def mark_committed_or_aborted(self, *, exact_call_id: str, outcome: str) -> None:
        self._check_owner(exact_call_id)
        if self._state != "COMMITTING" or outcome not in {
            "COMMITTED",
            "ABORTED_CLEAN",
            "POISONED_NO_RETRY",
        }:
            _reject("LIVE_LEASE_STATE_MISMATCH")
        self._state = outcome

    def close(self) -> None:
        self._source_rows = ()
        self._state = "CLOSED"


class _S4LeaseAdoptionV1:
    __slots__ = ("receipt", "_lease")

    def __init__(
        self, receipt: S4MutableStateAdmissionV1, lease: S4LiveMutableLeaseV1
    ) -> None:
        self.receipt = receipt
        self._lease = lease

    def _serialization_forbidden(self, *_args: object, **_kwargs: object) -> NoReturn:
        _reject("LIVE_LEASE_SERIALIZATION_FORBIDDEN")

    __copy__ = _serialization_forbidden
    __deepcopy__ = _serialization_forbidden
    __getstate__ = _serialization_forbidden
    __reduce__ = _serialization_forbidden
    __reduce_ex__ = _serialization_forbidden


class PreparedS4MutableStateAdmissionV1:
    """One-shot wrapper that keeps receipt and ephemeral lease inseparable."""

    __slots__ = ("receipt", "_live_lease", "_state")

    def __init__(
        self,
        receipt: S4MutableStateAdmissionV1,
        lease: S4LiveMutableLeaseV1,
    ) -> None:
        self.receipt = receipt
        self._live_lease: S4LiveMutableLeaseV1 | None = lease
        self._state = "OPEN"

    def _serialization_forbidden(self, *_args: object, **_kwargs: object) -> NoReturn:
        _reject("LIVE_LEASE_SERIALIZATION_FORBIDDEN")

    __copy__ = _serialization_forbidden
    __deepcopy__ = _serialization_forbidden
    __getstate__ = _serialization_forbidden
    __reduce__ = _serialization_forbidden
    __reduce_ex__ = _serialization_forbidden

    def begin_buffer_prepare(
        self,
        current_sources: dict[str, torch.Tensor],
        *,
        exact_call_id: str,
    ) -> _S4LeaseAdoptionV1:
        if self._state != "OPEN" or self._live_lease is None:
            _reject("LIVE_LEASE_ALREADY_TRANSFERRED")
        self._state = "PREPARING"
        lease = self._live_lease
        try:
            lease.revalidate_current_mapping(
                current_sources,
                expected_admission_hash=self.receipt.admission_hash,
                exact_call_id=exact_call_id,
                phase="buffer-prepare",
                require_content_unchanged=True,
            )
            lease.transfer_to_prepared_runtime(
                expected_admission_hash=self.receipt.admission_hash,
                exact_call_id=exact_call_id,
            )
        except BaseException:
            lease.close()
            self._live_lease = None
            self._state = "FAILED_CLOSED"
            raise
        self._live_lease = None
        self._state = "TRANSFERRED"
        return _S4LeaseAdoptionV1(self.receipt, lease)

    def close(self) -> None:
        if self._live_lease is not None:
            self._live_lease.close()
            self._live_lease = None
        self._state = "CLOSED"


def extract_s4_live_mutable_sources_v1(
    pre_result: object,
    topology: tuple[ProductionReluTopologyV4, ...],
) -> dict[str, torch.Tensor]:
    """Strictly extract the pinned provider's six α and six SparseBeta values."""

    if type(topology) is not tuple or any(
        type(link) is not ProductionReluTopologyV4 for link in topology
    ):
        _reject("TOPOLOGY_IDENTITY_MISMATCH")
    try:
        alpha_wrapper = object.__getattribute__(pre_result, "alphas_by_layer")
        beta_wrapper = object.__getattribute__(pre_result, "betas_by_layer")
        alpha_data = object.__getattribute__(alpha_wrapper, "_data")
        beta_data = object.__getattribute__(beta_wrapper, "_data")
    except (AttributeError, TypeError) as error:
        _reject("LIVE_SOURCE_NESTED_CONTAINER_TYPE_MISMATCH", cause=error)
    alpha_container_admitted = type(alpha_data) is dict or (
        type(alpha_data) is defaultdict
        and object.__getattribute__(alpha_data, "default_factory") is dict
    )
    if not alpha_container_admitted or type(beta_data) is not dict:
        _reject("LIVE_SOURCE_NESTED_CONTAINER_TYPE_MISMATCH")
    result: dict[str, torch.Tensor] = {}
    for slot_ordinal, link in enumerate(topology):
        try:
            link.validate()
        except ValueError as error:
            _reject(
                "TOPOLOGY_IDENTITY_MISMATCH", slot_ordinal=slot_ordinal, cause=error
            )
        if (
            link.provider_activation not in alpha_data
            or link.provider_preactivation not in beta_data
        ):
            _reject("LIVE_SOURCE_COVERAGE_MISMATCH", slot_ordinal=slot_ordinal)
        activation = alpha_data[link.provider_activation]
        beta_collection = beta_data[link.provider_preactivation]
        if type(activation) is not dict or type(beta_collection) is not list:
            _reject(
                "LIVE_SOURCE_NESTED_CONTAINER_TYPE_MISMATCH",
                slot_ordinal=slot_ordinal,
            )
        if link.provider_start_node not in activation or len(beta_collection) != 1:
            _reject("LIVE_SOURCE_COVERAGE_MISMATCH", slot_ordinal=slot_ordinal)
        alpha = activation[link.provider_start_node]
        sparse = beta_collection[0]
        try:
            beta = object.__getattribute__(sparse, "val")
        except (AttributeError, TypeError) as error:
            _reject(
                "LIVE_SOURCE_NESTED_CONTAINER_TYPE_MISMATCH",
                slot_ordinal=slot_ordinal,
                cause=error,
            )
        if type(alpha) is not torch.Tensor or type(beta) is not torch.Tensor:
            _reject(
                "LIVE_SOURCE_TENSOR_SUBCLASS_UNSUPPORTED", slot_ordinal=slot_ordinal
            )
        alpha_path = (
            f"alpha/{_encode_component(link.provider_activation)}/"
            f"{_encode_component(link.provider_start_node)}"
        )
        beta_path = f"beta/{_encode_component(link.provider_preactivation)}/0/value"
        if alpha_path in result or beta_path in result:
            _reject("LIVE_SOURCE_COVERAGE_MISMATCH", slot_ordinal=slot_ordinal)
        result[alpha_path] = alpha
        result[beta_path] = beta
    return result


def _ordered_topology(
    topology: tuple[ProductionReluTopologyV4, ...],
    plan: R31FullRegionPlanV1,
) -> tuple[ProductionReluTopologyV4, ...]:
    if type(topology) is not tuple or any(
        type(link) is not ProductionReluTopologyV4 for link in topology
    ):
        _reject("TOPOLOGY_IDENTITY_MISMATCH")
    by_native: dict[str, ProductionReluTopologyV4] = {}
    for link in topology:
        try:
            link.validate()
        except ValueError as error:
            _reject("TOPOLOGY_IDENTITY_MISMATCH", cause=error)
        if link.native_preactivation in by_native:
            _reject("TOPOLOGY_IDENTITY_MISMATCH")
        by_native[link.native_preactivation] = link
    if set(by_native) != {layout.native_preactivation for layout in plan.relu_layouts}:
        _reject("TOPOLOGY_IDENTITY_MISMATCH")
    ordered = tuple(
        by_native[layout.native_preactivation] for layout in plan.relu_layouts
    )
    for ordinal, (layout, link) in enumerate(zip(plan.relu_layouts, ordered)):
        alpha_path = (
            f"alpha/{_encode_component(link.provider_activation)}/"
            f"{_encode_component(link.provider_start_node)}"
        )
        beta_path = f"beta/{_encode_component(link.provider_preactivation)}/0/value"
        if (
            layout.provider_activation != link.provider_activation
            or layout.provider_preactivation != link.provider_preactivation
            or layout.alpha_path != alpha_path
            or layout.beta_path != beta_path
        ):
            _reject("TOPOLOGY_IDENTITY_MISMATCH", slot_ordinal=ordinal)
    return ordered


def _one_snapshot_tensor(
    tensor_map: dict[str, OwnedProductionTensorV4],
    path: str,
    *,
    role: ProductionTensorRole,
    ownership: ProductionTensorOwnership,
    slot_ordinal: int,
) -> OwnedProductionTensorV4:
    value = tensor_map.get(path)
    if value is None:
        _reject(
            "MUTABLE_STATE_COVERAGE_INCOMPLETE",
            slot_ordinal=slot_ordinal,
            semantic_path=path,
        )
    if value.role != role or value.ownership != ownership:
        _reject(
            "PLAN_SNAPSHOT_PROJECTION_MISMATCH",
            slot_ordinal=slot_ordinal,
            semantic_path=path,
        )
    return value


def _plan_tensor_spec(plan: R31FullRegionPlanV1, name: str):  # type: ignore[no-untyped-def]
    values = [spec for spec in plan.tensor_specs if spec.name == name]
    if len(values) != 1:
        _reject("PLAN_SNAPSHOT_PROJECTION_MISMATCH")
    return values[0]


def _beta_history_hash_and_validate(
    snapshot: ProductionStateSnapshotV4,
    layout: R31ReluLayoutV1,
    beta: OwnedProductionTensorV4,
    location: OwnedProductionTensorV4,
    sign: OwnedProductionTensorV4,
    *,
    slot_ordinal: int,
) -> str:
    if len(beta.value.shape) != 2 or beta.value.shape[0] != len(layout.beta_locations):
        _reject("BETA_HISTORY_WIDTH_MISMATCH", slot_ordinal=slot_ordinal)
    width = int(beta.value.shape[1])
    entries = sorted(
        (
            item
            for item in snapshot.history
            if item.layer_name == layout.provider_preactivation
        ),
        key=lambda item: item.domain_ordinal,
    )
    if len(entries) != beta.value.shape[0] or any(
        entry.domain_ordinal != ordinal
        or len(entry.locations) != width
        or tuple(int(value) for value in location.value[ordinal].tolist())
        != entry.locations
        or tuple(float(value) for value in sign.value[ordinal].tolist())
        != entry.coefficients
        or tuple(layout.beta_locations[ordinal]) != entry.locations
        for ordinal, entry in enumerate(entries)
    ):
        _reject("BETA_HISTORY_WIDTH_MISMATCH", slot_ordinal=slot_ordinal)
    return _canonical_hash([entry.to_dict() for entry in entries])


def _slot_from_sources(
    *,
    ordinal: int,
    layout: R31ReluLayoutV1,
    link: ProductionReluTopologyV4,
    snapshot: ProductionStateSnapshotV4,
    tensor_map: dict[str, OwnedProductionTensorV4],
    plan: R31FullRegionPlanV1,
    alpha_row: _LiveTensorRow,
    beta_row: _LiveTensorRow,
) -> S4MutableSlotV1:
    alpha = _one_snapshot_tensor(
        tensor_map,
        layout.alpha_path,
        role=ProductionTensorRole.ALPHA,
        ownership=ProductionTensorOwnership.MUTABLE_COPY_OUT,
        slot_ordinal=ordinal,
    )
    beta = _one_snapshot_tensor(
        tensor_map,
        layout.beta_path,
        role=ProductionTensorRole.BETA_VALUE,
        ownership=ProductionTensorOwnership.MUTABLE_COPY_OUT,
        slot_ordinal=ordinal,
    )
    location_path = layout.beta_path.removesuffix("/value") + "/location"
    sign_path = layout.beta_path.removesuffix("/value") + "/sign"
    location = _one_snapshot_tensor(
        tensor_map,
        location_path,
        role=ProductionTensorRole.BETA_LOCATION,
        ownership=ProductionTensorOwnership.COPY_IN,
        slot_ordinal=ordinal,
    )
    sign = _one_snapshot_tensor(
        tensor_map,
        sign_path,
        role=ProductionTensorRole.BETA_SIGN,
        ownership=ProductionTensorOwnership.COPY_IN,
        slot_ordinal=ordinal,
    )
    if not all(
        bool(torch.isfinite(value.value).all().item())
        for value in (alpha, beta, location, sign)
    ):
        _reject("NONFINITE_MUTABLE_STATE", slot_ordinal=ordinal)
    if beta.value.numel() and not bool((beta.value >= 0).all().item()):
        _reject("BETA_LOCATION_SIGN_HISTORY_MISMATCH", slot_ordinal=ordinal)
    if (
        alpha.axes[:3] != ("alpha_polarity", "start_spec", "domain")
        or tuple(alpha.value.shape[:3]) != (2, 1, plan.domain_count)
        or tuple(alpha.value.shape[3:]) != (len(layout.alpha_flat_indices),)
    ):
        _reject("ALPHA_MUTABLE_DIRECTION_MISMATCH", slot_ordinal=ordinal)
    alpha_spec = _plan_tensor_spec(plan, f"relu/{layout.native_preactivation}/alpha")
    beta_spec = _plan_tensor_spec(plan, f"relu/{layout.native_preactivation}/beta")
    if (
        tuple(alpha.value.shape) != alpha_spec.shape
        or str(alpha.value.dtype) != alpha_spec.dtype
        or alpha.content_sha256 != alpha_spec.content_sha256
        or tuple(beta.value.shape) != beta_spec.shape
        or str(beta.value.dtype) != beta_spec.dtype
        or beta.content_sha256 != beta_spec.content_sha256
    ):
        _reject("PLAN_SNAPSHOT_PROJECTION_MISMATCH", slot_ordinal=ordinal)
    for source, row in ((alpha, alpha_row), (beta, beta_row)):
        if (
            tuple(source.value.shape) != row.shape
            or str(source.value.dtype) != row.dtype
            or source.source_device != row.device
            or source.content_sha256 != row.content_hash
        ):
            _reject(
                "LIVE_SOURCE_CONTENT_MISMATCH",
                slot_ordinal=ordinal,
                semantic_path=source.semantic_path,
            )
        if not row.is_leaf:
            _reject(
                "LIVE_SOURCE_READINESS_MISMATCH",
                slot_ordinal=ordinal,
                semantic_path=source.semantic_path,
            )
    active = alpha.value[0, 0]
    preserved = alpha.value[1, 0]
    if tuple(active.shape) != (plan.domain_count, len(layout.alpha_flat_indices)):
        _reject("ALPHA_LAYOUT_IDENTITY_MISMATCH", slot_ordinal=ordinal)
    if location.value.shape != beta.value.shape or sign.value.shape != beta.value.shape:
        _reject("BETA_LOCATION_SIGN_HISTORY_MISMATCH", slot_ordinal=ordinal)
    history_hash = _beta_history_hash_and_validate(
        snapshot,
        layout,
        beta,
        location,
        sign,
        slot_ordinal=ordinal,
    )
    return S4MutableSlotV1(
        slot_ordinal=ordinal,
        native_preactivation=layout.native_preactivation,
        provider_activation=link.provider_activation,
        provider_preactivation=link.provider_preactivation,
        provider_start_node=link.provider_start_node,
        alpha_semantic_path=layout.alpha_path,
        alpha_source_axes=alpha.axes,
        alpha_source_shape=tuple(alpha.value.shape),
        alpha_source_dtype=str(alpha.value.dtype),
        alpha_source_device=alpha.source_device,
        alpha_source_hash=alpha.content_sha256,
        alpha_live_object_group=f"object:{2 * ordinal:06d}",
        alpha_live_storage_group=f"storage:{2 * ordinal:06d}",
        alpha_live_version=alpha_row.version,
        alpha_live_stride=alpha_row.stride,
        alpha_live_storage_offset=alpha_row.storage_offset,
        alpha_live_contiguous=alpha_row.contiguous,
        alpha_live_requires_grad=alpha_row.requires_grad,
        alpha_live_is_leaf=alpha_row.is_leaf,
        alpha_live_content_hash=alpha_row.content_hash,
        alpha_active_shape=tuple(active.shape),
        alpha_active_hash=production_tensor_sha256(active),
        alpha_active_element_count=active.numel(),
        alpha_preserved_shape=tuple(preserved.shape),
        alpha_preserved_hash=production_tensor_sha256(preserved),
        alpha_preserved_element_count=preserved.numel(),
        feature_shape=layout.feature_shape,
        alpha_flat_indices=layout.alpha_flat_indices,
        alpha_layout_hash=_canonical_hash(
            {
                "feature_shape": list(layout.feature_shape),
                "alpha_flat_indices": list(layout.alpha_flat_indices),
            }
        ),
        beta_semantic_path=layout.beta_path,
        beta_source_axes=beta.axes,
        beta_source_shape=tuple(beta.value.shape),
        beta_source_dtype=str(beta.value.dtype),
        beta_source_device=beta.source_device,
        beta_source_hash=beta.content_sha256,
        beta_live_object_group=f"object:{2 * ordinal + 1:06d}",
        beta_live_storage_group=(
            f"storage:{2 * ordinal + 1:06d}"
            if beta.value.numel()
            else f"empty:{2 * ordinal + 1:06d}"
        ),
        beta_live_version=beta_row.version,
        beta_live_stride=beta_row.stride,
        beta_live_storage_offset=beta_row.storage_offset,
        beta_live_contiguous=beta_row.contiguous,
        beta_live_requires_grad=beta_row.requires_grad,
        beta_live_is_leaf=beta_row.is_leaf,
        beta_live_content_hash=beta_row.content_hash,
        beta_location_hash=location.content_sha256,
        beta_sign_hash=sign.content_sha256,
        beta_history_hash=history_hash,
        beta_active=bool(beta.value.numel()),
        beta_element_count=beta.value.numel(),
    )


def _prepare_s4_mutable_state_admission_v1(
    snapshot: ProductionStateSnapshotV4,
    topology: tuple[ProductionReluTopologyV4, ...],
    production_plan: R31FullRegionPlanV1,
    live_mutable_sources: dict[str, torch.Tensor],
    *,
    exact_call_id: str,
    failure_hook: Callable[[str], None] | None,
) -> PreparedS4MutableStateAdmissionV1:
    if type(live_mutable_sources) is not dict:
        _reject("LIVE_SOURCE_CONTAINER_TYPE_MISMATCH")
    exact_call_id, exact_call_hash = _validate_exact_call_id(exact_call_id)
    _run_failure_hook(failure_hook, "after_input_envelope")
    if (
        type(snapshot) is not ProductionStateSnapshotV4
        or snapshot.schema_version != RVIR_V4_STATE_SCHEMA_VERSION
        or not snapshot.snapshot_id
    ):
        _reject("SNAPSHOT_SCHEMA_VERSION_MISMATCH")
    try:
        snapshot.validate()
    except (TypeError, ValueError, RuntimeError) as error:
        _reject("SNAPSHOT_SEMANTIC_INVALID", cause=error)
    if (
        type(production_plan) is not R31FullRegionPlanV1
        or production_plan.schema_version != R31_PLAN_SCHEMA
    ):
        _reject("PLAN_SCHEMA_VERSION_MISMATCH")
    try:
        production_plan.validate()
    except (TypeError, ValueError, RuntimeError) as error:
        _reject("PLAN_SEMANTIC_INVALID", cause=error)
    policy = snapshot.optimizer_policy
    if (
        not policy.bound_lower
        or policy.bound_upper
        or not policy.fix_intermediate_bounds
    ):
        _reject("OPTIMIZER_POLICY_MISMATCH")
    _run_failure_hook(failure_hook, "after_snapshot_validation")
    ordered_topology = _ordered_topology(topology, production_plan)
    ordered_paths = tuple(
        path
        for layout in production_plan.relu_layouts
        for path in (layout.alpha_path, layout.beta_path)
    )
    mutable_snapshot_paths = {
        item.semantic_path
        for item in snapshot.tensors
        if item.ownership == ProductionTensorOwnership.MUTABLE_COPY_OUT
    }
    if mutable_snapshot_paths != set(ordered_paths):
        _reject("MUTABLE_STATE_COVERAGE_INCOMPLETE")
    entry_rows = _capture_live_rows(live_mutable_sources, ordered_paths)
    entry_process_id = os.getpid()
    entry_thread_id = threading.get_ident()
    entry_devices = {row.device for row in entry_rows}
    if len(entry_devices) != 1:
        _reject("LIVE_SOURCE_SHAPE_DTYPE_DEVICE_MISMATCH")
    entry_device = next(iter(entry_devices))
    entry_stream_token = _current_stream_token(entry_device)
    _run_failure_hook(failure_hook, "after_first_live_capture")
    row_map = {row.semantic_path: row for row in entry_rows}
    tensor_map = snapshot.tensor_map()
    slots = tuple(
        _slot_from_sources(
            ordinal=ordinal,
            layout=layout,
            link=link,
            snapshot=snapshot,
            tensor_map=tensor_map,
            plan=production_plan,
            alpha_row=row_map[layout.alpha_path],
            beta_row=row_map[layout.beta_path],
        )
        for ordinal, (layout, link) in enumerate(
            zip(production_plan.relu_layouts, ordered_topology)
        )
    )
    expected_beta_activity = tuple(
        any(layout.beta_locations) for layout in production_plan.relu_layouts
    )
    if tuple(slot.beta_active for slot in slots) != expected_beta_activity:
        _reject("ACTIVE_BETA_COVERAGE_INCOMPLETE")
    topology_hash = _canonical_hash([link.to_dict() for link in ordered_topology])
    projection = {
        "plan_hash": production_plan.stable_hash(),
        "snapshot_hash": _snapshot_hash(snapshot),
        "topology_hash": topology_hash,
        "slots": [
            {
                "native": slot.native_preactivation,
                "alpha_path": slot.alpha_semantic_path,
                "beta_path": slot.beta_semantic_path,
                "feature_shape": list(slot.feature_shape),
                "alpha_flat_indices": list(slot.alpha_flat_indices),
                "alpha_source_hash": slot.alpha_source_hash,
                "beta_source_hash": slot.beta_source_hash,
                "beta_location_hash": slot.beta_location_hash,
                "beta_sign_hash": slot.beta_sign_hash,
                "beta_history_hash": slot.beta_history_hash,
            }
            for slot in slots
        ],
    }
    provisional = S4MutableStateAdmissionV1(
        snapshot_hash=_snapshot_hash(snapshot),
        production_plan_hash=production_plan.stable_hash(),
        plan_binding_projection_hash=_canonical_hash(projection),
        oracle_mapping_provenance_hash=production_plan.source_state_hash,
        topology_hash=topology_hash,
        optimizer_policy_hash=_canonical_hash(policy.to_dict()),
        exact_call_identity_hash=exact_call_hash,
        slots=slots,
        mutable_path_set_hash=_canonical_hash(sorted(ordered_paths)),
        alpha_source_count=len(slots),
        alpha_stored_element_count=sum(
            _shape_product(slot.alpha_source_shape) for slot in slots
        ),
        alpha_active_element_count=sum(
            slot.alpha_active_element_count for slot in slots
        ),
        alpha_preserved_element_count=sum(
            slot.alpha_preserved_element_count for slot in slots
        ),
        beta_slot_count=len(slots),
        active_beta_slot_count=sum(slot.beta_active for slot in slots),
        active_beta_element_count=sum(slot.beta_element_count for slot in slots),
        live_tensor_count=len(ordered_paths),
        live_element_count_per_pass=sum(
            _shape_product(row.shape) for row in entry_rows
        ),
        live_bytes_per_pass=sum(
            _shape_product(row.shape) * row.tensor.element_size() for row in entry_rows
        ),
        live_content_capture_pass_count=2,
        device_to_host_validation_copy_count=2 * len(ordered_paths),
        device_to_host_validation_bytes=2
        * sum(
            _shape_product(row.shape) * row.tensor.element_size() for row in entry_rows
        ),
        candidate_kernel_launch_count=0,
        candidate_cuda_allocation_count=0,
        dense_materialization_observed=False,
        timing_recorded=False,
        performance_claimed=False,
        process_global_query_exclusivity_validated=False,
        construction_model_hash=S4_MUTABLE_STATE_CONSTRUCTION_HASH_V4,
        admission_hash="0" * 64,
    )
    receipt = replace(
        provisional,
        admission_hash=_canonical_hash(provisional._payload_without_hash()),
    )
    receipt.validate()
    _run_failure_hook(failure_hook, "after_receipt_validation")
    _run_failure_hook(failure_hook, "before_second_live_capture")
    exit_rows = _capture_live_rows(live_mutable_sources, ordered_paths)
    if (
        os.getpid() != entry_process_id
        or threading.get_ident() != entry_thread_id
        or _current_stream_token(entry_device) != entry_stream_token
    ):
        _reject("LIVE_SOURCE_READ_RACE")
    _rows_equal(entry_rows, exit_rows, read_race=True)
    _run_failure_hook(failure_hook, "before_lease_publish")
    lease = S4LiveMutableLeaseV1(receipt.admission_hash, exact_call_id, exit_rows)
    return PreparedS4MutableStateAdmissionV1(receipt, lease)


def prepare_s4_mutable_state_admission_v1(
    snapshot: ProductionStateSnapshotV4,
    topology: tuple[ProductionReluTopologyV4, ...],
    production_plan: R31FullRegionPlanV1,
    live_mutable_sources: dict[str, torch.Tensor],
    *,
    exact_call_id: str,
) -> PreparedS4MutableStateAdmissionV1:
    """Admit all mutable α/β owners without compiling or launching a candidate."""

    return _prepare_s4_mutable_state_admission_v1(
        snapshot,
        topology,
        production_plan,
        live_mutable_sources,
        exact_call_id=exact_call_id,
        failure_hook=None,
    )


__all__ = [
    "S4MutableStateAdmissionError",
    "S4MutableSlotV1",
    "S4MutableStateAdmissionV1",
    "S4LiveMutableLeaseV1",
    "PreparedS4MutableStateAdmissionV1",
    "extract_s4_live_mutable_sources_v1",
    "prepare_s4_mutable_state_admission_v1",
]
