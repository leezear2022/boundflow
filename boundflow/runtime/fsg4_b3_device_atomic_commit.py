"""GPU-resident fail-closed atomic commit plan for FSG4/B3-C."""

# pylint: disable=too-many-arguments,too-many-locals,too-many-statements
# pylint: disable=too-many-boolean-expressions,missing-function-docstring
# pylint: disable=too-many-instance-attributes,protected-access

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from typing import cast, Mapping, MutableMapping, Sequence

import torch

from .native_alpha_beta_optimization_state import NativeAlphaBetaOptimizationState
from .rvir_v4_pre_state_initializer import ProductionReluTopologyV4
from .rvir_v4_production_state import (
    ProductionStateSnapshotV4,
    ProductionTensorOwnership,
    ProductionTensorRole,
    production_tensor_sha256,
)

FSG4_B3_DEVICE_ATOMIC_COMMIT_SCHEMA = "boundflow.fsg4-b3-device-atomic-commit/v1"


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _encode(value: str) -> str:
    return value.replace("%", "%25").replace("/", "%2F")


@dataclass(frozen=True)
class DeviceAtomicPathSpecV1:
    """One statically frozen provider-owned mutable GPU path."""

    semantic_path: str
    role: ProductionTensorRole
    shape: tuple[int, ...]
    dtype: str
    device: str
    alias_group: str
    rollback_ordinal: int

    def validate(self) -> None:
        if (
            not self.semantic_path
            or self.role
            not in {ProductionTensorRole.ALPHA, ProductionTensorRole.BETA_VALUE}
            or any(dimension < 0 for dimension in self.shape)
            or self.dtype not in {"torch.float16", "torch.float32", "torch.float64"}
            or torch.device(self.device).type != "cuda"
            or not self.alias_group
            or self.rollback_ordinal < 0
        ):
            raise ValueError("FSG4/B3-C device atomic path contract differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "semantic_path": self.semantic_path,
            "role": self.role.value,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "device": self.device,
            "alias_group": self.alias_group,
            "rollback_ordinal": self.rollback_ordinal,
        }


@dataclass(frozen=True)
class DeviceAtomicCommitPlanV1:
    """Reusable static ownership, placement, alias, and rollback contract."""

    plan_id: str
    prepared_template_hash: str
    paths: tuple[DeviceAtomicPathSpecV1, ...]
    schema_version: str = FSG4_B3_DEVICE_ATOMIC_COMMIT_SCHEMA

    @property
    def path_map(self) -> dict[str, DeviceAtomicPathSpecV1]:
        return {row.semantic_path: row for row in self.paths}

    @property
    def rollback_order(self) -> tuple[str, ...]:
        return tuple(
            row.semantic_path
            for row in sorted(self.paths, key=lambda item: item.rollback_ordinal)
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "prepared_template_hash": self.prepared_template_hash,
            "paths": [row.to_dict() for row in self.paths],
            "rollback_order": list(self.rollback_order),
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.identity_payload())

    def validate(self) -> None:
        for row in self.paths:
            row.validate()
        path_names = tuple(row.semantic_path for row in self.paths)
        roles = tuple(row.role for row in self.paths)
        ordinals = tuple(row.rollback_ordinal for row in self.paths)
        if (
            self.schema_version != FSG4_B3_DEVICE_ATOMIC_COMMIT_SCHEMA
            or not self.plan_id
            or not _is_sha256(self.prepared_template_hash)
            or len(self.paths) != 12
            or tuple(sorted(path_names)) != path_names
            or len(set(path_names)) != 12
            or roles.count(ProductionTensorRole.ALPHA) != 6
            or roles.count(ProductionTensorRole.BETA_VALUE) != 6
            or set(ordinals) != set(range(12))
            or len(self.rollback_order) != 12
        ):
            raise ValueError("FSG4/B3-C device atomic plan differs")


def compile_device_atomic_commit_plan_v1(
    *,
    plan_id: str,
    prepared_template_hash: str,
    paths: Sequence[DeviceAtomicPathSpecV1],
) -> DeviceAtomicCommitPlanV1:
    """Compile the exact static transaction contract before the live core call."""

    plan = DeviceAtomicCommitPlanV1(
        plan_id=plan_id,
        prepared_template_hash=prepared_template_hash,
        paths=tuple(sorted(paths, key=lambda row: row.semantic_path)),
    )
    plan.validate()
    return plan


@dataclass(frozen=True)
class DeviceAtomicCandidateV1:
    """One private candidate that never leaves its target CUDA device."""

    semantic_path: str
    role: ProductionTensorRole
    value: torch.Tensor

    def validate(self, spec: DeviceAtomicPathSpecV1) -> None:
        if (
            self.semantic_path != spec.semantic_path
            or self.role != spec.role
            or not torch.is_tensor(self.value)
            or tuple(self.value.shape) != spec.shape
            or str(self.value.dtype) != spec.dtype
            or str(self.value.device) != spec.device
            or not bool(torch.isfinite(self.value).all().item())
        ):
            raise ValueError("FSG4/B3-C device candidate differs")


def _tensor_version(value: torch.Tensor) -> int:
    version = int(value._version)  # pylint: disable=protected-access
    if version < 0:
        raise ValueError("FSG4/B3-C tensor version differs")
    return version


def _storage_token(value: torch.Tensor) -> tuple[str, int]:
    # Distinct empty CUDA tensors commonly have data_ptr()==0.  Object identity is
    # the only useful live alias discriminator for those zero-storage values.
    if value.numel() == 0:
        return ("empty-object", id(value))
    return (str(value.device), int(value.untyped_storage().data_ptr()))


def _validate_alias_contract(
    plan: DeviceAtomicCommitPlanV1, targets: Mapping[str, torch.Tensor]
) -> None:
    rows = plan.path_map
    names = tuple(sorted(rows))
    tokens = {path: _storage_token(targets[path]) for path in names}
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            expected_alias = rows[left].alias_group == rows[right].alias_group
            observed_alias = tokens[left] == tokens[right]
            if expected_alias != observed_alias:
                raise ValueError("FSG4/B3-C live target alias contract differs")


def validate_device_atomic_targets_v1(
    plan: DeviceAtomicCommitPlanV1,
    targets: Mapping[str, torch.Tensor],
) -> None:
    """Validate inventory and metadata without a GPU-to-CPU content transfer."""

    plan.validate()
    if set(targets) != set(plan.path_map):
        raise ValueError("FSG4/B3-C live target inventory differs")
    for path, spec in plan.path_map.items():
        value = targets[path]
        if (
            not torch.is_tensor(value)
            or tuple(value.shape) != spec.shape
            or str(value.dtype) != spec.dtype
            or str(value.device) != spec.device
        ):
            raise ValueError("FSG4/B3-C live target placement differs")
    _validate_alias_contract(plan, targets)


def _host_structure(value: object) -> object:
    if torch.is_tensor(value):
        tensor = cast(torch.Tensor, value)
        return {
            "kind": "tensor",
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "version": _tensor_version(tensor),
        }
    if isinstance(value, Mapping):
        return {
            str(key): _host_structure(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_host_structure(item) for item in value]
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TypeError(f"FSG4/B3-C host packet value differs: {type(value)}")


def _host_version(value: Mapping[str, object]) -> str:
    return _canonical_hash(_host_structure(value))


def _host_pre_version(value: Mapping[str, object]) -> str:
    retained: dict[str, object] = {}
    for key in ("depths", "history", "thresholds"):
        if key not in value:
            raise ValueError(f"FSG4/B3-C live host packet misses {key}")
        retained[key] = value[key]
    return _canonical_hash(
        {
            "key_inventory": sorted(str(key) for key in value),
            "retained": _host_structure(retained),
        }
    )


def _project_alpha_device(
    *,
    target: torch.Tensor,
    dense: torch.Tensor,
    snapshot: ProductionStateSnapshotV4,
    activation: str,
) -> torch.Tensor:
    prefix = f"alpha_layout/{_encode(activation)}"
    tensor_map = snapshot.tensor_map()
    indices: list[torch.Tensor] = []
    ordinal = 0
    while f"{prefix}/feature_index/{ordinal}" in tensor_map:
        indices.append(
            tensor_map[f"{prefix}/feature_index/{ordinal}"].value.to(
                device=dense.device, dtype=torch.long
            )
        )
        ordinal += 1
    projected = target.detach().clone()
    compressed = (
        dense[(slice(None),) + tuple(indices)]
        if indices
        else dense.reshape_as(target[0, 0])
    )
    if compressed.shape != target[0, 0].shape:
        raise ValueError("FSG4/B3-C alpha projection shape differs")
    projected[0, 0].copy_(compressed.to(dtype=target.dtype))
    return projected


def _project_beta_device(
    *,
    target: torch.Tensor,
    dense: torch.Tensor,
    location: torch.Tensor,
) -> torch.Tensor:
    flat = dense.reshape(int(dense.shape[0]), -1)
    device_location = location.to(device=dense.device, dtype=torch.long)
    projected = torch.stack(
        [flat[domain, device_location[domain]] for domain in range(int(dense.shape[0]))]
    ).to(dtype=target.dtype)
    if projected.shape != target.shape:
        raise ValueError("FSG4/B3-C beta projection shape differs")
    return projected


def _materialize_device_candidate(
    path: str, role: ProductionTensorRole, value: torch.Tensor
) -> DeviceAtomicCandidateV1:
    """Single explicit seam used by the physical activation counter."""

    return DeviceAtomicCandidateV1(path, role, value)


@dataclass(frozen=True)
class DeviceAtomicTransactionV1:
    """Dynamic GPU candidates and pre-mutation versions for one core call."""

    plan: DeviceAtomicCommitPlanV1
    core_instance_hash: str
    pre_snapshot_hash: str
    target_versions: tuple[tuple[str, int], ...]
    candidates: tuple[DeviceAtomicCandidateV1, ...]
    terminal_lower: torch.Tensor
    host_pre_version: str
    host_candidate: tuple[tuple[str, object], ...]
    transaction_version: str
    schema_version: str = FSG4_B3_DEVICE_ATOMIC_COMMIT_SCHEMA

    @property
    def candidate_values(self) -> dict[str, torch.Tensor]:
        return {
            candidate.semantic_path: candidate.value for candidate in self.candidates
        }

    @property
    def host_candidate_map(self) -> dict[str, object]:
        return dict(self.host_candidate)

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "plan_hash": self.plan.stable_hash(),
            "core_instance_hash": self.core_instance_hash,
            "pre_snapshot_hash": self.pre_snapshot_hash,
            "target_versions": [list(row) for row in self.target_versions],
            "candidate_metadata": [
                {
                    "semantic_path": candidate.semantic_path,
                    "role": candidate.role.value,
                    "shape": list(candidate.value.shape),
                    "dtype": str(candidate.value.dtype),
                    "device": str(candidate.value.device),
                }
                for candidate in self.candidates
            ],
            "terminal_lower": {
                "shape": list(self.terminal_lower.shape),
                "dtype": str(self.terminal_lower.dtype),
                "device": str(self.terminal_lower.device),
            },
            "host_pre_version": self.host_pre_version,
            "host_candidate_version": _host_version(self.host_candidate_map),
        }

    def validate(self) -> None:
        self.plan.validate()
        specs = self.plan.path_map
        versions = dict(self.target_versions)
        candidates = self.candidate_values
        for candidate in self.candidates:
            candidate.validate(specs[candidate.semantic_path])
        if (
            self.schema_version != FSG4_B3_DEVICE_ATOMIC_COMMIT_SCHEMA
            or not _is_sha256(self.core_instance_hash)
            or not _is_sha256(self.pre_snapshot_hash)
            or not _is_sha256(self.host_pre_version)
            or len(versions) != len(self.target_versions)
            or set(versions) != set(specs)
            or set(candidates) != set(specs)
            or len(self.candidates) != 12
            or tuple(candidate.semantic_path for candidate in self.candidates)
            != tuple(sorted(specs))
            or set(self.host_candidate_map) != {"depths", "history", "thresholds"}
            or tuple(self.terminal_lower.shape) != (6, 1)
            or str(self.terminal_lower.device) != self.plan.paths[0].device
            or not bool(torch.isfinite(self.terminal_lower).all().item())
            or self.transaction_version != _canonical_hash(self.identity_payload())
        ):
            raise ValueError("FSG4/B3-C device transaction differs")


def stage_device_atomic_transaction_v1(
    *,
    plan: DeviceAtomicCommitPlanV1,
    core_instance_hash: str,
    pre_snapshot_hash: str,
    pre_snapshot: ProductionStateSnapshotV4,
    live_targets: Mapping[str, torch.Tensor],
    terminal_state: NativeAlphaBetaOptimizationState,
    topology: tuple[ProductionReluTopologyV4, ...],
    terminal_lower: torch.Tensor,
    host_packet: Mapping[str, object],
    host_packet_candidate: Mapping[str, object],
) -> DeviceAtomicTransactionV1:
    """Stage twelve CUDA candidates without materializing a CPU candidate snapshot."""

    plan.validate()
    pre_snapshot.validate()
    terminal_state.validate()
    validate_device_atomic_targets_v1(plan, live_targets)
    if (
        pre_snapshot.stable_hash() != pre_snapshot_hash
        or len(topology) != 6
        or set(host_packet_candidate) != {"depths", "history", "thresholds"}
        or tuple(terminal_lower.shape) != (6, 1)
        or str(terminal_lower.device) != plan.paths[0].device
        or not bool(torch.isfinite(terminal_lower).all().item())
    ):
        raise ValueError("FSG4/B3-C device transaction source differs")
    pre_map = pre_snapshot.tensor_map()
    mutable_paths = {
        path
        for path, tensor in pre_map.items()
        if tensor.ownership == ProductionTensorOwnership.MUTABLE_COPY_OUT
    }
    if mutable_paths != set(plan.path_map):
        raise ValueError("FSG4/B3-C mutable snapshot inventory differs")
    candidates: dict[str, DeviceAtomicCandidateV1] = {}
    for link in topology:
        link.validate()
        native = link.native_preactivation
        alpha_path = (
            f"alpha/{_encode(link.provider_activation)}/"
            f"{_encode(link.provider_start_node)}"
        )
        beta_prefix = f"beta/{_encode(link.provider_preactivation)}/0"
        beta_path = f"{beta_prefix}/value"
        alpha_value = _project_alpha_device(
            target=live_targets[alpha_path],
            dense=terminal_state.alphas[native],
            snapshot=pre_snapshot,
            activation=link.provider_activation,
        )
        beta_value = _project_beta_device(
            target=live_targets[beta_path],
            dense=terminal_state.betas[native],
            location=pre_map[f"{beta_prefix}/location"].value,
        )
        for path, value in ((alpha_path, alpha_value), (beta_path, beta_value)):
            spec = plan.path_map[path]
            candidate = _materialize_device_candidate(path, spec.role, value)
            candidate.validate(spec)
            candidates[path] = candidate
    if set(candidates) != set(plan.path_map):
        raise ValueError("FSG4/B3-C candidate inventory differs")
    versions = tuple(
        (path, _tensor_version(live_targets[path])) for path in sorted(live_targets)
    )
    transaction = DeviceAtomicTransactionV1(
        plan=plan,
        core_instance_hash=core_instance_hash,
        pre_snapshot_hash=pre_snapshot_hash,
        target_versions=versions,
        candidates=tuple(candidates[path] for path in sorted(candidates)),
        terminal_lower=terminal_lower,
        host_pre_version=_host_pre_version(host_packet),
        host_candidate=tuple(
            sorted(host_packet_candidate.items(), key=lambda item: item[0])
        ),
        transaction_version="",
    )
    transaction = replace(
        transaction,
        transaction_version=_canonical_hash(transaction.identity_payload()),
    )
    transaction.validate()
    return transaction


def _copy_device_value(target: torch.Tensor, source: torch.Tensor) -> None:
    """Single direct-device mutation seam used by rollback fault injection."""

    if target.device != source.device or target.dtype != source.dtype:
        raise ValueError("FSG4/B3-C direct device copy placement differs")
    target.copy_(source)


def _replace_host_packet(
    target: MutableMapping[str, object], source: Mapping[str, object]
) -> None:
    target.clear()
    target.update(source)


def commit_device_atomic_transaction_v1(
    transaction: DeviceAtomicTransactionV1,
    *,
    live_targets: Mapping[str, torch.Tensor],
    host_packet: MutableMapping[str, object],
) -> dict[str, object]:
    """Commit all device values and host state, rolling both back on any failure."""

    transaction.validate()
    validate_device_atomic_targets_v1(transaction.plan, live_targets)
    versions = dict(transaction.target_versions)
    if (
        any(
            _tensor_version(live_targets[path]) != version
            for path, version in versions.items()
        )
        or _host_pre_version(host_packet) != transaction.host_pre_version
    ):
        raise ValueError("FSG4/B3-C transaction version is stale")
    candidates = transaction.candidate_values
    backups = {
        path: live_targets[path].detach().clone()
        for path in transaction.plan.rollback_order
    }
    host_backup = dict(host_packet)
    try:
        with torch.no_grad():
            for path in transaction.plan.rollback_order:
                _copy_device_value(live_targets[path], candidates[path])
        _replace_host_packet(host_packet, transaction.host_candidate_map)
        if any(
            _tensor_version(live_targets[path]) != versions[path] + 1
            or not torch.equal(live_targets[path], candidates[path])
            for path in transaction.plan.rollback_order
        ) or _host_version(host_packet) != _host_version(
            transaction.host_candidate_map
        ):
            raise ValueError("FSG4/B3-C post-commit transaction differs")
    except Exception:
        with torch.no_grad():
            for path in transaction.plan.rollback_order:
                _copy_device_value(live_targets[path], backups[path])
        _replace_host_packet(host_packet, host_backup)
        raise
    post_versions = tuple(
        (path, _tensor_version(live_targets[path]))
        for path in transaction.plan.rollback_order
    )
    receipt: dict[str, object] = {
        "schema_version": FSG4_B3_DEVICE_ATOMIC_COMMIT_SCHEMA,
        "plan_hash": transaction.plan.stable_hash(),
        "transaction_version": transaction.transaction_version,
        "core_instance_hash": transaction.core_instance_hash,
        "pre_snapshot_hash": transaction.pre_snapshot_hash,
        "host_candidate_version": _host_version(transaction.host_candidate_map),
        "pre_target_versions": [list(row) for row in transaction.target_versions],
        "post_target_versions": [list(row) for row in post_versions],
        "committed_paths": list(transaction.plan.rollback_order),
        "committed_path_count": 12,
        "device_rollback_backup_count": 12,
        "atomic_live_and_host_commit": True,
        "candidate_device_resident": True,
        "candidate_d2h_copy_count": 0,
        "content_audit_pending": True,
        "provider_callback_count": 0,
        "fallback_dispatch_count": 0,
        "performance_claimed": False,
    }
    receipt["commit_hash"] = _canonical_hash(receipt)
    return receipt


def audit_device_atomic_transaction_v1(
    transaction: DeviceAtomicTransactionV1,
    *,
    receipt: Mapping[str, object],
    pre_snapshot: ProductionStateSnapshotV4,
    live_targets: Mapping[str, torch.Tensor],
) -> dict[str, object]:
    """Generate content digests only after the synchronized headline query."""

    transaction.validate()
    pre_snapshot.validate()
    validate_device_atomic_targets_v1(transaction.plan, live_targets)
    receipt_payload = dict(receipt)
    commit_hash = receipt_payload.pop("commit_hash", None)
    if (
        commit_hash != _canonical_hash(receipt_payload)
        or receipt.get("schema_version") != FSG4_B3_DEVICE_ATOMIC_COMMIT_SCHEMA
        or receipt.get("plan_hash") != transaction.plan.stable_hash()
        or receipt.get("transaction_version") != transaction.transaction_version
        or receipt.get("core_instance_hash") != transaction.core_instance_hash
        or receipt.get("pre_snapshot_hash") != transaction.pre_snapshot_hash
        or receipt.get("committed_paths") != list(transaction.plan.rollback_order)
        or receipt.get("committed_path_count") != 12
        or receipt.get("device_rollback_backup_count") != 12
        or receipt.get("atomic_live_and_host_commit") is not True
        or receipt.get("candidate_device_resident") is not True
        or receipt.get("candidate_d2h_copy_count") != 0
        or receipt.get("content_audit_pending") is not True
        or receipt.get("provider_callback_count") != 0
        or receipt.get("fallback_dispatch_count") != 0
        or receipt.get("performance_claimed") is not False
        or pre_snapshot.stable_hash() != transaction.pre_snapshot_hash
        or receipt.get("host_candidate_version")
        != _host_version(transaction.host_candidate_map)
    ):
        raise ValueError("FSG4/B3-C audit receipt binding differs")
    pre_map = pre_snapshot.tensor_map()
    candidate = transaction.candidate_values
    rows: list[dict[str, object]] = []
    for path in transaction.plan.rollback_order:
        candidate_sha = production_tensor_sha256(candidate[path])
        committed_sha = production_tensor_sha256(live_targets[path])
        if candidate_sha != committed_sha:
            raise ValueError("FSG4/B3-C post-query committed content differs")
        rows.append(
            {
                "semantic_path": path,
                "before_sha256": pre_map[path].content_sha256,
                "candidate_sha256": candidate_sha,
                "committed_sha256": committed_sha,
                "changed": pre_map[path].content_sha256 != candidate_sha,
            }
        )
    audit: dict[str, object] = {
        "schema_version": FSG4_B3_DEVICE_ATOMIC_COMMIT_SCHEMA,
        "plan_hash": transaction.plan.stable_hash(),
        "transaction_version": transaction.transaction_version,
        "commit_hash": commit_hash,
        "path_digests": rows,
        "committed_path_count": len(rows),
        "post_query_synchronized": True,
        "headline_timing_excluded": True,
        "content_audit_complete": True,
        "performance_claimed": False,
    }
    audit["audit_hash"] = _canonical_hash(audit)
    return audit


__all__ = [
    "audit_device_atomic_transaction_v1",
    "commit_device_atomic_transaction_v1",
    "compile_device_atomic_commit_plan_v1",
    "DeviceAtomicCandidateV1",
    "DeviceAtomicCommitPlanV1",
    "DeviceAtomicPathSpecV1",
    "DeviceAtomicTransactionV1",
    "FSG4_B3_DEVICE_ATOMIC_COMMIT_SCHEMA",
    "stage_device_atomic_transaction_v1",
    "validate_device_atomic_targets_v1",
]
