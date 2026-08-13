"""Atomic RVIR-v4 native terminal-state projection into production ownership."""

# pylint: disable=too-many-locals,too-many-statements,too-many-arguments
# pylint: disable=too-many-boolean-expressions,missing-function-docstring
# pylint: disable=too-many-instance-attributes,duplicate-code

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping, MutableMapping

import torch

from .native_alpha_beta_optimization_state import NativeAlphaBetaOptimizationState
from .rvir_v4_pre_state_initializer import ProductionReluTopologyV4
from .rvir_v4_production_state import (
    OwnedProductionTensorV4,
    ProductionStateSnapshotV4,
    ProductionTensorOwnership,
    ProductionTensorRole,
    production_tensor_sha256,
)

RVIR_V4_ATOMIC_COPY_OUT_SCHEMA = "boundflow.rvir-v4-atomic-copy-out/v1"
RVIR_V4_LIVE_ATOMIC_COPY_OUT_SCHEMA = "boundflow.rvir-v4-live-atomic-copy-out/v1"
COPY_OUT_ATOL = 2e-4
COPY_OUT_RTOL = 2e-4


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _encode(value: str) -> str:
    return value.replace("%", "%25").replace("/", "%2F")


@dataclass(frozen=True)
class ProductionCopyOutPathReceiptV4:
    """One staged mutable path and its independently validated result."""

    semantic_path: str
    role: ProductionTensorRole
    before_sha256: str
    candidate_sha256: str
    expected_sha256: str
    maximum_absolute_difference: float
    sign_exact: bool

    def validate(self) -> None:
        if (
            not self.semantic_path
            or self.role
            not in {ProductionTensorRole.ALPHA, ProductionTensorRole.BETA_VALUE}
            or any(
                len(value) != 64
                for value in (
                    self.before_sha256,
                    self.candidate_sha256,
                    self.expected_sha256,
                )
            )
            or not math.isfinite(self.maximum_absolute_difference)
            or self.maximum_absolute_difference > COPY_OUT_ATOL
            or self.sign_exact is not True
        ):
            raise ValueError("RVIR-v4 copy-out path receipt differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "semantic_path": self.semantic_path,
            "role": self.role.value,
            "before_sha256": self.before_sha256,
            "candidate_sha256": self.candidate_sha256,
            "expected_sha256": self.expected_sha256,
            "maximum_absolute_difference": self.maximum_absolute_difference,
            "sign_exact": self.sign_exact,
        }


@dataclass(frozen=True)
class ProductionAtomicCopyOutV4:
    """Validated immutable post snapshot plus a twelve-path atomic commit receipt."""

    pre_snapshot_hash: str
    candidate_snapshot: ProductionStateSnapshotV4
    expected_post_snapshot_hash: str
    path_receipts: tuple[ProductionCopyOutPathReceiptV4, ...]
    lower_maximum_absolute_difference: float
    lower_sign_exact: bool
    provider_callback_count: int = 0
    fallback_dispatch_count: int = 0
    schema_version: str = RVIR_V4_ATOMIC_COPY_OUT_SCHEMA

    def validate(self) -> None:
        self.candidate_snapshot.validate()
        paths = [receipt.semantic_path for receipt in self.path_receipts]
        if (
            self.schema_version != RVIR_V4_ATOMIC_COPY_OUT_SCHEMA
            or len(self.pre_snapshot_hash) != 64
            or len(self.expected_post_snapshot_hash) != 64
            or len(self.path_receipts) != 12
            or len(set(paths)) != 12
            or not math.isfinite(self.lower_maximum_absolute_difference)
            or self.lower_maximum_absolute_difference > COPY_OUT_ATOL
            or self.lower_sign_exact is not True
            or self.provider_callback_count != 0
            or self.fallback_dispatch_count != 0
        ):
            raise ValueError("RVIR-v4 atomic copy-out gate failed")
        for receipt in self.path_receipts:
            receipt.validate()

    def metadata(self) -> dict[str, object]:
        self.validate()
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "pre_snapshot_hash": self.pre_snapshot_hash,
            "candidate_snapshot_hash": self.candidate_snapshot.stable_hash(),
            "expected_post_snapshot_hash": self.expected_post_snapshot_hash,
            "path_receipts": [receipt.to_dict() for receipt in self.path_receipts],
            "lower_maximum_absolute_difference": self.lower_maximum_absolute_difference,
            "lower_sign_exact": self.lower_sign_exact,
            "provider_callback_count": self.provider_callback_count,
            "fallback_dispatch_count": self.fallback_dispatch_count,
            "performance_claimed": False,
        }
        payload["copy_out_hash"] = _canonical_hash(payload)
        return payload


@dataclass(frozen=True)
class ProductionLiveCopyOutPathReceiptV4:
    """One production candidate path without comparator truth dependency."""

    semantic_path: str
    role: ProductionTensorRole
    before_sha256: str
    candidate_sha256: str
    changed: bool

    def validate(self) -> None:
        if (
            not self.semantic_path
            or self.role
            not in {ProductionTensorRole.ALPHA, ProductionTensorRole.BETA_VALUE}
            or len(self.before_sha256) != 64
            or len(self.candidate_sha256) != 64
            or self.changed != (self.before_sha256 != self.candidate_sha256)
        ):
            raise ValueError("RVIR-v4 live copy-out path receipt differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "semantic_path": self.semantic_path,
            "role": self.role.value,
            "before_sha256": self.before_sha256,
            "candidate_sha256": self.candidate_sha256,
            "changed": self.changed,
        }


@dataclass(frozen=True)
class ProductionLiveAtomicCopyOutV4:
    """Truth-independent private candidate for live state and host packet commit."""

    pre_snapshot_hash: str
    candidate_snapshot: ProductionStateSnapshotV4
    path_receipts: tuple[ProductionLiveCopyOutPathReceiptV4, ...]
    terminal_lower_sha256: str
    host_packet_pre_hash: str
    host_packet_candidate_hash: str
    host_packet_candidate: tuple[tuple[str, object], ...]
    provider_callback_count: int = 0
    fallback_dispatch_count: int = 0
    schema_version: str = RVIR_V4_LIVE_ATOMIC_COPY_OUT_SCHEMA

    def validate(self) -> None:
        self.candidate_snapshot.validate()
        paths = [receipt.semantic_path for receipt in self.path_receipts]
        candidate_keys = [key for key, _value in self.host_packet_candidate]
        if (
            self.schema_version != RVIR_V4_LIVE_ATOMIC_COPY_OUT_SCHEMA
            or len(self.pre_snapshot_hash) != 64
            or len(self.terminal_lower_sha256) != 64
            or len(self.host_packet_pre_hash) != 64
            or len(self.host_packet_candidate_hash) != 64
            or len(self.path_receipts) != 12
            or len(set(paths)) != 12
            or len(candidate_keys) != len(set(candidate_keys))
            or set(candidate_keys) != {"depths", "history", "thresholds"}
            or _host_packet_hash(dict(self.host_packet_candidate))
            != self.host_packet_candidate_hash
            or self.provider_callback_count != 0
            or self.fallback_dispatch_count != 0
        ):
            raise ValueError("RVIR-v4 live atomic copy-out gate failed")
        for receipt in self.path_receipts:
            receipt.validate()

    def metadata(self) -> dict[str, object]:
        self.validate()
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "pre_snapshot_hash": self.pre_snapshot_hash,
            "candidate_snapshot_hash": self.candidate_snapshot.stable_hash(),
            "path_receipts": [receipt.to_dict() for receipt in self.path_receipts],
            "terminal_lower_sha256": self.terminal_lower_sha256,
            "host_packet_pre_hash": self.host_packet_pre_hash,
            "host_packet_candidate_hash": self.host_packet_candidate_hash,
            "host_packet_candidate_keys": sorted(
                key for key, _value in self.host_packet_candidate
            ),
            "provider_callback_count": self.provider_callback_count,
            "fallback_dispatch_count": self.fallback_dispatch_count,
            "performance_claimed": False,
        }
        payload["live_copy_out_hash"] = _canonical_hash(payload)
        return payload


def _replacement(
    source: OwnedProductionTensorV4, value: torch.Tensor
) -> OwnedProductionTensorV4:
    owned = value.detach().cpu().contiguous().clone()
    result = OwnedProductionTensorV4(
        semantic_path=source.semantic_path,
        role=source.role,
        axes=source.axes,
        value=owned,
        content_sha256=production_tensor_sha256(owned),
        source_device=source.source_device,
        ownership=source.ownership,
        alias_group=source.alias_group,
    )
    result.validate()
    return result


def _copy_value(target: torch.Tensor, source: torch.Tensor) -> None:
    """Single mutation seam used by the rollback fault-injection test."""

    target.copy_(source.to(target.device))


def _replace_host_packet(
    target: MutableMapping[str, object], source: Mapping[str, object]
) -> None:
    """Single host mutation seam used by transaction rollback tests."""

    target.clear()
    target.update(source)


def _host_packet_metadata(value: object) -> object:
    if torch.is_tensor(value):
        return {
            "shape": [int(dimension) for dimension in value.shape],
            "dtype": str(value.dtype),
            "content_sha256": production_tensor_sha256(value),
        }
    if isinstance(value, Mapping):
        return {
            str(key): _host_packet_metadata(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_host_packet_metadata(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"RVIR-v4 live host packet value differs: {type(value)}")


def _host_packet_hash(value: Mapping[str, object]) -> str:
    return _canonical_hash(_host_packet_metadata(value))


def _host_packet_pre_hash(value: Mapping[str, object]) -> str:
    retained = {}
    for key in ("depths", "history", "thresholds"):
        if key not in value:
            raise ValueError(f"RVIR-v4 live host packet misses {key}")
        retained[key] = value[key]
    return _canonical_hash(
        {
            "key_inventory": sorted(str(key) for key in value),
            "retained": _host_packet_metadata(retained),
        }
    )


def _project_alpha(
    source: OwnedProductionTensorV4,
    dense: torch.Tensor,
    tensor_map: Mapping[str, OwnedProductionTensorV4],
    activation: str,
) -> torch.Tensor:
    prefix = f"alpha_layout/{_encode(activation)}"
    indices: list[torch.Tensor] = []
    ordinal = 0
    while f"{prefix}/feature_index/{ordinal}" in tensor_map:
        indices.append(tensor_map[f"{prefix}/feature_index/{ordinal}"].value)
        ordinal += 1
    projected = source.value.clone()
    compressed = (
        dense[(slice(None),) + tuple(indices)]
        if indices
        else dense.reshape_as(source.value[0, 0])
    )
    if compressed.shape != source.value[0, 0].shape:
        raise ValueError("RVIR-v4 copy-out alpha projection shape differs")
    projected[0, 0].copy_(compressed)
    return projected


def _project_beta(
    dense: torch.Tensor,
    location: OwnedProductionTensorV4,
    expected_shape: torch.Size,
) -> torch.Tensor:
    flat = dense.reshape(int(dense.shape[0]), -1)
    projected = torch.stack(
        [flat[domain, location.value[domain]] for domain in range(int(dense.shape[0]))]
    )
    if projected.shape != expected_shape:
        raise ValueError("RVIR-v4 copy-out beta projection shape differs")
    return projected


def stage_rvir_v4_atomic_copy_out(
    *,
    pre: ProductionStateSnapshotV4,
    terminal_state: NativeAlphaBetaOptimizationState,
    topology: tuple[ProductionReluTopologyV4, ...],
    expected_post: ProductionStateSnapshotV4,
    terminal_lower: torch.Tensor,
    expected_lower: torch.Tensor,
    candidate_snapshot_id: str,
) -> ProductionAtomicCopyOutV4:
    """Stage all paths privately and validate the complete candidate before commit."""

    pre.validate()
    terminal_state.validate()
    expected_post.validate()
    if not candidate_snapshot_id or len(topology) != 6:
        raise ValueError("RVIR-v4 copy-out candidate identity differs")
    pre_map = pre.tensor_map()
    expected_map = expected_post.tensor_map()
    if set(pre_map) != set(expected_map):
        raise ValueError("RVIR-v4 copy-out pre/post paths differ")
    replacements: dict[str, OwnedProductionTensorV4] = {}
    receipts: list[ProductionCopyOutPathReceiptV4] = []
    for link in topology:
        native = link.native_preactivation
        alpha_path = (
            f"alpha/{_encode(link.provider_activation)}/"
            f"{_encode(link.provider_start_node)}"
        )
        beta_prefix = f"beta/{_encode(link.provider_preactivation)}/0"
        beta_path = f"{beta_prefix}/value"
        alpha_source = pre_map[alpha_path]
        beta_source = pre_map[beta_path]
        alpha_value = _project_alpha(
            alpha_source,
            terminal_state.alphas[native],
            pre_map,
            link.provider_activation,
        )
        beta_value = _project_beta(
            terminal_state.betas[native],
            pre_map[f"{beta_prefix}/location"],
            beta_source.value.shape,
        )
        for source, value in ((alpha_source, alpha_value), (beta_source, beta_value)):
            expected = expected_map[source.semantic_path]
            if (
                source.ownership != ProductionTensorOwnership.MUTABLE_COPY_OUT
                or source.role != expected.role
                or value.shape != expected.value.shape
                or value.dtype != expected.value.dtype
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError("RVIR-v4 copy-out mutable schema differs")
            difference = (
                float((value - expected.value).abs().max().item())
                if value.numel()
                else 0.0
            )
            close = torch.allclose(
                value, expected.value, atol=COPY_OUT_ATOL, rtol=COPY_OUT_RTOL
            )
            sign_exact = torch.equal(torch.sign(value), torch.sign(expected.value))
            if not close:
                raise ValueError("RVIR-v4 copy-out mutable numeric parity differs")
            replacement = _replacement(source, value)
            replacements[source.semantic_path] = replacement
            receipts.append(
                ProductionCopyOutPathReceiptV4(
                    semantic_path=source.semantic_path,
                    role=source.role,
                    before_sha256=source.content_sha256,
                    candidate_sha256=replacement.content_sha256,
                    expected_sha256=expected.content_sha256,
                    maximum_absolute_difference=difference,
                    sign_exact=sign_exact,
                )
            )
    mutable_paths = {
        path
        for path, tensor in pre_map.items()
        if tensor.ownership == ProductionTensorOwnership.MUTABLE_COPY_OUT
    }
    if set(replacements) != mutable_paths or len(replacements) != 12:
        raise ValueError("RVIR-v4 copy-out mutable ownership differs")
    candidate = ProductionStateSnapshotV4(
        snapshot_id=candidate_snapshot_id,
        tensors=tuple(
            replacements.get(tensor.semantic_path, tensor)
            for tensor in sorted(pre.tensors, key=lambda item: item.semantic_path)
        ),
        history=pre.history,
        optimizer_policy=pre.optimizer_policy,
    )
    candidate.validate()
    for path in set(pre_map) - mutable_paths:
        if candidate.tensor_map()[path].content_sha256 != pre_map[path].content_sha256:
            raise ValueError("RVIR-v4 copy-out read-only state drift")
    lower_diff = float((terminal_lower - expected_lower).abs().max().item())
    if not torch.allclose(
        terminal_lower, expected_lower, atol=COPY_OUT_ATOL, rtol=COPY_OUT_RTOL
    ):
        raise ValueError("RVIR-v4 copy-out final lower parity differs")
    result = ProductionAtomicCopyOutV4(
        pre_snapshot_hash=pre.stable_hash(),
        candidate_snapshot=candidate,
        expected_post_snapshot_hash=expected_post.stable_hash(),
        path_receipts=tuple(sorted(receipts, key=lambda item: item.semantic_path)),
        lower_maximum_absolute_difference=lower_diff,
        lower_sign_exact=torch.equal(
            torch.sign(terminal_lower), torch.sign(expected_lower)
        ),
    )
    result.validate()
    return result


def stage_rvir_v4_live_atomic_copy_out(
    *,
    pre: ProductionStateSnapshotV4,
    terminal_state: NativeAlphaBetaOptimizationState,
    topology: tuple[ProductionReluTopologyV4, ...],
    terminal_lower: torch.Tensor,
    host_packet: Mapping[str, object],
    host_packet_candidate: Mapping[str, object],
    candidate_snapshot_id: str,
) -> ProductionLiveAtomicCopyOutV4:
    """Stage live mutable state and host pruning without comparator truth."""

    pre.validate()
    terminal_state.validate()
    if (
        not candidate_snapshot_id
        or len(topology) != 6
        or tuple(terminal_lower.shape) != (6, 1)
        or not torch.is_floating_point(terminal_lower)
        or not bool(torch.isfinite(terminal_lower).all())
        or set(host_packet_candidate) != {"depths", "history", "thresholds"}
    ):
        raise ValueError("RVIR-v4 live copy-out candidate identity differs")
    pre_map = pre.tensor_map()
    replacements: dict[str, OwnedProductionTensorV4] = {}
    receipts: list[ProductionLiveCopyOutPathReceiptV4] = []
    for link in topology:
        link.validate()
        native = link.native_preactivation
        alpha_path = (
            f"alpha/{_encode(link.provider_activation)}/"
            f"{_encode(link.provider_start_node)}"
        )
        beta_prefix = f"beta/{_encode(link.provider_preactivation)}/0"
        beta_path = f"{beta_prefix}/value"
        alpha_source = pre_map[alpha_path]
        beta_source = pre_map[beta_path]
        alpha_value = _project_alpha(
            alpha_source,
            terminal_state.alphas[native],
            pre_map,
            link.provider_activation,
        )
        beta_value = _project_beta(
            terminal_state.betas[native],
            pre_map[f"{beta_prefix}/location"],
            beta_source.value.shape,
        )
        for source, value in ((alpha_source, alpha_value), (beta_source, beta_value)):
            if (
                source.ownership != ProductionTensorOwnership.MUTABLE_COPY_OUT
                or value.shape != source.value.shape
                or value.dtype != source.value.dtype
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError("RVIR-v4 live copy-out mutable schema differs")
            replacement = _replacement(source, value)
            replacements[source.semantic_path] = replacement
            receipts.append(
                ProductionLiveCopyOutPathReceiptV4(
                    semantic_path=source.semantic_path,
                    role=source.role,
                    before_sha256=source.content_sha256,
                    candidate_sha256=replacement.content_sha256,
                    changed=source.content_sha256 != replacement.content_sha256,
                )
            )
    mutable_paths = {
        path
        for path, tensor in pre_map.items()
        if tensor.ownership == ProductionTensorOwnership.MUTABLE_COPY_OUT
    }
    if set(replacements) != mutable_paths or len(replacements) != 12:
        raise ValueError("RVIR-v4 live copy-out mutable ownership differs")
    candidate = ProductionStateSnapshotV4(
        snapshot_id=candidate_snapshot_id,
        tensors=tuple(
            replacements.get(tensor.semantic_path, tensor)
            for tensor in sorted(pre.tensors, key=lambda item: item.semantic_path)
        ),
        history=pre.history,
        optimizer_policy=pre.optimizer_policy,
    )
    candidate.validate()
    for path in set(pre_map) - mutable_paths:
        if candidate.tensor_map()[path].content_sha256 != pre_map[path].content_sha256:
            raise ValueError("RVIR-v4 live copy-out read-only state drift")
    staged = ProductionLiveAtomicCopyOutV4(
        pre_snapshot_hash=pre.stable_hash(),
        candidate_snapshot=candidate,
        path_receipts=tuple(sorted(receipts, key=lambda item: item.semantic_path)),
        terminal_lower_sha256=production_tensor_sha256(terminal_lower),
        host_packet_pre_hash=_host_packet_pre_hash(host_packet),
        host_packet_candidate_hash=_host_packet_hash(host_packet_candidate),
        host_packet_candidate=tuple(
            sorted(host_packet_candidate.items(), key=lambda item: item[0])
        ),
    )
    staged.validate()
    return staged


def commit_rvir_v4_live_atomic_copy_out(
    staged: ProductionLiveAtomicCopyOutV4,
    *,
    pre: ProductionStateSnapshotV4,
    live_targets: Mapping[str, torch.Tensor],
    host_packet: MutableMapping[str, object],
) -> dict[str, object]:
    """Atomically commit twelve live tensors and the pruned host packet."""

    staged.validate()
    pre.validate()
    pre_map = pre.tensor_map()
    candidate_map = staged.candidate_snapshot.tensor_map()
    paths = tuple(receipt.semantic_path for receipt in staged.path_receipts)
    if (
        staged.pre_snapshot_hash != pre.stable_hash()
        or set(live_targets) != set(paths)
        or _host_packet_pre_hash(host_packet) != staged.host_packet_pre_hash
    ):
        raise ValueError("RVIR-v4 live copy-out target inventory differs")
    for path in paths:
        live = live_targets[path]
        source = pre_map[path].value
        candidate = candidate_map[path].value
        if (
            not torch.is_tensor(live)
            or live.shape != source.shape
            or live.dtype != source.dtype
            or production_tensor_sha256(live) != production_tensor_sha256(source)
            or candidate.shape != live.shape
            or candidate.dtype != live.dtype
        ):
            raise ValueError("RVIR-v4 live copy-out tensor target differs")
    tensor_backups = {path: live_targets[path].detach().clone() for path in paths}
    host_backup = dict(host_packet)
    candidate_host = dict(staged.host_packet_candidate)
    try:
        with torch.no_grad():
            for path in paths:
                _copy_value(live_targets[path], candidate_map[path].value)
        _replace_host_packet(host_packet, candidate_host)
        if (
            any(
                production_tensor_sha256(live_targets[path])
                != candidate_map[path].content_sha256
                for path in paths
            )
            or _host_packet_hash(host_packet) != staged.host_packet_candidate_hash
        ):
            raise ValueError("RVIR-v4 live copy-out post-commit verification differs")
    except Exception:
        with torch.no_grad():
            for path in paths:
                _copy_value(live_targets[path], tensor_backups[path])
        _replace_host_packet(host_packet, host_backup)
        raise
    receipt: dict[str, object] = {
        "live_copy_out_hash": staged.metadata()["live_copy_out_hash"],
        "committed_path_count": len(paths),
        "changed_path_count": sum(receipt.changed for receipt in staged.path_receipts),
        "committed_paths": list(paths),
        "host_packet_candidate_hash": staged.host_packet_candidate_hash,
        "atomic_live_and_host_commit": True,
        "provider_callback_count": 0,
        "fallback_dispatch_count": 0,
        "performance_claimed": False,
    }
    receipt["commit_hash"] = _canonical_hash(receipt)
    return receipt


def commit_rvir_v4_atomic_copy_out(
    staged: ProductionAtomicCopyOutV4,
    *,
    pre: ProductionStateSnapshotV4,
    live_targets: Mapping[str, torch.Tensor],
) -> dict[str, object]:
    """Commit twelve already-validated values, rolling back any runtime copy failure."""

    staged.validate()
    pre.validate()
    pre_map = pre.tensor_map()
    candidate_map = staged.candidate_snapshot.tensor_map()
    paths = tuple(receipt.semantic_path for receipt in staged.path_receipts)
    if staged.pre_snapshot_hash != pre.stable_hash() or set(live_targets) != set(paths):
        raise ValueError("RVIR-v4 copy-out live target inventory differs")
    for path in paths:
        live = live_targets[path]
        source = pre_map[path].value
        candidate = candidate_map[path].value
        if (
            not torch.is_tensor(live)
            or live.shape != source.shape
            or live.dtype != source.dtype
            or production_tensor_sha256(live) != production_tensor_sha256(source)
            or candidate.shape != live.shape
            or candidate.dtype != live.dtype
        ):
            raise ValueError("RVIR-v4 copy-out live target differs")
    backups = {path: live_targets[path].detach().clone() for path in paths}
    try:
        with torch.no_grad():
            for path in paths:
                _copy_value(live_targets[path], candidate_map[path].value)
    except Exception:
        with torch.no_grad():
            for path in paths:
                _copy_value(live_targets[path], backups[path])
        raise
    if any(
        production_tensor_sha256(live_targets[path])
        != candidate_map[path].content_sha256
        for path in paths
    ):
        with torch.no_grad():
            for path in paths:
                _copy_value(live_targets[path], backups[path])
        raise ValueError("RVIR-v4 copy-out post-commit verification differs")
    receipt: dict[str, object] = {
        "copy_out_hash": staged.metadata()["copy_out_hash"],
        "committed_path_count": len(paths),
        "committed_paths": list(paths),
        "atomic_commit": True,
        "provider_callback_count": 0,
        "fallback_dispatch_count": 0,
        "performance_claimed": False,
    }
    receipt["commit_hash"] = _canonical_hash(receipt)
    return receipt


__all__ = [
    "commit_rvir_v4_atomic_copy_out",
    "commit_rvir_v4_live_atomic_copy_out",
    "COPY_OUT_ATOL",
    "COPY_OUT_RTOL",
    "ProductionAtomicCopyOutV4",
    "ProductionCopyOutPathReceiptV4",
    "ProductionLiveAtomicCopyOutV4",
    "ProductionLiveCopyOutPathReceiptV4",
    "stage_rvir_v4_atomic_copy_out",
    "stage_rvir_v4_live_atomic_copy_out",
]
