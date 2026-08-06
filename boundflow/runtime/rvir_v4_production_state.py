"""Typed production verifier state ownership for RVIR-v4."""

# pylint: disable=too-many-lines,too-many-arguments,too-many-locals
# pylint: disable=missing-function-docstring,too-many-branches
# pylint: disable=too-many-instance-attributes,too-many-boolean-expressions

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from collections.abc import Callable
from typing import Any, cast, Iterable, Mapping, Sequence

import torch

RVIR_V4_STATE_SCHEMA_VERSION = "boundflow.rvir-production-state/v4"


class ProductionTensorRole(str, Enum):
    """Semantic role of one owned production tensor."""

    INPUT_LOWER = "input_lower"
    INPUT_UPPER = "input_upper"
    LINEAR_SPEC = "linear_spec"
    INTERMEDIATE_LOWER = "intermediate_lower"
    INTERMEDIATE_UPPER = "intermediate_upper"
    REFERENCE_LOWER = "reference_lower"
    REFERENCE_UPPER = "reference_upper"
    ALPHA = "alpha"
    BETA_VALUE = "beta_value"
    BETA_LOCATION = "beta_location"
    BETA_SIGN = "beta_sign"
    BETA_BIAS = "beta_bias"
    DECISION_THRESHOLD = "decision_threshold"
    RESULT_LOWER = "result_lower"
    RESULT_UPPER = "result_upper"
    RESULT_LA = "result_lA"


class ProductionTensorOwnership(str, Enum):
    """Copy and mutation ownership at the solver-core boundary."""

    READ_ONLY = "read_only"
    COPY_IN = "copy_in"
    MUTABLE_COPY_OUT = "mutable_copy_out"


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _encode_component(value: str) -> str:
    return value.replace("%", "%25").replace("/", "%2F")


def _decode_component(value: str) -> str:
    return value.replace("%2F", "/").replace("%25", "%")


def production_tensor_sha256(value: torch.Tensor) -> str:
    """Hash tensor type, shape, and device-independent contiguous content."""

    if not torch.is_tensor(value):
        raise TypeError("RVIR-v4 production value must be a tensor")
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class OwnedProductionTensorV4:
    """One digest-bound tensor with semantic axes and alias ownership."""

    semantic_path: str
    role: ProductionTensorRole
    axes: tuple[str, ...]
    value: torch.Tensor
    content_sha256: str
    source_device: str
    ownership: ProductionTensorOwnership
    alias_group: str

    @classmethod
    def own(
        cls,
        *,
        semantic_path: str,
        role: ProductionTensorRole,
        axes: Sequence[str],
        value: torch.Tensor,
        ownership: ProductionTensorOwnership,
        alias_group: str,
    ) -> "OwnedProductionTensorV4":
        if not torch.is_tensor(value):
            raise TypeError("RVIR-v4 owned production value must be a tensor")
        owned = value.detach().cpu().contiguous().clone()
        result = cls(
            semantic_path=semantic_path,
            role=role,
            axes=tuple(axes),
            value=owned,
            content_sha256=production_tensor_sha256(owned),
            source_device=str(value.device),
            ownership=ownership,
            alias_group=alias_group,
        )
        result.validate()
        return result

    def validate(self) -> None:
        if (
            not self.semantic_path
            or not self.alias_group
            or not self.source_device
            or self.semantic_path.startswith("/")
            or "//" in self.semantic_path
        ):
            raise ValueError("RVIR-v4 production tensor identity differs")
        if len(self.axes) != self.value.ndim or any(not axis for axis in self.axes):
            raise ValueError("RVIR-v4 production tensor axes differ")
        if self.content_sha256 != production_tensor_sha256(self.value):
            raise ValueError("RVIR-v4 production tensor content differs")
        if self.value.is_floating_point() and not bool(
            torch.isfinite(self.value).all()
        ):
            raise ValueError("RVIR-v4 production tensor must be finite")

    def metadata(self) -> dict[str, object]:
        self.validate()
        return {
            "semantic_path": self.semantic_path,
            "role": self.role.value,
            "axes": list(self.axes),
            "shape": list(self.value.shape),
            "dtype": str(self.value.dtype),
            "source_device": self.source_device,
            "ownership": self.ownership.value,
            "alias_group": self.alias_group,
            "content_sha256": self.content_sha256,
        }


@dataclass(frozen=True)
class ProductionSplitHistoryEntryV4:
    """One layer's split history for one domain ordinal."""

    domain_ordinal: int
    layer_name: str
    locations: tuple[int, ...]
    coefficients: tuple[float, ...]
    biases: tuple[float, ...] | None = None
    scores: tuple[float, ...] | None = None
    depths: tuple[float, ...] | None = None

    def validate(self) -> None:
        if self.domain_ordinal < 0 or not self.layer_name:
            raise ValueError("RVIR-v4 split history identity differs")
        optional = (self.biases, self.scores, self.depths)
        if len(self.locations) != len(self.coefficients) or any(
            values is not None and len(values) != len(self.locations)
            for values in optional
        ):
            raise ValueError("RVIR-v4 split history lengths differ")
        if any(location < 0 for location in self.locations):
            raise ValueError("RVIR-v4 split history location is negative")
        values = self.coefficients + tuple(
            value for items in optional if items is not None for value in items
        )
        if not all(torch.isfinite(torch.tensor(value)).item() for value in values):
            raise ValueError("RVIR-v4 split history is non-finite")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "domain_ordinal": self.domain_ordinal,
            "layer_name": self.layer_name,
            "locations": list(self.locations),
            "coefficients": list(self.coefficients),
            "biases": None if self.biases is None else list(self.biases),
            "scores": None if self.scores is None else list(self.scores),
            "depths": None if self.depths is None else list(self.depths),
        }


@dataclass(frozen=True)
class ProductionOptimizerPolicyV4:
    """Optimizer and solver controls that own one production mutation."""

    iteration: int
    alpha_learning_rate: float
    beta_learning_rate: float
    bound_lower: bool
    bound_upper: bool
    fix_intermediate_bounds: bool
    deterministic: bool
    stop_criterion_id: str

    def validate(self) -> None:
        if (
            self.iteration < 0
            or self.alpha_learning_rate <= 0.0
            or self.beta_learning_rate <= 0.0
            or not (self.bound_lower or self.bound_upper)
            or not self.stop_criterion_id
            or not torch.isfinite(torch.tensor(self.alpha_learning_rate)).item()
            or not torch.isfinite(torch.tensor(self.beta_learning_rate)).item()
        ):
            raise ValueError("RVIR-v4 optimizer policy differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "iteration": self.iteration,
            "alpha_learning_rate": self.alpha_learning_rate,
            "beta_learning_rate": self.beta_learning_rate,
            "bound_lower": self.bound_lower,
            "bound_upper": self.bound_upper,
            "fix_intermediate_bounds": self.fix_intermediate_bounds,
            "deterministic": self.deterministic,
            "stop_criterion_id": self.stop_criterion_id,
        }


@dataclass(frozen=True)
class ProductionStateSnapshotV4:
    """Lossless tensor/history/policy snapshot at one production boundary."""

    snapshot_id: str
    tensors: tuple[OwnedProductionTensorV4, ...]
    history: tuple[ProductionSplitHistoryEntryV4, ...]
    optimizer_policy: ProductionOptimizerPolicyV4
    schema_version: str = RVIR_V4_STATE_SCHEMA_VERSION

    def tensor_map(self) -> dict[str, OwnedProductionTensorV4]:
        return {tensor.semantic_path: tensor for tensor in self.tensors}

    def validate(self) -> None:
        if self.schema_version != RVIR_V4_STATE_SCHEMA_VERSION or not self.snapshot_id:
            raise ValueError("RVIR-v4 production snapshot identity differs")
        self.optimizer_policy.validate()
        if not self.tensors:
            raise ValueError("RVIR-v4 production snapshot tensors are empty")
        for tensor in self.tensors:
            tensor.validate()
        paths = [tensor.semantic_path for tensor in self.tensors]
        if len(set(paths)) != len(paths):
            raise ValueError("RVIR-v4 production tensor paths repeat")
        for entry in self.history:
            entry.validate()
        history_keys = [
            (entry.domain_ordinal, entry.layer_name) for entry in self.history
        ]
        if len(set(history_keys)) != len(history_keys):
            raise ValueError("RVIR-v4 production history keys repeat")
        _validate_beta_tensor_groups(self.tensors)
        validate_beta_history_consistency(self.tensors, self.history)

    def metadata(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "snapshot_id": self.snapshot_id,
            "tensors": [tensor.metadata() for tensor in self.tensors],
            "history": [entry.to_dict() for entry in self.history],
            "optimizer_policy": self.optimizer_policy.to_dict(),
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.metadata())


@dataclass(frozen=True)
class ProductionStateMutationV4:
    """Pre/post digest closure for one mutable semantic path."""

    semantic_path: str
    before_sha256: str
    after_sha256: str
    changed: bool

    def validate(self) -> None:
        if (
            not self.semantic_path
            or len(self.before_sha256) != 64
            or len(self.after_sha256) != 64
            or self.changed != (self.before_sha256 != self.after_sha256)
        ):
            raise ValueError("RVIR-v4 state mutation receipt differs")


def _tensor_axes(prefix: Sequence[str], rank: int) -> tuple[str, ...]:
    if rank < len(prefix):
        raise ValueError("RVIR-v4 tensor rank is smaller than semantic axes")
    return tuple(prefix) + tuple(
        f"feature_axis_{ordinal}" for ordinal in range(rank - len(prefix))
    )


class ProductionStateBuilderV4:
    """Build one snapshot while preserving provider tensor alias groups."""

    def __init__(self) -> None:
        self._aliases: dict[int, str] = {}
        self._tensors: list[OwnedProductionTensorV4] = []

    def add(
        self,
        *,
        path: str,
        role: ProductionTensorRole,
        axes: Sequence[str],
        value: torch.Tensor,
        ownership: ProductionTensorOwnership,
    ) -> None:
        identity = id(value)
        alias = self._aliases.setdefault(identity, f"alias:{len(self._aliases):06d}")
        self._tensors.append(
            OwnedProductionTensorV4.own(
                semantic_path=path,
                role=role,
                axes=axes,
                value=value,
                ownership=ownership,
                alias_group=alias,
            )
        )

    def finish(self) -> tuple[OwnedProductionTensorV4, ...]:
        return tuple(sorted(self._tensors, key=lambda tensor: tensor.semantic_path))


def _mapping_items(value: object) -> Iterable[tuple[str, object]]:
    if isinstance(value, Mapping):
        return (
            (str(key), item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, (tuple, list)):
        return ((str(index), item) for index, item in enumerate(value))
    raise TypeError("RVIR-v4 production nested state must be a mapping or sequence")


def capture_alpha_state_v4(
    alpha_by_layer: Mapping[str, Mapping[str, torch.Tensor]],
    builder: ProductionStateBuilderV4,
    *,
    ownership: ProductionTensorOwnership = ProductionTensorOwnership.MUTABLE_COPY_OUT,
) -> None:
    """Capture activation/start-node keyed α with explicit axes."""

    for layer_name, start_nodes in sorted(alpha_by_layer.items()):
        if not isinstance(start_nodes, Mapping):
            raise TypeError("RVIR-v4 alpha layer state must be a mapping")
        for start_name, value in sorted(start_nodes.items()):
            if not torch.is_tensor(value):
                raise TypeError("RVIR-v4 alpha state must contain tensors")
            builder.add(
                path=(
                    f"alpha/{_encode_component(layer_name)}/"
                    f"{_encode_component(start_name)}"
                ),
                role=ProductionTensorRole.ALPHA,
                axes=_tensor_axes(
                    ("alpha_polarity", "start_spec", "domain"), value.ndim
                ),
                value=value,
                ownership=ownership,
            )


def capture_sparse_beta_state_v4(
    sparse_betas_by_layer: Mapping[str, object],
    builder: ProductionStateBuilderV4,
    *,
    ownership: ProductionTensorOwnership = ProductionTensorOwnership.MUTABLE_COPY_OUT,
) -> None:
    """Capture SparseBeta.val/loc/sign/bias using plural provider ownership."""

    for layer_name, collection in sorted(sparse_betas_by_layer.items()):
        for start_name, sparse_beta in _mapping_items(collection):
            if sparse_beta is None:
                continue
            fields = (
                ("value", "val", ProductionTensorRole.BETA_VALUE),
                ("location", "loc", ProductionTensorRole.BETA_LOCATION),
                ("sign", "sign", ProductionTensorRole.BETA_SIGN),
                ("bias", "bias", ProductionTensorRole.BETA_BIAS),
            )
            for field_name, attribute, role in fields:
                value = getattr(sparse_beta, attribute, None)
                if value is None and attribute == "bias":
                    continue
                if not torch.is_tensor(value):
                    raise TypeError(
                        f"RVIR-v4 SparseBeta {layer_name}/{start_name}/{attribute} differs"
                    )
                tensor = cast(torch.Tensor, value)
                builder.add(
                    path=(
                        f"beta/{_encode_component(layer_name)}/"
                        f"{_encode_component(start_name)}/{field_name}"
                    ),
                    role=role,
                    axes=_tensor_axes(("domain", "history_slot"), tensor.ndim),
                    value=tensor,
                    ownership=(
                        ownership
                        if role == ProductionTensorRole.BETA_VALUE
                        else ProductionTensorOwnership.COPY_IN
                    ),
                )


def capture_module_alpha_beta_state_v4(
    nodes: Iterable[object],
    *,
    require_beta: bool,
) -> tuple[OwnedProductionTensorV4, ...]:
    """Capture provider nodes and fail if plural SparseBeta ownership is omitted."""

    builder = ProductionStateBuilderV4()
    alpha: dict[str, Mapping[str, torch.Tensor]] = {}
    sparse_betas: dict[str, object] = {}
    for node in nodes:
        layer_name = str(getattr(node, "name", type(node).__name__))
        raw_alpha = getattr(node, "alpha", None)
        if isinstance(raw_alpha, Mapping) and raw_alpha:
            alpha[layer_name] = raw_alpha
        if hasattr(node, "sparse_betas"):
            raw_beta = getattr(node, "sparse_betas")
            if raw_beta is not None:
                sparse_betas[layer_name] = raw_beta
    if alpha:
        capture_alpha_state_v4(alpha, builder)
    if sparse_betas:
        capture_sparse_beta_state_v4(sparse_betas, builder)
    result = builder.finish()
    if require_beta and not any(
        tensor.role == ProductionTensorRole.BETA_VALUE for tensor in result
    ):
        raise ValueError(
            "RVIR-v4 beta phase omits provider node.sparse_betas plural state"
        )
    return result


def capture_split_history_v4(
    history: Sequence[Mapping[str, Sequence[object]]],
) -> tuple[ProductionSplitHistoryEntryV4, ...]:
    def values(
        raw_value: object, converter: Callable[[Any], object]
    ) -> tuple[object, ...]:
        if isinstance(raw_value, (str, bytes, Mapping)) or not isinstance(
            raw_value, Iterable
        ):
            raise ValueError("RVIR-v4 provider split history values differ")
        return tuple(converter(item) for item in raw_value)

    entries: list[ProductionSplitHistoryEntryV4] = []
    for domain_ordinal, domain in enumerate(history):
        for layer_name, raw in sorted(domain.items()):
            if not isinstance(raw, Sequence) or len(raw) < 2:
                raise ValueError("RVIR-v4 provider split history differs")
            raw_locations = raw[0]
            raw_coefficients = raw[1]
            locations = cast(tuple[int, ...], values(raw_locations, int))
            coefficients = cast(
                tuple[float, ...],
                values(raw_coefficients, float),
            )
            raw_biases = None if len(raw) < 3 else raw[2]
            raw_scores = None if len(raw) < 4 else raw[3]
            raw_depths = None if len(raw) < 5 else raw[4]
            biases = (
                None
                if raw_biases is None
                else cast(
                    tuple[float, ...],
                    values(raw_biases, float),
                )
            )
            scores = (
                None
                if raw_scores is None
                else cast(tuple[float, ...], values(raw_scores, float))
            )
            depths = (
                None
                if raw_depths is None
                else cast(tuple[float, ...], values(raw_depths, float))
            )
            entry = ProductionSplitHistoryEntryV4(
                domain_ordinal=domain_ordinal,
                layer_name=str(layer_name),
                locations=locations,
                coefficients=coefficients,
                biases=biases,
                scores=scores,
                depths=depths,
            )
            entry.validate()
            entries.append(entry)
    return tuple(entries)


def _beta_groups(
    tensors: Sequence[OwnedProductionTensorV4],
) -> dict[str, dict[ProductionTensorRole, OwnedProductionTensorV4]]:
    groups: dict[str, dict[ProductionTensorRole, OwnedProductionTensorV4]] = {}
    for tensor in tensors:
        if not tensor.semantic_path.startswith("beta/"):
            continue
        prefix, _field = tensor.semantic_path.rsplit("/", 1)
        groups.setdefault(prefix, {})[tensor.role] = tensor
    return groups


def _validate_beta_tensor_groups(
    tensors: Sequence[OwnedProductionTensorV4],
) -> None:
    for prefix, group in _beta_groups(tensors).items():
        required = {
            ProductionTensorRole.BETA_VALUE,
            ProductionTensorRole.BETA_LOCATION,
            ProductionTensorRole.BETA_SIGN,
        }
        if not required <= set(group):
            raise ValueError(f"RVIR-v4 SparseBeta fields are incomplete: {prefix}")
        value = group[ProductionTensorRole.BETA_VALUE].value
        location = group[ProductionTensorRole.BETA_LOCATION].value
        sign = group[ProductionTensorRole.BETA_SIGN].value
        bias_item = group.get(ProductionTensorRole.BETA_BIAS)
        if (
            value.shape != location.shape
            or value.shape != sign.shape
            or (bias_item is not None and value.shape != bias_item.value.shape)
            or location.dtype != torch.long
            or not torch.is_floating_point(value)
            or not torch.is_floating_point(sign)
            or bool((value < 0).any())
        ):
            raise ValueError(f"RVIR-v4 SparseBeta tensor schema differs: {prefix}")


def validate_beta_history_consistency(
    tensors: Sequence[OwnedProductionTensorV4],
    history: Sequence[ProductionSplitHistoryEntryV4],
) -> None:
    """Require each history prefix to equal SparseBeta location/sign[/bias]."""

    entries = {(item.domain_ordinal, item.layer_name): item for item in history}
    groups = _beta_groups(tensors)
    layers_with_beta = {_decode_component(prefix.split("/", 2)[1]) for prefix in groups}
    for (domain_ordinal, layer_name), entry in entries.items():
        if layer_name not in layers_with_beta:
            raise ValueError(
                f"RVIR-v4 history layer lacks SparseBeta ownership: {layer_name}"
            )
        layer_groups = [
            group
            for prefix, group in groups.items()
            if _decode_component(prefix.split("/", 2)[1]) == layer_name
        ]
        matched = False
        candidates: list[str] = []
        for group in layer_groups:
            location = group[ProductionTensorRole.BETA_LOCATION].value
            sign = group[ProductionTensorRole.BETA_SIGN].value
            if (
                domain_ordinal >= location.shape[0]
                or len(entry.locations) > location.shape[1]
            ):
                candidates.append(
                    f"shape={tuple(location.shape)},history={len(entry.locations)}"
                )
                continue
            count = len(entry.locations)
            observed_locations = tuple(
                int(item) for item in location[domain_ordinal, :count].tolist()
            )
            observed_signs = tuple(
                float(item) for item in sign[domain_ordinal, :count].tolist()
            )
            candidates.append(f"locations={observed_locations},signs={observed_signs}")
            if (
                observed_locations != entry.locations
                or observed_signs != entry.coefficients
            ):
                continue
            bias_item = group.get(ProductionTensorRole.BETA_BIAS)
            if entry.biases:
                if bias_item is None:
                    if any(value != 0.0 for value in entry.biases):
                        continue
                else:
                    observed_biases = tuple(
                        float(item)
                        for item in bias_item.value[domain_ordinal, :count].tolist()
                    )
                    if observed_biases != entry.biases:
                        continue
            matched = True
            break
        if not matched:
            raise ValueError(
                "RVIR-v4 SparseBeta/history content differs: "
                f"{domain_ordinal}/{layer_name}; expected locations={entry.locations},"
                f"signs={entry.coefficients}; candidates={candidates}"
            )


def diff_production_state_v4(
    before: ProductionStateSnapshotV4,
    after: ProductionStateSnapshotV4,
) -> tuple[ProductionStateMutationV4, ...]:
    """Close all mutable paths and reject path/schema drift."""

    before.validate()
    after.validate()
    before_map = before.tensor_map()
    after_map = after.tensor_map()
    mutable = {
        path
        for path, tensor in before_map.items()
        if tensor.ownership == ProductionTensorOwnership.MUTABLE_COPY_OUT
    }
    if set(before_map) != set(after_map):
        raise ValueError("RVIR-v4 pre/post production tensor paths differ")
    receipts: list[ProductionStateMutationV4] = []
    for path in sorted(mutable):
        left = before_map[path]
        right = after_map[path]
        if (
            left.role != right.role
            or left.axes != right.axes
            or left.value.shape != right.value.shape
            or left.value.dtype != right.value.dtype
        ):
            raise ValueError(f"RVIR-v4 mutable tensor schema drift: {path}")
        receipt = ProductionStateMutationV4(
            semantic_path=path,
            before_sha256=left.content_sha256,
            after_sha256=right.content_sha256,
            changed=left.content_sha256 != right.content_sha256,
        )
        receipt.validate()
        receipts.append(receipt)
    return tuple(receipts)


def production_snapshot_to_payload_v4(
    snapshot: ProductionStateSnapshotV4,
) -> dict[str, object]:
    """Serialize a snapshot to a torch.save-compatible plain mapping."""

    snapshot.validate()
    return {
        "schema_version": snapshot.schema_version,
        "snapshot_id": snapshot.snapshot_id,
        "tensors": [
            tensor.metadata() | {"value": tensor.value.detach().cpu().clone()}
            for tensor in snapshot.tensors
        ],
        "history": [entry.to_dict() for entry in snapshot.history],
        "optimizer_policy": snapshot.optimizer_policy.to_dict(),
        "snapshot_hash": snapshot.stable_hash(),
    }


def production_snapshot_from_payload_v4(
    payload: Mapping[str, object],
) -> ProductionStateSnapshotV4:
    """Reconstruct and validate a snapshot from a plain artifact mapping."""

    raw_tensors = payload.get("tensors")
    raw_history = payload.get("history")
    raw_policy = payload.get("optimizer_policy")
    if (
        not isinstance(raw_tensors, list)
        or not isinstance(raw_history, list)
        or not isinstance(raw_policy, Mapping)
    ):
        raise TypeError("RVIR-v4 snapshot payload structure differs")
    tensors: list[OwnedProductionTensorV4] = []
    for raw in raw_tensors:
        if not isinstance(raw, Mapping) or not torch.is_tensor(raw.get("value")):
            raise TypeError("RVIR-v4 snapshot tensor payload differs")
        value = cast(torch.Tensor, raw["value"])
        axes = raw.get("axes")
        if not isinstance(axes, list) or not all(
            isinstance(axis, str) for axis in axes
        ):
            raise TypeError("RVIR-v4 snapshot tensor axes payload differs")
        tensor = OwnedProductionTensorV4(
            semantic_path=str(raw.get("semantic_path", "")),
            role=ProductionTensorRole(str(raw.get("role", ""))),
            axes=tuple(axes),
            value=value.detach().cpu().contiguous().clone(),
            content_sha256=str(raw.get("content_sha256", "")),
            source_device=str(raw.get("source_device", "")),
            ownership=ProductionTensorOwnership(str(raw.get("ownership", ""))),
            alias_group=str(raw.get("alias_group", "")),
        )
        tensor.validate()
        tensors.append(tensor)
    history: list[ProductionSplitHistoryEntryV4] = []
    for raw in raw_history:
        if not isinstance(raw, Mapping):
            raise TypeError("RVIR-v4 snapshot history payload differs")
        locations = raw.get("locations")
        coefficients = raw.get("coefficients")
        biases = raw.get("biases")
        scores = raw.get("scores")
        depths = raw.get("depths")
        if (
            not isinstance(locations, list)
            or not isinstance(coefficients, list)
            or (biases is not None and not isinstance(biases, list))
            or (scores is not None and not isinstance(scores, list))
            or (depths is not None and not isinstance(depths, list))
        ):
            raise TypeError("RVIR-v4 snapshot history values differ")
        history.append(
            ProductionSplitHistoryEntryV4(
                domain_ordinal=int(raw.get("domain_ordinal", -1)),
                layer_name=str(raw.get("layer_name", "")),
                locations=tuple(int(item) for item in locations),
                coefficients=tuple(float(item) for item in coefficients),
                biases=(
                    None if biases is None else tuple(float(item) for item in biases)
                ),
                scores=(
                    None if scores is None else tuple(float(item) for item in scores)
                ),
                depths=(
                    None if depths is None else tuple(float(item) for item in depths)
                ),
            )
        )
    policy = ProductionOptimizerPolicyV4(
        iteration=int(raw_policy.get("iteration", -1)),
        alpha_learning_rate=float(raw_policy.get("alpha_learning_rate", 0.0)),
        beta_learning_rate=float(raw_policy.get("beta_learning_rate", 0.0)),
        bound_lower=bool(raw_policy.get("bound_lower", False)),
        bound_upper=bool(raw_policy.get("bound_upper", False)),
        fix_intermediate_bounds=bool(raw_policy.get("fix_intermediate_bounds", False)),
        deterministic=bool(raw_policy.get("deterministic", False)),
        stop_criterion_id=str(raw_policy.get("stop_criterion_id", "")),
    )
    snapshot = ProductionStateSnapshotV4(
        snapshot_id=str(payload.get("snapshot_id", "")),
        tensors=tuple(tensors),
        history=tuple(history),
        optimizer_policy=policy,
        schema_version=str(payload.get("schema_version", "")),
    )
    snapshot.validate()
    if payload.get("snapshot_hash") != snapshot.stable_hash():
        raise ValueError("RVIR-v4 snapshot payload hash differs")
    return snapshot


__all__ = [
    "RVIR_V4_STATE_SCHEMA_VERSION",
    "OwnedProductionTensorV4",
    "ProductionStateBuilderV4",
    "ProductionOptimizerPolicyV4",
    "ProductionSplitHistoryEntryV4",
    "ProductionStateMutationV4",
    "ProductionStateSnapshotV4",
    "ProductionTensorOwnership",
    "ProductionTensorRole",
    "capture_alpha_state_v4",
    "capture_module_alpha_beta_state_v4",
    "capture_sparse_beta_state_v4",
    "capture_split_history_v4",
    "diff_production_state_v4",
    "production_snapshot_from_payload_v4",
    "production_snapshot_to_payload_v4",
    "production_tensor_sha256",
    "validate_beta_history_consistency",
]
