"""Typed RVIR-v4 production pre-state to native dense-state mapping."""

# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=too-many-arguments,too-many-instance-attributes
# pylint: disable=too-many-boolean-expressions,missing-function-docstring

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import torch

from ..domains.interval import IntervalState
from .native_alpha_beta_optimization_state import (
    NativeAlphaBetaOptimizationState,
    NativeAlphaBetaStateScope,
)
from .rvir_v4_production_state import (
    OwnedProductionTensorV4,
    ProductionStateSnapshotV4,
    ProductionTensorRole,
    production_tensor_sha256,
)

RVIR_V4_PRE_STATE_MAPPING_SCHEMA = "boundflow.rvir-v4-pre-state-mapping/v1"
ROUND_TRIP_ATOL = 2e-4
ROUND_TRIP_RTOL = 2e-4


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


def _one_role(
    snapshot: ProductionStateSnapshotV4, role: ProductionTensorRole
) -> OwnedProductionTensorV4:
    values = [tensor for tensor in snapshot.tensors if tensor.role == role]
    if len(values) != 1:
        raise ValueError(f"RVIR-v4 pre-state requires one {role.value}")
    return values[0]


@dataclass(frozen=True)
class ProductionReluTopologyV4:
    """Exact provider activation/preactivation to native primal-value linkage."""

    provider_activation: str
    provider_preactivation: str
    native_preactivation: str
    provider_start_node: str

    def validate(self) -> None:
        if any(
            not value
            for value in (
                self.provider_activation,
                self.provider_preactivation,
                self.native_preactivation,
                self.provider_start_node,
            )
        ):
            raise ValueError("RVIR-v4 pre-state ReLU topology differs")

    def to_dict(self) -> dict[str, str]:
        self.validate()
        return {
            "provider_activation": self.provider_activation,
            "provider_preactivation": self.provider_preactivation,
            "native_preactivation": self.native_preactivation,
            "provider_start_node": self.provider_start_node,
        }


@dataclass(frozen=True)
class ProductionPreStateIdentityV4:
    """Independent snapshot/topology/history/intermediate-bound identity."""

    snapshot_hash: str
    topology_hash: str
    history_hash: str
    intermediate_bounds_hash: str

    def validate(self) -> None:
        if not all(
            _is_sha256(value)
            for value in (
                self.snapshot_hash,
                self.topology_hash,
                self.history_hash,
                self.intermediate_bounds_hash,
            )
        ):
            raise ValueError("RVIR-v4 pre-state identity differs")

    def to_dict(self) -> dict[str, str]:
        self.validate()
        return {
            "snapshot_hash": self.snapshot_hash,
            "topology_hash": self.topology_hash,
            "history_hash": self.history_hash,
            "intermediate_bounds_hash": self.intermediate_bounds_hash,
        }


@dataclass(frozen=True)
class ProductionStateRoundTripReceiptV4:
    """Compressed source → dense native → compressed/full source receipt."""

    semantic_path: str
    role: ProductionTensorRole
    mapped_source_hash: str
    mapped_round_trip_hash: str
    full_source_hash: str
    full_round_trip_hash: str
    mapped_element_count: int
    copy_through_element_count: int
    maximum_absolute_difference: float
    sign_exact: bool

    def validate(self) -> None:
        if (
            not self.semantic_path
            or self.role
            not in {ProductionTensorRole.ALPHA, ProductionTensorRole.BETA_VALUE}
            or not all(
                _is_sha256(value)
                for value in (
                    self.mapped_source_hash,
                    self.mapped_round_trip_hash,
                    self.full_source_hash,
                    self.full_round_trip_hash,
                )
            )
            or self.mapped_source_hash != self.mapped_round_trip_hash
            or self.full_source_hash != self.full_round_trip_hash
            or self.mapped_element_count < 0
            or self.copy_through_element_count < 0
            or not math.isfinite(self.maximum_absolute_difference)
            or self.maximum_absolute_difference > ROUND_TRIP_ATOL
            or self.sign_exact is not True
        ):
            raise ValueError("RVIR-v4 pre-state round-trip receipt differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "semantic_path": self.semantic_path,
            "role": self.role.value,
            "mapped_source_hash": self.mapped_source_hash,
            "mapped_round_trip_hash": self.mapped_round_trip_hash,
            "full_source_hash": self.full_source_hash,
            "full_round_trip_hash": self.full_round_trip_hash,
            "mapped_element_count": self.mapped_element_count,
            "copy_through_element_count": self.copy_through_element_count,
            "maximum_absolute_difference": self.maximum_absolute_difference,
            "sign_exact": self.sign_exact,
        }


@dataclass(frozen=True)
class ProductionNativePreStateV4:
    """Dense native α/β/split inputs and exact production ownership receipts."""

    identity: ProductionPreStateIdentityV4
    relu_pre_by_input: tuple[tuple[str, IntervalState], ...]
    split_by_relu_input: tuple[tuple[str, torch.Tensor], ...]
    alpha_by_relu_input: tuple[tuple[str, torch.Tensor], ...]
    beta_by_relu_input: tuple[tuple[str, torch.Tensor], ...]
    round_trip_receipts: tuple[ProductionStateRoundTripReceiptV4, ...]
    schema_version: str = RVIR_V4_PRE_STATE_MAPPING_SCHEMA

    @property
    def relu_pre(self) -> dict[str, IntervalState]:
        return dict(self.relu_pre_by_input)

    @property
    def splits(self) -> dict[str, torch.Tensor]:
        return dict(self.split_by_relu_input)

    @property
    def alphas(self) -> dict[str, torch.Tensor]:
        return dict(self.alpha_by_relu_input)

    @property
    def betas(self) -> dict[str, torch.Tensor]:
        return dict(self.beta_by_relu_input)

    def validate(self) -> None:
        self.identity.validate()
        relu_pre = self.relu_pre
        splits = self.splits
        alphas = self.alphas
        betas = self.betas
        if (
            self.schema_version != RVIR_V4_PRE_STATE_MAPPING_SCHEMA
            or len(relu_pre) != len(self.relu_pre_by_input)
            or len(relu_pre) != 6
            or len(splits) != len(self.split_by_relu_input)
            or len(splits) != 6
            or len(alphas) != len(self.alpha_by_relu_input)
            or len(alphas) != 6
            or len(betas) != len(self.beta_by_relu_input)
            or len(betas) != 6
            or set(relu_pre) != set(splits)
            or set(relu_pre) != set(alphas)
            or set(relu_pre) != set(betas)
            or len(self.round_trip_receipts) != 12
        ):
            raise ValueError("RVIR-v4 pre-state dense inventory differs")
        for name in sorted(splits):
            interval = relu_pre[name]
            interval.validate()
            split = splits[name]
            alpha = alphas[name]
            beta = betas[name]
            if (
                split.shape != alpha.shape
                or split.shape != beta.shape
                or split.shape != interval.lower.shape
                or split.dtype != torch.int8
                or not torch.is_floating_point(alpha)
                or alpha.dtype != beta.dtype
                or alpha.dtype != interval.lower.dtype
                or split.device != alpha.device
                or alpha.device != beta.device
                or alpha.device != interval.lower.device
                or not bool(torch.isfinite(interval.lower).all().item())
                or not bool(torch.isfinite(interval.upper).all().item())
                or not bool((interval.lower <= interval.upper).all().item())
                or not bool(((split >= -1) & (split <= 1)).all().item())
                or not bool(torch.isfinite(alpha).all().item())
                or not bool(torch.isfinite(beta).all().item())
                or not bool(((alpha >= 0) & (alpha <= 1)).all().item())
                or not bool((beta >= 0).all().item())
            ):
                raise ValueError(f"RVIR-v4 pre-state dense tensor differs: {name}")
        paths = [receipt.semantic_path for receipt in self.round_trip_receipts]
        if len(set(paths)) != len(paths):
            raise ValueError("RVIR-v4 pre-state round-trip paths repeat")
        for receipt in self.round_trip_receipts:
            receipt.validate()

    def metadata(self) -> dict[str, object]:
        self.validate()
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "identity": self.identity.to_dict(),
            "relu_states": {
                name: {
                    "intermediate_lower_hash": production_tensor_sha256(
                        self.relu_pre[name].lower
                    ),
                    "intermediate_upper_hash": production_tensor_sha256(
                        self.relu_pre[name].upper
                    ),
                    "split_hash": production_tensor_sha256(self.splits[name]),
                    "alpha_hash": production_tensor_sha256(self.alphas[name]),
                    "beta_hash": production_tensor_sha256(self.betas[name]),
                    "shape": list(self.splits[name].shape),
                    "dtype": str(self.alphas[name].dtype),
                }
                for name in sorted(self.splits)
            },
            "round_trip_receipts": [
                receipt.to_dict() for receipt in self.round_trip_receipts
            ],
            "performance_claimed": False,
        }
        payload["mapping_hash"] = _canonical_hash(payload)
        return payload

    def stable_hash(self) -> str:
        return str(self.metadata()["mapping_hash"])

    def to_native_state(
        self, scope: NativeAlphaBetaStateScope
    ) -> NativeAlphaBetaOptimizationState:
        """Bind mapped tensors to an independently built native scope."""

        self.validate()
        state = NativeAlphaBetaOptimizationState(
            scope=scope,
            split_by_relu_input=self.split_by_relu_input,
            alpha_by_relu_input=self.alpha_by_relu_input,
            beta_by_relu_input=self.beta_by_relu_input,
        )
        state.validate()
        return state


def _receipt(
    source: OwnedProductionTensorV4,
    mapped_source: torch.Tensor,
    mapped_round_trip: torch.Tensor,
    full_round_trip: torch.Tensor,
) -> ProductionStateRoundTripReceiptV4:
    if mapped_source.shape != mapped_round_trip.shape:
        raise ValueError("RVIR-v4 pre-state mapped round-trip shape differs")
    difference = (mapped_source - mapped_round_trip).abs()
    maximum = 0.0 if difference.numel() == 0 else float(difference.max().item())
    receipt = ProductionStateRoundTripReceiptV4(
        semantic_path=source.semantic_path,
        role=source.role,
        mapped_source_hash=production_tensor_sha256(mapped_source),
        mapped_round_trip_hash=production_tensor_sha256(mapped_round_trip),
        full_source_hash=source.content_sha256,
        full_round_trip_hash=production_tensor_sha256(full_round_trip),
        mapped_element_count=mapped_source.numel(),
        copy_through_element_count=source.value.numel() - mapped_source.numel(),
        maximum_absolute_difference=maximum,
        sign_exact=torch.equal(
            torch.sign(mapped_source), torch.sign(mapped_round_trip)
        ),
    )
    receipt.validate()
    return receipt


def _topology_identity(
    topology: tuple[ProductionReluTopologyV4, ...],
) -> tuple[dict[str, ProductionReluTopologyV4], str]:
    if len(topology) != 6:
        raise ValueError("RVIR-v4 pre-state topology count differs")
    for item in topology:
        item.validate()
    if (
        len({item.provider_activation for item in topology}) != len(topology)
        or len({item.provider_preactivation for item in topology}) != len(topology)
        or len({item.native_preactivation for item in topology}) != len(topology)
    ):
        raise ValueError("RVIR-v4 pre-state topology keys repeat")
    return (
        {item.native_preactivation: item for item in topology},
        _canonical_hash([item.to_dict() for item in topology]),
    )


def initialize_rvir_v4_native_pre_state(
    snapshot: ProductionStateSnapshotV4,
    topology: tuple[ProductionReluTopologyV4, ...],
    *,
    expected_identity: ProductionPreStateIdentityV4 | None = None,
) -> ProductionNativePreStateV4:
    """Restore dense lower α, SparseBeta, splits, and fixed intermediate bounds."""

    snapshot.validate()
    policy = snapshot.optimizer_policy
    if (
        policy.bound_lower is not True
        or policy.bound_upper is not False
        or policy.fix_intermediate_bounds is not True
    ):
        raise ValueError("RVIR-v4 pre-state production policy is not admitted")
    topology_by_native, topology_hash = _topology_identity(topology)
    tensor_map = snapshot.tensor_map()
    alpha_paths = {
        f"alpha/{_encode(item.provider_activation)}/{_encode(item.provider_start_node)}"
        for item in topology
    }
    observed_alpha_paths = {
        tensor.semantic_path
        for tensor in snapshot.tensors
        if tensor.role == ProductionTensorRole.ALPHA
    }
    beta_value_paths = {
        f"beta/{_encode(item.provider_preactivation)}/0/value" for item in topology
    }
    observed_beta_value_paths = {
        tensor.semantic_path
        for tensor in snapshot.tensors
        if tensor.role == ProductionTensorRole.BETA_VALUE
    }
    if (
        observed_alpha_paths != alpha_paths
        or observed_beta_value_paths != beta_value_paths
    ):
        raise ValueError("RVIR-v4 pre-state mutable path ownership differs")

    expected_intermediate_paths = {
        f"intermediate/{_encode(item.provider_preactivation)}/{polarity}"
        for item in topology
        for polarity in ("lower", "upper")
    }
    observed_intermediate_paths = {
        tensor.semantic_path
        for tensor in snapshot.tensors
        if tensor.role
        in {
            ProductionTensorRole.INTERMEDIATE_LOWER,
            ProductionTensorRole.INTERMEDIATE_UPPER,
        }
    }
    expected_shape_paths = {
        f"alpha_layout/{_encode(item.provider_activation)}/feature_shape"
        for item in topology
    }
    observed_shape_paths = {
        tensor.semantic_path
        for tensor in snapshot.tensors
        if tensor.role == ProductionTensorRole.ALPHA_FEATURE_SHAPE
    }
    allowed_layout_prefixes = {
        f"alpha_layout/{_encode(item.provider_activation)}/" for item in topology
    }
    allowed_lookup_paths = {
        f"alpha_layout/{_encode(item.provider_activation)}/spec_lookup/"
        f"{_encode(item.provider_start_node)}"
        for item in topology
    }
    layout_tensors = [
        tensor
        for tensor in snapshot.tensors
        if tensor.role
        in {
            ProductionTensorRole.ALPHA_FEATURE_INDEX,
            ProductionTensorRole.ALPHA_SPEC_LOOKUP_INDEX,
        }
    ]
    if (
        observed_intermediate_paths != expected_intermediate_paths
        or observed_shape_paths != expected_shape_paths
        or any(
            not any(
                tensor.semantic_path.startswith(prefix)
                for prefix in allowed_layout_prefixes
            )
            for tensor in layout_tensors
        )
        or any(
            tensor.role == ProductionTensorRole.ALPHA_SPEC_LOOKUP_INDEX
            and tensor.semantic_path not in allowed_lookup_paths
            for tensor in layout_tensors
        )
    ):
        raise ValueError("RVIR-v4 pre-state read-only layout ownership differs")

    input_lower = _one_role(snapshot, ProductionTensorRole.INPUT_LOWER)
    domain_count = int(input_lower.value.shape[0])
    if domain_count != 6:
        raise ValueError("RVIR-v4 pre-state domain count differs")
    history_keys = {
        (entry.domain_ordinal, entry.layer_name) for entry in snapshot.history
    }
    expected_history_keys = {
        (domain, item.provider_preactivation)
        for domain in range(domain_count)
        for item in topology
    }
    if history_keys != expected_history_keys:
        raise ValueError("RVIR-v4 pre-state history coverage differs")
    history_hash = _canonical_hash([entry.to_dict() for entry in snapshot.history])

    relu_pre: dict[str, IntervalState] = {}
    splits: dict[str, torch.Tensor] = {}
    alphas: dict[str, torch.Tensor] = {}
    betas: dict[str, torch.Tensor] = {}
    receipts: list[ProductionStateRoundTripReceiptV4] = []
    intermediate_identity: list[dict[str, object]] = []
    for native, link in topology_by_native.items():
        encoded_pre = _encode(link.provider_preactivation)
        encoded_activation = _encode(link.provider_activation)
        encoded_start = _encode(link.provider_start_node)
        lower_item = tensor_map[f"intermediate/{encoded_pre}/lower"]
        upper_item = tensor_map[f"intermediate/{encoded_pre}/upper"]
        if (
            lower_item.role != ProductionTensorRole.INTERMEDIATE_LOWER
            or upper_item.role != ProductionTensorRole.INTERMEDIATE_UPPER
        ):
            raise ValueError("RVIR-v4 pre-state intermediate roles differ")
        interval = IntervalState(lower_item.value.clone(), upper_item.value.clone())
        interval.validate()
        relu_pre[native] = interval
        intermediate_identity.append(
            {
                "provider_preactivation": link.provider_preactivation,
                "native_preactivation": native,
                "lower_hash": lower_item.content_sha256,
                "upper_hash": upper_item.content_sha256,
            }
        )

        alpha_item = tensor_map[f"alpha/{encoded_activation}/{encoded_start}"]
        feature_shape_item = tensor_map[
            f"alpha_layout/{encoded_activation}/feature_shape"
        ]
        if (
            alpha_item.role != ProductionTensorRole.ALPHA
            or feature_shape_item.role != ProductionTensorRole.ALPHA_FEATURE_SHAPE
        ):
            raise ValueError("RVIR-v4 pre-state alpha/layout roles differ")
        feature_shape = tuple(int(value) for value in feature_shape_item.value.tolist())
        if alpha_item.value.ndim != 4 or tuple(alpha_item.value.shape[:3]) != (
            2,
            1,
            domain_count,
        ):
            raise ValueError("RVIR-v4 pre-state alpha leading axes differ")
        indices: list[torch.Tensor] = []
        ordinal = 0
        while (
            f"alpha_layout/{encoded_activation}/feature_index/{ordinal}" in tensor_map
        ):
            index_item = tensor_map[
                f"alpha_layout/{encoded_activation}/feature_index/{ordinal}"
            ]
            if index_item.role != ProductionTensorRole.ALPHA_FEATURE_INDEX:
                raise ValueError("RVIR-v4 pre-state alpha index role differs")
            indices.append(index_item.value)
            ordinal += 1
        if (
            sum(
                tensor.role == ProductionTensorRole.ALPHA_FEATURE_INDEX
                and tensor.semantic_path.startswith(
                    f"alpha_layout/{encoded_activation}/feature_index/"
                )
                for tensor in snapshot.tensors
            )
            != ordinal
        ):
            raise ValueError("RVIR-v4 pre-state alpha index ordinals differ")
        lookup = tensor_map.get(
            f"alpha_layout/{encoded_activation}/spec_lookup/{encoded_start}"
        )
        if lookup is not None and (
            lookup.role != ProductionTensorRole.ALPHA_SPEC_LOOKUP_INDEX
            or not torch.equal(lookup.value, torch.zeros_like(lookup.value))
        ):
            raise ValueError("RVIR-v4 pre-state alpha spec lookup differs")
        compressed_alpha = alpha_item.value[0, 0]
        dense_alpha = torch.zeros(
            (domain_count,) + feature_shape, dtype=compressed_alpha.dtype
        )
        if indices:
            coordinates = torch.stack(indices, dim=1)
            if torch.unique(coordinates, dim=0).shape[0] != coordinates.shape[0]:
                raise ValueError("RVIR-v4 pre-state alpha coordinates repeat")
            dense_alpha[(slice(None),) + tuple(indices)] = compressed_alpha
            projected_alpha = dense_alpha[(slice(None),) + tuple(indices)]
        else:
            dense_alpha.copy_(compressed_alpha.reshape_as(dense_alpha))
            projected_alpha = dense_alpha.reshape_as(compressed_alpha)
        alpha_round_trip = alpha_item.value.clone()
        alpha_round_trip[0, 0].copy_(projected_alpha)
        receipts.append(
            _receipt(
                alpha_item,
                compressed_alpha,
                projected_alpha,
                alpha_round_trip,
            )
        )
        alphas[native] = dense_alpha

        split = torch.zeros_like(dense_alpha, dtype=torch.int8)
        flat_split = split.reshape(domain_count, -1)
        for domain in range(domain_count):
            entry = next(
                item
                for item in snapshot.history
                if item.domain_ordinal == domain
                and item.layer_name == link.provider_preactivation
            )
            if len(set(entry.locations)) != len(entry.locations):
                raise ValueError("RVIR-v4 pre-state history locations repeat")
            for location, coefficient in zip(entry.locations, entry.coefficients):
                if location >= flat_split.shape[1] or coefficient not in {-1.0, 1.0}:
                    raise ValueError("RVIR-v4 pre-state split value/location differs")
                flat_split[domain, location] = int(coefficient)
        splits[native] = split

        beta_prefix = f"beta/{encoded_pre}/0"
        beta_item = tensor_map[f"{beta_prefix}/value"]
        location_item = tensor_map[f"{beta_prefix}/location"]
        sign_item = tensor_map[f"{beta_prefix}/sign"]
        if (
            beta_item.role != ProductionTensorRole.BETA_VALUE
            or location_item.role != ProductionTensorRole.BETA_LOCATION
            or sign_item.role != ProductionTensorRole.BETA_SIGN
        ):
            raise ValueError("RVIR-v4 pre-state beta roles differ")
        locations = location_item.value
        dense_beta = torch.zeros_like(dense_alpha)
        flat_beta = dense_beta.reshape(domain_count, -1)
        for domain in range(domain_count):
            row_locations = locations[domain]
            if (
                bool((row_locations >= flat_beta.shape[1]).any().item())
                or torch.unique(row_locations).numel() != row_locations.numel()
            ):
                raise ValueError("RVIR-v4 pre-state beta locations differ")
            flat_beta[domain, row_locations] = beta_item.value[domain]
        projected_beta = torch.stack(
            [flat_beta[domain, locations[domain]] for domain in range(domain_count)]
        )
        beta_round_trip = projected_beta.clone()
        receipts.append(
            _receipt(beta_item, beta_item.value, projected_beta, beta_round_trip)
        )
        betas[native] = dense_beta

    identity = ProductionPreStateIdentityV4(
        snapshot_hash=snapshot.stable_hash(),
        topology_hash=topology_hash,
        history_hash=history_hash,
        intermediate_bounds_hash=_canonical_hash(intermediate_identity),
    )
    identity.validate()
    if (
        expected_identity is not None
        and identity.to_dict() != expected_identity.to_dict()
    ):
        raise ValueError("RVIR-v4 pre-state frozen identity differs")
    mapping = ProductionNativePreStateV4(
        identity=identity,
        relu_pre_by_input=tuple(sorted(relu_pre.items())),
        split_by_relu_input=tuple(sorted(splits.items())),
        alpha_by_relu_input=tuple(sorted(alphas.items())),
        beta_by_relu_input=tuple(sorted(betas.items())),
        round_trip_receipts=tuple(
            sorted(receipts, key=lambda receipt: receipt.semantic_path)
        ),
    )
    mapping.validate()
    return mapping


__all__ = [
    "ProductionNativePreStateV4",
    "ProductionPreStateIdentityV4",
    "ProductionReluTopologyV4",
    "ProductionStateRoundTripReceiptV4",
    "initialize_rvir_v4_native_pre_state",
]
