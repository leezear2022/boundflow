"""Typed IR for one differentiable lower-bound ReLU/affine region."""

# pylint: disable=too-many-instance-attributes,too-many-boolean-expressions
# pylint: disable=missing-function-docstring

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping

DIFFERENTIABLE_LOWER_REGION_IR_SCHEMA = "boundflow.differentiable-lower-region-ir/v1"
DIFFERENTIABLE_LOWER_REGION_INSTANCE_SCHEMA = (
    "boundflow.differentiable-lower-region-instance/v1"
)
DIFFERENTIABLE_LOWER_REGION_STAGE_ORDER = (
    "beta-sparse-scatter",
    "alpha-sparse-reconstruction",
    "lower-sign-select",
    "intercept-bias-reduction",
    "beta-signed-pre-add",
    "affine-right-contraction",
    "incoming-bias-carry",
)


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _json_value(value: object) -> object:
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError("differentiable lower-region attribute is not canonical JSON")


def freeze_ir_attribute(value: object) -> object:
    """Freeze a JSON attribute so a frozen IR cannot retain mutable lists."""

    if isinstance(value, list):
        return tuple(freeze_ir_attribute(item) for item in value)
    if isinstance(value, tuple):
        return tuple(freeze_ir_attribute(item) for item in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError("differentiable lower-region attribute is not canonical JSON")


@dataclass(frozen=True)
class DifferentiableTensorContractIRV1:
    """Static tensor schema; numerical payload and digest belong to an instance."""

    name: str
    role: str
    shape: tuple[int, ...]
    dtype: str
    device_kind: str
    layout: str
    strides: tuple[int, ...]
    requires_grad: bool
    present: bool = True

    def validate(self) -> None:
        if (
            not self.name
            or not self.role
            or not self.shape
            or any(dimension < 0 for dimension in self.shape)
            or self.dtype not in {"torch.float32", "torch.int64"}
            or self.device_kind != "cuda"
            or self.layout != "contiguous-strided"
            or len(self.strides) != len(self.shape)
            or any(stride < 0 for stride in self.strides)
            or self.present is not True
        ):
            raise ValueError(f"differentiable tensor contract differs: {self.name}")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "name": self.name,
            "role": self.role,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "device_kind": self.device_kind,
            "layout": self.layout,
            "strides": list(self.strides),
            "requires_grad": self.requires_grad,
            "present": self.present,
        }


@dataclass(frozen=True)
class DifferentiableLowerRegionIRV1:
    """Static, lower-only semantic plan for one production ReLU/affine region."""

    anchor_id: str
    anchor_hash: str
    provider_start_node: str
    native_preactivation: str
    provider_activation: str
    provider_preactivation: str
    producer_ordinal: int
    producer_name: str
    operator_kind: str
    domain_count: int
    spec_count: int
    relu_logical_shape: tuple[int, ...]
    affine_input_shape: tuple[int, ...]
    coefficient_shape: tuple[int, ...]
    result_coefficient_shape: tuple[int, ...]
    tensor_contracts: tuple[DifferentiableTensorContractIRV1, ...]
    operator_attributes: tuple[tuple[str, object], ...]
    alpha_feature_index_count: int
    alpha_spec_lookup_present: bool
    beta_active: bool
    beta_bias_present: bool
    source_state_hash: str
    primal_graph_hash: str
    split_state_hash: str
    topology_hash: str
    lineage_hash: str
    alpha_direction_index: int = 0
    alpha_spec_index: int = 0
    relu_lower_relaxation: str = "ambiguous-alpha-sign-select-v1"
    beta_pre_add_formula: str = "negative-value-times-split-sign-v1"
    lower_only: bool = True
    coefficient_representation: str = "dense"
    fanout: str = "single-consumer"
    stream_ownership: str = "current-default"
    alias_policy: str = "none"
    stage_order: tuple[str, ...] = DIFFERENTIABLE_LOWER_REGION_STAGE_ORDER
    schema_version: str = DIFFERENTIABLE_LOWER_REGION_IR_SCHEMA

    @property
    def tensor_contract_map(self) -> dict[str, DifferentiableTensorContractIRV1]:
        return {contract.name: contract for contract in self.tensor_contracts}

    @property
    def operator_attribute_map(self) -> dict[str, object]:
        return dict(self.operator_attributes)

    def validate(self) -> None:  # pylint: disable=too-many-branches
        contracts = self.tensor_contract_map
        attributes = self.operator_attribute_map
        production_beta_contract = contracts.get("value/production_beta")
        for contract in self.tensor_contracts:
            contract.validate()
        for _name, value in self.operator_attributes:
            _json_value(value)
        required_hashes = (
            self.anchor_hash,
            self.source_state_hash,
            self.primal_graph_hash,
            self.split_state_hash,
            self.topology_hash,
            self.lineage_hash,
        )
        expected_coefficient = (
            self.domain_count,
            self.spec_count,
            *self.relu_logical_shape,
        )
        expected_result = (
            self.domain_count,
            self.spec_count,
            *self.affine_input_shape,
        )
        if (
            self.schema_version != DIFFERENTIABLE_LOWER_REGION_IR_SCHEMA
            or not self.anchor_id
            or any(not _is_sha256(value) for value in required_hashes)
            or self.provider_start_node != "/49"
            or not self.native_preactivation
            or not self.provider_activation
            or not self.provider_preactivation
            or self.producer_ordinal < 0
            or not self.producer_name
            or self.operator_kind not in {"linear", "conv2d"}
            or self.domain_count < 1
            or self.spec_count < 1
            or not self.relu_logical_shape
            or not self.affine_input_shape
            or self.coefficient_shape != expected_coefficient
            or self.result_coefficient_shape != expected_result
            or len(contracts) != len(self.tensor_contracts)
            or tuple(sorted(contracts))
            != tuple(contract.name for contract in self.tensor_contracts)
            or len(attributes) != len(self.operator_attributes)
            or tuple(sorted(attributes))
            != tuple(name for name, _value in self.operator_attributes)
            or attributes.get("operator_kind") != self.operator_kind
            or attributes.get("input_shape") != self.affine_input_shape
            or attributes.get("output_shape") != self.relu_logical_shape
            or attributes.get("operator_bias_present")
            != ("amendment/operator_bias" in contracts)
            or self.alpha_feature_index_count != len(self.relu_logical_shape)
            or production_beta_contract is None
            or self.beta_active != (production_beta_contract.shape[-1] > 0)
            or self.alpha_direction_index != 0
            or self.alpha_spec_index != 0
            or self.relu_lower_relaxation != "ambiguous-alpha-sign-select-v1"
            or self.beta_pre_add_formula != "negative-value-times-split-sign-v1"
            or self.lower_only is not True
            or self.coefficient_representation != "dense"
            or self.fanout != "single-consumer"
            or self.stream_ownership != "current-default"
            or self.alias_policy != "none"
            or self.stage_order != DIFFERENTIABLE_LOWER_REGION_STAGE_ORDER
        ):
            raise ValueError("differentiable lower-region IR differs")
        required_contracts = {
            "value/incoming_lower_a",
            "value/preactivation_lower",
            "value/preactivation_upper",
            "value/production_alpha",
            "value/native_alpha",
            "value/production_beta",
            "value/native_beta",
            "value/operator_weight",
            "value/output_lower_a",
            "value/output_bias",
            "production_gradient/native_alpha",
            "amendment/incoming_lower_bias",
            "amendment/output_lower_a_gradient",
            "amendment/output_bias_gradient",
        }
        if not required_contracts.issubset(contracts):
            raise ValueError("differentiable lower-region tensor inventory differs")
        mapping_names = tuple(
            name.removeprefix("mapping/")
            for name in contracts
            if name.startswith("mapping/")
        )
        if (
            sum(name.endswith("/feature_shape") for name in mapping_names) != 1
            or sum("/feature_index/" in name for name in mapping_names)
            != self.alpha_feature_index_count
            or sum("/spec_lookup/" in name for name in mapping_names)
            != int(self.alpha_spec_lookup_present)
            or sum(name.endswith("/location") for name in mapping_names) != 1
            or sum(name.endswith("/sign") for name in mapping_names) != 1
        ):
            raise ValueError("differentiable lower-region sparse layout differs")
        if self.beta_active != (
            "value/relu_pre_add_coeff_l" in contracts
            and "production_gradient/native_beta" in contracts
        ):
            raise ValueError("differentiable lower-region beta ownership differs")
        if self.operator_kind == "conv2d":
            if not {
                "stride",
                "padding",
                "dilation",
                "groups",
                "output_padding",
            }.issubset(attributes):
                raise ValueError("differentiable Conv attributes differ")
        elif "output_padding" in attributes:
            raise ValueError("differentiable Linear attributes differ")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "anchor": {
                "anchor_id": self.anchor_id,
                "anchor_hash": self.anchor_hash,
                "provider_start_node": self.provider_start_node,
                "native_preactivation": self.native_preactivation,
                "provider_activation": self.provider_activation,
                "provider_preactivation": self.provider_preactivation,
                "producer_ordinal": self.producer_ordinal,
                "producer_name": self.producer_name,
            },
            "operator_kind": self.operator_kind,
            "domain_count": self.domain_count,
            "spec_count": self.spec_count,
            "relu_logical_shape": list(self.relu_logical_shape),
            "affine_input_shape": list(self.affine_input_shape),
            "coefficient_shape": list(self.coefficient_shape),
            "result_coefficient_shape": list(self.result_coefficient_shape),
            "tensor_contracts": [item.to_dict() for item in self.tensor_contracts],
            "operator_attributes": {
                name: _json_value(value) for name, value in self.operator_attributes
            },
            "sparse_state": {
                "alpha_feature_index_count": self.alpha_feature_index_count,
                "alpha_spec_lookup_present": self.alpha_spec_lookup_present,
                "beta_active": self.beta_active,
                "beta_bias_present": self.beta_bias_present,
                "alpha_direction_index": self.alpha_direction_index,
                "alpha_spec_index": self.alpha_spec_index,
                "beta_pre_add_formula": self.beta_pre_add_formula,
            },
            "relu_lower_relaxation": self.relu_lower_relaxation,
            "identity": {
                "source_state_hash": self.source_state_hash,
                "primal_graph_hash": self.primal_graph_hash,
                "split_state_hash": self.split_state_hash,
                "topology_hash": self.topology_hash,
                "lineage_hash": self.lineage_hash,
            },
            "lower_only": self.lower_only,
            "coefficient_representation": self.coefficient_representation,
            "fanout": self.fanout,
            "stream_ownership": self.stream_ownership,
            "alias_policy": self.alias_policy,
            "stage_order": list(self.stage_order),
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class DifferentiableLowerRegionInstanceV1:
    """One raw-bound execution instance of a static lower-region IR."""

    ir_hash: str
    reference_capture_hash: str
    base_capture_hash: str
    input_tensor_hashes: tuple[tuple[str, str], ...]
    schema_version: str = DIFFERENTIABLE_LOWER_REGION_INSTANCE_SCHEMA

    @property
    def input_tensor_hash_map(self) -> dict[str, str]:
        return dict(self.input_tensor_hashes)

    def validate_against(self, ir: DifferentiableLowerRegionIRV1) -> None:
        ir.validate()
        hashes = self.input_tensor_hash_map
        if (
            self.schema_version != DIFFERENTIABLE_LOWER_REGION_INSTANCE_SCHEMA
            or self.ir_hash != ir.stable_hash()
            or not _is_sha256(self.reference_capture_hash)
            or not _is_sha256(self.base_capture_hash)
            or len(hashes) != len(self.input_tensor_hashes)
            or tuple(sorted(hashes))
            != tuple(name for name, _value in self.input_tensor_hashes)
            or set(hashes) != set(ir.tensor_contract_map)
            or any(not _is_sha256(value) for value in hashes.values())
        ):
            raise ValueError("differentiable lower-region instance differs")

    def to_dict(self, ir: DifferentiableLowerRegionIRV1) -> dict[str, object]:
        self.validate_against(ir)
        return {
            "schema_version": self.schema_version,
            "ir_hash": self.ir_hash,
            "reference_capture_hash": self.reference_capture_hash,
            "base_capture_hash": self.base_capture_hash,
            "input_tensor_hashes": dict(self.input_tensor_hashes),
        }

    def stable_hash(self, ir: DifferentiableLowerRegionIRV1) -> str:
        return _canonical_hash(self.to_dict(ir))


def canonical_ir_attributes(
    attributes: Mapping[str, object],
) -> tuple[tuple[str, object], ...]:
    """Return a key-sorted, immutable attribute tuple for an IR instance."""

    return tuple(
        (name, freeze_ir_attribute(value)) for name, value in sorted(attributes.items())
    )


__all__ = [
    "DIFFERENTIABLE_LOWER_REGION_INSTANCE_SCHEMA",
    "DIFFERENTIABLE_LOWER_REGION_IR_SCHEMA",
    "DIFFERENTIABLE_LOWER_REGION_STAGE_ORDER",
    "DifferentiableLowerRegionIRV1",
    "DifferentiableLowerRegionInstanceV1",
    "DifferentiableTensorContractIRV1",
    "canonical_ir_attributes",
    "freeze_ir_attribute",
]
