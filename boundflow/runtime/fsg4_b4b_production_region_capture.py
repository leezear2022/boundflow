"""Typed B4-B production differentiable-region capture contracts."""

# pylint: disable=too-many-instance-attributes,too-many-boolean-expressions
# pylint: disable=too-many-arguments,missing-function-docstring

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping

import torch

from .rvir_v4_production_state import production_tensor_sha256

B4B_CAPTURE_SCHEMA = "boundflow.fsg4-b4b-production-region-capture/v1"
B4B_SHAPE_SOURCE = "correlation-parent-boundflow-operator"


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = 1
    result: list[int] = []
    for dimension in reversed(shape):
        result.append(stride)
        stride *= max(dimension, 1)
    return tuple(reversed(result))


@dataclass(frozen=True)
class DifferentiableRegionAnchorV1:
    """One preregistered semantic or performance capture anchor."""

    anchor_id: str
    role: str
    native_preactivation: str
    provider_activation: str
    provider_preactivation: str
    producer_op_ordinal: int
    producer_op_name: str
    producer_op_type: str
    coefficient_shape: tuple[int, ...]
    preactivation_shape: tuple[int, ...]
    production_alpha_path: str
    production_alpha_shape: tuple[int, ...]
    native_alpha_shape: tuple[int, ...]
    production_beta_path: str
    production_beta_shape: tuple[int, ...]
    native_beta_shape: tuple[int, ...]
    beta_must_be_nonempty: bool

    def validate(self) -> None:
        if (
            self.role not in {"semantic", "performance"}
            or not self.anchor_id
            or not self.native_preactivation
            or not self.provider_activation
            or not self.provider_preactivation
            or self.producer_op_ordinal < 0
            or not self.producer_op_name
            or self.producer_op_type not in {"linear", "conv2d"}
            or not self.coefficient_shape
            or not self.preactivation_shape
            or not self.production_alpha_path
            or not self.production_alpha_shape
            or self.native_alpha_shape != self.preactivation_shape
            or not self.production_beta_path
            or not self.production_beta_shape
            or self.native_beta_shape != self.preactivation_shape
            or any(dimension < 0 for dimension in self.production_beta_shape)
            or self.beta_must_be_nonempty != (math.prod(self.production_beta_shape) > 0)
        ):
            raise ValueError("FSG4/B4-B differentiable anchor differs")

    def metadata(self) -> dict[str, object]:
        self.validate()
        return {
            "anchor_id": self.anchor_id,
            "role": self.role,
            "native_preactivation": self.native_preactivation,
            "provider_activation": self.provider_activation,
            "provider_preactivation": self.provider_preactivation,
            "producer_op_ordinal": self.producer_op_ordinal,
            "producer_op_name": self.producer_op_name,
            "producer_op_type": self.producer_op_type,
            "coefficient_shape": list(self.coefficient_shape),
            "preactivation_shape": list(self.preactivation_shape),
            "production_alpha_path": self.production_alpha_path,
            "production_alpha_shape": list(self.production_alpha_shape),
            "native_alpha_shape": list(self.native_alpha_shape),
            "production_beta_path": self.production_beta_path,
            "production_beta_shape": list(self.production_beta_shape),
            "native_beta_shape": list(self.native_beta_shape),
            "beta_must_be_nonempty": self.beta_must_be_nonempty,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.metadata())


B4B_SEMANTIC_ANCHOR_V1 = DifferentiableRegionAnchorV1(
    anchor_id="semantic-active-beta-gemm-14",
    role="semantic",
    native_preactivation="31",
    provider_activation="/48",
    provider_preactivation="/input-28",
    producer_op_ordinal=14,
    producer_op_name="Gemm_14",
    producer_op_type="linear",
    coefficient_shape=(6, 1, 100),
    preactivation_shape=(6, 100),
    production_alpha_path="alpha/%2F48/%2F49",
    production_alpha_shape=(2, 1, 6, 27),
    native_alpha_shape=(6, 100),
    production_beta_path="beta/%2Finput-28/0/value",
    production_beta_shape=(6, 1),
    native_beta_shape=(6, 100),
    beta_must_be_nonempty=True,
)

B4B_PERFORMANCE_ANCHOR_V1 = DifferentiableRegionAnchorV1(
    anchor_id="performance-conv-8-candidate",
    role="performance",
    native_preactivation="25",
    provider_activation="/input-24",
    provider_preactivation="/input-20",
    producer_op_ordinal=8,
    producer_op_name="Conv_8",
    producer_op_type="conv2d",
    coefficient_shape=(6, 1, 16, 8, 8),
    preactivation_shape=(6, 16, 8, 8),
    production_alpha_path="alpha/%2Finput-24/%2F49",
    production_alpha_shape=(2, 1, 6, 86),
    native_alpha_shape=(6, 16, 8, 8),
    production_beta_path="beta/%2Finput-20/0/value",
    production_beta_shape=(6, 0),
    native_beta_shape=(6, 16, 8, 8),
    beta_must_be_nonempty=False,
)


def b4b_v1_anchors() -> tuple[DifferentiableRegionAnchorV1, ...]:
    """Return the two preregistered B4-B v1 anchors."""

    anchors = (B4B_SEMANTIC_ANCHOR_V1, B4B_PERFORMANCE_ANCHOR_V1)
    for anchor in anchors:
        anchor.validate()
    return anchors


@dataclass(frozen=True)
class CapturedCudaTensorV1:
    """One immutable CPU payload retaining its production CUDA identity."""

    name: str
    value: torch.Tensor
    source_shape: tuple[int, ...]
    source_dtype: str
    source_device: str
    source_strides: tuple[int, ...]
    source_requires_grad: bool
    content_sha256: str

    @classmethod
    def from_tensor(cls, name: str, value: torch.Tensor) -> "CapturedCudaTensorV1":
        if value.device.type != "cuda":
            raise ValueError("FSG4/B4-B capture requires a production CUDA tensor")
        if value.layout != torch.strided or not value.is_contiguous():
            raise ValueError("FSG4/B4-B capture requires contiguous strided tensors")
        payload = value.detach().cpu().contiguous().clone()
        return cls(
            name=name,
            value=payload,
            source_shape=tuple(int(dimension) for dimension in value.shape),
            source_dtype=str(value.dtype),
            source_device=str(value.device),
            source_strides=tuple(int(stride) for stride in value.stride()),
            source_requires_grad=bool(value.requires_grad),
            content_sha256=production_tensor_sha256(payload),
        )

    def validate(self) -> None:
        if (
            not self.name
            or self.value.device.type != "cpu"
            or self.value.layout != torch.strided
            or not self.value.is_contiguous()
            or tuple(self.value.shape) != self.source_shape
            or str(self.value.dtype) != self.source_dtype
            or not self.source_device.startswith("cuda:")
            or self.source_strides != _contiguous_strides(self.source_shape)
            or self.content_sha256 != production_tensor_sha256(self.value)
            or not _is_sha256(self.content_sha256)
            or (
                (self.value.is_floating_point() or self.value.is_complex())
                and not bool(torch.isfinite(self.value).all().item())
            )
        ):
            raise ValueError(f"FSG4/B4-B captured tensor differs: {self.name}")

    def metadata(self) -> dict[str, object]:
        self.validate()
        return {
            "name": self.name,
            "shape": list(self.source_shape),
            "dtype": self.source_dtype,
            "device": self.source_device,
            "layout": "contiguous-strided",
            "strides": list(self.source_strides),
            "requires_grad": self.source_requires_grad,
            "content_sha256": self.content_sha256,
        }


_REQUIRED_VALUES = frozenset(
    {
        "incoming_lower_a",
        "preactivation_lower",
        "preactivation_upper",
        "production_alpha",
        "native_alpha",
        "production_beta",
        "native_beta",
        "relu_pre_add_coeff_l",
        "operator_weight",
        "output_lower_a",
        "output_bias",
        "loss_seed",
    }
)
_REQUIRED_GRADIENTS = frozenset({"native_alpha", "native_beta"})
_OPTIONAL_GRADIENTS = frozenset({"incoming_lower_a"})


@dataclass(frozen=True)
class ProductionDifferentiableRegionCaptureV1:
    """One evaluation-0 exact-call capture with explicit gradient ownership."""

    source_state_hash: str
    primal_graph_hash: str
    split_state_hash: str
    topology_hash: str
    anchor: DifferentiableRegionAnchorV1
    values: tuple[tuple[str, CapturedCudaTensorV1], ...]
    gradients: tuple[tuple[str, CapturedCudaTensorV1], ...]
    operator_attributes: tuple[tuple[str, object], ...]
    evaluation_ordinal: int = 0
    phase: str = "optimizer"
    shape_source: str = B4B_SHAPE_SOURCE
    kernel_shape_inferred: bool = False
    capture_count: int = 1
    provider_callback_count: int = 0
    fallback_dispatch_count: int = 0
    eager_backward_fallback_count: int = 0
    schema_version: str = B4B_CAPTURE_SCHEMA

    @property
    def value_map(self) -> dict[str, CapturedCudaTensorV1]:
        return dict(self.values)

    @property
    def gradient_map(self) -> dict[str, CapturedCudaTensorV1]:
        return dict(self.gradients)

    @property
    def attribute_map(self) -> dict[str, object]:
        return dict(self.operator_attributes)

    def validate(self) -> None:  # pylint: disable=too-many-branches
        self.anchor.validate()
        values = self.value_map
        gradients = self.gradient_map
        attributes = self.attribute_map
        frozen_anchors = {anchor.stable_hash() for anchor in b4b_v1_anchors()}
        if (
            self.schema_version != B4B_CAPTURE_SCHEMA
            or not _is_sha256(self.source_state_hash)
            or not _is_sha256(self.primal_graph_hash)
            or not _is_sha256(self.split_state_hash)
            or not _is_sha256(self.topology_hash)
            or self.anchor.stable_hash() not in frozen_anchors
            or set(values) != _REQUIRED_VALUES
            or len(values) != len(self.values)
            or not _REQUIRED_GRADIENTS.issubset(gradients)
            or not set(gradients).issubset(_REQUIRED_GRADIENTS | _OPTIONAL_GRADIENTS)
            or len(gradients) != len(self.gradients)
            or len(attributes) != len(self.operator_attributes)
            or self.evaluation_ordinal != 0
            or self.phase != "optimizer"
            or self.shape_source != B4B_SHAPE_SOURCE
            or self.kernel_shape_inferred is not False
            or self.capture_count != 1
            or self.provider_callback_count != 0
            or self.fallback_dispatch_count != 0
            or self.eager_backward_fallback_count != 0
            or attributes.get("operator_kind") != self.anchor.producer_op_type
            or attributes.get("weight_shape")
            != list(values["operator_weight"].source_shape)
        ):
            raise ValueError("FSG4/B4-B production capture differs")
        if self.anchor.producer_op_type == "conv2d" and not {
            "stride",
            "padding",
            "dilation",
            "groups",
        }.issubset(attributes):
            raise ValueError("FSG4/B4-B Conv attributes are incomplete")
        expected_value_shapes = {
            "incoming_lower_a": self.anchor.coefficient_shape,
            "preactivation_lower": self.anchor.preactivation_shape,
            "preactivation_upper": self.anchor.preactivation_shape,
            "production_alpha": self.anchor.production_alpha_shape,
            "native_alpha": self.anchor.native_alpha_shape,
            "production_beta": self.anchor.production_beta_shape,
            "native_beta": self.anchor.native_beta_shape,
            "relu_pre_add_coeff_l": self.anchor.native_beta_shape,
        }
        expected_gradient_shapes = {
            "native_alpha": self.anchor.native_alpha_shape,
            "native_beta": self.anchor.native_beta_shape,
        }
        if "incoming_lower_a" in gradients:
            expected_gradient_shapes["incoming_lower_a"] = self.anchor.coefficient_shape
        for name, snapshot in values.items():
            snapshot.validate()
            if name != snapshot.name:
                raise ValueError("FSG4/B4-B value tensor name differs")
            expected = expected_value_shapes.get(name)
            if expected is not None and snapshot.source_shape != expected:
                raise ValueError(f"FSG4/B4-B value tensor shape differs: {name}")
        for name, snapshot in gradients.items():
            snapshot.validate()
            if (
                name != snapshot.name
                or snapshot.source_shape != expected_gradient_shapes[name]
            ):
                raise ValueError(f"FSG4/B4-B gradient tensor differs: {name}")
        if (
            values["native_alpha"].source_requires_grad is not True
            or values["native_beta"].source_requires_grad is not True
            or values["production_alpha"].source_requires_grad is not False
            or values["production_beta"].source_requires_grad is not False
            or values["relu_pre_add_coeff_l"].source_requires_grad is not True
            or any(snapshot.source_requires_grad for snapshot in gradients.values())
            or values["incoming_lower_a"].source_requires_grad
            != ("incoming_lower_a" in gradients)
            or self.anchor.beta_must_be_nonempty
            != (math.prod(values["production_beta"].source_shape) > 0)
        ):
            raise ValueError("FSG4/B4-B gradient ownership differs")

    def metadata(self) -> dict[str, object]:
        self.validate()
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "source_state_hash": self.source_state_hash,
            "primal_graph_hash": self.primal_graph_hash,
            "split_state_hash": self.split_state_hash,
            "topology_hash": self.topology_hash,
            "anchor": self.anchor.metadata(),
            "anchor_hash": self.anchor.stable_hash(),
            "values": {
                name: snapshot.metadata() for name, snapshot in sorted(self.values)
            },
            "gradients": {
                name: snapshot.metadata() for name, snapshot in sorted(self.gradients)
            },
            "operator_attributes": dict(sorted(self.operator_attributes)),
            "evaluation_ordinal": self.evaluation_ordinal,
            "phase": self.phase,
            "shape_source": self.shape_source,
            "kernel_shape_inferred": self.kernel_shape_inferred,
            "capture_count": self.capture_count,
            "provider_callback_count": self.provider_callback_count,
            "fallback_dispatch_count": self.fallback_dispatch_count,
            "eager_backward_fallback_count": self.eager_backward_fallback_count,
        }
        payload["capture_hash"] = _canonical_hash(payload)
        return payload


def capture_production_differentiable_region_v1(
    *,
    source_state_hash: str,
    primal_graph_hash: str,
    split_state_hash: str,
    topology_hash: str,
    anchor: DifferentiableRegionAnchorV1,
    values: Mapping[str, torch.Tensor],
    gradients: Mapping[str, torch.Tensor],
    operator_attributes: Mapping[str, object],
) -> ProductionDifferentiableRegionCaptureV1:
    """Copy one live CUDA exact call into a validated immutable capture."""

    capture = ProductionDifferentiableRegionCaptureV1(
        source_state_hash=source_state_hash,
        primal_graph_hash=primal_graph_hash,
        split_state_hash=split_state_hash,
        topology_hash=topology_hash,
        anchor=anchor,
        values=tuple(
            (name, CapturedCudaTensorV1.from_tensor(name, value))
            for name, value in sorted(values.items())
        ),
        gradients=tuple(
            (name, CapturedCudaTensorV1.from_tensor(name, value))
            for name, value in sorted(gradients.items())
        ),
        operator_attributes=tuple(sorted(operator_attributes.items())),
    )
    capture.validate()
    return capture


__all__ = [
    "B4B_CAPTURE_SCHEMA",
    "B4B_PERFORMANCE_ANCHOR_V1",
    "B4B_SEMANTIC_ANCHOR_V1",
    "CapturedCudaTensorV1",
    "DifferentiableRegionAnchorV1",
    "ProductionDifferentiableRegionCaptureV1",
    "b4b_v1_anchors",
    "capture_production_differentiable_region_v1",
]
