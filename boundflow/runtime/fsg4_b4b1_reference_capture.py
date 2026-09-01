"""B4-B1 live capture amendment for self-contained reference replay."""

# pylint: disable=too-many-instance-attributes,missing-function-docstring
# pylint: disable=too-many-arguments,too-many-boolean-expressions,duplicate-code

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import cast, Mapping

import torch

from .fsg4_b4b_production_region_capture import (
    B4BRegionLiveObserverV1,
    CapturedCudaTensorV1,
    LiveDifferentiableRegionObservationV1,
    ProductionDifferentiableRegionCaptureV1,
    production_differentiable_region_capture_from_payload_v1,
    production_differentiable_region_capture_to_payload_v1,
)

B4B1_REFERENCE_CAPTURE_SCHEMA = "boundflow.fsg4-b4b1-reference-capture/v1"


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _shape(value: object, *, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, int) or isinstance(item, bool) for item in value
    ):
        raise TypeError(f"FSG4/B4-B1 tensor shape differs: {name}")
    return tuple(value)


def _snapshot_from_payload(
    name: str, metadata: Mapping[str, object], raw: object
) -> CapturedCudaTensorV1:
    if not isinstance(raw, torch.Tensor):
        raise TypeError(f"FSG4/B4-B1 raw tensor differs: {name}")
    snapshot = CapturedCudaTensorV1(
        name=name,
        value=raw.detach().cpu().contiguous().clone(),
        source_shape=_shape(metadata.get("shape"), name=name),
        source_dtype=str(metadata.get("dtype", "")),
        source_device=str(metadata.get("device", "")),
        source_strides=_shape(metadata.get("strides"), name=f"{name}:strides"),
        source_requires_grad=metadata.get("requires_grad") is True,
        content_sha256=str(metadata.get("content_sha256", "")),
    )
    snapshot.validate()
    return snapshot


@dataclass(frozen=True)
class LiveDifferentiableReferenceObservationV1:
    """Evaluation-zero region values plus the missing bias/output adjoints."""

    base: LiveDifferentiableRegionObservationV1
    incoming_lower_bias: torch.Tensor
    operator_bias: torch.Tensor | None
    output_lower_a_gradient: torch.Tensor
    output_bias_gradient: torch.Tensor

    @property
    def operator_bias_present(self) -> bool:
        return self.operator_bias is not None

    def validate(self) -> None:
        self.base.validate()
        batch = int(self.base.incoming_lower_a.shape[0])
        spec = int(self.base.incoming_lower_a.shape[1])
        if (
            tuple(self.incoming_lower_bias.shape) != (batch, spec)
            or tuple(self.output_lower_a_gradient.shape)
            != tuple(self.base.output_lower_a.shape)
            or tuple(self.output_bias_gradient.shape)
            != tuple(self.base.output_bias.shape)
            or (self.operator_bias is None) != (not self.operator_bias_present)
        ):
            raise ValueError("FSG4/B4-B1 reference observation shape differs")
        tensors = [
            self.incoming_lower_bias,
            self.output_lower_a_gradient,
            self.output_bias_gradient,
        ]
        if self.operator_bias is not None:
            tensors.append(self.operator_bias)
            expected = int(self.base.operator_weight.shape[0])
            if tuple(self.operator_bias.shape) != (expected,):
                raise ValueError("FSG4/B4-B1 operator bias shape differs")
        if any(
            tensor.layout != torch.strided
            or not tensor.is_contiguous()
            or not torch.is_floating_point(tensor)
            or not bool(torch.isfinite(tensor).all().item())
            for tensor in tensors
        ):
            raise ValueError("FSG4/B4-B1 reference observation tensor differs")


class B4B1RegionLiveObserverV1:
    """Explicit opt-in observer that reconnects dense lower outputs for adjoints."""

    def __init__(self) -> None:
        self._base = B4BRegionLiveObserverV1()
        self._pending: dict[str, dict[str, torch.Tensor | None]] = {}
        self._observations: tuple[LiveDifferentiableReferenceObservationV1, ...] = ()

    @property
    def observations(self) -> tuple[LiveDifferentiableReferenceObservationV1, ...]:
        return self._observations

    def begin_evaluation(
        self,
        evaluation_ordinal: int,
        *,
        native_alphas: Mapping[str, torch.Tensor],
        native_betas: Mapping[str, torch.Tensor],
        relu_pre_add_coeff_l: Mapping[str, torch.Tensor],
    ) -> None:
        if evaluation_ordinal == 0 and (self._pending or self._observations):
            raise ValueError("FSG4/B4-B1 observer evaluation 0 repeats")
        self._base.begin_evaluation(
            evaluation_ordinal,
            native_alphas=native_alphas,
            native_betas=native_betas,
            relu_pre_add_coeff_l=relu_pre_add_coeff_l,
        )

    def wants(self, native_preactivation: str) -> bool:
        return self._base.wants(native_preactivation)

    def observe_relu_input(
        self,
        native_preactivation: str,
        *,
        incoming_lower_a: torch.Tensor,
        preactivation_lower: torch.Tensor,
        preactivation_upper: torch.Tensor,
        incoming_lower_bias: torch.Tensor,
    ) -> None:
        if native_preactivation in self._pending:
            raise ValueError("FSG4/B4-B1 observer ReLU repeats")
        if tuple(incoming_lower_bias.shape) != tuple(incoming_lower_a.shape[:2]):
            raise ValueError("FSG4/B4-B1 incoming lower bias shape differs")
        self._base.observe_relu_input(
            native_preactivation,
            incoming_lower_a=incoming_lower_a,
            preactivation_lower=preactivation_lower,
            preactivation_upper=preactivation_upper,
            incoming_lower_bias=incoming_lower_bias,
        )
        self._pending[native_preactivation] = {
            "incoming_lower_bias": incoming_lower_bias.contiguous(),
        }

    def observed_incoming_lower_a(self, native_preactivation: str) -> torch.Tensor:
        return self._base.observed_incoming_lower_a(native_preactivation)

    def observe_affine_output(
        self,
        native_preactivation: str,
        *,
        operator_weight: torch.Tensor,
        operator_bias: torch.Tensor | None,
        output_lower_a: torch.Tensor,
        output_bias: torch.Tensor,
        operator_attributes: Mapping[str, object],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pending = self._pending.get(native_preactivation)
        if pending is None or "output_lower_a" in pending:
            raise ValueError("FSG4/B4-B1 observer affine ownership differs")
        lower_a = output_lower_a.contiguous()
        lower_bias = output_bias.contiguous()
        if not lower_a.requires_grad or not lower_bias.requires_grad:
            raise ValueError("FSG4/B4-B1 observer output is not differentiable")
        lower_a.retain_grad()
        lower_bias.retain_grad()
        normalized_operator_bias = (
            None if operator_bias is None else operator_bias.contiguous()
        )
        self._base.observe_affine_output(
            native_preactivation,
            operator_weight=operator_weight,
            operator_bias=normalized_operator_bias,
            output_lower_a=lower_a,
            output_bias=lower_bias,
            operator_attributes=operator_attributes,
        )
        pending.update(
            {
                "operator_bias": normalized_operator_bias,
                "output_lower_a": lower_a,
                "output_bias": lower_bias,
            }
        )
        return lower_a, lower_bias

    def complete_evaluation(self, *, loss_seed: torch.Tensor) -> None:
        self._base.complete_evaluation(loss_seed=loss_seed)
        observations: list[LiveDifferentiableReferenceObservationV1] = []

        def freeze(value: torch.Tensor) -> torch.Tensor:
            frozen = value.detach().contiguous().clone()
            if value.requires_grad:
                frozen.requires_grad_(True)
            return frozen

        for base in self._base.observations:
            pending = self._pending.get(base.anchor.native_preactivation)
            if pending is None:
                raise ValueError("FSG4/B4-B1 observer pending region is absent")
            incoming_bias = pending.get("incoming_lower_bias")
            operator_bias = pending.get("operator_bias")
            output_lower_a = pending.get("output_lower_a")
            output_bias = pending.get("output_bias")
            if (
                not isinstance(incoming_bias, torch.Tensor)
                or (
                    operator_bias is not None
                    and not isinstance(operator_bias, torch.Tensor)
                )
                or not isinstance(output_lower_a, torch.Tensor)
                or not isinstance(output_bias, torch.Tensor)
            ):
                raise ValueError("FSG4/B4-B1 observer output adjoint differs")
            output_lower_a_gradient = output_lower_a.grad
            output_bias_gradient = output_bias.grad
            if output_lower_a_gradient is None or output_bias_gradient is None:
                raise ValueError("FSG4/B4-B1 observer output adjoint differs")
            observation = LiveDifferentiableReferenceObservationV1(
                base=base,
                incoming_lower_bias=freeze(incoming_bias),
                operator_bias=(
                    None if operator_bias is None else freeze(operator_bias)
                ),
                output_lower_a_gradient=freeze(output_lower_a_gradient),
                output_bias_gradient=freeze(output_bias_gradient),
            )
            observation.validate()
            observations.append(observation)
        self._observations = tuple(observations)


@dataclass(frozen=True)
class ProductionDifferentiableReferenceCaptureV1:
    """Self-contained B4-B1 amendment layered over one approved B4-B0 capture."""

    base: ProductionDifferentiableRegionCaptureV1
    incoming_lower_bias: CapturedCudaTensorV1
    operator_bias_present: bool
    operator_bias: CapturedCudaTensorV1 | None
    output_lower_a_gradient: CapturedCudaTensorV1
    output_bias_gradient: CapturedCudaTensorV1
    mapping_tensors: tuple[tuple[str, CapturedCudaTensorV1], ...]
    reference_attributes: tuple[tuple[str, object], ...]
    schema_version: str = B4B1_REFERENCE_CAPTURE_SCHEMA

    @property
    def mapping_tensor_map(self) -> dict[str, CapturedCudaTensorV1]:
        return dict(self.mapping_tensors)

    @property
    def reference_attribute_map(self) -> dict[str, object]:
        return dict(self.reference_attributes)

    def validate(self) -> None:  # pylint: disable=too-many-branches
        self.base.validate()
        anchor = self.base.anchor
        values = self.base.value_map
        mapping = self.mapping_tensor_map
        attributes = self.reference_attribute_map
        expected_attributes = build_b4b1_reference_attributes_v1(
            self.base,
            operator_bias_present=self.operator_bias_present,
        )
        lineage_hashes = self.base.production_lineage.source_hash_map
        required_mapping_paths = set(lineage_hashes) - {
            anchor.production_alpha_path,
            anchor.production_beta_path,
        }
        if (
            self.schema_version != B4B1_REFERENCE_CAPTURE_SCHEMA
            or self.operator_bias_present != (self.operator_bias is not None)
            or len(mapping) != len(self.mapping_tensors)
            or set(mapping) != required_mapping_paths
            or len(attributes) != len(self.reference_attributes)
            or attributes != expected_attributes
            or attributes.get("operator_kind") != anchor.producer_op_type
            or attributes.get("operator_bias_present") != self.operator_bias_present
            or attributes.get("input_shape")
            != list(values["output_lower_a"].source_shape[2:])
            or attributes.get("output_shape") != list(anchor.preactivation_shape[1:])
            or tuple(self.incoming_lower_bias.source_shape)
            != tuple(values["output_bias"].source_shape)
            or self.output_lower_a_gradient.source_shape
            != values["output_lower_a"].source_shape
            or self.output_bias_gradient.source_shape
            != values["output_bias"].source_shape
        ):
            raise ValueError("FSG4/B4-B1 reference capture differs")
        snapshots = [
            self.incoming_lower_bias,
            self.output_lower_a_gradient,
            self.output_bias_gradient,
            *mapping.values(),
        ]
        if self.operator_bias is not None:
            snapshots.append(self.operator_bias)
            if self.operator_bias.source_shape != (
                int(values["operator_weight"].source_shape[0]),
            ):
                raise ValueError("FSG4/B4-B1 operator bias shape differs")
        for snapshot in snapshots:
            snapshot.validate()
            if snapshot.source_device != values["incoming_lower_a"].source_device:
                raise ValueError("FSG4/B4-B1 reference capture device differs")
        if any(
            snapshot.content_sha256 != lineage_hashes[path]
            for path, snapshot in mapping.items()
        ):
            raise ValueError("FSG4/B4-B1 sparse mapping identity differs")
        if (
            self.output_lower_a_gradient.source_requires_grad
            or self.output_bias_gradient.source_requires_grad
            or (
                self.operator_bias is not None
                and self.operator_bias.source_requires_grad
            )
        ):
            raise ValueError("FSG4/B4-B1 captured adjoint ownership differs")
        if anchor.producer_op_type == "conv2d":
            if not {
                "stride",
                "padding",
                "dilation",
                "groups",
                "output_padding",
            }.issubset(attributes):
                raise ValueError("FSG4/B4-B1 Conv reference attributes differ")
        elif "output_padding" in attributes:
            raise ValueError("FSG4/B4-B1 Linear output padding is fabricated")

    def metadata(self) -> dict[str, object]:
        self.validate()
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "base_capture_hash": self.base.metadata()["capture_hash"],
            "incoming_lower_bias": self.incoming_lower_bias.metadata(),
            "operator_bias_present": self.operator_bias_present,
            "operator_bias": (
                None if self.operator_bias is None else self.operator_bias.metadata()
            ),
            "output_gradients": {
                "output_lower_a": self.output_lower_a_gradient.metadata(),
                "output_bias": self.output_bias_gradient.metadata(),
            },
            "mapping_tensors": {
                name: snapshot.metadata()
                for name, snapshot in sorted(self.mapping_tensors)
            },
            "reference_attributes": dict(sorted(self.reference_attributes)),
            "performance_claimed": False,
            "tir_admitted": False,
        }
        payload["reference_capture_hash"] = _canonical_hash(payload)
        return payload


def build_b4b1_reference_attributes_v1(
    base: ProductionDifferentiableRegionCaptureV1,
    *,
    operator_bias_present: bool,
) -> dict[str, object]:
    """Freeze affine logical shapes, bias presence, and Conv output padding."""

    base.validate()
    values = base.value_map
    attributes = dict(base.attribute_map)
    input_shape = tuple(values["output_lower_a"].source_shape[2:])
    output_shape = tuple(base.anchor.preactivation_shape[1:])
    attributes.update(
        {
            "operator_bias_present": operator_bias_present,
            "input_shape": list(input_shape),
            "output_shape": list(output_shape),
        }
    )
    if base.anchor.producer_op_type == "conv2d":
        if len(input_shape) != 3 or len(output_shape) != 3:
            raise ValueError("FSG4/B4-B1 Conv logical shape differs")
        stride = tuple(int(value) for value in cast(list[int], attributes["stride"]))
        padding = tuple(int(value) for value in cast(list[int], attributes["padding"]))
        dilation = tuple(
            int(value) for value in cast(list[int], attributes["dilation"])
        )
        weight_shape = values["operator_weight"].source_shape
        output_padding = tuple(
            int(input_shape[axis + 1])
            - (
                (int(output_shape[axis + 1]) - 1) * stride[axis]
                - 2 * padding[axis]
                + dilation[axis] * (int(weight_shape[axis + 2]) - 1)
                + 1
            )
            for axis in range(2)
        )
        if any(
            value < 0 or value >= stride[axis]
            for axis, value in enumerate(output_padding)
        ):
            raise ValueError("FSG4/B4-B1 Conv output padding differs")
        attributes["output_padding"] = list(output_padding)
    return dict(sorted(attributes.items()))


def capture_production_differentiable_reference_v1(
    *,
    base: ProductionDifferentiableRegionCaptureV1,
    observation: LiveDifferentiableReferenceObservationV1,
    mapping_tensors: Mapping[str, torch.Tensor],
    reference_attributes: Mapping[str, object],
) -> ProductionDifferentiableReferenceCaptureV1:
    """Copy the B4-B1 sufficiency amendment into an immutable CPU payload."""

    base.validate()
    observation.validate()
    if observation.base.anchor.stable_hash() != base.anchor.stable_hash():
        raise ValueError("FSG4/B4-B1 observation anchor differs")
    expected_attributes = build_b4b1_reference_attributes_v1(
        base,
        operator_bias_present=observation.operator_bias_present,
    )
    if dict(reference_attributes) != expected_attributes:
        raise ValueError("FSG4/B4-B1 reference attributes differ")
    capture = ProductionDifferentiableReferenceCaptureV1(
        base=base,
        incoming_lower_bias=CapturedCudaTensorV1.from_tensor(
            "incoming_lower_bias", observation.incoming_lower_bias
        ),
        operator_bias_present=observation.operator_bias_present,
        operator_bias=(
            None
            if observation.operator_bias is None
            else CapturedCudaTensorV1.from_tensor(
                "operator_bias", observation.operator_bias
            )
        ),
        output_lower_a_gradient=CapturedCudaTensorV1.from_tensor(
            "output_lower_a", observation.output_lower_a_gradient
        ),
        output_bias_gradient=CapturedCudaTensorV1.from_tensor(
            "output_bias", observation.output_bias_gradient
        ),
        mapping_tensors=tuple(
            (
                name,
                CapturedCudaTensorV1.from_tensor(name, tensor.contiguous()),
            )
            for name, tensor in sorted(mapping_tensors.items())
        ),
        reference_attributes=tuple(sorted(reference_attributes.items())),
    )
    capture.validate()
    return capture


def production_differentiable_reference_capture_to_payload_v1(
    capture: ProductionDifferentiableReferenceCaptureV1,
) -> dict[str, object]:
    """Serialize the base capture and all new raw B4-B1 amendment tensors."""

    metadata = capture.metadata()
    return {
        "metadata": metadata,
        "base": production_differentiable_region_capture_to_payload_v1(capture.base),
        "incoming_lower_bias": capture.incoming_lower_bias.value.clone(),
        "operator_bias": (
            None
            if capture.operator_bias is None
            else capture.operator_bias.value.clone()
        ),
        "output_gradients": {
            "output_lower_a": capture.output_lower_a_gradient.value.clone(),
            "output_bias": capture.output_bias_gradient.value.clone(),
        },
        "mapping_tensors": {
            name: snapshot.value.clone() for name, snapshot in capture.mapping_tensors
        },
    }


def production_differentiable_reference_capture_from_payload_v1(
    payload: Mapping[str, object],
) -> ProductionDifferentiableReferenceCaptureV1:
    """Reconstruct and semantically verify a B4-B1 amendment payload."""

    metadata = payload.get("metadata")
    base_payload = payload.get("base")
    output_raw = payload.get("output_gradients")
    mapping_raw = payload.get("mapping_tensors")
    if (
        not isinstance(metadata, Mapping)
        or not isinstance(base_payload, Mapping)
        or not isinstance(output_raw, Mapping)
        or not isinstance(mapping_raw, Mapping)
    ):
        raise TypeError("FSG4/B4-B1 reference payload envelope differs")
    incoming_metadata = metadata.get("incoming_lower_bias")
    operator_metadata = metadata.get("operator_bias")
    output_metadata = metadata.get("output_gradients")
    mapping_metadata = metadata.get("mapping_tensors")
    attributes = metadata.get("reference_attributes")
    if (
        not isinstance(incoming_metadata, Mapping)
        or not isinstance(output_metadata, Mapping)
        or not isinstance(mapping_metadata, Mapping)
        or not isinstance(attributes, Mapping)
        or set(output_raw) != {"output_lower_a", "output_bias"}
        or set(output_metadata) != set(output_raw)
        or set(mapping_metadata) != set(mapping_raw)
    ):
        raise TypeError("FSG4/B4-B1 reference payload inventory differs")
    operator_raw = payload.get("operator_bias")
    if (operator_metadata is None) != (operator_raw is None):
        raise ValueError("FSG4/B4-B1 operator bias presence differs")
    base = production_differentiable_region_capture_from_payload_v1(base_payload)
    capture = ProductionDifferentiableReferenceCaptureV1(
        base=base,
        incoming_lower_bias=_snapshot_from_payload(
            "incoming_lower_bias",
            incoming_metadata,
            payload.get("incoming_lower_bias"),
        ),
        operator_bias_present=metadata.get("operator_bias_present") is True,
        operator_bias=(
            None
            if operator_metadata is None
            else _snapshot_from_payload(
                "operator_bias",
                cast(Mapping[str, object], operator_metadata),
                operator_raw,
            )
        ),
        output_lower_a_gradient=_snapshot_from_payload(
            "output_lower_a",
            cast(Mapping[str, object], output_metadata["output_lower_a"]),
            output_raw["output_lower_a"],
        ),
        output_bias_gradient=_snapshot_from_payload(
            "output_bias",
            cast(Mapping[str, object], output_metadata["output_bias"]),
            output_raw["output_bias"],
        ),
        mapping_tensors=tuple(
            (
                str(name),
                _snapshot_from_payload(
                    str(name),
                    cast(Mapping[str, object], mapping_metadata[name]),
                    raw,
                ),
            )
            for name, raw in sorted(mapping_raw.items())
        ),
        reference_attributes=tuple(sorted(attributes.items())),
        schema_version=str(metadata.get("schema_version", "")),
    )
    capture.validate()
    if capture.metadata() != dict(metadata):
        raise ValueError("FSG4/B4-B1 reference semantic replay differs")
    return capture


__all__ = [
    "B4B1_REFERENCE_CAPTURE_SCHEMA",
    "B4B1RegionLiveObserverV1",
    "LiveDifferentiableReferenceObservationV1",
    "ProductionDifferentiableReferenceCaptureV1",
    "build_b4b1_reference_attributes_v1",
    "capture_production_differentiable_reference_v1",
    "production_differentiable_reference_capture_from_payload_v1",
    "production_differentiable_reference_capture_to_payload_v1",
]
