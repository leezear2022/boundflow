"""Independent pure-PyTorch semantics for the FSG4/B4-B1 lower region."""

# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=too-many-instance-attributes,too-many-boolean-expressions
# pylint: disable=missing-function-docstring,not-callable

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import cast

import torch
import torch.nn.functional as torch_functional

from boundflow.ir.differentiable_lower_region import (
    DifferentiableLowerRegionIRV1,
    DifferentiableLowerRegionInstanceV1,
    DifferentiableTensorContractIRV1,
    canonical_ir_attributes,
)

from .fsg4_b4b1_reference_capture import ProductionDifferentiableReferenceCaptureV1
from .fsg4_b4b_production_region_capture import CapturedCudaTensorV1
from .rvir_v4_production_state import production_tensor_sha256

B4B1_REFERENCE_RECEIPT_SCHEMA = "boundflow.fsg4-b4b1-reference-receipt/v1"
B4B1_REFERENCE_ATOL = 2.0e-4
B4B1_REFERENCE_RTOL = 2.0e-4


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _contract(
    name: str, role: str, snapshot: CapturedCudaTensorV1
) -> DifferentiableTensorContractIRV1:
    snapshot.validate()
    return DifferentiableTensorContractIRV1(
        name=name,
        role=role,
        shape=snapshot.source_shape,
        dtype=snapshot.source_dtype,
        device_kind=snapshot.source_device.split(":", 1)[0],
        layout="contiguous-strided",
        strides=snapshot.source_strides,
        requires_grad=snapshot.source_requires_grad,
    )


def _reference_tensor_inventory(
    capture: ProductionDifferentiableReferenceCaptureV1,
) -> tuple[tuple[DifferentiableTensorContractIRV1, ...], tuple[tuple[str, str], ...]]:
    base = capture.base
    rows: list[tuple[str, str, CapturedCudaTensorV1]] = []
    rows.extend(
        (f"value/{name}", "runtime-value", value) for name, value in base.values
    )
    rows.extend(
        (f"production_gradient/{name}", "production-gradient-target", value)
        for name, value in base.gradients
    )
    rows.extend(
        (
            (
                "amendment/incoming_lower_bias",
                "incoming-bias",
                capture.incoming_lower_bias,
            ),
            (
                "amendment/output_lower_a_gradient",
                "output-adjoint",
                capture.output_lower_a_gradient,
            ),
            (
                "amendment/output_bias_gradient",
                "output-adjoint",
                capture.output_bias_gradient,
            ),
        )
    )
    if capture.operator_bias is not None:
        rows.append(("amendment/operator_bias", "operator-bias", capture.operator_bias))
    rows.extend(
        (f"mapping/{name}", "sparse-layout", value)
        for name, value in capture.mapping_tensors
    )
    rows.sort(key=lambda item: item[0])
    contracts = tuple(_contract(name, role, snapshot) for name, role, snapshot in rows)
    hashes = tuple((name, snapshot.content_sha256) for name, _role, snapshot in rows)
    return contracts, hashes


def build_b4b1_differentiable_lower_ir_v1(
    capture: ProductionDifferentiableReferenceCaptureV1,
) -> DifferentiableLowerRegionIRV1:
    """Compile one capture schema into a static lower-region semantic IR."""

    capture.validate()
    base = capture.base
    anchor = base.anchor
    lineage = base.production_lineage
    if lineage.beta_bias_present or lineage.beta_update_mask_present:
        raise ValueError("B4-B1 v1 does not admit beta bias/update-mask semantics")
    contracts, _hashes = _reference_tensor_inventory(capture)
    attributes = canonical_ir_attributes(capture.reference_attribute_map)
    attribute_map = dict(attributes)
    affine_input_shape = cast(tuple[int, ...], attribute_map["input_shape"])
    relu_logical_shape = cast(tuple[int, ...], attribute_map["output_shape"])
    ir = DifferentiableLowerRegionIRV1(
        anchor_id=anchor.anchor_id,
        anchor_hash=anchor.stable_hash(),
        provider_start_node=lineage.provider_start_node,
        native_preactivation=anchor.native_preactivation,
        provider_activation=anchor.provider_activation,
        provider_preactivation=anchor.provider_preactivation,
        producer_ordinal=anchor.producer_op_ordinal,
        producer_name=anchor.producer_op_name,
        operator_kind=anchor.producer_op_type,
        domain_count=int(anchor.coefficient_shape[0]),
        spec_count=int(anchor.coefficient_shape[1]),
        relu_logical_shape=relu_logical_shape,
        affine_input_shape=affine_input_shape,
        coefficient_shape=anchor.coefficient_shape,
        result_coefficient_shape=(
            int(anchor.coefficient_shape[0]),
            int(anchor.coefficient_shape[1]),
            *affine_input_shape,
        ),
        tensor_contracts=contracts,
        operator_attributes=attributes,
        alpha_feature_index_count=lineage.alpha_feature_index_count,
        alpha_spec_lookup_present=lineage.alpha_spec_lookup_present,
        beta_active=anchor.beta_must_be_nonempty,
        beta_bias_present=lineage.beta_bias_present,
        source_state_hash=base.source_state_hash,
        primal_graph_hash=base.primal_graph_hash,
        split_state_hash=base.split_state_hash,
        topology_hash=base.topology_hash,
        lineage_hash=cast(str, lineage.metadata(anchor)["lineage_hash"]),
    )
    ir.validate()
    return ir


def build_b4b1_differentiable_lower_instance_v1(
    capture: ProductionDifferentiableReferenceCaptureV1,
    ir: DifferentiableLowerRegionIRV1,
) -> DifferentiableLowerRegionInstanceV1:
    """Bind the complete raw tensor inventory to a compiled IR."""

    capture.validate()
    ir.validate()
    _contracts, hashes = _reference_tensor_inventory(capture)
    instance = DifferentiableLowerRegionInstanceV1(
        ir_hash=ir.stable_hash(),
        reference_capture_hash=cast(str, capture.metadata()["reference_capture_hash"]),
        base_capture_hash=cast(str, capture.base.metadata()["capture_hash"]),
        input_tensor_hashes=hashes,
    )
    instance.validate_against(ir)
    return instance


def _mapping_by_suffix(
    capture: ProductionDifferentiableReferenceCaptureV1, suffix: str
) -> torch.Tensor:
    matches = [
        snapshot.value
        for name, snapshot in capture.mapping_tensors
        if name.endswith(suffix)
    ]
    if len(matches) != 1:
        raise ValueError(f"B4-B1 sparse mapping inventory differs: {suffix}")
    return matches[0]


def _reconstruct_alpha(
    capture: ProductionDifferentiableReferenceCaptureV1,
    ir: DifferentiableLowerRegionIRV1,
    production_alpha: torch.Tensor,
) -> torch.Tensor:
    feature_shape_raw = _mapping_by_suffix(capture, "/feature_shape")
    feature_shape = tuple(int(value) for value in feature_shape_raw.tolist())
    if feature_shape != ir.relu_logical_shape:
        raise ValueError("B4-B1 alpha feature shape differs")
    indices = [
        snapshot.value
        for name, snapshot in capture.mapping_tensors
        if "/feature_index/" in name
    ]
    if len(indices) != ir.alpha_feature_index_count:
        raise ValueError("B4-B1 alpha index count differs")
    indices = [
        snapshot.value
        for name, snapshot in sorted(capture.mapping_tensors)
        if "/feature_index/" in name
    ]
    lookups = [
        snapshot.value
        for name, snapshot in capture.mapping_tensors
        if "/spec_lookup/" in name
    ]
    if bool(lookups) != ir.alpha_spec_lookup_present or len(lookups) > 1:
        raise ValueError("B4-B1 alpha spec lookup differs")
    if lookups and (
        lookups[0].numel() != ir.spec_count
        or not torch.equal(lookups[0], torch.zeros_like(lookups[0]))
    ):
        raise ValueError("B4-B1 alpha spec selection differs")
    if production_alpha.ndim != 4 or tuple(production_alpha.shape[:3]) != (
        2,
        ir.spec_count,
        ir.domain_count,
    ):
        raise ValueError("B4-B1 compressed alpha shape differs")
    compressed = production_alpha[ir.alpha_direction_index, ir.alpha_spec_index]
    dense = torch.zeros(
        (ir.domain_count, *feature_shape),
        dtype=production_alpha.dtype,
        device=production_alpha.device,
    )
    if indices:
        coordinate_rows = torch.stack(indices, dim=1)
        if (
            any(index.dtype != torch.int64 or index.ndim != 1 for index in indices)
            or torch.unique(coordinate_rows, dim=0).shape[0] != coordinate_rows.shape[0]
            or any(
                bool((index < 0).any().item())
                or bool((index >= feature_shape[axis]).any().item())
                for axis, index in enumerate(indices)
            )
            or tuple(compressed.shape)
            != (ir.domain_count, int(coordinate_rows.shape[0]))
        ):
            raise ValueError("B4-B1 alpha sparse coordinates differ")
        dense[(slice(None), *indices)] = compressed  # type: ignore[index]
    elif compressed.numel() == dense.numel():
        dense = compressed.reshape_as(dense)
    else:
        raise ValueError("B4-B1 dense alpha layout differs")
    return dense


def _reconstruct_beta(
    capture: ProductionDifferentiableReferenceCaptureV1,
    ir: DifferentiableLowerRegionIRV1,
    production_beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    locations = _mapping_by_suffix(capture, "/location")
    signs = _mapping_by_suffix(capture, "/sign")
    if (
        locations.dtype != torch.int64
        or signs.dtype != production_beta.dtype
        or tuple(locations.shape) != tuple(production_beta.shape)
        or tuple(signs.shape) != tuple(production_beta.shape)
        or bool(((signs != -1) & (signs != 1)).any().item())
    ):
        raise ValueError("B4-B1 beta sparse schema differs")
    dense = torch.zeros(
        (ir.domain_count, *ir.relu_logical_shape),
        dtype=production_beta.dtype,
        device=production_beta.device,
    )
    split = torch.zeros_like(dense)
    flat_beta = dense.reshape(ir.domain_count, -1)
    flat_split = split.reshape(ir.domain_count, -1)
    for domain in range(ir.domain_count):
        row = locations[domain]
        if (
            bool((row < 0).any().item())
            or bool((row >= flat_beta.shape[1]).any().item())
            or torch.unique(row).numel() != row.numel()
        ):
            raise ValueError("B4-B1 beta sparse coordinates differ")
        flat_beta[domain, row] = production_beta[domain]
        flat_split[domain, row] = signs[domain]
    if ir.beta_active != (production_beta.numel() > 0):
        raise ValueError("B4-B1 beta presence differs")
    pre_add = -dense * split if ir.beta_active else None
    return dense, pre_add


@dataclass(frozen=True)
class DifferentiableLowerReferenceResultV1:
    """Raw independent reference values and local VJP outputs."""

    native_alpha: torch.Tensor
    native_beta: torch.Tensor
    relu_lower_a: torch.Tensor
    relu_bias: torch.Tensor
    output_lower_a: torch.Tensor
    output_bias: torch.Tensor
    native_alpha_gradient: torch.Tensor
    native_beta_gradient: torch.Tensor | None
    incoming_lower_a_gradient: torch.Tensor | None
    local_vjp: torch.Tensor


def run_b4b1_pytorch_reference_v1(
    capture: ProductionDifferentiableReferenceCaptureV1,
    ir: DifferentiableLowerRegionIRV1,
    instance: DifferentiableLowerRegionInstanceV1,
    *,
    force_incoming_a_gradient: bool = False,
) -> DifferentiableLowerReferenceResultV1:
    """Execute the frozen lower-only region using only public PyTorch operations."""

    capture.validate()
    ir.validate()
    instance.validate_against(ir)
    expected_ir = build_b4b1_differentiable_lower_ir_v1(capture)
    expected_instance = build_b4b1_differentiable_lower_instance_v1(
        capture, expected_ir
    )
    if ir.to_dict() != expected_ir.to_dict() or instance.to_dict(
        ir
    ) != expected_instance.to_dict(expected_ir):
        raise ValueError("B4-B1 reference IR/instance differs from capture")
    values = capture.base.value_map
    incoming_source = values["incoming_lower_a"]
    incoming_lower_a = (
        incoming_source.value.detach()
        .clone()
        .requires_grad_(
            incoming_source.source_requires_grad or force_incoming_a_gradient
        )
    )
    production_alpha = values["production_alpha"].value.detach().clone()
    production_alpha.requires_grad_(True)
    production_beta = values["production_beta"].value.detach().clone()
    production_beta.requires_grad_(ir.beta_active)
    native_alpha = _reconstruct_alpha(capture, ir, production_alpha)
    native_beta, beta_pre_add = _reconstruct_beta(capture, ir, production_beta)

    lower = values["preactivation_lower"].value
    upper = values["preactivation_upper"].value
    if (
        tuple(lower.shape) != (ir.domain_count, *ir.relu_logical_shape)
        or tuple(upper.shape) != tuple(lower.shape)
        or bool((lower > upper).any().item())
    ):
        raise ValueError("B4-B1 preactivation interval differs")
    positive = lower >= 0
    negative = upper <= 0
    ambiguous = (~positive) & (~negative)
    denominator = (upper - lower).clamp_min(torch.finfo(lower.dtype).eps)
    upper_slope = torch.where(
        positive,
        torch.ones_like(lower),
        torch.where(negative, torch.zeros_like(lower), upper / denominator),
    )
    upper_intercept = torch.where(
        ambiguous, -lower * upper_slope, torch.zeros_like(lower)
    )
    lower_slope = torch.where(
        ambiguous,
        native_alpha.clamp(0.0, 1.0),
        torch.where(positive, torch.ones_like(lower), torch.zeros_like(lower)),
    )
    incoming_flat = incoming_lower_a.reshape(ir.domain_count, ir.spec_count, -1)
    lower_slope_flat = lower_slope.reshape(ir.domain_count, -1)
    upper_slope_flat = upper_slope.reshape(ir.domain_count, -1)
    upper_intercept_flat = upper_intercept.reshape(ir.domain_count, -1)
    selected_slope = torch.where(
        incoming_flat >= 0,
        lower_slope_flat.unsqueeze(1),
        upper_slope_flat.unsqueeze(1),
    )
    selected_intercept = torch.where(
        incoming_flat >= 0,
        torch.zeros_like(upper_intercept_flat).unsqueeze(1),
        upper_intercept_flat.unsqueeze(1),
    )
    relu_flat = incoming_flat * selected_slope
    if beta_pre_add is not None:
        relu_flat = relu_flat + beta_pre_add.reshape(ir.domain_count, -1).unsqueeze(1)
    relu_bias = capture.incoming_lower_bias.value + (
        incoming_flat * selected_intercept
    ).sum(dim=2)
    relu_lower_a = relu_flat.reshape(ir.coefficient_shape)

    attributes = ir.operator_attribute_map
    weight = values["operator_weight"].value
    operator_bias = (
        None if capture.operator_bias is None else capture.operator_bias.value
    )
    if ir.operator_kind == "linear":
        output_lower_a = torch.matmul(relu_lower_a, weight)
        output_bias = relu_bias
        if operator_bias is not None:
            output_bias = output_bias + (
                relu_lower_a * operator_bias.reshape(1, 1, -1)
            ).sum(dim=2)
    else:
        batch_spec = ir.domain_count * ir.spec_count
        dense_input = relu_lower_a.reshape(batch_spec, *ir.relu_logical_shape)
        output_lower_a = torch_functional.conv_transpose2d(
            dense_input,
            weight,
            bias=None,
            stride=cast(tuple[int, int], attributes["stride"]),
            padding=cast(tuple[int, int], attributes["padding"]),
            output_padding=cast(tuple[int, int], attributes["output_padding"]),
            groups=cast(int, attributes["groups"]),
            dilation=cast(tuple[int, int], attributes["dilation"]),
        ).reshape(ir.result_coefficient_shape)
        output_bias = relu_bias
        if operator_bias is not None:
            bias_map = operator_bias.reshape(1, 1, -1, 1, 1)
            output_bias = output_bias + (relu_lower_a * bias_map).flatten(2).sum(2)
    if tuple(output_lower_a.shape) != ir.result_coefficient_shape or tuple(
        output_bias.shape
    ) != (ir.domain_count, ir.spec_count):
        raise ValueError("B4-B1 reference output shape differs")

    local_vjp = (output_lower_a * capture.output_lower_a_gradient.value).sum() + (
        output_bias * capture.output_bias_gradient.value
    ).sum()
    gradient_targets: list[torch.Tensor] = [native_alpha]
    if ir.beta_active:
        gradient_targets.append(native_beta)
    incoming_gradient_eligible = incoming_lower_a.requires_grad
    if incoming_gradient_eligible:
        gradient_targets.append(incoming_lower_a)
    gradients = torch.autograd.grad(local_vjp, gradient_targets)
    gradient_index = 0
    alpha_gradient = gradients[gradient_index]
    gradient_index += 1
    beta_gradient: torch.Tensor | None = None
    if ir.beta_active:
        beta_gradient = gradients[gradient_index]
        gradient_index += 1
    incoming_gradient = (
        gradients[gradient_index] if incoming_gradient_eligible else None
    )
    result = DifferentiableLowerReferenceResultV1(
        native_alpha=native_alpha.detach().contiguous(),
        native_beta=native_beta.detach().contiguous(),
        relu_lower_a=relu_lower_a.detach().contiguous(),
        relu_bias=relu_bias.detach().contiguous(),
        output_lower_a=output_lower_a.detach().contiguous(),
        output_bias=output_bias.detach().contiguous(),
        native_alpha_gradient=alpha_gradient.detach().contiguous(),
        native_beta_gradient=(
            None if beta_gradient is None else beta_gradient.detach().contiguous()
        ),
        incoming_lower_a_gradient=(
            None
            if incoming_gradient is None
            else incoming_gradient.detach().contiguous()
        ),
        local_vjp=local_vjp.detach().contiguous(),
    )
    return result


@dataclass(frozen=True)
class ReferenceParityMetricV1:
    """One deterministic tensor comparison in the reference receipt."""

    name: str
    element_count: int
    maximum_absolute_difference: float
    allclose: bool
    sign_exact: bool
    reference_hash: str
    production_hash: str

    def validate(self) -> None:
        if (
            not self.name
            or self.element_count < 1
            or not math.isfinite(self.maximum_absolute_difference)
            or self.maximum_absolute_difference < 0.0
            or len(self.reference_hash) != 64
            or len(self.production_hash) != 64
        ):
            raise ValueError("B4-B1 reference parity metric differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "name": self.name,
            "element_count": self.element_count,
            "maximum_absolute_difference": self.maximum_absolute_difference,
            "allclose": self.allclose,
            "sign_exact": self.sign_exact,
            "reference_hash": self.reference_hash,
            "production_hash": self.production_hash,
        }


@dataclass(frozen=True)
class DifferentiableLowerReferenceReceiptV1:
    """Hash-bound parity decision for one IR/instance reference execution."""

    ir_hash: str
    instance_hash: str
    reference_capture_hash: str
    anchor_id: str
    metrics: tuple[ReferenceParityMetricV1, ...]
    beta_gradient_present: bool
    incoming_lower_a_gradient_present: bool
    semantic_passed: bool
    atol: float = B4B1_REFERENCE_ATOL
    rtol: float = B4B1_REFERENCE_RTOL
    performance_claimed: bool = False
    tir_admitted: bool = False
    schema_version: str = B4B1_REFERENCE_RECEIPT_SCHEMA

    def validate(
        self,
        ir: DifferentiableLowerRegionIRV1,
        instance: DifferentiableLowerRegionInstanceV1,
    ) -> None:
        ir.validate()
        instance.validate_against(ir)
        for metric in self.metrics:
            metric.validate()
        if (
            self.schema_version != B4B1_REFERENCE_RECEIPT_SCHEMA
            or self.ir_hash != ir.stable_hash()
            or self.instance_hash != instance.stable_hash(ir)
            or self.reference_capture_hash != instance.reference_capture_hash
            or self.anchor_id != ir.anchor_id
            or len({metric.name for metric in self.metrics}) != len(self.metrics)
            or tuple(sorted(metric.name for metric in self.metrics))
            != tuple(metric.name for metric in self.metrics)
            or self.beta_gradient_present != ir.beta_active
            or self.semantic_passed
            != all(metric.allclose and metric.sign_exact for metric in self.metrics)
            or self.atol != B4B1_REFERENCE_ATOL
            or self.rtol != B4B1_REFERENCE_RTOL
            or self.performance_claimed is not False
            or self.tir_admitted is not False
        ):
            raise ValueError("B4-B1 reference receipt differs")

    def to_dict(
        self,
        ir: DifferentiableLowerRegionIRV1,
        instance: DifferentiableLowerRegionInstanceV1,
    ) -> dict[str, object]:
        self.validate(ir, instance)
        return {
            "schema_version": self.schema_version,
            "ir_hash": self.ir_hash,
            "instance_hash": self.instance_hash,
            "reference_capture_hash": self.reference_capture_hash,
            "anchor_id": self.anchor_id,
            "metrics": [metric.to_dict() for metric in self.metrics],
            "beta_gradient_present": self.beta_gradient_present,
            "incoming_lower_a_gradient_present": self.incoming_lower_a_gradient_present,
            "semantic_passed": self.semantic_passed,
            "atol": self.atol,
            "rtol": self.rtol,
            "performance_claimed": self.performance_claimed,
            "tir_admitted": self.tir_admitted,
        }

    def stable_hash(
        self,
        ir: DifferentiableLowerRegionIRV1,
        instance: DifferentiableLowerRegionInstanceV1,
    ) -> str:
        return _canonical_hash(self.to_dict(ir, instance))


def _metric(
    name: str, reference: torch.Tensor, production: torch.Tensor
) -> ReferenceParityMetricV1:
    if (
        reference.shape != production.shape
        or reference.dtype != production.dtype
        or not bool(torch.isfinite(reference).all().item())
        or not bool(torch.isfinite(production).all().item())
    ):
        raise ValueError(f"B4-B1 reference comparison schema differs: {name}")
    difference = float((reference - production).abs().max().item())
    return ReferenceParityMetricV1(
        name=name,
        element_count=reference.numel(),
        maximum_absolute_difference=difference,
        allclose=bool(
            torch.allclose(
                reference,
                production,
                atol=B4B1_REFERENCE_ATOL,
                rtol=B4B1_REFERENCE_RTOL,
            )
        ),
        sign_exact=bool(torch.equal(torch.sign(reference), torch.sign(production))),
        reference_hash=production_tensor_sha256(reference),
        production_hash=production_tensor_sha256(production),
    )


def build_b4b1_reference_receipt_v1(
    capture: ProductionDifferentiableReferenceCaptureV1,
    ir: DifferentiableLowerRegionIRV1,
    instance: DifferentiableLowerRegionInstanceV1,
    result: DifferentiableLowerReferenceResultV1,
) -> DifferentiableLowerReferenceReceiptV1:
    """Compare all eligible production values/gradients and freeze the decision."""

    values = capture.base.value_map
    gradients = capture.base.gradient_map
    rows = [
        _metric("native_alpha", result.native_alpha, values["native_alpha"].value),
        _metric("native_beta", result.native_beta, values["native_beta"].value),
        _metric(
            "native_alpha_gradient",
            result.native_alpha_gradient,
            gradients["native_alpha"].value,
        ),
        _metric("output_bias", result.output_bias, values["output_bias"].value),
        _metric(
            "output_lower_a", result.output_lower_a, values["output_lower_a"].value
        ),
    ]
    if ir.beta_active:
        if result.native_beta_gradient is None or "native_beta" not in gradients:
            raise ValueError("B4-B1 beta reference gradient is absent")
        rows.append(
            _metric(
                "native_beta_gradient",
                result.native_beta_gradient,
                gradients["native_beta"].value,
            )
        )
    elif result.native_beta_gradient is not None or "native_beta" in gradients:
        raise ValueError("B4-B1 empty beta gradient is fabricated")
    incoming_target = gradients.get("incoming_lower_a")
    if incoming_target is not None:
        if result.incoming_lower_a_gradient is None:
            raise ValueError("B4-B1 incoming-A reference gradient is absent")
        rows.append(
            _metric(
                "incoming_lower_a_gradient",
                result.incoming_lower_a_gradient,
                incoming_target.value,
            )
        )
    elif result.incoming_lower_a_gradient is not None:
        raise ValueError("B4-B1 ineligible incoming-A gradient is fabricated")
    rows.sort(key=lambda item: item.name)
    receipt = DifferentiableLowerReferenceReceiptV1(
        ir_hash=ir.stable_hash(),
        instance_hash=instance.stable_hash(ir),
        reference_capture_hash=instance.reference_capture_hash,
        anchor_id=ir.anchor_id,
        metrics=tuple(rows),
        beta_gradient_present=result.native_beta_gradient is not None,
        incoming_lower_a_gradient_present=result.incoming_lower_a_gradient is not None,
        semantic_passed=all(row.allclose and row.sign_exact for row in rows),
    )
    receipt.validate(ir, instance)
    return receipt


__all__ = [
    "B4B1_REFERENCE_ATOL",
    "B4B1_REFERENCE_RECEIPT_SCHEMA",
    "B4B1_REFERENCE_RTOL",
    "DifferentiableLowerReferenceReceiptV1",
    "DifferentiableLowerReferenceResultV1",
    "ReferenceParityMetricV1",
    "build_b4b1_differentiable_lower_instance_v1",
    "build_b4b1_differentiable_lower_ir_v1",
    "build_b4b1_reference_receipt_v1",
    "run_b4b1_pytorch_reference_v1",
]
