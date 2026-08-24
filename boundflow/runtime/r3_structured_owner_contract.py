"""Frozen R3-0 P-anchor contract bundle and semantic replay."""

# pylint: disable=too-many-arguments,duplicate-code

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping

from boundflow.ir.structured_lower_region import (
    BiasSplitWitnessV1,
    SavedTensorEntryV1,
    SavedTensorLedgerV1,
    ScratchIntervalV1,
    StructuredCoefficientHandleV1,
    StructuredLowerAttributeV1,
    StructuredLowerNodeV1,
    StructuredLowerOpKind,
    StructuredLowerR30ReceiptV1,
    StructuredLowerRegionInstanceV1,
    StructuredLowerRegionTemplateV1,
    StructuredTensorBindingV1,
    assert_tensor_free_context,
)

BUNDLE_SCHEMA = "boundflow.r3-0-contract-bundle/v1"
SUMMARY_SCHEMA = "boundflow.r3-0-contract-summary/v1"
START_NODE_ID = "25/Conv_8"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _attribute(name: str, value: str | tuple[int, ...]) -> StructuredLowerAttributeV1:
    if isinstance(value, str):
        return StructuredLowerAttributeV1(name=name, text=value)
    return StructuredLowerAttributeV1(name=name, integers=value)


def _node(
    node_id: str,
    ordinal: int,
    kind: StructuredLowerOpKind,
    inputs: tuple[str, ...],
    consumers: int,
    *,
    external: int = 0,
    attributes: tuple[StructuredLowerAttributeV1, ...] = (),
) -> StructuredLowerNodeV1:
    return StructuredLowerNodeV1(
        node_id=node_id,
        ordinal=ordinal,
        op_kind=kind,
        input_ids=inputs,
        output_shape=(6, 1, 16, 8, 8),
        source_op_ids=(f"r3-p-anchor-source-{ordinal}",),
        declared_consumer_count=consumers,
        external_consumer_count=external,
        attributes=attributes,
    )


def build_r30_template() -> StructuredLowerRegionTemplateV1:
    """Build the frozen contract-only P-anchor DAG, including a fanout witness."""

    nodes = (
        _node(
            "seed",
            0,
            StructuredLowerOpKind.SPEC_SEED,
            (),
            1,
            attributes=(_attribute("start_node", START_NODE_ID),),
        ),
        _node(
            "relu",
            1,
            StructuredLowerOpKind.RELU_LOWER_TRANSFORM,
            ("seed",),
            1,
            attributes=(
                _attribute("alpha_layout", "compressed-start-node-keyed"),
                _attribute("beta_layout", "empty-beta-not-dense-zero"),
            ),
        ),
        _node(
            "conv",
            2,
            StructuredLowerOpKind.CONV2D_RIGHT,
            ("relu",),
            1,
            attributes=(
                _attribute("stride", (1, 1)),
                _attribute("padding", (1, 1)),
                _attribute("dilation", (1, 1)),
                _attribute("groups", (1,)),
            ),
        ),
        _node(
            "bias_split",
            3,
            StructuredLowerOpKind.BIAS_SPLIT,
            ("conv",),
            2,
            attributes=(_attribute("token", "p-anchor-bias-token"),),
        ),
        _node(
            "slice",
            4,
            StructuredLowerOpKind.SLICE,
            ("bias_split",),
            1,
            attributes=(
                _attribute("axis", (2,)),
                _attribute("start", (0,)),
                _attribute("stop", (8,)),
            ),
        ),
        _node(
            "reshape",
            5,
            StructuredLowerOpKind.RESHAPE,
            ("bias_split",),
            1,
            attributes=(_attribute("source_shape", (6, 1, 16, 8, 8)),),
        ),
        _node("add", 6, StructuredLowerOpKind.ADD, ("slice", "reshape"), 1),
        _node(
            "input_concretize",
            7,
            StructuredLowerOpKind.INPUT_CONCRETIZE,
            ("add",),
            1,
            external=1,
            attributes=(_attribute("perturbation", "linf-center-radius"),),
        ),
    )
    template = StructuredLowerRegionTemplateV1(
        nodes=nodes,
        root_node_id="input_concretize",
        source_op_count=3,
        start_node_id=START_NODE_ID,
        source_hash=_hash("r3-0-contract-only-source-v1"),
        topology_hash=_hash([node.node_id for node in nodes]),
        lineage_hash=_hash([node.source_op_ids for node in nodes]),
        bias_witnesses=(
            BiasSplitWitnessV1(
                parent_node_id="bias_split",
                child_node_ids=("slice", "reshape"),
                numerators=(1, 1),
                denominator=2,
            ),
        ),
        scratch_intervals=(
            ScratchIntervalV1(0, 1, 2, 393216),
            ScratchIntervalV1(1, 2, 4, 393216),
            ScratchIntervalV1(0, 3, 6, 393216),
        ),
    )
    template.validate()
    return template


def _binding(
    name: str,
    role: str,
    pointer: int,
    *,
    shape: tuple[int, ...],
    requires_grad: bool = False,
) -> StructuredTensorBindingV1:
    return StructuredTensorBindingV1(
        name=name,
        role=role,
        shape=shape,
        strides=tuple(1 for _ in shape),
        dtype="torch.float32",
        device="cuda:0-contract-only",
        data_ptr=pointer,
        storage_ptr=pointer,
        version=0,
        requires_grad=requires_grad,
    )


def build_r30_instance(
    template: StructuredLowerRegionTemplateV1,
) -> StructuredLowerRegionInstanceV1:
    """Build deterministic metadata bindings; no live Tensor or production state is retained."""

    instance = StructuredLowerRegionInstanceV1(
        template_hash=template.stable_hash(),
        start_node_id=START_NODE_ID,
        evaluation_ordinal=0,
        mutation_ordinal=0,
        current_stream=7,
        split_history_hash=_hash("p-anchor-empty-split-history"),
        domain_hash=_hash("p-anchor-domain-zero"),
        bindings=(
            _binding(
                "alpha", "alpha", 4096, shape=(6, 1, 16, 8, 8), requires_grad=True
            ),
            _binding("beta", "beta", 8192, shape=(6, 0), requires_grad=True),
            _binding("lower", "bound", 12288, shape=(6, 16, 8, 8)),
            _binding("upper", "bound", 16384, shape=(6, 16, 8, 8)),
            _binding("weight", "weight", 20480, shape=(16, 16, 3, 3)),
            _binding("bias", "bias", 24576, shape=(16,)),
            _binding("scratch-0", "scratch", 28672, shape=(6, 1, 16, 8, 8)),
            _binding("scratch-1", "scratch", 32768, shape=(6, 1, 16, 8, 8)),
        ),
    )
    instance.validate()
    return instance


def build_r30_ledger() -> SavedTensorLedgerV1:
    """Build the future VJP saved-state budget with zero coefficient lineage."""

    entries = (
        SavedTensorEntryV1(
            "alpha",
            (6, 1, 16, 8, 8),
            "torch.float32",
            "cuda:0",
            "alpha",
            98304,
            0,
            False,
        ),
        SavedTensorEntryV1(
            "lower", (6, 16, 8, 8), "torch.float32", "cuda:0", "bounds", 98304, 0, False
        ),
        SavedTensorEntryV1(
            "upper", (6, 16, 8, 8), "torch.float32", "cuda:0", "bounds", 98304, 0, False
        ),
        SavedTensorEntryV1(
            "weight",
            (16, 16, 3, 3),
            "torch.float32",
            "cuda:0",
            "weight",
            9216,
            0,
            False,
        ),
    )
    ledger = SavedTensorLedgerV1(entries)
    ledger.validate()
    return ledger


@dataclass(frozen=True)
class _TensorFreeContextProbe:
    """Contract-side stand-in for allowed scalar autograd context metadata."""

    plan_key: str
    schema_version: str
    alpha_ordinal: int
    beta_ordinal: int


def build_r30_bundle() -> dict[str, object]:
    """Build and validate the complete R3-0 contract bundle."""

    template = build_r30_template()
    instance = build_r30_instance(template)
    ledger = build_r30_ledger()
    assert_tensor_free_context(
        _TensorFreeContextProbe(template.stable_hash(), TEMPLATE_CONTEXT_SCHEMA, 0, 1)
    )
    handle = StructuredCoefficientHandleV1(
        template_hash=template.stable_hash(),
        root_node_id=template.root_node_id,
        output_shape=(6, 1),
    )
    handle.validate()
    receipt = StructuredLowerR30ReceiptV1(
        template_hash=template.stable_hash(),
        instance_hash=instance.stable_hash(),
        node_count=len(template.nodes),
        source_op_count=template.source_op_count,
        edge_count=sum(len(node.input_ids) for node in template.nodes),
        root_node_id=template.root_node_id,
        scratch_slot_count=len(
            {interval.slot_id for interval in template.scratch_intervals}
        ),
        saved_logical_bytes=ledger.logical_bytes,
        saved_unique_storage_bytes=ledger.unique_storage_bytes,
        saved_coefficient_bytes=0,
        dense_escape_count=0,
        context_tensor_count=0,
    )
    receipt.validate(template=template, instance=instance, ledger=ledger)
    bundle: dict[str, object] = {
        "schema_version": BUNDLE_SCHEMA,
        "template": template.to_dict(),
        "instance": instance.to_dict(),
        "saved_tensor_ledger": ledger.to_dict(),
        "receipt": receipt.to_dict(),
    }
    bundle["bundle_hash"] = _hash(bundle)
    return bundle


TEMPLATE_CONTEXT_SCHEMA = "boundflow.r3-0-autograd-context-metadata/v1"


def validate_r30_bundle(payload: Mapping[str, object]) -> dict[str, object]:
    """Parse exact fields, recompute every derived receipt and return a canonical summary."""

    expected = {
        "schema_version",
        "template",
        "instance",
        "saved_tensor_ledger",
        "receipt",
        "bundle_hash",
    }
    if set(payload) != expected or payload.get("schema_version") != BUNDLE_SCHEMA:
        raise ValueError("R3-0 bundle fields differ")
    nested_names = ("template", "instance", "saved_tensor_ledger", "receipt")
    if any(not isinstance(payload.get(name), dict) for name in nested_names):
        raise ValueError("R3-0 bundle object differs")
    unsigned = {name: payload[name] for name in payload if name != "bundle_hash"}
    if payload.get("bundle_hash") != _hash(unsigned):
        raise ValueError("R3-0 bundle hash differs")

    template = StructuredLowerRegionTemplateV1.from_dict(
        cast_mapping(payload["template"], "template")
    )
    instance = StructuredLowerRegionInstanceV1.from_dict(
        cast_mapping(payload["instance"], "instance")
    )
    ledger = SavedTensorLedgerV1.from_dict(
        cast_mapping(payload["saved_tensor_ledger"], "saved_tensor_ledger")
    )
    receipt = StructuredLowerR30ReceiptV1.from_dict(
        cast_mapping(payload["receipt"], "receipt")
    )
    receipt.validate(template=template, instance=instance, ledger=ledger)

    expected_bundle = build_r30_bundle()
    if payload != expected_bundle:
        raise ValueError("R3-0 frozen contract differs")
    summary: dict[str, object] = {
        "schema_version": SUMMARY_SCHEMA,
        "status": receipt.status,
        "bundle_hash": payload["bundle_hash"],
        "template_hash": receipt.template_hash,
        "instance_hash": receipt.instance_hash,
        "node_count": receipt.node_count,
        "edge_count": receipt.edge_count,
        "scratch_slot_count": receipt.scratch_slot_count,
        "saved_logical_bytes": receipt.saved_logical_bytes,
        "saved_unique_storage_bytes": receipt.saved_unique_storage_bytes,
        "saved_coefficient_bytes": 0,
        "dense_escape_count": 0,
        "context_tensor_count": 0,
        "production_connected": False,
        "timing_recorded": False,
        "performance_claimed": False,
        "r3_1_open": True,
    }
    summary["summary_hash"] = _hash(summary)
    return summary


def cast_mapping(value: object, name: str) -> Mapping[str, object]:
    """Narrow a decoded JSON object without accepting non-string keys."""

    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"R3-0 {name} must be an object")
    return value


__all__ = [
    "BUNDLE_SCHEMA",
    "START_NODE_ID",
    "SUMMARY_SCHEMA",
    "build_r30_bundle",
    "build_r30_instance",
    "build_r30_ledger",
    "build_r30_template",
    "cast_mapping",
    "validate_r30_bundle",
]
