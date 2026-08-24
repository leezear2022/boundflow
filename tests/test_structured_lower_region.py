"""R3-0 structured owner contract and fail-closed validator tests."""

# pylint: disable=missing-function-docstring,too-many-arguments

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import cast

import pytest
import torch

from boundflow.ir.structured_lower_region import (
    BiasSplitWitnessV1,
    SavedTensorEntryV1,
    SavedTensorLedgerV1,
    ScratchIntervalV1,
    StructuredCoefficientHandleV1,
    StructuredDenseEscapeError,
    StructuredLowerAttributeV1,
    StructuredLowerNodeV1,
    StructuredLowerOpKind,
    StructuredLowerR30ReceiptV1,
    StructuredLowerRegionInstanceV1,
    StructuredLowerRegionTemplateV1,
    StructuredTensorBindingV1,
    assert_tensor_free_context,
)


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
        source_op_ids=(f"source-{ordinal}",),
        declared_consumer_count=consumers,
        external_consumer_count=external,
        attributes=attributes,
    )


def _template() -> StructuredLowerRegionTemplateV1:
    nodes = (
        _node(
            "seed",
            0,
            StructuredLowerOpKind.SPEC_SEED,
            (),
            1,
            attributes=(_attribute("start_node", "25/Conv_8"),),
        ),
        _node(
            "relu",
            1,
            StructuredLowerOpKind.RELU_LOWER_TRANSFORM,
            ("seed",),
            1,
            attributes=(
                _attribute("alpha_layout", "compressed-start-node"),
                _attribute("beta_layout", "empty"),
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
            "bias",
            3,
            StructuredLowerOpKind.BIAS_SPLIT,
            ("conv",),
            2,
            attributes=(_attribute("token", "conv-bias"),),
        ),
        _node(
            "slice",
            4,
            StructuredLowerOpKind.SLICE,
            ("bias",),
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
            ("bias",),
            1,
            attributes=(_attribute("source_shape", (6, 1, 16, 8, 8)),),
        ),
        _node("add", 6, StructuredLowerOpKind.ADD, ("slice", "reshape"), 1),
        _node(
            "root",
            7,
            StructuredLowerOpKind.INPUT_CONCRETIZE,
            ("add",),
            1,
            external=1,
            attributes=(_attribute("perturbation", "linf-center-radius"),),
        ),
    )
    return StructuredLowerRegionTemplateV1(
        nodes=nodes,
        root_node_id="root",
        source_op_count=3,
        start_node_id="25/Conv_8",
        source_hash="a" * 64,
        topology_hash="b" * 64,
        lineage_hash="c" * 64,
        bias_witnesses=(
            BiasSplitWitnessV1(
                parent_node_id="bias",
                child_node_ids=("slice", "reshape"),
                numerators=(1, 1),
                denominator=2,
            ),
        ),
        scratch_intervals=(
            ScratchIntervalV1(0, 1, 2, 65536),
            ScratchIntervalV1(1, 2, 4, 65536),
            ScratchIntervalV1(0, 3, 6, 65536),
        ),
    )


def _binding(
    name: str,
    role: str,
    pointer: int,
    *,
    shape: tuple[int, ...] = (6, 1),
    requires_grad: bool = False,
) -> StructuredTensorBindingV1:
    return StructuredTensorBindingV1(
        name=name,
        role=role,
        shape=shape,
        strides=tuple(1 for _ in shape),
        dtype="torch.float32",
        device="cuda:0",
        data_ptr=pointer,
        storage_ptr=pointer,
        version=0,
        requires_grad=requires_grad,
    )


def _instance(
    template: StructuredLowerRegionTemplateV1,
) -> StructuredLowerRegionInstanceV1:
    return StructuredLowerRegionInstanceV1(
        template_hash=template.stable_hash(),
        start_node_id=template.start_node_id,
        evaluation_ordinal=0,
        mutation_ordinal=0,
        current_stream=7,
        split_history_hash="d" * 64,
        domain_hash="e" * 64,
        bindings=(
            _binding("alpha", "alpha", 1000, requires_grad=True),
            _binding("beta", "beta", 2000, shape=(6, 0), requires_grad=True),
            _binding("bounds", "bound", 3000),
            _binding("weight", "weight", 4000),
            _binding("scratch-0", "scratch", 5000),
            _binding("scratch-1", "scratch", 6000),
        ),
    )


def _ledger() -> SavedTensorLedgerV1:
    return SavedTensorLedgerV1(
        (
            SavedTensorEntryV1(
                role="alpha",
                shape=(6, 1),
                dtype="torch.float32",
                device="cuda:0",
                storage_id="alpha-storage",
                logical_bytes=24,
                version=0,
                coefficient_lineage=False,
            ),
            SavedTensorEntryV1(
                role="weight-view-a",
                shape=(16, 16, 3, 3),
                dtype="torch.float32",
                device="cuda:0",
                storage_id="weight-storage",
                logical_bytes=9216,
                version=0,
                coefficient_lineage=False,
            ),
            SavedTensorEntryV1(
                role="weight-view-b",
                shape=(16, 16, 3, 3),
                dtype="torch.float32",
                device="cuda:0",
                storage_id="weight-storage",
                logical_bytes=9216,
                version=0,
                coefficient_lineage=False,
            ),
        )
    )


def _receipt(
    template: StructuredLowerRegionTemplateV1,
    instance: StructuredLowerRegionInstanceV1,
    ledger: SavedTensorLedgerV1,
) -> StructuredLowerR30ReceiptV1:
    return StructuredLowerR30ReceiptV1(
        template_hash=template.stable_hash(),
        instance_hash=instance.stable_hash(),
        node_count=len(template.nodes),
        source_op_count=template.source_op_count,
        edge_count=sum(len(node.input_ids) for node in template.nodes),
        root_node_id=template.root_node_id,
        scratch_slot_count=2,
        saved_logical_bytes=ledger.logical_bytes,
        saved_unique_storage_bytes=ledger.unique_storage_bytes,
        saved_coefficient_bytes=0,
        dense_escape_count=0,
        context_tensor_count=0,
    )


def test_template_instance_ledger_and_receipt_round_trip() -> None:
    template = _template()
    template.validate()
    parsed_template = StructuredLowerRegionTemplateV1.from_dict(template.to_dict())
    assert parsed_template == template
    assert parsed_template.stable_hash() == template.stable_hash()

    instance = _instance(template)
    instance.validate()
    parsed_instance = StructuredLowerRegionInstanceV1.from_dict(instance.to_dict())
    assert parsed_instance == instance

    ledger = _ledger()
    parsed_ledger = SavedTensorLedgerV1.from_dict(ledger.to_dict())
    assert parsed_ledger == ledger
    assert ledger.logical_bytes == 18456
    assert ledger.unique_storage_bytes == 9240

    receipt = _receipt(template, instance, ledger)
    parsed_receipt = StructuredLowerR30ReceiptV1.from_dict(receipt.to_dict())
    parsed_receipt.validate(template=template, instance=instance, ledger=ledger)


def test_template_rejects_non_topological_input() -> None:
    template = _template()
    nodes = list(template.nodes)
    nodes[1] = replace(nodes[1], input_ids=("conv",))
    with pytest.raises(ValueError, match="topology"):
        replace(template, nodes=tuple(nodes)).validate()


def test_template_rejects_escaped_non_root_consumer() -> None:
    template = _template()
    nodes = list(template.nodes)
    nodes[2] = replace(nodes[2], external_consumer_count=1, declared_consumer_count=2)
    with pytest.raises(ValueError, match="escaped consumer"):
        replace(template, nodes=tuple(nodes)).validate()


def test_template_rejects_consumer_count_and_disconnected_node() -> None:
    template = _template()
    nodes = list(template.nodes)
    nodes[3] = replace(nodes[3], declared_consumer_count=3)
    with pytest.raises(ValueError, match="consumer count"):
        replace(template, nodes=tuple(nodes)).validate()

    orphan = _node(
        "orphan",
        8,
        StructuredLowerOpKind.LINEAR_RIGHT,
        ("seed",),
        0,
        attributes=(_attribute("weight_layout", "out-in"),),
    )
    disconnected_nodes = list(template.nodes)
    disconnected_nodes[0] = replace(disconnected_nodes[0], declared_consumer_count=2)
    disconnected_nodes.append(orphan)
    with pytest.raises(ValueError, match="outside root closure"):
        replace(template, nodes=tuple(disconnected_nodes)).validate()


def test_template_rejects_superlinear_expansion_and_hash_tamper() -> None:
    template = _template()
    with pytest.raises(ValueError, match="template differs"):
        replace(template, source_op_count=1).validate()
    with pytest.raises(ValueError, match="template differs"):
        replace(template, source_hash="0" * 63).validate()


def test_bias_split_witness_is_exact_and_conservative() -> None:
    template = _template()
    witness = template.bias_witnesses[0]
    with pytest.raises(ValueError, match="ownership"):
        replace(
            template, bias_witnesses=(replace(witness, numerators=(1, 2)),)
        ).validate()
    with pytest.raises(ValueError, match="children"):
        replace(
            template,
            bias_witnesses=(replace(witness, child_node_ids=("reshape", "slice")),),
        ).validate()
    with pytest.raises(ValueError, match="coverage"):
        replace(template, bias_witnesses=()).validate()


def test_node_schema_rejects_wrong_arity_or_attributes() -> None:
    node = _template().nodes[1]
    with pytest.raises(ValueError, match="arity"):
        replace(node, input_ids=()).validate_local()
    with pytest.raises(ValueError, match="attributes"):
        replace(
            node, attributes=(_attribute("alpha_layout", "compressed"),)
        ).validate_local()


def test_template_parser_rejects_unknown_or_extra_fields() -> None:
    payload = _template().to_dict()
    cast(dict[str, object], payload)["unexpected"] = True
    with pytest.raises(ValueError, match="fields differ"):
        StructuredLowerRegionTemplateV1.from_dict(payload)

    payload = _template().to_dict()
    node = cast(list[dict[str, object]], payload["nodes"])[1]
    node["op_kind"] = "native_dense_escape"
    with pytest.raises(ValueError, match="op kind"):
        StructuredLowerRegionTemplateV1.from_dict(payload)


def test_scratch_liveness_rejects_overlap_and_invalid_slot() -> None:
    template = _template()
    intervals = template.scratch_intervals + (ScratchIntervalV1(0, 6, 7, 1024),)
    with pytest.raises(ValueError, match="overlap"):
        replace(template, scratch_intervals=intervals).validate()
    with pytest.raises(ValueError, match="scratch interval"):
        replace(
            template,
            scratch_intervals=(ScratchIntervalV1(2, 0, 1, 1024),),
        ).validate()


def test_structured_handle_fails_closed_on_dense_escape() -> None:
    template = _template()
    handle = StructuredCoefficientHandleV1(
        template_hash=template.stable_hash(),
        root_node_id=template.root_node_id,
        output_shape=(6, 1),
    )
    handle.validate()
    with pytest.raises(StructuredDenseEscapeError, match="cannot escape"):
        handle.to_dense()


@dataclass
class _NestedContext:
    plan_key: str
    payload: object


def test_context_reachability_accepts_metadata_and_rejects_nested_tensor() -> None:
    assert_tensor_free_context(_NestedContext("plan", {"ordinals": (1, 2)}))
    with pytest.raises(ValueError, match="context tensor reachable"):
        assert_tensor_free_context(
            _NestedContext("plan", {"executor": [torch.ones(1)]})
        )


def test_context_reachability_handles_cycles() -> None:
    cyclic: list[object] = []
    cyclic.append(cyclic)
    assert_tensor_free_context(cyclic)


def test_saved_tensor_ledger_rejects_dense_coefficient_lineage() -> None:
    entry = _ledger().entries[0]
    with pytest.raises(ValueError, match="dense coefficient"):
        SavedTensorLedgerV1((replace(entry, coefficient_lineage=True),)).validate()


def test_saved_tensor_ledger_parser_recomputes_derived_bytes() -> None:
    payload = _ledger().to_dict()
    payload["unique_storage_bytes"] = 999999
    with pytest.raises(ValueError, match="derivation"):
        SavedTensorLedgerV1.from_dict(payload)


def test_instance_rejects_missing_alpha_duplicate_names_and_excess_scratch() -> None:
    template = _template()
    instance = _instance(template)
    with pytest.raises(ValueError, match="lacks compressed alpha"):
        replace(
            instance,
            bindings=tuple(
                binding for binding in instance.bindings if binding.role != "alpha"
            ),
        ).validate()
    with pytest.raises(ValueError, match="binding names"):
        replace(
            instance, bindings=instance.bindings + (instance.bindings[0],)
        ).validate()
    with pytest.raises(ValueError, match="scratch count"):
        replace(
            instance,
            bindings=instance.bindings + (_binding("scratch-2", "scratch", 7000),),
        ).validate()


def test_instance_rejects_grad_on_non_optimizer_leaf() -> None:
    binding = _binding("weight", "weight", 1000, requires_grad=True)
    with pytest.raises(ValueError, match="binding differs"):
        binding.validate()


def test_instance_parser_rejects_stale_template_hash() -> None:
    template = _template()
    instance = replace(_instance(template), template_hash="f" * 64)
    ledger = _ledger()
    receipt = _receipt(template, instance, ledger)
    with pytest.raises(ValueError, match="receipt differs"):
        receipt.validate(template=template, instance=instance, ledger=ledger)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("saved_coefficient_bytes", 4),
        ("dense_escape_count", 1),
        ("context_tensor_count", 1),
        ("production_connected", True),
        ("timing_recorded", True),
        ("performance_claimed", True),
    ),
)
def test_r30_receipt_rejects_forbidden_claims(field: str, value: object) -> None:
    template = _template()
    instance = _instance(template)
    ledger = _ledger()
    receipt = _receipt(template, instance, ledger)
    with pytest.raises(ValueError, match="receipt differs"):
        replace(receipt, **{field: value}).validate(  # type: ignore[arg-type]
            template=template,
            instance=instance,
            ledger=ledger,
        )


def test_receipt_parser_rejects_extra_field() -> None:
    template = _template()
    instance = _instance(template)
    ledger = _ledger()
    payload = _receipt(template, instance, ledger).to_dict()
    payload["latency_ms"] = 1.0
    with pytest.raises(ValueError, match="fields differ"):
        StructuredLowerR30ReceiptV1.from_dict(payload)
