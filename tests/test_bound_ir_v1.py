"""Contracts for the first-class Bound IR v1 schema."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import ast
from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

import boundflow.ir.bound as bound_ir
from boundflow.domains.interval import IntervalState
from boundflow.ir.bound import (
    BOUND_IR_SCHEMA_VERSION,
    BFBoundGraph,
    BFBoundModule,
    BatchAxisKind,
    BoundBatchAxis,
    BoundDomainConfig,
    BoundMethodKind,
    BoundOp,
    BoundOpKind,
    BoundPolarity,
    BoundRepresentation,
    BoundTensorType,
    BoundValue,
    BoundValueRole,
    InputBindAttrs,
    NoBoundOpAttrs,
    ObjectiveKind,
    ObjectiveSpec,
    PerturbationKind,
    PerturbationSpec,
    ReluRelaxationAttrs,
    RepresentationChangeAttrs,
    ReshapeAttrs,
    VerificationSpec,
)


def _tensor_type(
    shape: tuple[int | None, ...] = (2, 4),
    *,
    dtype: str = "float32",
) -> BoundTensorType:
    return BoundTensorType(
        shape=shape,
        dtype=dtype,
        layout="contiguous",
        device="cuda",
        batch_axes=(BoundBatchAxis(BatchAxisKind.SPEC, 0),),
    )


def _value(
    value_id: str,
    *,
    representation: BoundRepresentation,
    polarity: BoundPolarity = BoundPolarity.LOWER,
    role: BoundValueRole = BoundValueRole.COEFFICIENT,
    tensor_type: BoundTensorType | None = None,
) -> BoundValue:
    return BoundValue(
        value_id=value_id,
        tensor_type=_tensor_type() if tensor_type is None else tensor_type,
        role=role,
        polarity=polarity,
        representation=representation,
        state_version="root-v1",
        source_primal_value_id="relu0",
    )


def _spec() -> VerificationSpec:
    return VerificationSpec(
        perturbations=(
            PerturbationSpec(
                perturbation_id="input-linf",
                input_primal_value_id="input",
                kind=PerturbationKind.LINF,
                radius=0.1,
            ),
        ),
        objectives=(
            ObjectiveSpec(
                objective_id="output-identity",
                output_primal_value_id="output",
                kind=ObjectiveKind.IDENTITY,
                num_objectives=2,
            ),
        ),
        requested_bounds=(BoundPolarity.LOWER,),
        numeric_policy="fp32_reference",
    )


def _materialization_module() -> BFBoundModule:
    structured = _value("a_structured", representation=BoundRepresentation.STRUCTURED)
    dense = _value("a_dense", representation=BoundRepresentation.DENSE)
    graph = BFBoundGraph(
        values=(structured, dense),
        ops=(
            BoundOp(
                op_id="materialize_relu0",
                kind=BoundOpKind.MATERIALIZE,
                inputs=(structured.value_id,),
                outputs=(dense.value_id,),
                attrs=RepresentationChangeAttrs(
                    source=BoundRepresentation.STRUCTURED,
                    target=BoundRepresentation.DENSE,
                    reason="dense_reference_boundary",
                ),
            ),
        ),
        inputs=(structured.value_id,),
        outputs=(dense.value_id,),
    )
    return BFBoundModule(
        module_id="plain-crown-materialize",
        primal_graph_hash="primal-sha256",
        spec=_spec(),
        domain=BoundDomainConfig(method=BoundMethodKind.CROWN),
        graph=graph,
    )


def test_bound_ir_v1_dump_and_hash_are_deterministic() -> None:
    module = _materialization_module()
    independent_module = _materialization_module()

    module.validate()
    payload = module.to_dict()
    encoded = module.canonical_json()

    assert payload["schema_version"] == BOUND_IR_SCHEMA_VERSION
    assert json.loads(encoded) == payload
    assert encoded == module.canonical_json()
    assert encoded == independent_module.canonical_json()
    assert module.stable_hash() == independent_module.stable_hash()
    assert len(module.stable_hash()) == 64


def test_bound_ir_v1_rejects_duplicate_value_and_op_ids() -> None:
    module = _materialization_module()
    structured, dense = module.graph.values
    op = module.graph.ops[0]

    duplicate_value_graph = replace(
        module.graph,
        values=(structured, replace(structured), dense),
    )
    with pytest.raises(ValueError, match="duplicate value IDs"):
        duplicate_value_graph.validate()

    duplicate_op_graph = replace(module.graph, ops=(op, replace(op)))
    with pytest.raises(ValueError, match="duplicate op IDs"):
        duplicate_op_graph.validate()


def test_bound_ir_v1_rejects_unknown_reference_and_use_before_definition() -> None:
    module = _materialization_module()
    structured, _ = module.graph.values
    unknown = replace(module.graph.ops[0], inputs=("missing",))

    with pytest.raises(ValueError, match="unknown value 'missing'"):
        replace(module.graph, ops=(unknown,)).validate()

    relu_value = replace(
        structured,
        value_id="relu_value",
        representation=BoundRepresentation.STRUCTURED,
    )
    final_value = replace(relu_value, value_id="final_value")
    first = BoundOp(
        op_id="uses_future",
        kind=BoundOpKind.RELU_RELAXATION,
        inputs=(relu_value.value_id,),
        outputs=(final_value.value_id,),
        attrs=ReluRelaxationAttrs(primal_node_id="relu1"),
    )
    second = BoundOp(
        op_id="defines_future",
        kind=BoundOpKind.RELU_RELAXATION,
        inputs=(structured.value_id,),
        outputs=(relu_value.value_id,),
        attrs=ReluRelaxationAttrs(primal_node_id="relu0"),
    )
    graph = BFBoundGraph(
        values=(structured, relu_value, final_value),
        ops=(first, second),
        inputs=(structured.value_id,),
        outputs=(final_value.value_id,),
    )
    with pytest.raises(ValueError, match="uses 'relu_value' before definition"):
        graph.validate()


def test_materialization_preserves_semantics_and_targets_dense() -> None:
    module = _materialization_module()
    structured, dense = module.graph.values
    op = module.graph.ops[0]

    changed_polarity = replace(dense, polarity=BoundPolarity.UPPER)
    with pytest.raises(ValueError, match="representation change alters semantics"):
        replace(module.graph, values=(structured, changed_polarity)).validate()

    still_structured = replace(
        dense,
        representation=BoundRepresentation.STRUCTURED,
    )
    source_dense = replace(structured, representation=BoundRepresentation.DENSE)
    structured_target_op = replace(
        op,
        attrs=RepresentationChangeAttrs(
            source=BoundRepresentation.DENSE,
            target=BoundRepresentation.STRUCTURED,
            reason="invalid_materialize_direction",
        ),
        inputs=(source_dense.value_id,),
        outputs=(still_structured.value_id,),
    )
    graph = BFBoundGraph(
        values=(source_dense, still_structured),
        ops=(structured_target_op,),
        inputs=(source_dense.value_id,),
        outputs=(still_structured.value_id,),
    )
    with pytest.raises(ValueError, match="materialize must produce a dense"):
        graph.validate()


def test_bound_ir_v1_accepts_explicit_fanout_and_residual_merge() -> None:
    source = _value("source", representation=BoundRepresentation.DENSE)
    left = replace(source, value_id="left")
    right = replace(source, value_id="right")
    merged = replace(source, value_id="merged")
    graph = BFBoundGraph(
        values=(source, left, right, merged),
        ops=(
            BoundOp(
                op_id="left_relu",
                kind=BoundOpKind.RELU_RELAXATION,
                inputs=(source.value_id,),
                outputs=(left.value_id,),
                attrs=ReluRelaxationAttrs(primal_node_id="left_relu"),
            ),
            BoundOp(
                op_id="right_relu",
                kind=BoundOpKind.RELU_RELAXATION,
                inputs=(source.value_id,),
                outputs=(right.value_id,),
                attrs=ReluRelaxationAttrs(primal_node_id="right_relu"),
            ),
            BoundOp(
                op_id="residual_add",
                kind=BoundOpKind.ADD,
                inputs=(left.value_id, right.value_id),
                outputs=(merged.value_id,),
                attrs=NoBoundOpAttrs(),
            ),
        ),
        inputs=(source.value_id,),
        outputs=(merged.value_id,),
    )

    graph.validate()
    assert graph.to_dict()["outputs"] == ["merged"]


def test_bound_ir_v1_rejects_polarity_and_tensor_type_mismatch() -> None:
    source = _value("source", representation=BoundRepresentation.DENSE)
    wrong_polarity = replace(
        source, value_id="wrong_polarity", polarity=BoundPolarity.UPPER
    )
    wrong_dtype = replace(
        source,
        value_id="wrong_dtype",
        tensor_type=_tensor_type(dtype="float64"),
    )

    polarity_op = BoundOp(
        op_id="polarity_relu",
        kind=BoundOpKind.RELU_RELAXATION,
        inputs=(source.value_id,),
        outputs=(wrong_polarity.value_id,),
        attrs=ReluRelaxationAttrs(primal_node_id="relu0"),
    )
    with pytest.raises(ValueError, match="changes lower/upper polarity"):
        BFBoundGraph(
            values=(source, wrong_polarity),
            ops=(polarity_op,),
            inputs=(source.value_id,),
            outputs=(wrong_polarity.value_id,),
        ).validate()

    dtype_op = replace(polarity_op, outputs=(wrong_dtype.value_id,))
    with pytest.raises(ValueError, match="matching tensor types"):
        BFBoundGraph(
            values=(source, wrong_dtype),
            ops=(dtype_op,),
            inputs=(source.value_id,),
            outputs=(wrong_dtype.value_id,),
        ).validate()


def test_bound_ir_v1_rejects_invalid_reshape_and_batch_axes() -> None:
    source = _value("source", representation=BoundRepresentation.DENSE)
    target = replace(
        source,
        value_id="target",
        tensor_type=_tensor_type((2, 3)),
    )
    reshape = BoundOp(
        op_id="bad_reshape",
        kind=BoundOpKind.RESHAPE,
        inputs=(source.value_id,),
        outputs=(target.value_id,),
        attrs=ReshapeAttrs(target_shape=(2, 3)),
    )
    with pytest.raises(ValueError, match="static element count"):
        BFBoundGraph(
            values=(source, target),
            ops=(reshape,),
            inputs=(source.value_id,),
            outputs=(target.value_id,),
        ).validate()

    invalid_axis_type = BoundTensorType(
        shape=(2, 4),
        dtype="float32",
        batch_axes=(
            BoundBatchAxis(BatchAxisKind.SPEC, 0),
            BoundBatchAxis(BatchAxisKind.SPEC, 1),
        ),
    )
    with pytest.raises(ValueError, match="duplicate batch-axis kinds"):
        invalid_axis_type.validate()

    out_of_rank = BoundTensorType(
        shape=(2, 4),
        dtype="float32",
        batch_axes=(BoundBatchAxis(BatchAxisKind.DOMAIN, 2),),
    )
    with pytest.raises(ValueError, match="outside rank"):
        out_of_rank.validate()


def test_bound_ir_v1_rejects_wrong_attrs_and_illegal_method_state() -> None:
    module = _materialization_module()
    op = replace(module.graph.ops[0], attrs=NoBoundOpAttrs())
    with pytest.raises(ValueError, match="expects RepresentationChangeAttrs"):
        replace(module.graph, ops=(op,)).validate()

    illegal_domain = BoundDomainConfig(
        method=BoundMethodKind.CROWN,
        alpha_enabled=True,
    )
    with pytest.raises(ValueError, match="cannot carry alpha/beta/split"):
        illegal_domain.validate()

    incomplete_alpha_beta = BoundDomainConfig(
        method=BoundMethodKind.ALPHA_BETA_CROWN,
        alpha_enabled=True,
        beta_enabled=False,
    )
    with pytest.raises(ValueError, match="requires alpha and beta"):
        incomplete_alpha_beta.validate()


def test_bound_ir_v1_resolves_bindings_against_typed_spec() -> None:
    perturbation = _value(
        "perturbation",
        representation=BoundRepresentation.DENSE,
        role=BoundValueRole.PERTURBATION,
    )
    interval = _value(
        "interval",
        representation=BoundRepresentation.DENSE,
        role=BoundValueRole.INTERVAL,
    )
    bind = BoundOp(
        op_id="input_bind",
        kind=BoundOpKind.INPUT_BIND,
        inputs=(perturbation.value_id,),
        outputs=(interval.value_id,),
        attrs=InputBindAttrs(
            primal_value_id="input",
            perturbation_id="missing-perturbation",
        ),
    )
    module = BFBoundModule(
        module_id="input-bind",
        primal_graph_hash="primal-sha256",
        spec=_spec(),
        domain=BoundDomainConfig(method=BoundMethodKind.CROWN),
        graph=BFBoundGraph(
            values=(perturbation, interval),
            ops=(bind,),
            inputs=(perturbation.value_id,),
            outputs=(interval.value_id,),
        ),
    )

    with pytest.raises(ValueError, match="unknown perturbation"):
        module.validate()

    valid_bind = replace(
        bind,
        attrs=InputBindAttrs(
            primal_value_id="input",
            perturbation_id="input-linf",
        ),
    )
    replace(module, graph=replace(module.graph, ops=(valid_bind,))).validate()


def test_bound_ir_v1_rejects_non_finite_spec_and_ambiguous_requested_bounds() -> None:
    bad_perturbation = replace(_spec().perturbations[0], radius=float("nan"))
    with pytest.raises(ValueError, match="finite and non-negative"):
        replace(_spec(), perturbations=(bad_perturbation,)).validate()

    ambiguous = replace(
        _spec(),
        requested_bounds=(BoundPolarity.BOTH, BoundPolarity.LOWER),
    )
    with pytest.raises(ValueError, match="BOTH cannot be combined"):
        ambiguous.validate()


def test_domain_state_compatibility_is_preserved() -> None:
    interval = IntervalState(
        lower=torch.tensor([0.0], dtype=torch.float32),
        upper=torch.tensor([1.0], dtype=torch.float32),
    )

    interval.validate()
    assert isinstance(interval, bound_ir.DomainState)


def test_bound_ir_module_has_no_runtime_or_backend_dependency() -> None:
    source_path = Path(bound_ir.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    forbidden = {"torch", "tvm", "boundflow.runtime", "boundflow.backends"}
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)

    assert not {
        module
        for module in imported
        if any(module == name or module.startswith(f"{name}.") for name in forbidden)
    }
