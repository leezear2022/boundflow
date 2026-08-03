"""Verified representation rewrites for Bound IR v1."""

# pylint: disable=duplicate-code

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from .bound import (
    BFBoundGraph,
    BFBoundModule,
    BoundOp,
    BoundOpKind,
    BoundRepresentation,
    BoundValue,
    BoundValueRole,
    RepresentationChangeAttrs,
)

_STRUCTURED_AFFINE_OPS = frozenset(
    {
        BoundOpKind.LINEAR_BACKWARD,
        BoundOpKind.CONV2D_BACKWARD,
        BoundOpKind.COEFFICIENT_COMPOSE,
        BoundOpKind.ADD_BACKWARD,
        BoundOpKind.CONCAT_BACKWARD,
        BoundOpKind.RESHAPE,
    }
)

_DENSE_BOUNDARY_OPS = frozenset(
    {
        BoundOpKind.RELU_RELAXATION,
        BoundOpKind.CONCRETIZE,
    }
)


def rewrite_plain_crown_structured_regions(module: BFBoundModule) -> BFBoundModule:
    """Wrap maximal affine regions in explicit dense/structured transitions.

    ReLU remains a dense semantic boundary in v1 because its sign-dependent
    relaxation consumes coefficient values. Affine, routing, concat, reshape,
    and contribution-compose operations execute with structured coefficients.
    """

    module.validate()
    _validate_dense_plain_crown_source(module)
    original_values = {value.value_id: value for value in module.graph.values}
    rewritten_values: list[BoundValue] = [
        original_values[value_id] for value_id in module.graph.inputs
    ]
    rewritten_ops: list[BoundOp] = []
    current_values = dict(original_values)
    serial = 0

    def transition_inputs(
        inputs: Sequence[str],
        *,
        target: BoundRepresentation,
        materialize: bool,
        consumer_id: str,
    ) -> tuple[str, ...]:
        nonlocal serial
        transitioned = list(inputs)
        for index, value_id in enumerate(inputs):
            if index % 4 not in {0, 2}:
                continue
            source = current_values[value_id]
            if source.role != BoundValueRole.COEFFICIENT:
                if source.role == BoundValueRole.SPLIT:
                    continue
                raise ValueError(f"affine port '{value_id}' is not a coefficient value")
            if source.representation == target:
                continue
            if BoundRepresentation.DENSE not in (source.representation, target):
                raise NotImplementedError(
                    "Bound IR v1 rewrite only transitions through dense"
                )
            serial += 1
            output_id = f"{value_id}.{target.value}.{serial:04d}"
            output = replace(source, value_id=output_id, representation=target)
            kind = (
                BoundOpKind.MATERIALIZE
                if materialize
                else BoundOpKind.REPRESENTATION_CAST
            )
            transition = BoundOp(
                op_id=f"{kind.value}.{consumer_id}.{serial:04d}",
                kind=kind,
                inputs=(value_id,),
                outputs=(output_id,),
                attrs=RepresentationChangeAttrs(
                    source=source.representation,
                    target=target,
                    reason=f"plain_crown_v1_boundary:{consumer_id}",
                ),
            )
            rewritten_values.append(output)
            rewritten_ops.append(transition)
            current_values[output_id] = output
            transitioned[index] = output_id
        return tuple(transitioned)

    for op in module.graph.ops:
        inputs = op.inputs
        output_representation: BoundRepresentation | None = None
        if op.kind in _STRUCTURED_AFFINE_OPS:
            inputs = transition_inputs(
                inputs,
                target=BoundRepresentation.STRUCTURED,
                materialize=False,
                consumer_id=op.op_id,
            )
            output_representation = BoundRepresentation.STRUCTURED
        elif op.kind in _DENSE_BOUNDARY_OPS:
            inputs = transition_inputs(
                inputs,
                target=BoundRepresentation.DENSE,
                materialize=True,
                consumer_id=op.op_id,
            )
            output_representation = BoundRepresentation.DENSE

        outputs: list[str] = []
        for index, output_id in enumerate(op.outputs):
            output = original_values[output_id]
            if (
                output_representation is not None
                and index % 4 in {0, 2}
                and output.role == BoundValueRole.COEFFICIENT
            ):
                output = replace(output, representation=output_representation)
            rewritten_values.append(output)
            current_values[output_id] = output
            outputs.append(output_id)
        rewritten_ops.append(replace(op, inputs=tuple(inputs), outputs=tuple(outputs)))

    rewritten = replace(
        module,
        module_id=f"{module.module_id}.structured-regions-v1",
        graph=BFBoundGraph(
            values=tuple(rewritten_values),
            ops=tuple(rewritten_ops),
            inputs=module.graph.inputs,
            outputs=module.graph.outputs,
        ),
    )
    rewritten.validate()
    return rewritten


def _validate_dense_plain_crown_source(module: BFBoundModule) -> None:
    """Reject graphs outside the deterministic v1 rewrite source contract."""

    if any(
        value.role == BoundValueRole.COEFFICIENT
        and value.representation != BoundRepresentation.DENSE
        for value in module.graph.values
    ):
        raise ValueError("structured-region rewrite expects a dense source module")
    supported = _STRUCTURED_AFFINE_OPS | _DENSE_BOUNDARY_OPS | {BoundOpKind.SPEC_BIND}
    unsupported = [op.kind.value for op in module.graph.ops if op.kind not in supported]
    if unsupported:
        raise NotImplementedError(
            f"structured-region rewrite found unsupported ops: {unsupported}"
        )
