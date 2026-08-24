"""Compile the frozen ResNet2B full-lower recurrence into R3-1b0 trace IR."""

# pylint: disable=missing-function-docstring,too-many-locals

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from ..frontends.plain_crown_bound_ir import plain_crown_primal_graph_hash
from ..ir.r3_bounded_arena import (
    R31BBranchV1,
    R31BBoundedArenaTraceV1,
    R31BStepKind,
    R31BTraceStepV1,
)
from ..ir.task import BFTaskModule, TaskOp
from .r3_structured_owner_custom_backward import R31FullRegionPlanV1


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _op_map(module: BFTaskModule) -> dict[str, TaskOp]:
    result = {op.name: op for op in module.get_entry_task().ops}
    if len(result) != len(module.get_entry_task().ops):
        raise ValueError("R3-1b primal op names repeat")
    return result


def _shape(
    program: Any, value_name: str, *, domain_count: int, spec_count: int
) -> tuple[int, ...]:
    values = getattr(program.graph, "values", None)
    if not isinstance(values, Mapping) or value_name not in values:
        raise ValueError(f"R3-1b primal value is absent: {value_name}")
    value_type = getattr(values[value_name], "type", None)
    raw_shape = getattr(value_type, "shape", None)
    if not isinstance(raw_shape, list) or len(raw_shape) < 2 or raw_shape[0] != 1:
        raise ValueError(f"R3-1b primal value shape differs: {value_name}")
    logical = tuple(int(dimension) for dimension in raw_shape[1:])
    if any(dimension <= 0 for dimension in logical):
        raise ValueError(f"R3-1b symbolic/nonpositive shape differs: {value_name}")
    return (domain_count, spec_count, *logical)


def _require_op(
    ops: Mapping[str, TaskOp],
    name: str,
    op_type: str,
    inputs: tuple[str, ...],
    output: str,
) -> None:
    op = ops.get(name)
    if (
        op is None
        or op.op_type != op_type
        or tuple(op.inputs[:1]) != inputs
        or tuple(op.outputs) != (output,)
    ):
        raise ValueError(f"R3-1b frozen primal op differs: {name}")


def compile_r31b_bounded_arena_trace_v1(
    program: Any,
    module: BFTaskModule,
    production_plan: R31FullRegionPlanV1,
) -> R31BBoundedArenaTraceV1:
    """Compile exact reverse steps and prove a two-slot fused residual schedule."""

    module.validate()
    production_plan.validate()
    if plain_crown_primal_graph_hash(module) != production_plan.primal_graph_hash:
        raise ValueError("R3-1b production graph identity differs")
    ops = _op_map(module)
    expected_names = (
        "Conv_0",
        "Relu_1",
        "Conv_2",
        "Relu_3",
        "Conv_4",
        "Conv_5",
        "Add_6",
        "Relu_7",
        "Conv_8",
        "Relu_9",
        "Conv_10",
        "Add_11",
        "Relu_12",
        "Flatten_13",
        "Gemm_14",
        "Relu_15",
        "Gemm_16",
    )
    if tuple(op.name for op in module.get_entry_task().ops) != expected_names:
        raise ValueError("R3-1b frozen primal order differs")
    frozen_ops = (
        ("Gemm_16", "linear", ("32",), "33"),
        ("Relu_15", "relu", ("31",), "32"),
        ("Gemm_14", "linear", ("30",), "31"),
        ("Flatten_13", "flatten", ("29",), "30"),
        ("Relu_12", "relu", ("28",), "29"),
        ("Add_11", "add", ("27",), "28"),
        ("Relu_7", "relu", ("23",), "24"),
        ("Add_6", "add", ("21",), "23"),
        ("Relu_1", "relu", ("17",), "18"),
        ("Conv_0", "conv2d", ("input.1",), "17"),
    )
    for values in frozen_ops:
        _require_op(ops, *values)
    if tuple(ops["Add_11"].inputs) != ("27", "24") or tuple(ops["Add_6"].inputs) != (
        "21",
        "22",
    ):
        raise ValueError("R3-1b residual input order differs")
    domain = production_plan.domain_count
    spec = production_plan.spec_count
    shape = lambda value: _shape(  # noqa: E731  # pylint: disable=unnecessary-lambda-assignment
        program, value, domain_count=domain, spec_count=spec
    )
    steps = (
        R31BTraceStepV1(
            0,
            R31BStepKind.SPEC_SEED,
            "spec",
            "33",
            (6, 1, 10),
            shape("33"),
            (),
            -1,
            0,
            False,
        ),
        R31BTraceStepV1(
            1,
            R31BStepKind.LINEAR_RIGHT,
            "33",
            "32",
            shape("33"),
            shape("32"),
            ("Gemm_16",),
            0,
            1,
            False,
        ),
        R31BTraceStepV1(
            2,
            R31BStepKind.RELU_LOWER,
            "32",
            "31",
            shape("32"),
            shape("31"),
            ("Relu_15",),
            1,
            1,
            True,
        ),
        R31BTraceStepV1(
            3,
            R31BStepKind.LINEAR_RIGHT,
            "31",
            "30",
            shape("31"),
            shape("30"),
            ("Gemm_14",),
            1,
            0,
            False,
        ),
        R31BTraceStepV1(
            4,
            R31BStepKind.RESHAPE_VIEW,
            "30",
            "29",
            shape("30"),
            shape("29"),
            ("Flatten_13",),
            0,
            0,
            True,
        ),
        R31BTraceStepV1(
            5,
            R31BStepKind.RELU_LOWER,
            "29",
            "28",
            shape("29"),
            shape("28"),
            ("Relu_12",),
            0,
            0,
            True,
        ),
        R31BTraceStepV1(
            6,
            R31BStepKind.RESIDUAL_REGION,
            "28",
            "24",
            shape("28"),
            shape("24"),
            ("Add_11", "Conv_10", "Relu_9", "Conv_8"),
            0,
            1,
            False,
            True,
            (
                R31BBranchV1("27", "24", ("Conv_10", "Relu_9", "Conv_8"), False),
                R31BBranchV1("24", "24", (), True),
            ),
        ),
        R31BTraceStepV1(
            7,
            R31BStepKind.RELU_LOWER,
            "24",
            "23",
            shape("24"),
            shape("23"),
            ("Relu_7",),
            1,
            1,
            True,
        ),
        R31BTraceStepV1(
            8,
            R31BStepKind.RESIDUAL_REGION,
            "23",
            "18",
            shape("23"),
            shape("18"),
            ("Add_6", "Conv_4", "Relu_3", "Conv_2", "Conv_5"),
            1,
            0,
            False,
            True,
            (
                R31BBranchV1("21", "18", ("Conv_4", "Relu_3", "Conv_2"), False),
                R31BBranchV1("22", "18", ("Conv_5",), False),
            ),
        ),
        R31BTraceStepV1(
            9,
            R31BStepKind.RELU_LOWER,
            "18",
            "17",
            shape("18"),
            shape("17"),
            ("Relu_1",),
            0,
            0,
            True,
        ),
        R31BTraceStepV1(
            10,
            R31BStepKind.CONV2D_RIGHT,
            "17",
            "input.1",
            shape("17"),
            shape("input.1"),
            ("Conv_0",),
            0,
            1,
            False,
        ),
        R31BTraceStepV1(
            11,
            R31BStepKind.INPUT_CONCRETIZE,
            "input.1",
            "lower",
            shape("input.1"),
            (6, 1),
            (),
            1,
            -1,
            False,
        ),
    )
    source_payload = [
        {
            "name": op.name,
            "type": op.op_type,
            "inputs": op.inputs,
            "outputs": op.outputs,
            "attrs": op.attrs,
        }
        for op in module.get_entry_task().ops
    ]
    trace = R31BBoundedArenaTraceV1(
        source_hash=_canonical_hash(source_payload),
        topology_hash=_canonical_hash([step.to_dict() for step in steps]),
        production_plan_hash=production_plan.stable_hash(),
        steps=steps,
        scratch_slot_count=2,
        scratch_capacity_elements=max(
            max(step.input_numel, step.output_numel) for step in steps
        ),
    )
    trace.validate()
    return trace


__all__ = ["compile_r31b_bounded_arena_trace_v1"]
