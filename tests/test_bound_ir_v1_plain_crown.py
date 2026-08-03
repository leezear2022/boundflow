"""Numerical closure tests for plain-CROWN Bound IR lowering/interpreter."""

# pylint: disable=missing-function-docstring,invalid-name,duplicate-code

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
from typing import Callable

import pytest
import torch

from boundflow.domains.interval import IntervalState
from boundflow.frontends.plain_crown_bound_ir import (
    PlainCrownBoundIRBuild,
    build_plain_crown_bound_ir,
)
from boundflow.ir.bound import (
    BFBoundGraph,
    BoundOpKind,
    BoundRepresentation,
    BoundValueRole,
    IntermediateBoundSource,
    ReluLowerSlopePolicy,
)
from boundflow.ir.bound_rewrite import rewrite_plain_crown_structured_regions
from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime.bound_ir_interpreter import execute_plain_crown_bound_ir
from boundflow.runtime.crown_ibp import _forward_ibp_trace_mlp, run_crown_ibp_mlp
from boundflow.runtime.task_executor import InputSpec


def _module(ops: list[TaskOp], params: dict[str, torch.Tensor]) -> BFTaskModule:
    return BFTaskModule(
        tasks=[
            BoundTask(
                task_id="plain-crown",
                kind=TaskKind.INTERVAL_IBP,
                ops=ops,
                input_values=["input"],
                output_values=["out"],
            )
        ],
        entry_task_id="plain-crown",
        bindings={"params": params},
    )


def _mlp() -> BFTaskModule:
    return _module(
        [
            TaskOp("linear", "linear1", ["input", "W1", "b1"], ["h1"]),
            TaskOp("relu", "relu1", ["h1"], ["r1"]),
            TaskOp("linear", "linear2", ["r1", "W2", "b2"], ["out"]),
        ],
        {
            "W1": torch.tensor(
                [[1.0, -0.5, 0.25], [-0.25, 0.75, 1.0], [0.5, 0.5, -1.0]]
            ),
            "b1": torch.tensor([0.1, -0.2, 0.05]),
            "W2": torch.tensor([[0.75, -1.0, 0.5], [-0.5, 0.25, 1.25]]),
            "b2": torch.tensor([0.15, -0.1]),
        },
    )


def _residual_fanout() -> BFTaskModule:
    torch.manual_seed(7)
    return _module(
        [
            TaskOp("linear", "linear1", ["input", "W1", "b1"], ["h1"]),
            TaskOp("relu", "relu1", ["h1"], ["r1"]),
            TaskOp("linear", "linear2", ["r1", "W2", "b2"], ["h2"]),
            TaskOp("add", "residual", ["input", "h2"], ["sum"]),
            TaskOp("relu", "relu2", ["sum"], ["r2"]),
            TaskOp("linear", "linear3", ["r2", "W3", "b3"], ["out"]),
        ],
        {
            "W1": torch.randn(5, 4),
            "b1": torch.randn(5),
            "W2": torch.randn(4, 5),
            "b2": torch.randn(4),
            "W3": torch.randn(3, 4),
            "b3": torch.randn(3),
        },
    )


def _concat_fanout() -> BFTaskModule:
    torch.manual_seed(11)
    return _module(
        [
            TaskOp("linear", "left", ["input", "W1", "b1"], ["h1"]),
            TaskOp("relu", "left_relu", ["h1"], ["r1"]),
            TaskOp("linear", "right", ["input", "W2", "b2"], ["h2"]),
            TaskOp("relu", "right_relu", ["h2"], ["r2"]),
            TaskOp(
                "concat",
                "join",
                ["r1", "r2"],
                ["joined"],
                attrs={"axis": 1},
            ),
            TaskOp("linear", "output", ["joined", "W3", "b3"], ["out"]),
        ],
        {
            "W1": torch.randn(3, 4),
            "b1": torch.randn(3),
            "W2": torch.randn(2, 4),
            "b2": torch.randn(2),
            "W3": torch.randn(2, 5),
            "b3": torch.randn(2),
        },
    )


def _chain_cnn() -> BFTaskModule:
    torch.manual_seed(13)
    return _module(
        [
            TaskOp(
                "conv2d",
                "conv",
                ["input", "Wc", "bc"],
                ["conv_out"],
                attrs={"stride": 1, "padding": 0, "dilation": 1, "groups": 1},
            ),
            TaskOp("relu", "relu", ["conv_out"], ["relu_out"]),
            TaskOp(
                "flatten",
                "flatten",
                ["relu_out"],
                ["flat"],
                attrs={"start_dim": 1, "end_dim": -1},
            ),
            TaskOp("linear", "linear", ["flat", "Wl", "bl"], ["out"]),
        ],
        {
            "Wc": torch.randn(2, 1, 2, 2),
            "bc": torch.randn(2),
            "Wl": torch.randn(3, 8),
            "bl": torch.randn(3),
        },
    )


def _lower_and_execute(
    module: BFTaskModule,
    spec: InputSpec,
    *,
    C: torch.Tensor | None = None,
) -> tuple[PlainCrownBoundIRBuild, IntervalState, IntervalState]:
    interval_env, relu_pre = _forward_ibp_trace_mlp(module, spec)
    build = build_plain_crown_bound_ir(
        module,
        spec,
        interval_env=interval_env,
        relu_pre=relu_pre,
        linear_spec_C=C,
    )
    actual = execute_plain_crown_bound_ir(
        build.module,
        task_module=module,
        input_spec=spec,
        relu_pre=relu_pre,
        linear_spec_C=C,
    )
    expected = run_crown_ibp_mlp(module, spec, linear_spec_C=C)
    return build, actual, expected


@pytest.mark.parametrize("use_linear_spec", [False, True])
def test_plain_crown_bound_ir_matches_mlp(use_linear_spec: bool) -> None:
    module = _mlp()
    spec = InputSpec.linf(
        value_name="input",
        center=torch.tensor([[0.2, -0.1, 0.4], [-0.3, 0.5, 0.1]]),
        eps=0.25,
    )
    C = (
        torch.tensor([[1.0, -1.0], [-0.5, 0.25], [0.0, 1.0]])
        if use_linear_spec
        else None
    )

    build, actual, expected = _lower_and_execute(module, spec, C=C)

    torch.testing.assert_close(actual.lower, expected.lower, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual.upper, expected.upper, atol=1e-6, rtol=1e-6)
    repeated = build_plain_crown_bound_ir(
        module,
        spec,
        interval_env=_forward_ibp_trace_mlp(module, spec)[0],
        relu_pre=_forward_ibp_trace_mlp(module, spec)[1],
        linear_spec_C=C,
    )
    assert build.module.canonical_json() == repeated.module.canonical_json()
    assert build.module.stable_hash() == repeated.module.stable_hash()


def test_external_adaptive_relu_policy_is_explicit_and_changes_semantics() -> None:
    module = _mlp()
    spec = InputSpec.linf(
        value_name="input", center=torch.tensor([[0.2, -0.1, 0.4]]), eps=0.25
    )
    interval_env, relu_pre = _forward_ibp_trace_mlp(module, spec)
    common = {
        "task_module": module,
        "input_spec": spec,
        "interval_env": interval_env,
        "relu_pre": relu_pre,
        "linear_spec_C": torch.tensor([[1.0, -1.0]]),
    }
    zero = build_plain_crown_bound_ir(**common).module
    with pytest.raises(ValueError, match="exact SHA-256 identity"):
        build_plain_crown_bound_ir(
            **common,
            intermediate_bound_source=IntermediateBoundSource.EXTERNAL_VERIFIER,
            relu_lower_slope_policy=ReluLowerSlopePolicy.ADAPTIVE,
        )
    adaptive = build_plain_crown_bound_ir(
        **common,
        intermediate_bound_source=IntermediateBoundSource.EXTERNAL_VERIFIER,
        intermediate_bounds_hash="a" * 64,
        relu_lower_slope_policy=ReluLowerSlopePolicy.ADAPTIVE,
    ).module

    zero_result = execute_plain_crown_bound_ir(
        zero,
        task_module=module,
        input_spec=spec,
        relu_pre=relu_pre,
        linear_spec_C=common["linear_spec_C"],
    )
    adaptive_result = execute_plain_crown_bound_ir(
        adaptive,
        task_module=module,
        input_spec=spec,
        relu_pre=relu_pre,
        linear_spec_C=common["linear_spec_C"],
    )
    relu_attrs = next(
        op.attrs for op in adaptive.graph.ops if op.kind == BoundOpKind.RELU_RELAXATION
    )

    assert (
        relu_attrs.intermediate_bound_source
        == IntermediateBoundSource.EXTERNAL_VERIFIER
    )
    assert relu_attrs.lower_slope_policy == ReluLowerSlopePolicy.ADAPTIVE
    adaptive_values = {value.value_id: value for value in adaptive.graph.values}
    assert all(
        adaptive_values[value_id].state_version
        == f"external-intermediate-bounds:{'a' * 64}"
        for op in adaptive.graph.ops
        if op.kind == BoundOpKind.RELU_RELAXATION
        for value_id in op.outputs
    )
    assert adaptive.stable_hash() != zero.stable_hash()
    alternate_external_identity = build_plain_crown_bound_ir(
        **common,
        intermediate_bound_source=IntermediateBoundSource.EXTERNAL_VERIFIER,
        intermediate_bounds_hash="b" * 64,
        relu_lower_slope_policy=ReluLowerSlopePolicy.ADAPTIVE,
    ).module
    assert alternate_external_identity.stable_hash() != adaptive.stable_hash()
    assert adaptive.canonical_json().count('"lower_slope_policy":"adaptive"') == 1
    assert not (
        torch.equal(adaptive_result.lower, zero_result.lower)
        and torch.equal(adaptive_result.upper, zero_result.upper)
    )


@pytest.mark.parametrize(
    ("module_factory", "center"),
    [
        (_residual_fanout, torch.randn(2, 4)),
        (_concat_fanout, torch.randn(2, 4)),
    ],
)
def test_plain_crown_bound_ir_matches_fanout_dags(
    module_factory: Callable[[], BFTaskModule], center: torch.Tensor
) -> None:
    module = module_factory()
    spec = InputSpec.l2(value_name="input", center=center, eps=0.15)

    build, actual, expected = _lower_and_execute(module, spec)

    torch.testing.assert_close(actual.lower, expected.lower, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(actual.upper, expected.upper, atol=2e-6, rtol=2e-6)
    kinds = [op.kind for op in build.module.graph.ops]
    assert BoundOpKind.COEFFICIENT_COMPOSE in kinds
    expected_route = (
        BoundOpKind.ADD_BACKWARD
        if module_factory is _residual_fanout
        else BoundOpKind.CONCAT_BACKWARD
    )
    assert expected_route in kinds


def test_plain_crown_bound_ir_matches_chain_cnn() -> None:
    module = _chain_cnn()
    spec = InputSpec.box(
        value_name="input",
        lower=torch.full((1, 1, 3, 3), -0.4),
        upper=torch.full((1, 1, 3, 3), 0.6),
    )

    build, actual, expected = _lower_and_execute(module, spec)

    torch.testing.assert_close(actual.lower, expected.lower, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(actual.upper, expected.upper, atol=2e-6, rtol=2e-6)
    assert BoundOpKind.CONV2D_BACKWARD in {op.kind for op in build.module.graph.ops}


def test_plain_crown_bound_ir_fails_closed_on_missing_trace_and_stale_bindings() -> (
    None
):
    module = _mlp()
    spec = InputSpec.linf(value_name="input", center=torch.zeros(1, 3), eps=0.1)
    interval_env, relu_pre = _forward_ibp_trace_mlp(module, spec)

    with pytest.raises(KeyError, match="pre-activation"):
        build_plain_crown_bound_ir(
            module,
            spec,
            interval_env=interval_env,
            relu_pre={},
        )

    C = torch.eye(2)
    build = build_plain_crown_bound_ir(
        module,
        spec,
        interval_env=interval_env,
        relu_pre=relu_pre,
        linear_spec_C=C,
    )
    with pytest.raises(ValueError, match="objective payload"):
        execute_plain_crown_bound_ir(
            build.module,
            task_module=module,
            input_spec=spec,
            relu_pre=relu_pre,
            linear_spec_C=C + 1,
        )

    stale_params = dict(module.bindings["params"])
    stale_params["b2"] = stale_params["b2"] + 1
    stale_module = replace(module, bindings={"params": stale_params})
    with pytest.raises(ValueError, match="fingerprint"):
        execute_plain_crown_bound_ir(
            build.module,
            task_module=stale_module,
            input_spec=spec,
            relu_pre=relu_pre,
            linear_spec_C=C,
        )


def test_bound_ir_interpreter_does_not_import_crown_oracle() -> None:
    source = Path("boundflow/runtime/bound_ir_interpreter.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert not any(module.endswith("crown_ibp") for module in imported_modules)


@pytest.mark.parametrize("case", ["mlp", "residual", "concat", "cnn"])
def test_structured_region_rewrite_preserves_final_bounds(case: str) -> None:
    if case == "mlp":
        module = _mlp()
        spec = InputSpec.linf(value_name="input", center=torch.randn(2, 3), eps=0.2)
        C = torch.tensor([[1.0, -1.0], [0.25, 0.75]])
    elif case == "residual":
        module = _residual_fanout()
        spec = InputSpec.l2(value_name="input", center=torch.randn(2, 4), eps=0.15)
        C = None
    elif case == "concat":
        module = _concat_fanout()
        spec = InputSpec.linf(value_name="input", center=torch.randn(2, 4), eps=0.1)
        C = None
    else:
        module = _chain_cnn()
        spec = InputSpec.box(
            value_name="input",
            lower=torch.full((1, 1, 3, 3), -0.3),
            upper=torch.full((1, 1, 3, 3), 0.5),
        )
        C = None

    interval_env, relu_pre = _forward_ibp_trace_mlp(module, spec)
    dense = build_plain_crown_bound_ir(
        module,
        spec,
        interval_env=interval_env,
        relu_pre=relu_pre,
        linear_spec_C=C,
    ).module
    structured = rewrite_plain_crown_structured_regions(dense)
    dense_result = execute_plain_crown_bound_ir(
        dense,
        task_module=module,
        input_spec=spec,
        relu_pre=relu_pre,
        linear_spec_C=C,
    )
    structured_result = execute_plain_crown_bound_ir(
        structured,
        task_module=module,
        input_spec=spec,
        relu_pre=relu_pre,
        linear_spec_C=C,
    )

    torch.testing.assert_close(
        structured_result.lower, dense_result.lower, atol=2e-6, rtol=2e-6
    )
    torch.testing.assert_close(
        structured_result.upper, dense_result.upper, atol=2e-6, rtol=2e-6
    )
    kinds = [op.kind for op in structured.graph.ops]
    assert BoundOpKind.REPRESENTATION_CAST in kinds
    assert BoundOpKind.MATERIALIZE in kinds
    assert any(
        value.role == BoundValueRole.COEFFICIENT
        and value.representation == BoundRepresentation.STRUCTURED
        for value in structured.graph.values
    )
    repeated = rewrite_plain_crown_structured_regions(dense)
    assert structured.canonical_json() == repeated.canonical_json()
    assert structured.stable_hash() == repeated.stable_hash()


def test_affine_ops_require_explicit_representation_transitions() -> None:
    module = _mlp()
    spec = InputSpec.linf(value_name="input", center=torch.zeros(1, 3), eps=0.1)
    interval_env, relu_pre = _forward_ibp_trace_mlp(module, spec)
    dense = build_plain_crown_bound_ir(
        module, spec, interval_env=interval_env, relu_pre=relu_pre
    ).module
    linear = next(
        op for op in dense.graph.ops if op.kind == BoundOpKind.LINEAR_BACKWARD
    )
    output_ids = {linear.outputs[0], linear.outputs[2]}
    corrupted_values = tuple(
        (
            replace(value, representation=BoundRepresentation.STRUCTURED)
            if value.value_id in output_ids
            else value
        )
        for value in dense.graph.values
    )
    corrupted = replace(
        dense,
        graph=BFBoundGraph(
            values=corrupted_values,
            ops=dense.graph.ops,
            inputs=dense.graph.inputs,
            outputs=dense.graph.outputs,
        ),
    )

    with pytest.raises(ValueError, match="explicit representation transition"):
        corrupted.validate()

    structured = rewrite_plain_crown_structured_regions(dense)
    with pytest.raises(ValueError, match="dense source module"):
        rewrite_plain_crown_structured_regions(structured)
