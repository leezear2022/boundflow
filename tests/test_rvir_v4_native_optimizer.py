"""Real ResNet provider-independent optimizer parity for RVIR-v4 V4-2D."""

# pylint: disable=missing-function-docstring

from dataclasses import dataclass, replace
from pathlib import Path

import pytest
import torch

from boundflow.frontends.onnx.frontend import import_onnx
from boundflow.planner import plan_interval_ibp_v0
from boundflow.ir.task import BFTaskModule
from boundflow.runtime.native_alpha_beta_optimization_state import (
    build_native_alpha_beta_scope,
    NativeAlphaBetaOptimizationState,
)
from boundflow.runtime.rvir_v4_native_optimizer import (
    compare_rvir_v4_native_optimizer_trace,
    execute_rvir_v4_native_optimizer_trace,
)
from boundflow.runtime.rvir_v4_optimizer_mutation import (
    ProductionOptimizerStepTraceV4,
    production_optimizer_step_trace_from_payload_v4,
)
from boundflow.runtime.rvir_v4_pre_state_initializer import (
    initialize_rvir_v4_native_pre_state,
    ProductionNativePreStateV4,
)
from boundflow.runtime.rvir_v4_production_state import (
    production_snapshot_from_payload_v4,
    ProductionStateSnapshotV4,
    production_tensor_sha256,
    ProductionTensorRole,
)
from boundflow.runtime.task_executor import InputSpec
from scripts.run_rvir_v4_pre_state_artifact import TOPOLOGY

ROOT = Path(__file__).resolve().parents[1]
CAPTURE = (
    ROOT / "artifacts/rvir-v4-pre-state/resnet2b-core-pre-state-v1/source_capture.pt"
)
MODEL = ROOT.parent / "vnncomp2021/benchmarks/cifar10_resnet/onnx/resnet_2b.onnx"


@dataclass(frozen=True)
class _Fixture:
    module: BFTaskModule
    spec: InputSpec
    objective: torch.Tensor
    mapping: ProductionNativePreStateV4
    state: NativeAlphaBetaOptimizationState
    base: ProductionStateSnapshotV4
    production: ProductionOptimizerStepTraceV4


def _fixture() -> _Fixture:
    capture = torch.load(CAPTURE, map_location="cpu", weights_only=True)
    base = production_snapshot_from_payload_v4(capture["cores"][0]["pre_snapshot"])
    production = production_optimizer_step_trace_from_payload_v4(
        capture["optimizer_step_traces"][0]
    )
    program = import_onnx(str(MODEL), do_shape_infer=True, normalize=True)
    module = plan_interval_ibp_v0(program)
    tensor_map = base.tensor_map()
    lower = next(
        tensor.value
        for tensor in tensor_map.values()
        if tensor.role == ProductionTensorRole.INPUT_LOWER
    )
    upper = next(
        tensor.value
        for tensor in tensor_map.values()
        if tensor.role == ProductionTensorRole.INPUT_UPPER
    )
    objective = next(
        tensor.value
        for tensor in tensor_map.values()
        if tensor.role == ProductionTensorRole.LINEAR_SPEC
    )
    spec = InputSpec.box(value_name=program.graph.inputs[0], lower=lower, upper=upper)
    mapping = initialize_rvir_v4_native_pre_state(base, TOPOLOGY)
    native_policy = production.mutation_policy.to_native_policy()
    scope = build_native_alpha_beta_scope(
        module,
        spec,
        linear_spec_C=objective,
        relu_pre=mapping.relu_pre,
        relu_split_state=mapping.splits,
        policy=native_policy,
    )
    state = mapping.to_native_state(scope)
    return _Fixture(module, spec, objective, mapping, state, base, production)


def test_native_optimizer_matches_all_ten_production_evaluations() -> None:
    fixture = _fixture()

    native = execute_rvir_v4_native_optimizer_trace(
        fixture.module,
        fixture.spec,
        linear_spec_C=fixture.objective,
        relu_pre=fixture.mapping.relu_pre,
        initial_state=fixture.state,
        mutation_policy=fixture.production.mutation_policy,
    )
    parity = compare_rvir_v4_native_optimizer_trace(
        native,
        fixture.production,
        base_snapshot=fixture.base,
        topology=TOPOLOGY,
    )

    assert len(native.steps) == 10
    assert sum(step.update_after for step in native.steps) == 9
    assert native.metadata()["provider_callback_count"] == 0
    assert parity.lower_maximum_absolute_difference <= 2e-4
    assert parity.alpha_maximum_absolute_difference <= 2e-4
    assert parity.beta_maximum_absolute_difference <= 2e-4
    assert all(row["allclose"] is True for row in parity.step_rows)
    assert all(row["sign_exact"] is True for row in parity.step_rows)
    assert native.steps[0].alpha_learning_rate == 0.01
    assert native.steps[0].beta_learning_rate == 0.05
    assert native.steps[-1].alpha_learning_rate == pytest.approx(0.01 * 0.98**9)
    assert native.steps[-1].beta_learning_rate == pytest.approx(0.05 * 0.98**9)
    assert fixture.state.stable_hash() == native.source_state_hash


def test_parity_rejects_resigned_production_lower_drift() -> None:
    fixture = _fixture()
    native = execute_rvir_v4_native_optimizer_trace(
        fixture.module,
        fixture.spec,
        linear_spec_C=fixture.objective,
        relu_pre=fixture.mapping.relu_pre,
        initial_state=fixture.state,
        mutation_policy=fixture.production.mutation_policy,
    )
    lower = fixture.production.steps[4].lower.clone()
    lower[0, 0] += 1.0
    bad_step = replace(
        fixture.production.steps[4],
        lower=lower,
        lower_sha256=production_tensor_sha256(lower),
    )
    bad_trace = ProductionOptimizerStepTraceV4(
        mutation_policy=fixture.production.mutation_policy,
        steps=fixture.production.steps[:4] + (bad_step,) + fixture.production.steps[5:],
    )

    with pytest.raises(ValueError, match="parity gate failed"):
        compare_rvir_v4_native_optimizer_trace(
            native, bad_trace, base_snapshot=fixture.base, topology=TOPOLOGY
        )


def test_native_optimizer_rejects_scope_drift_before_execution() -> None:
    fixture = _fixture()
    bad_state = replace(
        fixture.state,
        scope=replace(fixture.state.scope, objective_hash="f" * 64),
    )

    with pytest.raises(ValueError, match="initial scope differs"):
        execute_rvir_v4_native_optimizer_trace(
            fixture.module,
            fixture.spec,
            linear_spec_C=fixture.objective,
            relu_pre=fixture.mapping.relu_pre,
            initial_state=bad_state,
            mutation_policy=fixture.production.mutation_policy,
        )
