"""S4 compiled optimizer at the existing RVIR/B4-A exact-call seam."""

# pylint: disable=missing-function-docstring,protected-access,too-many-locals
# pylint: disable=import-error,import-outside-toplevel,duplicate-code
# pylint: disable=redefined-outer-name

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import torch

from boundflow.frontends.onnx.frontend import import_onnx
from boundflow.planner import plan_interval_ibp_v0
from boundflow.runtime.asplos27_s4_exact_call_bridge import (
    compile_s4_exact_call_assets_v1,
    execute_s4_exact_call_handoff_v1,
    prepare_s4_exact_call_region_v1,
)
from boundflow.runtime.fsg4_b3_prepared_core import (
    instantiate_core_plan_v1,
    prepare_core_template_v1,
)
from boundflow.runtime.fsg4_b3_terminal_optimizer_schedule import (
    compile_terminal_optimizer_schedule_v1,
)
from boundflow.runtime.fsg4_b4a_terminal_lower_adjoint_handoff import (
    execute_terminal_optimizer_with_lower_adjoint_handoff_v1,
)
from boundflow.runtime.native_alpha_beta_optimization_state import (
    build_native_alpha_beta_scope,
)
from boundflow.runtime.rvir_v4_optimizer_mutation import (
    production_optimizer_step_trace_from_payload_v4,
)
from boundflow.runtime.rvir_v4_pre_state_initializer import (
    initialize_rvir_v4_native_pre_state,
)
from boundflow.runtime.rvir_v4_production_state import (
    production_snapshot_from_payload_v4,
    ProductionTensorOwnership,
    ProductionTensorRole,
)
from boundflow.runtime.task_executor import InputSpec
from scripts.run_rvir_v4_live_return_capture import _move_tensors
from scripts.run_rvir_v4_pre_state_artifact import TOPOLOGY

ROOT = Path(__file__).resolve().parents[1]
CAPTURE = (
    ROOT / "artifacts/rvir-v4-pre-state/resnet2b-core-pre-state-v1/source_capture.pt"
)
MODEL = ROOT.parent / "vnncomp2021/benchmarks/cifar10_resnet/onnx/resnet_2b.onnx"


def _one(snapshot: Any, role: ProductionTensorRole) -> torch.Tensor:
    values = [item.value for item in snapshot.tensors if item.role == role]
    assert len(values) == 1
    return cast(torch.Tensor, values[0])


@pytest.fixture(scope="module")
def exact_call_case():
    if not torch.cuda.is_available() or not CAPTURE.is_file() or not MODEL.is_file():
        pytest.skip("S4 exact-call production CUDA fixture is unavailable")
    device = torch.device("cuda", torch.cuda.current_device())
    raw = torch.load(CAPTURE, map_location="cpu", weights_only=True)
    snapshot = production_snapshot_from_payload_v4(raw["cores"][0]["pre_snapshot"])
    production = production_optimizer_step_trace_from_payload_v4(
        raw["optimizer_step_traces"][0]
    )
    mapping = initialize_rvir_v4_native_pre_state(snapshot, TOPOLOGY).to(
        device=device, dtype=torch.float32
    )
    program = import_onnx(str(MODEL), do_shape_infer=True, normalize=True)
    module = plan_interval_ibp_v0(program)
    module.bindings = cast(
        dict[str, Any],
        _move_tensors(module.bindings, device=device, dtype=torch.float32),
    )
    lower = _one(snapshot, ProductionTensorRole.INPUT_LOWER).to(device)
    upper = _one(snapshot, ProductionTensorRole.INPUT_UPPER).to(device)
    objective = _one(snapshot, ProductionTensorRole.LINEAR_SPEC).to(device)
    input_spec = InputSpec.box(
        value_name=program.graph.inputs[0], lower=lower, upper=upper
    )
    scope = build_native_alpha_beta_scope(
        module,
        input_spec,
        linear_spec_C=objective,
        relu_pre=mapping.relu_pre,
        relu_split_state=mapping.splits,
        policy=production.mutation_policy.to_native_policy(),
    )
    initial = mapping.to_native_state(scope)
    live_sources = {
        item.semantic_path: item.value.to(device).detach().clone().requires_grad_(True)
        for item in snapshot.tensors
        if item.ownership == ProductionTensorOwnership.MUTABLE_COPY_OUT
    }
    assets = compile_s4_exact_call_assets_v1(device=device)
    schedule = compile_terminal_optimizer_schedule_v1()
    candidate = execute_s4_exact_call_handoff_v1(
        program=program,
        module=module,
        snapshot=snapshot,
        mapping=mapping,
        live_sources=live_sources,
        exact_call_id="s4-rvir-exact-call-correctness",
        input_spec=input_spec,
        linear_spec_C=objective,
        relu_pre=mapping.relu_pre,
        initial_state=initial,
        mutation_policy=production.mutation_policy,
        schedule=schedule,
        topology=TOPOLOGY,
        stream=torch.cuda.Stream(device=device),
        assets=assets,
    )
    mutable_paths = tuple(
        sorted(
            item.semantic_path
            for item in snapshot.tensors
            if item.ownership == ProductionTensorOwnership.MUTABLE_COPY_OUT
        )
    )
    template = prepare_core_template_v1(
        template_id="s4-exact-call-test",
        program=program,
        module=module,
        topology=TOPOLOGY,
        device=str(device),
        dtype=torch.float32,
        input_shape=tuple(lower.shape),
        objective_shape=tuple(objective.shape),
        mutable_paths=mutable_paths,
    )
    instance = instantiate_core_plan_v1(
        template=template,
        topology=TOPOLOGY,
        snapshot=snapshot,
        mapping=mapping,
        input_spec=input_spec,
        linear_spec_C=objective,
        mutation_policy=production.mutation_policy,
    )
    prepared_live = {
        path: value.detach().clone().requires_grad_(True)
        for path, value in live_sources.items()
    }
    prepared_stream = torch.cuda.Stream(device=device)
    prepared_region = prepare_s4_exact_call_region_v1(
        program=program,
        module=module,
        snapshot=snapshot,
        mapping=mapping,
        live_sources=prepared_live,
        exact_call_id="s4-rvir-exact-call-prepared",
        topology=TOPOLOGY,
        stream=prepared_stream,
        assets=assets,
    )
    prepared = execute_s4_exact_call_handoff_v1(
        program=program,
        module=module,
        snapshot=snapshot,
        mapping=mapping,
        live_sources=prepared_live,
        exact_call_id="s4-rvir-exact-call-prepared",
        input_spec=input_spec,
        linear_spec_C=objective,
        relu_pre=mapping.relu_pre,
        initial_state=instance.initial_state,
        mutation_policy=production.mutation_policy,
        schedule=schedule,
        topology=TOPOLOGY,
        stream=prepared_stream,
        assets=assets,
        prevalidated_plan=instance,
        prepared_region=prepared_region,
    )
    control = execute_terminal_optimizer_with_lower_adjoint_handoff_v1(
        module,
        input_spec,
        linear_spec_C=objective,
        relu_pre=mapping.relu_pre,
        initial_state=initial,
        mutation_policy=production.mutation_policy,
        schedule=schedule,
        topology=TOPOLOGY,
    )
    return candidate, prepared, control


def test_s4_exact_call_matches_native_terminal_handoff(exact_call_case) -> None:
    candidate, _prepared, control = exact_call_case
    candidate_result = candidate.handoff_result
    assert torch.allclose(
        candidate_result.optimizer_result.terminal_lower,
        control.optimizer_result.terminal_lower,
        atol=2e-4,
        rtol=2e-4,
    )
    assert torch.equal(
        torch.sign(candidate_result.optimizer_result.terminal_lower),
        torch.sign(control.optimizer_result.terminal_lower),
    )
    candidate_adjoint = dict(
        candidate_result.handoff.lower_adjoint_by_native_preactivation
    )
    control_adjoint = dict(control.handoff.lower_adjoint_by_native_preactivation)
    assert candidate_adjoint.keys() == control_adjoint.keys()
    for name in candidate_adjoint:
        assert torch.allclose(
            candidate_adjoint[name], control_adjoint[name], atol=2e-4, rtol=2e-4
        )
        assert torch.equal(
            torch.sign(candidate_adjoint[name]), torch.sign(control_adjoint[name])
        )


def test_s4_exact_call_receipt_has_no_compile_or_fallback(exact_call_case) -> None:
    candidate, _prepared, _control = exact_call_case
    receipt = candidate.receipt
    receipt.validate()
    assert receipt.evaluation_count == 10
    assert receipt.mutation_count == 9
    assert receipt.compile_inside_exact_call_count == 0
    assert receipt.fallback_count == 0
    assert receipt.performance_claimed is False


def test_s4_prepared_exact_call_matches_dynamic_bridge(exact_call_case) -> None:
    candidate, prepared, _control = exact_call_case
    observed = prepared.handoff_result.optimizer_result.terminal_lower
    expected = candidate.handoff_result.optimizer_result.terminal_lower
    assert torch.allclose(observed, expected, atol=2e-4, rtol=2e-4)
    assert torch.equal(torch.sign(observed), torch.sign(expected))
    assert prepared.receipt.setup_ns < candidate.receipt.setup_ns
    assert prepared.receipt.compile_inside_exact_call_count == 0
