"""R3-2A dynamic rebind and 10/9 optimizer trajectory gates."""

# pylint: disable=missing-function-docstring,too-many-locals,protected-access

from dataclasses import replace
from inspect import getsource
from pathlib import Path

import pytest
import torch

from boundflow.frontends.onnx.frontend import import_onnx
from boundflow.planner import plan_interval_ibp_v0
from boundflow.runtime.fsg4_b3_terminal_optimizer_schedule import (
    compile_terminal_optimizer_schedule_v1,
)
from boundflow.runtime.r3_bounded_arena_trace_compiler import (
    compile_r31b_bounded_arena_trace_v1,
)
from boundflow.runtime.r3_optimizer_trajectory import (
    execute_r32a_optimizer_trajectory_v1,
    rebind_r32a_dynamic_instance_v1,
    R32AExecutionMode,
)
from boundflow.runtime.r3_structured_owner_custom_backward import (
    bind_r31_runtime_inputs_v1,
    compile_r31_full_region_plan_v1,
)
from boundflow.runtime.rvir_v4_pre_state_initializer import (
    initialize_rvir_v4_native_pre_state,
)
from boundflow.runtime.rvir_v4_production_state import (
    production_snapshot_from_payload_v4,
)
from scripts.run_rvir_v4_pre_state_artifact import TOPOLOGY

ROOT = Path(__file__).resolve().parents[1]
CAPTURE = (
    ROOT / "artifacts/rvir-v4-pre-state/resnet2b-core-pre-state-v1/source_capture.pt"
)
MODEL = Path(
    "/home/lee/Codes/vnncomp2021/benchmarks/cifar10_resnet/onnx/resnet_2b.onnx"
)


def _fixture():
    if not MODEL.is_file():
        pytest.skip("frozen ResNet2B checkout is unavailable")
    if not torch.cuda.is_available():
        pytest.skip("R3-2A CUDA device is unavailable")
    raw = torch.load(CAPTURE, map_location="cpu", weights_only=True)
    snapshot = production_snapshot_from_payload_v4(raw["cores"][0]["pre_snapshot"])
    mapping = initialize_rvir_v4_native_pre_state(snapshot, TOPOLOGY)
    program = import_onnx(str(MODEL), do_shape_infer=True, normalize=True)
    module = plan_interval_ibp_v0(program)
    plan = compile_r31_full_region_plan_v1(module, snapshot, mapping, TOPOLOGY)
    trace = compile_r31b_bounded_arena_trace_v1(program, module, plan)
    tensors = bind_r31_runtime_inputs_v1(
        plan, module, snapshot, device=torch.device("cuda:0")
    )
    return plan, trace, tensors


def test_r32a_dynamic_rebind_changes_only_p_alpha_instance_identity() -> None:
    plan, trace, tensors = _fixture()
    p_ordinal = plan.p_alpha_input_ordinal
    with torch.no_grad():
        tensors[p_ordinal].add_(0.001).clamp_(0.0, 1.0)
    rebound, rebound_trace, receipt = rebind_r32a_dynamic_instance_v1(
        plan,
        trace,
        tensors,
        trajectory_id="1" * 64,
        evaluation_ordinal=1,
        alpha_learning_rate=0.0098,
    )
    changed = [
        ordinal
        for ordinal, (left, right) in enumerate(
            zip(plan.tensor_specs, rebound.tensor_specs)
        )
        if left != right
    ]
    assert changed == [p_ordinal]
    assert rebound_trace.production_plan_hash == rebound.stable_hash()
    assert receipt.rebound_plan_hash == rebound.stable_hash()


def test_r32a_dynamic_rebind_rejects_immutable_drift() -> None:
    plan, trace, tensors = _fixture()
    values = list(tensors)
    values[0] = values[0].clone()
    values[0][0, 0, 0, 0] += 0.01
    with pytest.raises(ValueError, match="immutable tensor drifted"):
        rebind_r32a_dynamic_instance_v1(
            plan,
            trace,
            tuple(values),
            trajectory_id="2" * 64,
            evaluation_ordinal=0,
            alpha_learning_rate=0.01,
        )


def test_r32a_native_candidate_ten_nine_trajectory_matches() -> None:
    plan, trace, native_tensors = _fixture()
    _, _, candidate_tensors = _fixture()
    schedule = compile_terminal_optimizer_schedule_v1()
    native = execute_r32a_optimizer_trajectory_v1(
        plan,
        trace,
        native_tensors,
        schedule=schedule,
        mode=R32AExecutionMode.NATIVE,
    )
    candidate = execute_r32a_optimizer_trajectory_v1(
        plan,
        trace,
        candidate_tensors,
        schedule=schedule,
        mode=R32AExecutionMode.CANDIDATE,
    )
    assert native.trajectory_id == candidate.trajectory_id
    assert native.optimizer_mutation_count == candidate.optimizer_mutation_count == 9
    for control, compiled in zip(native.steps, candidate.steps):
        assert torch.allclose(control.lower, compiled.lower, atol=2e-4, rtol=2e-4)
        assert torch.equal(torch.sign(control.lower), torch.sign(compiled.lower))
        assert torch.allclose(control.gradient, compiled.gradient, atol=2e-4, rtol=2e-4)
        assert torch.equal(torch.sign(control.gradient), torch.sign(compiled.gradient))
        assert torch.allclose(
            control.alpha_after, compiled.alpha_after, atol=2e-5, rtol=2e-5
        )
        assert torch.allclose(
            control.optimizer_after.exp_avg,
            compiled.optimizer_after.exp_avg,
            atol=2e-5,
            rtol=2e-5,
        )
        assert torch.allclose(
            control.optimizer_after.exp_avg_sq,
            compiled.optimizer_after.exp_avg_sq,
            atol=2e-5,
            rtol=2e-5,
        )
    assert torch.allclose(
        native.terminal_alpha, candidate.terminal_alpha, atol=2e-5, rtol=2e-5
    )
    assert all(step.compiled_receipt is None for step in native.steps)
    assert all(step.compiled_receipt is not None for step in candidate.steps)


def test_r32a_candidate_source_has_no_native_shadow() -> None:
    source = getsource(execute_r32a_optimizer_trajectory_v1)
    candidate_branch = source.split("else:", 1)[1]
    assert "execute_r31_native_oracle_v1" not in candidate_branch
    assert "run_crown_ibp" not in candidate_branch


def test_r32a_tampered_schedule_fails_closed() -> None:
    plan, trace, tensors = _fixture()
    schedule = compile_terminal_optimizer_schedule_v1()
    bad = replace(schedule, actions=schedule.actions[:-1])
    with pytest.raises(ValueError, match="terminal optimizer schedule differs"):
        execute_r32a_optimizer_trajectory_v1(
            plan,
            trace,
            tensors,
            schedule=bad,
            mode=R32AExecutionMode.NATIVE,
        )
