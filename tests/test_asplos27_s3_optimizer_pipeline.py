"""Correctness and fail-closed gates for the ASPLOS'27 S3 wrapper."""

# pylint: disable=missing-function-docstring,too-many-locals,protected-access

from dataclasses import replace
from inspect import getsource
from pathlib import Path

import pytest
import torch

from boundflow.frontends.onnx.frontend import import_onnx
from boundflow.planner import plan_interval_ibp_v0
from boundflow.runtime.asplos27_s2_crown_pipeline import PreparedS2CrownProgramV1
from boundflow.runtime.asplos27_s3_optimizer_pipeline import (
    execute_asplos27_s3_optimizer_v1,
)
from boundflow.runtime.fsg4_b3_terminal_optimizer_schedule import (
    compile_terminal_optimizer_schedule_v1,
)
from boundflow.runtime.r3_bounded_arena_trace_compiler import (
    compile_r31b_bounded_arena_trace_v1,
)
from boundflow.runtime.r3_structured_owner_custom_backward import (
    _evaluate_full_region,
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
MODEL = ROOT.parent / "vnncomp2021/benchmarks/cifar10_resnet/onnx/resnet_2b.onnx"


def _fixture():  # type: ignore[no-untyped-def]
    if not MODEL.is_file() or not CAPTURE.is_file() or not torch.cuda.is_available():
        pytest.skip("S3 frozen CUDA fixture is unavailable")
    raw = torch.load(CAPTURE, map_location="cpu", weights_only=True)
    snapshot = production_snapshot_from_payload_v4(raw["cores"][0]["pre_snapshot"])
    mapping = initialize_rvir_v4_native_pre_state(snapshot, TOPOLOGY)
    program = import_onnx(str(MODEL), do_shape_infer=True, normalize=True)
    module = plan_interval_ibp_v0(program)
    plan = compile_r31_full_region_plan_v1(module, snapshot, mapping, TOPOLOGY)
    trace = compile_r31b_bounded_arena_trace_v1(program, module, plan)

    def bind():  # type: ignore[no-untyped-def]
        return bind_r31_runtime_inputs_v1(
            plan, module, snapshot, device=torch.device("cuda:0")
        )

    return plan, trace, bind(), bind()


def _native_capture(plan, tensors, schedule):  # type: ignore[no-untyped-def]
    alpha = tensors[plan.p_alpha_input_ordinal]
    optimizer = torch.optim.Adam([alpha], lr=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    rows = []
    for action in schedule.actions:
        alpha_before = alpha.detach().cpu().clone()
        lower = _evaluate_full_region(plan, tensors)
        gradient = torch.autograd.grad(-lower.sum(), alpha)[0]
        if action.update_after:
            optimizer.zero_grad(set_to_none=True)
            alpha.grad = gradient
            optimizer.step()
            with torch.no_grad():
                alpha.clamp_(0.0, 1.0)
            scheduler.step()
        state = optimizer.state[alpha]
        rows.append(
            {
                "alpha_before": alpha_before,
                "lower": lower.detach().cpu().clone(),
                "gradient": gradient.detach().cpu().clone(),
                "alpha_after": alpha.detach().cpu().clone(),
                "step": float(state["step"].item()),
                "exp_avg": state["exp_avg"].detach().cpu().clone(),
                "exp_avg_sq": state["exp_avg_sq"].detach().cpu().clone(),
            }
        )
    return rows


def test_s3_direct_vjp_matches_native_full_optimizer_trajectory() -> None:
    plan, trace, native_tensors, candidate_tensors = _fixture()
    schedule = compile_terminal_optimizer_schedule_v1()
    stream = torch.cuda.Stream(device="cuda:0")
    with torch.cuda.stream(stream):
        native = _native_capture(plan, native_tensors, schedule)
    stream.synchronize()
    candidate = PreparedS2CrownProgramV1(plan, trace, candidate_tensors)
    with torch.cuda.stream(stream):
        result = execute_asplos27_s3_optimizer_v1(
            plan, candidate_tensors, schedule, candidate, capture=True
        )
    stream.synchronize()
    assert len(result.steps) == len(native) == 10
    for expected, observed in zip(native, result.steps):
        for name, tolerance in (
            ("alpha_before", 2e-5),
            ("lower", 2e-4),
            ("gradient", 2e-5),
            ("alpha_after", 2e-5),
            ("exp_avg", 2e-5),
            ("exp_avg_sq", 2e-5),
        ):
            actual = getattr(observed, f"optimizer_{name}", None)
            if actual is None:
                actual = getattr(observed, name)
            assert torch.allclose(
                actual, expected[name], atol=tolerance, rtol=tolerance
            )
        assert observed.optimizer_step == expected["step"]
        assert torch.equal(torch.sign(observed.lower), torch.sign(expected["lower"]))
        assert torch.equal(
            torch.sign(observed.gradient), torch.sign(expected["gradient"])
        )
    assert torch.allclose(
        result.terminal_lower.cpu(), native[-1]["lower"], atol=2e-4, rtol=2e-4
    )
    assert torch.allclose(
        result.terminal_alpha.cpu(), native[-1]["alpha_after"], atol=2e-5, rtol=2e-5
    )


def test_s3_receipt_proves_direct_vjp_and_host_policy_cut() -> None:
    plan, trace, _, tensors = _fixture()
    schedule = compile_terminal_optimizer_schedule_v1()
    candidate = PreparedS2CrownProgramV1(plan, trace, tensors)
    stream = torch.cuda.Stream(device="cuda:0")
    with torch.cuda.stream(stream):
        result = execute_asplos27_s3_optimizer_v1(
            plan, tensors, schedule, candidate, capture=False
        )
    stream.synchronize()
    receipt = result.receipt
    assert receipt.custom_forward_count == receipt.custom_backward_count == 10
    assert receipt.forward_graph_replay_count == 10
    assert receipt.selected_graph_replay_count == 0
    assert receipt.selected_vm_invocation_count == 10
    assert receipt.selected_output_copy_count == 10
    assert receipt.warm_dlpack_view_count == 0
    assert receipt.host_policy_cut_count == 10
    assert receipt.autograd_function_count == receipt.executor_registry_count == 0
    assert receipt.fallback_count == receipt.eager_candidate_count == 0
    assert receipt.saved_dense_a_count == 0
    assert receipt.performance_claimed is False
    payload = receipt.to_dict()
    assert len(str(payload["receipt_hash"])) == 64


def test_s3_receipt_rejects_resigned_counter_and_claim_tamper() -> None:
    plan, trace, _, tensors = _fixture()
    schedule = compile_terminal_optimizer_schedule_v1()
    candidate = PreparedS2CrownProgramV1(plan, trace, tensors)
    stream = torch.cuda.Stream(device="cuda:0")
    with torch.cuda.stream(stream):
        receipt = execute_asplos27_s3_optimizer_v1(
            plan, tensors, schedule, candidate
        ).receipt
    stream.synchronize()
    for changed in (
        replace(receipt, evaluation_count=9),
        replace(receipt, custom_backward_count=9),
        replace(receipt, host_policy_cut_count=9),
        replace(receipt, selected_vm_invocation_count=9),
        replace(receipt, autograd_function_count=1),
        replace(receipt, fallback_count=1),
        replace(receipt, saved_dense_a_count=1),
        replace(receipt, performance_claimed=True),
    ):
        with pytest.raises(ValueError, match="S3 optimizer execution receipt differs"):
            changed.validate()


def test_s3_hot_wrapper_has_no_legacy_autograd_registry_path() -> None:
    source = getsource(execute_asplos27_s3_optimizer_v1)
    assert "_candidate_evaluate" not in source
    assert "_EXECUTOR_REGISTRY" not in source
    assert "autograd.grad" not in source
    assert "candidate.forward()" in source
    assert "candidate.backward(candidate.upstream_gradient)" in source
