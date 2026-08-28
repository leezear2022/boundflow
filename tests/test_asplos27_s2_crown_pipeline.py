"""Correctness, ownership, and fail-closed gates for ASPLOS'27 S2."""

# pylint: disable=missing-function-docstring,too-many-locals
# pylint: disable=protected-access,duplicate-code

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from boundflow.frontends.onnx.frontend import import_onnx
from boundflow.planner import plan_interval_ibp_v0
from boundflow.runtime.asplos27_s2_crown_pipeline import (
    PreparedS2CrownProgramV1,
)
from boundflow.runtime.r3_bounded_arena_trace_compiler import (
    compile_r31b_bounded_arena_trace_v1,
)
from boundflow.runtime.r3_d2b_staged_backward import (
    PreparedR3D2BStagedBackwardCandidateV1,
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


def _fixture():
    if not MODEL.is_file() or not CAPTURE.is_file() or not torch.cuda.is_available():
        pytest.skip("S2 frozen CUDA fixture is unavailable")
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


def _candidate_vjp(prepared):  # type: ignore[no-untyped-def]
    stream = torch.cuda.Stream(device=prepared.device)
    with torch.cuda.stream(stream):
        prepared.begin_sample()
        prepared.begin_evaluation(0)
        lower = prepared.forward()
        gradient = prepared.backward(prepared.upstream_gradient)
    stream.synchronize()
    return lower.detach().clone(), gradient.detach().clone()


def test_s2_matches_independent_native_and_previous_direct_vjp() -> None:
    plan, trace, native_tensors = _fixture()
    _, _, direct_tensors = _fixture()
    _, _, candidate_tensors = _fixture()
    native_alpha = native_tensors[plan.p_alpha_input_ordinal]
    native_lower = _evaluate_full_region(plan, native_tensors)
    native_gradient = torch.autograd.grad(-native_lower.sum(), native_alpha)[0]
    direct = PreparedR3D2BStagedBackwardCandidateV1(plan, trace, direct_tensors)
    direct_lower, direct_gradient = _candidate_vjp(direct)
    candidate = PreparedS2CrownProgramV1(plan, trace, candidate_tensors)
    stream = torch.cuda.Stream(device=candidate.device)
    with torch.cuda.stream(stream):
        candidate_lower, candidate_gradient = candidate.run_vjp()
    stream.synchronize()
    for expected_lower, expected_gradient in (
        (native_lower, native_gradient),
        (direct_lower, direct_gradient),
    ):
        assert torch.allclose(candidate_lower, expected_lower, atol=2e-4, rtol=2e-4)
        assert torch.allclose(
            candidate_gradient, expected_gradient, atol=2e-4, rtol=2e-4
        )
        assert torch.equal(torch.sign(candidate_lower), torch.sign(expected_lower))
        assert torch.equal(
            torch.sign(candidate_gradient), torch.sign(expected_gradient)
        )


def test_s2_receipt_proves_five_cudnn_calls_and_no_dense_saved_a() -> None:
    plan, trace, tensors = _fixture()
    prepared = PreparedS2CrownProgramV1(plan, trace, tensors)
    stream = torch.cuda.Stream(device=prepared.device)
    with torch.cuda.stream(stream):
        prepared.run_vjp()
    stream.synchronize()
    receipt = prepared.execution_receipt()
    assert receipt.cudnn_partition_function_count == 4
    assert receipt.cudnn_conv_call_count == 5
    assert receipt.selected_tir_count == 4
    assert receipt.selected_graph_replay_count == 1
    assert receipt.active_beta is True
    assert receipt.saved_dense_a_count == 0
    assert receipt.saved_autograd_history is False
    assert receipt.warm_dlpack_view_count == 0
    assert receipt.fallback_count == receipt.eager_candidate_count == 0
    assert receipt.performance_claimed is False
    assert receipt.output_pointer == prepared.pre25_value.data_ptr()


def test_s2_receipt_rejects_resigned_semantic_and_claim_tamper() -> None:
    plan, trace, tensors = _fixture()
    prepared = PreparedS2CrownProgramV1(plan, trace, tensors)
    stream = torch.cuda.Stream(device=prepared.device)
    with torch.cuda.stream(stream):
        prepared.run_vjp()
    stream.synchronize()
    receipt = prepared.execution_receipt()
    mutations = (
        replace(receipt, performance_claimed=True),
        replace(receipt, cudnn_conv_call_count=4),
        replace(receipt, selected_graph_replay_count=2),
        replace(receipt, active_beta=False),
        replace(receipt, saved_dense_a_count=1),
        replace(receipt, fallback_count=1),
        replace(receipt, selected_lowered_relax_ir_hash="0" * 63),
    )
    for changed in mutations:
        with pytest.raises(ValueError, match="S2 CROWN execution receipt differs"):
            changed.validate()


def test_s2_rejects_default_stream_before_graph_launch() -> None:
    plan, trace, tensors = _fixture()
    prepared = PreparedS2CrownProgramV1(plan, trace, tensors)
    before = prepared.selected_value.replay_count
    with torch.cuda.stream(torch.cuda.default_stream(prepared.device)):
        with pytest.raises(RuntimeError, match="non-default stream"):
            prepared.selected_value.replay()
    assert prepared.selected_value.replay_count == before


def test_s2_immutable_state_mutation_is_rejected_before_vjp() -> None:
    plan, trace, tensors = _fixture()
    prepared = PreparedS2CrownProgramV1(plan, trace, tensors)
    immutable = tensors[0]
    if immutable is tensors[plan.p_alpha_input_ordinal]:
        immutable = tensors[1]
    immutable.add_(1.0)
    stream = torch.cuda.Stream(device=prepared.device)
    with torch.cuda.stream(stream):
        with pytest.raises(ValueError, match="immutable version drifted"):
            prepared.run_vjp()
    assert prepared.selected_value.replay_count == 0
