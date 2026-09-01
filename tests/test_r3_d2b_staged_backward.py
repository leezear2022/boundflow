"""D2-B staged coefficient-sign correctness and ownership gates."""

# pylint: disable=missing-function-docstring,too-many-locals
# pylint: disable=protected-access,duplicate-code

from dataclasses import replace
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
from boundflow.runtime.r3_d1c_cumulative_wrapper import (
    PreparedR3D1CCumulativeCandidateV1,
)
from boundflow.runtime.r3_d2b_staged_backward import (
    PreparedR3D2BStagedBackwardCandidateV1,
)
from boundflow.runtime.r3_optimizer_trajectory_timing import (
    execute_r32b_wrapper_v1,
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
    if not MODEL.is_file() or not torch.cuda.is_available():
        pytest.skip("R3-D2B frozen CUDA fixture is unavailable")
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


def _execute(plan, tensors, candidate):  # type: ignore[no-untyped-def]
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        result = execute_r32b_wrapper_v1(
            plan,
            tensors,
            compile_terminal_optimizer_schedule_v1(),
            candidate=candidate,
        )
    stream.synchronize()
    return result


def test_r3d2b_staged_backward_matches_d1c_complete_wrapper() -> None:
    plan, trace, reference_tensors = _fixture()
    _, _, candidate_tensors = _fixture()
    reference = _execute(
        plan,
        reference_tensors,
        PreparedR3D1CCumulativeCandidateV1(plan, trace, reference_tensors),
    )
    prepared = PreparedR3D2BStagedBackwardCandidateV1(plan, trace, candidate_tensors)
    candidate = _execute(plan, candidate_tensors, prepared)
    assert torch.allclose(
        reference.terminal_lower, candidate.terminal_lower, atol=2e-4, rtol=2e-4
    )
    assert torch.equal(
        torch.sign(reference.terminal_lower), torch.sign(candidate.terminal_lower)
    )
    assert torch.allclose(
        reference.terminal_alpha, candidate.terminal_alpha, atol=2e-5, rtol=2e-5
    )
    assert candidate.evaluation_count == 10
    assert candidate.optimizer_mutation_count == candidate.scheduler_mutation_count == 9
    assert candidate.custom_forward_count == candidate.custom_backward_count == 10
    receipt = prepared.d2b_receipt()
    assert receipt.forward_staged_launch_count == 4
    assert receipt.backward_staged_launch_count == 4
    assert receipt.raw_b1_backward_launch_count == 13
    assert receipt.persistent_dense_a is False
    assert receipt.saved_autograd_history is False
    assert receipt.performance_claimed is False


def test_r3d2b_receipt_rejects_claim_ownership_launch_and_pointer_drift() -> None:
    plan, trace, tensors = _fixture()
    prepared = PreparedR3D2BStagedBackwardCandidateV1(plan, trace, tensors)
    prepared.d1c_launch_count = 4
    prepared.d2b_backward_staged_launch_count = 4
    prepared.d2b_backward_bias_inplace_alias_count = 2
    prepared.b1_backward_launch_count = 13
    receipt = prepared.d2b_receipt()
    mutations = (
        replace(receipt, performance_claimed=True),
        replace(receipt, persistent_dense_a=True),
        replace(receipt, saved_autograd_history=True),
        replace(receipt, backward_staged_launch_count=3),
        replace(receipt, scratch_region_pointers=(1, 1)),
        replace(receipt, fallback_count=1),
    )
    for changed in mutations:
        with pytest.raises(ValueError, match="staged backward receipt differs"):
            changed.validate()


def test_r3d2b_residual_abi_fails_closed() -> None:
    plan, trace, tensors = _fixture()
    prepared = PreparedR3D2BStagedBackwardCandidateV1(plan, trace, tensors)
    with pytest.raises(ValueError, match="residual11 ABI differs"):
        prepared._dispatch_d2b_residual11(())
    with pytest.raises(ValueError, match="residual6 ABI differs"):
        prepared._dispatch_d2b_residual6(())
