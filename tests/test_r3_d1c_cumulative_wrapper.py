"""D1-C cumulative wrapper correctness and ownership gates."""

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
from boundflow.runtime.r3_optimizer_trajectory_timing import (
    execute_r32b_wrapper_v1,
    PreparedR32BTimingCandidateV1,
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
        pytest.skip("R3-D1C frozen CUDA fixture is unavailable")
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


def test_r3d1c_cumulative_wrapper_matches_frozen_r3_2b_candidate() -> None:
    plan, trace, reference_tensors = _fixture()
    _, _, candidate_tensors = _fixture()
    reference = _execute(
        plan,
        reference_tensors,
        PreparedR32BTimingCandidateV1(plan, trace, reference_tensors),
    )
    prepared = PreparedR3D1CCumulativeCandidateV1(plan, trace, candidate_tensors)
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
    receipt = prepared.d1c_receipt()
    assert receipt.launch_count == 4
    assert receipt.scratch_region_count == 2
    assert receipt.persistent_dense_a is False
    assert receipt.wrapper_performance_claimed is False


def test_r3d1c_schedule_receipt_rejects_claim_or_ownership_drift() -> None:
    plan, trace, tensors = _fixture()
    prepared = PreparedR3D1CCumulativeCandidateV1(plan, trace, tensors)
    prepared.d1c_launch_count = 4
    prepared.d1c_bias_inplace_alias_count = 2
    receipt = prepared.d1c_receipt()
    with pytest.raises(ValueError, match="cumulative receipt differs"):
        replace(receipt, wrapper_performance_claimed=True).validate()
    with pytest.raises(ValueError, match="cumulative receipt differs"):
        replace(receipt, persistent_dense_a=True).validate()
