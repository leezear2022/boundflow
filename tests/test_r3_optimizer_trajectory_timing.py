"""R3-2B capture-free wrapper correctness and isolation gates."""

# pylint: disable=missing-function-docstring,protected-access,too-many-locals

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
        pytest.skip("R3-2B frozen CUDA fixture is unavailable")
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


def test_r32b_capture_free_wrapper_matches_native_terminal_state() -> None:
    plan, trace, native_tensors = _fixture()
    _, _, candidate_tensors = _fixture()
    schedule = compile_terminal_optimizer_schedule_v1()
    initial = native_tensors[plan.p_alpha_input_ordinal].detach().clone()
    native_stream = torch.cuda.Stream()
    with torch.cuda.stream(native_stream):
        native = execute_r32b_wrapper_v1(plan, native_tensors, schedule, candidate=None)
    native_stream.synchronize()
    prepared = PreparedR32BTimingCandidateV1(plan, trace, candidate_tensors)
    candidate_stream = torch.cuda.Stream()
    with torch.cuda.stream(candidate_stream):
        candidate = execute_r32b_wrapper_v1(
            plan, candidate_tensors, schedule, candidate=prepared
        )
    candidate_stream.synchronize()
    assert torch.allclose(
        native.terminal_lower, candidate.terminal_lower, atol=2e-4, rtol=2e-4
    )
    assert torch.equal(
        torch.sign(native.terminal_lower), torch.sign(candidate.terminal_lower)
    )
    assert torch.allclose(
        native.terminal_alpha, candidate.terminal_alpha, atol=2e-5, rtol=2e-5
    )
    assert not torch.equal(initial, native.terminal_alpha)
    assert candidate.custom_forward_count == candidate.custom_backward_count == 10


def test_r32b_timed_candidate_source_has_no_capture_or_native_shadow() -> None:
    source = getsource(execute_r32b_wrapper_v1)
    assert "production_tensor_sha256" not in source
    assert ".cpu(" not in source
    assert "synchronize" not in source
    assert "reset_peak_memory_stats" not in source
    candidate_branch = source.split("else:", 1)[1]
    assert "_evaluate_full_region" not in candidate_branch
