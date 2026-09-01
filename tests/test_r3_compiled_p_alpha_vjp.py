"""R3-1b2 compiled custom backward correctness and ownership gates."""

# pylint: disable=missing-function-docstring,too-many-locals,protected-access

from inspect import getsource
from pathlib import Path

import pytest
import torch

from boundflow.backends.tvm.r3_p_alpha_vjp import (
    R31B2_EXPORTED_SYMBOLS,
    build_r31b2_p_alpha_vjp_tir_v1,
    compile_r31b2_p_alpha_vjp_tir_v1,
)
from boundflow.frontends.onnx.frontend import import_onnx
from boundflow.planner import plan_interval_ibp_v0
from boundflow.runtime.r3_bounded_arena_trace_compiler import (
    compile_r31b_bounded_arena_trace_v1,
)
from boundflow.runtime.r3_compiled_p_alpha_vjp import (
    PreparedR31B2CompiledCustomBackwardV1,
    execute_r31b2_compiled_custom_backward_v1,
)
from boundflow.runtime.r3_structured_owner_custom_backward import (
    bind_r31_runtime_inputs_v1,
    compile_r31_full_region_plan_v1,
    execute_r31_native_oracle_v1,
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
        pytest.skip("R3-1b2 CUDA device is unavailable")
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


def test_r31b2_module_has_exact_symbols_and_no_global_workspace() -> None:
    if not torch.cuda.is_available():
        pytest.skip("R3-1b2 CUDA compiler is unavailable")
    module = build_r31b2_p_alpha_vjp_tir_v1()
    assert len(module.functions) == len(R31B2_EXPORTED_SYMBOLS) == 10
    assert tuple(sorted(str(value.name_hint) for value in module.functions)) == tuple(
        sorted(R31B2_EXPORTED_SYMBOLS)
    )
    compiled = compile_r31b2_p_alpha_vjp_tir_v1()
    assert compiled.exported_symbols == R31B2_EXPORTED_SYMBOLS
    assert compiled.global_workspace_bytes == 0
    assert all(symbol in compiled.device_source for symbol in R31B2_EXPORTED_SYMBOLS)


def test_r31b2_compiled_custom_backward_matches_native() -> None:
    plan, trace, tensors = _fixture()
    native_lower, native_gradient = execute_r31_native_oracle_v1(plan, tensors)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        candidate = execute_r31b2_compiled_custom_backward_v1(plan, trace, tensors)
    stream.synchronize()

    assert torch.allclose(candidate.final_lower, native_lower, atol=2e-4, rtol=2e-4)
    assert torch.equal(torch.sign(candidate.final_lower), torch.sign(native_lower))
    assert torch.allclose(
        candidate.compressed_alpha_gradient,
        native_gradient,
        atol=2e-4,
        rtol=2e-4,
    )
    assert torch.equal(
        torch.sign(candidate.compressed_alpha_gradient), torch.sign(native_gradient)
    )
    assert float((candidate.final_lower - native_lower).abs().max().item()) <= 5e-6
    assert (
        float(
            (candidate.compressed_alpha_gradient - native_gradient).abs().max().item()
        )
        <= 1e-7
    )
    assert (
        torch.count_nonzero(candidate.compressed_alpha_gradient)
        == torch.count_nonzero(native_gradient)
        == 281
    )
    receipt = candidate.receipt
    receipt.validate()
    assert receipt.sign_bitmap_bytes == 43_008
    assert receipt.saved_dense_a_count == 0
    assert receipt.warm_dynamic_allocated_bytes == 0
    assert receipt.compiled_vjp and receipt.custom_vjp
    assert not receipt.timing_recorded and not receipt.performance_claimed


def test_r31b2_default_stream_fails_closed() -> None:
    plan, trace, tensors = _fixture()
    prepared = PreparedR31B2CompiledCustomBackwardV1(plan, trace, tensors)
    torch.cuda.default_stream().synchronize()
    with pytest.raises(RuntimeError, match="non-default stream"):
        prepared.forward()


def test_r31b2_candidate_backward_does_not_call_autograd_or_oracle() -> None:
    source = getsource(PreparedR31B2CompiledCustomBackwardV1.backward)
    assert "torch.autograd.grad" not in source
    assert "_evaluate_full_region" not in source
    assert "execute_r31_native_oracle_v1" not in source
    assert "run_crown_ibp" not in source
