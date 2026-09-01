"""D1-A residual11 staged factorization correctness gates."""

# pylint: disable=missing-function-docstring,protected-access,too-many-locals

from pathlib import Path

import pytest
import torch
import torch.nn.functional as torch_functional

from boundflow.backends.tvm.r3_d1_residual11_staged import (
    build_r3d1_residual11_staged_modules_v1,
    R3D1_RESIDUAL11_SYMBOLS,
)
from boundflow.backends.tvm.r3_full_lower_forward import (
    R31B1_LINEAR14_SYMBOL,
    R31B1_LINEAR16_SYMBOL,
    R31B1_RESIDUAL11_SYMBOL,
    R31B1_SEED_SYMBOL,
)
from boundflow.frontends.onnx.frontend import import_onnx
from boundflow.planner import plan_interval_ibp_v0
from boundflow.runtime.r3_bounded_arena_trace_compiler import (
    compile_r31b_bounded_arena_trace_v1,
)
from boundflow.runtime.r3_d1_residual11_staged import (
    execute_r3d1_residual11_staged_v1,
    R3D1Residual11ModuleCacheV1,
)
from boundflow.runtime.r3_full_lower_forward_tir import (
    PreparedR31B1FullLowerForwardV1,
    R31B1ModuleCacheV1,
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


def _independent_residual11_oracle(
    incoming: torch.Tensor,
    weight10: torch.Tensor,
    lower25: torch.Tensor,
    upper25: torch.Tensor,
    alpha25: torch.Tensor,
    alpha_map25: torch.Tensor,
    weight8: torch.Tensor,
    bias10: torch.Tensor,
    bias8: torch.Tensor,
    bias_in: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Closed PyTorch formula; it does not call either TIR implementation."""

    incoming_image = incoming[:6144].reshape(6, 16, 8, 8)
    staged_image = torch_functional.conv_transpose2d(
        incoming_image, weight10, stride=1, padding=1
    )
    staged = staged_image.reshape(6, 1024)
    lookup = alpha_map25.to(torch.long)
    safe_lookup = lookup.clamp_min(0)
    compact = alpha25[0, 0, :, safe_lookup]
    lower_alpha = torch.where(
        lookup.unsqueeze(0) >= 0,
        compact.clamp(0.0, 1.0),
        torch.zeros_like(compact),
    )
    ambiguous = (lower25 < 0.0) & (upper25 > 0.0)
    lower_slope = torch.where(
        ambiguous,
        lower_alpha,
        torch.where(lower25 >= 0.0, 1.0, 0.0),
    )
    upper_slope = torch.where(
        lower25 >= 0.0,
        1.0,
        torch.where(
            upper25 <= 0.0,
            0.0,
            upper25 / (upper25 - lower25).clamp_min(torch.finfo(torch.float32).eps),
        ),
    )
    slope = torch.where(staged >= 0.0, lower_slope, upper_slope)
    intercept = torch.where(
        (staged < 0.0) & ambiguous,
        -lower25 * upper_slope,
        0.0,
    )
    coefficient = (staged * slope).reshape(6, 16, 8, 8)
    output = incoming_image + torch_functional.conv_transpose2d(
        coefficient, weight8, stride=1, padding=1
    )
    incoming_bias = (incoming_image * bias10.reshape(1, 16, 1, 1)).sum(dim=(1, 2, 3))
    staged_bias = (
        staged * intercept
        + staged * slope * bias8.repeat_interleave(64).reshape(1, 1024)
    ).sum(dim=1)
    return output.reshape(6144), bias_in + incoming_bias + staged_bias


@pytest.fixture(scope="module")
def production_objects():  # type: ignore[no-untyped-def]
    if not MODEL.is_file() or not torch.cuda.is_available():
        pytest.skip("R3-D1 production CUDA fixture is unavailable")
    properties = torch.cuda.get_device_properties(0)
    if (properties.major, properties.minor) != (8, 9):
        pytest.skip("R3-D1 frozen sm_89 device is unavailable")
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


def test_r3d1_residual11_builds_two_symbols_without_global_workspace() -> None:
    _unscheduled, scheduled = build_r3d1_residual11_staged_modules_v1()
    symbols = tuple(
        sorted(
            str(function.attrs["global_symbol"])
            for function in scheduled.functions.values()
        )
    )
    assert symbols == tuple(sorted(R3D1_RESIDUAL11_SYMBOLS))
    assert 'scope="global"' not in scheduled.script()


def test_r3d1_residual11_staged_matches_frozen_v1_on_production_state(
    production_objects,
) -> None:
    import tvm_ffi

    plan, trace, tensors = production_objects
    prepared = PreparedR31B1FullLowerForwardV1(
        plan, trace, tensors, cache=R31B1ModuleCacheV1()
    )
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        with tvm_ffi.use_torch_stream(torch.cuda.stream(stream)):
            s0 = prepared.scratch_0
            s1 = prepared.scratch_1
            bias = prepared.bias_accumulator
            prepared._launch(
                R31B1_SEED_SYMBOL, prepared._tensor("objective"), s0[:60], bias
            )
            prepared._launch(
                R31B1_LINEAR16_SYMBOL,
                s0[:60],
                prepared._tensor("param/linear2.weight"),
                prepared._tensor("param/linear2.bias"),
                bias,
                s1[:600],
                bias,
            )
            prepared._relu("31", s1[:600], bias, active_beta=True)
            prepared._launch(
                R31B1_LINEAR14_SYMBOL,
                s1[:600],
                prepared._tensor("param/linear1.weight"),
                prepared._tensor("param/linear1.bias"),
                bias,
                s0[:6144],
                bias,
            )
            prepared._relu("28", s0[:6144], bias)
            bias_before = bias.clone()
            prepared._launch(
                R31B1_RESIDUAL11_SYMBOL,
                s0,
                prepared._tensor("param/layer1.1.conv2.weight"),
                prepared._tensor("param/layer1.1.conv2.bias"),
                prepared._tensor("relu/25/lower").reshape(6, 1024),
                prepared._tensor("relu/25/upper").reshape(6, 1024),
                prepared._tensor("relu/25/alpha"),
                prepared.alpha_maps["25"],
                prepared._tensor("param/layer1.1.conv1.weight"),
                prepared._tensor("param/layer1.1.conv1.bias"),
                bias,
                s1,
            )
            reference_output = s1[:6144].clone()
            reference_bias = bias.clone()
            oracle_output, oracle_bias = _independent_residual11_oracle(
                s0,
                prepared._tensor("param/layer1.1.conv2.weight"),
                prepared._tensor("relu/25/lower").reshape(6, 1024),
                prepared._tensor("relu/25/upper").reshape(6, 1024),
                prepared._tensor("relu/25/alpha"),
                prepared.alpha_maps["25"],
                prepared._tensor("param/layer1.1.conv1.weight"),
                prepared._tensor("param/layer1.1.conv2.bias"),
                prepared._tensor("param/layer1.1.conv1.bias"),
                bias_before,
            )
        scratch = torch.empty(6144, device="cuda", dtype=torch.float32)
        output = torch.empty_like(scratch)
        candidate_bias = torch.empty_like(bias_before)
        receipt = execute_r3d1_residual11_staged_v1(
            s0,
            prepared._tensor("param/layer1.1.conv2.weight"),
            prepared._tensor("relu/25/lower").reshape(6, 1024),
            prepared._tensor("relu/25/upper").reshape(6, 1024),
            prepared._tensor("relu/25/alpha"),
            prepared.alpha_maps["25"],
            prepared._tensor("param/layer1.1.conv1.weight"),
            prepared._tensor("param/layer1.1.conv2.bias"),
            prepared._tensor("param/layer1.1.conv1.bias"),
            bias_before,
            scratch,
            output,
            candidate_bias,
            cache=R3D1Residual11ModuleCacheV1(),
        )
    stream.synchronize()
    assert torch.allclose(output, reference_output, atol=2e-4, rtol=2e-4)
    assert torch.equal(torch.sign(output), torch.sign(reference_output))
    assert float((output - reference_output).abs().max()) <= 2e-4
    assert torch.allclose(candidate_bias, reference_bias, atol=2e-4, rtol=2e-4)
    assert float((candidate_bias - reference_bias).abs().max()) <= 2e-4
    assert torch.allclose(output, oracle_output, atol=2e-4, rtol=2e-4)
    assert torch.allclose(reference_output, oracle_output, atol=2e-4, rtol=2e-4)
    assert torch.allclose(candidate_bias, oracle_bias, atol=2e-4, rtol=2e-4)
    assert torch.allclose(reference_bias, oracle_bias, atol=2e-4, rtol=2e-4)
    assert receipt.launch_count == 2
    assert receipt.scratch_count == 1
    assert receipt.performance_claimed is False


def test_r3d1_residual11_default_stream_fails_before_compile() -> None:
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        pytest.skip("R3-D1 CUDA device is unavailable")
    arguments = (
        torch.zeros(18_432, device=device),
        torch.zeros((16, 16, 3, 3), device=device),
        torch.zeros((6, 1024), device=device),
        torch.ones((6, 1024), device=device),
        torch.zeros((2, 1, 6, 86), device=device),
        torch.zeros(1024, device=device, dtype=torch.int32),
        torch.zeros((16, 16, 3, 3), device=device),
        torch.zeros(16, device=device),
        torch.zeros(16, device=device),
        torch.zeros(6, device=device),
        torch.zeros(6144, device=device),
        torch.zeros(6144, device=device),
        torch.zeros(6, device=device),
    )
    with pytest.raises(RuntimeError, match="non-default stream"):
        execute_r3d1_residual11_staged_v1(
            *arguments, cache=R3D1Residual11ModuleCacheV1()
        )


def test_r3d1_residual11_shape_dtype_nonfinite_device_and_alias_fail_closed() -> None:
    if not torch.cuda.is_available():
        pytest.skip("R3-D1 CUDA device is unavailable")
    device = torch.device("cuda:0")

    def valid_arguments():  # type: ignore[no-untyped-def]
        return [
            torch.zeros(18_432, device=device),
            torch.zeros((16, 16, 3, 3), device=device),
            torch.zeros((6, 1024), device=device),
            torch.ones((6, 1024), device=device),
            torch.zeros((2, 1, 6, 86), device=device),
            torch.zeros(1024, device=device, dtype=torch.int32),
            torch.zeros((16, 16, 3, 3), device=device),
            torch.zeros(16, device=device),
            torch.zeros(16, device=device),
            torch.zeros(6, device=device),
            torch.zeros(6144, device=device),
            torch.zeros(6144, device=device),
            torch.zeros(6, device=device),
        ]

    mutations = []
    shape = valid_arguments()
    shape[0] = shape[0][:-1]
    mutations.append((shape, "tensor contract"))
    dtype = valid_arguments()
    dtype[5] = dtype[5].to(torch.int64)
    mutations.append((dtype, "tensor contract"))
    nonfinite = valid_arguments()
    nonfinite[2][0, 0] = torch.nan
    mutations.append((nonfinite, "tensor contract"))
    wrong_device = valid_arguments()
    wrong_device[7] = wrong_device[7].cpu()
    mutations.append((wrong_device, "tensor contract"))
    alias = valid_arguments()
    alias[11] = alias[10]
    mutations.append((alias, "arena alias"))
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for arguments, error in mutations:
            with pytest.raises(ValueError, match=error):
                execute_r3d1_residual11_staged_v1(
                    *arguments, cache=R3D1Residual11ModuleCacheV1()
                )
