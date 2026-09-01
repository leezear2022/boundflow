"""D1-A residual6 staged factorization correctness gates."""

# pylint: disable=missing-function-docstring,protected-access,too-many-locals

from pathlib import Path

import pytest
import torch
import torch.nn.functional as torch_functional

from boundflow.backends.tvm.r3_d1_residual6_staged import (
    build_r3d1_residual6_staged_modules_v1,
    R3D1_RESIDUAL6_SYMBOLS,
)
from boundflow.backends.tvm.r3_full_lower_forward import (
    R31B1_LINEAR14_SYMBOL,
    R31B1_LINEAR16_SYMBOL,
    R31B1_RESIDUAL11_SYMBOL,
    R31B1_RESIDUAL6_SYMBOL,
    R31B1_SEED_SYMBOL,
)
from boundflow.frontends.onnx.frontend import import_onnx
from boundflow.planner import plan_interval_ibp_v0
from boundflow.runtime.r3_bounded_arena_trace_compiler import (
    compile_r31b_bounded_arena_trace_v1,
)
from boundflow.runtime.r3_d1_residual6_staged import (
    execute_r3d1_residual6_staged_v1,
    R3D1Residual6ModuleCacheV1,
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


@pytest.fixture(scope="module")
def production_objects():  # type: ignore[no-untyped-def]
    if not MODEL.is_file() or not torch.cuda.is_available():
        pytest.skip("R3-D1 residual6 production fixture is unavailable")
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


def _oracle(
    incoming: torch.Tensor,
    weight4: torch.Tensor,
    lower19: torch.Tensor,
    upper19: torch.Tensor,
    alpha19: torch.Tensor,
    alpha_map19: torch.Tensor,
    weight2: torch.Tensor,
    weight5: torch.Tensor,
    bias4: torch.Tensor,
    bias2: torch.Tensor,
    bias5: torch.Tensor,
    bias_in: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    incoming_image = incoming[:6144].reshape(6, 16, 8, 8)
    staged_image = torch_functional.conv_transpose2d(
        incoming_image, weight4, stride=1, padding=1
    )
    staged = staged_image.reshape(6, 1024)
    lookup = alpha_map19.long()
    compact = alpha19[0, 0, :, lookup.clamp_min(0)]
    lower_alpha = torch.where(
        lookup.unsqueeze(0) >= 0,
        compact.clamp(0.0, 1.0),
        torch.zeros_like(compact),
    )
    ambiguous = (lower19 < 0.0) & (upper19 > 0.0)
    lower_slope = torch.where(
        ambiguous, lower_alpha, torch.where(lower19 >= 0.0, 1.0, 0.0)
    )
    upper_slope = torch.where(
        lower19 >= 0.0,
        1.0,
        torch.where(
            upper19 <= 0.0,
            0.0,
            upper19 / (upper19 - lower19).clamp_min(torch.finfo(torch.float32).eps),
        ),
    )
    slope = torch.where(staged >= 0.0, lower_slope, upper_slope)
    intercept = torch.where((staged < 0.0) & ambiguous, -lower19 * upper_slope, 0.0)
    main = torch_functional.conv_transpose2d(
        (staged * slope).reshape(6, 16, 8, 8),
        weight2,
        stride=2,
        padding=1,
        output_padding=1,
    )
    shortcut = torch_functional.conv_transpose2d(
        incoming_image, weight5, stride=2, output_padding=1
    )
    incoming_bias = (incoming_image * (bias4 + bias5).reshape(1, 16, 1, 1)).sum(
        dim=(1, 2, 3)
    )
    staged_bias = (
        staged * intercept
        + staged * slope * bias2.repeat_interleave(64).reshape(1, 1024)
    ).sum(dim=1)
    return (main + shortcut).reshape(12_288), bias_in + incoming_bias + staged_bias


def test_r3d1_residual6_builds_two_symbols_without_global_workspace() -> None:
    _unscheduled, scheduled = build_r3d1_residual6_staged_modules_v1()
    symbols = tuple(
        sorted(
            str(function.attrs["global_symbol"])
            for function in scheduled.functions.values()
        )
    )
    assert symbols == tuple(sorted(R3D1_RESIDUAL6_SYMBOLS))
    assert 'scope="global"' not in scheduled.script()


def test_r3d1_residual6_staged_matches_v1_and_independent_oracle(
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
            s0, s1, bias = (
                prepared.scratch_0,
                prepared.scratch_1,
                prepared.bias_accumulator,
            )
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
            prepared._relu("23", s1[:6144], bias)
            bias_before = bias.clone()
            oracle_output, oracle_bias = _oracle(
                s1,
                prepared._tensor("param/layer1.0.conv2.weight"),
                prepared._tensor("relu/19/lower").reshape(6, 1024),
                prepared._tensor("relu/19/upper").reshape(6, 1024),
                prepared._tensor("relu/19/alpha"),
                prepared.alpha_maps["19"],
                prepared._tensor("param/layer1.0.conv1.weight"),
                prepared._tensor("param/layer1.0.shortcut.0.weight"),
                prepared._tensor("param/layer1.0.conv2.bias"),
                prepared._tensor("param/layer1.0.conv1.bias"),
                prepared._tensor("param/layer1.0.shortcut.0.bias"),
                bias_before,
            )
            prepared._launch(
                R31B1_RESIDUAL6_SYMBOL,
                s1,
                prepared._tensor("param/layer1.0.conv2.weight"),
                prepared._tensor("param/layer1.0.conv2.bias"),
                prepared._tensor("relu/19/lower").reshape(6, 1024),
                prepared._tensor("relu/19/upper").reshape(6, 1024),
                prepared._tensor("relu/19/alpha"),
                prepared.alpha_maps["19"],
                prepared._tensor("param/layer1.0.conv1.weight"),
                prepared._tensor("param/layer1.0.conv1.bias"),
                prepared._tensor("param/layer1.0.shortcut.0.weight"),
                prepared._tensor("param/layer1.0.shortcut.0.bias"),
                bias,
                s0,
            )
            reference_output = s0[:12_288].clone()
            reference_bias = bias.clone()
        scratch = torch.empty(6144, device="cuda")
        output = torch.empty(12_288, device="cuda")
        candidate_bias = torch.empty(6, device="cuda")
        receipt = execute_r3d1_residual6_staged_v1(
            s1,
            prepared._tensor("param/layer1.0.conv2.weight"),
            prepared._tensor("relu/19/lower").reshape(6, 1024),
            prepared._tensor("relu/19/upper").reshape(6, 1024),
            prepared._tensor("relu/19/alpha"),
            prepared.alpha_maps["19"],
            prepared._tensor("param/layer1.0.conv1.weight"),
            prepared._tensor("param/layer1.0.shortcut.0.weight"),
            prepared._tensor("param/layer1.0.conv2.bias"),
            prepared._tensor("param/layer1.0.conv1.bias"),
            prepared._tensor("param/layer1.0.shortcut.0.bias"),
            bias_before,
            scratch,
            output,
            candidate_bias,
            cache=R3D1Residual6ModuleCacheV1(),
        )
    stream.synchronize()
    for candidate, reference in (
        (output, reference_output),
        (output, oracle_output),
        (reference_output, oracle_output),
        (candidate_bias, reference_bias),
        (candidate_bias, oracle_bias),
        (reference_bias, oracle_bias),
    ):
        assert torch.allclose(candidate, reference, atol=2e-4, rtol=2e-4)
        assert float((candidate - reference).abs().max()) <= 2e-4
    assert torch.equal(torch.sign(output), torch.sign(oracle_output))
    assert receipt.launch_count == 2
    assert receipt.scratch_count == 1
    assert receipt.performance_claimed is False


def test_r3d1_residual6_contract_and_default_stream_fail_closed() -> None:
    if not torch.cuda.is_available():
        pytest.skip("R3-D1 residual6 CUDA device is unavailable")
    device = torch.device("cuda:0")

    def arguments():  # type: ignore[no-untyped-def]
        return [
            torch.zeros(18_432, device=device),
            torch.zeros((16, 16, 3, 3), device=device),
            torch.zeros((6, 1024), device=device),
            torch.ones((6, 1024), device=device),
            torch.zeros((2, 1, 6, 132), device=device),
            torch.zeros(1024, device=device, dtype=torch.int32),
            torch.zeros((16, 8, 3, 3), device=device),
            torch.zeros((16, 8, 1, 1), device=device),
            torch.zeros(16, device=device),
            torch.zeros(16, device=device),
            torch.zeros(16, device=device),
            torch.zeros(6, device=device),
            torch.zeros(6144, device=device),
            torch.zeros(12_288, device=device),
            torch.zeros(6, device=device),
        ]

    with pytest.raises(RuntimeError, match="non-default stream"):
        execute_r3d1_residual6_staged_v1(
            *arguments(), cache=R3D1Residual6ModuleCacheV1()
        )
    cases = []
    wrong_shape = arguments()
    wrong_shape[0] = wrong_shape[0][:-1]
    cases.append((wrong_shape, "tensor contract"))
    wrong_dtype = arguments()
    wrong_dtype[5] = wrong_dtype[5].long()
    cases.append((wrong_dtype, "tensor contract"))
    nonfinite = arguments()
    nonfinite[3][0, 0] = torch.inf
    cases.append((nonfinite, "tensor contract"))
    wrong_device = arguments()
    wrong_device[8] = wrong_device[8].cpu()
    cases.append((wrong_device, "tensor contract"))
    alias = arguments()
    alias[13] = alias[0][:12_288]
    cases.append((alias, "arena alias"))
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for values, error in cases:
            with pytest.raises(ValueError, match=error):
                execute_r3d1_residual6_staged_v1(
                    *values, cache=R3D1Residual6ModuleCacheV1()
                )
