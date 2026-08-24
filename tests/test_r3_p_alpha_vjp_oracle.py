"""Independent mathematical reduction gate before R3-1b2 TIR implementation."""

# pylint: disable=missing-function-docstring,protected-access,too-many-locals

from pathlib import Path

import pytest
import torch

from boundflow.frontends.onnx.frontend import import_onnx
from boundflow.planner import plan_interval_ibp_v0
from boundflow.runtime.alpha_beta_crown import (
    BetaState,
    _beta_to_relu_pre_add_coeff,
)
from boundflow.runtime.crown_ibp import (
    _forward_ibp_trace_mlp,
    run_crown_ibp_mlp_with_relu_lower_coefficients_from_forward_trace,
)
from boundflow.runtime.r3_bounded_arena_trace_compiler import (
    compile_r31b_bounded_arena_trace_v1,
)
from boundflow.runtime.r3_full_lower_forward_tir import (
    PreparedR31B1FullLowerForwardV1,
)
from boundflow.runtime.r3_p_alpha_vjp_oracle import (
    evaluate_r31b2_p_alpha_closed_form_v1,
)
from boundflow.runtime.r3_structured_owner_custom_backward import (
    _evaluate_full_region,
    _runtime_parts,
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


def test_r31b2_closed_form_matches_native_autograd_before_tir() -> None:
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
    runtime = _runtime_parts(plan, tensors)
    runtime_module, spec, objective, relu_pre, alphas, betas, splits = runtime
    interval_env, _ = _forward_ibp_trace_mlp(
        runtime_module, spec, relu_split_state=splits
    )
    beta_add = _beta_to_relu_pre_add_coeff(
        BetaState(betas), relu_pre=relu_pre, relu_split_state=splits
    )
    _bounds, coefficients = (
        run_crown_ibp_mlp_with_relu_lower_coefficients_from_forward_trace(
            runtime_module,
            spec,
            interval_env=interval_env,
            relu_pre=relu_pre,
            linear_spec_C=objective,
            relu_alpha=alphas,
            relu_pre_add_coeff_l=beta_add,
        )
    )
    p_alpha = tensors[plan.p_alpha_input_ordinal]
    native_lower = _evaluate_full_region(plan, tensors)
    native_gradient = torch.autograd.grad(-native_lower.sum(), p_alpha)[0]
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        prepared = PreparedR31B1FullLowerForwardV1(plan, trace, tensors)
        prepared.run()
    stream.synchronize()
    predicted = evaluate_r31b2_p_alpha_closed_form_v1(
        plan,
        tensors,
        input_lower_coefficient=prepared.scratch_1.reshape(6, 1, 3, 32, 32),
        relu_lower_coefficients=coefficients,
    )

    assert torch.allclose(predicted, native_gradient, atol=2e-4, rtol=2e-4)
    assert torch.equal(torch.sign(predicted), torch.sign(native_gradient))
    assert torch.count_nonzero(predicted) == torch.count_nonzero(native_gradient) == 281
    assert float((predicted - native_gradient).abs().max().item()) <= 5e-8
