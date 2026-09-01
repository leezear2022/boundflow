"""Activation-BaB input streaming TIR and four-segment owner gates."""

# pylint: disable=missing-function-docstring,too-many-locals,duplicate-code
# pylint: disable=unnecessary-lambda
# pylint: disable=no-value-for-parameter

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import torch

from boundflow.backends.tvm.bab_input_domain import (
    BAB_INPUT_DOMAIN_BACKWARD_SYMBOL,
    BAB_INPUT_DOMAIN_FORWARD_SYMBOL,
    BabInputDomainTemplateV1,
    build_bab_input_domain_modules_v1,
)
from boundflow.backends.tvm.bab_terminal_linear import BabTerminalLinearTemplateV1
from boundflow.backends.tvm.root_crown_projection import (
    RootCrownProjectionTemplateV1,
)
from boundflow.backends.tvm.root_crown_residual import RootCrownResidualTemplateV1
from boundflow.runtime.bab_full_region_owner import (
    bab_full_region_inputs_from_capture_v1,
    BabFullRegionDynamicV1,
    PreparedBabFullRegionOwnerV1,
)
from boundflow.runtime.bab_input_domain_tir import (
    BabInputDomainTensorsV1,
    BabInputDomainTIRExecutorV1,
    execute_bab_input_domain_tir_v1,
    validate_bab_input_domain_tensors_v1,
)
from boundflow.runtime.bab_terminal_tir import BabTerminalTIRExecutorV1
from boundflow.runtime.root_crown_projection_tir import (
    RootCrownProjectionTIRExecutorV1,
)
from boundflow.runtime.root_crown_residual_tir import RootCrownResidualTIRExecutorV1

CAPTURE = Path("artifacts/bab-full-region-capture/resnet2b-prop0-v1/capture.pt")


def _payload() -> dict[str, Any]:
    if not CAPTURE.is_file():
        pytest.skip("activation-BaB full-region capture is unavailable")
    if not torch.cuda.is_available():
        pytest.skip("activation-BaB input-domain TIR requires CUDA")
    value = torch.load(CAPTURE, map_location="cpu", weights_only=True)
    if value.get("schema_version") != "boundflow.activation-bab-full-region-tensors/v1":
        raise ValueError("activation-BaB fixture schema differs")
    return cast(dict[str, Any], value)


def _coordinates(values: list[torch.Tensor] | tuple[torch.Tensor, ...]):
    return cast(
        tuple[tuple[int, int, int], ...],
        tuple(
            tuple(int(values[axis][ordinal]) for axis in range(3))
            for ordinal in range(int(values[0].numel()))
        ),
    )


def _capability() -> str:
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}"


def _input_template(payload: dict[str, Any]) -> BabInputDomainTemplateV1:
    evaluation = payload["segments"]["input_domain"]["evaluations"][0]
    return BabInputDomainTemplateV1(
        spec_count=1,
        domain_count=6,
        output_channels=8,
        output_height=16,
        output_width=16,
        input_channels=3,
        input_height=32,
        input_width=32,
        alpha_coordinates=_coordinates(evaluation["alpha_feature_indices"]),
        compute_capability=_capability(),
        forward_symbol=BAB_INPUT_DOMAIN_FORWARD_SYMBOL,
        backward_symbol=BAB_INPUT_DOMAIN_BACKWARD_SYMBOL,
    )


def _cuda(value: torch.Tensor, *, requires_grad: bool = False) -> torch.Tensor:
    result = value.detach().cuda().contiguous().clone()
    result.requires_grad_(requires_grad)
    return result


def _input_tensors(evaluation: dict[str, Any]) -> BabInputDomainTensorsV1:
    return BabInputDomainTensorsV1(
        incoming_lower_a=_cuda(evaluation["incoming_lower_a"], requires_grad=True),
        preactivation_lower=_cuda(evaluation["preactivation_lower"]),
        preactivation_upper=_cuda(evaluation["preactivation_upper"]),
        raw_alpha=_cuda(evaluation["raw_alpha"], requires_grad=True),
        operator_weight=_cuda(evaluation["operator_weight"]),
        operator_bias=_cuda(evaluation["operator_bias"]),
        input_center=_cuda(evaluation["input_center"]),
        input_radius=(
            (_cuda(evaluation["input_upper"]) - _cuda(evaluation["input_lower"])) / 2
        ).contiguous(),
    )


def _difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().cpu() - right).abs().max().item())


def _sign_exact(left: torch.Tensor, right: torch.Tensor) -> bool:
    return bool(torch.equal(torch.sign(left.detach().cpu()), torch.sign(right)))


def _all_executors(payload: dict[str, Any]):
    segments = payload["segments"]
    terminal = segments["terminal"]["evaluations"][0]
    beta = segments["terminal"]["beta_evidence"][0]
    residual = segments["residual"]["evaluations"][0]
    projection = segments["projection"]["evaluations"][0]
    capability = _capability()
    terminal_template = BabTerminalLinearTemplateV1(
        spec_count=1,
        domain_count=6,
        current_features=100,
        previous_features=1024,
        alpha_feature_indices=tuple(
            int(value) for value in terminal["alpha_feature_indices"][0].tolist()
        ),
        beta_count=int(beta["value"].shape[1]),
        compute_capability=capability,
    )
    residual_template = RootCrownResidualTemplateV1(
        spec_count=1,
        domain_count=6,
        channels=16,
        height=8,
        width=8,
        entry_alpha_coordinates=_coordinates(residual["entry_alpha_feature_indices"]),
        inner_alpha_coordinates=_coordinates(residual["inner_alpha_feature_indices"]),
        compute_capability=capability,
    )
    projection_template = RootCrownProjectionTemplateV1(
        spec_count=1,
        domain_count=6,
        output_channels=16,
        output_height=8,
        output_width=8,
        input_channels=8,
        input_height=16,
        input_width=16,
        entry_alpha_coordinates=_coordinates(projection["entry_alpha_feature_indices"]),
        inner_alpha_coordinates=_coordinates(projection["inner_alpha_feature_indices"]),
        compute_capability=capability,
    )
    return (
        BabTerminalTIRExecutorV1(terminal_template),
        RootCrownResidualTIRExecutorV1(residual_template),
        RootCrownProjectionTIRExecutorV1(projection_template),
        BabInputDomainTIRExecutorV1(_input_template(payload)),
    )


def test_bab_input_streaming_matches_ten_real_evaluations_and_nine_vjps() -> None:
    payload = _payload()
    segment = payload["segments"]["input_domain"]
    template = _input_template(payload)
    executor = BabInputDomainTIRExecutorV1(template)
    maximum = 0.0
    for ordinal, evaluation in enumerate(segment["evaluations"]):
        tensors = _input_tensors(evaluation)
        validate_bab_input_domain_tensors_v1(tensors, template)
        concrete, bias = execute_bab_input_domain_tir_v1(tensors, executor)
        for observed, expected in (
            (concrete, evaluation["concrete_lower"]),
            (bias, evaluation["output_bias"]),
        ):
            maximum = max(maximum, _difference(observed, expected))
            assert _sign_exact(observed, expected)
        if ordinal < 9:
            incoming_gradient, alpha_gradient = torch.autograd.grad(
                (concrete, bias),
                (tensors.incoming_lower_a, tensors.raw_alpha),
                grad_outputs=(
                    _cuda(evaluation["concrete_lower_gradient"]),
                    _cuda(evaluation["output_bias_gradient"]),
                ),
            )
            for observed, expected in (
                (incoming_gradient, evaluation["incoming_lower_a_gradient"]),
                (alpha_gradient, evaluation["raw_alpha_gradient"]),
            ):
                maximum = max(maximum, _difference(observed, expected))
                assert _sign_exact(observed, expected)
    assert maximum <= 2e-6
    assert executor.forward_launch_count == 10
    assert executor.backward_launch_count == 9
    assert executor.fallback_count == 0
    assert executor.pointer_count == executor.pointer_exact_count == 227


def test_bab_input_schedule_has_no_dense_coefficient_workspace() -> None:
    payload = _payload()
    template = _input_template(payload)
    _unscheduled, scheduled, inventory = build_bab_input_domain_modules_v1(template)
    script = scheduled.script(show_meta=False)
    assert "input_coefficient_scratch" not in script
    assert "T.alloc_buffer((1, 6, 3, 32, 32)" not in script
    assert inventory == (
        ("adjoint", (1,)),
        ("bias_sum", (1,)),
        ("coefficient", (1,)),
        ("concrete_sum", (1,)),
        ("partial", (2, 128)),
        ("reduction", (2,)),
    )


def test_four_compiled_segments_are_consumed_by_single_full_owner() -> None:
    payload = _payload()
    segments = cast(dict[str, Any], payload["segments"])
    dynamic, static = bab_full_region_inputs_from_capture_v1(segments, 0, device="cuda")
    reference = PreparedBabFullRegionOwnerV1(static).evaluate(dynamic)
    terminal, residual, projection, input_domain = _all_executors(payload)
    owner = PreparedBabFullRegionOwnerV1(
        static,
        terminal_executor=terminal,
        residual_executor=residual,
        projection_executor=projection,
        input_executor=input_domain,
    )
    candidate = owner.evaluate(dynamic)
    assert torch.allclose(candidate, reference, atol=2e-5, rtol=2e-5)
    assert torch.equal(torch.sign(candidate), torch.sign(reference))
    candidate_gradients = torch.autograd.grad(-candidate.sum(), dynamic.tensors())
    reference_gradients = torch.autograd.grad(-reference.sum(), dynamic.tensors())
    for candidate_gradient, reference_gradient in zip(
        candidate_gradients, reference_gradients
    ):
        assert torch.allclose(
            candidate_gradient, reference_gradient, atol=3e-5, rtol=3e-5
        )
        assert torch.equal(
            torch.sign(candidate_gradient), torch.sign(reference_gradient)
        )
    receipt = owner.receipt()
    assert receipt.forward_count == receipt.backward_count == 1
    assert receipt.compiled_segment_count == 4
    assert receipt.compiled_forward_launch_count == 8
    assert receipt.compiled_backward_launch_count == 4
    assert receipt.saved_dense_coefficient_count == 0


def test_four_compiled_segments_replay_nine_native_adam_mutations() -> None:
    payload = _payload()
    segments = cast(dict[str, Any], payload["segments"])
    dynamic, static = bab_full_region_inputs_from_capture_v1(segments, 0, device="cuda")
    terminal, residual, projection, input_domain = _all_executors(payload)
    owner = PreparedBabFullRegionOwnerV1(
        static,
        terminal_executor=terminal,
        residual_executor=residual,
        projection_executor=projection,
        input_executor=input_domain,
    )
    parameters = list(dynamic.tensors()[1:])
    optimizer = torch.optim.Adam(
        (
            {"params": parameters[:6], "lr": 0.01},
            {"params": [parameters[6]], "lr": 0.05},
        ),
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    alpha_locations = (
        ("terminal", "raw_alpha"),
        ("residual", "entry_raw_alpha"),
        ("residual", "inner_raw_alpha"),
        ("projection", "entry_raw_alpha"),
        ("projection", "inner_raw_alpha"),
        ("input_domain", "raw_alpha"),
    )
    maximum = 0.0
    for ordinal in range(9):
        lower = owner.evaluate(
            BabFullRegionDynamicV1(dynamic.terminal_incoming, *parameters)
        )
        gradients = torch.autograd.grad(-lower.sum(), parameters)
        optimizer.zero_grad(set_to_none=True)
        for parameter, gradient in zip(parameters, gradients):
            parameter.grad = gradient
        optimizer.step()
        with torch.no_grad():
            for parameter in parameters[:6]:
                parameter.clamp_(0.0, 1.0)
            parameters[6].clamp_(min=0.0)
        scheduler.step()
        expected = tuple(
            segments[segment]["evaluations"][ordinal + 1][name].cuda()
            for segment, name in alpha_locations
        ) + (segments["terminal"]["beta_evidence"][ordinal + 1]["value"].cuda(),)
        for observed, reference in zip(parameters, expected):
            maximum = max(
                maximum,
                float((observed.detach() - reference).abs().max().item()),
            )
            assert torch.equal(torch.sign(observed), torch.sign(reference))
    assert maximum <= 4e-6
    receipt = owner.receipt()
    assert receipt.forward_count == receipt.backward_count == 9
    assert receipt.compiled_forward_launch_count == 72
    assert receipt.compiled_backward_launch_count == 36


@pytest.mark.parametrize(
    ("field", "mutator", "match"),
    (
        (
            "raw_alpha",
            lambda value: torch.full_like(value, float("nan")),
            "nonfinite tensor",
        ),
        (
            "preactivation_lower",
            lambda value: torch.full_like(value, 2.0),
            "legality differs",
        ),
        (
            "input_radius",
            lambda value: torch.full_like(value, -1.0),
            "legality differs",
        ),
    ),
)
def test_bab_input_admission_rejects_invalid_values(
    field: str, mutator: Any, match: str
) -> None:
    payload = _payload()
    evaluation = payload["segments"]["input_domain"]["evaluations"][0]
    tensors = _input_tensors(evaluation)
    object.__setattr__(tensors, field, mutator(getattr(tensors, field)))
    with pytest.raises(ValueError, match=match):
        validate_bab_input_domain_tensors_v1(tensors, _input_template(payload))
