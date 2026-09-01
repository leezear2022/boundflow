"""Beta-aware activation-BaB terminal TVM/TIR correctness gates."""

# pylint: disable=missing-function-docstring,too-many-locals,duplicate-code
# pylint: disable=unnecessary-lambda

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import torch

from boundflow.backends.tvm.bab_terminal_linear import (
    BabTerminalLinearTemplateV1,
    build_bab_terminal_tir_modules_v1,
)
from boundflow.backends.tvm.root_crown_projection import (
    RootCrownProjectionTemplateV1,
)
from boundflow.backends.tvm.root_crown_residual import RootCrownResidualTemplateV1
from boundflow.runtime.bab_full_region_owner import (
    bab_full_region_inputs_from_capture_v1,
    PreparedBabFullRegionOwnerV1,
)
from boundflow.runtime.bab_terminal_tir import (
    BabTerminalTensorsV1,
    BabTerminalTIRExecutorV1,
    execute_bab_terminal_tir_v1,
    validate_bab_terminal_tensors_v1,
)
from boundflow.runtime.root_crown_projection_tir import (
    RootCrownProjectionTIRExecutorV1,
)
from boundflow.runtime.root_crown_residual_tir import RootCrownResidualTIRExecutorV1

CAPTURE = Path("artifacts/bab-full-region-capture/resnet2b-prop0-v1/capture.pt")


def _payload() -> dict[str, Any]:
    if not CAPTURE.is_file():
        pytest.skip("activation-BaB full-region capture is unavailable")
    if not torch.cuda.is_available():
        pytest.skip("activation-BaB terminal TIR requires CUDA")
    value = torch.load(CAPTURE, map_location="cpu", weights_only=True)
    if value.get("schema_version") != "boundflow.activation-bab-full-region-tensors/v1":
        raise ValueError("activation-BaB fixture schema differs")
    return cast(dict[str, Any], value)


def _template(payload: dict[str, Any]) -> BabTerminalLinearTemplateV1:
    evaluation = payload["segments"]["terminal"]["evaluations"][0]
    beta = payload["segments"]["terminal"]["beta_evidence"][0]
    major, minor = torch.cuda.get_device_capability()
    incoming = evaluation["incoming_lower_a"]
    weight = evaluation["operator_weight"]
    return BabTerminalLinearTemplateV1(
        spec_count=int(incoming.shape[0]),
        domain_count=int(incoming.shape[1]),
        current_features=int(incoming.shape[2]),
        previous_features=int(weight.shape[1]),
        alpha_feature_indices=tuple(
            int(value) for value in evaluation["alpha_feature_indices"][0].tolist()
        ),
        beta_count=int(beta["value"].shape[1]),
        compute_capability=f"sm_{major}{minor}",
    )


def _cuda(value: torch.Tensor, *, requires_grad: bool = False) -> torch.Tensor:
    result = value.detach().cuda().contiguous().clone()
    result.requires_grad_(requires_grad)
    return result


def _tensors(evaluation: dict[str, Any], beta: dict[str, Any]) -> BabTerminalTensorsV1:
    return BabTerminalTensorsV1(
        incoming_lower_a=_cuda(evaluation["incoming_lower_a"], requires_grad=True),
        preactivation_lower=_cuda(evaluation["preactivation_lower"]),
        preactivation_upper=_cuda(evaluation["preactivation_upper"]),
        compressed_alpha=_cuda(evaluation["raw_alpha"], requires_grad=True),
        sparse_beta=_cuda(beta["value"], requires_grad=True),
        beta_location=_cuda(beta["location"]).to(torch.int64),
        beta_sign=_cuda(beta["sign"]),
        linear_weight=_cuda(evaluation["operator_weight"]),
        linear_bias=_cuda(evaluation["operator_bias"]),
    )


def _difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().cpu() - right).abs().max().item())


def _sign_exact(left: torch.Tensor, right: torch.Tensor) -> bool:
    return bool(torch.equal(torch.sign(left.detach().cpu()), torch.sign(right)))


def _coordinates(values: list[torch.Tensor]) -> tuple[tuple[int, int, int], ...]:
    return cast(
        tuple[tuple[int, int, int], ...],
        tuple(
            tuple(int(values[axis][ordinal]) for axis in range(3))
            for ordinal in range(int(values[0].numel()))
        ),
    )


def _downstream_executors(
    payload: dict[str, Any],
) -> tuple[RootCrownResidualTIRExecutorV1, RootCrownProjectionTIRExecutorV1]:
    residual = payload["segments"]["residual"]["evaluations"][0]
    projection = payload["segments"]["projection"]["evaluations"][0]
    major, minor = torch.cuda.get_device_capability()
    capability = f"sm_{major}{minor}"
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
        RootCrownResidualTIRExecutorV1(residual_template),
        RootCrownProjectionTIRExecutorV1(projection_template),
    )


def test_bab_terminal_tir_matches_ten_real_evaluations_and_nine_vjps() -> None:
    payload = _payload()
    terminal = payload["segments"]["terminal"]
    template = _template(payload)
    executor = BabTerminalTIRExecutorV1(template)
    maximum = 0.0
    for ordinal, (evaluation, beta) in enumerate(
        zip(terminal["evaluations"], terminal["beta_evidence"])
    ):
        tensors = _tensors(evaluation, beta)
        validate_bab_terminal_tensors_v1(tensors, template)
        output_a, output_bias = execute_bab_terminal_tir_v1(tensors, executor)
        references = (
            (output_a, evaluation["output_lower_a"]),
            (
                output_bias,
                evaluation["relu_lower_bias"] + evaluation["linear_lower_bias"],
            ),
        )
        for observed, expected in references:
            maximum = max(maximum, _difference(observed, expected))
            assert _sign_exact(observed, expected)
        if ordinal < 9:
            gradients = torch.autograd.grad(
                (output_a, output_bias),
                (
                    tensors.incoming_lower_a,
                    tensors.compressed_alpha,
                    tensors.sparse_beta,
                ),
                grad_outputs=(
                    _cuda(evaluation["output_lower_a_gradient"]),
                    _cuda(evaluation["output_bias_gradient"]),
                ),
            )
            for observed, expected in (
                (gradients[1], evaluation["raw_alpha_gradient"]),
                (gradients[2], beta["gradient"]),
            ):
                maximum = max(maximum, _difference(observed, expected))
                assert _sign_exact(observed, expected)
            assert torch.isfinite(gradients[0]).all()
    assert maximum <= 2e-6
    assert executor.forward_launch_count == 10
    assert executor.backward_launch_count == 9
    assert executor.fallback_count == 0
    assert executor.pointer_count == executor.pointer_exact_count == 264
    assert executor.compiled.workspace_inventory == (
        ("terminal_linear_adjoint", (1, 6, 100)),
    )


def test_three_compiled_segments_are_consumed_by_single_full_owner() -> None:
    payload = _payload()
    segments = cast(dict[str, Any], payload["segments"])
    dynamic, static = bab_full_region_inputs_from_capture_v1(segments, 0, device="cuda")
    reference = PreparedBabFullRegionOwnerV1(static).evaluate(dynamic)
    terminal_executor = BabTerminalTIRExecutorV1(_template(payload))
    residual_executor, projection_executor = _downstream_executors(payload)
    owner = PreparedBabFullRegionOwnerV1(
        static,
        terminal_executor=terminal_executor,
        residual_executor=residual_executor,
        projection_executor=projection_executor,
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
    assert receipt.terminal_backend == "tvm-beta-terminal-v1"
    assert receipt.forward_count == receipt.backward_count == 1
    assert receipt.terminal_forward_launch_count == 2
    assert receipt.terminal_backward_launch_count == 1
    assert receipt.compiled_segment_count == 3
    assert receipt.compiled_forward_launch_count == 6
    assert receipt.compiled_backward_launch_count == 3
    assert receipt.saved_dense_coefficient_count == 0


def test_bab_terminal_tir_has_no_dense_coefficient_workspace() -> None:
    payload = _payload()
    template = _template(payload)
    _unscheduled, scheduled = build_bab_terminal_tir_modules_v1(template)
    script = scheduled.script(show_meta=False)
    for forbidden in (
        "relu_output_lower_a",
        "post_beta_lower_a",
        "dense_alpha",
        "adjoint_matmul",
    ):
        assert forbidden not in script
    assert "terminal_linear_adjoint" in script


@pytest.mark.parametrize(
    ("field", "mutator", "match"),
    (
        (
            "beta_location",
            lambda value: torch.full_like(value, 100),
            "sparse-state legality differs",
        ),
        (
            "beta_sign",
            lambda value: torch.zeros_like(value),
            "sparse-state legality differs",
        ),
        (
            "sparse_beta",
            lambda value: torch.full_like(value, float("nan")),
            "nonfinite tensor",
        ),
    ),
)
def test_bab_terminal_tir_rejects_sparse_state_tamper(
    field: str, mutator: Any, match: str
) -> None:
    payload = _payload()
    evaluation = payload["segments"]["terminal"]["evaluations"][0]
    beta = payload["segments"]["terminal"]["beta_evidence"][0]
    tensors = _tensors(evaluation, beta)
    object.__setattr__(tensors, field, mutator(getattr(tensors, field)))
    with pytest.raises(ValueError, match=match):
        validate_bab_terminal_tensors_v1(tensors, _template(payload))
