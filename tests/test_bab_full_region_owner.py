"""Activation-BaB full-owner correctness and 10/9 trajectory tests."""

# pylint: disable=too-many-locals,missing-function-docstring
# pylint: disable=no-value-for-parameter,duplicate-code

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import torch

from boundflow.runtime.bab_full_region_owner import (
    bab_full_region_inputs_from_capture_v1,
    BabFullRegionDynamicV1,
    BabFullRegionOwnerReceiptV1,
    evaluate_bab_full_region_trace_v1,
    PreparedBabFullRegionOwnerV1,
)

CAPTURE = Path("artifacts/bab-full-region-capture/resnet2b-prop0-v1/capture.pt")
ALPHA_LOCATIONS = (
    ("terminal", "raw_alpha", "raw_alpha_gradient"),
    ("residual", "entry_raw_alpha", "entry_raw_alpha_gradient"),
    ("residual", "inner_raw_alpha", "inner_raw_alpha_gradient"),
    ("projection", "entry_raw_alpha", "entry_raw_alpha_gradient"),
    ("projection", "inner_raw_alpha", "inner_raw_alpha_gradient"),
    ("input_domain", "raw_alpha", "raw_alpha_gradient"),
)


def _payload() -> dict[str, Any]:
    if not CAPTURE.is_file():
        pytest.skip("activation-BaB full-region capture is unavailable")
    if not torch.cuda.is_available():
        pytest.skip("activation-BaB full-region correctness requires CUDA")
    result = torch.load(CAPTURE, map_location="cpu", weights_only=True)
    if (
        result.get("schema_version")
        != "boundflow.activation-bab-full-region-tensors/v1"
    ):
        raise ValueError("activation-BaB fixture schema differs")
    return cast(dict[str, Any], result)


def _difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().cpu() - right).abs().max().item())


def _sign_exact(left: torch.Tensor, right: torch.Tensor) -> bool:
    return bool(torch.equal(torch.sign(left.detach().cpu()), torch.sign(right)))


def _captured_alpha_gradients(
    segments: dict[str, Any], ordinal: int
) -> tuple[torch.Tensor, ...]:
    return tuple(
        cast(
            torch.Tensor,
            segments[segment]["evaluations"][ordinal][gradient_name],
        )
        for segment, _value_name, gradient_name in ALPHA_LOCATIONS
    )


def test_bab_full_region_receipt_rejects_performance_or_dense_save() -> None:
    valid = BabFullRegionOwnerReceiptV1(forward_count=10, backward_count=9)
    valid.validate()
    with pytest.raises(ValueError, match="owner receipt differs"):
        BabFullRegionOwnerReceiptV1(
            forward_count=10,
            backward_count=9,
            saved_dense_coefficient_count=1,
        ).validate()
    with pytest.raises(ValueError, match="owner receipt differs"):
        BabFullRegionOwnerReceiptV1(
            forward_count=10,
            backward_count=9,
            performance_claimed=True,
        ).validate()


def test_bab_full_region_trace_and_custom_backward_match_real_capture() -> None:
    payload = _payload()
    segments = cast(dict[str, Any], payload["segments"])
    maximum = 0.0
    owner = None
    for ordinal in range(10):
        dynamic, static = bab_full_region_inputs_from_capture_v1(
            segments, ordinal, device="cuda"
        )
        trace = evaluate_bab_full_region_trace_v1(dynamic, static)
        terminal = segments["terminal"]["evaluations"][ordinal]
        residual = segments["residual"]["evaluations"][ordinal]
        projection = segments["projection"]["evaluations"][ordinal]
        input_domain = segments["input_domain"]["evaluations"][ordinal]
        references = (
            (trace.terminal_a, terminal["output_lower_a"]),
            (
                trace.terminal_bias,
                terminal["relu_lower_bias"] + terminal["linear_lower_bias"],
            ),
            (trace.residual_a, residual["output_lower_a"]),
            (trace.residual_bias, residual["output_bias"]),
            (trace.projection_a, projection["output_lower_a"]),
            (trace.projection_bias, projection["output_bias"]),
            (trace.concrete, input_domain["concrete_lower"]),
            (trace.input_bias, input_domain["output_bias"]),
        )
        for observed, expected in references:
            maximum = max(maximum, _difference(observed, expected))
            assert _sign_exact(observed, expected)
        assert maximum <= 2e-5

        owner = PreparedBabFullRegionOwnerV1(static)
        lower = owner.evaluate(dynamic)
        expected_lower = input_domain["concrete_lower"] + (
            terminal["relu_lower_bias"]
            + terminal["linear_lower_bias"]
            + residual["output_bias"]
            + projection["output_bias"]
            + input_domain["output_bias"]
        ).transpose(0, 1)
        assert _difference(lower, expected_lower) <= 2e-5
        assert _sign_exact(lower, expected_lower)
        if ordinal < 9:
            observed_gradients = torch.autograd.grad(-lower.sum(), dynamic.tensors())
            references_gradient = (
                None,
                *_captured_alpha_gradients(segments, ordinal),
                segments["terminal"]["beta_evidence"][ordinal]["gradient"],
            )
            for observed, expected in zip(observed_gradients, references_gradient):
                if expected is None:
                    assert torch.isfinite(observed).all()
                    continue
                assert _difference(observed, expected) <= 3e-5
                assert _sign_exact(observed, expected)
    assert owner is not None
    receipt = owner.receipt()
    assert receipt.forward_count == 1
    assert receipt.backward_count == 0
    assert receipt.saved_dense_coefficient_count == 0
    assert receipt.frozen_bound_gradient_count == 0


def test_bab_full_region_replays_nine_native_adam_mutations() -> None:
    payload = _payload()
    segments = cast(dict[str, Any], payload["segments"])
    dynamic, static = bab_full_region_inputs_from_capture_v1(segments, 0, device="cuda")
    owner = PreparedBabFullRegionOwnerV1(static)
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
        expected_alpha = tuple(
            segments[segment]["evaluations"][ordinal + 1][value_name].cuda()
            for segment, value_name, _gradient_name in ALPHA_LOCATIONS
        )
        expected_beta = segments["terminal"]["beta_evidence"][ordinal + 1][
            "value"
        ].cuda()
        for observed, expected in zip(parameters, (*expected_alpha, expected_beta)):
            # PyTorch ConvTranspose reduction order differs slightly between the
            # isolated recompute owner and the host graph.  The per-step VJP
            # signs are exact; accumulated 9-step state drift remains below
            # 2e-6, one order tighter than the existing production gate.
            assert torch.allclose(observed, expected, atol=2e-6, rtol=0.0)
            assert torch.equal(torch.sign(observed), torch.sign(expected))
    receipt = owner.receipt()
    assert receipt.forward_count == 9
    assert receipt.backward_count == 9
    assert receipt.mutable_owner_count == 7
    assert receipt.performance_claimed is False


def test_bab_full_region_rejects_beta_location_tamper() -> None:
    payload = _payload()
    segments = cast(dict[str, Any], payload["segments"])
    dynamic, static = bab_full_region_inputs_from_capture_v1(segments, 0, device="cuda")
    object.__setattr__(
        static,
        "beta_locations",
        torch.full_like(static.beta_locations, dynamic.terminal_incoming.shape[-1]),
    )
    with pytest.raises(ValueError, match="sparse beta legality differs"):
        PreparedBabFullRegionOwnerV1(static).evaluate(dynamic)
