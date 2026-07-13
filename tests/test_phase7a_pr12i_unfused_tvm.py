"""Correctness and workspace contracts for the PR-12I TVM-unfused baseline."""

import pytest
import torch

from boundflow.backends.tvm.fused_crown_conv2d import FusedCrownConv2dSignature
from boundflow.backends.tvm.fused_crown_linear import FusedCrownLinearKey
from boundflow.backends.tvm.unfused_crown import (
    build_unfused_crown_conv2d_primfunc,
    build_unfused_crown_linear_primfunc,
    explicit_workspace_bytes,
)
from boundflow.runtime.fused_crown import (
    FusedCrownExecutionContext,
    FusedReluAffineRequest,
    TVMUnfusedCrownExecutor,
    TorchDenseFusedCrownReference,
)


def _request(kind: str, device: torch.device) -> FusedReluAffineRequest:
    torch.manual_seed(1209 + int(kind == "conv2d"))
    domain, spec = 2, 3
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    if kind == "linear":
        input_shape = (4,)
        output_shape = (7,)
        weight = torch.randn(7, 4, device=device)
        bias = torch.randn(7, device=device)
        attrs = {}
    else:
        input_shape = (2, 7, 7)
        output_shape = (3, 4, 4)
        weight = torch.randn(3, 2, 3, 3, device=device)
        bias = torch.randn(3, device=device)
        attrs = {
            "stride": (2, 2),
            "padding": (1, 1),
            "dilation": (1, 1),
            "groups": 1,
            "output_padding": (0, 0),
        }
    features = int(torch.tensor(output_shape).prod())
    return FusedReluAffineRequest(
        kind=kind,  # type: ignore[arg-type]
        A_u=torch.randn(domain, spec, features, device=device),
        A_l=torch.randn(domain, spec, features, device=device),
        alpha_u=torch.rand(domain, features, device=device),
        alpha_l=torch.rand(domain, features, device=device),
        beta_u=torch.randn(domain, features, device=device),
        beta_l=torch.randn(domain, features, device=device),
        weight=weight,
        bias=bias,
        input_shape=input_shape,
        output_shape=output_shape,
        attrs=attrs,
    )


def test_unfused_primfuncs_expose_scaled_coefficient_outputs() -> None:
    linear = build_unfused_crown_linear_primfunc(FusedCrownLinearKey(1, 2, 4, 3))
    conv = build_unfused_crown_conv2d_primfunc(
        FusedCrownConv2dSignature(
            1,
            2,
            2,
            4,
            4,
            3,
            4,
            4,
            3,
            3,
            (1, 1),
            (1, 1),
        )
    )

    for primfunc in (linear, conv):
        names = {buffer.name for buffer in primfunc.buffer_map.values()}
        assert {"scaled_u", "scaled_l"} <= names
    assert explicit_workspace_bytes((2, 3, 7)) == 2 * 2 * 3 * 7 * 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("kind", ["linear", "conv2d"])
def test_tvm_unfused_matches_dense_region_reference(kind: str) -> None:
    request = _request(kind, torch.device("cuda"))
    expected = TorchDenseFusedCrownReference().run(request)
    executor = TVMUnfusedCrownExecutor()

    assert executor.supports(request, FusedCrownExecutionContext())
    actual = executor.run(request)

    for actual_tensor, expected_tensor in zip(
        (
            actual.A_prev_u,
            actual.A_prev_l,
            actual.bias_delta_u,
            actual.bias_delta_l,
        ),
        (
            expected.A_prev_u,
            expected.A_prev_l,
            expected.bias_delta_u,
            expected.bias_delta_l,
        ),
    ):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=2e-4, atol=2e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_tvm_unfused_honors_custom_stream_without_global_sync() -> None:
    request = _request("linear", torch.device("cuda"))
    expected = TorchDenseFusedCrownReference().run(request)
    stream = torch.cuda.Stream()
    # Inputs were produced on the current/default stream. The caller must make
    # this producer dependency explicit before an external runtime consumes
    # them on a different stream; the executor is responsible for launching on
    # that stream, not for guessing input provenance.
    stream.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(stream):
        actual = TVMUnfusedCrownExecutor().run(request, stream=stream)
        snapshot = actual.A_prev_u.clone()
    stream.synchronize()

    torch.testing.assert_close(snapshot, expected.A_prev_u, rtol=2e-4, atol=2e-4)
