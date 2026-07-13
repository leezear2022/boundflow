"""Correctness and stream contracts for the PR-12G chunked eager candidate."""

import pytest
import torch

from boundflow.ir.task import TaskOp
from boundflow.runtime.fused_crown import (
    FusedReluAffineRequest,
    TorchChunkedFusedCrownExecutor,
    TorchDenseFusedCrownReference,
    build_fused_crown_runtime_selection,
    validate_fused_crown_execution_steps,
)


def _request(kind: str, device: torch.device) -> FusedReluAffineRequest:
    torch.manual_seed(1270 + int(kind == "conv2d"))
    domain, spec = 2, 5
    output_shape: tuple[int, ...]
    input_shape: tuple[int, ...]
    if kind == "linear":
        output_shape = (7,)
        input_shape = (4,)
        weight = torch.randn(7, 4, device=device)
        bias = torch.randn(7, device=device)
        attrs = {}
    else:
        output_shape = (3, 4, 4)
        input_shape = (2, 4, 4)
        weight = torch.randn(3, 2, 3, 3, device=device)
        bias = torch.randn(3, device=device)
        attrs = {
            "stride": (1, 1),
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


@pytest.mark.parametrize("kind", ["linear", "conv2d"])
def test_chunked_executor_matches_dense_across_domain_boundaries(kind: str) -> None:
    request = _request(kind, torch.device("cpu"))

    expected = TorchDenseFusedCrownReference().run(request)
    actual = TorchChunkedFusedCrownExecutor(chunk_rows=3).run(request)

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
        torch.testing.assert_close(actual_tensor, expected_tensor)


def test_chunked_executor_rejects_invalid_chunk_size() -> None:
    with pytest.raises(ValueError, match="chunk_rows"):
        TorchChunkedFusedCrownExecutor(chunk_rows=0)


def test_runtime_selection_preserves_planner_backend_in_step_contract() -> None:
    ops = [
        TaskOp("linear", "linear", ["input", "w", "b"], ["h"]),
        TaskOp("relu", "relu", ["h"], ["r"]),
    ]

    selection = build_fused_crown_runtime_selection(
        ops, backend="pytorch_chunked", chunk_rows=17
    )

    assert isinstance(selection.executor, TorchChunkedFusedCrownExecutor)
    assert selection.executor.chunk_rows == 17
    assert selection.steps[0].backend == "pytorch_chunked"
    assert validate_fused_crown_execution_steps(ops, selection.steps) == selection.steps

    eager = build_fused_crown_runtime_selection(ops, backend="pytorch_eager")
    assert eager.executor is None and eager.steps == ()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_chunked_executor_honors_custom_stream_without_global_sync() -> None:
    device = torch.device("cuda")
    request = _request("linear", device)
    expected = TorchDenseFusedCrownReference().run(request)
    executor = TorchChunkedFusedCrownExecutor(chunk_rows=3)
    default = torch.cuda.default_stream(device)
    custom = torch.cuda.Stream(device=device)

    with torch.cuda.stream(default):
        torch.cuda._sleep(200_000_000)
    with torch.cuda.stream(custom):
        actual = executor.run(request, stream=custom)
        snapshot = actual.A_prev_u.clone()
    custom.synchronize()

    torch.testing.assert_close(snapshot, expected.A_prev_u)
    default.synchronize()
