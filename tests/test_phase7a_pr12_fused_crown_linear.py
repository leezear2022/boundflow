"""CUDA correctness and mechanism tests for the PR-12 fused Linear task."""

import pytest
import torch

from boundflow.backends.tvm.fused_crown_linear import (
    FusedCrownLinearKey,
    allocated_intermediate_buffers,
    build_fused_crown_linear_module,
    build_fused_crown_linear_relax_ir_module,
)
from boundflow.backends.tvm.relax_analysis import collect_relax_ir_stats


def _reference(
    coeff_u: torch.Tensor,
    coeff_l: torch.Tensor,
    alpha_u: torch.Tensor,
    beta_u: torch.Tensor,
    alpha_l: torch.Tensor,
    beta_l: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    scaled_u = torch.where(
        coeff_u >= 0, coeff_u * alpha_u[:, None], coeff_u * alpha_l[:, None]
    )
    scaled_l = torch.where(
        coeff_l >= 0, coeff_l * alpha_l[:, None], coeff_l * alpha_u[:, None]
    )
    relu_bias_u = torch.where(
        coeff_u >= 0, coeff_u * beta_u[:, None], coeff_u * beta_l[:, None]
    ).sum(dim=2)
    relu_bias_l = torch.where(
        coeff_l >= 0, coeff_l * beta_l[:, None], coeff_l * beta_u[:, None]
    ).sum(dim=2)
    return (
        scaled_u @ weight,
        scaled_l @ weight,
        relu_bias_u + (scaled_u * bias).sum(dim=2),
        relu_bias_l + (scaled_l * bias).sum(dim=2),
    )


def test_fused_linear_primfunc_has_no_scaled_a_allocation() -> None:
    key = FusedCrownLinearKey(2, 8, 16, 12)

    assert allocated_intermediate_buffers(key) == ()


def test_fused_linear_has_one_thin_relax_call_tir_wrapper() -> None:
    module = build_fused_crown_linear_relax_ir_module(FusedCrownLinearKey(2, 8, 16, 12))

    stats = collect_relax_ir_stats(module)
    assert stats["relax_funcs"] == 1
    assert stats["tir_funcs"] == 1
    assert stats["call_tir"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_linear_cuda_matches_dense_reference() -> None:
    import tvm

    torch.manual_seed(1201)
    key = FusedCrownLinearKey(2, 8, 16, 12)
    tensors = [
        torch.randn(2, 8, 16, dtype=torch.float32),
        torch.randn(2, 8, 16, dtype=torch.float32),
        torch.rand(2, 16, dtype=torch.float32),
        torch.randn(2, 16, dtype=torch.float32),
        torch.rand(2, 16, dtype=torch.float32),
        torch.randn(2, 16, dtype=torch.float32),
        torch.randn(16, 12, dtype=torch.float32),
        torch.randn(16, dtype=torch.float32),
    ]
    tensors[0][0, 0, :3] = torch.tensor([0.0, 1.0, -1.0])
    tensors[1][0, 0, :3] = torch.tensor([0.0, -1.0, 1.0])
    expected = _reference(*tensors)
    device = tvm.cuda(0)
    inputs = [tvm.runtime.tensor(tensor.numpy(), device=device) for tensor in tensors]
    outputs = [
        tvm.runtime.empty((2, 8, 12), "float32", device),
        tvm.runtime.empty((2, 8, 12), "float32", device),
        tvm.runtime.empty((2, 8), "float32", device),
        tvm.runtime.empty((2, 8), "float32", device),
    ]

    compiled = build_fused_crown_linear_module(key)
    assert compiled is build_fused_crown_linear_module(key)
    compiled(*inputs, *outputs)
    device.sync()

    for actual, reference in zip(outputs, expected):
        assert torch.isfinite(torch.from_numpy(actual.numpy())).all()
        torch.testing.assert_close(
            torch.from_numpy(actual.numpy()), reference, rtol=2e-5, atol=2e-5
        )


@pytest.mark.parametrize(
    "updates",
    [
        {"dtype": "float16"},
        {"target": "llvm"},
        {"compute_capability": "invalid"},
        {"schedule_id": "unknown"},
    ],
)
def test_fused_linear_rejects_unsupported_compile_keys(
    updates: dict[str, object],
) -> None:
    values = {
        "domain_batch": 1,
        "spec_batch": 1,
        "current_features": 4,
        "previous_features": 3,
        **updates,
    }
    with pytest.raises((NotImplementedError, ValueError)):
        FusedCrownLinearKey(**values).validate()  # type: ignore[arg-type]
