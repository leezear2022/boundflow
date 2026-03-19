import pytest
import torch

from boundflow.runtime.linear_operator import DenseLinearOperator
from boundflow.runtime.perturbation import LpBallPerturbation


@pytest.mark.parametrize("p", ["inf", 2, 1])
def test_highdim_concretize_matches_manual_flatten(p: str | int) -> None:
    torch.manual_seed(0)
    batch = 2
    specs = 3
    center = torch.randn(batch, 2, 3, 4, dtype=torch.float32)
    A_high = torch.randn(batch, specs, 2, 3, 4, dtype=torch.float32)
    b = torch.randn(batch, specs, dtype=torch.float32)
    perturbation = LpBallPerturbation(p=p, eps=0.25)

    lb_high, ub_high = perturbation.concretize_affine(center=center, A=A_high, b=b)

    center_flat = center.flatten(1)
    A_flat = A_high.flatten(2)
    lb_flat, ub_flat = perturbation.concretize_affine(center=center_flat, A=A_flat, b=b)
    lb_op, ub_op = perturbation.concretize_affine(center=center, A=DenseLinearOperator(A_high), b=b)

    assert torch.allclose(lb_high, lb_flat)
    assert torch.allclose(ub_high, ub_flat)
    assert torch.allclose(lb_high, lb_op)
    assert torch.allclose(ub_high, ub_op)


def test_highdim_concretize_rejects_shape_mismatch() -> None:
    perturbation = LpBallPerturbation(p="inf", eps=0.2)
    center = torch.randn(2, 1, 4, 4, dtype=torch.float32)
    op = DenseLinearOperator(torch.randn(2, 3, 1, 3, 4, dtype=torch.float32))

    with pytest.raises(ValueError, match="affine shape mismatch"):
        perturbation.concretize_affine(center=center, A=op)

