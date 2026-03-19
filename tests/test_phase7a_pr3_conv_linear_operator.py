import pytest
import torch
import torch.nn.functional as F

from boundflow.runtime.linear_operator import Conv2dLinearOperator, DenseLinearOperator


def _conv_output_shape(
    input_shape: tuple[int, int, int],
    weight: torch.Tensor,
    *,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
) -> tuple[int, int, int]:
    in_c, in_h, in_w = input_shape
    if int(weight.shape[1]) != in_c:
        raise ValueError("weight input channels must match input_shape")
    k_h = int(weight.shape[2])
    k_w = int(weight.shape[3])
    out_h = ((in_h + 2 * padding[0] - dilation[0] * (k_h - 1) - 1) // stride[0]) + 1
    out_w = ((in_w + 2 * padding[1] - dilation[1] * (k_w - 1) - 1) // stride[1]) + 1
    return (int(weight.shape[0]), out_h, out_w)


def _dense_conv_reference(
    coeffs: torch.Tensor,
    weight: torch.Tensor,
    *,
    input_shape: tuple[int, int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    groups: int,
) -> torch.Tensor:
    input_numel = input_shape[0] * input_shape[1] * input_shape[2]
    eye = torch.eye(input_numel, device=coeffs.device, dtype=coeffs.dtype).view(input_numel, *input_shape)
    conv_basis = F.conv2d(eye, weight, bias=None, stride=stride, padding=padding, dilation=dilation, groups=groups)
    conv_basis = conv_basis.unsqueeze(0).expand(int(coeffs.shape[0]), -1, -1, -1, -1)
    return torch.einsum("bko,bio->bki", coeffs.flatten(2), conv_basis.flatten(2))


def test_conv2d_linear_operator_matches_dense_reference() -> None:
    torch.manual_seed(0)
    batch = 2
    specs = 4
    input_shape = (2, 5, 4)
    weight = torch.randn(3, 2, 3, 2, dtype=torch.float32)
    stride = (2, 1)
    padding = (1, 0)
    dilation = (1, 1)
    groups = 1
    output_shape = _conv_output_shape(input_shape, weight, stride=stride, padding=padding, dilation=dilation)
    coeffs = torch.randn(batch, specs, *output_shape, dtype=torch.float32)

    base = DenseLinearOperator(coeffs)
    op = base.conv2d_right(
        weight,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        input_shape=input_shape,
    )

    assert isinstance(op, Conv2dLinearOperator)
    dense_ref = _dense_conv_reference(
        coeffs,
        weight,
        input_shape=input_shape,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    center = torch.randn(batch, *input_shape, dtype=torch.float32)
    vec = torch.randn(*input_shape, dtype=torch.float32)

    assert torch.allclose(op.to_dense(), dense_ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(op.center_term(center), torch.einsum("bki,bi->bk", dense_ref, center.flatten(1)), atol=1e-5, rtol=1e-5)
    assert torch.allclose(op.contract_input(vec), torch.einsum("bki,i->bk", dense_ref, vec.flatten()), atol=1e-5, rtol=1e-5)


def test_conv2d_linear_operator_rejects_shape_mismatch() -> None:
    torch.manual_seed(0)
    weight = torch.randn(3, 2, 3, 3, dtype=torch.float32)
    coeffs = torch.randn(2, 4, 3, 4, 4, dtype=torch.float32)
    op = DenseLinearOperator(coeffs).conv2d_right(
        weight,
        stride=(1, 1),
        padding=(1, 1),
        dilation=(1, 1),
        groups=1,
        input_shape=(2, 4, 4),
    )

    with pytest.raises(ValueError, match="shape mismatch"):
        op.center_term(torch.randn(2, 2, 3, 4, dtype=torch.float32))

