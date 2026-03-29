from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

import torch
import torch.nn.functional as F


def _prod(shape: Sequence[int]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return int(total)


def _normalize_input_shape(shape: Sequence[int] | None, *, default_last_dim: int | None = None) -> tuple[int, ...]:
    if shape is None:
        if default_last_dim is None:
            raise ValueError("input_shape is required when default_last_dim is not provided")
        shape = (int(default_last_dim),)
    out = tuple(int(dim) for dim in shape)
    if not out:
        raise ValueError("input_shape must be non-empty")
    if any(dim <= 0 for dim in out):
        raise ValueError(f"input_shape must be positive, got {out}")
    return out


def _ensure_float_matrix(
    value: torch.Tensor,
    *,
    name: str,
    rows: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value)}")
    if value.dim() != 2:
        raise ValueError(f"{name} expects rank-2 matrix, got {tuple(value.shape)}")
    if int(value.shape[0]) != int(rows):
        raise ValueError(f"{name} shape mismatch: expected first dim {rows}, got {tuple(value.shape)}")
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} expects floating tensor, got dtype={value.dtype}")
    if value.device != device:
        raise ValueError(f"{name} device mismatch: expected {device}, got {value.device}")
    if value.dtype != dtype:
        raise TypeError(f"{name} dtype mismatch: expected {dtype}, got {value.dtype}")
    return value


def _flatten_center(
    center: torch.Tensor,
    *,
    name: str,
    batch: int,
    input_shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not torch.is_tensor(center):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(center)}")
    if not torch.is_floating_point(center):
        raise TypeError(f"{name} expects floating tensor, got dtype={center.dtype}")
    if center.device != device:
        raise ValueError(f"{name} device mismatch: expected {device}, got {center.device}")
    if center.dtype != dtype:
        raise TypeError(f"{name} dtype mismatch: expected {dtype}, got {center.dtype}")
    expected_shape = (int(batch),) + tuple(input_shape)
    if tuple(int(dim) for dim in center.shape) != expected_shape:
        raise ValueError(f"{name} shape mismatch: expected {expected_shape}, got {tuple(center.shape)}")
    return center.reshape(int(batch), -1)


def _flatten_contract_input(
    value: torch.Tensor,
    *,
    name: str,
    batch: int,
    input_shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value)}")
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} expects floating tensor, got dtype={value.dtype}")
    if value.device != device:
        raise ValueError(f"{name} device mismatch: expected {device}, got {value.device}")
    if value.dtype != dtype:
        raise TypeError(f"{name} dtype mismatch: expected {dtype}, got {value.dtype}")
    flat_dim = _prod(input_shape)
    unbatched_shape = tuple(input_shape)
    batched_shape = (int(batch),) + tuple(input_shape)

    if tuple(int(dim) for dim in value.shape) == unbatched_shape:
        return value.reshape(1, flat_dim).expand(int(batch), -1)
    if tuple(int(dim) for dim in value.shape) == batched_shape:
        return value.reshape(int(batch), flat_dim)
    if len(input_shape) == 1 and value.dim() == 1 and int(value.shape[0]) == flat_dim:
        return value.reshape(1, flat_dim).expand(int(batch), -1)
    if len(input_shape) == 1 and value.dim() == 2 and int(value.shape[0]) == int(batch) and int(value.shape[1]) == flat_dim:
        return value
    raise ValueError(
        f"{name} shape mismatch: expected {unbatched_shape} or {batched_shape}, got {tuple(value.shape)}"
    )


def _conv2d_output_shape(
    input_shape: tuple[int, int, int],
    weight: torch.Tensor,
    *,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    groups: int,
) -> tuple[int, int, int]:
    if len(input_shape) != 3:
        raise ValueError(f"conv2d input_shape expects (C,H,W), got {input_shape}")
    if weight.dim() != 4:
        raise ValueError(f"conv2d weight expects rank-4 [O,I/groups,KH,KW], got {tuple(weight.shape)}")
    in_channels, in_h, in_w = (int(dim) for dim in input_shape)
    if in_channels != int(weight.shape[1]) * int(groups):
        raise ValueError(
            f"conv2d input channels mismatch: input_shape={input_shape} weight={tuple(weight.shape)} groups={groups}"
        )
    k_h = int(weight.shape[2])
    k_w = int(weight.shape[3])
    out_h = ((in_h + 2 * int(padding[0]) - int(dilation[0]) * (k_h - 1) - 1) // int(stride[0])) + 1
    out_w = ((in_w + 2 * int(padding[1]) - int(dilation[1]) * (k_w - 1) - 1) // int(stride[1])) + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError(
            f"conv2d output spatial dims must be positive, got {(out_h, out_w)} from input_shape={input_shape}"
        )
    return (int(weight.shape[0]), int(out_h), int(out_w))


def _conv_transpose_output_padding(
    *,
    input_shape: tuple[int, int, int],
    output_shape: tuple[int, int, int],
    weight: torch.Tensor,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
) -> tuple[int, int]:
    if len(input_shape) != 3 or len(output_shape) != 3:
        raise ValueError(f"conv_transpose expects 3D shapes, got input={input_shape} output={output_shape}")
    k_h = int(weight.shape[2])
    k_w = int(weight.shape[3])
    out_pad_h = int(input_shape[1]) - (
        (int(output_shape[1]) - 1) * int(stride[0]) - 2 * int(padding[0]) + int(dilation[0]) * (k_h - 1) + 1
    )
    out_pad_w = int(input_shape[2]) - (
        (int(output_shape[2]) - 1) * int(stride[1]) - 2 * int(padding[1]) + int(dilation[1]) * (k_w - 1) + 1
    )
    if out_pad_h < 0 or out_pad_w < 0:
        raise ValueError(
            f"invalid conv_transpose output_padding {(out_pad_h, out_pad_w)} for input={input_shape} output={output_shape}"
        )
    return (out_pad_h, out_pad_w)


def _normalize_conv_weight(
    weight: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not torch.is_tensor(weight):
        raise TypeError(f"conv2d weight must be a torch.Tensor, got {type(weight)}")
    if weight.dim() != 4:
        raise ValueError(f"conv2d weight expects rank-4 [O,I/groups,KH,KW], got {tuple(weight.shape)}")
    if not torch.is_floating_point(weight):
        raise TypeError(f"conv2d weight expects floating tensor, got dtype={weight.dtype}")
    if weight.device != device:
        raise ValueError(f"conv2d weight device mismatch: expected {device}, got {weight.device}")
    if weight.dtype != dtype:
        raise TypeError(f"conv2d weight dtype mismatch: expected {dtype}, got {weight.dtype}")
    return weight


def _normalize_pair(value: int | Sequence[int], *, name: str) -> tuple[int, int]:
    if isinstance(value, int):
        return (int(value), int(value))
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return (int(value[0]), int(value[0]))
        if len(value) == 2:
            return (int(value[0]), int(value[1]))
    raise ValueError(f"{name} expects int or pair, got {value}")


@runtime_checkable
class LinearOperator(Protocol):
    @property
    def shape(self) -> tuple[int, int, int]:
        """Logical shape [B, K, I]."""

    @property
    def input_shape(self) -> tuple[int, ...]:
        """Logical input shape excluding batch."""

    @property
    def input_numel(self) -> int: ...

    @property
    def spec_dim(self) -> int: ...

    @property
    def device(self) -> torch.device: ...

    @property
    def dtype(self) -> torch.dtype: ...

    def center_term(self, center: torch.Tensor) -> torch.Tensor:
        """Return A @ center with shape [B, K]."""

    def row_abs_sum(self) -> torch.Tensor:
        """Return row-wise L1 norm with shape [B, K]."""

    def row_l2_norm(self) -> torch.Tensor:
        """Return row-wise L2 norm with shape [B, K]."""

    def row_abs_max(self) -> torch.Tensor:
        """Return row-wise Linf norm with shape [B, K]."""

    def contract_input(self, vec: torch.Tensor) -> torch.Tensor:
        """Return row-wise contraction against vec with shape [B, K]."""

    def contract_last_dim(self, vec: torch.Tensor) -> torch.Tensor:
        """Compatibility alias for flat-input contraction."""

    def matmul_right(self, rhs: torch.Tensor) -> "LinearOperator":
        """Return lazy operator for A @ rhs, where rhs:[I,J] and A:[B,K,I]."""

    def reshape_input(self, new_input_shape: Sequence[int]) -> "LinearOperator":
        """View the logical input axis with a different non-batch shape."""

    def slice_input(self, new_input_shape: Sequence[int], *, start: int, stop: int) -> "LinearOperator":
        """Take a contiguous slice over the first logical input axis."""

    def add(self, other: "LinearOperator") -> "LinearOperator":
        """Return a lazy operator for self + other."""

    def conv2d_right(
        self,
        weight: torch.Tensor,
        *,
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        input_shape: Sequence[int],
    ) -> "LinearOperator":
        """Return lazy operator for A @ Conv2d(x)."""

    def split_pos_neg(self) -> tuple["LinearOperator", "LinearOperator"]:
        """Return exact coefficient-wise positive/negative decomposition."""

    def to_dense(self) -> torch.Tensor:
        """Materialize as a dense tensor with flat input axis [B, K, I]."""


def _validate_slice_input(
    *,
    input_shape: Sequence[int],
    new_input_shape: Sequence[int],
    start: int,
    stop: int,
) -> tuple[tuple[int, ...], tuple[int, ...], int, int]:
    base_shape = _normalize_input_shape(tuple(input_shape))
    sliced_shape = _normalize_input_shape(tuple(new_input_shape))
    start_i = int(start)
    stop_i = int(stop)
    if start_i < 0 or stop_i < start_i:
        raise ValueError(f"slice_input expects 0 <= start <= stop, got start={start_i} stop={stop_i}")
    if stop_i > int(base_shape[0]):
        raise ValueError(f"slice_input stop={stop_i} exceeds input_shape[0]={int(base_shape[0])}")
    expected_shape = (int(stop_i - start_i),) + tuple(int(dim) for dim in base_shape[1:])
    if tuple(sliced_shape) != tuple(expected_shape):
        raise ValueError(f"slice_input new_input_shape mismatch: expected {expected_shape}, got {sliced_shape}")
    return base_shape, sliced_shape, start_i, stop_i


def _add_operator_pair(lhs: LinearOperator, rhs: LinearOperator) -> "AddLinearOperator":
    return AddLinearOperator(lhs=lhs, rhs=rhs)


def _slice_rows(op: LinearOperator, *, start: int, stop: int) -> torch.Tensor:
    batch = int(op.shape[0])
    spec_dim = int(op.spec_dim)
    if len(op.input_shape) == 3:
        rows = _materialize_feature_map_rows(op, expected_input_shape=tuple(int(dim) for dim in op.input_shape))
        return rows[:, start:stop, ...]
    dense = op.to_dense().reshape(batch, spec_dim, *op.input_shape)
    return dense[:, :, start:stop, ...].reshape(batch, spec_dim, -1)


def _split_pos_neg_dense(op: LinearOperator) -> tuple["DenseLinearOperator", "DenseLinearOperator"]:
    dense = op.to_dense().reshape(int(op.shape[0]), int(op.spec_dim), *op.input_shape)
    return (
        DenseLinearOperator(dense.clamp_min(0.0), input_shape=op.input_shape),
        DenseLinearOperator(dense.clamp_max(0.0), input_shape=op.input_shape),
    )


def _coerce_per_input_tensor(
    value: torch.Tensor | float | int,
    *,
    batch: int,
    input_shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    name: str,
    require_nonnegative: bool = False,
) -> torch.Tensor:
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value, device=device, dtype=dtype)
    tensor = tensor.to(device=device, dtype=dtype)
    flat_dim = _prod(input_shape)
    target_shape = (int(batch), int(flat_dim))

    if tensor.dim() == 0:
        out = tensor.reshape(1, 1).expand(target_shape)
    elif tuple(int(dim) for dim in tensor.shape) == tuple(input_shape):
        out = tensor.reshape(1, flat_dim).expand(target_shape)
    elif tensor.dim() == 1 and int(tensor.shape[0]) == flat_dim:
        out = tensor.reshape(1, flat_dim).expand(target_shape)
    elif tensor.dim() == len(input_shape) + 1 and int(tensor.shape[0]) == 1 and tuple(int(dim) for dim in tensor.shape[1:]) == tuple(input_shape):
        out = tensor.reshape(1, flat_dim).expand(target_shape)
    elif tensor.dim() == 2 and int(tensor.shape[0]) == 1 and int(tensor.shape[1]) == flat_dim:
        out = tensor.expand(target_shape)
    elif tensor.dim() == len(input_shape) + 1 and int(tensor.shape[0]) == batch and tuple(int(dim) for dim in tensor.shape[1:]) == tuple(input_shape):
        out = tensor.reshape(batch, flat_dim)
    elif tensor.dim() == 2 and int(tensor.shape[0]) == batch and int(tensor.shape[1]) == flat_dim:
        out = tensor
    else:
        raise ValueError(
            f"{name} shape {tuple(int(dim) for dim in tensor.shape)} cannot broadcast to input_shape={input_shape} with batch={batch}"
        )
    if require_nonnegative and bool((out < 0).any().item()):
        raise ValueError(f"{name} must be nonnegative")
    return out


def _normalize_index_permutation(
    value: torch.Tensor | Sequence[int],
    *,
    length: int,
    device: torch.device,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    index = value if torch.is_tensor(value) else torch.as_tensor(value, device=device)
    if index.dim() != 1:
        raise ValueError(f"{name} expects rank-1 permutation, got {tuple(index.shape)}")
    if int(index.shape[0]) != int(length):
        raise ValueError(f"{name} length mismatch: expected {length}, got {tuple(index.shape)}")
    if torch.is_floating_point(index):
        raise TypeError(f"{name} expects integer permutation, got dtype={index.dtype}")
    index = index.to(device=device, dtype=torch.long)
    expected = torch.arange(int(length), device=device, dtype=torch.long)
    if not torch.equal(index.sort().values, expected):
        raise ValueError(f"{name} must be a permutation of [0, {int(length) - 1}]")
    inverse = torch.empty_like(index)
    inverse[index] = expected
    return index, inverse


@dataclass(frozen=True)
class DenseLinearOperator:
    coeffs: torch.Tensor
    input_shape: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not torch.is_tensor(self.coeffs):
            raise TypeError(f"DenseLinearOperator.coeffs must be a torch.Tensor, got {type(self.coeffs)}")
        if self.coeffs.dim() < 3:
            raise ValueError(
                f"DenseLinearOperator expects rank>=3 coeffs [B,K,*input_shape], got {tuple(self.coeffs.shape)}"
            )
        if not torch.is_floating_point(self.coeffs):
            raise TypeError(f"DenseLinearOperator expects floating coeffs, got dtype={self.coeffs.dtype}")
        inferred_shape = (
            tuple(int(dim) for dim in self.coeffs.shape[2:])
            if self.coeffs.dim() > 3
            else (int(self.coeffs.shape[2]),)
        )
        normalized_shape = _normalize_input_shape(
            self.input_shape,
            default_last_dim=int(self.coeffs.reshape(int(self.coeffs.shape[0]), int(self.coeffs.shape[1]), -1).shape[2]),
        )
        if _prod(normalized_shape) != int(self.coeffs.reshape(int(self.coeffs.shape[0]), int(self.coeffs.shape[1]), -1).shape[2]):
            raise ValueError(
                f"DenseLinearOperator input_shape {normalized_shape} does not match coeffs shape {tuple(self.coeffs.shape)}"
            )
        if self.input_shape is None and inferred_shape:
            normalized_shape = inferred_shape
        object.__setattr__(self, "input_shape", normalized_shape)

    @property
    def shape(self) -> tuple[int, int, int]:
        flat = self.coeffs.reshape(int(self.coeffs.shape[0]), int(self.coeffs.shape[1]), -1)
        return tuple(int(v) for v in flat.shape)

    @property
    def input_numel(self) -> int:
        return _prod(self.input_shape)

    @property
    def spec_dim(self) -> int:
        return int(self.shape[1])

    @property
    def device(self) -> torch.device:
        return self.coeffs.device

    @property
    def dtype(self) -> torch.dtype:
        return self.coeffs.dtype

    def _flat_coeffs(self) -> torch.Tensor:
        return self.coeffs.reshape(self.shape)

    def center_term(self, center: torch.Tensor) -> torch.Tensor:
        flat = _flatten_center(
            center,
            name="center",
            batch=self.shape[0],
            input_shape=self.input_shape,
            device=self.device,
            dtype=self.dtype,
        )
        return torch.einsum("bki,bi->bk", self._flat_coeffs(), flat)

    def row_abs_sum(self) -> torch.Tensor:
        return self._flat_coeffs().abs().sum(dim=2)

    def row_l2_norm(self) -> torch.Tensor:
        return torch.linalg.vector_norm(self._flat_coeffs(), ord=2, dim=2)

    def row_abs_max(self) -> torch.Tensor:
        return self._flat_coeffs().abs().amax(dim=2)

    def contract_input(self, vec: torch.Tensor) -> torch.Tensor:
        flat = _flatten_contract_input(
            vec,
            name="vec",
            batch=self.shape[0],
            input_shape=self.input_shape,
            device=self.device,
            dtype=self.dtype,
        )
        return torch.einsum("bki,bi->bk", self._flat_coeffs(), flat)

    def contract_last_dim(self, vec: torch.Tensor) -> torch.Tensor:
        if len(self.input_shape) != 1:
            raise NotImplementedError(
                f"contract_last_dim only supports flat input_shape, got {self.input_shape}"
            )
        return self.contract_input(vec)

    def matmul_right(self, rhs: torch.Tensor) -> LinearOperator:
        if len(self.input_shape) != 1:
            raise NotImplementedError(f"matmul_right only supports flat input_shape, got {self.input_shape}")
        rr = _ensure_float_matrix(rhs, name="rhs", rows=self.input_numel, device=self.device, dtype=self.dtype)
        return RightMatmulLinearOperator(base=self, rhs=rr)

    def reshape_input(self, new_input_shape: Sequence[int]) -> LinearOperator:
        shape = _normalize_input_shape(tuple(new_input_shape))
        if _prod(shape) != self.input_numel:
            raise ValueError(f"reshape_input shape mismatch: {shape} does not match input_numel={self.input_numel}")
        if tuple(shape) == tuple(self.input_shape):
            return self
        return ReshapeInputLinearOperator(base=self, input_shape=shape)

    def slice_input(self, new_input_shape: Sequence[int], *, start: int, stop: int) -> LinearOperator:
        _base_shape, sliced_shape, start_i, stop_i = _validate_slice_input(
            input_shape=self.input_shape,
            new_input_shape=new_input_shape,
            start=start,
            stop=stop,
        )
        dense = self.coeffs.reshape(int(self.shape[0]), int(self.shape[1]), *self.input_shape)
        sliced = dense[:, :, start_i:stop_i, ...].contiguous()
        return DenseLinearOperator(sliced, input_shape=sliced_shape)

    def add(self, other: LinearOperator) -> LinearOperator:
        return _add_operator_pair(self, other)

    def conv2d_right(
        self,
        weight: torch.Tensor,
        *,
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        input_shape: Sequence[int],
    ) -> LinearOperator:
        in_shape = _normalize_input_shape(tuple(input_shape))
        if len(in_shape) != 3:
            raise ValueError(f"conv2d_right expects NCHW input_shape without batch, got {in_shape}")
        w = _normalize_conv_weight(weight, device=self.device, dtype=self.dtype)
        stride_2 = _normalize_pair(stride, name="stride")
        padding_2 = _normalize_pair(padding, name="padding")
        dilation_2 = _normalize_pair(dilation, name="dilation")
        out_shape = _conv2d_output_shape(in_shape, w, stride=stride_2, padding=padding_2, dilation=dilation_2, groups=int(groups))
        base: LinearOperator = self
        if tuple(base.input_shape) != tuple(out_shape):
            if base.input_numel != _prod(out_shape):
                raise ValueError(
                    f"conv2d_right output shape mismatch: base.input_shape={base.input_shape} conv_output_shape={out_shape}"
                )
            base = base.reshape_input(out_shape)
        return Conv2dLinearOperator(
            base=base,
            weight=w,
            stride=stride_2,
            padding=padding_2,
            dilation=dilation_2,
            groups=int(groups),
            input_shape=in_shape,
        )

    def split_pos_neg(self) -> tuple[LinearOperator, LinearOperator]:
        coeffs = self.coeffs.reshape(int(self.shape[0]), int(self.spec_dim), *self.input_shape)
        return (
            DenseLinearOperator(coeffs.clamp_min(0.0), input_shape=self.input_shape),
            DenseLinearOperator(coeffs.clamp_max(0.0), input_shape=self.input_shape),
        )

    def to_dense(self) -> torch.Tensor:
        return self._flat_coeffs()


@dataclass(frozen=True)
class RightMatmulLinearOperator:
    base: LinearOperator
    rhs: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.base, LinearOperator):
            raise TypeError(f"RightMatmulLinearOperator.base must satisfy LinearOperator, got {type(self.base)}")
        if len(self.base.input_shape) != 1:
            raise NotImplementedError(
                f"RightMatmulLinearOperator requires flat base input_shape, got {self.base.input_shape}"
            )
        _ensure_float_matrix(
            self.rhs,
            name="rhs",
            rows=self.base.input_numel,
            device=self.base.device,
            dtype=self.base.dtype,
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        return (int(self.base.shape[0]), int(self.base.shape[1]), int(self.rhs.shape[1]))

    @property
    def input_shape(self) -> tuple[int, ...]:
        return (int(self.rhs.shape[1]),)

    @property
    def input_numel(self) -> int:
        return int(self.rhs.shape[1])

    @property
    def spec_dim(self) -> int:
        return int(self.base.spec_dim)

    @property
    def device(self) -> torch.device:
        return self.base.device

    @property
    def dtype(self) -> torch.dtype:
        return self.base.dtype

    def center_term(self, center: torch.Tensor) -> torch.Tensor:
        flat = _flatten_center(
            center,
            name="center",
            batch=self.shape[0],
            input_shape=self.input_shape,
            device=self.device,
            dtype=self.dtype,
        )
        projected = flat.matmul(self.rhs.t())
        return self.base.center_term(projected)

    def row_abs_sum(self) -> torch.Tensor:
        return self.to_dense().abs().sum(dim=2)

    def row_l2_norm(self) -> torch.Tensor:
        return torch.linalg.vector_norm(self.to_dense(), ord=2, dim=2)

    def row_abs_max(self) -> torch.Tensor:
        return self.to_dense().abs().amax(dim=2)

    def contract_input(self, vec: torch.Tensor) -> torch.Tensor:
        flat = _flatten_contract_input(
            vec,
            name="vec",
            batch=self.shape[0],
            input_shape=self.input_shape,
            device=self.device,
            dtype=self.dtype,
        )
        projected = flat.matmul(self.rhs.t())
        return self.base.contract_input(projected)

    def contract_last_dim(self, vec: torch.Tensor) -> torch.Tensor:
        return self.contract_input(vec)

    def matmul_right(self, rhs: torch.Tensor) -> LinearOperator:
        rr = _ensure_float_matrix(rhs, name="rhs", rows=self.input_numel, device=self.device, dtype=self.dtype)
        return RightMatmulLinearOperator(base=self.base, rhs=self.rhs.matmul(rr))

    def reshape_input(self, new_input_shape: Sequence[int]) -> LinearOperator:
        shape = _normalize_input_shape(tuple(new_input_shape))
        if _prod(shape) != self.input_numel:
            raise ValueError(f"reshape_input shape mismatch: {shape} does not match input_numel={self.input_numel}")
        if tuple(shape) == tuple(self.input_shape):
            return self
        return ReshapeInputLinearOperator(base=self, input_shape=shape)

    def slice_input(self, new_input_shape: Sequence[int], *, start: int, stop: int) -> LinearOperator:
        return SliceInputLinearOperator(base=self, input_shape=tuple(new_input_shape), start=int(start), stop=int(stop))

    def add(self, other: LinearOperator) -> LinearOperator:
        return _add_operator_pair(self, other)

    def conv2d_right(
        self,
        weight: torch.Tensor,
        *,
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        input_shape: Sequence[int],
    ) -> LinearOperator:
        return self.reshape_input(
            _conv2d_output_shape(
                _normalize_input_shape(tuple(input_shape)),
                _normalize_conv_weight(weight, device=self.device, dtype=self.dtype),
                stride=_normalize_pair(stride, name="stride"),
                padding=_normalize_pair(padding, name="padding"),
                dilation=_normalize_pair(dilation, name="dilation"),
                groups=int(groups),
            )
        ).conv2d_right(
            weight,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            input_shape=input_shape,
        )

    def split_pos_neg(self) -> tuple[LinearOperator, LinearOperator]:
        return _split_pos_neg_dense(self)

    def to_dense(self) -> torch.Tensor:
        return torch.einsum("bko,oi->bki", self.base.to_dense(), self.rhs)


@dataclass(frozen=True)
class ReshapeInputLinearOperator:
    base: LinearOperator
    input_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.base, LinearOperator):
            raise TypeError(f"ReshapeInputLinearOperator.base must satisfy LinearOperator, got {type(self.base)}")
        shape = _normalize_input_shape(self.input_shape)
        if _prod(shape) != self.base.input_numel:
            raise ValueError(
                f"ReshapeInputLinearOperator input_shape {shape} does not match base.input_numel={self.base.input_numel}"
            )
        object.__setattr__(self, "input_shape", shape)

    @property
    def shape(self) -> tuple[int, int, int]:
        return (int(self.base.shape[0]), int(self.base.shape[1]), int(self.base.shape[2]))

    @property
    def input_numel(self) -> int:
        return _prod(self.input_shape)

    @property
    def spec_dim(self) -> int:
        return int(self.base.spec_dim)

    @property
    def device(self) -> torch.device:
        return self.base.device

    @property
    def dtype(self) -> torch.dtype:
        return self.base.dtype

    def _reshape_for_base(self, flat: torch.Tensor) -> torch.Tensor:
        return flat.view(int(flat.shape[0]), *self.base.input_shape)

    def center_term(self, center: torch.Tensor) -> torch.Tensor:
        flat = _flatten_center(
            center,
            name="center",
            batch=self.shape[0],
            input_shape=self.input_shape,
            device=self.device,
            dtype=self.dtype,
        )
        return self.base.center_term(self._reshape_for_base(flat))

    def row_abs_sum(self) -> torch.Tensor:
        return self.base.row_abs_sum()

    def row_l2_norm(self) -> torch.Tensor:
        return self.base.row_l2_norm()

    def row_abs_max(self) -> torch.Tensor:
        return self.base.row_abs_max()

    def contract_input(self, vec: torch.Tensor) -> torch.Tensor:
        flat = _flatten_contract_input(
            vec,
            name="vec",
            batch=self.shape[0],
            input_shape=self.input_shape,
            device=self.device,
            dtype=self.dtype,
        )
        return self.base.contract_input(self._reshape_for_base(flat))

    def contract_last_dim(self, vec: torch.Tensor) -> torch.Tensor:
        if len(self.input_shape) != 1:
            raise NotImplementedError(
                f"contract_last_dim only supports flat input_shape, got {self.input_shape}"
            )
        return self.contract_input(vec)

    def matmul_right(self, rhs: torch.Tensor) -> LinearOperator:
        if len(self.input_shape) != 1:
            raise NotImplementedError(f"matmul_right only supports flat input_shape, got {self.input_shape}")
        rr = _ensure_float_matrix(rhs, name="rhs", rows=self.input_numel, device=self.device, dtype=self.dtype)
        return RightMatmulLinearOperator(base=self, rhs=rr)

    def reshape_input(self, new_input_shape: Sequence[int]) -> LinearOperator:
        shape = _normalize_input_shape(tuple(new_input_shape))
        if _prod(shape) != self.input_numel:
            raise ValueError(f"reshape_input shape mismatch: {shape} does not match input_numel={self.input_numel}")
        if tuple(shape) == tuple(self.input_shape):
            return self
        return ReshapeInputLinearOperator(base=self.base, input_shape=shape)

    def slice_input(self, new_input_shape: Sequence[int], *, start: int, stop: int) -> LinearOperator:
        return SliceInputLinearOperator(base=self, input_shape=tuple(new_input_shape), start=int(start), stop=int(stop))

    def add(self, other: LinearOperator) -> LinearOperator:
        return _add_operator_pair(self, other)

    def conv2d_right(
        self,
        weight: torch.Tensor,
        *,
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        input_shape: Sequence[int],
    ) -> LinearOperator:
        in_shape = _normalize_input_shape(tuple(input_shape))
        w = _normalize_conv_weight(weight, device=self.device, dtype=self.dtype)
        stride_2 = _normalize_pair(stride, name="stride")
        padding_2 = _normalize_pair(padding, name="padding")
        dilation_2 = _normalize_pair(dilation, name="dilation")
        out_shape = _conv2d_output_shape(in_shape, w, stride=stride_2, padding=padding_2, dilation=dilation_2, groups=int(groups))
        base: LinearOperator = self
        if tuple(base.input_shape) != tuple(out_shape):
            if base.input_numel != _prod(out_shape):
                raise ValueError(
                    f"conv2d_right output shape mismatch: base.input_shape={base.input_shape} conv_output_shape={out_shape}"
                )
            base = base.reshape_input(out_shape)
        return Conv2dLinearOperator(
            base=base,
            weight=w,
            stride=stride_2,
            padding=padding_2,
            dilation=dilation_2,
            groups=int(groups),
            input_shape=in_shape,
        )

    def split_pos_neg(self) -> tuple[LinearOperator, LinearOperator]:
        return _split_pos_neg_dense(self)

    def to_dense(self) -> torch.Tensor:
        return self.base.to_dense()


@dataclass(frozen=True)
class ReindexInputLinearOperator:
    base: LinearOperator
    input_shape: tuple[int, ...]
    gather_index: torch.Tensor
    scatter_index: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.base, LinearOperator):
            raise TypeError(f"ReindexInputLinearOperator.base must satisfy LinearOperator, got {type(self.base)}")
        shape = _normalize_input_shape(self.input_shape)
        if _prod(shape) != self.base.input_numel:
            raise ValueError(
                f"ReindexInputLinearOperator input_shape {shape} does not match base.input_numel={self.base.input_numel}"
            )
        gather_index, scatter_index = _normalize_index_permutation(
            self.gather_index,
            length=int(self.base.input_numel),
            device=self.base.device,
            name="gather_index",
        )
        object.__setattr__(self, "input_shape", shape)
        object.__setattr__(self, "gather_index", gather_index)
        object.__setattr__(self, "scatter_index", scatter_index)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.base.shape)

    @property
    def input_numel(self) -> int:
        return _prod(self.input_shape)

    @property
    def spec_dim(self) -> int:
        return int(self.base.spec_dim)

    @property
    def device(self) -> torch.device:
        return self.base.device

    @property
    def dtype(self) -> torch.dtype:
        return self.base.dtype

    def _gather_center_for_base(self, center: torch.Tensor) -> torch.Tensor:
        flat = _flatten_center(
            center,
            name="center",
            batch=self.shape[0],
            input_shape=self.input_shape,
            device=self.device,
            dtype=self.dtype,
        )
        return flat.index_select(1, self.gather_index).view(int(self.shape[0]), *self.base.input_shape)

    def _gather_input_for_base(self, value: torch.Tensor, *, name: str) -> torch.Tensor:
        flat = _flatten_contract_input(
            value,
            name=name,
            batch=self.shape[0],
            input_shape=self.input_shape,
            device=self.device,
            dtype=self.dtype,
        )
        return flat.index_select(1, self.gather_index).view(int(self.shape[0]), *self.base.input_shape)

    def center_term(self, center: torch.Tensor) -> torch.Tensor:
        return self.base.center_term(self._gather_center_for_base(center))

    def row_abs_sum(self) -> torch.Tensor:
        return self.base.row_abs_sum()

    def row_l2_norm(self) -> torch.Tensor:
        return self.base.row_l2_norm()

    def row_abs_max(self) -> torch.Tensor:
        return self.base.row_abs_max()

    def contract_input(self, vec: torch.Tensor) -> torch.Tensor:
        return self.base.contract_input(self._gather_input_for_base(vec, name="vec"))

    def contract_last_dim(self, vec: torch.Tensor) -> torch.Tensor:
        if len(self.input_shape) != 1:
            raise NotImplementedError(f"contract_last_dim only supports flat input_shape, got {self.input_shape}")
        return self.contract_input(vec)

    def matmul_right(self, rhs: torch.Tensor) -> LinearOperator:
        if len(self.input_shape) != 1:
            raise NotImplementedError(f"matmul_right only supports flat input_shape, got {self.input_shape}")
        rr = _ensure_float_matrix(rhs, name="rhs", rows=self.input_numel, device=self.device, dtype=self.dtype)
        return RightMatmulLinearOperator(base=self, rhs=rr)

    def reshape_input(self, new_input_shape: Sequence[int]) -> LinearOperator:
        shape = _normalize_input_shape(tuple(new_input_shape))
        if _prod(shape) != self.input_numel:
            raise ValueError(f"reshape_input shape mismatch: {shape} does not match input_numel={self.input_numel}")
        if tuple(shape) == tuple(self.input_shape):
            return self
        return ReshapeInputLinearOperator(base=self, input_shape=shape)

    def slice_input(self, new_input_shape: Sequence[int], *, start: int, stop: int) -> LinearOperator:
        return SliceInputLinearOperator(base=self, input_shape=tuple(new_input_shape), start=int(start), stop=int(stop))

    def add(self, other: LinearOperator) -> LinearOperator:
        return _add_operator_pair(self, other)

    def conv2d_right(
        self,
        weight: torch.Tensor,
        *,
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        input_shape: Sequence[int],
    ) -> LinearOperator:
        in_shape = _normalize_input_shape(tuple(input_shape))
        w = _normalize_conv_weight(weight, device=self.device, dtype=self.dtype)
        stride_2 = _normalize_pair(stride, name="stride")
        padding_2 = _normalize_pair(padding, name="padding")
        dilation_2 = _normalize_pair(dilation, name="dilation")
        out_shape = _conv2d_output_shape(in_shape, w, stride=stride_2, padding=padding_2, dilation=dilation_2, groups=int(groups))
        base: LinearOperator = self
        if tuple(base.input_shape) != tuple(out_shape):
            if base.input_numel != _prod(out_shape):
                raise ValueError(
                    f"conv2d_right output shape mismatch: base.input_shape={base.input_shape} conv_output_shape={out_shape}"
                )
            base = base.reshape_input(out_shape)
        return Conv2dLinearOperator(
            base=base,
            weight=w,
            stride=stride_2,
            padding=padding_2,
            dilation=dilation_2,
            groups=int(groups),
            input_shape=in_shape,
        )

    def split_pos_neg(self) -> tuple[LinearOperator, LinearOperator]:
        base_pos, base_neg = self.base.split_pos_neg()
        return (
            ReindexInputLinearOperator(base=base_pos, input_shape=self.input_shape, gather_index=self.gather_index),
            ReindexInputLinearOperator(base=base_neg, input_shape=self.input_shape, gather_index=self.gather_index),
        )

    def to_dense(self) -> torch.Tensor:
        return self.base.to_dense().index_select(2, self.scatter_index)


@dataclass(frozen=True)
class Conv2dLinearOperator:
    base: LinearOperator
    weight: torch.Tensor
    stride: tuple[int, int]
    padding: tuple[int, int]
    dilation: tuple[int, int]
    groups: int
    input_shape: tuple[int, int, int]

    def __post_init__(self) -> None:
        if not isinstance(self.base, LinearOperator):
            raise TypeError(f"Conv2dLinearOperator.base must satisfy LinearOperator, got {type(self.base)}")
        w = _normalize_conv_weight(self.weight, device=self.base.device, dtype=self.base.dtype)
        input_shape = _normalize_input_shape(self.input_shape)
        if len(input_shape) != 3:
            raise ValueError(f"Conv2dLinearOperator expects NCHW input_shape without batch, got {input_shape}")
        stride = _normalize_pair(self.stride, name="stride")
        padding = _normalize_pair(self.padding, name="padding")
        dilation = _normalize_pair(self.dilation, name="dilation")
        output_shape = _conv2d_output_shape(input_shape, w, stride=stride, padding=padding, dilation=dilation, groups=int(self.groups))
        base = self.base
        if tuple(base.input_shape) != tuple(output_shape):
            if base.input_numel != _prod(output_shape):
                raise ValueError(
                    f"Conv2dLinearOperator base.input_shape={base.input_shape} does not match conv output_shape={output_shape}"
                )
            base = base.reshape_input(output_shape)
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "weight", w)
        object.__setattr__(self, "stride", stride)
        object.__setattr__(self, "padding", padding)
        object.__setattr__(self, "dilation", dilation)
        object.__setattr__(self, "groups", int(self.groups))
        object.__setattr__(self, "input_shape", tuple(int(dim) for dim in input_shape))

    @property
    def shape(self) -> tuple[int, int, int]:
        return (int(self.base.shape[0]), int(self.base.shape[1]), int(self.input_numel))

    @property
    def input_numel(self) -> int:
        return _prod(self.input_shape)

    @property
    def spec_dim(self) -> int:
        return int(self.base.spec_dim)

    @property
    def device(self) -> torch.device:
        return self.base.device

    @property
    def dtype(self) -> torch.dtype:
        return self.base.dtype

    def _run_conv(self, flat_input: torch.Tensor) -> torch.Tensor:
        x = flat_input.view(int(flat_input.shape[0]), *self.input_shape)
        return F.conv2d(
            x,
            self.weight,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def center_term(self, center: torch.Tensor) -> torch.Tensor:
        flat = _flatten_center(
            center,
            name="center",
            batch=self.shape[0],
            input_shape=self.input_shape,
            device=self.device,
            dtype=self.dtype,
        )
        return self.base.center_term(self._run_conv(flat))

    def row_abs_sum(self) -> torch.Tensor:
        rows = _materialize_feature_map_rows(self, expected_input_shape=self.input_shape)
        return _reduce_feature_map_rows(rows, batch=int(self.shape[0]), spec_dim=self.spec_dim, reduce="l1")

    def row_l2_norm(self) -> torch.Tensor:
        rows = _materialize_feature_map_rows(self, expected_input_shape=self.input_shape)
        return _reduce_feature_map_rows(rows, batch=int(self.shape[0]), spec_dim=self.spec_dim, reduce="l2")

    def row_abs_max(self) -> torch.Tensor:
        rows = _materialize_feature_map_rows(self, expected_input_shape=self.input_shape)
        return _reduce_feature_map_rows(rows, batch=int(self.shape[0]), spec_dim=self.spec_dim, reduce="linf")

    def contract_input(self, vec: torch.Tensor) -> torch.Tensor:
        flat = _flatten_contract_input(
            vec,
            name="vec",
            batch=self.shape[0],
            input_shape=self.input_shape,
            device=self.device,
            dtype=self.dtype,
        )
        return self.base.contract_input(self._run_conv(flat))

    def contract_last_dim(self, vec: torch.Tensor) -> torch.Tensor:
        if len(self.input_shape) != 1:
            raise NotImplementedError(
                f"contract_last_dim only supports flat input_shape, got {self.input_shape}"
            )
        return self.contract_input(vec)

    def matmul_right(self, rhs: torch.Tensor) -> LinearOperator:
        raise NotImplementedError(f"matmul_right is only defined for flat input_shape, got {self.input_shape}")

    def reshape_input(self, new_input_shape: Sequence[int]) -> LinearOperator:
        shape = _normalize_input_shape(tuple(new_input_shape))
        if _prod(shape) != self.input_numel:
            raise ValueError(f"reshape_input shape mismatch: {shape} does not match input_numel={self.input_numel}")
        if tuple(shape) == tuple(self.input_shape):
            return self
        return ReshapeInputLinearOperator(base=self, input_shape=shape)

    def slice_input(self, new_input_shape: Sequence[int], *, start: int, stop: int) -> LinearOperator:
        return SliceInputLinearOperator(base=self, input_shape=tuple(new_input_shape), start=int(start), stop=int(stop))

    def add(self, other: LinearOperator) -> LinearOperator:
        return _add_operator_pair(self, other)

    def conv2d_right(
        self,
        weight: torch.Tensor,
        *,
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        input_shape: Sequence[int],
    ) -> LinearOperator:
        in_shape = _normalize_input_shape(tuple(input_shape))
        w = _normalize_conv_weight(weight, device=self.device, dtype=self.dtype)
        stride_2 = _normalize_pair(stride, name="stride")
        padding_2 = _normalize_pair(padding, name="padding")
        dilation_2 = _normalize_pair(dilation, name="dilation")
        out_shape = _conv2d_output_shape(in_shape, w, stride=stride_2, padding=padding_2, dilation=dilation_2, groups=int(groups))
        base: LinearOperator = self
        if tuple(base.input_shape) != tuple(out_shape):
            if base.input_numel != _prod(out_shape):
                raise ValueError(
                    f"conv2d_right output shape mismatch: base.input_shape={base.input_shape} conv_output_shape={out_shape}"
                )
            base = base.reshape_input(out_shape)
        return Conv2dLinearOperator(
            base=base,
            weight=w,
            stride=stride_2,
            padding=padding_2,
            dilation=dilation_2,
            groups=int(groups),
            input_shape=in_shape,
        )

    def split_pos_neg(self) -> tuple[LinearOperator, LinearOperator]:
        return _split_pos_neg_dense(self)

    def to_dense(self) -> torch.Tensor:
        output_shape = tuple(int(dim) for dim in self.base.input_shape)
        if len(output_shape) != 3:
            raise NotImplementedError(
                f"Conv2dLinearOperator.to_dense expects base.input_shape rank-3 conv output, got {output_shape}"
            )
        base_dense = self.base.to_dense().view(int(self.shape[0]) * int(self.spec_dim), *output_shape)
        output_padding = _conv_transpose_output_padding(
            input_shape=self.input_shape,
            output_shape=output_shape,
            weight=self.weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        pieces = F.conv_transpose2d(
            base_dense,
            self.weight,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            output_padding=output_padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return pieces.view(int(self.shape[0]), int(self.spec_dim), -1)


@dataclass(frozen=True)
class AddLinearOperator:
    lhs: LinearOperator
    rhs: LinearOperator

    def __post_init__(self) -> None:
        if not isinstance(self.lhs, LinearOperator):
            raise TypeError(f"AddLinearOperator.lhs must satisfy LinearOperator, got {type(self.lhs)}")
        if not isinstance(self.rhs, LinearOperator):
            raise TypeError(f"AddLinearOperator.rhs must satisfy LinearOperator, got {type(self.rhs)}")
        if tuple(self.lhs.shape) != tuple(self.rhs.shape):
            raise ValueError(f"add shape mismatch: lhs={self.lhs.shape} rhs={self.rhs.shape}")
        if tuple(self.lhs.input_shape) != tuple(self.rhs.input_shape):
            raise ValueError(f"add input_shape mismatch: lhs={self.lhs.input_shape} rhs={self.rhs.input_shape}")
        if self.lhs.device != self.rhs.device:
            raise ValueError(f"add device mismatch: lhs={self.lhs.device} rhs={self.rhs.device}")
        if self.lhs.dtype != self.rhs.dtype:
            raise TypeError(f"add dtype mismatch: lhs={self.lhs.dtype} rhs={self.rhs.dtype}")

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.lhs.shape)

    @property
    def input_shape(self) -> tuple[int, ...]:
        return tuple(int(dim) for dim in self.lhs.input_shape)

    @property
    def input_numel(self) -> int:
        return int(self.lhs.input_numel)

    @property
    def spec_dim(self) -> int:
        return int(self.lhs.spec_dim)

    @property
    def device(self) -> torch.device:
        return self.lhs.device

    @property
    def dtype(self) -> torch.dtype:
        return self.lhs.dtype

    def center_term(self, center: torch.Tensor) -> torch.Tensor:
        return self.lhs.center_term(center) + self.rhs.center_term(center)

    def row_abs_sum(self) -> torch.Tensor:
        if len(self.input_shape) == 3:
            rows = _materialize_feature_map_rows(self.lhs, expected_input_shape=self.input_shape) + _materialize_feature_map_rows(
                self.rhs,
                expected_input_shape=self.input_shape,
            )
            return _reduce_feature_map_rows(rows, batch=int(self.shape[0]), spec_dim=self.spec_dim, reduce="l1")
        dense = self.to_dense()
        return dense.abs().sum(dim=2)

    def row_l2_norm(self) -> torch.Tensor:
        if len(self.input_shape) == 3:
            rows = _materialize_feature_map_rows(self.lhs, expected_input_shape=self.input_shape) + _materialize_feature_map_rows(
                self.rhs,
                expected_input_shape=self.input_shape,
            )
            return _reduce_feature_map_rows(rows, batch=int(self.shape[0]), spec_dim=self.spec_dim, reduce="l2")
        dense = self.to_dense()
        return torch.linalg.vector_norm(dense, ord=2, dim=2)

    def row_abs_max(self) -> torch.Tensor:
        if len(self.input_shape) == 3:
            rows = _materialize_feature_map_rows(self.lhs, expected_input_shape=self.input_shape) + _materialize_feature_map_rows(
                self.rhs,
                expected_input_shape=self.input_shape,
            )
            return _reduce_feature_map_rows(rows, batch=int(self.shape[0]), spec_dim=self.spec_dim, reduce="linf")
        dense = self.to_dense()
        return dense.abs().amax(dim=2)

    def contract_input(self, vec: torch.Tensor) -> torch.Tensor:
        return self.lhs.contract_input(vec) + self.rhs.contract_input(vec)

    def contract_last_dim(self, vec: torch.Tensor) -> torch.Tensor:
        if len(self.input_shape) != 1:
            raise NotImplementedError(f"contract_last_dim only supports flat input_shape, got {self.input_shape}")
        return self.contract_input(vec)

    def matmul_right(self, rhs: torch.Tensor) -> LinearOperator:
        if len(self.input_shape) != 1:
            raise NotImplementedError(f"matmul_right only supports flat input_shape, got {self.input_shape}")
        rr = _ensure_float_matrix(rhs, name="rhs", rows=self.input_numel, device=self.device, dtype=self.dtype)
        return RightMatmulLinearOperator(base=self, rhs=rr)

    def reshape_input(self, new_input_shape: Sequence[int]) -> LinearOperator:
        shape = _normalize_input_shape(tuple(new_input_shape))
        if _prod(shape) != self.input_numel:
            raise ValueError(f"reshape_input shape mismatch: {shape} does not match input_numel={self.input_numel}")
        if tuple(shape) == tuple(self.input_shape):
            return self
        return ReshapeInputLinearOperator(base=self, input_shape=shape)

    def slice_input(self, new_input_shape: Sequence[int], *, start: int, stop: int) -> LinearOperator:
        return SliceInputLinearOperator(base=self, input_shape=tuple(new_input_shape), start=int(start), stop=int(stop))

    def add(self, other: LinearOperator) -> LinearOperator:
        return _add_operator_pair(self, other)

    def conv2d_right(
        self,
        weight: torch.Tensor,
        *,
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        input_shape: Sequence[int],
    ) -> LinearOperator:
        in_shape = _normalize_input_shape(tuple(input_shape))
        w = _normalize_conv_weight(weight, device=self.device, dtype=self.dtype)
        stride_2 = _normalize_pair(stride, name="stride")
        padding_2 = _normalize_pair(padding, name="padding")
        dilation_2 = _normalize_pair(dilation, name="dilation")
        out_shape = _conv2d_output_shape(in_shape, w, stride=stride_2, padding=padding_2, dilation=dilation_2, groups=int(groups))
        base: LinearOperator = self
        if tuple(base.input_shape) != tuple(out_shape):
            if base.input_numel != _prod(out_shape):
                raise ValueError(
                    f"conv2d_right output shape mismatch: base.input_shape={base.input_shape} conv_output_shape={out_shape}"
                )
            base = base.reshape_input(out_shape)
        return Conv2dLinearOperator(
            base=base,
            weight=w,
            stride=stride_2,
            padding=padding_2,
            dilation=dilation_2,
            groups=int(groups),
            input_shape=in_shape,
        )

    def split_pos_neg(self) -> tuple[LinearOperator, LinearOperator]:
        return _split_pos_neg_dense(self)

    def to_dense(self) -> torch.Tensor:
        return self.lhs.to_dense() + self.rhs.to_dense()


@dataclass(frozen=True)
class SliceInputLinearOperator:
    base: LinearOperator
    input_shape: tuple[int, ...]
    start: int
    stop: int

    def __post_init__(self) -> None:
        if not isinstance(self.base, LinearOperator):
            raise TypeError(f"SliceInputLinearOperator.base must satisfy LinearOperator, got {type(self.base)}")
        _base_shape, sliced_shape, start_i, stop_i = _validate_slice_input(
            input_shape=self.base.input_shape,
            new_input_shape=self.input_shape,
            start=self.start,
            stop=self.stop,
        )
        object.__setattr__(self, "input_shape", sliced_shape)
        object.__setattr__(self, "start", start_i)
        object.__setattr__(self, "stop", stop_i)

    @property
    def shape(self) -> tuple[int, int, int]:
        return (int(self.base.shape[0]), int(self.base.shape[1]), int(self.input_numel))

    @property
    def input_numel(self) -> int:
        return _prod(self.input_shape)

    @property
    def spec_dim(self) -> int:
        return int(self.base.spec_dim)

    @property
    def device(self) -> torch.device:
        return self.base.device

    @property
    def dtype(self) -> torch.dtype:
        return self.base.dtype

    def _embed_input(self, value: torch.Tensor, *, name: str) -> torch.Tensor:
        flat = _flatten_contract_input(
            value,
            name=name,
            batch=self.shape[0],
            input_shape=self.input_shape,
            device=self.device,
            dtype=self.dtype,
        )
        embedded = torch.zeros((int(self.shape[0]), *self.base.input_shape), device=self.device, dtype=self.dtype)
        embedded[:, self.start:self.stop, ...] = flat.view(int(self.shape[0]), *self.input_shape)
        return embedded

    def center_term(self, center: torch.Tensor) -> torch.Tensor:
        return self.base.center_term(self._embed_input(center, name="center"))

    def row_abs_sum(self) -> torch.Tensor:
        if len(self.base.input_shape) == 3:
            rows = _slice_rows(self.base, start=self.start, stop=self.stop)
            return _reduce_feature_map_rows(rows, batch=int(self.shape[0]), spec_dim=self.spec_dim, reduce="l1")
        dense = self.to_dense()
        return dense.abs().sum(dim=2)

    def row_l2_norm(self) -> torch.Tensor:
        if len(self.base.input_shape) == 3:
            rows = _slice_rows(self.base, start=self.start, stop=self.stop)
            return _reduce_feature_map_rows(rows, batch=int(self.shape[0]), spec_dim=self.spec_dim, reduce="l2")
        dense = self.to_dense()
        return torch.linalg.vector_norm(dense, ord=2, dim=2)

    def row_abs_max(self) -> torch.Tensor:
        if len(self.base.input_shape) == 3:
            rows = _slice_rows(self.base, start=self.start, stop=self.stop)
            return _reduce_feature_map_rows(rows, batch=int(self.shape[0]), spec_dim=self.spec_dim, reduce="linf")
        dense = self.to_dense()
        return dense.abs().amax(dim=2)

    def contract_input(self, vec: torch.Tensor) -> torch.Tensor:
        return self.base.contract_input(self._embed_input(vec, name="vec"))

    def contract_last_dim(self, vec: torch.Tensor) -> torch.Tensor:
        if len(self.input_shape) != 1:
            raise NotImplementedError(f"contract_last_dim only supports flat input_shape, got {self.input_shape}")
        return self.contract_input(vec)

    def matmul_right(self, rhs: torch.Tensor) -> LinearOperator:
        if len(self.input_shape) != 1:
            raise NotImplementedError(f"matmul_right only supports flat input_shape, got {self.input_shape}")
        rr = _ensure_float_matrix(rhs, name="rhs", rows=self.input_numel, device=self.device, dtype=self.dtype)
        return RightMatmulLinearOperator(base=self, rhs=rr)

    def reshape_input(self, new_input_shape: Sequence[int]) -> LinearOperator:
        shape = _normalize_input_shape(tuple(new_input_shape))
        if _prod(shape) != self.input_numel:
            raise ValueError(f"reshape_input shape mismatch: {shape} does not match input_numel={self.input_numel}")
        if tuple(shape) == tuple(self.input_shape):
            return self
        return ReshapeInputLinearOperator(base=self, input_shape=shape)

    def slice_input(self, new_input_shape: Sequence[int], *, start: int, stop: int) -> LinearOperator:
        return SliceInputLinearOperator(base=self, input_shape=tuple(new_input_shape), start=int(start), stop=int(stop))

    def add(self, other: LinearOperator) -> LinearOperator:
        return _add_operator_pair(self, other)

    def conv2d_right(
        self,
        weight: torch.Tensor,
        *,
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        input_shape: Sequence[int],
    ) -> LinearOperator:
        in_shape = _normalize_input_shape(tuple(input_shape))
        w = _normalize_conv_weight(weight, device=self.device, dtype=self.dtype)
        stride_2 = _normalize_pair(stride, name="stride")
        padding_2 = _normalize_pair(padding, name="padding")
        dilation_2 = _normalize_pair(dilation, name="dilation")
        out_shape = _conv2d_output_shape(in_shape, w, stride=stride_2, padding=padding_2, dilation=dilation_2, groups=int(groups))
        base: LinearOperator = self
        if tuple(base.input_shape) != tuple(out_shape):
            if base.input_numel != _prod(out_shape):
                raise ValueError(
                    f"conv2d_right output shape mismatch: base.input_shape={base.input_shape} conv_output_shape={out_shape}"
                )
            base = base.reshape_input(out_shape)
        return Conv2dLinearOperator(
            base=base,
            weight=w,
            stride=stride_2,
            padding=padding_2,
            dilation=dilation_2,
            groups=int(groups),
            input_shape=in_shape,
        )

    def split_pos_neg(self) -> tuple[LinearOperator, LinearOperator]:
        return _split_pos_neg_dense(self)

    def to_dense(self) -> torch.Tensor:
        return _slice_rows(self.base, start=self.start, stop=self.stop).reshape(int(self.shape[0]), int(self.spec_dim), -1)


@dataclass(frozen=True)
class ScaledInputLinearOperator:
    base: LinearOperator
    scale: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.base, LinearOperator):
            raise TypeError(f"ScaledInputLinearOperator.base must satisfy LinearOperator, got {type(self.base)}")
        scale = _coerce_per_input_tensor(
            self.scale,
            batch=int(self.base.shape[0]),
            input_shape=tuple(int(dim) for dim in self.base.input_shape),
            device=self.base.device,
            dtype=self.base.dtype,
            name="scale",
            require_nonnegative=True,
        )
        object.__setattr__(self, "scale", scale)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.base.shape)

    @property
    def input_shape(self) -> tuple[int, ...]:
        return tuple(int(dim) for dim in self.base.input_shape)

    @property
    def input_numel(self) -> int:
        return int(self.base.input_numel)

    @property
    def spec_dim(self) -> int:
        return int(self.base.spec_dim)

    @property
    def device(self) -> torch.device:
        return self.base.device

    @property
    def dtype(self) -> torch.dtype:
        return self.base.dtype

    def _logical_scale(self) -> torch.Tensor:
        return self.scale.view(int(self.shape[0]), *self.input_shape)

    def _scale_value(self, value: torch.Tensor, *, name: str) -> torch.Tensor:
        flat = _flatten_contract_input(
            value,
            name=name,
            batch=int(self.shape[0]),
            input_shape=self.input_shape,
            device=self.device,
            dtype=self.dtype,
        )
        return (flat * self.scale).view(int(self.shape[0]), *self.input_shape)

    def center_term(self, center: torch.Tensor) -> torch.Tensor:
        return self.base.center_term(self._scale_value(center, name="center"))

    def row_abs_sum(self) -> torch.Tensor:
        return self.to_dense().abs().sum(dim=2)

    def row_l2_norm(self) -> torch.Tensor:
        return torch.linalg.vector_norm(self.to_dense(), ord=2, dim=2)

    def row_abs_max(self) -> torch.Tensor:
        return self.to_dense().abs().amax(dim=2)

    def contract_input(self, vec: torch.Tensor) -> torch.Tensor:
        return self.base.contract_input(self._scale_value(vec, name="vec"))

    def contract_last_dim(self, vec: torch.Tensor) -> torch.Tensor:
        if len(self.input_shape) != 1:
            raise NotImplementedError(f"contract_last_dim only supports flat input_shape, got {self.input_shape}")
        return self.contract_input(vec)

    def matmul_right(self, rhs: torch.Tensor) -> LinearOperator:
        if len(self.input_shape) != 1:
            raise NotImplementedError(f"matmul_right only supports flat input_shape, got {self.input_shape}")
        rr = _ensure_float_matrix(rhs, name="rhs", rows=self.input_numel, device=self.device, dtype=self.dtype)
        return RightMatmulLinearOperator(base=self, rhs=rr)

    def reshape_input(self, new_input_shape: Sequence[int]) -> LinearOperator:
        shape = _normalize_input_shape(tuple(new_input_shape))
        if _prod(shape) != self.input_numel:
            raise ValueError(f"reshape_input shape mismatch: {shape} does not match input_numel={self.input_numel}")
        if tuple(shape) == tuple(self.input_shape):
            return self
        return ScaledInputLinearOperator(base=self.base.reshape_input(shape), scale=self.scale)

    def slice_input(self, new_input_shape: Sequence[int], *, start: int, stop: int) -> LinearOperator:
        _base_shape, sliced_shape, start_i, stop_i = _validate_slice_input(
            input_shape=self.input_shape,
            new_input_shape=new_input_shape,
            start=start,
            stop=stop,
        )
        logical = self._logical_scale()
        sliced = logical[:, start_i:stop_i, ...].contiguous()
        return ScaledInputLinearOperator(
            base=self.base.slice_input(sliced_shape, start=start_i, stop=stop_i),
            scale=sliced,
        )

    def add(self, other: LinearOperator) -> LinearOperator:
        return _add_operator_pair(self, other)

    def conv2d_right(
        self,
        weight: torch.Tensor,
        *,
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        input_shape: Sequence[int],
    ) -> LinearOperator:
        in_shape = _normalize_input_shape(tuple(input_shape))
        w = _normalize_conv_weight(weight, device=self.device, dtype=self.dtype)
        stride_2 = _normalize_pair(stride, name="stride")
        padding_2 = _normalize_pair(padding, name="padding")
        dilation_2 = _normalize_pair(dilation, name="dilation")
        out_shape = _conv2d_output_shape(in_shape, w, stride=stride_2, padding=padding_2, dilation=dilation_2, groups=int(groups))
        base: LinearOperator = self
        if tuple(base.input_shape) != tuple(out_shape):
            if base.input_numel != _prod(out_shape):
                raise ValueError(
                    f"conv2d_right output shape mismatch: base.input_shape={base.input_shape} conv_output_shape={out_shape}"
                )
            base = base.reshape_input(out_shape)
        return Conv2dLinearOperator(
            base=base,
            weight=w,
            stride=stride_2,
            padding=padding_2,
            dilation=dilation_2,
            groups=int(groups),
            input_shape=in_shape,
        )

    def split_pos_neg(self) -> tuple[LinearOperator, LinearOperator]:
        return _split_pos_neg_dense(self)

    def to_dense(self) -> torch.Tensor:
        return self.base.to_dense() * self.scale.unsqueeze(1)


@dataclass(frozen=True)
class RepeatedRowLinearOperator:
    coeffs: torch.Tensor
    spec_dim_size: int
    input_shape: tuple[int, ...]
    batch_size: int

    def __post_init__(self) -> None:
        input_shape = _normalize_input_shape(self.input_shape)
        if int(self.batch_size) <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if int(self.spec_dim_size) <= 0:
            raise ValueError(f"spec_dim_size must be positive, got {self.spec_dim_size}")
        coeffs = self.coeffs
        if not torch.is_tensor(coeffs):
            coeffs = torch.as_tensor(coeffs)
        if not torch.is_floating_point(coeffs):
            raise TypeError(f"RepeatedRowLinearOperator.coeffs expects floating tensor, got dtype={coeffs.dtype}")
        coeffs = _coerce_per_input_tensor(
            coeffs,
            batch=int(self.batch_size),
            input_shape=input_shape,
            device=coeffs.device,
            dtype=coeffs.dtype,
            name="coeffs",
        )
        object.__setattr__(self, "coeffs", coeffs)
        object.__setattr__(self, "input_shape", input_shape)
        object.__setattr__(self, "batch_size", int(self.batch_size))
        object.__setattr__(self, "spec_dim_size", int(self.spec_dim_size))

    @property
    def shape(self) -> tuple[int, int, int]:
        return (int(self.batch_size), int(self.spec_dim_size), int(self.input_numel))

    @property
    def input_numel(self) -> int:
        return _prod(self.input_shape)

    @property
    def spec_dim(self) -> int:
        return int(self.spec_dim_size)

    @property
    def device(self) -> torch.device:
        return self.coeffs.device

    @property
    def dtype(self) -> torch.dtype:
        return self.coeffs.dtype

    def _logical_coeffs(self) -> torch.Tensor:
        return self.coeffs.view(int(self.batch_size), *self.input_shape)

    def _row_contract(self, value: torch.Tensor, *, name: str) -> torch.Tensor:
        flat = _flatten_contract_input(
            value,
            name=name,
            batch=int(self.batch_size),
            input_shape=self.input_shape,
            device=self.device,
            dtype=self.dtype,
        )
        return (self.coeffs * flat).sum(dim=1)

    def center_term(self, center: torch.Tensor) -> torch.Tensor:
        row = self._row_contract(center, name="center")
        return row.unsqueeze(1).expand(int(self.batch_size), int(self.spec_dim_size))

    def row_abs_sum(self) -> torch.Tensor:
        row = self.coeffs.abs().sum(dim=1)
        return row.unsqueeze(1).expand(int(self.batch_size), int(self.spec_dim_size))

    def row_l2_norm(self) -> torch.Tensor:
        row = torch.linalg.vector_norm(self.coeffs, ord=2, dim=1)
        return row.unsqueeze(1).expand(int(self.batch_size), int(self.spec_dim_size))

    def row_abs_max(self) -> torch.Tensor:
        row = self.coeffs.abs().amax(dim=1)
        return row.unsqueeze(1).expand(int(self.batch_size), int(self.spec_dim_size))

    def contract_input(self, vec: torch.Tensor) -> torch.Tensor:
        row = self._row_contract(vec, name="vec")
        return row.unsqueeze(1).expand(int(self.batch_size), int(self.spec_dim_size))

    def contract_last_dim(self, vec: torch.Tensor) -> torch.Tensor:
        if len(self.input_shape) != 1:
            raise NotImplementedError(f"contract_last_dim only supports flat input_shape, got {self.input_shape}")
        return self.contract_input(vec)

    def matmul_right(self, rhs: torch.Tensor) -> LinearOperator:
        if len(self.input_shape) != 1:
            raise NotImplementedError(f"matmul_right only supports flat input_shape, got {self.input_shape}")
        rr = _ensure_float_matrix(rhs, name="rhs", rows=self.input_numel, device=self.device, dtype=self.dtype)
        return RepeatedRowLinearOperator(
            coeffs=self.coeffs.matmul(rr),
            spec_dim_size=self.spec_dim_size,
            input_shape=(int(rr.shape[1]),),
            batch_size=self.batch_size,
        )

    def reshape_input(self, new_input_shape: Sequence[int]) -> LinearOperator:
        shape = _normalize_input_shape(tuple(new_input_shape))
        if _prod(shape) != self.input_numel:
            raise ValueError(f"reshape_input shape mismatch: {shape} does not match input_numel={self.input_numel}")
        if tuple(shape) == tuple(self.input_shape):
            return self
        return RepeatedRowLinearOperator(
            coeffs=self.coeffs,
            spec_dim_size=self.spec_dim_size,
            input_shape=shape,
            batch_size=self.batch_size,
        )

    def slice_input(self, new_input_shape: Sequence[int], *, start: int, stop: int) -> LinearOperator:
        _base_shape, sliced_shape, start_i, stop_i = _validate_slice_input(
            input_shape=self.input_shape,
            new_input_shape=new_input_shape,
            start=start,
            stop=stop,
        )
        logical = self._logical_coeffs()
        sliced = logical[:, start_i:stop_i, ...].contiguous()
        return RepeatedRowLinearOperator(
            coeffs=sliced,
            spec_dim_size=self.spec_dim_size,
            input_shape=sliced_shape,
            batch_size=self.batch_size,
        )

    def add(self, other: LinearOperator) -> LinearOperator:
        return _add_operator_pair(self, other)

    def conv2d_right(
        self,
        weight: torch.Tensor,
        *,
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        input_shape: Sequence[int],
    ) -> LinearOperator:
        in_shape = _normalize_input_shape(tuple(input_shape))
        w = _normalize_conv_weight(weight, device=self.device, dtype=self.dtype)
        stride_2 = _normalize_pair(stride, name="stride")
        padding_2 = _normalize_pair(padding, name="padding")
        dilation_2 = _normalize_pair(dilation, name="dilation")
        out_shape = _conv2d_output_shape(in_shape, w, stride=stride_2, padding=padding_2, dilation=dilation_2, groups=int(groups))
        base: LinearOperator = self
        if tuple(base.input_shape) != tuple(out_shape):
            if base.input_numel != _prod(out_shape):
                raise ValueError(
                    f"conv2d_right output shape mismatch: base.input_shape={base.input_shape} conv_output_shape={out_shape}"
                )
            base = base.reshape_input(out_shape)
        return Conv2dLinearOperator(
            base=base,
            weight=w,
            stride=stride_2,
            padding=padding_2,
            dilation=dilation_2,
            groups=int(groups),
            input_shape=in_shape,
        )

    def split_pos_neg(self) -> tuple[LinearOperator, LinearOperator]:
        return (
            RepeatedRowLinearOperator(
                coeffs=self.coeffs.clamp_min(0.0),
                spec_dim_size=self.spec_dim_size,
                input_shape=self.input_shape,
                batch_size=self.batch_size,
            ),
            RepeatedRowLinearOperator(
                coeffs=self.coeffs.clamp_max(0.0),
                spec_dim_size=self.spec_dim_size,
                input_shape=self.input_shape,
                batch_size=self.batch_size,
            ),
        )

    def to_dense(self) -> torch.Tensor:
        return self.coeffs.unsqueeze(1).expand(int(self.batch_size), int(self.spec_dim_size), int(self.input_numel))


def _materialize_feature_map_rows(
    op: LinearOperator,
    *,
    expected_input_shape: tuple[int, int, int],
) -> torch.Tensor:
    if len(expected_input_shape) != 3:
        raise ValueError(f"expected_input_shape must be rank-3 (C,H,W), got {expected_input_shape}")

    batch = int(op.shape[0])
    spec_dim = int(op.spec_dim)
    if _prod(expected_input_shape) != int(op.shape[2]):
        raise ValueError(
            f"feature-map materialization shape mismatch: expected_input_shape={expected_input_shape} op.shape={op.shape}"
        )

    if isinstance(op, DenseLinearOperator):
        return op.coeffs.reshape(batch * spec_dim, *expected_input_shape)

    if isinstance(op, AddLinearOperator):
        return _materialize_feature_map_rows(op.lhs, expected_input_shape=expected_input_shape) + _materialize_feature_map_rows(
            op.rhs,
            expected_input_shape=expected_input_shape,
        )

    if isinstance(op, ReshapeInputLinearOperator):
        if tuple(op.input_shape) != tuple(expected_input_shape):
            raise ValueError(
                f"reshape operator input_shape mismatch: expected={expected_input_shape} actual={op.input_shape}"
            )
        if len(op.base.input_shape) == 3:
            base_rows = _materialize_feature_map_rows(
                op.base,
                expected_input_shape=tuple(int(dim) for dim in op.base.input_shape),
            )
            return base_rows.reshape(batch * spec_dim, *expected_input_shape)
        return op.base.to_dense().reshape(batch * spec_dim, *expected_input_shape)

    if isinstance(op, ReindexInputLinearOperator):
        if tuple(op.input_shape) != tuple(expected_input_shape):
            raise ValueError(
                f"reindex operator input_shape mismatch: expected={expected_input_shape} actual={op.input_shape}"
            )
        if len(op.base.input_shape) == 3:
            base_rows = _materialize_feature_map_rows(
                op.base,
                expected_input_shape=tuple(int(dim) for dim in op.base.input_shape),
            )
            flat = base_rows.reshape(batch * spec_dim, -1).index_select(1, op.scatter_index)
            return flat.reshape(batch * spec_dim, *expected_input_shape)
        return op.base.to_dense().index_select(2, op.scatter_index).reshape(batch * spec_dim, *expected_input_shape)

    if isinstance(op, SliceInputLinearOperator):
        if tuple(op.input_shape) != tuple(expected_input_shape):
            raise ValueError(
                f"slice operator input_shape mismatch: expected={expected_input_shape} actual={op.input_shape}"
            )
        if len(op.base.input_shape) != 3:
            return op.to_dense().reshape(batch * spec_dim, *expected_input_shape)
        base_rows = _materialize_feature_map_rows(op.base, expected_input_shape=tuple(int(dim) for dim in op.base.input_shape))
        return base_rows[:, op.start:op.stop, ...]

    if isinstance(op, Conv2dLinearOperator):
        output_shape = tuple(int(dim) for dim in op.base.input_shape)
        if len(output_shape) != 3:
            raise NotImplementedError(
                f"Conv2dLinearOperator base.input_shape must be rank-3 for feature-map materialization, got {output_shape}"
            )
        base_rows = _materialize_feature_map_rows(op.base, expected_input_shape=output_shape)
        output_padding = _conv_transpose_output_padding(
            input_shape=tuple(int(dim) for dim in op.input_shape),
            output_shape=output_shape,
            weight=op.weight,
            stride=op.stride,
            padding=op.padding,
            dilation=op.dilation,
        )
        rows = F.conv_transpose2d(
            base_rows,
            op.weight,
            bias=None,
            stride=op.stride,
            padding=op.padding,
            output_padding=output_padding,
            dilation=op.dilation,
            groups=op.groups,
        )
        return rows.reshape(batch * spec_dim, *expected_input_shape)

    return op.to_dense().reshape(batch * spec_dim, *expected_input_shape)


def _reduce_feature_map_rows(
    rows: torch.Tensor,
    *,
    batch: int,
    spec_dim: int,
    reduce: str,
) -> torch.Tensor:
    if rows.dim() != 4:
        raise ValueError(f"rows must be rank-4 [B*K,C,H,W], got {tuple(rows.shape)}")
    if int(rows.shape[0]) != int(batch) * int(spec_dim):
        raise ValueError(
            f"rows batch/spec mismatch: rows[0]={int(rows.shape[0])} expected={int(batch) * int(spec_dim)}"
        )

    if reduce == "l1":
        return rows.abs().sum(dim=(1, 2, 3)).view(int(batch), int(spec_dim))
    if reduce == "l2":
        return torch.linalg.vector_norm(rows.reshape(int(batch) * int(spec_dim), -1), ord=2, dim=1).view(
            int(batch), int(spec_dim)
        )
    if reduce == "linf":
        return rows.abs().amax(dim=(1, 2, 3)).view(int(batch), int(spec_dim))
    raise ValueError(f"unsupported feature-map reduction: {reduce}")


def as_linear_operator(A: torch.Tensor | LinearOperator) -> LinearOperator:
    if torch.is_tensor(A):
        return DenseLinearOperator(A)
    if isinstance(A, LinearOperator):
        return A
    raise TypeError(f"expected torch.Tensor or LinearOperator, got {type(A)}")
