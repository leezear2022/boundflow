from __future__ import annotations

from typing import Any, Sequence


def normalize_concat_axis(axis_raw: Any, *, rank_with_batch: int, caller: str) -> int:
    axis = int(axis_raw)
    if rank_with_batch == 2:
        if axis in (1, -1):
            return 1
        raise NotImplementedError(f"{caller} only supports feature-axis concat for rank-2 [B,F], got axis={axis}")
    if rank_with_batch == 4:
        if axis in (1, -3):
            return 1
        raise NotImplementedError(
            f"{caller} only supports NCHW channel-axis concat for rank-4 [B,C,H,W], got axis={axis}"
        )
    raise NotImplementedError(
        f"{caller} only supports concat on rank-2 [B,F] or rank-4 [B,C,H,W], got rank={rank_with_batch}"
    )


def validate_concat_tensor_shapes(
    input_shapes: Sequence[Sequence[int]],
    *,
    axis: int,
    caller: str,
) -> int:
    if not input_shapes:
        raise ValueError(f"{caller} expects at least 1 shape")
    ref_shape = tuple(int(dim) for dim in input_shapes[0])
    total = int(ref_shape[axis])
    for idx, shape_raw in enumerate(input_shapes[1:], start=1):
        shape = tuple(int(dim) for dim in shape_raw)
        if len(shape) != len(ref_shape):
            raise NotImplementedError(
                f"{caller} only supports concat with equal-rank inputs, got {ref_shape} and {shape} at input {idx}"
            )
        for dim_i, (lhs, rhs) in enumerate(zip(ref_shape, shape)):
            if dim_i == axis:
                continue
            if lhs != rhs:
                raise NotImplementedError(
                    f"{caller} only supports concat with exact same-shape non-axis dims, got {ref_shape} and {shape}"
                )
        total += int(shape[axis])
    return total


def validate_concat_value_shapes(
    input_shapes: Sequence[Sequence[int]],
    *,
    caller: str,
) -> int:
    if not input_shapes:
        raise ValueError(f"{caller} expects at least 1 shape")
    ref_shape = tuple(int(dim) for dim in input_shapes[0])
    total = int(ref_shape[0])
    for idx, shape_raw in enumerate(input_shapes[1:], start=1):
        shape = tuple(int(dim) for dim in shape_raw)
        if len(shape) != len(ref_shape):
            raise NotImplementedError(
                f"{caller} only supports concat with equal-rank inputs, got {ref_shape} and {shape} at input {idx}"
            )
        for dim_i, (lhs, rhs) in enumerate(zip(ref_shape, shape)):
            if dim_i == 0:
                continue
            if lhs != rhs:
                raise NotImplementedError(
                    f"{caller} only supports concat with exact same-shape non-axis dims, got {ref_shape} and {shape}"
                )
        total += int(shape[0])
    return total
