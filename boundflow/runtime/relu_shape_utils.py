from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import torch

from ..domains.interval import IntervalState


def shape_numel(shape: Sequence[int]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return int(total)


def relu_input_shapes(relu_pre: Dict[str, IntervalState]) -> Dict[str, Tuple[int, ...]]:
    return {name: tuple(int(dim) for dim in pre.lower.shape[1:]) for name, pre in relu_pre.items()}


def coerce_relu_param_shape(
    value_raw: Any,
    *,
    shape: Tuple[int, ...],
    batch_size: int,
    per_batch: bool,
    name: str,
    label: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    value = value_raw if torch.is_tensor(value_raw) else torch.as_tensor(value_raw, device=device, dtype=dtype)
    value = value.to(device=device, dtype=dtype)
    flat_dim = shape_numel(shape)
    batched_shape = (int(batch_size),) + tuple(shape)

    if value.dim() == 0:
        return value.expand(batched_shape if per_batch else shape).clone()

    if tuple(value.shape) == tuple(shape):
        if per_batch:
            return value.unsqueeze(0).expand(batched_shape).clone()
        return value.clone()

    if value.dim() == 1 and int(value.shape[0]) == flat_dim:
        reshaped = value.reshape(shape)
        if per_batch:
            return reshaped.unsqueeze(0).expand(batched_shape).clone()
        return reshaped.clone()

    if value.dim() == len(shape) + 1 and int(value.shape[0]) == 1 and tuple(value.shape[1:]) == tuple(shape):
        if per_batch:
            return value.expand(batched_shape).clone()
        return value.reshape(shape).clone()

    if value.dim() == 2 and int(value.shape[0]) == 1 and int(value.shape[1]) == flat_dim:
        reshaped = value.reshape((1,) + tuple(shape))
        if per_batch:
            return reshaped.expand(batched_shape).clone()
        return reshaped.reshape(shape).clone()

    if per_batch:
        if tuple(value.shape) == batched_shape:
            return value.clone()
        if value.dim() == 2 and int(value.shape[0]) == int(batch_size) and int(value.shape[1]) == flat_dim:
            return value.reshape(batched_shape).clone()
        raise ValueError(
            f"{label}[{name}] shape {tuple(value.shape)} does not match expected shared {shape} or per-batch {batched_shape}"
        )

    if tuple(value.shape) == batched_shape:
        raise ValueError(
            f"{label}[{name}] shape {tuple(value.shape)} is batch-specific; expected shared logical shape {shape}"
        )
    if value.dim() == 2 and int(value.shape[0]) == int(batch_size) and int(value.shape[1]) == flat_dim:
        raise ValueError(
            f"{label}[{name}] shape {tuple(value.shape)} is batch-specific; expected shared logical shape {shape}"
        )
    raise ValueError(f"{label}[{name}] shape {tuple(value.shape)} does not match expected shared logical shape {shape}")


def broadcast_relu_split_like_pre(
    split_raw: Any,
    *,
    pre: IntervalState,
    x_name: str,
    device: torch.device,
) -> torch.Tensor:
    split = split_raw if torch.is_tensor(split_raw) else torch.as_tensor(split_raw, device=device)
    split = split.to(device=device)
    if split.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError(f"relu split for {x_name} must be integer tensor, got dtype={split.dtype}")

    logical_shape = tuple(int(dim) for dim in pre.lower.shape[1:])
    flat_dim = shape_numel(logical_shape)
    batch = int(pre.lower.shape[0])
    target_shape = (batch, flat_dim)

    if split.dim() == 0:
        return split.reshape(1, 1).expand(target_shape)
    if tuple(split.shape) == logical_shape:
        return split.reshape(1, flat_dim).expand(target_shape)
    if split.dim() == 1 and int(split.shape[0]) == flat_dim:
        return split.reshape(1, flat_dim).expand(target_shape)
    if split.dim() == len(logical_shape) + 1 and int(split.shape[0]) == 1 and tuple(split.shape[1:]) == logical_shape:
        return split.reshape(1, flat_dim).expand(target_shape)
    if split.dim() == 2 and int(split.shape[0]) == 1 and int(split.shape[1]) == flat_dim:
        return split.expand(target_shape)
    if split.dim() == len(logical_shape) + 1 and int(split.shape[0]) == batch and tuple(split.shape[1:]) == logical_shape:
        return split.reshape(batch, flat_dim)
    if split.dim() == 2 and int(split.shape[0]) == batch and int(split.shape[1]) == flat_dim:
        return split
    raise ValueError(
        f"relu_split_state[{x_name}] shape {tuple(split.shape)} cannot broadcast to logical shape {logical_shape} with batch {batch}"
    )
