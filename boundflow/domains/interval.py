from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from .base import AbstractDomain
from ..ir.bound import DomainState


@dataclass
class IntervalState(DomainState):
    lower: torch.Tensor
    upper: torch.Tensor

    def validate(self) -> None:
        if self.lower.shape != self.upper.shape:
            raise ValueError(f"interval shape mismatch: {self.lower.shape} vs {self.upper.shape}")
        if self.lower.dtype != self.upper.dtype:
            raise ValueError(f"interval dtype mismatch: {self.lower.dtype} vs {self.upper.dtype}")
        if self.lower.device != self.upper.device:
            raise ValueError(f"interval device mismatch: {self.lower.device} vs {self.upper.device}")


class IntervalDomain(AbstractDomain):
    @property
    def domain_id(self) -> str:
        return "interval"

    def affine_transformer(
        self, state_in: DomainState, weight: Any, bias: Any, **attrs: Any
    ) -> DomainState:
        """
        Interval bound propagation for affine ops.

        v0.1: supports:
        - linear: y = x @ weight^T + bias (weight: [out, in])
        - conv2d: y = conv2d(x, weight, bias, stride, padding, dilation, groups)
        """
        if not isinstance(state_in, IntervalState):
            raise TypeError(f"IntervalDomain expects IntervalState, got {type(state_in)}")

        x_l, x_u = state_in.lower, state_in.upper
        w = weight
        b = bias
        if not torch.is_tensor(w):
            w = torch.as_tensor(w, device=x_l.device)
        if b is None:
            b = 0.0
        if not torch.is_tensor(b):
            b = torch.as_tensor(b, device=x_l.device)

        op = attrs.get("op")

        if op == "conv2d" or (op is None and x_l.dim() == 4 and w.dim() == 4):
            stride = _as_int_tuple(attrs.get("stride", 1), dim=2)
            padding = _as_int_tuple(attrs.get("padding", 0), dim=2)
            dilation = _as_int_tuple(attrs.get("dilation", 1), dim=2)
            groups = int(attrs.get("groups", 1))

            w_pos = torch.clamp(w, min=0.0)
            w_neg = torch.clamp(w, max=0.0)
            y_l = F.conv2d(x_l, w_pos, bias=None, stride=stride, padding=padding, dilation=dilation, groups=groups) + F.conv2d(
                x_u, w_neg, bias=None, stride=stride, padding=padding, dilation=dilation, groups=groups
            )
            y_u = F.conv2d(x_u, w_pos, bias=None, stride=stride, padding=padding, dilation=dilation, groups=groups) + F.conv2d(
                x_l, w_neg, bias=None, stride=stride, padding=padding, dilation=dilation, groups=groups
            )
            if torch.is_tensor(b):
                y_l = y_l + b.view(1, -1, 1, 1)
                y_u = y_u + b.view(1, -1, 1, 1)
            return IntervalState(lower=y_l, upper=y_u)

        if op not in (None, "linear"):
            raise NotImplementedError(f"unsupported affine op for IntervalDomain: {op}")

        w_pos = torch.clamp(w, min=0.0)
        w_neg = torch.clamp(w, max=0.0)
        y_l = x_l.matmul(w_pos.t()) + x_u.matmul(w_neg.t())
        y_u = x_u.matmul(w_pos.t()) + x_l.matmul(w_neg.t())
        if torch.is_tensor(b):
            y_l = y_l + b
            y_u = y_u + b
        return IntervalState(lower=y_l, upper=y_u)

    def relu_transformer(self, state_in: DomainState) -> DomainState:
        if not isinstance(state_in, IntervalState):
            raise TypeError(f"IntervalDomain expects IntervalState, got {type(state_in)}")
        return IntervalState(lower=torch.relu(state_in.lower), upper=torch.relu(state_in.upper))

    def elementwise_transformer(self, states_in: List[DomainState], op: str) -> DomainState:
        if any(not isinstance(s, IntervalState) for s in states_in):
            raise TypeError("IntervalDomain expects IntervalState inputs")
        if len(states_in) != 2:
            raise ValueError(f"elementwise '{op}' expects 2 inputs, got {len(states_in)}")

        a, b = states_in[0], states_in[1]
        if op == "add":
            return IntervalState(lower=a.lower + b.lower, upper=a.upper + b.upper)
        if op == "mul":
            candidates = torch.stack(
                [
                    a.lower * b.lower,
                    a.lower * b.upper,
                    a.upper * b.lower,
                    a.upper * b.upper,
                ],
                dim=0,
            )
            return IntervalState(lower=candidates.min(dim=0).values, upper=candidates.max(dim=0).values)

        raise NotImplementedError(f"unsupported elementwise op: {op}")


IntTupleLike = Union[int, Sequence[int]]


def _as_int_tuple(value: IntTupleLike, *, dim: int) -> Tuple[int, ...]:
    if isinstance(value, int):
        return (value,) * dim
    if isinstance(value, (list, tuple)):
        if len(value) == dim:
            return tuple(int(v) for v in value)
        if len(value) == 1:
            return (int(value[0]),) * dim
    raise ValueError(f"invalid int tuple: {value} (expected dim={dim})")
