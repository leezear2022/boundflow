from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class LinearSpec:
    """
    线性性质（Property）规格：y = C @ f(x)

    对齐 auto_LiRPA 的 `compute_bounds(C=...)` 语义：
    - C shape: [batch, n_spec, out_dim]
    - 输出 bounds shape: [batch, n_spec]
    """

    C: torch.Tensor
    name: str = "linear_spec"
    rhs: Optional[torch.Tensor] = None
    sense: str = ">="  # placeholder

    def validate(self) -> None:
        if self.C.dim() != 3:
            raise ValueError(f"LinearSpec.C must be rank-3 [B,S,O], got {tuple(self.C.shape)}")

