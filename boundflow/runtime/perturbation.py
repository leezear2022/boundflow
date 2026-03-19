from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, Tuple, Union

import torch

from ..ir.bound import DomainState
from .linear_operator import LinearOperator, as_linear_operator


class PerturbationSet(Protocol):
    @property
    def perturbation_id(self) -> str: ...

    def bounding_box(self, center: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: ...

    def concretize_matmul(
        self, *, center: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Concretize affine form y = center @ weight^T + bias, where x = center + δ, δ ∈ S.

        Supported shapes:
        - center: [B, I], weight: [O, I], bias: [O] or [B, O]
        - center: [B, I], weight: [B, O, I], bias: [B, O]
        """

    def concretize_affine(
        self, *, center: torch.Tensor, A: torch.Tensor | LinearOperator, b: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Concretize affine form y = A @ x + b, where x = center + δ, δ ∈ S.

        Supported shapes:
        - center: [B, *input_shape]
        - A: [B, K, I] / [B, K, *input_shape], or a LinearOperator with the same logical shape
        - b: [B, K] (optional; defaults to 0)
        Returns:
        - (lb, ub): both [B, K]
        """


@dataclass
class InputPerturbationState(DomainState):
    center: torch.Tensor
    perturbation: PerturbationSet


_PNorm = Union[Literal["inf"], Literal["l_inf"], Literal["linf"], Literal["L_inf"], Literal["Linf"], float, int]


@dataclass(frozen=True)
class LpBallPerturbation(PerturbationSet):
    p: _PNorm
    eps: float

    @property
    def perturbation_id(self) -> str:
        p = self._normalize_p()
        return f"lp(p={p},eps={float(self.eps)})"

    def _normalize_p(self) -> str:
        p = self.p
        if isinstance(p, str):
            s = p.strip().lower()
            if s in {"inf", "l_inf", "linf"}:
                return "inf"
            raise ValueError(f"unsupported p string for LpBallPerturbation: {self.p!r}")
        if int(p) == p:
            pi = int(p)
            if pi in (1, 2):
                return str(pi)
        if float(p) == float("inf"):
            return "inf"
        raise ValueError(f"unsupported p for LpBallPerturbation: {self.p!r}")

    def bounding_box(self, center: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        eps = float(self.eps)
        # For p in {1,2,inf}: ||δ||_p <= eps => ||δ||_inf <= eps.
        return center - eps, center + eps

    def concretize_matmul(
        self, *, center: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if center.dim() != 2:
            raise ValueError(f"concretize_matmul expects center rank-2 [B,I], got {tuple(center.shape)}")

        p = self._normalize_p()
        eps = float(self.eps)

        if weight.dim() == 2:
            # center: [B,I], weight: [O,I] -> out: [B,O]
            if weight.shape[1] != center.shape[1]:
                raise ValueError(f"matmul shape mismatch: center={tuple(center.shape)} weight={tuple(weight.shape)}")
            out_center = center.matmul(weight.t())
            if p == "inf":
                row_norm = weight.abs().sum(dim=1)  # [O]
            elif p == "2":
                row_norm = torch.linalg.vector_norm(weight, ord=2, dim=1)  # [O]
            elif p == "1":
                row_norm = weight.abs().amax(dim=1)  # [O]
            else:
                raise AssertionError(f"unreachable p: {p}")
            deviation = eps * row_norm.unsqueeze(0)  # [1,O] -> broadcast [B,O]
            lb = out_center - deviation
            ub = out_center + deviation
            if bias is not None:
                b = bias
                if not torch.is_tensor(b):
                    b = torch.as_tensor(b, device=center.device)
                lb = lb + b
                ub = ub + b
            return lb, ub

        if weight.dim() == 3:
            # center: [B,I], weight: [B,O,I] -> out: [B,O]
            if weight.shape[0] != center.shape[0] or weight.shape[2] != center.shape[1]:
                raise ValueError(f"batched matmul mismatch: center={tuple(center.shape)} weight={tuple(weight.shape)}")
            out_center = torch.bmm(center.unsqueeze(1), weight.transpose(1, 2)).squeeze(1)
            if p == "inf":
                row_norm = weight.abs().sum(dim=2)  # [B,O]
            elif p == "2":
                row_norm = torch.linalg.vector_norm(weight, ord=2, dim=2)  # [B,O]
            elif p == "1":
                row_norm = weight.abs().amax(dim=2)  # [B,O]
            else:
                raise AssertionError(f"unreachable p: {p}")
            deviation = eps * row_norm
            lb = out_center - deviation
            ub = out_center + deviation
            if bias is not None:
                b = bias
                if not torch.is_tensor(b):
                    b = torch.as_tensor(b, device=center.device)
                lb = lb + b
                ub = ub + b
            return lb, ub

        raise ValueError(f"concretize_matmul expects weight rank-2/3, got {tuple(weight.shape)}")

    def concretize_affine(
        self, *, center: torch.Tensor, A: torch.Tensor | LinearOperator, b: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if center.dim() < 2:
            raise ValueError(f"concretize_affine expects center rank>=2 [B,*input_shape], got {tuple(center.shape)}")
        op = as_linear_operator(A)
        center_shape = tuple(int(dim) for dim in center.shape[1:])
        if op.shape[0] != center.shape[0] or tuple(op.input_shape) != center_shape:
            raise ValueError(
                f"affine shape mismatch: center={tuple(center.shape)} input_shape={tuple(op.input_shape)} A={tuple(op.shape)}"
            )
        if op.device != center.device:
            raise ValueError(f"affine device mismatch: center={center.device} A={op.device}")
        if op.dtype != center.dtype:
            raise TypeError(f"affine dtype mismatch: center={center.dtype} A={op.dtype}")

        p = self._normalize_p()
        eps = float(self.eps)

        out_center = op.center_term(center)
        if p == "inf":
            row_norm = op.row_abs_sum()
        elif p == "2":
            row_norm = op.row_l2_norm()
        elif p == "1":
            row_norm = op.row_abs_max()
        else:
            raise AssertionError(f"unreachable p: {p}")
        deviation = eps * row_norm
        lb = out_center - deviation
        ub = out_center + deviation
        if b is not None:
            bb = b
            if not torch.is_tensor(bb):
                bb = torch.as_tensor(bb, device=center.device)
            lb = lb + bb
            ub = ub + bb
        return lb, ub
