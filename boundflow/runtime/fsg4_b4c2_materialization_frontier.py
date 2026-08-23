"""Provider ownership for production ReLU lower materialization frontiers."""

# pylint: disable=missing-function-docstring,too-many-boolean-expressions

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import torch

B4C2_RELU_FRONTIERS = ("31", "28", "25", "23", "19", "17")


@dataclass(frozen=True)
class B4C2MaterializationFrontierReceiptV1:
    """Exact activation ledger for ten optimizer evaluations."""

    evaluation_count: int
    materialization_count: int
    per_frontier_count: dict[str, int]
    fallback_count: int
    eager_recompute_count: int
    exact_call: bool
    performance_claimed: bool = False

    def validate(self) -> None:
        if (
            self.evaluation_count != 10
            or self.materialization_count != 60
            or self.per_frontier_count != {name: 10 for name in B4C2_RELU_FRONTIERS}
            or self.fallback_count != 0
            or self.eager_recompute_count != 0
            or self.exact_call is not True
            or self.performance_claimed is not False
        ):
            raise ValueError("B4-C2 materialization frontier receipt differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)


class B4C2MaterializationFrontierObserverV1:
    """Admit all six production lower ReLU materialization sites."""

    def __init__(self) -> None:
        self._evaluation_ordinal = -1
        self._events: list[tuple[int, str]] = []
        self.fallback_count = 0
        self.eager_recompute_count = 0

    def begin_evaluation(
        self,
        evaluation_ordinal: int,
        *,
        native_alphas: Mapping[str, torch.Tensor],
        native_betas: Mapping[str, torch.Tensor],
        relu_pre_add_coeff_l: Mapping[str, torch.Tensor],
    ) -> None:
        del native_alphas, native_betas, relu_pre_add_coeff_l
        if evaluation_ordinal != self._evaluation_ordinal + 1:
            self.fallback_count += 1
            raise ValueError("B4-C2 evaluation order differs")
        self._evaluation_ordinal = evaluation_ordinal

    def wants(self, native_preactivation: str) -> bool:
        del native_preactivation
        return False

    def provider_owns_affine_output(self, native_preactivation: str) -> bool:
        del native_preactivation
        return False

    def provider_owns_relu_lower_materialization(self, x_name: str) -> bool:
        return x_name in B4C2_RELU_FRONTIERS

    def record_relu_lower_materialization(self, x_name: str) -> None:
        expected = B4C2_RELU_FRONTIERS[len(self._events) % len(B4C2_RELU_FRONTIERS)]
        if x_name != expected:
            self.fallback_count += 1
            raise ValueError("B4-C2 materialization frontier order differs")
        self._events.append((self._evaluation_ordinal, x_name))

    def complete_evaluation(self, *, loss_seed: torch.Tensor) -> None:
        if self._evaluation_ordinal != 0 or tuple(loss_seed.shape) != (6, 1):
            self.fallback_count += 1
            raise ValueError("B4-C2 evaluation-zero closure differs")

    def receipt(self) -> B4C2MaterializationFrontierReceiptV1:
        receipt = B4C2MaterializationFrontierReceiptV1(
            evaluation_count=self._evaluation_ordinal + 1,
            materialization_count=len(self._events),
            per_frontier_count={
                name: sum(event_name == name for _, event_name in self._events)
                for name in B4C2_RELU_FRONTIERS
            },
            fallback_count=self.fallback_count,
            eager_recompute_count=self.eager_recompute_count,
            exact_call=True,
        )
        receipt.validate()
        return receipt


__all__ = [
    "B4C2MaterializationFrontierObserverV1",
    "B4C2MaterializationFrontierReceiptV1",
    "B4C2_RELU_FRONTIERS",
]
