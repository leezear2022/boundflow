"""Typed RVIR-v4 production optimizer mutation policy contracts."""

# pylint: disable=too-many-boolean-expressions

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from .native_alpha_beta_optimization_state import NativeAlphaBetaOptimizerPolicy
from .rvir_v4_production_state import ProductionOptimizerPolicyV4

PRODUCTION_ITERATION_SEMANTICS_V4 = "evaluate-n-update-n-minus-one/v1"


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ProductionMutationPolicyV4:
    """Admitted lower-only production policy and exact loop cardinalities."""

    production: ProductionOptimizerPolicyV4
    iteration_semantics: str = PRODUCTION_ITERATION_SEMANTICS_V4

    @property
    def evaluation_count(self) -> int:
        """Number of production bound evaluations."""

        return self.production.iteration

    @property
    def update_count(self) -> int:
        """Number of production backward/optimizer updates."""

        return max(self.production.iteration - 1, 0)

    def validate(self) -> None:
        """Reject policy variants outside the preregistered V4-2 subset."""

        self.production.validate()
        if (
            self.iteration_semantics != PRODUCTION_ITERATION_SEMANTICS_V4
            or self.production.iteration <= 0
            or self.production.bound_lower is not True
            or self.production.bound_upper is not False
            or self.production.fix_intermediate_bounds is not True
            or "stop_criterion_batch_any" not in self.production.stop_criterion_id
        ):
            raise ValueError("RVIR-v4 production mutation policy is not admitted")
        if self.evaluation_count != self.update_count + 1:
            raise ValueError("RVIR-v4 production optimizer loop cardinality differs")

    def to_dict(self) -> dict[str, object]:
        """Return the canonical mutation-policy payload."""

        self.validate()
        return {
            "production": self.production.to_dict(),
            "iteration_semantics": self.iteration_semantics,
            "evaluation_count": self.evaluation_count,
            "update_count": self.update_count,
        }

    def stable_hash(self) -> str:
        """Return the canonical policy and loop-semantics identity."""

        return _canonical_hash(self.to_dict())

    def to_native_policy(self) -> NativeAlphaBetaOptimizerPolicy:
        """Map 10 production evaluations to nine native optimizer updates."""

        self.validate()
        policy = NativeAlphaBetaOptimizerPolicy(
            steps=self.update_count,
            lr=self.production.alpha_learning_rate,
            beta_lr=self.production.beta_learning_rate,
            objective="lower",
            spec_reduce="mean",
        )
        policy.validate()
        return policy


__all__ = [
    "PRODUCTION_ITERATION_SEMANTICS_V4",
    "ProductionMutationPolicyV4",
]
