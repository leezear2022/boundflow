"""Typed RVIR-v4 production optimizer mutation policy contracts."""

# pylint: disable=too-many-boolean-expressions,too-many-instance-attributes

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import sys
from typing import Any, Mapping

from .native_alpha_beta_optimization_state import NativeAlphaBetaOptimizerPolicy
from .rvir_v4_production_state import ProductionOptimizerPolicyV4

PRODUCTION_ITERATION_SEMANTICS_V4 = "evaluate-n-update-n-minus-one/v1"


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _callable_id(value: object) -> str:
    module = getattr(value, "__module__", "")
    qualname = getattr(value, "__qualname__", type(value).__name__)
    loaded = sys.modules.get(module)
    if loaded is not None:
        aliases = sorted(
            name
            for name, candidate in vars(loaded).items()
            if candidate is value and not name.startswith("_")
        )
        if aliases:
            qualname = aliases[0]
    return f"{module}.{qualname}"


@dataclass(frozen=True)
class ProductionOptimizerControlsV4:
    """Full optimize-bound controls observed at the production core boundary."""

    optimizer: str
    lr_decay: float
    keep_best: bool
    loss_reduction_id: str
    early_stop_patience: int
    start_save_best: float
    use_float64_in_last_iteration: bool
    pruning_in_iteration: bool
    pruning_in_iteration_threshold: float
    max_time: float
    enable_alpha_crown: bool
    enable_beta_crown: bool
    init_alpha: bool
    use_shared_alpha: bool
    apply_output_constraints_to: tuple[str, ...]
    directly_optimize: tuple[str, ...]
    tighten_input_bounds: bool
    cuts_enabled: bool

    def validate(self) -> None:
        """Validate a lossless, finite production controls snapshot."""

        if (
            not self.optimizer
            or not self.loss_reduction_id
            or self.early_stop_patience < 0
            or not 0.0 <= self.start_save_best <= 1.0
            or self.pruning_in_iteration_threshold < 0.0
            or self.max_time <= 0.0
            or not all(
                math.isfinite(value)
                for value in (
                    self.lr_decay,
                    self.start_save_best,
                    self.pruning_in_iteration_threshold,
                    self.max_time,
                )
            )
            or self.lr_decay <= 0.0
        ):
            raise ValueError("RVIR-v4 production optimizer controls differ")

    def to_dict(self) -> dict[str, object]:
        """Return the canonical production controls payload."""

        self.validate()
        return {
            "optimizer": self.optimizer,
            "lr_decay": self.lr_decay,
            "keep_best": self.keep_best,
            "loss_reduction_id": self.loss_reduction_id,
            "early_stop_patience": self.early_stop_patience,
            "start_save_best": self.start_save_best,
            "use_float64_in_last_iteration": self.use_float64_in_last_iteration,
            "pruning_in_iteration": self.pruning_in_iteration,
            "pruning_in_iteration_threshold": self.pruning_in_iteration_threshold,
            "max_time": self.max_time,
            "enable_alpha_crown": self.enable_alpha_crown,
            "enable_beta_crown": self.enable_beta_crown,
            "init_alpha": self.init_alpha,
            "use_shared_alpha": self.use_shared_alpha,
            "apply_output_constraints_to": list(self.apply_output_constraints_to),
            "directly_optimize": list(self.directly_optimize),
            "tighten_input_bounds": self.tighten_input_bounds,
            "cuts_enabled": self.cuts_enabled,
        }

    def stable_hash(self) -> str:
        """Return the full production optimizer controls identity."""

        return _canonical_hash(self.to_dict())


def capture_production_optimizer_controls_v4(
    optimize_bound_args: Mapping[str, Any], *, cuts_enabled: bool
) -> ProductionOptimizerControlsV4:
    """Capture every V4-2-relevant control from a live BoundedModule."""

    required = {
        "optimizer",
        "lr_decay",
        "keep_best",
        "loss_reduction_func",
        "early_stop_patience",
        "start_save_best",
        "use_float64_in_last_iteration",
        "pruning_in_iteration",
        "pruning_in_iteration_threshold",
        "max_time",
        "enable_alpha_crown",
        "enable_beta_crown",
        "init_alpha",
        "use_shared_alpha",
        "apply_output_constraints_to",
        "directly_optimize",
        "tighten_input_bounds",
    }
    if not required.issubset(optimize_bound_args):
        missing = sorted(required - set(optimize_bound_args))
        raise ValueError(f"RVIR-v4 production optimizer controls missing: {missing}")

    def strings(name: str) -> tuple[str, ...]:
        raw = optimize_bound_args[name]
        if not isinstance(raw, (tuple, list)) or not all(
            isinstance(value, str) for value in raw
        ):
            raise TypeError(f"RVIR-v4 production optimizer {name} differs")
        return tuple(raw)

    controls = ProductionOptimizerControlsV4(
        optimizer=str(optimize_bound_args["optimizer"]),
        lr_decay=float(optimize_bound_args["lr_decay"]),
        keep_best=bool(optimize_bound_args["keep_best"]),
        loss_reduction_id=_callable_id(optimize_bound_args["loss_reduction_func"]),
        early_stop_patience=int(optimize_bound_args["early_stop_patience"]),
        start_save_best=float(optimize_bound_args["start_save_best"]),
        use_float64_in_last_iteration=bool(
            optimize_bound_args["use_float64_in_last_iteration"]
        ),
        pruning_in_iteration=bool(optimize_bound_args["pruning_in_iteration"]),
        pruning_in_iteration_threshold=float(
            optimize_bound_args["pruning_in_iteration_threshold"]
        ),
        max_time=float(optimize_bound_args["max_time"]),
        enable_alpha_crown=bool(optimize_bound_args["enable_alpha_crown"]),
        enable_beta_crown=bool(optimize_bound_args["enable_beta_crown"]),
        init_alpha=bool(optimize_bound_args["init_alpha"]),
        use_shared_alpha=bool(optimize_bound_args["use_shared_alpha"]),
        apply_output_constraints_to=strings("apply_output_constraints_to"),
        directly_optimize=strings("directly_optimize"),
        tighten_input_bounds=bool(optimize_bound_args["tighten_input_bounds"]),
        cuts_enabled=bool(cuts_enabled),
    )
    controls.validate()
    return controls


def production_optimizer_controls_from_payload_v4(
    payload: Mapping[str, object],
) -> ProductionOptimizerControlsV4:
    """Rebuild and validate a canonical controls payload for replay."""

    expected_keys = {
        "optimizer",
        "lr_decay",
        "keep_best",
        "loss_reduction_id",
        "early_stop_patience",
        "start_save_best",
        "use_float64_in_last_iteration",
        "pruning_in_iteration",
        "pruning_in_iteration_threshold",
        "max_time",
        "enable_alpha_crown",
        "enable_beta_crown",
        "init_alpha",
        "use_shared_alpha",
        "apply_output_constraints_to",
        "directly_optimize",
        "tighten_input_bounds",
        "cuts_enabled",
    }
    if set(payload) != expected_keys:
        raise ValueError("RVIR-v4 optimizer controls payload fields differ")
    bool_fields = (
        "keep_best",
        "use_float64_in_last_iteration",
        "pruning_in_iteration",
        "enable_alpha_crown",
        "enable_beta_crown",
        "init_alpha",
        "use_shared_alpha",
        "tighten_input_bounds",
        "cuts_enabled",
    )
    if not all(isinstance(payload[name], bool) for name in bool_fields):
        raise TypeError("RVIR-v4 optimizer controls boolean fields differ")

    def strings(name: str) -> tuple[str, ...]:
        raw = payload[name]
        if not isinstance(raw, list) or not all(
            isinstance(value, str) for value in raw
        ):
            raise TypeError(f"RVIR-v4 optimizer controls {name} payload differs")
        return tuple(raw)

    def number(name: str) -> float:
        raw = payload[name]
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            raise TypeError(f"RVIR-v4 optimizer controls {name} payload differs")
        return float(raw)

    raw_patience = payload["early_stop_patience"]
    if not isinstance(raw_patience, int) or isinstance(raw_patience, bool):
        raise TypeError("RVIR-v4 optimizer controls patience payload differs")

    controls = ProductionOptimizerControlsV4(
        optimizer=str(payload["optimizer"]),
        lr_decay=number("lr_decay"),
        keep_best=payload["keep_best"],  # type: ignore[arg-type]
        loss_reduction_id=str(payload["loss_reduction_id"]),
        early_stop_patience=raw_patience,
        start_save_best=number("start_save_best"),
        use_float64_in_last_iteration=payload[  # type: ignore[arg-type]
            "use_float64_in_last_iteration"
        ],
        pruning_in_iteration=payload["pruning_in_iteration"],  # type: ignore[arg-type]
        pruning_in_iteration_threshold=number("pruning_in_iteration_threshold"),
        max_time=number("max_time"),
        enable_alpha_crown=payload["enable_alpha_crown"],  # type: ignore[arg-type]
        enable_beta_crown=payload["enable_beta_crown"],  # type: ignore[arg-type]
        init_alpha=payload["init_alpha"],  # type: ignore[arg-type]
        use_shared_alpha=payload["use_shared_alpha"],  # type: ignore[arg-type]
        apply_output_constraints_to=strings("apply_output_constraints_to"),
        directly_optimize=strings("directly_optimize"),
        tighten_input_bounds=payload["tighten_input_bounds"],  # type: ignore[arg-type]
        cuts_enabled=payload["cuts_enabled"],  # type: ignore[arg-type]
    )
    controls.validate()
    if controls.to_dict() != dict(payload):
        raise ValueError("RVIR-v4 optimizer controls payload canonicalization differs")
    return controls


@dataclass(frozen=True)
class ProductionMutationPolicyV4:
    """Admitted lower-only production policy and exact loop cardinalities."""

    production: ProductionOptimizerPolicyV4
    controls: ProductionOptimizerControlsV4
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
        self.controls.validate()
        if (
            self.iteration_semantics != PRODUCTION_ITERATION_SEMANTICS_V4
            or self.production.iteration <= 0
            or self.production.bound_lower is not True
            or self.production.bound_upper is not False
            or self.production.fix_intermediate_bounds is not True
            or "stop_criterion_batch_any" not in self.production.stop_criterion_id
            or self.controls.optimizer != "adam"
            or self.controls.keep_best is not True
            or not self.controls.loss_reduction_id.endswith("reduction_sum")
            or self.controls.use_float64_in_last_iteration is not False
            or self.controls.enable_alpha_crown is not True
            or self.controls.enable_beta_crown is not True
            or self.controls.init_alpha is not True
            or self.controls.use_shared_alpha is not False
            or self.controls.apply_output_constraints_to
            or self.controls.directly_optimize
            or self.controls.tighten_input_bounds is not False
            or self.controls.cuts_enabled is not False
        ):
            raise ValueError("RVIR-v4 production mutation policy is not admitted")
        if self.evaluation_count != self.update_count + 1:
            raise ValueError("RVIR-v4 production optimizer loop cardinality differs")

    def to_dict(self) -> dict[str, object]:
        """Return the canonical mutation-policy payload."""

        self.validate()
        return {
            "production": self.production.to_dict(),
            "controls": self.controls.to_dict(),
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
    "ProductionOptimizerControlsV4",
    "capture_production_optimizer_controls_v4",
    "production_optimizer_controls_from_payload_v4",
]
