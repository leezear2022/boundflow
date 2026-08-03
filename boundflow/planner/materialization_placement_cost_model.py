"""Profile-guided interaction model for multi-barrier placement."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Mapping, Sequence, Tuple

import torch

from .materialization_static_features import StaticBarrierSummary

PLACEMENT_COST_MODEL_SCHEMA_VERSION = (
    "boundflow.materialization_placement_cost_model/v3"
)
PEAK_FEATURE_NAMES = (
    "intercept",
    "dense_baseline_peak_gib",
    "dense_candidate_gib",
    "structured_candidate_gib",
    "max_structured_candidate_gib",
    "dense_live_weighted_gib",
    "structured_live_weighted_gib",
    "structured_fanout_weighted_gib",
    "structured_merge_weighted_gib",
    "structured_downstream_merge_weighted_gib",
    "structured_reuse_weighted_gib",
    "structured_fraction",
    "structured_fraction_squared",
    "action_transition_fraction",
    "barrier_count",
)
LATENCY_FEATURE_NAMES = (
    "intercept",
    "dense_baseline_latency_ms",
    "dense_candidate_gib",
    "structured_candidate_gib",
    "dense_live_weighted_gib",
    "structured_live_weighted_gib",
    "structured_fanout_weighted_gib",
    "structured_merge_weighted_gib",
    "structured_downstream_merge_weighted_gib",
    "structured_depth_weighted_gib",
    "structured_reuse_weighted_gib",
    "structured_fraction",
    "structured_fraction_squared",
    "structured_position_sum",
    "structured_position_squared_sum",
    "longest_structured_run_fraction",
    "action_transition_fraction",
    "barrier_count",
)


@dataclass(frozen=True)
class PlacementFeatures:  # pylint: disable=too-many-instance-attributes
    """Profile-independent features for one barrier action combination."""

    dense_baseline_peak_bytes: int
    dense_baseline_latency_ms: float
    dense_candidate_bytes: int
    structured_candidate_bytes: int
    max_structured_candidate_bytes: int
    dense_live_weighted_bytes: int
    structured_live_weighted_bytes: int
    structured_fanout_weighted_bytes: int
    structured_merge_weighted_bytes: int
    structured_downstream_merge_weighted_bytes: int
    structured_depth_weighted_bytes: int
    structured_reuse_weighted_bytes: int
    action_transition_count: int
    structured_count: int
    barrier_count: int
    structured_position_sum: float
    structured_position_squared_sum: float
    longest_structured_run: int

    def validate(self) -> None:
        """Validate feature domains and normalized position values."""

        if int(self.dense_baseline_peak_bytes) <= 0:
            raise ValueError("dense_baseline_peak_bytes must be > 0")
        if (
            not math.isfinite(float(self.dense_baseline_latency_ms))
            or float(self.dense_baseline_latency_ms) < 0.0
        ):
            raise ValueError("dense_baseline_latency_ms must be finite and >= 0")
        for name in (
            "dense_candidate_bytes",
            "structured_candidate_bytes",
            "max_structured_candidate_bytes",
            "dense_live_weighted_bytes",
            "structured_live_weighted_bytes",
            "structured_fanout_weighted_bytes",
            "structured_merge_weighted_bytes",
            "structured_downstream_merge_weighted_bytes",
            "structured_depth_weighted_bytes",
            "structured_reuse_weighted_bytes",
            "action_transition_count",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be >= 0")
        if int(self.barrier_count) <= 0:
            raise ValueError("barrier_count must be > 0")
        if not 0 <= int(self.structured_count) <= int(self.barrier_count):
            raise ValueError("structured_count must be in [0, barrier_count]")
        if not 0 <= int(self.longest_structured_run) <= int(self.barrier_count):
            raise ValueError("longest_structured_run must be in [0, barrier_count]")
        if not 0 <= int(self.action_transition_count) < int(self.barrier_count):
            raise ValueError("action_transition_count must be in [0, barrier_count)")
        for name in (
            "structured_position_sum",
            "structured_position_squared_sum",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0")

    @property
    def structured_fraction(self) -> float:
        """Return the fraction of barriers kept structured."""

        return float(self.structured_count) / float(self.barrier_count)

    def peak_vector(self) -> Tuple[float, ...]:
        """Return the stable peak-model feature vector."""

        fraction = self.structured_fraction
        return (
            1.0,
            float(self.dense_baseline_peak_bytes) / float(1 << 30),
            float(self.dense_candidate_bytes) / float(1 << 30),
            float(self.structured_candidate_bytes) / float(1 << 30),
            float(self.max_structured_candidate_bytes) / float(1 << 30),
            float(self.dense_live_weighted_bytes) / float(1 << 30),
            float(self.structured_live_weighted_bytes) / float(1 << 30),
            float(self.structured_fanout_weighted_bytes) / float(1 << 30),
            float(self.structured_merge_weighted_bytes) / float(1 << 30),
            float(self.structured_downstream_merge_weighted_bytes) / float(1 << 30),
            float(self.structured_reuse_weighted_bytes) / float(1 << 30),
            fraction,
            fraction * fraction,
            float(self.action_transition_count) / float(self.barrier_count),
            float(self.barrier_count),
        )

    def latency_vector(self) -> Tuple[float, ...]:
        """Return the stable latency-model feature vector."""

        fraction = self.structured_fraction
        return (
            1.0,
            float(self.dense_baseline_latency_ms),
            float(self.dense_candidate_bytes) / float(1 << 30),
            float(self.structured_candidate_bytes) / float(1 << 30),
            float(self.dense_live_weighted_bytes) / float(1 << 30),
            float(self.structured_live_weighted_bytes) / float(1 << 30),
            float(self.structured_fanout_weighted_bytes) / float(1 << 30),
            float(self.structured_merge_weighted_bytes) / float(1 << 30),
            float(self.structured_downstream_merge_weighted_bytes) / float(1 << 30),
            float(self.structured_depth_weighted_bytes) / float(1 << 30),
            float(self.structured_reuse_weighted_bytes) / float(1 << 30),
            fraction,
            fraction * fraction,
            float(self.structured_position_sum),
            float(self.structured_position_squared_sum),
            float(self.longest_structured_run) / float(self.barrier_count),
            float(self.action_transition_count) / float(self.barrier_count),
            float(self.barrier_count),
        )


@dataclass(frozen=True)
class PlacementCalibrationSample:
    """One successful measured placement used for model fitting."""

    features: PlacementFeatures
    peak_bytes: int
    latency_ms: float

    def validate(self) -> None:
        """Validate measured targets and their feature vector."""

        self.features.validate()
        if int(self.peak_bytes) <= 0:
            raise ValueError("peak_bytes must be > 0")
        if not math.isfinite(float(self.latency_ms)) or float(self.latency_ms) < 0.0:
            raise ValueError("latency_ms must be finite and >= 0")


@dataclass(frozen=True)
class PlacementCostPrediction:
    """Predicted peak and latency for one combination."""

    peak_bytes: int
    latency_ms: float


@dataclass(frozen=True)
class PlacementInteractionCostModel:
    """Ridge-linear peak/latency interaction model."""

    schema_version: str
    ridge: float
    peak_coefficients: Tuple[float, ...]
    latency_coefficients: Tuple[float, ...]
    training_samples: int

    def validate(self) -> None:
        """Validate schema, coefficient dimensions, and finite values."""

        if self.schema_version != PLACEMENT_COST_MODEL_SCHEMA_VERSION:
            raise ValueError(f"unsupported placement cost model: {self.schema_version}")
        if not math.isfinite(float(self.ridge)) or float(self.ridge) < 0.0:
            raise ValueError("ridge must be finite and >= 0")
        if len(self.peak_coefficients) != len(PEAK_FEATURE_NAMES):
            raise ValueError("peak coefficient dimension mismatch")
        if len(self.latency_coefficients) != len(LATENCY_FEATURE_NAMES):
            raise ValueError("latency coefficient dimension mismatch")
        if not all(
            math.isfinite(float(value))
            for value in self.peak_coefficients + self.latency_coefficients
        ):
            raise ValueError("placement cost coefficients must be finite")
        if int(self.training_samples) <= 0:
            raise ValueError("training_samples must be > 0")

    def predict(self, features: PlacementFeatures) -> PlacementCostPrediction:
        """Predict non-negative peak bytes and latency milliseconds."""

        self.validate()
        features.validate()
        peak_gib = _dot(self.peak_coefficients, features.peak_vector())
        latency = _dot(self.latency_coefficients, features.latency_vector())
        return PlacementCostPrediction(
            peak_bytes=max(0, int(round(peak_gib * float(1 << 30)))),
            latency_ms=max(0.0, float(latency)),
        )

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic JSON representation."""

        self.validate()
        return {
            **asdict(self),
            "peak_feature_names": list(PEAK_FEATURE_NAMES),
            "latency_feature_names": list(LATENCY_FEATURE_NAMES),
            "peak_coefficients": list(self.peak_coefficients),
            "latency_coefficients": list(self.latency_coefficients),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, object]
    ) -> "PlacementInteractionCostModel":
        """Load a model while checking its frozen feature-name contract."""

        peak_names = payload.get("peak_feature_names")
        latency_names = payload.get("latency_feature_names")
        peak_values = payload.get("peak_coefficients")
        latency_values = payload.get("latency_coefficients")
        ridge = payload.get("ridge")
        training_samples = payload.get("training_samples")
        if not isinstance(peak_names, (list, tuple)):
            raise ValueError("peak feature names must be a sequence")
        if not isinstance(latency_names, (list, tuple)):
            raise ValueError("latency feature names must be a sequence")
        if tuple(peak_names) != PEAK_FEATURE_NAMES:
            raise ValueError("peak feature names do not match the frozen model")
        if tuple(latency_names) != LATENCY_FEATURE_NAMES:
            raise ValueError("latency feature names do not match the frozen model")
        if not isinstance(peak_values, (list, tuple)) or not all(
            isinstance(value, (int, float)) for value in peak_values
        ):
            raise ValueError("peak coefficients must be a numeric sequence")
        if not isinstance(latency_values, (list, tuple)) or not all(
            isinstance(value, (int, float)) for value in latency_values
        ):
            raise ValueError("latency coefficients must be a numeric sequence")
        if not isinstance(ridge, (int, float)):
            raise ValueError("ridge must be numeric")
        if not isinstance(training_samples, int):
            raise ValueError("training_samples must be an integer")
        model = cls(
            schema_version=str(payload["schema_version"]),
            ridge=float(ridge),
            peak_coefficients=tuple(float(value) for value in peak_values),
            latency_coefficients=tuple(float(value) for value in latency_values),
            training_samples=training_samples,
        )
        model.validate()
        return model


def placement_features_from_static(  # pylint: disable=too-many-locals
    barriers: Sequence[StaticBarrierSummary],
    actions: Sequence[str],
    *,
    dense_baseline_peak_bytes: int,
    dense_baseline_latency_ms: float,
) -> PlacementFeatures:
    """Aggregate candidate-independent barrier metadata for one action pattern."""

    materialized_barriers = tuple(barriers)
    materialized_actions = tuple(str(action) for action in actions)
    if not materialized_barriers:
        raise ValueError("static placement features require at least one barrier")
    if len(materialized_actions) != len(materialized_barriers):
        raise ValueError("placement action count does not match static barriers")
    if set(materialized_actions) - {"dense", "structured"}:
        raise ValueError("static placement actions must be dense or structured")
    for barrier in materialized_barriers:
        barrier.validate()
    barrier_count = len(materialized_actions)
    structured_positions = [
        float(index + 1) / float(barrier_count)
        for index, action in enumerate(materialized_actions)
        if action == "structured"
    ]
    longest_run = 0
    current_run = 0
    for action in materialized_actions:
        if action == "structured":
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 0
    features = PlacementFeatures(
        dense_baseline_peak_bytes=int(dense_baseline_peak_bytes),
        dense_baseline_latency_ms=float(dense_baseline_latency_ms),
        dense_candidate_bytes=sum(
            barrier.coefficient_bytes
            for action, barrier in zip(materialized_actions, materialized_barriers)
            if action == "dense"
        ),
        structured_candidate_bytes=sum(
            barrier.coefficient_bytes
            for action, barrier in zip(materialized_actions, materialized_barriers)
            if action == "structured"
        ),
        max_structured_candidate_bytes=max(
            (
                barrier.coefficient_bytes
                for action, barrier in zip(materialized_actions, materialized_barriers)
                if action == "structured"
            ),
            default=0,
        ),
        dense_live_weighted_bytes=sum(
            barrier.coefficient_bytes * max(1, barrier.direct_live_span)
            for action, barrier in zip(materialized_actions, materialized_barriers)
            if action == "dense"
        ),
        structured_live_weighted_bytes=sum(
            barrier.coefficient_bytes * max(1, barrier.direct_live_span)
            for action, barrier in zip(materialized_actions, materialized_barriers)
            if action == "structured"
        ),
        structured_fanout_weighted_bytes=sum(
            barrier.coefficient_bytes * max(0, barrier.direct_consumer_count - 1)
            for action, barrier in zip(materialized_actions, materialized_barriers)
            if action == "structured"
        ),
        structured_merge_weighted_bytes=sum(
            barrier.coefficient_bytes
            for action, barrier in zip(materialized_actions, materialized_barriers)
            if action == "structured" and barrier.is_merge_output
        ),
        structured_downstream_merge_weighted_bytes=sum(
            barrier.coefficient_bytes * barrier.downstream_merge_count
            for action, barrier in zip(materialized_actions, materialized_barriers)
            if action == "structured"
        ),
        structured_depth_weighted_bytes=sum(
            barrier.coefficient_bytes * barrier.downstream_depth
            for action, barrier in zip(materialized_actions, materialized_barriers)
            if action == "structured"
        ),
        structured_reuse_weighted_bytes=sum(
            barrier.coefficient_bytes * max(0, barrier.reuse_count - 1)
            for action, barrier in zip(materialized_actions, materialized_barriers)
            if action == "structured"
        ),
        action_transition_count=sum(
            lhs != rhs
            for lhs, rhs in zip(materialized_actions, materialized_actions[1:])
        ),
        structured_count=materialized_actions.count("structured"),
        barrier_count=barrier_count,
        structured_position_sum=sum(structured_positions),
        structured_position_squared_sum=sum(
            position * position for position in structured_positions
        ),
        longest_structured_run=longest_run,
    )
    features.validate()
    return features


def _dot(coefficients: Tuple[float, ...], features: Tuple[float, ...]) -> float:
    return sum(
        float(coefficient) * float(feature)
        for coefficient, feature in zip(coefficients, features)
    )


def _fit(
    features: list[Tuple[float, ...]], targets: list[float], *, ridge: float
) -> Tuple[float, ...]:
    design = torch.tensor(features, dtype=torch.float64)
    response = torch.tensor(targets, dtype=torch.float64)
    identity = torch.eye(int(design.shape[1]), dtype=torch.float64)
    identity[0, 0] = 0.0
    gram = design.T @ design + float(ridge) * identity
    rhs = design.T @ response
    try:
        coefficients = torch.linalg.solve(gram, rhs)  # pylint: disable=not-callable
    except torch.linalg.LinAlgError:
        coefficients = torch.linalg.pinv(gram) @ rhs  # pylint: disable=not-callable
    return tuple(float(value) for value in coefficients.tolist())


def fit_placement_interaction_cost_model(
    samples: Iterable[PlacementCalibrationSample], *, ridge: float = 1e-3
) -> PlacementInteractionCostModel:
    """Fit the frozen v1 interaction model from calibration-only samples."""

    materialized = list(samples)
    if not materialized:
        raise ValueError("placement calibration samples must be non-empty")
    for sample in materialized:
        sample.validate()
    model = PlacementInteractionCostModel(
        schema_version=PLACEMENT_COST_MODEL_SCHEMA_VERSION,
        ridge=float(ridge),
        peak_coefficients=_fit(
            [sample.features.peak_vector() for sample in materialized],
            [float(sample.peak_bytes) / float(1 << 30) for sample in materialized],
            ridge=float(ridge),
        ),
        latency_coefficients=_fit(
            [sample.features.latency_vector() for sample in materialized],
            [float(sample.latency_ms) for sample in materialized],
            ridge=float(ridge),
        ),
        training_samples=len(materialized),
    )
    model.validate()
    return model


__all__ = [
    "LATENCY_FEATURE_NAMES",
    "PEAK_FEATURE_NAMES",
    "PLACEMENT_COST_MODEL_SCHEMA_VERSION",
    "PlacementCalibrationSample",
    "PlacementCostPrediction",
    "PlacementFeatures",
    "PlacementInteractionCostModel",
    "fit_placement_interaction_cost_model",
    "placement_features_from_static",
]
