"""Explainable calibration models for PR-11 materialization planning."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Iterable, Tuple

import torch

from .materialization import (
    BoundMethod,
    MaterializationAction,
    MaterializationContext,
    MaterializationObservation,
)

COST_MODEL_SCHEMA_VERSION = "boundflow.materialization_cost_model/v1"
FEATURE_NAMES = (
    "intercept",
    "action_bytes_gib",
    "spec_size_over_128",
    "domain_batch_over_32",
)


@dataclass(frozen=True)
class MaterializationCalibrationSample:
    """One context/action measurement used only in the calibration split."""

    context: MaterializationContext
    observation: MaterializationObservation


@dataclass(frozen=True)
class ActionCostModel:
    """Piecewise peak and latency models for one method/action regime."""

    bound_method: BoundMethod
    action: MaterializationAction
    peak_coefficients: Tuple[float, ...]
    latency_coefficients: Tuple[float, ...]
    training_samples: int

    def validate(self) -> None:
        """Validate coefficient dimensions and numeric values."""

        if self.action == MaterializationAction.REDUCE_BATCH:
            raise ValueError("cost models only support dense/structured actions")
        for label, coefficients in (
            ("peak", self.peak_coefficients),
            ("latency", self.latency_coefficients),
        ):
            if len(coefficients) != len(FEATURE_NAMES):
                raise ValueError(
                    f"{label} coefficients must have {len(FEATURE_NAMES)} values"
                )
            if not all(math.isfinite(float(value)) for value in coefficients):
                raise ValueError(f"{label} coefficients must be finite")
        if int(self.training_samples) <= 0:
            raise ValueError("training_samples must be > 0")


@dataclass(frozen=True)
class MaterializationCostModel:
    """Frozen, JSON-serializable models partitioned by method and action."""

    schema_version: str
    ridge: float
    actions: Tuple[ActionCostModel, ...]

    def validate(self) -> None:
        """Validate schema and complete dense/structured pairs per method."""

        if self.schema_version != COST_MODEL_SCHEMA_VERSION:
            raise ValueError(f"unsupported cost model schema: {self.schema_version}")
        if not math.isfinite(float(self.ridge)) or float(self.ridge) < 0.0:
            raise ValueError("ridge must be finite and >= 0")
        keys = {(model.bound_method, model.action) for model in self.actions}
        if len(keys) != len(self.actions):
            raise ValueError("cost model contains duplicate method/action models")
        methods = {model.bound_method for model in self.actions}
        for method in methods:
            method_actions = {
                action for model_method, action in keys if model_method == method
            }
            if method_actions != {
                MaterializationAction.DENSE,
                MaterializationAction.STRUCTURED,
            }:
                raise ValueError(
                    f"cost model requires dense/structured for method={method.value}"
                )
        for model in self.actions:
            model.validate()

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON representation for manifests and cache keys."""

        self.validate()
        return {
            "schema_version": self.schema_version,
            "ridge": float(self.ridge),
            "feature_names": list(FEATURE_NAMES),
            "actions": [
                {
                    **asdict(model),
                    "bound_method": model.bound_method.value,
                    "action": model.action.value,
                    "peak_coefficients": list(model.peak_coefficients),
                    "latency_coefficients": list(model.latency_coefficients),
                }
                for model in sorted(
                    self.actions,
                    key=lambda item: (item.bound_method.value, item.action.value),
                )
            ],
        }

    def predict(self, context: MaterializationContext) -> MaterializationContext:
        """Return a context whose cost summary contains calibrated predictions."""

        self.validate()
        context.validate()
        predictions: dict[MaterializationAction, tuple[int, float]] = {}
        matching = tuple(
            model
            for model in self.actions
            if model.bound_method == context.bound_method
        )
        if len(matching) != 2:
            raise ValueError(
                f"cost model has no complete model for {context.bound_method.value}"
            )
        for model in matching:
            features = _feature_vector(context, model.action)
            peak_gib = _dot(model.peak_coefficients, features)
            latency_log = _dot(model.latency_coefficients, features)
            peak = max(0, int(round(peak_gib * float(1 << 30))))
            latency = max(0.0, math.expm1(min(latency_log, 60.0)))
            predictions[model.action] = (peak, latency)
        dense_peak, dense_latency = predictions[MaterializationAction.DENSE]
        structured_peak, structured_latency = predictions[
            MaterializationAction.STRUCTURED
        ]
        summary = replace(
            context.operator_summary,
            dense_a_bytes=dense_peak,
            structured_base_bytes=structured_peak,
            scale_bytes=0,
            temporary_bytes=0,
            alpha_state_bytes=0,
            beta_state_bytes=0,
            dense_latency_ms=dense_latency,
            structured_latency_ms=structured_latency,
        )
        return replace(context, operator_summary=summary)


def _action_bytes(
    context: MaterializationContext, action: MaterializationAction
) -> int:
    return context.operator_summary.predicted_peak_bytes(action)


def _feature_vector(
    context: MaterializationContext, action: MaterializationAction
) -> Tuple[float, ...]:
    return (
        1.0,
        float(_action_bytes(context, action)) / float(1 << 30),
        float(context.spec_size) / 128.0,
        float(context.domain_batch_size) / 32.0,
    )


def _dot(coefficients: Tuple[float, ...], features: Tuple[float, ...]) -> float:
    return sum(
        float(coefficient) * float(feature)
        for coefficient, feature in zip(coefficients, features)
    )


def _fit_coefficients(
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


def fit_materialization_cost_model(
    samples: Iterable[MaterializationCalibrationSample], *, ridge: float = 1e-6
) -> MaterializationCostModel:
    """Fit separate explainable action models from successful calibration samples."""

    if not math.isfinite(float(ridge)) or float(ridge) < 0.0:
        raise ValueError("ridge must be finite and >= 0")
    grouped: dict[
        tuple[BoundMethod, MaterializationAction],
        list[MaterializationCalibrationSample],
    ] = {}
    for sample in samples:
        sample.context.validate()
        sample.observation.validate()
        if sample.observation.status != "ok":
            continue
        if sample.observation.action == MaterializationAction.REDUCE_BATCH:
            continue
        key = (sample.context.bound_method, sample.observation.action)
        grouped.setdefault(key, []).append(sample)

    models: list[ActionCostModel] = []
    methods = {method for method, _action in grouped}
    if not methods:
        raise ValueError("no successful calibration samples")
    for method in sorted(methods, key=lambda item: item.value):
        for action in (
            MaterializationAction.DENSE,
            MaterializationAction.STRUCTURED,
        ):
            action_samples = grouped.get((method, action), [])
            if not action_samples:
                raise ValueError(
                    "no successful calibration samples for "
                    f"{method.value}/{action.value}"
                )
            features = [
                _feature_vector(sample.context, action) for sample in action_samples
            ]
            peak_targets = [
                float(sample.observation.peak_bytes) / float(1 << 30)
                for sample in action_samples
                if sample.observation.peak_bytes is not None
            ]
            latency_targets = [
                math.log1p(float(sample.observation.latency_ms))
                for sample in action_samples
                if sample.observation.latency_ms is not None
            ]
            if len(peak_targets) != len(features) or len(latency_targets) != len(
                features
            ):
                raise ValueError(
                    "incomplete successful calibration sample for "
                    f"{method.value}/{action.value}"
                )
            models.append(
                ActionCostModel(
                    bound_method=method,
                    action=action,
                    peak_coefficients=_fit_coefficients(
                        features, peak_targets, ridge=float(ridge)
                    ),
                    latency_coefficients=_fit_coefficients(
                        features, latency_targets, ridge=float(ridge)
                    ),
                    training_samples=len(action_samples),
                )
            )
    result = MaterializationCostModel(
        schema_version=COST_MODEL_SCHEMA_VERSION,
        ridge=float(ridge),
        actions=tuple(models),
    )
    result.validate()
    return result


__all__ = [
    "ActionCostModel",
    "COST_MODEL_SCHEMA_VERSION",
    "FEATURE_NAMES",
    "MaterializationCalibrationSample",
    "MaterializationCostModel",
    "fit_materialization_cost_model",
]
