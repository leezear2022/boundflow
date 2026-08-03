"""Calibration-only backend selection for fused plain-CROWN regions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable, Mapping, Sequence, Tuple

from .execution_candidate import BackendVariant, OperatorFamily


@dataclass(frozen=True)
class FusedCrownBackendObservation:  # pylint: disable=too-many-instance-attributes
    """One correct calibration measurement used by the PR-12 backend planner."""

    case_id: str
    family: OperatorFamily
    backend: BackendVariant
    boundary_bytes: int
    region_count: int
    warm_latency_ms: float
    peak_allocated_bytes: int
    eligible: bool
    correct: bool

    def validate(self) -> None:
        """Reject incomplete or invalid measurements before model fitting."""

        if not self.case_id:
            raise ValueError("case_id must be non-empty")
        if self.boundary_bytes <= 0 or self.region_count <= 0:
            raise ValueError("boundary_bytes and region_count must be positive")
        if not math.isfinite(self.warm_latency_ms) or self.warm_latency_ms <= 0:
            raise ValueError("warm_latency_ms must be finite and positive")
        if self.peak_allocated_bytes < 0:
            raise ValueError("peak_allocated_bytes must be non-negative")


@dataclass(frozen=True)
class FusedCrownBackendDecision:  # pylint: disable=too-many-instance-attributes
    """Explainable selection made without consulting held-out measurements."""

    backend: BackendVariant
    reason: str
    calibration_case_id: str | None
    predicted_fused_over_eager: float | None
    predicted_eager_peak_bytes: int | None
    predicted_fused_peak_bytes: int | None
    budget_bytes: int
    eligible: bool

    @property
    def use_fused(self) -> bool:
        """Return whether the selected runtime should execute fused TIR."""

        return self.backend == BackendVariant.TVM_FUSED_TIR

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible decision dump."""

        payload = asdict(self)
        payload["backend"] = self.backend.value
        payload["use_fused"] = self.use_fused
        return payload


@dataclass(frozen=True)
class _CalibrationPair:  # pylint: disable=too-many-instance-attributes
    case_id: str
    family: OperatorFamily
    boundary_bytes: int
    region_count: int
    eager_latency_ms: float
    fused_latency_ms: float
    eager_peak_bytes: int
    fused_peak_bytes: int

    @property
    def fused_over_eager(self) -> float:
        """Return the measured warm-latency ratio for this pair."""

        return self.fused_latency_ms / self.eager_latency_ms


class FusedCrownBackendPlanner:
    """Nearest-scale planner fitted exclusively from frozen calibration rows.

    The v1 model intentionally stays explainable: it chooses the same-family
    calibration pair nearest in log bytes-per-region, scales the two measured
    peaks linearly with boundary bytes, then selects the faster feasible backend.
    Eligibility always takes precedence over performance predictions.
    """

    def __init__(self, pairs: Tuple[_CalibrationPair, ...]) -> None:
        if not pairs:
            raise ValueError("at least one paired calibration case is required")
        self._pairs = pairs

    @classmethod
    def fit(
        cls, observations: Iterable[FusedCrownBackendObservation]
    ) -> "FusedCrownBackendPlanner":
        """Pair eager/fused correct observations by case without data leakage."""

        grouped: dict[
            tuple[str, OperatorFamily],
            dict[BackendVariant, FusedCrownBackendObservation],
        ] = {}
        for observation in observations:
            observation.validate()
            if not observation.correct or not observation.eligible:
                continue
            if observation.backend not in {
                BackendVariant.PYTORCH_EAGER,
                BackendVariant.TVM_FUSED_TIR,
            }:
                continue
            grouped.setdefault((observation.case_id, observation.family), {})[
                observation.backend
            ] = observation

        pairs: list[_CalibrationPair] = []
        for (case_id, family), candidates in sorted(
            grouped.items(), key=lambda item: (item[0][1].value, item[0][0])
        ):
            eager = candidates.get(BackendVariant.PYTORCH_EAGER)
            fused = candidates.get(BackendVariant.TVM_FUSED_TIR)
            if eager is None or fused is None:
                continue
            if (
                eager.boundary_bytes != fused.boundary_bytes
                or eager.region_count != fused.region_count
            ):
                raise ValueError(f"calibration candidate metadata mismatch: {case_id}")
            pairs.append(
                _CalibrationPair(
                    case_id=case_id,
                    family=family,
                    boundary_bytes=eager.boundary_bytes,
                    region_count=eager.region_count,
                    eager_latency_ms=eager.warm_latency_ms,
                    fused_latency_ms=fused.warm_latency_ms,
                    eager_peak_bytes=eager.peak_allocated_bytes,
                    fused_peak_bytes=fused.peak_allocated_bytes,
                )
            )
        return cls(tuple(pairs))

    def decide(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        *,
        family: OperatorFamily,
        boundary_bytes: int,
        region_count: int,
        budget_bytes: int,
        eligible: bool,
    ) -> FusedCrownBackendDecision:
        """Select eager or fused execution using calibration data only."""

        if boundary_bytes <= 0 or region_count <= 0 or budget_bytes <= 0:
            raise ValueError(
                "boundary_bytes, region_count and budget_bytes must be positive"
            )
        if not eligible:
            return FusedCrownBackendDecision(
                backend=BackendVariant.PYTORCH_EAGER,
                reason="capability_or_graph_ineligible_fallback",
                calibration_case_id=None,
                predicted_fused_over_eager=None,
                predicted_eager_peak_bytes=None,
                predicted_fused_peak_bytes=None,
                budget_bytes=budget_bytes,
                eligible=False,
            )
        compatible = tuple(pair for pair in self._pairs if pair.family == family)
        if not compatible:
            return FusedCrownBackendDecision(
                backend=BackendVariant.PYTORCH_EAGER,
                reason="no_same_family_calibration_fallback",
                calibration_case_id=None,
                predicted_fused_over_eager=None,
                predicted_eager_peak_bytes=None,
                predicted_fused_peak_bytes=None,
                budget_bytes=budget_bytes,
                eligible=True,
            )

        query_scale = boundary_bytes / region_count
        nearest = min(
            compatible,
            key=lambda pair: abs(
                math.log(query_scale)
                - math.log(pair.boundary_bytes / pair.region_count)
            ),
        )
        scale = boundary_bytes / nearest.boundary_bytes
        eager_peak = int(math.ceil(nearest.eager_peak_bytes * scale))
        fused_peak = int(math.ceil(nearest.fused_peak_bytes * scale))
        eager_feasible = eager_peak <= budget_bytes
        fused_feasible = fused_peak <= budget_bytes
        ratio = nearest.fused_over_eager

        if fused_feasible and not eager_feasible:
            backend = BackendVariant.TVM_FUSED_TIR
            reason = "fused_only_budget_feasible"
        elif eager_feasible and not fused_feasible:
            backend = BackendVariant.PYTORCH_EAGER
            reason = "eager_only_budget_feasible"
        elif eager_feasible and fused_feasible and ratio < 1.0:
            backend = BackendVariant.TVM_FUSED_TIR
            reason = "calibration_predicts_fused_faster"
        elif eager_feasible and fused_feasible:
            backend = BackendVariant.PYTORCH_EAGER
            reason = "calibration_predicts_eager_faster"
        elif fused_peak < eager_peak:
            backend = BackendVariant.TVM_FUSED_TIR
            reason = "budget_violated_choose_lower_predicted_peak"
        else:
            backend = BackendVariant.PYTORCH_EAGER
            reason = "budget_violated_choose_lower_predicted_peak"

        return FusedCrownBackendDecision(
            backend=backend,
            reason=reason,
            calibration_case_id=nearest.case_id,
            predicted_fused_over_eager=ratio,
            predicted_eager_peak_bytes=eager_peak,
            predicted_fused_peak_bytes=fused_peak,
            budget_bytes=budget_bytes,
            eligible=True,
        )

    def to_dict(self) -> Mapping[str, object]:
        """Dump the frozen calibration pairs used by the planner."""

        return {
            "model": "same_family_nearest_log_bytes_per_region_v1",
            "pairs": [
                {
                    **asdict(pair),
                    "family": pair.family.value,
                    "fused_over_eager": pair.fused_over_eager,
                }
                for pair in self._pairs
            ],
        }


@dataclass(frozen=True)
class FusedCrownMultiBackendDecision:  # pylint: disable=too-many-instance-attributes
    """V2 decision across eager, chunked eager, and fused TIR candidates."""

    backend: BackendVariant
    reason: str
    calibration_case_id: str
    predicted_latency_ratio: float
    predicted_peak_bytes: int
    budget_bytes: int
    eligible_backends: Tuple[BackendVariant, ...]
    predictions: Mapping[str, Mapping[str, object]]

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible decision dump."""

        return {
            **asdict(self),
            "backend": self.backend.value,
            "eligible_backends": [backend.value for backend in self.eligible_backends],
            "predictions": dict(self.predictions),
        }


@dataclass(frozen=True)
class _BackendPair:
    case_id: str
    family: OperatorFamily
    backend: BackendVariant
    boundary_bytes: int
    region_count: int
    latency_ratio: float
    peak_bytes: int


class FusedCrownMultiBackendPlanner:
    """Nearest-scale v2 planner over all measured runtime candidates."""

    def __init__(self, pairs: Tuple[_BackendPair, ...]) -> None:
        if not pairs:
            raise ValueError("at least one multi-backend calibration pair is required")
        self._pairs = pairs

    @classmethod
    def fit(
        cls, observations: Iterable[FusedCrownBackendObservation]
    ) -> "FusedCrownMultiBackendPlanner":
        """Pair every correct candidate with eager from the same calibration case."""

        grouped: dict[
            tuple[str, OperatorFamily],
            dict[BackendVariant, FusedCrownBackendObservation],
        ] = {}
        for observation in observations:
            observation.validate()
            if observation.correct and observation.eligible:
                grouped.setdefault((observation.case_id, observation.family), {})[
                    observation.backend
                ] = observation
        pairs: list[_BackendPair] = []
        for (case_id, family), candidates in sorted(
            grouped.items(), key=lambda item: (item[0][1].value, item[0][0])
        ):
            eager = candidates.get(BackendVariant.PYTORCH_EAGER)
            if eager is None:
                continue
            for backend, candidate in sorted(
                candidates.items(), key=lambda item: item[0].value
            ):
                if (
                    eager.boundary_bytes != candidate.boundary_bytes
                    or eager.region_count != candidate.region_count
                ):
                    raise ValueError(
                        f"calibration candidate metadata mismatch: {case_id}"
                    )
                pairs.append(
                    _BackendPair(
                        case_id=case_id,
                        family=family,
                        backend=backend,
                        boundary_bytes=eager.boundary_bytes,
                        region_count=eager.region_count,
                        latency_ratio=(
                            candidate.warm_latency_ms / eager.warm_latency_ms
                        ),
                        peak_bytes=candidate.peak_allocated_bytes,
                    )
                )
        available = {pair.backend for pair in pairs}
        if BackendVariant.PYTORCH_EAGER not in available or len(available) < 2:
            raise ValueError(
                "multi-backend calibration requires eager and another backend"
            )
        return cls(tuple(pairs))

    def decide(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        *,
        family: OperatorFamily,
        boundary_bytes: int,
        region_count: int,
        budget_bytes: int,
        eligible_backends: Sequence[BackendVariant],
    ) -> FusedCrownMultiBackendDecision:
        """Choose the predicted-fastest budget-feasible eligible backend."""

        if min(boundary_bytes, region_count, budget_bytes) <= 0:
            raise ValueError(
                "boundary_bytes, region_count and budget_bytes must be positive"
            )
        eligible = tuple(dict.fromkeys(eligible_backends))
        if BackendVariant.PYTORCH_EAGER not in eligible:
            raise ValueError("pytorch_eager must remain an eligible fallback")
        query_scale = boundary_bytes / region_count
        predictions: dict[BackendVariant, tuple[_BackendPair, int]] = {}
        for backend in eligible:
            compatible = tuple(
                pair
                for pair in self._pairs
                if pair.family == family and pair.backend == backend
            )
            if not compatible:
                continue
            nearest = min(
                compatible,
                key=lambda pair: abs(
                    math.log(query_scale)
                    - math.log(pair.boundary_bytes / pair.region_count)
                ),
            )
            predicted_peak = int(
                math.ceil(nearest.peak_bytes * boundary_bytes / nearest.boundary_bytes)
            )
            predictions[backend] = nearest, predicted_peak
        if BackendVariant.PYTORCH_EAGER not in predictions:
            raise ValueError(f"no eager calibration for family {family.value}")
        feasible = {
            backend: prediction
            for backend, prediction in predictions.items()
            if prediction[1] <= budget_bytes
        }
        pool = feasible or predictions
        selection_key = (
            (lambda item: (item[1][0].latency_ratio, item[1][1], item[0].value))
            if feasible
            else (lambda item: (item[1][1], item[1][0].latency_ratio, item[0].value))
        )
        selected_backend, (selected_pair, selected_peak) = min(
            pool.items(), key=selection_key
        )
        reason = (
            "predicted_fastest_budget_feasible"
            if feasible
            else "budget_violated_predicted_fastest_lower_peak_tiebreak"
        )
        payload = {
            backend.value: {
                "calibration_case_id": pair.case_id,
                "predicted_latency_ratio": pair.latency_ratio,
                "predicted_peak_bytes": peak,
                "budget_feasible": peak <= budget_bytes,
            }
            for backend, (pair, peak) in predictions.items()
        }
        return FusedCrownMultiBackendDecision(
            backend=selected_backend,
            reason=reason,
            calibration_case_id=selected_pair.case_id,
            predicted_latency_ratio=selected_pair.latency_ratio,
            predicted_peak_bytes=selected_peak,
            budget_bytes=budget_bytes,
            eligible_backends=eligible,
            predictions=payload,
        )

    def to_dict(self) -> Mapping[str, object]:
        """Dump all candidate ratios fitted without held-out observations."""

        return {
            "model": "same_family_nearest_log_bytes_per_region_multibackend_v2",
            "pairs": [
                {
                    **asdict(pair),
                    "family": pair.family.value,
                    "backend": pair.backend.value,
                }
                for pair in self._pairs
            ],
        }


@dataclass(frozen=True)
class CompileAwareBackendObservation:  # pylint: disable=too-many-instance-attributes
    """One calibration-only E2E observation for compile-aware selection."""

    case_id: str
    family: OperatorFamily
    backend: BackendVariant
    boundary_bytes: int
    region_count: int
    warm_latency_ms: float
    peak_allocated_bytes: int
    fresh_setup_ms: float
    disk_setup_ms: float | None
    eligible: bool
    correct: bool

    def validate(self) -> None:
        """Reject invalid timing, memory, or identity fields before fitting."""

        if not self.case_id:
            raise ValueError("case_id must be non-empty")
        if min(self.boundary_bytes, self.region_count) <= 0:
            raise ValueError("boundary_bytes and region_count must be positive")
        if not math.isfinite(self.warm_latency_ms) or self.warm_latency_ms <= 0:
            raise ValueError("warm_latency_ms must be finite and positive")
        if self.peak_allocated_bytes < 0 or self.fresh_setup_ms < 0:
            raise ValueError("peak and setup values must be non-negative")
        if self.disk_setup_ms is not None and self.disk_setup_ms < 0:
            raise ValueError("disk_setup_ms must be non-negative when present")


@dataclass(frozen=True)
class CompileAwareQueryPolicy:
    """Expected repeated-query/cache regime supplied to the Planner."""

    expected_reuse_queries: int
    memory_cache_hit_probability: float
    disk_cache_hit_probability: float

    def validate(self) -> None:
        """Validate the probability simplex and repeated-query count."""

        if self.expected_reuse_queries <= 0:
            raise ValueError("expected_reuse_queries must be positive")
        probabilities = (
            self.memory_cache_hit_probability,
            self.disk_cache_hit_probability,
        )
        if any(not math.isfinite(value) or value < 0 for value in probabilities):
            raise ValueError("cache probabilities must be finite and non-negative")
        if sum(probabilities) > 1.0 + 1e-12:
            raise ValueError("cache probabilities must sum to at most one")

    @property
    def fresh_probability(self) -> float:
        """Return the residual probability of a fresh compilation path."""

        return max(
            0.0,
            1.0 - self.memory_cache_hit_probability - self.disk_cache_hit_probability,
        )


@dataclass(frozen=True)
class CompileAwareBackendDecision:  # pylint: disable=too-many-instance-attributes
    """Explainable capability→budget→risk→amortized-latency decision."""

    backend: BackendVariant
    reason: str
    budget_bytes: int
    selected_budget_feasible: bool
    selected_risk_tier: int
    selected_amortized_latency_ratio: float
    policy: CompileAwareQueryPolicy
    eligible_backends: Tuple[BackendVariant, ...]
    predictions: Mapping[str, Mapping[str, object]]

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible decision dump."""

        return {
            "backend": self.backend.value,
            "reason": self.reason,
            "budget_bytes": self.budget_bytes,
            "selected_budget_feasible": self.selected_budget_feasible,
            "selected_risk_tier": self.selected_risk_tier,
            "selected_amortized_latency_ratio": (self.selected_amortized_latency_ratio),
            "policy": asdict(self.policy),
            "eligible_backends": [backend.value for backend in self.eligible_backends],
            "predictions": dict(self.predictions),
        }


@dataclass(frozen=True)
class _CompileAwarePair:  # pylint: disable=too-many-instance-attributes
    case_id: str
    family: OperatorFamily
    backend: BackendVariant
    boundary_bytes: int
    region_count: int
    warm_latency_ratio: float
    eager_warm_latency_ms: float
    peak_allocated_bytes: int
    fresh_setup_ms: float
    disk_setup_ms: float | None


class CompileAwareFusedCrownPlanner:
    """Calibration-only repeated-query Planner with explicit setup/cache cost."""

    def __init__(self, pairs: Tuple[_CompileAwarePair, ...]) -> None:
        if not pairs:
            raise ValueError("at least one compile-aware calibration pair is required")
        self._pairs = pairs

    @classmethod
    def fit(
        cls, observations: Iterable[CompileAwareBackendObservation]
    ) -> "CompileAwareFusedCrownPlanner":
        """Pair correct candidates with eager without consulting held-out rows."""

        grouped: dict[
            tuple[str, OperatorFamily],
            dict[BackendVariant, CompileAwareBackendObservation],
        ] = {}
        for observation in observations:
            observation.validate()
            if observation.correct and observation.eligible:
                grouped.setdefault((observation.case_id, observation.family), {})[
                    observation.backend
                ] = observation
        pairs: list[_CompileAwarePair] = []
        for (case_id, family), candidates in sorted(
            grouped.items(), key=lambda item: (item[0][1].value, item[0][0])
        ):
            eager = candidates.get(BackendVariant.PYTORCH_EAGER)
            if eager is None:
                continue
            for backend, candidate in sorted(
                candidates.items(), key=lambda item: item[0].value
            ):
                if (
                    candidate.boundary_bytes != eager.boundary_bytes
                    or candidate.region_count != eager.region_count
                ):
                    raise ValueError(f"calibration metadata mismatch: {case_id}")
                pairs.append(
                    _CompileAwarePair(
                        case_id=case_id,
                        family=family,
                        backend=backend,
                        boundary_bytes=candidate.boundary_bytes,
                        region_count=candidate.region_count,
                        warm_latency_ratio=(
                            candidate.warm_latency_ms / eager.warm_latency_ms
                        ),
                        eager_warm_latency_ms=eager.warm_latency_ms,
                        peak_allocated_bytes=candidate.peak_allocated_bytes,
                        fresh_setup_ms=candidate.fresh_setup_ms,
                        disk_setup_ms=candidate.disk_setup_ms,
                    )
                )
        available = {pair.backend for pair in pairs}
        if BackendVariant.PYTORCH_EAGER not in available or len(available) < 2:
            raise ValueError("compile-aware fitting requires eager and another backend")
        return cls(tuple(pairs))

    def decide(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        *,
        family: OperatorFamily,
        boundary_bytes: int,
        region_count: int,
        budget_bytes: int,
        eligible_backends: Sequence[BackendVariant],
        policy: CompileAwareQueryPolicy,
    ) -> CompileAwareBackendDecision:
        """Apply capability→budget→risk→amortized-latency lexicographically."""

        if min(boundary_bytes, region_count, budget_bytes) <= 0:
            raise ValueError(
                "boundary_bytes, region_count and budget_bytes must be positive"
            )
        policy.validate()
        eligible = tuple(dict.fromkeys(eligible_backends))
        if BackendVariant.PYTORCH_EAGER not in eligible:
            raise ValueError("pytorch_eager must remain an eligible fallback")
        query_scale = boundary_bytes / region_count
        predictions: dict[BackendVariant, dict[str, Any]] = {}
        for backend in eligible:
            compatible = tuple(
                pair
                for pair in self._pairs
                if pair.family == family and pair.backend == backend
            )
            if not compatible:
                continue
            nearest = min(
                compatible,
                key=lambda pair: abs(
                    math.log(query_scale)
                    - math.log(pair.boundary_bytes / pair.region_count)
                ),
            )
            distance = abs(
                math.log(query_scale)
                - math.log(nearest.boundary_bytes / nearest.region_count)
            )
            predicted_peak = int(
                math.ceil(
                    nearest.peak_allocated_bytes
                    * boundary_bytes
                    / nearest.boundary_bytes
                )
            )
            disk_fallback = (
                policy.disk_cache_hit_probability > 0 and nearest.disk_setup_ms is None
            )
            disk_setup = (
                nearest.fresh_setup_ms
                if nearest.disk_setup_ms is None
                else nearest.disk_setup_ms
            )
            expected_setup = (
                policy.fresh_probability * nearest.fresh_setup_ms
                + policy.disk_cache_hit_probability * disk_setup
            )
            amortized_ratio = nearest.warm_latency_ratio + expected_setup / (
                policy.expected_reuse_queries * nearest.eager_warm_latency_ms
            )
            risk_tier = int(distance > math.log(4.0)) + int(disk_fallback)
            predictions[backend] = {
                "calibration_case_id": nearest.case_id,
                "predicted_peak_bytes": predicted_peak,
                "budget_feasible": predicted_peak <= budget_bytes,
                "warm_latency_ratio": nearest.warm_latency_ratio,
                "fresh_setup_ms": nearest.fresh_setup_ms,
                "disk_setup_ms": nearest.disk_setup_ms,
                "disk_cache_falls_back_to_fresh": disk_fallback,
                "expected_setup_ms": expected_setup,
                "amortized_latency_ratio": amortized_ratio,
                "log_scale_distance": distance,
                "risk_tier": risk_tier,
            }
        if BackendVariant.PYTORCH_EAGER not in predictions:
            raise ValueError(f"no eager calibration for family {family.value}")
        feasible = {
            backend: prediction
            for backend, prediction in predictions.items()
            if bool(prediction["budget_feasible"])
        }
        if feasible:
            selected_backend, selected = min(
                feasible.items(),
                key=lambda item: (
                    int(item[1]["risk_tier"]),
                    float(item[1]["amortized_latency_ratio"]),
                    int(item[1]["predicted_peak_bytes"]),
                    item[0].value,
                ),
            )
            reason = "lowest_risk_amortized_latency_within_budget"
        else:
            selected_backend, selected = min(
                predictions.items(),
                key=lambda item: (
                    int(item[1]["predicted_peak_bytes"]),
                    int(item[1]["risk_tier"]),
                    float(item[1]["amortized_latency_ratio"]),
                    item[0].value,
                ),
            )
            reason = "no_budget_feasible_candidate_choose_lowest_predicted_peak"
        return CompileAwareBackendDecision(
            backend=selected_backend,
            reason=reason,
            budget_bytes=budget_bytes,
            selected_budget_feasible=bool(selected["budget_feasible"]),
            selected_risk_tier=int(selected["risk_tier"]),
            selected_amortized_latency_ratio=float(selected["amortized_latency_ratio"]),
            policy=policy,
            eligible_backends=eligible,
            predictions={
                backend.value: prediction for backend, prediction in predictions.items()
            },
        )

    def to_dict(self) -> Mapping[str, object]:
        """Dump the immutable calibration-only model."""

        return {
            "model": "compile_aware_same_family_nearest_scale_v1",
            "objective": "capability_then_budget_then_risk_then_amortized_latency",
            "pairs": [
                {
                    **asdict(pair),
                    "family": pair.family.value,
                    "backend": pair.backend.value,
                }
                for pair in self._pairs
            ],
        }


__all__ = [
    "CompileAwareBackendDecision",
    "CompileAwareBackendObservation",
    "CompileAwareFusedCrownPlanner",
    "CompileAwareQueryPolicy",
    "FusedCrownBackendDecision",
    "FusedCrownBackendObservation",
    "FusedCrownBackendPlanner",
    "FusedCrownMultiBackendDecision",
    "FusedCrownMultiBackendPlanner",
]
