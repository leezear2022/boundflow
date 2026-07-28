"""Measured typed workloads and leakage-free calibration for IR-5 evaluation."""

# Measurement code deliberately keeps raw and modeled facts in one typed record.
# pylint: disable=too-many-arguments,too-many-instance-attributes,too-many-locals

from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time
from typing import Mapping, Sequence

import torch

from ..frontends.plain_crown_bound_ir import tensor_content_hash
from ..ir.plan import BackendKind
from ..runtime.task_backend_dispatch import TypedTaskBackendRegistry
from ..runtime.task_ir_executor import execute_task_ir_semantics
from .adaptive_plan_evaluator import AdaptivePlanObservation
from .typed_benchmark_workloads import (
    PreparedTypedBenchmark,
    build_cnn_candidate,
    build_mlp_candidate,
)

MEASURED_BENCHMARK_SCHEMA_VERSION = "boundflow.ir5-measured-candidate/v1"


@dataclass(frozen=True)
class TypedWorkloadSpec:
    """Frozen MLP shape and seed, independent of backend choice."""

    workload_id: str
    split: str
    batch: int
    input_dim: int
    hidden_dim: int
    output_dim: int
    seed: int

    def validate(self) -> None:
        """Reject mutable/ambiguous workload identities."""

        if not self.workload_id or self.split not in {"calibration", "heldout"}:
            raise ValueError("measured workload identity/split is invalid")
        if min(self.batch, self.input_dim, self.hidden_dim, self.output_dim) <= 0:
            raise ValueError("measured workload dimensions must be positive")
        if self.seed < 0:
            raise ValueError("measured workload seed must be nonnegative")

    @property
    def work_units(self) -> int:
        """Return a backend-independent shape feature for calibration."""

        self.validate()
        return self.batch * self.output_dim * self.hidden_dim * self.input_dim

    def to_dict(self) -> dict[str, object]:
        """Return canonical JSON fields."""

        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class TypedCNNWorkloadSpec:
    """Frozen chain-CNN family used for architecture-held-out evaluation."""

    workload_id: str
    split: str
    batch: int
    input_channels: int
    image_size: int
    conv1_channels: int
    conv2_channels: int
    output_dim: int
    seed: int

    def validate(self) -> None:
        """Reject invalid CNN shapes or ambiguous split identity."""

        dimensions = (
            self.batch,
            self.input_channels,
            self.image_size,
            self.conv1_channels,
            self.conv2_channels,
            self.output_dim,
        )
        if not self.workload_id or self.split not in {"calibration", "heldout"}:
            raise ValueError("measured CNN workload identity/split is invalid")
        if min(dimensions) <= 0 or self.image_size % 2:
            raise ValueError(
                "measured CNN dimensions must be positive with even image size"
            )
        if self.seed < 0:
            raise ValueError("measured CNN workload seed must be nonnegative")

    @property
    def work_units(self) -> int:
        """Return an architecture-independent multiply-accumulate proxy."""

        self.validate()
        half = self.image_size // 2
        conv1 = (
            self.batch
            * self.conv1_channels
            * self.image_size
            * self.image_size
            * self.input_channels
            * 9
        )
        conv2 = self.batch * self.conv2_channels * half * half * self.conv1_channels * 9
        head = self.batch * self.output_dim * self.conv2_channels * half * half
        return conv1 + conv2 + head

    def to_dict(self) -> dict[str, object]:
        """Return canonical architecture-family fields."""

        self.validate()
        return {"family": "chain_cnn", **asdict(self)}


MeasuredWorkloadSpec = TypedWorkloadSpec | TypedCNNWorkloadSpec


@dataclass(frozen=True)
class CandidateMeasurement:  # pylint: disable=too-many-instance-attributes
    """Raw execution facts for one immutable workload/backend candidate."""

    schema_version: str
    workload: MeasuredWorkloadSpec
    backend: BackendKind
    plan_instance_hash: str
    bound_module_hash: str
    task_module_hash: str
    schedule_hash: str
    compiled_artifact_key: str | None
    cold_latency_ms: float
    warm_latency_ms: tuple[float, ...]
    measured_compile_setup_ms: float
    resident_baseline_bytes: int
    incremental_peak_bytes: int
    measured_peak_bytes: int
    lower_hash: str
    upper_hash: str
    reference_lower_hash: str
    reference_upper_hash: str
    lower_max_abs_diff: float
    upper_max_abs_diff: float
    semantic_allclose: bool
    cache_events: tuple[Mapping[str, object], ...]

    @property
    def plan_id(self) -> str:
        """Return a workload-local policy identifier."""

        return self.backend.value

    def validate(self) -> None:
        """Reject incomplete or semantically invalid benchmark evidence."""

        self.workload.validate()
        if self.schema_version != MEASURED_BENCHMARK_SCHEMA_VERSION:
            raise ValueError("unsupported measured benchmark schema")
        if any(
            len(value) != 64
            for value in (
                self.plan_instance_hash,
                self.bound_module_hash,
                self.task_module_hash,
                self.schedule_hash,
                self.lower_hash,
                self.upper_hash,
                self.reference_lower_hash,
                self.reference_upper_hash,
            )
        ):
            raise ValueError("measured benchmark hash identity is invalid")
        numeric = (
            self.cold_latency_ms,
            self.measured_compile_setup_ms,
            self.lower_max_abs_diff,
            self.upper_max_abs_diff,
            *self.warm_latency_ms,
        )
        invalid_numeric = (
            not self.warm_latency_ms
            or any(not _finite_nonnegative(value) for value in numeric)
            or self.resident_baseline_bytes < 0
            or self.incremental_peak_bytes < 0
            or self.measured_peak_bytes <= 0
        )
        invalid_semantics = (
            self.measured_peak_bytes
            != self.resident_baseline_bytes + self.incremental_peak_bytes
            or not self.semantic_allclose
        )
        if invalid_numeric or invalid_semantics:
            raise ValueError("measured benchmark evidence is invalid")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe raw evidence."""

        self.validate()
        return {
            "schema_version": self.schema_version,
            "workload": self.workload.to_dict(),
            "backend": self.backend.value,
            "plan_id": self.plan_id,
            "plan_instance_hash": self.plan_instance_hash,
            "bound_module_hash": self.bound_module_hash,
            "task_module_hash": self.task_module_hash,
            "schedule_hash": self.schedule_hash,
            "compiled_artifact_key": self.compiled_artifact_key,
            "cold_latency_ms": self.cold_latency_ms,
            "warm_latency_ms": list(self.warm_latency_ms),
            "measured_compile_setup_ms": self.measured_compile_setup_ms,
            "resident_baseline_bytes": self.resident_baseline_bytes,
            "incremental_peak_bytes": self.incremental_peak_bytes,
            "measured_peak_bytes": self.measured_peak_bytes,
            "lower_hash": self.lower_hash,
            "upper_hash": self.upper_hash,
            "reference_lower_hash": self.reference_lower_hash,
            "reference_upper_hash": self.reference_upper_hash,
            "lower_max_abs_diff": self.lower_max_abs_diff,
            "upper_max_abs_diff": self.upper_max_abs_diff,
            "semantic_allclose": self.semantic_allclose,
            "cache_events": list(self.cache_events),
        }


@dataclass(frozen=True)
class BackendCalibration:
    """Prediction coefficients fitted only from calibration workloads."""

    backend: BackendKind
    latency_ms_per_work_unit: float
    compile_setup_ms: float
    calibration_workload_ids: tuple[str, ...]
    calibration_measurement_hash: str

    def predict(self, workload: MeasuredWorkloadSpec) -> tuple[float, float]:
        """Predict steady latency/setup without reading held-out measurements."""

        workload.validate()
        return (
            self.latency_ms_per_work_unit * workload.work_units,
            self.compile_setup_ms,
        )

    def to_dict(self) -> dict[str, object]:
        """Return canonical model provenance."""

        return {
            "backend": self.backend.value,
            "latency_ms_per_work_unit": self.latency_ms_per_work_unit,
            "compile_setup_ms": self.compile_setup_ms,
            "calibration_workload_ids": list(self.calibration_workload_ids),
            "calibration_measurement_hash": self.calibration_measurement_hash,
        }


def measure_workload(  # pylint: disable=too-many-statements
    workload: MeasuredWorkloadSpec,
    backends: Sequence[BackendKind],
    *,
    device: str,
    warm_samples: int,
    cache_root: Path,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> tuple[CandidateMeasurement, ...]:
    """Execute identical typed semantics for every backend and check reference."""

    workload.validate()
    if not backends or backends[0] != BackendKind.REFERENCE:
        raise ValueError("measured backend order must begin with reference")
    if len(set(backends)) != len(backends) or warm_samples <= 0:
        raise ValueError("measured backend list/samples are invalid")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA measurement requested but CUDA is unavailable")

    measurements: list[CandidateMeasurement] = []
    reference_lower: torch.Tensor | None = None
    reference_upper: torch.Tensor | None = None
    for backend in backends:
        prepared = _build_candidate(workload, backend=backend, device=device)
        registry = TypedTaskBackendRegistry(
            tvm_cache_dir=cache_root / workload.workload_id / backend.value
        )
        resident_baseline = _memory_allocated(device)
        _reset_peak(device)
        cold_started = time.perf_counter_ns()
        result, _trace = _execute(prepared, registry)
        _synchronize(device)
        cold_ms = _elapsed_ms(cold_started)
        cold_peak = _max_memory_allocated(device)
        warm: list[float] = []
        peak = cold_peak
        for _index in range(warm_samples):
            _reset_peak(device)
            started = time.perf_counter_ns()
            result, _trace = _execute(prepared, registry)
            _synchronize(device)
            warm.append(_elapsed_ms(started))
            peak = max(peak, _max_memory_allocated(device))
        lower = result.lower.detach()
        upper = result.upper.detach()
        if reference_lower is None:
            reference_lower = lower.clone()
            reference_upper = upper.clone()
        assert reference_upper is not None
        lower_diff = float((lower - reference_lower).abs().max().item())
        upper_diff = float((upper - reference_upper).abs().max().item())
        allclose = bool(
            torch.allclose(lower, reference_lower, rtol=rtol, atol=atol)
            and torch.allclose(upper, reference_upper, rtol=rtol, atol=atol)
        )
        cache_events = _cache_events(registry)
        compile_setup_ms = _compile_setup_ms(cache_events)
        artifact_keys = sorted(
            {
                candidate.compiled_artifact_key
                for candidate in prepared.template.backend_candidates
                if candidate.compiled_artifact_key is not None
                and candidate.candidate_id
                in {
                    task.backend.backend_candidate_id
                    for task in prepared.task_module.tasks
                }
            }
        )
        artifact_key = None if not artifact_keys else _artifact_set_key(artifact_keys)
        incremental_peak = max(0, peak - resident_baseline)
        measurement = CandidateMeasurement(
            schema_version=MEASURED_BENCHMARK_SCHEMA_VERSION,
            workload=workload,
            backend=backend,
            plan_instance_hash=prepared.instance.stable_hash(
                template=prepared.template,
                bound_module=prepared.bound_module,
            ),
            bound_module_hash=prepared.bound_module.stable_hash(),
            task_module_hash=prepared.task_module.stable_hash(
                bound_module=prepared.bound_module,
                template=prepared.template,
                instance=prepared.instance,
            ),
            schedule_hash=prepared.schedule.stable_hash(
                bound_module=prepared.bound_module,
                template=prepared.template,
                instance=prepared.instance,
            ),
            compiled_artifact_key=artifact_key,
            cold_latency_ms=cold_ms,
            warm_latency_ms=tuple(warm),
            measured_compile_setup_ms=compile_setup_ms,
            resident_baseline_bytes=resident_baseline,
            incremental_peak_bytes=incremental_peak,
            measured_peak_bytes=max(1, resident_baseline + incremental_peak),
            lower_hash=tensor_content_hash(lower),
            upper_hash=tensor_content_hash(upper),
            reference_lower_hash=tensor_content_hash(reference_lower),
            reference_upper_hash=tensor_content_hash(reference_upper),
            lower_max_abs_diff=lower_diff,
            upper_max_abs_diff=upper_diff,
            semantic_allclose=allclose,
            cache_events=cache_events,
        )
        measurement.validate()
        measurements.append(measurement)
        del prepared, registry, result, lower, upper
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    return tuple(measurements)


def fit_backend_calibrations(
    measurements: Sequence[CandidateMeasurement],
) -> Mapping[BackendKind, BackendCalibration]:
    """Fit robust per-backend coefficients from calibration evidence only."""

    if not measurements:
        raise ValueError("calibration measurements are empty")
    grouped: dict[BackendKind, list[CandidateMeasurement]] = {}
    for item in measurements:
        item.validate()
        if item.workload.split != "calibration":
            raise ValueError("held-out measurement leaked into calibration fit")
        grouped.setdefault(item.backend, []).append(item)
    models: dict[BackendKind, BackendCalibration] = {}
    for backend, items in grouped.items():
        encoded = json.dumps(
            [item.to_dict() for item in items],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        models[backend] = BackendCalibration(
            backend=backend,
            latency_ms_per_work_unit=statistics.median(
                statistics.median(item.warm_latency_ms) / item.workload.work_units
                for item in items
            ),
            compile_setup_ms=statistics.median(
                item.measured_compile_setup_ms for item in items
            ),
            calibration_workload_ids=tuple(
                sorted(item.workload.workload_id for item in items)
            ),
            calibration_measurement_hash=hashlib.sha256(encoded).hexdigest(),
        )
    return models


def build_heldout_observations(
    measurements: Sequence[CandidateMeasurement],
    calibrations: Mapping[BackendKind, BackendCalibration],
) -> tuple[AdaptivePlanObservation, ...]:
    """Join held-out outcomes with calibration-only predictions."""

    observations: list[AdaptivePlanObservation] = []
    for item in measurements:
        item.validate()
        if item.workload.split != "heldout":
            raise ValueError("adaptive observation is not held-out")
        model = calibrations.get(item.backend)
        if model is None:
            raise ValueError(f"missing calibration for {item.backend.value}")
        predicted_latency, predicted_compile = model.predict(item.workload)
        observations.append(
            AdaptivePlanObservation(
                plan_id=item.plan_id,
                plan_instance_hash=item.plan_instance_hash,
                predicted_latency_ms=predicted_latency,
                predicted_compile_ms=predicted_compile,
                local_score_ms=predicted_latency,
                measured_latency_ms=item.warm_latency_ms,
                measured_compile_ms=item.measured_compile_setup_ms,
                measured_peak_bytes=item.measured_peak_bytes,
                compiled_artifact_key=item.compiled_artifact_key,
            )
        )
    return tuple(observations)


def _execute(
    prepared: PreparedTypedBenchmark,
    registry: TypedTaskBackendRegistry,
):
    return execute_task_ir_semantics(
        prepared.task_module,
        prepared.schedule,
        bound_module=prepared.bound_module,
        template=prepared.template,
        instance=prepared.instance,
        legacy_task_module=prepared.legacy_module,
        input_spec=prepared.input_spec,
        relu_pre=prepared.relu_pre,
        backend=registry,
    )


def _cache_events(
    registry: TypedTaskBackendRegistry,
) -> tuple[Mapping[str, object], ...]:
    cache = registry.tvm.cache
    if cache is None:
        return ()
    return tuple(event.to_dict() for event in cache.events)


def _artifact_set_key(keys: Sequence[str]) -> str:
    encoded = json.dumps(sorted(keys), separators=(",", ":")).encode("utf-8")
    return "typed-artifacts:" + hashlib.sha256(encoded).hexdigest()


def _compile_setup_ms(events: Sequence[Mapping[str, object]]) -> float:
    total = 0.0
    for event in events:
        value = event.get("total_ms")
        if event.get("event") not in {"miss", "disk_hit"}:
            continue
        if not isinstance(value, (int, float)):
            raise TypeError("TVM cache event total_ms is not numeric")
        total += float(value)
    return total


def _finite_nonnegative(value: float) -> bool:
    return 0.0 <= value < float("inf")


def _synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _reset_peak(device: str) -> None:
    _synchronize(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()


def _memory_allocated(device: str) -> int:
    return int(torch.cuda.memory_allocated()) if device == "cuda" else 0


def _max_memory_allocated(device: str) -> int:
    return int(torch.cuda.max_memory_allocated()) if device == "cuda" else 1


def _elapsed_ms(started_ns: int) -> float:
    return (time.perf_counter_ns() - started_ns) / 1e6


__all__ = [
    "BackendCalibration",
    "CandidateMeasurement",
    "MEASURED_BENCHMARK_SCHEMA_VERSION",
    "MeasuredWorkloadSpec",
    "TypedCNNWorkloadSpec",
    "TypedWorkloadSpec",
    "build_heldout_observations",
    "fit_backend_calibrations",
    "measure_workload",
]


def _build_candidate(
    workload: MeasuredWorkloadSpec,
    *,
    backend: BackendKind,
    device: str,
) -> PreparedTypedBenchmark:
    if isinstance(workload, TypedCNNWorkloadSpec):
        return build_cnn_candidate(
            workload_id=workload.workload_id,
            backend=backend,
            device=device,
            batch=workload.batch,
            input_channels=workload.input_channels,
            image_size=workload.image_size,
            conv1_channels=workload.conv1_channels,
            conv2_channels=workload.conv2_channels,
            output_dim=workload.output_dim,
            seed=workload.seed,
        )
    return build_mlp_candidate(
        workload_id=workload.workload_id,
        backend=backend,
        device=device,
        batch=workload.batch,
        input_dim=workload.input_dim,
        hidden_dim=workload.hidden_dim,
        output_dim=workload.output_dim,
        seed=workload.seed,
    )
