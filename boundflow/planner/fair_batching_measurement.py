"""Physical-batching baselines and normalized observations for IR-5C3."""

# pylint: disable=too-many-arguments,too-many-instance-attributes,too-many-locals

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import statistics
import time

import torch

from ..frontends.plain_crown_bound_ir import tensor_content_hash
from ..ir.plan import BackendKind
from ..runtime.crown_ibp import (
    _forward_ibp_trace_mlp,
    run_crown_ibp_mlp,
    run_crown_ibp_mlp_from_forward_trace,
)
from ..runtime.task_backend_dispatch import TypedTaskBackendRegistry
from ..runtime.task_ir_executor import execute_task_ir_semantics
from .adaptive_plan_evaluator import AdaptivePlanObservation
from .measured_adaptive_benchmark import (
    BackendCalibration,
    CandidateMeasurement,
    TypedCNNWorkloadSpec,
)
from .typed_benchmark_workloads import PreparedTypedBenchmark

BATCHED_ORIGINAL_SCHEMA_VERSION = "boundflow.ir5-batched-original/v1"


@dataclass(frozen=True)
class BatchedOriginalMeasurement:  # pylint: disable=too-many-instance-attributes
    """Legacy original solver measured at the same physical sample batch."""

    schema_version: str
    variant: str
    workload: TypedCNNWorkloadSpec
    semantic_hash: str
    cold_batch_latency_ms: float
    warm_batch_latency_ms: tuple[float, ...]
    resident_baseline_bytes: int
    incremental_peak_bytes: int
    measured_peak_bytes: int
    lower_hash: str
    upper_hash: str
    typed_reference_lower_hash: str
    typed_reference_upper_hash: str
    lower_max_abs_diff: float
    upper_max_abs_diff: float
    semantic_allclose: bool

    @property
    def warm_per_query_latency_ms(self) -> tuple[float, ...]:
        """Normalize physical-batch latency by the exact query count."""

        return tuple(
            value / self.workload.batch for value in self.warm_batch_latency_ms
        )

    def validate(self) -> None:
        """Reject invalid timing, memory, identity, or numeric evidence."""

        self.workload.validate()
        hashes = (
            self.semantic_hash,
            self.lower_hash,
            self.upper_hash,
            self.typed_reference_lower_hash,
            self.typed_reference_upper_hash,
        )
        numeric = (
            self.cold_batch_latency_ms,
            self.lower_max_abs_diff,
            self.upper_max_abs_diff,
            *self.warm_batch_latency_ms,
        )
        invalid_identity = (
            self.schema_version != BATCHED_ORIGINAL_SCHEMA_VERSION
            or self.variant
            not in {
                "batched_original",
                "batched_original_from_forward_trace",
            }
            or any(len(value) != 64 for value in hashes)
            or not self.warm_batch_latency_ms
        )
        invalid_measurement = (
            any(not 0.0 <= value < float("inf") for value in numeric)
            or self.resident_baseline_bytes < 0
            or self.incremental_peak_bytes < 0
            or self.measured_peak_bytes <= 0
        )
        invalid_semantics = (
            self.measured_peak_bytes
            != self.resident_baseline_bytes + self.incremental_peak_bytes
            or not self.semantic_allclose
        )
        if invalid_identity or invalid_measurement or invalid_semantics:
            raise ValueError("batched-original measurement is invalid")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe raw evidence without hiding batch normalization."""

        self.validate()
        return {
            "schema_version": self.schema_version,
            "workload": self.workload.to_dict(),
            "variant": self.variant,
            "physical_batch_size": self.workload.batch,
            "semantic_hash": self.semantic_hash,
            "cold_batch_latency_ms": self.cold_batch_latency_ms,
            "warm_batch_latency_ms": list(self.warm_batch_latency_ms),
            "warm_per_query_latency_ms": list(self.warm_per_query_latency_ms),
            "resident_baseline_bytes": self.resident_baseline_bytes,
            "incremental_peak_bytes": self.incremental_peak_bytes,
            "measured_peak_bytes": self.measured_peak_bytes,
            "lower_hash": self.lower_hash,
            "upper_hash": self.upper_hash,
            "typed_reference_lower_hash": self.typed_reference_lower_hash,
            "typed_reference_upper_hash": self.typed_reference_upper_hash,
            "lower_max_abs_diff": self.lower_max_abs_diff,
            "upper_max_abs_diff": self.upper_max_abs_diff,
            "semantic_allclose": self.semantic_allclose,
        }


def measure_batched_original(
    prepared_reference: PreparedTypedBenchmark,
    workload: TypedCNNWorkloadSpec,
    *,
    device: str,
    warm_samples: int,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> BatchedOriginalMeasurement:
    """Measure legacy plain CROWN against the identical typed reference input."""

    return _measure_batched_original(
        prepared_reference,
        workload,
        device=device,
        warm_samples=warm_samples,
        rtol=rtol,
        atol=atol,
        from_forward_trace=False,
    )


def measure_batched_original_from_forward_trace(
    prepared_reference: PreparedTypedBenchmark,
    workload: TypedCNNWorkloadSpec,
    *,
    device: str,
    warm_samples: int,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> BatchedOriginalMeasurement:
    """Measure legacy backward from the same precomputed trace as typed IR."""

    return _measure_batched_original(
        prepared_reference,
        workload,
        device=device,
        warm_samples=warm_samples,
        rtol=rtol,
        atol=atol,
        from_forward_trace=True,
    )


def _measure_batched_original(
    prepared_reference: PreparedTypedBenchmark,
    workload: TypedCNNWorkloadSpec,
    *,
    device: str,
    warm_samples: int,
    rtol: float,
    atol: float,
    from_forward_trace: bool,
) -> BatchedOriginalMeasurement:
    workload.validate()
    if (
        prepared_reference.backend != BackendKind.REFERENCE
        or prepared_reference.workload_id != workload.workload_id
        or warm_samples <= 0
    ):
        raise ValueError("batched-original prepared reference is invalid")
    if int(prepared_reference.input_spec.center.shape[0]) != workload.batch:
        raise ValueError("batched-original physical batch does not match workload")
    typed, _trace = execute_task_ir_semantics(
        prepared_reference.task_module,
        prepared_reference.schedule,
        bound_module=prepared_reference.bound_module,
        template=prepared_reference.template,
        instance=prepared_reference.instance,
        legacy_task_module=prepared_reference.legacy_module,
        input_spec=prepared_reference.input_spec,
        relu_pre=prepared_reference.relu_pre,
        backend=TypedTaskBackendRegistry(),
    )
    _synchronize(device)
    interval_env = None
    relu_pre = None
    if from_forward_trace:
        interval_env, relu_pre = _forward_ibp_trace_mlp(
            prepared_reference.legacy_module,
            prepared_reference.input_spec,
        )

    def run_original():
        if interval_env is None or relu_pre is None:
            return run_crown_ibp_mlp(
                prepared_reference.legacy_module,
                prepared_reference.input_spec,
            )
        return run_crown_ibp_mlp_from_forward_trace(
            prepared_reference.legacy_module,
            prepared_reference.input_spec,
            interval_env=interval_env,
            relu_pre=relu_pre,
        )

    baseline = _memory_allocated(device)
    _reset_peak(device)
    cold_started = time.perf_counter_ns()
    result = run_original()
    _synchronize(device)
    cold_ms = _elapsed_ms(cold_started)
    peak = _max_memory_allocated(device)
    warm: list[float] = []
    for _index in range(warm_samples):
        _reset_peak(device)
        started = time.perf_counter_ns()
        result = run_original()
        _synchronize(device)
        warm.append(_elapsed_ms(started))
        peak = max(peak, _max_memory_allocated(device))
    lower_diff = float((result.lower - typed.lower).abs().max().item())
    upper_diff = float((result.upper - typed.upper).abs().max().item())
    allclose = bool(
        torch.allclose(result.lower, typed.lower, rtol=rtol, atol=atol)
        and torch.allclose(result.upper, typed.upper, rtol=rtol, atol=atol)
    )
    incremental = max(0, peak - baseline)
    measurement = BatchedOriginalMeasurement(
        schema_version=BATCHED_ORIGINAL_SCHEMA_VERSION,
        variant=(
            "batched_original_from_forward_trace"
            if from_forward_trace
            else "batched_original"
        ),
        workload=workload,
        semantic_hash=_baseline_semantic_hash(
            prepared_reference,
            variant=(
                "batched_original_from_forward_trace"
                if from_forward_trace
                else "batched_original"
            ),
        ),
        cold_batch_latency_ms=cold_ms,
        warm_batch_latency_ms=tuple(warm),
        resident_baseline_bytes=baseline,
        incremental_peak_bytes=incremental,
        measured_peak_bytes=max(1, baseline + incremental),
        lower_hash=tensor_content_hash(result.lower),
        upper_hash=tensor_content_hash(result.upper),
        typed_reference_lower_hash=tensor_content_hash(typed.lower),
        typed_reference_upper_hash=tensor_content_hash(typed.upper),
        lower_max_abs_diff=lower_diff,
        upper_max_abs_diff=upper_diff,
        semantic_allclose=allclose,
    )
    measurement.validate()
    return measurement


def compiler_candidate_observation(
    measurement: CandidateMeasurement,
    calibration: BackendCalibration,
) -> AdaptivePlanObservation:
    """Normalize a physically batched compiler candidate to per-query latency."""

    if not isinstance(measurement.workload, TypedCNNWorkloadSpec):
        raise TypeError("fair compiler observation requires CNN held-out workload")
    if measurement.workload.split != "heldout":
        raise ValueError("fair compiler observation is not held-out")
    predicted_batch, predicted_compile = calibration.predict(measurement.workload)
    batch = measurement.workload.batch
    return AdaptivePlanObservation(
        plan_id=f"compiler:{measurement.backend.value}",
        plan_instance_hash=measurement.plan_instance_hash,
        predicted_latency_ms=predicted_batch / batch,
        predicted_compile_ms=predicted_compile,
        local_score_ms=predicted_batch / batch,
        measured_latency_ms=tuple(
            value / batch for value in measurement.warm_latency_ms
        ),
        measured_compile_ms=measurement.measured_compile_setup_ms,
        measured_peak_bytes=measurement.measured_peak_bytes,
        compiled_artifact_key=measurement.compiled_artifact_key,
    )


def ordinary_batching_observation(
    reference_measurement: CandidateMeasurement,
) -> AdaptivePlanObservation:
    """Expose the typed reference plan as an ordinary physical-batch baseline."""

    observation = _normalized_reference(
        reference_measurement,
        plan_id="ordinary-batching",
        divisor=_cnn_batch(reference_measurement),
    )
    observation.validate()
    return observation


def fixed_single_observation(
    single_reference_measurement: CandidateMeasurement,
) -> AdaptivePlanObservation:
    """Expose one-query typed reference repeated by TTV expected query count."""

    if _cnn_batch(single_reference_measurement) != 1:
        raise ValueError("fixed-single observation requires physical batch one")
    observation = _normalized_reference(
        single_reference_measurement,
        plan_id="fixed-single",
        divisor=1,
    )
    observation.validate()
    return observation


def batched_original_observation(
    measurement: BatchedOriginalMeasurement,
) -> AdaptivePlanObservation:
    """Convert legacy physical-batch evidence to the common observation schema."""

    measurement.validate()
    per_query = measurement.warm_per_query_latency_ms
    median = statistics.median(per_query)
    observation = AdaptivePlanObservation(
        plan_id=measurement.variant.replace("_", "-"),
        plan_instance_hash=measurement.semantic_hash,
        predicted_latency_ms=median,
        predicted_compile_ms=0.0,
        local_score_ms=median,
        measured_latency_ms=per_query,
        measured_compile_ms=0.0,
        measured_peak_bytes=measurement.measured_peak_bytes,
    )
    observation.validate()
    return observation


def verify_single_query_matches_batch(
    batched_reference: PreparedTypedBenchmark,
    single_reference: PreparedTypedBenchmark,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> dict[str, object]:
    """Prove batch-one baseline is the first query of the batched workload."""

    if (
        batched_reference.backend != BackendKind.REFERENCE
        or single_reference.backend != BackendKind.REFERENCE
        or int(single_reference.input_spec.center.shape[0]) != 1
        or tuple(single_reference.input_spec.center.shape[1:])
        != tuple(batched_reference.input_spec.center.shape[1:])
    ):
        raise ValueError("fixed/batched semantic comparison inputs are invalid")
    batched, _batched_trace = _execute_reference(batched_reference)
    single, _single_trace = _execute_reference(single_reference)
    lower = batched.lower[:1]
    upper = batched.upper[:1]
    lower_diff = float((single.lower - lower).abs().max().item())
    upper_diff = float((single.upper - upper).abs().max().item())
    allclose = bool(
        torch.allclose(single.lower, lower, rtol=rtol, atol=atol)
        and torch.allclose(single.upper, upper, rtol=rtol, atol=atol)
    )
    if not allclose:
        raise ValueError("fixed-single semantics differ from batched first query")
    return {
        "semantic_allclose": True,
        "single_lower_hash": tensor_content_hash(single.lower),
        "single_upper_hash": tensor_content_hash(single.upper),
        "batched_first_lower_hash": tensor_content_hash(lower),
        "batched_first_upper_hash": tensor_content_hash(upper),
        "lower_max_abs_diff": lower_diff,
        "upper_max_abs_diff": upper_diff,
    }


def _normalized_reference(
    measurement: CandidateMeasurement,
    *,
    plan_id: str,
    divisor: int,
) -> AdaptivePlanObservation:
    measurement.validate()
    if measurement.backend != BackendKind.REFERENCE:
        raise ValueError("batching baseline must use typed reference backend")
    normalized = tuple(value / divisor for value in measurement.warm_latency_ms)
    median = statistics.median(normalized)
    return AdaptivePlanObservation(
        plan_id=plan_id,
        plan_instance_hash=measurement.plan_instance_hash,
        predicted_latency_ms=median,
        predicted_compile_ms=0.0,
        local_score_ms=median,
        measured_latency_ms=normalized,
        measured_compile_ms=0.0,
        measured_peak_bytes=measurement.measured_peak_bytes,
    )


def _cnn_batch(measurement: CandidateMeasurement) -> int:
    if not isinstance(measurement.workload, TypedCNNWorkloadSpec):
        raise TypeError("fair batching baseline requires a CNN workload")
    return measurement.workload.batch


def _execute_reference(prepared: PreparedTypedBenchmark):
    return execute_task_ir_semantics(
        prepared.task_module,
        prepared.schedule,
        bound_module=prepared.bound_module,
        template=prepared.template,
        instance=prepared.instance,
        legacy_task_module=prepared.legacy_module,
        input_spec=prepared.input_spec,
        relu_pre=prepared.relu_pre,
        backend=TypedTaskBackendRegistry(),
    )


def _baseline_semantic_hash(prepared: PreparedTypedBenchmark, *, variant: str) -> str:
    payload = {
        "kind": variant,
        "bound_module_hash": prepared.bound_module.stable_hash(),
        "input_shape": list(prepared.input_spec.center.shape),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


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
    "BATCHED_ORIGINAL_SCHEMA_VERSION",
    "BatchedOriginalMeasurement",
    "batched_original_observation",
    "compiler_candidate_observation",
    "fixed_single_observation",
    "measure_batched_original",
    "measure_batched_original_from_forward_trace",
    "ordinary_batching_observation",
    "verify_single_query_matches_batch",
]
