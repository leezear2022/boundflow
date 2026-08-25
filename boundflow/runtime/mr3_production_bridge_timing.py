"""Mechanical gates for MR3 single-site production bridge timing."""

# pylint: disable=missing-function-docstring,too-many-locals,too-many-statements
# pylint: disable=too-many-branches,too-many-boolean-expressions

from __future__ import annotations

import hashlib
import json
import math
import random
from statistics import fmean
from typing import Any, Mapping, Sequence, cast

from .mr3_production_bridge_formal import _compare_payload

TIMING_SCHEMA = "boundflow.mr3-production-bridge-timing-formal/v1"
WORKER_SCHEMA = "boundflow.mr3-production-bridge-timing-worker/v1"
SOURCE_COMMIT = "2d788ad6608d7d4da9ac9937efa3cdeb11d36f27"
ABCROWN_COMMIT = "e5c7e17bf0488843acb77b7519f59876717a49f4"
AUTO_LIRPA_COMMIT = "5a098e8f9fb5786a428a024981d833d303921f2d"
VNNCOMP_COMMIT = "90419aadcf06cf543ce5c1706cae1059dc9fa6cf"
MODEL_SHA256 = "791aa24d77917ecda16809fbbd48e7739616f88ebf74cf358b2d1bf911dc4a6d"
PROPERTY_SHA256 = "89edf0665d74397670d0562d513db694a49a84edaf5cf3d64c9c6fa63c3769ff"
GO_STATUS = "VALIDATED-MR3-P-PRODUCTION-BRIDGE-PHYSICS"
NO_GO_STATUS = "VALIDATED-NO-GO-MR3-P-PRODUCTION-BRIDGE-PHYSICS"
EXPECTED_RUNS = tuple(
    (pair, position, mode)
    for pair, modes in enumerate(
        (
            ("provider", "bridge"),
            ("bridge", "provider"),
            ("provider", "bridge"),
            ("bridge", "provider"),
            ("provider", "bridge"),
            ("bridge", "provider"),
        )
    )
    for position, mode in enumerate(modes)
)
HOST_GEOMEAN_GATE = 1.05
BOOTSTRAP_LOWER_GATE = 1.00
WORST_PAIR_GATE = 0.98
MEMORY_RATIO_GATE = 1.05
BOOTSTRAP_SEED = 20260826
BOOTSTRAP_SAMPLES = 10_000
ATOL = 2.0e-4
RTOL = 2.0e-4


def canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _geomean(values: Sequence[float]) -> float:
    if not values or any(value <= 0.0 or not math.isfinite(value) for value in values):
        raise ValueError("MR3 timing ratio differs")
    return math.exp(fmean(math.log(value) for value in values))


def _bootstrap_lower(values: Sequence[float]) -> float:
    generator = random.Random(BOOTSTRAP_SEED)
    samples = sorted(
        _geomean([values[generator.randrange(len(values))] for _ in values])
        for _ in range(BOOTSTRAP_SAMPLES)
    )
    return samples[int(0.025 * BOOTSTRAP_SAMPLES)]


def _validate_source(source: object) -> None:
    if source != {
        "abcrown_commit": ABCROWN_COMMIT,
        "auto_lirpa_commit": AUTO_LIRPA_COMMIT,
        "vnncomp_commit": VNNCOMP_COMMIT,
        "model_sha256": MODEL_SHA256,
        "property_sha256": PROPERTY_SHA256,
    }:
        raise ValueError("MR3 timing worker source differs")


def _validate_protocol(protocol: object) -> None:
    if protocol != {
        "device": "cuda",
        "seed": 100,
        "max_iterations": 1,
        "batch_size": 64,
        "alpha_steps": 5,
        "beta_steps": 10,
        "formal_observation_enabled": False,
        "compile_timed": False,
        "dummy_module_warm_timed": False,
    }:
        raise ValueError("MR3 timing worker protocol differs")


def _validate_gpu_snapshot(value: object) -> None:
    expected = {
        "name",
        "driver_version",
        "temperature.gpu",
        "power.draw",
        "clocks.current.graphics",
        "clocks.current.memory",
        "power.limit",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("MR3 timing GPU snapshot differs")
    for key, item in value.items():
        if (
            not isinstance(item, str)
            or not item
            or (key == "name" and "NVIDIA" not in item)
        ):
            raise ValueError("MR3 timing GPU snapshot value differs")


def _validate_measurement(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "host_ns",
        "cuda_event_ms",
        "device_before",
        "device_after",
        "stream_before",
        "stream_after",
        "base_allocated_bytes",
        "base_reserved_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
    }:
        raise ValueError("MR3 timing measurement structure differs")
    integer_fields = (
        "host_ns",
        "device_before",
        "device_after",
        "stream_before",
        "stream_after",
        "base_allocated_bytes",
        "base_reserved_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
    )
    if any(not isinstance(value[field], int) for field in integer_fields):
        raise ValueError("MR3 timing measurement type differs")
    if (
        value["host_ns"] <= 0
        or not isinstance(value["cuda_event_ms"], float)
        or not math.isfinite(value["cuda_event_ms"])
        or value["cuda_event_ms"] <= 0.0
        or value["device_before"] != value["device_after"]
        or value["stream_before"] != value["stream_after"]
        or value["base_allocated_bytes"] < 0
        or value["base_reserved_bytes"] < 0
        or value["peak_allocated_bytes"] < value["base_allocated_bytes"]
        or value["peak_reserved_bytes"] < value["base_reserved_bytes"]
    ):
        raise ValueError("MR3 timing measurement invariant differs")


def _validate_module_receipt(value: object) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("MR3 timing module receipt is absent")
    unsigned = dict(value)
    receipt_hash = unsigned.pop("receipt_hash", None)
    if (
        receipt_hash != canonical_hash(unsigned)
        or not _is_sha256(value.get("module_hash"))
        or not _is_sha256(value.get("device_source_hash"))
        or not isinstance(value.get("tvm_version"), str)
        or value.get("exported_symbols")
        != [
            "boundflow_cibc_dense_exact_forward_v3",
            "boundflow_cibc_dense_exact_backward_v3",
        ]
        or value.get("global_workspace_bytes") != 0
        or value.get("dummy_forward_launch_count") != 1
        or value.get("dummy_backward_launch_count") != 1
        or value.get("dummy_fallback_count") != 0
        or value.get("dummy_eager_count") != 0
    ):
        raise ValueError("MR3 timing module receipt differs")


def _validate_bridge_receipt(value: object) -> None:
    expected = {
        "evaluation_count": 10,
        "forward_launch_count": 10,
        "backward_launch_count": 9,
        "empty_beta_tensor_count": 10,
        "empty_beta_numel": 0,
        "relu_conv_content_match_count": 10,
        "relu_conv_pointer_match_count": 0,
        "persistent_dense_a_count": 0,
        "fallback_count": 0,
        "eager_count": 0,
        "native_shadow_count": 0,
        "timing_recorded": False,
        "performance_claimed": False,
    }
    if value != expected:
        raise ValueError("MR3 timing bridge receipt differs")


def validate_worker(worker: Mapping[str, Any], *, mode: str) -> None:
    unsigned = dict(worker)
    worker_hash = unsigned.pop("worker_hash", None)
    if (
        worker.get("schema_version") != WORKER_SCHEMA
        or worker.get("mode") != mode
        or worker_hash != canonical_hash(unsigned)
        or worker.get("timing_recorded") is not True
        or worker.get("performance_claimed") is not False
        or worker.get("solver_result")
        != {"status": "verified", "success": True, "visited_domains": [6]}
        or worker.get("device_before") != worker.get("device_after")
        or worker.get("stream_before") != worker.get("stream_after")
    ):
        raise ValueError("MR3 timing worker envelope differs")
    _validate_source(worker.get("source"))
    _validate_protocol(worker.get("protocol"))
    _validate_gpu_snapshot(worker.get("gpu_before"))
    _validate_gpu_snapshot(worker.get("gpu_after"))
    _validate_measurement(worker.get("measurement"))
    measurement = worker["measurement"]
    if (
        worker["device_before"] != measurement["device_before"]
        or worker["device_after"] != measurement["device_after"]
        or worker["stream_before"] != measurement["stream_before"]
        or worker["stream_after"] != measurement["stream_after"]
    ):
        raise ValueError("MR3 timing worker device/stream projection differs")
    if mode == "provider":
        if (
            worker.get("candidate_module_receipt") is not None
            or worker.get("bridge_receipt") is not None
        ):
            raise ValueError("MR3 timing provider unexpectedly has bridge state")
    else:
        _validate_module_receipt(worker.get("candidate_module_receipt"))
        _validate_bridge_receipt(worker.get("bridge_receipt"))


def _pair_metric(
    pair_index: int, provider: Mapping[str, Any], bridge: Mapping[str, Any]
) -> dict[str, object]:
    maximum = 0.0
    element_count = 0
    for field in (
        "solver_result",
        "outer_result_state",
        "final_target_alpha_state",
        "final_module_state",
    ):
        field_maximum, field_count = _compare_payload(
            provider[field], bridge[field], atol=ATOL, rtol=RTOL, path=field
        )
        maximum = max(maximum, field_maximum)
        element_count += field_count
    provider_measurement = provider["measurement"]
    bridge_measurement = bridge["measurement"]
    host_speedup = provider_measurement["host_ns"] / bridge_measurement["host_ns"]
    event_speedup = (
        provider_measurement["cuda_event_ms"] / bridge_measurement["cuda_event_ms"]
    )
    provider_allocated_increment = (
        provider_measurement["peak_allocated_bytes"]
        - provider_measurement["base_allocated_bytes"]
    )
    bridge_allocated_increment = (
        bridge_measurement["peak_allocated_bytes"]
        - bridge_measurement["base_allocated_bytes"]
    )
    provider_reserved_increment = (
        provider_measurement["peak_reserved_bytes"]
        - provider_measurement["base_reserved_bytes"]
    )
    bridge_reserved_increment = (
        bridge_measurement["peak_reserved_bytes"]
        - bridge_measurement["base_reserved_bytes"]
    )
    metric: dict[str, object] = {
        "pair_index": pair_index,
        "semantic_maximum_absolute_difference": maximum,
        "semantic_element_count": element_count,
        "sign_exact": True,
        "allclose": True,
        "provider_host_ns": provider_measurement["host_ns"],
        "bridge_host_ns": bridge_measurement["host_ns"],
        "host_speedup": host_speedup,
        "provider_cuda_event_ms": provider_measurement["cuda_event_ms"],
        "bridge_cuda_event_ms": bridge_measurement["cuda_event_ms"],
        "cuda_event_speedup": event_speedup,
        "host_event_direction_consistent": (host_speedup >= 1.0)
        == (event_speedup >= 1.0),
        "absolute_peak_allocated_ratio": bridge_measurement["peak_allocated_bytes"]
        / provider_measurement["peak_allocated_bytes"],
        "absolute_peak_reserved_ratio": bridge_measurement["peak_reserved_bytes"]
        / provider_measurement["peak_reserved_bytes"],
        "provider_incremental_allocated_bytes": provider_allocated_increment,
        "bridge_incremental_allocated_bytes": bridge_allocated_increment,
        "provider_incremental_reserved_bytes": provider_reserved_increment,
        "bridge_incremental_reserved_bytes": bridge_reserved_increment,
        "incremental_allocated_ratio": (
            bridge_allocated_increment / provider_allocated_increment
            if provider_allocated_increment > 0
            else None
        ),
        "incremental_reserved_ratio": (
            bridge_reserved_increment / provider_reserved_increment
            if provider_reserved_increment > 0
            else None
        ),
    }
    metric["metric_hash"] = canonical_hash(metric)
    return metric


def derive_summary(raw: Mapping[str, Any]) -> dict[str, object]:
    unsigned = dict(raw)
    raw_hash = unsigned.pop("raw_hash", None)
    runs_value = raw.get("runs")
    if (
        raw.get("schema_version") != TIMING_SCHEMA
        or raw.get("source_commit") != SOURCE_COMMIT
        or raw.get("run_order") != [list(run) for run in EXPECTED_RUNS]
        or raw_hash != canonical_hash(unsigned)
        or not isinstance(runs_value, list)
        or len(runs_value) != len(EXPECTED_RUNS)
    ):
        raise ValueError("MR3 timing raw provenance differs")
    runs: list[Mapping[str, Any]] = []
    module_receipts: list[object] = []
    gpu_identities: list[tuple[object, object]] = []
    for expected, wrapper in zip(EXPECTED_RUNS, runs_value):
        if (
            not isinstance(wrapper, Mapping)
            or (wrapper.get("pair_index"), wrapper.get("position"), wrapper.get("mode"))
            != expected
            or not isinstance(wrapper.get("worker"), Mapping)
        ):
            raise ValueError("MR3 timing run order differs")
        worker = wrapper["worker"]
        validate_worker(worker, mode=expected[2])
        runs.append(wrapper)
        gpu_identities.append(
            (worker["gpu_before"]["name"], worker["gpu_before"]["driver_version"])
        )
        if expected[2] == "bridge":
            module_receipts.append(worker["candidate_module_receipt"])
    if len(set(gpu_identities)) != 1:
        raise ValueError("MR3 timing GPU identity differs across workers")
    module_hashes = {canonical_hash(receipt) for receipt in module_receipts}
    if len(module_receipts) != 6 or len(module_hashes) != 1:
        raise ValueError("MR3 timing candidate module differs across workers")
    metrics: list[dict[str, object]] = []
    for pair_index in range(6):
        pair = [run for run in runs if run["pair_index"] == pair_index]
        provider = next(run["worker"] for run in pair if run["mode"] == "provider")
        bridge = next(run["worker"] for run in pair if run["mode"] == "bridge")
        metrics.append(_pair_metric(pair_index, provider, bridge))
    host_speedups = [cast(float, metric["host_speedup"]) for metric in metrics]
    geomean = _geomean(host_speedups)
    bootstrap_lower = _bootstrap_lower(host_speedups)
    worst_pair = min(host_speedups)
    worst_allocated = max(
        cast(float, metric["absolute_peak_allocated_ratio"]) for metric in metrics
    )
    worst_reserved = max(
        cast(float, metric["absolute_peak_reserved_ratio"]) for metric in metrics
    )
    direction_count = sum(
        bool(metric["host_event_direction_consistent"]) for metric in metrics
    )
    gates = {
        "pair_count": len(metrics) == 6,
        "correctness": all(bool(metric["allclose"]) for metric in metrics),
        "host_geomean": geomean >= HOST_GEOMEAN_GATE,
        "bootstrap_lower": bootstrap_lower >= BOOTSTRAP_LOWER_GATE,
        "worst_pair": worst_pair >= WORST_PAIR_GATE,
        "absolute_peak_allocated": worst_allocated <= MEMORY_RATIO_GATE,
        "absolute_peak_reserved": worst_reserved <= MEMORY_RATIO_GATE,
        "host_event_direction": direction_count == 6,
        "module_stability": len(module_hashes) == 1,
    }
    go = all(gates.values())
    summary: dict[str, object] = {
        "schema_version": TIMING_SCHEMA,
        "status": GO_STATUS if go else NO_GO_STATUS,
        "source_commit": SOURCE_COMMIT,
        "run_count": len(runs),
        "pair_count": len(metrics),
        "pair_metrics": metrics,
        "host_speedup_geomean": geomean,
        "host_speedup_bootstrap_95_lower": bootstrap_lower,
        "host_speedup_worst_pair": worst_pair,
        "absolute_peak_allocated_worst_ratio": worst_allocated,
        "absolute_peak_reserved_worst_ratio": worst_reserved,
        "host_event_direction_consistent_count": direction_count,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "gates": gates,
        "candidate_module_receipt_hash": next(iter(module_hashes)),
        "performance_claimed": go,
        "same_solver_complete_query_timing_open": go,
    }
    summary["summary_hash"] = canonical_hash(summary)
    return summary


__all__ = [
    "EXPECTED_RUNS",
    "GO_STATUS",
    "NO_GO_STATUS",
    "SOURCE_COMMIT",
    "TIMING_SCHEMA",
    "canonical_hash",
    "derive_summary",
    "validate_worker",
]
