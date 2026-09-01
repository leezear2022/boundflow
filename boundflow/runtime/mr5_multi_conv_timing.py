"""Mechanical gates for MR5 multi-site production bridge timing."""

# pylint: disable=missing-function-docstring,too-many-locals,too-many-statements
# pylint: disable=too-many-branches,too-many-boolean-expressions,protected-access

from __future__ import annotations

from typing import Any, Mapping, cast

from boundflow.runtime import mr3_production_bridge_timing as legacy
from boundflow.runtime.mr3_provider_hook_feasibility import canonical_hash
from boundflow.runtime.mr5_multi_conv_formal import _validate_module_receipts

TIMING_SCHEMA = "boundflow.mr5-multi-conv-timing-formal/v1"
WORKER_SCHEMA = "boundflow.mr5-multi-conv-timing-worker/v1"
SOURCE_COMMIT = "24a208140b73ed943d983ea73f2a20f842a19015"
GO_STATUS = "VALIDATED-MR5-MULTI-CONV-PRODUCTION-BRIDGE-PHYSICS"
NO_GO_STATUS = "VALIDATED-NO-GO-MR5-MULTI-CONV-PRODUCTION-BRIDGE-PHYSICS"
SITE_ORDER = ("C2", "C1", "C0")
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


def _validate_protocol(value: object) -> None:
    if value != {
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
        raise ValueError("MR5 timing protocol differs")


def _signature_map(modules: Mapping[str, Any]) -> dict[str, object]:
    return {
        site: receipt.get("signature_hash")
        for site, receipt in modules.items()
        if isinstance(receipt, Mapping)
    }


def _validate_candidate_receipt(value: object) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("MR5 timing candidate receipt absent")
    unsigned = dict(value)
    receipt_hash = unsigned.pop("receipt_hash", None)
    modules = value.get("module_receipts")
    if (
        receipt_hash != canonical_hash(unsigned)
        or value.get("site_order") != list(SITE_ORDER)
        or value.get("dummy_forward_launch_count") != 3
        or value.get("dummy_backward_launch_count") != 3
        or value.get("dummy_fallback_count") != 0
        or value.get("dummy_eager_count") != 0
        or not isinstance(modules, Mapping)
    ):
        raise ValueError("MR5 timing candidate receipt differs")
    _validate_module_receipts(modules, _signature_map(modules))


def _site_map(value: object, expected: int, *, range_max: int | None = None) -> bool:
    if not isinstance(value, Mapping) or set(value) != set(SITE_ORDER):
        return False
    if range_max is not None:
        return all(
            isinstance(item, int) and item in range(range_max + 1)
            for item in value.values()
        )
    return all(item == expected for item in value.values())


def _validate_bridge_receipt(
    value: object, candidate_receipt: Mapping[str, Any]
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("MR5 timing bridge receipt absent")
    modules = value.get("module_receipts")
    signatures = value.get("signature_hashes")
    if (
        value.get("evaluation_count") != 10
        or value.get("site_order_count") != 30
        or not _site_map(value.get("forward_launches"), 10)
        or not _site_map(value.get("backward_launches"), 9)
        or not _site_map(value.get("beta_tensor_count"), 10)
        or not _site_map(value.get("beta_numel"), 0)
        or not _site_map(value.get("handoff_content_count"), 10)
        or not _site_map(value.get("handoff_pointer_count"), 0, range_max=10)
        or not _site_map(value.get("cache_miss_count"), 0)
        or not _site_map(value.get("cache_hit_count"), 10)
        or value.get("pending_site_count") != 0
        or value.get("fallback_count") != 0
        or value.get("eager_count") != 0
        or value.get("native_shadow_count") != 0
        or value.get("prewarmed_before_outer") is not True
        or value.get("timing_recorded") is not True
        or value.get("performance_claimed") is not False
        or not isinstance(modules, Mapping)
        or modules != candidate_receipt.get("module_receipts")
        or not isinstance(signatures, Mapping)
        or signatures != _signature_map(modules)
    ):
        raise ValueError("MR5 timing bridge receipt differs")
    _validate_module_receipts(modules, signatures)


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
        raise ValueError("MR5 timing worker envelope differs")
    legacy._validate_source(worker.get("source"))
    _validate_protocol(worker.get("protocol"))
    legacy._validate_gpu_snapshot(worker.get("gpu_before"))
    legacy._validate_gpu_snapshot(worker.get("gpu_after"))
    legacy._validate_measurement(worker.get("measurement"))
    measurement = worker["measurement"]
    if (
        worker["device_before"] != measurement["device_before"]
        or worker["device_after"] != measurement["device_after"]
        or worker["stream_before"] != measurement["stream_before"]
        or worker["stream_after"] != measurement["stream_after"]
    ):
        raise ValueError("MR5 timing device/stream projection differs")
    if mode == "provider":
        if (
            worker.get("candidate_module_receipt") is not None
            or worker.get("bridge_receipt") is not None
        ):
            raise ValueError("MR5 timing provider has candidate state")
    else:
        candidate = worker.get("candidate_module_receipt")
        _validate_candidate_receipt(candidate)
        assert isinstance(candidate, Mapping)
        _validate_bridge_receipt(worker.get("bridge_receipt"), candidate)


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
        raise ValueError("MR5 timing raw provenance differs")
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
            raise ValueError("MR5 timing run order differs")
        worker = wrapper["worker"]
        validate_worker(worker, mode=expected[2])
        runs.append(wrapper)
        gpu_identities.append(
            (worker["gpu_before"]["name"], worker["gpu_before"]["driver_version"])
        )
        if expected[2] == "bridge":
            module_receipts.append(worker["candidate_module_receipt"])
    if len(set(gpu_identities)) != 1:
        raise ValueError("MR5 timing GPU identity differs across workers")
    module_hashes = {canonical_hash(receipt) for receipt in module_receipts}
    if len(module_receipts) != 6 or len(module_hashes) != 1:
        raise ValueError("MR5 timing candidate modules differ across workers")
    metrics: list[dict[str, object]] = []
    for pair_index in range(6):
        pair = [run for run in runs if run["pair_index"] == pair_index]
        provider = next(run["worker"] for run in pair if run["mode"] == "provider")
        bridge = next(run["worker"] for run in pair if run["mode"] == "bridge")
        metrics.append(legacy._pair_metric(pair_index, provider, bridge))
    host_speedups = [cast(float, metric["host_speedup"]) for metric in metrics]
    geomean = legacy._geomean(host_speedups)
    bootstrap_lower = legacy._bootstrap_lower(host_speedups)
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
        "bootstrap_seed": legacy.BOOTSTRAP_SEED,
        "bootstrap_samples": legacy.BOOTSTRAP_SAMPLES,
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
    "derive_summary",
    "validate_worker",
]
