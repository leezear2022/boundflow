"""Mechanical validation and routing for MR7-R unprofiled host recovery."""

# pylint: disable=missing-function-docstring,too-many-locals,too-many-branches
# pylint: disable=too-many-statements,too-many-boolean-expressions,duplicate-code
# pylint: disable=protected-access

from __future__ import annotations

import statistics
from typing import Any, Mapping, Sequence, cast

from boundflow.runtime import mr3_production_bridge_timing as timing_math
from boundflow.runtime.mr3_provider_hook_feasibility import canonical_hash
from boundflow.runtime.mr5_multi_conv_timing import validate_worker as validate_base
from boundflow.runtime.mr6_guard_attribution import (
    validate_guard_receipt,
    validate_worker as validate_mr6_worker,
)
from boundflow.runtime.mr7_launch_materialization_attribution import (
    WORKER_SCHEMA as MR7_WORKER_SCHEMA,
    required_region_speedup,
    validate_host_receipt,
)

RAW_SCHEMA = "boundflow.mr7r-unprofiled-host-recovery-formal/v1"
EXPECTED_RUNS = tuple(
    (pair, position, role)
    for pair, roles in enumerate(
        (
            ("baseline", "ledger"),
            ("ledger", "baseline"),
            ("baseline", "ledger"),
            ("ledger", "baseline"),
            ("baseline", "ledger"),
        )
    )
    for position, role in enumerate(roles)
)
PAIR_COUNT = 5
PARITY_TARGET = 1.107412
HOST_RATIO_MEDIAN_MIN = 0.95
HOST_RATIO_MEDIAN_MAX = 1.05
HOST_RATIO_RUN_MIN = 0.90
HOST_RATIO_RUN_MAX = 1.10
HOST_CLOSURE_ERROR_MAX = 0.02
BOUNDARY_SHARE_MIN = 0.15
BOUNDARY_ABSOLUTE_NS_MIN = 15_000_000
BOUNDARY_QUALIFYING_RUNS_MIN = 4
MAXIMUM_REQUIRED_REGION_SPEEDUP = 10.0
INVALID_STATUS = "INVALID_MR7R_LEDGER_PERTURBATION"
OPPORTUNITY_STATUS = "VALIDATED_MR7R_HOST_BOUNDARY_OPPORTUNITY"
NO_GO_STATUS = "VALIDATED_NO_GO_MR7R_HOST_BOUNDARY"

_BOUNDARY_CATEGORIES = (
    "ffi_dlpack_stream",
    "layout_materialization",
    "post_output_guard",
)
_EXPECTED_LAUNCH_MARKERS = {
    "forward.C0": 10,
    "forward.C1": 10,
    "forward.C2": 10,
    "backward.C0": 9,
    "backward.C1": 9,
    "backward.C2": 9,
}
_EXPECTED_HOST_CALLS = {
    "admission_handoff": 150,
    "layout_materialization": 117,
    "ffi_dlpack_stream": 57,
    "post_output_guard": 30,
    "optimizer_and_residual": 0,
}


def _median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("MR7-R median input absent")
    return float(statistics.median(values))


def _validate_ledger_worker(
    value: object,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if not isinstance(value, Mapping):
        raise ValueError("MR7-R ledger worker absent")
    unsigned = dict(value)
    worker_hash = unsigned.pop("worker_hash", None)
    base = value.get("base_worker")
    host = validate_host_receipt(value.get("host_receipt"))
    events = value.get("device_events")
    marker_totals = value.get("device_marker_totals")
    if (
        value.get("schema_version") != MR7_WORKER_SCHEMA
        or value.get("kind") != "control"
        or worker_hash != canonical_hash(unsigned)
        or not isinstance(base, Mapping)
        or value.get("launch_marker_counts") != _EXPECTED_LAUNCH_MARKERS
        or value.get("device_event_hash") != canonical_hash([])
        or events != []
        or marker_totals != {}
        or value.get("timing_recorded") is not True
        or value.get("production_admitted") is not False
        or value.get("performance_claimed") is not False
        or host.get("category_calls") != _EXPECTED_HOST_CALLS
        or host.get("outer_host_ns") != base.get("measurement", {}).get("host_ns")
    ):
        raise ValueError("MR7-R ledger worker differs")
    validate_guard_receipt(value.get("guard_receipt"), mode="diagnostic")
    validate_base(base, mode="bridge")
    return base, host


def _module_hash(base: Mapping[str, Any]) -> str:
    receipt = base.get("candidate_module_receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("MR7-R candidate module receipt absent")
    return canonical_hash(receipt)


def _physical_identity(base: Mapping[str, Any]) -> tuple[object, object, object]:
    gpu = base.get("gpu_before")
    if not isinstance(gpu, Mapping):
        raise ValueError("MR7-R GPU identity absent")
    return gpu.get("name"), gpu.get("driver_version"), base.get("device_before")


def _pair_metric(
    pair_index: int,
    baseline: Mapping[str, Any],
    ledger: Mapping[str, Any],
    host: Mapping[str, Any],
) -> dict[str, object]:
    semantic = timing_math._pair_metric(pair_index, baseline, ledger)
    baseline_measurement = cast(Mapping[str, Any], baseline["measurement"])
    ledger_measurement = cast(Mapping[str, Any], ledger["measurement"])
    baseline_host_ns = cast(int, baseline_measurement["host_ns"])
    ledger_host_ns = cast(int, ledger_measurement["host_ns"])
    baseline_event_ms = cast(float, baseline_measurement["cuda_event_ms"])
    ledger_event_ms = cast(float, ledger_measurement["cuda_event_ms"])
    host_ratio = ledger_host_ns / baseline_host_ns
    event_ratio = ledger_event_ms / baseline_event_ms
    category_ns = cast(Mapping[str, int], host["category_ns"])
    boundary_ns = sum(category_ns[name] for name in _BOUNDARY_CATEGORIES)
    boundary_share = boundary_ns / ledger_host_ns
    metric: dict[str, object] = {
        "pair_index": pair_index,
        "semantic_allclose": semantic["allclose"],
        "semantic_sign_exact": semantic["sign_exact"],
        "semantic_maximum_absolute_difference": semantic[
            "semantic_maximum_absolute_difference"
        ],
        "semantic_element_count": semantic["semantic_element_count"],
        "baseline_host_ns": baseline_host_ns,
        "ledger_host_ns": ledger_host_ns,
        "ledger_baseline_host_ratio": host_ratio,
        "baseline_cuda_event_ms": baseline_event_ms,
        "ledger_cuda_event_ms": ledger_event_ms,
        "ledger_baseline_cuda_event_ratio": event_ratio,
        "host_event_direction_consistent": (host_ratio >= 1.0) == (event_ratio >= 1.0),
        "host_closure_error_ratio": host["closure_error_ratio"],
        "host_category_ns": dict(category_ns),
        "boundary_ns": boundary_ns,
        "boundary_share": boundary_share,
        "boundary_share_qualifies": boundary_share >= BOUNDARY_SHARE_MIN,
        "boundary_absolute_qualifies": boundary_ns >= BOUNDARY_ABSOLUTE_NS_MIN,
    }
    metric["metric_hash"] = canonical_hash(metric)
    return metric


def derive_summary(raw: Mapping[str, Any]) -> dict[str, object]:
    unsigned = dict(raw)
    raw_hash = unsigned.pop("raw_hash", None)
    runs_value = raw.get("runs")
    source_commit = raw.get("source_commit")
    expected_order = [list(item) for item in EXPECTED_RUNS]
    if (
        raw.get("schema_version") != RAW_SCHEMA
        or not isinstance(source_commit, str)
        or len(source_commit) != 40
        or raw.get("run_order") != expected_order
        or raw_hash != canonical_hash(unsigned)
        or not isinstance(runs_value, list)
        or len(runs_value) != len(EXPECTED_RUNS)
    ):
        raise ValueError("MR7-R raw provenance differs")

    pair_workers: dict[int, dict[str, Mapping[str, Any]]] = {}
    pair_hosts: dict[int, Mapping[str, Any]] = {}
    module_hashes: set[str] = set()
    physical_identities: set[tuple[object, object, object]] = set()
    for expected, wrapper in zip(expected_order, runs_value):
        if (
            not isinstance(wrapper, Mapping)
            or [
                wrapper.get("pair_index"),
                wrapper.get("position"),
                wrapper.get("role"),
            ]
            != expected
        ):
            raise ValueError("MR7-R run order differs")
        pair_index = cast(int, wrapper["pair_index"])
        role = cast(str, wrapper["role"])
        worker = wrapper.get("worker")
        if role == "baseline":
            base = validate_mr6_worker(worker, mode="diagnostic")
        elif role == "ledger":
            base, host = _validate_ledger_worker(worker)
            pair_hosts[pair_index] = host
        else:
            raise ValueError("MR7-R worker role differs")
        pair_workers.setdefault(pair_index, {})[role] = base
        module_hashes.add(_module_hash(base))
        physical_identities.add(_physical_identity(base))
    if (
        set(pair_workers) != set(range(PAIR_COUNT))
        or set(pair_hosts) != set(range(PAIR_COUNT))
        or any(
            set(workers) != {"baseline", "ledger"} for workers in pair_workers.values()
        )
        or len(module_hashes) != 1
        or len(physical_identities) != 1
    ):
        raise ValueError("MR7-R pair/module/physical identity differs")

    pair_metrics = [
        _pair_metric(
            pair,
            pair_workers[pair]["baseline"],
            pair_workers[pair]["ledger"],
            pair_hosts[pair],
        )
        for pair in range(PAIR_COUNT)
    ]
    host_ratios = [
        cast(float, metric["ledger_baseline_host_ratio"]) for metric in pair_metrics
    ]
    boundary_shares = [cast(float, metric["boundary_share"]) for metric in pair_metrics]
    boundary_ns_values = [
        float(cast(int, metric["boundary_ns"])) for metric in pair_metrics
    ]
    host_ratio_median = _median(host_ratios)
    boundary_share_median = _median(boundary_shares)
    boundary_ns_median = _median(boundary_ns_values)
    direction_count = sum(
        bool(metric["host_event_direction_consistent"]) for metric in pair_metrics
    )
    closure_count = sum(
        cast(float, metric["host_closure_error_ratio"]) <= HOST_CLOSURE_ERROR_MAX
        for metric in pair_metrics
    )
    qualifying_count = sum(
        bool(
            metric["boundary_share_qualifies"] and metric["boundary_absolute_qualifies"]
        )
        for metric in pair_metrics
    )
    semantic_exact = all(
        bool(metric["semantic_allclose"] and metric["semantic_sign_exact"])
        for metric in pair_metrics
    )
    perturbation_gates = {
        "pair_count": len(pair_metrics) == PAIR_COUNT,
        "semantic_exact": semantic_exact,
        "launch_cache_module_stream_fallback_exact": True,
        "host_closure": closure_count == PAIR_COUNT,
        "host_ratio_median": (
            HOST_RATIO_MEDIAN_MIN <= host_ratio_median <= HOST_RATIO_MEDIAN_MAX
        ),
        "host_ratio_worst": all(
            HOST_RATIO_RUN_MIN <= ratio <= HOST_RATIO_RUN_MAX for ratio in host_ratios
        ),
        "host_event_direction": direction_count == PAIR_COUNT,
    }
    ledger_attribution_valid = all(perturbation_gates.values())
    required = required_region_speedup(
        share=boundary_share_median, target=PARITY_TARGET
    )
    opportunity_gates = {
        "ledger_attribution_valid": ledger_attribution_valid,
        "boundary_share_median": boundary_share_median >= BOUNDARY_SHARE_MIN,
        "boundary_absolute_median": boundary_ns_median >= BOUNDARY_ABSOLUTE_NS_MIN,
        "boundary_qualifying_runs": qualifying_count >= BOUNDARY_QUALIFYING_RUNS_MIN,
        "amdahl_reachable": required is not None
        and required <= MAXIMUM_REQUIRED_REGION_SPEEDUP,
    }
    opportunity_open = all(opportunity_gates.values())
    status = (
        INVALID_STATUS
        if not ledger_attribution_valid
        else OPPORTUNITY_STATUS if opportunity_open else NO_GO_STATUS
    )
    summary: dict[str, object] = {
        "schema_version": RAW_SCHEMA,
        "source_commit": source_commit,
        "status": status,
        "run_count": len(runs_value),
        "pair_count": len(pair_metrics),
        "pair_metrics": pair_metrics,
        "ledger_baseline_host_ratios": host_ratios,
        "ledger_baseline_host_ratio_geomean": timing_math._geomean(host_ratios),
        "ledger_baseline_host_ratio_median": host_ratio_median,
        "ledger_baseline_host_ratio_minimum": min(host_ratios),
        "ledger_baseline_host_ratio_maximum": max(host_ratios),
        "host_event_direction_consistent_count": direction_count,
        "host_closure_qualifying_count": closure_count,
        "boundary_shares": boundary_shares,
        "boundary_ns": [int(value) for value in boundary_ns_values],
        "boundary_share_median": boundary_share_median,
        "boundary_ns_median": boundary_ns_median,
        "boundary_qualifying_run_count": qualifying_count,
        "required_parity_region_speedup": required,
        "parity_target": PARITY_TARGET,
        "candidate_module_receipt_hash": next(iter(module_hashes)),
        "perturbation_gates": perturbation_gates,
        "opportunity_gates": opportunity_gates,
        "ledger_attribution_valid": ledger_attribution_valid,
        "compiled_region_correctness_open": opportunity_open,
        "timing_open": False,
        "performance_claimed": False,
    }
    summary["summary_hash"] = canonical_hash(summary)
    return summary


__all__ = [
    "EXPECTED_RUNS",
    "INVALID_STATUS",
    "NO_GO_STATUS",
    "OPPORTUNITY_STATUS",
    "RAW_SCHEMA",
    "derive_summary",
]
