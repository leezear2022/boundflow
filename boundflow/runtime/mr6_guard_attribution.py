"""Mechanical validation and routing gates for MR6 guard attribution."""

# pylint: disable=missing-function-docstring,protected-access,too-many-locals
# pylint: disable=too-many-branches,too-many-statements,duplicate-code
# pylint: disable=too-many-boolean-expressions

from __future__ import annotations

from typing import Any, Mapping, cast

from boundflow.runtime import mr3_production_bridge_timing as timing_math
from boundflow.runtime.mr3_provider_hook_feasibility import canonical_hash
from boundflow.runtime.mr5_multi_conv_timing import validate_worker as validate_base

WORKER_SCHEMA = "boundflow.mr6-hot-path-guard-attribution-worker/v1"
RAW_SCHEMA = "boundflow.mr6-hot-path-guard-attribution-formal/v1"
SOURCE_COMMIT = "fb3c245fc8de1be08471d91b97b026ded9ce204b"
MODES = ("provider", "full", "diagnostic")
EXPECTED_RUNS = tuple(
    (triplet, position, mode)
    for triplet, modes in enumerate(
        (
            ("provider", "full", "diagnostic"),
            ("full", "diagnostic", "provider"),
            ("diagnostic", "provider", "full"),
        )
    )
    for position, mode in enumerate(modes)
)
FULL_DIAGNOSTIC_GEOMEAN_GATE = 1.10
PROVIDER_DIAGNOSTIC_GEOMEAN_GATE = 0.98
PROVIDER_DIAGNOSTIC_WORST_GATE = 0.95
GO_STATUS = "VALIDATED-MR6-GUARD-DOMINANT-OPPORTUNITY"
NO_GO_STATUS = "VALIDATED-NO-GO-MR6-GUARD-DOMINANCE"


def validate_guard_receipt(value: object, *, mode: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("MR6 guard receipt absent")
    unsigned = dict(value)
    receipt_hash = unsigned.pop("receipt_hash", None)
    if receipt_hash != canonical_hash(unsigned) or value.get("policy") != mode:
        raise ValueError("MR6 guard receipt hash differs")
    expected = {
        "provider": {
            "site_evaluations": 0,
            "validation_calls": 0,
            "content_calls": 0,
            "input_value_guards_executed": 0,
            "handoff_content_guards_executed": 0,
            "output_finite_guards_executed": 0,
            "input_value_guards_elided": 0,
            "handoff_content_guards_elided": 0,
            "synchronizing_guards_executed": 0,
            "production_admitted": True,
        },
        "full": {
            "site_evaluations": 30,
            "validation_calls": 30,
            "content_calls": 30,
            "input_value_guards_executed": 270,
            "handoff_content_guards_executed": 30,
            "output_finite_guards_executed": 60,
            "input_value_guards_elided": 0,
            "handoff_content_guards_elided": 0,
            "synchronizing_guards_executed": 360,
            "production_admitted": True,
        },
        "diagnostic": {
            "site_evaluations": 30,
            "validation_calls": 30,
            "content_calls": 30,
            "input_value_guards_executed": 0,
            "handoff_content_guards_executed": 0,
            "output_finite_guards_executed": 60,
            "input_value_guards_elided": 270,
            "handoff_content_guards_elided": 30,
            "synchronizing_guards_executed": 60,
            "production_admitted": False,
        },
    }
    if (
        mode not in expected
        or any(value.get(key) != item for key, item in expected[mode].items())
        or value.get("performance_claimed") is not False
    ):
        raise ValueError("MR6 guard receipt differs")


def validate_worker(value: object, *, mode: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("MR6 worker absent")
    unsigned = dict(value)
    worker_hash = unsigned.pop("worker_hash", None)
    base = value.get("base_worker")
    if (
        value.get("schema_version") != WORKER_SCHEMA
        or value.get("mode") != mode
        or worker_hash != canonical_hash(unsigned)
        or value.get("timing_recorded") is not True
        or value.get("production_admitted") is not (mode != "diagnostic")
        or value.get("performance_claimed") is not False
        or not isinstance(base, Mapping)
    ):
        raise ValueError("MR6 worker envelope differs")
    validate_guard_receipt(value.get("guard_receipt"), mode=mode)
    validate_base(base, mode="provider" if mode == "provider" else "bridge")
    return base


def _metric(
    triplet: int,
    numerator: Mapping[str, Any],
    denominator: Mapping[str, Any],
) -> dict[str, object]:
    return timing_math._pair_metric(triplet, numerator, denominator)


def derive_summary(raw: Mapping[str, Any]) -> dict[str, object]:
    unsigned = dict(raw)
    raw_hash = unsigned.pop("raw_hash", None)
    runs_value = raw.get("runs")
    if (
        raw.get("schema_version") != RAW_SCHEMA
        or raw.get("source_commit") != SOURCE_COMMIT
        or raw.get("run_order") != [list(item) for item in EXPECTED_RUNS]
        or raw_hash != canonical_hash(unsigned)
        or not isinstance(runs_value, list)
        or len(runs_value) != len(EXPECTED_RUNS)
    ):
        raise ValueError("MR6 raw provenance differs")
    runs: list[Mapping[str, Any]] = []
    module_hashes: set[str] = set()
    gpu_identities: set[tuple[object, object]] = set()
    for expected, wrapper in zip(EXPECTED_RUNS, runs_value):
        if (
            not isinstance(wrapper, Mapping)
            or (
                wrapper.get("triplet_index"),
                wrapper.get("position"),
                wrapper.get("mode"),
            )
            != expected
        ):
            raise ValueError("MR6 run order differs")
        mode = cast(str, wrapper["mode"])
        base = validate_worker(wrapper.get("worker"), mode=mode)
        gpu_identities.add(
            (base["gpu_before"]["name"], base["gpu_before"]["driver_version"])
        )
        if mode != "provider":
            module_hashes.add(canonical_hash(base["candidate_module_receipt"]))
        runs.append({**wrapper, "base": base})
    if len(gpu_identities) != 1 or len(module_hashes) != 1:
        raise ValueError("MR6 physical/module identity differs")

    metrics: list[dict[str, object]] = []
    for triplet_index in range(3):
        triplet = [row for row in runs if row["triplet_index"] == triplet_index]
        by_mode = {
            cast(str, row["mode"]): cast(Mapping[str, Any], row["base"])
            for row in triplet
        }
        provider_full = _metric(triplet_index, by_mode["provider"], by_mode["full"])
        provider_diagnostic = _metric(
            triplet_index, by_mode["provider"], by_mode["diagnostic"]
        )
        full_diagnostic = _metric(triplet_index, by_mode["full"], by_mode["diagnostic"])
        metrics.append(
            {
                "triplet_index": triplet_index,
                "provider_full": provider_full,
                "provider_diagnostic": provider_diagnostic,
                "full_diagnostic": full_diagnostic,
            }
        )

    def nested(row: Mapping[str, object], name: str) -> Mapping[str, Any]:
        return cast(Mapping[str, Any], row[name])

    full_diag = [
        cast(float, nested(row, "full_diagnostic")["host_speedup"]) for row in metrics
    ]
    provider_diag = [
        cast(float, nested(row, "provider_diagnostic")["host_speedup"])
        for row in metrics
    ]
    provider_full_speedups = [
        cast(float, nested(row, "provider_full")["host_speedup"]) for row in metrics
    ]
    semantic_exact = all(
        bool(nested(row, name)["allclose"] and nested(row, name)["sign_exact"])
        for row in metrics
        for name in ("provider_full", "provider_diagnostic", "full_diagnostic")
    )
    direction_count = sum(
        bool(nested(row, name)["host_event_direction_consistent"])
        for row in metrics
        for name in ("provider_full", "provider_diagnostic", "full_diagnostic")
    )
    full_diag_geomean = timing_math._geomean(full_diag)
    provider_diag_geomean = timing_math._geomean(provider_diag)
    gates = {
        "triplet_count": len(metrics) == 3,
        "semantic_exact": semantic_exact,
        "guard_counts": True,
        "module_stability": len(module_hashes) == 1,
        "host_event_direction": direction_count == 9,
        "full_diagnostic_geomean": (full_diag_geomean >= FULL_DIAGNOSTIC_GEOMEAN_GATE),
        "provider_diagnostic_geomean": (
            provider_diag_geomean >= PROVIDER_DIAGNOSTIC_GEOMEAN_GATE
        ),
        "provider_diagnostic_worst": (
            min(provider_diag) >= PROVIDER_DIAGNOSTIC_WORST_GATE
        ),
    }
    route_open = all(gates.values())
    summary: dict[str, object] = {
        "schema_version": RAW_SCHEMA,
        "source_commit": SOURCE_COMMIT,
        "status": GO_STATUS if route_open else NO_GO_STATUS,
        "run_count": len(runs),
        "triplet_count": len(metrics),
        "triplet_metrics": metrics,
        "full_diagnostic_host_geomean": full_diag_geomean,
        "full_diagnostic_bootstrap_95_lower": timing_math._bootstrap_lower(full_diag),
        "full_diagnostic_worst": min(full_diag),
        "provider_diagnostic_host_geomean": provider_diag_geomean,
        "provider_diagnostic_bootstrap_95_lower": timing_math._bootstrap_lower(
            provider_diag
        ),
        "provider_diagnostic_worst": min(provider_diag),
        "provider_full_host_geomean": timing_math._geomean(provider_full_speedups),
        "host_event_direction_consistent_count": direction_count,
        "candidate_module_receipt_hash": next(iter(module_hashes)),
        "gates": gates,
        "safe_guard_fusion_open": route_open,
        "performance_claimed": False,
    }
    summary["summary_hash"] = canonical_hash(summary)
    return summary


__all__ = [
    "EXPECTED_RUNS",
    "GO_STATUS",
    "NO_GO_STATUS",
    "RAW_SCHEMA",
    "SOURCE_COMMIT",
    "derive_summary",
    "validate_guard_receipt",
    "validate_worker",
]
