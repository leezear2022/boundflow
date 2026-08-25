"""Deterministic MR0 explicit CUDA-event budget derivation."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from typing import Any, cast, Mapping, Sequence

MR0_SCHEMA = "boundflow.mr0-explicit-event-budget/v1"
MR0_BUDGETS = (1, 4, 8, 17)
MR0_WORKER_COUNT = 5
MR0_GROUP_COUNT = 20
MR0_REPEATS = 100
MR0_WARMUP = 20
MR0_ORDERS = ("CI", "IC", "CI", "IC", "CI")
MR0_GEOMEAN_GATE = 1.05
MR0_BOOTSTRAP_UPPER_GATE = 1.05
MR0_WORST_GATE = 1.08
MR0_BOOTSTRAP_SEED = 20260826
MR0_BOOTSTRAP_SAMPLES = 10_000


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def geomean(values: Sequence[float]) -> float:
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("MR0 geomean input differs")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def derive_budget_row(
    *,
    budget: int,
    control_ms: Sequence[float],
    instrumented_ms: Sequence[float],
) -> dict[str, object]:
    if (
        budget not in MR0_BUDGETS
        or len(control_ms) != MR0_GROUP_COUNT
        or len(instrumented_ms) != MR0_GROUP_COUNT
        or any(
            not math.isfinite(value) or value <= 0.0
            for value in (*control_ms, *instrumented_ms)
        )
    ):
        raise ValueError("MR0 budget samples differ")
    control_median = statistics.median(control_ms)
    instrumented_median = statistics.median(instrumented_ms)
    row: dict[str, object] = {
        "budget": budget,
        "control_ms": list(control_ms),
        "instrumented_ms": list(instrumented_ms),
        "control_median_ms": control_median,
        "instrumented_median_ms": instrumented_median,
        "overhead_ratio": instrumented_median / control_median,
        "logical_event_records_per_group": 2 + 2 * budget * MR0_REPEATS,
        "event_objects": 2 + 2 * max(MR0_BUDGETS),
    }
    row["row_hash"] = canonical_hash(row)
    return row


def validate_budget_row(value: Mapping[str, Any]) -> dict[str, object]:
    expected = {
        "budget",
        "control_ms",
        "instrumented_ms",
        "control_median_ms",
        "instrumented_median_ms",
        "overhead_ratio",
        "logical_event_records_per_group",
        "event_objects",
        "row_hash",
    }
    if set(value) != expected:
        raise ValueError("MR0 budget row fields differ")
    rebuilt = derive_budget_row(
        budget=int(value["budget"]),
        control_ms=[float(item) for item in value["control_ms"]],
        instrumented_ms=[float(item) for item in value["instrumented_ms"]],
    )
    if rebuilt != dict(value):
        raise ValueError("MR0 budget row derivation differs")
    return rebuilt


def bootstrap_geomean_upper(values: Sequence[float]) -> float:
    if len(values) != MR0_WORKER_COUNT:
        raise ValueError("MR0 bootstrap worker count differs")
    generator = random.Random(MR0_BOOTSTRAP_SEED)
    samples = []
    for _ in range(MR0_BOOTSTRAP_SAMPLES):
        sample = [values[generator.randrange(len(values))] for _ in values]
        samples.append(geomean(sample))
    samples.sort()
    return samples[min(len(samples) - 1, int(0.975 * len(samples)))]


def derive_summary(workers: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    if len(workers) != MR0_WORKER_COUNT:
        raise ValueError("MR0 worker count differs")
    rows_by_budget: dict[int, list[dict[str, object]]] = {
        budget: [] for budget in MR0_BUDGETS
    }
    for ordinal, worker in enumerate(workers):
        if (
            worker.get("run_ordinal") != ordinal
            or worker.get("order") != MR0_ORDERS[ordinal]
            or worker.get("semantic_admitted") is not True
            or worker.get("stream_admitted") is not True
            or worker.get("performance_claimed") is not False
        ):
            raise ValueError("MR0 worker admission differs")
        budget_rows = worker.get("budget_rows")
        if not isinstance(budget_rows, list) or len(budget_rows) != len(MR0_BUDGETS):
            raise ValueError("MR0 worker budget inventory differs")
        for expected_budget, raw_row in zip(MR0_BUDGETS, budget_rows):
            if not isinstance(raw_row, Mapping):
                raise TypeError("MR0 worker budget row differs")
            row = validate_budget_row(raw_row)
            if row["budget"] != expected_budget:
                raise ValueError("MR0 worker budget order differs")
            rows_by_budget[expected_budget].append(row)
    budget_summaries = []
    for budget in MR0_BUDGETS:
        ratios = [cast(float, row["overhead_ratio"]) for row in rows_by_budget[budget]]
        budget_summaries.append(
            {
                "budget": budget,
                "worker_overhead_ratios": ratios,
                "geomean_overhead_ratio": geomean(ratios),
                "bootstrap_95_upper": bootstrap_geomean_upper(ratios),
                "worst_worker_overhead_ratio": max(ratios),
                "decision_budget": budget == max(MR0_BUDGETS),
            }
        )
    decision = cast(dict[str, Any], budget_summaries[-1])
    gates = {
        "geomean": cast(float, decision["geomean_overhead_ratio"]) <= MR0_GEOMEAN_GATE,
        "bootstrap_upper": cast(float, decision["bootstrap_95_upper"])
        <= MR0_BOOTSTRAP_UPPER_GATE,
        "worst_worker": cast(float, decision["worst_worker_overhead_ratio"])
        <= MR0_WORST_GATE,
        "all_workers_semantic": True,
        "all_workers_stream": True,
    }
    admitted = all(gates.values())
    result: dict[str, object] = {
        "schema_version": MR0_SCHEMA,
        "worker_count": MR0_WORKER_COUNT,
        "pair_group_count": MR0_WORKER_COUNT * len(MR0_BUDGETS) * MR0_GROUP_COUNT,
        "replay_count_per_side": MR0_WORKER_COUNT
        * len(MR0_BUDGETS)
        * MR0_GROUP_COUNT
        * MR0_REPEATS,
        "budget_summaries": budget_summaries,
        "decision_budget": max(MR0_BUDGETS),
        "gates": gates,
        "mr0_admitted": admitted,
        "verdict": (
            "VALIDATED-MR0-EXPLICIT-EVENT-BUDGET"
            if admitted
            else "VALIDATED-NO-GO-MR0-EXPLICIT-EVENT-BUDGET"
        ),
        "mr1_internal_boundary_correctness_open": admitted,
        "same_solver_open": False,
        "r2_open": False,
        "performance_claimed": False,
    }
    result["summary_hash"] = canonical_hash(result)
    return result


__all__ = [
    "MR0_BOOTSTRAP_SAMPLES",
    "MR0_BOOTSTRAP_SEED",
    "MR0_BOOTSTRAP_UPPER_GATE",
    "MR0_BUDGETS",
    "MR0_GEOMEAN_GATE",
    "MR0_GROUP_COUNT",
    "MR0_ORDERS",
    "MR0_REPEATS",
    "MR0_WARMUP",
    "MR0_WORKER_COUNT",
    "MR0_WORST_GATE",
    "bootstrap_geomean_upper",
    "canonical_hash",
    "derive_budget_row",
    "derive_summary",
    "geomean",
    "validate_budget_row",
]
