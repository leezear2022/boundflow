#!/usr/bin/env python3
"""Run fully re-signed semantic tamper probes for the MR3 formal artifact."""

# pylint: disable=missing-function-docstring,too-many-statements
# pylint: disable=wrong-import-position

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boundflow.runtime.mr3_production_bridge_formal import (  # noqa: E402
    canonical_hash,
    derive_summary,
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("MR3 tamper raw root differs")
    return value


def _resign(raw: dict[str, Any], worker_indexes: tuple[int, ...] = ()) -> None:
    for index in worker_indexes:
        worker = raw["runs"][index]["worker"]
        unsigned_worker = dict(worker)
        unsigned_worker.pop("worker_hash", None)
        worker["worker_hash"] = canonical_hash(unsigned_worker)
    unsigned_raw = dict(raw)
    unsigned_raw.pop("raw_hash", None)
    raw["raw_hash"] = canonical_hash(unsigned_raw)


def _expect_rejected(
    original: dict[str, Any],
    name: str,
    mutate: Callable[[dict[str, Any]], tuple[int, ...]],
) -> dict[str, object]:
    candidate = deepcopy(original)
    worker_indexes = mutate(candidate)
    _resign(candidate, worker_indexes)
    try:
        derive_summary(candidate)
    except (ValueError, TypeError, KeyError) as error:
        return {"name": name, "rejected": True, "error": str(error)}
    raise AssertionError(f"MR3 tamper unexpectedly admitted: {name}")


def run(raw: dict[str, Any]) -> dict[str, object]:
    cases: list[tuple[str, Callable[[dict[str, Any]], tuple[int, ...]]]] = []

    def case(name: str):
        def register(function: Callable[[dict[str, Any]], tuple[int, ...]]):
            cases.append((name, function))
            return function

        return register

    @case("source_commit")
    def _(value):
        value["source_commit"] = "f" * 40
        return ()

    @case("run_order")
    def _(value):
        value["run_order"][0], value["run_order"][1] = (
            value["run_order"][1],
            value["run_order"][0],
        )
        return ()

    @case("wrapper_mode")
    def _(value):
        value["runs"][0]["mode"] = "bridge"
        return ()

    @case("worker_source")
    def _(value):
        value["runs"][0]["worker"]["source"]["abcrown_commit"] = "0" * 40
        return (0,)

    @case("model_digest")
    def _(value):
        value["runs"][0]["worker"]["source"]["model_sha256"] = "0" * 64
        return (0,)

    @case("timing_claim")
    def _(value):
        value["runs"][1]["worker"]["timing_recorded"] = True
        return (1,)

    @case("solver_verdict")
    def _(value):
        value["runs"][1]["worker"]["solver_result"]["status"] = "unknown"
        return (1,)

    @case("forward_count")
    def _(value):
        value["runs"][1]["worker"]["bridge_receipt"]["forward_launch_count"] = 9
        return (1,)

    @case("fallback_count")
    def _(value):
        value["runs"][1]["worker"]["bridge_receipt"]["fallback_count"] = 1
        return (1,)

    @case("region_value")
    def _(value):
        value["runs"][1]["worker"]["region_states"][0]["lower_a"]["values"][0] += 0.1
        return (1,)

    @case("aggregate_loss")
    def _(value):
        value["runs"][1]["worker"]["evaluation_trajectory"][0]["aggregate_loss"] += 0.1
        return (1,)

    @case("gradient")
    def _(value):
        value["runs"][1]["worker"]["mutation_trajectory"][0]["gradient"]["values"][
            0
        ] += 0.01
        return (1,)

    @case("adam_moment")
    def _(value):
        value["runs"][1]["worker"]["mutation_trajectory"][0]["exp_avg"]["values"][
            0
        ] += 0.01
        return (1,)

    @case("learning_rate")
    def _(value):
        value["runs"][1]["worker"]["mutation_trajectory"][0]["lr_used"] = 0.02
        return (1,)

    @case("clamp_mask")
    def _(value):
        value["runs"][1]["worker"]["mutation_trajectory"][0]["clamp_mask"][
            "zero_count"
        ] += 1
        return (1,)

    @case("final_alpha")
    def _(value):
        value["runs"][1]["worker"]["final_target_alpha_state"]["values"][0] += 0.01
        return (1,)

    @case("atomic_commit")
    def _(value):
        value["runs"][1]["worker"]["atomic_receipt"]["atomic_commit_count"] = 0
        return (1,)

    @case("rollback_pointer")
    def _(value):
        value["rollback_probe"]["atomic_receipt"]["owner_pointer_hash_after"] = "f" * 64
        unsigned = dict(value["rollback_probe"])
        unsigned.pop("worker_hash", None)
        value["rollback_probe"]["worker_hash"] = canonical_hash(unsigned)
        return ()

    results = [_expect_rejected(raw, name, mutate) for name, mutate in cases]
    report: dict[str, object] = {
        "case_count": len(results),
        "rejected_count": sum(bool(item["rejected"]) for item in results),
        "fully_resigned": True,
        "results": results,
    }
    report["report_hash"] = canonical_hash(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(_load(args.artifact / "raw.json"))
    args.output.write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"rejected={report['rejected_count']}/{report['case_count']}")


if __name__ == "__main__":
    main()
