#!/usr/bin/env python3
"""Attribute the fully warm-matched BAB4 complete-verifier prelude gap."""

# pylint: disable=protected-access,too-many-locals,wrong-import-position
# pylint: disable=too-many-statements,too-many-boolean-expressions

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from typing import Any, Mapping

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts import run_asplos27_s4_same_solver_five_fresh as formal

CONTROL = "B4-A-WARM"
CANDIDATE = "BAB4-WARM"
PAIR_ORDERS = (
    (CONTROL, CANDIDATE),
    (CANDIDATE, CONTROL),
    (CONTROL, CANDIDATE),
)
PHASES = (
    "complete_verifier",
    "cuda_empty_cache",
    "gc_collect",
    "complete_bab",
    "prepare_for_act_bab",
    "general_bab",
)


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"BAB4 complete-prelude {label} differs")
    return value


def _integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise TypeError(f"BAB4 complete-prelude {label} differs")
    return value


def _phase(payload: Mapping[str, Any], name: str) -> tuple[int, int]:
    diagnostics = _mapping(payload.get("diagnostics"), "diagnostics")
    timings = _mapping(
        diagnostics.get("complete_prelude_timings"), "complete prelude timings"
    )
    aggregates = _mapping(timings.get("aggregates"), "aggregates")
    row = _mapping(aggregates.get(name), f"phase {name}")
    return (
        _integer(row.get("inclusive_ns"), f"{name} inclusive"),
        _integer(row.get("exclusive_ns"), f"{name} exclusive"),
    )


def _gc_receipt(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    diagnostics = _mapping(payload.get("diagnostics"), "diagnostics")
    receipt = _mapping(diagnostics.get("prepared_gc_isolation"), "GC receipt")
    if (
        receipt.get("schema_version") != "boundflow.prepared-gc-isolation/v1"
        or receipt.get("full_prepare_collection") is not True
        or receipt.get("prepared_old_generation_scan_excluded") is not True
        or receipt.get("query_collection_preserved") is not True
        or receipt.get("query_timing_excluded") is not True
        or receipt.get("query_collect_generation") != 1
        or receipt.get("query_collect_call_count") != 1
        or receipt.get("restored") is not True
        or receipt.get("performance_claimed") is not False
    ):
        raise ValueError("BAB4 complete-prelude GC receipt differs")
    for name in (
        "prepare_collect_ns",
        "query_collect_ns",
        "restore_collect_ns",
        "prepare_collected_object_count",
        "query_collected_object_count",
        "restore_collected_object_count",
    ):
        _integer(receipt.get(name), f"GC receipt {name}")
    return receipt


def _run_worker(
    args: argparse.Namespace,
    *,
    pair_ordinal: int,
    sequence_position: int,
    configuration: str,
) -> Mapping[str, Any]:
    output = args.output / f"pair-{pair_ordinal:02d}" / configuration
    output.mkdir(parents=True, exist_ok=False)
    attempts: list[dict[str, object]] = []
    for attempt in range(args.max_environment_attempts):
        attempt_output = output / f"attempt-{attempt:02d}"
        attempt_output.mkdir()
        result = attempt_output / "worker.json"
        command = (
            str(args.python),
            str(REPOSITORY_ROOT / "scripts/run_asplos27_s4_same_solver_worker.py"),
            "--configuration",
            configuration,
            "--mode",
            "control",
            "--run-id",
            f"bab4-prelude-{pair_ordinal:02d}-{configuration.lower()}-{attempt:02d}",
            "--block-index",
            str(pair_ordinal),
            "--sequence-position",
            str(sequence_position),
            "--benchmark-root",
            str(args.benchmark_root),
            "--abcrown-root",
            str(args.abcrown_root),
            "--model",
            str(args.model),
            "--property",
            str(args.property),
            "--result",
            str(result),
            "--attribute-complete-prelude",
        )
        completed = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            env=os.environ.copy(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        stdout_path = attempt_output / "stdout.txt"
        stderr_path = attempt_output / "stderr.txt"
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(
                f"BAB4 complete-prelude worker failed: {configuration}: "
                f"{completed.stderr[-2000:]}"
            )
        payload = _mapping(json.loads(result.read_text(encoding="utf-8")), "worker")
        run = _mapping(payload.get("run"), "run")
        environment = _mapping(run.get("environment"), "environment")
        admitted = environment.get("admitted") is True
        attempts.append({"attempt": attempt, "environment_admitted": admitted})
        if not admitted:
            continue
        for phase in PHASES:
            _phase(payload, phase)
        if configuration.endswith("-GC"):
            _gc_receipt(payload)
        for source in (result, stdout_path, stderr_path):
            shutil.copy2(source, output / source.name)
        (output / "selection.json").write_text(
            json.dumps(
                {
                    "selected_attempt": attempt,
                    "attempts": attempts,
                    "all_attempts_preserved": True,
                },
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return payload
    raise RuntimeError("BAB4 complete-prelude environment did not admit")


def _geomean(values: list[float]) -> float:
    if not values or any(value <= 0 for value in values):
        raise ValueError("BAB4 complete-prelude geomean input differs")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _summarize(
    workers: Mapping[tuple[int, str], Mapping[str, Any]],
) -> dict[str, object]:
    pairs: list[dict[str, object]] = []
    deltas: dict[str, list[float]] = {}
    query_speedups: list[float] = []
    core_speedups: list[float] = []
    allocated_ratios: list[float] = []
    reserved_ratios: list[float] = []
    semantic_rows: list[dict[str, Any]] = []
    for pair_ordinal, order in enumerate(PAIR_ORDERS):
        control = workers[(pair_ordinal, CONTROL)]
        candidate = workers[(pair_ordinal, CANDIDATE)]
        control_run = _mapping(control.get("run"), "control run")
        candidate_run = _mapping(candidate.get("run"), "candidate run")
        control_metrics = _mapping(control_run.get("metrics"), "control metrics")
        candidate_metrics = _mapping(candidate_run.get("metrics"), "candidate metrics")
        control_diagnostics = _mapping(
            control.get("diagnostics"), "control diagnostics"
        )
        candidate_diagnostics = _mapping(
            candidate.get("diagnostics"), "candidate diagnostics"
        )
        control_query_phase = _mapping(
            control_diagnostics.get("query_phase_timing"), "control query phase"
        )
        candidate_query_phase = _mapping(
            candidate_diagnostics.get("query_phase_timing"), "candidate query phase"
        )
        control_query = _integer(control_metrics.get("query_wall_ns"), "control query")
        candidate_query = _integer(
            candidate_metrics.get("query_wall_ns"), "candidate query"
        )
        control_core = _integer(control_metrics.get("core_wall_ns"), "control core")
        candidate_core = _integer(
            candidate_metrics.get("core_wall_ns"), "candidate core"
        )
        query_speedups.append(control_query / candidate_query)
        core_speedups.append(control_core / candidate_core)
        control_allocated = _integer(
            control_metrics.get("peak_allocated_bytes"), "control allocated"
        )
        candidate_allocated = _integer(
            candidate_metrics.get("peak_allocated_bytes"), "candidate allocated"
        )
        control_reserved = _integer(
            control_metrics.get("peak_reserved_bytes"), "control reserved"
        )
        candidate_reserved = _integer(
            candidate_metrics.get("peak_reserved_bytes"), "candidate reserved"
        )
        allocated_ratios.append(candidate_allocated / control_allocated)
        reserved_ratios.append(candidate_reserved / control_reserved)
        semantics = formal._semantic_pair(
            _mapping(control_run.get("semantics"), "control semantics"),
            _mapping(candidate_run.get("semantics"), "candidate semantics"),
        )
        semantic_rows.append(semantics)
        pair: dict[str, object] = {
            "pair_ordinal": pair_ordinal,
            "order": list(order),
            "query_speedup": control_query / candidate_query,
            "core_speedup": control_core / candidate_core,
            "peak_allocated_ratio": candidate_allocated / control_allocated,
            "peak_reserved_ratio": candidate_reserved / control_reserved,
            **semantics,
        }
        pre_delta = (
            _integer(candidate_query_phase.get("pre_core_ns"), "candidate pre-core")
            - _integer(control_query_phase.get("pre_core_ns"), "control pre-core")
        ) / 1e6
        deltas.setdefault("pre_core_ms", []).append(pre_delta)
        pair["pre_core_candidate_minus_control_ms"] = pre_delta
        for phase in PHASES:
            control_inclusive, control_exclusive = _phase(control, phase)
            candidate_inclusive, candidate_exclusive = _phase(candidate, phase)
            for suffix, candidate_value, control_value in (
                ("inclusive", candidate_inclusive, control_inclusive),
                ("exclusive", candidate_exclusive, control_exclusive),
            ):
                key = f"{phase}_{suffix}_ms"
                delta = (candidate_value - control_value) / 1e6
                deltas.setdefault(key, []).append(delta)
                pair[f"{key}_candidate_minus_control"] = delta
        if CONTROL.endswith("-GC") and CANDIDATE.endswith("-GC"):
            control_gc = _gc_receipt(control)
            candidate_gc = _gc_receipt(candidate)
            for name in (
                "prepare_collect_ns",
                "query_collect_ns",
                "restore_collect_ns",
            ):
                key = f"gc_receipt_{name}_ms"
                delta = (
                    _integer(candidate_gc.get(name), f"candidate {name}")
                    - _integer(control_gc.get(name), f"control {name}")
                ) / 1e6
                deltas.setdefault(key, []).append(delta)
                pair[f"{key}_candidate_minus_control"] = delta
        pairs.append(pair)
    summary: dict[str, object] = {
        "schema_version": "boundflow.bab4-complete-prelude-attribution/v1",
        "pair_count": len(PAIR_ORDERS),
        "control_configuration": CONTROL,
        "candidate_configuration": CANDIDATE,
        "query_speedup_geomean": _geomean(query_speedups),
        "core_speedup_geomean": _geomean(core_speedups),
        "all_discrete_semantics_exact": all(
            bool(row["discrete_semantics_exact"]) for row in semantic_rows
        ),
        "lower_max_abs_diff": max(
            float(row["lower_max_abs_diff"]) for row in semantic_rows
        ),
        "lower_sign_exact": all(bool(row["lower_sign_exact"]) for row in semantic_rows),
        "peak_allocated_ratio_geomean": _geomean(allocated_ratios),
        "peak_allocated_ratio_worst": max(allocated_ratios),
        "peak_reserved_ratio_geomean": _geomean(reserved_ratios),
        "peak_reserved_ratio_worst": max(reserved_ratios),
        "candidate_minus_control_median_ms": {
            key: statistics.median(values) for key, values in sorted(deltas.items())
        },
        "candidate_minus_control_pairs_ms": dict(sorted(deltas.items())),
        "pairs": pairs,
        "profile_timing_claimed": False,
        "performance_claimed": False,
    }
    summary["summary_hash"] = formal._canonical_hash(summary)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--abcrown-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--property", type=Path, required=True)
    parser.add_argument("--max-environment-attempts", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    """Run three alternating fresh pairs and emit an attribution-only summary."""

    args = _parse_args()
    args.python = args.python.absolute()
    if args.max_environment_attempts <= 0:
        raise ValueError("BAB4 complete-prelude retry count must be positive")
    for name in ("output", "benchmark_root", "abcrown_root", "model", "property"):
        setattr(args, name, getattr(args, name).resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    workers: dict[tuple[int, str], Mapping[str, Any]] = {}
    for pair_ordinal, order in enumerate(PAIR_ORDERS):
        for sequence_position, configuration in enumerate(order):
            workers[(pair_ordinal, configuration)] = _run_worker(
                args,
                pair_ordinal=pair_ordinal,
                sequence_position=sequence_position,
                configuration=configuration,
            )
    summary = _summarize(workers)
    (args.output / "summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
