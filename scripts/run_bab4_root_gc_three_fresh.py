#!/usr/bin/env python3
"""Run three alternating cumulative root-CROWN plus BAB4 GC pairs."""

# pylint: disable=too-many-locals,too-many-statements,wrong-import-position
# pylint: disable=protected-access,duplicate-code,too-many-boolean-expressions

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping, cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts import run_asplos27_s4_same_solver_five_fresh as formal

WORKER = REPOSITORY_ROOT / "scripts/run_bab4_root_gc_worker.py"
CONTROL = "B4-A-GC"
CANDIDATE = "BAB4-GC-ROOT"
PAIR_ORDERS = (
    (CONTROL, CANDIDATE),
    (CANDIDATE, CONTROL),
    (CONTROL, CANDIDATE),
)


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"BAB4 cumulative root {label} differs")
    return cast(Mapping[str, Any], value)


def _integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise TypeError(f"BAB4 cumulative root {label} differs")
    return value


def _geomean(values: list[float]) -> float:
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("BAB4 cumulative root geomean input differs")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _duration(payload: Mapping[str, Any], scope: str) -> int:
    run = _mapping(payload.get("run"), "run")
    metrics = _mapping(run.get("metrics"), "metrics")
    if scope == "query":
        return _integer(metrics.get("query_wall_ns"), "query")
    if scope == "core":
        return _integer(metrics.get("core_wall_ns"), "core")
    diagnostics = _mapping(payload.get("diagnostics"), "diagnostics")
    root = _mapping(diagnostics.get("root_incomplete_timings"), "root timings")
    aggregates = _mapping(root.get("aggregates"), "root aggregates")
    row = _mapping(aggregates.get("root_incomplete"), "root incomplete")
    return _integer(row.get("inclusive_ns"), "root incomplete")


def _run_worker(
    args: argparse.Namespace,
    *,
    pair: int,
    sequence: int,
    configuration: str,
) -> Mapping[str, Any]:
    output = args.output / f"pair-{pair:02d}" / configuration
    output.mkdir(parents=True, exist_ok=False)
    attempts: list[dict[str, object]] = []
    for attempt in range(args.max_environment_attempts):
        attempt_dir = output / f"attempt-{attempt:02d}"
        attempt_dir.mkdir()
        result = attempt_dir / "worker.json"
        command = [
            str(args.python),
            str(WORKER),
            "--configuration",
            configuration,
            "--run-id",
            f"bab4-root-{pair:02d}-{configuration.lower()}-{attempt:02d}",
            "--block-index",
            str(pair),
            "--sequence-position",
            str(sequence),
            "--benchmark-root",
            str(args.benchmark_root),
            "--abcrown-root",
            str(args.abcrown_root),
            "--model",
            str(args.model),
            "--property",
            str(args.property),
            "--residual-capture",
            str(args.residual_capture),
            "--projection-capture",
            str(args.projection_capture),
            "--input-capture",
            str(args.input_capture),
            "--result",
            str(result),
        ]
        if args.attribute_root_segments:
            command.append("--attribute-root-segments")
        if args.direct_root_backward and configuration == CANDIDATE:
            command.append("--direct-root-backward")
        completed = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            check=False,
        )
        stdout = attempt_dir / "stdout.txt"
        stderr = attempt_dir / "stderr.txt"
        stdout.write_text(completed.stdout, encoding="utf-8")
        stderr.write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(
                f"BAB4 cumulative root worker failed: {configuration}: "
                f"{completed.stderr[-2000:]}"
            )
        payload = _mapping(json.loads(result.read_text(encoding="utf-8")), "worker")
        run = _mapping(payload.get("run"), "run")
        environment = _mapping(run.get("environment"), "environment")
        admitted = environment.get("admitted") is True
        attempts.append({"attempt": attempt, "environment_admitted": admitted})
        if not admitted:
            continue
        for source in (result, stdout, stderr):
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
    raise RuntimeError("BAB4 cumulative root environment did not admit")


def _summarize(
    workers: Mapping[tuple[int, str], Mapping[str, Any]],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    speedups: dict[str, list[float]] = {
        scope: [] for scope in ("query", "root", "core")
    }
    memory: dict[str, list[float]] = {name: [] for name in ("allocated", "reserved")}
    direct_modes: list[bool] = []
    for pair, order in enumerate(PAIR_ORDERS):
        control = workers[(pair, CONTROL)]
        candidate = workers[(pair, CANDIDATE)]
        control_run = _mapping(control.get("run"), "control run")
        candidate_run = _mapping(candidate.get("run"), "candidate run")
        semantics = formal._semantic_pair(
            _mapping(control_run.get("semantics"), "control semantics"),
            _mapping(candidate_run.get("semantics"), "candidate semantics"),
        )
        row: dict[str, object] = {"pair": pair, "order": list(order), **semantics}
        for scope in speedups:
            control_ns = _duration(control, scope)
            candidate_ns = _duration(candidate, scope)
            speedup = control_ns / candidate_ns
            speedups[scope].append(speedup)
            row[f"control_{scope}_ns"] = control_ns
            row[f"candidate_{scope}_ns"] = candidate_ns
            row[f"{scope}_speedup"] = speedup
        control_metrics = _mapping(control_run.get("metrics"), "control metrics")
        candidate_metrics = _mapping(candidate_run.get("metrics"), "candidate metrics")
        for name, field in (
            ("allocated", "peak_allocated_bytes"),
            ("reserved", "peak_reserved_bytes"),
        ):
            ratio = _integer(candidate_metrics.get(field), field) / _integer(
                control_metrics.get(field), field
            )
            memory[name].append(ratio)
            row[f"peak_{name}_ratio"] = ratio
        receipts = _mapping(candidate.get("root_receipts"), "root receipts")
        warmup_receipts = _mapping(
            candidate.get("root_warmup_receipts"), "root warmup receipts"
        )
        input_receipt = _mapping(receipts.get("input_domain"), "input receipt")
        warmup_input_receipt = _mapping(
            warmup_receipts.get("input_domain"), "root warmup input receipt"
        )
        if (
            candidate.get("root_query_install_count") != 1
            or input_receipt.get("forward_launch_count") != 5
            or input_receipt.get("backward_launch_count") != 4
            or input_receipt.get("dense_input_a_external_materialization_count") != 0
            or input_receipt.get("fallback_count") != 0
            or warmup_input_receipt.get("forward_launch_count") != 5
            or warmup_input_receipt.get("backward_launch_count") != 4
            or warmup_input_receipt.get("fallback_count") != 0
        ):
            raise ValueError("BAB4 cumulative root production activation differs")
        backward_receipt = receipts.get("backward_general")
        direct_mode = candidate.get("root_direct_backward_enabled") is True
        direct_modes.append(direct_mode)
        if direct_mode:
            backward = _mapping(backward_receipt, "backward receipt")
            if (
                backward.get("call_count") != 5
                or backward.get("native_deque_traversal_count") != 0
                or backward.get("fallback_count") != 0
            ):
                raise ValueError("BAB4 direct backward activation differs")
        rows.append(row)
    if any(direct_modes) != all(direct_modes):
        raise ValueError("BAB4 cumulative root direct mode differs across pairs")
    result: dict[str, object] = {
        "schema_version": "boundflow.bab4-root-gc-three-fresh/v1",
        "pair_count": len(rows),
        "control_configuration": CONTROL,
        "candidate_configuration": CANDIDATE,
        "direct_root_backward_enabled": all(direct_modes),
        "rows": rows,
        "all_discrete_semantics_exact": True,
        "lower_max_abs_diff": max(
            cast(float, row["lower_max_abs_diff"]) for row in rows
        ),
        "lower_sign_exact": all(bool(row["lower_sign_exact"]) for row in rows),
        "peak_allocated_ratio_geomean": _geomean(memory["allocated"]),
        "peak_allocated_ratio_worst": max(memory["allocated"]),
        "peak_reserved_ratio_geomean": _geomean(memory["reserved"]),
        "peak_reserved_ratio_worst": max(memory["reserved"]),
        "performance_claimed": False,
    }
    for scope, values in speedups.items():
        result[f"{scope}_speedup_geomean"] = _geomean(values)
        result[f"{scope}_speedup_worst"] = min(values)
    result["query_research_gate_qualified"] = (
        cast(float, result["query_speedup_geomean"]) >= 1.15
    )
    result["summary_hash"] = formal._canonical_hash(result)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--abcrown-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--property", type=Path, required=True)
    parser.add_argument("--residual-capture", type=Path, required=True)
    parser.add_argument("--projection-capture", type=Path, required=True)
    parser.add_argument("--input-capture", type=Path, required=True)
    parser.add_argument("--max-environment-attempts", type=int, default=6)
    parser.add_argument("--attribute-root-segments", action="store_true")
    parser.add_argument("--direct-root-backward", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the cumulative diagnostic and print its summary."""

    args = _parse_args()
    for name in (
        "output",
        "python",
        "benchmark_root",
        "abcrown_root",
        "model",
        "property",
        "residual_capture",
        "projection_capture",
        "input_capture",
    ):
        setattr(args, name, getattr(args, name).absolute())
    if args.max_environment_attempts <= 0:
        raise ValueError("BAB4 cumulative root retry count differs")
    args.output.mkdir(parents=True, exist_ok=False)
    workers: dict[tuple[int, str], Mapping[str, Any]] = {}
    for pair, order in enumerate(PAIR_ORDERS):
        for sequence, configuration in enumerate(order):
            workers[(pair, configuration)] = _run_worker(
                args,
                pair=pair,
                sequence=sequence,
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
