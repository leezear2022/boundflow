#!/usr/bin/env python3
"""Run three fresh paired native/expanded root CROWN measurements."""

# pylint: disable=too-many-locals,duplicate-code

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
from typing import Any, cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKER = REPOSITORY_ROOT / "scripts/run_root_crown_expanded_live_worker.py"


def _geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _duration(payload: dict[str, Any], scope: str) -> int:
    if scope == "query":
        return int(payload["run"]["metrics"]["query_wall_ns"])
    aggregates = payload["diagnostics"]["root_incomplete_timings"]["aggregates"]
    if scope == "root":
        return int(aggregates["root_incomplete"]["inclusive_ns"])
    if scope == "optimizer":
        return int(aggregates["optimized_bounds_transaction"]["inclusive_ns"])
    if scope == "backward":
        return int(aggregates["autograd_backward"]["inclusive_ns"])
    raise ValueError(f"root expanded timing scope differs: {scope}")


def _semantic_difference(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_semantics = left["run"]["semantics"]
    right_semantics = right["run"]["semantics"]
    exact_fields = (
        "batch_size",
        "depths",
        "final_decision",
        "history_count",
        "lower_shape",
        "n_splits",
        "n_verified",
        "queue_accepted",
        "queue_after",
        "queue_before",
        "queue_input",
        "queue_pruned",
        "split_depth",
        "status",
        "success",
        "upper_positive_infinity_mask",
        "upper_shape",
        "visited_domains",
    )
    if any(left_semantics[name] != right_semantics[name] for name in exact_fields):
        raise ValueError("root expanded discrete semantics differ")
    left_values = cast(list[float], left_semantics["lower_values"])
    right_values = cast(list[float], right_semantics["lower_values"])
    if len(left_values) != len(right_values):
        raise ValueError("root expanded lower cardinality differs")
    return max(
        abs(left_value - right_value)
        for left_value, right_value in zip(left_values, right_values)
    )


def _command(
    args: argparse.Namespace, mode: str, pair: int, sequence: int, result: Path
) -> list[str]:
    return [
        str(args.python),
        str(WORKER),
        "--mode",
        mode,
        "--run-id",
        f"expanded-p{pair}-{mode}",
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
        "--result",
        str(result),
    ]


def _run(args: argparse.Namespace) -> dict[str, object]:
    args.output.mkdir(parents=True, exist_ok=True)
    orders = (
        ("control", "candidate-single"),
        ("candidate-single", "control"),
        ("control", "candidate-single"),
    )
    rows: list[dict[str, object]] = []
    for pair, order in enumerate(orders):
        payloads: dict[str, dict[str, Any]] = {}
        for sequence, mode in enumerate(order):
            result = args.output / f"pair-{pair}-{mode}.json"
            completed = subprocess.run(
                _command(args, mode, pair, sequence, result),
                cwd=REPOSITORY_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            (args.output / f"pair-{pair}-{mode}.stdout.log").write_text(
                completed.stdout, encoding="utf-8"
            )
            payloads[mode] = json.loads(result.read_text(encoding="utf-8"))
        control = payloads["control"]
        candidate = payloads["candidate-single"]
        row: dict[str, object] = {
            "pair": pair,
            "order": list(order),
            "lower_max_abs_diff": _semantic_difference(control, candidate),
        }
        for scope in ("query", "root", "optimizer", "backward"):
            control_ns = _duration(control, scope)
            candidate_ns = _duration(candidate, scope)
            row[f"control_{scope}_ns"] = control_ns
            row[f"candidate_{scope}_ns"] = candidate_ns
            row[f"{scope}_speedup"] = control_ns / candidate_ns
        rows.append(row)
    summary: dict[str, object] = {
        "schema_version": "boundflow.root-crown-expanded-three-fresh/v1",
        "pair_count": len(rows),
        "rows": rows,
        "maximum_lower_absolute_difference": max(
            cast(float, row["lower_max_abs_diff"]) for row in rows
        ),
        "all_discrete_semantics_exact": True,
        "cumulative_autograd_owner_count": 1,
        "performance_claimed": False,
    }
    for scope in ("query", "root", "optimizer", "backward"):
        values = [cast(float, row[f"{scope}_speedup"]) for row in rows]
        summary[f"{scope}_geomean_speedup"] = _geomean(values)
        summary[f"{scope}_worst_speedup"] = min(values)
    summary_path = args.output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--abcrown-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--property", type=Path, required=True)
    parser.add_argument("--residual-capture", type=Path, required=True)
    parser.add_argument("--projection-capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Run the frozen alternating three-pair sequence."""

    _run(_parse_args())


if __name__ == "__main__":
    main()
