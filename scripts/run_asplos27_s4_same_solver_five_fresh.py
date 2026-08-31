#!/usr/bin/env python3
"""Generate or replay five fresh B4-A/S4 same-solver AOT pairs."""

# pylint: disable=too-many-locals,too-many-statements,wrong-import-position
# pylint: disable=too-many-boolean-expressions,duplicate-code

from __future__ import annotations

import argparse
import hashlib
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

ARTIFACT_SCHEMA = "boundflow.asplos27-s4-aot-five-fresh/v1"
WORKER_SCHEMA = "boundflow.asplos27-s4-same-solver-worker/v1"
PAIR_ORDERS = (
    ("B4-A", "S4"),
    ("S4", "B4-A"),
    ("B4-A", "S4"),
    ("S4", "B4-A"),
    ("B4-A", "S4"),
)
CODE_PATHS = (
    "boundflow/runtime/asplos27_s4_exact_call_bridge.py",
    "boundflow/runtime/asplos27_s4_exact_call_plan_template.py",
    "scripts/generate_asplos27_s4_exact_call_plan_template.py",
    "scripts/run_asplos27_s4_same_solver_worker.py",
    "scripts/run_asplos27_s4_same_solver_five_fresh.py",
    "scripts/run_fsg3_same_solver_timing.py",
    "artifacts/asplos27-s4-exact-call-plan/resnet2b-prop0-v1/plan_template.json",
)


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"S4 five-fresh {label} differs")
    return cast(Mapping[str, Any], value)


def _integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"S4 five-fresh {label} differs")
    return value


def _geomean(values: list[float]) -> float:
    if not values or any(value <= 0.0 or not math.isfinite(value) for value in values):
        raise ValueError("S4 five-fresh geomean input differs")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _semantic_pair(
    control: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, object]:
    control_copy = dict(control)
    candidate_copy = dict(candidate)
    control_lower = control_copy.pop("lower_values", None)
    candidate_lower = candidate_copy.pop("lower_values", None)
    if control_copy != candidate_copy:
        raise ValueError("S4 five-fresh discrete semantics differ")
    if (
        not isinstance(control_lower, list)
        or not isinstance(candidate_lower, list)
        or len(control_lower) != len(candidate_lower)
        or not control_lower
    ):
        raise TypeError("S4 five-fresh lower payload differs")
    differences = [
        abs(float(left) - float(right))
        for left, right in zip(control_lower, candidate_lower)
    ]
    signs_exact = all(
        (float(left) < 0.0) == (float(right) < 0.0)
        and (float(left) == 0.0) == (float(right) == 0.0)
        for left, right in zip(control_lower, candidate_lower)
    )
    maximum = max(differences)
    if maximum > 2e-4 or not signs_exact:
        raise ValueError("S4 five-fresh lower semantics differ")
    return {
        "lower_max_abs_diff": maximum,
        "lower_sign_exact": signs_exact,
        "discrete_semantics_exact": True,
    }


def _validate_worker(
    payload: Mapping[str, Any], configuration: str
) -> tuple[Mapping[str, Any], Mapping[str, Any] | None]:
    if (
        payload.get("schema_version") != WORKER_SCHEMA
        or payload.get("configuration") != configuration
        or payload.get("mode") != "control"
        or payload.get("performance_claimed") is not False
    ):
        raise ValueError("S4 five-fresh worker envelope differs")
    run = _mapping(payload.get("run"), "run")
    environment = _mapping(run.get("environment"), "environment")
    if environment.get("admitted") is not True:
        raise ValueError("S4 five-fresh environment is not admitted")
    receipts = payload.get("s4_exact_call_receipts")
    expected_count = int(configuration == "S4")
    if not isinstance(receipts, list) or len(receipts) != expected_count:
        raise ValueError("S4 five-fresh receipt cardinality differs")
    receipt = None if not receipts else _mapping(receipts[0], "receipt")
    if receipt is not None and (
        receipt.get("source_capture_runtime_dependency") is not False
        or receipt.get("static_prepare_excluded_from_query") is not True
        or receipt.get("performance_claimed") is not False
        or receipt.get("fallback_count") != 0
        or receipt.get("compile_inside_exact_call_count") != 0
        or receipt.get("provider_callback_count") != 0
    ):
        raise ValueError("S4 five-fresh AOT activation differs")
    return run, receipt


def _summary(
    protocol: Mapping[str, Any], workers: Mapping[str, Mapping[str, Any]]
) -> dict[str, object]:
    core_speedups: list[float] = []
    query_speedups: list[float] = []
    semantic_rows: list[dict[str, Any]] = []
    static_prepare_ns: list[int] = []
    template_hashes: set[str] = set()
    pair_rows: list[dict[str, Any]] = []
    for pair_ordinal, order in enumerate(PAIR_ORDERS):
        rows = {
            configuration: workers[f"pair-{pair_ordinal:02d}/{configuration}"]
            for configuration in order
        }
        control_run, _ = _validate_worker(rows["B4-A"], "B4-A")
        candidate_run, receipt = _validate_worker(rows["S4"], "S4")
        assert receipt is not None
        control_metrics = _mapping(control_run.get("metrics"), "control metrics")
        candidate_metrics = _mapping(candidate_run.get("metrics"), "candidate metrics")
        control_core = _integer(control_metrics.get("core_wall_ns"), "control core")
        candidate_core = _integer(
            candidate_metrics.get("core_wall_ns"), "candidate core"
        )
        control_query = _integer(control_metrics.get("query_wall_ns"), "control query")
        candidate_query = _integer(
            candidate_metrics.get("query_wall_ns"), "candidate query"
        )
        semantics = _semantic_pair(
            _mapping(control_run.get("semantics"), "control semantics"),
            _mapping(candidate_run.get("semantics"), "candidate semantics"),
        )
        core_speedup = control_core / candidate_core
        query_speedup = control_query / candidate_query
        core_speedups.append(core_speedup)
        query_speedups.append(query_speedup)
        semantic_rows.append(semantics)
        static_prepare_ns.append(
            _integer(receipt.get("static_prepare_ns"), "static prepare")
        )
        template_hash = receipt.get("region_template_hash")
        if not isinstance(template_hash, str) or len(template_hash) != 64:
            raise ValueError("S4 five-fresh template hash differs")
        template_hashes.add(template_hash)
        pair_rows.append(
            {
                "pair_ordinal": pair_ordinal,
                "order": list(order),
                "control_core_wall_ns": control_core,
                "candidate_core_wall_ns": candidate_core,
                "core_speedup": core_speedup,
                "control_query_wall_ns": control_query,
                "candidate_query_wall_ns": candidate_query,
                "query_speedup": query_speedup,
                **semantics,
            }
        )
    if len(template_hashes) != 1:
        raise ValueError("S4 five-fresh template identity drifts")
    core_geomean = _geomean(core_speedups)
    query_geomean = _geomean(query_speedups)
    mean_static_prepare = sum(static_prepare_ns) / len(static_prepare_ns)
    mean_query_saved = sum(
        row["control_query_wall_ns"] - row["candidate_query_wall_ns"]
        for row in pair_rows
    ) / len(pair_rows)
    break_even_queries = (
        None if mean_query_saved <= 0 else mean_static_prepare / mean_query_saved
    )
    summary: dict[str, object] = {
        "schema_version": ARTIFACT_SCHEMA,
        "protocol_hash": _canonical_hash(protocol),
        "pair_count": len(PAIR_ORDERS),
        "worker_count": 2 * len(PAIR_ORDERS),
        "all_environment_admitted": True,
        "all_discrete_semantics_exact": True,
        "lower_max_abs_diff": max(
            float(row["lower_max_abs_diff"]) for row in semantic_rows
        ),
        "lower_sign_exact": all(bool(row["lower_sign_exact"]) for row in semantic_rows),
        "core_speedup_geomean": core_geomean,
        "core_speedup_worst": min(core_speedups),
        "query_speedup_geomean": query_geomean,
        "query_speedup_worst": min(query_speedups),
        "query_parity_qualified": query_geomean >= 1.0,
        "query_research_gate_qualified": query_geomean >= 1.15,
        "core_research_gate_qualified": core_geomean >= 1.20,
        "mean_static_prepare_ns": mean_static_prepare,
        "mean_query_saved_ns": mean_query_saved,
        "cold_break_even_queries": break_even_queries,
        "region_template_hash": next(iter(template_hashes)),
        "pairs": pair_rows,
        "performance_claimed": False,
    }
    summary["summary_hash"] = _canonical_hash(summary)
    return summary


def _protocol(args: argparse.Namespace) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": ARTIFACT_SCHEMA,
        "source_git_head": _git_head(REPOSITORY_ROOT),
        "abcrown_commit": _git_head(args.abcrown_root),
        "vnncomp_commit": _git_head(args.benchmark_root),
        "model_sha256": _file_hash(args.model),
        "property_sha256": _file_hash(args.property),
        "code_revision": {
            path: _file_hash(REPOSITORY_ROOT / path) for path in CODE_PATHS
        },
        "python_executable": str(args.python),
        "pair_orders": [list(order) for order in PAIR_ORDERS],
        "fresh_process_per_worker": True,
        "max_environment_attempts": args.max_environment_attempts,
        "environment_window": "post-static-prepare-cool-idle-to-query-end",
        "lower_atol": 2e-4,
        "lower_sign_exact": True,
        "query_parity_gate": 1.0,
        "query_research_gate": 1.15,
        "core_research_gate": 1.20,
        "source_capture_runtime_dependency": False,
        "performance_claimed": False,
    }
    payload["protocol_hash"] = _canonical_hash(payload)
    return payload


def _run_worker(
    args: argparse.Namespace,
    *,
    pair_ordinal: int,
    sequence_position: int,
    configuration: str,
    output: Path,
) -> Mapping[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (
            str(REPOSITORY_ROOT),
            str(args.abcrown_root),
            environment.get("PYTHONPATH", ""),
        )
    )
    attempt_rows: list[dict[str, object]] = []
    for attempt in range(args.max_environment_attempts):
        attempt_dir = output / f"attempt-{attempt:02d}"
        attempt_dir.mkdir()
        result = attempt_dir / "worker.json"
        command = [
            str(args.python),
            str(REPOSITORY_ROOT / "scripts/run_asplos27_s4_same_solver_worker.py"),
            "--configuration",
            configuration,
            "--mode",
            "control",
            "--run-id",
            (
                f"s4-aot-pair-{pair_ordinal:02d}-{configuration.lower()}"
                f"-attempt-{attempt:02d}"
            ),
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
        ]
        completed = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        (attempt_dir / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
        (attempt_dir / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(
                f"S4 five-fresh worker failed: pair={pair_ordinal} "
                f"configuration={configuration} attempt={attempt} "
                f"exit={completed.returncode}"
            )
        raw = json.loads(result.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise TypeError("S4 five-fresh worker root differs")
        payload = cast(Mapping[str, Any], raw)
        run = _mapping(payload.get("run"), "attempt run")
        admitted = _mapping(run.get("environment"), "attempt environment").get(
            "admitted"
        )
        attempt_rows.append(
            {
                "attempt": attempt,
                "environment_admitted": admitted,
                "worker_sha256": _file_hash(result),
            }
        )
        if admitted is True:
            shutil.copy2(result, output / "worker.json")
            shutil.copy2(attempt_dir / "stdout.txt", output / "stdout.txt")
            shutil.copy2(attempt_dir / "stderr.txt", output / "stderr.txt")
            selection = {
                "selected_attempt": attempt,
                "attempts": attempt_rows,
                "all_attempts_preserved": True,
            }
            (output / "selection.json").write_text(
                json.dumps(selection, sort_keys=True, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            return payload
    raise RuntimeError(
        f"S4 five-fresh environment did not admit: pair={pair_ordinal} "
        f"configuration={configuration} attempts={args.max_environment_attempts}"
    )


def _manifest(root: Path) -> dict[str, object]:
    files = {
        str(path.relative_to(root)): _file_hash(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    result: dict[str, object] = {
        "schema_version": ARTIFACT_SCHEMA,
        "files": files,
        "performance_claimed": False,
    }
    result["manifest_hash"] = _canonical_hash(result)
    return result


def _generate(args: argparse.Namespace) -> None:
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    protocol = _protocol(args)
    (root / "protocol.json").write_text(
        json.dumps(protocol, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    workers: dict[str, Mapping[str, Any]] = {}
    for pair_ordinal, order in enumerate(PAIR_ORDERS):
        for sequence_position, configuration in enumerate(order):
            key = f"pair-{pair_ordinal:02d}/{configuration}"
            workers[key] = _run_worker(
                args,
                pair_ordinal=pair_ordinal,
                sequence_position=sequence_position,
                configuration=configuration,
                output=root / "raw" / f"pair-{pair_ordinal:02d}" / configuration,
            )
    summary = _summary(protocol, workers)
    (root / "summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    manifest = _manifest(root)
    (root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))


def _replay(args: argparse.Namespace) -> None:
    root = args.artifact.resolve()
    protocol_raw = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    summary_raw = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    manifest_raw = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    protocol = _mapping(protocol_raw, "protocol")
    expected_manifest = _manifest(root)
    if manifest_raw != expected_manifest:
        raise ValueError("S4 five-fresh manifest differs")
    workers = {
        f"pair-{pair_ordinal:02d}/{configuration}": _mapping(
            json.loads(
                (
                    root
                    / "raw"
                    / f"pair-{pair_ordinal:02d}"
                    / configuration
                    / "worker.json"
                ).read_text(encoding="utf-8")
            ),
            "replay worker",
        )
        for pair_ordinal, order in enumerate(PAIR_ORDERS)
        for configuration in order
    }
    recomputed = _summary(protocol, workers)
    if summary_raw != recomputed:
        raise ValueError("S4 five-fresh summary differs")
    print(
        json.dumps(
            {
                "artifact": str(root),
                "manifest_hash": expected_manifest["manifest_hash"],
                "summary_hash": recomputed["summary_hash"],
                "status": "PASS",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--python", type=Path, required=True)
    generate.add_argument("--benchmark-root", type=Path, required=True)
    generate.add_argument("--abcrown-root", type=Path, required=True)
    generate.add_argument("--model", type=Path, required=True)
    generate.add_argument("--property", type=Path, required=True)
    generate.add_argument("--max-environment-attempts", type=int, default=4)
    replay = subparsers.add_parser("replay")
    replay.add_argument("--artifact", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Dispatch artifact generation or deterministic replay."""

    args = _parse_args()
    if args.command == "generate":
        if args.max_environment_attempts <= 0:
            raise ValueError("S4 five-fresh retry count must be positive")
        args.python = args.python.absolute()
        for name in ("benchmark_root", "abcrown_root", "model", "property"):
            setattr(args, name, getattr(args, name).resolve())
        _generate(args)
    else:
        _replay(args)


if __name__ == "__main__":
    main()
