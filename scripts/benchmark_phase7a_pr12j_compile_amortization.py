#!/usr/bin/env python
"""Measure fused CROWN compile/load/cache phases and repeated-query amortization."""

# mypy: disable-error-code=import-untyped
# pylint: disable=broad-exception-caught,duplicate-code,import-outside-toplevel
# pylint: disable=too-many-arguments,too-many-branches,too-many-lines
# pylint: disable=too-many-locals,too-many-statements

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Optional, Sequence

import torch

from boundflow.backends.tvm.fused_crown_cache import FusedCrownModuleCache
from boundflow.planner.execution_candidate import BackendVariant
from boundflow.runtime.crown_ibp import run_crown_ibp_mlp
from boundflow.runtime.fused_crown import (
    TVMFusedCrownExecutor,
    build_fused_crown_runtime_selection,
)
from scripts.benchmark_phase7a_pr12_runtime_pareto import (
    _build_query,
    _environment,
    _event_call,
    _max_abs_diff,
    _max_rel_diff,
    _sha256,
    _summary,
    _warm_groups,
    _workload,
    _write_jsonl,
)

SCHEMA_VERSION = "boundflow.pr12j-amortization/v1"
MANIFEST_SCHEMA_VERSION = "boundflow.pr12j-amortization-manifest/v1"
QUERY_COUNTS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
DEFAULT_CASE_IDS = (
    "linear-memory-sensitive",
    "conv-unseen-width",
    "mini-resnet-unseen-width",
)
BASELINE_BACKENDS = (
    BackendVariant.PYTORCH_EAGER.value,
    BackendVariant.PYTORCH_CHUNKED.value,
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _records(split: dict[str, Any], case_ids: Sequence[str]) -> list[dict[str, Any]]:
    requested = set(case_ids)
    records = [
        record for record in split["calibration"] if record.get("case_id") in requested
    ]
    missing = requested - {str(record["case_id"]) for record in records}
    if missing:
        raise ValueError(
            f"case ids are not consumed calibration cases: {sorted(missing)}"
        )
    return records


def _baseline_latency(raw_path: Path) -> dict[tuple[str, str], float]:
    rows = [
        json.loads(line)
        for line in raw_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    values: dict[tuple[str, str], float] = {}
    for row in rows:
        if not (
            row["status"] == "ok"
            and row["candidate"]["stream"] == "default"
            and row["benchmark_contract"]["level"] == "end_to_end_final_bound"
            and row["candidate"]["backend"] in BASELINE_BACKENDS
        ):
            continue
        if not row["benchmark_contract"].get("compliant", False):
            raise ValueError("PR-12J requires compliant PR-12I baseline rows")
        values[(row["workload"]["case_id"], row["candidate"]["backend"])] = float(
            row["runtime"]["host_group_per_query"]["median_ms"]
        )
    return values


def _break_even(
    setup_ms: float, candidate_ms: float, baseline_ms: float
) -> dict[str, Any]:
    if candidate_ms >= baseline_ms:
        return {
            "status": "not_amortizable",
            "queries": None,
            "reason": "candidate_warm_not_faster",
        }
    queries = max(1, math.ceil(setup_ms / (baseline_ms - candidate_ms)))
    return {
        "status": "within_sweep" if queries <= QUERY_COUNTS[-1] else "beyond_sweep",
        "queries": queries,
        "reason": None,
    }


def _query_totals(
    setup_ms: float, candidate_ms: float, baselines: dict[str, float]
) -> list[dict[str, Any]]:
    return [
        {
            "queries": queries,
            "fresh_or_disk_candidate_ms": setup_ms + queries * candidate_ms,
            "memory_cache_candidate_ms": queries * candidate_ms,
            "baselines_ms": {
                backend: queries * latency for backend, latency in baselines.items()
            },
        }
        for queries in QUERY_COUNTS
    ]


def _run_query(
    workload,
    cache_dir: Path,
    *,
    warmup: int,
    groups: int,
    repeats: int,
) -> tuple[dict[str, Any], Any]:
    device = torch.device("cuda")
    module, input_spec = _build_query(workload, device)
    expected = run_crown_ibp_mlp(module, input_spec)
    cache = FusedCrownModuleCache(cache_dir)
    executor = TVMFusedCrownExecutor(compile_cache=cache)
    stream = torch.cuda.default_stream(device)

    def call() -> Any:
        selection = build_fused_crown_runtime_selection(
            module.get_entry_task().ops,
            backend=BackendVariant.TVM_FUSED_TIR.value,
        )
        return run_crown_ibp_mlp(
            module,
            input_spec,
            fused_crown_executor=executor,
            fused_crown_steps=selection.steps,
        )

    plan_started = time.perf_counter_ns()
    standalone_selection = build_fused_crown_runtime_selection(
        module.get_entry_task().ops,
        backend=BackendVariant.TVM_FUSED_TIR.value,
    )
    plan_ms = (time.perf_counter_ns() - plan_started) / 1e6
    first_wall, first_cuda, first_result = _event_call(call, stream)
    first_event_count = len(cache.events)
    cold_wall, cold_cuda, cold_result = _event_call(call, stream)
    del cold_result
    second_event_count = len(cache.events) - first_event_count
    warm_host, warm_cuda = _warm_groups(
        call, stream, warmup=warmup, groups=groups, repeats=repeats
    )
    warm_summary = _summary(warm_host)
    correctness = {
        "max_abs_diff": _max_abs_diff(first_result, expected),
        "max_rel_diff": _max_rel_diff(first_result, expected),
        "finite": bool(
            torch.isfinite(first_result.lower).all()
            and torch.isfinite(first_result.upper).all()
        ),
        "lower_le_upper": bool((first_result.lower <= first_result.upper).all()),
        "allclose": bool(
            torch.allclose(first_result.lower, expected.lower, rtol=2e-4, atol=2e-4)
            and torch.allclose(first_result.upper, expected.upper, rtol=2e-4, atol=2e-4)
        ),
        "rtol": 2e-4,
        "atol": 2e-4,
    }
    del first_result
    events = [event.to_dict() for event in cache.events]
    return (
        {
            "planner_ir_construction_ms": plan_ms,
            "planned_regions": len(standalone_selection.steps),
            "first_query_wall_ms": first_wall,
            "first_query_cuda_event_ms": first_cuda,
            "cold_memory_cache_wall_ms": cold_wall,
            "cold_memory_cache_cuda_event_ms": cold_cuda,
            "warm_host_per_query": warm_summary,
            "warm_cuda_per_query": _summary(warm_cuda),
            "warm_host_samples_ms": warm_host,
            "warm_cuda_samples_ms": warm_cuda,
            "warmup": warmup,
            "groups": groups,
            "repeats": repeats,
            "first_query_cache_event_count": first_event_count,
            "second_query_cache_event_count": second_event_count,
            "cache_events": events,
            "correctness": correctness,
        },
        expected,
    )


def _worker(split_path: Path, case_id: str, cache_dir: Path, output_path: Path) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    record = _records(split, [case_id])[0]
    workload = _workload(record, split_role="compile_worker")
    device = torch.device("cuda")
    module, input_spec = _build_query(workload, device)
    cache = FusedCrownModuleCache(cache_dir)
    executor = TVMFusedCrownExecutor(compile_cache=cache)
    stream = torch.cuda.default_stream(device)

    def call() -> Any:
        selection = build_fused_crown_runtime_selection(
            module.get_entry_task().ops,
            backend=BackendVariant.TVM_FUSED_TIR.value,
        )
        return run_crown_ibp_mlp(
            module,
            input_spec,
            fused_crown_executor=executor,
            fused_crown_steps=selection.steps,
        )

    first_wall, first_cuda, actual = _event_call(call, stream)
    query_completed_ns = time.perf_counter_ns()
    expected = run_crown_ibp_mlp(module, input_spec)
    payload = {
        "schema_version": "boundflow.pr12j-process-worker/v1",
        "case_id": case_id,
        "first_query_wall_ms": first_wall,
        "first_query_cuda_event_ms": first_cuda,
        "cache_events": [event.to_dict() for event in cache.events],
        "all_cache_events_disk_or_memory": all(
            event.event in {"disk_hit", "memory_hit"} for event in cache.events
        ),
        "query_completed_perf_counter_ns": query_completed_ns,
        "correctness": {
            "max_abs_diff": _max_abs_diff(actual, expected),
            "max_rel_diff": _max_rel_diff(actual, expected),
            "allclose": bool(
                torch.allclose(actual.lower, expected.lower, rtol=2e-4, atol=2e-4)
                and torch.allclose(actual.upper, expected.upper, rtol=2e-4, atol=2e-4)
            ),
        },
    }
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


def _restart_worker(
    split_path: Path, case_id: str, cache_dir: Path, worker_path: Path
) -> dict[str, Any]:
    started = time.perf_counter_ns()
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--split-file",
            str(split_path.resolve()),
            "--case-id",
            case_id,
            "--cache-dir",
            str(cache_dir.resolve()),
            "--worker-output",
            str(worker_path.resolve()),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    process_wall_ms = (time.perf_counter_ns() - started) / 1e6
    if completed.returncode != 0:
        raise RuntimeError(
            f"process-restart worker failed rc={completed.returncode}: "
            f"{completed.stderr[-4000:]}"
        )
    payload = json.loads(worker_path.read_text(encoding="utf-8"))
    payload["process_wall_ms"] = process_wall_ms
    payload["stdout"] = completed.stdout[-2000:]
    payload["stderr"] = completed.stderr[-2000:]
    return payload


def _controller(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    split = json.loads(args.split_file.read_text(encoding="utf-8"))
    case_ids = tuple(item for item in args.case_ids.split(",") if item)
    records = _records(split, case_ids)
    workloads = [
        _workload(record, split_role="compile_amortization") for record in records
    ]
    baseline_values = _baseline_latency(args.baseline_raw)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    for workload in workloads:
        cache_dir = args.out_dir / "cache" / workload.case_id
        cache_dir.mkdir(parents=True, exist_ok=True)
        runtime, _ = _run_query(
            workload,
            cache_dir,
            warmup=args.warmup,
            groups=args.groups,
            repeats=args.repeats,
        )
        worker_path = args.out_dir / f"worker_{workload.case_id}.json"
        restart = _restart_worker(
            args.split_file, workload.case_id, cache_dir, worker_path
        )
        misses = [
            event for event in runtime["cache_events"] if event["event"] == "miss"
        ]
        warm_ms = float(runtime["warm_host_per_query"]["median_ms"])
        fresh_setup_ms = max(0.0, float(runtime["first_query_wall_ms"]) - warm_ms)
        disk_setup_ms = max(0.0, float(restart["first_query_wall_ms"]) - warm_ms)
        process_setup_ms = max(0.0, float(restart["process_wall_ms"]) - warm_ms)
        baselines = {
            backend: baseline_values[(workload.case_id, backend)]
            for backend in BASELINE_BACKENDS
        }
        correct = bool(
            runtime["correctness"]["allclose"]
            and runtime["correctness"]["finite"]
            and runtime["correctness"]["lower_le_upper"]
            and restart["correctness"]["allclose"]
            and restart["all_cache_events_disk_or_memory"]
            and misses
        )
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "status": "ok" if correct else "fail",
                "workload": {
                    "case_id": workload.case_id,
                    "family": workload.family,
                    "domain": workload.domain,
                    "spec": workload.spec,
                    "config": workload.config,
                },
                "runtime": runtime,
                "compile_phase_totals_ms": {
                    name: sum(float(event[name]) for event in misses)
                    for name in (
                        "cache_lookup_ms",
                        "tir_generation_ms",
                        "schedule_ms",
                        "tvm_compile_ms",
                        "serialization_ms",
                        "module_load_ms",
                        "total_ms",
                    )
                },
                "unique_compiled_modules": len(misses),
                "process_restart_disk_cache": restart,
                "baseline_warm_ms": baselines,
                "amortization": {
                    "candidate_warm_ms": warm_ms,
                    "fresh_setup_ms": fresh_setup_ms,
                    "disk_first_query_setup_ms": disk_setup_ms,
                    "process_restart_setup_ms": process_setup_ms,
                    "fresh_break_even": {
                        backend: _break_even(fresh_setup_ms, warm_ms, latency)
                        for backend, latency in baselines.items()
                    },
                    "disk_break_even": {
                        backend: _break_even(disk_setup_ms, warm_ms, latency)
                        for backend, latency in baselines.items()
                    },
                    "process_restart_break_even": {
                        backend: _break_even(process_setup_ms, warm_ms, latency)
                        for backend, latency in baselines.items()
                    },
                    "fresh_query_totals": _query_totals(
                        fresh_setup_ms, warm_ms, baselines
                    ),
                    "disk_query_totals": _query_totals(
                        disk_setup_ms, warm_ms, baselines
                    ),
                    "process_restart_query_totals": _query_totals(
                        process_setup_ms, warm_ms, baselines
                    ),
                },
            }
        )
    raw_path = args.out_dir / "raw.jsonl"
    _write_jsonl(raw_path, rows)
    outputs = {"raw.jsonl": _sha256(raw_path)}
    for path in sorted(args.out_dir.glob("worker_*.json")):
        outputs[path.name] = _file_sha256(path)
    cache_files = sorted((args.out_dir / "cache").rglob("*"))
    outputs.update(
        {
            str(path.relative_to(args.out_dir)): _file_sha256(path)
            for path in cache_files
            if path.is_file()
        }
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "split_id": split["split_id"],
        "split_sha256": _sha256(args.split_file),
        "baseline_raw_sha256": _sha256(args.baseline_raw),
        "environment": _environment(),
        "case_ids": [workload.case_id for workload in workloads],
        "query_counts": list(QUERY_COUNTS),
        "measurement": {
            "warmup": args.warmup,
            "groups": args.groups,
            "repeats": args.repeats,
        },
        "row_count": len(rows),
        "status_counts": {
            status: sum(row["status"] == status for row in rows)
            for status in sorted({str(row["status"]) for row in rows})
        },
        "outputs": outputs,
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run controller or one isolated process-restart cache worker."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--case-id", default="")
    parser.add_argument("--case-ids", default=",".join(DEFAULT_CASE_IDS))
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--baseline-raw", type=Path)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--groups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args(argv)
    if args.worker:
        if not args.case_id or args.cache_dir is None or args.worker_output is None:
            parser.error("worker requires case-id, cache-dir and worker-output")
        return _worker(
            args.split_file, args.case_id, args.cache_dir, args.worker_output
        )
    if args.out_dir is None or args.baseline_raw is None:
        parser.error("controller requires out-dir and baseline-raw")
    if min(args.warmup, args.groups, args.repeats) <= 0:
        parser.error("warmup/groups/repeats must be positive")
    return _controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
