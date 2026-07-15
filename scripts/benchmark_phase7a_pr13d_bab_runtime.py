"""PR-13D fixed-stream and true-E2E same-solver BaB benchmark."""

# The runner keeps one explicit matrix in a single artifact-producing module.
# pylint: disable=duplicate-code

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from functools import partial
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Callable, Optional, Sequence, TypeVar

import torch

from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime.bab import BabConfig, BabResult, solve_bab_mlp
from boundflow.runtime.bab_query import (
    BoundQueryRequest,
    BoundQueryResult,
    FixedBabQueryRecorder,
    build_query_batch,
    compare_query_results,
    execute_bound_query,
)
from boundflow.runtime.bab_query_runtime import (
    SameSolverQueryRuntime,
    SameSolverRuntimeConfig,
)
from boundflow.runtime.query_executor import execute_alpha_beta_query_batch
from boundflow.runtime.task_executor import InputSpec

SCHEMA_VERSION = "boundflow.pr13d-bab-runtime/v2"
MANIFEST_SCHEMA_VERSION = "boundflow.pr13d-bab-runtime-manifest/v2"


@dataclass(frozen=True)
class Workload:
    """One deterministic CNN verification query."""

    name: str
    module: BFTaskModule
    input_spec: InputSpec
    threshold: float


@dataclass(frozen=True)
class TimedRun:
    """One synchronized execution sample."""

    latency_ms: float
    peak_delta_bytes: int


@dataclass(frozen=True)
class TimingStats:
    """Aggregate timing and allocator statistics for one variant."""

    runs: int
    latency_ms_p50: float
    latency_ms_p90: float
    latency_ms_p99: float
    peak_delta_bytes_p50: float
    peak_delta_bytes_max: int


T = TypeVar("T")


def _make_chain_cnn(
    *,
    device: torch.device,
    dtype: torch.dtype,
    output_bias: float,
    stable: bool,
) -> BFTaskModule:
    generator = torch.Generator(device=device)
    generator.manual_seed(13)
    if stable:
        w1 = torch.full((2, 1, 3, 3), 0.05, device=device, dtype=dtype)
        b1 = torch.ones(2, device=device, dtype=dtype)
        w2 = torch.full((2, 2, 3, 3), 0.03, device=device, dtype=dtype)
        b2 = torch.ones(2, device=device, dtype=dtype)
        w3 = torch.zeros((1, 2 * 6 * 6), device=device, dtype=dtype)
    else:
        w1 = (
            torch.randn((2, 1, 3, 3), device=device, dtype=dtype, generator=generator)
            * 0.15
        )
        b1 = torch.zeros(2, device=device, dtype=dtype)
        w2 = (
            torch.randn((2, 2, 3, 3), device=device, dtype=dtype, generator=generator)
            * 0.10
        )
        b2 = torch.zeros(2, device=device, dtype=dtype)
        w3 = (
            torch.randn(
                (1, 2 * 6 * 6),
                device=device,
                dtype=dtype,
                generator=generator,
            )
            * 0.05
        )
    b3 = torch.tensor([output_bias], device=device, dtype=dtype)
    task = BoundTask(
        task_id="cnn",
        kind=TaskKind.INTERVAL_IBP,
        ops=[
            TaskOp(
                op_type="conv2d",
                name="conv1",
                inputs=["input", "W1", "b1"],
                outputs=["h1"],
                attrs={
                    "stride": (1, 1),
                    "padding": (1, 1),
                    "dilation": (1, 1),
                    "groups": 1,
                },
            ),
            TaskOp(op_type="relu", name="relu1", inputs=["h1"], outputs=["r1"]),
            TaskOp(
                op_type="conv2d",
                name="conv2",
                inputs=["r1", "W2", "b2"],
                outputs=["h2"],
                attrs={
                    "stride": (1, 1),
                    "padding": (1, 1),
                    "dilation": (1, 1),
                    "groups": 1,
                },
            ),
            TaskOp(op_type="relu", name="relu2", inputs=["h2"], outputs=["r2"]),
            TaskOp(
                op_type="flatten",
                name="flatten",
                inputs=["r2"],
                outputs=["flat"],
                attrs={"start_dim": 1, "end_dim": -1},
            ),
            TaskOp(
                op_type="linear",
                name="linear",
                inputs=["flat", "W3", "b3"],
                outputs=["out"],
            ),
        ],
        input_values=["input"],
        output_values=["out"],
    )
    return BFTaskModule(
        tasks=[task],
        entry_task_id="cnn",
        bindings={
            "params": {
                "W1": w1,
                "b1": b1,
                "W2": w2,
                "b2": b2,
                "W3": w3,
                "b3": b3,
            }
        },
    )


def _workloads(device: torch.device) -> list[Workload]:
    dtype = torch.float32
    hard_module = _make_chain_cnn(
        device=device, dtype=dtype, output_bias=0.0, stable=False
    )
    safe_module = _make_chain_cnn(
        device=device, dtype=dtype, output_bias=5.0, stable=False
    )
    unsafe_module = _make_chain_cnn(
        device=device, dtype=dtype, output_bias=-1.0, stable=True
    )
    ambiguous = InputSpec.linf(
        value_name="input",
        center=torch.zeros((1, 1, 6, 6), device=device, dtype=dtype),
        eps=0.2,
    )
    stable = InputSpec.linf(
        value_name="input",
        center=torch.ones((1, 1, 6, 6), device=device, dtype=dtype),
        eps=0.01,
    )
    return [
        Workload("cnn_hard_max_nodes", hard_module, ambiguous, 10.0),
        Workload("cnn_safe_root", safe_module, ambiguous, 0.0),
        Workload("cnn_unsafe_root", unsafe_module, stable, 0.0),
    ]


def _config(
    *, max_nodes: int, node_batch_size: int, alpha_steps: int, threshold: float
) -> BabConfig:
    return BabConfig(
        max_nodes=max_nodes,
        oracle="alpha_beta",
        node_batch_size=node_batch_size,
        enable_node_eval_cache=False,
        enable_batch_infeasible_prune=False,
        alpha_steps=alpha_steps,
        alpha_lr=0.2,
        alpha_init=0.5,
        beta_init=0.0,
        threshold=threshold,
        tol=1e-6,
    )


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(device: torch.device, fn: Callable[[], T]) -> tuple[TimedRun, T]:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        _sync(device)
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        baseline = 0
    start_ns = time.perf_counter_ns()
    value = fn()
    _sync(device)
    latency_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
    peak_delta = (
        max(0, torch.cuda.max_memory_allocated(device) - baseline)
        if device.type == "cuda"
        else 0
    )
    return TimedRun(latency_ms, int(peak_delta)), value


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return float(ordered[int(round((len(ordered) - 1) * quantile))])


def _stats(samples: Sequence[TimedRun]) -> TimingStats:
    latencies = [sample.latency_ms for sample in samples]
    peaks = [sample.peak_delta_bytes for sample in samples]
    return TimingStats(
        runs=len(samples),
        latency_ms_p50=_percentile(latencies, 0.50),
        latency_ms_p90=_percentile(latencies, 0.90),
        latency_ms_p99=_percentile(latencies, 0.99),
        peak_delta_bytes_p50=_percentile(peaks, 0.50),
        peak_delta_bytes_max=max(peaks, default=0),
    )


def _runtime(max_batch_size: int, memory_budget_bytes: int) -> SameSolverQueryRuntime:
    return SameSolverQueryRuntime(
        SameSolverRuntimeConfig(
            max_batch_size=max_batch_size,
            memory_budget_bytes=memory_budget_bytes,
        )
    )


def _check_results(
    expected_by_id: dict[str, BoundQueryResult],
    actual: Sequence[tuple[str, BoundQueryResult]],
) -> dict[str, object]:
    comparisons = [
        compare_query_results(query_id, expected_by_id[query_id], result)
        for query_id, result in actual
    ]
    return {
        "count": len(comparisons),
        "passed": sum(item.passed for item in comparisons),
        "failed": sum(not item.passed for item in comparisons),
        "state_version_mismatches": sum(
            not item.state_version_match for item in comparisons
        ),
        "state_value_mismatches": sum(
            not item.state_values_allclose for item in comparisons
        ),
        "max_abs_diff": max((item.max_abs_diff for item in comparisons), default=0.0),
    }


def _run_fixed_stream(  # pylint: disable=too-many-arguments,too-many-locals
    workload: Workload,
    *,
    max_nodes: int,
    alpha_steps: int,
    batch_size: int,
    memory_budget_bytes: int,
    warmup: int,
    repeats: int,
) -> list[dict[str, object]]:
    device = workload.input_spec.center.device
    recorder = FixedBabQueryRecorder()
    solve_bab_mlp(
        workload.module,
        workload.input_spec,
        config=_config(
            max_nodes=max_nodes,
            node_batch_size=1,
            alpha_steps=alpha_steps,
            threshold=workload.threshold,
        ),
        query_recorder=recorder,
    )
    recorder.validate_complete()
    requests = [
        BoundQueryRequest(entry.query, entry.payload) for entry in recorder.entries
    ]
    expected_by_id = {
        entry.query.query_id: entry.result
        for entry in recorder.entries
        if entry.result is not None
    }
    physical_batches = [
        build_query_batch(
            requests[start : start + batch_size],
            estimated_peak_bytes=0,
            memory_budget_bytes=memory_budget_bytes,
        )
        for start in range(0, len(requests), batch_size)
    ]

    def per_node() -> list[tuple[str, BoundQueryResult]]:
        return [
            (
                request.query.query_id,
                execute_bound_query(workload.module, request.query, request.payload),
            )
            for request in requests
        ]

    def batched_original() -> list[tuple[str, BoundQueryResult]]:
        results: list[tuple[str, BoundQueryResult]] = []
        for batch in physical_batches:
            results.extend(execute_alpha_beta_query_batch(workload.module, batch))
        return results

    def runtime_batch() -> list[tuple[str, BoundQueryResult]]:
        runtime = _runtime(batch_size, memory_budget_bytes)
        result = runtime.execute(workload.module, requests)
        runtime_audits.append(runtime.audit())
        return result

    rows: list[dict[str, object]] = []
    runtime_audits: list[dict[str, object]] = []
    for name, function in (
        ("per_node_original", per_node),
        ("batched_original", batched_original),
        ("boundflow_runtime_dense", runtime_batch),
    ):
        for _ in range(warmup):
            function()
            _sync(device)
        samples: list[TimedRun] = []
        last_results: list[tuple[str, BoundQueryResult]] = []
        for _ in range(repeats):
            sample, value = _timed(device, function)
            samples.append(sample)
            last_results = value
        timing = _stats(samples)
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "mode": "fixed_stream",
                "workload": workload.name,
                "variant": name,
                "query_count": len(requests),
                "batch_size": 1 if name == "per_node_original" else batch_size,
                "timing": asdict(timing),
                "time_per_query_ms_p50": timing.latency_ms_p50 / len(requests),
                "query_throughput_per_s_p50": (
                    len(requests) * 1_000.0 / timing.latency_ms_p50
                ),
                "correctness": _check_results(expected_by_id, last_results),
                "runtime_audit_last": (
                    runtime_audits[-1]
                    if name == "boundflow_runtime_dense" and runtime_audits
                    else None
                ),
            }
        )
    return rows


def _result_fields(result: BabResult) -> dict[str, object]:
    return {
        "status": result.status,
        "nodes_visited": result.nodes_visited,
        "nodes_evaluated": result.nodes_evaluated,
        "nodes_expanded": result.nodes_expanded,
        "max_queue": result.max_queue,
        "batch_rounds": result.batch_rounds,
        "avg_batch_fill_rate": result.avg_batch_fill_rate,
        "best_lower": result.best_lower,
        "best_upper": result.best_upper,
    }


def _execute_e2e_once(
    workload: Workload,
    config: BabConfig,
    *,
    use_runtime: bool,
    batch_size: int,
    memory_budget_bytes: int,
) -> tuple[BabResult, Optional[dict[str, object]]]:
    """Execute one solver run with an optional same-solver runtime adapter."""

    runtime = _runtime(batch_size, memory_budget_bytes) if use_runtime else None
    result = solve_bab_mlp(
        workload.module,
        workload.input_spec,
        config=config,
        query_runtime=runtime,
    )
    return result, None if runtime is None else runtime.audit()


def _run_e2e(  # pylint: disable=too-many-arguments,too-many-locals
    workload: Workload,
    *,
    max_nodes: int,
    alpha_steps: int,
    batch_size: int,
    memory_budget_bytes: int,
    warmup: int,
    repeats: int,
) -> list[dict[str, object]]:
    device = workload.input_spec.center.device
    rows: list[dict[str, object]] = []
    variants = (
        ("per_node_original", 1, False),
        ("batched_original", batch_size, False),
        ("boundflow_runtime_dense", batch_size, True),
    )
    for name, node_batch_size, use_runtime in variants:
        config = _config(
            max_nodes=max_nodes,
            node_batch_size=node_batch_size,
            alpha_steps=alpha_steps,
            threshold=workload.threshold,
        )

        execute = partial(
            _execute_e2e_once,
            workload,
            config,
            use_runtime=use_runtime,
            batch_size=batch_size,
            memory_budget_bytes=memory_budget_bytes,
        )

        for _ in range(warmup):
            execute()
            _sync(device)
        samples: list[TimedRun] = []
        results: list[BabResult] = []
        runtime_audits: list[dict[str, object]] = []
        for _ in range(repeats):
            sample, value = _timed(device, execute)
            result, audit = value
            samples.append(sample)
            results.append(result)
            if audit is not None:
                runtime_audits.append(audit)
        reference = _result_fields(results[0])
        stable = all(_result_fields(result) == reference for result in results)
        timing = _stats(samples)
        latency_p50 = timing.latency_ms_p50
        nodes_value = reference["nodes_evaluated"]
        if not isinstance(nodes_value, int):
            raise TypeError("nodes_evaluated must be an integer")
        nodes = nodes_value
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "mode": "true_e2e",
                "workload": workload.name,
                "variant": name,
                "node_batch_size": node_batch_size,
                "timing": asdict(timing),
                "solver": reference,
                "stable_across_repeats": stable,
                "nodes_per_second_p50": (
                    0.0 if latency_p50 <= 0.0 else nodes * 1_000.0 / latency_p50
                ),
                "runtime_audit_last": (runtime_audits[-1] if runtime_audits else None),
            }
        )
    return rows


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True, encoding="utf-8").strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row_p50(row: dict[str, object]) -> float:
    timing = row.get("timing")
    if not isinstance(timing, dict):
        raise TypeError("benchmark row is missing timing")
    value = timing.get("latency_ms_p50")
    if not isinstance(value, (int, float)):
        raise TypeError("timing.latency_ms_p50 must be numeric")
    return float(value)


def _numeric_field(values: dict[str, object], name: str) -> float:
    value = values.get(name)
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    return float(value)


def _solver_nodes(row: dict[str, object]) -> object:
    solver = row.get("solver")
    return solver.get("nodes_evaluated") if isinstance(solver, dict) else None


def _comparison_summary(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Summarize runtime speedups without hiding the batched-original baseline."""

    groups = sorted({(str(row["mode"]), str(row["workload"])) for row in rows})
    comparisons: list[dict[str, object]] = []
    for mode, workload in groups:
        variants = {
            str(row["variant"]): row
            for row in rows
            if row["mode"] == mode and row["workload"] == workload
        }
        required = {
            "per_node_original",
            "batched_original",
            "boundflow_runtime_dense",
        }
        if not required.issubset(variants):
            continue
        runtime_ms = _row_p50(variants["boundflow_runtime_dense"])
        item: dict[str, object] = {
            "mode": mode,
            "workload": workload,
            "runtime_speedup_vs_per_node": (
                _row_p50(variants["per_node_original"]) / runtime_ms
            ),
            "runtime_speedup_vs_batched_original": (
                _row_p50(variants["batched_original"]) / runtime_ms
            ),
        }
        if mode == "true_e2e":
            solvers = [variants[name].get("solver") for name in sorted(required)]
            item["status_consistent"] = (
                all(isinstance(solver, dict) for solver in solvers)
                and len(
                    {solver["status"] for solver in solvers if isinstance(solver, dict)}
                )
                == 1
            )
            item["node_counts"] = {
                name: _solver_nodes(variants[name]) for name in sorted(required)
            }
        comparisons.append(item)
    return comparisons


def run(args: argparse.Namespace) -> None:  # pylint: disable=too-many-locals
    """Run reduced fixed/E2E matrices and write raw JSONL plus manifest."""

    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty artifact: {args.out_dir}"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    rows: list[dict[str, object]] = []
    workloads = _workloads(device)
    hard = workloads[0]
    rows.extend(
        _run_fixed_stream(
            hard,
            max_nodes=args.max_nodes,
            alpha_steps=args.alpha_steps,
            batch_size=args.batch_size,
            memory_budget_bytes=args.memory_budget_bytes,
            warmup=args.warmup,
            repeats=args.repeats,
        )
    )
    for workload in workloads:
        rows.extend(
            _run_e2e(
                workload,
                max_nodes=args.max_nodes,
                alpha_steps=args.alpha_steps,
                batch_size=args.batch_size,
                memory_budget_bytes=args.memory_budget_bytes,
                warmup=args.warmup,
                repeats=args.repeats,
            )
        )
    raw_path = args.out_dir / "raw.jsonl"
    raw_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows
        ),
        encoding="utf-8",
    )
    correctness_failures = 0
    unstable_e2e = 0
    for row in rows:
        correctness = row.get("correctness")
        if isinstance(correctness, dict):
            failed = correctness.get("failed", 0)
            if not isinstance(failed, int):
                raise TypeError("correctness.failed must be an integer")
            correctness_failures += failed
        if row.get("mode") == "true_e2e" and row.get("stable_across_repeats") is False:
            unstable_e2e += 1
    comparisons = _comparison_summary(rows)
    fixed_hard = next(
        item
        for item in comparisons
        if item["mode"] == "fixed_stream" and item["workload"] == "cnn_hard_max_nodes"
    )
    e2e_hard = next(
        item
        for item in comparisons
        if item["mode"] == "true_e2e" and item["workload"] == "cnn_hard_max_nodes"
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": ("ok" if correctness_failures == 0 and unstable_e2e == 0 else "fail"),
        "scope": "reduced chain-CNN; not VNN-COMP/full closure",
        "rows": len(rows),
        "correctness_failures": correctness_failures,
        "unstable_e2e_rows": unstable_e2e,
        "comparisons": comparisons,
        "research_gate": {
            "fixed_speedup_vs_per_node_ge_1_3": (
                _numeric_field(fixed_hard, "runtime_speedup_vs_per_node") >= 1.3
            ),
            "hard_e2e_speedup_vs_per_node_ge_1_2": (
                _numeric_field(e2e_hard, "runtime_speedup_vs_per_node") >= 1.2
            ),
            "hard_runtime_within_10pct_of_batched_original": (
                _numeric_field(e2e_hard, "runtime_speedup_vs_batched_original") >= 0.9
            ),
            "non_toy_workload": False,
        },
        "closure_recommendation": "VALIDATED-REDUCED",
        "limitations": [
            "alpha-beta/split queries capability-filter to dense executor",
            "PR-12 plain-CROWN multi-backend Planner is not eligible",
            "no compile-time component for PyTorch eager dense executor",
        ],
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "command": " ".join(["python", __file__, *args.argv]),
        "config": {
            key: value
            for key, value in vars(args).items()
            if key not in {"argv", "out_dir"}
        },
        "torch": torch.__version__,
        "python": platform.python_version(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "files": {
            raw_path.name: _sha256(raw_path),
            summary_path.name: _sha256(summary_path),
        },
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if summary["status"] != "ok":
        raise RuntimeError(f"PR-13D benchmark failed: {summary}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse reduced benchmark controls."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-nodes", type=int, default=16)
    parser.add_argument("--alpha-steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--memory-budget-bytes", type=int, default=1 << 30)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    args.argv = list(argv) if argv is not None else []
    return args


def main() -> None:
    """Run the command-line benchmark."""

    args = parse_args(sys.argv[1:])
    run(args)


if __name__ == "__main__":
    main()
