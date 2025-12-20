#!/usr/bin/env python
from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import os
import platform
import socket
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

try:
    import signal

    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
except Exception:
    pass

from boundflow.frontends.pytorch.frontend import import_torch
from boundflow.planner.core import PlannerConfig
from boundflow.planner.options import PartitionOptions, PartitionPolicy, PlannerDebugOptions
from boundflow.planner.pipeline import plan
from boundflow.planner.storage_reuse import ReuseKeyMode, ReusePolicy, StorageReuseOptions
from boundflow.runtime.scheduler import run_ibp_scheduled
from boundflow.runtime.task_executor import LinfInputSpec, PythonTaskExecutor
from boundflow.runtime.tvm_executor import MemoryPlanMode, TVMExecutorOptions, TVMTaskExecutor
from boundflow.ir.liveness import buffer_size_bytes


def _git_short_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return ""


def _utc_now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


def _try_import_auto_lirpa():
    try:
        import auto_LiRPA  # type: ignore

        from auto_LiRPA.perturbations import PerturbationLpNorm  # type: ignore

        return auto_LiRPA, auto_LiRPA.BoundedModule, auto_LiRPA.BoundedTensor, PerturbationLpNorm
    except Exception:
        return None


def _percentiles_ms(samples_ms: List[float]) -> Dict[str, float]:
    if not samples_ms:
        return {"p50": 0.0, "p95": 0.0}
    xs = sorted(samples_ms)
    p50 = xs[int(0.50 * (len(xs) - 1))]
    p95 = xs[int(0.95 * (len(xs) - 1))]
    return {"p50": float(p50), "p95": float(p95)}


def _jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if dataclasses.is_dataclass(obj):
        return {f.name: _jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return repr(obj)


def _aggregate_task_compile_stats(compile_stats: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    entries = [v for v in (compile_stats or {}).values() if isinstance(v, dict) and v.get("kind") == "task_relax_ops"]
    out: Dict[str, Any] = {
        "num_compiled_tasks": int(len(entries)),
        "compile_ms_total": float(sum(float(e.get("compile_ms", 0.0)) for e in entries)),
        "call_tir_after_legalize_sum": 0,
        "call_tir_after_fuse_tir_sum": 0,
        "memory_by_scan_alloc_storage_total_bytes_sum": 0,
        "memory_by_scan_alloc_storage_nonconst_bytes_sum": 0,
        "memory_by_scan_alloc_storage_max_bytes_max": 0,
        "memory_by_tvm_estimator_stage": None,
        "memory_by_tvm_estimator_render": None,
    }
    estimator_stage: Optional[str] = None
    estimator_render: Optional[str] = None
    for e in entries:
        ir_stats = e.get("ir_stats") or {}
        if "after_legalize" in ir_stats:
            out["call_tir_after_legalize_sum"] += int((ir_stats["after_legalize"] or {}).get("call_tir", 0))
        if "after_fuse_tir" in ir_stats:
            out["call_tir_after_fuse_tir_sum"] += int((ir_stats["after_fuse_tir"] or {}).get("call_tir", 0))

        mem = e.get("memory_stats") or {}
        if estimator_stage is None and isinstance(mem, dict):
            estimator_stage = mem.get("by_tvm_estimator_stage")
        if estimator_render is None and isinstance(mem, dict):
            s = mem.get("by_tvm_estimator")
            if isinstance(s, str):
                estimator_render = s
        by_scan = (mem.get("by_scan") or {}) if isinstance(mem, dict) else {}
        out["memory_by_scan_alloc_storage_total_bytes_sum"] += int(by_scan.get("alloc_storage_total_bytes", 0))
        out["memory_by_scan_alloc_storage_nonconst_bytes_sum"] += int(by_scan.get("alloc_storage_nonconst_bytes", 0))
        out["memory_by_scan_alloc_storage_max_bytes_max"] = max(
            int(out["memory_by_scan_alloc_storage_max_bytes_max"]),
            int(by_scan.get("alloc_storage_max_bytes", 0)),
        )

    out["memory_by_tvm_estimator_stage"] = estimator_stage
    out["memory_by_tvm_estimator_render"] = estimator_render
    return out


def _build_workload(*, name: str) -> Tuple[nn.Module, torch.Tensor, float]:
    torch.manual_seed(0)
    if name == "mlp":
        model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))
        x0 = torch.randn(4, 16)
        eps = 0.1
        return model, x0, eps
    raise ValueError(f"unsupported workload: {name}")


def _time_run(
    fn,
    *,
    warmup: int,
    iters: int,
) -> Tuple[float, Dict[str, float]]:
    for _ in range(int(warmup)):
        fn()
    samples_ms: List[float] = []
    for _ in range(int(iters)):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        samples_ms.append((t1 - t0) / 1e6)
    avg = float(sum(samples_ms) / max(1, len(samples_ms)))
    pct = _percentiles_ms(samples_ms)
    return avg, pct


def _bench_one(
    *,
    workload: str,
    batch: Optional[int],
    eps_override: Optional[float],
    partition_policy: PartitionPolicy,
    min_tasks: int,
    reuse_on: bool,
    memory_plan_mode: MemoryPlanMode,
    fusion_on: bool,
    warmup: int,
    iters: int,
    include_auto_lirpa: bool,
    check_correctness: bool,
) -> Dict[str, Any]:
    model, x0, eps = _build_workload(name=workload)
    if batch is not None:
        if workload != "mlp":
            raise ValueError("--batch only supported for workload=mlp currently")
        x0 = torch.randn(int(batch), x0.shape[1])
    if eps_override is not None:
        eps = float(eps_override)
    model.eval()
    with torch.no_grad():
        program = import_torch(model, (x0,), export_mode="export", normalize=True)

    cfg = PlannerConfig(
        enable_task_graph=(partition_policy != PartitionPolicy.V0_SINGLE_TASK),
        enable_storage_reuse=bool(reuse_on),
        partition=PartitionOptions(policy=partition_policy, min_tasks=int(min_tasks)),
        storage_reuse=StorageReuseOptions(
            enabled=bool(reuse_on),
            key_mode=ReuseKeyMode.STRICT,
            policy=ReusePolicy.LIFO,
        ),
        debug=PlannerDebugOptions(dump_config=True, validate_after_each_pass=False, dump_plan=False),
    )

    t0 = time.perf_counter_ns()
    bundle = plan(program, config=cfg)
    t1 = time.perf_counter_ns()
    plan_ms_total = (t1 - t0) / 1e6

    module = bundle.task_module
    input_spec = LinfInputSpec(value_name=program.graph.inputs[0], center=x0, eps=eps)

    # Reference (PythonTaskExecutor).
    py_out = run_ibp_scheduled(module, input_spec, executor=PythonTaskExecutor())

    # TVM executor (task-level RELAX_OPS).
    tvm_exec = TVMTaskExecutor(
        options=TVMExecutorOptions(
            target="llvm",
            kernel_style="relax",
            enable_task_relax_ops=True,
            enable_task_fusion_pipeline=bool(fusion_on),
            task_fuse_opt_level=2,
            memory_plan_mode=memory_plan_mode,
            tir_var_upper_bound=None,
        )
    )

    def _run_tvm_once() -> None:
        _ = run_ibp_scheduled(module, input_spec, executor=tvm_exec)

    # Trigger compilation once (counted separately from steady-state runtime).
    cache0 = tvm_exec.get_task_compile_cache_stats()
    t_compile0 = time.perf_counter_ns()
    _ = _run_tvm_once()
    t_compile1 = time.perf_counter_ns()
    compile_first_run_ms = (t_compile1 - t_compile0) / 1e6
    cache1 = tvm_exec.get_task_compile_cache_stats()

    def _delta(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
        keys = set(a.keys()) | set(b.keys())
        return {k: int(b.get(k, 0)) - int(a.get(k, 0)) for k in sorted(keys)}

    run_ms_avg, run_pct = _time_run(_run_tvm_once, warmup=warmup, iters=iters)
    compile_stats = tvm_exec.get_compile_stats()
    compile_agg = _aggregate_task_compile_stats(compile_stats)
    compile_cache_stats = tvm_exec.get_task_compile_cache_stats()

    tvm_out = run_ibp_scheduled(module, input_spec, executor=tvm_exec)

    correctness: Dict[str, Any] = {
        "bounds_allclose_to_python": None,
        "bounds_allclose_to_auto_lirpa": None,
        "python_vs_tvm_max_abs_diff_lb": None,
        "python_vs_tvm_max_abs_diff_ub": None,
        "python_vs_tvm_max_rel_diff_lb": None,
        "python_vs_tvm_max_rel_diff_ub": None,
        "python_vs_auto_lirpa_max_abs_diff_lb": None,
        "python_vs_auto_lirpa_max_abs_diff_ub": None,
        "python_vs_auto_lirpa_max_rel_diff_lb": None,
        "python_vs_auto_lirpa_max_rel_diff_ub": None,
    }
    if check_correctness:
        lb_diff = (py_out.lower - tvm_out.lower).abs()
        ub_diff = (py_out.upper - tvm_out.upper).abs()
        lb_rel = lb_diff / py_out.lower.abs().clamp_min(1e-12)
        ub_rel = ub_diff / py_out.upper.abs().clamp_min(1e-12)
        correctness["bounds_allclose_to_python"] = bool(
            torch.allclose(py_out.lower, tvm_out.lower, atol=1e-5, rtol=1e-5)
            and torch.allclose(py_out.upper, tvm_out.upper, atol=1e-5, rtol=1e-5)
        )
        correctness["python_vs_tvm_max_abs_diff_lb"] = float(lb_diff.max().item())
        correctness["python_vs_tvm_max_abs_diff_ub"] = float(ub_diff.max().item())
        correctness["python_vs_tvm_max_rel_diff_lb"] = float(lb_rel.max().item())
        correctness["python_vs_tvm_max_rel_diff_ub"] = float(ub_rel.max().item())

    baseline: Dict[str, Any] = {"auto_lirpa": None}
    if include_auto_lirpa:
        lirpa = _try_import_auto_lirpa()
        if lirpa is not None:
            _, BoundedModule, BoundedTensor, PerturbationLpNorm = lirpa
            t_setup0 = time.perf_counter_ns()
            lirpa_model = BoundedModule(model, torch.empty_like(x0), device=x0.device)
            ptb = PerturbationLpNorm(norm=float("inf"), eps=float(eps))
            bounded_x = BoundedTensor(x0, ptb)
            t_setup1 = time.perf_counter_ns()
            setup_ms = (t_setup1 - t_setup0) / 1e6

            def _run_lirpa_once() -> None:
                _ = lirpa_model.compute_bounds(x=(bounded_x,), method="IBP")

            lirpa_ms_avg, lirpa_pct = _time_run(_run_lirpa_once, warmup=warmup, iters=iters)
            lb, ub = lirpa_model.compute_bounds(x=(bounded_x,), method="IBP")
            baseline["auto_lirpa"] = {
                "method": "IBP",
                "setup_ms": float(setup_ms),
                "compute_bounds_ms_avg": float(lirpa_ms_avg),
                "compute_bounds_ms_p50": float(lirpa_pct["p50"]),
                "compute_bounds_ms_p95": float(lirpa_pct["p95"]),
            }
            if check_correctness:
                correctness["bounds_allclose_to_auto_lirpa"] = bool(
                    torch.allclose(py_out.lower, lb, atol=1e-6, rtol=1e-5)
                    and torch.allclose(py_out.upper, ub, atol=1e-6, rtol=1e-5)
                )
                l2 = (py_out.lower - lb).abs()
                u2 = (py_out.upper - ub).abs()
                l2_rel = l2 / py_out.lower.abs().clamp_min(1e-12)
                u2_rel = u2 / py_out.upper.abs().clamp_min(1e-12)
                correctness["python_vs_auto_lirpa_max_abs_diff_lb"] = float(l2.max().item())
                correctness["python_vs_auto_lirpa_max_abs_diff_ub"] = float(u2.max().item())
                correctness["python_vs_auto_lirpa_max_rel_diff_lb"] = float(l2_rel.max().item())
                correctness["python_vs_auto_lirpa_max_rel_diff_ub"] = float(u2_rel.max().item())
        else:
            baseline["auto_lirpa"] = {"available": False}

    out: Dict[str, Any] = {
        "schema_version": "0.1",
        "meta": {
            "git_commit": _git_short_sha(),
            "timestamp": int(time.time()),
            "time_utc": _utc_now_iso(),
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "torch": getattr(torch, "__version__", ""),
            "torch_num_threads": int(torch.get_num_threads()),
            "tvm_home": os.environ.get("TVM_HOME", ""),
            "seed": 0,
            "tvm": None,
        },
        "workload": {
            "model": workload,
            "input_shape": list(x0.shape),
            "eps": float(eps),
            "spec": "none",
            "domain": "interval_ibp",
        },
        "config": {
            "planner_config_dump": bundle.meta.get("config_dump"),
            "tvm_options": _jsonable(tvm_exec.options),
        },
        "planner": {
            "num_tasks": int(len(module.tasks)),
            "num_edges": int(len(module.task_graph.edges) if module.task_graph is not None else 0),
            "timings_ms": bundle.meta.get("timings_ms"),
            "plan_ms_total": float(plan_ms_total),
            "storage": {
                "logical_buffers": int(module.storage_plan.num_logical_buffers()),
                "physical_buffers": int(module.storage_plan.num_physical_buffers()),
                "logical_bytes_est": int(
                    sum(
                        buffer_size_bytes(spec) or 0
                        for spec in module.storage_plan.buffers.values()
                        if str(getattr(spec, "scope", "global")) == "global"
                    )
                ),
                "physical_bytes_est": int(
                    sum(
                        buffer_size_bytes(spec) or 0
                        for spec in (module.storage_plan.physical_buffers or module.storage_plan.buffers).values()
                        if str(getattr(spec, "scope", "global")) == "global"
                    )
                ),
                "reuse_stats": _jsonable(bundle.meta.get("reuse_stats")),
            },
        },
        "tvm": {
            "tir_var_upper_bound": None,
            "tir_var_upper_bound_source": "none",
            "tir_var_upper_bound_scope": "func_signature_only",
            "compile_stats_agg": compile_agg,
            "compile_cache_stats": compile_cache_stats,
            "compile_cache_stats_delta_compile_first_run": _delta(cache0, cache1),
        },
        "runtime": {
            "warmup": int(warmup),
            "iters": int(iters),
            "compile_first_run_ms": float(compile_first_run_ms),
            "run_ms_avg": float(run_ms_avg),
            "run_ms_p50": float(run_pct["p50"]),
            "run_ms_p95": float(run_pct["p95"]),
        },
        "baseline": baseline,
        "correctness": correctness,
    }
    try:
        import tvm  # noqa: PLC0415

        out["meta"]["tvm"] = getattr(tvm, "__version__", None)
    except Exception:
        pass
    return out


def iter_default_matrix() -> Iterable[Tuple[PartitionPolicy, bool, MemoryPlanMode, bool]]:
    for partition_policy in (PartitionPolicy.V0_SINGLE_TASK, PartitionPolicy.V2_BASELINE):
        for reuse_on in (False, True):
            for memory_plan_mode in (MemoryPlanMode.DEFAULT, MemoryPlanMode.DISABLE_STATIC_PLAN):
                for fusion_on in (False, True):
                    yield partition_policy, reuse_on, memory_plan_mode, fusion_on


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--workload", choices=["mlp"], default="mlp")
    p.add_argument("--batch", type=int, default=None, help="Override batch size for the workload input (mlp only).")
    p.add_argument("--eps", type=float, default=None, help="Override Linf eps for the workload.")
    p.add_argument("--matrix", choices=["default", "small"], default="default")
    p.add_argument("--min-tasks", type=int, default=2)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--no-auto-lirpa", action="store_true")
    p.add_argument("--no-check", action="store_true")
    p.add_argument("--output", type=str, default="")
    args = p.parse_args(argv)

    runs: List[Dict[str, Any]] = []
    matrix_iter: Iterable[Tuple[PartitionPolicy, bool, MemoryPlanMode, bool]]
    if args.matrix == "small":
        matrix_iter = [
            (PartitionPolicy.V2_BASELINE, False, MemoryPlanMode.DEFAULT, True),
        ]
    else:
        matrix_iter = iter_default_matrix()

    for partition_policy, reuse_on, memory_plan_mode, fusion_on in matrix_iter:
        runs.append(
            _bench_one(
                workload=str(args.workload),
                batch=args.batch,
                eps_override=args.eps,
                partition_policy=partition_policy,
                min_tasks=int(args.min_tasks),
                reuse_on=bool(reuse_on),
                memory_plan_mode=memory_plan_mode,
                fusion_on=bool(fusion_on),
                warmup=int(args.warmup),
                iters=int(args.iters),
                include_auto_lirpa=not bool(args.no_auto_lirpa),
                check_correctness=not bool(args.no_check),
            )
        )

    # JSONL output (one line per run).
    lines = [json.dumps(r, ensure_ascii=False) for r in runs]
    text = "\n".join(lines) + "\n"
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        try:
            sys.stdout.write(text)
        except BrokenPipeError:
            # Allow piping to tools like `head` without raising.
            try:
                sys.stdout.close()
            except Exception:
                pass
            return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
