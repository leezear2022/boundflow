#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from typing import Any, Dict, List

import torch
import torch.nn as nn

from boundflow.frontends.pytorch.frontend import import_torch
from boundflow.planner.core import PlannerConfig
from boundflow.planner.options import PartitionOptions, PartitionPolicy
from boundflow.planner.pipeline import plan
from boundflow.planner.storage_reuse import ReuseKeyMode, ReusePolicy, StorageReuseOptions
from boundflow.runtime.scheduler import run_ibp_scheduled
from boundflow.runtime.task_executor import LinfInputSpec, PythonTaskExecutor
from boundflow.runtime.tvm_executor import MemoryPlanMode, TVMExecutorOptions, TVMTaskExecutor


def _git_short_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return ""


def _jsonable_reuse_stats(stats: Any) -> Any:
    if stats is None:
        return None
    miss = getattr(stats, "miss_reasons", None) or {}
    overlap = getattr(stats, "overlap_blockers", None) or {}
    return {
        "pool_hit": int(getattr(stats, "pool_hit", 0)),
        "pool_miss": int(getattr(stats, "pool_miss", 0)),
        "bytes_saved_est": int(getattr(stats, "bytes_saved_est", 0)),
        "unknown_bytes_buffers": int(getattr(stats, "unknown_bytes_buffers", 0)),
        "max_free_pool_keys": int(getattr(stats, "max_free_pool_keys", 0)),
        "max_free_pool_buffers": int(getattr(stats, "max_free_pool_buffers", 0)),
        "miss_reasons": {str(k.value if hasattr(k, "value") else k): int(v) for k, v in miss.items()},
        "overlap_blockers": {str(k): int(v) for k, v in overlap.items()},
    }


def _aggregate_task_compile_stats(compile_stats: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    entries = [v for v in (compile_stats or {}).values() if isinstance(v, dict) and v.get("kind") == "task_relax_ops"]
    out: Dict[str, Any] = {
        "num_compiled_tasks": int(len(entries)),
        "compile_ms_sum": float(sum(float(e.get("compile_ms", 0.0)) for e in entries)),
        "call_tir_after_legalize_sum": 0,
        "call_tir_after_fuse_tir_sum": 0,
        "memory_alloc_storage_sum": 0,
        "memory_alloc_tensor_sum": 0,
        "memory_alloc_storage_total_bytes_sum": 0,
        "memory_alloc_storage_max_bytes_max": 0,
        "memory_alloc_storage_nonconst_bytes_sum": 0,
    }
    for e in entries:
        ir_stats = e.get("ir_stats") or {}
        if "after_legalize" in ir_stats:
            out["call_tir_after_legalize_sum"] += int((ir_stats["after_legalize"] or {}).get("call_tir", 0))
        if "after_fuse_tir" in ir_stats:
            out["call_tir_after_fuse_tir_sum"] += int((ir_stats["after_fuse_tir"] or {}).get("call_tir", 0))
        mem = e.get("memory_stats") or {}
        out["memory_alloc_storage_sum"] += int(mem.get("alloc_storage", 0))
        out["memory_alloc_tensor_sum"] += int(mem.get("alloc_tensor", 0))
        out["memory_alloc_storage_total_bytes_sum"] += int(mem.get("alloc_storage_total_bytes", 0))
        out["memory_alloc_storage_max_bytes_max"] = max(
            int(out["memory_alloc_storage_max_bytes_max"]),
            int(mem.get("alloc_storage_max_bytes", 0)),
        )
        out["memory_alloc_storage_nonconst_bytes_sum"] += int(mem.get("alloc_storage_nonconst_bytes", 0))
    return out


def _run_one(
    *,
    reuse_enabled: bool,
    static_plan_enabled: bool,
    min_tasks: int,
    iters: int,
    warmup: int,
    check_correctness: bool,
) -> Dict[str, Any]:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8), nn.ReLU())
    x0 = torch.randn(4, 16)
    program = import_torch(model, (x0,), export_mode="export", normalize=True)

    cfg = PlannerConfig(
        enable_task_graph=True,
        enable_storage_reuse=bool(reuse_enabled),
        partition=PartitionOptions(policy=PartitionPolicy("v2_baseline"), min_tasks=int(min_tasks)),
        storage_reuse=StorageReuseOptions(
            enabled=bool(reuse_enabled),
            key_mode=ReuseKeyMode.STRICT,
            policy=ReusePolicy.LIFO,
        ),
    )
    bundle = plan(program, config=cfg)
    module = bundle.task_module

    input_spec = LinfInputSpec(value_name=program.graph.inputs[0], center=x0, eps=0.1)

    if check_correctness:
        py = run_ibp_scheduled(module, input_spec, executor=PythonTaskExecutor())
    else:
        py = None

    mem_mode = MemoryPlanMode.DEFAULT if static_plan_enabled else MemoryPlanMode.DISABLE_STATIC_PLAN
    tvm_exec = TVMTaskExecutor(
        options=TVMExecutorOptions(
            target="llvm",
            kernel_style="relax",
            enable_task_relax_ops=True,
            enable_task_fusion_pipeline=True,
            task_fuse_opt_level=2,
            memory_plan_mode=mem_mode,
        )
    )

    # Warmup (also triggers compilation).
    out = None
    for _ in range(int(warmup)):
        out = run_ibp_scheduled(module, input_spec, executor=tvm_exec)

    if check_correctness and py is not None and out is not None:
        assert torch.allclose(py.lower, out.lower, atol=1e-5, rtol=1e-5)
        assert torch.allclose(py.upper, out.upper, atol=1e-5, rtol=1e-5)

    # Timed run (post-compile).
    t0 = time.perf_counter_ns()
    for _ in range(int(iters)):
        out = run_ibp_scheduled(module, input_spec, executor=tvm_exec)
    t1 = time.perf_counter_ns()
    run_ms_avg = ((t1 - t0) / 1e6) / max(1, int(iters))

    compile_stats = tvm_exec.get_compile_stats()
    compile_agg = _aggregate_task_compile_stats(compile_stats)

    return {
        "reuse_enabled": bool(reuse_enabled),
        "static_plan_enabled": bool(static_plan_enabled),
        "planner": {
            "num_tasks": int(len(module.tasks)),
            "num_edges": int(len(module.task_graph.edges) if module.task_graph is not None else 0),
            "storage_plan": {
                "logical": int(module.storage_plan.num_logical_buffers()),
                "physical": int(module.storage_plan.num_physical_buffers()),
            },
            "reuse_stats": _jsonable_reuse_stats(bundle.meta.get("reuse_stats")),
            "planner_steps": bundle.meta.get("planner_steps"),
            "timings_ms": bundle.meta.get("timings_ms"),
            "config_dump": bundle.meta.get("config_dump"),
        },
        "tvm": {
            "memory_plan_mode": str(mem_mode.value),
            "compile_tasks": compile_agg,
        },
        "runtime": {
            "run_ms_avg": float(run_ms_avg),
            "iters": int(iters),
            "warmup": int(warmup),
        },
        "env": {
            "git": _git_short_sha(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": getattr(torch, "__version__", ""),
            "tvm_home": os.environ.get("TVM_HOME", ""),
        },
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--min-tasks", type=int, default=2)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--check-correctness", action="store_true")
    args = p.parse_args()

    runs: List[Dict[str, Any]] = []
    for reuse_enabled in (False, True):
        for static_plan_enabled in (False, True):
            runs.append(
                _run_one(
                    reuse_enabled=reuse_enabled,
                    static_plan_enabled=static_plan_enabled,
                    min_tasks=int(args.min_tasks),
                    iters=int(args.iters),
                    warmup=int(args.warmup),
                    check_correctness=bool(args.check_correctness),
                )
            )

    print(json.dumps({"runs": runs}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

