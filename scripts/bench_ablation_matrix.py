#!/usr/bin/env python
from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import os
import platform
import traceback
import socket
import subprocess
import sys
import time
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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


class _AutoLiRPABaselineCacheEntry(Tuple[Dict[str, Any], torch.Tensor, torch.Tensor]):
    """
    Internal cache entry: (jsonable_payload, lb, ub).

    Note: lb/ub are torch tensors and must not be written to JSONL directly.
    """


_AUTO_LIRPA_BASELINE_CACHE: Dict[Tuple[Any, ...], Tuple[Dict[str, Any], torch.Tensor, torch.Tensor]] = {}


def _stable_jsonable_digest(x: Any) -> str:
    try:
        s = json.dumps(x, ensure_ascii=False, sort_keys=True, default=repr)
    except Exception:
        s = repr(x)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _auto_lirpa_baseline_cached(
    *,
    workload: str,
    model: nn.Module,
    x0: torch.Tensor,
    eps: float,
    spec: Any,
    method: str,
    warmup: int,
    iters: int,
) -> Tuple[Dict[str, Any], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Compute auto_LiRPA baseline once per (workload,input_shape,eps,method,warmup,iters) and reuse across matrix points.
    """
    key = (
        str(workload),
        tuple(int(d) for d in x0.shape),
        str(x0.dtype),
        str(x0.device),
        float(eps),
        str(_stable_jsonable_digest(spec)),
        str(method),
        int(warmup),
        int(iters),
    )
    baseline_key = hashlib.sha256(repr(key).encode("utf-8")).hexdigest()[:16]
    if key in _AUTO_LIRPA_BASELINE_CACHE:
        payload, lb, ub = _AUTO_LIRPA_BASELINE_CACHE[key]
        payload2 = dict(payload)
        payload2["cache_hit"] = True
        payload2["baseline_key"] = baseline_key
        payload2["spec_hash"] = str(key[5])
        return payload2, lb, ub

    lirpa = _try_import_auto_lirpa()
    if lirpa is None:
        payload = {
            "available": False,
            "reason": "import_failed",
            "method": str(method),
            "device": str(x0.device),
            "cache_hit": False,
            "baseline_key": baseline_key,
            "spec_hash": str(key[5]),
        }
        _AUTO_LIRPA_BASELINE_CACHE[key] = (payload, torch.empty(0), torch.empty(0))
        return dict(payload), None, None

    auto_LiRPA, BoundedModule, BoundedTensor, PerturbationLpNorm = lirpa
    version = getattr(auto_LiRPA, "__version__", "") or ""

    try:
        t_init0 = time.perf_counter_ns()
        lirpa_model = BoundedModule(model, torch.empty_like(x0), device=x0.device)
        ptb = PerturbationLpNorm(norm=float("inf"), eps=float(eps))
        bounded_x = BoundedTensor(x0, ptb)
        t_init1 = time.perf_counter_ns()
        init_ms = (t_init1 - t_init0) / 1e6

        # Cold run: one compute_bounds call with outputs captured.
        t_cold0 = time.perf_counter_ns()
        lb, ub = lirpa_model.compute_bounds(x=(bounded_x,), method=str(method))
        t_cold1 = time.perf_counter_ns()
        run_ms_cold = (t_cold1 - t_cold0) / 1e6

        def _run_once() -> None:
            _ = lirpa_model.compute_bounds(x=(bounded_x,), method=str(method))

        run_ms_avg, run_pct = _time_run(_run_once, warmup=warmup, iters=iters)

        payload = {
            "available": True,
            "reason": "",
            "version": str(version),
            "method": str(method),
            "device": str(x0.device),
            "init_ms": float(init_ms),
            "run_ms_cold": float(run_ms_cold),
            "run_ms_avg": float(run_ms_avg),
            "run_ms_p50": float(run_pct["p50"]),
            "run_ms_p95": float(run_pct["p95"]),
            "warmup": int(warmup),
            "iters": int(iters),
            "cache_hit": False,
            "baseline_key": baseline_key,
            "spec_hash": str(key[5]),
        }
        _AUTO_LIRPA_BASELINE_CACHE[key] = (payload, lb.detach().cpu(), ub.detach().cpu())
        return dict(payload), lb.detach().cpu(), ub.detach().cpu()
    except Exception as e:
        payload = {
            "available": False,
            "reason": f"compute_failed:{type(e).__name__}",
            "method": str(method),
            "device": str(x0.device),
            "cache_hit": False,
            "baseline_key": baseline_key,
            "spec_hash": str(key[5]),
        }
        _AUTO_LIRPA_BASELINE_CACHE[key] = (payload, torch.empty(0), torch.empty(0))
        return dict(payload), None, None


def _try_import_tvm_version() -> Optional[str]:
    try:
        import tvm  # type: ignore

        return getattr(tvm, "__version__", None)
    except Exception:
        return None


def _sha256_text(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()


def _digest_text(xs: List[str]) -> str:
    h = hashlib.sha256()
    for x in xs:
        h.update((x or "").encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:16]


def _compile_keyset_summary(compile_stats: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    keys = sorted([str(k) for k in (compile_stats or {}).keys()])
    return {
        "compile_keyset_size": int(len(keys)),
        "compile_keyset_digest": _digest_text(keys),
    }


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


def _build_workload(*, name: str, batch: Optional[int] = None, eps_override: Optional[float] = None) -> Tuple[nn.Module, torch.Tensor, float]:
    torch.manual_seed(0)
    if name == "mlp":
        model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))
        b = int(batch) if batch is not None else 4
        x0 = torch.randn(b, 16)
        eps = float(eps_override) if eps_override is not None else 0.1
        return model, x0, eps
    if name == "mnist_cnn":
        # Minimal CNN-like workload (conv2d + relu + flatten + linear).
        model = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(4 * 28 * 28, 10),
        )
        b = int(batch) if batch is not None else 4
        x0 = torch.randn(b, 1, 28, 28)
        eps = float(eps_override) if eps_override is not None else 0.05
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


def _device_fingerprint() -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": None,
        "cuda_name": None,
    }
    if torch.cuda.is_available():
        try:
            out["cuda_device"] = int(torch.cuda.current_device())
            out["cuda_name"] = str(torch.cuda.get_device_name(out["cuda_device"]))
        except Exception:
            pass
    return out


def _env_flags() -> Dict[str, Any]:
    keys = [
        "PYTHONHASHSEED",
        "BOUNDFLOW_QUIET",
        "TVM_HOME",
        "TVM_FFI_DISABLE_TORCH_C_DLPACK",
        "TVM_FFI_CACHE_DIR",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    ]
    return {k: os.environ.get(k) for k in keys if k in os.environ}


def _bench_one_safe(
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
    tvm_enabled: bool,
    tol_atol: float,
    tol_rtol: float,
) -> Dict[str, Any]:
    requested_config = {
        "partition_policy": str(getattr(partition_policy, "value", partition_policy)),
        "min_tasks": int(min_tasks),
        "reuse_on": bool(reuse_on),
        "memory_plan_mode": str(getattr(memory_plan_mode, "value", memory_plan_mode)),
        "fusion_on": bool(fusion_on),
        "warmup": int(warmup),
        "iters": int(iters),
        "include_auto_lirpa": bool(include_auto_lirpa),
        "check_correctness": bool(check_correctness),
        "tvm_enabled": bool(tvm_enabled),
        "tol": {"atol": float(tol_atol), "rtol": float(tol_rtol)},
    }

    out: Dict[str, Any] = {
        "schema_version": "0.1",
        "status": "ok",
        "error": None,
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
            "tvm": _try_import_tvm_version(),
            "device": _device_fingerprint(),
            "env_flags": _env_flags(),
            "seed": 0,
        },
        "workload": {
            "model": str(workload),
            "input_shape": None,
            "eps": None,
            "spec": "none",
            "domain": "interval_ibp",
        },
        "config": {
            "requested": requested_config,
            "planner_config_dump": None,
            "tvm_options": None,
        },
        "planner": None,
        "tvm": {
            "available": None,
            "tir_var_upper_bound": None,
            "tir_var_upper_bound_source": "none",
            "tir_var_upper_bound_scope": "func_signature_only",
            "compile_stats_agg": None,
            "compile_cache_stats": None,
            "compile_cache_stats_delta_compile_first_run": None,
        },
        "runtime": {
            "warmup": int(warmup),
            "iters": int(iters),
            "compile_first_run_ms": None,
            "run_ms_cold": None,
            "run_ms_avg": None,
            "run_ms_p50": None,
            "run_ms_p95": None,
        },
        "baseline": {"auto_lirpa": None},
        "correctness": {
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
            "python_vs_tvm_gate": None,
            "python_vs_auto_lirpa_gate": None,
        },
    }

    try:
        model, x0, eps = _build_workload(name=workload, batch=batch, eps_override=eps_override)
        out["workload"]["input_shape"] = list(x0.shape)
        out["workload"]["eps"] = float(eps)

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

        out["config"]["planner_config_dump"] = bundle.meta.get("config_dump")
        out["planner"] = {
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
        }

        # Reference (PythonTaskExecutor).
        py_out = run_ibp_scheduled(module, input_spec, executor=PythonTaskExecutor())

        # Optional auto_LiRPA baseline.
        if include_auto_lirpa:
            base, lb, ub = _auto_lirpa_baseline_cached(
                workload=str(workload),
                model=model,
                x0=x0,
                eps=float(eps),
                spec=out.get("workload", {}).get("spec"),
                method="IBP",
                warmup=warmup,
                iters=iters,
            )
            # Backward-compatible aliases kept for older postprocess/notes.
            if base.get("available") is True:
                base.setdefault("setup_ms", base.get("init_ms"))
                base.setdefault("compute_bounds_ms_avg", base.get("run_ms_avg"))
                base.setdefault("compute_bounds_ms_p50", base.get("run_ms_p50"))
                base.setdefault("compute_bounds_ms_p95", base.get("run_ms_p95"))
            out["baseline"]["auto_lirpa"] = base
            if check_correctness and lb is not None and ub is not None:
                lb_t = lb.to(py_out.lower.device)
                ub_t = ub.to(py_out.upper.device)
                ok = bool(
                    torch.allclose(py_out.lower, lb_t, atol=float(tol_atol), rtol=float(tol_rtol))
                    and torch.allclose(py_out.upper, ub_t, atol=float(tol_atol), rtol=float(tol_rtol))
                )
                out["correctness"]["bounds_allclose_to_auto_lirpa"] = ok
                l2 = (py_out.lower - lb_t).abs()
                u2 = (py_out.upper - ub_t).abs()
                l2_rel = l2 / py_out.lower.abs().clamp_min(1e-12)
                u2_rel = u2 / py_out.upper.abs().clamp_min(1e-12)
                out["correctness"]["python_vs_auto_lirpa_max_abs_diff_lb"] = float(l2.max().item())
                out["correctness"]["python_vs_auto_lirpa_max_abs_diff_ub"] = float(u2.max().item())
                out["correctness"]["python_vs_auto_lirpa_max_rel_diff_lb"] = float(l2_rel.max().item())
                out["correctness"]["python_vs_auto_lirpa_max_rel_diff_ub"] = float(u2_rel.max().item())
                out["correctness"]["python_vs_auto_lirpa_gate"] = {
                    "ref": "auto_lirpa",
                    "ok": ok,
                    "tol": {"atol": float(tol_atol), "rtol": float(tol_rtol)},
                    "max_abs_diff_lb": float(l2.max().item()),
                    "max_abs_diff_ub": float(u2.max().item()),
                    "max_rel_diff_lb": float(l2_rel.max().item()),
                    "max_rel_diff_ub": float(u2_rel.max().item()),
                }

        if not tvm_enabled:
            out["tvm"]["available"] = False
            out["tvm"]["reason"] = "disabled_by_flag"
            return out

        # TVM executor (task-level RELAX_OPS).
        out["tvm"]["available"] = True
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
        out["config"]["tvm_options"] = _jsonable(tvm_exec.options)

        def _run_tvm_once() -> None:
            _ = run_ibp_scheduled(module, input_spec, executor=tvm_exec)

        # Trigger compilation once (counted separately from steady-state runtime).
        cache0 = tvm_exec.get_task_compile_cache_stats()
        t_compile0 = time.perf_counter_ns()
        _run_tvm_once()
        t_compile1 = time.perf_counter_ns()
        compile_first_run_ms = (t_compile1 - t_compile0) / 1e6
        cache1 = tvm_exec.get_task_compile_cache_stats()

        # Cold run after compilation (captures VM init/cache effects, but should exclude compilation).
        t_cold0 = time.perf_counter_ns()
        _run_tvm_once()
        t_cold1 = time.perf_counter_ns()
        run_ms_cold = (t_cold1 - t_cold0) / 1e6

        def _delta(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
            keys = set(a.keys()) | set(b.keys())
            return {k: int(b.get(k, 0)) - int(a.get(k, 0)) for k in sorted(keys)}

        run_ms_avg, run_pct = _time_run(_run_tvm_once, warmup=warmup, iters=iters)
        compile_stats = tvm_exec.get_compile_stats()
        compile_agg = _aggregate_task_compile_stats(compile_stats)
        compile_cache_stats = tvm_exec.get_task_compile_cache_stats()
        keyset = _compile_keyset_summary(compile_stats)

        tvm_out = run_ibp_scheduled(module, input_spec, executor=tvm_exec)

        out["tvm"].update(
            {
                "compile_stats_agg": compile_agg,
                "compile_cache_stats": compile_cache_stats,
                "compile_cache_stats_delta_compile_first_run": _delta(cache0, cache1),
                "compile_cache_tag": str(getattr(tvm_exec.options, "compile_cache_tag", "")),
                **keyset,
            }
        )
        out["runtime"].update(
            {
                "compile_first_run_ms": float(compile_first_run_ms),
                "run_ms_cold": float(run_ms_cold),
                "run_ms_avg": float(run_ms_avg),
                "run_ms_p50": float(run_pct["p50"]),
                "run_ms_p95": float(run_pct["p95"]),
            }
        )

        if check_correctness:
            lb_diff = (py_out.lower - tvm_out.lower).abs()
            ub_diff = (py_out.upper - tvm_out.upper).abs()
            lb_rel = lb_diff / py_out.lower.abs().clamp_min(1e-12)
            ub_rel = ub_diff / py_out.upper.abs().clamp_min(1e-12)
            ok = bool(
                torch.allclose(py_out.lower, tvm_out.lower, atol=float(tol_atol), rtol=float(tol_rtol))
                and torch.allclose(py_out.upper, tvm_out.upper, atol=float(tol_atol), rtol=float(tol_rtol))
            )
            out["correctness"]["bounds_allclose_to_python"] = ok
            out["correctness"]["python_vs_tvm_max_abs_diff_lb"] = float(lb_diff.max().item())
            out["correctness"]["python_vs_tvm_max_abs_diff_ub"] = float(ub_diff.max().item())
            out["correctness"]["python_vs_tvm_max_rel_diff_lb"] = float(lb_rel.max().item())
            out["correctness"]["python_vs_tvm_max_rel_diff_ub"] = float(ub_rel.max().item())
            out["correctness"]["python_vs_tvm_gate"] = {
                "ref": "python_task_executor",
                "ok": ok,
                "tol": {"atol": float(tol_atol), "rtol": float(tol_rtol)},
                "max_abs_diff_lb": float(lb_diff.max().item()),
                "max_abs_diff_ub": float(ub_diff.max().item()),
                "max_rel_diff_lb": float(lb_rel.max().item()),
                "max_rel_diff_ub": float(ub_rel.max().item()),
            }

        return out
    except Exception as e:
        tb = traceback.format_exc()
        out["status"] = "fail"
        out["error"] = {
            "error_type": type(e).__name__,
            "error_msg": str(e),
            "traceback_hash": _sha256_text(tb),
            "traceback": tb,
        }
        # Best-effort: mark TVM unavailable if import fails, even if tvm_enabled requested.
        if out.get("tvm", {}).get("available") is None:
            out["tvm"]["available"] = False
            out["tvm"]["reason"] = "error"
        return out


def iter_default_matrix() -> Iterable[Tuple[PartitionPolicy, bool, MemoryPlanMode, bool]]:
    for partition_policy in (PartitionPolicy.V0_SINGLE_TASK, PartitionPolicy.V2_BASELINE):
        for reuse_on in (False, True):
            for memory_plan_mode in (MemoryPlanMode.DEFAULT, MemoryPlanMode.DISABLE_STATIC_PLAN):
                for fusion_on in (False, True):
                    yield partition_policy, reuse_on, memory_plan_mode, fusion_on


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--workload", choices=["mlp", "mnist_cnn"], default="mlp")
    p.add_argument("--batch", type=int, default=None, help="Override batch size for the workload input.")
    p.add_argument("--eps", type=float, default=None, help="Override Linf eps for the workload (Linf).")
    p.add_argument("--matrix", choices=["default", "small"], default="default")
    p.add_argument("--min-tasks", type=int, default=2)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--no-auto-lirpa", action="store_true")
    p.add_argument("--no-tvm", action="store_true", help="Skip TVM executor (python-only).")
    p.add_argument("--no-check", action="store_true")
    p.add_argument("--tol-atol", type=float, default=1e-5)
    p.add_argument("--tol-rtol", type=float, default=1e-5)
    p.add_argument(
        "--exit-nonzero-on-fail",
        action="store_true",
        help="Exit with non-zero code if any matrix point fails (rows are still written).",
    )
    p.add_argument("--output", type=str, default="")
    args = p.parse_args(argv)

    matrix_iter: Iterable[Tuple[PartitionPolicy, bool, MemoryPlanMode, bool]]
    if args.matrix == "small":
        matrix_iter = [
            (PartitionPolicy.V2_BASELINE, False, MemoryPlanMode.DEFAULT, True),
        ]
    else:
        matrix_iter = iter_default_matrix()

    any_fail = False
    out_f = None
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        out_f = open(args.output, "w", encoding="utf-8")
    try:
        for partition_policy, reuse_on, memory_plan_mode, fusion_on in matrix_iter:
            row = _bench_one_safe(
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
                tvm_enabled=not bool(args.no_tvm),
                tol_atol=float(args.tol_atol),
                tol_rtol=float(args.tol_rtol),
            )
            if row.get("status") != "ok":
                any_fail = True
            line = json.dumps(row, ensure_ascii=False)
            if out_f is not None:
                out_f.write(line + "\n")
                out_f.flush()
            else:
                try:
                    sys.stdout.write(line + "\n")
                    sys.stdout.flush()
                except BrokenPipeError:
                    try:
                        sys.stdout.close()
                    except Exception:
                        pass
                    return 0
    finally:
        if out_f is not None:
            out_f.close()

    if any_fail and bool(args.exit_nonzero_on_fail):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
