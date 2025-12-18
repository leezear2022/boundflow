#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from pathlib import Path
import subprocess

import torch
import torch.nn as nn

from boundflow.frontends.pytorch.frontend import import_torch
from boundflow.ir.liveness import compute_liveness_task_level
from boundflow.planner.interval_v2 import IntervalV2PartitionConfig, plan_interval_ibp_v2
from boundflow.planner.passes import apply_conservative_buffer_reuse
from boundflow.planner.storage_reuse import ReuseKeyMode, ReusePolicy, StorageReuseOptions, estimate_bytes_saved


def _mnist_cnn():
    auto_LiRPA = __import__("auto_LiRPA")
    Flatten = __import__("auto_LiRPA.utils", fromlist=["Flatten"]).Flatten
    _ = auto_LiRPA
    return nn.Sequential(
        nn.Conv2d(1, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32 * 7 * 7, 100),
        nn.ReLU(),
        nn.Linear(100, 10),
    )

def _git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(Path(__file__).resolve().parent.parent))
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"

def _env_vars_whitelist() -> dict[str, str]:
    import os

    keys = [
        "BOUNDFLOW_ROOT",
        "TVM_HOME",
        "TVM_FFI_DISABLE_TORCH_C_DLPACK",
        "TVM_FFI_CACHE_DIR",
        "TMPDIR",
        "PYTHONPATH",
    ]
    return {k: os.environ.get(k, "") for k in keys if k in os.environ}


def _versions() -> dict[str, str]:
    out = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "torch": getattr(torch, "__version__", "unknown"),
    }
    try:
        import tvm  # noqa: PLC0415

        out["tvm"] = getattr(tvm, "__version__", "unknown")
    except Exception:
        out["tvm"] = "unavailable"
    try:
        import tvm_ffi  # noqa: PLC0415

        out["tvm_ffi"] = getattr(tvm_ffi, "__version__", "unknown")
    except Exception:
        out["tvm_ffi"] = "unavailable"
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["mlp", "cnn"], default="mlp")
    p.add_argument("--min-tasks", type=int, default=2)
    p.add_argument("--key-mode", choices=["strict", "ignore_layout"], default="strict")
    p.add_argument("--policy", choices=["lifo", "fifo"], default="lifo")
    p.add_argument("--format", choices=["text", "json", "csv"], default="text")
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()

    torch.manual_seed(0)

    if args.model == "mlp":
        model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))
        x0 = torch.randn(4, 16)
    else:
        model = _mnist_cnn()
        x0 = torch.randn(2, 1, 28, 28)

    program = import_torch(model, (x0,), export_mode="export", normalize=True)
    module = plan_interval_ibp_v2(program, config=IntervalV2PartitionConfig(min_tasks=int(args.min_tasks)))
    module.validate()

    before_saved, _ = estimate_bytes_saved(module.storage_plan)
    commit = _git_commit()
    versions = _versions()
    env_vars = _env_vars_whitelist()
    num_edges = int(len(module.task_graph.edges)) if module.task_graph is not None else 0
    before = {
        "phase": "before",
        "model": args.model,
        "min_tasks": int(args.min_tasks),
        "key_mode": args.key_mode,
        "policy": args.policy,
        "respect_memory_effect": False,
        "git_commit": commit,
        "versions": versions,
        "env_vars": env_vars,
        "input_shape": list(x0.shape),
        "num_tasks": int(len(module.tasks)),
        "num_edges": num_edges,
        "logical_buffers": int(module.storage_plan.num_logical_buffers()),
        "physical_buffers": int(module.storage_plan.num_physical_buffers()),
        "bytes_saved_est": int(before_saved),
    }

    opt = StorageReuseOptions(
        enabled=True,
        key_mode=ReuseKeyMode(args.key_mode),
        policy=ReusePolicy(args.policy),
    )
    stats = apply_conservative_buffer_reuse(module, options=opt)
    liveness = compute_liveness_task_level(module)
    after_saved, _ = estimate_bytes_saved(module.storage_plan)
    miss_reasons = {k.value: int(v) for k, v in (stats.miss_reasons or {}).items()}
    why_topk = sorted(miss_reasons.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
    overlap_blockers = {k: int(v) for k, v in (getattr(stats, "overlap_blockers", {}) or {}).items()}
    overlap_blockers_topk = sorted(overlap_blockers.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
    key_hist: dict[str, int] = {}
    for lt in liveness.lifetimes.values():
        k = str(lt.key)
        key_hist[k] = int(key_hist.get(k, 0)) + 1
    reuse_key_topk = sorted(key_hist.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
    after = {
        "phase": "after",
        "model": args.model,
        "min_tasks": int(args.min_tasks),
        "key_mode": args.key_mode,
        "policy": args.policy,
        "respect_memory_effect": bool(opt.respect_memory_effect),
        "git_commit": commit,
        "versions": versions,
        "env_vars": env_vars,
        "input_shape": list(x0.shape),
        "num_tasks": int(len(module.tasks)),
        "num_edges": num_edges,
        "logical_buffers": int(module.storage_plan.num_logical_buffers()),
        "physical_buffers": int(module.storage_plan.num_physical_buffers()),
        "bytes_saved_est": int(after_saved),
        "pool_hit": int(stats.pool_hit),
        "pool_miss": int(stats.pool_miss),
        "tasks": int(len(liveness.topo_order)),
        "max_free_pool_keys": int(getattr(stats, "max_free_pool_keys", 0)),
        "max_free_pool_buffers": int(getattr(stats, "max_free_pool_buffers", 0)),
        "miss_reasons": miss_reasons,
        "why_not_reused_topk": why_topk,
        "reuse_key_topk": reuse_key_topk,
        "overlap_blockers_topk": overlap_blockers_topk,
    }

    fmt = args.format
    if fmt == "text":
        print(
            f"[before] logical={before['logical_buffers']} physical={before['physical_buffers']} bytes_saved_est={before['bytes_saved_est']}"
        )
        print(
            f"[after]  logical={after['logical_buffers']} physical={after['physical_buffers']} bytes_saved_est={after['bytes_saved_est']} "
            f"pool_hit={after['pool_hit']} pool_miss={after['pool_miss']} tasks={after['tasks']}"
        )
        if after["miss_reasons"]:
            print("miss_reasons:", after["miss_reasons"])
    elif fmt == "json":
        out = {"before": before, "after": after}
        s = json.dumps(out, ensure_ascii=False)
        if args.out:
            Path(args.out).write_text(s + "\n", encoding="utf-8")
        else:
            print(s)
    elif fmt == "csv":
        # Flatten versions/env vars into stable CSV columns.
        def _flatten_row(r: dict) -> dict:
            rr = dict(r)
            versions = rr.pop("versions", {}) or {}
            env_vars = rr.pop("env_vars", {}) or {}
            for k, v in versions.items():
                rr[f"ver_{k}"] = v
            for k, v in env_vars.items():
                rr[f"env_{k}"] = v
            return rr

        before_csv = _flatten_row(before)
        after_csv = _flatten_row(after)
        rows = [before_csv, after_csv]
        fieldnames = sorted(
            {
                k
                for r in rows
                for k in r.keys()
                if k
                not in (
                    "miss_reasons",
                    "why_not_reused_topk",
                    "reuse_key_topk",
                    "overlap_blockers_topk",
                )
            }
        )
        # Flatten miss_reasons for CSV.
        miss_keys = sorted({k for r in rows for k in (r.get("miss_reasons") or {}).keys()})
        fieldnames += [f"miss_{k}" for k in miss_keys]

        output_path = Path(args.out) if args.out else None
        f = output_path.open("w", newline="", encoding="utf-8") if output_path else None
        try:
            writer = csv.DictWriter(f or sys.stdout, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                row = {k: r.get(k, "") for k in fieldnames}
                miss = r.get("miss_reasons") or {}
                for k in miss_keys:
                    row[f"miss_{k}"] = miss.get(k, 0)
                writer.writerow(row)
        finally:
            if f is not None:
                f.close()
    else:
        raise ValueError(f"unsupported format: {fmt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
