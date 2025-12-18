#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

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
    before = {
        "phase": "before",
        "model": args.model,
        "min_tasks": int(args.min_tasks),
        "key_mode": args.key_mode,
        "policy": args.policy,
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
    after = {
        "phase": "after",
        "model": args.model,
        "min_tasks": int(args.min_tasks),
        "key_mode": args.key_mode,
        "policy": args.policy,
        "logical_buffers": int(module.storage_plan.num_logical_buffers()),
        "physical_buffers": int(module.storage_plan.num_physical_buffers()),
        "bytes_saved_est": int(after_saved),
        "pool_hit": int(stats.pool_hit),
        "pool_miss": int(stats.pool_miss),
        "tasks": int(len(liveness.topo_order)),
        "miss_reasons": {k.value: int(v) for k, v in (stats.miss_reasons or {}).items()},
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
        rows = [before, after]
        fieldnames = sorted({k for r in rows for k in r.keys() if k != "miss_reasons"})
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
