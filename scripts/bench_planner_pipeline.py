#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

import torch
import torch.nn as nn

from boundflow.frontends.pytorch.frontend import import_torch
from boundflow.planner.core import PlannerConfig
from boundflow.planner.options import PartitionOptions, PartitionPolicy
from boundflow.planner.pipeline import plan
from boundflow.planner.storage_reuse import ReuseKeyMode, ReusePolicy, StorageReuseOptions


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["mlp"], default="mlp")
    p.add_argument("--min-tasks", type=int, default=2)
    p.add_argument("--policy", choices=["v0_single_task", "v2_baseline"], default="v2_baseline")
    p.add_argument("--enable-reuse", action="store_true")
    p.add_argument("--reuse-key-mode", choices=["strict", "ignore_layout"], default="strict")
    p.add_argument("--reuse-policy", choices=["lifo", "fifo"], default="lifo")
    args = p.parse_args()

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))
    x0 = torch.randn(4, 16)
    program = import_torch(model, (x0,), export_mode="export", normalize=True)

    cfg = PlannerConfig(
        enable_task_graph=(args.policy != "v0_single_task"),
        enable_storage_reuse=bool(args.enable_reuse),
        partition=PartitionOptions(
            policy=PartitionPolicy(args.policy),
            min_tasks=int(args.min_tasks),
        ),
        storage_reuse=StorageReuseOptions(
            enabled=bool(args.enable_reuse),
            key_mode=ReuseKeyMode(args.reuse_key_mode),
            policy=ReusePolicy(args.reuse_policy),
        ),
    )

    bundle = plan(program, config=cfg)
    out = {
        "num_tasks": len(bundle.task_module.tasks),
        "num_edges": len(bundle.task_module.task_graph.edges) if bundle.task_module.task_graph is not None else 0,
        "config_dump": bundle.meta.get("config_dump"),
        "planner_steps": bundle.meta.get("planner_steps"),
        "reuse_stats": {
            "pool_hit": int(getattr(bundle.meta.get("reuse_stats"), "pool_hit", 0)),
            "pool_miss": int(getattr(bundle.meta.get("reuse_stats"), "pool_miss", 0)),
        }
        if bundle.meta.get("reuse_stats") is not None
        else None,
    }
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

