#!/usr/bin/env python
from __future__ import annotations

import argparse

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
    print(f"[before] logical={module.storage_plan.num_logical_buffers()} physical={module.storage_plan.num_physical_buffers()} bytes_saved_est={before_saved}")

    opt = StorageReuseOptions(
        enabled=True,
        key_mode=ReuseKeyMode(args.key_mode),
        policy=ReusePolicy(args.policy),
    )
    stats = apply_conservative_buffer_reuse(module, options=opt)
    liveness = compute_liveness_task_level(module)
    after_saved, _ = estimate_bytes_saved(module.storage_plan)
    print(
        f"[after]  logical={module.storage_plan.num_logical_buffers()} physical={module.storage_plan.num_physical_buffers()} "
        f"bytes_saved_est={after_saved} pool_hit={stats.pool_hit} pool_miss={stats.pool_miss} tasks={len(liveness.topo_order)}"
    )
    if stats.miss_reasons:
        print("miss_reasons:", {k.value: v for k, v in stats.miss_reasons.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
