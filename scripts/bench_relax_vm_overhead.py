#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from boundflow.frontends.pytorch.frontend import import_torch
from boundflow.planner.interval_v2 import IntervalV2PartitionConfig, plan_interval_ibp_v2
from boundflow.backends.tvm.relax_interval_task_ops import build_interval_task_relax_ops_ir_module


def _ms(ns: int) -> float:
    return ns / 1e6


def _timeit(fn, *, iters: int, warmup: int) -> dict:
    for _ in range(int(warmup)):
        fn()
    t0 = time.perf_counter_ns()
    for _ in range(int(iters)):
        fn()
    t1 = time.perf_counter_ns()
    return {"iters": int(iters), "warmup": int(warmup), "total_ms": _ms(t1 - t0), "avg_ms": _ms(t1 - t0) / iters}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--include-return", action="store_true")
    p.add_argument("--format", choices=["json"], default="json")
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 8), nn.ReLU())
    x0 = torch.randn(4, 16)
    program = import_torch(model, (x0,), export_mode="export", normalize=True)
    module = plan_interval_ibp_v2(program, config=IntervalV2PartitionConfig(min_tasks=1))
    task = module.get_entry_task()

    import tvm
    from tvm import relax
    from tvm.runtime import _tensor as rt

    ir_mod, spec = build_interval_task_relax_ops_ir_module(task, storage_plan=module.storage_plan, target="llvm", func_name="main")
    ex = relax.build(ir_mod, target="llvm")
    dev = tvm.cpu(0)
    vm = relax.VirtualMachine(ex, dev)

    # Build fixed inputs (interval lanes) + params.
    params = module.bindings.get("params", {}) or {}
    x_l = x0 - 0.1
    x_u = x0 + 0.1
    args_tvm = [
        rt.tensor(x_l.detach().cpu().numpy(), device=dev),
        rt.tensor(x_u.detach().cpu().numpy(), device=dev),
    ]
    for p_name in spec.param_values:
        t = params[p_name]
        if not torch.is_tensor(t):
            t = torch.as_tensor(t)
        args_tvm.append(rt.tensor(t.detach().cpu().numpy(), device=dev))

    fn_lookup_each_time = lambda: vm[spec.func_name](*args_tvm)
    fn_cached = vm[spec.func_name]
    fn_cached_call = lambda: fn_cached(*args_tvm)

    results = {
        "tvm_version": getattr(tvm, "__version__", "unknown"),
        "iters": int(args.iters),
        "warmup": int(args.warmup),
        "include_return": bool(args.include_return),
        "methods": {},
    }
    results["methods"]["vm_lookup_each_time"] = _timeit(fn_lookup_each_time, iters=args.iters, warmup=args.warmup)
    results["methods"]["vm_cached_packedfunc"] = _timeit(fn_cached_call, iters=args.iters, warmup=args.warmup)

    # save_function closure: bind current inputs once, then call a zero-arg function.
    saved_name = "main_saved"
    vm.save_function(spec.func_name, saved_name, *args_tvm, include_return=bool(args.include_return))
    saved_fn = vm[saved_name]
    results["methods"]["vm_save_function_closure"] = _timeit(lambda: saved_fn(), iters=args.iters, warmup=args.warmup)

    s = json.dumps(results, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(s + "\n", encoding="utf-8")
    else:
        print(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

