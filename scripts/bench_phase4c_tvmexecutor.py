from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn

from boundflow.frontends.pytorch.frontend import import_torch
from boundflow.planner import plan_interval_ibp_v0
from boundflow.runtime.task_executor import LinfInputSpec, PythonTaskExecutor
from boundflow.runtime.tvm_executor import TVMExecutorOptions, TVMTaskExecutor


def _build_mlp() -> nn.Module:
    return nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))


def _build_mnist_cnn() -> nn.Module:
    from auto_LiRPA.utils import Flatten

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


def _time_it(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    return (t1 - t0) / float(iters)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mlp", "cnn"], default="mlp")
    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--target", type=str, default="llvm")
    args = parser.parse_args()

    torch.manual_seed(0)
    if args.model == "mlp":
        model = _build_mlp()
        x0 = torch.randn(4, 16)
    else:
        __import__("auto_LiRPA")  # make error message clearer
        model = _build_mnist_cnn()
        x0 = torch.randn(2, 1, 28, 28)

    program = import_torch(model, (x0,), export_mode="export", normalize=True)
    task_module = plan_interval_ibp_v0(program)
    input_spec = LinfInputSpec(value_name="input", center=x0, eps=args.eps)

    py_exe = PythonTaskExecutor()
    tvm_exe = TVMTaskExecutor(options=TVMExecutorOptions(target=args.target))

    def run_py():
        py_exe.run_ibp(task_module, input_spec)

    def run_tvm():
        tvm_exe.run_ibp(task_module, input_spec)

    py_s = _time_it(run_py, warmup=args.warmup, iters=args.iters)
    tvm_s = _time_it(run_tvm, warmup=args.warmup, iters=args.iters)

    print(f"model={args.model} eps={args.eps} iters={args.iters} warmup={args.warmup} target={args.target}")
    print(f"PythonTaskExecutor: {py_s * 1e3:.3f} ms/iter")
    print(f"TVMTaskExecutor:    {tvm_s * 1e3:.3f} ms/iter")
    if tvm_exe.last_stats is not None:
        print(f"tvm_ops={tvm_exe.last_stats.tvm_ops}")
        print(f"fallback_ops={tvm_exe.last_stats.fallback_ops}")
        print(f"linear_cache={tvm_exe.last_stats.linear_kernel_cache}")
        print(f"conv2d_cache={tvm_exe.last_stats.conv2d_kernel_cache}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

