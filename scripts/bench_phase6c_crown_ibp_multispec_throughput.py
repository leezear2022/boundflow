from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Literal, Tuple

import torch

from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime.crown_ibp import run_crown_ibp_mlp
from boundflow.runtime.task_executor import InputSpec


@dataclass(frozen=True)
class Row:
    workload: str
    device: str
    dtype: str
    batch: int
    in_dim: int
    hidden: int
    out_dim: int
    eps: float
    p: str
    specs: int
    batch_ms_p50: float
    serial_ms_p50: float
    speedup: float


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def _make_mlp_module(*, w1: torch.Tensor, b1: torch.Tensor, w2: torch.Tensor, b2: torch.Tensor) -> BFTaskModule:
    task = BoundTask(
        task_id="t0",
        kind=TaskKind.INTERVAL_IBP,
        ops=[
            TaskOp(op_type="linear", name="linear1", inputs=["input", "W1", "b1"], outputs=["h1"]),
            TaskOp(op_type="relu", name="relu1", inputs=["h1"], outputs=["r1"]),
            TaskOp(op_type="linear", name="linear2", inputs=["r1", "W2", "b2"], outputs=["out"]),
        ],
        input_values=["input"],
        output_values=["out"],
    )
    return BFTaskModule(
        tasks=[task],
        entry_task_id="t0",
        bindings={"params": {"W1": w1, "b1": b1, "W2": w2, "b2": b2}},
    )


def _parse_specs_list(s: str) -> List[int]:
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise ValueError("empty --specs-list")
    return out


def _percentile_ms(samples_s: Iterable[float], q: float) -> float:
    xs = sorted(samples_s)
    if not xs:
        return 0.0
    k = int(round((len(xs) - 1) * q))
    return float(xs[k]) * 1000.0


_TimerMode = Literal["perf_counter", "torch_benchmark"]


def _time_call_perf_counter(fn, *, warmup: int, iters: int, sync_cuda: bool) -> Tuple[float, Dict[str, Any]]:
    with torch.inference_mode():
        for _ in range(int(warmup)):
            fn()
        if sync_cuda:
            torch.cuda.synchronize()
        times: List[float] = []
        for _ in range(int(iters)):
            t0 = time.perf_counter()
            fn()
            if sync_cuda:
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    return _percentile_ms(times, 0.5), {"n_warmup": int(warmup), "n_repeat": int(iters)}


def _time_call_torch_benchmark(
    fn, *, warmup: int, sync_cuda: bool, min_run_time_s: float
) -> Tuple[float, Dict[str, Any]]:
    import torch.utils.benchmark as benchmark

    def wrapped() -> None:
        fn()
        if sync_cuda:
            torch.cuda.synchronize()

    with torch.inference_mode():
        for _ in range(int(warmup)):
            wrapped()
        m = benchmark.Timer(stmt="wrapped()", globals={"wrapped": wrapped}).blocked_autorange(min_run_time=float(min_run_time_s))
    return float(m.median) * 1000.0, {
        "n_warmup": int(warmup),
        "torch_benchmark": True,
        "min_run_time_s": float(min_run_time_s),
        "n_repeat": int(getattr(m, "number_per_run", 0)),
        "median_s": float(m.median),
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 6C microbench: multi-spec batch vs serial for CROWN-IBP MLP.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--in-dim", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--out-dim", type=int, default=10)
    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--p", type=str, default="linf", choices=["linf", "l2", "l1"])
    parser.add_argument("--specs-list", type=str, default="1,4,16,64", help="comma-separated S values")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--serial-warmup", type=int, default=None, help="defaults to --warmup")
    parser.add_argument("--serial-iters", type=int, default=None, help="defaults to --iters")
    parser.add_argument(
        "--timer",
        type=str,
        default="perf_counter",
        choices=["perf_counter", "torch_benchmark"],
        help="timing backend (torch_benchmark uses blocked_autorange)",
    )
    parser.add_argument("--torch-benchmark-min-run-time-s", type=float, default=0.2)
    args = parser.parse_args(argv)

    os.environ.setdefault("PYTHONHASHSEED", "0")

    torch.manual_seed(int(args.seed))
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    batch = int(args.batch)
    in_dim = int(args.in_dim)
    hidden = int(args.hidden)
    out_dim = int(args.out_dim)
    eps = float(args.eps)
    warmup = int(args.warmup)
    iters = int(args.iters)
    serial_warmup = warmup if args.serial_warmup is None else int(args.serial_warmup)
    serial_iters = iters if args.serial_iters is None else int(args.serial_iters)
    timer_mode: _TimerMode = args.timer

    _eprint("NOTE: serial_ms_* includes Python loop + slicing overhead (models a real per-spec serial usage).")

    x0 = torch.randn(batch, in_dim, device=device, dtype=dtype)
    w1 = torch.randn(hidden, in_dim, device=device, dtype=dtype)
    b1 = torch.randn(hidden, device=device, dtype=dtype)
    w2 = torch.randn(out_dim, hidden, device=device, dtype=dtype)
    b2 = torch.randn(out_dim, device=device, dtype=dtype)
    module = _make_mlp_module(w1=w1, b1=b1, w2=w2, b2=b2)

    if args.p == "linf":
        spec = InputSpec.linf(value_name="input", center=x0, eps=eps)
        p_norm = "linf"
    elif args.p == "l2":
        spec = InputSpec.l2(value_name="input", center=x0, eps=eps)
        p_norm = "l2"
    else:
        spec = InputSpec.l1(value_name="input", center=x0, eps=eps)
        p_norm = "l1"

    rows: List[Row] = []
    per_run_stats: Dict[int, Dict[str, Any]] = {}
    for s in _parse_specs_list(args.specs_list):
        C = torch.randn(batch, int(s), out_dim, device=device, dtype=dtype)

        def _batch() -> None:
            run_crown_ibp_mlp(module, spec, linear_spec_C=C)

        def _serial() -> None:
            for i in range(int(s)):
                run_crown_ibp_mlp(module, spec, linear_spec_C=C[:, i : i + 1, :])

        sync_cuda = device.type == "cuda"
        if timer_mode == "torch_benchmark":
            batch_ms, batch_stats = _time_call_torch_benchmark(
                _batch,
                warmup=warmup,
                sync_cuda=sync_cuda,
                min_run_time_s=float(args.torch_benchmark_min_run_time_s),
            )
            serial_ms, serial_stats = _time_call_torch_benchmark(
                _serial,
                warmup=serial_warmup,
                sync_cuda=sync_cuda,
                min_run_time_s=float(args.torch_benchmark_min_run_time_s),
            )
        else:
            batch_ms, batch_stats = _time_call_perf_counter(_batch, warmup=warmup, iters=iters, sync_cuda=sync_cuda)
            serial_ms, serial_stats = _time_call_perf_counter(
                _serial,
                warmup=serial_warmup,
                iters=serial_iters,
                sync_cuda=sync_cuda,
            )
        per_run_stats[int(s)] = {"batch": batch_stats, "serial": serial_stats}
        speedup = float("inf") if batch_ms == 0.0 else serial_ms / batch_ms
        rows.append(
            Row(
                workload="mlp",
                device=str(device),
                dtype=str(dtype).replace("torch.", ""),
                batch=batch,
                in_dim=in_dim,
                hidden=hidden,
                out_dim=out_dim,
                eps=eps,
                p=p_norm,
                specs=int(s),
                batch_ms_p50=float(batch_ms),
                serial_ms_p50=float(serial_ms),
                speedup=float(speedup),
            )
        )
        _eprint(f"S={s}: batch_p50_ms={batch_ms:.3f} serial_p50_ms={serial_ms:.3f} speedup={speedup:.2f}x")

    meta: Dict[str, Any] = {
        "script": "bench_phase6c_crown_ibp_multispec_throughput",
        "torch_version": torch.__version__,
        "seed": int(args.seed),
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "batch": batch,
        "in_dim": in_dim,
        "hidden": hidden,
        "out_dim": out_dim,
        "eps": eps,
        "p": p_norm,
        "specs_list": _parse_specs_list(args.specs_list),
        "timer": timer_mode,
        "warmup": warmup,
        "iters": iters,
        "serial_warmup": serial_warmup,
        "serial_iters": serial_iters,
        "serial_includes_python_loop_overhead": True,
        "per_specs_stats": per_run_stats,
    }
    print(json.dumps({"meta": meta, "rows": [asdict(r) for r in rows]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
