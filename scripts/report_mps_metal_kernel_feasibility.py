from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.bench_phase7a_shared_crown_path_attribution import _device_meta, _device_name, _make_device


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(_ROOT),
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
        )
    except Exception:
        return "unknown"


def _parse_sizes(raw: str) -> List[int]:
    out = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not out:
        raise ValueError("empty --sizes")
    return out


def _percentile_ms(samples_s: Iterable[float], q: float = 0.5) -> float:
    xs = sorted(float(x) for x in samples_s)
    if not xs:
        return 0.0
    k = int(round((len(xs) - 1) * q))
    return xs[k] * 1000.0


def _time(fn: Callable[[], None], *, warmup: int, iters: int) -> float:
    for _ in range(int(warmup)):
        fn()
    torch.mps.synchronize()
    samples: List[float] = []
    for _ in range(int(iters)):
        t0 = time.perf_counter()
        fn()
        torch.mps.synchronize()
        samples.append(time.perf_counter() - t0)
    return _percentile_ms(samples)


def _compile_library():
    source = """
kernel void axpy(
    device const float* x,
    device float* out,
    constant float& a,
    constant float& b,
    uint idx [[thread_position_in_grid]]
) {
    out[idx] = a * x[idx] + b;
}

kernel void signed_weighted(
    device const float* x,
    device const float* pos_w,
    device const float* neg_w,
    device float* out,
    uint idx [[thread_position_in_grid]]
) {
    float v = x[idx];
    out[idx] = v >= 0.0f ? v * pos_w[idx] : v * neg_w[idx];
}
"""
    return torch.mps.compile_shader(source)


def _run_axpy(lib: Any, *, size: int, warmup: int, iters: int, device: torch.device) -> Dict[str, Any]:
    x = torch.randn(size, device=device, dtype=torch.float32)
    out = torch.empty_like(x)
    a = 1.75
    b = -0.25

    def torch_fn() -> None:
        y = x * a + b
        out.copy_(y)

    def metal_fn() -> None:
        lib.axpy(x, out, a, b)

    expected = x * a + b
    metal_fn()
    torch.mps.synchronize()
    max_abs_diff = float(torch.max(torch.abs(out - expected)).item())
    torch_ms = _time(torch_fn, warmup=warmup, iters=iters)
    metal_ms = _time(metal_fn, warmup=warmup, iters=iters)
    return {
        "kernel": "axpy",
        "size": int(size),
        "torch_ms_p50": float(torch_ms),
        "metal_ms_p50": float(metal_ms),
        "speedup": float("inf") if metal_ms == 0.0 else float(torch_ms / metal_ms),
        "max_abs_diff": max_abs_diff,
        "allclose": bool(max_abs_diff <= 1e-6),
    }


def _run_signed_weighted(lib: Any, *, size: int, warmup: int, iters: int, device: torch.device) -> Dict[str, Any]:
    x = torch.randn(size, device=device, dtype=torch.float32)
    pos_w = torch.randn(size, device=device, dtype=torch.float32)
    neg_w = torch.randn(size, device=device, dtype=torch.float32)
    out = torch.empty_like(x)

    def torch_fn() -> None:
        y = torch.where(x >= 0, x * pos_w, x * neg_w)
        out.copy_(y)

    def metal_fn() -> None:
        lib.signed_weighted(x, pos_w, neg_w, out)

    expected = torch.where(x >= 0, x * pos_w, x * neg_w)
    metal_fn()
    torch.mps.synchronize()
    max_abs_diff = float(torch.max(torch.abs(out - expected)).item())
    torch_ms = _time(torch_fn, warmup=warmup, iters=iters)
    metal_ms = _time(metal_fn, warmup=warmup, iters=iters)
    return {
        "kernel": "signed_weighted",
        "size": int(size),
        "torch_ms_p50": float(torch_ms),
        "metal_ms_p50": float(metal_ms),
        "speedup": float("inf") if metal_ms == 0.0 else float(torch_ms / metal_ms),
        "max_abs_diff": max_abs_diff,
        "allclose": bool(max_abs_diff <= 1e-6),
    }


def _geomean(values: Iterable[float]) -> Optional[float]:
    xs = [float(v) for v in values if float(v) > 0.0 and math.isfinite(float(v))]
    if not xs:
        return None
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Minimal custom Metal kernel feasibility gate for MPS.")
    parser.add_argument("--device", type=str, default="mps", choices=["mps"])
    parser.add_argument("--sizes", type=str, default="4096,65536,1048576")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args(argv)

    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"):
        raise RuntimeError("PYTORCH_ENABLE_MPS_FALLBACK must be disabled for Metal feasibility evidence")
    device = _make_device(str(args.device), dtype_name="float32", allow_mps_fallback=False)
    if not hasattr(torch.mps, "compile_shader"):
        raise RuntimeError("torch.mps.compile_shader is not available in this PyTorch build")

    lib = _compile_library()
    rows: List[Dict[str, Any]] = []
    for size in _parse_sizes(str(args.sizes)):
        rows.append(_run_axpy(lib, size=size, warmup=int(args.warmup), iters=int(args.iters), device=device))
        rows.append(_run_signed_weighted(lib, size=size, warmup=int(args.warmup), iters=int(args.iters), device=device))

    payload = {
        "meta": {
            "schema_version": "mps_metal_kernel_feasibility.v1",
            "script": "report_mps_metal_kernel_feasibility",
            "git_sha": _git_sha(),
            "torch_version": torch.__version__,
            "device": str(device),
            "device_name": _device_name(device),
            "device_meta": _device_meta(device, allow_mps_fallback=False),
            "sizes": _parse_sizes(str(args.sizes)),
            "warmup": int(args.warmup),
            "iters": int(args.iters),
        },
        "rows": rows,
        "summary": {
            "rows": len(rows),
            "allclose": all(bool(row["allclose"]) for row in rows),
            "geomean_speedup": _geomean(row["speedup"] for row in rows),
            "best_speedup": max((float(row["speedup"]) for row in rows), default=0.0),
            "worst_speedup": min((float(row["speedup"]) for row in rows), default=0.0),
        },
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if payload["summary"]["allclose"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
