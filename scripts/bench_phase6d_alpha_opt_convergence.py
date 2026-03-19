from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Tuple

import torch

from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime.crown_ibp import run_crown_ibp_mlp
from boundflow.runtime.task_executor import InputSpec


AlphaObjective = Literal["lower", "upper", "gap", "both"]


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def _loss(bounds, objective: AlphaObjective) -> torch.Tensor:
    if objective == "lower":
        return -bounds.lower.mean()
    if objective == "upper":
        return bounds.upper.mean()
    if objective == "gap":
        return (bounds.upper - bounds.lower).mean()
    if objective == "both":
        return bounds.upper.mean() - bounds.lower.mean()
    raise AssertionError(f"unreachable objective: {objective}")


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


@dataclass(frozen=True)
class TraceRow:
    step: int
    lower_mean: float
    upper_mean: float
    alpha_min: float
    alpha_max: float
    alpha_mean: float


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 6D microbench: alpha optimization convergence (CROWN-IBP MLP).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--in-dim", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--out-dim", type=int, default=10)
    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--p", type=str, default="linf", choices=["linf", "l2", "l1"])
    parser.add_argument("--specs", type=int, default=32, help="S for linear_spec_C (0 to disable)")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--alpha-init", type=float, default=0.5)
    parser.add_argument("--objective", type=str, default="lower", choices=["lower", "upper", "gap", "both"])
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
    objective: AlphaObjective = args.objective

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

    S = int(args.specs)
    C = None if S <= 0 else torch.randn(batch, S, out_dim, device=device, dtype=dtype)

    # One ReLU node in this workload: alpha for "h1" (shape [H]).
    alpha = torch.full((hidden,), float(args.alpha_init), device=device, dtype=dtype, requires_grad=True)
    opt = torch.optim.Adam([alpha], lr=float(args.lr))

    trace: List[TraceRow] = []
    for step in range(int(args.steps) + 1):
        bounds = run_crown_ibp_mlp(module, spec, linear_spec_C=C, relu_alpha={"h1": alpha})
        trace.append(
            TraceRow(
                step=step,
                lower_mean=float(bounds.lower.mean().detach().cpu().item()),
                upper_mean=float(bounds.upper.mean().detach().cpu().item()),
                alpha_min=float(alpha.detach().min().cpu().item()),
                alpha_max=float(alpha.detach().max().cpu().item()),
                alpha_mean=float(alpha.detach().mean().cpu().item()),
            )
        )
        if step == int(args.steps):
            break

        loss = _loss(bounds, objective)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            alpha.clamp_(0.0, 1.0)

        if device.type == "cuda":
            torch.cuda.synchronize()

        if step % 5 == 0:
            _eprint(
                f"step={step:03d} loss={float(loss.detach().cpu().item()):.6f} "
                f"lb_mean={trace[-1].lower_mean:.6f} ub_mean={trace[-1].upper_mean:.6f} "
                f"alpha_mean={trace[-1].alpha_mean:.4f}"
            )

    meta: Dict[str, Any] = {
        "script": "bench_phase6d_alpha_opt_convergence",
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
        "specs": S,
        "objective": objective,
        "steps": int(args.steps),
        "lr": float(args.lr),
        "alpha_init": float(args.alpha_init),
    }
    print(json.dumps({"meta": meta, "trace": [t.__dict__ for t in trace]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

