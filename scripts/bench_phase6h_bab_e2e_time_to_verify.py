from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

import platform
import subprocess
import torch

from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime import alpha_beta_crown as ab
from boundflow.runtime import bab as bab_mod
from boundflow.runtime.task_executor import InputSpec


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def _percentile_ms(samples_s: Iterable[float], q: float) -> float:
    xs = sorted(samples_s)
    if not xs:
        return 0.0
    k = int(round((len(xs) - 1) * q))
    return float(xs[k]) * 1000.0


_TimerMode = Literal["perf_counter", "torch_benchmark"]

@dataclass(frozen=True)
class _TimeStats:
    ms_p50: float
    ms_p90: float
    ms_p99: float
    runs_count: int
    valid_runs_count: int
    timeouts_count: int


class _TimeoutCtx:
    def __init__(self, seconds: float) -> None:
        self._seconds = float(seconds)
        self._enabled = self._seconds > 0.0
        self._old_handler = None
        self._signal = None

    def __enter__(self) -> None:
        if not self._enabled:
            return
        import signal  # standard lib

        if not hasattr(signal, "SIGALRM"):
            raise RuntimeError("timeout not supported on this platform (no SIGALRM)")
        self._signal = signal
        self._old_handler = signal.getsignal(signal.SIGALRM)

        def _handler(_signum, _frame) -> None:  # pragma: no cover
            raise TimeoutError(f"timeout after {self._seconds}s")

        signal.signal(signal.SIGALRM, _handler)
        signal.setitimer(signal.ITIMER_REAL, self._seconds)

    def __exit__(self, exc_type, exc, tb) -> bool:
        if not self._enabled:
            return False
        assert self._signal is not None
        self._signal.setitimer(self._signal.ITIMER_REAL, 0.0)
        if self._old_handler is not None:
            self._signal.signal(self._signal.SIGALRM, self._old_handler)
        return False


def _time_call_perf_counter(
    fn,
    *,
    warmup: int,
    iters: int,
    sync_cuda: bool,
    timeout_s: float,
) -> Tuple[_TimeStats, Dict[str, Any]]:
    warmup_timeouts = 0
    for _ in range(int(warmup)):
        try:
            with _TimeoutCtx(float(timeout_s)):
                fn()
        except TimeoutError:
            warmup_timeouts += 1
    if sync_cuda:
        torch.cuda.synchronize()
    times: List[float] = []
    timeouts = 0
    for _ in range(int(iters)):
        t0 = time.perf_counter()
        try:
            with _TimeoutCtx(float(timeout_s)):
                fn()
            if sync_cuda:
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        except TimeoutError:
            timeouts += 1
    stats = _TimeStats(
        ms_p50=_percentile_ms(times, 0.5),
        ms_p90=_percentile_ms(times, 0.9),
        ms_p99=_percentile_ms(times, 0.99),
        runs_count=int(iters),
        valid_runs_count=len(times),
        timeouts_count=int(timeouts),
    )
    meta = {
        "n_warmup": int(warmup),
        "n_repeat": int(iters),
        "timeout_s": float(timeout_s),
        "warmup_timeouts": int(warmup_timeouts),
    }
    return stats, meta


def _time_call_torch_benchmark(
    fn,
    *,
    warmup: int,
    sync_cuda: bool,
    min_run_time_s: float,
    repeats: int,
) -> Tuple[_TimeStats, Dict[str, Any]]:
    import torch.utils.benchmark as benchmark

    def wrapped() -> None:
        fn()
        if sync_cuda:
            torch.cuda.synchronize()

    for _ in range(int(warmup)):
        wrapped()

    medians_s: List[float] = []
    number_per_run: List[int] = []
    for _ in range(max(1, int(repeats))):
        m = benchmark.Timer(stmt="wrapped()", globals={"wrapped": wrapped}).blocked_autorange(
            min_run_time=float(min_run_time_s)
        )
        medians_s.append(float(m.median))
        try:
            number_per_run.append(int(getattr(m, "number_per_run")))
        except Exception:
            pass

    stats = _TimeStats(
        ms_p50=_percentile_ms(medians_s, 0.5),
        ms_p90=_percentile_ms(medians_s, 0.9),
        ms_p99=_percentile_ms(medians_s, 0.99),
        runs_count=int(repeats),
        valid_runs_count=len(medians_s),
        timeouts_count=0,
    )
    meta = {
        "n_warmup": int(warmup),
        "torch_benchmark": True,
        "min_run_time_s": float(min_run_time_s),
        "repeats": int(repeats),
        "number_per_run": (number_per_run[0] if number_per_run else None),
    }
    return stats, meta


def _make_single_relu_mlp(*, w1: torch.Tensor, b1: torch.Tensor, w2: torch.Tensor, b2: torch.Tensor) -> BFTaskModule:
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
    return BFTaskModule(tasks=[task], entry_task_id="t0", bindings={"params": {"W1": w1, "b1": b1, "W2": w2, "b2": b2}})


def _make_1d_relu_identity(*, device: torch.device, dtype: torch.dtype) -> BFTaskModule:
    w1 = torch.tensor([[1.0]], dtype=dtype, device=device)
    b1 = torch.tensor([0.0], dtype=dtype, device=device)
    w2 = torch.tensor([[1.0]], dtype=dtype, device=device)
    b2 = torch.tensor([0.0], dtype=dtype, device=device)
    return _make_single_relu_mlp(w1=w1, b1=b1, w2=w2, b2=b2)


def _make_three_direction_first_layer_module(*, t: float = 0.49, device: torch.device, dtype: torch.dtype) -> BFTaskModule:
    a1 = torch.tensor([1.0, 0.0], dtype=dtype, device=device)
    a2 = torch.tensor([-0.5, 0.8660254], dtype=dtype, device=device)
    a3 = torch.tensor([-0.5, -0.8660254], dtype=dtype, device=device)
    w1 = torch.stack([a1, a2, a3], dim=0)
    b1 = torch.full((3,), -float(t), dtype=dtype, device=device)
    w2 = torch.eye(3, dtype=dtype, device=device)
    b2 = torch.zeros(3, dtype=dtype, device=device)
    return _make_single_relu_mlp(w1=w1, b1=b1, w2=w2, b2=b2)


def _make_chain_mlp(
    *,
    device: torch.device,
    dtype: torch.dtype,
    in_dim: int,
    hidden_dims: List[int],
    out_dim: int,
    seed: int,
) -> BFTaskModule:
    """
    Build a chain-structured MLP:
      input -> (Linear -> ReLU) * L -> Linear -> out
    with deterministic random weights controlled by `seed`.
    """
    if in_dim <= 0 or out_dim <= 0:
        raise ValueError("in_dim/out_dim must be positive")
    if not hidden_dims:
        raise ValueError("hidden_dims must be non-empty")
    if any(int(h) <= 0 for h in hidden_dims):
        raise ValueError(f"invalid hidden_dims: {hidden_dims}")

    g = torch.Generator(device=device)
    g.manual_seed(int(seed))

    ops: List[TaskOp] = []
    params: Dict[str, torch.Tensor] = {}

    cur_val = "input"
    cur_dim = int(in_dim)
    relu_idx = 0
    linear_idx = 0

    for layer_i, h in enumerate(hidden_dims, start=1):
        linear_idx += 1
        W_name = f"W{linear_idx}"
        b_name = f"b{linear_idx}"
        out_name = f"h{layer_i}"
        W = torch.randn(int(h), int(cur_dim), device=device, dtype=dtype, generator=g) * 0.5
        b = torch.randn(int(h), device=device, dtype=dtype, generator=g) * 0.1
        params[W_name] = W
        params[b_name] = b
        ops.append(TaskOp(op_type="linear", name=f"linear{linear_idx}", inputs=[cur_val, W_name, b_name], outputs=[out_name]))

        relu_idx += 1
        relu_out = f"r{layer_i}"
        ops.append(TaskOp(op_type="relu", name=f"relu{relu_idx}", inputs=[out_name], outputs=[relu_out]))

        cur_val = relu_out
        cur_dim = int(h)

    # output linear
    linear_idx += 1
    W_name = f"W{linear_idx}"
    b_name = f"b{linear_idx}"
    W = torch.randn(int(out_dim), int(cur_dim), device=device, dtype=dtype, generator=g) * 0.5
    b = torch.randn(int(out_dim), device=device, dtype=dtype, generator=g) * 0.1
    params[W_name] = W
    params[b_name] = b
    ops.append(TaskOp(op_type="linear", name=f"linear{linear_idx}", inputs=[cur_val, W_name, b_name], outputs=["out"]))

    task = BoundTask(
        task_id="t0",
        kind=TaskKind.INTERVAL_IBP,
        ops=ops,
        input_values=["input"],
        output_values=["out"],
    )
    return BFTaskModule(tasks=[task], entry_task_id="t0", bindings={"params": params})


def _parse_bool_list(s: str) -> List[bool]:
    out: List[bool] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if part in ("1", "true", "True", "yes", "y"):
            out.append(True)
        elif part in ("0", "false", "False", "no", "n"):
            out.append(False)
        else:
            raise ValueError(f"invalid boolean list element: {part}")
    if not out:
        raise ValueError("empty boolean list")
    return out


def _get_git_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.getcwd(), stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def _get_device_name(device: torch.device) -> str:
    if device.type == "cuda":
        try:
            return str(torch.cuda.get_device_name(0))
        except Exception:
            return "cuda:unknown"
    cpu = platform.processor() or platform.machine() or "cpu"
    return cpu


def _platform_meta() -> Dict[str, Any]:
    try:
        u = platform.uname()._asdict()
    except Exception:
        u = {"system": platform.system(), "release": platform.release(), "version": platform.version()}
    return {
        "python_version": sys.version.split()[0],
        "platform": u,
    }


@dataclass
class _Counters:
    oracle_calls: int = 0
    evaluated_nodes_count: int = 0
    forward_trace_calls_oracle: int = 0
    forward_trace_calls_branch: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    pruned_infeasible_count: int = 0


def _instrument_runtime(counters: _Counters):
    orig_ab_run = ab.run_alpha_beta_crown_mlp
    orig_bab_run = bab_mod.run_alpha_beta_crown_mlp
    orig_ab_forward = ab._forward_ibp_trace_mlp
    orig_bab_forward = bab_mod._forward_ibp_trace_mlp
    orig_cache_get = bab_mod.NodeEvalCache.get
    orig_prune = bab_mod.prune_infeasible_first_layer_items

    def _run(*args, **kwargs):
        spec = args[1]
        counters.oracle_calls += 1
        counters.evaluated_nodes_count += int(spec.center.shape[0])
        return orig_ab_run(*args, **kwargs)

    def _ab_forward(*args, **kwargs):
        counters.forward_trace_calls_oracle += 1
        return orig_ab_forward(*args, **kwargs)

    def _bab_forward(*args, **kwargs):
        counters.forward_trace_calls_branch += 1
        return orig_bab_forward(*args, **kwargs)

    def _cache_get(self, *, split_state):  # type: ignore[no-redef]
        v = orig_cache_get(self, split_state=split_state)
        if v is None:
            counters.cache_misses += 1
        else:
            counters.cache_hits += 1
        return v

    def _prune_wrapper(module, input_spec, *, items, cache, cfg):  # type: ignore[no-redef]
        kept, pruned = orig_prune(module, input_spec, items=items, cache=cache, cfg=cfg)
        counters.pruned_infeasible_count += len(pruned)
        return kept, pruned

    ab.run_alpha_beta_crown_mlp = _run  # type: ignore[assignment]
    bab_mod.run_alpha_beta_crown_mlp = _run  # type: ignore[assignment]
    ab._forward_ibp_trace_mlp = _ab_forward  # type: ignore[assignment]
    bab_mod._forward_ibp_trace_mlp = _bab_forward  # type: ignore[assignment]
    bab_mod.NodeEvalCache.get = _cache_get  # type: ignore[assignment]
    bab_mod.prune_infeasible_first_layer_items = _prune_wrapper  # type: ignore[assignment]

    def _restore() -> None:
        ab.run_alpha_beta_crown_mlp = orig_ab_run  # type: ignore[assignment]
        bab_mod.run_alpha_beta_crown_mlp = orig_bab_run  # type: ignore[assignment]
        ab._forward_ibp_trace_mlp = orig_ab_forward  # type: ignore[assignment]
        bab_mod._forward_ibp_trace_mlp = orig_bab_forward  # type: ignore[assignment]
        bab_mod.NodeEvalCache.get = orig_cache_get  # type: ignore[assignment]
        bab_mod.prune_infeasible_first_layer_items = orig_prune  # type: ignore[assignment]

    return _restore


@dataclass(frozen=True)
class Row:
    workload: str
    device: str
    dtype: str
    p: str
    eps: float
    specs: int
    node_batch_size: int
    max_nodes: int
    oracle: str
    steps: int
    lr: float
    alpha_init: float
    beta_init: float
    threshold: float
    enable_node_eval_cache: int
    use_branch_hint: int
    enable_batch_infeasible_prune: int
    comparable: int
    note_code: str
    note: str
    batch_ms_p50: float
    batch_ms_p90: float
    batch_ms_p99: float
    serial_ms_p50: float
    serial_ms_p90: float
    serial_ms_p99: float
    speedup: float
    speedup_p90: float
    speedup_p99: float
    batch_runs_count: int
    batch_valid_runs_count: int
    batch_timeouts_count: int
    serial_runs_count: int
    serial_valid_runs_count: int
    serial_timeouts_count: int
    batch_verdict: str
    serial_verdict: str
    batch_stats: Dict[str, Any]
    serial_stats: Dict[str, Any]
    counts_batch: Dict[str, Any]
    counts_serial: Dict[str, Any]


def _run_solve(
    *,
    module: BFTaskModule,
    spec: InputSpec,
    C: Optional[torch.Tensor],
    cfg: bab_mod.BabConfig,
    counters: _Counters,
) -> bab_mod.BabResult:
    restore = _instrument_runtime(counters)
    try:
        return bab_mod.solve_bab_mlp(module, spec, config=cfg, linear_spec_C=C)
    finally:
        restore()


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 6H PR-1: E2E BaB time-to-verify ablation bench (switch matrix + counters).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument(
        "--workload",
        type=str,
        default="1d_relu",
        choices=[
            "1d_relu",
            "3dir_l2",
            "mlp2d_2x16",
            "mlp3d_3x32",
        ],
    )
    parser.add_argument("--specs", type=int, default=16)
    parser.add_argument("--eps", type=float, default=1.0)
    parser.add_argument("--p", type=str, default="linf", choices=["linf", "l2", "l1"])
    parser.add_argument("--max-nodes", type=int, default=256)
    parser.add_argument("--node-batch-size", type=int, default=32)
    parser.add_argument("--oracle", type=str, default="alpha_beta", choices=["alpha", "alpha_beta"])
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--alpha-init", type=float, default=0.5)
    parser.add_argument("--beta-init", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--timeout-s", type=float, default=0.0, help="best-effort per-run timeout for perf_counter timer (0 disables)")
    parser.add_argument("--timer", type=str, default="perf_counter", choices=["perf_counter", "torch_benchmark"])
    parser.add_argument("--torch-benchmark-min-run-time-s", type=float, default=0.2)
    parser.add_argument("--torch-benchmark-repeats", type=int, default=5, help="repeat blocked_autorange N times to estimate p90/p99")
    parser.add_argument("--enable-node-eval-cache", type=str, default="0,1", help="comma list of {0,1}")
    parser.add_argument("--use-branch-hint", type=str, default="0,1", help="comma list of {0,1}")
    parser.add_argument("--enable-batch-infeasible-prune", type=str, default="0,1", help="comma list of {0,1}")
    parser.add_argument("--jsonl-out", type=str, default="", help="optional JSONL output path (append one line per run)")
    args = parser.parse_args(argv)

    os.environ.setdefault("PYTHONHASHSEED", "0")
    if args.device == "cpu":
        warnings.filterwarnings(
            "ignore",
            message=r"CUDA initialization: Unexpected error from cudaGetDeviceCount\(\).*",
        )

    torch.manual_seed(int(args.seed))
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    sync_cuda = device.type == "cuda"
    git_sha = _get_git_sha()
    device_name = _get_device_name(device)
    plat = _platform_meta()

    specs = int(args.specs)
    eps = float(args.eps)
    p_norm = str(args.p)
    if args.workload == "1d_relu":
        module = _make_1d_relu_identity(device=device, dtype=dtype)
        x0 = torch.zeros(1, 1, device=device, dtype=dtype)
        out_dim = 1
    else:
        if args.workload == "3dir_l2":
            module = _make_three_direction_first_layer_module(device=device, dtype=dtype)
            x0 = torch.zeros(1, 2, device=device, dtype=dtype)
            out_dim = 3
        elif args.workload == "mlp2d_2x16":
            module = _make_chain_mlp(device=device, dtype=dtype, in_dim=2, hidden_dims=[16, 16], out_dim=4, seed=int(args.seed) + 101)
            x0 = torch.zeros(1, 2, device=device, dtype=dtype)
            out_dim = 4
        elif args.workload == "mlp3d_3x32":
            module = _make_chain_mlp(device=device, dtype=dtype, in_dim=3, hidden_dims=[32, 32, 32], out_dim=6, seed=int(args.seed) + 202)
            x0 = torch.zeros(1, 3, device=device, dtype=dtype)
            out_dim = 6
        else:  # pragma: no cover
            raise AssertionError(f"unhandled workload: {args.workload}")

    if p_norm == "linf":
        spec = InputSpec.linf(value_name="input", center=x0, eps=eps)
    elif p_norm == "l2":
        spec = InputSpec.l2(value_name="input", center=x0, eps=eps)
    else:
        spec = InputSpec.l1(value_name="input", center=x0, eps=eps)

    C = None
    # Provide a multi-spec C for realism. Use deterministic randomness (seed already set).
    C = torch.randn(1, specs, out_dim, device=device, dtype=dtype)

    cache_modes = _parse_bool_list(args.enable_node_eval_cache)
    hint_modes = _parse_bool_list(args.use_branch_hint)
    prune_modes = _parse_bool_list(args.enable_batch_infeasible_prune)

    timer_mode: _TimerMode = args.timer

    _eprint("NOTE: serial_ms_* includes Python loop + priority-queue overhead.")

    rows: List[Row] = []
    per_run_stats: List[Dict[str, Any]] = []

    for enable_cache, use_hint, enable_prune in product(cache_modes, hint_modes, prune_modes):
        cfg_batch = bab_mod.BabConfig(
            max_nodes=int(args.max_nodes),
            oracle=str(args.oracle),  # type: ignore[arg-type]
            node_batch_size=int(args.node_batch_size),
            enable_node_eval_cache=bool(enable_cache),
            use_branch_hint=bool(use_hint),
            enable_batch_infeasible_prune=bool(enable_prune),
            alpha_steps=int(args.steps),
            alpha_lr=float(args.lr),
            alpha_init=float(args.alpha_init),
            beta_init=float(args.beta_init),
            objective="lower",
            spec_reduce="mean",
            threshold=float(args.threshold),
            tol=float(args.tol),
        )
        cfg_serial = bab_mod.BabConfig(
            max_nodes=int(args.max_nodes),
            oracle=str(args.oracle),  # type: ignore[arg-type]
            node_batch_size=1,
            enable_node_eval_cache=bool(enable_cache),
            use_branch_hint=bool(use_hint),
            enable_batch_infeasible_prune=bool(enable_prune),
            alpha_steps=int(args.steps),
            alpha_lr=float(args.lr),
            alpha_init=float(args.alpha_init),
            beta_init=float(args.beta_init),
            objective="lower",
            spec_reduce="mean",
            threshold=float(args.threshold),
            tol=float(args.tol),
        )

        def _batch_call() -> None:
            c = _Counters()
            _ = _run_solve(module=module, spec=spec, C=C, cfg=cfg_batch, counters=c)

        def _serial_call() -> None:
            c = _Counters()
            _ = _run_solve(module=module, spec=spec, C=C, cfg=cfg_serial, counters=c)

        if timer_mode == "torch_benchmark":
            batch_time, batch_timer = _time_call_torch_benchmark(
                _batch_call,
                warmup=int(args.warmup),
                sync_cuda=sync_cuda,
                min_run_time_s=float(args.torch_benchmark_min_run_time_s),
                repeats=int(args.torch_benchmark_repeats),
            )
            serial_time, serial_timer = _time_call_torch_benchmark(
                _serial_call,
                warmup=int(args.warmup),
                sync_cuda=sync_cuda,
                min_run_time_s=float(args.torch_benchmark_min_run_time_s),
                repeats=int(args.torch_benchmark_repeats),
            )
        else:
            batch_time, batch_timer = _time_call_perf_counter(
                _batch_call,
                warmup=int(args.warmup),
                iters=int(args.iters),
                sync_cuda=sync_cuda,
                timeout_s=float(args.timeout_s),
            )
            serial_time, serial_timer = _time_call_perf_counter(
                _serial_call,
                warmup=int(args.warmup),
                iters=int(args.iters),
                sync_cuda=sync_cuda,
                timeout_s=float(args.timeout_s),
            )

        # Representative run for stats/counters (avoid noise across repeats).
        counters_b = _Counters()
        res_b = _run_solve(module=module, spec=spec, C=C, cfg=cfg_batch, counters=counters_b)
        counters_s = _Counters()
        res_s = _run_solve(module=module, spec=spec, C=C, cfg=cfg_serial, counters=counters_s)

        def _counts(c: _Counters) -> Dict[str, Any]:
            total = int(c.cache_hits) + int(c.cache_misses)
            return {
                "oracle_calls": int(c.oracle_calls),
                "forward_trace_calls": int(c.forward_trace_calls_oracle + c.forward_trace_calls_branch),
                "forward_trace_calls_oracle": int(c.forward_trace_calls_oracle),
                "forward_trace_calls_branch": int(c.forward_trace_calls_branch),
                "cache_hits": int(c.cache_hits),
                "cache_misses": int(c.cache_misses),
                "cache_hit_rate": 0.0 if total == 0 else float(c.cache_hits) / float(total),
                "pruned_infeasible_count": int(c.pruned_infeasible_count),
                "evaluated_nodes_count": int(c.evaluated_nodes_count),
            }

        batch_stats = {
            "verdict": str(res_b.status),
            "popped_nodes_total": int(res_b.nodes_visited),
            "popped_nodes": int(res_b.nodes_visited),
            "evaluated_nodes": int(res_b.nodes_evaluated),
            "expanded_nodes": int(res_b.nodes_expanded),
            "queue_peak": int(res_b.max_queue),
            "max_queue_size": int(res_b.max_queue),
            "avg_batch_fill_rate": float(res_b.avg_batch_fill_rate),
            "batch_rounds": int(res_b.batch_rounds),
            "best_lower": float(res_b.best_lower),
            "best_upper": float(res_b.best_upper),
        }
        serial_stats = {
            "verdict": str(res_s.status),
            "popped_nodes_total": int(res_s.nodes_visited),
            "popped_nodes": int(res_s.nodes_visited),
            "evaluated_nodes": int(res_s.nodes_evaluated),
            "expanded_nodes": int(res_s.nodes_expanded),
            "queue_peak": int(res_s.max_queue),
            "max_queue_size": int(res_s.max_queue),
            "avg_batch_fill_rate": float(res_s.avg_batch_fill_rate),
            "batch_rounds": int(res_s.batch_rounds),
            "best_lower": float(res_s.best_lower),
            "best_upper": float(res_s.best_upper),
        }

        verdict_equal = str(res_b.status) == str(res_s.status)
        timing_ok = (
            int(batch_time.valid_runs_count) > 0
            and int(serial_time.valid_runs_count) > 0
            and int(batch_time.timeouts_count) == 0
            and int(serial_time.timeouts_count) == 0
        )
        comparable = verdict_equal and timing_ok
        speedup = 0.0 if batch_time.ms_p50 <= 0.0 else float(serial_time.ms_p50) / float(batch_time.ms_p50)
        speedup_p90 = 0.0 if batch_time.ms_p90 <= 0.0 else float(serial_time.ms_p90) / float(batch_time.ms_p90)
        speedup_p99 = 0.0 if batch_time.ms_p99 <= 0.0 else float(serial_time.ms_p99) / float(batch_time.ms_p99)
        note_code = ""
        note = ""
        if not verdict_equal:
            note_code = "verdict_mismatch"
            note = "verdict_mismatch (search order / budget effects); speedup is reference-only"
        elif not timing_ok:
            note_code = "timing_invalid"
            note = "timing_invalid (timeouts or no valid samples); speedup is reference-only"
        rows.append(
            Row(
                workload=f"phase6h_bab_e2e_{args.workload}",
                device=str(device),
                dtype=str(dtype).replace("torch.", ""),
                p=str(p_norm),
                eps=float(eps),
                specs=int(specs),
                node_batch_size=int(args.node_batch_size),
                max_nodes=int(args.max_nodes),
                oracle=str(args.oracle),
                steps=int(args.steps),
                lr=float(args.lr),
                alpha_init=float(args.alpha_init),
                beta_init=float(args.beta_init),
                threshold=float(args.threshold),
                enable_node_eval_cache=int(bool(enable_cache)),
                use_branch_hint=int(bool(use_hint)),
                enable_batch_infeasible_prune=int(bool(enable_prune)),
                comparable=int(bool(comparable)),
                note_code=note_code,
                note=note,
                batch_ms_p50=float(batch_time.ms_p50),
                batch_ms_p90=float(batch_time.ms_p90),
                batch_ms_p99=float(batch_time.ms_p99),
                serial_ms_p50=float(serial_time.ms_p50),
                serial_ms_p90=float(serial_time.ms_p90),
                serial_ms_p99=float(serial_time.ms_p99),
                speedup=float(speedup),
                speedup_p90=float(speedup_p90),
                speedup_p99=float(speedup_p99),
                batch_runs_count=int(batch_time.runs_count),
                batch_valid_runs_count=int(batch_time.valid_runs_count),
                batch_timeouts_count=int(batch_time.timeouts_count),
                serial_runs_count=int(serial_time.runs_count),
                serial_valid_runs_count=int(serial_time.valid_runs_count),
                serial_timeouts_count=int(serial_time.timeouts_count),
                batch_verdict=str(res_b.status),
                serial_verdict=str(res_s.status),
                batch_stats=batch_stats,
                serial_stats=serial_stats,
                counts_batch=_counts(counters_b),
                counts_serial=_counts(counters_s),
            )
        )
        per_run_stats.append(
            {
                "enable_node_eval_cache": int(bool(enable_cache)),
                "use_branch_hint": int(bool(use_hint)),
                "enable_batch_infeasible_prune": int(bool(enable_prune)),
                "timer": {"batch": batch_timer, "serial": serial_timer},
                "time": {"batch": batch_time.__dict__, "serial": serial_time.__dict__},
            }
        )

    out: Dict[str, Any] = {
        "rows": [r.__dict__ for r in rows],
        "meta": {
            "schema_version": "phase6h_e2e_v2",
            "seed": int(args.seed),
            "git_sha": git_sha,
            "device": str(device),
            "device_name": device_name,
            "dtype": str(dtype).replace("torch.", ""),
            "workload": str(args.workload),
            "p": str(p_norm),
            "eps": float(eps),
            "specs": int(specs),
            "max_nodes": int(args.max_nodes),
            "node_batch_size": int(args.node_batch_size),
            "oracle": str(args.oracle),
            "steps": int(args.steps),
            "lr": float(args.lr),
            "spec_reduce": "mean",
            "alpha_init": float(args.alpha_init),
            "beta_init": float(args.beta_init),
            "threshold": float(args.threshold),
            "tol": float(args.tol),
            "timer": str(timer_mode),
            "warmup": int(args.warmup),
            "iters": int(args.iters),
            "timeout_s": float(args.timeout_s),
            "torch_version": str(torch.__version__),
            "torch_num_threads": int(torch.get_num_threads()),
            **plat,
            "torch_benchmark_min_run_time_s": float(args.torch_benchmark_min_run_time_s),
            "torch_benchmark_repeats": int(args.torch_benchmark_repeats),
            "per_run_timer_stats": per_run_stats,
        },
    }
    s = json.dumps(out)
    print(s)
    if args.jsonl_out:
        with open(str(args.jsonl_out), "a", encoding="utf-8") as f:
            f.write(s + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
