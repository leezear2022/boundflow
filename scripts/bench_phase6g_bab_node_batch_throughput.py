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

import torch

from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime import alpha_beta_crown as ab
from boundflow.runtime import bab as bab_mod
from boundflow.runtime.task_executor import InputSpec


@dataclass(frozen=True)
class Row:
    workload: str
    device: str
    dtype: str
    node_batch_size: int
    nodes: int
    in_dim: int
    hidden: int
    out_dim: int
    specs: int
    eps: float
    p: str
    steps: int
    enable_node_eval_cache: int
    use_branch_hint: int
    enable_batch_infeasible_prune: int
    batch_ms_p50: float
    serial_ms_p50: float
    speedup: float
    counts_batch: Dict[str, Any]
    counts_serial: Dict[str, Any]


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


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


def _make_three_direction_first_layer_module(
    *,
    t: float = 0.49,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> BFTaskModule:
    # input_dim=2, hidden=3. First layer constraints are:
    #   h_i = a_i · x - t
    # with directions 0/120/240 deg. All active splits are infeasible on unit L2 ball (convex-combo certificate).
    a1 = torch.tensor([1.0, 0.0], dtype=dtype, device=device)
    a2 = torch.tensor([-0.5, 0.8660254], dtype=dtype, device=device)
    a3 = torch.tensor([-0.5, -0.8660254], dtype=dtype, device=device)
    w1 = torch.stack([a1, a2, a3], dim=0)
    b1 = torch.full((3,), -float(t), dtype=dtype, device=device)
    w2 = torch.eye(3, dtype=dtype, device=device)
    b2 = torch.zeros(3, dtype=dtype, device=device)
    return _make_single_relu_mlp(w1=w1, b1=b1, w2=w2, b2=b2)


def _percentile_ms(samples_s: Iterable[float], q: float) -> float:
    xs = sorted(samples_s)
    if not xs:
        return 0.0
    k = int(round((len(xs) - 1) * q))
    return float(xs[k]) * 1000.0


_TimerMode = Literal["perf_counter", "torch_benchmark"]


def _time_call_perf_counter(fn, *, warmup: int, iters: int, sync_cuda: bool) -> Tuple[float, Dict[str, Any]]:
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


def _time_call_torch_benchmark(fn, *, warmup: int, sync_cuda: bool, min_run_time_s: float) -> Tuple[float, Dict[str, Any]]:
    import torch.utils.benchmark as benchmark

    def wrapped() -> None:
        fn()
        if sync_cuda:
            torch.cuda.synchronize()

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


@dataclass
class _Counters:
    oracle_calls: int = 0
    oracle_evaluated_nodes: int = 0
    oracle_forward_trace_calls: int = 0
    branch_forward_trace_calls: int = 0
    branch_pick_calls: int = 0
    pruned_infeasible_count: int = 0


def _instrument_runtime(counters: _Counters):
    orig_ab_run = ab.run_alpha_beta_crown_mlp
    orig_bab_run = bab_mod.run_alpha_beta_crown_mlp
    orig_ab_forward = ab._forward_ibp_trace_mlp
    orig_bab_forward = bab_mod._forward_ibp_trace_mlp

    def _run(*args, **kwargs):
        spec = args[1]
        counters.oracle_calls += 1
        counters.oracle_evaluated_nodes += int(spec.center.shape[0])
        return orig_ab_run(*args, **kwargs)

    def _ab_forward(*args, **kwargs):
        counters.oracle_forward_trace_calls += 1
        return orig_ab_forward(*args, **kwargs)

    def _bab_forward(*args, **kwargs):
        counters.branch_forward_trace_calls += 1
        return orig_bab_forward(*args, **kwargs)

    ab.run_alpha_beta_crown_mlp = _run  # type: ignore[assignment]
    bab_mod.run_alpha_beta_crown_mlp = _run  # type: ignore[assignment]
    ab._forward_ibp_trace_mlp = _ab_forward  # type: ignore[assignment]
    bab_mod._forward_ibp_trace_mlp = _bab_forward  # type: ignore[assignment]

    def _restore() -> None:
        ab.run_alpha_beta_crown_mlp = orig_ab_run  # type: ignore[assignment]
        bab_mod.run_alpha_beta_crown_mlp = orig_bab_run  # type: ignore[assignment]
        ab._forward_ibp_trace_mlp = orig_ab_forward  # type: ignore[assignment]
        bab_mod._forward_ibp_trace_mlp = orig_bab_forward  # type: ignore[assignment]

    return _restore


def _build_split_states(
    *,
    nodes: int,
    device: torch.device,
    rng: torch.Generator,
    dup_frac: float,
    infeasible_frac: float,
) -> List[bab_mod.ReluSplitState]:
    infeasible_n = int(round(float(infeasible_frac) * int(nodes)))
    infeasible_n = max(0, min(int(nodes), infeasible_n))
    splits: List[bab_mod.ReluSplitState] = []
    infeasible_state = bab_mod.ReluSplitState(by_relu_input={"h1": torch.tensor([1, 1, 1], device=device, dtype=torch.int8)})
    feasible_base = bab_mod.ReluSplitState(by_relu_input={"h1": torch.tensor([0, 0, 0], device=device, dtype=torch.int8)})

    for i in range(int(nodes)):
        if i < infeasible_n:
            splits.append(infeasible_state)
            continue
        # optionally repeat a previous state to exercise cache
        if splits and torch.rand((), generator=rng, device=device).item() < float(dup_frac):
            j = int(torch.randint(low=0, high=len(splits), size=(), generator=rng, device=device).item())
            splits.append(splits[j])
            continue
        # random split on 3 neurons: {-1,0,+1}
        s = torch.randint(low=0, high=3, size=(3,), generator=rng, device=device, dtype=torch.int64).to(torch.int8) - 1
        splits.append(bab_mod.ReluSplitState(by_relu_input={"h1": s}))
    return splits


def _run_bounding_workload(
    *,
    module: BFTaskModule,
    base_spec: InputSpec,
    C1: torch.Tensor,
    split_states: List[bab_mod.ReluSplitState],
    node_batch_size: int,
    steps: int,
    lr: float,
    alpha_init: float,
    beta_init: float,
    enable_node_eval_cache: bool,
    use_branch_hint: bool,
    enable_batch_infeasible_prune: bool,
    serial: bool,
) -> Tuple[_Counters, Dict[str, Any]]:
    cfg = bab_mod.BabConfig(
        oracle="alpha_beta",
        node_batch_size=int(node_batch_size),
        enable_node_eval_cache=bool(enable_node_eval_cache),
        enable_batch_infeasible_prune=bool(enable_batch_infeasible_prune),
        alpha_steps=int(steps),
        alpha_lr=float(lr),
        alpha_init=float(alpha_init),
        beta_init=float(beta_init),
        objective="lower",
        spec_reduce="mean",
    )
    cache: Optional[bab_mod.NodeEvalCache] = None
    if enable_node_eval_cache:
        cache = bab_mod.NodeEvalCache(module=module, input_spec=base_spec, linear_spec_C=C1, cfg=cfg)
    cache_by_example = None if cache is None else {0: cache}

    counters = _Counters()
    restore = _instrument_runtime(counters)
    try:
        if serial:
            for st in split_states:
                items = [(0, bab_mod._QueueItem(priority=0.0, node_id=0, example_idx=0, split_state=st))]
                kept, pruned = bab_mod.prune_infeasible_first_layer_items(
                    module, base_spec, items=items, cache_by_example=cache_by_example, cfg=cfg
                )
                counters.pruned_infeasible_count += len(pruned)
                if not kept:
                    continue
                _, it = kept[0]
                b, a, be, stats, branch_hint, _hit = bab_mod.eval_bab_alpha_beta_node(
                    module,
                    base_spec,
                    linear_spec_C=C1,
                    split_state=it.split_state,
                    warm_start_alpha=None,
                    warm_start_beta=None,
                    cfg=cfg,
                    cache=cache,
                )
                _ = (b, a, be, stats)
                counters.branch_pick_calls += 1
                if use_branch_hint and branch_hint is not None:
                    continue
                _ = bab_mod._pick_branch(module, base_spec, split_state=it.split_state)
            stats_out: Dict[str, Any] = {}
            stats_out["cache_hits"] = 0
            stats_out["cache_misses"] = 0
            stats_out["cache_hit_rate"] = 0.0
            if cache is not None:
                stats_out["cache_hits"] = int(cache.hits)
                stats_out["cache_misses"] = int(cache.misses)
                total = int(cache.hits) + int(cache.misses)
                stats_out["cache_hit_rate"] = 0.0 if total == 0 else float(cache.hits) / float(total)
            return counters, stats_out

        # batched path
        K = int(node_batch_size)
        for off in range(0, len(split_states), K):
            chunk = split_states[off : off + K]
            items = [
                (i, bab_mod._QueueItem(priority=0.0, node_id=i, example_idx=0, split_state=st))
                for i, st in enumerate(chunk)
            ]

            # prune infeasible (and write infeasible nodes to cache)
            kept, pruned = bab_mod.prune_infeasible_first_layer_items(
                module, base_spec, items=items, cache_by_example=cache_by_example, cfg=cfg
            )
            counters.pruned_infeasible_count += len(pruned)
            if not kept:
                continue

            # filter cache hits
            to_eval: List[Tuple[int, bab_mod._QueueItem]] = []
            branch_hints: Dict[int, Optional[Tuple[str, int]]] = {}
            for orig_i, it in kept:
                if cache is not None:
                    v = cache.get(split_state=it.split_state)
                    if v is not None:
                        branch_hints[orig_i] = v.branch
                        continue
                to_eval.append((orig_i, it))

            if to_eval:
                center_b = base_spec.center.expand(len(to_eval), -1).clone()
                batch_spec = type(base_spec)(
                    value_name=base_spec.value_name, center=center_b, perturbation=base_spec.perturbation
                )
                C_b = C1.expand(len(to_eval), -1, -1).clone()
                split_b = torch.stack([it.split_state.by_relu_input["h1"] for _i, it in to_eval], dim=0)
                bounds_b, alpha_b, beta_b, stats = ab.run_alpha_beta_crown_mlp(
                    module,
                    batch_spec,
                    linear_spec_C=C_b,
                    relu_split_state={"h1": split_b},
                    steps=int(steps),
                    lr=float(lr),
                    alpha_init=float(alpha_init),
                    beta_init=float(beta_init),
                    objective="lower",
                    per_batch_params=True,
                )
                for j, (orig_i, it) in enumerate(to_eval):
                    if cache is not None:
                        cache.put(
                            split_state=it.split_state,
                            value=bab_mod.NodeEvalCacheValue(
                                bounds=bab_mod.IntervalState(
                                    lower=bounds_b.lower[j : j + 1].detach().clone(),
                                    upper=bounds_b.upper[j : j + 1].detach().clone(),
                                ),
                                best_alpha=bab_mod.AlphaState(
                                    {k: v[j].detach().clone() for k, v in alpha_b.alpha_by_relu_input.items()}
                                ),
                                best_beta=ab.BetaState(
                                    {k: v[j].detach().clone() for k, v in beta_b.beta_by_relu_input.items()}
                                ),
                                stats=stats,
                                branch=(stats.branch_choices[j] if stats.branch_choices else None),
                            ),
                        )
                    branch_hints[orig_i] = stats.branch_choices[j] if stats.branch_choices else None

            for orig_i, it in kept:
                counters.branch_pick_calls += 1
                if use_branch_hint and branch_hints.get(orig_i) is not None:
                    continue
                _ = bab_mod._pick_branch(module, base_spec, split_state=it.split_state)

        stats_out2: Dict[str, Any] = {}
        stats_out2["cache_hits"] = 0
        stats_out2["cache_misses"] = 0
        stats_out2["cache_hit_rate"] = 0.0
        if cache is not None:
            stats_out2["cache_hits"] = int(cache.hits)
            stats_out2["cache_misses"] = int(cache.misses)
            total = int(cache.hits) + int(cache.misses)
            stats_out2["cache_hit_rate"] = 0.0 if total == 0 else float(cache.hits) / float(total)
        return counters, stats_out2
    finally:
        restore()


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 6G microbench: BaB node-batch switch matrix + counters.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--nodes", type=int, default=32)
    parser.add_argument("--node-batch-size", type=int, default=32, help="K: node-batch size for batched eval")
    parser.add_argument("--specs", type=int, default=16)
    parser.add_argument("--eps", type=float, default=1.0)
    parser.add_argument("--p", type=str, default="l2", choices=["linf", "l2", "l1"])
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--alpha-init", type=float, default=0.5)
    parser.add_argument("--beta-init", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--timer", type=str, default="perf_counter", choices=["perf_counter", "torch_benchmark"])
    parser.add_argument("--torch-benchmark-min-run-time-s", type=float, default=0.2)
    parser.add_argument("--dup-frac", type=float, default=0.3, help="fraction of nodes that repeat an earlier split state")
    parser.add_argument("--infeasible-frac", type=float, default=0.2, help="fraction of nodes that are a known infeasible split")
    parser.add_argument("--enable-node-eval-cache", type=str, default="0,1", help="comma list of {0,1}")
    parser.add_argument("--use-branch-hint", type=str, default="0,1", help="comma list of {0,1}")
    parser.add_argument("--enable-batch-infeasible-prune", type=str, default="0,1", help="comma list of {0,1}")
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
    nodes = int(args.nodes)
    node_batch_size = int(args.node_batch_size)
    specs = int(args.specs)
    eps = float(args.eps)
    steps = int(args.steps)
    lr = float(args.lr)
    warmup = int(args.warmup)
    iters = int(args.iters)
    timer_mode: _TimerMode = args.timer

    _eprint("NOTE: serial_ms_* includes Python loop + branch-pick overhead.")

    if args.p == "linf":
        x0 = torch.zeros(1, 2, device=device, dtype=dtype)
        base_spec = InputSpec.linf(value_name="input", center=x0, eps=eps)
        p_norm = "linf"
    elif args.p == "l2":
        x0 = torch.zeros(1, 2, device=device, dtype=dtype)
        base_spec = InputSpec.l2(value_name="input", center=x0, eps=eps)
        p_norm = "l2"
    else:
        x0 = torch.zeros(1, 2, device=device, dtype=dtype)
        base_spec = InputSpec.l1(value_name="input", center=x0, eps=eps)
        p_norm = "l1"

    sync_cuda = device.type == "cuda"

    module = _make_three_direction_first_layer_module(device=device, dtype=dtype)
    hidden = 3
    in_dim = 2
    out_dim = 3
    C1 = torch.randn(1, specs, out_dim, device=device, dtype=dtype)

    rng = torch.Generator(device=device)
    rng.manual_seed(int(args.seed) + 123)
    split_states = _build_split_states(
        nodes=nodes,
        device=device,
        rng=rng,
        dup_frac=float(args.dup_frac),
        infeasible_frac=float(args.infeasible_frac),
    )

    cache_modes = _parse_bool_list(args.enable_node_eval_cache)
    hint_modes = _parse_bool_list(args.use_branch_hint)
    prune_modes = _parse_bool_list(args.enable_batch_infeasible_prune)

    rows: List[Row] = []
    per_run_stats: List[Dict[str, Any]] = []

    for enable_cache, use_hint, enable_prune in product(cache_modes, hint_modes, prune_modes):
        def _batch() -> None:
            _run_bounding_workload(
                module=module,
                base_spec=base_spec,
                C1=C1,
                split_states=split_states,
                node_batch_size=node_batch_size,
                steps=steps,
                lr=lr,
                alpha_init=float(args.alpha_init),
                beta_init=float(args.beta_init),
                enable_node_eval_cache=bool(enable_cache),
                use_branch_hint=bool(use_hint),
                enable_batch_infeasible_prune=bool(enable_prune),
                serial=False,
            )

        def _serial() -> None:
            _run_bounding_workload(
                module=module,
                base_spec=base_spec,
                C1=C1,
                split_states=split_states,
                node_batch_size=1,
                steps=steps,
                lr=lr,
                alpha_init=float(args.alpha_init),
                beta_init=float(args.beta_init),
                enable_node_eval_cache=bool(enable_cache),
                use_branch_hint=bool(use_hint),
                enable_batch_infeasible_prune=bool(enable_prune),
                serial=True,
            )

        if timer_mode == "torch_benchmark":
            batch_ms, batch_stats = _time_call_torch_benchmark(
                _batch,
                warmup=warmup,
                sync_cuda=sync_cuda,
                min_run_time_s=float(args.torch_benchmark_min_run_time_s),
            )
            serial_ms, serial_stats = _time_call_torch_benchmark(
                _serial,
                warmup=warmup,
                sync_cuda=sync_cuda,
                min_run_time_s=float(args.torch_benchmark_min_run_time_s),
            )
        else:
            batch_ms, batch_stats = _time_call_perf_counter(_batch, warmup=warmup, iters=iters, sync_cuda=sync_cuda)
            serial_ms, serial_stats = _time_call_perf_counter(_serial, warmup=warmup, iters=iters, sync_cuda=sync_cuda)

        # Capture counters from a single representative run (counts should be stable across repeats).
        c_batch, stats_batch = _run_bounding_workload(
            module=module,
            base_spec=base_spec,
            C1=C1,
            split_states=split_states,
            node_batch_size=node_batch_size,
            steps=steps,
            lr=lr,
            alpha_init=float(args.alpha_init),
            beta_init=float(args.beta_init),
            enable_node_eval_cache=bool(enable_cache),
            use_branch_hint=bool(use_hint),
            enable_batch_infeasible_prune=bool(enable_prune),
            serial=False,
        )
        c_serial, stats_serial = _run_bounding_workload(
            module=module,
            base_spec=base_spec,
            C1=C1,
            split_states=split_states,
            node_batch_size=1,
            steps=steps,
            lr=lr,
            alpha_init=float(args.alpha_init),
            beta_init=float(args.beta_init),
            enable_node_eval_cache=bool(enable_cache),
            use_branch_hint=bool(use_hint),
            enable_batch_infeasible_prune=bool(enable_prune),
            serial=True,
        )

        counts_batch = {
            "oracle_calls": c_batch.oracle_calls,
            "evaluated_nodes_count": c_batch.oracle_evaluated_nodes,
            "forward_trace_calls": c_batch.oracle_forward_trace_calls + c_batch.branch_forward_trace_calls,
            "forward_trace_calls_oracle": c_batch.oracle_forward_trace_calls,
            "forward_trace_calls_branch": c_batch.branch_forward_trace_calls,
            "branch_pick_calls": c_batch.branch_pick_calls,
            "pruned_infeasible_count": c_batch.pruned_infeasible_count,
            **stats_batch,
        }
        counts_serial = {
            "oracle_calls": c_serial.oracle_calls,
            "evaluated_nodes_count": c_serial.oracle_evaluated_nodes,
            "forward_trace_calls": c_serial.oracle_forward_trace_calls + c_serial.branch_forward_trace_calls,
            "forward_trace_calls_oracle": c_serial.oracle_forward_trace_calls,
            "forward_trace_calls_branch": c_serial.branch_forward_trace_calls,
            "branch_pick_calls": c_serial.branch_pick_calls,
            "pruned_infeasible_count": c_serial.pruned_infeasible_count,
            **stats_serial,
        }

        speedup = float("inf") if batch_ms == 0.0 else serial_ms / batch_ms
        rows.append(
            Row(
                workload="phase6g_bab_node_batch_switch_matrix",
                device=str(device),
                dtype=str(dtype).replace("torch.", ""),
                node_batch_size=int(node_batch_size),
                nodes=nodes,
                in_dim=in_dim,
                hidden=hidden,
                out_dim=out_dim,
                specs=specs,
                eps=eps,
                p=p_norm,
                steps=steps,
                enable_node_eval_cache=int(bool(enable_cache)),
                use_branch_hint=int(bool(use_hint)),
                enable_batch_infeasible_prune=int(bool(enable_prune)),
                batch_ms_p50=float(batch_ms),
                serial_ms_p50=float(serial_ms),
                speedup=float(speedup),
                counts_batch=counts_batch,
                counts_serial=counts_serial,
            )
        )
        per_run_stats.append(
            {
                "enable_node_eval_cache": int(bool(enable_cache)),
                "use_branch_hint": int(bool(use_hint)),
                "enable_batch_infeasible_prune": int(bool(enable_prune)),
                "timer": {"batch": batch_stats, "serial": serial_stats},
            }
        )

    out: Dict[str, Any] = {
        "rows": [r.__dict__ for r in rows],
        "meta": {
            "schema_version": "phase6g_node_eval_v1",
            "seed": int(args.seed),
            "device": str(device),
            "dtype": str(dtype).replace("torch.", ""),
            "nodes": int(nodes),
            "node_batch_size": int(node_batch_size),
            "specs": int(specs),
            "eps": float(eps),
            "p": str(p_norm),
            "steps": int(steps),
            "lr": float(lr),
            "dup_frac": float(args.dup_frac),
            "infeasible_frac": float(args.infeasible_frac),
            "timer": str(timer_mode),
            "warmup": int(warmup),
            "iters": int(iters),
            "torch_version": str(torch.__version__),
            "per_run_timer_stats": per_run_stats,
        },
    }
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
