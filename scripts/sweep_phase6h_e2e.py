from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def _csv_list(s: str) -> List[str]:
    return [p.strip() for p in s.split(",") if p.strip()]


def _int_list(s: str) -> List[int]:
    return [int(p.strip()) for p in s.split(",") if p.strip()]


def _float_list(s: str) -> List[float]:
    return [float(p.strip()) for p in s.split(",") if p.strip()]


@dataclass(frozen=True)
class SweepConfig:
    devices: Sequence[str]
    dtypes: Sequence[str]
    workloads: Sequence[str]
    ps: Sequence[str]
    specs_list: Sequence[int]
    max_nodes_list: Sequence[int]
    node_batch_sizes: Sequence[int]
    oracles: Sequence[str]
    steps_list: Sequence[int]
    lrs: Sequence[float]
    eps_list: Sequence[float]
    timers: Sequence[str]


def _run_one(
    *,
    python: str,
    script_path: str,
    out_jsonl: Path,
    device: str,
    dtype: str,
    workload: str,
    p: str,
    specs: int,
    eps: float,
    max_nodes: int,
    node_batch_size: int,
    oracle: str,
    steps: int,
    lr: float,
    timer: str,
    warmup: int,
    iters: int,
    torch_benchmark_min_run_time_s: float,
    seed: int,
    enable_node_eval_cache: str,
    use_branch_hint: str,
    enable_batch_infeasible_prune: str,
) -> None:
    cmd = [
        python,
        script_path,
        "--device",
        device,
        "--dtype",
        dtype,
        "--workload",
        workload,
        "--p",
        p,
        "--specs",
        str(specs),
        "--eps",
        str(eps),
        "--max-nodes",
        str(max_nodes),
        "--node-batch-size",
        str(node_batch_size),
        "--oracle",
        oracle,
        "--steps",
        str(steps),
        "--lr",
        str(lr),
        "--timer",
        timer,
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
        "--torch-benchmark-min-run-time-s",
        str(torch_benchmark_min_run_time_s),
        "--seed",
        str(seed),
        "--enable-node-eval-cache",
        enable_node_eval_cache,
        "--use-branch-hint",
        use_branch_hint,
        "--enable-batch-infeasible-prune",
        enable_batch_infeasible_prune,
    ]
    env = dict(os.environ)
    env.setdefault("PYTHONHASHSEED", "0")
    proc = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"bench_failed rc={proc.returncode}", proc.returncode, proc.stdout, proc.stderr)
    line = proc.stdout.strip()
    if not line:
        raise RuntimeError("bench_empty_stdout", proc.returncode, proc.stdout, proc.stderr)
    obj = json.loads(line)
    if "rows" not in obj or "meta" not in obj:
        raise RuntimeError("bench_invalid_schema", proc.returncode, proc.stdout, proc.stderr)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _write_failure_record(
    *,
    out_jsonl: Path,
    combo: Dict[str, Any],
    err: BaseException,
    stderr: str,
    stdout: str,
    returncode: int | None,
) -> None:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    tail = stderr[-4000:] if stderr else ""
    obj = {
        "rows": [],
        "meta": {
            "run_status": "error",
            "error": repr(err),
            "returncode": returncode,
            "stderr_tail": tail,
            "stdout_tail": (stdout[-2000:] if stdout else ""),
            **combo,
        },
    }
    with out_jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def _iter_product(cfg: SweepConfig) -> Iterable[dict]:
    for (
        device,
        dtype,
        workload,
        p,
        specs,
        eps,
        max_nodes,
        node_batch_size,
        oracle,
        steps,
        lr,
        timer,
    ) in itertools.product(
        cfg.devices,
        cfg.dtypes,
        cfg.workloads,
        cfg.ps,
        cfg.specs_list,
        cfg.eps_list,
        cfg.max_nodes_list,
        cfg.node_batch_sizes,
        cfg.oracles,
        cfg.steps_list,
        cfg.lrs,
        cfg.timers,
    ):
        yield {
            "device": device,
            "dtype": dtype,
            "workload": workload,
            "p": p,
            "specs": specs,
            "eps": eps,
            "max_nodes": max_nodes,
            "node_batch_size": node_batch_size,
            "oracle": oracle,
            "steps": steps,
            "lr": lr,
            "timer": timer,
        }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 6H PR-2: sweep E2E BaB bench and append JSONL.")
    parser.add_argument("--out-jsonl", type=str, required=True)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--bench-script", type=str, default="scripts/bench_phase6h_bab_e2e_time_to_verify.py")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-runs", type=int, default=0, help="0 means no limit")

    parser.add_argument("--devices", type=str, default="cpu")
    parser.add_argument("--dtypes", type=str, default="float32")
    parser.add_argument("--workloads", type=str, default="1d_relu")
    parser.add_argument("--ps", type=str, default="linf")
    parser.add_argument("--specs-list", type=str, default="16")
    parser.add_argument("--eps-list", type=str, default="1.0")
    parser.add_argument("--max-nodes-list", type=str, default="256")
    parser.add_argument("--node-batch-sizes", type=str, default="32")
    parser.add_argument("--oracles", type=str, default="alpha_beta")
    parser.add_argument("--steps-list", type=str, default="0")
    parser.add_argument("--lrs", type=str, default="0.2")
    parser.add_argument("--timers", type=str, default="perf_counter")

    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--torch-benchmark-min-run-time-s", type=float, default=0.2)

    # Keep bench schema stable; these are passed verbatim to PR-1 bench.
    parser.add_argument("--enable-node-eval-cache", type=str, default="0,1")
    parser.add_argument("--use-branch-hint", type=str, default="0,1")
    parser.add_argument("--enable-batch-infeasible-prune", type=str, default="0,1")

    parser.add_argument("--fail-fast", action="store_true", help="stop sweep on first failure")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    cfg = SweepConfig(
        devices=_csv_list(args.devices),
        dtypes=_csv_list(args.dtypes),
        workloads=_csv_list(args.workloads),
        ps=_csv_list(args.ps),
        specs_list=_int_list(args.specs_list),
        eps_list=_float_list(args.eps_list),
        max_nodes_list=_int_list(args.max_nodes_list),
        node_batch_sizes=_int_list(args.node_batch_sizes),
        oracles=_csv_list(args.oracles),
        steps_list=_int_list(args.steps_list),
        lrs=_float_list(args.lrs),
        timers=_csv_list(args.timers),
    )

    out_jsonl = Path(args.out_jsonl)
    bench_script = str(Path(args.bench_script))
    runs = 0
    failures = 0
    for combo in _iter_product(cfg):
        runs += 1
        if int(args.max_runs) > 0 and runs > int(args.max_runs):
            break
        print(f"[{runs}] {combo}", file=sys.stderr)
        if args.dry_run:
            continue
        try:
            _run_one(
                python=str(args.python),
                script_path=bench_script,
                out_jsonl=out_jsonl,
                seed=int(args.seed),
                warmup=int(args.warmup),
                iters=int(args.iters),
                torch_benchmark_min_run_time_s=float(args.torch_benchmark_min_run_time_s),
                enable_node_eval_cache=str(args.enable_node_eval_cache),
                use_branch_hint=str(args.use_branch_hint),
                enable_batch_infeasible_prune=str(args.enable_batch_infeasible_prune),
                **combo,
            )
        except BaseException as e:
            failures += 1
            rc: int | None = None
            stdout = ""
            stderr = ""
            if isinstance(e, RuntimeError) and len(e.args) >= 4 and isinstance(e.args[1], int):
                rc = int(e.args[1])
                stdout = str(e.args[2] or "")
                stderr = str(e.args[3] or "")
            _write_failure_record(out_jsonl=out_jsonl, combo=combo, err=e, stderr=stderr, stdout=stdout, returncode=rc)
            print(f"[{runs}] ERROR: {e!r}", file=sys.stderr)
            if args.fail_fast:
                raise

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
