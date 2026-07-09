from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


_CASES: Dict[str, Dict[str, Optional[str]]] = {
    "default": {
        "PYTORCH_MPS_PREFER_METAL": None,
        "PYTORCH_MPS_FAST_MATH": None,
    },
    "prefer_metal": {
        "PYTORCH_MPS_PREFER_METAL": "1",
        "PYTORCH_MPS_FAST_MATH": None,
    },
    "fast_math": {
        "PYTORCH_MPS_PREFER_METAL": None,
        "PYTORCH_MPS_FAST_MATH": "1",
    },
    "both": {
        "PYTORCH_MPS_PREFER_METAL": "1",
        "PYTORCH_MPS_FAST_MATH": "1",
    },
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(_repo_root()),
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
        )
    except Exception:
        return "unknown"


def _parse_cases(raw: str) -> List[str]:
    if raw.strip() == "all":
        return list(_CASES)
    out: List[str] = []
    for part in raw.split(","):
        case = part.strip()
        if not case:
            continue
        if case not in _CASES:
            raise ValueError(f"unknown env case: {case}")
        out.append(case)
    if not out:
        raise ValueError("empty --cases")
    return out


def _tail(text: str, *, lines: int = 20) -> str:
    return "\n".join(text.splitlines()[-lines:])


def _json_from_stdout(stdout: str) -> Dict[str, Any]:
    raw = stdout.strip()
    if not raw:
        raise ValueError("child benchmark produced empty stdout")
    return json.loads(raw.splitlines()[-1])


def _case_env(
    *,
    base_env: Dict[str, str],
    case: str,
    allow_mps_fallback: bool,
    set_kmp_duplicate_lib_ok: bool,
    omp_num_threads: Optional[str],
) -> Tuple[Dict[str, str], Dict[str, Optional[str]]]:
    env = dict(base_env)
    applied: Dict[str, Optional[str]] = {}
    for key, value in _CASES[case].items():
        applied[key] = value
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value

    if not allow_mps_fallback:
        applied["PYTORCH_ENABLE_MPS_FALLBACK"] = None
        env.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)

    if set_kmp_duplicate_lib_ok:
        applied["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    if omp_num_threads is not None:
        applied["OMP_NUM_THREADS"] = str(omp_num_threads)
        env["OMP_NUM_THREADS"] = str(omp_num_threads)

    root = str(_repo_root())
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = root if not existing_pythonpath else root + os.pathsep + existing_pythonpath
    applied["PYTHONPATH_PREFIX"] = root
    return env, applied


def _matrix_command(args: argparse.Namespace) -> List[str]:
    root = _repo_root()
    cmd = [
        str(args.python),
        str(root / "scripts" / "bench_phase7b_crossover_matrix.py"),
        "--device",
        "mps",
        "--dtype",
        "float32",
        "--workloads",
        str(args.workloads),
        "--scales",
        str(args.scales),
        "--policies",
        str(args.policies),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--seed",
        str(args.seed),
        "--timer",
        str(args.timer),
        "--torch-benchmark-min-run-time-s",
        str(args.torch_benchmark_min_run_time_s),
    ]
    if bool(args.allow_mps_fallback):
        cmd.append("--allow-mps-fallback")
    return cmd


def _compact_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "workload": row["workload"],
        "scale_id": row["scale_id"],
        "policy_request": row["policy_request"],
        "compare_target": row["compare_target"],
        "planner_decision": row["planner_decision"],
        "metrics": row["metrics"],
    }


def _run_case(args: argparse.Namespace, case: str) -> Dict[str, Any]:
    env, applied_env = _case_env(
        base_env=os.environ,
        case=case,
        allow_mps_fallback=bool(args.allow_mps_fallback),
        set_kmp_duplicate_lib_ok=bool(args.set_kmp_duplicate_lib_ok),
        omp_num_threads=args.omp_num_threads,
    )
    cmd = _matrix_command(args)
    if bool(args.dry_run):
        return {
            "case": case,
            "status": "dry_run",
            "env": applied_env,
            "command": cmd,
        }

    completed = subprocess.run(
        cmd,
        cwd=str(_repo_root()),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return {
            "case": case,
            "status": "fail",
            "returncode": int(completed.returncode),
            "env": applied_env,
            "command": cmd,
            "stdout_tail": _tail(completed.stdout),
            "stderr_tail": _tail(completed.stderr),
        }

    try:
        matrix_payload = _json_from_stdout(completed.stdout)
    except Exception as exc:
        return {
            "case": case,
            "status": "fail",
            "returncode": int(completed.returncode),
            "env": applied_env,
            "command": cmd,
            "parse_error": f"{type(exc).__name__}: {exc}",
            "stdout_tail": _tail(completed.stdout),
            "stderr_tail": _tail(completed.stderr),
        }

    return {
        "case": case,
        "status": "ok",
        "returncode": int(completed.returncode),
        "env": applied_env,
        "command": cmd,
        "matrix_meta": matrix_payload["meta"],
        "matrix_summary": matrix_payload["summary"],
        "rows": [_compact_row(row) for row in matrix_payload["rows"]],
        "stderr_tail": _tail(completed.stderr, lines=8),
    }


def _geomean(values: Iterable[float]) -> Optional[float]:
    clean = [float(v) for v in values if float(v) > 0.0 and math.isfinite(float(v))]
    if not clean:
        return None
    return math.exp(sum(math.log(v) for v in clean) / len(clean))


def _case_summary(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    default = next((result for result in results if result["case"] == "default" and result["status"] == "ok"), None)
    default_rows = {}
    if default is not None:
        default_rows = {
            (row["workload"], row["scale_id"], row["policy_request"]): row
            for row in default.get("rows", [])
        }

    summaries: List[Dict[str, Any]] = []
    for result in results:
        summary: Dict[str, Any] = {
            "case": result["case"],
            "status": result["status"],
        }
        if result["status"] != "ok":
            summaries.append(summary)
            continue

        structured_gains: List[float] = []
        baseline_gains: List[float] = []
        structured_vs_baseline: List[float] = []
        rows_compared = 0
        for row in result.get("rows", []):
            metrics = row["metrics"]
            structured_vs_baseline.append(float(metrics["speedup"]))
            key = (row["workload"], row["scale_id"], row["policy_request"])
            old = default_rows.get(key)
            if old is None:
                continue
            old_metrics = old["metrics"]
            new_structured = float(metrics["structured_ms_p50"])
            new_baseline = float(metrics["baseline_ms_p50"])
            if new_structured > 0.0:
                structured_gains.append(float(old_metrics["structured_ms_p50"]) / new_structured)
            if new_baseline > 0.0:
                baseline_gains.append(float(old_metrics["baseline_ms_p50"]) / new_baseline)
            rows_compared += 1

        summary.update(
            {
                "rows": len(result.get("rows", [])),
                "rows_compared_to_default": rows_compared,
                "geomean_structured_abs_gain_vs_default": _geomean(structured_gains),
                "geomean_baseline_abs_gain_vs_default": _geomean(baseline_gains),
                "geomean_structured_vs_dense_speedup": _geomean(structured_vs_baseline),
            }
        )
        summaries.append(summary)
    return summaries


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Sweep PyTorch MPS env vars over the Phase 7B crossover matrix.")
    parser.add_argument("--cases", type=str, default="all")
    parser.add_argument("--workloads", type=str, default="all")
    parser.add_argument("--scales", type=str, default="smoke")
    parser.add_argument("--policies", type=str, default="structured,dense_barrier,auto")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timer", type=str, default="perf_counter", choices=["perf_counter", "torch_benchmark"])
    parser.add_argument("--torch-benchmark-min-run-time-s", type=float, default=0.2)
    parser.add_argument("--allow-mps-fallback", action="store_true")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--set-kmp-duplicate-lib-ok", action="store_true")
    parser.add_argument("--omp-num-threads", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    cases = _parse_cases(str(args.cases))
    results = [_run_case(args, case) for case in cases]
    payload = {
        "meta": {
            "schema_version": "mps_env_var_sweep.v1",
            "script": "report_mps_env_var_sweep",
            "git_sha": _git_sha(),
            "python": str(args.python),
            "cases": cases,
            "workloads": str(args.workloads),
            "scales": str(args.scales),
            "policies": str(args.policies),
            "warmup": int(args.warmup),
            "iters": int(args.iters),
            "seed": int(args.seed),
            "timer": str(args.timer),
            "torch_benchmark_min_run_time_s": float(args.torch_benchmark_min_run_time_s),
            "allow_mps_fallback": bool(args.allow_mps_fallback),
            "set_kmp_duplicate_lib_ok": bool(args.set_kmp_duplicate_lib_ok),
            "omp_num_threads": args.omp_num_threads,
            "dry_run": bool(args.dry_run),
        },
        "results": results,
        "summary": {
            "ok": sum(1 for result in results if result["status"] == "ok"),
            "fail": sum(1 for result in results if result["status"] == "fail"),
            "dry_run": sum(1 for result in results if result["status"] == "dry_run"),
            "by_case": _case_summary(results),
        },
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if payload["summary"]["fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
