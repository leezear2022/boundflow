from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_ENV_CASES = {
    "default": {
        "PYTORCH_MPS_PREFER_METAL": None,
        "PYTORCH_MPS_FAST_MATH": None,
    },
    "prefer_metal": {
        "PYTORCH_MPS_PREFER_METAL": "1",
        "PYTORCH_MPS_FAST_MATH": None,
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


def _parse_csv(raw: str, allowed: Optional[set[str]] = None) -> List[str]:
    out: List[str] = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        if allowed is not None and item not in allowed:
            raise ValueError(f"unknown item: {item}")
        out.append(item)
    if not out:
        raise ValueError(f"empty csv value: {raw}")
    return out


def _child_env(
    *,
    case: str,
    set_kmp_duplicate_lib_ok: bool,
    omp_num_threads: Optional[str],
) -> Tuple[Dict[str, str], Dict[str, Optional[str]]]:
    env = dict(os.environ)
    applied: Dict[str, Optional[str]] = {}
    for key, value in _ENV_CASES[case].items():
        applied[key] = value
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
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


def _json_from_stdout(stdout: str) -> Dict[str, Any]:
    raw = stdout.strip()
    if not raw:
        raise ValueError("child produced empty stdout")
    return json.loads(raw.splitlines()[-1])


def _tail(text: str, lines: int = 16) -> str:
    return "\n".join(text.splitlines()[-lines:])


def _run_child(args: argparse.Namespace, *, case: str, workload: str, scale: str, policy: str, seed: int) -> Dict[str, Any]:
    env, applied_env = _child_env(
        case=case,
        set_kmp_duplicate_lib_ok=bool(args.set_kmp_duplicate_lib_ok),
        omp_num_threads=args.omp_num_threads,
    )
    cmd = [
        str(args.python),
        str(Path(__file__).resolve()),
        "--_child",
        "--case",
        case,
        "--workload",
        workload,
        "--scale",
        scale,
        "--policy",
        policy,
        "--seed",
        str(seed),
    ]
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
            "env": applied_env,
            "returncode": int(completed.returncode),
            "stdout_tail": _tail(completed.stdout),
            "stderr_tail": _tail(completed.stderr),
        }
    try:
        payload = _json_from_stdout(completed.stdout)
    except Exception as exc:
        return {
            "case": case,
            "status": "fail",
            "env": applied_env,
            "returncode": int(completed.returncode),
            "parse_error": f"{type(exc).__name__}: {exc}",
            "stdout_tail": _tail(completed.stdout),
            "stderr_tail": _tail(completed.stderr),
        }
    payload["env"] = applied_env
    return payload


def _compare_bounds(default: Dict[str, Any], prefer: Dict[str, Any], *, atol: float, rtol: float) -> Dict[str, Any]:
    import math

    def flatten(xs: Any) -> List[float]:
        if isinstance(xs, list):
            out: List[float] = []
            for x in xs:
                out.extend(flatten(x))
            return out
        return [float(xs)]

    lower_default = flatten(default["bounds"]["lower"])
    lower_prefer = flatten(prefer["bounds"]["lower"])
    upper_default = flatten(default["bounds"]["upper"])
    upper_prefer = flatten(prefer["bounds"]["upper"])
    diffs = [abs(a - b) for a, b in zip(lower_default, lower_prefer)]
    diffs.extend(abs(a - b) for a, b in zip(upper_default, upper_prefer))
    max_abs_diff = max(diffs) if diffs else 0.0

    allclose = True
    for a, b in zip(lower_default + upper_default, lower_prefer + upper_prefer):
        if not math.isclose(a, b, rel_tol=float(rtol), abs_tol=float(atol)):
            allclose = False
            break

    return {
        "allclose": bool(allclose),
        "max_abs_diff": float(max_abs_diff),
        "cert_decision_match": default["certified_decisions"] == prefer["certified_decisions"],
        "default_certified_count": int(default["certified_count"]),
        "prefer_metal_certified_count": int(prefer["certified_count"]),
    }


def _unknown_calls(payload: Dict[str, Any]) -> int:
    attr = payload["counts"]["operator_attribution"]
    return int(attr["materialization"]["by_reason"].get("unknown_materialization", {}).get("calls", 0))


def _run_parent(args: argparse.Namespace) -> int:
    workloads = _parse_csv(args.workloads)
    scales = _parse_csv(args.scales, {"smoke", "small", "bench"})
    policies = _parse_csv(args.policies, {"structured", "dense_barrier", "auto"})
    rows: List[Dict[str, Any]] = []
    case_idx = 0
    for scale in scales:
        for workload in workloads:
            for policy in policies:
                seed = int(args.seed) + case_idx
                default = _run_child(args, case="default", workload=workload, scale=scale, policy=policy, seed=seed)
                prefer = _run_child(args, case="prefer_metal", workload=workload, scale=scale, policy=policy, seed=seed)
                row = {
                    "workload": workload,
                    "scale_id": scale,
                    "policy_request": policy,
                    "seed": int(seed),
                    "default": default,
                    "prefer_metal": prefer,
                }
                if default.get("status") == "ok" and prefer.get("status") == "ok":
                    comparison = _compare_bounds(default, prefer, atol=float(args.atol), rtol=float(args.rtol))
                    comparison["default_unknown_materialization_calls"] = _unknown_calls(default)
                    comparison["prefer_metal_unknown_materialization_calls"] = _unknown_calls(prefer)
                    comparison["fallback_disabled"] = (
                        not bool(default["device_meta"]["mps_fallback_enabled"])
                        and not bool(prefer["device_meta"]["mps_fallback_enabled"])
                    )
                    row["comparison"] = comparison
                    row["status"] = (
                        "ok"
                        if comparison["allclose"]
                        and comparison["cert_decision_match"]
                        and comparison["default_unknown_materialization_calls"] == 0
                        and comparison["prefer_metal_unknown_materialization_calls"] == 0
                        and comparison["fallback_disabled"]
                        else "fail"
                    )
                else:
                    row["status"] = "fail"
                rows.append(row)
            case_idx += 1

    summary = {
        "ok": sum(1 for row in rows if row["status"] == "ok"),
        "fail": sum(1 for row in rows if row["status"] == "fail"),
        "max_abs_diff": max(
            (float(row.get("comparison", {}).get("max_abs_diff", 0.0)) for row in rows),
            default=0.0,
        ),
    }
    payload = {
        "meta": {
            "schema_version": "mps_prefer_metal_guardrails.v1",
            "script": "report_mps_prefer_metal_guardrails",
            "git_sha": _git_sha(),
            "python": str(args.python),
            "workloads": workloads,
            "scales": scales,
            "policies": policies,
            "seed": int(args.seed),
            "atol": float(args.atol),
            "rtol": float(args.rtol),
            "set_kmp_duplicate_lib_ok": bool(args.set_kmp_duplicate_lib_ok),
            "omp_num_threads": args.omp_num_threads,
        },
        "rows": rows,
        "summary": summary,
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if summary["fail"] == 0 else 1


def _certified_decisions(lower: Any, upper: Any) -> Tuple[List[int], int]:
    import torch

    decisions: List[int] = []
    certified = 0
    for lower_row, upper_row in zip(lower, upper):
        winner = int(torch.argmax(lower_row).item())
        if int(lower_row.numel()) <= 1:
            margin = lower_row[winner]
        else:
            mask = torch.ones_like(upper_row, dtype=torch.bool)
            mask[winner] = False
            margin = lower_row[winner] - torch.max(upper_row[mask])
        if bool((margin > 0).item()):
            certified += 1
            decisions.append(winner)
        else:
            decisions.append(-1)
    return decisions, certified


def _run_child_mode(args: argparse.Namespace) -> int:
    import torch

    from boundflow.runtime.bound_planner import plan_phase7b_shared_crown
    from scripts.bench_phase7a_shared_crown_path_attribution import (
        _build_case,
        _collect_counts,
        _device_meta,
        _device_name,
        _make_device,
        _run_variant_once,
    )

    dtype = torch.float32
    device = _make_device("mps", dtype_name="float32", allow_mps_fallback=False)
    compare_target, module, spec = _build_case(
        str(args.workload),
        device=device,
        dtype=dtype,
        profile=str(args.scale),
        seed=int(args.seed),
    )
    planner = plan_phase7b_shared_crown(
        compare_target=compare_target,
        workload=str(args.workload),
        scale_id=str(args.scale),
        device=str(device),
        requested_final_concretization_policy=str(args.policy),  # type: ignore[arg-type]
    )
    bounds = _run_variant_once(
        module,
        spec,
        variant="structured",
        final_policy=planner.final_concretization_policy,
    )
    counts = _collect_counts(
        module,
        spec,
        variant="structured",
        use_dense_cache=planner.use_dense_cache,
        final_policy=planner.final_concretization_policy,
    )
    lower = bounds.lower.detach().cpu()
    upper = bounds.upper.detach().cpu()
    decisions, certified_count = _certified_decisions(lower, upper)
    payload = {
        "case": str(args.case),
        "status": "ok",
        "torch_version": torch.__version__,
        "device": str(device),
        "device_name": _device_name(device),
        "device_meta": _device_meta(device, allow_mps_fallback=False),
        "workload": str(args.workload),
        "scale_id": str(args.scale),
        "policy_request": str(args.policy),
        "planner_decision": planner.to_jsonable(),
        "bounds": {
            "lower": lower.tolist(),
            "upper": upper.tolist(),
            "shape": list(lower.shape),
        },
        "certified_decisions": decisions,
        "certified_count": int(certified_count),
        "counts": counts,
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Default vs PYTORCH_MPS_PREFER_METAL guardrails for Phase 7B.")
    parser.add_argument("--workloads", type=str, default="all")
    parser.add_argument("--scales", type=str, default="small,bench")
    parser.add_argument("--policies", type=str, default="structured,dense_barrier,auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--set-kmp-duplicate-lib-ok", action="store_true")
    parser.add_argument("--omp-num-threads", type=str, default=None)
    parser.add_argument("--_child", action="store_true")
    parser.add_argument("--case", type=str, default="default", choices=["default", "prefer_metal"])
    parser.add_argument("--workload", type=str, default="permute_reshape_linear")
    parser.add_argument("--scale", type=str, default="smoke", choices=["smoke", "small", "bench"])
    parser.add_argument("--policy", type=str, default="auto", choices=["structured", "dense_barrier", "auto"])
    args = parser.parse_args(argv)
    if bool(args._child):
        return _run_child_mode(args)
    if str(args.workloads).strip() == "all":
        args.workloads = "relu_heavy_mlp,residual_relu_mlp,concat_relu_mlp,permute_reshape_linear"
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
