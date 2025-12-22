#!/usr/bin/env python
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
import sys
import time
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _utc_now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


def _git_short_sha() -> str:
    try:
        import subprocess

        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return ""


def _read_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            raise ValueError(f"jsonl row is not an object at {path}:{i}")
        rows.append(obj)
    return rows


def _sha256_file(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    _ensure_parent(dst)
    shutil.copy2(src, dst)
    return True


def _write_manifest(
    out_dir: Path,
    *,
    run_id: str,
    command_argv: Sequence[str],
    bench_commands: Sequence[Tuple[str, Sequence[str]]],
    postprocess_argv: Sequence[str],
    jsonl_paths: Sequence[Path],
    jsonl_path: Path,
    post_out_dir: Path,
    claimed_paths: Sequence[Tuple[str, Path]],
) -> None:
    sha = _git_short_sha()
    rows = _read_jsonl_rows(jsonl_path)
    first_meta = (rows[0].get("meta") if rows else None) or {}

    def _kv(k: str, v: Any) -> str:
        return f"{k}: {v}"

    lines: List[str] = []
    lines.append(_kv("artifact", "boundflow-phase5d"))
    lines.append(_kv("run_id", run_id))
    lines.append(_kv("time_utc", _utc_now_iso()))
    lines.append(_kv("git_commit", sha))
    lines.append(_kv("command", " ".join(command_argv)))
    for name, argv in bench_commands:
        lines.append(_kv(f"bench_command.{name}", "python scripts/bench_ablation_matrix.py " + " ".join(argv)))
    lines.append(_kv("postprocess_command", "python scripts/postprocess_ablation_jsonl.py " + " ".join(postprocess_argv)))
    lines.append(_kv("rows", len(rows)))
    if isinstance(first_meta, dict) and first_meta:
        for k in ("python", "torch", "tvm", "torch_num_threads", "platform", "host"):
            if k in first_meta:
                lines.append(_kv(f"meta.{k}", first_meta.get(k)))

    lines.append("")
    lines.append("outputs:")
    outputs = [
        jsonl_path,
        *list(jsonl_paths),
        post_out_dir / "ablation.csv",
        post_out_dir / "tables" / "ablation_summary.csv",
        post_out_dir / "tables" / "table_main.csv",
        post_out_dir / "MANIFEST.txt",
    ]
    for p in outputs:
        lines.append(f"- {p}")
    lines.append("")
    lines.append("paper_facing_outputs:")
    for name, p in claimed_paths:
        lines.append(f"- {name}: {p}")

    lines.append("")
    lines.append("sha256:")
    for p in outputs:
        dig = _sha256_file(Path(p))
        if dig is not None:
            lines.append(f"- {p}: {dig}")
    for _, p in claimed_paths:
        dig = _sha256_file(Path(p))
        if dig is not None:
            lines.append(f"- {p}: {dig}")

    (out_dir / "MANIFEST.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _apply_repro_env(*, torch_num_threads: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ.setdefault("TVM_FFI_DISABLE_TORCH_C_DLPACK", "1")
    os.environ.setdefault("BOUNDFLOW_QUIET", "1")
    # Force a headless matplotlib backend to avoid Qt/Wayland warnings in artifact runs.
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("OMP_NUM_THREADS", str(torch_num_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(torch_num_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(torch_num_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(torch_num_threads))

    # Best-effort: torch threads.
    try:
        import torch

        torch.set_num_threads(int(torch_num_threads))
    except Exception:
        pass


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["quick", "full"], default="quick")
    p.add_argument("--run-id", type=str, default="")
    p.add_argument("--out-root", type=str, default="artifacts/phase5d")
    p.add_argument("--workload", choices=["mlp", "mnist_cnn", "all"], default="mlp")
    p.add_argument("--torch-num-threads", type=int, default=1)
    p.add_argument(
        "--allow-no-tvm",
        action="store_true",
        help="Allow generating a python-only artifact when TVM is unavailable (TVM-related claims are not covered).",
    )
    p.add_argument("--keep-intermediate", action="store_true")
    args = p.parse_args(argv)

    run_id = str(args.run_id or f"{int(time.time())}_{os.getpid()}")
    out_dir = Path(args.out_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    _apply_repro_env(torch_num_threads=int(args.torch_num_threads))

    # TVM is required by default (it measures compile/run + cache stats), but can be downgraded.
    tvm_ok = True
    try:
        import tvm  # noqa: F401
    except Exception as e:
        tvm_ok = False
        if not bool(args.allow_no_tvm):
            sys.stderr.write(
                "ERROR: TVM python package is not importable; Phase5D artifact requires TVM by default.\n"
                "Hint: activate conda env and source env.sh, then rebuild TVM if needed.\n"
                "Or re-run with --allow-no-tvm to generate a python-only artifact.\n"
                f"Underlying error: {e}\n"
            )
            return 2

    jsonl_path = out_dir / "results.jsonl"
    post_out_dir = out_dir / "_postprocess"

    from scripts.bench_ablation_matrix import main as bench_main
    from scripts.postprocess_ablation_jsonl import main as post_main

    # Bench args: keep schema stable; only tweak warmup/iters/matrix for quick/full.
    if args.mode == "quick":
        matrix = "small"
        warmup = "1"
        iters = "1"
    else:
        matrix = "default"
        warmup = "3"
        iters = "20"

    workloads: List[str]
    if str(args.workload) == "all":
        workloads = ["mlp", "mnist_cnn"]
    else:
        workloads = [str(args.workload)]

    bench_commands: List[Tuple[str, List[str]]] = []
    jsonl_inputs: List[Path] = []
    # Always write a merged JSONL at out_dir/results.jsonl for downstream tooling.
    jsonl_path.write_text("", encoding="utf-8")
    for w in workloads:
        per_dir = out_dir / "runs" / w
        per_dir.mkdir(parents=True, exist_ok=True)
        per_jsonl = per_dir / "results.jsonl"
        bench_argv = [
            "--workload",
            str(w),
            "--matrix",
            matrix,
            "--warmup",
            warmup,
            "--iters",
            iters,
            "--output",
            str(per_jsonl),
        ]
        if not tvm_ok:
            bench_argv.append("--no-tvm")
        rc = bench_main(list(bench_argv))
        if rc != 0:
            return int(rc)
        bench_commands.append((str(w), list(bench_argv)))
        jsonl_inputs.append(per_jsonl)
        # Append to merged JSONL.
        merged = jsonl_path
        merged.parent.mkdir(parents=True, exist_ok=True)
        with merged.open("a", encoding="utf-8") as f_out:
            for line in per_jsonl.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                # Hardening: ensure workload label exists in every merged row.
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        wl = obj.get("workload")
                        if not isinstance(wl, dict):
                            wl = {}
                            obj["workload"] = wl
                        wl.setdefault("model", str(w))
                        line = json.dumps(obj, ensure_ascii=False)
                except Exception:
                    pass
                f_out.write(line.rstrip("\n") + "\n")

    postprocess_argv = [
        str(jsonl_path),
        "--out-dir",
        str(post_out_dir),
    ]
    rc = post_main(list(postprocess_argv))
    if rc != 0:
        return int(rc)

    # Normalize "paper-facing" filenames and directory structure.
    claimed: List[Tuple[str, Path]] = []
    if _copy_if_exists(post_out_dir / "ablation.csv", out_dir / "results_flat.csv"):
        claimed.append(("results_flat.csv", out_dir / "results_flat.csv"))

    if _copy_if_exists(post_out_dir / "tables" / "ablation_summary.csv", out_dir / "tables" / "table_ablation.csv"):
        claimed.append(("tables/table_ablation.csv", out_dir / "tables" / "table_ablation.csv"))

    if _copy_if_exists(post_out_dir / "tables" / "table_main.csv", out_dir / "tables" / "table_main.csv"):
        claimed.append(("tables/table_main.csv", out_dir / "tables" / "table_main.csv"))

    # Figures are best-effort (matplotlib optional).
    if _copy_if_exists(
        post_out_dir / "figures" / "cache_miss_vs_compile_first_run.png",
        out_dir / "figures" / "fig_cache_miss_vs_compile_first_run.png",
    ):
        claimed.append(
            ("figures/fig_cache_miss_vs_compile_first_run.png", out_dir / "figures" / "fig_cache_miss_vs_compile_first_run.png")
        )

    if _copy_if_exists(
        post_out_dir / "figures" / "call_tir_vs_fusion.png",
        out_dir / "figures" / "fig_call_tir_vs_fusion.png",
    ):
        claimed.append(("figures/fig_call_tir_vs_fusion.png", out_dir / "figures" / "fig_call_tir_vs_fusion.png"))

    if _copy_if_exists(
        post_out_dir / "figures" / "mem_bytes_vs_runtime.png",
        out_dir / "figures" / "fig_mem_bytes_vs_runtime.png",
    ):
        claimed.append(("figures/fig_mem_bytes_vs_runtime.png", out_dir / "figures" / "fig_mem_bytes_vs_runtime.png"))

    if _copy_if_exists(
        post_out_dir / "figures" / "mem_bytes_vs_runtime_by_workload.png",
        out_dir / "figures" / "fig_mem_bytes_vs_runtime_by_workload.png",
    ):
        claimed.append(
            (
                "figures/fig_mem_bytes_vs_runtime_by_workload.png",
                out_dir / "figures" / "fig_mem_bytes_vs_runtime_by_workload.png",
            )
        )

    if _copy_if_exists(
        post_out_dir / "figures" / "speedup_hot_vs_auto_lirpa_by_workload.png",
        out_dir / "figures" / "fig_speedup_hot_vs_auto_lirpa_by_workload.png",
    ):
        claimed.append(
            (
                "figures/fig_speedup_hot_vs_auto_lirpa_by_workload.png",
                out_dir / "figures" / "fig_speedup_hot_vs_auto_lirpa_by_workload.png",
            )
        )

    if _copy_if_exists(
        post_out_dir / "figures" / "runtime_cold_vs_hot.png",
        out_dir / "figures" / "fig_runtime_cold_vs_hot.png",
    ):
        claimed.append(("figures/fig_runtime_cold_vs_hot.png", out_dir / "figures" / "fig_runtime_cold_vs_hot.png"))

    if _copy_if_exists(
        post_out_dir / "figures" / "mem_quadrants.png",
        out_dir / "figures" / "fig_mem_quadrants.png",
    ):
        claimed.append(("figures/fig_mem_quadrants.png", out_dir / "figures" / "fig_mem_quadrants.png"))

    if _copy_if_exists(
        post_out_dir / "figures" / "runtime_breakdown.png",
        out_dir / "figures" / "fig_runtime_breakdown.png",
    ):
        claimed.append(("figures/fig_runtime_breakdown.png", out_dir / "figures" / "fig_runtime_breakdown.png"))

    # Bundle claims doc into the artifact directory for AE convenience (source of truth lives in gemini_doc/).
    repo_root = Path(__file__).resolve().parents[1]
    claims_src = repo_root / "gemini_doc" / "artifact_claims_phase5d.md"
    if claims_src.exists():
        if _copy_if_exists(claims_src, out_dir / "CLAIMS.md"):
            claimed.append(("CLAIMS.md", out_dir / "CLAIMS.md"))

    appendix_src = repo_root / "gemini_doc" / "artifact_appendix_phase5d.md"
    if appendix_src.exists():
        if _copy_if_exists(appendix_src, out_dir / "APPENDIX.md"):
            claimed.append(("APPENDIX.md", out_dir / "APPENDIX.md"))

    _write_manifest(
        out_dir,
        run_id=run_id,
        command_argv=[Path(sys.argv[0]).name, *(argv or [])],
        bench_commands=bench_commands,
        postprocess_argv=postprocess_argv,
        jsonl_paths=jsonl_inputs,
        jsonl_path=jsonl_path,
        post_out_dir=post_out_dir,
        claimed_paths=claimed,
    )

    if not bool(args.keep_intermediate):
        shutil.rmtree(post_out_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
