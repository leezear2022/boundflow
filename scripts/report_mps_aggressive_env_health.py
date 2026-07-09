from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _tail(text: str, lines: int = 16) -> str:
    return "\n".join(text.splitlines()[-lines:])


def _run(cmd: List[str], *, env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    completed = subprocess.run(
        cmd,
        cwd=str(_repo_root()),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "command": cmd,
        "returncode": int(completed.returncode),
        "ok": completed.returncode == 0,
        "stdout_tail": _tail(completed.stdout),
        "stderr_tail": _tail(completed.stderr),
    }


def _conda_run(env_name: str, code: str, *, extra_env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    return _run(["conda", "run", "-n", env_name, "python", "-c", code], env=env)


def _conda_openmp_packages(env_name: str) -> Dict[str, Any]:
    completed = subprocess.run(
        ["conda", "list", "-n", env_name],
        cwd=str(_repo_root()),
        text=True,
        capture_output=True,
        check=False,
    )
    packages = [
        line
        for line in completed.stdout.splitlines()
        if any(token in line.lower() for token in ("omp", "openmp", "openblas", "mkl"))
    ]
    return {
        "command": ["conda", "list", "-n", env_name],
        "returncode": int(completed.returncode),
        "ok": completed.returncode == 0,
        "detected_packages": packages,
        "stderr_tail": _tail(completed.stderr),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Health report for the aggressive macOS MPS PyTorch env.")
    parser.add_argument("--conda-env", type=str, default="boundflow-mps-aggressive")
    args = parser.parse_args(argv)

    torch_probe = (
        "import torch, json, os; "
        "print(json.dumps({"
        "'torch_version': torch.__version__, "
        "'mps_built': torch.backends.mps.is_built(), "
        "'mps_available': torch.backends.mps.is_available(), "
        "'fallback': os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'), "
        "'prefer_metal': os.environ.get('PYTORCH_MPS_PREFER_METAL'), "
        "'fast_math': os.environ.get('PYTORCH_MPS_FAST_MATH')"
        "}))"
    )
    clean = _conda_run(str(args.conda_env), torch_probe)
    kmp = _conda_run(str(args.conda_env), torch_probe, extra_env={"KMP_DUPLICATE_LIB_OK": "TRUE"})
    prefer_metal = _conda_run(
        str(args.conda_env),
        torch_probe,
        extra_env={"PYTORCH_MPS_PREFER_METAL": "1"},
    )
    packages = _conda_openmp_packages(str(args.conda_env))

    payload = {
        "meta": {
            "schema_version": "mps_aggressive_env_health.v1",
            "script": "report_mps_aggressive_env_health",
            "conda_env": str(args.conda_env),
        },
        "checks": {
            "import_torch_clean": clean,
            "import_torch_with_kmp_duplicate_lib_ok": kmp,
            "import_torch_prefer_metal_clean": prefer_metal,
            "conda_openmp_package_tail": {
                "ok": packages["ok"],
                "returncode": packages["returncode"],
                "detected_packages": packages["detected_packages"],
            },
        },
        "summary": {
            "clean_import_ok": bool(clean["ok"]),
            "kmp_workaround_needed": (not bool(clean["ok"])) and bool(kmp["ok"]),
            "prefer_metal_clean_import_ok": bool(prefer_metal["ok"]),
        },
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if payload["summary"]["clean_import_ok"] and payload["summary"]["prefer_metal_clean_import_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
