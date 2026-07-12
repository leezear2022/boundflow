from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def command_output(*command: str) -> str | None:
    executable = shutil.which(command[0])
    if executable is None:
        return None
    result = subprocess.run(
        (executable, *command[1:]), text=True, capture_output=True, check=False
    )
    output = (result.stdout or result.stderr).strip()
    return output if output else None


def package_info(name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(name)
    except Exception as error:  # doctor 必须记录 ABI/import 错误，而不是吞掉现场。
        return {"available": False, "error": repr(error)}
    return {
        "available": True,
        "version": getattr(module, "__version__", None),
        "file": getattr(module, "__file__", None),
    }


def collect() -> dict[str, Any]:
    packages = {
        name: package_info(name)
        for name in ("torch", "torchvision", "tvm_ffi", "tvm", "auto_LiRPA")
    }
    torch_cuda = None
    if packages["torch"]["available"]:
        import torch

        torch_cuda = {
            "runtime": torch.version.cuda,
            "available": torch.cuda.is_available(),
            "device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        }
    return {
        "platform": platform.platform(),
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "tools": {
            name: command_output(name, "--version")
            for name in (
                "cmake",
                "ninja",
                "ccache",
                "llvm-config",
                "clang",
                "gcc",
                "nvcc",
            )
        },
        "nvidia_smi": command_output(
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ),
        "packages": packages,
        "torch_cuda": torch_cuda,
    }


def strict_errors(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report["conda_env"] != "boundflow":
        errors.append("CONDA_DEFAULT_ENV 不是 boundflow")
    for package in ("torch", "torchvision", "tvm_ffi", "tvm", "auto_LiRPA"):
        if not report["packages"][package]["available"]:
            errors.append(f"无法导入 {package}")
    torch = report["packages"]["torch"]
    if torch["available"] and not str(torch["version"]).startswith("2.12.1"):
        errors.append(f"torch 版本不是 2.12.1: {torch['version']}")
    if report["torch_cuda"] and report["torch_cuda"]["runtime"] != "13.2":
        errors.append(
            f"PyTorch CUDA runtime 不是 13.2: {report['torch_cuda']['runtime']}"
        )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    report = collect()
    errors = strict_errors(report) if args.strict else []
    report["strict_errors"] = errors
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
