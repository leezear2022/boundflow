from __future__ import annotations

import os
import sys
from typing import List, Tuple

import pytest


def _env_summary_lines() -> List[str]:
    lines = [
        f"Python Executable: {sys.executable}",
        f"CONDA_DEFAULT_ENV: {os.environ.get('CONDA_DEFAULT_ENV', 'Not Set')}",
        f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not Set')}",
    ]
    if os.environ.get("CONDA_DEFAULT_ENV") != "boundflow":
        lines.append(
            "[HINT] 建议在 conda 环境 'boundflow' 下运行：`conda activate boundflow && python tests/test_env.py`"
        )
    lines.append("-" * 40)
    return lines


def _try_import(display_name: str, package_name: str) -> Tuple[bool, str]:
    try:
        __import__(package_name)
        return True, f"[OK] Import {display_name} success"
    except ImportError as e:
        return False, f"[FAIL] Import {display_name} failed: {e}"
    except Exception as e:
        # e.g. compiled library missing
        return False, f"[FAIL] Import {display_name} error: {e}"


def _check_env(required: List[Tuple[str, str]]) -> Tuple[bool, List[str]]:
    messages: List[str] = []
    messages.extend(_env_summary_lines())

    ok = True
    for display_name, package_name in required:
        success, msg = _try_import(display_name, package_name)
        messages.append(msg)
        ok &= success

    messages.append("-" * 40)
    messages.append("Environment verification passed!" if ok else "Environment verification FAILED.")
    return ok, messages


def test_env_smoke_imports():
    # Core (minimal) import smoke test: should pass in a clean environment.
    ok, messages = _check_env(
        [
            ("PyTorch", "torch"),
            ("BoundFlow", "boundflow"),
        ]
    )
    # pytest 默认捕获 stdout；需要查看输出可用 `pytest -s tests/test_env.py`
    print("\n".join(messages))
    assert ok


def test_env_optional_imports():
    # Optional deps used by parts of the repo. Skip if not installed.
    missing: List[str] = []
    for display_name, package_name in [
        ("Auto_LiRPA", "auto_LiRPA"),
        ("TVM", "tvm"),
    ]:
        try:
            __import__(package_name)
        except ImportError:
            missing.append(display_name)
    if missing:
        pytest.skip(f"optional deps not installed: {', '.join(missing)}")

    ok, messages = _check_env(
        [
            ("PyTorch", "torch"),
            ("Auto_LiRPA", "auto_LiRPA"),
            ("BoundFlow", "boundflow"),
            ("TVM", "tvm"),
        ]
    )
    print("\n".join(messages))
    assert ok


def main() -> int:
    # CLI mode: strict check.
    ok, messages = _check_env(
        [
            ("PyTorch", "torch"),
            ("Auto_LiRPA", "auto_LiRPA"),
            ("BoundFlow", "boundflow"),
            ("TVM", "tvm"),
        ]
    )
    print("\n".join(messages))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
