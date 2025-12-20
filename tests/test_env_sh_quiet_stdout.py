from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_env_sh_does_not_pollute_stdout() -> None:
    root = _repo_root()
    env = os.environ.copy()
    env.pop("BOUNDFLOW_QUIET", None)
    proc = subprocess.run(
        ["bash", "-c", "source ./env.sh"],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout == ""
    assert "BoundFlow environment configured." in proc.stderr
    assert "TVM_HOME=" in proc.stderr


def test_env_sh_can_be_silenced() -> None:
    root = _repo_root()
    env = os.environ.copy()
    env["BOUNDFLOW_QUIET"] = "1"
    proc = subprocess.run(
        ["bash", "-c", "source ./env.sh"],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout == ""
    assert proc.stderr == ""

