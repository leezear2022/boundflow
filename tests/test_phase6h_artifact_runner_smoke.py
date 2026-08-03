"""Smoke tests for the Phase6H artifact runner."""

import os
from pathlib import Path
import subprocess
import sys


def test_phase6h_artifact_runner_smoke(tmp_path: Path) -> None:
    """Run all stages with an explicit Python independent of PATH."""
    out_dir = tmp_path / "artifact"
    environment = os.environ.copy()
    environment["PATH"] = "/usr/bin:/bin"
    environment["PHASE6H_PYTHON"] = sys.executable
    subprocess.check_call(
        ["bash", "scripts/run_phase6h_artifact.sh", str(out_dir)],
        env=environment,
    )

    assert (out_dir / "phase6h_e2e.jsonl").exists()
    assert (out_dir / "phase6h_e2e.csv").exists()
    assert (out_dir / "phase6h_e2e_summary.md").exists()
    assert (out_dir / "env.txt").exists()
    assert (out_dir / "pip_freeze.txt").exists()
    assert (out_dir / "conda_list.txt").exists()

    assert (out_dir / "phase6h_e2e.jsonl").stat().st_size > 0
    assert (out_dir / "phase6h_e2e.csv").stat().st_size > 0
    assert (out_dir / "phase6h_e2e_summary.md").stat().st_size > 0
    assert f"python: {sys.executable}" in (out_dir / "env.txt").read_text(
        encoding="utf-8"
    )


def test_phase6h_artifact_runner_rejects_missing_python(tmp_path: Path) -> None:
    """Reject a missing explicit interpreter before producing artifacts."""
    environment = os.environ.copy()
    environment["PHASE6H_PYTHON"] = str(tmp_path / "missing-python")

    completed = subprocess.run(
        ["bash", "scripts/run_phase6h_artifact.sh", str(tmp_path / "artifact")],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "Python interpreter not found" in completed.stderr
