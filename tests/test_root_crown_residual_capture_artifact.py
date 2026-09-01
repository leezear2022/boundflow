"""Replay tests for the root CROWN residual full-VJP capture."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/root-crown-residual-capture/resnet2b-prop0-v1"
REPLAY = ROOT / "scripts/package_root_crown_residual_capture.py"


def _replay(artifact: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(REPLAY),
            "replay",
            "--artifact",
            str(artifact),
            "--repository",
            str(ROOT),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_root_crown_residual_capture_replays() -> None:
    result = _replay(ARTIFACT)
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["evaluation_count"] == 5
    assert summary["full_vjp_output_count"] == 7
    assert summary["performance_claimed"] is False


def test_root_crown_residual_capture_rejects_receipt_tamper(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    shutil.copytree(ARTIFACT, artifact)
    receipt_path = artifact / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["backward_count"] = 0
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    result = _replay(artifact)
    assert result.returncode != 0
    assert "artifact file differs" in result.stderr
