"""Replay tests for the root CROWN terminal five-pair artifact."""

# pylint: disable=missing-function-docstring,duplicate-code

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = REPOSITORY_ROOT / "artifacts/root-crown-terminal-tir/resnet2b-prop0-v1"
REPLAY = REPOSITORY_ROOT / "scripts/package_root_crown_terminal_five_pair.py"


def _replay(artifact: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(REPLAY),
            "replay",
            "--artifact",
            str(artifact),
            "--repository",
            str(REPOSITORY_ROOT),
        ],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_root_crown_terminal_five_pair_artifact_replays() -> None:
    result = _replay(ARTIFACT)
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["fresh_process_count"] == 10
    assert summary["discrete_semantics_exact"] is True
    assert summary["performance_claimed"] is False
    assert summary["decision"] == "mechanism-correct-no-stable-query-speedup"


def test_root_crown_terminal_five_pair_artifact_rejects_summary_tamper(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    shutil.copytree(ARTIFACT, artifact)
    summary_path = artifact / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["speedups"]["query_wall_ns"]["geomean"] = 10.0
    summary_path.write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    result = _replay(artifact)
    assert result.returncode != 0
    assert "artifact file differs" in result.stderr


def test_root_crown_terminal_five_pair_artifact_has_no_local_paths() -> None:
    for path in ARTIFACT.iterdir():
        if path.is_file():
            payload = path.read_bytes()
            assert b"/home/" not in payload
            assert b"/tmp/" not in payload
