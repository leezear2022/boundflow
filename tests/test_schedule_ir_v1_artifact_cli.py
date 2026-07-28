"""Fresh-process artifact contract for Schedule IR v1."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def test_schedule_ir_reference_artifact_generates_and_replays(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "schedule-ir-reference"
    script = Path("scripts/run_schedule_ir_v1_reference_artifact.py")
    generated_process = subprocess.run(
        [
            sys.executable,
            str(script),
            "generate",
            "--out-dir",
            str(artifact),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    replayed_process = subprocess.run(
        [
            sys.executable,
            str(script),
            "replay",
            "--artifact-dir",
            str(artifact),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    generated = json.loads(generated_process.stdout.strip().splitlines()[-1])
    replayed = json.loads(replayed_process.stdout.strip().splitlines()[-1])
    assert generated["status"] == "generated"
    assert replayed["status"] == "replayed"
    assert generated["schedule_hash"] == replayed["schedule_hash"]
    assert generated["trace_hash"] == replayed["trace_hash"]
    assert generated["task_module_hash"] == replayed["task_module_hash"]
    assert generated["task_trace_hash"] == replayed["task_trace_hash"]
