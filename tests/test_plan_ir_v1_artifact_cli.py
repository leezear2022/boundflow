"""Independent-process contract for the Plan IR v1 reference artifact."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def test_plan_ir_reference_artifact_generates_and_replays_in_fresh_processes(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "plan-ir-reference"
    script = Path("scripts/run_plan_ir_v1_reference_artifact.py")
    generate = subprocess.run(
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
    replay = subprocess.run(
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
    generated = json.loads(generate.stdout.strip().splitlines()[-1])
    replayed = json.loads(replay.stdout.strip().splitlines()[-1])
    assert generated["status"] == "generated"
    assert replayed["status"] == "replayed"
    assert generated["bound_module_hash"] == replayed["bound_module_hash"]
    assert generated["plan_template_hash"] == replayed["plan_template_hash"]
    assert generated["plan_instance_hash"] == replayed["plan_instance_hash"]
