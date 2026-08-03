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
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    task_trace = json.loads((artifact / "task_trace.json").read_text(encoding="utf-8"))
    bound_result = json.loads(
        (artifact / "bound_result.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == "boundflow.schedule-ir-artifact/v2"
    assert all(event["output_value_hashes"] for event in task_trace["events"])
    assert bound_result["schema_version"] == "boundflow.bound-result-evidence/v1"
    assert len(bound_result["lower"]["sha256"]) == 64
    assert len(bound_result["upper"]["sha256"]) == 64


def test_schedule_ir_reference_artifact_rejects_semantic_result_tamper(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "schedule-ir-reference"
    script = Path("scripts/run_schedule_ir_v1_reference_artifact.py")
    subprocess.run(
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
    result_path = artifact / "bound_result.json"
    result_path.write_text(
        result_path.read_text(encoding="utf-8").replace(
            '"sha256":"', '"sha256":"tampered'
        ),
        encoding="utf-8",
    )
    replayed = subprocess.run(
        [
            sys.executable,
            str(script),
            "replay",
            "--artifact-dir",
            str(artifact),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert replayed.returncode != 0
    assert "Bound result replay mismatch" in replayed.stderr
