"""Fresh-process replay contract for the frozen real-verifier IR artifact."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def test_frozen_real_verifier_ir_artifact_replays() -> None:
    """The committed artifact must recompile all 394 calls in a fresh process."""

    artifact = Path("artifacts/rvir/rvir-cpu-correctness-v1-20260803")
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_real_verifier_ir_artifact.py",
            "replay",
            "--artifact-dir",
            str(artifact),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result == {
        "activation_call_count": 394,
        "performance_claimed": False,
        "status": "replayed",
    }
