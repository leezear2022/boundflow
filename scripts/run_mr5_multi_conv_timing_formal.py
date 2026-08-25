#!/usr/bin/env python3
"""Generate or replay the MR5 multi-site production timing artifact."""

# pylint: disable=protected-access,wrong-import-position,missing-function-docstring

from __future__ import annotations

import subprocess
import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boundflow.runtime import (
    mr3_production_bridge_timing as legacy_runtime,
)  # noqa: E402
from boundflow.runtime.mr5_multi_conv_timing import (  # noqa: E402
    BOOTSTRAP_LOWER_GATE,
    EXPECTED_RUNS,
    HOST_GEOMEAN_GATE,
    MEMORY_RATIO_GATE,
    SOURCE_COMMIT,
    TIMING_SCHEMA,
    WORST_PAIR_GATE,
    derive_summary,
)
from scripts import run_mr3_production_bridge_timing_formal as legacy  # noqa: E402

ARTIFACT_SCHEMA = "boundflow.mr5-multi-conv-timing-artifact/v1"
PROTOCOL_SCHEMA = "boundflow.mr5-multi-conv-timing-protocol/v1"
WORKER = "scripts/run_mr5_multi_conv_timing_worker.py"
CORRECTNESS_ARTIFACT = (
    ROOT / "artifacts/measurement-recovery/mr5-multi-conv-production-bridge-v1"
)
CODE_PATHS = (
    "boundflow/runtime/mr5_multi_conv_production_bridge.py",
    "boundflow/runtime/mr5_multi_conv_timing.py",
    WORKER,
    "scripts/run_mr5_multi_conv_timing_formal.py",
    "scripts/probe_mr5_multi_conv_timing_tamper.py",
    "tests/test_mr5_multi_conv_timing.py",
)


def _correctness_identity() -> dict[str, object]:
    replay = subprocess.run(
        (
            sys.executable,
            str(ROOT / "scripts/run_mr5_multi_conv_formal.py"),
            "--artifact",
            str(CORRECTNESS_ARTIFACT),
            "--replay",
        ),
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    manifest = legacy._load_json(CORRECTNESS_ARTIFACT / "manifest.json")
    status = "VALIDATED-MR5-MULTI-CONV-PRODUCTION-BRIDGE-CORRECTNESS"
    if status not in replay.stdout:
        raise ValueError("MR5 correctness prerequisite did not replay")
    return {
        "manifest_file_sha256": legacy._sha256(CORRECTNESS_ARTIFACT / "manifest.json"),
        "manifest_hash": manifest["manifest_hash"],
        "summary_file_sha256": manifest["files"]["summary.json"],
        "replay_status": status,
    }


def _configure_legacy_generator() -> None:
    legacy.ARTIFACT_SCHEMA = ARTIFACT_SCHEMA
    legacy.PROTOCOL_SCHEMA = PROTOCOL_SCHEMA
    legacy.WORKER = WORKER
    legacy.CORRECTNESS_ARTIFACT = CORRECTNESS_ARTIFACT
    setattr(legacy, "CODE_PATHS", CODE_PATHS)
    legacy.EXPECTED_RUNS = EXPECTED_RUNS
    legacy.SOURCE_COMMIT = SOURCE_COMMIT
    legacy.TIMING_SCHEMA = TIMING_SCHEMA
    legacy.HOST_GEOMEAN_GATE = HOST_GEOMEAN_GATE
    legacy.BOOTSTRAP_SEED = legacy_runtime.BOOTSTRAP_SEED
    legacy.BOOTSTRAP_SAMPLES = legacy_runtime.BOOTSTRAP_SAMPLES
    setattr(legacy, "BOOTSTRAP_LOWER_GATE", BOOTSTRAP_LOWER_GATE)
    legacy.WORST_PAIR_GATE = WORST_PAIR_GATE
    legacy.MEMORY_RATIO_GATE = MEMORY_RATIO_GATE
    legacy.derive_summary = derive_summary
    legacy._correctness_identity = _correctness_identity


def replay_artifact(artifact: Path) -> dict[str, object]:
    _configure_legacy_generator()
    return legacy.replay_artifact(artifact)


def refresh_manifest(artifact: Path) -> dict[str, object]:
    _configure_legacy_generator()
    return legacy.refresh_manifest(artifact)


def _load_json(path: Path):
    return legacy._load_json(path)


def _write_json(path: Path, value: object) -> None:
    legacy._write_json(path, value)


def main() -> None:
    _configure_legacy_generator()
    legacy.main()


if __name__ == "__main__":
    main()
