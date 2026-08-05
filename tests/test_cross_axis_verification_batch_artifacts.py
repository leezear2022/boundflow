"""Frozen NRIR-43 Phase-A artifact replay and tamper tests."""

# pylint: disable=missing-function-docstring,protected-access,duplicate-code

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.run_cross_axis_verification_batch_formal import (
    _canonical_hash,
    _file_sha256,
    _plan,
    _validate_cross_batch,
    validate_formal,
    validate_worker,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / (
    "artifacts/cross-axis-verification-batch/"
    "vnncomp21-resnet2b-property0-three-repeat-cpu-phase-a-v1"
)
BENCHMARK_ROOT = Path("/tmp/boundflow-vnncomp2021-nrir43")


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _rehash_worker(worker: dict) -> None:
    for raw in worker["raw_runs"]:
        raw["raw_hash"] = _canonical_hash(
            {key: value for key, value in raw.items() if key != "raw_hash"}
        )
    worker["worker_hash"] = _canonical_hash(
        {key: value for key, value in worker.items() if key != "worker_hash"}
    )


def test_nrir43_formal_manifest_and_workers_validate() -> None:
    formal = _load(ARTIFACT / "formal.json")
    manifest = _load(ARTIFACT / "manifest.json")
    validate_formal(formal)
    assert formal["status"] == "validated-no-go"
    assert formal["decision"]["phase_a_go"] is False
    assert formal["decision"]["next_route"] == "stop_cross_axis_batching"
    assert formal["decision"]["parity_passed"] is True
    assert formal["decision"]["launch_gate_passed"] is True
    assert formal["decision"]["timing_gate_passed"] is False
    assert formal["performance_claimed"] is False
    for relative, expected in manifest["files"].items():
        assert _file_sha256(ARTIFACT / relative) == expected
    plan = _plan()
    for repeat_index in range(3):
        validate_worker(
            _load(ARTIFACT / "shards" / f"repeat-{repeat_index}.json"),
            plan=plan,
            repeat_index=repeat_index,
        )


def test_nrir43_replay_command_passes() -> None:
    if not BENCHMARK_ROOT.exists():
        pytest.skip("frozen VNN-COMP checkout is unavailable")
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_cross_axis_verification_batch_formal.py"),
            "replay",
            "--benchmark-root",
            str(BENCHMARK_ROOT),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout
    assert '"replay":"passed"' in completed.stdout


def test_synchronized_outer_rehash_does_not_mask_launch_tamper() -> None:
    worker = _load(ARTIFACT / "shards/repeat-0.json")
    candidate = next(
        raw for raw in worker["raw_runs"] if raw["row"]["mode"] == "cross_axis"
    )
    candidate["row"]["scorer_launch_count"] = 15
    candidate["row_hash"] = _canonical_hash(candidate["row"])
    _rehash_worker(worker)
    with pytest.raises(ValueError, match="evidence row differs"):
        validate_worker(worker, plan=_plan(), repeat_index=0)


def test_synchronized_outer_rehash_does_not_mask_segment_tamper() -> None:
    worker = _load(ARTIFACT / "shards/repeat-0.json")
    candidate = next(
        raw for raw in worker["raw_runs"] if raw["row"]["mode"] == "cross_axis"
    )
    batch = deepcopy(candidate["cross_batches"][1])
    batch["plan"]["segments"][1]["child_domain_offset"] = 1
    candidate["cross_batches"][1] = batch
    _rehash_worker(worker)
    with pytest.raises(ValueError, match="segments are not packed"):
        validate_worker(worker, plan=_plan(), repeat_index=0)


def test_typed_cross_batch_rejects_objective_owner_tamper() -> None:
    worker = _load(ARTIFACT / "shards/repeat-0.json")
    candidate = next(
        raw for raw in worker["raw_runs"] if raw["row"]["mode"] == "cross_axis"
    )
    batch = deepcopy(candidate["cross_batches"][0])
    batch["plan"]["segments"][0]["objective_hash"] = "0" * 64
    with pytest.raises(ValueError, match="Plan IR is invalid|Instance IR differs"):
        _validate_cross_batch(batch)
