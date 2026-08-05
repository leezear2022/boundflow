"""Frozen NRIR44 Phase-A/Phase-B replay and tamper tests."""

# pylint: disable=missing-function-docstring,protected-access,duplicate-code

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.run_root_projection_floor_formal import (
    _canonical_hash,
    _validate_projection_payload,
    validate_formal as validate_phase_a,
    validate_worker as validate_phase_a_worker,
)
from scripts.run_root_projection_floor_global_formal import (
    _semantic_worker,
    _validate_worker as validate_phase_b_worker,
    validate_formal as validate_phase_b,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/root-projection-floor"
PHASE_A = ARTIFACT_ROOT / ("vnncomp21-resnet2b-property0-three-repeat-cpu-phase-a-v1")
PHASE_B = ARTIFACT_ROOT / ("vnncomp21-resnet2b-property0-three-repeat-cpu-phase-b-v1")
BENCHMARK_ROOT = Path("/tmp/boundflow-vnncomp2021-nrir43")


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_nrir44_phase_a_formal_closes_projection_gate() -> None:
    formal = _load(PHASE_A / "formal.json")

    validate_phase_a(formal)
    assert formal["status"] == "validated-reduced"
    assert formal["decision"]["phase_a_go"] is True
    assert formal["decision"]["next_route"] == "run_nrir44_phase_b"
    assert formal["metrics"]["maximum_projected_repeat_ns"] <= 11_000_000_000
    assert formal["metrics"]["projected_to_nrir42_median_ratio"] <= 0.50
    assert formal["performance_claimed"] is False


def test_nrir44_phase_b_formal_closes_fixed_resnet_admission() -> None:
    formal = _load(PHASE_B / "formal.json")

    validate_phase_b(formal)
    payload = formal["formal_payload"]
    assert formal["status"] == "validated-reduced"
    assert payload["timing_gate_passed"] is True
    assert payload["projected_to_nrir42_median_ratio"] <= 0.82
    assert payload["accepted_nodes"] == [[31, 31]] * 3
    assert (
        payload["worst_active_lowers"]
        == [[-35.53092575073242, -30.258447647094727]] * 3
    )
    assert formal["performance_claimed"] is False


def test_nrir44_replay_commands_pass() -> None:
    if not BENCHMARK_ROOT.exists():
        pytest.skip("frozen VNN-COMP checkout is unavailable")
    for script in (
        "scripts/run_root_projection_floor_formal.py",
        "scripts/run_root_projection_floor_global_formal.py",
    ):
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / script),
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
        assert "validated-reduced" in completed.stdout


def test_nrir44_synchronized_outer_rehash_does_not_mask_budget_tamper() -> None:
    worker = _load(PHASE_A / "shards/repeat-0.json")
    projected = next(row for row in worker["rows"] if row["mode"] == "projected")
    projected["objective_evaluation_count"] = 10
    projected["row_hash"] = _canonical_hash(
        {key: value for key, value in projected.items() if key != "row_hash"}
    )
    worker["worker_hash"] = _canonical_hash(
        {key: value for key, value in worker.items() if key != "worker_hash"}
    )

    with pytest.raises(ValueError, match="evidence row differs"):
        validate_phase_a_worker(worker, repeat_index=0)


def test_nrir44_typed_projection_rejects_consumer_tamper() -> None:
    worker = _load(PHASE_A / "shards/repeat-0.json")
    projected = next(row for row in worker["rows"] if row["mode"] == "projected")
    payload = deepcopy(projected["projection"])
    payload["plan"]["consumed_result_fields"][0] = "deep_queue.lower"

    with pytest.raises(ValueError, match="Plan IR is invalid"):
        _validate_projection_payload(payload)


def test_nrir44_synchronized_outer_rehash_does_not_mask_whole_time_tamper() -> None:
    worker = _load(PHASE_B / "shards/repeat-0.json")
    worker["runtime_trace"]["elapsed_ns"] = 48_000_000_001
    worker["result_hash"] = _canonical_hash(_semantic_worker(worker))

    with pytest.raises(ValueError, match="worker result differs"):
        validate_phase_b_worker(worker)
