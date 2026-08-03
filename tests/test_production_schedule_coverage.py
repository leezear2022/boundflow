"""Production Schedule IR ownership and memory P0 gate tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from boundflow.planner.production_schedule_coverage import (
    generate_production_schedule_coverage_artifact,
    replay_production_schedule_coverage_artifact,
)

ROOT = Path(__file__).resolve().parents[1]
IR5 = ROOT / "artifacts/ir5/residual-final-v3-20260728"
RVIR = ROOT / "artifacts/rvir/rvir-cpu-correctness-v2-20260803"


def test_p0_gate_distinguishes_schedule_ownership_from_production_admission(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "p0"
    coverage = generate_production_schedule_coverage_artifact(
        artifact_dir,
        ir5_artifact_dir=IR5,
        rvir_artifact_dir=RVIR,
    )

    assert coverage["verdict"] == "NO_GO"
    assert coverage["gates"]["residual_schedule_owns_full_bound_graph"] is True
    assert coverage["gates"]["residual_schedule_owns_arena_lifecycle"] is True
    assert (
        coverage["gates"]["residual_schedule_exercises_materialization_transition"]
        is False
    )
    assert coverage["gates"]["residual_template_has_storage_choice"] is False
    assert coverage["gates"]["real_resnet_is_native_multi_region_bound_ir"] is False
    assert coverage["gates"]["current_plan_decisions_change_with_budget"] is False
    assert coverage["real_resnet_schedule_path"]["activation_call_count"] == 51
    assert (
        coverage["real_resnet_schedule_path"]["representative"][
            "schedule_action_counts"
        ]["launch"]
        == 1
    )
    assert all(
        case["budget_probe"]["plan_instance_hashes_differ"]
        and not case["budget_probe"]["decisions_switched"]
        and case["budget_probe"]["below_peak_rejected"]
        for case in coverage["residual_schedule_path"]["cases"]
    )
    assert (
        replay_production_schedule_coverage_artifact(
            artifact_dir,
            ir5_artifact_dir=IR5,
            rvir_artifact_dir=RVIR,
        )
        == coverage
    )


def test_p0_replay_rejects_semantic_tamper_even_with_updated_digest(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "p0"
    generate_production_schedule_coverage_artifact(
        artifact_dir,
        ir5_artifact_dir=IR5,
        rvir_artifact_dir=RVIR,
    )
    coverage_path = artifact_dir / "coverage.json"
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    coverage["verdict"] = "GO"
    coverage_path.write_text(
        json.dumps(coverage, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["coverage.json"] = hashlib.sha256(
        coverage_path.read_bytes()
    ).hexdigest()
    manifest["verdict"] = "GO"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="semantic replay mismatch"):
        replay_production_schedule_coverage_artifact(
            artifact_dir,
            ir5_artifact_dir=IR5,
            rvir_artifact_dir=RVIR,
        )
