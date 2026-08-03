"""Frozen NRIR-2 artifact contract and synchronized semantic tamper gates."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.run_native_real_network_memory_plans_artifact import (
    MEMORY_ARTIFACT_SCHEMA_VERSION,
    validate_memory_plan_evidence,
)

ARTIFACT_DIR = (
    Path(__file__).resolve().parents[1] / "artifacts/native-real-network-memory-plans/"
    "vnncomp21-resnet2b-prop0-cpu-v1"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evidence() -> dict:
    return json.loads((ARTIFACT_DIR / "plans.json").read_text(encoding="utf-8"))


def test_frozen_native_memory_artifact_has_exact_budget_switch_contract() -> None:
    manifest = json.loads((ARTIFACT_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == MEMORY_ARTIFACT_SCHEMA_VERSION
    assert manifest["status"] == "ok"
    assert manifest["performance_claimed"] is False
    assert manifest["files"] == {"plans.json": _sha256(ARTIFACT_DIR / "plans.json")}

    evidence = _evidence()
    validate_memory_plan_evidence(evidence)
    assert evidence["retain_all"]["schedule_peak_bytes"] == 1_860_912
    assert evidence["lifetime_reuse"]["schedule_peak_bytes"] == 442_656
    assert evidence["lifetime_reuse"]["early_release_count"] == 85
    assert evidence["lifetime_reuse"]["physical_alias_pair_count"] == 386
    assert evidence["semantics"]["retain_vs_external"]["max_abs_diff"] == (
        7.152557373046875e-07
    )


def test_native_memory_evidence_rejects_rehashed_semantic_tamper() -> None:
    tampered = deepcopy(_evidence())
    tampered["lifetime_reuse"]["early_release_count"] = 0
    tampered["gates"]["runtime_releases_before_final_task"] = True
    with pytest.raises(ValueError, match="lacks aliases or early release"):
        validate_memory_plan_evidence(tampered)
