"""Frozen NRIR-42 Phase-A/Phase-B replay and synchronized tamper gates."""

# pylint: disable=missing-function-docstring,protected-access,duplicate-code

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.run_objective_branch_scorer_ownership_formal import (
    _canonical_hash as phase_a_hash,
    _plan_from_dict,
    _row_from_dict,
    _validate_serialized_capsule,
    _validate_worker as validate_phase_a_worker,
    validate_formal as validate_phase_a,
)
from scripts.run_objective_branch_scorer_ownership_global_formal import (
    _canonical_hash as phase_b_hash,
    _semantic_worker as phase_b_worker_semantics,
    validate_formal as validate_phase_b,
)

ROOT = Path(__file__).resolve().parents[1]
PHASE_A = (
    ROOT
    / "artifacts/objective-branch-scorer-ownership"
    / "vnncomp21-resnet2b-property0-three-repeat-cpu-v1/formal.json"
)
PHASE_B = (
    ROOT
    / "artifacts/objective-branch-scorer-ownership-global"
    / "vnncomp21-resnet2b-property0-three-repeat-cpu-formal-v1/formal.json"
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_frozen_phase_a_passes_exact_ownership_and_cost_gates() -> None:
    formal = _load(PHASE_A)
    validate_phase_a(formal)

    assert formal["formal_hash"] == (
        "0d310c2ffc96844648a83f9921bc7f353ec8425986bccb36f75e6d1cd2b25b58"
    )
    assert formal["decision"]["phase_a_go"] is True
    assert formal["decision"]["next_route"] == "run_phase_b_global_60s"
    assert [
        metric["median_ratio"] for metric in formal["decision"]["clause_metrics"]
    ] == [0.7068884167986379, 0.6984858551708761]
    rows = formal["paired_rows"]
    assert all(
        row["enumeration_call_count"] == (341 if row["mode"] == "historical" else 31)
        for row in rows
    )
    assert all(parity["exact"] is True for parity in formal["parities"])


def test_synchronized_capsule_token_tamper_is_rejected_by_typed_replay() -> None:
    formal = _load(PHASE_A)
    raw = next(
        item
        for item in formal["workers"][0]["raw_runs"]
        if item["row"]["mode"] == "prevalidated"
    )
    capsule = deepcopy(raw["capsules"][0])
    capsule["capsule"]["candidate_source"] = "forged_reenumeration"
    capsule["capsule"]["semantic_token"] = phase_a_hash(
        {
            key: value
            for key, value in capsule["capsule"].items()
            if key != "semantic_token"
        }
    )

    with pytest.raises(ValueError, match="capsule"):
        _validate_serialized_capsule(capsule, raw["branch_semantics"][0])


def test_synchronized_score_and_call_count_tamper_are_rejected() -> None:
    formal = _load(PHASE_A)
    worker = deepcopy(formal["workers"][0])
    raw = worker["raw_runs"][0]
    score = raw["branch_semantics"][0]["scores"][0]
    score["inactive_lower"] += 0.25
    raw["branch_semantics"][0]["score_hash"] = phase_a_hash(
        raw["branch_semantics"][0]["scores"]
    )
    raw["row"]["branch_semantic_hash"] = phase_a_hash(raw["branch_semantics"])
    raw["row_hash"] = _row_from_dict(raw["row"]).stable_hash()
    raw["raw_hash"] = phase_a_hash(
        {key: value for key, value in raw.items() if key != "raw_hash"}
    )
    worker["worker_hash"] = phase_a_hash(
        {key: value for key, value in worker.items() if key != "worker_hash"}
    )
    with pytest.raises(ValueError, match="score semantic"):
        validate_phase_a_worker(worker, _row_plan(formal))

    row = deepcopy(formal["paired_rows"][1])
    assert row["mode"] == "prevalidated"
    row["enumeration_call_count"] = 30
    with pytest.raises(ValueError, match="enumeration ownership"):
        _row_from_dict(row).validate()


def _row_plan(formal: dict[str, Any]):
    return _plan_from_dict(formal["plan"])


def test_frozen_phase_b_recovers_all_global_60s_production_coverage() -> None:
    formal = _load(PHASE_B)
    validate_phase_b(formal)
    payload = formal["formal_payload"]

    assert formal["formal_payload_hash"] == (
        "7274e834b3bf08a9e138fa3284b70222620cf3c571395331e1a87ed5fee7d759"
    )
    assert formal["status"] == "validated-reduced"
    assert payload["all_correctness_gates_passed"] is True
    assert payload["all_production_gates_passed"] is True
    assert payload["selected_original_clause_indices"] == [[2, 3]] * 3
    assert payload["accepted_nodes"] == [[31, 31]] * 3
    assert all(value <= 70_000_000_000 for value in payload["whole_elapsed_ns"])
    assert all(
        packed["capsule_count"] == 31
        and packed["compile_enumeration_count"] == 31
        and packed["execute_enumeration_count"] == 0
        for result in payload["repeat_results"]
        for packed in (item["packed"] for item in result["slices"])
    )


def test_rehashed_phase_b_deadline_tamper_cannot_keep_production_pass() -> None:
    formal = deepcopy(_load(PHASE_B))
    result = formal["formal_payload"]["repeat_results"][0]
    result["runtime_trace"]["elapsed_ns"] = 70_000_000_001
    result["result_hash"] = phase_b_hash(phase_b_worker_semantics(result))
    formal["formal_payload"]["whole_elapsed_ns"][0] = 70_000_000_001
    formal["formal_payload_hash"] = phase_b_hash(formal["formal_payload"])

    with pytest.raises(ValueError, match="worker result"):
        validate_phase_b(formal)
