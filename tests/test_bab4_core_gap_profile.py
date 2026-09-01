"""BAB4 core-gap attribution tooling contracts."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.run_bab4_core_gap_profile import (
    FORMAL_ARTIFACT,
    unprofiled_optimizer_samples_ns,
)
from scripts import run_asplos27_s4_same_solver_worker as live_worker
from scripts import run_bab4_rfactor_prepared_five_fresh as prepared


def test_frozen_bab4_optimizer_samples_are_five_admitted_live_runs() -> None:
    values = unprofiled_optimizer_samples_ns()
    assert len(values) == 5
    assert min(values) > 0
    summary = json.loads((FORMAL_ARTIFACT / "summary.json").read_text(encoding="utf-8"))
    assert summary["pair_count"] == len(values)
    assert summary["candidate_configuration"] == "BAB4"
    assert summary["performance_claimed"] is False


def test_matched_preparation_control_moves_only_static_request_outside_query() -> None:
    args = argparse.Namespace(
        configuration="B4-A-PREP",
        mode="control",
        run_id="test",
        block_index=0,
        sequence_position=0,
        benchmark_root=Path("benchmark"),
        abcrown_root=Path("abcrown"),
        model=Path("model.onnx"),
        property=Path("property.vnnlib"),
        attribute_root_incomplete=False,
    )
    namespace = live_worker._base_namespace(args, Path("result.json"))
    assert namespace.configuration == "B4-A"
    assert namespace.prepare_static_request is True
    assert namespace.prepare_root_optimizer_warmup is False


def test_rfactor_protocol_uses_matched_preparation_and_five_alternating_pairs() -> None:
    prepared.configure()
    assert prepared.CONTROL_CONFIGURATION == "B4-A-PREP"
    assert prepared.CANDIDATE_CONFIGURATION == "BAB4"
    assert len(prepared.PAIR_ORDERS) == 5
    assert all(set(order) == {"B4-A-PREP", "BAB4"} for order in prepared.PAIR_ORDERS)
