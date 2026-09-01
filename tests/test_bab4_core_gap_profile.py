"""BAB4 core-gap attribution tooling contracts."""

# pylint: disable=missing-function-docstring,protected-access

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.run_bab4_core_gap_profile import (
    FORMAL_ARTIFACT,
    unprofiled_optimizer_samples_ns,
)
from scripts import run_asplos27_s4_same_solver_worker as live_worker
from scripts import run_asplos27_s4_same_solver_five_fresh as formal_implementation
from scripts import run_bab4_gc_five_fresh as gc_formal
from scripts import run_bab4_rfactor_prepared_five_fresh as prepared
from scripts import run_bab4_rfactor_warm_five_fresh as warm


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


def test_warm_matched_protocol_primes_native_root_on_both_sides() -> None:
    for configuration in ("B4-A-WARM", "BAB4-WARM"):
        args = argparse.Namespace(
            configuration=configuration,
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
        assert namespace.prepare_static_request is True
        assert namespace.prepare_root_optimizer_warmup is True
    assert "B4-A-WARM" not in live_worker.CANDIDATE_CONFIGURATIONS
    assert "BAB4-WARM" in live_worker.CANDIDATE_CONFIGURATIONS
    assert "BAB4-WARM" in live_worker.FOUR_SEGMENT_CONFIGURATIONS

    warm.configure()
    assert warm.implementation._is_four_segment_candidate() is True
    assert len(warm.PAIR_ORDERS) == 5
    assert all(set(order) == {"B4-A-WARM", "BAB4-WARM"} for order in warm.PAIR_ORDERS)


def test_gc_protocol_is_symmetric_and_binds_prepared_gc_source() -> None:
    for configuration in ("B4-A-GC", "BAB4-GC"):
        args = argparse.Namespace(
            configuration=configuration,
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
        assert namespace.prepare_static_request is True
        assert namespace.prepare_root_optimizer_warmup is True
        assert namespace.prepare_gc_isolation is True

    gc_formal.configure()
    assert gc_formal.implementation._is_four_segment_candidate() is True
    assert len(gc_formal.PAIR_ORDERS) == 5
    assert all(set(order) == {"B4-A-GC", "BAB4-GC"} for order in gc_formal.PAIR_ORDERS)
    assert "boundflow/runtime/prepared_gc_isolation.py" in gc_formal.CODE_PATHS


def test_formal_gc_receipt_validation_is_fail_closed() -> None:
    receipt = {
        "schema_version": "boundflow.prepared-gc-isolation/v1",
        "full_prepare_collection": True,
        "prepared_old_generation_scan_excluded": True,
        "query_collection_preserved": True,
        "query_timing_excluded": True,
        "query_collect_generation": 1,
        "query_collect_call_count": 1,
        "restored": True,
        "performance_claimed": False,
        "prepare_collect_ns": 3,
        "query_collect_ns": 2,
        "restore_collect_ns": 4,
        "prepare_collected_object_count": 0,
        "query_collected_object_count": 0,
        "restore_collected_object_count": 0,
    }
    payload = {"diagnostics": {"prepared_gc_isolation": receipt}}
    formal_implementation._validate_prepared_gc_receipt(payload, "BAB4-GC")
    receipt["restored"] = False
    try:
        formal_implementation._validate_prepared_gc_receipt(payload, "BAB4-GC")
    except ValueError as error:
        assert "prepared GC receipt" in str(error)
    else:
        raise AssertionError("mutated prepared GC receipt was accepted")
