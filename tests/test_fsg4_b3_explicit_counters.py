"""Contracts for the FSG4/B3 explicit-counter diagnostic."""

# pylint: disable=missing-function-docstring,protected-access

import copy

import pytest

from boundflow.runtime.fsg4_b3_explicit_counters import (
    COUNTER_NAMES,
    EXPECTED_B2_FIXED_COUNTERS,
    events_from_rows,
    Fsg4B3CounterRecorder,
    Fsg4B3CounterSnapshot,
    fsg4_b3_counter_snapshot_from_dict,
)
from scripts import run_fsg4_b3_counter_diagnostic as diagnostic
from scripts import probe_fsg4_b3_counter_artifact_tamper as tamper
from scripts import run_rvir_v4_live_return_capture as live_runner


def _complete_recorder() -> Fsg4B3CounterRecorder:
    recorder = Fsg4B3CounterRecorder()
    for name in COUNTER_NAMES:
        value = EXPECTED_B2_FIXED_COUNTERS.get(name, 1)
        if value:
            recorder.add(name, amount=value, detail=f"test {name}")
    return recorder


def _snapshot(recorder: Fsg4B3CounterRecorder) -> Fsg4B3CounterSnapshot:
    return Fsg4B3CounterSnapshot(
        counts_by_name=tuple(sorted(recorder.counts().items())),
        semantic_hash="1" * 64,
        worker_result_sha256="2" * 64,
        provider_core_call_count=0,
        provider_compute_bounds_call_count=0,
        provider_update_bounds_call_count=0,
        fallback_dispatch_count=0,
        environment_admitted=True,
    )


def test_event_journal_rebuilds_all_counters() -> None:
    recorder = _complete_recorder()
    rows = [event.to_dict() for event in recorder.events]
    events = events_from_rows(rows)
    rebuilt = Fsg4B3CounterRecorder(events=list(events))
    assert rebuilt.counts() == recorder.counts()
    assert set(rebuilt.counts()) == set(COUNTER_NAMES)


def test_counter_snapshot_round_trips_and_hash_binds() -> None:
    payload = _snapshot(_complete_recorder()).to_dict()
    restored = fsg4_b3_counter_snapshot_from_dict(payload)
    assert restored.counts == _complete_recorder().counts()
    tampered = copy.deepcopy(payload)
    tampered["counts"]["forward_trace_build_count"] = 4  # type: ignore[index]
    with pytest.raises(ValueError, match="hash differs"):
        fsg4_b3_counter_snapshot_from_dict(tampered)


@pytest.mark.parametrize(
    ("counter", "replacement"),
    (
        ("forward_trace_build_count", 4),
        ("optimizer_evaluation_count", 9),
        ("optimizer_update_count", 10),
        ("timed_candidate_d2h_copy_count", 11),
        ("commit_copy_call_count", 13),
    ),
)
def test_fixed_b2_counter_mismatch_fails_closed(counter: str, replacement: int) -> None:
    counts = _complete_recorder().counts()
    counts[counter] = replacement
    snapshot = Fsg4B3CounterSnapshot(
        counts_by_name=tuple(sorted(counts.items())),
        semantic_hash="1" * 64,
        worker_result_sha256="2" * 64,
        provider_core_call_count=0,
        provider_compute_bounds_call_count=0,
        provider_update_bounds_call_count=0,
        fallback_dispatch_count=0,
        environment_admitted=True,
    )
    with pytest.raises(ValueError, match="gate failed"):
        snapshot.validate()
    assert any(counter in failure for failure in snapshot.gate_failures())


def test_profiler_callback_is_not_part_of_counter_contract() -> None:
    assert "profile_callback_count" not in COUNTER_NAMES
    assert "provider_core_call_count" not in COUNTER_NAMES


def test_named_instrumentation_patches_and_restores_without_profiler() -> None:
    recorder = Fsg4B3CounterRecorder()
    original = live_runner._move_tensors
    with diagnostic._instrument_b2(recorder):
        assert live_runner._move_tensors is not original
    assert live_runner._move_tensors is original
    assert not recorder.events


def test_boolean_counter_is_rejected_instead_of_coerced() -> None:
    payload = _snapshot(_complete_recorder()).to_dict()
    payload["counts"]["forward_trace_build_count"] = True  # type: ignore[index]
    snapshot_payload = dict(payload)
    snapshot_payload.pop("snapshot_hash")
    payload["snapshot_hash"] = diagnostic.canonical_hash(snapshot_payload)
    with pytest.raises(TypeError, match="must be an integer"):
        fsg4_b3_counter_snapshot_from_dict(payload)


def test_tamper_probe_covers_outer_resigned_semantics_and_counters() -> None:
    assert {name for name, _attack in tamper.ATTACKS} == {
        "counter-report-only-outer-resign",
        "counter-and-journal-outer-resign",
        "delete-journal-event-outer-resign",
        "worker-semantic-outer-resign",
        "provider-count-outer-resign",
        "code-revision-outer-resign",
    }
