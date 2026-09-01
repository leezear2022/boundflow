"""Runner-level contracts for the FSG4/B4-0 attribution artifact."""

# pylint: disable=missing-function-docstring,protected-access

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from boundflow.runtime.fsg4_b4_kernel_attribution import canonical_hash
from scripts import run_fsg4_b4_kernel_attribution as runner


def _semantics(lower_values: list[float]) -> dict[str, object]:
    return {
        "status": "verified",
        "success": True,
        "visited_domains": [6],
        "queue_before": 0,
        "queue_input": 6,
        "queue_accepted": 6,
        "queue_pruned": 0,
        "queue_after": 6,
        "depths": [1] * 6,
        "history_count": 6,
        "lower_shape": [6, 1],
        "lower_values": lower_values,
        "upper_shape": [6, 1],
        "upper_values": [0.0] * 6,
        "upper_positive_infinity_mask": [True] * 6,
        "final_decision": [[5, 27], [5, 32], [5, 90]] * 2,
        "split_depth": 1,
        "batch_size": 6,
        "n_verified": 0,
        "n_splits": 6,
    }


def test_gzip_jsonl_is_deterministic_and_binds_uncompressed_content(
    tmp_path: Path,
) -> None:
    rows = [{"ordinal": 0}, {"ordinal": 1}]
    first = tmp_path / "first.jsonl.gz"
    second = tmp_path / "second.jsonl.gz"
    first_content_hash = runner._write_jsonl_gzip(first, rows)
    second_content_hash = runner._write_jsonl_gzip(second, rows)
    observed, observed_content_hash = runner._load_jsonl_gzip(first)
    assert observed == rows
    assert observed_content_hash == first_content_hash == second_content_hash
    assert first.read_bytes() == second.read_bytes()


def test_worker_hash_rejects_unsigned_payload_drift() -> None:
    worker = {
        "schema_version": runner.WORKER_SCHEMA,
        "kind": "profile",
        "performance_claimed": False,
        "event_count": 1,
    }
    worker["worker_hash"] = canonical_hash(worker)
    runner._validate_worker(worker, kind="profile")
    worker["event_count"] = 2
    with pytest.raises(ValueError, match="worker binding"):
        runner._validate_worker(worker, kind="profile")


def test_log_sanitizer_removes_machine_paths() -> None:
    benchmark_root = Path("/home/lee/Codes/vnncomp2021")
    abcrown_root = Path("/home/lee/Codes/alpha-beta-CROWN")
    args = argparse.Namespace(
        benchmark_root=benchmark_root,
        abcrown_root=abcrown_root,
        abcrown_python=abcrown_root / ".venv/bin/python",
        model=benchmark_root / "bench/model.onnx",
        property=benchmark_root / "bench/property.vnnlib",
    )
    raw = (
        f"{args.abcrown_python} {args.model} {args.property} "
        f"{runner.REPOSITORY_ROOT} /tmp/private-run"
    )
    sanitized = runner._sanitize_text(raw, args)
    assert "/home/" not in sanitized
    assert "/tmp/" not in sanitized
    assert "$ABCROWN_PYTHON" in sanitized
    assert "$VNNCOMP_ROOT/bench/model.onnx" in sanitized
    assert "$TMP/private-run" in sanitized


def test_worker_interpreter_path_preserves_virtualenv_symlink(tmp_path: Path) -> None:
    target = tmp_path / "base-python"
    target.touch()
    virtualenv_python = tmp_path / "venv-python"
    virtualenv_python.symlink_to(target)
    observed = runner._absolute_without_symlink_resolution(virtualenv_python)
    assert observed == virtualenv_python.absolute()
    assert observed != virtualenv_python.resolve()


def test_semantic_pair_uses_frozen_b3_tolerance_and_exact_discrete_state() -> None:
    control = _semantics([-0.4] * 6)
    profiled = _semantics([-0.400001] * 6)
    report = runner._semantic_pair_report(control, profiled)
    assert report["passed"] is True
    assert report["discrete_exact"] is True
    assert report["lower_sign_exact"] is True
    assert report["lower_max_abs_diff"] == pytest.approx(1.0e-6)
    profiled["queue_after"] = 5
    with pytest.raises(ValueError, match="queue_after:differs"):
        runner._semantic_pair_report(control, profiled)


def test_semantic_pair_rejects_sign_drift_inside_numeric_tolerance() -> None:
    control = _semantics([-1.0e-8] * 6)
    profiled = _semantics([1.0e-8] * 6)
    with pytest.raises(ValueError, match="lower sign differs"):
        runner._semantic_pair_report(control, profiled)


def test_tamper_probe_set_is_fixed() -> None:
    assert [label for label, _ in runner._tamper_mutations()] == [
        "marker-count",
        "raw-phase",
        "raw-ordinal",
        "raw-duration",
        "raw-delete",
        "semantic-lower",
        "protocol-code",
        "worker-kind",
        "summary-opportunity",
    ]
