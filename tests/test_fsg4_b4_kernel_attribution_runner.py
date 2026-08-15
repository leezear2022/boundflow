"""Runner-level contracts for the FSG4/B4-0 attribution artifact."""

# pylint: disable=missing-function-docstring,protected-access

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from boundflow.runtime.fsg4_b4_kernel_attribution import canonical_hash
from scripts import run_fsg4_b4_kernel_attribution as runner


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
