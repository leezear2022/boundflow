from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from scripts.bench_ablation_matrix import main as bench_main


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def test_pr13d_bench_jsonl_schema_contract(tmp_path: Path) -> None:
    pytest.importorskip("tvm")
    out = tmp_path / "ablation.jsonl"

    # Keep it tiny: one config point, no auto_LiRPA baseline. Enable correctness to ensure diff fields exist.
    rc = bench_main(
        [
            "--matrix",
            "small",
            "--warmup",
            "1",
            "--iters",
            "1",
            "--no-auto-lirpa",
            "--output",
            str(out),
        ]
    )
    assert rc == 0
    assert out.exists()

    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines, "expected at least one JSONL row"

    for i, line in enumerate(lines, 1):
        row = json.loads(line)
        assert isinstance(row, dict), f"line {i} is not a JSON object"

        assert row.get("schema_version") == "1.0"
        assert row.get("status") == "ok"
        assert row.get("error") is None

        meta = row.get("meta") or {}
        assert isinstance(meta, dict)
        assert isinstance(meta.get("time_utc"), str)
        assert isinstance(meta.get("env_flags"), dict)
        assert isinstance(meta.get("device"), dict)

        runtime = row.get("runtime") or {}
        assert isinstance(runtime, dict)
        assert _is_number(runtime.get("compile_first_run_ms"))
        assert float(runtime["compile_first_run_ms"]) >= 0.0
        assert _is_number(runtime.get("run_ms_cold"))
        assert float(runtime["run_ms_cold"]) >= 0.0
        for k in ("run_ms_avg", "run_ms_p50", "run_ms_p95"):
            assert _is_number(runtime.get(k))
            assert float(runtime[k]) >= 0.0

        tvm = row.get("tvm") or {}
        assert isinstance(tvm, dict)
        delta = tvm.get("compile_cache_stats_delta_compile_first_run") or {}
        assert isinstance(delta, dict)
        for k in ("task_compile_cache_hit", "task_compile_cache_miss", "task_compile_fail"):
            assert isinstance(delta.get(k), int)
            assert int(delta[k]) >= 0
        assert isinstance(tvm.get("compile_cache_tag"), str)
        assert isinstance(tvm.get("compile_keyset_size"), int)
        assert int(tvm["compile_keyset_size"]) >= 0
        assert isinstance(tvm.get("compile_keyset_digest"), str)
        assert tvm["compile_keyset_digest"], "expected non-empty compile_keyset_digest"

        corr: Dict[str, Any] = row.get("correctness") or {}
        assert isinstance(corr, dict)
        gate = corr.get("python_vs_tvm_gate") or {}
        assert isinstance(gate, dict)
        assert gate.get("ref") == "python_task_executor"
        assert isinstance(gate.get("ok"), bool)
        tol = gate.get("tol") or {}
        assert isinstance(tol, dict)
        assert _is_number(tol.get("atol"))
        assert _is_number(tol.get("rtol"))
        for k in (
            "python_vs_tvm_max_abs_diff_lb",
            "python_vs_tvm_max_abs_diff_ub",
            "python_vs_tvm_max_rel_diff_lb",
            "python_vs_tvm_max_rel_diff_ub",
        ):
            assert _is_number(corr.get(k))
            assert float(corr[k]) >= 0.0


def test_pr13d_bench_jsonl_schema_contract_no_check_still_has_keys(tmp_path: Path) -> None:
    pytest.importorskip("tvm")
    out = tmp_path / "ablation.jsonl"
    rc = bench_main(
        [
            "--matrix",
            "small",
            "--warmup",
            "1",
            "--iters",
            "1",
            "--no-auto-lirpa",
            "--no-check",
            "--output",
            str(out),
        ]
    )
    assert rc == 0
    line = out.read_text(encoding="utf-8").splitlines()[0]
    row = json.loads(line)
    corr = row.get("correctness") or {}
    for k in (
        "python_vs_tvm_max_abs_diff_lb",
        "python_vs_tvm_max_abs_diff_ub",
        "python_vs_tvm_max_rel_diff_lb",
        "python_vs_tvm_max_rel_diff_ub",
    ):
        assert k in corr
        assert corr[k] is None


def test_pr13d_bench_jsonl_schema_contract_no_tvm_mode_still_writes_rows(tmp_path: Path) -> None:
    out = tmp_path / "ablation.jsonl"
    rc = bench_main(
        [
            "--matrix",
            "small",
            "--warmup",
            "1",
            "--iters",
            "1",
            "--no-auto-lirpa",
            "--no-tvm",
            "--output",
            str(out),
        ]
    )
    assert rc == 0
    line = out.read_text(encoding="utf-8").splitlines()[0]
    row = json.loads(line)
    assert row.get("schema_version") == "1.0"
    assert row.get("status") in ("ok", "fail")
    assert row.get("tvm", {}).get("available") is False
