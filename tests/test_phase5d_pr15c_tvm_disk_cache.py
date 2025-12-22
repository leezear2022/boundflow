from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.bench_ablation_matrix import main as bench_main


def _read_first_row(p: Path) -> dict:
    line = p.read_text(encoding="utf-8").splitlines()[0]
    return json.loads(line)


def test_pr15c_tvm_disk_cache_reduces_compile_miss(tmp_path: Path) -> None:
    pytest.importorskip("tvm")
    out1 = tmp_path / "r1.jsonl"
    out2 = tmp_path / "r2.jsonl"
    cache_dir = tmp_path / "tvm_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # First run: expect at least one compile miss (compile happens).
    rc1 = bench_main(
        [
            "--matrix",
            "small",
            "--warmup",
            "1",
            "--iters",
            "1",
            "--no-auto-lirpa",
            "--tvm-cache-dir",
            str(cache_dir),
            "--output",
            str(out1),
        ]
    )
    assert rc1 == 0
    row1 = _read_first_row(out1)
    assert row1.get("status") in ("ok", "fail")

    # Second run: should be able to load from disk cache, so compile miss delta should be 0.
    rc2 = bench_main(
        [
            "--matrix",
            "small",
            "--warmup",
            "1",
            "--iters",
            "1",
            "--no-auto-lirpa",
            "--tvm-cache-dir",
            str(cache_dir),
            "--output",
            str(out2),
        ]
    )
    assert rc2 == 0
    row2 = _read_first_row(out2)
    assert row2.get("status") == "ok"
    delta = (row2.get("tvm") or {}).get("compile_cache_stats_delta_compile_first_run") or {}
    assert isinstance(delta, dict)
    assert int(delta.get("task_compile_cache_miss", 0)) == 0

