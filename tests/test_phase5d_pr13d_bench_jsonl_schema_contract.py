from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from scripts.bench_ablation_matrix import main as bench_main


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def test_pr13d_bench_jsonl_schema_contract(tmp_path: Path) -> None:
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

        assert row.get("schema_version") == "0.1"

        meta = row.get("meta") or {}
        assert isinstance(meta, dict)
        assert isinstance(meta.get("time_utc"), str)

        runtime = row.get("runtime") or {}
        assert isinstance(runtime, dict)
        assert _is_number(runtime.get("compile_first_run_ms"))
        assert float(runtime["compile_first_run_ms"]) >= 0.0
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

        corr: Dict[str, Any] = row.get("correctness") or {}
        assert isinstance(corr, dict)
        for k in (
            "python_vs_tvm_max_abs_diff_lb",
            "python_vs_tvm_max_abs_diff_ub",
            "python_vs_tvm_max_rel_diff_lb",
            "python_vs_tvm_max_rel_diff_ub",
        ):
            assert _is_number(corr.get(k))
            assert float(corr[k]) >= 0.0
