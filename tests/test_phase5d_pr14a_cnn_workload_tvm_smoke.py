from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.bench_ablation_matrix import main as bench_main


def test_pr14a_mnist_cnn_bench_tvm_smoke(tmp_path: Path) -> None:
    pytest.importorskip("tvm")
    out = tmp_path / "ablation.jsonl"
    rc = bench_main(
        [
            "--workload",
            "mnist_cnn",
            "--matrix",
            "small",
            "--warmup",
            "1",
            "--iters",
            "1",
            "--no-auto-lirpa",
            "--exit-nonzero-on-fail",
            "--output",
            str(out),
        ]
    )
    assert rc == 0
    line = out.read_text(encoding="utf-8").splitlines()[0]
    row = json.loads(line)
    assert row.get("status") == "ok"
    assert row.get("workload", {}).get("model") == "mnist_cnn"
    assert row.get("tvm", {}).get("available") is True
    # Ensure the timing fields are filled in TVM mode.
    assert isinstance(row.get("runtime", {}).get("compile_first_run_ms"), (int, float))
    assert isinstance(row.get("runtime", {}).get("run_ms_p50"), (int, float))

