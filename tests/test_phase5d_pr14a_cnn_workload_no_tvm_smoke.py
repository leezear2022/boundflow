from __future__ import annotations

import json
from pathlib import Path

from scripts.bench_ablation_matrix import main as bench_main


def test_pr14a_mnist_cnn_bench_no_tvm_smoke(tmp_path: Path) -> None:
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
            "--no-tvm",
            "--output",
            str(out),
        ]
    )
    assert rc == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines
    row = json.loads(lines[0])
    assert row.get("status") in ("ok", "fail")
    assert row.get("workload", {}).get("model") == "mnist_cnn"
    assert row.get("tvm", {}).get("available") is False

