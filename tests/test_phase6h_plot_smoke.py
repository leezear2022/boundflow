from pathlib import Path

import pytest


def test_phase6h_plot_smoke(tmp_path: Path) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    _ = matplotlib

    import subprocess
    import sys
    import json

    repo_root = Path(__file__).resolve().parents[1]
    jsonl = tmp_path / "in.jsonl"
    out_dir = tmp_path / "figs"

    out = subprocess.check_output(
        [
            sys.executable,
            "scripts/bench_phase6h_bab_e2e_time_to_verify.py",
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--workload",
            "1d_relu",
            "--oracle",
            "alpha_beta",
            "--steps",
            "0",
            "--max-nodes",
            "64",
            "--node-batch-size",
            "8",
            "--warmup",
            "1",
            "--iters",
            "1",
            "--enable-node-eval-cache",
            "0,1",
            "--use-branch-hint",
            "0,1",
            "--enable-batch-infeasible-prune",
            "0,1",
        ],
        cwd=str(repo_root),
    ).decode("utf-8")
    payload = json.loads(out)
    jsonl.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    subprocess.check_call(
        [sys.executable, "scripts/plot_phase6h_e2e.py", "--in-jsonl", str(jsonl), "--out-dir", str(out_dir)],
        cwd=str(repo_root),
    )
    assert out_dir.exists()
    pngs = list(out_dir.glob("*.png"))
    assert len(pngs) >= 3
    assert all(p.stat().st_size > 0 for p in pngs[:3])

