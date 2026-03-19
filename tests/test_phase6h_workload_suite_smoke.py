import json
from pathlib import Path


def _run(workload: str) -> None:
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    out = subprocess.check_output(
        [
            sys.executable,
            "scripts/bench_phase6h_bab_e2e_time_to_verify.py",
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--workload",
            workload,
            "--oracle",
            "alpha_beta",
            "--steps",
            "0",
            "--max-nodes",
            "32",
            "--node-batch-size",
            "8",
            "--warmup",
            "1",
            "--iters",
            "1",
            "--enable-node-eval-cache",
            "0",
            "--use-branch-hint",
            "1",
            "--enable-batch-infeasible-prune",
            "0",
        ],
        cwd=str(repo_root),
    ).decode("utf-8")
    payload = json.loads(out)
    assert "meta" in payload and "rows" in payload
    assert payload["meta"]["workload"] == workload
    assert len(payload["rows"]) == 1


def test_phase6h_workload_suite_smoke() -> None:
    for w in ("1d_relu", "3dir_l2", "mlp2d_2x16", "mlp3d_3x32"):
        _run(w)

