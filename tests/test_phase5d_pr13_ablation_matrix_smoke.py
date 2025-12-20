import json

from scripts.bench_ablation_matrix import main as bench_main


def test_pr13_ablation_matrix_smoke_jsonl():
    # Keep it tiny: one config point, no auto_LiRPA baseline.
    # This test is primarily to ensure the bench script stays runnable and emits JSONL.
    out_lines = []

    def _capture_stdout_write(s: str) -> None:
        out_lines.append(s)

    import sys

    old = sys.stdout
    try:
        sys.stdout = type("W", (), {"write": staticmethod(_capture_stdout_write)})()  # type: ignore[assignment]
        rc = bench_main(["--matrix", "small", "--warmup", "1", "--iters", "1", "--no-auto-lirpa", "--no-check"])
    finally:
        sys.stdout = old  # type: ignore[assignment]

    assert rc == 0
    text = "".join(out_lines).strip()
    assert text

    # JSONL: one line in small matrix.
    line = text.splitlines()[0]
    row = json.loads(line)
    assert "meta" in row
    assert "planner" in row
    assert "tvm" in row
    assert "runtime" in row
