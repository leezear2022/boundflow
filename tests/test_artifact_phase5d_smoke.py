from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_phase5d_artifact import main as artifact_main


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        assert isinstance(obj, dict)
        rows.append(obj)
    return rows


def _manifest_claimed_paths(path: Path) -> list[Path]:
    text = path.read_text(encoding="utf-8").splitlines()
    in_section = False
    out: list[Path] = []
    for line in text:
        if line.strip() == "paper_facing_outputs:":
            in_section = True
            continue
        if in_section and line.strip() == "sha256:":
            break
        if not in_section:
            continue
        if not line.startswith("- "):
            continue
        if ":" not in line:
            continue
        _, rest = line.split(":", 1)
        p = rest.strip()
        if p:
            out.append(Path(p))
    return out


def test_phase5d_artifact_runner_quick_smoke(tmp_path: Path) -> None:
    pytest.importorskip("tvm")
    out_root = tmp_path / "artifacts"
    run_id = "smoke"

    rc = artifact_main(
        [
            "--mode",
            "quick",
            "--workload",
            "all",
            "--run-id",
            run_id,
            "--out-root",
            str(out_root),
            "--torch-num-threads",
            "1",
        ]
    )
    assert rc == 0

    out_dir = out_root / run_id
    assert out_dir.exists()

    jsonl = out_dir / "results.jsonl"
    assert jsonl.exists()
    rows = _read_jsonl(jsonl)
    assert len(rows) >= 2, "expected at least one JSONL row per workload"
    assert {r.get("workload", {}).get("model") for r in rows} >= {"mlp", "mnist_cnn"}
    for row in rows:
        for k in ("meta", "workload", "config", "planner", "tvm", "runtime", "baseline", "correctness"):
            assert k in row

    assert (out_dir / "results_flat.csv").exists()
    assert (out_dir / "tables" / "table_ablation.csv").exists()
    assert (out_dir / "tables" / "table_main.csv").exists()
    assert (out_dir / "MANIFEST.txt").exists()
    for p in _manifest_claimed_paths(out_dir / "MANIFEST.txt"):
        assert p.exists(), f"manifest claims missing path: {p}"
    assert (out_dir / "runs" / "mlp" / "results.jsonl").exists()
    assert (out_dir / "runs" / "mnist_cnn" / "results.jsonl").exists()

    # Claims doc is best-effort (depends on repo file), but should exist in normal repo state.
    assert (out_dir / "CLAIMS.md").exists()
    assert (out_dir / "APPENDIX.md").exists()


def test_phase5d_artifact_runner_allow_no_tvm_smoke(tmp_path: Path) -> None:
    # This smoke test is primarily for environments without TVM.
    try:
        import tvm  # noqa: F401

        pytest.skip("TVM is available; skip allow-no-tvm smoke to avoid duplicate compilation cost.")
    except Exception:
        pass

    out_root = tmp_path / "artifacts"
    run_id = "smoke_no_tvm"
    rc = artifact_main(
        [
            "--mode",
            "quick",
            "--allow-no-tvm",
            "--workload",
            "all",
            "--run-id",
            run_id,
            "--out-root",
            str(out_root),
            "--torch-num-threads",
            "1",
        ]
    )
    assert rc == 0

    out_dir = out_root / run_id
    assert (out_dir / "results.jsonl").exists()
    rows = _read_jsonl(out_dir / "results.jsonl")
    assert len(rows) >= 2
    assert {r.get("workload", {}).get("model") for r in rows} >= {"mlp", "mnist_cnn"}
    for row in rows:
        assert row.get("tvm", {}).get("available") is False
        assert row.get("status") in ("ok", "fail")

    assert (out_dir / "results_flat.csv").exists()
    assert (out_dir / "tables" / "table_main.csv").exists()
    assert (out_dir / "MANIFEST.txt").exists()
    for p in _manifest_claimed_paths(out_dir / "MANIFEST.txt"):
        assert p.exists(), f"manifest claims missing path: {p}"
    assert (out_dir / "CLAIMS.md").exists()
    assert (out_dir / "APPENDIX.md").exists()
    assert (out_dir / "runs" / "mlp" / "results.jsonl").exists()
    assert (out_dir / "runs" / "mnist_cnn" / "results.jsonl").exists()
