from pathlib import Path


def test_phase6h_artifact_runner_smoke(tmp_path: Path) -> None:
    import subprocess

    out_dir = tmp_path / "artifact"
    subprocess.check_call(["bash", "scripts/run_phase6h_artifact.sh", str(out_dir)])

    assert (out_dir / "phase6h_e2e.jsonl").exists()
    assert (out_dir / "phase6h_e2e.csv").exists()
    assert (out_dir / "phase6h_e2e_summary.md").exists()
    assert (out_dir / "env.txt").exists()
    assert (out_dir / "pip_freeze.txt").exists()
    assert (out_dir / "conda_list.txt").exists()

    assert (out_dir / "phase6h_e2e.jsonl").stat().st_size > 0
    assert (out_dir / "phase6h_e2e.csv").stat().st_size > 0
    assert (out_dir / "phase6h_e2e_summary.md").stat().st_size > 0
