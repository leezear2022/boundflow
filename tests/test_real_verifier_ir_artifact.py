"""Fresh-process replay contract for the frozen real-verifier IR artifact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def _replay(artifact: Path) -> dict[str, object]:
    """Replay one committed artifact in a fresh process."""

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_real_verifier_ir_artifact.py",
            "replay",
            "--artifact-dir",
            str(artifact),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _replay_failure(artifact: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "scripts/run_real_verifier_ir_artifact.py",
            "replay",
            "--artifact-dir",
            str(artifact),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_frozen_real_verifier_ir_v2_artifact_replays_online_raw_rows() -> None:
    """V2 must replay 394 admissions plus all 377 online query/record rows."""

    artifact = Path("artifacts/rvir/rvir-cpu-correctness-v2-20260803")
    assert _replay(artifact) == {
        "activation_call_count": 394,
        "performance_claimed": False,
        "status": "replayed",
    }
    assert (artifact / "online_queries.jsonl").read_text(encoding="utf-8").count(
        "\n"
    ) == 377
    assert (artifact / "online_typed_ir.jsonl").read_text(encoding="utf-8").count(
        "\n"
    ) == 377


def test_frozen_real_verifier_ir_v1_artifact_remains_replayable() -> None:
    """The audited v1 artifact remains a supported immutable historical record."""

    artifact = Path("artifacts/rvir/rvir-cpu-correctness-v1-20260803")
    assert _replay(artifact) == {
        "activation_call_count": 394,
        "performance_claimed": False,
        "status": "replayed",
    }


def test_v2_replay_rejects_rehashed_online_ir_record_tamper(tmp_path: Path) -> None:
    """Semantic replay must reject a forged IR hash even after digest rewrites."""

    artifact = tmp_path / "artifact"
    shutil.copytree("artifacts/rvir/rvir-cpu-correctness-v2-20260803", artifact)
    records_path = artifact / "online_typed_ir.jsonl"
    record_lines = records_path.read_text(encoding="utf-8").splitlines()
    first_record = json.loads(record_lines[0])
    first_record["ir_hashes"]["schedule_hash"] = "0" * 64
    record_lines[0] = json.dumps(first_record, sort_keys=True, allow_nan=False)
    records_path.write_text("\n".join(record_lines) + "\n", encoding="utf-8")

    summary_path = artifact / "online_execution.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["source_digests"]["typed_ir.jsonl"] = _sha256(records_path)
    _write_json(summary_path, summary)
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["online_typed_ir.jsonl"] = _sha256(records_path)
    manifest["files"]["online_execution.json"] = _sha256(summary_path)
    _write_json(manifest_path, manifest)

    completed = _replay_failure(artifact)
    assert completed.returncode != 0
    assert "online typed-IR replay mismatch at row 0" in completed.stderr


def test_v2_replay_rejects_rehashed_parent_order_tamper(tmp_path: Path) -> None:
    """Parent-before-child is replayed from raw queries, not trusted from summary."""

    artifact = tmp_path / "artifact"
    shutil.copytree("artifacts/rvir/rvir-cpu-correctness-v2-20260803", artifact)
    queries_path = artifact / "online_queries.jsonl"
    query_lines = queries_path.read_text(encoding="utf-8").splitlines()
    queries = [json.loads(line) for line in query_lines]
    queries[2]["parent_query_id"] = queries[-1]["query_id"]
    queries_path.write_text(
        "".join(
            json.dumps(query, sort_keys=True, allow_nan=False) + "\n"
            for query in queries
        ),
        encoding="utf-8",
    )

    summary_path = artifact / "online_execution.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["source_digests"]["queries.jsonl"] = _sha256(queries_path)
    _write_json(summary_path, summary)
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["online_queries.jsonl"] = _sha256(queries_path)
    manifest["files"]["online_execution.json"] = _sha256(summary_path)
    _write_json(manifest_path, manifest)

    completed = _replay_failure(artifact)
    assert completed.returncode != 0
    assert "online query parent does not precede its child" in completed.stderr
