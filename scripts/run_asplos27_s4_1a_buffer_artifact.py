#!/usr/bin/env python3
"""Generate the S4-1A five-positive plus seven-fault formal artifact."""

# pylint: disable=too-many-locals,too-many-statements,wrong-import-position
# pylint: disable=missing-function-docstring,unidiomatic-typecheck
# pylint: disable=protected-access

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import replay_asplos27_s4_1a_buffer_stdlib as replay_tool
from scripts.run_asplos27_s4_1a_buffer_worker import FAULTS

CODE_PATHS = (
    "boundflow/runtime/asplos27_s4_mutable_state_admission.py",
    "boundflow/runtime/asplos27_s4_ordered_buffer_abi.py",
    "scripts/run_asplos27_s4_admission_worker.py",
    "scripts/run_asplos27_s4_1a_buffer_worker.py",
    "scripts/run_asplos27_s4_1a_buffer_artifact.py",
    "scripts/replay_asplos27_s4_1a_buffer_stdlib.py",
    "scripts/probe_asplos27_s4_1a_buffer_tamper.py",
    "tests/test_asplos27_s4_ordered_buffer_abi.py",
    "tests/test_asplos27_s4_1a_buffer_artifact.py",
)
POSITIVE_MARKERS = (
    "::test_construction_model_hash_is_recomputed_exactly",
    "::test_positive_order_counts_content_views_and_claim_boundary",
    "::test_empty_beta_tokens_have_no_physical_resource",
)


def _run(command: list[str], *, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, check=True, text=True, capture_output=True)


def _git(*arguments: str) -> str:
    return _run(["git", *arguments]).stdout.strip()


def _blob_sha256(revision: str, relative: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{revision}:{relative}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return hashlib.sha256(result.stdout).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(replay_tool.canonical(value) + "\n", encoding="utf-8")


def _write_manifest(root: Path) -> None:
    files = {
        path.relative_to(root).as_posix(): replay_tool.file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    manifest: dict[str, object] = {
        "schema_version": replay_tool.MANIFEST_SCHEMA,
        "artifact_schema": replay_tool.ARTIFACT_SCHEMA,
        "files": files,
        "performance_claimed": False,
    }
    manifest["manifest_hash"] = replay_tool.canonical_hash(manifest)
    _write_json(root / "manifest.json", manifest)


def _assert_committed_clean() -> tuple[str, dict[str, str]]:
    revision = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain", "--", *CODE_PATHS)
    if dirty:
        raise RuntimeError(f"S4-1A formal code paths are dirty:\n{dirty}")
    return revision, {
        relative: _blob_sha256(revision, relative) for relative in CODE_PATHS
    }


def _negative_registry() -> tuple[dict[str, object], str]:
    collect = _run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "tests/test_asplos27_s4_ordered_buffer_abi.py",
        ]
    )
    nodes = [
        line
        for line in collect.stdout.splitlines()
        if line.startswith("tests/test_asplos27_s4_ordered_buffer_abi.py::")
    ]
    negative = [
        node for node in nodes if not any(marker in node for marker in POSITIVE_MARKERS)
    ]
    if len(negative) < 68 or len(set(negative)) != len(negative):
        raise RuntimeError(f"S4-1A negative inventory differs: {len(negative)}")
    targeted = _run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_asplos27_s4_ordered_buffer_abi.py",
        ]
    )
    registry: dict[str, object] = {
        "schema_version": replay_tool.NEGATIVE_SCHEMA,
        "minimum_required": 68,
        "case_count": len(negative),
        "cases": [
            {
                "ordinal": ordinal,
                "nodeid": node,
                "exact_detail_and_reason_asserted": True,
            }
            for ordinal, node in enumerate(negative)
        ],
        "targeted_result": "pass",
        "targeted_stdout_sha256": hashlib.sha256(
            targeted.stdout.encode("utf-8")
        ).hexdigest(),
        "performance_claimed": False,
    }
    return registry, targeted.stdout


def _worker_command(
    args: argparse.Namespace, ordinal: int, fault: str, result: Path
) -> list[str]:
    return [
        str(args.external_python),
        str(ROOT / "scripts/run_asplos27_s4_1a_buffer_worker.py"),
        "--benchmark-root",
        str(args.benchmark_root),
        "--abcrown-root",
        str(args.abcrown_root),
        "--model",
        str(args.model),
        "--property",
        str(args.property),
        "--run-ordinal",
        str(ordinal),
        "--fault",
        fault,
        "--result",
        str(result),
    ]


def _extract_binary(output: Path, row: dict[str, Any], ordinal: int) -> None:
    payload = row.get("admission")
    if type(payload) is not dict or payload.get("mode") != "positive":
        return
    records = payload.pop("binary_records", None)
    if type(records) is not list or len(records) != 8:
        raise ValueError("S4-1A positive binary record inventory differs")
    binary = bytearray()
    index: list[dict[str, object]] = []
    for record in records:
        if type(record) is not dict:
            raise TypeError("S4-1A binary record differs")
        source = bytes.fromhex(str(record.pop("source_hex")))
        candidate = bytes.fromhex(str(record.pop("candidate_hex")))
        source_offset = len(binary)
        binary.extend(source)
        candidate_offset = len(binary)
        binary.extend(candidate)
        index.append(
            {
                **record,
                "source_offset": source_offset,
                "candidate_offset": candidate_offset,
            }
        )
    relative = f"raw/binary/run-{ordinal:02d}.bin"
    path = output / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(binary))
    payload["binary_sidecar"] = {
        "relative_path": relative,
        "byte_count": len(binary),
        "sha256": replay_tool.file_sha256(path),
    }
    payload["binary_index"] = index
    payload["worker_payload_hash"] = (
        replay_tool._hash_without(  # pylint: disable=protected-access
            payload, "worker_payload_hash"
        )
    )
    row["raw_hash"] = replay_tool._hash_without(
        row, "raw_hash"
    )  # pylint: disable=protected-access


def generate(args: argparse.Namespace) -> dict[str, Any]:
    revision, code_revision = _assert_committed_clean()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"S4-1A artifact output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    sequence = ["none"] * 5 + list(FAULTS)
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="boundflow-s4-1a-workers-") as tmp:
        temp_root = Path(tmp)
        for ordinal, fault in enumerate(sequence):
            result_path = temp_root / f"run-{ordinal:02d}.json"
            completed = _run(_worker_command(args, ordinal, fault, result_path))
            if '"status":"captured"' not in completed.stdout:
                raise RuntimeError(f"S4-1A worker {ordinal} completion differs")
            row = json.loads(result_path.read_text(encoding="utf-8"))
            if type(row) is not dict:
                raise TypeError("S4-1A worker root differs")
            _extract_binary(output, row, ordinal)
            rows.append(row)
    workers = output / "raw/workers.jsonl"
    workers.parent.mkdir(parents=True, exist_ok=True)
    workers.write_text(
        "".join(replay_tool.canonical(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    registry, targeted_stdout = _negative_registry()
    _write_json(output / "negative_registry.json", registry)
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "targeted-pytest.txt").write_text(targeted_stdout, encoding="utf-8")
    source = rows[0]["source"]
    if any(row["source"] != source for row in rows):
        raise ValueError("S4-1A source changed across fresh workers")
    protocol: dict[str, object] = {
        "schema_version": replay_tool.PROTOCOL_SCHEMA,
        "artifact_schema": replay_tool.ARTIFACT_SCHEMA,
        "source_revision": revision,
        "code_revision": code_revision,
        "source": source,
        "source_hash": replay_tool.canonical_hash(source),
        "worker_sequence": sequence,
        "fresh_process_count": 12,
        "positive_process_count": 5,
        "isolated_fault_process_count": 7,
        "negative_case_minimum": 68,
        "workers_jsonl_sha256": replay_tool.file_sha256(workers),
        "negative_registry_sha256": replay_tool.file_sha256(
            output / "negative_registry.json"
        ),
        "exact_call_identity_raw_persisted": False,
        "buffer_prepare": True,
        "candidate_execute": False,
        "mutation": False,
        "timing_recorded": False,
        "performance_claimed": False,
    }
    protocol["protocol_hash"] = replay_tool.canonical_hash(protocol)
    _write_json(output / "protocol.json", protocol)
    summary = replay_tool._derive_summary(  # pylint: disable=protected-access
        output, rows, protocol, registry
    )
    _write_json(output / "summary.json", summary)
    (output / "README.md").write_text(
        "# ASPLOS'27 S4-1A ordered buffer artifact\n\n"
        "Five fresh real-provider processes prepare and close the ordered compressed "
        "alpha/active-beta buffer ABI. Seven additional fresh processes inject one "
        "isolated construction fault each. No CROWN evaluator, optimizer mutation, "
        "timing, fallback, or performance claim is present.\n",
        encoding="utf-8",
    )
    _write_manifest(output)
    return replay_tool.replay(output)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--abcrown-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--property", type=Path, required=True)
    parser.add_argument("--external-python", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    for name in (
        "output",
        "benchmark_root",
        "abcrown_root",
        "model",
        "property",
        "external_python",
    ):
        setattr(args, name, Path(os.path.abspath(getattr(args, name))))
    summary = generate(args)
    print(
        "S4-1A buffer artifact PASS: "
        f"workers={summary['fresh_process_count']} "
        f"negative={summary['negative_case_count']} "
        f"status={summary['status']}"
    )


if __name__ == "__main__":
    main()
