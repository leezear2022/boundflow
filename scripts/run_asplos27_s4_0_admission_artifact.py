#!/usr/bin/env python3
"""Generate the S4-0 five-fresh real-provider admission artifact."""

# pylint: disable=too-many-locals,too-many-statements,duplicate-code
# pylint: disable=wrong-import-position,missing-function-docstring
# pylint: disable=unidiomatic-typecheck

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

from scripts import replay_asplos27_s4_0_admission_stdlib as replay_tool

CODE_PATHS = (
    "boundflow/runtime/asplos27_s4_mutable_state_admission.py",
    "scripts/run_asplos27_s4_admission_worker.py",
    "scripts/run_asplos27_s4_0_admission_artifact.py",
    "scripts/replay_asplos27_s4_0_admission_stdlib.py",
    "scripts/probe_asplos27_s4_0_admission_tamper.py",
    "tests/test_asplos27_s4_mutable_state_admission.py",
)
POSITIVE_NODE_MARKERS = (
    "::test_formal_admission_counts_policy_and_claim_boundary",
    "::test_provider_source_readiness_is_recorded_not_rewritten",
    "::test_receipt_is_canonical_tensor_free_and_json_serializable",
    "::test_tensor_free_walker_accepts_immutable_dag_sharing",
    "::test_topology_and_snapshot_storage_order_do_not_change_receipt",
    "::test_exact_call_identity_changes_only_identity_bound_receipt",
    "::test_lease_holds_strong_tensor_ownership_until_close",
    "::test_provider_adapter_extracts_exact_builtin_structure",
    "::test_provider_adapter_accepts_exact_stdlib_defaultdict_alpha_owner",
)
EXCLUDED_NEGATIVE_MARKERS = ("::test_failure_injection_never_publishes_partial_owner",)


def _run(command: list[str], *, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    )


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


def _write_manifest(root: Path) -> dict[str, object]:
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
    return manifest


def _assert_code_is_committed_clean() -> tuple[str, dict[str, str]]:
    revision = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain", "--", *CODE_PATHS)
    if dirty:
        raise RuntimeError(f"S4 formal code paths are dirty:\n{dirty}")
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
            "tests/test_asplos27_s4_mutable_state_admission.py",
        ]
    )
    nodes = [
        line
        for line in collect.stdout.splitlines()
        if line.startswith("tests/test_asplos27_s4_mutable_state_admission.py::")
    ]
    negative = [
        node
        for node in nodes
        if not any(marker in node for marker in POSITIVE_NODE_MARKERS)
        and not any(marker in node for marker in EXCLUDED_NEGATIVE_MARKERS)
    ]
    if len(negative) < 56 or len(set(negative)) != len(negative):
        raise RuntimeError(f"S4 negative registry inventory differs: {len(negative)}")
    targeted = _run([sys.executable, "-m", "pytest", "-q", *negative])
    registry: dict[str, object] = {
        "schema_version": replay_tool.NEGATIVE_SCHEMA,
        "minimum_required": 56,
        "case_count": len(negative),
        "cases": [
            {
                "ordinal": ordinal,
                "nodeid": node,
                "fresh_pytest_case": True,
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


def _worker_command(args: argparse.Namespace, ordinal: int, result: Path) -> list[str]:
    return [
        str(args.external_python),
        str(ROOT / "scripts/run_asplos27_s4_admission_worker.py"),
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
        "--result",
        str(result),
    ]


def generate(args: argparse.Namespace) -> dict[str, Any]:
    revision, code_revision = _assert_code_is_committed_clean()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"S4 artifact output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="boundflow-s4-admission-workers-") as tmp:
        temp_root = Path(tmp)
        for ordinal in range(5):
            result_path = temp_root / f"run-{ordinal:02d}.json"
            completed = _run(_worker_command(args, ordinal, result_path))
            if '"status":"captured"' not in completed.stdout:
                raise RuntimeError(f"S4 worker {ordinal} completion marker differs")
            value = json.loads(result_path.read_text(encoding="utf-8"))
            if type(value) is not dict:
                raise TypeError("S4 worker result root differs")
            rows.append(value)
    workers_path = output / "raw/workers.jsonl"
    workers_path.parent.mkdir(parents=True, exist_ok=True)
    workers_path.write_text(
        "".join(replay_tool.canonical(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    registry, targeted_stdout = _negative_registry()
    _write_json(output / "negative_registry.json", registry)
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "negative-pytest.txt").write_text(targeted_stdout, encoding="utf-8")

    source = rows[0]["source"]
    worker_protocol = rows[0]["protocol"]
    if any(row["source"] != source for row in rows) or any(
        row["protocol"] != worker_protocol for row in rows
    ):
        raise ValueError("S4 worker source/protocol changed across fresh processes")
    protocol: dict[str, object] = {
        "schema_version": replay_tool.PROTOCOL_SCHEMA,
        "artifact_schema": replay_tool.ARTIFACT_SCHEMA,
        "source_revision": revision,
        "code_revision": code_revision,
        "source": source,
        "source_hash": replay_tool.canonical_hash(source),
        "worker_protocol": worker_protocol,
        "worker_protocol_hash": replay_tool.canonical_hash(worker_protocol),
        "fresh_process_count": 5,
        "negative_case_minimum": 56,
        "workers_jsonl_sha256": replay_tool.file_sha256(workers_path),
        "negative_registry_sha256": replay_tool.file_sha256(
            output / "negative_registry.json"
        ),
        "exact_call_identity_raw_persisted": False,
        "buffer_prepare": False,
        "candidate_execute": False,
        "mutation": False,
        "timing_recorded": False,
        "performance_claimed": False,
    }
    protocol["protocol_hash"] = replay_tool.canonical_hash(protocol)
    _write_json(output / "protocol.json", protocol)

    summary = replay_tool._derive_summary(  # pylint: disable=protected-access
        rows, protocol, registry
    )
    _write_json(output / "summary.json", summary)
    (output / "README.md").write_text(
        "# ASPLOS'27 S4-0 mutable-state admission artifact\n\n"
        "Five fresh real αβ-CROWN provider processes stop immediately after "
        "tensor-free admission and lease close. No buffer prepare, candidate "
        "execution, optimizer mutation, timing, or performance claim occurs.\n\n"
        "Replay with `python scripts/replay_asplos27_s4_0_admission_stdlib.py "
        "--artifact <this-directory>`.\n",
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
        "S4-0 admission artifact PASS: "
        f"workers={summary['fresh_process_count']} "
        f"negative={summary['negative_case_count']} "
        f"status={summary['status']}"
    )


if __name__ == "__main__":
    main()
