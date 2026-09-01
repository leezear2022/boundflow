#!/usr/bin/env python3
"""Generate the frozen eleven-process S4-1B0 correctness artifact."""

# pylint: disable=too-many-locals,too-many-statements,missing-function-docstring
# pylint: disable=too-many-branches,duplicate-code

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_SCHEMA = "boundflow.asplos27-s4-1b0-ternary-summary/v1"
MANIFEST_SCHEMA = "boundflow.asplos27-s4-1b0-ternary-manifest/v1"
PROTOCOL_SCHEMA = "boundflow.asplos27-s4-1b0-ternary-protocol/v1"
WORKERS = (
    *(f"positive-{ordinal:02d}" for ordinal in range(5)),
    "cache-00",
    "fault-classifier-policy",
    "fault-cache-source",
    "fault-descriptor-dlpack",
    "fault-stream-launch",
    "fault-invalid-selector-claim",
)
FAULT_REASON = {
    "fault-classifier-policy": "TERNARY_ENDPOINT_MIDPOINT_POLICY_MISMATCH",
    "fault-cache-source": "TERNARY_ENDPOINT_DEVICE_SOURCE_MISMATCH",
    "fault-descriptor-dlpack": "TERNARY_ENDPOINT_DLPACK_IDENTITY_MISMATCH",
    "fault-stream-launch": "TERNARY_ENDPOINT_STREAM_IDENTITY_MISMATCH",
    "fault-invalid-selector-claim": "TERNARY_ENDPOINT_INVALID_SELECTOR_NOT_POISONED",
}
CODE_PATHS = (
    "boundflow/backends/tvm/asplos27_s4_ternary_endpoint.py",
    "scripts/run_asplos27_s4_1b0_ternary_worker.py",
    "scripts/run_asplos27_s4_1b0_ternary_artifact.py",
    "scripts/replay_asplos27_s4_1b0_ternary_stdlib.py",
    "scripts/probe_asplos27_s4_1b0_ternary_tamper.py",
    "tests/test_asplos27_s4_ternary_endpoint.py",
    "tests/test_asplos27_s4_1b0_ternary_artifact.py",
)


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_hash(value: object) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical(value) + "\n", encoding="utf-8")


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, check=True, text=True, capture_output=True)


def _git(*args: str) -> str:
    return _run(["git", *args]).stdout.strip()


def _blob_hash(revision: str, path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{revision}:{path}"], cwd=ROOT, check=True, capture_output=True
    )
    return hashlib.sha256(result.stdout).hexdigest()


def _source_identity() -> tuple[str, dict[str, str]]:
    revision = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain=v1", "--", *CODE_PATHS)
    if dirty:
        raise RuntimeError(f"formal code paths are dirty:\n{dirty}")
    return revision, {path: _blob_hash(revision, path) for path in CODE_PATHS}


def _protocol(revision: str, blobs: dict[str, str]) -> dict[str, Any]:
    source_capture = Path(
        "artifacts/asplos27-s3-streamed-suffix/resnet2b-rvir-v1/inputs/suffix-boundary.pt"
    )
    dependencies = (
        "gemini_doc/BOUNDFLOW_ASPLOS27_S4_1B0_IEEE_BIT_FIXTURES_V1_2026_08_30.json",
        "gemini_doc/BOUNDFLOW_ASPLOS27_S4_1B0_NEGATIVE_CONTRACT_V1_2026_08_30.json",
        "gemini_doc/BOUNDFLOW_ASPLOS27_S4_1B0_ABI_CONTRACT_V1_2026_08_30.json",
        "gemini_doc/BOUNDFLOW_ASPLOS27_S4_1B0_FORMAL_ARTIFACT_CONTRACT_V1_2026_08_30.json",
    )
    return {
        "schema_version": PROTOCOL_SCHEMA,
        "source_revision": revision,
        "code_blobs": blobs,
        "construction_model_hash": (
            "5056d302aa27785ab8a22bd8f5665ebef0a4aba2ca22bc72ce28581144dbcc2a"
        ),
        "source_capture": {
            "path": source_capture.as_posix(),
            "sha256": file_sha256(ROOT / source_capture),
        },
        "dependencies": {path: file_sha256(ROOT / path) for path in dependencies},
        "external_commits": {
            "abcrown": "e5c7e17bf0488843acb77b7519f59876717a49f4",
            "auto_lirpa": "5a098e8f9fb5786a428a024981d833d303921f2d",
            "vnncomp": "90419aadcf06cf543ce5c1706cae1059dc9fa6cf",
        },
        "model": {
            "sha256": "791aa24d77917ecda16809fbbd48e7739616f88ebf74cf358b2d1bf911dc4a6d"
        },
        "property": {
            "sha256": "89edf0665d74397670d0562d513db694a49a84edaf5cf3d64c9c6fa63c3769ff"
        },
        "worker_sequence": list(WORKERS),
        "numel": 18432,
        "timing_recorded": False,
        "performance_claimed": False,
    }


def _worker_command(name: str, ordinal: int, protocol: Path, temp: Path) -> list[str]:
    mode = (
        "positive"
        if name.startswith("positive")
        else "cache" if name == "cache-00" else "fault"
    )
    command = [
        sys.executable,
        "scripts/run_asplos27_s4_1b0_ternary_worker.py",
        "--mode",
        mode,
        "--worker-name",
        name,
        "--ordinal",
        str(ordinal),
        "--protocol",
        str(protocol),
    ]
    if mode == "positive":
        command += [
            "--binary-output",
            str(temp / f"{name}.bin"),
            "--module-output",
            str(temp / f"module-{name}"),
        ]
    elif mode == "fault":
        command += ["--fault", name.removeprefix("fault-")]
    return command


def _manifest(root: Path) -> None:
    files = {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    value: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA,
        "artifact_schema": ARTIFACT_SCHEMA,
        "files": files,
        "performance_claimed": False,
    }
    value["manifest_hash"] = canonical_hash(value)
    _write_json(root / "manifest.json", value)


def generate(output: Path) -> dict[str, Any]:
    if output.exists():
        raise RuntimeError(
            "artifact output already exists; partial resume is forbidden"
        )
    revision, blobs = _source_identity()
    output.mkdir(parents=True)
    protocol = _protocol(revision, blobs)
    _write_json(output / "protocol.json", protocol)
    negative_contract = json.loads(
        (
            ROOT
            / "gemini_doc/BOUNDFLOW_ASPLOS27_S4_1B0_NEGATIVE_CONTRACT_V1_2026_08_30.json"
        ).read_text(encoding="utf-8")
    )
    registry = {
        "schema_version": "boundflow.asplos27-s4-1b0-ternary-negative-registry/v1",
        "reasons": negative_contract["stable_reasons"],
        "reason_count": 20,
        "performance_claimed": False,
    }
    _write_json(output / "negative_registry.json", registry)
    rows = []
    (output / "raw/binary").mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="s4-1b0-workers-") as temporary:
        temp = Path(temporary)
        for ordinal, name in enumerate(WORKERS):
            result = _run(
                _worker_command(name, ordinal, output / "protocol.json", temp)
            )
            row = json.loads(result.stdout)
            rows.append(row)
            if name.startswith("positive"):
                shutil.copy2(
                    temp / f"{name}.bin", output / "raw/binary" / f"{name}.bin"
                )
                module_dir = temp / f"module-{name}"
                if name == "positive-00":
                    shutil.copytree(module_dir, output / "module")
                else:
                    for module_file in module_dir.iterdir():
                        if file_sha256(module_file) != file_sha256(
                            output / "module" / module_file.name
                        ):
                            raise RuntimeError("fresh module identity differs")
    (output / "raw").mkdir(exist_ok=True)
    (output / "raw/workers.jsonl").write_text(
        "".join(canonical(row) + "\n" for row in rows), encoding="utf-8"
    )
    positives = rows[:5]
    cache = rows[5]
    faults = rows[6:]
    if len({row["pid"] for row in rows}) != 11:
        raise RuntimeError("fresh worker pid inventory differs")
    expected_counts = {"positive": 8689, "negative": 9137, "zero": 606, "invalid": 0}
    if any(
        row["counts"] != expected_counts or not row["selected_bitwise_exact"]
        for row in positives
    ):
        raise RuntimeError("positive semantics differ")
    if any(
        row["binary"]["sha256"] != positives[0]["binary"]["sha256"] for row in positives
    ):
        raise RuntimeError("positive binary determinism differs")
    if (cache["events"], cache["compile_count"], cache["hit_count"]) != (
        ["miss", "hit"],
        1,
        1,
    ):
        raise RuntimeError("cache worker differs")
    for row in faults:
        if row["result"]["reason"] != FAULT_REASON[row["worker_name"]]:
            raise RuntimeError("fault reason differs")
    targeted = _run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_asplos27_s4_ternary_endpoint.py",
        ]
    )
    (output / "logs").mkdir()
    (output / "logs/targeted-pytest.txt").write_text(targeted.stdout, encoding="utf-8")
    summary: dict[str, Any] = {
        "schema_version": ARTIFACT_SCHEMA,
        "source_revision": revision,
        "worker_count": 11,
        "worker_sequence": list(WORKERS),
        "fresh_pid_count": 11,
        "positive_count": 5,
        "cache_count": 1,
        "fault_count": 5,
        "selector_counts": expected_counts,
        "old_binary_zero_misclassified": 606,
        "selected_bitwise_exact": True,
        "positive_sidecar_sha256": positives[0]["binary"]["sha256"],
        "positive_sidecar_byte_count": 313344,
        "module_receipt_hash": positives[0]["module_receipt_hash"],
        "cache_events": ["miss", "hit"],
        "fault_reasons": [row["result"]["reason"] for row in faults],
        "targeted_result": "pass",
        "status": "FORMAL-CANDIDATE-PASS-PENDING-EXTERNAL-AUDIT-S4-1B0",
        "timing_recorded": False,
        "performance_claimed": False,
    }
    summary["summary_hash"] = canonical_hash(summary)
    _write_json(output / "summary.json", summary)
    (output / "README.md").write_text(
        "# S4-1B0 ternary endpoint formal candidate\n\n"
        "Five positive, one cache, and five fault workers. No timing or performance claim.\n",
        encoding="utf-8",
    )
    _manifest(output)
    replay = _run(
        [
            sys.executable,
            "scripts/replay_asplos27_s4_1b0_ternary_stdlib.py",
            str(output),
        ]
    )
    if "PASS" not in replay.stdout:
        raise RuntimeError("stdlib replay did not pass")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/asplos27-s4-1b0-ternary/resnet2b-prop0-v1"),
    )
    args = parser.parse_args()
    print(canonical(generate(args.output.resolve())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
