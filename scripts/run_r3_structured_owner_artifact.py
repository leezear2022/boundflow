#!/usr/bin/env python3
"""Generate or replay the contract-only R3-0 structured-owner artifact."""

# pylint: disable=missing-function-docstring,too-many-boolean-expressions

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Mapping

from boundflow.runtime.r3_structured_owner_contract import (
    build_r30_bundle,
    validate_r30_bundle,
)

PROTOCOL_SCHEMA = "boundflow.r3-0-contract-protocol/v1"
MANIFEST_SCHEMA = "boundflow.r3-0-contract-manifest/v1"
CODE_PATHS = (
    "boundflow/ir/structured_lower_region.py",
    "boundflow/runtime/r3_structured_owner_contract.py",
    "scripts/probe_r3_structured_owner_tamper.py",
    "scripts/run_r3_structured_owner_artifact.py",
    "tests/test_r3_structured_owner_contract.py",
    "tests/test_structured_lower_region.py",
)
ALLOWED_DIRTY_PATHS = (".docops/ev.jsonl",)


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash(value: object) -> str:
    return _hash_bytes(_canonical(value).encode("utf-8"))


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    return value


def _write(path: Path, value: object) -> None:
    path.write_text(_canonical(value) + "\n", encoding="utf-8")


def _git(*args: str) -> str:
    result = subprocess.run(
        ("git", *args), check=True, capture_output=True, text=True, encoding="utf-8"
    )
    return result.stdout.strip()


def _tracked_status() -> tuple[str, ...]:
    result = subprocess.run(
        ("git", "status", "--porcelain=v1", "--untracked-files=no"),
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return _parse_porcelain(result.stdout)


def _parse_porcelain(output: str) -> tuple[str, ...]:
    """Preserve the leading status column and dot-prefixed repository paths."""

    paths: list[str] = []
    for line in output.splitlines():
        if len(line) < 4:
            raise ValueError("R3-0 git porcelain differs")
        path = line[3:]
        if " -> " in path:
            path = path.rsplit(" -> ", 1)[1]
        paths.append(path)
    return tuple(sorted(paths))


def _blob_digest(commit: str, path: str) -> str:
    result = subprocess.run(
        ("git", "show", f"{commit}:{path}"), check=True, capture_output=True
    )
    return _hash_bytes(result.stdout)


def _protocol() -> dict[str, object]:
    head = _git("rev-parse", "HEAD")
    dirty = _tracked_status()
    if any(path not in ALLOWED_DIRTY_PATHS for path in dirty):
        raise ValueError(f"R3-0 formal source is dirty: {dirty}")
    code_revision = {path: _blob_digest(head, path) for path in CODE_PATHS}
    protocol: dict[str, object] = {
        "schema_version": PROTOCOL_SCHEMA,
        "source_git_head": head,
        "source_clean": True,
        "allowed_dirty_paths": list(ALLOWED_DIRTY_PATHS),
        "observed_dirty_paths": list(dirty),
        "code_revision": code_revision,
        "run_kind": "contract-only",
        "production_connected": False,
        "timing_recorded": False,
        "performance_claimed": False,
        "r3_1_open_only_after_replay": True,
    }
    protocol["protocol_hash"] = _hash(protocol)
    return protocol


def _validate_protocol(protocol: Mapping[str, object]) -> None:
    expected = {
        "schema_version",
        "source_git_head",
        "source_clean",
        "allowed_dirty_paths",
        "observed_dirty_paths",
        "code_revision",
        "run_kind",
        "production_connected",
        "timing_recorded",
        "performance_claimed",
        "r3_1_open_only_after_replay",
        "protocol_hash",
    }
    if set(protocol) != expected:
        raise ValueError("R3-0 protocol fields differ")
    unsigned = {name: protocol[name] for name in protocol if name != "protocol_hash"}
    if (
        protocol["schema_version"] != PROTOCOL_SCHEMA
        or protocol["protocol_hash"] != _hash(unsigned)
        or protocol["source_clean"] is not True
        or protocol["allowed_dirty_paths"] != list(ALLOWED_DIRTY_PATHS)
        or protocol["run_kind"] != "contract-only"
        or protocol["production_connected"] is not False
        or protocol["timing_recorded"] is not False
        or protocol["performance_claimed"] is not False
        or protocol["r3_1_open_only_after_replay"] is not True
    ):
        raise ValueError("R3-0 protocol differs")
    commit = protocol["source_git_head"]
    revision = protocol["code_revision"]
    if (
        not isinstance(commit, str)
        or not isinstance(revision, dict)
        or set(revision) != set(CODE_PATHS)
    ):
        raise ValueError("R3-0 code revision differs")
    for path, digest in revision.items():
        if not isinstance(path, str) or digest != _blob_digest(commit, path):
            raise ValueError("R3-0 source blob differs")


def _manifest(
    root: Path, protocol: Mapping[str, object], summary: Mapping[str, object]
) -> dict[str, object]:
    files = {
        str(path.relative_to(root)): _hash_bytes(path.read_bytes())
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    manifest: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA,
        "protocol_hash": protocol["protocol_hash"],
        "summary_hash": summary["summary_hash"],
        "files": files,
    }
    manifest["manifest_hash"] = _hash(manifest)
    return manifest


def _validate_manifest(root: Path, manifest: Mapping[str, object]) -> None:
    expected = {
        "schema_version",
        "protocol_hash",
        "summary_hash",
        "files",
        "manifest_hash",
    }
    if set(manifest) != expected or manifest["schema_version"] != MANIFEST_SCHEMA:
        raise ValueError("R3-0 manifest fields differ")
    unsigned = {name: manifest[name] for name in manifest if name != "manifest_hash"}
    if manifest["manifest_hash"] != _hash(unsigned):
        raise ValueError("R3-0 manifest hash differs")
    files = manifest["files"]
    if not isinstance(files, dict):
        raise ValueError("R3-0 manifest files differ")
    actual_names = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if set(files) != actual_names:
        raise ValueError("R3-0 manifest inventory differs")
    for name, digest in files.items():
        if not isinstance(name, str) or digest != _hash_bytes(
            (root / name).read_bytes()
        ):
            raise ValueError("R3-0 manifest file digest differs")


def generate_artifact(root: Path) -> dict[str, object]:
    """Create a raw-first artifact atomically from a clean committed source."""

    if root.exists():
        raise ValueError("R3-0 artifact target already exists")
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        protocol = _protocol()
        bundle = build_r30_bundle()
        summary = validate_r30_bundle(bundle)
        _write(temporary / "protocol.json", protocol)
        _write(temporary / "bundle.json", bundle)
        _write(temporary / "summary.json", summary)
        _write(temporary / "manifest.json", _manifest(temporary, protocol, summary))
        os.replace(temporary, root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return summary


def replay_artifact(root: Path) -> dict[str, object]:
    """Replay manifest, committed blobs, typed schemas and all semantic derivations."""

    protocol = _load(root / "protocol.json")
    bundle = _load(root / "bundle.json")
    stored_summary = _load(root / "summary.json")
    manifest = _load(root / "manifest.json")
    _validate_manifest(root, manifest)
    _validate_protocol(protocol)
    summary = validate_r30_bundle(bundle)
    if stored_summary != summary:
        raise ValueError("R3-0 summary differs")
    if manifest["protocol_hash"] != protocol["protocol_hash"]:
        raise ValueError("R3-0 manifest protocol differs")
    if manifest["summary_hash"] != summary["summary_hash"]:
        raise ValueError("R3-0 manifest summary differs")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    summary = (
        replay_artifact(args.artifact)
        if args.replay
        else generate_artifact(args.artifact)
    )
    print(_canonical(summary))


if __name__ == "__main__":
    main()
