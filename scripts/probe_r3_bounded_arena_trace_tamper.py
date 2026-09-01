#!/usr/bin/env python3
"""Probe fully re-signed R3-1b0 trace mutations against frozen replay."""

# pylint: disable=wrong-import-position,protected-access,missing-function-docstring

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Callable, cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts import run_r3_bounded_arena_trace_artifact as artifact


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        artifact._canonical_json(payload, indent=2) + "\n", encoding="utf-8"
    )


def _resign_trace(root: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    path = root / "trace.json"
    trace = artifact._load_json(path)
    mutate(trace)
    semantic = {name: trace[name] for name in trace if name != "trace_hash"}
    trace["trace_hash"] = artifact._canonical_hash(semantic)
    _write_json(path, trace)


def _resign_manifest(root: Path) -> None:
    manifest = artifact._load_json(root / "manifest.json")
    manifest["files"] = {
        name: artifact._file_sha256(root / name) for name in artifact.ARTIFACT_FILES
    }
    manifest.pop("manifest_hash", None)
    manifest["manifest_hash"] = artifact._canonical_hash(manifest)
    _write_json(root / "manifest.json", manifest)


def _shape(trace: dict[str, Any]) -> None:
    cast(list[dict[str, Any]], trace["steps"])[10]["output_shape"] = [6, 1, 3, 31, 32]


def _scratch(trace: dict[str, Any]) -> None:
    trace["scratch_slot_count"] = 3


def _slot(trace: dict[str, Any]) -> None:
    cast(list[dict[str, Any]], trace["steps"])[6]["output_slot"] = 0


def _branch(trace: dict[str, Any]) -> None:
    steps = cast(list[dict[str, Any]], trace["steps"])
    cast(list[dict[str, Any]], steps[6]["branches"])[0]["join_value"] = "wrong"


def _compiled(trace: dict[str, Any]) -> None:
    trace["compiled_region"] = True


def _summary(root: Path) -> None:
    summary = artifact._load_json(root / "summary.json")
    summary["b1_open"] = False
    summary.pop("summary_hash", None)
    summary["summary_hash"] = artifact._canonical_hash(summary)
    _write_json(root / "summary.json", summary)
    result = artifact._result(summary)
    (root / "replay_stdout.txt").write_text(
        artifact._canonical_json(result) + "\n", encoding="utf-8"
    )
    manifest = artifact._load_json(root / "manifest.json")
    manifest["summary_hash"] = summary["summary_hash"]
    _write_json(root / "manifest.json", manifest)


def _probe(source: Path) -> dict[str, object]:
    probes: tuple[tuple[str, Callable[[Path], None]], ...] = (
        ("shape", lambda root: _resign_trace(root, _shape)),
        ("scratch-count", lambda root: _resign_trace(root, _scratch)),
        ("slot", lambda root: _resign_trace(root, _slot)),
        ("branch-join", lambda root: _resign_trace(root, _branch)),
        ("compiled-claim", lambda root: _resign_trace(root, _compiled)),
        ("summary-gate", _summary),
    )
    rows = []
    with tempfile.TemporaryDirectory(prefix="r3-1b0-tamper-") as temporary:
        base = Path(temporary)
        for ordinal, (name, mutate) in enumerate(probes):
            target = base / f"probe-{ordinal:02d}"
            shutil.copytree(source, target)
            mutate(target)
            _resign_manifest(target)
            rejected = False
            error = ""
            try:
                artifact._replay(target)
            except (TypeError, ValueError) as caught:
                rejected = True
                error = str(caught)
            rows.append({"name": name, "rejected": rejected, "error": error})
    result: dict[str, object] = {
        "probe_count": len(rows),
        "rejected_count": sum(bool(row["rejected"]) for row in rows),
        "rows": rows,
    }
    result["tamper_hash"] = artifact._canonical_hash(result)
    if result["rejected_count"] != result["probe_count"]:
        raise ValueError("R3-1b0 tamper probe was admitted")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    result = _probe(args.artifact.resolve())
    print(json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
