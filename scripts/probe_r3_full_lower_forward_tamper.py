#!/usr/bin/env python3
"""Fully re-sign semantic tampering against the R3-1b1 artifact replay."""

# pylint: disable=wrong-import-position,missing-function-docstring,protected-access

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Callable

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts import run_r3_full_lower_forward_artifact as artifact


def _load_record(path: Path) -> dict[str, Any]:
    value = json.loads((path / "raw.jsonl").read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("R3-1b1 tamper raw root differs")
    return value


def _unsafe_resign(path: Path, record: dict[str, Any]) -> None:
    metric = record["metric"]
    compilation = record["compilation_receipt"]
    launch = record["launch_receipt"]
    summary = artifact._load_json(
        path / "summary.json"
    )  # pylint: disable=protected-access
    summary.update(
        trace_hash=record["trace_hash"],
        production_plan_hash=record["production_plan_hash"],
        module_hash=compilation["module_hash"],
        device_source_hash=compilation["device_source_hash"],
        compilation_receipt_hash=record["compilation_receipt_hash"],
        native_lower=metric["native_lower"],
        candidate_lower=metric["candidate_lower"],
        max_abs_diff=metric["max_abs_diff"],
        sign_exact=metric["sign_exact"],
        launch_count=launch["launch_count"],
        coefficient_scratch_count=launch["coefficient_scratch_count"],
        scratch_capacity_elements=launch["scratch_capacity_elements"],
        dlpack_pointer_count=launch["dlpack_pointer_count"],
        dlpack_pointer_exact_count=launch["dlpack_pointer_exact_count"],
        warm_dynamic_allocated_bytes=launch["warm_dynamic_allocated_bytes"],
        python_visible_intermediate_coefficient_count=launch[
            "python_visible_intermediate_coefficient_count"
        ],
        compiled_region=launch["compiled_region"],
    )
    summary.pop("summary_hash", None)
    summary["summary_hash"] = artifact._hash(
        summary
    )  # pylint: disable=protected-access
    artifact._write_json(
        path / "summary.json", summary
    )  # pylint: disable=protected-access
    (path / "raw.jsonl").write_text(
        artifact._canonical(record) + "\n",
        encoding="utf-8",  # pylint: disable=protected-access
    )
    result = artifact._result(summary)  # pylint: disable=protected-access
    (path / "replay_stdout.txt").write_text(
        artifact._canonical(result) + "\n",
        encoding="utf-8",  # pylint: disable=protected-access
    )
    manifest = {
        "schema_version": artifact.ARTIFACT_SCHEMA,
        "files": {
            name: artifact._file_hash(path / name)  # pylint: disable=protected-access
            for name in artifact.SIGNED_FILES
        },
    }
    manifest["manifest_hash"] = artifact._hash(
        manifest
    )  # pylint: disable=protected-access
    artifact._write_json(
        path / "manifest.json", manifest
    )  # pylint: disable=protected-access


def _mutations() -> tuple[tuple[str, Callable[[dict[str, Any]], None]], ...]:
    return (
        (
            "candidate-lower",
            lambda record: record["metric"]["candidate_lower"].__setitem__(0, 0.0),
        ),
        (
            "native-lower",
            lambda record: record["metric"]["native_lower"].__setitem__(0, 0.0),
        ),
        (
            "module-hash",
            lambda record: record["compilation_receipt"].__setitem__(
                "module_hash", "0" * 64
            ),
        ),
        (
            "device-source-hash",
            lambda record: record["compilation_receipt"].__setitem__(
                "device_source_hash", "0" * 64
            ),
        ),
        (
            "launch-count",
            lambda record: record["launch_receipt"].__setitem__("launch_count", 14),
        ),
        (
            "scratch-alias",
            lambda record: record["launch_receipt"]["scratch_pointers"].__setitem__(
                1, record["launch_receipt"]["scratch_pointers"][0]
            ),
        ),
        (
            "warm-allocation",
            lambda record: record["launch_receipt"].__setitem__(
                "warm_dynamic_allocated_bytes", 4
            ),
        ),
        (
            "stream-mismatch",
            lambda record: record["launch_receipt"].__setitem__(
                "tvm_ffi_stream_id", record["launch_receipt"]["stream_id"] + 1
            ),
        ),
        (
            "compiled-region",
            lambda record: record["launch_receipt"].__setitem__(
                "compiled_region", False
            ),
        ),
        (
            "dlpack-exact",
            lambda record: record["launch_receipt"].__setitem__(
                "dlpack_pointer_exact_count", 69
            ),
        ),
    )


def probe(source: Path) -> dict[str, object]:
    rejected: list[str] = []
    with tempfile.TemporaryDirectory(prefix="r31b1-tamper-") as temporary:
        root = Path(temporary)
        original = _load_record(source)
        for name, mutate in _mutations():
            target = root / name
            shutil.copytree(source, target)
            changed = copy.deepcopy(original)
            mutate(changed)
            _unsafe_resign(target, changed)
            try:
                artifact.replay(target)
            except (ValueError, TypeError, KeyError):
                rejected.append(name)
            else:
                raise RuntimeError(f"R3-1b1 tamper was accepted: {name}")
    return {
        "status": "tamper-passed",
        "attack_count": len(_mutations()),
        "rejected_count": len(rejected),
        "rejected": rejected,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    print(artifact._canonical(probe(args.artifact)))  # pylint: disable=protected-access
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
