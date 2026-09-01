#!/usr/bin/env python3
"""Outer-resigned tamper probes for the B4-B2 B2-5 formal artifact."""

# pylint: disable=wrong-import-position,missing-function-docstring,duplicate-code

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Callable

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.run_fsg4_b4b2_b2_5_artifact import (
    canonical_hash,
    collect_files,
    load_json,
    write_json,
)


def _resign(artifact: Path) -> None:
    manifest = load_json(artifact / "manifest.json")
    manifest["files"] = collect_files(artifact)
    manifest.pop("manifest_hash", None)
    manifest["manifest_hash"] = canonical_hash(manifest)
    write_json(artifact / "manifest.json", manifest)


def _mutate_latency(artifact: Path) -> None:
    path = artifact / "timing/run_00.json"
    value = load_json(path)
    value["result"]["pairs"][0]["candidate_ms"] *= 0.1
    result = value["result"]
    result.pop("worker_hash", None)
    result["worker_hash"] = canonical_hash(result)
    value.pop("envelope_hash", None)
    value["envelope_hash"] = canonical_hash(value)
    write_json(path, value)


def _mutate_winner(artifact: Path) -> None:
    path = artifact / "calibration.json"
    value = load_json(path)
    value["result"]["winner_ordinal"] = (
        int(value["result"]["winner_ordinal"]) + 1
    ) % 12
    value.pop("envelope_hash", None)
    value["envelope_hash"] = canonical_hash(value)
    write_json(path, value)


def _mutate_kernel_inventory(artifact: Path) -> None:
    path = artifact / "timing/run_01.json"
    value = load_json(path)
    value["result"]["kernel_inventory"]["total_kernel_count"] = 1
    result = value["result"]
    result.pop("worker_hash", None)
    result["worker_hash"] = canonical_hash(result)
    value.pop("envelope_hash", None)
    value["envelope_hash"] = canonical_hash(value)
    write_json(path, value)


def _mutate_semantic(artifact: Path) -> None:
    path = artifact / "correctness/P_00.json"
    value = load_json(path)
    value["result"]["semantic_passed"] = False
    result = value["result"]
    result.pop("worker_hash", None)
    result["worker_hash"] = canonical_hash(result)
    value.pop("envelope_hash", None)
    value["envelope_hash"] = canonical_hash(value)
    write_json(path, value)


def _mutate_source(artifact: Path) -> None:
    path = artifact / "protocol.json"
    value = load_json(path)
    revisions = value["code_revision"]
    first = sorted(revisions)[0]
    revisions[first] = "0" * 64
    value.pop("protocol_hash", None)
    value["protocol_hash"] = canonical_hash(value)
    write_json(path, value)


def _delete_correctness(artifact: Path) -> None:
    (artifact / "correctness/S_04.json").unlink()


def _append_candidate(artifact: Path) -> None:
    path = artifact / "calibration.json"
    value = load_json(path)
    row = deepcopy(value["result"]["rows"][-1])
    row["candidate_ordinal"] = 12
    value["result"]["rows"].append(row)
    value["result"]["candidate_count"] = 13
    value.pop("envelope_hash", None)
    value["envelope_hash"] = canonical_hash(value)
    write_json(path, value)


def _mutate_module_hash(artifact: Path) -> None:
    path = artifact / "timing/run_02.json"
    value = load_json(path)
    value["result"]["module_receipt_hash"] = "f" * 64
    result = value["result"]
    result.pop("worker_hash", None)
    result["worker_hash"] = canonical_hash(result)
    value.pop("envelope_hash", None)
    value["envelope_hash"] = canonical_hash(value)
    write_json(path, value)


CASES: tuple[tuple[str, Callable[[Path], None]], ...] = (
    ("latency-raw-outer-resign", _mutate_latency),
    ("winner-outer-resign", _mutate_winner),
    ("kernel-inventory-outer-resign", _mutate_kernel_inventory),
    ("semantic-outer-resign", _mutate_semantic),
    ("source-revision-outer-resign", _mutate_source),
    ("missing-correctness-outer-resign", _delete_correctness),
    ("thirteenth-candidate-outer-resign", _append_candidate),
    ("module-hash-outer-resign", _mutate_module_hash),
)


def run_probe(artifact: Path) -> dict[str, object]:
    rows = []
    with tempfile.TemporaryDirectory(prefix="b4b2-b2-5-tamper-") as directory:
        root = Path(directory)
        for name, mutate in CASES:
            target = root / name
            shutil.copytree(artifact, target)
            mutate(target)
            _resign(target)
            completed = subprocess.run(
                (
                    sys.executable,
                    str(REPOSITORY_ROOT / "scripts/run_fsg4_b4b2_b2_5_artifact.py"),
                    "replay",
                    "--artifact-dir",
                    str(target),
                    "--no-recompile",
                ),
                cwd=REPOSITORY_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            rows.append(
                {
                    "case": name,
                    "rejected": completed.returncode != 0,
                    "returncode": completed.returncode,
                    "error_tail": (
                        completed.stderr.splitlines()[-1]
                        if completed.stderr.splitlines()
                        else ""
                    ),
                }
            )
    return {
        "schema_version": "boundflow.fsg4-b4b2-b2-5-tamper-report/v1",
        "case_count": len(rows),
        "rejected_count": sum(bool(row["rejected"]) for row in rows),
        "all_rejected": all(bool(row["rejected"]) for row in rows),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = run_probe(args.artifact_dir)
    if not result["all_rejected"]:
        raise RuntimeError(json.dumps(result, sort_keys=True))
    write_json(args.report, result)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
