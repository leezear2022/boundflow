#!/usr/bin/env python3
"""Probe fully re-signed R3-1 M0 artifact mutations against semantic replay."""

# pylint: disable=wrong-import-position,protected-access,too-many-locals
# pylint: disable=missing-function-docstring

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Callable, cast

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from boundflow.runtime.rvir_v4_production_state import production_tensor_sha256
from scripts import run_r3_structured_owner_five_fresh_artifact as artifact


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        artifact._canonical_json(payload, indent=2) + "\n", encoding="utf-8"
    )


def _resign_manifest(root: Path) -> None:
    manifest = artifact._load_json(root / "manifest.json")
    manifest["files"] = {
        name: artifact._file_sha256(root / name) for name in artifact._artifact_files()
    }
    manifest.pop("manifest_hash", None)
    manifest["manifest_hash"] = artifact._canonical_hash(manifest)
    _write_json(root / "manifest.json", manifest)


def _mutate_raw(
    root: Path, mutator: Callable[[dict[str, Any]], None], *, mode: str = "candidate"
) -> None:
    path = root / artifact._raw_name(0, mode)
    row = artifact._load_torch(path)
    mutator(row)
    torch.save(row, path)


def _lower(row: dict[str, Any]) -> None:
    value = cast(torch.Tensor, row["final_lower"])
    value[0, 0] += 0.1
    row["final_lower_sha256"] = production_tensor_sha256(value)


def _gradient(row: dict[str, Any]) -> None:
    value = cast(torch.Tensor, row["compressed_alpha_gradient"])
    value[0, 0, 0, 0] += 0.1
    row["compressed_alpha_gradient_sha256"] = production_tensor_sha256(value)


def _memory(row: dict[str, Any]) -> None:
    memory = cast(dict[str, int], row["memory"])
    memory["peak_allocated"] = 1
    memory["peak_allocated_increment"] = 0


def _compiled(row: dict[str, Any]) -> None:
    cast(dict[str, object], row["execution_receipt"])["compiled_region"] = True


def _dense_saved(row: dict[str, Any]) -> None:
    cast(dict[str, object], row["execution_receipt"])["saved_dense_a_count"] = 1


def _claim(row: dict[str, Any]) -> None:
    row["performance_claimed"] = True


def _version(row: dict[str, Any]) -> None:
    cast(list[int], row["alpha_versions_after"])[0] += 1


def _summary_gate(root: Path) -> None:
    summary = artifact._load_json(root / "summary.json")
    summary["status"] = "validated-r3-1-m0"
    summary["r3_1_admitted"] = True
    summary["r3_2a_open"] = True
    summary["all_peak_allocated_passed"] = True
    summary["all_compiled_region_passed"] = True
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
        ("final-lower", lambda root: _mutate_raw(root, _lower)),
        ("compressed-alpha-gradient", lambda root: _mutate_raw(root, _gradient)),
        ("peak-allocated", lambda root: _mutate_raw(root, _memory)),
        ("compiled-region", lambda root: _mutate_raw(root, _compiled)),
        ("saved-dense-a", lambda root: _mutate_raw(root, _dense_saved)),
        ("performance-claim", lambda root: _mutate_raw(root, _claim)),
        ("alpha-version", lambda root: _mutate_raw(root, _version)),
        ("summary-admission", _summary_gate),
    )
    rows = []
    with tempfile.TemporaryDirectory(prefix="r3-1-m0-tamper-") as temporary:
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
        raise ValueError("R3-1 M0 tamper probe was admitted")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = _probe(args.artifact.resolve())
    print(json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
