#!/usr/bin/env python3
"""Probe fully re-signed R3-1b3 artifact semantic mutations."""

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
from scripts import run_r3_compiled_five_fresh_artifact as artifact


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


def _mutate_raw(root: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    path = root / artifact._raw_name(0, "candidate")
    row = artifact._load_torch(path)
    mutate(row)
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
    memory["peak_allocated"] *= 20
    memory["peak_allocated_increment"] = (
        memory["peak_allocated"] - memory["allocated_before"]
    )


def _receipt(row: dict[str, Any], name: str, value: object) -> None:
    cast(dict[str, object], row["execution_receipt"])[name] = value


def _summary(root: Path) -> None:
    summary = artifact._load_json(root / "summary.json")
    summary["r3_1_admitted"] = False
    summary["r3_2a_open"] = False
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


def _pair_order(root: Path) -> None:
    protocol = artifact._load_json(root / "protocol.json")
    protocol["pair_orders"][0] = "CN"
    protocol.pop("protocol_hash", None)
    protocol["protocol_hash"] = artifact._canonical_hash(protocol)
    _write_json(root / "protocol.json", protocol)
    manifest = artifact._load_json(root / "manifest.json")
    manifest["protocol_hash"] = protocol["protocol_hash"]
    _write_json(root / "manifest.json", manifest)


def probe(source: Path) -> dict[str, object]:
    probes: tuple[tuple[str, Callable[[Path], None]], ...] = (
        ("final-lower", lambda root: _mutate_raw(root, _lower)),
        ("gradient", lambda root: _mutate_raw(root, _gradient)),
        ("peak-allocated", lambda root: _mutate_raw(root, _memory)),
        (
            "saved-dense-a",
            lambda root: _mutate_raw(
                root, lambda row: _receipt(row, "saved_dense_a_count", 1)
            ),
        ),
        (
            "scratch-count",
            lambda root: _mutate_raw(
                root, lambda row: _receipt(row, "coefficient_scratch_count", 3)
            ),
        ),
        (
            "compiled-vjp",
            lambda root: _mutate_raw(
                root, lambda row: _receipt(row, "compiled_vjp", False)
            ),
        ),
        (
            "performance-claim",
            lambda root: _mutate_raw(
                root, lambda row: row.__setitem__("performance_claimed", True)
            ),
        ),
        ("summary-admission", _summary),
        ("pair-order", _pair_order),
    )
    rows = []
    with tempfile.TemporaryDirectory(prefix="r31b3-tamper-") as temporary:
        base = Path(temporary)
        for ordinal, (name, mutate) in enumerate(probes):
            target = base / f"probe-{ordinal:02d}"
            shutil.copytree(source, target)
            mutate(target)
            _resign_manifest(target)
            rejected = False
            error = ""
            try:
                artifact.replay(target)
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
        raise ValueError("R3-1b3 tamper probe was admitted")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            probe(args.artifact.resolve()), sort_keys=True, separators=(",", ":")
        )
    )


if __name__ == "__main__":
    main()
