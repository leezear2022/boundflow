#!/usr/bin/env python3
"""Probe fully re-signed R3-2A trajectory and protocol tampering."""

# pylint: disable=protected-access,too-many-locals,too-many-statements
# pylint: disable=missing-function-docstring,duplicate-code

from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path
import shutil
import tempfile
from typing import Callable

import torch

from scripts.run_r3_optimizer_trajectory_artifact import (
    _canonical,
    _file_hash,
    _hash,
    _load,
    _tensor_hash,
    replay,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = ROOT / "artifacts/r3-structured-owner/r3-2a-optimizer-trajectory-v1"


def _write_json(path: Path, value: object) -> None:
    path.write_text(_canonical(value) + "\n")


def _resign_worker(raw: dict[str, object]) -> None:
    metadata = raw["trajectory_metadata"]
    values = raw["trajectory_raw"]
    assert isinstance(metadata, dict) and isinstance(values, dict)
    initial = values["initial_alpha"]
    terminal = values["terminal_alpha"]
    steps = values["steps"]
    assert torch.is_tensor(initial) and torch.is_tensor(terminal)
    assert isinstance(steps, list)
    metadata["initial_alpha_sha256"] = _tensor_hash(initial)
    metadata["terminal_alpha_sha256"] = _tensor_hash(terminal)
    metadata_steps = []
    for step in steps:
        assert isinstance(step, dict)
        step_meta = step["metadata"]
        assert isinstance(step_meta, dict)
        for raw_name, hash_name in (
            ("alpha_before", "alpha_before_sha256"),
            ("lower", "lower_sha256"),
            ("gradient", "gradient_sha256"),
            ("alpha_after", "alpha_after_sha256"),
        ):
            tensor = step[raw_name]
            assert torch.is_tensor(tensor)
            step_meta[hash_name] = _tensor_hash(tensor)
        optimizer = step_meta["optimizer_after"]
        assert isinstance(optimizer, dict)
        for raw_name, hash_name in (
            ("optimizer_exp_avg", "exp_avg_sha256"),
            ("optimizer_exp_avg_sq", "exp_avg_sq_sha256"),
        ):
            tensor = step[raw_name]
            assert torch.is_tensor(tensor)
            optimizer[hash_name] = _tensor_hash(tensor)
        metadata_steps.append(step_meta)
    metadata["steps"] = metadata_steps
    unsigned = dict(metadata)
    unsigned.pop("trajectory_hash", None)
    metadata["trajectory_hash"] = _hash(unsigned)


def _resign_manifest(artifact: Path) -> None:
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    files = manifest["files"]
    assert isinstance(files, dict)
    for name in files:
        files[name] = _file_hash(artifact / name)
    protocol = json.loads((artifact / "protocol.json").read_text())
    summary = json.loads((artifact / "summary.json").read_text())
    manifest["protocol_hash"] = protocol["protocol_hash"]
    manifest["summary_hash"] = summary["summary_hash"]
    unsigned = dict(manifest)
    unsigned.pop("manifest_hash", None)
    manifest["manifest_hash"] = _hash(unsigned)
    _write_json(manifest_path, manifest)


def _candidate_raw(artifact: Path) -> tuple[Path, dict[str, object]]:
    path = artifact / "raw/run-00-1-candidate.pt"
    return path, _load(path)


def _mutate_worker(
    artifact: Path, mutation: Callable[[dict[str, object]], None]
) -> None:
    path, raw = _candidate_raw(artifact)
    mutation(raw)
    _resign_worker(raw)
    torch.save(raw, path)
    _resign_manifest(artifact)


def _step(raw: dict[str, object], ordinal: int) -> dict[str, object]:
    values = raw["trajectory_raw"]
    assert isinstance(values, dict)
    steps = values["steps"]
    assert isinstance(steps, list) and isinstance(steps[ordinal], dict)
    return steps[ordinal]


def _tensor_delta(raw: dict[str, object], ordinal: int, name: str) -> None:
    step = _step(raw, ordinal)
    tensor = step[name]
    assert torch.is_tensor(tensor)
    changed = tensor.clone()
    changed.reshape(-1)[0] += 0.01
    step[name] = changed


def _receipt_value(raw: dict[str, object], name: str, value: object) -> None:
    metadata = _step(raw, 3)["metadata"]
    assert isinstance(metadata, dict)
    receipt = metadata["compiled_receipt"]
    assert isinstance(receipt, dict)
    receipt[name] = value


def _ordinal(raw: dict[str, object]) -> None:
    metadata = _step(raw, 4)["metadata"]
    assert isinstance(metadata, dict)
    metadata["evaluation_ordinal"] = 5
    rebind = metadata["rebind"]
    assert isinstance(rebind, dict)
    rebind["evaluation_ordinal"] = 5


def _final_update(raw: dict[str, object]) -> None:
    metadata = _step(raw, 9)["metadata"]
    assert isinstance(metadata, dict)
    metadata["update_after"] = True


def _immutable_identity(raw: dict[str, object]) -> None:
    metadata = raw["trajectory_metadata"]
    assert isinstance(metadata, dict)
    changed = "f" * 64
    metadata["immutable_content_hash"] = changed
    values = raw["trajectory_raw"]
    assert isinstance(values, dict)
    steps = values["steps"]
    assert isinstance(steps, list)
    for step in steps:
        assert isinstance(step, dict)
        step_meta = step["metadata"]
        assert isinstance(step_meta, dict)
        rebind = step_meta["rebind"]
        assert isinstance(rebind, dict)
        rebind["immutable_content_hash"] = changed


def _memory(raw: dict[str, object]) -> None:
    memory = raw["memory"]
    metadata = raw["trajectory_metadata"]
    assert isinstance(memory, dict) and isinstance(metadata, dict)
    memory["peak_allocated"] = int(memory["peak_allocated"]) * 100
    metadata["peak_allocated_bytes"] = memory["peak_allocated"]


def _summary_admission(artifact: Path) -> None:
    path = artifact / "summary.json"
    summary = json.loads(path.read_text())
    summary["trajectory_correctness_admitted"] = False
    unsigned = dict(summary)
    unsigned.pop("summary_hash", None)
    summary["summary_hash"] = _hash(unsigned)
    _write_json(path, summary)
    _resign_manifest(artifact)


def _protocol_tolerance(artifact: Path) -> None:
    path = artifact / "protocol.json"
    protocol = json.loads(path.read_text())
    protocol["state_tolerance"]["atol"] = 1.0
    unsigned = dict(protocol)
    unsigned.pop("protocol_hash", None)
    protocol["protocol_hash"] = _hash(unsigned)
    _write_json(path, protocol)
    _resign_manifest(artifact)


def probe(artifact: Path) -> dict[str, object]:
    worker_cases: dict[str, Callable[[dict[str, object]], None]] = {
        "intermediate_lower": lambda raw: _tensor_delta(raw, 4, "lower"),
        "intermediate_gradient": lambda raw: _tensor_delta(raw, 4, "gradient"),
        "alpha_after": lambda raw: _tensor_delta(raw, 4, "alpha_after"),
        "adam_exp_avg": lambda raw: _tensor_delta(raw, 4, "optimizer_exp_avg"),
        "saved_dense_a": lambda raw: _receipt_value(raw, "saved_dense_a_count", 1),
        "fallback_count": lambda raw: _receipt_value(raw, "fallback_count", 1),
        "evaluation_ordinal": _ordinal,
        "terminal_update": _final_update,
        "immutable_identity": _immutable_identity,
        "memory_peak": _memory,
    }
    cases: dict[str, Callable[[Path], None]] = {
        name: partial(_mutate_worker, mutation=mutation)
        for name, mutation in worker_cases.items()
    }
    cases.update(
        summary_admission=_summary_admission,
        protocol_tolerance=_protocol_tolerance,
    )
    rejected: dict[str, str] = {}
    for name, mutation in cases.items():
        root = Path(tempfile.mkdtemp(prefix=f"r3-2a-tamper-{name}-")) / "artifact"
        shutil.copytree(artifact, root)
        mutation(root)
        try:
            replay(root)
        except (ValueError, TypeError, RuntimeError) as error:
            rejected[name] = f"{type(error).__name__}: {error}"
        else:
            raise RuntimeError(f"R3-2A tamper was accepted: {name}")
        finally:
            shutil.rmtree(root.parent, ignore_errors=True)
    result: dict[str, object] = {
        "schema_version": "boundflow.r3-2a-tamper-report/v1",
        "case_count": len(cases),
        "rejected_count": len(rejected),
        "rejected": rejected,
        "all_rejected": len(rejected) == len(cases),
        "performance_claimed": False,
    }
    result["report_hash"] = _hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = probe(args.artifact.resolve())
    output = args.output or args.artifact / "tamper_report.json"
    _write_json(output.resolve(), result)
    print(
        f"R3-2A tamper PASS: rejected={result['rejected_count']}/{result['case_count']} "
        "performance_claimed=false"
    )


if __name__ == "__main__":
    main()
