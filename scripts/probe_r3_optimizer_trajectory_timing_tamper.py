#!/usr/bin/env python3
"""Probe fully re-signed R3-2B timing artifact tampering."""

# pylint: disable=protected-access,missing-function-docstring,duplicate-code
# pylint: disable=too-many-locals

from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path
import shutil
import statistics
import tempfile
from typing import Callable

import torch

from scripts.run_r3_optimizer_trajectory_timing_artifact import (
    _canonical,
    _file_hash,
    _hash,
    _load,
    _tensor_hash,
    replay,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = ROOT / "artifacts/r3-structured-owner/r3-2b-wrapper-timing-v1"


def _write(path: Path, value: object) -> None:
    path.write_text(_canonical(value) + "\n")


def _resign_manifest(root: Path) -> None:
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    for name in manifest["files"]:
        manifest["files"][name] = _file_hash(root / name)
    protocol = json.loads((root / "protocol.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    manifest["protocol_hash"] = protocol["protocol_hash"]
    manifest["summary_hash"] = summary["summary_hash"]
    unsigned = dict(manifest)
    unsigned.pop("manifest_hash", None)
    manifest["manifest_hash"] = _hash(unsigned)
    _write(path, manifest)


def _mutate_worker(root: Path, mutation: Callable[[dict[str, object]], None]) -> None:
    path = root / "raw/run-00-1-candidate.pt"
    raw = _load(path)
    mutation(raw)
    torch.save(raw, path)
    _resign_manifest(root)


def _latency(raw: dict[str, object]) -> None:
    samples = raw["latency_ns"]
    assert isinstance(samples, list)
    changed = [max(1, int(value) // 10) for value in samples]
    raw["latency_ns"] = changed
    raw["median_latency_ns"] = statistics.median(changed)


def _terminal(raw: dict[str, object], name: str) -> None:
    tensor = raw[name]
    assert torch.is_tensor(tensor)
    changed = tensor.clone()
    changed.reshape(-1)[0] += 0.01
    raw[name] = changed
    raw[f"{name}_sha256"] = _tensor_hash(changed)


def _execution(raw: dict[str, object]) -> None:
    receipt = raw["execution"]
    assert isinstance(receipt, dict)
    receipt["custom_forward_count"] = 9


def _memory(raw: dict[str, object]) -> None:
    receipt = raw["memory"]
    assert isinstance(receipt, dict)
    receipt["peak_allocated"] = int(receipt["peak_allocated"]) * 100


def _sample_count(raw: dict[str, object]) -> None:
    samples = raw["latency_ns"]
    assert isinstance(samples, list)
    raw["latency_ns"] = samples[:-1]
    raw["sample_count"] = 29
    raw["median_latency_ns"] = statistics.median(samples[:-1])


def _clock(raw: dict[str, object]) -> None:
    raw["clock"] = "cuda-event-only"


def _protocol_threshold(root: Path) -> None:
    path = root / "protocol.json"
    protocol = json.loads(path.read_text())
    protocol["geomean_threshold"] = 0.1
    unsigned = dict(protocol)
    unsigned.pop("protocol_hash", None)
    protocol["protocol_hash"] = _hash(unsigned)
    _write(path, protocol)
    _resign_manifest(root)


def _summary_value(root: Path, name: str, value: object) -> None:
    path = root / "summary.json"
    summary = json.loads(path.read_text())
    summary[name] = value
    unsigned = dict(summary)
    unsigned.pop("summary_hash", None)
    summary["summary_hash"] = _hash(unsigned)
    _write(path, summary)
    _resign_manifest(root)


def probe(artifact: Path) -> dict[str, object]:
    workers: dict[str, Callable[[dict[str, object]], None]] = {
        "latency_samples": _latency,
        "terminal_lower": partial(_terminal, name="terminal_lower"),
        "terminal_alpha": partial(_terminal, name="terminal_alpha"),
        "execution_counter": _execution,
        "memory_peak": _memory,
        "sample_count": _sample_count,
        "clock": _clock,
    }
    cases: dict[str, Callable[[Path], None]] = {
        name: partial(_mutate_worker, mutation=mutation)
        for name, mutation in workers.items()
    }
    cases.update(
        protocol_threshold=_protocol_threshold,
        summary_go=partial(_summary_value, name="r3_2b_go", value=True),
        summary_order=partial(_summary_value, name="order", value=[]),
    )
    rejected: dict[str, str] = {}
    for name, mutation in cases.items():
        root = Path(tempfile.mkdtemp(prefix=f"r3-2b-tamper-{name}-")) / "artifact"
        shutil.copytree(artifact, root)
        mutation(root)
        try:
            replay(root)
        except (ValueError, TypeError, RuntimeError) as error:
            rejected[name] = f"{type(error).__name__}: {error}"
        else:
            raise RuntimeError(f"R3-2B tamper was accepted: {name}")
        finally:
            shutil.rmtree(root.parent, ignore_errors=True)
    result: dict[str, object] = {
        "schema_version": "boundflow.r3-2b-tamper-report/v1",
        "case_count": len(cases),
        "rejected_count": len(rejected),
        "rejected": rejected,
        "all_rejected": len(rejected) == len(cases),
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
    _write(output.resolve(), result)
    print(
        f"R3-2B tamper PASS: rejected={result['rejected_count']}/{result['case_count']}"
    )


if __name__ == "__main__":
    main()
