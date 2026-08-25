#!/usr/bin/env python3
"""Generate or replay the five-fresh D1-B fixed schedule artifact."""

# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=missing-function-docstring,duplicate-code,too-many-boolean-expressions

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts/r3-structured-owner/r3-d1b-schedule-formal-v1"
WORKER = ROOT / "scripts/run_r3_d1b_schedule_worker.py"
CALIBRATION = ROOT / "artifacts/r3-structured-owner/r3-d1b-serial-calibration-v1.json"
RESIDUAL11 = ROOT / "artifacts/r3-structured-owner/r3-d1a-residual11-staged-v1"
RESIDUAL6 = ROOT / "artifacts/r3-structured-owner/r3-d1a-residual6-staged-v1"
RUN_COUNT = 5
GATE = 15.5
CODE_PATHS = (
    "boundflow/backends/tvm/r3_d1b_serial_schedule.py",
    "scripts/probe_r3_d1b_serial_schedule.py",
    "scripts/run_r3_d1b_schedule_worker.py",
    "scripts/run_r3_d1b_schedule_artifact.py",
)


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("R3-D1B JSON differs")
    return value


def _write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _clean() -> None:
    dirty = [
        line
        for line in _git("status", "--porcelain").splitlines()
        if not line.endswith("docs/CIBC_for_DAC.pdf") and ".docops/ev.jsonl" not in line
    ]
    if dirty:
        raise RuntimeError(f"R3-D1B formal source is dirty: {dirty}")


def _median(values: object) -> float:
    if not isinstance(values, list) or len(values) != 10:
        raise ValueError("R3-D1B timing sample inventory differs")
    numbers = [float(value) for value in values]
    if any(not math.isfinite(value) or value <= 0.0 for value in numbers):
        raise ValueError("R3-D1B timing sample differs")
    return statistics.median(numbers)


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "source_git_head",
        "run_count",
        "warmup_count",
        "sample_count",
        "winner_threads_per_block",
        "winner_schedule_kind",
        "isolated_opportunity_gate",
        "residual11_manifest_sha256",
        "residual6_manifest_sha256",
        "calibration_sha256",
        "code_revision",
        "wrapper_performance_claimed",
        "protocol_hash",
    }
    if (
        set(protocol) != required
        or protocol["schema_version"] != "boundflow.r3-d1b-schedule-protocol/v1"
        or protocol["run_count"] != RUN_COUNT
        or protocol["warmup_count"] != 2
        or protocol["sample_count"] != 10
        or protocol["winner_threads_per_block"] != 256
        or protocol["winner_schedule_kind"] != "two-kernel-serial-reduction"
        or protocol["isolated_opportunity_gate"] != GATE
        or protocol["wrapper_performance_claimed"] is not False
    ):
        raise ValueError("R3-D1B protocol contract differs")


def _validate_raw(
    raw: Mapping[str, Any], protocol: Mapping[str, Any]
) -> dict[str, float]:
    expected = {
        "schema_version",
        "run_index",
        "source_git_head",
        "residual11_manifest_sha256",
        "residual6_manifest_sha256",
        "calibration_sha256",
        "environment",
        "measurement",
        "winner_frozen",
        "isolated_opportunity_gate",
        "isolated_performance_claimed",
        "wrapper_performance_claimed",
    }
    if (
        set(raw) != expected
        or raw["schema_version"] != "boundflow.r3-d1b-schedule-worker/v1"
        or raw["source_git_head"] != protocol["source_git_head"]
        or raw["residual11_manifest_sha256"] != protocol["residual11_manifest_sha256"]
        or raw["residual6_manifest_sha256"] != protocol["residual6_manifest_sha256"]
        or raw["calibration_sha256"] != protocol["calibration_sha256"]
        or raw["winner_frozen"] is not True
        or raw["isolated_opportunity_gate"] != GATE
        or raw["isolated_performance_claimed"] is not True
        or raw["wrapper_performance_claimed"] is not False
    ):
        raise ValueError("R3-D1B worker envelope differs")
    row = raw["measurement"]
    if not isinstance(row, Mapping):
        raise TypeError("R3-D1B measurement differs")
    required = {
        "threads_per_block",
        "schedule_kind",
        "baseline_ms",
        "candidate_ms",
        "baseline_median_ms",
        "candidate_median_ms",
        "speedup",
        "maximum_diff",
        "sign_exact",
        "candidate_scheduled_tir_hash",
        "candidate_device_source_hash",
        "baseline_scheduled_tir_hash",
        "baseline_device_source_hash",
        "launch_count",
        "scratch_count",
        "persistent_dense_a",
    }
    if (
        set(row) != required
        or row["threads_per_block"] != 256
        or row["schedule_kind"] != "two-kernel-serial-reduction"
        or row["launch_count"] != 4
        or row["scratch_count"] != 2
        or row["persistent_dense_a"] is not False
        or row["sign_exact"] is not True
        or float(row["maximum_diff"]) > 2.0e-4
    ):
        raise ValueError("R3-D1B schedule or semantic receipt differs")
    hashes = (
        row["candidate_scheduled_tir_hash"],
        row["candidate_device_source_hash"],
        row["baseline_scheduled_tir_hash"],
        row["baseline_device_source_hash"],
    )
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in hashes
    ):
        raise ValueError("R3-D1B compiler receipt differs")
    baseline = _median(row["baseline_ms"])
    candidate = _median(row["candidate_ms"])
    speedup = baseline / candidate
    if (
        abs(baseline - float(row["baseline_median_ms"])) > 1.0e-12
        or abs(candidate - float(row["candidate_median_ms"])) > 1.0e-12
        or abs(speedup - float(row["speedup"])) > 1.0e-12
    ):
        raise ValueError("R3-D1B timing derivation differs")
    return {
        "baseline_median_ms": baseline,
        "candidate_median_ms": candidate,
        "speedup": speedup,
    }


def _summarize(
    raws: list[dict[str, Any]], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    if len(raws) != RUN_COUNT or [raw["run_index"] for raw in raws] != list(
        range(RUN_COUNT)
    ):
        raise ValueError("R3-D1B fresh process inventory differs")
    metrics = [_validate_raw(raw, protocol) for raw in raws]
    receipt_names = (
        "candidate_scheduled_tir_hash",
        "candidate_device_source_hash",
        "baseline_scheduled_tir_hash",
        "baseline_device_source_hash",
    )
    receipts = {
        tuple(raw["measurement"][name] for name in receipt_names) for raw in raws
    }
    environments = {_canonical(raw["environment"]) for raw in raws}
    if len(receipts) != 1 or len(environments) != 1:
        raise ValueError("R3-D1B fresh compiler/environment receipt differs")
    speedups = [metric["speedup"] for metric in metrics]
    summary: dict[str, Any] = {
        "schema_version": "boundflow.r3-d1b-schedule-summary/v1",
        "run_count": RUN_COUNT,
        "winner_threads_per_block": 256,
        "winner_schedule_kind": "two-kernel-serial-reduction",
        "metrics": metrics,
        "geomean_speedup": math.prod(speedups) ** (1.0 / len(speedups)),
        "worst_speedup": min(speedups),
        "isolated_opportunity_gate": GATE,
        "isolated_gate_pass": min(speedups) >= GATE,
        "correctness_pass": True,
        "d1b_schedule_qualification": True,
        "d1c_wrapper_open": min(speedups) >= GATE,
        "isolated_performance_claimed": True,
        "wrapper_performance_claimed": False,
    }
    summary["summary_hash"] = _hash(summary)
    return summary


def _protocol() -> dict[str, Any]:
    calibration = _json(CALIBRATION)
    if (
        calibration.get("winner_threads_per_block") != 256
        or calibration.get("winner_gate_pass") is not True
    ):
        raise ValueError("R3-D1B calibration admission differs")
    protocol: dict[str, Any] = {
        "schema_version": "boundflow.r3-d1b-schedule-protocol/v1",
        "source_git_head": _git("rev-parse", "HEAD"),
        "run_count": RUN_COUNT,
        "warmup_count": 2,
        "sample_count": 10,
        "winner_threads_per_block": 256,
        "winner_schedule_kind": "two-kernel-serial-reduction",
        "isolated_opportunity_gate": GATE,
        "residual11_manifest_sha256": _file_hash(RESIDUAL11 / "manifest.json"),
        "residual6_manifest_sha256": _file_hash(RESIDUAL6 / "manifest.json"),
        "calibration_sha256": _file_hash(CALIBRATION),
        "code_revision": {path: _file_hash(ROOT / path) for path in CODE_PATHS},
        "wrapper_performance_claimed": False,
    }
    protocol["protocol_hash"] = _hash(protocol)
    return protocol


def generate(output: Path) -> dict[str, Any]:
    _clean()
    protocol = _protocol()
    with tempfile.TemporaryDirectory(prefix="boundflow-r3d1b-formal-") as temporary:
        temporary_root = Path(temporary)
        raw_paths = []
        for ordinal in range(RUN_COUNT):
            path = temporary_root / f"run-{ordinal:02d}.json"
            subprocess.run(
                (
                    sys.executable,
                    str(WORKER),
                    "--run-index",
                    str(ordinal),
                    "--result",
                    str(path),
                ),
                cwd=ROOT,
                check=True,
            )
            raw_paths.append(path)
        raws = [_json(path) for path in raw_paths]
        summary = _summarize(raws, protocol)
        if output.exists():
            shutil.rmtree(output)
        (output / "raw").mkdir(parents=True)
        for path in raw_paths:
            shutil.copy2(path, output / "raw" / path.name)
    _write(output / "protocol.json", protocol)
    _write(output / "summary.json", summary)
    files = {
        str(path.relative_to(output)): _file_hash(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    manifest: dict[str, Any] = {
        "schema_version": "boundflow.r3-d1b-schedule-manifest/v1",
        "protocol_hash": protocol["protocol_hash"],
        "summary_hash": summary["summary_hash"],
        "files": files,
    }
    manifest["manifest_hash"] = _hash(manifest)
    _write(output / "manifest.json", manifest)
    return replay(output)


def replay(output: Path) -> dict[str, Any]:
    protocol = _json(output / "protocol.json")
    summary = _json(output / "summary.json")
    manifest = _json(output / "manifest.json")
    protocol_copy = dict(protocol)
    protocol_hash = protocol_copy.pop("protocol_hash", None)
    if protocol_hash != _hash(protocol_copy):
        raise ValueError("R3-D1B protocol hash differs")
    _validate_protocol(protocol)
    summary_copy = dict(summary)
    summary_hash = summary_copy.pop("summary_hash", None)
    if summary_hash != _hash(summary_copy):
        raise ValueError("R3-D1B summary hash differs")
    manifest_copy = dict(manifest)
    manifest_hash = manifest_copy.pop("manifest_hash", None)
    if manifest_hash != _hash(manifest_copy):
        raise ValueError("R3-D1B manifest hash differs")
    if (
        manifest.get("protocol_hash") != protocol_hash
        or manifest.get("summary_hash") != summary_hash
    ):
        raise ValueError("R3-D1B manifest linkage differs")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or any(
        _file_hash(output / name) != digest for name, digest in files.items()
    ):
        raise ValueError("R3-D1B file digest differs")
    raws = [_json(path) for path in sorted((output / "raw").glob("*.json"))]
    if _summarize(raws, protocol) != summary:
        raise ValueError("R3-D1B semantic replay differs")
    print(
        f"R3-D1B replay PASS: geomean={summary['geomean_speedup']:.4f}x "
        f"worst={summary['worst_speedup']:.4f}x wrapper_performance_claimed=false",
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    if args.replay:
        replay(args.output.absolute())
    else:
        generate(args.output.absolute())


if __name__ == "__main__":
    main()
