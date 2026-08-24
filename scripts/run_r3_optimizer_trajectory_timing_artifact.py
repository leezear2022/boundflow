#!/usr/bin/env python3
"""Generate or replay the formal R3-2B wrapper timing artifact."""

# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=missing-function-docstring,duplicate-code,too-many-boolean-expressions
# pylint: disable=import-outside-toplevel

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
from typing import Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAPTURE = (
    ROOT / "artifacts/rvir-v4-pre-state/resnet2b-core-pre-state-v1/source_capture.pt"
)
DEFAULT_MODEL = Path(
    "/home/lee/Codes/vnncomp2021/benchmarks/cifar10_resnet/onnx/resnet_2b.onnx"
)
DEFAULT_OUTPUT = ROOT / "artifacts/r3-structured-owner/r3-2b-wrapper-timing-v1"
WORKER = ROOT / "scripts/run_r3_optimizer_trajectory_timing_worker.py"
PROTOCOL_SCHEMA = "boundflow.r3-2b-wrapper-timing-protocol/v1"
MANIFEST_SCHEMA = "boundflow.r3-2b-wrapper-timing-manifest/v1"
ORDER = (("native", "candidate"), ("candidate", "native")) * 2 + (
    ("native", "candidate"),
)
CODE_PATHS = (
    "boundflow/runtime/r3_optimizer_trajectory_timing.py",
    "boundflow/runtime/r3_compiled_p_alpha_vjp.py",
    "boundflow/runtime/r3_full_lower_forward_tir.py",
    "boundflow/backends/tvm/r3_p_alpha_vjp.py",
    "boundflow/backends/tvm/r3_full_lower_forward.py",
    "scripts/run_r3_optimizer_trajectory_timing_worker.py",
    "scripts/run_r3_optimizer_trajectory_timing_artifact.py",
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


def _tensor_hash(value: torch.Tensor) -> str:
    from boundflow.runtime.rvir_v4_production_state import production_tensor_sha256

    return production_tensor_sha256(value)


def _git(*args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def _tracked_source_is_clean() -> bool:
    status = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=no"),
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    rows = status.splitlines()
    return all(
        row.endswith(" .docops/ev.jsonl") and row[:2].strip() == "M" for row in rows
    )


def _load(path: Path) -> dict[str, object]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise TypeError("R3-2B raw root differs")
    return value


def _validate_worker(raw: Mapping[str, object]) -> None:
    expected = {
        "schema_version",
        "run_index",
        "mode",
        "source_capture_sha256",
        "model_sha256",
        "plan_hash",
        "trace_hash",
        "warmup_count",
        "sample_count",
        "latency_ns",
        "median_latency_ns",
        "terminal_lower",
        "terminal_alpha",
        "terminal_lower_sha256",
        "terminal_alpha_sha256",
        "execution",
        "memory",
        "environment",
        "clock",
        "performance_claimed",
    }
    if (
        set(raw) != expected
        or raw["schema_version"] != "boundflow.r3-2b-wrapper-timing-worker/v1"
        or raw["mode"] not in {"native", "candidate"}
        or raw["warmup_count"] != 3
        or raw["sample_count"] != 30
        or raw["clock"] != "host-perf-counter-ns-with-device-boundary-sync"
        or raw["performance_claimed"] is not False
    ):
        raise ValueError("R3-2B worker envelope differs")
    samples = raw["latency_ns"]
    if (
        not isinstance(samples, list)
        or len(samples) != 30
        or any(not isinstance(value, int) or value <= 0 for value in samples)
        or raw["median_latency_ns"] != statistics.median(samples)
    ):
        raise ValueError("R3-2B latency samples differ")
    lower = raw["terminal_lower"]
    alpha = raw["terminal_alpha"]
    if (
        not torch.is_tensor(lower)
        or not torch.is_tensor(alpha)
        or tuple(lower.shape) != (6, 1)
        or tuple(alpha.shape) != (2, 1, 6, 86)
        or not bool(torch.isfinite(lower).all())
        or not bool(torch.isfinite(alpha).all())
        or _tensor_hash(lower) != raw["terminal_lower_sha256"]
        or _tensor_hash(alpha) != raw["terminal_alpha_sha256"]
    ):
        raise ValueError("R3-2B terminal raw differs")
    execution = raw["execution"]
    if not isinstance(execution, dict):
        raise TypeError("R3-2B execution receipt differs")
    expected_execution = {
        "evaluation_count": 10,
        "optimizer_mutation_count": 9,
        "scheduler_mutation_count": 9,
        "custom_forward_count": 10 if raw["mode"] == "candidate" else 0,
        "custom_backward_count": 10 if raw["mode"] == "candidate" else 0,
        "fallback_count": 0,
        "eager_candidate_count": 0,
        "native_shadow_count": 0,
        "timing_capture_count": 0,
    }
    if execution != expected_execution:
        raise ValueError("R3-2B execution counters differ")
    memory = raw["memory"]
    if not isinstance(memory, dict) or set(memory) != {
        "allocated_before",
        "reserved_before",
        "peak_allocated",
        "peak_reserved",
    }:
        raise TypeError("R3-2B memory receipt differs")
    if any(not isinstance(value, int) or value < 0 for value in memory.values()):
        raise ValueError("R3-2B memory value differs")


def _compare_pair(
    native: Mapping[str, object], candidate: Mapping[str, object]
) -> dict[str, float]:
    _validate_worker(native)
    _validate_worker(candidate)
    if (
        native["mode"] != "native"
        or candidate["mode"] != "candidate"
        or native["run_index"] != candidate["run_index"]
    ):
        raise ValueError("R3-2B pair identity differs")
    for name in (
        "source_capture_sha256",
        "model_sha256",
        "plan_hash",
        "trace_hash",
        "environment",
    ):
        if native[name] != candidate[name]:
            raise ValueError(f"R3-2B pair {name} differs")
    nlower, clower = native["terminal_lower"], candidate["terminal_lower"]
    nalpha, calpha = native["terminal_alpha"], candidate["terminal_alpha"]
    assert torch.is_tensor(nlower) and torch.is_tensor(clower)
    assert torch.is_tensor(nalpha) and torch.is_tensor(calpha)
    lower_diff = float((nlower - clower).abs().max())
    alpha_diff = float((nalpha - calpha).abs().max())
    if not torch.allclose(nlower, clower, atol=2e-4, rtol=2e-4) or not torch.equal(
        torch.sign(nlower), torch.sign(clower)
    ):
        raise ValueError("R3-2B terminal lower differs")
    if not torch.allclose(nalpha, calpha, atol=2e-5, rtol=2e-5):
        raise ValueError("R3-2B terminal alpha differs")
    native_median = float(native["median_latency_ns"])  # type: ignore[arg-type]
    candidate_median = float(candidate["median_latency_ns"])  # type: ignore[arg-type]
    nmem, cmem = native["memory"], candidate["memory"]
    assert isinstance(nmem, dict) and isinstance(cmem, dict)
    allocated_ratio = float(cmem["peak_allocated"]) / float(nmem["peak_allocated"])
    reserved_ratio = float(cmem["peak_reserved"]) / float(nmem["peak_reserved"])
    if allocated_ratio > 1.0 or reserved_ratio > 1.0:
        raise ValueError("R3-2B memory gate differs")
    return {
        "native_median_ns": native_median,
        "candidate_median_ns": candidate_median,
        "speedup": native_median / candidate_median,
        "lower_diff": lower_diff,
        "alpha_diff": alpha_diff,
        "allocated_ratio": allocated_ratio,
        "reserved_ratio": reserved_ratio,
    }


def _summary(raws: list[dict[str, object]]) -> dict[str, object]:
    pairs = []
    for run_index in range(5):
        by_mode = {raw["mode"]: raw for raw in raws if raw["run_index"] == run_index}
        if set(by_mode) != {"native", "candidate"}:
            raise ValueError("R3-2B pair mode inventory differs")
        pairs.append(_compare_pair(by_mode["native"], by_mode["candidate"]))
    speedups = [row["speedup"] for row in pairs]
    geomean = math.exp(sum(math.log(value) for value in speedups) / len(speedups))
    worst = min(speedups)
    go = geomean >= 1.20 and worst >= 0.98
    result: dict[str, object] = {
        "pair_count": 5,
        "worker_count": 10,
        "warmup_count_per_worker": 3,
        "sample_count_per_worker": 30,
        "order": [list(value) for value in ORDER],
        "pair_metrics": pairs,
        "geomean_speedup": geomean,
        "worst_pair_speedup": worst,
        "maximum_lower_max_abs_diff": max(row["lower_diff"] for row in pairs),
        "maximum_alpha_max_abs_diff": max(row["alpha_diff"] for row in pairs),
        "worst_allocated_ratio": max(row["allocated_ratio"] for row in pairs),
        "worst_reserved_ratio": max(row["reserved_ratio"] for row in pairs),
        "go_threshold_geomean": 1.20,
        "go_threshold_worst": 0.98,
        "r3_2b_go": go,
        "single_site_killed": not go,
        "r3_3_open": go,
        "performance_claimed": go,
    }
    result["summary_hash"] = _hash(result)
    return result


def _protocol(source_revision: str, capture: Path, model: Path) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": PROTOCOL_SCHEMA,
        "source_revision": source_revision,
        "source_capture": {
            "repo_path": str(capture.relative_to(ROOT)),
            "sha256": _file_hash(capture),
        },
        "model": {
            "public_id": "vnncomp2021/cifar10_resnet/resnet_2b.onnx",
            "sha256": _file_hash(model),
        },
        "pair_order": [list(value) for value in ORDER],
        "warmup_count": 3,
        "sample_count": 30,
        "evaluation_count": 10,
        "mutation_count": 9,
        "clock": "host-perf-counter-ns-with-device-boundary-sync",
        "geomean_threshold": 1.20,
        "worst_pair_threshold": 0.98,
        "memory_ratio_max": 1.0,
        "code_revision": {name: _file_hash(ROOT / name) for name in CODE_PATHS},
    }
    payload["protocol_hash"] = _hash(payload)
    return payload


def _validate_protocol(
    protocol: Mapping[str, object], manifest: Mapping[str, object]
) -> None:
    expected = {
        "schema_version": PROTOCOL_SCHEMA,
        "pair_order": [list(value) for value in ORDER],
        "warmup_count": 3,
        "sample_count": 30,
        "evaluation_count": 10,
        "mutation_count": 9,
        "clock": "host-perf-counter-ns-with-device-boundary-sync",
        "geomean_threshold": 1.20,
        "worst_pair_threshold": 0.98,
        "memory_ratio_max": 1.0,
    }
    if any(protocol.get(name) != value for name, value in expected.items()):
        raise ValueError("R3-2B frozen protocol semantics differ")
    if protocol.get("source_revision") != manifest.get("source_revision"):
        raise ValueError("R3-2B source revision differs")
    if protocol.get("code_revision") != {
        name: _file_hash(ROOT / name) for name in CODE_PATHS
    }:
        raise ValueError("R3-2B code revision differs")


def generate(output: Path, capture: Path, model: Path) -> None:
    if output.exists():
        raise FileExistsError(f"R3-2B artifact output already exists: {output}")
    if not _tracked_source_is_clean():
        raise RuntimeError("R3-2B formal generation requires a clean worktree")
    revision = _git("rev-parse", "HEAD")
    temp = Path(tempfile.mkdtemp(prefix="r3-2b-artifact-", dir=output.parent))
    raws = []
    try:
        raw_dir = temp / "raw"
        raw_dir.mkdir(parents=True)
        for run_index, pair in enumerate(ORDER):
            for sequence, mode in enumerate(pair):
                target = raw_dir / f"run-{run_index:02d}-{sequence}-{mode}.pt"
                subprocess.run(
                    (
                        sys.executable,
                        str(WORKER),
                        "--source-capture",
                        str(capture),
                        "--model",
                        str(model),
                        "--mode",
                        mode,
                        "--run-index",
                        str(run_index),
                        "--result",
                        str(target),
                    ),
                    cwd=ROOT,
                    check=True,
                    env=os.environ.copy(),
                )
                raws.append(_load(target))
        summary = _summary(raws)
        protocol = _protocol(revision, capture, model)
        (temp / "protocol.json").write_text(_canonical(protocol) + "\n")
        (temp / "summary.json").write_text(_canonical(summary) + "\n")
        files = sorted(
            path.relative_to(temp).as_posix()
            for path in temp.rglob("*")
            if path.is_file()
        )
        manifest: dict[str, object] = {
            "schema_version": MANIFEST_SCHEMA,
            "source_revision": revision,
            "protocol_hash": protocol["protocol_hash"],
            "summary_hash": summary["summary_hash"],
            "files": {name: _file_hash(temp / name) for name in files},
        }
        manifest["manifest_hash"] = _hash(manifest)
        (temp / "manifest.json").write_text(_canonical(manifest) + "\n")
        replay(temp)
        temp.rename(output)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def replay(artifact: Path) -> dict[str, object]:
    manifest = json.loads((artifact / "manifest.json").read_text())
    unsigned_manifest = dict(manifest)
    manifest_hash = unsigned_manifest.pop("manifest_hash", None)
    if (
        manifest_hash != _hash(unsigned_manifest)
        or manifest.get("schema_version") != MANIFEST_SCHEMA
    ):
        raise ValueError("R3-2B manifest differs")
    files = manifest.get("files")
    expected_files = {"protocol.json", "summary.json"} | {
        f"raw/run-{run_index:02d}-{sequence}-{mode}.pt"
        for run_index, pair in enumerate(ORDER)
        for sequence, mode in enumerate(pair)
    }
    if (
        not isinstance(files, dict)
        or set(files) != expected_files
        or any(_file_hash(artifact / name) != digest for name, digest in files.items())
    ):
        raise ValueError("R3-2B manifest file digest differs")
    protocol = json.loads((artifact / "protocol.json").read_text())
    unsigned_protocol = dict(protocol)
    protocol_hash = unsigned_protocol.pop("protocol_hash", None)
    if (
        protocol_hash != _hash(unsigned_protocol)
        or protocol_hash != manifest["protocol_hash"]
    ):
        raise ValueError("R3-2B protocol hash differs")
    _validate_protocol(protocol, manifest)
    raws = [_load(path) for path in sorted((artifact / "raw").glob("*.pt"))]
    summary = _summary(raws)
    if (
        summary != json.loads((artifact / "summary.json").read_text())
        or summary["summary_hash"] != manifest["summary_hash"]
    ):
        raise ValueError("R3-2B semantic summary replay differs")
    print(
        f"R3-2B replay PASS: geomean={summary['geomean_speedup']} "
        f"worst={summary['worst_pair_speedup']} go={summary['r3_2b_go']}"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args()
    if args.replay:
        replay(args.replay.resolve())
    else:
        generate(
            args.output.resolve(), args.source_capture.resolve(), args.model.resolve()
        )


if __name__ == "__main__":
    main()
