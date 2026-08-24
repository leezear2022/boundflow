#!/usr/bin/env python3
"""Generate or replay the formal R3-D0 microphysics attribution artifact."""

# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=too-many-boolean-expressions,missing-function-docstring
# pylint: disable=import-outside-toplevel

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAPTURE = (
    ROOT / "artifacts/rvir-v4-pre-state/resnet2b-core-pre-state-v1/source_capture.pt"
)
DEFAULT_MODEL = Path(
    "/home/lee/Codes/vnncomp2021/benchmarks/cifar10_resnet/onnx/resnet_2b.onnx"
)
DEFAULT_OUTPUT = ROOT / "artifacts/r3-structured-owner/r3-d0-microphysics-v1"
R32B_ARTIFACT = ROOT / "artifacts/r3-structured-owner/r3-2b-wrapper-timing-v1"
WORKER = ROOT / "scripts/run_r3_d0_microphysics_worker.py"
PROTOCOL_SCHEMA = "boundflow.r3-d0-microphysics-protocol/v1"
MANIFEST_SCHEMA = "boundflow.r3-d0-microphysics-manifest/v1"
ORDER = (("native", "candidate"), ("candidate", "native")) * 2 + (
    ("native", "candidate"),
)
CODE_PATHS = (
    "boundflow/runtime/r3_d0_microphysics_attribution.py",
    "scripts/run_r3_d0_microphysics_worker.py",
    "scripts/run_r3_d0_microphysics_artifact.py",
    "boundflow/runtime/r3_optimizer_trajectory_timing.py",
    "boundflow/runtime/r3_compiled_p_alpha_vjp.py",
    "boundflow/runtime/r3_full_lower_forward_tir.py",
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


def _git(*args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def _tracked_source_is_clean() -> bool:
    rows = _git("status", "--porcelain", "--untracked-files=no").splitlines()
    return all(
        row.endswith(" .docops/ev.jsonl") and row[:2].strip() == "M" for row in rows
    )


def _load(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise TypeError("R3-D0 raw root differs")
    return value


def _json_load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("R3-D0 JSON root differs")
    return value


def _json_write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _tensor_hash(value: torch.Tensor) -> str:
    from boundflow.runtime.rvir_v4_production_state import production_tensor_sha256

    return production_tensor_sha256(value)


def _validate_worker(raw: Mapping[str, Any]) -> dict[str, object]:
    from boundflow.runtime.r3_d0_microphysics_attribution import (
        canonical_hash,
        derive_worker_ledger,
        event_from_dict,
    )

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
        "profiled_host_wall_ns",
        "cuda_event_elapsed_ns",
        "terminal_lower",
        "terminal_alpha",
        "terminal_lower_sha256",
        "terminal_alpha_sha256",
        "events",
        "event_hash",
        "ledger",
        "environment",
        "performance_claimed",
    }
    if (
        set(raw) != expected
        or raw["schema_version"] != "boundflow.r3-d0-microphysics-worker/v1"
        or raw["mode"] not in {"native", "candidate"}
        or raw["warmup_count"] != 3
        or raw["sample_count"] != 30
        or raw["performance_claimed"] is not False
    ):
        raise ValueError("R3-D0 worker envelope differs")
    samples = raw["latency_ns"]
    if (
        not isinstance(samples, list)
        or len(samples) != 30
        or any(not isinstance(value, int) or value <= 0 for value in samples)
        or raw["median_latency_ns"] != round(statistics.median(samples))
    ):
        raise ValueError("R3-D0 latency sample differs")
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
        raise ValueError("R3-D0 terminal tensor differs")
    rows = raw["events"]
    if not isinstance(rows, list):
        raise TypeError("R3-D0 event rows differ")
    events = tuple(event_from_dict(row) for row in rows)
    if canonical_hash(rows) != raw["event_hash"]:
        raise ValueError("R3-D0 event hash differs")
    rebuilt = derive_worker_ledger(
        events,
        mode=str(raw["mode"]),
        unprofiled_median_ns=int(raw["median_latency_ns"]),
        profiled_host_wall_ns=int(raw["profiled_host_wall_ns"]),
        cuda_event_elapsed_ns=int(raw["cuda_event_elapsed_ns"]),
    )
    if rebuilt != raw["ledger"] or rebuilt["calibration_admitted"] is not True:
        raise ValueError("R3-D0 worker ledger replay differs")
    return rebuilt


def _reference_raw(pair: int, mode: str) -> dict[str, Any]:
    position = ORDER[pair].index(mode)
    return _load(R32B_ARTIFACT / f"raw/run-{pair:02d}-{position}-{mode}.pt")


def _summarize(raws: list[dict[str, Any]]) -> dict[str, object]:
    from boundflow.runtime.r3_d0_microphysics_attribution import derive_pair_route

    if len(raws) != 10:
        raise ValueError("R3-D0 worker count differs")
    pair_metrics = []
    for pair, order in enumerate(ORDER):
        by_mode = {
            mode: next(
                raw for raw in raws if raw["run_index"] == pair and raw["mode"] == mode
            )
            for mode in order
        }
        native = by_mode["native"]
        candidate = by_mode["candidate"]
        native_ledger = _validate_worker(native)
        candidate_ledger = _validate_worker(candidate)
        for mode, raw in by_mode.items():
            reference = _reference_raw(pair, mode)
            reference_median = float(reference["median_latency_ns"])
            drift = abs(float(raw["median_latency_ns"]) / reference_median - 1.0)
            if drift > 0.15:
                raise ValueError("R3-D0 unprofiled sanity differs from R3-2B")
        lower_diff = float(
            (native["terminal_lower"] - candidate["terminal_lower"]).abs().max()
        )
        alpha_diff = float(
            (native["terminal_alpha"] - candidate["terminal_alpha"]).abs().max()
        )
        if (
            lower_diff > 2.0e-4
            or alpha_diff > 2.0e-6
            or not torch.equal(
                torch.sign(native["terminal_lower"]),
                torch.sign(candidate["terminal_lower"]),
            )
        ):
            raise ValueError("R3-D0 native/candidate semantics differ")
        route = derive_pair_route(native_ledger, candidate_ledger)
        pair_metrics.append(
            {
                "pair_index": pair,
                "order": list(order),
                "native_median_ns": native["median_latency_ns"],
                "candidate_median_ns": candidate["median_latency_ns"],
                "lower_max_abs_diff": lower_diff,
                "alpha_max_abs_diff": alpha_diff,
                "native_kernel_union_ns": native_ledger["kernel_union_ns"],
                "candidate_kernel_union_ns": candidate_ledger["kernel_union_ns"],
                "native_host_residual_ns": native_ledger["host_residual_ns"],
                "candidate_host_residual_ns": candidate_ledger["host_residual_ns"],
                "candidate_compiled_region": candidate_ledger["compiled_region"],
                "route": route,
            }
        )
    compiled_routes = [
        metric["route"]["compiled_region_route"] for metric in pair_metrics
    ]
    all_compiled = all(route["within_10x"] for route in compiled_routes)
    all_graph = all(metric["route"]["graph_physical"] for metric in pair_metrics)
    verdict = (
        "VALIDATED-R3-D0-COMPILED-REGION-SCHEDULE-OPPORTUNITY"
        if all_compiled
        else "VALIDATED-NO-GO-R3-SO-CVJP-PERFORMANCE"
    )
    summary: dict[str, object] = {
        "pair_count": 5,
        "worker_count": 10,
        "order": [list(value) for value in ORDER],
        "warmup_count_per_worker": 3,
        "sample_count_per_worker": 30,
        "pair_metrics": pair_metrics,
        "all_graph_physical": all_graph,
        "graph_route_open": False,
        "all_compiled_region_within_10x": all_compiled,
        "worst_compiled_region_required_speedup": max(
            float(route["required_speedup"]) for route in compiled_routes
        ),
        "verdict": verdict,
        "r3_d1_open": all_compiled,
        "r3_3_open": False,
        "performance_claimed": False,
    }
    summary["summary_hash"] = _hash(summary)
    return summary


def _protocol() -> dict[str, object]:
    protocol: dict[str, object] = {
        "schema_version": PROTOCOL_SCHEMA,
        "source_git_head": _git("rev-parse", "HEAD"),
        "code_revision": {path: _file_hash(ROOT / path) for path in CODE_PATHS},
        "order": [list(value) for value in ORDER],
        "warmup_count": 3,
        "sample_count": 30,
        "target_speedup": 1.20,
        "sanity_relative_tolerance": 0.15,
        "host_calibration_threshold": "max(5ms,5pct-host-wall)",
        "cuda_calibration_threshold": "max(2ms,5pct-cuda-event)",
        "containment_fallback_max_fraction": 0.05,
        "unattributed_kernel_max": 0,
        "max_required_region_speedup": 10.0,
        "r3_2b_manifest_file_sha256": _file_hash(R32B_ARTIFACT / "manifest.json"),
        "source_capture_sha256": _file_hash(DEFAULT_CAPTURE),
        "model_sha256": _file_hash(DEFAULT_MODEL),
        "performance_claimed": False,
    }
    protocol["protocol_hash"] = _hash(protocol)
    return protocol


def generate(output: Path, capture: Path, model: Path) -> None:
    if output.exists():
        raise FileExistsError(f"R3-D0 artifact output already exists: {output}")
    if not _tracked_source_is_clean():
        raise RuntimeError("R3-D0 formal generation requires a clean worktree")
    protocol = _protocol()
    with tempfile.TemporaryDirectory(prefix="boundflow-r3d0-formal-") as temporary:
        root = Path(temporary)
        raw_root = root / "raw"
        raw_root.mkdir()
        raws = []
        for pair, order in enumerate(ORDER):
            for position, mode in enumerate(order):
                result = raw_root / f"run-{pair:02d}-{position}-{mode}.pt"
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
                        str(pair),
                        "--result",
                        str(result),
                    ),
                    cwd=ROOT,
                    check=True,
                )
                raws.append(_load(result))
        summary = _summarize(raws)
        _json_write(root / "protocol.json", protocol)
        _json_write(root / "summary.json", summary)
        files = {
            str(path.relative_to(root)): _file_hash(path)
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }
        manifest: dict[str, object] = {
            "schema_version": MANIFEST_SCHEMA,
            "source_git_head": protocol["source_git_head"],
            "protocol_hash": protocol["protocol_hash"],
            "summary_hash": summary["summary_hash"],
            "files": files,
            "performance_claimed": False,
        }
        manifest["manifest_hash"] = _hash(manifest)
        _json_write(root / "manifest.json", manifest)
        shutil.copytree(root, output)
    replay(output)


def replay(output: Path) -> dict[str, object]:
    protocol = _json_load(output / "protocol.json")
    summary = _json_load(output / "summary.json")
    manifest = _json_load(output / "manifest.json")
    if (
        protocol.get("schema_version") != PROTOCOL_SCHEMA
        or manifest.get("schema_version") != MANIFEST_SCHEMA
        or protocol.get("performance_claimed") is not False
        or summary.get("performance_claimed") is not False
        or manifest.get("performance_claimed") is not False
    ):
        raise ValueError("R3-D0 artifact boundary differs")
    protocol_copy = dict(protocol)
    protocol_hash = protocol_copy.pop("protocol_hash", None)
    if _hash(protocol_copy) != protocol_hash:
        raise ValueError("R3-D0 protocol hash differs")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or any(
        _file_hash(output / str(path)) != digest for path, digest in files.items()
    ):
        raise ValueError("R3-D0 manifest file digest differs")
    manifest_copy = dict(manifest)
    manifest_hash = manifest_copy.pop("manifest_hash", None)
    if _hash(manifest_copy) != manifest_hash:
        raise ValueError("R3-D0 manifest hash differs")
    raws = [_load(path) for path in sorted((output / "raw").glob("*.pt"))]
    rebuilt = _summarize(raws)
    if rebuilt != summary or _hash(
        {k: v for k, v in summary.items() if k != "summary_hash"}
    ) != summary.get("summary_hash"):
        raise ValueError("R3-D0 summary replay differs")
    print(
        f"R3-D0 replay PASS: verdict={summary['verdict']} "
        f"worst_required={summary['worst_compiled_region_required_speedup']} "
        "performance_claimed=false",
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    output = args.output.absolute()
    if args.replay:
        replay(output)
    else:
        generate(output, args.source_capture.absolute(), args.model.absolute())


if __name__ == "__main__":
    main()
