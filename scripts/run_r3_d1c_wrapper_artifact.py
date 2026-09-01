#!/usr/bin/env python3
"""Generate or replay the formal five-fresh D1-C cumulative wrapper artifact."""

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
import time
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAPTURE = (
    ROOT / "artifacts/rvir-v4-pre-state/resnet2b-core-pre-state-v1/source_capture.pt"
)
DEFAULT_MODEL = Path(
    "/home/lee/Codes/vnncomp2021/benchmarks/cifar10_resnet/onnx/resnet_2b.onnx"
)
DEFAULT_OUTPUT = ROOT / "artifacts/r3-structured-owner/r3-d1c-wrapper-formal-v1"
WORKER = ROOT / "scripts/run_r3_d1c_wrapper_worker.py"
D1B_ARTIFACT = ROOT / "artifacts/r3-structured-owner/r3-d1b-schedule-formal-v1"
R3_2B_ARTIFACT = ROOT / "artifacts/r3-structured-owner/r3-2b-wrapper-timing-v1"
ORDER = (
    ("native", "d1c", "b3"),
    ("d1c", "native", "b3"),
    ("b3", "native", "d1c"),
    ("d1c", "b3", "native"),
    ("native", "b3", "d1c"),
)
CODE_PATHS = (
    "boundflow/backends/tvm/r3_d1c_wrapper_schedule.py",
    "boundflow/runtime/r3_d1c_cumulative_wrapper.py",
    "boundflow/runtime/r3_optimizer_trajectory_timing.py",
    "boundflow/runtime/r3_compiled_p_alpha_vjp.py",
    "scripts/run_r3_d1c_wrapper_worker.py",
    "scripts/run_r3_d1c_wrapper_artifact.py",
    "scripts/probe_r3_d1c_wrapper_tamper.py",
)
WRAPPER_GEOMEAN_GATE = 1.20
WRAPPER_WORST_GATE = 1.00
CUMULATIVE_GATE = 9.3181
COOLDOWN_SECONDS = 30


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
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _clean() -> None:
    dirty = [
        line
        for line in _git("status", "--porcelain").splitlines()
        if not line.endswith("docs/CIBC_for_DAC.pdf") and ".docops/ev.jsonl" not in line
    ]
    if dirty:
        raise RuntimeError(f"R3-D1C formal source is dirty: {dirty}")


def _load(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise TypeError("R3-D1C raw root differs")
    return value


def _validate_worker(raw: Mapping[str, Any]) -> None:
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
        "d1c_receipt",
        "arena_pointers",
        "memory",
        "environment",
        "clock",
        "formal_performance_claimed",
    }
    if (
        set(raw) != expected
        or raw["schema_version"] != "boundflow.r3-d1c-wrapper-worker/v1"
        or raw["mode"] not in {"native", "b3", "d1c"}
        or raw["warmup_count"] != 3
        or raw["sample_count"] != 30
        or raw["clock"] != "host-perf-counter-ns-with-device-boundary-sync"
        or raw["formal_performance_claimed"] is not False
    ):
        raise ValueError("R3-D1C worker envelope differs")
    samples = raw["latency_ns"]
    if (
        not isinstance(samples, list)
        or len(samples) != 30
        or any(not isinstance(value, int) or value <= 0 for value in samples)
        or raw["median_latency_ns"] != statistics.median(samples)
    ):
        raise ValueError("R3-D1C timing samples differ")
    lower, alpha = raw["terminal_lower"], raw["terminal_alpha"]
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
        raise ValueError("R3-D1C terminal payload differs")
    mode = raw["mode"]
    execution = raw["execution"]
    expected_execution = {
        "evaluation_count": 10,
        "optimizer_mutation_count": 9,
        "scheduler_mutation_count": 9,
        "custom_forward_count": 0 if mode == "native" else 10,
        "custom_backward_count": 0 if mode == "native" else 10,
        "forward_launch_count_last_evaluation": {"native": 0, "b3": 15, "d1c": 17}[
            mode
        ],
        "fallback_count": 0,
        "eager_candidate_count": 0,
        "native_shadow_count": 0,
        "timing_capture_count": 0,
    }
    if execution != expected_execution:
        raise ValueError("R3-D1C execution receipt differs")
    receipt = raw["d1c_receipt"]
    arena = raw["arena_pointers"]
    if mode == "d1c":
        expected_receipt = {
            "scheduled_tir_hash",
            "device_source_hash",
            "exported_symbols",
            "threads_per_block",
            "reduction_kind",
            "vector_width",
            "launch_count",
            "existing_arena_count",
            "scratch_region_count",
            "scratch_region_pointers",
            "bias_inplace_alias_count",
            "persistent_dense_a",
            "global_workspace_bytes",
            "fallback_count",
            "eager_candidate_count",
            "native_shadow_count",
            "wrapper_performance_claimed",
        }
        hashes = (
            receipt.get("scheduled_tir_hash") if isinstance(receipt, Mapping) else None,
            receipt.get("device_source_hash") if isinstance(receipt, Mapping) else None,
        )
        if (
            not isinstance(receipt, Mapping)
            or set(receipt) != expected_receipt
            or any(
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in hashes
            )
            or tuple(receipt.get("exported_symbols", ()))
            != (
                "boundflow_r3d1c_residual11_stage1",
                "boundflow_r3d1c_residual11_stage2",
                "boundflow_r3d1c_residual6_stage1",
                "boundflow_r3d1c_residual6_stage2",
            )
            or receipt.get("threads_per_block") != 256
            or receipt.get("reduction_kind") != "serial-reference"
            or receipt.get("vector_width") != 1
            or receipt.get("launch_count") != 4
            or receipt.get("existing_arena_count") != 2
            or receipt.get("scratch_region_count") != 2
            or receipt.get("bias_inplace_alias_count") != 2
            or receipt.get("persistent_dense_a") is not False
            or receipt.get("global_workspace_bytes") != 0
            or receipt.get("wrapper_performance_claimed") is not False
            or not isinstance(arena, tuple)
            or len(arena) != 2
            or tuple(receipt.get("scratch_region_pointers", ()))
            != (arena[1] + 6144 * 4, arena[0] + 12288 * 4)
        ):
            raise ValueError("R3-D1C ownership receipt differs")
    elif receipt is not None or arena is not None:
        raise ValueError("R3-D1C noncandidate receipt differs")
    memory = raw["memory"]
    if (
        not isinstance(memory, Mapping)
        or set(memory)
        != {"allocated_before", "reserved_before", "peak_allocated", "peak_reserved"}
        or any(not isinstance(value, int) or value < 0 for value in memory.values())
    ):
        raise ValueError("R3-D1C memory receipt differs")


def _tensor_diff(left: Any, right: Any) -> tuple[float, bool]:
    if not torch.is_tensor(left) or not torch.is_tensor(right):
        raise TypeError("R3-D1C comparison tensor differs")
    return float((left - right).abs().max()), torch.equal(
        torch.sign(left), torch.sign(right)
    )


def _compare_triplet(
    by_mode: Mapping[str, Mapping[str, Any]],
) -> dict[str, float | bool]:
    if set(by_mode) != {"native", "b3", "d1c"}:
        raise ValueError("R3-D1C triplet inventory differs")
    native, b3, d1c = by_mode["native"], by_mode["b3"], by_mode["d1c"]
    for raw in (native, b3, d1c):
        _validate_worker(raw)
    for name in (
        "run_index",
        "source_capture_sha256",
        "model_sha256",
        "plan_hash",
        "trace_hash",
        "environment",
    ):
        if native[name] != b3[name] or native[name] != d1c[name]:
            raise ValueError(f"R3-D1C triplet {name} differs")
    native_lower_diff, native_sign = _tensor_diff(
        native["terminal_lower"], d1c["terminal_lower"]
    )
    b3_lower_diff, b3_sign = _tensor_diff(b3["terminal_lower"], d1c["terminal_lower"])
    native_alpha_diff, _ = _tensor_diff(native["terminal_alpha"], d1c["terminal_alpha"])
    b3_alpha_diff, _ = _tensor_diff(b3["terminal_alpha"], d1c["terminal_alpha"])
    if (
        native_lower_diff > 2e-4
        or b3_lower_diff > 2e-4
        or not native_sign
        or not b3_sign
        or native_alpha_diff > 2e-5
        or b3_alpha_diff > 2e-5
    ):
        raise ValueError("R3-D1C terminal semantics differ")
    native_ns = float(native["median_latency_ns"])
    b3_ns = float(b3["median_latency_ns"])
    d1c_ns = float(d1c["median_latency_ns"])
    b3_memory, d1c_memory = b3["memory"], d1c["memory"]
    assert isinstance(b3_memory, Mapping) and isinstance(d1c_memory, Mapping)
    return {
        "native_median_ns": native_ns,
        "b3_median_ns": b3_ns,
        "d1c_median_ns": d1c_ns,
        "wrapper_speedup": native_ns / d1c_ns,
        "cumulative_candidate_speedup": b3_ns / d1c_ns,
        "native_lower_diff": native_lower_diff,
        "b3_lower_diff": b3_lower_diff,
        "native_alpha_diff": native_alpha_diff,
        "b3_alpha_diff": b3_alpha_diff,
        "sign_exact": native_sign and b3_sign,
        "allocated_ratio_to_b3": float(d1c_memory["peak_allocated"])
        / float(b3_memory["peak_allocated"]),
        "reserved_ratio_to_b3": float(d1c_memory["peak_reserved"])
        / float(b3_memory["peak_reserved"]),
    }


def _summary(raws: list[dict[str, Any]]) -> dict[str, Any]:
    d1c_receipts = [raw["d1c_receipt"] for raw in raws if raw["mode"] == "d1c"]
    receipt_signature = {
        (
            receipt["scheduled_tir_hash"],
            receipt["device_source_hash"],
            tuple(receipt["exported_symbols"]),
            receipt["threads_per_block"],
            receipt["reduction_kind"],
            receipt["vector_width"],
        )
        for receipt in d1c_receipts
        if isinstance(receipt, Mapping)
    }
    if len(d1c_receipts) != 5 or len(receipt_signature) != 1:
        raise ValueError("R3-D1C fresh compiler receipt differs")
    rows = []
    for run_index in range(5):
        by_mode = {raw["mode"]: raw for raw in raws if raw["run_index"] == run_index}
        rows.append(_compare_triplet(by_mode))
    wrapper = [float(row["wrapper_speedup"]) for row in rows]
    cumulative = [float(row["cumulative_candidate_speedup"]) for row in rows]
    wrapper_geomean = math.exp(sum(math.log(value) for value in wrapper) / len(wrapper))
    cumulative_geomean = math.exp(
        sum(math.log(value) for value in cumulative) / len(cumulative)
    )
    wrapper_go = (
        wrapper_geomean >= WRAPPER_GEOMEAN_GATE and min(wrapper) >= WRAPPER_WORST_GATE
    )
    cumulative_go = min(cumulative) >= CUMULATIVE_GATE
    memory_go = all(
        float(row["allocated_ratio_to_b3"]) <= 1.0
        and float(row["reserved_ratio_to_b3"]) <= 1.0
        for row in rows
    )
    go = wrapper_go and cumulative_go and memory_go
    result: dict[str, Any] = {
        "schema_version": "boundflow.r3-d1c-wrapper-summary/v1",
        "triplet_count": 5,
        "worker_count": 15,
        "order": [list(row) for row in ORDER],
        "triplet_metrics": rows,
        "wrapper_geomean_speedup": wrapper_geomean,
        "wrapper_worst_speedup": min(wrapper),
        "cumulative_geomean_speedup": cumulative_geomean,
        "cumulative_worst_speedup": min(cumulative),
        "wrapper_geomean_gate": WRAPPER_GEOMEAN_GATE,
        "wrapper_worst_gate": WRAPPER_WORST_GATE,
        "cumulative_gate": CUMULATIVE_GATE,
        "wrapper_gate_pass": wrapper_go,
        "cumulative_gate_pass": cumulative_go,
        "memory_gate_pass": memory_go,
        "d1c_go": go,
        "status": (
            "VALIDATED-R3-D1-P-LOCAL-WRAPPER"
            if go
            else "VALIDATED-NO-GO-R3-D1C-CUMULATIVE-WRAPPER"
        ),
        "r3_3_open": go,
        "backward_attribution_open": not go,
        "performance_claimed": go,
    }
    result["summary_hash"] = _hash(result)
    return result


def _protocol(revision: str, capture: Path, model: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": "boundflow.r3-d1c-wrapper-protocol/v1",
        "source_revision": revision,
        "source_capture_sha256": _file_hash(capture),
        "model_sha256": _file_hash(model),
        "d1b_manifest_sha256": _file_hash(D1B_ARTIFACT / "manifest.json"),
        "r3_2b_manifest_sha256": _file_hash(R3_2B_ARTIFACT / "manifest.json"),
        "order": [list(row) for row in ORDER],
        "warmup_count": 3,
        "sample_count": 30,
        "cooldown_seconds": COOLDOWN_SECONDS,
        "evaluation_count": 10,
        "optimizer_mutation_count": 9,
        "scheduler_mutation_count": 9,
        "wrapper_geomean_gate": WRAPPER_GEOMEAN_GATE,
        "wrapper_worst_gate": WRAPPER_WORST_GATE,
        "cumulative_gate": CUMULATIVE_GATE,
        "memory_ratio_max": 1.0,
        "code_revision": {name: _file_hash(ROOT / name) for name in CODE_PATHS},
    }
    result["protocol_hash"] = _hash(result)
    return result


def generate(output: Path, capture: Path, model: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"R3-D1C artifact exists: {output}")
    _clean()
    revision = _git("rev-parse", "HEAD")
    protocol = _protocol(revision, capture, model)
    temporary = Path(tempfile.mkdtemp(prefix="r3-d1c-formal-", dir=output.parent))
    try:
        raw_dir = temporary / "raw"
        raw_dir.mkdir(parents=True)
        raws = []
        worker_ordinal = 0
        for run_index, triplet in enumerate(ORDER):
            for sequence, mode in enumerate(triplet):
                if worker_ordinal:
                    time.sleep(COOLDOWN_SECONDS)
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
                worker_ordinal += 1
        summary = _summary(raws)
        (temporary / "protocol.json").write_text(
            _canonical(protocol) + "\n", encoding="utf-8"
        )
        (temporary / "summary.json").write_text(
            _canonical(summary) + "\n", encoding="utf-8"
        )
        files = {
            str(path.relative_to(temporary)): _file_hash(path)
            for path in sorted(temporary.rglob("*"))
            if path.is_file()
        }
        manifest: dict[str, Any] = {
            "schema_version": "boundflow.r3-d1c-wrapper-manifest/v1",
            "source_revision": revision,
            "protocol_hash": protocol["protocol_hash"],
            "summary_hash": summary["summary_hash"],
            "files": files,
        }
        manifest["manifest_hash"] = _hash(manifest)
        (temporary / "manifest.json").write_text(
            _canonical(manifest) + "\n", encoding="utf-8"
        )
        replay(temporary)
        temporary.rename(output)
        return summary
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def replay(artifact: Path) -> dict[str, Any]:
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    unsigned_manifest = dict(manifest)
    manifest_hash = unsigned_manifest.pop("manifest_hash", None)
    if (
        manifest_hash != _hash(unsigned_manifest)
        or manifest.get("schema_version") != "boundflow.r3-d1c-wrapper-manifest/v1"
    ):
        raise ValueError("R3-D1C manifest differs")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or any(
        _file_hash(artifact / name) != digest for name, digest in files.items()
    ):
        raise ValueError("R3-D1C file digest differs")
    protocol = json.loads((artifact / "protocol.json").read_text(encoding="utf-8"))
    unsigned_protocol = dict(protocol)
    protocol_hash = unsigned_protocol.pop("protocol_hash", None)
    if (
        protocol_hash != _hash(unsigned_protocol)
        or protocol_hash != manifest["protocol_hash"]
    ):
        raise ValueError("R3-D1C protocol differs")
    frozen = {
        "order": [list(row) for row in ORDER],
        "warmup_count": 3,
        "sample_count": 30,
        "cooldown_seconds": COOLDOWN_SECONDS,
        "evaluation_count": 10,
        "optimizer_mutation_count": 9,
        "scheduler_mutation_count": 9,
        "wrapper_geomean_gate": WRAPPER_GEOMEAN_GATE,
        "wrapper_worst_gate": WRAPPER_WORST_GATE,
        "cumulative_gate": CUMULATIVE_GATE,
        "memory_ratio_max": 1.0,
    }
    if any(protocol.get(name) != value for name, value in frozen.items()):
        raise ValueError("R3-D1C frozen protocol semantics differ")
    raws = [_load(path) for path in sorted((artifact / "raw").glob("*.pt"))]
    summary = _summary(raws)
    if (
        summary != json.loads((artifact / "summary.json").read_text(encoding="utf-8"))
        or summary["summary_hash"] != manifest["summary_hash"]
    ):
        raise ValueError("R3-D1C semantic replay differs")
    print(
        f"R3-D1C replay PASS: wrapper={summary['wrapper_geomean_speedup']:.4f}x "
        f"cumulative={summary['cumulative_geomean_speedup']:.4f}x status={summary['status']}",
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    if args.replay:
        replay(args.output.absolute())
    else:
        generate(
            args.output.absolute(),
            args.source_capture.absolute(),
            args.model.absolute(),
        )


if __name__ == "__main__":
    main()
