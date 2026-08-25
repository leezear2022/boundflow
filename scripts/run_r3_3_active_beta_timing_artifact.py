#!/usr/bin/env python3
"""Generate or replay the formal R3-3 active-beta isolated timing artifact."""

# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=too-many-boolean-expressions,missing-function-docstring
# pylint: disable=duplicate-code

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/r3-structured-owner/r3-3-active-beta-timing-v1"
WORKER = ROOT / "scripts/run_r3_3_active_beta_timing_worker.py"
CAPTURE = ROOT / "artifacts/fsg4-b4b1-reference-five-fresh/resnet2b-prop0-v1"
RUN_COUNT = 6
ORDERS = ("AB", "BA", "AB", "BA", "AB", "BA")
WARMUPS = 10
PAIR_COUNT = 30
SPEEDUP_GATE = 1.05
BOOTSTRAP_GATE = 1.0
WORST_GATE = 0.98
MEMORY_GATE = 1.05
BOOTSTRAP_COUNT = 10_000
BOOTSTRAP_SEED = 20260826
CODE_PATHS = (
    "boundflow/ir/differentiable_lower_sparse_linear_tir.py",
    "boundflow/backends/tvm/differentiable_lower_sparse_linear.py",
    "boundflow/runtime/fsg4_b4b2_sparse_linear_tir.py",
    "boundflow/runtime/r3_3_active_beta_timing.py",
    "scripts/run_r3_3_active_beta_timing_worker.py",
    "scripts/run_r3_3_active_beta_timing_artifact.py",
    "scripts/probe_r3_3_active_beta_timing_tamper.py",
    "gemini_doc/BOUNDFLOW_R3_3_ACTIVE_BETA_ISOLATED_TIMING_PLAN_2026_08_26.md",
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
        raise TypeError("R3-3 timing JSON root differs")
    return value


def _write(path: Path, value: object) -> None:
    path.write_text(_canonical(value) + "\n", encoding="utf-8")


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _git_blob_hash(revision: str, path: str) -> str:
    content = subprocess.check_output(("git", "show", f"{revision}:{path}"), cwd=ROOT)
    return hashlib.sha256(content).hexdigest()


def _clean() -> None:
    ignored = ("docs/CIBC_for_DAC.pdf", ".docops/ev.jsonl")
    dirty = [
        row
        for row in _git("status", "--porcelain").splitlines()
        if not any(row.endswith(name) for name in ignored)
    ]
    if dirty:
        raise RuntimeError(f"R3-3 timing source is dirty: {dirty}")


def _geomean(values: Sequence[float]) -> float:
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("R3-3 timing geomean input differs")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _bootstrap_lower(values: Sequence[float]) -> float:
    generator = random.Random(BOOTSTRAP_SEED)
    rows = []
    for _ in range(BOOTSTRAP_COUNT):
        sample = [values[generator.randrange(len(values))] for _ in values]
        rows.append(_geomean(sample))
    rows.sort()
    return rows[int(0.025 * len(rows))]


def _memory_ratio(
    candidate: Mapping[str, Any], baseline: Mapping[str, Any], key: str
) -> float:
    left, right = float(candidate[key]), float(baseline[key])
    if left < 0 or right <= 0:
        raise ValueError("R3-3 timing memory differs")
    return left / right


def _validate_worker(
    raw: Mapping[str, Any], ordinal: int, protocol: Mapping[str, Any]
) -> dict[str, float]:
    payload = dict(raw)
    worker_hash = payload.pop("worker_hash", None)
    expected_keys = {
        "schema_version",
        "run_ordinal",
        "capture_ordinal",
        "capture_sha256",
        "order",
        "warmup_count",
        "pair_count",
        "pairs",
        "baseline_median_ms",
        "candidate_median_ms",
        "paired_speedup",
        "parity",
        "baseline_memory",
        "candidate_memory",
        "template_hash",
        "schedule_hash",
        "module_receipt_hash",
        "forbidden_workspace_count",
        "module_call_count",
        "fallback_count",
        "eager_backward_count",
        "gpu_before",
        "gpu_after",
        "compile_excluded",
        "performance_claimed",
        "worker_hash",
    }
    captures = protocol.get("capture_sha256")
    pairs = raw.get("pairs")
    parity = raw.get("parity")
    baseline_memory = raw.get("baseline_memory")
    candidate_memory = raw.get("candidate_memory")
    if (
        set(raw) != expected_keys
        or worker_hash != _hash(payload)
        or raw.get("schema_version") != "boundflow.r3-3-active-beta-timing-worker/v1"
        or raw.get("run_ordinal") != ordinal
        or raw.get("capture_ordinal") != ordinal % 5
        or not isinstance(captures, dict)
        or raw.get("capture_sha256") != captures[f"run_{ordinal % 5:02d}.pt"]
        or raw.get("order") != ORDERS[ordinal]
        or raw.get("warmup_count") != WARMUPS
        or raw.get("pair_count") != PAIR_COUNT
        or not isinstance(pairs, list)
        or len(pairs) != PAIR_COUNT
        or not isinstance(parity, dict)
        or parity.get("allclose") is not True
        or parity.get("sign_exact") is not True
        or float(parity.get("maximum_absolute_difference", math.inf)) > 2.0e-4
        or parity.get("element_count") != 6318
        or not isinstance(baseline_memory, dict)
        or not isinstance(candidate_memory, dict)
        or raw.get("forbidden_workspace_count") != 0
        or raw.get("module_call_count") != {"forward": 1, "backward": 1}
        or raw.get("fallback_count") != 0
        or raw.get("eager_backward_count") != 0
        or raw.get("compile_excluded") is not True
        or raw.get("performance_claimed") is not False
    ):
        raise ValueError("R3-3 timing worker envelope differs")
    baseline_samples, candidate_samples = [], []
    for pair_ordinal, row in enumerate(pairs):
        if (
            not isinstance(row, dict)
            or set(row)
            != {
                "pair_ordinal",
                "order",
                "baseline_ms",
                "candidate_ms",
                "speedup",
            }
            or row.get("pair_ordinal") != pair_ordinal
            or row.get("order") != ORDERS[ordinal]
        ):
            raise ValueError("R3-3 timing pair envelope differs")
        baseline_ms, candidate_ms = float(row["baseline_ms"]), float(
            row["candidate_ms"]
        )
        if (
            not math.isfinite(baseline_ms)
            or not math.isfinite(candidate_ms)
            or baseline_ms <= 0.0
            or candidate_ms <= 0.0
            or float(row["speedup"]) != baseline_ms / candidate_ms
        ):
            raise ValueError("R3-3 timing pair derivation differs")
        baseline_samples.append(baseline_ms)
        candidate_samples.append(candidate_ms)
    baseline_median = statistics.median(baseline_samples)
    candidate_median = statistics.median(candidate_samples)
    speedup = baseline_median / candidate_median
    if (
        float(raw["baseline_median_ms"]) != baseline_median
        or float(raw["candidate_median_ms"]) != candidate_median
        or float(raw["paired_speedup"]) != speedup
    ):
        raise ValueError("R3-3 timing worker summary differs")
    for memory in (baseline_memory, candidate_memory):
        if set(memory) != {
            "base_allocated_bytes",
            "peak_allocated_bytes",
            "incremental_allocated_bytes",
            "peak_reserved_bytes",
        } or memory["incremental_allocated_bytes"] != max(
            0, memory["peak_allocated_bytes"] - memory["base_allocated_bytes"]
        ):
            raise ValueError("R3-3 timing memory derivation differs")
    return {
        "speedup": speedup,
        "allocated_ratio": _memory_ratio(
            candidate_memory, baseline_memory, "peak_allocated_bytes"
        ),
        "reserved_ratio": _memory_ratio(
            candidate_memory, baseline_memory, "peak_reserved_bytes"
        ),
        "incremental_ratio": _memory_ratio(
            candidate_memory, baseline_memory, "incremental_allocated_bytes"
        ),
        "parity_maximum": float(parity["maximum_absolute_difference"]),
    }


def _summary(raws: list[dict[str, Any]], protocol: Mapping[str, Any]) -> dict[str, Any]:
    if len(raws) != RUN_COUNT:
        raise ValueError("R3-3 timing worker count differs")
    rows = [
        _validate_worker(raw, ordinal, protocol) for ordinal, raw in enumerate(raws)
    ]
    template_hashes = {raw["template_hash"] for raw in raws}
    schedule_hashes = {raw["schedule_hash"] for raw in raws}
    module_hashes = {raw["module_receipt_hash"] for raw in raws}
    if (
        len(template_hashes) != 1
        or len(schedule_hashes) != 1
        or len(module_hashes) != 1
    ):
        raise ValueError("R3-3 timing compiler identity differs")
    speedups = [row["speedup"] for row in rows]
    geomean = _geomean(speedups)
    bootstrap = _bootstrap_lower(speedups)
    worst = min(speedups)
    allocated = max(row["allocated_ratio"] for row in rows)
    reserved = max(row["reserved_ratio"] for row in rows)
    gates = {
        "geomean": geomean >= SPEEDUP_GATE,
        "bootstrap_lower": bootstrap >= BOOTSTRAP_GATE,
        "worst_worker": worst >= WORST_GATE,
        "allocated": allocated <= MEMORY_GATE,
        "reserved": reserved <= MEMORY_GATE,
    }
    passed = all(gates.values())
    result: dict[str, Any] = {
        "schema_version": "boundflow.r3-3-active-beta-timing-summary/v1",
        "worker_count": RUN_COUNT,
        "pair_count": RUN_COUNT * PAIR_COUNT,
        "worker_speedups": speedups,
        "paired_speedup_geomean": geomean,
        "bootstrap_95_lower": bootstrap,
        "worst_worker_speedup": worst,
        "maximum_parity_difference": max(row["parity_maximum"] for row in rows),
        "maximum_allocated_ratio": allocated,
        "maximum_reserved_ratio": reserved,
        "maximum_incremental_allocated_ratio": max(
            row["incremental_ratio"] for row in rows
        ),
        "gates": gates,
        "provisional_verdict": (
            "VALIDATED-R3-3-S-ISOLATED-PHYSICS-PENDING-TAMPER"
            if passed
            else "VALIDATED-NO-GO-R3-3-S-ISOLATED-PHYSICS-PENDING-TAMPER"
        ),
        "template_hash": next(iter(template_hashes)),
        "schedule_hash": next(iter(schedule_hashes)),
        "module_receipt_hash": next(iter(module_hashes)),
        "timing_closure_pending_tamper": True,
        "r3_4_open": False,
        "same_solver_open": False,
        "performance_claimed": False,
    }
    result["summary_hash"] = _hash(result)
    return result


def _protocol(revision: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": "boundflow.r3-3-active-beta-timing-protocol/v1",
        "source_revision": revision,
        "worker_count": RUN_COUNT,
        "orders": list(ORDERS),
        "capture_ordinals": [0, 1, 2, 3, 4, 0],
        "warmups_per_side": WARMUPS,
        "pairs_per_worker": PAIR_COUNT,
        "speedup_geomean_gate": SPEEDUP_GATE,
        "bootstrap_lower_gate": BOOTSTRAP_GATE,
        "worst_worker_gate": WORST_GATE,
        "memory_ratio_gate": MEMORY_GATE,
        "bootstrap_sample_count": BOOTSTRAP_COUNT,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "compile_cache_miss_excluded": True,
        "dense_reconstruction_included": True,
        "autograd_wrapper_included": True,
        "performance_claimed": False,
        "capture_sha256": {
            f"run_{ordinal:02d}.pt": _file_hash(CAPTURE / f"run_{ordinal:02d}.pt")
            for ordinal in range(5)
        },
        "code_revision": {name: _file_hash(ROOT / name) for name in CODE_PATHS},
    }
    result["protocol_hash"] = _hash(result)
    return result


def generate(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"R3-3 timing artifact exists: {output}")
    _clean()
    protocol = _protocol(_git("rev-parse", "HEAD"))
    temporary = Path(tempfile.mkdtemp(prefix="r3-3-timing-", dir=output.parent))
    try:
        raw_dir, log_dir = temporary / "raw", temporary / "logs"
        raw_dir.mkdir(parents=True)
        log_dir.mkdir()
        raws = []
        for ordinal, order in enumerate(ORDERS):
            target = raw_dir / f"run_{ordinal:02d}.json"
            completed = subprocess.run(
                (
                    sys.executable,
                    str(WORKER),
                    "--run-ordinal",
                    str(ordinal),
                    "--order",
                    order,
                    "--output",
                    str(target),
                ),
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
                env=os.environ.copy(),
            )
            (log_dir / f"run_{ordinal:02d}.stdout.txt").write_text(
                completed.stdout, encoding="utf-8"
            )
            (log_dir / f"run_{ordinal:02d}.stderr.txt").write_text(
                completed.stderr, encoding="utf-8"
            )
            raws.append(_json(target))
        summary = _summary(raws, protocol)
        _write(temporary / "protocol.json", protocol)
        _write(temporary / "summary.json", summary)
        files = {
            str(path.relative_to(temporary)): _file_hash(path)
            for path in sorted(temporary.rglob("*"))
            if path.is_file()
        }
        manifest: dict[str, Any] = {
            "schema_version": "boundflow.r3-3-active-beta-timing-manifest/v1",
            "source_revision": protocol["source_revision"],
            "protocol_hash": protocol["protocol_hash"],
            "summary_hash": summary["summary_hash"],
            "files": files,
        }
        manifest["manifest_hash"] = _hash(manifest)
        _write(temporary / "manifest.json", manifest)
        replay(temporary)
        temporary.rename(output)
        return summary
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def replay(artifact: Path) -> dict[str, Any]:
    manifest = _json(artifact / "manifest.json")
    unsigned_manifest = dict(manifest)
    manifest_hash = unsigned_manifest.pop("manifest_hash", None)
    if manifest.get(
        "schema_version"
    ) != "boundflow.r3-3-active-beta-timing-manifest/v1" or manifest_hash != _hash(
        unsigned_manifest
    ):
        raise ValueError("R3-3 timing manifest differs")
    files = manifest.get("files")
    if not isinstance(files, dict) or any(
        _file_hash(artifact / name) != digest for name, digest in files.items()
    ):
        raise ValueError("R3-3 timing file digest differs")
    protocol = _json(artifact / "protocol.json")
    unsigned_protocol = dict(protocol)
    protocol_hash = unsigned_protocol.pop("protocol_hash", None)
    if (
        protocol_hash != _hash(unsigned_protocol)
        or protocol_hash != manifest["protocol_hash"]
    ):
        raise ValueError("R3-3 timing protocol hash differs")
    frozen = {
        "worker_count": RUN_COUNT,
        "orders": list(ORDERS),
        "capture_ordinals": [0, 1, 2, 3, 4, 0],
        "warmups_per_side": WARMUPS,
        "pairs_per_worker": PAIR_COUNT,
        "speedup_geomean_gate": SPEEDUP_GATE,
        "bootstrap_lower_gate": BOOTSTRAP_GATE,
        "worst_worker_gate": WORST_GATE,
        "memory_ratio_gate": MEMORY_GATE,
        "bootstrap_sample_count": BOOTSTRAP_COUNT,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "compile_cache_miss_excluded": True,
        "dense_reconstruction_included": True,
        "autograd_wrapper_included": True,
        "performance_claimed": False,
    }
    if any(protocol.get(name) != value for name, value in frozen.items()):
        raise ValueError("R3-3 timing frozen protocol differs")
    revision, code = protocol.get("source_revision"), protocol.get("code_revision")
    if (
        not isinstance(revision, str)
        or not isinstance(code, dict)
        or set(code) != set(CODE_PATHS)
        or any(
            _git_blob_hash(revision, name) != digest for name, digest in code.items()
        )
    ):
        raise ValueError("R3-3 timing source binding differs")
    raws = [_json(path) for path in sorted((artifact / "raw").glob("*.json"))]
    summary = _summary(raws, protocol)
    if (
        summary != _json(artifact / "summary.json")
        or summary["summary_hash"] != manifest["summary_hash"]
    ):
        raise ValueError("R3-3 timing semantic replay differs")
    print(
        f"R3-3 timing replay PASS: geomean={summary['paired_speedup_geomean']:.6f} "
        f"worst={summary['worst_worker_speedup']:.6f} "
        f"verdict={summary['provisional_verdict']}",
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    if args.replay:
        replay(args.output.absolute())
    else:
        generate(args.output.absolute())


if __name__ == "__main__":
    main()
