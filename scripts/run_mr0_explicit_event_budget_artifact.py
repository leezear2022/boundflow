#!/usr/bin/env python3
"""Generate or replay the formal MR0 explicit CUDA-event budget artifact."""

# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=too-many-boolean-expressions,missing-function-docstring
# pylint: disable=duplicate-code,wrong-import-position

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, cast, Mapping

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/measurement-recovery/mr0-explicit-event-budget-resnet2b-v1"
WORKER = ROOT / "scripts/run_mr0_explicit_event_budget_worker.py"
SOURCE_CAPTURE = (
    ROOT
    / "artifacts/rvir-v4-native-optimizer/resnet2b-core-step-parity-v1/source_capture.pt"
)
MODEL = Path(
    "/home/lee/Codes/vnncomp2021/benchmarks/cifar10_resnet/onnx/resnet_2b.onnx"
)
SOURCE_CAPTURE_SHA256 = (
    "f42229dd06be8521865d14af4cc51450ec8bec4e792934b457c276aad50126dc"
)
MODEL_SHA256 = "791aa24d77917ecda16809fbbd48e7739616f88ebf74cf358b2d1bf911dc4a6d"
CODE_PATHS = (
    "boundflow/runtime/mr0_explicit_event_budget.py",
    "scripts/run_mr0_explicit_event_budget_worker.py",
    "scripts/run_mr0_explicit_event_budget_artifact.py",
    "scripts/probe_mr0_explicit_event_budget_tamper.py",
    "gemini_doc/BOUNDFLOW_MR0_LOW_PERTURBATION_EVENT_BUDGET_PLAN_2026_08_26.md",
)

from boundflow.runtime.mr0_explicit_event_budget import (  # noqa: E402
    MR0_BOOTSTRAP_SAMPLES,
    MR0_BOOTSTRAP_SEED,
    MR0_BOOTSTRAP_UPPER_GATE,
    MR0_BUDGETS,
    MR0_GEOMEAN_GATE,
    MR0_GROUP_COUNT,
    MR0_ORDERS,
    MR0_REPEATS,
    MR0_WARMUP,
    MR0_WORKER_COUNT,
    MR0_WORST_GATE,
    canonical_hash,
    derive_summary,
    validate_budget_row,
)


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("MR0 JSON root differs")
    return value


def _write(path: Path, value: object) -> None:
    path.write_text(_canonical(value) + "\n", encoding="utf-8")


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _git_blob_hash(revision: str, path: str) -> str:
    content = subprocess.check_output(("git", "show", f"{revision}:{path}"), cwd=ROOT)
    return hashlib.sha256(content).hexdigest()


def _portable_log(value: str) -> str:
    replacements = (
        (str(ROOT), "<repo>"),
        (str(MODEL.parents[4]), "<vnncomp-checkout>"),
        (sys.prefix, "<python-prefix>"),
    )
    for source, target in replacements:
        value = value.replace(source, target)
    return value


def _clean() -> None:
    ignored = ("docs/CIBC_for_DAC.pdf", ".docops/ev.jsonl")
    dirty = [
        row
        for row in _git("status", "--porcelain").splitlines()
        if not any(row.endswith(path) for path in ignored)
    ]
    if dirty:
        raise RuntimeError(f"MR0 source is dirty: {dirty}")


def _validate_worker(raw: Mapping[str, Any], ordinal: int) -> dict[str, Any]:
    unsigned = dict(raw)
    worker_hash = unsigned.pop("worker_hash", None)
    expected = {
        "schema_version",
        "run_ordinal",
        "order",
        "source_capture_sha256",
        "model_sha256",
        "threads_per_block",
        "topology_inventory",
        "budget_rows",
        "warmup_count",
        "group_count",
        "repeats_per_side",
        "event_budgets",
        "event_object_count",
        "semantic_receipt",
        "semantic_admitted",
        "stream_before",
        "stream_after",
        "stream_admitted",
        "gpu_before",
        "gpu_after",
        "environment",
        "compile_excluded",
        "cuda_graph",
        "input_copy_included",
        "performance_claimed",
        "worker_hash",
    }
    rows = raw.get("budget_rows")
    semantic = raw.get("semantic_receipt")
    environment = raw.get("environment")
    if (
        set(raw) != expected
        or worker_hash != canonical_hash(unsigned)
        or raw.get("schema_version") != "boundflow.mr0-explicit-event-budget-worker/v1"
        or raw.get("run_ordinal") != ordinal
        or raw.get("order") != MR0_ORDERS[ordinal]
        or raw.get("source_capture_sha256") != SOURCE_CAPTURE_SHA256
        or raw.get("model_sha256") != MODEL_SHA256
        or raw.get("threads_per_block") != 128
        or raw.get("topology_inventory")
        != {"add": 2, "conv2d": 6, "flatten": 1, "linear": 2, "relu": 6}
        or raw.get("warmup_count") != MR0_WARMUP
        or raw.get("group_count") != MR0_GROUP_COUNT
        or raw.get("repeats_per_side") != MR0_REPEATS
        or raw.get("event_budgets") != list(MR0_BUDGETS)
        or raw.get("event_object_count") != 36
        or not isinstance(rows, list)
        or len(rows) != len(MR0_BUDGETS)
        or not isinstance(semantic, dict)
        or semantic.get("exact") is not True
        or semantic.get("maximum_absolute_difference") != 0.0
        or semantic.get("pointer_stable") is not True
        or semantic.get("contract_stable") is not True
        or semantic.get("candidate_conv_launch_count") != 6
        or semantic.get("fallback_count") != 0
        or semantic.get("eager_shadow_count") != 0
        or raw.get("semantic_admitted") is not True
        or raw.get("stream_before") != raw.get("stream_after")
        or raw.get("stream_admitted") is not True
        or not isinstance(environment, dict)
        or environment.get("measurement_backend") != "torch-cuda-event-no-profiler"
        or raw.get("compile_excluded") is not True
        or raw.get("cuda_graph") is not True
        or raw.get("input_copy_included") is not True
        or raw.get("performance_claimed") is not False
    ):
        raise ValueError("MR0 worker envelope differs")
    rebuilt_rows = []
    for budget, row in zip(MR0_BUDGETS, rows):
        if not isinstance(row, Mapping):
            raise TypeError("MR0 worker budget row differs")
        rebuilt = validate_budget_row(row)
        if rebuilt["budget"] != budget:
            raise ValueError("MR0 worker budget order differs")
        rebuilt_rows.append(rebuilt)
    return {**dict(raw), "budget_rows": rebuilt_rows}


def _summary(raws: list[dict[str, Any]]) -> dict[str, object]:
    workers = [_validate_worker(raw, ordinal) for ordinal, raw in enumerate(raws)]
    return derive_summary(workers)


def _protocol(revision: str) -> dict[str, object]:
    if _file_hash(SOURCE_CAPTURE) != SOURCE_CAPTURE_SHA256:
        raise ValueError("MR0 source capture digest differs")
    if _file_hash(MODEL) != MODEL_SHA256:
        raise ValueError("MR0 model digest differs")
    result: dict[str, object] = {
        "schema_version": "boundflow.mr0-explicit-event-budget-protocol/v1",
        "source_revision": revision,
        "source_capture_sha256": SOURCE_CAPTURE_SHA256,
        "model_sha256": MODEL_SHA256,
        "worker_count": MR0_WORKER_COUNT,
        "orders": list(MR0_ORDERS),
        "event_budgets": list(MR0_BUDGETS),
        "decision_budget": 17,
        "warmup_count": MR0_WARMUP,
        "group_count": MR0_GROUP_COUNT,
        "repeats_per_side": MR0_REPEATS,
        "bootstrap_seed": MR0_BOOTSTRAP_SEED,
        "bootstrap_samples": MR0_BOOTSTRAP_SAMPLES,
        "geomean_gate": MR0_GEOMEAN_GATE,
        "bootstrap_upper_gate": MR0_BOOTSTRAP_UPPER_GATE,
        "worst_worker_gate": MR0_WORST_GATE,
        "threads_per_block": 128,
        "production_op_count": 17,
        "input_copy_included": True,
        "compile_excluded": True,
        "cuda_graph": True,
        "measurement_backend": "torch-cuda-event-no-profiler",
        "same_solver_open": False,
        "r2_open": False,
        "performance_claimed": False,
        "code_revision": {path: _file_hash(ROOT / path) for path in CODE_PATHS},
    }
    result["protocol_hash"] = canonical_hash(result)
    return result


def generate(output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"MR0 artifact exists: {output}")
    _clean()
    protocol = _protocol(_git("rev-parse", "HEAD"))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="mr0-event-budget-", dir=output.parent))
    try:
        raw_dir, log_dir = temporary / "raw", temporary / "logs"
        raw_dir.mkdir()
        log_dir.mkdir()
        raws = []
        for ordinal, order in enumerate(MR0_ORDERS):
            target = raw_dir / f"run_{ordinal:02d}_{order.lower()}.json"
            completed = subprocess.run(
                (
                    sys.executable,
                    str(WORKER),
                    "--source-capture",
                    str(SOURCE_CAPTURE),
                    "--model",
                    str(MODEL),
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
                _portable_log(completed.stdout), encoding="utf-8"
            )
            (log_dir / f"run_{ordinal:02d}.stderr.txt").write_text(
                _portable_log(completed.stderr), encoding="utf-8"
            )
            raws.append(_json(target))
        summary = _summary(raws)
        _write(temporary / "protocol.json", protocol)
        _write(temporary / "summary.json", summary)
        files = {
            str(path.relative_to(temporary)): _file_hash(path)
            for path in sorted(temporary.rglob("*"))
            if path.is_file()
        }
        manifest: dict[str, object] = {
            "schema_version": "boundflow.mr0-explicit-event-budget-manifest/v1",
            "source_revision": protocol["source_revision"],
            "protocol_hash": protocol["protocol_hash"],
            "summary_hash": summary["summary_hash"],
            "files": files,
        }
        manifest["manifest_hash"] = canonical_hash(manifest)
        _write(temporary / "manifest.json", manifest)
        replay(temporary)
        temporary.rename(output)
        return summary
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def replay(artifact: Path) -> dict[str, object]:
    manifest = _json(artifact / "manifest.json")
    unsigned_manifest = dict(manifest)
    manifest_hash = unsigned_manifest.pop("manifest_hash", None)
    if manifest.get(
        "schema_version"
    ) != "boundflow.mr0-explicit-event-budget-manifest/v1" or manifest_hash != canonical_hash(
        unsigned_manifest
    ):
        raise ValueError("MR0 manifest differs")
    files = manifest.get("files")
    if not isinstance(files, dict) or any(
        _file_hash(artifact / str(name)) != digest for name, digest in files.items()
    ):
        raise ValueError("MR0 file digest differs")
    for name in files:
        if str(name).endswith(".txt"):
            text = (artifact / str(name)).read_text(encoding="utf-8")
            if _portable_log(text) != text or any(
                token in text for token in ("/home/", "file://", "\\Users\\")
            ):
                raise ValueError("MR0 log leaks a local path")
    protocol = _json(artifact / "protocol.json")
    unsigned_protocol = dict(protocol)
    protocol_hash = unsigned_protocol.pop("protocol_hash", None)
    if protocol_hash != canonical_hash(
        unsigned_protocol
    ) or protocol_hash != manifest.get("protocol_hash"):
        raise ValueError("MR0 protocol hash differs")
    frozen = {
        "source_capture_sha256": SOURCE_CAPTURE_SHA256,
        "model_sha256": MODEL_SHA256,
        "worker_count": MR0_WORKER_COUNT,
        "orders": list(MR0_ORDERS),
        "event_budgets": list(MR0_BUDGETS),
        "decision_budget": 17,
        "warmup_count": MR0_WARMUP,
        "group_count": MR0_GROUP_COUNT,
        "repeats_per_side": MR0_REPEATS,
        "bootstrap_seed": MR0_BOOTSTRAP_SEED,
        "bootstrap_samples": MR0_BOOTSTRAP_SAMPLES,
        "geomean_gate": MR0_GEOMEAN_GATE,
        "bootstrap_upper_gate": MR0_BOOTSTRAP_UPPER_GATE,
        "worst_worker_gate": MR0_WORST_GATE,
        "threads_per_block": 128,
        "production_op_count": 17,
        "input_copy_included": True,
        "compile_excluded": True,
        "cuda_graph": True,
        "measurement_backend": "torch-cuda-event-no-profiler",
        "same_solver_open": False,
        "r2_open": False,
        "performance_claimed": False,
    }
    if any(protocol.get(name) != value for name, value in frozen.items()):
        raise ValueError("MR0 frozen protocol differs")
    revision, code = protocol.get("source_revision"), protocol.get("code_revision")
    if (
        not isinstance(revision, str)
        or not isinstance(code, dict)
        or set(code) != set(CODE_PATHS)
        or any(
            _git_blob_hash(revision, path) != digest for path, digest in code.items()
        )
    ):
        raise ValueError("MR0 source binding differs")
    raws = [_json(path) for path in sorted((artifact / "raw").glob("*.json"))]
    summary = _summary(raws)
    if summary != _json(artifact / "summary.json") or summary[
        "summary_hash"
    ] != manifest.get("summary_hash"):
        raise ValueError("MR0 semantic replay differs")
    decision = cast(list[Mapping[str, Any]], summary["budget_summaries"])[-1]
    print(
        f"MR0 replay PASS: budget17 geomean="
        f"{decision['geomean_overhead_ratio']:.6f} "
        f"upper={decision['bootstrap_95_upper']:.6f} "
        f"verdict={summary['verdict']}",
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
