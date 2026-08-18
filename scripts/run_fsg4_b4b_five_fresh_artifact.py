#!/usr/bin/env python3
"""Generate or replay the B4-B0 five-fresh dual-anchor artifact."""

# pylint: disable=wrong-import-position,protected-access,too-many-locals
# pylint: disable=too-many-statements,too-many-branches,too-many-boolean-expressions
# pylint: disable=missing-function-docstring,duplicate-code

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
from typing import Any, Mapping, cast

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from boundflow.runtime.fsg4_b4b_production_region_capture import (
    ProductionDifferentiableRegionCaptureV1,
    production_differentiable_region_capture_from_payload_v1,
)
from scripts import run_fsg4_b4b_capture_worker as worker

ARTIFACT_SCHEMA = "boundflow.fsg4-b4b0-five-fresh-artifact/v1"
PROTOCOL_SCHEMA = "boundflow.fsg4-b4b0-five-fresh-protocol/v1"
SUMMARY_SCHEMA = "boundflow.fsg4-b4b0-five-fresh-summary/v1"
RUN_COUNT = 5
ATOL = 2e-4
RTOL = 2e-4
RUN_FILES = tuple(f"run_{index:02d}.pt" for index in range(RUN_COUNT))
STDOUT_FILES = tuple(f"run_{index:02d}.stdout.txt" for index in range(RUN_COUNT))
ARTIFACT_FILES = (
    ("protocol.json",)
    + RUN_FILES
    + STDOUT_FILES
    + ("summary.json", "replay_stdout.txt", "README.md")
)
CODE_PATHS = (
    "boundflow/runtime/crown_ibp.py",
    "boundflow/runtime/fsg4_b3_terminal_optimizer_schedule.py",
    "boundflow/runtime/fsg4_b4b_production_region_capture.py",
    "scripts/run_fsg4_b4b_capture_worker.py",
    "scripts/run_fsg4_b4b_five_fresh_artifact.py",
    "scripts/probe_fsg4_b4b_five_fresh_tamper.py",
)


def _canonical_json(value: object, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
        allow_nan=False,
    )


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(_canonical_json(value, indent=2) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"FSG4/B4-B0 JSON root differs: {path}")
    return value


def _load_torch(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise TypeError(f"FSG4/B4-B0 torch root differs: {path}")
    return value


def _git(*args: str) -> str:
    completed = subprocess.run(
        ("git", *args),
        cwd=REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def _code_revision() -> dict[str, str]:
    return {path: _file_sha256(REPOSITORY_ROOT / path) for path in CODE_PATHS}


def _verify_code_provenance(manifest: Mapping[str, Any]) -> None:
    source = manifest.get("source_git_head")
    revision = manifest.get("code_revision")
    if not isinstance(source, str) or not isinstance(revision, Mapping):
        raise ValueError("FSG4/B4-B0 code provenance differs")
    if _git("rev-parse", "HEAD") == source:
        observed = _code_revision()
    else:
        observed = {
            path: hashlib.sha256(
                subprocess.run(
                    ("git", "show", f"{source}:{path}"),
                    cwd=REPOSITORY_ROOT,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                ).stdout
            ).hexdigest()
            for path in CODE_PATHS
        }
    if dict(revision) != observed:
        raise ValueError("FSG4/B4-B0 code revision differs")


def _protocol(source_capture: Path, model: Path) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": PROTOCOL_SCHEMA,
        "source_git_head": _git("rev-parse", "HEAD"),
        "code_revision": _code_revision(),
        "run_count": RUN_COUNT,
        "run_indices": list(range(RUN_COUNT)),
        "process_isolation": "one-fresh-subprocess-per-capture",
        "evaluation_ordinal": 0,
        "anchor_count_per_run": 2,
        "direct_semantic_atol": ATOL,
        "direct_semantic_rtol": RTOL,
        "sign_exact": True,
        "default_cuda_stream_required": True,
        "source_alias_pairs_required": [],
        "source_capture_sha256": _file_sha256(source_capture),
        "model_sha256": _file_sha256(model),
        "performance_claimed": False,
        "tir_admitted": False,
    }
    payload["protocol_hash"] = _canonical_hash(payload)
    return payload


def _validate_protocol(value: Mapping[str, Any]) -> None:
    semantic = dict(value)
    claimed = semantic.pop("protocol_hash", None)
    if (
        value.get("schema_version") != PROTOCOL_SCHEMA
        or value.get("run_count") != RUN_COUNT
        or value.get("run_indices") != list(range(RUN_COUNT))
        or value.get("process_isolation") != "one-fresh-subprocess-per-capture"
        or value.get("evaluation_ordinal") != 0
        or value.get("anchor_count_per_run") != 2
        or value.get("direct_semantic_atol") != ATOL
        or value.get("direct_semantic_rtol") != RTOL
        or value.get("sign_exact") is not True
        or value.get("default_cuda_stream_required") is not True
        or value.get("source_alias_pairs_required") != []
        or value.get("performance_claimed") is not False
        or value.get("tir_admitted") is not False
        or claimed != _canonical_hash(semantic)
    ):
        raise ValueError("FSG4/B4-B0 protocol differs")


def _captures(
    payload: Mapping[str, Any],
) -> list[ProductionDifferentiableRegionCaptureV1]:
    raw = payload.get("captures")
    if not isinstance(raw, list) or len(raw) != 2:
        raise ValueError("FSG4/B4-B0 capture inventory differs")
    captures = [
        production_differentiable_region_capture_from_payload_v1(item)
        for item in raw
        if isinstance(item, Mapping)
    ]
    if len(captures) != 2 or [item.anchor.role for item in captures] != [
        "semantic",
        "performance",
    ]:
        raise ValueError("FSG4/B4-B0 anchor ordering differs")
    return captures


def _validate_run(
    payload: Mapping[str, Any], *, run_index: int, protocol: Mapping[str, Any]
) -> list[ProductionDifferentiableRegionCaptureV1]:
    environment = payload.get("environment")
    if (
        payload.get("schema_version") != worker.WORKER_SCHEMA
        or payload.get("run_index") != run_index
        or payload.get("source_capture_sha256") != protocol.get("source_capture_sha256")
        or payload.get("model_sha256") != protocol.get("model_sha256")
        or payload.get("evaluation_count") != 10
        or payload.get("update_count") != 9
        or payload.get("performance_claimed") is not False
        or not isinstance(environment, Mapping)
        or environment.get("device_index") != 0
        or environment.get("compute_capability") != [8, 9]
    ):
        raise ValueError("FSG4/B4-B0 worker envelope differs")
    return _captures(payload)


def _discrete_projection(
    capture: ProductionDifferentiableRegionCaptureV1,
) -> dict[str, object]:
    metadata = capture.metadata()
    metadata.pop("capture_hash")
    metadata.pop("source_cuda_stream_id")
    for category in ("values", "gradients"):
        records = cast(dict[str, dict[str, object]], metadata[category])
        for record in records.values():
            record.pop("content_sha256")
    return metadata


def _compare_tensor(
    actual: torch.Tensor, expected: torch.Tensor, *, label: str
) -> tuple[int, float]:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError(f"FSG4/B4-B0 tensor structure differs: {label}")
    if not torch.allclose(actual, expected, atol=ATOL, rtol=RTOL):
        raise ValueError(f"FSG4/B4-B0 tensor numeric differs: {label}")
    if not torch.equal(torch.sign(actual), torch.sign(expected)):
        raise ValueError(f"FSG4/B4-B0 tensor sign differs: {label}")
    maximum = float((actual - expected).abs().max().item()) if actual.numel() else 0.0
    return actual.numel(), maximum


def _compare_capture(
    actual: ProductionDifferentiableRegionCaptureV1,
    expected: ProductionDifferentiableRegionCaptureV1,
    *,
    run_index: int,
) -> dict[str, object]:
    if _discrete_projection(actual) != _discrete_projection(expected):
        raise ValueError("FSG4/B4-B0 five-fresh discrete structure differs")
    comparisons = 0
    elements = 0
    maximum = 0.0
    for category, left, right in (
        ("value", actual.value_map, expected.value_map),
        ("gradient", actual.gradient_map, expected.gradient_map),
    ):
        if set(left) != set(right):
            raise ValueError(f"FSG4/B4-B0 {category} inventory differs")
        for name in sorted(left):
            count, difference = _compare_tensor(
                left[name].value,
                right[name].value,
                label=f"run-{run_index}:{actual.anchor.anchor_id}:{category}:{name}",
            )
            comparisons += 1
            elements += count
            maximum = max(maximum, difference)
    return {
        "run_index": run_index,
        "anchor_id": actual.anchor.anchor_id,
        "tensor_comparison_count": comparisons,
        "element_comparison_count": elements,
        "maximum_absolute_difference": maximum,
        "all_sign_exact": True,
        "discrete_structure_exact": True,
    }


def _summary(
    runs: list[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> dict[str, object]:
    if len(runs) != RUN_COUNT:
        raise ValueError("FSG4/B4-B0 run count differs")
    captures = [
        _validate_run(payload, run_index=index, protocol=protocol)
        for index, payload in enumerate(runs)
    ]
    source_state_hash = runs[0].get("source_state_hash")
    schedule_hash = runs[0].get("schedule_hash")
    environment = runs[0].get("environment")
    if any(
        run.get("source_state_hash") != source_state_hash
        or run.get("schedule_hash") != schedule_hash
        or run.get("environment") != environment
        for run in runs[1:]
    ):
        raise ValueError("FSG4/B4-B0 run identity differs")
    comparisons = [
        _compare_capture(captures[index][anchor], captures[0][anchor], run_index=index)
        for index in range(1, RUN_COUNT)
        for anchor in range(2)
    ]
    summary: dict[str, object] = {
        "schema_version": SUMMARY_SCHEMA,
        "status": "validated-b4-b0-five-fresh-capture",
        "protocol_hash": protocol["protocol_hash"],
        "run_count": RUN_COUNT,
        "capture_count": RUN_COUNT * 2,
        "semantic_anchor_count": RUN_COUNT,
        "performance_anchor_count": RUN_COUNT,
        "source_state_hash": source_state_hash,
        "schedule_hash": schedule_hash,
        "environment": environment,
        "comparisons": comparisons,
        "tensor_comparison_count": sum(
            cast(int, row["tensor_comparison_count"]) for row in comparisons
        ),
        "element_comparison_count": sum(
            cast(int, row["element_comparison_count"]) for row in comparisons
        ),
        "maximum_absolute_difference": max(
            cast(float, row["maximum_absolute_difference"]) for row in comparisons
        ),
        "all_discrete_structure_exact": True,
        "all_numeric_within_tolerance": True,
        "all_sign_exact": True,
        "root_raw_replay_passed": True,
        "performance_claimed": False,
        "tir_admitted": False,
    }
    summary["summary_hash"] = _canonical_hash(summary)
    return summary


def _result(summary: Mapping[str, Any]) -> dict[str, object]:
    return {
        "status": "replay-passed",
        "evidence_status": summary["status"],
        "run_count": summary["run_count"],
        "capture_count": summary["capture_count"],
        "maximum_absolute_difference": summary["maximum_absolute_difference"],
        "all_sign_exact": summary["all_sign_exact"],
        "summary_hash": summary["summary_hash"],
        "performance_claimed": False,
        "tir_admitted": False,
    }


def _readme() -> str:
    return (
        "# FSG4/B4-B0 Five-Fresh Production Capture\n\n"
        "Five isolated CUDA processes capture the evaluation-0 semantic Gemm and "
        "performance Conv anchors. Root replay rebuilds each typed capture from raw "
        "tensor payload and checks discrete structure, tolerance, and sign. This is "
        "capture correctness only; performance_claimed=false and tir_admitted=false.\n"
    )


def _run_worker(
    *, python: Path, source_capture: Path, model: Path, run_index: int, result: Path
) -> str:
    environment = dict(os.environ)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT)
    completed = subprocess.run(
        (
            str(python),
            str(REPOSITORY_ROOT / "scripts/run_fsg4_b4b_capture_worker.py"),
            "--source-capture",
            str(source_capture),
            "--model",
            str(model),
            "--run-index",
            str(run_index),
            "--result",
            str(result),
        ),
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=180,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"FSG4/B4-B0 worker failed run={run_index}:\n{completed.stdout}"
        )
    if "/home/" in completed.stdout or "/tmp/" in completed.stdout:
        raise ValueError("FSG4/B4-B0 worker log contains host-local path")
    return completed.stdout


def _all_files(root: Path) -> dict[str, str]:
    return {name: _file_sha256(root / name) for name in ARTIFACT_FILES}


def _generate(args: argparse.Namespace) -> dict[str, object]:
    if _git("status", "--porcelain=v1", "--", *CODE_PATHS):
        raise ValueError("FSG4/B4-B0 code paths must be committed")
    output = args.artifact_dir.resolve()
    if output.exists():
        raise FileExistsError(f"FSG4/B4-B0 artifact exists: {output}")
    source_capture = args.source_capture.resolve()
    model = args.model.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.incomplete-", dir=output.parent
    ) as temporary:
        root = Path(temporary)
        protocol = _protocol(source_capture, model)
        _write_json(root / "protocol.json", protocol)
        runs = []
        for run_index in range(RUN_COUNT):
            stdout = _run_worker(
                python=args.python.expanduser().absolute(),
                source_capture=source_capture,
                model=model,
                run_index=run_index,
                result=root / RUN_FILES[run_index],
            )
            (root / STDOUT_FILES[run_index]).write_text(stdout, encoding="utf-8")
            runs.append(_load_torch(root / RUN_FILES[run_index]))
        summary = _summary(cast(list[Mapping[str, Any]], runs), protocol)
        _write_json(root / "summary.json", summary)
        result = _result(summary)
        (root / "replay_stdout.txt").write_text(
            _canonical_json(result) + "\n", encoding="utf-8"
        )
        (root / "README.md").write_text(_readme(), encoding="utf-8")
        manifest: dict[str, object] = {
            "schema_version": ARTIFACT_SCHEMA,
            "source_git_head": _git("rev-parse", "HEAD"),
            "code_revision": _code_revision(),
            "protocol_hash": protocol["protocol_hash"],
            "summary_hash": summary["summary_hash"],
            "files": _all_files(root),
            "performance_claimed": False,
            "tir_admitted": False,
        }
        manifest["manifest_hash"] = _canonical_hash(manifest)
        _write_json(root / "manifest.json", manifest)
        shutil.move(root, output)
    _verify_static_artifact(output)
    return result


def _verify_static_artifact(
    artifact: Path,
) -> tuple[list[dict[str, Any]], dict[str, object], dict[str, object]]:
    artifact = artifact.resolve()
    manifest = _load_json(artifact / "manifest.json")
    semantic = dict(manifest)
    claimed = semantic.pop("manifest_hash", None)
    if (
        manifest.get("schema_version") != ARTIFACT_SCHEMA
        or claimed != _canonical_hash(semantic)
        or manifest.get("performance_claimed") is not False
        or manifest.get("tir_admitted") is not False
    ):
        raise ValueError("FSG4/B4-B0 manifest differs")
    _verify_code_provenance(manifest)
    files = manifest.get("files")
    if not isinstance(files, Mapping) or dict(files) != _all_files(artifact):
        raise ValueError("FSG4/B4-B0 artifact inventory differs")
    protocol = _load_json(artifact / "protocol.json")
    _validate_protocol(protocol)
    runs = [_load_torch(artifact / name) for name in RUN_FILES]
    summary = _summary(cast(list[Mapping[str, Any]], runs), protocol)
    if (
        _load_json(artifact / "summary.json") != summary
        or manifest.get("protocol_hash") != protocol["protocol_hash"]
        or manifest.get("summary_hash") != summary["summary_hash"]
    ):
        raise ValueError("FSG4/B4-B0 semantic replay differs")
    result = _result(summary)
    if (artifact / "replay_stdout.txt").read_text(encoding="utf-8") != (
        _canonical_json(result) + "\n"
    ) or (artifact / "README.md").read_text(encoding="utf-8") != _readme():
        raise ValueError("FSG4/B4-B0 projection replay differs")
    return runs, summary, result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--artifact-dir", type=Path, required=True)
    generate.add_argument("--source-capture", type=Path, required=True)
    generate.add_argument("--model", type=Path, required=True)
    generate.add_argument("--python", type=Path, required=True)
    replay = commands.add_parser("replay")
    replay.add_argument("--artifact-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "generate":
        result = _generate(args)
    else:
        _runs, _summary_payload, result = _verify_static_artifact(args.artifact_dir)
    print(_canonical_json(result), flush=True)


if __name__ == "__main__":
    main()
