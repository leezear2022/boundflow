#!/usr/bin/env python3
"""Generate or replay the B4-B1 reference-sufficiency five-fresh artifact."""

# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=too-many-boolean-expressions,protected-access,duplicate-code
# pylint: disable=missing-function-docstring

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, cast, Mapping

import torch

from boundflow.runtime.fsg4_b4b1_reference_capture import (
    ProductionDifferentiableReferenceCaptureV1,
    production_differentiable_reference_capture_from_payload_v1,
)
from scripts import run_fsg4_b4b_capture_worker as b4b0_worker
from scripts import run_fsg4_b4b_five_fresh_artifact as b4b0_artifact
from scripts import run_fsg4_b4b1_reference_capture_worker as worker

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_SCHEMA = "boundflow.fsg4-b4b1-reference-five-fresh-artifact/v1"
PROTOCOL_SCHEMA = "boundflow.fsg4-b4b1-reference-five-fresh-protocol/v1"
SUMMARY_SCHEMA = "boundflow.fsg4-b4b1-reference-five-fresh-summary/v1"
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
    "boundflow/runtime/fsg4_b4b1_reference_capture.py",
    "scripts/run_fsg4_b4b_capture_worker.py",
    "scripts/run_fsg4_b4b_five_fresh_artifact.py",
    "scripts/run_fsg4_b4b1_reference_capture_worker.py",
    "scripts/run_fsg4_b4b1_reference_five_fresh_artifact.py",
    "scripts/probe_fsg4_b4b1_reference_capture_integrity.py",
)


def _canonical_json(value: object, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        allow_nan=False,
        indent=indent,
    )


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        raise ValueError("FSG4/B4-B1 code provenance differs")
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
        raise ValueError("FSG4/B4-B1 code revision differs")


def _protocol(source_capture: Path, model: Path) -> dict[str, object]:
    base_protocol = b4b0_artifact._protocol(source_capture, model)
    payload: dict[str, object] = {
        "schema_version": PROTOCOL_SCHEMA,
        "source_git_head": _git("rev-parse", "HEAD"),
        "code_revision": _code_revision(),
        "base_protocol": base_protocol,
        "base_protocol_hash": base_protocol["protocol_hash"],
        "source_capture_sha256": _file_sha256(source_capture),
        "model_sha256": _file_sha256(model),
        "run_count": RUN_COUNT,
        "run_indices": list(range(RUN_COUNT)),
        "process_isolation": "one-fresh-subprocess-per-capture",
        "evaluation_ordinal": 0,
        "anchors": [
            "semantic-active-beta-gemm-14",
            "performance-conv-8-candidate",
        ],
        "required_amendment_values": [
            "incoming_lower_bias",
            "operator_bias",
            "output_lower_a_gradient",
            "output_bias_gradient",
            "sparse_mapping_tensors",
        ],
        "atol": ATOL,
        "rtol": RTOL,
        "sign_exact": True,
        "performance_claimed": False,
        "tir_admitted": False,
    }
    payload["protocol_hash"] = _canonical_hash(payload)
    return payload


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    semantic = dict(protocol)
    claimed = semantic.pop("protocol_hash", None)
    base = protocol.get("base_protocol")
    if not isinstance(base, Mapping):
        raise TypeError("FSG4/B4-B1 base protocol differs")
    b4b0_artifact._validate_protocol(base)
    if (
        protocol.get("schema_version") != PROTOCOL_SCHEMA
        or protocol.get("source_git_head") != base.get("source_git_head")
        or protocol.get("base_protocol_hash") != base.get("protocol_hash")
        or protocol.get("source_capture_sha256")
        != b4b0_artifact.FROZEN_SOURCE_IDENTITY["source_capture_sha256"]
        or protocol.get("model_sha256")
        != b4b0_artifact.FROZEN_SOURCE_IDENTITY["model_sha256"]
        or protocol.get("run_count") != RUN_COUNT
        or protocol.get("run_indices") != list(range(RUN_COUNT))
        or protocol.get("process_isolation") != "one-fresh-subprocess-per-capture"
        or protocol.get("evaluation_ordinal") != 0
        or protocol.get("anchors")
        != ["semantic-active-beta-gemm-14", "performance-conv-8-candidate"]
        or protocol.get("atol") != ATOL
        or protocol.get("rtol") != RTOL
        or protocol.get("sign_exact") is not True
        or protocol.get("performance_claimed") is not False
        or protocol.get("tir_admitted") is not False
        or claimed != _canonical_hash(semantic)
    ):
        raise ValueError("FSG4/B4-B1 protocol differs")


def _load_torch(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise TypeError(f"FSG4/B4-B1 tensor payload differs: {path.name}")
    return cast(dict[str, Any], value)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"FSG4/B4-B1 JSON payload differs: {path.name}")
    return cast(dict[str, Any], value)


def _write_json(path: Path, value: object) -> None:
    path.write_text(_canonical_json(value, indent=2) + "\n", encoding="utf-8")


def _reference_captures(
    payload: Mapping[str, Any],
) -> list[ProductionDifferentiableReferenceCaptureV1]:
    raw = payload.get("captures")
    if not isinstance(raw, list) or len(raw) != 2:
        raise ValueError("FSG4/B4-B1 worker capture inventory differs")
    captures = [
        production_differentiable_reference_capture_from_payload_v1(item)
        for item in raw
        if isinstance(item, Mapping)
    ]
    if len(captures) != 2 or [
        capture.base.anchor.anchor_id for capture in captures
    ] != ["semantic-active-beta-gemm-14", "performance-conv-8-candidate"]:
        raise ValueError("FSG4/B4-B1 worker anchor order differs")
    return captures


def _project_base_run(payload: Mapping[str, Any]) -> dict[str, Any]:
    projected = dict(payload)
    projected["schema_version"] = b4b0_worker.WORKER_SCHEMA
    projected["captures"] = [
        cast(Mapping[str, Any], item)["base"]
        for item in cast(list[object], payload["captures"])
    ]
    projected.pop("tir_admitted", None)
    return projected


def _validate_run(
    payload: Mapping[str, Any], *, run_index: int, protocol: Mapping[str, Any]
) -> list[ProductionDifferentiableReferenceCaptureV1]:
    environment = payload.get("environment")
    if (
        payload.get("schema_version") != worker.WORKER_SCHEMA
        or payload.get("run_index") != run_index
        or payload.get("source_capture_sha256") != protocol.get("source_capture_sha256")
        or payload.get("model_sha256") != protocol.get("model_sha256")
        or payload.get("source_state_hash")
        != b4b0_artifact.FROZEN_SOURCE_IDENTITY["source_state_hash"]
        or payload.get("schedule_hash")
        != b4b0_artifact.FROZEN_SOURCE_IDENTITY["schedule_hash"]
        or payload.get("evaluation_count") != 10
        or payload.get("update_count") != 9
        or payload.get("performance_claimed") is not False
        or payload.get("tir_admitted") is not False
        or not isinstance(environment, Mapping)
        or environment.get("compute_capability") != [8, 9]
    ):
        raise ValueError("FSG4/B4-B1 worker envelope differs")
    return _reference_captures(payload)


def _discrete_projection(
    capture: ProductionDifferentiableReferenceCaptureV1,
) -> dict[str, object]:
    metadata = capture.metadata()

    def scrub(value: object) -> object:
        if isinstance(value, dict):
            return {
                key: scrub(item)
                for key, item in sorted(value.items())
                if key
                not in {
                    "content_sha256",
                    "capture_hash",
                    "base_capture_hash",
                    "reference_capture_hash",
                    "source_cuda_stream_id",
                }
            }
        if isinstance(value, list):
            return [scrub(item) for item in value]
        return value

    projected = scrub(metadata)
    if not isinstance(projected, dict):
        raise TypeError("FSG4/B4-B1 discrete projection differs")
    return cast(dict[str, object], projected)


def _amendment_tensors(
    capture: ProductionDifferentiableReferenceCaptureV1,
) -> dict[str, torch.Tensor]:
    tensors = {
        "incoming_lower_bias": capture.incoming_lower_bias.value,
        "output_gradient:output_lower_a": capture.output_lower_a_gradient.value,
        "output_gradient:output_bias": capture.output_bias_gradient.value,
        **{
            f"mapping:{name}": snapshot.value
            for name, snapshot in capture.mapping_tensors
        },
    }
    if capture.operator_bias is not None:
        tensors["operator_bias"] = capture.operator_bias.value
    return tensors


def _compare_tensor(
    reference: torch.Tensor, candidate: torch.Tensor
) -> tuple[float, bool]:
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        raise ValueError("FSG4/B4-B1 amendment tensor structure differs")
    if not (reference.is_floating_point() or reference.is_complex()):
        if not torch.equal(reference, candidate):
            raise ValueError("FSG4/B4-B1 amendment discrete tensor differs")
        return 0.0, True
    difference = (reference - candidate).abs()
    maximum = 0.0 if difference.numel() == 0 else float(difference.max().item())
    if not torch.allclose(reference, candidate, atol=ATOL, rtol=RTOL):
        raise ValueError("FSG4/B4-B1 amendment numeric tensor differs")
    sign_exact = torch.equal(torch.sign(reference), torch.sign(candidate))
    if not sign_exact:
        raise ValueError("FSG4/B4-B1 amendment tensor sign differs")
    return maximum, sign_exact


def _summary(
    runs: list[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> dict[str, object]:
    if len(runs) != RUN_COUNT:
        raise ValueError("FSG4/B4-B1 run count differs")
    captures = [
        _validate_run(run, run_index=index, protocol=protocol)
        for index, run in enumerate(runs)
    ]
    base_runs = [_project_base_run(run) for run in runs]
    base_summary = b4b0_artifact._summary(
        cast(list[Mapping[str, Any]], base_runs),
        cast(Mapping[str, Any], protocol["base_protocol"]),
    )
    discrete = [_discrete_projection(item) for item in captures[0]]
    maximum = 0.0
    tensor_comparison_count = 0
    element_comparison_count = 0
    for run in captures:
        for anchor_index, capture in enumerate(run):
            if _discrete_projection(capture) != discrete[anchor_index]:
                raise ValueError("FSG4/B4-B1 amendment discrete structure differs")
            reference_tensors = _amendment_tensors(captures[0][anchor_index])
            candidate_tensors = _amendment_tensors(capture)
            if set(reference_tensors) != set(candidate_tensors):
                raise ValueError("FSG4/B4-B1 amendment tensor inventory differs")
            for name in sorted(reference_tensors):
                difference, _sign_exact = _compare_tensor(
                    reference_tensors[name], candidate_tensors[name]
                )
                maximum = max(maximum, difference)
                tensor_comparison_count += 1
                element_comparison_count += reference_tensors[name].numel()
    summary: dict[str, object] = {
        "schema_version": SUMMARY_SCHEMA,
        "status": "validated-b4-b1a-five-fresh-capture-sufficiency",
        "protocol_hash": protocol["protocol_hash"],
        "base_summary_hash": base_summary["summary_hash"],
        "run_count": RUN_COUNT,
        "capture_count": RUN_COUNT * 2,
        "semantic_anchor_count": RUN_COUNT,
        "performance_anchor_count": RUN_COUNT,
        "amendment_tensor_comparison_count": tensor_comparison_count,
        "amendment_element_comparison_count": element_comparison_count,
        "maximum_amendment_absolute_difference": maximum,
        "all_amendment_sign_exact": True,
        "root_raw_replay_passed": True,
        "bias_and_output_adjoint_present": True,
        "sparse_mapping_raw_present": True,
        "operator_bias_present": [True, True],
        "base_maximum_absolute_difference": base_summary["maximum_absolute_difference"],
        "base_all_sign_exact": base_summary["all_sign_exact"],
        "run_reference_capture_hashes": [
            [capture.metadata()["reference_capture_hash"] for capture in run]
            for run in captures
        ],
        "performance_claimed": False,
        "tir_admitted": False,
    }
    summary["summary_hash"] = _canonical_hash(summary)
    return summary


def _result(summary: Mapping[str, object]) -> dict[str, object]:
    return {
        "status": "replay-passed",
        "evidence_status": summary["status"],
        "run_count": summary["run_count"],
        "capture_count": summary["capture_count"],
        "amendment_tensor_comparison_count": summary[
            "amendment_tensor_comparison_count"
        ],
        "amendment_element_comparison_count": summary[
            "amendment_element_comparison_count"
        ],
        "maximum_amendment_absolute_difference": summary[
            "maximum_amendment_absolute_difference"
        ],
        "all_amendment_sign_exact": summary["all_amendment_sign_exact"],
        "summary_hash": summary["summary_hash"],
        "performance_claimed": False,
        "tir_admitted": False,
    }


def _readme() -> str:
    return (
        "# FSG4/B4-B1a Five-Fresh Reference Capture\n\n"
        "Five isolated CUDA processes capture the approved B4-B0 base plus incoming "
        "bias, operator bias, region output adjoints, and sparse mapping raw. Root "
        "replay rebuilds both typed layers and checks discrete structure, tolerance, "
        "and sign. This is capture sufficiency only; no typed-reference, TIR, or "
        "performance claim.\n"
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
            str(REPOSITORY_ROOT / "scripts/run_fsg4_b4b1_reference_capture_worker.py"),
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
            f"FSG4/B4-B1 worker failed run={run_index}:\n{completed.stdout}"
        )
    if "/home/" in completed.stdout or "/tmp/" in completed.stdout:
        raise ValueError("FSG4/B4-B1 worker log contains host-local path")
    return completed.stdout


def _all_files(root: Path) -> dict[str, str]:
    return {name: _file_sha256(root / name) for name in ARTIFACT_FILES}


def _generate(args: argparse.Namespace) -> dict[str, object]:
    if _git("status", "--porcelain=v1", "--", *CODE_PATHS):
        raise ValueError("FSG4/B4-B1 code paths must be committed")
    output = args.artifact_dir.resolve()
    if output.exists():
        raise FileExistsError(f"FSG4/B4-B1 artifact exists: {output}")
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
            "base_frozen_source_identity_hash": _canonical_hash(
                b4b0_artifact.FROZEN_SOURCE_IDENTITY
            ),
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
        or manifest.get("base_frozen_source_identity_hash")
        != _canonical_hash(b4b0_artifact.FROZEN_SOURCE_IDENTITY)
        or manifest.get("performance_claimed") is not False
        or manifest.get("tir_admitted") is not False
    ):
        raise ValueError("FSG4/B4-B1 manifest differs")
    _verify_code_provenance(manifest)
    files = manifest.get("files")
    if not isinstance(files, Mapping) or dict(files) != _all_files(artifact):
        raise ValueError("FSG4/B4-B1 artifact inventory differs")
    protocol = _load_json(artifact / "protocol.json")
    _validate_protocol(protocol)
    if manifest.get("source_git_head") != protocol.get(
        "source_git_head"
    ) or manifest.get("code_revision") != protocol.get("code_revision"):
        raise ValueError("FSG4/B4-B1 manifest protocol identity differs")
    runs = [_load_torch(artifact / name) for name in RUN_FILES]
    summary = _summary(cast(list[Mapping[str, Any]], runs), protocol)
    if (
        _load_json(artifact / "summary.json") != summary
        or manifest.get("protocol_hash") != protocol["protocol_hash"]
        or manifest.get("summary_hash") != summary["summary_hash"]
    ):
        raise ValueError("FSG4/B4-B1 semantic replay differs")
    result = _result(summary)
    if (artifact / "replay_stdout.txt").read_text(encoding="utf-8") != (
        _canonical_json(result) + "\n"
    ) or (artifact / "README.md").read_text(encoding="utf-8") != _readme():
        raise ValueError("FSG4/B4-B1 projection replay differs")
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
