#!/usr/bin/env python3
"""Generate or replay the B4-B1 typed pure-PyTorch reference artifact."""

# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=too-many-boolean-expressions,protected-access,duplicate-code
# pylint: disable=missing-function-docstring,import-error

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, cast, Mapping, Sequence

import torch

from boundflow.runtime.fsg4_b4b1_pytorch_reference import (
    B4B1_REFERENCE_ATOL,
    B4B1_REFERENCE_RTOL,
    build_b4b1_differentiable_lower_instance_v1,
    build_b4b1_differentiable_lower_ir_v1,
    build_b4b1_reference_receipt_v1,
    run_b4b1_pytorch_reference_v1,
)
from scripts import run_fsg4_b4b1_reference_five_fresh_artifact as source_artifact

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_SCHEMA = "boundflow.fsg4-b4b1-pytorch-reference-artifact/v1"
PROTOCOL_SCHEMA = "boundflow.fsg4-b4b1-pytorch-reference-protocol/v1"
SUMMARY_SCHEMA = "boundflow.fsg4-b4b1-pytorch-reference-summary/v1"
RUN_COUNT = 5
REFERENCE_TORCH_THREADS = 1
REFERENCE_DETERMINISTIC_DEBUG_MODE = 2
ANCHORS = (
    "semantic-active-beta-gemm-14",
    "performance-conv-8-candidate",
)
ARTIFACT_FILES = (
    "protocol.json",
    "reference_records.jsonl",
    "summary.json",
    "replay_stdout.txt",
    "README.md",
)
CODE_PATHS = (
    "boundflow/ir/differentiable_lower_region.py",
    "boundflow/runtime/fsg4_b4b1_pytorch_reference.py",
    "boundflow/runtime/fsg4_b4b1_reference_capture.py",
    "boundflow/runtime/fsg4_b4b_production_region_capture.py",
    "scripts/run_fsg4_b4b1_reference_five_fresh_artifact.py",
    "scripts/run_fsg4_b4b1_pytorch_reference_artifact.py",
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
        raise ValueError("FSG4/B4-B1 reference code provenance differs")
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
        raise ValueError("FSG4/B4-B1 reference code revision differs")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"FSG4/B4-B1 reference JSON differs: {path.name}")
    return cast(dict[str, Any], value)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if any(not isinstance(row, dict) for row in rows):
        raise TypeError("FSG4/B4-B1 reference JSONL differs")
    return cast(list[dict[str, Any]], rows)


def _write_json(path: Path, value: object) -> None:
    path.write_text(_canonical_json(value, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(_canonical_json(row) + "\n" for row in rows), encoding="utf-8"
    )


def _source_identity(capture_artifact: Path) -> dict[str, object]:
    _runs, summary, _result = source_artifact._verify_static_artifact(capture_artifact)
    manifest = _load_json(capture_artifact / "manifest.json")
    return {
        "schema_version": manifest["schema_version"],
        "manifest_hash": manifest["manifest_hash"],
        "protocol_hash": manifest["protocol_hash"],
        "summary_hash": summary["summary_hash"],
        "run_file_hashes": {
            name: manifest["files"][name] for name in source_artifact.RUN_FILES
        },
    }


def _protocol(capture_artifact: Path) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": PROTOCOL_SCHEMA,
        "source_git_head": _git("rev-parse", "HEAD"),
        "code_revision": _code_revision(),
        "source_capture_artifact": _source_identity(capture_artifact),
        "run_count": RUN_COUNT,
        "anchors": list(ANCHORS),
        "reference_engine": "public-pytorch-ops-only",
        "private_crown_helper_imported": False,
        "alpha_direction_index": 0,
        "alpha_spec_index": 0,
        "beta_pre_add_formula": "negative-value-times-split-sign-v1",
        "torch_num_threads": REFERENCE_TORCH_THREADS,
        "torch_deterministic_algorithms": True,
        "torch_deterministic_debug_mode": REFERENCE_DETERMINISTIC_DEBUG_MODE,
        "torch_deterministic_state_restore": "exact-debug-mode-v1",
        "torch_float32_matmul_precision": "highest",
        "torch_mkldnn_enabled": False,
        "receipt_metric_inventory": "exact-ir-contract-target-v1",
        "atol": B4B1_REFERENCE_ATOL,
        "rtol": B4B1_REFERENCE_RTOL,
        "sign_exact": True,
        "performance_claimed": False,
        "tir_admitted": False,
    }
    payload["protocol_hash"] = _canonical_hash(payload)
    return payload


def _validate_protocol(protocol: Mapping[str, Any], capture_artifact: Path) -> None:
    semantic = dict(protocol)
    claimed = semantic.pop("protocol_hash", None)
    if (
        protocol.get("schema_version") != PROTOCOL_SCHEMA
        or protocol.get("source_capture_artifact") != _source_identity(capture_artifact)
        or protocol.get("run_count") != RUN_COUNT
        or protocol.get("anchors") != list(ANCHORS)
        or protocol.get("reference_engine") != "public-pytorch-ops-only"
        or protocol.get("private_crown_helper_imported") is not False
        or protocol.get("alpha_direction_index") != 0
        or protocol.get("alpha_spec_index") != 0
        or protocol.get("beta_pre_add_formula") != "negative-value-times-split-sign-v1"
        or protocol.get("torch_num_threads") != REFERENCE_TORCH_THREADS
        or protocol.get("torch_deterministic_algorithms") is not True
        or protocol.get("torch_deterministic_debug_mode")
        != REFERENCE_DETERMINISTIC_DEBUG_MODE
        or protocol.get("torch_deterministic_state_restore") != "exact-debug-mode-v1"
        or protocol.get("torch_float32_matmul_precision") != "highest"
        or protocol.get("torch_mkldnn_enabled") is not False
        or protocol.get("receipt_metric_inventory") != "exact-ir-contract-target-v1"
        or protocol.get("atol") != B4B1_REFERENCE_ATOL
        or protocol.get("rtol") != B4B1_REFERENCE_RTOL
        or protocol.get("sign_exact") is not True
        or protocol.get("performance_claimed") is not False
        or protocol.get("tir_admitted") is not False
        or claimed != _canonical_hash(semantic)
    ):
        raise ValueError("FSG4/B4-B1 reference protocol differs")


def _records(
    runs: list[Mapping[str, Any]], source_protocol: Mapping[str, Any]
) -> list[dict[str, object]]:
    if len(runs) != RUN_COUNT:
        raise ValueError("FSG4/B4-B1 reference run count differs")
    records: list[dict[str, object]] = []
    for run_index, run in enumerate(runs):
        captures = source_artifact._validate_run(
            run,
            run_index=run_index,
            protocol=source_protocol,
        )
        for capture in captures:
            ir = build_b4b1_differentiable_lower_ir_v1(capture)
            instance = build_b4b1_differentiable_lower_instance_v1(capture, ir)
            result = run_b4b1_pytorch_reference_v1(capture, ir, instance)
            receipt = build_b4b1_reference_receipt_v1(capture, ir, instance, result)
            if receipt.semantic_passed is not True:
                raise ValueError(
                    f"FSG4/B4-B1 numerical semantics differ: {ir.anchor_id}"
                )
            row: dict[str, object] = {
                "run_index": run_index,
                "anchor_id": ir.anchor_id,
                "ir": ir.to_dict(),
                "ir_hash": ir.stable_hash(),
                "instance": instance.to_dict(ir),
                "instance_hash": instance.stable_hash(ir),
                "receipt": receipt.to_dict(ir, instance),
                "receipt_hash": receipt.stable_hash(ir, instance),
            }
            row["record_hash"] = _canonical_hash(row)
            records.append(row)
    if [(row["run_index"], row["anchor_id"]) for row in records] != [
        (run_index, anchor) for run_index in range(RUN_COUNT) for anchor in ANCHORS
    ]:
        raise ValueError("FSG4/B4-B1 reference record order differs")
    return records


def _records_from_source(
    capture_artifact: Path, protocol: Mapping[str, Any]
) -> list[dict[str, object]]:
    _validate_protocol(protocol, capture_artifact)
    runs, _summary, _result = source_artifact._verify_static_artifact(capture_artifact)
    source_protocol = source_artifact._load_json(capture_artifact / "protocol.json")
    with _reference_execution_policy():
        return _records(cast(list[Mapping[str, Any]], runs), source_protocol)


@contextmanager
def _reference_execution_policy():
    """Freeze and restore the CPU policy used by exact derived-record replay."""

    previous_threads = torch.get_num_threads()
    previous_deterministic_debug_mode = torch.get_deterministic_debug_mode()
    previous_precision = torch.get_float32_matmul_precision()
    previous_mkldnn = torch.backends.mkldnn.enabled
    try:
        torch.set_num_threads(REFERENCE_TORCH_THREADS)
        torch.set_deterministic_debug_mode(REFERENCE_DETERMINISTIC_DEBUG_MODE)
        torch.set_float32_matmul_precision("highest")
        torch.backends.mkldnn.enabled = False
        yield
    finally:
        torch.backends.mkldnn.enabled = previous_mkldnn
        torch.set_float32_matmul_precision(previous_precision)
        torch.set_deterministic_debug_mode(previous_deterministic_debug_mode)
        torch.set_num_threads(previous_threads)


def _summary(
    records: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> dict[str, object]:
    if len(records) != RUN_COUNT * len(ANCHORS):
        raise ValueError("FSG4/B4-B1 reference record count differs")
    metrics = [
        metric
        for row in records
        for metric in cast(Mapping[str, Any], row["receipt"])["metrics"]
    ]
    ir_hashes = {
        anchor: sorted(
            {cast(str, row["ir_hash"]) for row in records if row["anchor_id"] == anchor}
        )
        for anchor in ANCHORS
    }
    if any(len(values) != 1 for values in ir_hashes.values()):
        raise ValueError("FSG4/B4-B1 static IR varies across fresh runs")
    maximum = max(float(metric["maximum_absolute_difference"]) for metric in metrics)
    summary: dict[str, object] = {
        "schema_version": SUMMARY_SCHEMA,
        "status": "validated-b4b1-typed-pytorch-reference-pending-external-audit",
        "protocol_hash": protocol["protocol_hash"],
        "run_count": RUN_COUNT,
        "capture_count": len(records),
        "anchor_capture_counts": {
            anchor: sum(row["anchor_id"] == anchor for row in records)
            for anchor in ANCHORS
        },
        "static_ir_hashes": {anchor: values[0] for anchor, values in ir_hashes.items()},
        "instance_count": len({row["instance_hash"] for row in records}),
        "receipt_count": len({row["receipt_hash"] for row in records}),
        "metric_comparison_count": len(metrics),
        "element_comparison_count": sum(
            int(metric["element_count"]) for metric in metrics
        ),
        "maximum_absolute_difference": maximum,
        "all_metrics_allclose": all(metric["allclose"] is True for metric in metrics),
        "all_metrics_sign_exact": all(
            metric["sign_exact"] is True for metric in metrics
        ),
        "s_native_beta_gradient_count": sum(
            metric["name"] == "native_beta_gradient" for metric in metrics
        ),
        "p_incoming_a_gradient_count": sum(
            metric["name"] == "incoming_lower_a_gradient" for metric in metrics
        ),
        "s_incoming_a_micro_gate": "covered-by-targeted-test",
        "coordinated_rewrite_integrity": "pending-separate-probe",
        "performance_claimed": False,
        "tir_admitted": False,
    }
    if (
        summary["anchor_capture_counts"] != {anchor: RUN_COUNT for anchor in ANCHORS}
        or summary["all_metrics_allclose"] is not True
        or summary["all_metrics_sign_exact"] is not True
        or summary["s_native_beta_gradient_count"] != RUN_COUNT
        or summary["p_incoming_a_gradient_count"] != RUN_COUNT
    ):
        raise ValueError("FSG4/B4-B1 reference summary gate differs")
    summary["summary_hash"] = _canonical_hash(summary)
    return summary


def _result(summary: Mapping[str, object]) -> dict[str, object]:
    return {
        "status": "replay-passed",
        "evidence_status": summary["status"],
        "run_count": summary["run_count"],
        "capture_count": summary["capture_count"],
        "metric_comparison_count": summary["metric_comparison_count"],
        "element_comparison_count": summary["element_comparison_count"],
        "maximum_absolute_difference": summary["maximum_absolute_difference"],
        "all_metrics_sign_exact": summary["all_metrics_sign_exact"],
        "summary_hash": summary["summary_hash"],
        "performance_claimed": False,
        "tir_admitted": False,
    }


def _readme() -> str:
    return (
        "# FSG4/B4-B1 Typed Pure-PyTorch Reference\n\n"
        "Root replay reads the hash-bound B4-B1a five-fresh raw, recompiles static "
        "typed IR and per-capture instances, executes only public PyTorch operators, "
        "and compares sparse reconstruction, forward outputs, and eligible local VJP "
        "gradients. This is correctness evidence pending external audit; no TIR, "
        "performance, memory, or ASPLOS-ready claim.\n"
    )


def _all_files(root: Path) -> dict[str, str]:
    return {name: _file_sha256(root / name) for name in ARTIFACT_FILES}


def _generate(args: argparse.Namespace) -> dict[str, object]:
    if _git("status", "--porcelain=v1", "--", *CODE_PATHS):
        raise ValueError("FSG4/B4-B1 reference code paths must be committed")
    output = args.artifact_dir.resolve()
    capture_artifact = args.capture_artifact.resolve()
    if output.exists():
        raise FileExistsError(f"FSG4/B4-B1 reference artifact exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.incomplete-", dir=output.parent
    ) as temporary:
        root = Path(temporary)
        protocol = _protocol(capture_artifact)
        _validate_protocol(protocol, capture_artifact)
        records = _records_from_source(capture_artifact, protocol)
        summary = _summary(records, protocol)
        _write_json(root / "protocol.json", protocol)
        _write_jsonl(root / "reference_records.jsonl", records)
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
            "source_capture_artifact": protocol["source_capture_artifact"],
            "protocol_hash": protocol["protocol_hash"],
            "summary_hash": summary["summary_hash"],
            "files": _all_files(root),
            "performance_claimed": False,
            "tir_admitted": False,
        }
        manifest["manifest_hash"] = _canonical_hash(manifest)
        _write_json(root / "manifest.json", manifest)
        shutil.move(root, output)
    _verify_static_artifact(output, capture_artifact)
    return result


def _verify_static_artifact(
    artifact: Path, capture_artifact: Path
) -> tuple[list[dict[str, Any]], dict[str, object], dict[str, object]]:
    artifact = artifact.resolve()
    capture_artifact = capture_artifact.resolve()
    manifest = _load_json(artifact / "manifest.json")
    semantic = dict(manifest)
    claimed = semantic.pop("manifest_hash", None)
    if (
        manifest.get("schema_version") != ARTIFACT_SCHEMA
        or claimed != _canonical_hash(semantic)
        or manifest.get("source_capture_artifact") != _source_identity(capture_artifact)
        or manifest.get("performance_claimed") is not False
        or manifest.get("tir_admitted") is not False
    ):
        raise ValueError("FSG4/B4-B1 reference manifest differs")
    _verify_code_provenance(manifest)
    files = manifest.get("files")
    if not isinstance(files, Mapping) or dict(files) != _all_files(artifact):
        raise ValueError("FSG4/B4-B1 reference artifact inventory differs")
    protocol = _load_json(artifact / "protocol.json")
    _validate_protocol(protocol, capture_artifact)
    if (
        manifest.get("source_git_head") != protocol.get("source_git_head")
        or manifest.get("code_revision") != protocol.get("code_revision")
        or manifest.get("protocol_hash") != protocol.get("protocol_hash")
    ):
        raise ValueError("FSG4/B4-B1 reference manifest protocol differs")
    records = _records_from_source(capture_artifact, protocol)
    if _load_jsonl(artifact / "reference_records.jsonl") != records:
        raise ValueError("FSG4/B4-B1 reference record replay differs")
    summary = _summary(records, protocol)
    if (
        _load_json(artifact / "summary.json") != summary
        or manifest.get("summary_hash") != summary["summary_hash"]
    ):
        raise ValueError("FSG4/B4-B1 reference semantic replay differs")
    result = _result(summary)
    if (artifact / "replay_stdout.txt").read_text(encoding="utf-8") != (
        _canonical_json(result) + "\n"
    ) or (artifact / "README.md").read_text(encoding="utf-8") != _readme():
        raise ValueError("FSG4/B4-B1 reference projection replay differs")
    return records, summary, result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--artifact-dir", type=Path, required=True)
    generate.add_argument("--capture-artifact", type=Path, required=True)
    replay = commands.add_parser("replay")
    replay.add_argument("--artifact-dir", type=Path, required=True)
    replay.add_argument("--capture-artifact", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "generate":
        result = _generate(args)
    else:
        _records_payload, _summary_payload, result = _verify_static_artifact(
            args.artifact_dir, args.capture_artifact
        )
    print(_canonical_json(result), flush=True)


if __name__ == "__main__":
    main()
