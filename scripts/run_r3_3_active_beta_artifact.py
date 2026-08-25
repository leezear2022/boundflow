#!/usr/bin/env python3
"""Generate or replay the formal R3-3 S-anchor active-beta artifact."""

# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=too-many-boolean-expressions,missing-function-docstring
# pylint: disable=duplicate-code,protected-access

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, cast, Mapping

import torch

from boundflow.ir.differentiable_lower_sparse_linear_tir import (
    SPARSE_LINEAR_OUTPUT_NAMES,
    DifferentiableLowerSparseLinearGradientProjectionReceiptV1,
    DifferentiableLowerSparseLinearTIRInstanceV1,
    DifferentiableLowerSparseLinearTIRLaunchReceiptV1,
    DifferentiableLowerSparseLinearTIRModuleReceiptV1,
    DifferentiableLowerSparseLinearTIRScheduleV1,
    DifferentiableLowerSparseLinearTIRTemplateV1,
)
from boundflow.runtime.rvir_v4_production_state import production_tensor_sha256
from boundflow.runtime.fsg4_b4b1_pytorch_reference import (
    build_b4b1_differentiable_lower_instance_v1,
    build_b4b1_differentiable_lower_ir_v1,
    run_b4b1_pytorch_reference_v1,
)
from boundflow.runtime.fsg4_b4b1_reference_capture import (
    production_differentiable_reference_capture_from_payload_v1,
)
from boundflow.runtime import fsg4_b4b2_sparse_linear_tir as sparse_linear
from scripts.run_fsg4_b4b1_pytorch_reference_artifact import (
    _reference_execution_policy,
)

ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / "artifacts/fsg4-b4b1-reference-five-fresh/resnet2b-prop0-v1"
OUTPUT = ROOT / "artifacts/r3-structured-owner/r3-3-active-beta-correctness-v1"
WORKER = ROOT / "scripts/run_r3_3_active_beta_worker.py"
CACHE_PROBE = ROOT / "scripts/run_fsg4_b4b2_sparse_linear_tir_correctness.py"
PLAN = ROOT / "gemini_doc/BOUNDFLOW_R3_3_S_ANCHOR_ACTIVE_BETA_PLAN_2026_08_26.md"
RUN_COUNT = 5
TOLERANCE = 2.0e-4
CODE_PATHS = (
    "boundflow/backends/tvm/differentiable_lower_sparse_linear.py",
    "boundflow/ir/differentiable_lower_sparse_linear_tir.py",
    "boundflow/runtime/fsg4_b4b1_pytorch_reference.py",
    "boundflow/runtime/fsg4_b4b2_sparse_linear_tir.py",
    "scripts/run_fsg4_b4b2_sparse_linear_tir_correctness.py",
    "scripts/run_fsg4_b4b1_pytorch_reference_artifact.py",
    "scripts/run_r3_3_active_beta_worker.py",
    "scripts/run_r3_3_active_beta_artifact.py",
    "scripts/probe_r3_3_active_beta_tamper.py",
    "gemini_doc/BOUNDFLOW_R3_3_S_ANCHOR_ACTIVE_BETA_PLAN_2026_08_26.md",
)
WORKER_KEYS = {
    "schema_version",
    "run_ordinal",
    "capture_sha256",
    "template_hash",
    "schedule_hash",
    "module_receipt_hash",
    "template",
    "schedule",
    "instance",
    "metrics",
    "outputs",
    "references",
    "native_gradients",
    "output_hashes",
    "reference_hashes",
    "native_gradient_hashes",
    "projection_receipt",
    "module_receipt",
    "launch_receipt",
    "alpha_feature_indices",
    "beta_locations",
    "beta_signs",
    "empty_beta_specialization_rejected",
    "unowned_native_zero_exact",
    "beta_nonzero_count",
    "timing_recorded",
    "performance_claimed",
}


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
        raise RuntimeError(f"R3-3 formal source is dirty: {dirty}")


def _load(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise TypeError("R3-3 raw root differs")
    return value


def _tensor_map(raw: Mapping[str, Any], name: str) -> dict[str, torch.Tensor]:
    value = raw.get(name)
    if not isinstance(value, dict) or set(value) != set(SPARSE_LINEAR_OUTPUT_NAMES):
        raise TypeError(f"R3-3 {name} inventory differs")
    result: dict[str, torch.Tensor] = {}
    for tensor_name, tensor in value.items():
        if not torch.is_tensor(tensor) or not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"R3-3 {name} tensor differs: {tensor_name}")
        result[tensor_name] = tensor.contiguous()
    return result


def _hash_map(
    raw: Mapping[str, Any], name: str, tensors: Mapping[str, torch.Tensor]
) -> dict[str, str]:
    value = raw.get(name)
    if not isinstance(value, dict) or set(value) != set(tensors):
        raise TypeError(f"R3-3 {name} inventory differs")
    rebuilt = {key: production_tensor_sha256(tensor) for key, tensor in tensors.items()}
    if value != rebuilt:
        raise ValueError(f"R3-3 {name} differs")
    return rebuilt


def _max_diff(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    return float((candidate - reference).abs().max().item())


def _validate_metrics(
    raw: Mapping[str, Any],
    outputs: Mapping[str, torch.Tensor],
    references: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    metrics = raw.get("metrics")
    if not isinstance(metrics, list) or len(metrics) != len(SPARSE_LINEAR_OUTPUT_NAMES):
        raise TypeError("R3-3 metric inventory differs")
    maxima: dict[str, float] = {}
    for name, metric in zip(SPARSE_LINEAR_OUTPUT_NAMES, metrics):
        if not isinstance(metric, dict) or set(metric) != {
            "name",
            "element_count",
            "maximum_absolute_difference",
            "allclose",
            "sign_exact",
            "reference_hash",
            "candidate_hash",
        }:
            raise TypeError("R3-3 metric envelope differs")
        candidate, reference = outputs[name], references[name]
        maximum = _max_diff(candidate, reference)
        expected = {
            "name": name,
            "element_count": candidate.numel(),
            "maximum_absolute_difference": maximum,
            "allclose": bool(
                torch.allclose(candidate, reference, atol=TOLERANCE, rtol=TOLERANCE)
            ),
            "sign_exact": bool(
                torch.equal(torch.sign(candidate), torch.sign(reference))
            ),
            "reference_hash": production_tensor_sha256(reference),
            "candidate_hash": production_tensor_sha256(candidate),
        }
        if (
            metric != expected
            or expected["allclose"] is not True
            or expected["sign_exact"] is not True
        ):
            raise ValueError(f"R3-3 semantic metric differs: {name}")
        maxima[name] = maximum
    return maxima


def _validate_worker(
    raw: Mapping[str, Any], capture_hashes: Mapping[str, str]
) -> dict[str, float]:
    ordinal = raw.get("run_ordinal")
    if (
        set(raw) != WORKER_KEYS
        or raw.get("schema_version") != "boundflow.r3-3-active-beta-worker/v1"
        or ordinal not in range(RUN_COUNT)
        or raw.get("capture_sha256") != capture_hashes[f"run_{ordinal:02d}.pt"]
        or raw.get("timing_recorded") is not False
        or raw.get("performance_claimed") is not False
        or raw.get("empty_beta_specialization_rejected") is not True
        or raw.get("unowned_native_zero_exact") is not True
    ):
        raise ValueError("R3-3 worker envelope differs")
    template_raw, schedule_raw, instance_raw = (
        raw.get("template"),
        raw.get("schedule"),
        raw.get("instance"),
    )
    if not all(
        isinstance(value, dict) for value in (template_raw, schedule_raw, instance_raw)
    ):
        raise TypeError("R3-3 IR payload differs")
    template = DifferentiableLowerSparseLinearTIRTemplateV1.from_dict(
        cast(Mapping[str, object], template_raw)
    )
    schedule = DifferentiableLowerSparseLinearTIRScheduleV1.from_dict(
        cast(Mapping[str, object], schedule_raw), template
    )
    instance = DifferentiableLowerSparseLinearTIRInstanceV1.from_dict(
        cast(Mapping[str, object], instance_raw), template
    )
    if (
        raw.get("template_hash") != template.stable_hash()
        or raw.get("schedule_hash") != schedule.stable_hash(template)
        or raw.get("alpha_feature_indices") != list(template.alpha_feature_indices)
        or raw.get("beta_locations") != list(template.beta_locations)
        or raw.get("beta_signs") != list(template.beta_signs)
        or len(template.alpha_feature_indices) != 27
        or tuple(sorted(set(template.alpha_feature_indices)))
        != template.alpha_feature_indices
        or len(template.beta_locations) != 6
        or len(template.beta_signs) != 6
        or any(sign not in {-1, 1} for sign in template.beta_signs)
    ):
        raise ValueError("R3-3 template identity differs")
    outputs = _tensor_map(raw, "outputs")
    references = _tensor_map(raw, "references")
    output_hashes = _hash_map(raw, "output_hashes", outputs)
    reference_hashes = _hash_map(raw, "reference_hashes", references)
    maxima = _validate_metrics(raw, outputs, references)
    native = raw.get("native_gradients")
    if not isinstance(native, dict) or set(native) != {
        "native_alpha_gradient",
        "native_beta_gradient",
    }:
        raise TypeError("R3-3 native gradient inventory differs")
    native_alpha, native_beta = native.values()
    if (
        not torch.is_tensor(native_alpha)
        or not torch.is_tensor(native_beta)
        or not bool(torch.isfinite(native_alpha).all().item())
        or not bool(torch.isfinite(native_beta).all().item())
    ):
        raise ValueError("R3-3 native gradient differs")
    native_tensors = {key: value.contiguous() for key, value in native.items()}
    native_hashes = _hash_map(raw, "native_gradient_hashes", native_tensors)
    capture_payload = torch.load(
        CAPTURE / f"run_{ordinal:02d}.pt",
        map_location="cpu",
        weights_only=False,
    )
    capture = production_differentiable_reference_capture_from_payload_v1(
        capture_payload["captures"][0]
    )
    lower_ir = build_b4b1_differentiable_lower_ir_v1(capture)
    lower_instance = build_b4b1_differentiable_lower_instance_v1(capture, lower_ir)
    with _reference_execution_policy():
        oracle = run_b4b1_pytorch_reference_v1(capture, lower_ir, lower_instance)
    if oracle.native_beta_gradient is None:
        raise RuntimeError("R3-3 replay native beta gradient is absent")
    oracle_alpha, oracle_beta, oracle_unowned_zero = (
        sparse_linear._reference_compressed_gradients(oracle, template)
    )
    oracle_tensors = {
        "output_lower_a": oracle.output_lower_a,
        "output_bias": oracle.output_bias,
        "compressed_alpha_gradient": oracle_alpha,
        "compressed_beta_gradient": oracle_beta,
    }
    oracle_native = {
        "native_alpha_gradient": oracle.native_alpha_gradient,
        "native_beta_gradient": oracle.native_beta_gradient,
    }
    if (
        oracle_unowned_zero is not True
        or any(
            production_tensor_sha256(references[name])
            != production_tensor_sha256(value)
            for name, value in oracle_tensors.items()
        )
        or any(
            production_tensor_sha256(native_tensors[name])
            != production_tensor_sha256(value)
            for name, value in oracle_native.items()
        )
    ):
        raise ValueError("R3-3 independent oracle replay differs")
    alpha_indices = torch.tensor(template.alpha_feature_indices, dtype=torch.int64)
    if alpha_indices.max().item() >= native_alpha.shape[1]:
        raise ValueError("R3-3 alpha location differs")
    alpha_mask = torch.ones_like(native_alpha, dtype=torch.bool)
    alpha_mask[:, alpha_indices] = False
    beta_mask = torch.ones_like(native_beta, dtype=torch.bool)
    for domain, location in enumerate(template.beta_locations):
        if location < 0 or location >= native_beta.shape[1]:
            raise ValueError("R3-3 beta location differs")
        beta_mask[domain, location] = False
    if (
        torch.count_nonzero(native_alpha[alpha_mask]).item() != 0
        or torch.count_nonzero(native_beta[beta_mask]).item() != 0
        or tuple(outputs["compressed_alpha_gradient"].shape) != (6, 27)
        or tuple(outputs["compressed_beta_gradient"].shape) != (6, 1)
        or raw.get("beta_nonzero_count") != 6
        or torch.count_nonzero(outputs["compressed_beta_gradient"]).item() != 6
    ):
        raise ValueError("R3-3 ownership projection differs")
    module_raw = raw.get("module_receipt")
    projection_raw = raw.get("projection_receipt")
    launch_raw = raw.get("launch_receipt")
    if not all(
        isinstance(value, dict) for value in (module_raw, projection_raw, launch_raw)
    ):
        raise TypeError("R3-3 receipt payload differs")
    module = DifferentiableLowerSparseLinearTIRModuleReceiptV1.from_dict(
        cast(Mapping[str, object], module_raw), template, schedule
    )
    projection = DifferentiableLowerSparseLinearGradientProjectionReceiptV1.from_dict(
        cast(Mapping[str, object], projection_raw), template, instance
    )
    launch = DifferentiableLowerSparseLinearTIRLaunchReceiptV1.from_dict(
        cast(Mapping[str, object], launch_raw),
        template,
        instance,
        schedule,
        module,
        projection,
    )
    projected_alpha = torch.zeros_like(native_alpha)
    projected_alpha[:, alpha_indices] = outputs["compressed_alpha_gradient"]
    projected_beta = torch.zeros_like(native_beta)
    for domain, location in enumerate(template.beta_locations):
        projected_beta[domain, location] = outputs["compressed_beta_gradient"][
            domain, 0
        ]
    if (
        raw.get("module_receipt_hash") != module.stable_hash(template, schedule)
        or projection.reference_native_alpha_gradient_hash
        != native_hashes["native_alpha_gradient"]
        or projection.reference_native_beta_gradient_hash
        != native_hashes["native_beta_gradient"]
        or projection.reference_compressed_alpha_gradient_hash
        != reference_hashes["compressed_alpha_gradient"]
        or projection.reference_compressed_beta_gradient_hash
        != reference_hashes["compressed_beta_gradient"]
        or projection.candidate_compressed_alpha_gradient_hash
        != output_hashes["compressed_alpha_gradient"]
        or projection.candidate_compressed_beta_gradient_hash
        != output_hashes["compressed_beta_gradient"]
        or projection.projected_native_alpha_gradient_hash
        != production_tensor_sha256(projected_alpha)
        or projection.projected_native_beta_gradient_hash
        != production_tensor_sha256(projected_beta)
        or module.forbidden_workspace_count != 0
        or launch.cache_event != "miss"
        or launch.forward_launch_count != 1
        or launch.backward_launch_count != 1
        or launch.dlpack_pointer_count != 21
        or launch.dlpack_pointer_exact_count != 21
        or launch.fallback_count != 0
        or launch.eager_backward_count != 0
        or launch.performance_claimed is not False
    ):
        raise ValueError("R3-3 receipt binding differs")
    return maxima


def _validate_cache_probe(raw: Mapping[str, Any]) -> None:
    rows = raw.get("rows")
    if (
        raw.get("status") != "validated-b2-2-sparse-source-linear-correctness"
        or raw.get("run_count") != RUN_COUNT
        or raw.get("metric_count") != RUN_COUNT * 4
        or raw.get("allclose") is not True
        or raw.get("sign_exact") is not True
        or raw.get("forbidden_workspace_count") != 0
        or raw.get("sparse_source_admitted") is not True
        or raw.get("performance_claimed") is not False
        or not isinstance(rows, list)
        or [row.get("cache_event") for row in rows]
        != ["miss", "hit", "hit", "hit", "hit"]
    ):
        raise ValueError("R3-3 cache-sequence probe differs")


def _summary(
    raws: list[dict[str, Any]],
    cache_probe: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    if len(raws) != RUN_COUNT or sorted(raw["run_ordinal"] for raw in raws) != list(
        range(RUN_COUNT)
    ):
        raise ValueError("R3-3 raw inventory differs")
    capture_hashes = protocol["capture_sha256"]
    if not isinstance(capture_hashes, dict):
        raise TypeError("R3-3 capture protocol differs")
    maxima = [_validate_worker(raw, capture_hashes) for raw in raws]
    _validate_cache_probe(cache_probe)
    template_hashes = {raw["template_hash"] for raw in raws}
    schedule_hashes = {raw["schedule_hash"] for raw in raws}
    module_hashes = {raw["module_receipt_hash"] for raw in raws}
    if (
        len(template_hashes) != 1
        or len(schedule_hashes) != 1
        or len(module_hashes) != 1
    ):
        raise ValueError("R3-3 receipt stability differs")
    result: dict[str, Any] = {
        "schema_version": "boundflow.r3-3-active-beta-summary/v1",
        "worker_count": RUN_COUNT,
        "metric_count": RUN_COUNT * 4,
        "maximum_absolute_difference": max(max(row.values()) for row in maxima),
        "maximum_by_output": {
            name: max(row[name] for row in maxima)
            for name in SPARSE_LINEAR_OUTPUT_NAMES
        },
        "beta_nonzero_count": sum(int(raw["beta_nonzero_count"]) for raw in raws),
        "fresh_worker_cache_events": [
            raw["launch_receipt"]["cache_event"] for raw in raws
        ],
        "cache_sequence_events": [row["cache_event"] for row in cache_probe["rows"]],
        "template_hash": next(iter(template_hashes)),
        "schedule_hash": next(iter(schedule_hashes)),
        "module_receipt_hash": next(iter(module_hashes)),
        "active_beta_correctness_admitted": True,
        "ownership_admitted": True,
        "isolated_timing_open": True,
        "r3_4_open": False,
        "same_solver_open": False,
        "timing_recorded": False,
        "performance_claimed": False,
    }
    result["summary_hash"] = _hash(result)
    return result


def _protocol(revision: str) -> dict[str, Any]:
    captures = {
        f"run_{ordinal:02d}.pt": _file_hash(CAPTURE / f"run_{ordinal:02d}.pt")
        for ordinal in range(RUN_COUNT)
    }
    result: dict[str, Any] = {
        "schema_version": "boundflow.r3-3-active-beta-protocol/v1",
        "source_revision": revision,
        "run_count": RUN_COUNT,
        "anchor_ordinal": 0,
        "empty_beta_control_anchor_ordinal": 1,
        "output_names": list(SPARSE_LINEAR_OUTPUT_NAMES),
        "tolerance": {"atol": TOLERANCE, "rtol": TOLERANCE, "sign_exact": True},
        "active_beta_shape": [6, 1],
        "active_beta_location_count": 6,
        "alpha_feature_count_per_domain": 27,
        "fresh_process_cache_policy": "cold-miss-each",
        "cache_sequence_events": ["miss", "hit", "hit", "hit", "hit"],
        "capture_sha256": captures,
        "plan_sha256": _file_hash(PLAN),
        "timing_recorded": False,
        "performance_claimed": False,
        "code_revision": {name: _file_hash(ROOT / name) for name in CODE_PATHS},
    }
    result["protocol_hash"] = _hash(result)
    return result


def _write_json(path: Path, value: object) -> None:
    path.write_text(_canonical(value) + "\n", encoding="utf-8")


def generate(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"R3-3 artifact exists: {output}")
    _clean()
    protocol = _protocol(_git("rev-parse", "HEAD"))
    temporary = Path(tempfile.mkdtemp(prefix="r3-3-formal-", dir=output.parent))
    try:
        raw_dir = temporary / "raw"
        log_dir = temporary / "logs"
        raw_dir.mkdir(parents=True)
        log_dir.mkdir()
        raws = []
        for ordinal in range(RUN_COUNT):
            target = raw_dir / f"run_{ordinal:02d}.pt"
            completed = subprocess.run(
                (
                    sys.executable,
                    str(WORKER),
                    "--run-ordinal",
                    str(ordinal),
                    "--result",
                    str(target),
                ),
                cwd=ROOT,
                check=True,
                text=True,
                capture_output=True,
                env=os.environ.copy(),
            )
            (log_dir / f"worker_{ordinal:02d}.stdout.txt").write_text(
                completed.stdout, encoding="utf-8"
            )
            (log_dir / f"worker_{ordinal:02d}.stderr.txt").write_text(
                completed.stderr, encoding="utf-8"
            )
            raws.append(_load(target))
        cache_completed = subprocess.run(
            (sys.executable, str(CACHE_PROBE)),
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
            env=os.environ.copy(),
        )
        (log_dir / "cache_probe.stdout.txt").write_text(
            cache_completed.stdout, encoding="utf-8"
        )
        (log_dir / "cache_probe.stderr.txt").write_text(
            cache_completed.stderr, encoding="utf-8"
        )
        json_lines = [
            line for line in cache_completed.stdout.splitlines() if line.startswith("{")
        ]
        if len(json_lines) != 1:
            raise ValueError("R3-3 cache probe stdout differs")
        cache_probe = json.loads(json_lines[0])
        _write_json(temporary / "cache_probe.json", cache_probe)
        summary = _summary(raws, cache_probe, protocol)
        _write_json(temporary / "protocol.json", protocol)
        _write_json(temporary / "summary.json", summary)
        files = {
            str(path.relative_to(temporary)): _file_hash(path)
            for path in sorted(temporary.rglob("*"))
            if path.is_file()
        }
        manifest: dict[str, Any] = {
            "schema_version": "boundflow.r3-3-active-beta-manifest/v1",
            "source_revision": protocol["source_revision"],
            "protocol_hash": protocol["protocol_hash"],
            "summary_hash": summary["summary_hash"],
            "files": files,
        }
        manifest["manifest_hash"] = _hash(manifest)
        _write_json(temporary / "manifest.json", manifest)
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
    if manifest.get(
        "schema_version"
    ) != "boundflow.r3-3-active-beta-manifest/v1" or manifest_hash != _hash(
        unsigned_manifest
    ):
        raise ValueError("R3-3 manifest differs")
    files = manifest.get("files")
    if not isinstance(files, dict) or any(
        _file_hash(artifact / name) != digest for name, digest in files.items()
    ):
        raise ValueError("R3-3 file digest differs")
    protocol = json.loads((artifact / "protocol.json").read_text(encoding="utf-8"))
    unsigned_protocol = dict(protocol)
    protocol_hash = unsigned_protocol.pop("protocol_hash", None)
    if (
        protocol_hash != _hash(unsigned_protocol)
        or protocol_hash != manifest["protocol_hash"]
    ):
        raise ValueError("R3-3 protocol hash differs")
    frozen = {
        "run_count": RUN_COUNT,
        "anchor_ordinal": 0,
        "empty_beta_control_anchor_ordinal": 1,
        "output_names": list(SPARSE_LINEAR_OUTPUT_NAMES),
        "tolerance": {"atol": TOLERANCE, "rtol": TOLERANCE, "sign_exact": True},
        "active_beta_shape": [6, 1],
        "active_beta_location_count": 6,
        "alpha_feature_count_per_domain": 27,
        "fresh_process_cache_policy": "cold-miss-each",
        "cache_sequence_events": ["miss", "hit", "hit", "hit", "hit"],
        "timing_recorded": False,
        "performance_claimed": False,
    }
    if any(protocol.get(name) != value for name, value in frozen.items()):
        raise ValueError("R3-3 frozen protocol differs")
    revision = protocol.get("source_revision")
    code_revision = protocol.get("code_revision")
    if (
        not isinstance(revision, str)
        or not isinstance(code_revision, dict)
        or set(code_revision) != set(CODE_PATHS)
        or any(
            _git_blob_hash(revision, name) != digest
            for name, digest in code_revision.items()
        )
        or protocol.get("plan_sha256") != code_revision[str(PLAN.relative_to(ROOT))]
    ):
        raise ValueError("R3-3 source binding differs")
    cache_probe = json.loads(
        (artifact / "cache_probe.json").read_text(encoding="utf-8")
    )
    raws = [_load(path) for path in sorted((artifact / "raw").glob("*.pt"))]
    summary = _summary(raws, cache_probe, protocol)
    if (
        summary != json.loads((artifact / "summary.json").read_text(encoding="utf-8"))
        or summary["summary_hash"] != manifest["summary_hash"]
    ):
        raise ValueError("R3-3 semantic replay differs")
    if not math.isfinite(float(summary["maximum_absolute_difference"])):
        raise ValueError("R3-3 summary numeric differs")
    print(
        f"R3-3 replay PASS: workers={summary['worker_count']} "
        f"max_diff={summary['maximum_absolute_difference']} "
        f"timing_open={summary['isolated_timing_open']}",
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
