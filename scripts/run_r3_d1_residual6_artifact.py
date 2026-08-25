#!/usr/bin/env python3
"""Generate or replay the D1-A five-fresh residual6 correctness artifact."""

# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=too-many-boolean-expressions,missing-function-docstring
# pylint: disable=not-callable,duplicate-code

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping

import torch
import torch.nn.functional as torch_functional

from scripts.run_r3_d1_residual11_artifact import (
    _clean,
    _file_hash,
    _git,
    _hash,
    _json_load,
    _json_write,
    _load,
    _tensor_hash,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAPTURE = (
    ROOT / "artifacts/rvir-v4-pre-state/resnet2b-core-pre-state-v1/source_capture.pt"
)
DEFAULT_MODEL = Path(
    "/home/lee/Codes/vnncomp2021/benchmarks/cifar10_resnet/onnx/resnet_2b.onnx"
)
DEFAULT_OUTPUT = ROOT / "artifacts/r3-structured-owner/r3-d1a-residual6-staged-v1"
RESIDUAL11_ARTIFACT = ROOT / "artifacts/r3-structured-owner/r3-d1a-residual11-staged-v1"
WORKER = ROOT / "scripts/run_r3_d1_residual6_worker.py"
PROTOCOL_SCHEMA = "boundflow.r3-d1a-residual6-protocol/v1"
MANIFEST_SCHEMA = "boundflow.r3-d1a-residual6-manifest/v1"
CODE_PATHS = (
    "boundflow/backends/tvm/r3_d1_residual6_staged.py",
    "boundflow/runtime/r3_d1_residual6_staged.py",
    "scripts/run_r3_d1_residual6_worker.py",
    "scripts/run_r3_d1_residual6_artifact.py",
    "scripts/run_r3_d1_residual11_artifact.py",
)
ATOL = 2.0e-4


def _oracle(inputs: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    incoming = inputs["incoming"].double()[:6144].reshape(6, 16, 8, 8)
    staged_image = torch_functional.conv_transpose2d(
        incoming, inputs["weight4"].double(), stride=1, padding=1
    )
    staged = staged_image.reshape(6, 1024)
    lower = inputs["lower19"].double()
    upper = inputs["upper19"].double()
    alpha = inputs["alpha19"].double()
    lookup = inputs["alpha_map19"].long()
    compact = alpha[0, 0, :, lookup.clamp_min(0)]
    lower_alpha = torch.where(
        lookup.unsqueeze(0) >= 0,
        compact.clamp(0.0, 1.0),
        torch.zeros_like(compact),
    )
    ambiguous = (lower < 0.0) & (upper > 0.0)
    lower_slope = torch.where(
        ambiguous, lower_alpha, torch.where(lower >= 0.0, 1.0, 0.0)
    )
    upper_slope = torch.where(
        lower >= 0.0,
        1.0,
        torch.where(
            upper <= 0.0,
            0.0,
            upper / (upper - lower).clamp_min(torch.finfo(torch.float32).eps),
        ),
    )
    slope = torch.where(staged >= 0.0, lower_slope, upper_slope)
    intercept = torch.where((staged < 0.0) & ambiguous, -lower * upper_slope, 0.0)
    main_path = torch_functional.conv_transpose2d(
        (staged * slope).reshape(6, 16, 8, 8),
        inputs["weight2"].double(),
        stride=2,
        padding=1,
        output_padding=1,
    )
    shortcut = torch_functional.conv_transpose2d(
        incoming,
        inputs["weight5"].double(),
        stride=2,
        output_padding=1,
    )
    incoming_bias = (
        incoming * (inputs["bias4"] + inputs["bias5"]).double().reshape(1, 16, 1, 1)
    ).sum(dim=(1, 2, 3))
    staged_bias = (
        staged * intercept
        + staged
        * slope
        * inputs["bias2"].double().repeat_interleave(64).reshape(1, 1024)
    ).sum(dim=1)
    return (
        (main_path + shortcut).reshape(12_288),
        inputs["bias_in"].double() + incoming_bias + staged_bias,
    )


def _validate_raw(raw: Mapping[str, Any]) -> dict[str, float]:
    expected = {
        "schema_version",
        "run_index",
        "source_capture_sha256",
        "model_sha256",
        "plan_hash",
        "trace_hash",
        "inputs",
        "reference_output",
        "reference_bias",
        "candidate_output",
        "candidate_bias",
        "tensor_hashes",
        "receipt",
        "timing_recorded",
        "performance_claimed",
    }
    if (
        set(raw) != expected
        or raw["schema_version"] != "boundflow.r3-d1-residual6-worker/v1"
        or raw["timing_recorded"] is not False
        or raw["performance_claimed"] is not False
    ):
        raise ValueError("R3-D1 residual6 worker envelope differs")
    inputs = raw["inputs"]
    expected_inputs = {
        "incoming",
        "weight4",
        "lower19",
        "upper19",
        "alpha19",
        "alpha_map19",
        "weight2",
        "weight5",
        "bias4",
        "bias2",
        "bias5",
        "bias_in",
    }
    if (
        not isinstance(inputs, Mapping)
        or set(inputs) != expected_inputs
        or any(not torch.is_tensor(value) for value in inputs.values())
    ):
        raise ValueError("R3-D1 residual6 input payload differs")
    names = ("reference_output", "reference_bias", "candidate_output", "candidate_bias")
    hashes = raw["tensor_hashes"]
    if not isinstance(hashes, Mapping) or set(hashes) != set(names):
        raise ValueError("R3-D1 residual6 tensor hash inventory differs")
    for name in names:
        value = raw[name]
        if not torch.is_tensor(value) or _tensor_hash(value) != hashes[name]:
            raise ValueError("R3-D1 residual6 tensor hash differs")
    receipt = raw["receipt"]
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("exported_symbols")
        != (
            "boundflow_r3d1_residual6_stage1",
            "boundflow_r3d1_residual6_stage2",
        )
        or receipt.get("launch_count") != 2
        or receipt.get("scratch_count") != 1
        or receipt.get("persistent_dense_a") is not False
        or receipt.get("fallback_count") != 0
        or receipt.get("timing_recorded") is not False
        or receipt.get("performance_claimed") is not False
    ):
        raise ValueError("R3-D1 residual6 receipt differs")
    oracle_output, oracle_bias = _oracle(inputs)
    reference_output = raw["reference_output"].double()
    reference_bias = raw["reference_bias"].double()
    candidate_output = raw["candidate_output"].double()
    candidate_bias = raw["candidate_bias"].double()
    metrics = {
        "candidate_reference_output_diff": float(
            (candidate_output - reference_output).abs().max()
        ),
        "candidate_reference_bias_diff": float(
            (candidate_bias - reference_bias).abs().max()
        ),
        "candidate_oracle_output_diff": float(
            (candidate_output - oracle_output).abs().max()
        ),
        "candidate_oracle_bias_diff": float((candidate_bias - oracle_bias).abs().max()),
        "reference_oracle_output_diff": float(
            (reference_output - oracle_output).abs().max()
        ),
        "reference_oracle_bias_diff": float((reference_bias - oracle_bias).abs().max()),
    }
    if any(value > ATOL for value in metrics.values()) or not torch.equal(
        torch.sign(candidate_output), torch.sign(oracle_output)
    ):
        raise ValueError("R3-D1 residual6 semantics differ")
    return metrics


def _summarize(raws: list[dict[str, Any]]) -> dict[str, object]:
    if len(raws) != 5 or [raw["run_index"] for raw in raws] != list(range(5)):
        raise ValueError("R3-D1 residual6 worker inventory differs")
    metrics = [_validate_raw(raw) for raw in raws]
    for name in ("unscheduled_tir_hash", "scheduled_tir_hash", "device_source_hash"):
        if len({str(raw["receipt"][name]) for raw in raws}) != 1:
            raise ValueError("R3-D1 residual6 compiler receipt drifted")
    summary: dict[str, object] = {
        "run_count": 5,
        "element_count": 5 * (12_288 + 6) * 2,
        "metrics": metrics,
        "maximum_diff": max(value for row in metrics for value in row.values()),
        "sign_exact": True,
        "launch_count_per_run": 2,
        "scratch_count": 1,
        "persistent_dense_a": False,
        "d1a_residual6_correctness": True,
        "d1b_schedule_qualification_open": True,
        "d1c_wrapper_open": False,
        "timing_recorded": False,
        "performance_claimed": False,
    }
    summary["summary_hash"] = _hash(summary)
    return summary


def _protocol() -> dict[str, object]:
    protocol: dict[str, object] = {
        "schema_version": PROTOCOL_SCHEMA,
        "source_git_head": _git("rev-parse", "HEAD"),
        "code_revision": {path: _file_hash(ROOT / path) for path in CODE_PATHS},
        "run_count": 5,
        "atol": ATOL,
        "sign_exact": True,
        "timing_forbidden": True,
        "residual11_manifest_sha256": _file_hash(RESIDUAL11_ARTIFACT / "manifest.json"),
        "source_capture_sha256": _file_hash(DEFAULT_CAPTURE),
        "model_sha256": _file_hash(DEFAULT_MODEL),
        "performance_claimed": False,
    }
    protocol["protocol_hash"] = _hash(protocol)
    return protocol


def generate(output: Path, capture: Path, model: Path) -> None:
    if output.exists():
        raise FileExistsError(f"R3-D1 residual6 artifact already exists: {output}")
    if not _clean():
        raise RuntimeError("R3-D1 residual6 formal requires a clean worktree")
    protocol = _protocol()
    with tempfile.TemporaryDirectory(prefix="boundflow-r3d1a-residual6-") as temporary:
        root = Path(temporary)
        raw_root = root / "raw"
        raw_root.mkdir()
        raws = []
        for run_index in range(5):
            result = raw_root / f"run-{run_index:02d}.pt"
            subprocess.run(
                (
                    sys.executable,
                    str(WORKER),
                    "--source-capture",
                    str(capture),
                    "--model",
                    str(model),
                    "--run-index",
                    str(run_index),
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
            "timing_recorded": False,
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
        or manifest.get("timing_recorded") is not False
    ):
        raise ValueError("R3-D1 residual6 artifact boundary differs")
    protocol_copy = dict(protocol)
    protocol_hash = protocol_copy.pop("protocol_hash", None)
    if (
        _hash(protocol_copy) != protocol_hash
        or manifest.get("protocol_hash") != protocol_hash
        or manifest.get("summary_hash") != summary.get("summary_hash")
    ):
        raise ValueError("R3-D1 residual6 protocol hash differs")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or any(
        _file_hash(output / str(path)) != digest for path, digest in files.items()
    ):
        raise ValueError("R3-D1 residual6 manifest file digest differs")
    manifest_copy = dict(manifest)
    manifest_hash = manifest_copy.pop("manifest_hash", None)
    if _hash(manifest_copy) != manifest_hash:
        raise ValueError("R3-D1 residual6 manifest hash differs")
    raws = [_load(path) for path in sorted((output / "raw").glob("*.pt"))]
    rebuilt = _summarize(raws)
    if rebuilt != summary:
        raise ValueError("R3-D1 residual6 summary replay differs")
    print(
        f"R3-D1-A residual6 replay PASS: max_diff={summary['maximum_diff']} "
        "timing_recorded=false performance_claimed=false",
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
