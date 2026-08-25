#!/usr/bin/env python3
"""Generate or replay the D1-A five-fresh residual11 correctness artifact."""

# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=too-many-boolean-expressions,missing-function-docstring
# pylint: disable=import-outside-toplevel,not-callable
# pylint: disable=duplicate-code

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping

import torch
import torch.nn.functional as torch_functional

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAPTURE = (
    ROOT / "artifacts/rvir-v4-pre-state/resnet2b-core-pre-state-v1/source_capture.pt"
)
DEFAULT_MODEL = Path(
    "/home/lee/Codes/vnncomp2021/benchmarks/cifar10_resnet/onnx/resnet_2b.onnx"
)
DEFAULT_OUTPUT = ROOT / "artifacts/r3-structured-owner/r3-d1a-residual11-staged-v1"
D0_ARTIFACT = ROOT / "artifacts/r3-structured-owner/r3-d0-microphysics-v1"
WORKER = ROOT / "scripts/run_r3_d1_residual11_worker.py"
PROTOCOL_SCHEMA = "boundflow.r3-d1a-residual11-protocol/v1"
MANIFEST_SCHEMA = "boundflow.r3-d1a-residual11-manifest/v1"
CODE_PATHS = (
    "boundflow/backends/tvm/r3_d1_residual11_staged.py",
    "boundflow/runtime/r3_d1_residual11_staged.py",
    "scripts/run_r3_d1_residual11_worker.py",
    "scripts/run_r3_d1_residual11_artifact.py",
)
ATOL = 2.0e-4


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


def _clean() -> bool:
    rows = _git("status", "--porcelain", "--untracked-files=no").splitlines()
    return all(
        row.endswith(" .docops/ev.jsonl") and row[:2].strip() == "M" for row in rows
    )


def _load(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise TypeError("R3-D1 raw root differs")
    return value


def _json_load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("R3-D1 JSON root differs")
    return value


def _json_write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _tensor_hash(value: torch.Tensor) -> str:
    from boundflow.runtime.rvir_v4_production_state import production_tensor_sha256

    return production_tensor_sha256(value)


def _oracle(inputs: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    incoming = inputs["incoming"].double()[:6144].reshape(6, 16, 8, 8)
    weight10 = inputs["weight10"].double()
    staged_image = torch_functional.conv_transpose2d(
        incoming, weight10, stride=1, padding=1
    )
    staged = staged_image.reshape(6, 1024)
    lower = inputs["lower25"].double()
    upper = inputs["upper25"].double()
    alpha = inputs["alpha25"].double()
    lookup = inputs["alpha_map25"].long()
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
    output = incoming + torch_functional.conv_transpose2d(
        (staged * slope).reshape(6, 16, 8, 8),
        inputs["weight8"].double(),
        stride=1,
        padding=1,
    )
    incoming_bias = (incoming * inputs["bias10"].double().reshape(1, 16, 1, 1)).sum(
        dim=(1, 2, 3)
    )
    staged_bias = (
        staged * intercept
        + staged
        * slope
        * inputs["bias8"].double().repeat_interleave(64).reshape(1, 1024)
    ).sum(dim=1)
    bias = inputs["bias_in"].double() + incoming_bias + staged_bias
    return output.reshape(6144), bias


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
        or raw["schema_version"] != "boundflow.r3-d1-residual11-worker/v1"
        or raw["timing_recorded"] is not False
        or raw["performance_claimed"] is not False
    ):
        raise ValueError("R3-D1 worker envelope differs")
    inputs = raw["inputs"]
    if not isinstance(inputs, Mapping) or set(inputs) != {
        "incoming",
        "weight10",
        "lower25",
        "upper25",
        "alpha25",
        "alpha_map25",
        "weight8",
        "bias10",
        "bias8",
        "bias_in",
    }:
        raise ValueError("R3-D1 input payload differs")
    if any(not torch.is_tensor(value) for value in inputs.values()):
        raise TypeError("R3-D1 input tensor differs")
    names = ("reference_output", "reference_bias", "candidate_output", "candidate_bias")
    hashes = raw["tensor_hashes"]
    if not isinstance(hashes, Mapping) or set(hashes) != set(names):
        raise ValueError("R3-D1 tensor hash inventory differs")
    for name in names:
        value = raw[name]
        if not torch.is_tensor(value) or _tensor_hash(value) != hashes[name]:
            raise ValueError("R3-D1 output tensor hash differs")
    receipt = raw["receipt"]
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("exported_symbols")
        != (
            "boundflow_r3d1_residual11_stage1",
            "boundflow_r3d1_residual11_stage2",
        )
        or receipt.get("launch_count") != 2
        or receipt.get("scratch_count") != 1
        or receipt.get("persistent_dense_a") is not False
        or receipt.get("fallback_count") != 0
        or receipt.get("timing_recorded") is not False
        or receipt.get("performance_claimed") is not False
    ):
        raise ValueError("R3-D1 receipt differs")
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
        raise ValueError("R3-D1 residual11 semantics differ")
    return metrics


def _summarize(raws: list[dict[str, Any]]) -> dict[str, object]:
    if len(raws) != 5 or [raw["run_index"] for raw in raws] != list(range(5)):
        raise ValueError("R3-D1 worker inventory differs")
    metrics = [_validate_raw(raw) for raw in raws]
    by_name = {
        name: {str(raw["receipt"][name]) for raw in raws}
        for name in ("unscheduled_tir_hash", "scheduled_tir_hash", "device_source_hash")
    }
    if any(len(values) != 1 for values in by_name.values()):
        raise ValueError("R3-D1 compiler receipt drifted")
    summary: dict[str, object] = {
        "run_count": 5,
        "element_count": 5 * (6144 + 6) * 2,
        "metrics": metrics,
        "maximum_diff": max(value for row in metrics for value in row.values()),
        "sign_exact": True,
        "launch_count_per_run": 2,
        "scratch_count": 1,
        "persistent_dense_a": False,
        "d1a_residual11_correctness": True,
        "d1b_timing_open": False,
        "residual6_open": True,
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
        "d0_manifest_sha256": _file_hash(D0_ARTIFACT / "manifest.json"),
        "source_capture_sha256": _file_hash(DEFAULT_CAPTURE),
        "model_sha256": _file_hash(DEFAULT_MODEL),
        "performance_claimed": False,
    }
    protocol["protocol_hash"] = _hash(protocol)
    return protocol


def generate(output: Path, capture: Path, model: Path) -> None:
    if output.exists():
        raise FileExistsError(f"R3-D1 artifact output already exists: {output}")
    if not _clean():
        raise RuntimeError("R3-D1 formal generation requires a clean worktree")
    protocol = _protocol()
    with tempfile.TemporaryDirectory(prefix="boundflow-r3d1a-") as temporary:
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
        raise ValueError("R3-D1 artifact boundary differs")
    protocol_copy = dict(protocol)
    protocol_hash = protocol_copy.pop("protocol_hash", None)
    if (
        _hash(protocol_copy) != protocol_hash
        or manifest.get("protocol_hash") != protocol_hash
        or manifest.get("summary_hash") != summary.get("summary_hash")
    ):
        raise ValueError("R3-D1 protocol hash differs")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or any(
        _file_hash(output / str(path)) != digest for path, digest in files.items()
    ):
        raise ValueError("R3-D1 manifest file digest differs")
    manifest_copy = dict(manifest)
    manifest_hash = manifest_copy.pop("manifest_hash", None)
    if _hash(manifest_copy) != manifest_hash:
        raise ValueError("R3-D1 manifest hash differs")
    raws = [_load(path) for path in sorted((output / "raw").glob("*.pt"))]
    rebuilt = _summarize(raws)
    if rebuilt != summary:
        raise ValueError("R3-D1 summary replay differs")
    print(
        f"R3-D1-A replay PASS: max_diff={summary['maximum_diff']} "
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
