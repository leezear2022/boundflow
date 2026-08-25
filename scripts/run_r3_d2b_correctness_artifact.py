#!/usr/bin/env python3
"""Generate or replay the formal five-pair D2-B correctness artifact."""

# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=missing-function-docstring,duplicate-code,too-many-boolean-expressions
# pylint: disable=import-outside-toplevel

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
from typing import Any, Mapping

import torch

from boundflow.runtime.r3_d1c_cumulative_wrapper import (
    R3D1CCumulativeReceiptV1,
)
from boundflow.runtime.r3_d2b_staged_backward import (
    R3D2BStagedBackwardReceiptV1,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAPTURE = (
    ROOT / "artifacts/rvir-v4-pre-state/resnet2b-core-pre-state-v1/source_capture.pt"
)
DEFAULT_MODEL = Path(
    "/home/lee/Codes/vnncomp2021/benchmarks/cifar10_resnet/onnx/resnet_2b.onnx"
)
DEFAULT_OUTPUT = ROOT / "artifacts/r3-structured-owner/r3-d2b-correctness-v1"
WORKER = ROOT / "scripts/run_r3_d2b_correctness_worker.py"
RUN_COUNT = 5
ORDER = (("d1c", "d2b"), ("d2b", "d1c")) * 2 + (("d1c", "d2b"),)
TENSOR_NAMES = (
    "alpha_before",
    "lower",
    "gradient",
    "alpha_after",
    "optimizer_exp_avg",
    "optimizer_exp_avg_sq",
)
CODE_PATHS = (
    "boundflow/runtime/r3_d1c_cumulative_wrapper.py",
    "boundflow/runtime/r3_d2b_staged_backward.py",
    "boundflow/runtime/r3_optimizer_trajectory_timing.py",
    "scripts/run_r3_d2b_correctness_worker.py",
    "scripts/run_r3_d2b_correctness_artifact.py",
    "scripts/probe_r3_d2b_correctness_tamper.py",
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


def _tensor_hash(value: torch.Tensor) -> str:
    from boundflow.runtime.rvir_v4_production_state import production_tensor_sha256

    return production_tensor_sha256(value)


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _clean() -> None:
    dirty = [
        row
        for row in _git("status", "--porcelain").splitlines()
        if not row.endswith("docs/CIBC_for_DAC.pdf") and ".docops/ev.jsonl" not in row
    ]
    if dirty:
        raise RuntimeError(f"R3-D2B formal source is dirty: {dirty}")


def _load(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise TypeError("R3-D2B raw root differs")
    return value


def _step_projection(step: Mapping[str, Any]) -> dict[str, Any]:
    return {name: value for name, value in step.items() if name not in TENSOR_NAMES}


def _validate_worker(raw: Mapping[str, Any]) -> None:
    expected = {
        "schema_version",
        "run_index",
        "mode",
        "source_capture_sha256",
        "model_sha256",
        "plan_hash",
        "trace_hash",
        "metadata",
        "initial_alpha",
        "terminal_alpha",
        "steps",
        "environment",
        "timing_recorded",
        "performance_claimed",
    }
    if (
        set(raw) != expected
        or raw["schema_version"] != "boundflow.r3-d2b-correctness-worker/v1"
        or raw["run_index"] not in range(RUN_COUNT)
        or raw["mode"] not in {"d1c", "d2b"}
        or raw["timing_recorded"] is not False
        or raw["performance_claimed"] is not False
    ):
        raise ValueError("R3-D2B worker envelope differs")
    metadata = raw["metadata"]
    steps = raw["steps"]
    initial = raw["initial_alpha"]
    terminal = raw["terminal_alpha"]
    if (
        not isinstance(metadata, dict)
        or not isinstance(steps, list)
        or len(steps) != 10
        or not torch.is_tensor(initial)
        or not torch.is_tensor(terminal)
        or tuple(initial.shape) != (2, 1, 6, 86)
        or tuple(terminal.shape) != (2, 1, 6, 86)
        or _tensor_hash(initial) != metadata.get("initial_alpha_sha256")
        or _tensor_hash(terminal) != metadata.get("terminal_alpha_sha256")
        or metadata.get("evaluation_count") != 10
        or metadata.get("optimizer_mutation_count") != 9
        or metadata.get("scheduler_mutation_count") != 9
        or metadata.get("timing_recorded") is not False
        or metadata.get("performance_claimed") is not False
    ):
        raise ValueError("R3-D2B trajectory envelope differs")
    unsigned_metadata = dict(metadata)
    trajectory_hash = unsigned_metadata.pop("trajectory_hash", None)
    if trajectory_hash != _hash(unsigned_metadata):
        raise ValueError("R3-D2B trajectory hash differs")
    previous = initial
    rebuilt_step_hashes = []
    for ordinal, step in enumerate(steps):
        if (
            not isinstance(step, dict)
            or set(step)
            != {
                "evaluation_ordinal",
                "update_after",
                "alpha_learning_rate",
                *TENSOR_NAMES,
                "optimizer_step",
                "d1c_receipt",
                "d2b_receipt",
                "tensor_hashes",
            }
            or step["evaluation_ordinal"] != ordinal
            or step["update_after"] != (ordinal < 9)
            or abs(float(step["alpha_learning_rate"]) - 0.01 * 0.98**ordinal) > 1e-15
            or float(step["optimizer_step"]) != float(min(ordinal + 1, 9))
        ):
            raise ValueError("R3-D2B step envelope differs")
        hashes = step["tensor_hashes"]
        if not isinstance(hashes, dict) or set(hashes) != set(TENSOR_NAMES):
            raise TypeError("R3-D2B step hash inventory differs")
        for name in TENSOR_NAMES:
            tensor = step[name]
            if (
                not torch.is_tensor(tensor)
                or not bool(torch.isfinite(tensor).all().item())
                or _tensor_hash(tensor) != hashes[name]
            ):
                raise ValueError(f"R3-D2B step tensor differs: {name}")
        if not torch.equal(step["alpha_before"], previous):
            raise ValueError("R3-D2B alpha lineage differs")
        previous = step["alpha_after"]
        d1c = step["d1c_receipt"]
        d2b = step["d2b_receipt"]
        if not isinstance(d1c, dict):
            raise TypeError("R3-D2B D1-C receipt differs")
        R3D1CCumulativeReceiptV1(**d1c).validate()
        if raw["mode"] == "d2b":
            if not isinstance(d2b, dict):
                raise TypeError("R3-D2B staged receipt differs")
            R3D2BStagedBackwardReceiptV1(**d2b).validate()
        elif d2b is not None:
            raise ValueError("R3-D2B control staged receipt is present")
        rebuilt_step_hashes.append(_hash(_step_projection(step)))
    if not torch.equal(previous, terminal) or rebuilt_step_hashes != metadata.get(
        "step_hashes"
    ):
        raise ValueError("R3-D2B terminal/step replay differs")


def _max_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).abs().max().item())


def _compare_pair(
    control: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, float]:
    _validate_worker(control)
    _validate_worker(candidate)
    if (
        control["mode"] != "d1c"
        or candidate["mode"] != "d2b"
        or control["run_index"] != candidate["run_index"]
    ):
        raise ValueError("R3-D2B pair mode/index differs")
    for name in (
        "source_capture_sha256",
        "model_sha256",
        "plan_hash",
        "trace_hash",
        "environment",
    ):
        if control[name] != candidate[name]:
            raise ValueError(f"R3-D2B pair identity differs: {name}")
    if control["metadata"]["trajectory_id"] != candidate["metadata"]["trajectory_id"]:
        raise ValueError("R3-D2B pair trajectory identity differs")
    maxima = {
        name: 0.0 for name in ("lower", "gradient", "alpha", "exp_avg", "exp_avg_sq")
    }
    for left_step, right_step in zip(control["steps"], candidate["steps"]):
        for raw_name, key, tolerance in (
            ("lower", "lower", 2e-4),
            ("gradient", "gradient", 2e-4),
            ("alpha_after", "alpha", 2e-5),
            ("optimizer_exp_avg", "exp_avg", 2e-5),
            ("optimizer_exp_avg_sq", "exp_avg_sq", 2e-5),
        ):
            left, right = left_step[raw_name], right_step[raw_name]
            maxima[key] = max(maxima[key], _max_diff(left, right))
            if not torch.allclose(left, right, atol=tolerance, rtol=tolerance):
                raise ValueError(f"R3-D2B pair allclose differs: {raw_name}")
            if raw_name in {"lower", "gradient"} and not torch.equal(
                torch.sign(left), torch.sign(right)
            ):
                raise ValueError(f"R3-D2B pair sign differs: {raw_name}")
    return maxima


def _summary(raws: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = []
    for run_index in range(RUN_COUNT):
        by_mode = {raw["mode"]: raw for raw in raws if raw["run_index"] == run_index}
        if set(by_mode) != {"d1c", "d2b"}:
            raise ValueError("R3-D2B pair inventory differs")
        pairs.append(_compare_pair(by_mode["d1c"], by_mode["d2b"]))
    result: dict[str, Any] = {
        "schema_version": "boundflow.r3-d2b-correctness-summary/v1",
        "pair_count": RUN_COUNT,
        "worker_count": RUN_COUNT * 2,
        "order": [list(value) for value in ORDER],
        "maximum_lower_max_abs_diff": max(row["lower"] for row in pairs),
        "maximum_gradient_max_abs_diff": max(row["gradient"] for row in pairs),
        "maximum_alpha_max_abs_diff": max(row["alpha"] for row in pairs),
        "maximum_exp_avg_max_abs_diff": max(row["exp_avg"] for row in pairs),
        "maximum_exp_avg_sq_max_abs_diff": max(row["exp_avg_sq"] for row in pairs),
        "trajectory_correctness_admitted": True,
        "ownership_admitted": True,
        "d2b_timing_open": True,
        "r3_3_open": False,
        "same_solver_open": False,
        "timing_recorded": False,
        "performance_claimed": False,
    }
    result["summary_hash"] = _hash(result)
    return result


def _protocol(revision: str, capture: Path, model: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": "boundflow.r3-d2b-correctness-protocol/v1",
        "source_revision": revision,
        "pair_count": RUN_COUNT,
        "order": [list(value) for value in ORDER],
        "evaluation_count": 10,
        "mutation_count": 9,
        "lower_gradient_tolerance": {"atol": 2e-4, "rtol": 2e-4, "sign_exact": True},
        "state_tolerance": {"atol": 2e-5, "rtol": 2e-5},
        "source_capture_sha256": _file_hash(capture),
        "model_sha256": _file_hash(model),
        "timing_recorded": False,
        "performance_claimed": False,
        "code_revision": {name: _file_hash(ROOT / name) for name in CODE_PATHS},
    }
    result["protocol_hash"] = _hash(result)
    return result


def generate(output: Path, capture: Path, model: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"R3-D2B artifact exists: {output}")
    _clean()
    protocol = _protocol(_git("rev-parse", "HEAD"), capture, model)
    temporary = Path(tempfile.mkdtemp(prefix="r3-d2b-formal-", dir=output.parent))
    try:
        raw_dir = temporary / "raw"
        raw_dir.mkdir(parents=True)
        raws = []
        for run_index, modes in enumerate(ORDER):
            for mode in modes:
                target = raw_dir / f"run-{run_index:02d}-{mode}.pt"
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
            "schema_version": "boundflow.r3-d2b-correctness-manifest/v1",
            "source_revision": protocol["source_revision"],
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
        or manifest.get("schema_version") != "boundflow.r3-d2b-correctness-manifest/v1"
    ):
        raise ValueError("R3-D2B manifest differs")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or any(
        _file_hash(artifact / name) != digest for name, digest in files.items()
    ):
        raise ValueError("R3-D2B file digest differs")
    protocol = json.loads((artifact / "protocol.json").read_text(encoding="utf-8"))
    unsigned_protocol = dict(protocol)
    protocol_hash = unsigned_protocol.pop("protocol_hash", None)
    if (
        protocol_hash != _hash(unsigned_protocol)
        or protocol_hash != manifest["protocol_hash"]
    ):
        raise ValueError("R3-D2B protocol hash differs")
    frozen = {
        "pair_count": RUN_COUNT,
        "order": [list(value) for value in ORDER],
        "evaluation_count": 10,
        "mutation_count": 9,
        "lower_gradient_tolerance": {"atol": 2e-4, "rtol": 2e-4, "sign_exact": True},
        "state_tolerance": {"atol": 2e-5, "rtol": 2e-5},
        "timing_recorded": False,
        "performance_claimed": False,
    }
    if any(protocol.get(name) != value for name, value in frozen.items()):
        raise ValueError("R3-D2B frozen protocol differs")
    raws = [_load(path) for path in sorted((artifact / "raw").glob("*.pt"))]
    summary = _summary(raws)
    if (
        summary != json.loads((artifact / "summary.json").read_text(encoding="utf-8"))
        or summary["summary_hash"] != manifest["summary_hash"]
    ):
        raise ValueError("R3-D2B semantic replay differs")
    print(
        f"R3-D2B replay PASS: pairs={summary['pair_count']} "
        f"gradient_max={summary['maximum_gradient_max_abs_diff']} "
        f"timing_open={summary['d2b_timing_open']}",
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
