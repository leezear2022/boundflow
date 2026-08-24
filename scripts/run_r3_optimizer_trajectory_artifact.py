#!/usr/bin/env python3
"""Generate or replay the formal R3-2A five-fresh trajectory artifact."""

# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=missing-function-docstring,import-outside-toplevel
# pylint: disable=too-many-boolean-expressions,duplicate-code

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
from typing import Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAPTURE = (
    ROOT / "artifacts/rvir-v4-pre-state/resnet2b-core-pre-state-v1/source_capture.pt"
)
DEFAULT_MODEL = Path(
    "/home/lee/Codes/vnncomp2021/benchmarks/cifar10_resnet/onnx/resnet_2b.onnx"
)
DEFAULT_OUTPUT = ROOT / "artifacts/r3-structured-owner/r3-2a-optimizer-trajectory-v1"
WORKER = ROOT / "scripts/run_r3_optimizer_trajectory_worker.py"
PROTOCOL_SCHEMA = "boundflow.r3-2a-optimizer-trajectory-protocol/v1"
MANIFEST_SCHEMA = "boundflow.r3-2a-optimizer-trajectory-manifest/v1"
ORDER = (("native", "candidate"), ("candidate", "native")) * 2 + (
    ("native", "candidate"),
)
CODE_PATHS = (
    "boundflow/runtime/r3_optimizer_trajectory.py",
    "boundflow/runtime/r3_compiled_p_alpha_vjp.py",
    "boundflow/runtime/r3_full_lower_forward_tir.py",
    "boundflow/backends/tvm/r3_p_alpha_vjp.py",
    "boundflow/backends/tvm/r3_full_lower_forward.py",
    "boundflow/ir/r3_bounded_arena.py",
    "scripts/run_r3_optimizer_trajectory_worker.py",
    "scripts/run_r3_optimizer_trajectory_artifact.py",
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
    return subprocess.run(
        ("git", *args), cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def _load(path: Path) -> dict[str, object]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise TypeError("R3-2A raw root differs")
    return value


def _validate_worker(raw: Mapping[str, object]) -> None:
    expected = {
        "schema_version",
        "run_index",
        "mode",
        "source_capture_sha256",
        "model_sha256",
        "plan_hash",
        "trace_hash",
        "trajectory_metadata",
        "trajectory_raw",
        "memory",
        "environment",
        "timing_recorded",
        "performance_claimed",
    }
    if (
        set(raw) != expected
        or raw["schema_version"] != "boundflow.r3-2a-optimizer-trajectory-worker/v1"
        or raw["mode"] not in {"native", "candidate"}
        or raw["timing_recorded"] is not False
        or raw["performance_claimed"] is not False
    ):
        raise ValueError("R3-2A worker envelope differs")
    metadata = raw["trajectory_metadata"]
    values = raw["trajectory_raw"]
    memory = raw["memory"]
    if (
        not isinstance(metadata, dict)
        or not isinstance(values, dict)
        or not isinstance(memory, dict)
    ):
        raise TypeError("R3-2A worker payload structure differs")
    if (
        metadata.get("mode") != raw["mode"]
        or metadata.get("performance_claimed") is not False
    ):
        raise ValueError("R3-2A trajectory mode/claim differs")
    metadata_without_hash = dict(metadata)
    claimed_hash = metadata_without_hash.pop("trajectory_hash", None)
    if claimed_hash != _hash(metadata_without_hash):
        raise ValueError("R3-2A trajectory metadata hash differs")
    steps = values.get("steps")
    metadata_steps = metadata.get("steps")
    if (
        not isinstance(steps, list)
        or not isinstance(metadata_steps, list)
        or len(steps) != len(metadata_steps)
        or len(steps) != 10
    ):
        raise ValueError("R3-2A raw step cardinality differs")
    initial = values.get("initial_alpha")
    terminal = values.get("terminal_alpha")
    if not torch.is_tensor(initial) or not torch.is_tensor(terminal):
        raise TypeError("R3-2A alpha raw differs")
    if _tensor_hash(initial) != metadata.get("initial_alpha_sha256") or _tensor_hash(
        terminal
    ) != metadata.get("terminal_alpha_sha256"):
        raise ValueError("R3-2A alpha raw hash differs")
    previous = initial
    for ordinal, (step, step_meta) in enumerate(zip(steps, metadata_steps)):
        if (
            not isinstance(step, dict)
            or not isinstance(step_meta, dict)
            or step.get("metadata") != step_meta
        ):
            raise ValueError("R3-2A step metadata projection differs")
        if step_meta.get("evaluation_ordinal") != ordinal or step_meta.get(
            "update_after"
        ) != (ordinal < 9):
            raise ValueError("R3-2A step order differs")
        tensors = {
            "alpha_before": "alpha_before_sha256",
            "lower": "lower_sha256",
            "gradient": "gradient_sha256",
            "alpha_after": "alpha_after_sha256",
        }
        for raw_name, hash_name in tensors.items():
            tensor = step.get(raw_name)
            if (
                not torch.is_tensor(tensor)
                or _tensor_hash(tensor) != step_meta.get(hash_name)
                or not bool(torch.isfinite(tensor).all())
            ):
                raise ValueError(f"R3-2A step {raw_name} differs")
        if not torch.equal(step["alpha_before"], previous):
            raise ValueError("R3-2A raw alpha lineage differs")
        previous = step["alpha_after"]
        opt = step_meta.get("optimizer_after")
        if not isinstance(opt, dict):
            raise TypeError("R3-2A optimizer metadata differs")
        for raw_name, hash_name in (
            ("optimizer_exp_avg", "exp_avg_sha256"),
            ("optimizer_exp_avg_sq", "exp_avg_sq_sha256"),
        ):
            tensor = step.get(raw_name)
            if not torch.is_tensor(tensor) or _tensor_hash(tensor) != opt.get(
                hash_name
            ):
                raise ValueError("R3-2A optimizer raw hash differs")
        if float(opt.get("step", -1.0)) != float(min(ordinal + 1, 9)):
            raise ValueError("R3-2A optimizer step differs")
        receipt = step_meta.get("compiled_receipt")
        if raw["mode"] == "candidate":
            if (
                not isinstance(receipt, dict)
                or receipt.get("saved_dense_a_count") != 0
                or receipt.get("coefficient_scratch_count") != 2
                or receipt.get("warm_dynamic_allocated_bytes") != 0
                or receipt.get("fallback_count") != 0
                or receipt.get("native_shadow_count") != 0
            ):
                raise ValueError("R3-2A candidate ownership receipt differs")
        elif receipt is not None:
            raise ValueError("R3-2A native compiled receipt differs")
    if not torch.equal(previous, terminal):
        raise ValueError("R3-2A terminal raw lineage differs")
    if (
        metadata.get("optimizer_mutation_count") != 9
        or metadata.get("scheduler_mutation_count") != 9
    ):
        raise ValueError("R3-2A mutation cardinality differs")
    for name in (
        "allocated_before",
        "reserved_before",
        "peak_allocated",
        "peak_reserved",
    ):
        if not isinstance(memory.get(name), int) or int(memory[name]) < 0:
            raise ValueError("R3-2A memory field differs")
    if memory["peak_allocated"] != metadata.get("peak_allocated_bytes") or memory[
        "peak_reserved"
    ] != metadata.get("peak_reserved_bytes"):
        raise ValueError("R3-2A memory projection differs")


def _max_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).abs().max().item())


def _compare_pair(
    native: Mapping[str, object], candidate: Mapping[str, object]
) -> dict[str, float]:
    _validate_worker(native)
    _validate_worker(candidate)
    if (
        native["mode"] != "native"
        or candidate["mode"] != "candidate"
        or native["run_index"] != candidate["run_index"]
    ):
        raise ValueError("R3-2A pair identity differs")
    for name in (
        "source_capture_sha256",
        "model_sha256",
        "plan_hash",
        "trace_hash",
        "environment",
    ):
        if native[name] != candidate[name]:
            raise ValueError(f"R3-2A pair {name} differs")
    nmeta = native["trajectory_metadata"]
    cmeta = candidate["trajectory_metadata"]
    nraw = native["trajectory_raw"]
    craw = candidate["trajectory_raw"]
    assert isinstance(nmeta, dict) and isinstance(cmeta, dict)
    assert isinstance(nraw, dict) and isinstance(craw, dict)
    if (
        nmeta["trajectory_id"] != cmeta["trajectory_id"]
        or nmeta["immutable_content_hash"] != cmeta["immutable_content_hash"]
    ):
        raise ValueError("R3-2A pair trajectory identity differs")
    maxima = {
        "lower": 0.0,
        "gradient": 0.0,
        "alpha": 0.0,
        "exp_avg": 0.0,
        "exp_avg_sq": 0.0,
    }
    for nstep, cstep in zip(nraw["steps"], craw["steps"]):
        assert isinstance(nstep, dict) and isinstance(cstep, dict)
        for name, atol in (
            ("lower", 2e-4),
            ("gradient", 2e-4),
            ("alpha_after", 2e-5),
            ("optimizer_exp_avg", 2e-5),
            ("optimizer_exp_avg_sq", 2e-5),
        ):
            left, right = nstep[name], cstep[name]
            assert torch.is_tensor(left) and torch.is_tensor(right)
            key = {
                "alpha_after": "alpha",
                "optimizer_exp_avg": "exp_avg",
                "optimizer_exp_avg_sq": "exp_avg_sq",
            }.get(name, name)
            maxima[key] = max(maxima[key], _max_diff(left, right))
            if not torch.allclose(left, right, atol=atol, rtol=atol):
                raise ValueError(f"R3-2A pair {name} allclose differs")
            if name in {"lower", "gradient"} and not torch.equal(
                torch.sign(left), torch.sign(right)
            ):
                raise ValueError(f"R3-2A pair {name} sign differs")
    nmem, cmem = native["memory"], candidate["memory"]
    assert isinstance(nmem, dict) and isinstance(cmem, dict)
    allocated_ratio = float(cmem["peak_allocated"]) / float(nmem["peak_allocated"])
    reserved_ratio = float(cmem["peak_reserved"]) / float(nmem["peak_reserved"])
    if allocated_ratio > 1.0 or reserved_ratio > 1.0:
        raise ValueError("R3-2A pair memory gate differs")
    return maxima | {
        "allocated_ratio": allocated_ratio,
        "reserved_ratio": reserved_ratio,
    }


def _summary(raws: list[dict[str, object]]) -> dict[str, object]:
    pairs = []
    for run_index in range(5):
        by_mode = {raw["mode"]: raw for raw in raws if raw["run_index"] == run_index}
        if set(by_mode) != {"native", "candidate"}:
            raise ValueError("R3-2A pair mode inventory differs")
        pairs.append(_compare_pair(by_mode["native"], by_mode["candidate"]))
    result: dict[str, object] = {
        "pair_count": 5,
        "worker_count": 10,
        "order": [list(value) for value in ORDER],
        "maximum_lower_max_abs_diff": max(float(row["lower"]) for row in pairs),
        "maximum_gradient_max_abs_diff": max(float(row["gradient"]) for row in pairs),
        "maximum_alpha_max_abs_diff": max(float(row["alpha"]) for row in pairs),
        "maximum_exp_avg_max_abs_diff": max(float(row["exp_avg"]) for row in pairs),
        "maximum_exp_avg_sq_max_abs_diff": max(
            float(row["exp_avg_sq"]) for row in pairs
        ),
        "worst_allocated_ratio": max(float(row["allocated_ratio"]) for row in pairs),
        "worst_reserved_ratio": max(float(row["reserved_ratio"]) for row in pairs),
        "trajectory_correctness_admitted": True,
        "r3_2b_open": True,
        "timing_recorded": False,
        "performance_claimed": False,
    }
    result["summary_hash"] = _hash(result)
    return result


def _protocol(source_revision: str, capture: Path, model: Path) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": PROTOCOL_SCHEMA,
        "source_revision": source_revision,
        "source_capture": {
            "repo_path": str(capture.relative_to(ROOT)),
            "sha256": _file_hash(capture),
        },
        "model": {
            "public_id": "vnncomp2021/cifar10_resnet/resnet_2b.onnx",
            "sha256": _file_hash(model),
        },
        "pair_order": [list(value) for value in ORDER],
        "evaluation_count": 10,
        "mutation_count": 9,
        "alpha_lr": 0.01,
        "lr_decay": 0.98,
        "lower_gradient_tolerance": {"atol": 2e-4, "rtol": 2e-4, "sign_exact": True},
        "state_tolerance": {"atol": 2e-5, "rtol": 2e-5},
        "memory_ratio_max": 1.0,
        "timing_recorded": False,
        "performance_claimed": False,
        "code_revision": {name: _file_hash(ROOT / name) for name in CODE_PATHS},
    }
    payload["protocol_hash"] = _hash(payload)
    return payload


def generate(output: Path, capture: Path, model: Path) -> None:
    if output.exists():
        raise FileExistsError(f"R3-2A artifact output already exists: {output}")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("R3-2A formal generation requires a clean worktree")
    revision = _git("rev-parse", "HEAD")
    temp = Path(tempfile.mkdtemp(prefix="r3-2a-artifact-", dir=output.parent))
    raws: list[dict[str, object]] = []
    try:
        raw_dir = temp / "raw"
        raw_dir.mkdir(parents=True)
        for run_index, pair in enumerate(ORDER):
            for sequence, mode in enumerate(pair):
                target = raw_dir / f"run-{run_index:02d}-{sequence}-{mode}.pt"
                command = (
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
                )
                subprocess.run(command, cwd=ROOT, check=True, env=os.environ.copy())
                raws.append(_load(target))
        summary = _summary(raws)
        protocol = _protocol(revision, capture, model)
        (temp / "protocol.json").write_text(_canonical(protocol) + "\n")
        (temp / "summary.json").write_text(_canonical(summary) + "\n")
        files = sorted(
            path.relative_to(temp).as_posix()
            for path in temp.rglob("*")
            if path.is_file()
        )
        manifest: dict[str, object] = {
            "schema_version": MANIFEST_SCHEMA,
            "source_revision": revision,
            "protocol_hash": protocol["protocol_hash"],
            "summary_hash": summary["summary_hash"],
            "files": {name: _file_hash(temp / name) for name in files},
        }
        manifest["manifest_hash"] = _hash(manifest)
        (temp / "manifest.json").write_text(_canonical(manifest) + "\n")
        replay(temp)
        temp.rename(output)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def replay(artifact: Path) -> dict[str, object]:
    manifest = json.loads((artifact / "manifest.json").read_text())
    manifest_without_hash = dict(manifest)
    claimed = manifest_without_hash.pop("manifest_hash", None)
    if (
        claimed != _hash(manifest_without_hash)
        or manifest.get("schema_version") != MANIFEST_SCHEMA
    ):
        raise ValueError("R3-2A manifest hash/schema differs")
    files = manifest.get("files")
    if not isinstance(files, dict) or any(
        _file_hash(artifact / name) != digest for name, digest in files.items()
    ):
        raise ValueError("R3-2A manifest file digest differs")
    protocol = json.loads((artifact / "protocol.json").read_text())
    protocol_without_hash = dict(protocol)
    protocol_hash = protocol_without_hash.pop("protocol_hash", None)
    if (
        protocol_hash != _hash(protocol_without_hash)
        or protocol_hash != manifest["protocol_hash"]
    ):
        raise ValueError("R3-2A protocol hash differs")
    raws = [_load(path) for path in sorted((artifact / "raw").glob("*.pt"))]
    summary = _summary(raws)
    frozen = json.loads((artifact / "summary.json").read_text())
    if summary != frozen or summary["summary_hash"] != manifest["summary_hash"]:
        raise ValueError("R3-2A semantic summary replay differs")
    print(
        "R3-2A replay PASS: pairs=5 evaluations=10 mutations=9 "
        f"lower_max={summary['maximum_lower_max_abs_diff']} "
        f"gradient_max={summary['maximum_gradient_max_abs_diff']} "
        "performance_claimed=false"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args()
    if args.replay is not None:
        replay(args.replay.resolve())
    else:
        generate(
            args.output.resolve(), args.source_capture.resolve(), args.model.resolve()
        )


if __name__ == "__main__":
    main()
