#!/usr/bin/env python3
"""Generate or replay the R3-1 M0 five-fresh correctness/memory kill gate."""

# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=wrong-import-position,duplicate-code,missing-function-docstring
# pylint: disable=too-many-arguments,too-many-boolean-expressions

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

from boundflow.runtime.rvir_v4_production_state import production_tensor_sha256

ARTIFACT_SCHEMA = "boundflow.r3-1-m0-five-fresh-artifact/v1"
PROTOCOL_SCHEMA = "boundflow.r3-1-m0-five-fresh-protocol/v1"
SUMMARY_SCHEMA = "boundflow.r3-1-m0-five-fresh-summary/v1"
WORKER_SCHEMA = "boundflow.r3-1-full-region-worker/v1"
RUN_COUNT = 5
PAIR_ORDERS = ("NC", "CN", "NC", "CN", "NC")
ATOL = 2.0e-4
RTOL = 2.0e-4
CODE_PATHS = (
    "boundflow/runtime/r3_structured_owner_custom_backward.py",
    "scripts/run_r3_structured_owner_worker.py",
    "scripts/run_r3_structured_owner_five_fresh_artifact.py",
    "tests/test_r3_structured_owner_custom_backward.py",
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


def _write_json(path: Path, payload: object) -> None:
    path.write_text(_canonical_json(payload, indent=2) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"R3-1 JSON root differs: {path.name}")
    return cast(dict[str, Any], value)


def _load_torch(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise TypeError(f"R3-1 raw root differs: {path.name}")
    return cast(dict[str, Any], value)


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
    return {name: _file_sha256(REPOSITORY_ROOT / name) for name in CODE_PATHS}


def _raw_name(run_index: int, mode: str) -> str:
    return f"run_{run_index:02d}_{mode}.pt"


def _stdout_name(run_index: int, mode: str) -> str:
    return f"run_{run_index:02d}_{mode}.stdout.txt"


def _artifact_files() -> tuple[str, ...]:
    names = ["protocol.json", "summary.json", "replay_stdout.txt", "README.md"]
    for run_index in range(RUN_COUNT):
        for mode in ("native", "candidate"):
            names.extend((_raw_name(run_index, mode), _stdout_name(run_index, mode)))
    return tuple(names)


def _protocol(source_capture: Path, model: Path) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": PROTOCOL_SCHEMA,
        "source_git_head": _git("rev-parse", "HEAD"),
        "code_revision": _code_revision(),
        "source_capture_sha256": _file_sha256(source_capture),
        "model_sha256": _file_sha256(model),
        "run_count": RUN_COUNT,
        "pair_orders": list(PAIR_ORDERS),
        "process_isolation": "one-mode-per-fresh-subprocess",
        "start_node_id": "25/Conv_8",
        "evaluation_count": 1,
        "optimizer_mutation_count": 0,
        "atol": ATOL,
        "rtol": RTOL,
        "sign_exact": True,
        "memory_peak_allocated_ratio_max": 1.0,
        "memory_peak_reserved_ratio_max": 1.0,
        "compiled_region_required": True,
        "saved_dense_a_count_max": 0,
        "scratch_slot_count_max": 2,
        "custom_forward_count": 1,
        "custom_backward_count": 1,
        "timing_recorded": False,
        "performance_claimed": False,
    }
    payload["protocol_hash"] = _canonical_hash(payload)
    return payload


def _run_worker(
    *,
    python: Path,
    source_capture: Path,
    model: Path,
    mode: str,
    run_index: int,
    result: Path,
) -> str:
    environment = dict(os.environ)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT)
    completed = subprocess.run(
        (
            str(python),
            str(REPOSITORY_ROOT / "scripts/run_r3_structured_owner_worker.py"),
            "--source-capture",
            str(source_capture),
            "--model",
            str(model),
            "--mode",
            mode,
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
            f"R3-1 worker failed run={run_index} mode={mode}:\n{completed.stdout}"
        )
    if "/home/" in completed.stdout or "/tmp/" in completed.stdout:
        raise ValueError("R3-1 worker stdout leaks a host-local path")
    return completed.stdout


def _validate_worker(
    row: Mapping[str, Any], protocol: Mapping[str, Any], *, run_index: int, mode: str
) -> None:
    lower = row.get("final_lower")
    gradient = row.get("compressed_alpha_gradient")
    receipt = row.get("execution_receipt")
    memory = row.get("memory")
    if (
        row.get("schema_version") != WORKER_SCHEMA
        or row.get("run_index") != run_index
        or row.get("mode") != mode
        or row.get("source_capture_sha256") != protocol.get("source_capture_sha256")
        or row.get("model_sha256") != protocol.get("model_sha256")
        or not isinstance(lower, torch.Tensor)
        or not isinstance(gradient, torch.Tensor)
        or tuple(lower.shape) != (6, 1)
        or tuple(gradient.shape) != (2, 1, 6, 86)
        or not isinstance(receipt, dict)
        or not isinstance(memory, dict)
        or row.get("final_lower_sha256") != production_tensor_sha256(lower)
        or row.get("compressed_alpha_gradient_sha256")
        != production_tensor_sha256(gradient)
        or row.get("timing_recorded") is not False
        or row.get("performance_claimed") is not False
        or row.get("alpha_versions_before") != row.get("alpha_versions_after")
        or row.get("beta_versions_before") != row.get("beta_versions_after")
    ):
        raise ValueError(f"R3-1 raw worker differs: run={run_index} mode={mode}")
    required_memory = {
        "allocated_before",
        "reserved_before",
        "peak_allocated",
        "peak_reserved",
        "peak_allocated_increment",
        "peak_reserved_increment",
    }
    if set(memory) != required_memory or any(
        not isinstance(memory[name], int) or memory[name] < 0
        for name in required_memory
    ):
        raise ValueError("R3-1 raw memory receipt differs")
    if mode == "candidate":
        required = {
            "forward_count": 1,
            "backward_count": 1,
            "custom_backward_count": 1,
            "saved_dense_a_count": 0,
            "scratch_slot_count": 2,
            "fallback_count": 0,
            "eager_escape_count": 0,
            "native_shadow_count": 0,
            "optimizer_mutation_count": 0,
            "production_connected": True,
            "timing_recorded": False,
            "performance_claimed": False,
            "compiled_region": False,
            "python_eager_rematerialization": True,
        }
        if any(receipt.get(name) != value for name, value in required.items()):
            raise ValueError("R3-1 candidate execution receipt differs")
    elif (
        receipt.get("execution_kind") != "independent-native-oracle"
        or receipt.get("forward_count") != 1
        or receipt.get("backward_count") != 1
        or receipt.get("optimizer_mutation_count") != 0
    ):
        raise ValueError("R3-1 native receipt differs")


def _maximum_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    difference = (left - right).abs()
    return 0.0 if difference.numel() == 0 else float(difference.max().item())


def _summary(
    runs: Mapping[tuple[int, str], Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> dict[str, object]:
    rows = []
    maximum_lower = 0.0
    maximum_gradient = 0.0
    worst_allocated_ratio = 0.0
    worst_reserved_ratio = 0.0
    all_semantic = True
    all_allocated = True
    all_reserved = True
    all_structure = True
    all_compiled = True
    plan_hashes: set[str] = set()
    state_hashes: set[str] = set()
    for run_index in range(RUN_COUNT):
        native = runs[(run_index, "native")]
        candidate = runs[(run_index, "candidate")]
        _validate_worker(native, protocol, run_index=run_index, mode="native")
        _validate_worker(candidate, protocol, run_index=run_index, mode="candidate")
        plan_hashes.update((native["plan_hash"], candidate["plan_hash"]))
        state_hashes.update(
            (native["source_state_hash"], candidate["source_state_hash"])
        )
        native_lower = cast(torch.Tensor, native["final_lower"])
        candidate_lower = cast(torch.Tensor, candidate["final_lower"])
        native_gradient = cast(torch.Tensor, native["compressed_alpha_gradient"])
        candidate_gradient = cast(torch.Tensor, candidate["compressed_alpha_gradient"])
        lower_difference = _maximum_difference(native_lower, candidate_lower)
        gradient_difference = _maximum_difference(native_gradient, candidate_gradient)
        lower_close = torch.allclose(
            native_lower, candidate_lower, atol=ATOL, rtol=RTOL
        )
        gradient_close = torch.allclose(
            native_gradient, candidate_gradient, atol=ATOL, rtol=RTOL
        )
        sign_exact = torch.equal(
            torch.sign(native_lower), torch.sign(candidate_lower)
        ) and torch.equal(torch.sign(native_gradient), torch.sign(candidate_gradient))
        semantic = bool(lower_close and gradient_close and sign_exact)
        native_memory = cast(Mapping[str, int], native["memory"])
        candidate_memory = cast(Mapping[str, int], candidate["memory"])
        allocated_ratio = (
            candidate_memory["peak_allocated"] / native_memory["peak_allocated"]
        )
        reserved_ratio = (
            candidate_memory["peak_reserved"] / native_memory["peak_reserved"]
        )
        receipt = cast(Mapping[str, Any], candidate["execution_receipt"])
        structure = (
            receipt["saved_dense_a_count"] == 0
            and receipt["scratch_slot_count"] <= 2
            and receipt["forward_count"] == 1
            and receipt["custom_backward_count"] == 1
            and receipt["optimizer_mutation_count"] == 0
        )
        compiled = receipt["compiled_region"] is True
        maximum_lower = max(maximum_lower, lower_difference)
        maximum_gradient = max(maximum_gradient, gradient_difference)
        worst_allocated_ratio = max(worst_allocated_ratio, allocated_ratio)
        worst_reserved_ratio = max(worst_reserved_ratio, reserved_ratio)
        all_semantic = all_semantic and semantic
        all_allocated = all_allocated and allocated_ratio <= 1.0
        all_reserved = all_reserved and reserved_ratio <= 1.0
        all_structure = all_structure and structure
        all_compiled = all_compiled and compiled
        rows.append(
            {
                "run_index": run_index,
                "pair_order": PAIR_ORDERS[run_index],
                "lower_maximum_absolute_difference": lower_difference,
                "gradient_maximum_absolute_difference": gradient_difference,
                "semantic_passed": semantic,
                "peak_allocated_ratio": allocated_ratio,
                "peak_reserved_ratio": reserved_ratio,
                "allocated_gate_passed": allocated_ratio <= 1.0,
                "reserved_gate_passed": reserved_ratio <= 1.0,
                "structure_gate_passed": structure,
                "compiled_region_gate_passed": compiled,
            }
        )
    if len(plan_hashes) != 1 or len(state_hashes) != 1:
        raise ValueError("R3-1 frozen plan/state identity differs across workers")
    passed = (
        all_semantic
        and all_allocated
        and all_reserved
        and all_structure
        and all_compiled
    )
    status = (
        "validated-r3-1-m0"
        if passed
        else "validated-no-go-r3-1-m0-python-rematerialization"
    )
    summary: dict[str, object] = {
        "schema_version": SUMMARY_SCHEMA,
        "status": status,
        "run_count": RUN_COUNT,
        "worker_count": RUN_COUNT * 2,
        "plan_hash": next(iter(plan_hashes)),
        "source_state_hash": next(iter(state_hashes)),
        "maximum_lower_absolute_difference": maximum_lower,
        "maximum_compressed_alpha_gradient_absolute_difference": maximum_gradient,
        "all_semantic_passed": all_semantic,
        "all_structure_passed": all_structure,
        "all_peak_allocated_passed": all_allocated,
        "all_peak_reserved_passed": all_reserved,
        "all_compiled_region_passed": all_compiled,
        "worst_peak_allocated_ratio": worst_allocated_ratio,
        "worst_peak_reserved_ratio": worst_reserved_ratio,
        "r3_1_admitted": passed,
        "r3_2a_open": passed,
        "rows": rows,
        "timing_recorded": False,
        "performance_claimed": False,
    }
    summary["summary_hash"] = _canonical_hash(summary)
    return summary


def _result(summary: Mapping[str, object]) -> dict[str, object]:
    return {
        "status": "replay-passed",
        "evidence_status": summary["status"],
        "run_count": summary["run_count"],
        "worker_count": summary["worker_count"],
        "maximum_lower_absolute_difference": summary[
            "maximum_lower_absolute_difference"
        ],
        "maximum_compressed_alpha_gradient_absolute_difference": summary[
            "maximum_compressed_alpha_gradient_absolute_difference"
        ],
        "worst_peak_allocated_ratio": summary["worst_peak_allocated_ratio"],
        "worst_peak_reserved_ratio": summary["worst_peak_reserved_ratio"],
        "r3_1_admitted": summary["r3_1_admitted"],
        "r3_2a_open": summary["r3_2a_open"],
        "summary_hash": summary["summary_hash"],
        "timing_recorded": False,
        "performance_claimed": False,
    }


def _readme() -> str:
    return (
        "# R3-1 M0 Five-Fresh Kill Gate\n\n"
        "Five independent native/candidate pairs validate final lower, compressed "
        "P-alpha VJP, Function ownership, and peak allocated/reserved memory. The "
        "frozen R3-1 gate also requires a compiled bounded-arena region. No latency "
        "is recorded and no performance claim is allowed.\n"
    )


def _all_files(root: Path) -> dict[str, str]:
    return {name: _file_sha256(root / name) for name in _artifact_files()}


def _generate(args: argparse.Namespace) -> dict[str, object]:
    if _git("status", "--porcelain=v1", "--", *CODE_PATHS):
        raise ValueError("R3-1 formal code paths must be committed")
    output = args.artifact.resolve()
    if output.exists():
        raise FileExistsError(f"R3-1 artifact exists: {output}")
    source_capture = args.source_capture.resolve()
    model = args.model.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.incomplete-", dir=output.parent
    ) as temporary:
        root = Path(temporary)
        protocol = _protocol(source_capture, model)
        _write_json(root / "protocol.json", protocol)
        runs: dict[tuple[int, str], Mapping[str, Any]] = {}
        for run_index, order in enumerate(PAIR_ORDERS):
            modes = (
                ("native", "candidate") if order == "NC" else ("candidate", "native")
            )
            for mode in modes:
                raw_path = root / _raw_name(run_index, mode)
                stdout = _run_worker(
                    python=args.python.expanduser().absolute(),
                    source_capture=source_capture,
                    model=model,
                    mode=mode,
                    run_index=run_index,
                    result=raw_path,
                )
                (root / _stdout_name(run_index, mode)).write_text(
                    stdout, encoding="utf-8"
                )
                runs[(run_index, mode)] = _load_torch(raw_path)
        summary = _summary(runs, protocol)
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
            "timing_recorded": False,
            "performance_claimed": False,
        }
        manifest["manifest_hash"] = _canonical_hash(manifest)
        _write_json(root / "manifest.json", manifest)
        shutil.move(root, output)
    _verify_static(output)
    return result


def _verify_static(
    artifact: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[tuple[int, str], Mapping[str, Any]]]:
    manifest = _load_json(artifact / "manifest.json")
    unsigned = dict(manifest)
    claimed_manifest_hash = unsigned.pop("manifest_hash", None)
    files = manifest.get("files")
    if (
        manifest.get("schema_version") != ARTIFACT_SCHEMA
        or claimed_manifest_hash != _canonical_hash(unsigned)
        or manifest.get("timing_recorded") is not False
        or manifest.get("performance_claimed") is not False
        or not isinstance(files, dict)
        or set(files) != set(_artifact_files())
        or any(files[name] != _file_sha256(artifact / name) for name in files)
    ):
        raise ValueError("R3-1 artifact manifest differs")
    protocol = _load_json(artifact / "protocol.json")
    unsigned_protocol = dict(protocol)
    claimed_protocol_hash = unsigned_protocol.pop("protocol_hash", None)
    if (
        protocol.get("schema_version") != PROTOCOL_SCHEMA
        or claimed_protocol_hash != _canonical_hash(unsigned_protocol)
        or manifest.get("protocol_hash") != claimed_protocol_hash
        or protocol.get("run_count") != RUN_COUNT
        or protocol.get("pair_orders") != list(PAIR_ORDERS)
    ):
        raise ValueError("R3-1 artifact protocol differs")
    runs: dict[tuple[int, str], Mapping[str, Any]] = {}
    for run_index in range(RUN_COUNT):
        for mode in ("native", "candidate"):
            row = _load_torch(artifact / _raw_name(run_index, mode))
            _validate_worker(row, protocol, run_index=run_index, mode=mode)
            runs[(run_index, mode)] = row
    return manifest, protocol, runs


def _replay(artifact: Path) -> dict[str, object]:
    manifest, protocol, runs = _verify_static(artifact)
    expected = _summary(runs, protocol)
    observed = _load_json(artifact / "summary.json")
    if observed != expected or manifest.get("summary_hash") != expected["summary_hash"]:
        raise ValueError("R3-1 artifact summary differs")
    result = _result(expected)
    if (artifact / "replay_stdout.txt").read_text(encoding="utf-8") != (
        _canonical_json(result) + "\n"
    ):
        raise ValueError("R3-1 replay stdout differs")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--source-capture", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument(
        "--python", type=Path, default=Path(sys.executable), help="worker interpreter"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.replay:
        result = _replay(args.artifact.resolve())
    else:
        if args.source_capture is None or args.model is None:
            raise ValueError("R3-1 generation requires source capture and model")
        result = _generate(args)
    print(_canonical_json(result), flush=True)


if __name__ == "__main__":
    main()
