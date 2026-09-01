#!/usr/bin/env python3
"""Generate or replay five fresh B3/B4-A correctness pairs."""

# pylint: disable=wrong-import-position,protected-access,too-many-locals
# pylint: disable=too-many-statements,too-many-branches,too-many-arguments
# pylint: disable=too-many-boolean-expressions,missing-function-docstring
# pylint: disable=duplicate-code

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
from typing import Any, Mapping, cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from boundflow.runtime.fsg3_same_solver_timing import (
    _semantic_pair_failures,
    canonical_hash,
)
from boundflow.runtime.fsg4_b3_same_solver_timing import (
    fsg4_b3_timing_run_from_dict,
)
from scripts import run_fsg4_b4a_same_solver_worker as worker
from scripts import run_rvir_v4_production_state_capture as capture_runner

PROTOCOL_SCHEMA = "boundflow.fsg4-b4a-five-fresh-protocol/v1"
REPORT_SCHEMA = "boundflow.fsg4-b4a-five-fresh-report/v1"
MANIFEST_SCHEMA = "boundflow.fsg4-b4a-five-fresh-manifest/v1"
PAIR_SCHEDULE = (
    ("B3", "B4-A"),
    ("B4-A", "B3"),
    ("B3", "B4-A"),
    ("B4-A", "B3"),
    ("B3", "B4-A"),
)
CODE_PATHS = worker.CODE_PATHS + ("scripts/run_fsg4_b4a_correctness_pairs.py",)
ATOL = 2e-4
RTOL = 2e-4


def _canonical_json(value: object, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
        allow_nan=False,
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical_json(value, indent=2) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"FSG4/B4-A JSON root differs: {path}")
    return value


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


def _schedule_payload() -> list[dict[str, object]]:
    return [
        {"pair_index": index, "positions": list(configurations)}
        for index, configurations in enumerate(PAIR_SCHEDULE)
    ]


def _protocol(args: argparse.Namespace) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": PROTOCOL_SCHEMA,
        "source_git_head": _git("rev-parse", "HEAD"),
        "code_revision": _code_revision(),
        "schedule": _schedule_payload(),
        "pair_count": 5,
        "worker_count": 10,
        "process_isolation": "one-fresh-subprocess-per-position",
        "benchmark_id": "vnncomp2021/cifar10_resnet/resnet2b-prop0",
        "model_sha256": capture_runner.file_sha256(args.model.resolve()),
        "property_sha256": capture_runner.file_sha256(args.property.resolve()),
        "direct_semantic_atol": ATOL,
        "direct_semantic_rtol": RTOL,
        "terminal_export_payload_audit_excluded_from_timing": True,
        "log_path_policy": "deterministic-root-aliases-no-host-local-paths",
        "timing_admitted": False,
        "performance_claimed": False,
    }
    value["protocol_hash"] = canonical_hash(value)
    return value


def _validate_protocol(value: Mapping[str, Any]) -> None:
    payload = dict(value)
    claimed = payload.pop("protocol_hash", None)
    if (
        value.get("schema_version") != PROTOCOL_SCHEMA
        or value.get("schedule") != _schedule_payload()
        or value.get("pair_count") != 5
        or value.get("worker_count") != 10
        or value.get("process_isolation") != "one-fresh-subprocess-per-position"
        or value.get("direct_semantic_atol") != ATOL
        or value.get("direct_semantic_rtol") != RTOL
        or value.get("terminal_export_payload_audit_excluded_from_timing") is not True
        or value.get("log_path_policy")
        != "deterministic-root-aliases-no-host-local-paths"
        or value.get("timing_admitted") is not False
        or value.get("performance_claimed") is not False
        or claimed != canonical_hash(payload)
    ):
        raise ValueError("FSG4/B4-A five-fresh protocol differs")


def _run_root(root: Path, pair_index: int, position: int, config: str) -> Path:
    return (
        root
        / "runs"
        / f"pair-{pair_index:02d}"
        / f"position-{position}-{config.lower()}"
    )


def _command(
    args: argparse.Namespace,
    *,
    pair_index: int,
    position: int,
    configuration: str,
    result: Path,
) -> tuple[str, ...]:
    return (
        str(args.abcrown_python.expanduser().absolute()),
        str(REPOSITORY_ROOT / "scripts/run_fsg4_b4a_same_solver_worker.py"),
        "--configuration",
        configuration,
        "--mode",
        "control",
        "--run-id",
        f"b4a-pair-{pair_index:02d}-position-{position}",
        "--block-index",
        str(pair_index),
        "--sequence-position",
        str(position),
        "--benchmark-root",
        str(args.benchmark_root.resolve()),
        "--abcrown-root",
        str(args.abcrown_root.resolve()),
        "--model",
        str(args.model.resolve()),
        "--property",
        str(args.property.resolve()),
        "--result",
        str(result),
    )


def _sanitize_log(value: str, args: argparse.Namespace) -> str:
    replacements = (
        (str(args.abcrown_python.expanduser().absolute()), "$ABCROWN_PYTHON"),
        (str(args.abcrown_root.resolve()), "$ABCROWN_ROOT"),
        (str(args.benchmark_root.resolve()), "$BENCHMARK_ROOT"),
        (str(REPOSITORY_ROOT), "$BOUNDFLOW_ROOT"),
    )
    sanitized = value
    for source, replacement in replacements:
        sanitized = sanitized.replace(source, replacement)
    if "/home/" in sanitized or "/tmp/" in sanitized:
        raise ValueError("FSG4/B4-A worker log contains a host-local path")
    return sanitized


def _tensor_values(value: object, label: str) -> tuple[tuple[int, ...], list[float]]:
    if not isinstance(value, Mapping):
        raise TypeError(f"FSG4/B4-A tensor payload differs: {label}")
    shape = value.get("shape")
    encoded = value.get("content_base64")
    if (
        not isinstance(shape, list)
        or any(not isinstance(item, int) or item < 0 for item in shape)
        or value.get("dtype") != "torch.float32"
        or not isinstance(encoded, str)
    ):
        raise ValueError(f"FSG4/B4-A tensor schema differs: {label}")
    raw = base64.b64decode(encoded, validate=True)
    element_count = math.prod(shape)
    if len(raw) != element_count * 4:
        raise ValueError(f"FSG4/B4-A tensor byte count differs: {label}")
    values = [item[0] for item in struct.iter_unpack("<f", raw)]
    if len(values) != element_count or any(not math.isfinite(item) for item in values):
        raise ValueError(f"FSG4/B4-A tensor value differs: {label}")
    return tuple(shape), values


def _tensor_pair(actual: object, expected: object, label: str) -> tuple[float, bool]:
    actual_shape, actual_values = _tensor_values(actual, f"{label}:actual")
    expected_shape, expected_values = _tensor_values(expected, f"{label}:expected")
    if actual_shape != expected_shape or len(actual_values) != len(expected_values):
        raise ValueError(f"FSG4/B4-A tensor pair shape differs: {label}")
    maximum = 0.0
    sign_exact = True
    for observed, reference in zip(actual_values, expected_values):
        difference = abs(observed - reference)
        maximum = max(maximum, difference)
        if difference > ATOL + RTOL * abs(reference):
            raise ValueError(f"FSG4/B4-A tensor pair numeric differs: {label}")
        sign_exact = sign_exact and (observed > 0) - (observed < 0) == (
            (reference > 0) - (reference < 0)
        )
    if not sign_exact:
        raise ValueError(f"FSG4/B4-A tensor pair sign differs: {label}")
    return maximum, sign_exact


def _one_payload(worker_value: Mapping[str, Any]) -> Mapping[str, Any]:
    diagnostics = worker_value.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise TypeError("FSG4/B4-A worker diagnostics differs")
    payloads = diagnostics.get("native_backward_export_payloads")
    if (
        not isinstance(payloads, list)
        or len(payloads) != 1
        or not isinstance(payloads[0], Mapping)
    ):
        raise ValueError("FSG4/B4-A export payload cardinality differs")
    if (
        diagnostics.get("terminal_export_audit_excluded_from_timing") is not True
        or not isinstance(diagnostics.get("terminal_export_audit_ns"), int)
        or int(diagnostics["terminal_export_audit_ns"]) <= 0
    ):
        raise ValueError("FSG4/B4-A export audit timing boundary differs")
    return cast(Mapping[str, Any], payloads[0])


def _export_pair(
    control: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, object]:
    maxima: list[float] = []
    control_lower = control.get("lower")
    candidate_lower = candidate.get("lower")
    maximum, _sign = _tensor_pair(candidate_lower, control_lower, "terminal-lower")
    maxima.append(maximum)
    for category in ("lAs", "intermediates"):
        left = control.get(category)
        right = candidate.get(category)
        if (
            not isinstance(left, Mapping)
            or not isinstance(right, Mapping)
            or set(left) != set(right)
        ):
            raise ValueError(f"FSG4/B4-A export inventory differs: {category}")
        for name in sorted(left):
            if category == "lAs":
                maximum, _sign = _tensor_pair(right[name], left[name], f"lA:{name}")
                maxima.append(maximum)
            else:
                left_interval = left[name]
                right_interval = right[name]
                if not isinstance(left_interval, Mapping) or not isinstance(
                    right_interval, Mapping
                ):
                    raise TypeError("FSG4/B4-A intermediate interval differs")
                for side in ("lower", "upper"):
                    maximum, _sign = _tensor_pair(
                        right_interval.get(side),
                        left_interval.get(side),
                        f"intermediate:{name}:{side}",
                    )
                    maxima.append(maximum)
    return {
        "tensor_count": len(maxima),
        "maximum_absolute_difference": max(maxima, default=0.0),
        "all_sign_exact": True,
    }


def _load_worker(path: Path, configuration: str) -> tuple[Any, dict[str, Any]]:
    value = _load_json(path)
    run_payload = value.get("run")
    activation = value.get("activation")
    if (
        value.get("schema_version") != worker.WORKER_SCHEMA
        or value.get("configuration") != configuration
        or value.get("mode") != "control"
        or value.get("performance_claimed") is not False
        or not isinstance(run_payload, Mapping)
        or not isinstance(activation, Mapping)
    ):
        raise ValueError("FSG4/B4-A worker envelope differs")
    run = fsg4_b3_timing_run_from_dict(cast(Mapping[str, object], run_payload))
    if not run.environment.admitted or run.mode.value != "control":
        raise ValueError("FSG4/B4-A worker environment differs")
    return run, value


def _pair_row(root: Path, pair_index: int) -> dict[str, object]:
    values: dict[str, tuple[Any, dict[str, Any]]] = {}
    for position, configuration in enumerate(PAIR_SCHEDULE[pair_index]):
        run_root = _run_root(root, pair_index, position, configuration)
        values[configuration] = _load_worker(run_root / "worker.json", configuration)
    if set(values) != {"B3", "B4-A"}:
        raise ValueError("FSG4/B4-A pair inventory differs")
    control_run, control_worker = values["B3"]
    candidate_run, candidate_worker = values["B4-A"]
    failures = _semantic_pair_failures(
        control_run.semantics,
        candidate_run.semantics,
        label=f"b4a-pair-{pair_index:02d}",
    )
    control_runtime = cast(Mapping[str, Any], control_worker["diagnostics"])[
        "runtime_environment"
    ]
    candidate_runtime = cast(Mapping[str, Any], candidate_worker["diagnostics"])[
        "runtime_environment"
    ]
    candidate_activation = cast(Mapping[str, Any], candidate_worker["activation"])
    if (
        failures
        or control_run.source_identity != candidate_run.source_identity
        or control_run.environment.gpu_uuid != candidate_run.environment.gpu_uuid
        or control_run.environment.gpu_name != candidate_run.environment.gpu_name
        or control_runtime != candidate_runtime
        or candidate_activation.get("terminal_lower_adjoint_handoff_count") != 1
        or candidate_activation.get("terminal_export_crown_rerun_count") != 0
        or candidate_activation.get("lineage_count") != 6
        or candidate_activation.get("provider_callback_count") != 0
        or candidate_activation.get("fallback_dispatch_count") != 0
    ):
        raise ValueError("FSG4/B4-A direct pair gate differs: " + ",".join(failures))
    export_pair = _export_pair(
        _one_payload(control_worker), _one_payload(candidate_worker)
    )
    return {
        "pair_index": pair_index,
        "schedule": list(PAIR_SCHEDULE[pair_index]),
        "source_identity": control_run.source_identity,
        "gpu_uuid": control_run.environment.gpu_uuid,
        "runtime_environment_hash": canonical_hash(control_runtime),
        "semantic_failures": [],
        "export_pair": export_pair,
        "b3_activation": control_worker["activation"],
        "b4a_activation": candidate_worker["activation"],
        "environment_admitted": True,
        "provider_fallback_zero": True,
        "performance_claimed": False,
    }


def _report(root: Path) -> dict[str, object]:
    protocol = _load_json(root / "protocol.json")
    _validate_protocol(protocol)
    pairs = [_pair_row(root, index) for index in range(5)]
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA,
        "source_git_head": protocol["source_git_head"],
        "protocol_hash": protocol["protocol_hash"],
        "pair_count": 5,
        "worker_count": 10,
        "pairs": pairs,
        "all_direct_semantic_pairs_passed": True,
        "all_terminal_export_pairs_passed": True,
        "all_lineage_and_counter_gates_passed": True,
        "maximum_export_absolute_difference": max(
            float(
                cast(Mapping[str, Any], pair["export_pair"])[
                    "maximum_absolute_difference"
                ]
            )
            for pair in pairs
        ),
        "timing_admitted": True,
        "performance_claimed": False,
    }
    report["report_hash"] = canonical_hash(report)
    return report


def _all_files(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != root / "manifest.json"
    }


def _generate(args: argparse.Namespace) -> None:
    root = args.artifact_dir.resolve()
    if root.exists():
        raise FileExistsError(f"FSG4/B4-A artifact exists: {root}")
    dirty = _git("status", "--porcelain=v1", "--", *CODE_PATHS)
    if dirty:
        raise RuntimeError("FSG4/B4-A correctness code paths must be committed")
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{root.name}.incomplete-", dir=root.parent
    ) as raw:
        staging = Path(raw)
        _write_json(staging / "protocol.json", _protocol(args))
        for pair_index, configurations in enumerate(PAIR_SCHEDULE):
            for position, configuration in enumerate(configurations):
                run_root = _run_root(staging, pair_index, position, configuration)
                run_root.mkdir(parents=True, exist_ok=True)
                completed = subprocess.run(
                    _command(
                        args,
                        pair_index=pair_index,
                        position=position,
                        configuration=configuration,
                        result=run_root / "worker.json",
                    ),
                    cwd=REPOSITORY_ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )
                (run_root / "stdout.txt").write_text(
                    _sanitize_log(completed.stdout, args), encoding="utf-8"
                )
                (run_root / "stderr.txt").write_text(
                    _sanitize_log(completed.stderr, args), encoding="utf-8"
                )
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"FSG4/B4-A worker failed pair={pair_index} position={position}:\n"
                        + completed.stdout
                        + completed.stderr
                    )
        report = _report(staging)
        _write_json(staging / "report.json", report)
        (staging / "README.md").write_text(
            "# FSG4/B4-A five-fresh correctness\n\n"
            "Five independent B3/B4-A pairs. Correctness only; performance_claimed=false.\n",
            encoding="utf-8",
        )
        manifest: dict[str, object] = {
            "schema_version": MANIFEST_SCHEMA,
            "source_git_head": _git("rev-parse", "HEAD"),
            "code_revision": _code_revision(),
            "protocol_hash": _load_json(staging / "protocol.json")["protocol_hash"],
            "report_hash": report["report_hash"],
            "files": _all_files(staging),
            "timing_admitted": True,
            "performance_claimed": False,
        }
        manifest["manifest_hash"] = canonical_hash(manifest)
        _write_json(staging / "manifest.json", manifest)
        shutil.move(staging, root)
    _replay(root)


def _replay(root: Path) -> dict[str, object]:
    root = root.resolve()
    manifest = _load_json(root / "manifest.json")
    semantic_manifest = dict(manifest)
    claimed_manifest_hash = semantic_manifest.pop("manifest_hash", None)
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA
        or claimed_manifest_hash != canonical_hash(semantic_manifest)
        or manifest.get("performance_claimed") is not False
        or manifest.get("timing_admitted") is not True
    ):
        raise ValueError("FSG4/B4-A manifest differs")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or dict(files) != _all_files(root):
        raise ValueError("FSG4/B4-A file inventory differs")
    rebuilt = _report(root)
    report = _load_json(root / "report.json")
    if report != rebuilt or report.get("report_hash") != manifest.get("report_hash"):
        raise ValueError("FSG4/B4-A semantic replay differs")
    result = {
        "status": "replay-passed",
        "source_git_head": manifest["source_git_head"],
        "pair_count": report["pair_count"],
        "worker_count": report["worker_count"],
        "maximum_export_absolute_difference": report[
            "maximum_export_absolute_difference"
        ],
        "manifest_hash": claimed_manifest_hash,
        "timing_admitted": True,
        "performance_claimed": False,
    }
    print(_canonical_json(result), flush=True)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--artifact-dir", type=Path, required=True)
    generate.add_argument("--benchmark-root", type=Path, required=True)
    generate.add_argument("--abcrown-root", type=Path, required=True)
    generate.add_argument("--abcrown-python", type=Path, required=True)
    generate.add_argument("--model", type=Path, required=True)
    generate.add_argument("--property", type=Path, required=True)
    replay = commands.add_parser("replay")
    replay.add_argument("--artifact-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "generate":
        _generate(args)
    else:
        _replay(args.artifact_dir)


if __name__ == "__main__":
    main()
