#!/usr/bin/env python3
"""Generate, resume, or replay the 24-process B4-A formal timing artifact."""

# pylint: disable=wrong-import-position,protected-access,duplicate-code
# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=too-many-arguments,too-many-lines,too-many-boolean-expressions
# pylint: disable=missing-function-docstring

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence, cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from boundflow.runtime.fsg3_same_solver_timing import (
    _semantic_pair_failures,
    canonical_hash,
)
from boundflow.runtime.fsg4_b3_same_solver_timing import (
    FSG4B3TimingRun,
    fsg4_b3_timing_run_from_dict,
)
from scripts import run_fsg3_same_solver_experiment as base_experiment
from scripts import run_fsg4_b4a_correctness_pairs as correctness
from scripts import run_fsg4_b4a_same_solver_worker as worker

ARTIFACT_SCHEMA = "boundflow.fsg4-b4a-formal-timing-artifact/v1"
PROTOCOL_SCHEMA = "boundflow.fsg4-b4a-formal-timing-protocol/v1"
CORRECTNESS_ARTIFACT = (
    REPOSITORY_ROOT / "artifacts/fsg4-b4a-five-fresh/resnet2b-prop0-v1"
)
SEQUENCE = (
    (("B3", "control"), ("B3", "profile"), ("B4-A", "control"), ("B4-A", "profile")),
    (("B4-A", "profile"), ("B3", "control"), ("B3", "profile"), ("B4-A", "control")),
    (("B4-A", "control"), ("B4-A", "profile"), ("B3", "control"), ("B3", "profile")),
    (("B3", "profile"), ("B4-A", "control"), ("B4-A", "profile"), ("B3", "control")),
    (("B3", "control"), ("B4-A", "profile"), ("B4-A", "control"), ("B3", "profile")),
    (("B4-A", "profile"), ("B3", "profile"), ("B3", "control"), ("B4-A", "control")),
)
CODE_PATHS = tuple(
    dict.fromkeys(
        (
            *worker.CODE_PATHS,
            "scripts/run_fsg4_b4a_correctness_pairs.py",
            "scripts/run_fsg4_b4a_formal_timing.py",
            "scripts/probe_fsg4_b4a_formal_timing_tamper.py",
        )
    )
)
METRICS = (
    "core_wall_ns",
    "query_wall_ns",
    "core_gpu_ns",
    "query_gpu_ns",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
)
CORE_GATE = 1.03
QUERY_WORST_GATE = 0.98
ATOL = 2e-4
RTOL = 2e-4
B4A_PREFLIGHT_TEMPERATURE_LIMIT_C = 45
WORKER_TIMEOUT_SECONDS = base_experiment.WORKER_SUBPROCESS_TIMEOUT_SECONDS
FORMAL_PREFLIGHT_CONTRACT = {
    **base_experiment.FORMAL_PREFLIGHT_CONTRACT,
    "temperature_limit_celsius": B4A_PREFLIGHT_TEMPERATURE_LIMIT_C,
    "software_thermal_signal_must_be_inactive": True,
}


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


def _write_jsonl(path: Path, rows: Sequence[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(_canonical_json(row) + "\n" for row in rows), encoding="utf-8"
    )


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"FSG4/B4-A formal JSON root differs: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"FSG4/B4-A formal JSONL row differs: {path}")
        rows.append(value)
    return rows


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", *args),
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def _code_revision() -> dict[str, str]:
    return {path: _file_sha256(REPOSITORY_ROOT / path) for path in CODE_PATHS}


def _historical_revision(source: str, paths: Sequence[str]) -> dict[str, str]:
    if _git(REPOSITORY_ROOT, "rev-parse", "HEAD") == source:
        return {path: _file_sha256(REPOSITORY_ROOT / path) for path in paths}
    return {
        path: hashlib.sha256(
            subprocess.run(
                ("git", "show", f"{source}:{path}"),
                cwd=REPOSITORY_ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ).stdout
        ).hexdigest()
        for path in paths
    }


def _historical_code_revision(source: str) -> dict[str, str]:
    return _historical_revision(source, CODE_PATHS)


def _sequence_payload() -> list[dict[str, object]]:
    return [
        {
            "block_index": block,
            "positions": [
                {"configuration": configuration, "mode": mode}
                for configuration, mode in positions
            ],
        }
        for block, positions in enumerate(SEQUENCE)
    ]


def _correctness_identity() -> dict[str, object]:
    manifest = _load_json(CORRECTNESS_ARTIFACT / "manifest.json")
    semantic = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    if (
        manifest.get("manifest_hash") != canonical_hash(semantic)
        or manifest.get("timing_admitted") is not True
        or manifest.get("performance_claimed") is not False
    ):
        raise ValueError("FSG4/B4-A correctness admission differs")
    return {
        "manifest_file_sha256": _file_sha256(CORRECTNESS_ARTIFACT / "manifest.json"),
        "manifest_hash": manifest["manifest_hash"],
        "source_git_head": manifest["source_git_head"],
    }


def _protocol(args: argparse.Namespace) -> dict[str, object]:
    benchmark = args.benchmark_root.resolve()
    abcrown = args.abcrown_root.resolve()
    model = args.model.resolve()
    property_path = args.property.resolve()
    value: dict[str, object] = {
        "schema_version": PROTOCOL_SCHEMA,
        "source_git_head": _git(REPOSITORY_ROOT, "rev-parse", "HEAD"),
        "code_revision": _code_revision(),
        "sequence": _sequence_payload(),
        "worker_count": 24,
        "block_count": 6,
        "formal_preflight_contract": FORMAL_PREFLIGHT_CONTRACT,
        "correctness_identity": _correctness_identity(),
        "abcrown_commit": _git(abcrown, "rev-parse", "HEAD"),
        "auto_lirpa_commit": _git(abcrown / "auto_LiRPA", "rev-parse", "HEAD"),
        "vnncomp_commit": _git(benchmark, "rev-parse", "HEAD"),
        "model_name": model.name,
        "model_sha256": _file_sha256(model),
        "property_name": property_path.name,
        "property_sha256": _file_sha256(property_path),
        "python_name": args.abcrown_python.name,
        "headline_mode": "control-only",
        "profile_role": "attribution-only",
        "core_wall_geomean_gate": CORE_GATE,
        "query_wall_worst_pair_gate": QUERY_WORST_GATE,
        "resume_policy": "accept-complete-source-bound-worker-only",
        "performance_claimed": False,
    }
    value["protocol_hash"] = canonical_hash(value)
    return value


def _validate_protocol(value: Mapping[str, Any]) -> None:
    payload = dict(value)
    claimed = payload.pop("protocol_hash", None)
    if (
        claimed != canonical_hash(payload)
        or value.get("schema_version") != PROTOCOL_SCHEMA
        or value.get("sequence") != _sequence_payload()
        or value.get("worker_count") != 24
        or value.get("block_count") != 6
        or value.get("formal_preflight_contract") != FORMAL_PREFLIGHT_CONTRACT
        or value.get("correctness_identity") != _correctness_identity()
        or value.get("headline_mode") != "control-only"
        or value.get("profile_role") != "attribution-only"
        or value.get("core_wall_geomean_gate") != CORE_GATE
        or value.get("query_wall_worst_pair_gate") != QUERY_WORST_GATE
        or value.get("performance_claimed") is not False
    ):
        raise ValueError("FSG4/B4-A formal protocol differs")
    source = value.get("source_git_head")
    revision = value.get("code_revision")
    if not isinstance(source, str) or not isinstance(revision, Mapping):
        raise TypeError("FSG4/B4-A formal source identity differs")
    if dict(revision) != _historical_code_revision(source):
        raise ValueError("FSG4/B4-A formal code revision differs")


def _run_id(block: int, position: int, configuration: str, mode: str) -> str:
    return f"block-{block:02d}-pos-{position:02d}-{configuration}-{mode}"


def _worker_command(
    *,
    args: argparse.Namespace,
    result: Path,
    block: int,
    position: int,
    configuration: str,
    mode: str,
) -> tuple[str, ...]:
    return (
        str(args.abcrown_python.expanduser().absolute()),
        str(REPOSITORY_ROOT / "scripts/run_fsg4_b4a_same_solver_worker.py"),
        "--configuration",
        configuration,
        "--mode",
        mode,
        "--run-id",
        _run_id(block, position, configuration, mode),
        "--block-index",
        str(block),
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


def _command_projection(
    block: int, position: int, configuration: str, mode: str
) -> dict[str, object]:
    return {
        "worker": "scripts/run_fsg4_b4a_same_solver_worker.py",
        "block_index": block,
        "sequence_position": position,
        "configuration": configuration,
        "mode": mode,
        "run_id": _run_id(block, position, configuration, mode),
        "benchmark_root": "$VNNCOMP_ROOT",
        "abcrown_root": "$ABCROWN_ROOT",
        "model": "$MODEL",
        "property": "$PROPERTY",
        "result": f"workers/run_{block * 4 + position:02d}.json",
    }


def _sanitize(value: str, *, benchmark: Path, abcrown: Path, python: Path) -> str:
    replacements = sorted(
        (
            (str(REPOSITORY_ROOT), "$BOUNDFLOW_ROOT"),
            (str(benchmark), "$VNNCOMP_ROOT"),
            (str(abcrown), "$ABCROWN_ROOT"),
            (str(python), f"$PYTHON/{python.name}"),
        ),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    for source, replacement in replacements:
        value = value.replace(source, replacement)
    if "/home/" in value or "/tmp/" in value:
        raise ValueError("FSG4/B4-A formal log contains a host-local path")
    return value


def _normalize_preflight(value: Mapping[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(value, allow_nan=False))
    if not isinstance(normalized, dict) or not isinstance(
        normalized.get("samples"), list
    ):
        raise TypeError("FSG4/B4-A formal preflight differs")
    for sample in normalized["samples"]:
        if not isinstance(sample, dict) or not isinstance(
            sample.get("compute_processes"), list
        ):
            raise TypeError("FSG4/B4-A formal preflight sample differs")
        for process in sample["compute_processes"]:
            if isinstance(process, dict) and isinstance(process.get("name"), str):
                process["name"] = Path(process["name"]).name
    _validate_formal_preflight(normalized)
    return normalized


def _validate_formal_preflight(value: Mapping[str, Any]) -> None:
    base_value = dict(value)
    base_value["temperature_limit_celsius"] = (
        base_experiment.PREFLIGHT_TEMPERATURE_LIMIT_C
    )
    base_experiment._validate_formal_preflight(base_value)
    samples = value.get("samples")
    if (
        value.get("temperature_limit_celsius") != B4A_PREFLIGHT_TEMPERATURE_LIMIT_C
        or not isinstance(samples, list)
        or not samples
        or not isinstance(samples[-1], Mapping)
    ):
        raise ValueError("FSG4/B4-A strict preflight payload differs")
    last = cast(Mapping[str, Any], samples[-1])
    snapshot = last.get("gpu_snapshot")
    if (
        not isinstance(snapshot, Mapping)
        or int(last.get("temperature_celsius", -1)) > B4A_PREFLIGHT_TEMPERATURE_LIMIT_C
        or snapshot.get("sw_thermal_slowdown") != "Not Active"
    ):
        raise ValueError("FSG4/B4-A strict preflight admission differs")


def _wait_for_formal_environment() -> dict[str, object]:
    started = time.monotonic_ns()
    samples: list[dict[str, object]] = []
    while True:
        observed = base_experiment._wait_for_formal_environment()
        observed_samples = observed.get("samples")
        if not isinstance(observed_samples, list) or not observed_samples:
            raise TypeError("FSG4/B4-A strict preflight samples differ")
        samples.extend(cast(list[dict[str, object]], observed_samples))
        last = samples[-1]
        snapshot = last.get("gpu_snapshot")
        ready = (
            isinstance(snapshot, Mapping)
            and int(str(last.get("temperature_celsius", -1)))
            <= B4A_PREFLIGHT_TEMPERATURE_LIMIT_C
            and snapshot.get("sw_thermal_slowdown") == "Not Active"
        )
        if ready:
            value: dict[str, object] = {
                "temperature_limit_celsius": B4A_PREFLIGHT_TEMPERATURE_LIMIT_C,
                "poll_seconds": base_experiment.PREFLIGHT_POLL_SECONDS,
                "timeout_seconds": base_experiment.PREFLIGHT_TIMEOUT_SECONDS,
                "sample_count": len(samples),
                "wait_ns": time.monotonic_ns() - started,
                "samples": samples,
                "admitted": True,
            }
            _validate_formal_preflight(value)
            return value
        if (
            time.monotonic_ns() - started
            > base_experiment.PREFLIGHT_TIMEOUT_SECONDS * 1_000_000_000
        ):
            raise TimeoutError("FSG4/B4-A strict preflight did not reach cool idle")
        print(
            _canonical_json(
                {
                    "preflight": "waiting-for-b4a-cool-idle",
                    "temperature_celsius": last.get("temperature_celsius"),
                    "software_thermal_slowdown": (
                        snapshot.get("sw_thermal_slowdown")
                        if isinstance(snapshot, Mapping)
                        else None
                    ),
                    "sample_count": len(samples),
                }
            ),
            flush=True,
        )
        time.sleep(base_experiment.PREFLIGHT_POLL_SECONDS)


def _validate_worker(
    value: Mapping[str, Any],
    *,
    block: int,
    position: int,
    configuration: str,
    mode: str,
    source_git_head: str,
) -> FSG4B3TimingRun:
    run_value = value.get("run")
    activation = value.get("activation")
    protocol = value.get("protocol")
    diagnostics = value.get("diagnostics")
    if (
        value.get("schema_version") != worker.WORKER_SCHEMA
        or value.get("configuration") != configuration
        or value.get("mode") != mode
        or value.get("performance_claimed") is not False
        or not isinstance(run_value, Mapping)
        or not isinstance(activation, Mapping)
        or not isinstance(protocol, Mapping)
        or not isinstance(diagnostics, Mapping)
    ):
        raise ValueError("FSG4/B4-A formal worker envelope differs")
    run = fsg4_b3_timing_run_from_dict(cast(Mapping[str, object], run_value))
    if (
        run.block_index != block
        or run.sequence_position != position
        or run.mode.value != mode
        or run.configuration.value != "B3"
        or not run.environment.admitted
        or run.execution.provider_core_call_count != 0
        or run.execution.provider_compute_bounds_call_count != 0
        or run.execution.provider_update_bounds_call_count != 0
        or run.execution.fallback_dispatch_count != 0
    ):
        raise ValueError("FSG4/B4-A formal worker run differs")
    expected_handoff = 0 if configuration == "B3" else 1
    expected_rerun = 1 if configuration == "B3" else 0
    protocol_payload = dict(protocol)
    protocol_hash = protocol_payload.pop("protocol_hash", None)
    worker_source = protocol.get("source_git_head")
    worker_revision = protocol.get("code_revision")
    activation_payload = dict(activation)
    activation_hash = activation_payload.pop("activation_hash", None)
    profile_counts = activation.get("profile_counter_counts")
    expected_profile = worker.EXPECTED_B3C_FIXED_COUNTERS
    if mode == "profile":
        if (
            not isinstance(profile_counts, Mapping)
            or activation.get("profile_counter_counts_hash")
            != canonical_hash(dict(sorted(profile_counts.items())))
            or any(
                profile_counts.get(name) != value
                for name, value in expected_profile.items()
            )
            or activation.get("forward_trace_build_count")
            != expected_profile["forward_trace_build_count"]
        ):
            raise ValueError("FSG4/B4-A formal physical profile counter differs")
    elif any(
        activation.get(name) is not None
        for name in (
            "profile_counter_counts",
            "profile_counter_counts_hash",
            "forward_trace_build_count",
        )
    ):
        raise ValueError("FSG4/B4-A formal control profile receipt differs")
    if (
        activation.get("terminal_lower_adjoint_handoff_count") != expected_handoff
        or activation.get("terminal_export_crown_rerun_count") != expected_rerun
        or activation.get("provider_callback_count") != 0
        or activation.get("fallback_dispatch_count") != 0
        or (configuration == "B4-A" and activation.get("lineage_count") != 6)
        or activation_hash != canonical_hash(activation_payload)
        or protocol_hash != canonical_hash(protocol_payload)
        or not isinstance(worker_source, str)
        or not isinstance(worker_revision, Mapping)
        or worker_source != source_git_head
        or dict(worker_revision)
        != _historical_revision(worker_source, worker.CODE_PATHS)
        or protocol.get("base_b3_protocol_identity") != run.protocol_identity
        or protocol.get("configuration") != configuration
        or protocol.get("feature")
        != (
            "b3-terminal-export-crown-rerun"
            if configuration == "B3"
            else "b4a-terminal-lower-adjoint-handoff"
        )
        or protocol.get("same_solver") is not True
        or protocol.get("performance_claimed") is not False
        or canonical_hash(diagnostics.get("runtime_environment"))
        != run.environment.runtime_identity
        or diagnostics.get("terminal_export_audit_excluded_from_timing") is not True
    ):
        raise ValueError("FSG4/B4-A formal activation differs")
    return run


def _run_worker(
    *,
    artifact: Path,
    index: int,
    command: tuple[str, ...],
    projection: Mapping[str, object],
    preflight: Mapping[str, Any],
    benchmark: Path,
    abcrown: Path,
    python: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = artifact / f"workers/run_{index:02d}.json"
    stdout = artifact / f"logs/run_{index:02d}.stdout.txt"
    stderr = artifact / f"logs/run_{index:02d}.stderr.txt"
    result.parent.mkdir(parents=True, exist_ok=True)
    stdout.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT) + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    before = base_experiment._host_snapshot()
    started = time.monotonic_ns()
    try:
        completed = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=WORKER_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        ended = time.monotonic_ns()

        def timeout_text(value: bytes | str | None) -> str:
            if value is None:
                return ""
            return value.decode(errors="replace") if isinstance(value, bytes) else value

        stdout.write_text(
            _sanitize(
                timeout_text(error.stdout),
                benchmark=benchmark,
                abcrown=abcrown,
                python=python,
            ),
            encoding="utf-8",
        )
        stderr.write_text(
            _sanitize(
                timeout_text(error.stderr),
                benchmark=benchmark,
                abcrown=abcrown,
                python=python,
            ),
            encoding="utf-8",
        )
        _write_json(
            artifact / "failed_worker.json",
            {
                "index": index,
                "command": dict(projection),
                "returncode": None,
                "timed_out": True,
                "timeout_seconds": WORKER_TIMEOUT_SECONDS,
                "started_monotonic_ns": started,
                "ended_monotonic_ns": ended,
                "duration_ns": ended - started,
                "host_before": before,
                "host_after": base_experiment._host_snapshot(),
                "formal_preflight": dict(preflight),
                "result_file": str(result.relative_to(artifact)),
                "stdout_file": str(stdout.relative_to(artifact)),
                "stderr_file": str(stderr.relative_to(artifact)),
                "performance_claimed": False,
            },
        )
        raise RuntimeError(f"FSG4/B4-A formal worker {index} timed out") from error
    ended = time.monotonic_ns()
    after = base_experiment._host_snapshot()
    stdout.write_text(
        _sanitize(
            completed.stdout, benchmark=benchmark, abcrown=abcrown, python=python
        ),
        encoding="utf-8",
    )
    stderr.write_text(
        _sanitize(
            completed.stderr, benchmark=benchmark, abcrown=abcrown, python=python
        ),
        encoding="utf-8",
    )
    metadata: dict[str, Any] = {
        "index": index,
        "command": dict(projection),
        "returncode": completed.returncode,
        "timed_out": False,
        "timeout_seconds": WORKER_TIMEOUT_SECONDS,
        "started_monotonic_ns": started,
        "ended_monotonic_ns": ended,
        "duration_ns": ended - started,
        "host_before": before,
        "host_after": after,
        "formal_preflight": dict(preflight),
        "result_file": str(result.relative_to(artifact)),
        "stdout_file": str(stdout.relative_to(artifact)),
        "stderr_file": str(stderr.relative_to(artifact)),
        "performance_claimed": False,
    }
    if completed.returncode != 0 or not result.is_file():
        _write_json(artifact / "failed_worker.json", metadata)
        raise RuntimeError(f"FSG4/B4-A formal worker {index} failed")
    return _load_json(result), metadata


def _load_complete(
    *,
    artifact: Path,
    index: int,
    block: int,
    position: int,
    configuration: str,
    mode: str,
    source_git_head: str,
) -> tuple[dict[str, Any], dict[str, Any], FSG4B3TimingRun] | None:
    worker_path = artifact / f"workers/run_{index:02d}.json"
    metadata_path = artifact / f"metadata/run_{index:02d}.json"
    stdout = artifact / f"logs/run_{index:02d}.stdout.txt"
    stderr = artifact / f"logs/run_{index:02d}.stderr.txt"
    exists = [path.exists() for path in (worker_path, metadata_path, stdout, stderr)]
    if not any(exists):
        return None
    if not all(exists):
        raise ValueError(f"FSG4/B4-A partial worker cannot resume: {index}")
    envelope = _load_json(worker_path)
    metadata = _load_json(metadata_path)
    run = _validate_worker(
        envelope,
        block=block,
        position=position,
        configuration=configuration,
        mode=mode,
        source_git_head=source_git_head,
    )
    if (
        metadata.get("index") != index
        or metadata.get("returncode") != 0
        or metadata.get("timed_out") is not False
        or metadata.get("command")
        != _command_projection(block, position, configuration, mode)
        or not isinstance(metadata.get("formal_preflight"), Mapping)
    ):
        raise ValueError(f"FSG4/B4-A resume metadata differs: {index}")
    _validate_formal_preflight(cast(Mapping[str, Any], metadata["formal_preflight"]))
    return envelope, metadata, run


def _geomean(values: Sequence[float]) -> float:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("FSG4/B4-A ratio inventory differs")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _pair_rows(
    envelopes: Sequence[Mapping[str, Any]], runs: Sequence[FSG4B3TimingRun]
) -> list[dict[str, object]]:
    indexed: dict[tuple[int, str, str], tuple[Mapping[str, Any], FSG4B3TimingRun]] = {}
    for envelope, run in zip(envelopes, runs):
        indexed[
            (run.block_index, str(envelope["configuration"]), str(envelope["mode"]))
        ] = (
            envelope,
            run,
        )
    rows: list[dict[str, object]] = []
    for block in range(6):
        b3_envelope, b3 = indexed[(block, "B3", "control")]
        b4a_envelope, b4a = indexed[(block, "B4-A", "control")]
        failures = _semantic_pair_failures(
            b3.semantics, b4a.semantics, label=f"b4a-formal-{block}"
        )
        b3_runtime = cast(Mapping[str, Any], b3_envelope["diagnostics"])[
            "runtime_environment"
        ]
        b4a_runtime = cast(Mapping[str, Any], b4a_envelope["diagnostics"])[
            "runtime_environment"
        ]
        export_pair = correctness._export_pair(
            correctness._one_payload(b3_envelope),
            correctness._one_payload(b4a_envelope),
        )
        if (
            failures
            or b3.source_identity != b4a.source_identity
            or b3.environment.gpu_uuid != b4a.environment.gpu_uuid
            or b3_runtime != b4a_runtime
        ):
            raise ValueError("FSG4/B4-A formal pair differs: " + ",".join(failures))
        row: dict[str, object] = {
            "block_index": block,
            "b3_run_id": b3.run_id,
            "b4a_run_id": b4a.run_id,
            "b3_run_hash": b3.stable_hash(),
            "b4a_run_hash": b4a.stable_hash(),
            "ratios": {
                name: getattr(b3.metrics, name) / getattr(b4a.metrics, name)
                for name in METRICS
            },
            "semantic_failures": [],
            "export_pair": export_pair,
            "environment_admitted": True,
            "performance_claimed": False,
        }
        row["pair_hash"] = canonical_hash(row)
        rows.append(row)
    return rows


def _profile_rows(
    envelopes: Sequence[Mapping[str, Any]], runs: Sequence[FSG4B3TimingRun]
) -> list[dict[str, object]]:
    return [
        {
            "configuration": envelope["configuration"],
            "run_id": run.run_id,
            "block_index": run.block_index,
            "closure_error": run.profile_closure_error,
            "residual_share": run.profile_residual_share,
            "spans": [span.to_dict() for span in run.profile_spans],
            "performance_claimed": False,
        }
        for envelope, run in zip(envelopes, runs)
        if envelope["mode"] == "profile"
    ]


def _summary(
    pair_rows: Sequence[Mapping[str, Any]],
    profile_rows: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    metric_ratios = {
        name: [float(cast(Mapping[str, Any], row["ratios"])[name]) for row in pair_rows]
        for name in METRICS
    }
    geomeans = {name: _geomean(values) for name, values in metric_ratios.items()}
    core_passed = geomeans["core_wall_ns"] >= CORE_GATE
    query_worst = min(metric_ratios["query_wall_ns"])
    query_passed = query_worst >= QUERY_WORST_GATE
    profile_closed = len(profile_rows) == 12 and all(
        isinstance(row.get("closure_error"), (int, float))
        and float(cast(float, row["closure_error"])) <= 0.01
        and isinstance(row.get("residual_share"), (int, float))
        and float(cast(float, row["residual_share"])) <= 0.03
        for row in profile_rows
    )
    admitted = core_passed and query_passed and profile_closed
    value: dict[str, object] = {
        "status": (
            "validated-b4a-performance-candidate"
            if admitted
            else "validated-no-go-b4a-performance"
        ),
        "run_count": 24,
        "control_pair_count": 6,
        "profile_run_count": 12,
        "correctness_passed": True,
        "environment_passed": True,
        "measurement_auditable": True,
        "profile_closed": profile_closed,
        "metric_pair_ratios": metric_ratios,
        "metric_geomeans": geomeans,
        "core_wall_geomean_gate": CORE_GATE,
        "core_wall_geomean_passed": core_passed,
        "query_wall_worst_pair": query_worst,
        "query_wall_worst_pair_gate": QUERY_WORST_GATE,
        "query_wall_worst_pair_passed": query_passed,
        "performance_candidate_admitted": admitted,
        "kernel_launch_delta": "DEFERRED-TO-B4-A-KERNEL-DELTA",
        "performance_claimed": False,
    }
    value["summary_hash"] = canonical_hash(value)
    return value


def _derived(
    envelopes: Sequence[Mapping[str, Any]], runs: Sequence[FSG4B3TimingRun]
) -> dict[str, object]:
    if (
        len(envelopes) != 24
        or len(runs) != 24
        or len({run.run_id for run in runs}) != 24
        or len({run.source_identity for run in runs}) != 1
        or len({run.protocol_identity for run in runs}) != 1
        or len(
            {
                (
                    run.environment.gpu_uuid,
                    run.environment.gpu_name,
                    run.environment.runtime_identity,
                )
                for run in runs
            }
        )
        != 1
    ):
        raise ValueError("FSG4/B4-A formal run inventory differs")
    pairs = _pair_rows(envelopes, runs)
    profiles = _profile_rows(envelopes, runs)
    return {
        "paired_runs.jsonl": pairs,
        "profile_runs.jsonl": profiles,
        "summary.json": _summary(pairs, profiles),
    }


def _readme() -> str:
    return (
        "# FSG4/B4-A formal timing\n\n"
        "Twenty-four fresh B3/B4-A control/profile workers. Control pairs determine "
        "the preregistered core/query gates; profile rows are attribution only. "
        "performance_claimed remains false pending external audit.\n"
    )


def _replay_result(summary: Mapping[str, Any]) -> dict[str, object]:
    return {
        "status": summary["status"],
        "run_count": summary["run_count"],
        "summary_hash": summary["summary_hash"],
        "core_wall_geomean": cast(Mapping[str, Any], summary["metric_geomeans"])[
            "core_wall_ns"
        ],
        "query_wall_worst_pair": summary["query_wall_worst_pair"],
        "performance_candidate_admitted": summary["performance_candidate_admitted"],
        "performance_claimed": False,
    }


def _generate(args: argparse.Namespace) -> dict[str, object]:
    if _git(REPOSITORY_ROOT, "status", "--porcelain=v1", "--", *CODE_PATHS):
        raise ValueError("FSG4/B4-A formal code paths must be clean")
    artifact = args.artifact_dir.resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    expected_protocol = _protocol(args)
    protocol_path = artifact / "protocol.json"
    if protocol_path.exists():
        observed = _load_json(protocol_path)
        _validate_protocol(observed)
        if observed != expected_protocol:
            raise ValueError("FSG4/B4-A formal resume protocol differs")
    elif list(artifact.iterdir()):
        raise FileExistsError("FSG4/B4-A formal artifact lacks protocol")
    else:
        _write_json(protocol_path, expected_protocol)
    benchmark = args.benchmark_root.resolve()
    abcrown = args.abcrown_root.resolve()
    python = args.abcrown_python.expanduser().absolute()
    envelopes: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    runs: list[FSG4B3TimingRun] = []
    for block, positions in enumerate(SEQUENCE):
        for position, (configuration, mode) in enumerate(positions):
            index = block * 4 + position
            loaded = _load_complete(
                artifact=artifact,
                index=index,
                block=block,
                position=position,
                configuration=configuration,
                mode=mode,
                source_git_head=cast(str, expected_protocol["source_git_head"]),
            )
            if loaded is None:
                preflight = _normalize_preflight(_wait_for_formal_environment())
                envelope, metadata = _run_worker(
                    artifact=artifact,
                    index=index,
                    command=_worker_command(
                        args=args,
                        result=artifact / f"workers/run_{index:02d}.json",
                        block=block,
                        position=position,
                        configuration=configuration,
                        mode=mode,
                    ),
                    projection=_command_projection(
                        block, position, configuration, mode
                    ),
                    preflight=preflight,
                    benchmark=benchmark,
                    abcrown=abcrown,
                    python=python,
                )
                run = _validate_worker(
                    envelope,
                    block=block,
                    position=position,
                    configuration=configuration,
                    mode=mode,
                    source_git_head=cast(str, expected_protocol["source_git_head"]),
                )
                _write_json(artifact / f"metadata/run_{index:02d}.json", metadata)
                print(
                    _canonical_json({"completed_worker": index, "run_id": run.run_id}),
                    flush=True,
                )
            else:
                envelope, metadata, run = loaded
                print(
                    _canonical_json({"resumed_worker": index, "run_id": run.run_id}),
                    flush=True,
                )
            envelopes.append(envelope)
            metadata_rows.append(metadata)
            runs.append(run)
            _write_jsonl(
                artifact / "worker_runs.jsonl", [item.to_dict() for item in runs]
            )
            _write_jsonl(artifact / "run_metadata.jsonl", metadata_rows)
    derived = _derived(envelopes, runs)
    for name, value in derived.items():
        if name.endswith(".jsonl"):
            _write_jsonl(artifact / name, cast(Sequence[object], value))
        else:
            _write_json(artifact / name, value)
    summary = cast(Mapping[str, Any], derived["summary.json"])
    replay = _replay_result(summary)
    (artifact / "replay_stdout.txt").write_text(
        _canonical_json(replay) + "\n", encoding="utf-8"
    )
    (artifact / "README.md").write_text(_readme(), encoding="utf-8")
    files = sorted(
        str(path.relative_to(artifact))
        for path in artifact.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    manifest: dict[str, object] = {
        "schema_version": ARTIFACT_SCHEMA,
        "source_git_head": expected_protocol["source_git_head"],
        "code_revision": expected_protocol["code_revision"],
        "files": {name: _file_sha256(artifact / name) for name in files},
        "protocol_hash": expected_protocol["protocol_hash"],
        "summary_hash": summary["summary_hash"],
        "status": summary["status"],
        "worker_count": 24,
        "performance_claimed": False,
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    _write_json(artifact / "manifest.json", manifest)
    return replay


def _verify(artifact: Path) -> tuple[dict[str, Any], dict[str, object]]:
    manifest = _load_json(artifact / "manifest.json")
    semantic = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    if (
        manifest.get("schema_version") != ARTIFACT_SCHEMA
        or manifest.get("manifest_hash") != canonical_hash(semantic)
        or manifest.get("worker_count") != 24
        or manifest.get("performance_claimed") is not False
    ):
        raise ValueError("FSG4/B4-A formal manifest differs")
    protocol = _load_json(artifact / "protocol.json")
    _validate_protocol(protocol)
    if (
        manifest.get("source_git_head") != protocol["source_git_head"]
        or manifest.get("code_revision") != protocol["code_revision"]
        or manifest.get("protocol_hash") != protocol["protocol_hash"]
    ):
        raise ValueError("FSG4/B4-A formal protocol binding differs")
    files = manifest.get("files")
    observed = {
        str(path.relative_to(artifact))
        for path in artifact.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if not isinstance(files, Mapping) or set(files) != observed:
        raise ValueError("FSG4/B4-A formal file inventory differs")
    for name, digest in files.items():
        if not isinstance(name, str) or digest != _file_sha256(artifact / name):
            raise ValueError(f"FSG4/B4-A formal digest differs: {name}")
    envelopes: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    runs: list[FSG4B3TimingRun] = []
    for block, positions in enumerate(SEQUENCE):
        for position, (configuration, mode) in enumerate(positions):
            index = block * 4 + position
            loaded = _load_complete(
                artifact=artifact,
                index=index,
                block=block,
                position=position,
                configuration=configuration,
                mode=mode,
                source_git_head=cast(str, protocol["source_git_head"]),
            )
            if loaded is None:
                raise ValueError(f"FSG4/B4-A formal worker missing: {index}")
            envelope, metadata, run = loaded
            envelopes.append(envelope)
            metadata_rows.append(metadata)
            runs.append(run)
    if _load_jsonl(artifact / "worker_runs.jsonl") != [run.to_dict() for run in runs]:
        raise ValueError("FSG4/B4-A formal worker aggregate differs")
    if _load_jsonl(artifact / "run_metadata.jsonl") != metadata_rows:
        raise ValueError("FSG4/B4-A formal metadata aggregate differs")
    derived = _derived(envelopes, runs)
    for name, expected in derived.items():
        observed_value: object = (
            _load_jsonl(artifact / name)
            if name.endswith(".jsonl")
            else _load_json(artifact / name)
        )
        if observed_value != expected:
            raise ValueError(f"FSG4/B4-A formal derived replay differs: {name}")
    summary = cast(dict[str, Any], derived["summary.json"])
    if (
        manifest.get("summary_hash") != summary["summary_hash"]
        or manifest.get("status") != summary["status"]
    ):
        raise ValueError("FSG4/B4-A formal summary binding differs")
    replay = _replay_result(summary)
    if (artifact / "replay_stdout.txt").read_text(encoding="utf-8") != _canonical_json(
        replay
    ) + "\n":
        raise ValueError("FSG4/B4-A formal replay stdout differs")
    if (artifact / "README.md").read_text(encoding="utf-8") != _readme():
        raise ValueError("FSG4/B4-A formal README differs")
    return summary, replay


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
        result = _generate(args)
    else:
        _summary_value, result = _verify(args.artifact_dir.resolve())
    print(_canonical_json(result))


if __name__ == "__main__":
    main()
