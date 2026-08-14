#!/usr/bin/env python3
"""Generate, resume, or replay the 36-process FSG4 B0/B2/B3 artifact."""

# pylint: disable=wrong-import-position,protected-access,duplicate-code
# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=too-many-arguments,too-many-lines,too-many-boolean-expressions

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence, cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from boundflow.runtime.fsg3_same_solver_timing import canonical_hash, FSG3Mode
from boundflow.runtime.fsg4_b3_same_solver_timing import (
    derive_fsg4_b3_timing_evidence,
    expected_fsg4_b3_sequence,
    FSG4B3TimingConfiguration,
    FSG4B3TimingRun,
    fsg4_b3_timing_run_from_dict,
)
from scripts import run_fsg3_same_solver_experiment as base_experiment
from scripts import run_fsg4_b3_counter_diagnostic as diagnostic
from scripts import run_fsg4_b3_same_solver_timing as worker

ARTIFACT_SCHEMA = "boundflow.fsg4-b3-same-solver-artifact/v1"
PROTOCOL_SCHEMA = "boundflow.fsg4-b3-same-solver-protocol/v1"
CODE_PATHS = tuple(
    dict.fromkeys(
        (
            "boundflow/runtime/fsg3_same_solver_timing.py",
            "boundflow/runtime/fsg4_b3_explicit_counters.py",
            "boundflow/runtime/fsg4_b3_same_solver_timing.py",
            "scripts/run_fsg3_same_solver_timing.py",
            "scripts/run_fsg3_same_solver_experiment.py",
            "scripts/run_fsg4_b3_counter_diagnostic.py",
            "scripts/run_fsg4_b3_same_solver_timing.py",
            "scripts/run_fsg4_b3_same_solver_experiment.py",
            "scripts/probe_fsg4_b3_same_solver_artifact_tamper.py",
            *diagnostic.B3C_CODE_PATHS,
        )
    )
)
METRIC_NAMES = (
    "cold_total_ns",
    "query_wall_ns",
    "core_wall_ns",
    "query_gpu_ns",
    "core_gpu_ns",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
)
WORKER_SUBPROCESS_TIMEOUT_SECONDS = base_experiment.WORKER_SUBPROCESS_TIMEOUT_SECONDS
FORMAL_PREFLIGHT_CONTRACT = dict(base_experiment.FORMAL_PREFLIGHT_CONTRACT)


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
        raise TypeError(f"FSG4/B3 JSON root differs: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"FSG4/B3 JSONL row differs: {path}")
        rows.append(value)
    return rows


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(root: Path, *args: str) -> str:
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


def _code_paths_clean() -> bool:
    return not _git_value(
        REPOSITORY_ROOT, "status", "--porcelain=v1", "--", *CODE_PATHS
    )


def _historical_code_revision(source: str) -> dict[str, str]:
    if _git_value(REPOSITORY_ROOT, "rev-parse", "HEAD") == source:
        return _code_revision()
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
        for path in CODE_PATHS
    }


def _expected_sequence_payload() -> list[list[object]]:
    return [
        [block, position, configuration.value, mode.value]
        for block, position, configuration, mode in expected_fsg4_b3_sequence()
    ]


def _protocol(args: argparse.Namespace) -> dict[str, object]:
    benchmark = args.benchmark_root.resolve()
    abcrown = args.abcrown_root.resolve()
    model = args.model.resolve()
    property_path = args.property.resolve()
    protocol: dict[str, object] = {
        "schema_version": PROTOCOL_SCHEMA,
        "source_git_head": _git_value(REPOSITORY_ROOT, "rev-parse", "HEAD"),
        "code_revision": _code_revision(),
        "expected_sequence": _expected_sequence_payload(),
        "worker_count": 36,
        "block_count": 6,
        "formal_preflight_contract": FORMAL_PREFLIGHT_CONTRACT,
        "worker_protocol_identity": worker._protocol_identity(),
        "five_fresh_manifest_file_sha256": (worker.FIVE_FRESH_MANIFEST_FILE_SHA256),
        "five_fresh_manifest_hash": worker.FIVE_FRESH_MANIFEST_HASH,
        "abcrown_commit": _git_value(abcrown, "rev-parse", "HEAD"),
        "auto_lirpa_commit": _git_value(abcrown / "auto_LiRPA", "rev-parse", "HEAD"),
        "vnncomp_commit": _git_value(benchmark, "rev-parse", "HEAD"),
        "model_name": model.name,
        "model_sha256": _file_sha256(model),
        "property_name": property_path.name,
        "property_sha256": _file_sha256(property_path),
        "python_name": args.abcrown_python.name,
        "control_counter_instrumentation": False,
        "profile_counter_instrumentation": "B2/B3-lightweight-no-journal",
        "resume_policy": "accept-complete-source-bound-worker-only",
        "performance_claimed": False,
    }
    protocol["protocol_hash"] = canonical_hash(protocol)
    return protocol


def _validate_protocol(value: Mapping[str, Any]) -> None:
    payload = dict(value)
    claimed = payload.pop("protocol_hash", None)
    if (
        claimed != canonical_hash(payload)
        or value.get("schema_version") != PROTOCOL_SCHEMA
        or value.get("expected_sequence") != _expected_sequence_payload()
        or value.get("worker_count") != 36
        or value.get("block_count") != 6
        or value.get("formal_preflight_contract") != FORMAL_PREFLIGHT_CONTRACT
        or value.get("worker_protocol_identity") != worker._protocol_identity()
        or value.get("five_fresh_manifest_file_sha256")
        != worker.FIVE_FRESH_MANIFEST_FILE_SHA256
        or value.get("five_fresh_manifest_hash") != worker.FIVE_FRESH_MANIFEST_HASH
        or value.get("control_counter_instrumentation") is not False
        or value.get("performance_claimed") is not False
    ):
        raise ValueError("FSG4/B3 protocol differs")
    source = value.get("source_git_head")
    revision = value.get("code_revision")
    if not isinstance(source, str) or not isinstance(revision, Mapping):
        raise TypeError("FSG4/B3 protocol source differs")
    if dict(revision) != _historical_code_revision(source):
        raise ValueError("FSG4/B3 protocol code revision differs")


def _run_id(
    block: int,
    position: int,
    configuration: FSG4B3TimingConfiguration,
    mode: FSG3Mode,
) -> str:
    return f"block-{block:02d}-pos-{position:02d}-{configuration.value}-{mode.value}"


def _worker_command(
    *,
    python: Path,
    benchmark: Path,
    abcrown: Path,
    model: Path,
    property_path: Path,
    result: Path,
    block: int,
    position: int,
    configuration: FSG4B3TimingConfiguration,
    mode: FSG3Mode,
) -> tuple[str, ...]:
    return (
        str(python),
        str(REPOSITORY_ROOT / "scripts/run_fsg4_b3_same_solver_timing.py"),
        "--configuration",
        configuration.value,
        "--mode",
        mode.value,
        "--run-id",
        _run_id(block, position, configuration, mode),
        "--block-index",
        str(block),
        "--sequence-position",
        str(position),
        "--benchmark-root",
        str(benchmark),
        "--abcrown-root",
        str(abcrown),
        "--model",
        str(model),
        "--property",
        str(property_path),
        "--result",
        str(result),
    )


def _command_projection(
    *,
    block: int,
    position: int,
    configuration: FSG4B3TimingConfiguration,
    mode: FSG3Mode,
) -> dict[str, object]:
    return {
        "worker": "scripts/run_fsg4_b3_same_solver_timing.py",
        "configuration": configuration.value,
        "mode": mode.value,
        "run_id": _run_id(block, position, configuration, mode),
        "block_index": block,
        "sequence_position": position,
        "benchmark_root": "$VNNCOMP_ROOT",
        "abcrown_root": "$ABCROWN_ROOT",
        "model": "$MODEL",
        "property": "$PROPERTY",
        "result": f"workers/run_{block * 6 + position:02d}.json",
    }


def _sanitize_text(
    value: str,
    *,
    benchmark: Path,
    abcrown: Path,
    python: Path,
) -> str:
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
    for original, replacement in replacements:
        value = value.replace(original, replacement)
    return value


def _normalize_process_names(rows: object) -> list[Any]:
    if not isinstance(rows, list):
        raise TypeError("FSG4/B3 preflight process list differs")
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("name"), str):
            row["name"] = Path(row["name"]).name
    return rows


def _normalize_preflight(value: Mapping[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(value, allow_nan=False))
    if not isinstance(normalized, dict) or not isinstance(
        normalized.get("samples"), list
    ):
        raise TypeError("FSG4/B3 formal preflight differs")
    for sample in normalized["samples"]:
        if not isinstance(sample, dict):
            raise TypeError("FSG4/B3 formal preflight sample differs")
        sample["compute_processes"] = _normalize_process_names(
            sample.get("compute_processes")
        )
    base_experiment._validate_formal_preflight(normalized)
    return normalized


def _run_worker(
    *,
    artifact: Path,
    index: int,
    command: tuple[str, ...],
    command_projection: Mapping[str, object],
    benchmark: Path,
    abcrown: Path,
    python: Path,
) -> tuple[dict[str, Any], dict[str, object]]:
    result_relative = f"workers/run_{index:02d}.json"
    stdout_relative = f"logs/run_{index:02d}.stdout.txt"
    stderr_relative = f"logs/run_{index:02d}.stderr.txt"
    result_path = artifact / result_relative
    result_path.parent.mkdir(parents=True, exist_ok=True)
    (artifact / "logs").mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["PYTHONNOUSERSITE"] = "1"
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT) + (
        os.pathsep + existing if existing else ""
    )
    before = base_experiment._host_snapshot()
    started_ns = time.monotonic_ns()
    try:
        completed = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=WORKER_SUBPROCESS_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        ended_ns = time.monotonic_ns()

        def timeout_text(value: bytes | str | None) -> str:
            if value is None:
                return ""
            return value.decode(errors="replace") if isinstance(value, bytes) else value

        (artifact / stdout_relative).write_text(
            _sanitize_text(
                timeout_text(error.stdout),
                benchmark=benchmark,
                abcrown=abcrown,
                python=python,
            ),
            encoding="utf-8",
        )
        (artifact / stderr_relative).write_text(
            _sanitize_text(
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
                "command": dict(command_projection),
                "returncode": None,
                "timeout_seconds": WORKER_SUBPROCESS_TIMEOUT_SECONDS,
                "timed_out": True,
                "started_monotonic_ns": started_ns,
                "ended_monotonic_ns": ended_ns,
                "duration_ns": ended_ns - started_ns,
                "host_before": before,
                "host_after": base_experiment._host_snapshot(),
                "result_file": result_relative,
                "stdout_file": stdout_relative,
                "stderr_file": stderr_relative,
                "performance_claimed": False,
            },
        )
        raise RuntimeError(f"FSG4/B3 worker {index} timed out") from error
    ended_ns = time.monotonic_ns()
    after = base_experiment._host_snapshot()
    (artifact / stdout_relative).write_text(
        _sanitize_text(
            completed.stdout, benchmark=benchmark, abcrown=abcrown, python=python
        ),
        encoding="utf-8",
    )
    (artifact / stderr_relative).write_text(
        _sanitize_text(
            completed.stderr, benchmark=benchmark, abcrown=abcrown, python=python
        ),
        encoding="utf-8",
    )
    metadata: dict[str, object] = {
        "index": index,
        "command": dict(command_projection),
        "returncode": completed.returncode,
        "timeout_seconds": WORKER_SUBPROCESS_TIMEOUT_SECONDS,
        "timed_out": False,
        "started_monotonic_ns": started_ns,
        "ended_monotonic_ns": ended_ns,
        "duration_ns": ended_ns - started_ns,
        "host_before": before,
        "host_after": after,
        "result_file": result_relative,
        "stdout_file": stdout_relative,
        "stderr_file": stderr_relative,
        "performance_claimed": False,
    }
    if completed.returncode != 0 or not result_path.is_file():
        _write_json(artifact / "failed_worker.json", metadata)
        raise RuntimeError(
            f"FSG4/B3 worker {index} failed with {completed.returncode}:\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    return _load_json(result_path), metadata


def _validate_envelope(
    value: Mapping[str, Any],
    *,
    block: int,
    position: int,
    configuration: FSG4B3TimingConfiguration,
    mode: FSG3Mode,
) -> FSG4B3TimingRun:
    if (
        value.get("schema_version") != worker.WORKER_ENVELOPE_SCHEMA
        or value.get("performance_claimed") is not False
        or not isinstance(value.get("run"), Mapping)
        or not isinstance(value.get("diagnostics"), Mapping)
    ):
        raise ValueError("FSG4/B3 worker envelope differs")
    run = fsg4_b3_timing_run_from_dict(cast(Mapping[str, Any], value["run"]))
    if (run.block_index, run.sequence_position, run.configuration, run.mode) != (
        block,
        position,
        configuration,
        mode,
    ):
        raise ValueError("FSG4/B3 worker sequence differs")
    diagnostics = cast(Mapping[str, Any], value["diagnostics"])
    if (
        diagnostics.get("fsg4_configuration") != configuration.value
        or diagnostics.get("base_measurement_configuration")
        != ("B0" if configuration == FSG4B3TimingConfiguration.B0 else "B2")
        or diagnostics.get("profile_counter_instrumentation")
        is not (
            mode == FSG3Mode.PROFILE
            and configuration
            in {FSG4B3TimingConfiguration.B2, FSG4B3TimingConfiguration.B3}
        )
        or diagnostics.get("activation_receipt_hash")
        != canonical_hash(run.activation.to_dict())
        or diagnostics.get("five_fresh_manifest_file_sha256")
        != worker.FIVE_FRESH_MANIFEST_FILE_SHA256
        or diagnostics.get("five_fresh_manifest_hash")
        != worker.FIVE_FRESH_MANIFEST_HASH
    ):
        raise ValueError("FSG4/B3 worker diagnostic binding differs")
    return run


def _paired_rows(runs: Sequence[FSG4B3TimingRun]) -> list[dict[str, object]]:
    indexed = {(run.block_index, run.configuration, run.mode): run for run in runs}
    rows: list[dict[str, object]] = []
    comparisons = (
        (FSG4B3TimingConfiguration.B0, FSG4B3TimingConfiguration.B2),
        (FSG4B3TimingConfiguration.B0, FSG4B3TimingConfiguration.B3),
        (FSG4B3TimingConfiguration.B2, FSG4B3TimingConfiguration.B3),
    )
    for block in range(6):
        for numerator, denominator in comparisons:
            reference = indexed[(block, numerator, FSG3Mode.CONTROL)]
            candidate = indexed[(block, denominator, FSG3Mode.CONTROL)]
            row: dict[str, object] = {
                "block_index": block,
                "numerator": numerator.value,
                "denominator": denominator.value,
                "numerator_run_id": reference.run_id,
                "denominator_run_id": candidate.run_id,
                "numerator_hash": reference.stable_hash(),
                "denominator_hash": candidate.stable_hash(),
                "ratio": {
                    name: getattr(reference.metrics, name)
                    / getattr(candidate.metrics, name)
                    for name in METRIC_NAMES
                },
                "performance_claimed": False,
            }
            row["pair_hash"] = canonical_hash(row)
            rows.append(row)
    return rows


def _profile_rows(runs: Sequence[FSG4B3TimingRun]) -> list[dict[str, object]]:
    return [
        {
            "run_id": run.run_id,
            "run_hash": run.stable_hash(),
            "configuration": run.configuration.value,
            "block_index": run.block_index,
            "span_index": index,
            "span": span.to_dict(),
        }
        for run in runs
        if run.mode == FSG3Mode.PROFILE
        for index, span in enumerate(run.profile_spans)
    ]


def _activation_rows(runs: Sequence[FSG4B3TimingRun]) -> list[dict[str, object]]:
    return [
        {
            "run_id": run.run_id,
            "run_hash": run.stable_hash(),
            "configuration": run.configuration.value,
            "mode": run.mode.value,
            "activation": run.activation.to_dict(),
            "activation_hash": canonical_hash(run.activation.to_dict()),
            "performance_claimed": False,
        }
        for run in runs
    ]


def _closure(runs: Sequence[FSG4B3TimingRun]) -> dict[str, object]:
    rows = [
        {
            "run_id": run.run_id,
            "run_hash": run.stable_hash(),
            "configuration": run.configuration.value,
            "block_index": run.block_index,
            "closure_error": run.profile_closure_error,
            "residual_share": run.profile_residual_share,
        }
        for run in runs
        if run.mode == FSG3Mode.PROFILE
    ]
    return {
        "rows": rows,
        "closure_limit": 0.01,
        "residual_limit": 0.03,
        "all_closed": all(
            cast(float, row["closure_error"]) <= 0.01
            and cast(float, row["residual_share"]) <= 0.03
            for row in rows
        ),
        "performance_claimed": False,
    }


def _derived_payloads(
    runs: Sequence[FSG4B3TimingRun],
    envelopes: Sequence[Mapping[str, Any]],
    metadata: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    summary = derive_fsg4_b3_timing_evidence(runs)
    return {
        "paired_runs.jsonl": _paired_rows(runs),
        "profile_spans.jsonl": _profile_rows(runs),
        "activation_receipts.jsonl": _activation_rows(runs),
        "closure.json": _closure(runs),
        "summary.json": summary,
        "failure_rows.jsonl": [
            {"failure": item} for item in cast(list[str], summary["failure_rows"])
        ],
        "environment.json": base_experiment._environment(envelopes, metadata),
    }


def _readme() -> str:
    return (
        "# FSG4 B3 same-solver timing artifact\n\n"
        "This directory contains 36 fresh B0/B2/B3 control/profile workers in "
        "the preregistered six-permutation order. Raw rows never claim "
        "performance; replay recomputes semantics, activation, environment, "
        "profile closure, paired ratios, and the B3 decision.\n"
    )


def _replay_result(summary: Mapping[str, Any]) -> dict[str, object]:
    return {
        "status": summary["status"],
        "run_count": summary["run_count"],
        "summary_hash": summary["summary_hash"],
        "correctness_passed": summary["correctness_passed"],
        "environment_passed": summary["environment_passed"],
        "measurement_auditable": summary["measurement_auditable"],
        "decision_inputs": summary["decision_inputs"],
        "performance_claimed": False,
    }


def _load_complete_worker(
    *,
    artifact: Path,
    index: int,
    block: int,
    position: int,
    configuration: FSG4B3TimingConfiguration,
    mode: FSG3Mode,
) -> tuple[dict[str, Any], dict[str, Any], FSG4B3TimingRun] | None:
    envelope_path = artifact / f"workers/run_{index:02d}.json"
    metadata_path = artifact / f"metadata/run_{index:02d}.json"
    stdout_path = artifact / f"logs/run_{index:02d}.stdout.txt"
    stderr_path = artifact / f"logs/run_{index:02d}.stderr.txt"
    exists = [
        path.exists()
        for path in (envelope_path, metadata_path, stdout_path, stderr_path)
    ]
    if not any(exists):
        return None
    if not all(exists):
        raise ValueError(f"FSG4/B3 partial worker cannot resume: {index}")
    envelope = _load_json(envelope_path)
    metadata = _load_json(metadata_path)
    run = _validate_envelope(
        envelope,
        block=block,
        position=position,
        configuration=configuration,
        mode=mode,
    )
    if (
        metadata.get("index") != index
        or metadata.get("returncode") != 0
        or metadata.get("timed_out") is not False
        or metadata.get("command")
        != _command_projection(
            block=block,
            position=position,
            configuration=configuration,
            mode=mode,
        )
        or not isinstance(metadata.get("formal_preflight"), Mapping)
    ):
        raise ValueError(f"FSG4/B3 resume metadata differs: {index}")
    base_experiment._validate_formal_preflight(
        cast(Mapping[str, Any], metadata["formal_preflight"])
    )
    return envelope, metadata, run


def _generate(args: argparse.Namespace) -> dict[str, object]:
    if not _code_paths_clean():
        raise ValueError("FSG4/B3 code paths must be clean before formal generation")
    artifact = args.artifact_dir.resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    expected_protocol = _protocol(args)
    protocol_path = artifact / "protocol.json"
    if protocol_path.exists():
        observed_protocol = _load_json(protocol_path)
        _validate_protocol(observed_protocol)
        if observed_protocol != expected_protocol:
            raise ValueError("FSG4/B3 resume protocol differs from current source")
    else:
        unexpected = list(artifact.iterdir())
        if unexpected:
            raise FileExistsError("FSG4/B3 artifact lacks root protocol")
        _write_json(protocol_path, expected_protocol)
    benchmark = args.benchmark_root.resolve()
    abcrown = args.abcrown_root.resolve()
    python = args.abcrown_python.expanduser().absolute()
    model = args.model.resolve()
    property_path = args.property.resolve()
    envelopes: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    runs: list[FSG4B3TimingRun] = []
    for index, (block, position, configuration, mode) in enumerate(
        expected_fsg4_b3_sequence()
    ):
        existing = _load_complete_worker(
            artifact=artifact,
            index=index,
            block=block,
            position=position,
            configuration=configuration,
            mode=mode,
        )
        if existing is not None:
            envelope, outer, run = existing
            print(
                _canonical_json({"resumed_worker": index, "run_id": run.run_id}),
                flush=True,
            )
        else:
            preflight = _normalize_preflight(
                base_experiment._wait_for_formal_environment()
            )
            result = artifact / f"workers/run_{index:02d}.json"
            projection = _command_projection(
                block=block,
                position=position,
                configuration=configuration,
                mode=mode,
            )
            envelope, outer = _run_worker(
                artifact=artifact,
                index=index,
                command=_worker_command(
                    python=python,
                    benchmark=benchmark,
                    abcrown=abcrown,
                    model=model,
                    property_path=property_path,
                    result=result,
                    block=block,
                    position=position,
                    configuration=configuration,
                    mode=mode,
                ),
                command_projection=projection,
                benchmark=benchmark,
                abcrown=abcrown,
                python=python,
            )
            outer["formal_preflight"] = preflight
            run = _validate_envelope(
                envelope,
                block=block,
                position=position,
                configuration=configuration,
                mode=mode,
            )
            _write_json(artifact / f"metadata/run_{index:02d}.json", outer)
            print(
                _canonical_json(
                    {
                        "completed_worker": index,
                        "run_id": run.run_id,
                        "environment_admitted": run.environment.admitted,
                    }
                ),
                flush=True,
            )
        envelopes.append(envelope)
        metadata.append(outer)
        runs.append(run)
        _write_jsonl(artifact / "worker_runs.jsonl", [item.to_dict() for item in runs])
        _write_jsonl(artifact / "run_metadata.jsonl", metadata)
    derived = _derived_payloads(runs, envelopes, metadata)
    for name, payload in derived.items():
        if name.endswith(".jsonl"):
            _write_jsonl(artifact / name, cast(Sequence[object], payload))
        else:
            _write_json(artifact / name, payload)
    summary = cast(Mapping[str, Any], derived["summary.json"])
    if (
        summary.get("measurement_auditable") is not True
        or summary.get("correctness_passed") is not True
        or summary.get("environment_passed") is not True
    ):
        _write_json(artifact / "failed_summary.json", summary)
        raise ValueError("FSG4/B3 formal measurement admission failed")
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
        "worker_count": 36,
        "performance_claimed": False,
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    _write_json(artifact / "manifest.json", manifest)
    return replay


def _verify_static_artifact(
    artifact: Path,
) -> tuple[list[FSG4B3TimingRun], dict[str, Any], dict[str, object]]:
    manifest = _load_json(artifact / "manifest.json")
    semantic_manifest = {
        key: value for key, value in manifest.items() if key != "manifest_hash"
    }
    if (
        manifest.get("schema_version") != ARTIFACT_SCHEMA
        or manifest.get("manifest_hash") != canonical_hash(semantic_manifest)
        or manifest.get("worker_count") != 36
        or manifest.get("performance_claimed") is not False
    ):
        raise ValueError("FSG4/B3 manifest differs")
    protocol = _load_json(artifact / "protocol.json")
    _validate_protocol(protocol)
    if (
        manifest.get("source_git_head") != protocol["source_git_head"]
        or manifest.get("code_revision") != protocol["code_revision"]
        or manifest.get("protocol_hash") != protocol["protocol_hash"]
    ):
        raise ValueError("FSG4/B3 manifest protocol binding differs")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise TypeError("FSG4/B3 manifest file inventory differs")
    observed_files = {
        str(path.relative_to(artifact))
        for path in artifact.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if set(files) != observed_files:
        raise ValueError("FSG4/B3 artifact inventory differs")
    for name, digest in files.items():
        if not isinstance(name, str) or digest != _file_sha256(artifact / name):
            raise ValueError(f"FSG4/B3 artifact digest differs: {name}")
    envelopes: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    runs: list[FSG4B3TimingRun] = []
    for index, (block, position, configuration, mode) in enumerate(
        expected_fsg4_b3_sequence()
    ):
        loaded = _load_complete_worker(
            artifact=artifact,
            index=index,
            block=block,
            position=position,
            configuration=configuration,
            mode=mode,
        )
        if loaded is None:
            raise ValueError(f"FSG4/B3 worker missing: {index}")
        envelope, outer, run = loaded
        envelopes.append(envelope)
        metadata.append(outer)
        runs.append(run)
    if _load_jsonl(artifact / "worker_runs.jsonl") != [run.to_dict() for run in runs]:
        raise ValueError("FSG4/B3 worker aggregate differs")
    if _load_jsonl(artifact / "run_metadata.jsonl") != metadata:
        raise ValueError("FSG4/B3 metadata aggregate differs")
    derived = _derived_payloads(runs, envelopes, metadata)
    for name, expected in derived.items():
        observed: object = (
            _load_jsonl(artifact / name)
            if name.endswith(".jsonl")
            else _load_json(artifact / name)
        )
        if observed != expected:
            raise ValueError(f"FSG4/B3 derived replay differs: {name}")
    summary = cast(dict[str, Any], derived["summary.json"])
    if (
        summary.get("measurement_auditable") is not True
        or summary.get("correctness_passed") is not True
        or summary.get("environment_passed") is not True
    ):
        raise ValueError("FSG4/B3 formal measurement admission differs")
    if (
        manifest.get("summary_hash") != summary["summary_hash"]
        or manifest.get("status") != summary["status"]
    ):
        raise ValueError("FSG4/B3 manifest summary binding differs")
    replay = _replay_result(summary)
    if (artifact / "replay_stdout.txt").read_text(encoding="utf-8") != (
        _canonical_json(replay) + "\n"
    ):
        raise ValueError("FSG4/B3 replay stdout differs")
    if (artifact / "README.md").read_text(encoding="utf-8") != _readme():
        raise ValueError("FSG4/B3 README differs")
    return runs, summary, replay


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
    """Generate/resume or independently replay a formal FSG4 artifact."""

    args = _parse_args()
    if args.command == "generate":
        result = _generate(args)
    else:
        _runs, _summary, result = _verify_static_artifact(args.artifact_dir.resolve())
    print(_canonical_json(result))


if __name__ == "__main__":
    main()
