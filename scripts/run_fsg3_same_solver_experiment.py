#!/usr/bin/env python3
"""Generate or replay the preregistered 36-process FSG3 timing artifact."""

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

from boundflow.runtime.fsg3_same_solver_timing import (
    _semantic_pair_failures,
    canonical_hash,
    derive_fsg3_timing_evidence,
    expected_fsg3_sequence,
    FSG3Configuration,
    FSG3Mode,
    FSG3TimingRun,
    fsg3_timing_run_from_dict,
)
from scripts import run_fsg3_same_solver_timing as worker

ARTIFACT_SCHEMA = "boundflow.fsg3-same-solver-artifact/v1"
CODE_PATHS = (
    "boundflow/runtime/fsg3_same_solver_timing.py",
    "scripts/run_fsg3_same_solver_timing.py",
    "scripts/run_fsg3_same_solver_experiment.py",
    "scripts/run_rvir_v4_live_return_capture.py",
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
PREFLIGHT_TEMPERATURE_LIMIT_C = 50
PREFLIGHT_POLL_SECONDS = 5
PREFLIGHT_TIMEOUT_SECONDS = 900
FORMAL_PREFLIGHT_CONTRACT = {
    "temperature_limit_celsius": PREFLIGHT_TEMPERATURE_LIMIT_C,
    "poll_seconds": PREFLIGHT_POLL_SECONDS,
    "timeout_seconds": PREFLIGHT_TIMEOUT_SECONDS,
    "external_compute_processes_forbidden": True,
    "ac_power_required": True,
    "thermal_reason_must_be_inactive": True,
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


def _write_jsonl(path: Path, values: Sequence[object]) -> None:
    path.write_text(
        "".join(_canonical_json(value) + "\n" for value in values),
        encoding="utf-8",
    )


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"FSG3 JSON root differs: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"FSG3 JSONL row differs: {path}")
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


def _verify_code_provenance(manifest: Mapping[str, Any]) -> None:
    source = manifest.get("source_git_head")
    revision = manifest.get("code_revision")
    if not isinstance(source, str) or not isinstance(revision, Mapping):
        raise ValueError("FSG3 code provenance differs")
    if _git_value(REPOSITORY_ROOT, "rev-parse", "HEAD") == source:
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
        raise ValueError("FSG3 code revision differs")


def _host_snapshot() -> dict[str, object]:
    governors = sorted(
        {
            path.read_text(encoding="utf-8").strip()
            for path in Path("/sys/devices/system/cpu").glob(
                "cpu[0-9]*/cpufreq/scaling_governor"
            )
            if path.is_file()
        }
    )
    ac_path = Path("/sys/class/power_supply/ACAD/online")
    return {
        "monotonic_ns": time.monotonic_ns(),
        "load_average": list(os.getloadavg()),
        "cpu_governors": governors,
        "ac_powered": ac_path.is_file()
        and ac_path.read_text(encoding="utf-8").strip() == "1",
    }


def _temperature_celsius(snapshot: Mapping[str, object]) -> int:
    try:
        return int(str(snapshot["temperature"]).split()[0])
    except (IndexError, ValueError) as error:
        raise ValueError("FSG3 preflight temperature differs") from error


def _external_compute_processes(
    processes: Sequence[Mapping[str, object]],
) -> list[str]:
    external: list[str] = []
    for row in processes:
        name = str(row["name"])
        raw_memory = row["used_memory_mib"]
        if not isinstance(raw_memory, int) or isinstance(raw_memory, bool):
            raise TypeError("FSG3 preflight process memory differs")
        memory = raw_memory
        if Path(name).name in worker.ALLOWED_GRAPHICS_PROCESSES and memory < 64:
            continue
        external.append(f"{row['pid']}:{name}:{memory}MiB")
    return external


def _wait_for_formal_environment() -> dict[str, object]:
    """Wait for a frozen cool/idle preflight without discarding any run."""

    started_ns = time.monotonic_ns()
    samples: list[dict[str, object]] = []
    while True:
        snapshot = worker._nvidia_snapshot()
        processes = worker._compute_processes()
        external = _external_compute_processes(processes)
        if external:
            raise RuntimeError(
                "FSG3 formal preflight found external CUDA compute processes: "
                + ", ".join(external)
            )
        thermal_active = any(
            snapshot[name] != "Not Active"
            for name in ("sw_thermal_slowdown", "hw_thermal_slowdown")
        )
        temperature = _temperature_celsius(snapshot)
        sample = {
            "elapsed_ns": time.monotonic_ns() - started_ns,
            "temperature_celsius": temperature,
            "thermal_active": thermal_active,
            "gpu_snapshot": snapshot,
            "compute_processes": processes,
            "ac_powered": worker._ac_powered(),
        }
        samples.append(sample)
        ready = (
            temperature <= PREFLIGHT_TEMPERATURE_LIMIT_C
            and not thermal_active
            and sample["ac_powered"] is True
        )
        if ready:
            return {
                "temperature_limit_celsius": PREFLIGHT_TEMPERATURE_LIMIT_C,
                "poll_seconds": PREFLIGHT_POLL_SECONDS,
                "timeout_seconds": PREFLIGHT_TIMEOUT_SECONDS,
                "sample_count": len(samples),
                "wait_ns": time.monotonic_ns() - started_ns,
                "samples": samples,
                "admitted": True,
            }
        if time.monotonic_ns() - started_ns > PREFLIGHT_TIMEOUT_SECONDS * 1_000_000_000:
            raise TimeoutError("FSG3 formal preflight did not reach cool idle state")
        print(
            _canonical_json(
                {
                    "preflight": "waiting",
                    "temperature_celsius": temperature,
                    "thermal_active": thermal_active,
                    "sample_count": len(samples),
                }
            ),
            flush=True,
        )
        time.sleep(PREFLIGHT_POLL_SECONDS)


def _validate_formal_preflight(value: Mapping[str, Any]) -> None:
    expected = {
        "temperature_limit_celsius",
        "poll_seconds",
        "timeout_seconds",
        "sample_count",
        "wait_ns",
        "samples",
        "admitted",
    }
    if (
        set(value) != expected
        or value["temperature_limit_celsius"] != PREFLIGHT_TEMPERATURE_LIMIT_C
        or value["poll_seconds"] != PREFLIGHT_POLL_SECONDS
        or value["timeout_seconds"] != PREFLIGHT_TIMEOUT_SECONDS
        or value["admitted"] is not True
        or not isinstance(value["samples"], list)
        or value["sample_count"] != len(value["samples"])
        or not value["samples"]
        or int(value["wait_ns"]) < 0
    ):
        raise ValueError("FSG3 formal preflight payload differs")
    last = value["samples"][-1]
    if not isinstance(last, Mapping):
        raise TypeError("FSG3 formal preflight sample differs")
    processes = last.get("compute_processes")
    if not isinstance(processes, list):
        raise TypeError("FSG3 formal preflight process list differs")
    if (
        int(last.get("temperature_celsius", -1)) > PREFLIGHT_TEMPERATURE_LIMIT_C
        or last.get("thermal_active") is not False
        or last.get("ac_powered") is not True
        or _external_compute_processes(cast(Sequence[Mapping[str, object]], processes))
    ):
        raise ValueError("FSG3 formal preflight admission differs")


def _run_id(
    block: int,
    position: int,
    configuration: FSG3Configuration,
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
    configuration: FSG3Configuration,
    mode: FSG3Mode,
) -> tuple[str, ...]:
    return (
        str(python),
        str(REPOSITORY_ROOT / "scripts/run_fsg3_same_solver_timing.py"),
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


def _run_worker(
    *,
    artifact: Path,
    index: int,
    command: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, object]]:
    stdout_relative = f"logs/run_{index:02d}.stdout.txt"
    stderr_relative = f"logs/run_{index:02d}.stderr.txt"
    result_relative = f"workers/run_{index:02d}.json"
    result_path = artifact / result_relative
    result_path.parent.mkdir(parents=True, exist_ok=True)
    (artifact / "logs").mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["PYTHONNOUSERSITE"] = "1"
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT) + (
        os.pathsep + existing if existing else ""
    )
    before = _host_snapshot()
    started_ns = time.monotonic_ns()
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=180,
    )
    ended_ns = time.monotonic_ns()
    after = _host_snapshot()
    (artifact / stdout_relative).write_text(completed.stdout, encoding="utf-8")
    (artifact / stderr_relative).write_text(completed.stderr, encoding="utf-8")
    metadata: dict[str, object] = {
        "index": index,
        "command": list(command),
        "returncode": completed.returncode,
        "timeout_seconds": 180,
        "timed_out": False,
        "started_monotonic_ns": started_ns,
        "ended_monotonic_ns": ended_ns,
        "duration_ns": ended_ns - started_ns,
        "host_before": before,
        "host_after": after,
        "result_file": result_relative,
        "stdout_file": stdout_relative,
        "stderr_file": stderr_relative,
    }
    if completed.returncode != 0 or not result_path.is_file():
        raise RuntimeError(
            f"FSG3 worker {index} failed with {completed.returncode}:\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    return _load_json(result_path), metadata


def _paired_rows(runs: Sequence[FSG3TimingRun]) -> list[dict[str, object]]:
    indexed = {(run.block_index, run.configuration, run.mode): run for run in runs}
    rows: list[dict[str, object]] = []
    for block in range(6):
        baseline = indexed[(block, FSG3Configuration.B0, FSG3Mode.CONTROL)]
        for candidate in (FSG3Configuration.B1, FSG3Configuration.B2):
            observed = indexed[(block, candidate, FSG3Mode.CONTROL)]
            row: dict[str, object] = {
                "block_index": block,
                "baseline_run_id": baseline.run_id,
                "candidate_run_id": observed.run_id,
                "candidate": candidate.value,
                "baseline_hash": baseline.stable_hash(),
                "candidate_hash": observed.stable_hash(),
                "speedup_b0_over_candidate": {
                    name: getattr(baseline.metrics, name)
                    / getattr(observed.metrics, name)
                    for name in METRIC_NAMES
                },
                "performance_claimed": False,
            }
            row["pair_hash"] = canonical_hash(row)
            rows.append(row)
    return rows


def _profile_rows(runs: Sequence[FSG3TimingRun]) -> list[dict[str, object]]:
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


def _closure(runs: Sequence[FSG3TimingRun]) -> dict[str, object]:
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


def _environment(
    envelopes: Sequence[Mapping[str, Any]],
    metadata: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for envelope, outer in zip(envelopes, metadata):
        run = cast(Mapping[str, Any], envelope["run"])
        diagnostics = cast(Mapping[str, Any], envelope["diagnostics"])
        gate = cast(Mapping[str, Any], run["environment"])
        runtime_environment = diagnostics["runtime_environment"]
        if gate["runtime_identity"] != canonical_hash(runtime_environment):
            raise ValueError("FSG3 runtime environment identity differs")
        preflight = outer.get("formal_preflight")
        if preflight is not None:
            if not isinstance(preflight, Mapping):
                raise TypeError("FSG3 formal preflight metadata differs")
            _validate_formal_preflight(cast(Mapping[str, Any], preflight))
        rows.append(
            {
                "run_id": run["run_id"],
                "gate": gate,
                "runtime_environment": runtime_environment,
                "gpu_before": diagnostics["environment_before"],
                "gpu_after": diagnostics["environment_after"],
                "compute_processes_before": diagnostics["compute_processes_before"],
                "compute_processes_after": diagnostics["compute_processes_after"],
                "host_before": outer["host_before"],
                "host_after": outer["host_after"],
                "formal_preflight": preflight,
            }
        )
    runtime_hashes = {canonical_hash(row["runtime_environment"]) for row in rows}
    return {
        "schema_version": ARTIFACT_SCHEMA,
        "run_count": len(rows),
        "rows": rows,
        "runtime_identity_count": len(runtime_hashes),
        "all_workers_environment_admitted": all(
            cast(Mapping[str, Any], row["gate"])["admitted"] is True for row in rows
        ),
        "performance_claimed": False,
    }


def _readme() -> str:
    return (
        "# FSG3 B0/B1/B2 Same-Solver Timing\n\n"
        "This artifact contains the preregistered 36 fresh processes: six full "
        "configuration permutations, each with adjacent control/profile workers. "
        "Headline latency uses control only; profile spans are attribution-only. "
        "Replay recomputes sequence, semantics, no-fallback, environment, closure, "
        "perturbation, paired statistics and the baseline status.\n"
    )


def _replay_result(summary: Mapping[str, Any]) -> dict[str, object]:
    return {
        "status": "replay-passed",
        "evidence_status": summary["status"],
        "run_count": summary["run_count"],
        "measurement_auditable": summary["measurement_auditable"],
        "correctness_passed": summary["correctness_passed"],
        "environment_passed": summary["environment_passed"],
        "summary_hash": summary["summary_hash"],
        "performance_claimed": False,
    }


def _derived_payloads(
    runs: Sequence[FSG3TimingRun],
    envelopes: Sequence[Mapping[str, Any]],
    metadata: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    summary = derive_fsg3_timing_evidence(runs)
    return {
        "paired_runs.jsonl": _paired_rows(runs),
        "profile_spans.jsonl": _profile_rows(runs),
        "closure.json": _closure(runs),
        "summary.json": summary,
        "failure_rows.jsonl": [
            {"failure": item} for item in cast(list[str], summary["failure_rows"])
        ],
        "environment.json": _environment(envelopes, metadata),
    }


def _generate(args: argparse.Namespace) -> dict[str, object]:
    if not _code_paths_clean():
        raise ValueError("FSG3 code paths must be clean before formal generation")
    artifact = args.artifact_dir.resolve()
    if artifact.exists() and any(artifact.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {artifact}")
    artifact.mkdir(parents=True, exist_ok=True)
    benchmark = args.benchmark_root.resolve()
    abcrown = args.abcrown_root.resolve()
    python = args.abcrown_python.expanduser().absolute()
    model = args.model.resolve()
    property_path = args.property.resolve()
    envelopes: list[dict[str, Any]] = []
    metadata: list[dict[str, object]] = []
    runs: list[FSG3TimingRun] = []
    for index, (block, position, configuration, mode) in enumerate(
        expected_fsg3_sequence()
    ):
        preflight = _wait_for_formal_environment()
        result = artifact / f"workers/run_{index:02d}.json"
        command = _worker_command(
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
        )
        envelope, outer = _run_worker(artifact=artifact, index=index, command=command)
        outer["formal_preflight"] = preflight
        if (
            envelope.get("schema_version") != worker.WORKER_ENVELOPE_SCHEMA
            or envelope.get("performance_claimed") is not False
            or not isinstance(envelope.get("run"), Mapping)
            or not isinstance(envelope.get("diagnostics"), Mapping)
        ):
            raise ValueError(f"FSG3 worker envelope differs: {index}")
        run = fsg3_timing_run_from_dict(cast(Mapping[str, Any], envelope["run"]))
        if (run.block_index, run.sequence_position, run.configuration, run.mode) != (
            block,
            position,
            configuration,
            mode,
        ):
            raise ValueError(f"FSG3 worker sequence differs: {index}")
        envelopes.append(envelope)
        metadata.append(outer)
        runs.append(run)
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
    _write_jsonl(artifact / "worker_runs.jsonl", [run.to_dict() for run in runs])
    _write_jsonl(artifact / "run_metadata.jsonl", metadata)
    derived = _derived_payloads(runs, envelopes, metadata)
    for name, payload in derived.items():
        if name.endswith(".jsonl"):
            _write_jsonl(artifact / name, cast(Sequence[object], payload))
        else:
            _write_json(artifact / name, payload)
    summary = cast(Mapping[str, Any], derived["summary.json"])
    replay_result = _replay_result(summary)
    (artifact / "replay_stdout.txt").write_text(
        _canonical_json(replay_result) + "\n", encoding="utf-8"
    )
    (artifact / "README.md").write_text(_readme(), encoding="utf-8")
    files = sorted(
        str(path.relative_to(artifact))
        for path in artifact.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    manifest: dict[str, object] = {
        "schema_version": ARTIFACT_SCHEMA,
        "source_git_head": _git_value(REPOSITORY_ROOT, "rev-parse", "HEAD"),
        "code_revision": _code_revision(),
        "files": {name: _file_sha256(artifact / name) for name in files},
        "expected_sequence": [
            [block, position, configuration.value, mode.value]
            for block, position, configuration, mode in expected_fsg3_sequence()
        ],
        "formal_preflight_contract": FORMAL_PREFLIGHT_CONTRACT,
        "abcrown_commit": _git_value(abcrown, "rev-parse", "HEAD"),
        "auto_lirpa_commit": _git_value(abcrown / "auto_LiRPA", "rev-parse", "HEAD"),
        "vnncomp_commit": _git_value(benchmark, "rev-parse", "HEAD"),
        "model_sha256": _file_sha256(model),
        "property_sha256": _file_sha256(property_path),
        "source_identity": runs[0].source_identity,
        "protocol_identity": runs[0].protocol_identity,
        "summary_hash": summary["summary_hash"],
        "status": summary["status"],
        "performance_claimed": False,
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    _write_json(artifact / "manifest.json", manifest)
    return replay_result


def _generate_smoke(args: argparse.Namespace) -> dict[str, object]:
    """Run block zero sequentially without creating a formal artifact claim."""

    output = args.artifact_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    benchmark = args.benchmark_root.resolve()
    abcrown = args.abcrown_root.resolve()
    python = args.abcrown_python.expanduser().absolute()
    model = args.model.resolve()
    property_path = args.property.resolve()
    runs: list[FSG3TimingRun] = []
    envelopes: list[dict[str, Any]] = []
    metadata: list[dict[str, object]] = []
    for index, (block, position, configuration, mode) in enumerate(
        expected_fsg3_sequence()[:6]
    ):
        result_path = output / f"workers/run_{index:02d}.json"
        envelope, outer = _run_worker(
            artifact=output,
            index=index,
            command=_worker_command(
                python=python,
                benchmark=benchmark,
                abcrown=abcrown,
                model=model,
                property_path=property_path,
                result=result_path,
                block=block,
                position=position,
                configuration=configuration,
                mode=mode,
            ),
        )
        run = fsg3_timing_run_from_dict(cast(Mapping[str, Any], envelope["run"]))
        if (run.block_index, run.sequence_position, run.configuration, run.mode) != (
            block,
            position,
            configuration,
            mode,
        ):
            raise ValueError(f"FSG3 smoke worker sequence differs: {index}")
        runs.append(run)
        envelopes.append(envelope)
        metadata.append(outer)
    indexed = {(run.configuration, run.mode): run for run in runs}
    failures: list[str] = []
    for mode in FSG3Mode:
        baseline = indexed[(FSG3Configuration.B0, mode)]
        passthrough = indexed[(FSG3Configuration.B1, mode)]
        if (
            baseline.execution.provider_compute_bounds_call_count
            != passthrough.execution.provider_compute_bounds_call_count
            or baseline.execution.provider_update_bounds_call_count
            != passthrough.execution.provider_update_bounds_call_count
        ):
            failures.append(f"{mode.value}:B1-provider-count-differs")
        for configuration in (FSG3Configuration.B1, FSG3Configuration.B2):
            failures.extend(
                _semantic_pair_failures(
                    baseline.semantics,
                    indexed[(configuration, mode)].semantics,
                    label=f"{mode.value}:{configuration.value}",
                )
            )
    for configuration in FSG3Configuration:
        failures.extend(
            _semantic_pair_failures(
                indexed[(configuration, FSG3Mode.CONTROL)].semantics,
                indexed[(configuration, FSG3Mode.PROFILE)].semantics,
                label=f"{configuration.value}:profile-control",
            )
        )
    environment = _environment(envelopes, metadata)
    closure = _closure(runs)
    result: dict[str, object] = {
        "schema_version": ARTIFACT_SCHEMA,
        "status": "smoke-passed" if not failures else "smoke-failed",
        "formal_artifact": False,
        "run_count": len(runs),
        "sequence": [run.run_id for run in runs],
        "semantic_failure_rows": failures,
        "environment_admitted_count": sum(run.environment.admitted for run in runs),
        "all_profile_core_closed": closure["all_closed"],
        "runtime_identity_count": environment["runtime_identity_count"],
        "performance_claimed": False,
    }
    result["smoke_hash"] = canonical_hash(result)
    _write_jsonl(output / "worker_runs.jsonl", [run.to_dict() for run in runs])
    _write_jsonl(output / "run_metadata.jsonl", metadata)
    _write_json(output / "environment.json", environment)
    _write_json(output / "closure.json", closure)
    _write_json(output / "smoke_summary.json", result)
    return result


def _verify_static_artifact(
    artifact: Path,
) -> tuple[list[FSG3TimingRun], dict[str, Any], dict[str, object]]:
    manifest = _load_json(artifact / "manifest.json")
    semantic_manifest = {
        key: value for key, value in manifest.items() if key != "manifest_hash"
    }
    expected_sequence = [
        [block, position, configuration.value, mode.value]
        for block, position, configuration, mode in expected_fsg3_sequence()
    ]
    if (
        manifest.get("schema_version") != ARTIFACT_SCHEMA
        or manifest.get("manifest_hash") != canonical_hash(semantic_manifest)
        or manifest.get("expected_sequence") != expected_sequence
        or manifest.get("formal_preflight_contract") != FORMAL_PREFLIGHT_CONTRACT
        or manifest.get("performance_claimed") is not False
    ):
        raise ValueError("FSG3 manifest differs")
    _verify_code_provenance(manifest)
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise TypeError("FSG3 manifest files differ")
    observed_files = {
        str(path.relative_to(artifact))
        for path in artifact.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if set(files) != observed_files:
        raise ValueError("FSG3 artifact inventory differs")
    for name, digest in files.items():
        if not isinstance(name, str) or digest != _file_sha256(artifact / name):
            raise ValueError(f"FSG3 artifact digest differs: {name}")
    run_payloads = _load_jsonl(artifact / "worker_runs.jsonl")
    runs = [fsg3_timing_run_from_dict(value) for value in run_payloads]
    metadata = _load_jsonl(artifact / "run_metadata.jsonl")
    envelopes = [
        _load_json(artifact / f"workers/run_{index:02d}.json")
        for index in range(len(expected_fsg3_sequence()))
    ]
    if len(metadata) != len(runs) or len(envelopes) != len(runs):
        raise ValueError("FSG3 artifact run coverage differs")
    for index, (run, envelope, outer) in enumerate(zip(runs, envelopes, metadata)):
        if envelope.get("run") != run.to_dict():
            raise ValueError(f"FSG3 envelope/run projection differs: {index}")
        if (
            outer.get("index") != index
            or outer.get("returncode") != 0
            or outer.get("timed_out") is not False
        ):
            raise ValueError(f"FSG3 worker metadata differs: {index}")
    derived = _derived_payloads(runs, envelopes, metadata)
    for name, expected in derived.items():
        observed: object = (
            _load_jsonl(artifact / name)
            if name.endswith(".jsonl")
            else _load_json(artifact / name)
        )
        if observed != expected:
            raise ValueError(f"FSG3 derived replay differs: {name}")
    summary = cast(dict[str, Any], derived["summary.json"])
    if (
        manifest.get("source_identity") != runs[0].source_identity
        or manifest.get("protocol_identity") != runs[0].protocol_identity
        or manifest.get("summary_hash") != summary["summary_hash"]
        or manifest.get("status") != summary["status"]
    ):
        raise ValueError("FSG3 manifest semantic identity differs")
    result = _replay_result(summary)
    if (artifact / "replay_stdout.txt").read_text(encoding="utf-8") != (
        _canonical_json(result) + "\n"
    ):
        raise ValueError("FSG3 replay stdout differs")
    if (artifact / "README.md").read_text(encoding="utf-8") != _readme():
        raise ValueError("FSG3 README differs")
    return runs, summary, result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def add_run_arguments(command: argparse.ArgumentParser) -> None:
        command.add_argument("--artifact-dir", type=Path, required=True)
        command.add_argument("--benchmark-root", type=Path, required=True)
        command.add_argument("--abcrown-root", type=Path, required=True)
        command.add_argument("--abcrown-python", type=Path, required=True)
        command.add_argument("--model", type=Path, required=True)
        command.add_argument("--property", type=Path, required=True)

    generate = commands.add_parser("generate")
    add_run_arguments(generate)
    smoke = commands.add_parser("smoke")
    add_run_arguments(smoke)
    replay = commands.add_parser("replay")
    replay.add_argument("--artifact-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Generate or replay one complete FSG3 artifact."""

    args = _parse_args()
    if args.command == "generate":
        result = _generate(args)
    elif args.command == "smoke":
        result = _generate_smoke(args)
    else:
        _runs, _summary, result = _verify_static_artifact(args.artifact_dir.resolve())
    print(_canonical_json(result))


if __name__ == "__main__":
    main()
