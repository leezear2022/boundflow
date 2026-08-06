#!/usr/bin/env python3
"""Generate or replay the FSG1 official αβ-CROWN B0 GPU baseline."""

# pylint: disable=too-many-lines,too-many-locals,too-many-statements
# pylint: disable=too-many-branches,too-many-arguments,import-outside-toplevel
# pylint: disable=protected-access,missing-function-docstring,line-too-long

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, cast, Iterator, Mapping, Sequence

ARTIFACT_SCHEMA_VERSION = "boundflow.fsg1-official-control-artifact/v1"
OFFICIAL_CONTROL_WORKER_SCHEMA_VERSION = "boundflow.fsg1-official-control-worker/v1"
MANIFEST_FILE = "manifest.json"
VNNCOMP_COMMIT = "90419aadcf06cf543ce5c1706cae1059dc9fa6cf"
ABCROWN_COMMIT = "e5c7e17bf0488843acb77b7519f59876717a49f4"
AUTO_LIRPA_COMMIT = "5a098e8f9fb5786a428a024981d833d303921f2d"
DEFAULT_REPEAT_COUNT = 5
DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_ALPHA_STEPS = 5
DEFAULT_BETA_STEPS = 10
DEFAULT_BATCH_SIZE = 64
WORKLOADS = (
    {
        "workload_id": "cifar10_resnet:000",
        "model": "benchmarks/cifar10_resnet/onnx/resnet_2b.onnx",
        "property": (
            "benchmarks/cifar10_resnet/vnnlib_properties_pgd_filtered/"
            "resnet2b_pgd_filtered/prop_0_eps_0.008.vnnlib"
        ),
        "model_sha256": "791aa24d77917ecda16809fbbd48e7739616f88ebf74cf358b2d1bf911dc4a6d",
        "property_sha256": "89edf0665d74397670d0562d513db694a49a84edaf5cf3d64c9c6fa63c3769ff",
    },
    {
        "workload_id": "mnistfc:2",
        "model": "benchmarks/mnistfc/mnist-net_256x2.onnx",
        "property": "benchmarks/mnistfc/prop_2_0.03.vnnlib",
        "model_sha256": "3a5c9730d60bbf1f9b030e731b438436581efd7c00a28ab683c1ec4b6d3449c4",
        "property_sha256": "0c36c00722b6c1701f4d5f17b9d28117711f351c6773e450653cc728a2dd224b",
    },
)
ARTIFACT_FILES = (
    "environment.json",
    "workloads.jsonl",
    "worker_runs.jsonl",
    "paired_runs.jsonl",
    "raw_events.jsonl",
    "normalized_spans.jsonl",
    "dependency_edges.jsonl",
    "closure.json",
    "ablation.json",
    "summary.json",
    "failure_rows.jsonl",
    "replay_stdout.txt",
    "README.md",
)
CODE_PATHS = (
    "boundflow/runtime/gpu_attribution.py",
    "boundflow/runtime/official_control_attribution.py",
    "scripts/run_fsg1_official_control_baseline.py",
)


def canonical_json(value: object, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
        allow_nan=False,
    )


def canonical_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _git_value(root: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _git_revision(root: Path) -> str:
    return _git_value(root, "rev-parse", "HEAD")


def _write_json(path: Path, value: object) -> None:
    path.write_text(canonical_json(value, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(canonical_json(dict(row)) + "\n" for row in rows),
        encoding="utf-8",
    )


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"JSONL row must be an object: {path}")
        rows.append(value)
    return rows


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _code_revision() -> dict[str, str]:
    root = _repo_root()
    return {path: file_sha256(root / path) for path in CODE_PATHS}


def _code_paths_clean() -> bool:
    root = _repo_root()
    return not _git_value(root, "status", "--porcelain=v1", "--", *CODE_PATHS)


def _verify_code_provenance(manifest: Mapping[str, Any]) -> None:
    root = _repo_root()
    source_head = manifest.get("source_git_head")
    revision = manifest.get("code_revision")
    if not isinstance(source_head, str) or not isinstance(revision, Mapping):
        raise ValueError("FSG1 source provenance differs")
    current_head = _git_revision(root)
    if current_head == source_head:
        observed = _code_revision()
    else:
        observed = {}
        for path in CODE_PATHS:
            blob = subprocess.run(
                ("git", "show", f"{source_head}:{path}"),
                cwd=root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ).stdout
            observed[path] = _sha256_bytes(blob)
    if dict(revision) != observed:
        raise ValueError("FSG1 source code revision differs")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--benchmark-root", type=Path, required=True)
    generate.add_argument("--abcrown-root", type=Path, required=True)
    generate.add_argument("--abcrown-python", type=Path, required=True)
    generate.add_argument("--artifact-dir", type=Path, required=True)
    generate.add_argument("--repeats", type=int, default=DEFAULT_REPEAT_COUNT)
    generate.add_argument(
        "--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS
    )
    generate.add_argument("--workload", action="append")
    replay = commands.add_parser("replay")
    replay.add_argument("--artifact-dir", type=Path, required=True)
    worker = commands.add_parser("worker")
    worker.add_argument("--mode", choices=("control", "profile"), required=True)
    worker.add_argument("--run-id", required=True)
    worker.add_argument("--workload-id", required=True)
    worker.add_argument("--repeat-index", type=int, required=True)
    worker.add_argument("--pair-order", required=True)
    worker.add_argument("--model", type=Path, required=True)
    worker.add_argument("--property", type=Path, required=True)
    worker.add_argument("--model-relative-path", required=True)
    worker.add_argument("--property-relative-path", required=True)
    worker.add_argument("--benchmark-root", type=Path, required=True)
    worker.add_argument("--abcrown-root", type=Path, required=True)
    worker.add_argument("--result-json", type=Path, required=True)
    worker.add_argument("--timeout-seconds", type=int, required=True)
    worker.add_argument("--alpha-steps", type=int, default=DEFAULT_ALPHA_STEPS)
    worker.add_argument("--beta-steps", type=int, default=DEFAULT_BETA_STEPS)
    worker.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    return parser.parse_args()


def _phase_from_stack(method: str) -> tuple[str, str]:
    external = "unclassified_compute_bounds"
    for frame in inspect.stack(context=0)[2:20]:
        filename = frame.filename.replace("\\", "/")
        if frame.function == "update_bounds_core":
            external = "activation_bab_bound"
            return "beta_split", external
        if "/input_split/" in filename:
            external = "input_bab_bound"
            return "beta_split", external
        if "incomplete_verifier" in filename:
            external = "incomplete_verification"
            return "initial_crown", external
        if "beta_CROWN_solver" in filename:
            external = "alpha_crown_initialization"
            normalized = method.lower().replace("_", "-")
            return (
                "alpha_optimize" if "optimized" in normalized else "initial_crown",
                external,
            )
    normalized = method.lower().replace("_", "-")
    if "optimized" in normalized:
        return "alpha_optimize", external
    return "unclassified", external


class _OfficialCallObserver:
    def __init__(self, torch_module: Any, scope_started_ns: int, anchor: Any) -> None:
        self._torch = torch_module
        self._scope_started_ns = scope_started_ns
        self._anchor = anchor
        self._stack: list[int] = []
        self._pending: list[dict[str, Any]] = []

    @contextmanager
    def instrument(self, bounded_module: Any) -> Iterator[None]:
        original = bounded_module.compute_bounds

        def wrapped(instance: Any, *args: Any, **kwargs: Any) -> Any:
            call_id = len(self._pending)
            parent = self._stack[-1] if self._stack else None
            method = str(kwargs.get("method", "backward"))
            phase, external_phase = _phase_from_stack(method)
            stream = self._torch.cuda.current_stream()
            start_event = self._torch.cuda.Event(enable_timing=True)
            end_event = self._torch.cuda.Event(enable_timing=True)
            host_start = time.perf_counter_ns() - self._scope_started_ns
            start_event.record(stream)
            pending = {
                "call_id": call_id,
                "parent_call_id": parent,
                "depth": len(self._stack),
                "method": method,
                "phase": phase,
                "external_phase": external_phase,
                "host_start_ns": host_start,
                "host_end_ns": None,
                "cuda_start_event": start_event,
                "cuda_end_event": end_event,
                "stream_id": str(stream.cuda_stream),
                "memory_allocated_before_bytes": int(
                    self._torch.cuda.memory_allocated()
                ),
                "memory_allocated_after_bytes": None,
                "memory_reserved_before_bytes": int(self._torch.cuda.memory_reserved()),
                "memory_reserved_after_bytes": None,
                "bound_lower": bool(kwargs.get("bound_lower", True)),
                "bound_upper": bool(kwargs.get("bound_upper", True)),
                "kwargs_keys": sorted(kwargs),
            }
            self._pending.append(pending)
            self._stack.append(call_id)
            try:
                return original(instance, *args, **kwargs)
            finally:
                end_event.record(stream)
                pending["host_end_ns"] = time.perf_counter_ns() - self._scope_started_ns
                pending["memory_allocated_after_bytes"] = int(
                    self._torch.cuda.memory_allocated()
                )
                pending["memory_reserved_after_bytes"] = int(
                    self._torch.cuda.memory_reserved()
                )
                popped = self._stack.pop()
                if popped != call_id:
                    raise RuntimeError("FSG1 observer call stack differs")

        bounded_module.compute_bounds = wrapped
        try:
            yield
        finally:
            bounded_module.compute_bounds = original

    def finish(self, *, scope_ns: int) -> list[dict[str, object]]:
        calls: list[dict[str, object]] = []
        for pending in sorted(self._pending, key=lambda item: int(item["call_id"])):
            cuda_start = int(
                round(self._anchor.elapsed_time(pending["cuda_start_event"]) * 1e6)
            )
            cuda_end = int(
                round(self._anchor.elapsed_time(pending["cuda_end_event"]) * 1e6)
            )
            calls.append(
                {
                    key: value
                    for key, value in pending.items()
                    if key not in {"cuda_start_event", "cuda_end_event"}
                }
                | {
                    "cuda_start_ns": min(cuda_start, scope_ns),
                    "cuda_end_ns": min(max(cuda_start, cuda_end), scope_ns),
                }
            )
        return calls


def _visited_domains(result: Any) -> list[int]:
    stats = getattr(result, "stats", None)
    if not isinstance(stats, dict) or not isinstance(stats.get("bab"), list):
        return []
    return [
        int(row[2])
        for row in stats["bab"]
        if isinstance(row, (tuple, list)) and len(row) >= 3
    ]


def _worker(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(args.abcrown_root / "complete_verifier"))
    sys.path.insert(0, str(args.abcrown_root))
    import torch

    # pylint: disable-next=import-error
    from abcrown import ABCrownSolver, ConfigBuilder, IOConstraints  # type: ignore[import-not-found]
    from auto_LiRPA import BoundedModule  # type: ignore[import-untyped]

    if not torch.cuda.is_available():
        raise RuntimeError("FSG1 official worker requires CUDA")
    if _git_revision(args.abcrown_root) != ABCROWN_COMMIT:
        raise ValueError("FSG1 alpha-beta-CROWN commit differs")
    if _git_revision(args.abcrown_root / "auto_LiRPA") != AUTO_LIRPA_COMMIT:
        raise ValueError("FSG1 auto_LiRPA commit differs")
    if _git_revision(args.benchmark_root) != VNNCOMP_COMMIT:
        raise ValueError("FSG1 VNN-COMP commit differs")
    # Kept alive across solver construction/verification and closed explicitly below.
    # pylint: disable-next=consider-using-with
    property_workspace = tempfile.TemporaryDirectory(prefix="boundflow-fsg1-property-")
    isolated_property = Path(property_workspace.name) / args.property.name
    shutil.copy2(args.property, isolated_property)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    anchor = torch.cuda.Event(enable_timing=True)
    finish_event = torch.cuda.Event(enable_timing=True)
    stream = torch.cuda.current_stream()
    anchor.record(stream)
    scope_started_ns = time.perf_counter_ns()
    observer = _OfficialCallObserver(torch, scope_started_ns, anchor)
    config = (
        ConfigBuilder.from_defaults()
        .set("general/device", "cuda")
        .set("general/complete_verifier", "bab")
        .set("attack/pgd_order", "skip")
        .set("bab/timeout", args.timeout_seconds)
        .set("solver/batch_size", args.batch_size)
        .set("solver/alpha-crown/iteration", args.alpha_steps)
        .set("solver/beta-crown/iteration", args.beta_steps)
    )
    context = (
        observer.instrument(BoundedModule) if args.mode == "profile" else nullcontext()
    )
    with context:
        solver = ABCrownSolver(str(args.model), config=config)
        result = solver.verify(
            constraints=IOConstraints(vnnlib_path=str(isolated_property))
        )
    finish_event.record(stream)
    torch.cuda.synchronize()
    host_scope_ns = time.perf_counter_ns() - scope_started_ns
    cuda_scope_ns = int(round(anchor.elapsed_time(finish_event) * 1e6))
    scope_ns = max(host_scope_ns, cuda_scope_ns)
    calls = observer.finish(scope_ns=scope_ns) if args.mode == "profile" else []
    properties = torch.cuda.get_device_properties(0)
    record = {
        "schema_version": OFFICIAL_CONTROL_WORKER_SCHEMA_VERSION,
        "run_id": args.run_id,
        "configuration_id": "B0",
        "workload_id": args.workload_id,
        "mode": args.mode,
        "repeat_index": args.repeat_index,
        "pair_order": args.pair_order,
        "source": {
            "abcrown_commit": _git_revision(args.abcrown_root),
            "auto_lirpa_commit": _git_revision(args.abcrown_root / "auto_LiRPA"),
            "vnncomp_commit": _git_revision(args.benchmark_root),
            "model_relative_path": args.model_relative_path,
            "property_relative_path": args.property_relative_path,
            "model_sha256": file_sha256(args.model),
            "property_sha256": file_sha256(args.property),
        },
        "protocol": {
            "device": "cuda",
            "timeout_seconds": args.timeout_seconds,
            "alpha_steps": args.alpha_steps,
            "beta_steps": args.beta_steps,
            "batch_size": args.batch_size,
            "complete_verifier": "bab",
            "attack_policy": "skip",
            "synchronize_outer_scope": True,
            "property_cache": "cold_isolated_copy",
        },
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torch_cuda": str(torch.version.cuda),
            "gpu_name": properties.name,
            "gpu_total_memory": int(properties.total_memory),
        },
        "result": {
            "status": str(result.status),
            "success": bool(result.success),
            "visited_domains": _visited_domains(result),
        },
        "scope_ns": scope_ns,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "calls": calls,
        "performance_claimed": False,
    }
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.result_json, record)
    print(
        canonical_json(
            {
                "status": record["result"]["status"],
                "run_id": args.run_id,
                "scope_ns": scope_ns,
                "call_count": len(calls),
            }
        ),
        flush=True,
    )
    property_workspace.cleanup()


def _external_env() -> dict[str, str]:
    environment = dict(os.environ)
    for name in (
        "BOUNDFLOW_ROOT",
        "PYTHONPATH",
        "TVM_HOME",
        "TVM_LIBRARY_PATH",
    ):
        environment.pop(name, None)
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def _selected_workloads(names: Sequence[str] | None) -> tuple[Mapping[str, str], ...]:
    selected = set(names or [str(item["workload_id"]) for item in WORKLOADS])
    known = {str(item["workload_id"]) for item in WORKLOADS}
    if not selected or not selected <= known:
        raise ValueError("FSG1 selected workload differs")
    return tuple(item for item in WORKLOADS if item["workload_id"] in selected)


def _validate_inputs(
    benchmark_root: Path,
    abcrown_root: Path,
    abcrown_python: Path,
    workloads: Sequence[Mapping[str, str]],
) -> None:
    if _git_revision(benchmark_root) != VNNCOMP_COMMIT:
        raise ValueError("FSG1 VNN-COMP commit differs")
    if _git_revision(abcrown_root) != ABCROWN_COMMIT:
        raise ValueError("FSG1 alpha-beta-CROWN commit differs")
    if _git_revision(abcrown_root / "auto_LiRPA") != AUTO_LIRPA_COMMIT:
        raise ValueError("FSG1 auto_LiRPA commit differs")
    if not abcrown_python.is_file():
        raise FileNotFoundError("FSG1 official Python is missing")
    for workload in workloads:
        model = benchmark_root / workload["model"]
        property_path = benchmark_root / workload["property"]
        if (
            not model.is_file()
            or not property_path.is_file()
            or file_sha256(model) != workload["model_sha256"]
            or file_sha256(property_path) != workload["property_sha256"]
        ):
            raise ValueError("FSG1 workload source differs")


def _run_external_worker(
    *,
    abcrown_python: Path,
    abcrown_root: Path,
    benchmark_root: Path,
    workload: Mapping[str, str],
    mode: str,
    repeat_index: int,
    pair_order: str,
    timeout_seconds: int,
    result_path: Path,
) -> dict[str, Any]:
    run_id = f"{workload['workload_id'].replace(':', '-')}-r{repeat_index}-{mode}"
    command = (
        str(abcrown_python),
        str(Path(__file__).resolve()),
        "worker",
        "--mode",
        mode,
        "--run-id",
        run_id,
        "--workload-id",
        workload["workload_id"],
        "--repeat-index",
        str(repeat_index),
        "--pair-order",
        pair_order,
        "--model",
        str(benchmark_root / workload["model"]),
        "--property",
        str(benchmark_root / workload["property"]),
        "--model-relative-path",
        workload["model"],
        "--property-relative-path",
        workload["property"],
        "--benchmark-root",
        str(benchmark_root),
        "--abcrown-root",
        str(abcrown_root),
        "--result-json",
        str(result_path),
        "--timeout-seconds",
        str(timeout_seconds),
    )
    completed = subprocess.run(
        command,
        cwd=_repo_root(),
        env=_external_env(),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_seconds + 120,
    )
    if completed.returncode != 0 or not result_path.is_file():
        raise RuntimeError(
            f"FSG1 worker {run_id} failed with {completed.returncode}: "
            f"{completed.stdout[-8000:]}"
        )
    record = _load_json(result_path)
    print(completed.stdout.strip()[-2000:], flush=True)
    return record


def _derived_payloads(
    worker_records: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    from boundflow.runtime.official_control_attribution import (
        derive_official_control_evidence,
    )

    evidence = derive_official_control_evidence(worker_records)
    runs = [dict(item) for item in cast(Sequence[Mapping[str, Any]], evidence["runs"])]
    run_summaries = [
        dict(item)
        for item in cast(Sequence[Mapping[str, Any]], evidence["run_summaries"])
    ]
    pairs = [
        dict(item) for item in cast(Sequence[Mapping[str, Any]], evidence["pairs"])
    ]
    profile_records = {
        str(record["run_id"]): record
        for record in worker_records
        if record["mode"] == "profile"
    }
    raw_events: list[dict[str, object]] = []
    normalized: list[dict[str, object]] = []
    edges: list[dict[str, object]] = []
    for run in runs:
        run_id = str(run["run_id"])
        for span in run["spans"]:
            raw_events.append({"run_id": run_id, "span": span})
            parent = span["parent_span_id"]
            if parent is not None:
                edges.append(
                    {
                        "run_id": run_id,
                        "kind": "parent",
                        "source": parent,
                        "target": span["span_id"],
                    }
                )
            for dependency in span["dependency_span_ids"]:
                edges.append(
                    {
                        "run_id": run_id,
                        "kind": "dependency",
                        "source": dependency,
                        "target": span["span_id"],
                    }
                )
        for segment in run["critical_path"]:
            normalized.append({"run_id": run_id, "segment": segment})
    workloads: list[dict[str, object]] = []
    for workload_id in sorted({str(row["workload_id"]) for row in worker_records}):
        rows = [row for row in worker_records if row["workload_id"] == workload_id]
        first = rows[0]
        workloads.append(
            {
                "workload_id": workload_id,
                "source": first["source"],
                "protocol": first["protocol"],
                "expected_result": first["result"],
            }
        )
    environment_rows = [dict(record["environment"]) for record in worker_records]
    if any(row != environment_rows[0] for row in environment_rows[1:]):
        raise ValueError("FSG1 worker environments differ")
    summary = {
        key: value
        for key, value in evidence.items()
        if key not in {"pairs", "runs", "run_summaries"}
    }
    summary["profile_run_count"] = len(runs)
    summary["control_run_count"] = sum(
        record["mode"] == "control" for record in worker_records
    )
    summary["profile_call_counts"] = {
        run_id: len(record["calls"])
        for run_id, record in sorted(profile_records.items())
    }
    summary["summary_hash"] = canonical_hash(summary)
    return {
        "environment.json": {
            "environment": environment_rows[0],
            "source": workloads[0]["source"],
            "performance_claimed": False,
        },
        "workloads.jsonl": workloads,
        "paired_runs.jsonl": pairs,
        "raw_events.jsonl": raw_events,
        "normalized_spans.jsonl": normalized,
        "dependency_edges.jsonl": edges,
        "closure.json": {
            "runs": run_summaries,
            "all_attribution_passed": all(
                item["attribution_passed"] is True for item in run_summaries
            ),
            "performance_claimed": False,
        },
        "ablation.json": {
            "status": "not_applicable_b0_control_only",
            "configuration_id": "B0",
            "performance_claimed": False,
        },
        "summary.json": summary,
        "failure_rows.jsonl": [],
    }


def _payload_text(name: str, value: object) -> str:
    if name.endswith(".jsonl"):
        if not isinstance(value, list):
            raise TypeError(f"FSG1 JSONL payload differs: {name}")
        return "".join(canonical_json(item) + "\n" for item in value)
    return canonical_json(value, indent=2) + "\n"


def _replay_result(summary: Mapping[str, Any]) -> dict[str, object]:
    return {
        "status": "replay-passed",
        "evidence_status": summary["status"],
        "pair_count": summary["pair_count"],
        "profile_run_count": summary["profile_run_count"],
        "summary_hash": summary["summary_hash"],
        "performance_claimed": False,
    }


def _readme() -> str:
    return (
        "# FSG1 Official αβ-CROWN B0 Full-Stack Control\n\n"
        "This artifact records five fresh control/profile pairs per frozen workload.\n"
        "It validates the B0 denominator and attribution instrumentation only.\n"
        "It does not claim BoundFlow performance.\n"
    )


def _generate(args: argparse.Namespace) -> dict[str, object]:
    if args.repeats < 1:
        raise ValueError("FSG1 repeats must be positive")
    if not _code_paths_clean():
        raise ValueError(
            "FSG1 source code paths must be clean before formal generation"
        )
    artifact_dir = args.artifact_dir.resolve()
    if artifact_dir.exists() and any(artifact_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {artifact_dir}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    benchmark_root = args.benchmark_root.resolve()
    abcrown_root = args.abcrown_root.resolve()
    abcrown_python = Path(os.path.abspath(args.abcrown_python))
    workloads = _selected_workloads(args.workload)
    _validate_inputs(benchmark_root, abcrown_root, abcrown_python, workloads)
    worker_records: list[dict[str, Any]] = []
    failures: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="boundflow-fsg1-") as temporary:
        temporary_root = Path(temporary)
        for workload in workloads:
            for repeat_index in range(args.repeats):
                modes = (
                    ("control", "profile")
                    if repeat_index % 2 == 0
                    else ("profile", "control")
                )
                pair_order = "-".join(modes)
                for mode in modes:
                    result_path = temporary_root / (
                        f"{workload['workload_id'].replace(':', '-')}-"
                        f"r{repeat_index}-{mode}.json"
                    )
                    try:
                        worker_records.append(
                            _run_external_worker(
                                abcrown_python=abcrown_python,
                                abcrown_root=abcrown_root,
                                benchmark_root=benchmark_root,
                                workload=workload,
                                mode=mode,
                                repeat_index=repeat_index,
                                pair_order=pair_order,
                                timeout_seconds=args.timeout_seconds,
                                result_path=result_path,
                            )
                        )
                    except Exception as error:  # pylint: disable=broad-exception-caught
                        failures.append(
                            {
                                "workload_id": workload["workload_id"],
                                "repeat_index": repeat_index,
                                "mode": mode,
                                "error_type": type(error).__name__,
                                "error": str(error),
                            }
                        )
                        break
                if failures:
                    break
            if failures:
                break
    _write_jsonl(artifact_dir / "worker_runs.jsonl", worker_records)
    _write_jsonl(artifact_dir / "failure_rows.jsonl", failures)
    if failures:
        raise RuntimeError("FSG1 formal worker failed; see failure_rows.jsonl")
    payloads = _derived_payloads(worker_records)
    for name, payload in payloads.items():
        (artifact_dir / name).write_text(_payload_text(name, payload), encoding="utf-8")
    summary = _load_json(artifact_dir / "summary.json")
    replay_result = _replay_result(summary)
    (artifact_dir / "replay_stdout.txt").write_text(
        canonical_json(replay_result) + "\n", encoding="utf-8"
    )
    (artifact_dir / "README.md").write_text(_readme(), encoding="utf-8")
    root = _repo_root()
    manifest: dict[str, object] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": summary["status"],
        "source_git_head": _git_revision(root),
        "source_dirty_paths": (
            _git_value(root, "status", "--porcelain=v1") or ""
        ).splitlines(),
        "code_revision": _code_revision(),
        "files": {name: file_sha256(artifact_dir / name) for name in ARTIFACT_FILES},
        "summary_hash": summary["summary_hash"],
        "pair_count": summary["pair_count"],
        "performance_claimed": False,
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    _write_json(artifact_dir / MANIFEST_FILE, manifest)
    return replay_result


def _replay(artifact_dir: Path) -> dict[str, object]:
    manifest = _load_json(artifact_dir / MANIFEST_FILE)
    semantic_manifest = {
        key: value for key, value in manifest.items() if key != "manifest_hash"
    }
    envelope_checks = (
        manifest.get("schema_version") == ARTIFACT_SCHEMA_VERSION,
        manifest.get("performance_claimed") is False,
        manifest.get("manifest_hash") == canonical_hash(semantic_manifest),
    )
    if not all(envelope_checks):
        raise ValueError("FSG1 manifest envelope differs")
    _verify_code_provenance(manifest)
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != set(ARTIFACT_FILES):
        raise ValueError("FSG1 artifact file inventory differs")
    for name, digest in files.items():
        if file_sha256(artifact_dir / name) != digest:
            raise ValueError("FSG1 artifact file digest differs")
    worker_records = _read_jsonl(artifact_dir / "worker_runs.jsonl")
    payloads = _derived_payloads(worker_records)
    for name, payload in payloads.items():
        if (artifact_dir / name).read_text(encoding="utf-8") != _payload_text(
            name, payload
        ):
            raise ValueError(f"FSG1 semantic replay differs: {name}")
    summary = _load_json(artifact_dir / "summary.json")
    if (
        manifest.get("status") != summary["status"]
        or manifest.get("summary_hash") != summary["summary_hash"]
        or manifest.get("pair_count") != summary["pair_count"]
    ):
        raise ValueError("FSG1 manifest summary projection differs")
    result = _replay_result(summary)
    if (artifact_dir / "replay_stdout.txt").read_text(encoding="utf-8") != (
        canonical_json(result) + "\n"
    ):
        raise ValueError("FSG1 replay stdout differs")
    if (artifact_dir / "README.md").read_text(encoding="utf-8") != _readme():
        raise ValueError("FSG1 README differs")
    return result


def main() -> None:
    args = _parse_args()
    if args.command == "worker":
        _worker(args)
        return
    if args.command == "generate":
        result = _generate(args)
    else:
        result = _replay(args.artifact_dir.resolve())
    print(canonical_json(result))


if __name__ == "__main__":
    main()
