#!/usr/bin/env python3
"""Generate or replay the FSG4/B4-0 raw kernel-attribution artifact."""

# pylint: disable=wrong-import-position,protected-access,too-many-locals
# pylint: disable=too-many-statements,too-many-branches,too-many-arguments
# pylint: disable=duplicate-code,import-error,missing-function-docstring
# pylint: disable=import-outside-toplevel,too-many-boolean-expressions

from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
from functools import wraps
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Callable, Iterator, Mapping, MutableMapping, Sequence, cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from boundflow.runtime.fsg3_same_solver_timing import canonical_hash as b3_hash
from boundflow.runtime.fsg4_b4_kernel_attribution import (
    b4_profiler_event_from_dict,
    canonical_hash,
    derive_b4_attribution,
    extract_profiler_events,
)
from scripts import run_fsg4_b3_same_solver_timing as b3_worker

ARTIFACT_SCHEMA = "boundflow.fsg4-b4-kernel-attribution-artifact/v1"
PROTOCOL_SCHEMA = "boundflow.fsg4-b4-kernel-attribution-protocol/v1"
WORKER_SCHEMA = "boundflow.fsg4-b4-kernel-attribution-worker/v1"
B3_FORMAL_ARTIFACT = Path("artifacts/fsg4-b3-same-solver-timing/resnet2b-prop0-v1")
CODE_PATHS = (
    "boundflow/runtime/fsg4_b4_kernel_attribution.py",
    "scripts/run_fsg4_b4_kernel_attribution.py",
    "scripts/run_fsg4_b3_same_solver_timing.py",
    "scripts/run_fsg4_b3_counter_diagnostic.py",
    "scripts/run_rvir_v4_live_return_capture.py",
    "boundflow/runtime/fsg4_b3_terminal_optimizer_schedule.py",
    "boundflow/runtime/rvir_v4_native_backward_export.py",
    "boundflow/runtime/rvir_v4_native_kfsb.py",
    "boundflow/runtime/fsg4_b3_device_atomic_commit.py",
    "boundflow/runtime/fsg4_b3_device_live_return.py",
)


def _json(value: object, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
        allow_nan=False,
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json(value, indent=2) + "\n", encoding="utf-8")


def _write_jsonl_gzip(path: Path, rows: Sequence[object]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("wb") as raw_stream:
        with gzip.GzipFile(
            filename="", mode="wb", compresslevel=9, mtime=0, fileobj=raw_stream
        ) as stream:
            for row in rows:
                encoded = (_json(row) + "\n").encode("utf-8")
                digest.update(encoded)
                stream.write(encoded)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"FSG4/B4 JSON root differs: {path.name}")
    return value


def _load_jsonl_gzip(path: Path) -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as stream:
        for encoded in stream:
            digest.update(encoded)
            value = json.loads(encoded)
            if not isinstance(value, dict):
                raise TypeError(f"FSG4/B4 JSONL row differs: {path.name}")
            rows.append(value)
    return rows, digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def _code_revision() -> dict[str, str]:
    return {path: _sha256(REPOSITORY_ROOT / path) for path in CODE_PATHS}


def _absolute_without_symlink_resolution(path: Path) -> Path:
    """Make a CLI path absolute while retaining virtualenv interpreter links."""

    return path.expanduser().absolute()


def _historical_code_revision(source: str) -> dict[str, str]:
    if _git(REPOSITORY_ROOT, "rev-parse", "HEAD") == source:
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


@contextmanager
def _patch(owner: Any, name: str, value: Any) -> Iterator[None]:
    original = getattr(owner, name)
    setattr(owner, name, value)
    try:
        yield
    finally:
        setattr(owner, name, original)


def _marked(name: str, function: Callable[..., Any]) -> Callable[..., Any]:
    from torch.profiler import record_function

    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with record_function(f"boundflow::b4::{name}"):
            return function(*args, **kwargs)

    return wrapped


def _counted_marked(
    prefix: str,
    function: Callable[..., Any],
    counts: MutableMapping[str, int],
) -> Callable[..., Any]:
    from torch.profiler import record_function

    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        ordinal = counts.get(prefix, 0)
        counts[prefix] = ordinal + 1
        with record_function(f"boundflow::b4::{prefix}.{ordinal:02d}"):
            return function(*args, **kwargs)

    return wrapped


@contextmanager
def _instrument_b4_markers() -> Iterator[MutableMapping[str, int]]:
    from boundflow.runtime import fsg4_b3_device_atomic_commit as atomic
    from boundflow.runtime import fsg4_b3_device_live_return as device_return
    from boundflow.runtime import fsg4_b3_prepared_core as prepared
    from boundflow.runtime import fsg4_b3_terminal_optimizer_schedule as terminal
    from boundflow.runtime import rvir_v4_native_backward_export as backward
    from boundflow.runtime import rvir_v4_native_kfsb as kfsb
    from boundflow.runtime import rvir_v4_pre_state_initializer as prestate
    from scripts import run_rvir_v4_live_return_capture as live

    counts: MutableMapping[str, int] = {}
    patches: tuple[tuple[Any, str, str], ...] = (
        (live._LiveExecutor, "execute", "core"),
        (prestate, "initialize_rvir_v4_native_pre_state", "pre_state.initialize"),
        (prepared, "instantiate_core_plan_v1", "pre_state.plan_instance"),
        (terminal, "execute_terminal_optimizer_schedule_v1", "optimizer"),
        (backward, "export_rvir_v4_native_backward", "terminal_export"),
        (kfsb, "evaluate_rvir_v4_native_kfsb", "kfsb"),
        (atomic, "stage_device_atomic_transaction_v1", "atomic_stage"),
        (device_return, "assemble_device_live_core_return_v1", "atomic_assembly"),
        (device_return, "commit_device_live_core_return_v1", "atomic_commit"),
    )
    with ExitStack() as stack:
        for owner, function_name, marker in patches:
            original = getattr(owner, function_name)
            stack.enter_context(_patch(owner, function_name, _marked(marker, original)))
        counted: tuple[tuple[Any, str, str], ...] = (
            (terminal, "run_crown_ibp_mlp_from_forward_trace", "optimizer.crown"),
            (
                backward,
                "run_crown_ibp_mlp_with_relu_lower_coefficients_from_forward_trace",
                "terminal_export.crown",
            ),
            (kfsb, "run_crown_ibp_mlp_from_forward_trace", "kfsb.crown"),
            (terminal, "_forward_ibp_trace_mlp", "optimizer.forward"),
            (kfsb, "_forward_ibp_trace_mlp", "kfsb.forward"),
        )
        for owner, function_name, marker in counted:
            original = getattr(owner, function_name)
            stack.enter_context(
                _patch(
                    owner,
                    function_name,
                    _counted_marked(marker, original, counts),
                )
            )
        yield counts


def _base_args(args: argparse.Namespace, result: Path) -> argparse.Namespace:
    return argparse.Namespace(
        configuration="B3",
        mode="control",
        run_id=f"b4-0-{args.kind}",
        block_index=0,
        sequence_position=0,
        benchmark_root=args.benchmark_root,
        abcrown_root=args.abcrown_root,
        model=args.model,
        property=args.property,
        result=result,
    )


def _validate_b3_envelope(value: Mapping[str, Any]) -> Mapping[str, Any]:
    run = value.get("run")
    if (
        value.get("schema_version") != b3_worker.WORKER_ENVELOPE_SCHEMA
        or value.get("performance_claimed") is not False
        or not isinstance(run, Mapping)
        or run.get("configuration") != "B3"
        or run.get("mode") != "control"
        or run.get("performance_claimed") is not False
    ):
        raise ValueError("FSG4/B4 inherited B3 worker differs")
    return cast(Mapping[str, Any], run)


def _worker(args: argparse.Namespace) -> dict[str, object]:
    import torch
    from torch.profiler import profile, ProfilerActivity

    with tempfile.TemporaryDirectory(prefix="boundflow-fsg4-b4-worker-") as raw:
        base_result = Path(raw) / "b3-worker.json"
        profiler = None
        counts: Mapping[str, int] = {}
        if args.kind == "profile":
            torch.cuda.synchronize()
            with _instrument_b4_markers() as observed:
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=False,
                ) as profiler:
                    _marked("worker", b3_worker._worker)(_base_args(args, base_result))
                counts = dict(observed)
            torch.cuda.synchronize()
        else:
            b3_worker._worker(_base_args(args, base_result))
        envelope = _load_json(base_result)
    _validate_b3_envelope(envelope)
    event_payloads: list[dict[str, object]] = []
    if profiler is not None:
        extracted = extract_profiler_events(profiler.events())
        event_payloads = [event.to_dict() for event in extracted]
    raw_event_jsonl_sha256 = _write_jsonl_gzip(args.events, event_payloads)
    result: dict[str, object] = {
        "schema_version": WORKER_SCHEMA,
        "kind": args.kind,
        "b3_envelope": envelope,
        "marker_counts": dict(sorted(counts.items())),
        "event_count": len(event_payloads),
        "raw_event_hash": canonical_hash(event_payloads),
        "raw_event_jsonl_sha256": raw_event_jsonl_sha256,
        "performance_claimed": False,
    }
    result["worker_hash"] = canonical_hash(result)
    _write_json(args.result, result)
    return result


def _protocol(args: argparse.Namespace) -> dict[str, object]:
    b3_manifest = REPOSITORY_ROOT / B3_FORMAL_ARTIFACT / "manifest.json"
    b3_manifest_payload = _load_json(b3_manifest)
    protocol: dict[str, object] = {
        "schema_version": PROTOCOL_SCHEMA,
        "source_git_head": _git(REPOSITORY_ROOT, "rev-parse", "HEAD"),
        "code_revision": _code_revision(),
        "worker_sequence": ["control", "profile"],
        "b3_formal_manifest_file_sha256": _sha256(b3_manifest),
        "b3_formal_manifest_hash": b3_manifest_payload.get("manifest_hash"),
        "abcrown_commit": _git(args.abcrown_root, "rev-parse", "HEAD"),
        "auto_lirpa_commit": _git(
            args.abcrown_root / "auto_LiRPA", "rev-parse", "HEAD"
        ),
        "vnncomp_commit": _git(args.benchmark_root, "rev-parse", "HEAD"),
        "model_name": args.model.name,
        "model_sha256": _sha256(args.model),
        "property_name": args.property.name,
        "property_sha256": _sha256(args.property),
        "profiler": {
            "activities": ["CPU", "CUDA"],
            "record_shapes": True,
            "profile_memory": True,
            "with_stack": False,
        },
        "b4_0_attribution_only": True,
        "performance_claimed": False,
    }
    protocol["protocol_hash"] = canonical_hash(protocol)
    return protocol


def _validate_protocol(value: Mapping[str, Any]) -> None:
    payload = dict(value)
    claimed = payload.pop("protocol_hash", None)
    source = value.get("source_git_head")
    revision = value.get("code_revision")
    if (
        claimed != canonical_hash(payload)
        or value.get("schema_version") != PROTOCOL_SCHEMA
        or value.get("worker_sequence") != ["control", "profile"]
        or value.get("performance_claimed") is not False
        or not isinstance(source, str)
        or not isinstance(revision, Mapping)
        or dict(revision) != _historical_code_revision(source)
    ):
        raise ValueError("FSG4/B4 protocol differs")


def _worker_command(
    args: argparse.Namespace, *, kind: str, result: Path, events: Path
) -> tuple[str, ...]:
    return (
        str(args.abcrown_python),
        str(Path(__file__).resolve()),
        "worker",
        "--kind",
        kind,
        "--benchmark-root",
        str(args.benchmark_root),
        "--abcrown-root",
        str(args.abcrown_root),
        "--model",
        str(args.model),
        "--property",
        str(args.property),
        "--result",
        str(result),
        "--events",
        str(events),
    )


def _run_fresh_worker(
    args: argparse.Namespace, *, artifact: Path, kind: str
) -> dict[str, Any]:
    result = artifact / "workers" / f"{kind}.json"
    events = artifact / "events" / f"{kind}.jsonl.gz"
    command = _worker_command(args, kind=kind, result=result, events=events)
    environment = dict(os.environ)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT)
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=900,
    )
    (artifact / "logs").mkdir(parents=True, exist_ok=True)
    stdout = _sanitize_text(completed.stdout, args)
    stderr = _sanitize_text(completed.stderr, args)
    (artifact / "logs" / f"{kind}.stdout.txt").write_text(stdout, encoding="utf-8")
    (artifact / "logs" / f"{kind}.stderr.txt").write_text(stderr, encoding="utf-8")
    if completed.returncode != 0 or not result.is_file() or not events.is_file():
        raise RuntimeError(
            f"FSG4/B4 {kind} worker failed with {completed.returncode}:\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    return _load_json(result)


def _validate_worker_python(args: argparse.Namespace) -> None:
    environment = dict(os.environ)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT)
    completed = subprocess.run(
        (
            str(args.abcrown_python),
            "-c",
            "import boundflow, torch; print(torch.__version__)",
        ),
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=60,
    )
    if completed.returncode != 0:
        raise ValueError(
            "FSG4/B4 worker interpreter cannot import torch/boundflow: "
            + _sanitize_text(completed.stderr, args)
        )


def _sanitize_text(value: str, args: argparse.Namespace) -> str:
    replacements = (
        (
            str(args.model),
            f"$VNNCOMP_ROOT/{args.model.relative_to(args.benchmark_root)}",
        ),
        (
            str(args.property),
            f"$VNNCOMP_ROOT/{args.property.relative_to(args.benchmark_root)}",
        ),
        (str(args.abcrown_python), "$ABCROWN_PYTHON"),
        (str(args.benchmark_root), "$VNNCOMP_ROOT"),
        (str(args.abcrown_root), "$ABCROWN_ROOT"),
        (str(REPOSITORY_ROOT), "$BOUNDFLOW_ROOT"),
        ("/tmp/", "$TMP/"),
    )
    sanitized = value
    for original, replacement in replacements:
        sanitized = sanitized.replace(original, replacement)
    return sanitized


def _validate_worker(value: Mapping[str, Any], *, kind: str) -> None:
    payload = dict(value)
    claimed = payload.pop("worker_hash", None)
    if (
        claimed != canonical_hash(payload)
        or value.get("schema_version") != WORKER_SCHEMA
        or value.get("kind") != kind
        or value.get("performance_claimed") is not False
    ):
        raise ValueError(f"FSG4/B4 {kind} worker binding differs")


def _run_payload(worker: Mapping[str, Any]) -> Mapping[str, Any]:
    envelope = worker.get("b3_envelope")
    if not isinstance(envelope, Mapping):
        raise TypeError("FSG4/B4 worker envelope differs")
    return _validate_b3_envelope(cast(Mapping[str, Any], envelope))


def _derive(artifact: Path) -> dict[str, object]:
    control = _load_json(artifact / "workers/control.json")
    profiled = _load_json(artifact / "workers/profile.json")
    _validate_worker(control, kind="control")
    _validate_worker(profiled, kind="profile")
    if (
        control.get("event_count") != 0
        or control.get("performance_claimed") is not False
        or profiled.get("performance_claimed") is not False
    ):
        raise ValueError("FSG4/B4 worker pair differs")
    control_run = _run_payload(control)
    profile_run = _run_payload(profiled)
    if control_run["semantics"] != profile_run["semantics"]:
        raise ValueError("FSG4/B4 control/profile semantics differ")
    event_payloads, jsonl_sha256 = _load_jsonl_gzip(
        artifact / "events/profile.jsonl.gz"
    )
    events = tuple(b4_profiler_event_from_dict(row) for row in event_payloads)
    if (
        profiled.get("event_count") != len(events)
        or profiled.get("raw_event_hash")
        != canonical_hash([event.to_dict() for event in events])
        or profiled.get("raw_event_jsonl_sha256") != jsonl_sha256
    ):
        raise ValueError("FSG4/B4 profile raw event binding differs")
    profile_metrics = cast(Mapping[str, Any], profile_run["metrics"])
    control_metrics = cast(Mapping[str, Any], control_run["metrics"])
    attribution = derive_b4_attribution(
        events,
        run_id=str(profile_run["run_id"]),
        source_identity=str(profile_run["source_identity"]),
        protocol_identity=str(profile_run["protocol_identity"]),
        query_wall_ns=int(profile_metrics["query_wall_ns"]),
        core_wall_ns=int(profile_metrics["core_wall_ns"]),
    )
    marker_counts = profiled.get("marker_counts")
    if not isinstance(marker_counts, Mapping):
        raise TypeError("FSG4/B4 marker counters differ")
    expected_counts = {
        "optimizer.crown": 10,
        "optimizer.forward": 1,
        "terminal_export.crown": 1,
        "kfsb.crown": 3,
        "kfsb.forward": 3,
    }
    if {name: int(marker_counts.get(name, -1)) for name in expected_counts} != (
        expected_counts
    ):
        raise ValueError("FSG4/B4 14-call marker coverage differs")
    summary: dict[str, object] = {
        "schema_version": ARTIFACT_SCHEMA,
        "status": "measured-attribution-only",
        "control_worker_hash": control["worker_hash"],
        "profile_worker_hash": profiled["worker_hash"],
        "semantic_hash": b3_hash(control_run["semantics"]),
        "marker_counts": dict(
            sorted((str(k), int(v)) for k, v in marker_counts.items())
        ),
        "profile_over_control": {
            metric: int(profile_metrics[metric]) / int(control_metrics[metric])
            for metric in ("query_wall_ns", "core_wall_ns")
        },
        "attribution": attribution,
        "b4_0_attribution_only": True,
        "performance_claimed": False,
    }
    summary["summary_hash"] = canonical_hash(summary)
    return summary


def _readme() -> str:
    return (
        "# FSG4/B4-0 kernel attribution artifact\n\n"
        "This artifact contains one fresh unprofiled B3 control and one fresh "
        "B3 worker observed with PyTorch CPU/CUDA profiling. Profile time is "
        "attribution-only and never a performance claim. Replay rebuilds the "
        "14-call marker coverage, raw kernel aggregation, and Amdahl gates.\n"
    )


def _generate(args: argparse.Namespace) -> dict[str, object]:
    if _git(REPOSITORY_ROOT, "status", "--porcelain=v1", "--", *CODE_PATHS):
        raise ValueError("FSG4/B4 code paths must be clean before generation")
    _validate_worker_python(args)
    artifact = args.artifact_dir.resolve()
    if artifact.exists() and any(artifact.iterdir()):
        raise FileExistsError(f"FSG4/B4 artifact already exists: {artifact.name}")
    artifact.mkdir(parents=True, exist_ok=True)
    protocol = _protocol(args)
    _write_json(artifact / "protocol.json", protocol)
    for kind in ("control", "profile"):
        _run_fresh_worker(args, artifact=artifact, kind=kind)
    summary = _derive(artifact)
    _write_json(artifact / "summary.json", summary)
    replay = {
        "status": summary["status"],
        "summary_hash": summary["summary_hash"],
        "cuda_kernel_count": cast(Mapping[str, Any], summary["attribution"])[
            "cuda_kernel_count"
        ],
        "performance_claimed": False,
    }
    (artifact / "replay_stdout.txt").write_text(_json(replay) + "\n", encoding="utf-8")
    (artifact / "README.md").write_text(_readme(), encoding="utf-8")
    for path in artifact.rglob("*"):
        if path.is_file() and path.suffix != ".gz":
            value = path.read_text(encoding="utf-8")
            if any(token in value for token in ("/home/", "/tmp/", "file://")):
                raise ValueError(f"FSG4/B4 artifact leaks a local path: {path.name}")
    files = sorted(
        str(path.relative_to(artifact))
        for path in artifact.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    manifest: dict[str, object] = {
        "schema_version": ARTIFACT_SCHEMA,
        "source_git_head": protocol["source_git_head"],
        "protocol_hash": protocol["protocol_hash"],
        "summary_hash": summary["summary_hash"],
        "files": {name: _sha256(artifact / name) for name in files},
        "worker_count": 2,
        "performance_claimed": False,
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    _write_json(artifact / "manifest.json", manifest)
    return replay


def _replay(artifact: Path) -> dict[str, object]:
    artifact = artifact.resolve()
    manifest = _load_json(artifact / "manifest.json")
    payload = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    if (
        manifest.get("manifest_hash") != canonical_hash(payload)
        or manifest.get("schema_version") != ARTIFACT_SCHEMA
        or manifest.get("worker_count") != 2
        or manifest.get("performance_claimed") is not False
    ):
        raise ValueError("FSG4/B4 manifest differs")
    protocol = _load_json(artifact / "protocol.json")
    _validate_protocol(protocol)
    if (
        manifest.get("source_git_head") != protocol["source_git_head"]
        or manifest.get("protocol_hash") != protocol["protocol_hash"]
    ):
        raise ValueError("FSG4/B4 manifest protocol binding differs")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise TypeError("FSG4/B4 manifest file inventory differs")
    observed = {
        str(path.relative_to(artifact))
        for path in artifact.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if set(files) != observed:
        raise ValueError("FSG4/B4 artifact file inventory differs")
    for name, digest in files.items():
        if not isinstance(name, str) or digest != _sha256(artifact / name):
            raise ValueError(f"FSG4/B4 artifact digest differs: {name}")
    summary = _derive(artifact)
    if _load_json(artifact / "summary.json") != summary:
        raise ValueError("FSG4/B4 summary replay differs")
    if manifest.get("summary_hash") != summary["summary_hash"]:
        raise ValueError("FSG4/B4 manifest summary binding differs")
    result = {
        "status": summary["status"],
        "summary_hash": summary["summary_hash"],
        "cuda_kernel_count": cast(Mapping[str, Any], summary["attribution"])[
            "cuda_kernel_count"
        ],
        "performance_claimed": False,
    }
    if (artifact / "replay_stdout.txt").read_text(encoding="utf-8") != (
        _json(result) + "\n"
    ) or (artifact / "README.md").read_text(encoding="utf-8") != _readme():
        raise ValueError("FSG4/B4 replay projection differs")
    return result


def _path_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--abcrown-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--property", type=Path, required=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    worker = commands.add_parser("worker")
    worker.add_argument("--kind", choices=("control", "profile"), required=True)
    _path_args(worker)
    worker.add_argument("--result", type=Path, required=True)
    worker.add_argument("--events", type=Path, required=True)
    generate = commands.add_parser("generate")
    _path_args(generate)
    generate.add_argument("--abcrown-python", type=Path, required=True)
    generate.add_argument("--artifact-dir", type=Path, required=True)
    replay = commands.add_parser("replay")
    replay.add_argument("--artifact-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "worker":
        args.benchmark_root = args.benchmark_root.resolve()
        args.abcrown_root = args.abcrown_root.resolve()
        args.model = args.model.resolve()
        args.property = args.property.resolve()
        args.result = args.result.resolve()
        args.events = args.events.resolve()
        result = _worker(args)
    elif args.command == "generate":
        args.benchmark_root = args.benchmark_root.resolve()
        args.abcrown_root = args.abcrown_root.resolve()
        args.abcrown_python = _absolute_without_symlink_resolution(args.abcrown_python)
        args.model = args.model.resolve()
        args.property = args.property.resolve()
        result = _generate(args)
    else:
        result = _replay(args.artifact_dir)
    print(_json(result))


if __name__ == "__main__":
    main()
