#!/usr/bin/env python3
"""Generate or replay one explicit-counter FSG4/B3 B2 diagnostic artifact."""

# pylint: disable=protected-access,too-many-locals,too-many-statements
# pylint: disable=too-many-branches,wrong-import-position,import-outside-toplevel
# pylint: disable=too-many-boolean-expressions,missing-function-docstring

from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
from functools import wraps
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Callable, Iterator, Mapping, cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from boundflow.runtime.fsg3_same_solver_timing import (
    _semantic_pair_failures,
    canonical_hash,
    expected_fsg3_sequence,
    FSG3Configuration,
    FSG3Mode,
    fsg3_timing_run_from_dict,
)
from boundflow.runtime.fsg4_b3_explicit_counters import (
    events_from_rows,
    Fsg4B3CounterRecorder,
    Fsg4B3CounterSnapshot,
    fsg4_b3_counter_snapshot_from_dict,
)
from scripts import run_fsg3_same_solver_timing as fsg3_worker
from scripts import run_rvir_v4_production_state_capture as capture_runner

ARTIFACT_SCHEMA = "boundflow.fsg4-b3-counter-diagnostic-artifact/v1"
MANIFEST_SCHEMA = "boundflow.fsg4-b3-counter-diagnostic-manifest/v1"
FSG3_REFERENCE_ARTIFACT = (
    REPOSITORY_ROOT / "artifacts/fsg3-same-solver-timing/resnet2b-prop0-v5"
)
FSG3_REFERENCE_MANIFEST_HASH = (
    "9089e2019eb5e98cac228151cb061c0f6aceefa0ad6c6b3e298584bcede21e85"
)
FSG3_REFERENCE_SUMMARY_HASH = (
    "df852590d99be09962c1287e7166b421edb260416403a3c91545dca6e2e1318e"
)
CODE_PATHS = (
    "boundflow/runtime/fsg3_same_solver_timing.py",
    "boundflow/runtime/fsg4_b3_explicit_counters.py",
    "scripts/run_fsg4_b3_counter_diagnostic.py",
    "scripts/run_fsg3_same_solver_timing.py",
    "scripts/run_rvir_v4_live_return_capture.py",
    "boundflow/runtime/rvir_v4_native_optimizer.py",
    "boundflow/runtime/rvir_v4_native_backward_export.py",
    "boundflow/runtime/rvir_v4_native_kfsb.py",
    "boundflow/runtime/rvir_v4_atomic_copy_out.py",
    "boundflow/runtime/rvir_v4_live_return.py",
)


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


def _write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(_canonical_json(value) + "\n" for value in values),
        encoding="utf-8",
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


def _integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"FSG4/B3 {label} must be an integer")
    return value


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


def _verify_code_revision(manifest: Mapping[str, Any]) -> None:
    source = manifest.get("source_git_head")
    revision = manifest.get("code_revision")
    if not isinstance(source, str) or not isinstance(revision, Mapping):
        raise ValueError("FSG4/B3 source provenance differs")
    if _git("rev-parse", "HEAD") == source:
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
        raise ValueError("FSG4/B3 code revision differs")


def _fsg3_reference_controls() -> tuple[Any, ...]:
    reference_manifest = _load_json(FSG3_REFERENCE_ARTIFACT / "manifest.json")
    semantic_manifest = dict(reference_manifest)
    claimed_manifest_hash = semantic_manifest.pop("manifest_hash", None)
    if (
        claimed_manifest_hash != FSG3_REFERENCE_MANIFEST_HASH
        or claimed_manifest_hash != canonical_hash(semantic_manifest)
        or reference_manifest.get("summary_hash") != FSG3_REFERENCE_SUMMARY_HASH
    ):
        raise ValueError("FSG4/B3 frozen FSG3 reference identity differs")
    files = reference_manifest.get("files")
    if not isinstance(files, Mapping):
        raise TypeError("FSG4/B3 frozen FSG3 file inventory differs")
    observed_files = {
        str(path.relative_to(FSG3_REFERENCE_ARTIFACT))
        for path in FSG3_REFERENCE_ARTIFACT.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if set(files) != observed_files:
        raise ValueError("FSG4/B3 frozen FSG3 file inventory differs")
    for name, digest in files.items():
        if not isinstance(name, str) or digest != _file_sha256(
            FSG3_REFERENCE_ARTIFACT / name
        ):
            raise ValueError(f"FSG4/B3 frozen FSG3 file digest differs: {name}")
    summary = _load_json(FSG3_REFERENCE_ARTIFACT / "summary.json")
    if summary.get("summary_hash") != FSG3_REFERENCE_SUMMARY_HASH:
        raise ValueError("FSG4/B3 frozen FSG3 summary identity differs")
    references = tuple(
        fsg3_timing_run_from_dict(row)
        for row in _load_jsonl(FSG3_REFERENCE_ARTIFACT / "worker_runs.jsonl")
    )
    expected = tuple(expected_fsg3_sequence())
    if len(references) != len(expected) or any(
        (
            run.block_index,
            run.sequence_position,
            run.configuration,
            run.mode,
        )
        != row
        for run, row in zip(references, expected)
    ):
        raise ValueError("FSG4/B3 frozen FSG3 run sequence differs")
    controls = tuple(
        reference
        for reference in references
        if reference.configuration == FSG3Configuration.B2
        and reference.mode == FSG3Mode.CONTROL
    )
    if len(controls) != 6:
        raise ValueError("FSG4/B3 frozen B2 control coverage differs")
    return controls


def _verify_fsg3_semantic_reference(run: Any) -> tuple[str, ...]:
    controls = _fsg3_reference_controls()
    failures = tuple(
        failure
        for index, reference in enumerate(controls)
        for failure in _semantic_pair_failures(
            reference.semantics,
            run.semantics,
            label=f"frozen-B2-control-{index}",
        )
    )
    if failures:
        raise ValueError(
            "FSG4/B3 worker differs from frozen FSG3 semantics: " + ",".join(failures)
        )
    return tuple(reference.run_id for reference in controls)


@contextmanager
def _patch_attribute(owner: Any, name: str, replacement: object) -> Iterator[None]:
    original = getattr(owner, name)
    setattr(owner, name, replacement)
    try:
        yield
    finally:
        setattr(owner, name, original)


def _counted_function(
    recorder: Fsg4B3CounterRecorder,
    counter: str,
    detail: str,
    function: Callable[..., Any],
) -> Callable[..., Any]:
    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        recorder.add(counter, detail=detail)
        return function(*args, **kwargs)

    return wrapped


@contextmanager
def _instrument_b2(recorder: Fsg4B3CounterRecorder) -> Iterator[None]:
    """Patch named seams only; do not install a Python or CUDA profiler."""

    import torch

    from boundflow import planner
    from boundflow.runtime import native_alpha_beta_optimization_state as native_state
    from boundflow.runtime import rvir_v4_atomic_copy_out as atomic
    from boundflow.runtime import rvir_v4_live_return as live_return
    from boundflow.runtime import rvir_v4_native_backward_export as backward
    from boundflow.runtime import rvir_v4_native_kfsb as kfsb
    from boundflow.runtime import rvir_v4_native_optimizer as optimizer
    from boundflow.runtime import rvir_v4_optimizer_mutation as mutation
    from boundflow.runtime import rvir_v4_production_state as production
    from scripts import run_rvir_v4_live_return_capture as live_runner

    stack = ExitStack()

    def patch_function(owner: Any, name: str, counter: str, detail: str) -> None:
        original = cast(Callable[..., Any], getattr(owner, name))
        stack.enter_context(
            _patch_attribute(
                owner,
                name,
                _counted_function(recorder, counter, detail, original),
            )
        )

    patch_function(
        planner,
        "plan_interval_ibp_v0",
        "template_compile_count",
        "cold planner template compile",
    )
    patch_function(
        native_state,
        "build_native_alpha_beta_scope",
        "scope_construction_count",
        "live executor dynamic scope construction",
    )
    patch_function(
        optimizer,
        "build_native_alpha_beta_scope",
        "scope_construction_count",
        "optimizer scope equivalence reconstruction",
    )
    for module_owner, detail in (
        (optimizer, "optimizer forward trace"),
        (backward, "terminal backward forward trace"),
        (kfsb, "KFSB child forward trace"),
    ):
        patch_function(
            module_owner,
            "_forward_ibp_trace_mlp",
            "forward_trace_build_count",
            detail,
        )
    patch_function(
        optimizer,
        "run_crown_ibp_mlp_from_forward_trace",
        "optimizer_bound_evaluation_call_count",
        "optimizer CROWN evaluation",
    )
    patch_function(
        kfsb,
        "_evaluate_children",
        "kfsb_child_batch_count",
        "one KFSB candidate child batch",
    )

    original_move = live_runner._move_tensors
    move_depth = 0

    @wraps(original_move)
    def counted_move(*args: Any, **kwargs: Any) -> Any:
        nonlocal move_depth
        root = move_depth == 0
        move_depth += 1
        try:
            result = original_move(*args, **kwargs)
        finally:
            move_depth -= 1
        if root:
            recorder.add(
                "module_binding_move_in_core_count",
                detail="root recursive module.bindings device move",
            )
        return result

    stack.enter_context(_patch_attribute(live_runner, "_move_tensors", counted_move))

    original_optimizer = optimizer.execute_rvir_v4_native_optimizer_trace

    @wraps(original_optimizer)
    def counted_optimizer(*args: Any, **kwargs: Any) -> Any:
        result = original_optimizer(*args, **kwargs)
        evaluations = len(result.steps)
        updates = sum(int(step.update_after) for step in result.steps)
        recorder.add(
            "optimizer_trace_call_count", detail="one production optimizer trace"
        )
        recorder.add(
            "optimizer_evaluation_count",
            amount=evaluations,
            detail="materialized optimizer evaluations",
        )
        recorder.add(
            "optimizer_update_count",
            amount=updates,
            detail="optimizer steps marked update_after",
        )
        recorder.add(
            "full_optimizer_step_snapshot_count",
            amount=evaluations,
            detail="full lower/alpha/beta step snapshots retained",
        )
        return result

    stack.enter_context(
        _patch_attribute(
            optimizer, "execute_rvir_v4_native_optimizer_trace", counted_optimizer
        )
    )

    original_kfsb = kfsb.evaluate_rvir_v4_native_kfsb

    @wraps(original_kfsb)
    def counted_kfsb(*args: Any, **kwargs: Any) -> Any:
        result = original_kfsb(*args, **kwargs)
        recorder.add(
            "kfsb_evaluation_call_count", detail="one complete KFSB evaluation"
        )
        recorder.add(
            "kfsb_candidate_count",
            amount=len(result.candidate_splits),
            detail="materialized KFSB candidate splits",
        )
        return result

    stack.enter_context(
        _patch_attribute(kfsb, "evaluate_rvir_v4_native_kfsb", counted_kfsb)
    )

    original_replacement = atomic._replacement

    original_project_alpha = atomic._project_alpha

    @wraps(original_project_alpha)
    def counted_project_alpha(*args: Any, **kwargs: Any) -> Any:
        dense = args[1] if len(args) > 1 else kwargs["dense"]
        result = original_project_alpha(*args, **kwargs)
        if dense.device.type == "cuda" and result.device.type == "cpu":
            recorder.add(
                "timed_candidate_d2h_copy_count",
                detail="alpha projection GPU dense into CPU-owned sparse layout",
            )
        return result

    stack.enter_context(
        _patch_attribute(atomic, "_project_alpha", counted_project_alpha)
    )

    @wraps(original_replacement)
    def counted_replacement(source: Any, value: Any) -> Any:
        result = original_replacement(source, value)
        recorder.add(
            "candidate_snapshot_materialization_count",
            detail=f"candidate path {source.semantic_path}",
        )
        if value.device.type == "cuda" and result.value.device.type == "cpu":
            recorder.add(
                "timed_candidate_d2h_copy_count",
                detail=f"candidate D2H path {source.semantic_path}",
            )
        return result

    stack.enter_context(_patch_attribute(atomic, "_replacement", counted_replacement))

    original_copy = atomic._copy_value

    @wraps(original_copy)
    def counted_copy(target: Any, source: Any) -> None:
        original_copy(target, source)
        recorder.add(
            "live_tensor_copy_call_count",
            detail=f"live copy {tuple(target.shape)} {target.device}",
        )

    stack.enter_context(_patch_attribute(atomic, "_copy_value", counted_copy))

    original_stage = atomic.stage_rvir_v4_live_atomic_copy_out

    @wraps(original_stage)
    def counted_stage(*args: Any, **kwargs: Any) -> Any:
        result = original_stage(*args, **kwargs)
        recorder.add(
            "atomic_stage_call_count", detail="one live atomic stage completed"
        )
        return result

    stack.enter_context(
        _patch_attribute(atomic, "stage_rvir_v4_live_atomic_copy_out", counted_stage)
    )

    original_commit = atomic.commit_rvir_v4_live_atomic_copy_out

    @wraps(original_commit)
    def counted_commit(*args: Any, **kwargs: Any) -> Any:
        staged = args[0] if args else kwargs["staged"]
        paths = len(staged.path_receipts)
        copies_before = recorder.counts()["live_tensor_copy_call_count"]
        try:
            result = original_commit(*args, **kwargs)
        except Exception:
            copies_after = recorder.counts()["live_tensor_copy_call_count"]
            rollback_copies = max(copies_after - copies_before - paths, 0)
            if rollback_copies:
                recorder.add(
                    "rollback_copy_call_count",
                    amount=rollback_copies,
                    detail="failed transaction rollback copies",
                )
            raise
        copies_after = recorder.counts()["live_tensor_copy_call_count"]
        recorder.add(
            "atomic_commit_call_count", detail="one live atomic commit completed"
        )
        recorder.add(
            "device_rollback_backup_count",
            amount=paths,
            detail="unconditional per-path device rollback backups",
        )
        recorder.add(
            "commit_copy_call_count",
            amount=copies_after - copies_before,
            detail="successful live tensor commit copies",
        )
        recorder.add(
            "committed_mutable_path_count",
            amount=_integer(result["committed_path_count"], "committed path count"),
            detail="receipt-confirmed committed mutable paths",
        )
        return result

    stack.enter_context(
        _patch_attribute(
            live_return, "commit_rvir_v4_live_atomic_copy_out", counted_commit
        )
    )

    def patch_tensor_hash(owner: Any, name: str, detail: str) -> None:
        original = cast(Callable[..., Any], getattr(owner, name))

        @wraps(original)
        def wrapped(value: Any) -> Any:
            recorder.add("tensor_content_hash_count", detail=detail)
            if torch.is_tensor(value) and value.device.type == "cuda":
                recorder.add("gpu_tensor_content_hash_count", detail=detail)
            return original(value)

        stack.enter_context(_patch_attribute(owner, name, wrapped))

    for module_owner, detail in (
        (production, "production-state tensor digest"),
        (atomic, "atomic-copy tensor digest"),
        (live_return, "live-return tensor digest"),
    ):
        patch_tensor_hash(module_owner, "production_tensor_sha256", detail)

    validate_types = (
        production.ProductionStateSnapshotV4,
        native_state.NativeAlphaBetaStateScope,
        native_state.NativeAlphaBetaOptimizationState,
        mutation.ProductionMutationPolicyV4,
        optimizer.NativeProductionOptimizerStepV4,
        optimizer.NativeProductionOptimizerTraceV4,
        backward.NativeBackwardExportV4,
        kfsb.NativeKfsbEvaluationV4,
        atomic.ProductionLiveAtomicCopyOutV4,
        live_return.LiveCoreReturnAssemblyV4,
    )
    for validate_type in validate_types:
        patch_function(
            validate_type,
            "validate",
            "typed_validate_call_count",
            f"{validate_type.__name__}.validate",
        )

    hash_types = (
        production.ProductionStateSnapshotV4,
        native_state.NativeAlphaBetaStateScope,
        native_state.NativeAlphaBetaOptimizationState,
        mutation.ProductionMutationPolicyV4,
    )
    for hash_type in hash_types:
        patch_function(
            hash_type,
            "stable_hash",
            "stable_hash_call_count",
            f"{hash_type.__name__}.stable_hash",
        )

    with stack:
        yield


def _worker_namespace(args: argparse.Namespace, result: Path) -> argparse.Namespace:
    return argparse.Namespace(
        configuration=FSG3Configuration.B2.value,
        mode=FSG3Mode.CONTROL.value,
        run_id="fsg4-b3-0-b2-control",
        block_index=0,
        sequence_position=4,
        benchmark_root=args.benchmark_root.resolve(),
        abcrown_root=args.abcrown_root.resolve(),
        model=args.model.resolve(),
        property=args.property.resolve(),
        result=result.resolve(),
    )


def _validate_worker_envelope(value: Mapping[str, Any], worker_sha: str) -> Any:
    if value.get("schema_version") != fsg3_worker.WORKER_ENVELOPE_SCHEMA:
        raise ValueError("FSG4/B3 worker envelope schema differs")
    raw_run = value.get("run")
    if not isinstance(raw_run, Mapping):
        raise TypeError("FSG4/B3 worker run differs")
    run = fsg3_timing_run_from_dict(cast(Mapping[str, object], raw_run))
    if (
        run.configuration != FSG3Configuration.B2
        or run.mode != FSG3Mode.CONTROL
        or not run.environment.admitted
        or run.execution.provider_core_call_count != 0
        or run.execution.provider_compute_bounds_call_count != 0
        or run.execution.provider_update_bounds_call_count != 0
        or run.execution.fallback_dispatch_count != 0
        or value.get("performance_claimed") is not False
        or len(worker_sha) != 64
    ):
        raise ValueError("FSG4/B3 worker gate failed")
    return run


def _generate(args: argparse.Namespace) -> None:
    artifact = args.artifact_dir.resolve()
    if artifact.exists():
        raise FileExistsError(f"FSG4/B3 artifact already exists: {artifact}")
    dirty = _git("status", "--porcelain=v1", "--", *CODE_PATHS)
    if dirty:
        raise RuntimeError("FSG4/B3 diagnostic code paths must be committed")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{artifact.name}.incomplete-", dir=artifact.parent
    ) as raw:
        staging = Path(raw)
        worker_path = staging / "worker.json"
        recorder = Fsg4B3CounterRecorder()
        with _instrument_b2(recorder):
            fsg3_worker._worker(_worker_namespace(args, worker_path))
        worker_sha = _file_sha256(worker_path)
        envelope = _load_json(worker_path)
        run = _validate_worker_envelope(envelope, worker_sha)
        reference_run_ids = _verify_fsg3_semantic_reference(run)
        events = [event.to_dict() for event in recorder.events]
        _write_jsonl(staging / "events.jsonl", events)
        counts = recorder.counts()
        print(
            _canonical_json(
                {
                    "counter_gate": "observed-before-validation",
                    "counts": counts,
                    "performance_claimed": False,
                }
            ),
            flush=True,
        )
        snapshot = Fsg4B3CounterSnapshot(
            counts_by_name=tuple(sorted(counts.items())),
            semantic_hash=canonical_hash(run.semantics.to_dict()),
            worker_result_sha256=worker_sha,
            provider_core_call_count=run.execution.provider_core_call_count,
            provider_compute_bounds_call_count=(
                run.execution.provider_compute_bounds_call_count
            ),
            provider_update_bounds_call_count=(
                run.execution.provider_update_bounds_call_count
            ),
            fallback_dispatch_count=run.execution.fallback_dispatch_count,
            environment_admitted=run.environment.admitted,
        )
        snapshot_payload = snapshot.to_dict()
        report: dict[str, object] = {
            "schema_version": ARTIFACT_SCHEMA,
            "source_git_head": _git("rev-parse", "HEAD"),
            "source_identity": run.source_identity,
            "protocol_identity": run.protocol_identity,
            "fsg3_reference_manifest_hash": FSG3_REFERENCE_MANIFEST_HASH,
            "fsg3_reference_summary_hash": FSG3_REFERENCE_SUMMARY_HASH,
            "fsg3_reference_b2_control_run_ids": list(reference_run_ids),
            "fsg3_reference_semantic_failures": [],
            "model_sha256": capture_runner.file_sha256(args.model),
            "property_sha256": capture_runner.file_sha256(args.property),
            "event_count": len(events),
            "event_journal_sha256": _file_sha256(staging / "events.jsonl"),
            "snapshot": snapshot_payload,
            "fixed_counter_expectations_passed": True,
            "correctness_passed": True,
            "environment_passed": True,
            "diagnostic_timing_claimed": False,
            "performance_claimed": False,
        }
        report["report_hash"] = canonical_hash(report)
        _write_json(staging / "report.json", report)
        files = {
            name: _file_sha256(staging / name)
            for name in ("events.jsonl", "report.json", "worker.json")
        }
        manifest: dict[str, object] = {
            "schema_version": MANIFEST_SCHEMA,
            "source_git_head": _git("rev-parse", "HEAD"),
            "code_revision": _code_revision(),
            "files": files,
            "report_hash": report["report_hash"],
            "fsg3_reference_manifest_hash": FSG3_REFERENCE_MANIFEST_HASH,
            "fsg3_reference_summary_hash": FSG3_REFERENCE_SUMMARY_HASH,
            "performance_claimed": False,
        }
        manifest["manifest_hash"] = canonical_hash(manifest)
        _write_json(staging / "manifest.json", manifest)
        _replay(staging)
        staging.rename(artifact)


def _replay(artifact: Path) -> dict[str, object]:
    artifact = artifact.resolve()
    manifest = _load_json(artifact / "manifest.json")
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("FSG4/B3 manifest schema differs")
    payload = dict(manifest)
    claimed_manifest_hash = payload.pop("manifest_hash", None)
    if claimed_manifest_hash != canonical_hash(payload):
        raise ValueError("FSG4/B3 manifest hash differs")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != {
        "events.jsonl",
        "report.json",
        "worker.json",
    }:
        raise ValueError("FSG4/B3 manifest file inventory differs")
    for name, expected in files.items():
        if _file_sha256(artifact / str(name)) != expected:
            raise ValueError(f"FSG4/B3 file digest differs: {name}")
    _verify_code_revision(manifest)
    report = _load_json(artifact / "report.json")
    report_payload = dict(report)
    claimed_report_hash = report_payload.pop("report_hash", None)
    if (
        report.get("schema_version") != ARTIFACT_SCHEMA
        or report.get("source_git_head") != manifest.get("source_git_head")
        or report.get("fsg3_reference_manifest_hash") != FSG3_REFERENCE_MANIFEST_HASH
        or report.get("fsg3_reference_summary_hash") != FSG3_REFERENCE_SUMMARY_HASH
        or manifest.get("fsg3_reference_manifest_hash") != FSG3_REFERENCE_MANIFEST_HASH
        or manifest.get("fsg3_reference_summary_hash") != FSG3_REFERENCE_SUMMARY_HASH
        or report.get("fsg3_reference_semantic_failures") != []
        or claimed_report_hash != canonical_hash(report_payload)
        or claimed_report_hash != manifest.get("report_hash")
        or report.get("performance_claimed") is not False
        or report.get("diagnostic_timing_claimed") is not False
        or report.get("correctness_passed") is not True
        or report.get("environment_passed") is not True
        or report.get("fixed_counter_expectations_passed") is not True
        or manifest.get("performance_claimed") is not False
    ):
        raise ValueError("FSG4/B3 report gate failed")
    raw_snapshot = report.get("snapshot")
    if not isinstance(raw_snapshot, Mapping):
        raise TypeError("FSG4/B3 report snapshot differs")
    snapshot = fsg4_b3_counter_snapshot_from_dict(raw_snapshot)
    worker_path = artifact / "worker.json"
    envelope = _load_json(worker_path)
    run = _validate_worker_envelope(envelope, _file_sha256(worker_path))
    reference_run_ids = _verify_fsg3_semantic_reference(run)
    if (
        snapshot.semantic_hash != canonical_hash(run.semantics.to_dict())
        or snapshot.worker_result_sha256 != _file_sha256(worker_path)
        or report.get("source_identity") != run.source_identity
        or report.get("protocol_identity") != run.protocol_identity
        or report.get("fsg3_reference_b2_control_run_ids") != list(reference_run_ids)
    ):
        raise ValueError("FSG4/B3 worker/report binding differs")
    rows = _load_jsonl(artifact / "events.jsonl")
    events = events_from_rows(rows)
    rebuilt = Fsg4B3CounterRecorder(events=list(events)).counts()
    if (
        rebuilt != snapshot.counts
        or report.get("event_count") != len(events)
        or report.get("event_journal_sha256") != _file_sha256(artifact / "events.jsonl")
    ):
        raise ValueError("FSG4/B3 event-derived counters differ")
    result = {
        "status": "replay-passed",
        "source_git_head": manifest["source_git_head"],
        "event_count": len(events),
        "counts": rebuilt,
        "report_hash": claimed_report_hash,
        "manifest_hash": claimed_manifest_hash,
        "performance_claimed": False,
    }
    print(_canonical_json(result), flush=True)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--artifact-dir", type=Path, required=True)
    run.add_argument("--benchmark-root", type=Path, required=True)
    run.add_argument("--abcrown-root", type=Path, required=True)
    run.add_argument("--model", type=Path, required=True)
    run.add_argument("--property", type=Path, required=True)
    replay = commands.add_parser("replay")
    replay.add_argument("--artifact-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "run":
        _generate(args)
    else:
        _replay(args.artifact_dir)


if __name__ == "__main__":
    main()
