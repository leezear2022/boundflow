#!/usr/bin/env python3
"""Run one real FSG4 B0/B2/B3 same-solver timing worker."""

# pylint: disable=wrong-import-position,protected-access,duplicate-code
# pylint: disable=too-many-locals,too-many-statements,import-error

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Optional, cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from boundflow.runtime.fsg3_same_solver_timing import (
    canonical_hash,
    FSG3Configuration,
    FSG3Mode,
    fsg3_timing_run_from_dict,
)
from boundflow.runtime.fsg4_b3_explicit_counters import Fsg4B3CounterRecorder
from boundflow.runtime.fsg4_b3_same_solver_timing import (
    FSG4B3ActivationReceipt,
    FSG4B3ExecutionCounters,
    FSG4B3TimingConfiguration,
    FSG4B3TimingRun,
)
from scripts import run_fsg3_same_solver_timing as base_worker
from scripts import run_fsg4_b3_counter_diagnostic as diagnostic

WORKER_ENVELOPE_SCHEMA = "boundflow.fsg4-b3-same-solver-worker-envelope/v1"
FIVE_FRESH_MANIFEST_FILE_SHA256 = (
    "bf8b3ecccea992cce9dca56c963518510af8dc8d410c0d02b94513160189cb98"
)
FIVE_FRESH_MANIFEST_HASH = (
    "457ab1adc8488c5353ec66294583e7a2bedf2e92fca5901a72a41e8321df1573"
)


def _protocol_identity() -> str:
    """Bind the inherited solver protocol and B3 admission evidence."""

    return canonical_hash(
        {
            "base_protocol_identity": base_worker._protocol_identity(
                FSG3Configuration.B2
            ),
            "configuration_set": ["B0", "B2", "B3"],
            "control_measurement_instrumentation": "base-observers-only",
            "profile_counter_instrumentation": "lightweight-no-event-journal",
            "five_fresh_manifest_file_sha256": FIVE_FRESH_MANIFEST_FILE_SHA256,
            "five_fresh_manifest_hash": FIVE_FRESH_MANIFEST_HASH,
            "performance_claimed": False,
        }
    )


def _list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"FSG4/B3 {label} differs")
    return value


def _one_mapping(value: object, label: str) -> Mapping[str, Any]:
    rows = _list(value, label)
    if len(rows) != 1 or not isinstance(rows[0], Mapping):
        raise ValueError(f"FSG4/B3 {label} cardinality differs")
    return cast(Mapping[str, Any], rows[0])


def _normalize_process_names(rows: object, label: str) -> list[Any]:
    normalized = _list(rows, label)
    for row in normalized:
        if isinstance(row, dict) and isinstance(row.get("name"), str):
            row["name"] = Path(row["name"]).name
    return normalized


def _normalize_diagnostics(value: Mapping[str, Any]) -> dict[str, Any]:
    """Remove host-local executable paths before binding runtime identity."""

    normalized = json.loads(json.dumps(value, allow_nan=False))
    if not isinstance(normalized, dict):
        raise TypeError("FSG4/B3 diagnostics normalization differs")
    runtime = normalized.get("runtime_environment")
    if not isinstance(runtime, dict) or not isinstance(
        runtime.get("python_executable"), str
    ):
        raise TypeError("FSG4/B3 runtime environment differs")
    runtime["python_executable"] = Path(runtime["python_executable"]).name
    for name in ("compute_processes_before", "compute_processes_after"):
        normalized[name] = _normalize_process_names(normalized.get(name), name)
    preflight = normalized.get("worker_preflight")
    if not isinstance(preflight, dict) or not isinstance(
        preflight.get("samples"), list
    ):
        raise TypeError("FSG4/B3 worker preflight differs")
    for sample in preflight["samples"]:
        if not isinstance(sample, dict):
            raise TypeError("FSG4/B3 worker preflight sample differs")
        sample["compute_processes"] = _normalize_process_names(
            sample.get("compute_processes"), "worker preflight processes"
        )
    return normalized


def _activation_receipt(
    *,
    configuration: FSG4B3TimingConfiguration,
    mode: FSG3Mode,
    diagnostics: Mapping[str, Any],
    recorder: Optional[Fsg4B3CounterRecorder],
) -> FSG4B3ActivationReceipt:
    templates = _list(
        diagnostics.get("prepared_core_template_hashes"), "prepared templates"
    )
    instances = _list(
        diagnostics.get("prepared_core_instance_hashes"), "prepared instances"
    )
    schedules = _list(
        diagnostics.get("terminal_optimizer_schedule_hashes"), "terminal schedules"
    )
    assemblies = _list(diagnostics.get("assembly_metadata"), "assemblies")
    receipts = _list(diagnostics.get("commit_receipts"), "commit receipts")
    audits = _list(diagnostics.get("device_commit_audits"), "device audits")
    post_audit_ns = int(diagnostics.get("post_query_audit_ns", -1))
    excluded = diagnostics.get("post_query_audit_excluded_from_timing") is True
    headline_digest: Optional[int] = None
    candidate_d2h: Optional[int] = None
    if configuration == FSG4B3TimingConfiguration.B3:
        assembly = _one_mapping(assemblies, "assemblies")
        receipt = _one_mapping(receipts, "commit receipts")
        _one_mapping(audits, "device audits")
        headline_digest = int(assembly.get("headline_content_digest_count", -1))
        candidate_d2h = int(receipt.get("candidate_d2h_copy_count", -1))
    detailed_counts = None if recorder is None else recorder.counts()
    activation = FSG4B3ActivationReceipt(
        prepared_core_template_count=len(templates),
        prepared_core_instance_count=len(instances),
        terminal_optimizer_schedule_count=len(schedules),
        assembly_count=len(assemblies),
        commit_receipt_count=len(receipts),
        device_commit_audit_count=len(audits),
        post_query_audit_ns=post_audit_ns,
        post_query_audit_excluded_from_timing=excluded,
        headline_content_digest_count=headline_digest,
        candidate_d2h_copy_count=candidate_d2h,
        detailed_counts_by_name=(
            None if detailed_counts is None else tuple(sorted(detailed_counts.items()))
        ),
    )
    activation.validate(configuration, mode)
    return activation


def _base_namespace(args: argparse.Namespace, result: Path) -> argparse.Namespace:
    configuration = FSG4B3TimingConfiguration(args.configuration)
    underlying = (
        FSG3Configuration.B0
        if configuration == FSG4B3TimingConfiguration.B0
        else FSG3Configuration.B2
    )
    return argparse.Namespace(
        configuration=underlying.value,
        mode=args.mode,
        run_id=args.run_id,
        block_index=args.block_index,
        sequence_position=args.sequence_position,
        benchmark_root=args.benchmark_root,
        abcrown_root=args.abcrown_root,
        model=args.model,
        property=args.property,
        result=result,
        prepare_static_request=bool(getattr(args, "prepare_static_request", False)),
        attribute_root_incomplete=bool(
            getattr(args, "attribute_root_incomplete", False)
        ),
        attribute_complete_prelude=bool(
            getattr(args, "attribute_complete_prelude", False)
        ),
        prepare_root_optimizer_warmup=bool(
            getattr(args, "prepare_root_optimizer_warmup", False)
        ),
    )


def _worker(args: argparse.Namespace) -> None:
    configuration = FSG4B3TimingConfiguration(args.configuration)
    mode = FSG3Mode(args.mode)
    recorder = (
        Fsg4B3CounterRecorder(retain_events=False)
        if mode == FSG3Mode.PROFILE
        and configuration
        in {FSG4B3TimingConfiguration.B2, FSG4B3TimingConfiguration.B3}
        else None
    )
    with tempfile.TemporaryDirectory(prefix="boundflow-fsg4-b3-base-worker-") as raw:
        base_result = Path(raw) / "base-worker.json"
        with ExitStack() as stack:
            if recorder is not None:
                stack.enter_context(diagnostic._instrument_b2(recorder))
            if configuration == FSG4B3TimingConfiguration.B3:
                stack.enter_context(diagnostic._use_prepared_executor("B3-C"))
            base_worker._worker(_base_namespace(args, base_result))
        base_envelope = json.loads(base_result.read_text(encoding="utf-8"))
    if (
        not isinstance(base_envelope, Mapping)
        or base_envelope.get("schema_version") != base_worker.WORKER_ENVELOPE_SCHEMA
        or base_envelope.get("performance_claimed") is not False
    ):
        raise ValueError("FSG4/B3 base worker envelope differs")
    base_run_payload = base_envelope.get("run")
    diagnostics = base_envelope.get("diagnostics")
    if not isinstance(base_run_payload, Mapping) or not isinstance(
        diagnostics, Mapping
    ):
        raise TypeError("FSG4/B3 base worker payload differs")
    base_run = fsg3_timing_run_from_dict(cast(Mapping[str, Any], base_run_payload))
    expected_base = (
        FSG3Configuration.B0
        if configuration == FSG4B3TimingConfiguration.B0
        else FSG3Configuration.B2
    )
    if base_run.configuration != expected_base or base_run.mode != mode:
        raise ValueError("FSG4/B3 base measurement configuration differs")
    normalized_diagnostics = _normalize_diagnostics(
        cast(Mapping[str, Any], diagnostics)
    )
    normalized_environment = replace(
        base_run.environment,
        runtime_identity=canonical_hash(normalized_diagnostics["runtime_environment"]),
    )
    execution = FSG4B3ExecutionCounters(
        typed_validation_count=base_run.execution.typed_validation_count,
        provider_core_call_count=base_run.execution.provider_core_call_count,
        provider_compute_bounds_call_count=(
            base_run.execution.provider_compute_bounds_call_count
        ),
        provider_update_bounds_call_count=(
            base_run.execution.provider_update_bounds_call_count
        ),
        fallback_dispatch_count=base_run.execution.fallback_dispatch_count,
        backend_kind=base_run.execution.backend_kind,
        replacement_mode={
            FSG4B3TimingConfiguration.B0: "original_provider",
            FSG4B3TimingConfiguration.B2: "whole_call_reference",
            FSG4B3TimingConfiguration.B3: "b3_ir_graph_plan_schedule",
        }[configuration],
    )
    activation = _activation_receipt(
        configuration=configuration,
        mode=mode,
        diagnostics=normalized_diagnostics,
        recorder=recorder,
    )
    run = FSG4B3TimingRun(
        run_id=base_run.run_id,
        block_index=base_run.block_index,
        sequence_position=base_run.sequence_position,
        configuration=configuration,
        mode=mode,
        source_identity=base_run.source_identity,
        protocol_identity=_protocol_identity(),
        metrics=base_run.metrics,
        semantics=base_run.semantics,
        execution=execution,
        environment=normalized_environment,
        activation=activation,
        profile_spans=base_run.profile_spans,
        profile_closure_error=base_run.profile_closure_error,
        profile_residual_share=base_run.profile_residual_share,
    )
    run.validate()
    extended_diagnostics = dict(normalized_diagnostics)
    extended_diagnostics.update(
        {
            "base_measurement_configuration": expected_base.value,
            "fsg4_configuration": configuration.value,
            "profile_counter_instrumentation": recorder is not None,
            "activation_receipt_hash": canonical_hash(activation.to_dict()),
            "five_fresh_manifest_file_sha256": FIVE_FRESH_MANIFEST_FILE_SHA256,
            "five_fresh_manifest_hash": FIVE_FRESH_MANIFEST_HASH,
        }
    )
    envelope = {
        "schema_version": WORKER_ENVELOPE_SCHEMA,
        "run": run.to_dict(),
        "diagnostics": extended_diagnostics,
        "performance_claimed": False,
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(
        json.dumps(envelope, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "run_id": run.run_id,
                "configuration": configuration.value,
                "query_wall_ns": run.metrics.query_wall_ns,
                "core_wall_ns": run.metrics.core_wall_ns,
                "environment_admitted": run.environment.admitted,
                "performance_claimed": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", choices=("B0", "B2", "B3"), required=True)
    parser.add_argument("--mode", choices=("control", "profile"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--block-index", type=int, required=True)
    parser.add_argument("--sequence-position", type=int, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--abcrown-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--property", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Run one fail-closed FSG4 timing worker."""

    args = _parse_args()
    args.benchmark_root = args.benchmark_root.resolve()
    args.abcrown_root = args.abcrown_root.resolve()
    args.model = args.model.resolve()
    args.property = args.property.resolve()
    args.result = args.result.resolve()
    _worker(args)


if __name__ == "__main__":
    main()
