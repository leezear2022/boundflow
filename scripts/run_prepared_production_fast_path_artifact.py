#!/usr/bin/env python3
"""Generate or replay the NRIR-16 prepared production fast-path artifact."""

# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=too-many-arguments,too-many-boolean-expressions,duplicate-code
# pylint: disable=missing-function-docstring,line-too-long

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Mapping, cast

import torch

from boundflow.ir.bound import IntermediateBoundSource
from boundflow.runtime.abcrown_adapter import file_sha256
from boundflow.runtime.complete_verifier_query import (
    CompleteVerifierQueryPolicy,
    execute_complete_verifier_query,
)
from boundflow.runtime.native_prepared_complete_query import (
    NATIVE_PREPARED_COMPLETE_QUERY_COMPILER_VERSION,
    execute_native_prepared_complete_query,
    prepare_native_root_complete_query,
)
from scripts.run_end_to_end_tightness_performance_baseline import (
    ARTIFACT_SCHEMA_VERSION as BASELINE_ARTIFACT_SCHEMA_VERSION,
)
from scripts.run_end_to_end_tightness_performance_baseline import (
    EXTERNAL_ADAPTIVE_POLICY,
    QUEUE_CONFIG,
    SEARCH_POLICY,
    _build_context,
    validate_evidence as validate_baseline_evidence,
)
from scripts.run_native_real_network_ir_artifact import (
    ABCROWN_COMMIT,
    INTERMEDIATE_BOUNDS_SHA256,
    MODEL_SHA256,
    VNNCOMP_COMMIT,
    VNNLIB_SHA256,
)

ARTIFACT_SCHEMA_VERSION = "boundflow.prepared-production-fast-path-artifact/v1"
EVIDENCE_SCHEMA_VERSION = "boundflow.prepared-production-fast-path-evidence/v1"
ARTIFACT_FILE = "prepared_fast_path.json"
MANIFEST_FILE = "manifest.json"
QUERY_ID = "vnncomp21-resnet2b-prop0-native-ir16-prepared-production"
AUDIT_QUERY_ID = "vnncomp21-resnet2b-prop0-native-ir16-audit-control"
TIMING_GROUPS = 3
TIMING_WARMUPS = 1
NUMERIC_ATOL = 2e-4
NUMERIC_RTOL = 2e-4
MIN_WARM_DIAGNOSTIC_REDUCTION = 10.0
TIMING_ORDERS = (
    ("audit", "prepared_production"),
    ("prepared_production", "audit"),
    ("audit", "prepared_production"),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("generate", "replay"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--model", type=Path, required=True)
        subparser.add_argument("--source-artifact-dir", type=Path, required=True)
        subparser.add_argument("--local-artifact-dir", type=Path, required=True)
        subparser.add_argument("--baseline-artifact-dir", type=Path, required=True)
        subparser.add_argument("--artifact-dir", type=Path, required=True)
        subparser.add_argument("--torch-threads", type=int, default=8)
    return parser.parse_args()


def _canonical_json(value: object, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
        allow_nan=False,
    )


def canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a list")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_baseline_reference(baseline_dir: Path) -> dict[str, object]:
    manifest_path = baseline_dir / "manifest.json"
    artifact_path = baseline_dir / "baseline.json"
    manifest = _load_json(manifest_path)
    artifact = _load_json(artifact_path)
    if (
        manifest.get("schema_version") != BASELINE_ARTIFACT_SCHEMA_VERSION
        or manifest.get("status") != "ok"
        or manifest.get("performance_claimed") is not False
        or manifest.get("files") != {"baseline.json": file_sha256(artifact_path)}
        or artifact.get("schema_version") != BASELINE_ARTIFACT_SCHEMA_VERSION
        or artifact.get("status") != "ok"
    ):
        raise ValueError("NRIR-16 baseline artifact differs")
    evidence = _mapping(artifact.get("evidence"), "baseline evidence")
    validate_baseline_evidence(evidence)
    if manifest.get("evidence_hash") != canonical_hash(evidence):
        raise ValueError("NRIR-16 baseline evidence hash differs")
    semantic = _mapping(evidence.get("semantic"), "baseline semantic")
    external = _mapping(semantic.get("external_adaptive"), "baseline external")
    return {
        "manifest_sha256": file_sha256(manifest_path),
        "artifact_sha256": file_sha256(artifact_path),
        "evidence_hash": manifest["evidence_hash"],
        "query_trace_hash": external["query_trace_hash"],
        "rows": external["rows"],
        "unresolved_clause_indices": external["unresolved_clause_indices"],
        "verified_clause_indices": external["verified_clause_indices"],
    }


def _prepare_context(args: argparse.Namespace) -> tuple[dict[str, Any], Any, int]:
    context = _build_context(
        model=args.model,
        source_artifact_dir=args.source_artifact_dir,
        local_artifact_dir=args.local_artifact_dir,
    )
    tensors = _mapping(context["tensors"], "context tensors")
    thresholds = torch.zeros(9, dtype=tensors["linear_spec_c"].dtype)
    started_ns = time.perf_counter_ns()
    prepared = prepare_native_root_complete_query(
        context["module"],
        context["input_spec"],
        linear_spec_C=tensors["linear_spec_c"],
        thresholds=thresholds,
        query_id=QUERY_ID,
        search_policy=SEARCH_POLICY,
        optimizer_policy=EXTERNAL_ADAPTIVE_POLICY,
        relu_pre_override=context["external_relu_pre"],
        intermediate_bound_source=IntermediateBoundSource.EXTERNAL_VERIFIER,
    )
    return context, prepared, time.perf_counter_ns() - started_ns


def _prepared_payload_bytes(prepared: Any) -> int:
    storages: dict[tuple[str, int], int] = {}

    def add(value: torch.Tensor) -> None:
        storage = value.untyped_storage()
        key = (str(value.device), storage.data_ptr())
        storages[key] = max(storages.get(key, 0), storage.nbytes())

    for clause in prepared.clauses:
        program = clause.prepared_optimizer.program
        for value in program.initial_state.splits.values():
            add(value)
        for value in program.initial_state.alphas.values():
            add(value)
        for value in program.initial_state.betas.values():
            add(value)
        for state in program.interval_env.values():
            add(state.lower)
            add(state.upper)
        for state in program.relu_pre.values():
            add(state.lower)
            add(state.upper)
    return sum(storages.values())


def _execute_prepared(context: Mapping[str, Any], prepared: Any) -> Any:
    tensors = _mapping(context["tensors"], "context tensors")
    thresholds = torch.zeros(9, dtype=tensors["linear_spec_c"].dtype)
    return execute_native_prepared_complete_query(
        prepared,
        context["module"],
        context["input_spec"],
        linear_spec_C=tensors["linear_spec_c"],
        thresholds=thresholds,
    )


def _execute_audit(context: Mapping[str, Any]) -> Any:
    tensors = _mapping(context["tensors"], "context tensors")
    return execute_complete_verifier_query(
        context["module"],
        context["input_spec"],
        linear_spec_C=tensors["linear_spec_c"],
        thresholds=torch.zeros(9, dtype=tensors["linear_spec_c"].dtype),
        query_id=AUDIT_QUERY_ID,
        query_policy=CompleteVerifierQueryPolicy(),
        search_policy=SEARCH_POLICY,
        queue_config=QUEUE_CONFIG,
        optimizer_policy=EXTERNAL_ADAPTIVE_POLICY,
        relu_pre_override=context["external_relu_pre"],
        intermediate_bound_source=IntermediateBoundSource.EXTERNAL_VERIFIER,
    )


def _production_semantic(execution: Any) -> dict[str, object]:
    trace = execution.trace
    return {
        "trace": trace.to_dict(),
        "trace_hash": trace.stable_hash(),
        "rows": [
            {
                "clause_index": item.clause_index,
                "lower": item.lower,
                "upper": item.upper,
                "candidate_best": item.candidate_best,
                "status": item.status,
                "best_iteration": item.best_iteration,
                "lower_hash": item.lower_hash,
                "upper_hash": item.upper_hash,
            }
            for item in trace.completed_clauses
        ],
    }


def _compare_semantics(
    production: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, object]:
    actual_rows = _list(production.get("rows"), "production rows")
    expected_rows = _list(baseline.get("rows"), "baseline rows")
    if len(actual_rows) != len(expected_rows):
        raise ValueError("NRIR-16 production/baseline clause count differs")
    lower_diffs: list[float] = []
    candidate_diffs: list[float] = []
    statuses_same = True
    for actual_raw, expected_raw in zip(actual_rows, expected_rows):
        actual = _mapping(actual_raw, "production row")
        expected = _mapping(expected_raw, "baseline row")
        if actual.get("clause_index") != expected.get("clause_index"):
            raise ValueError("NRIR-16 production/baseline clause order differs")
        lower_diffs.append(abs(float(actual["lower"]) - float(expected["lower"])))
        candidate_diffs.append(
            abs(float(actual["candidate_best"]) - float(expected["candidate_best"]))
        )
        statuses_same = statuses_same and actual.get("status") == expected.get("status")
    lower_scale = max(
        1.0,
        max(abs(float(_mapping(row, "row")["lower"])) for row in expected_rows),
    )
    return {
        "clause_order_same": True,
        "statuses_same": statuses_same,
        "lower_max_abs_diff": max(lower_diffs),
        "lower_allclose": max(lower_diffs) <= NUMERIC_ATOL + NUMERIC_RTOL * lower_scale,
        "candidate_max_abs_diff": max(candidate_diffs),
        "candidate_exact": max(candidate_diffs) == 0.0,
        "atol": NUMERIC_ATOL,
        "rtol": NUMERIC_RTOL,
    }


def _percentile_90(values: list[int]) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.9 * len(ordered)) - 1)]


def _timing_summary(samples: list[dict[str, object]]) -> dict[str, object]:
    by_variant: dict[str, list[int]] = {}
    for sample in samples:
        by_variant.setdefault(cast(str, sample["variant"]), []).append(
            cast(int, sample["elapsed_ns"])
        )
    return {
        variant: {
            "runs": len(values),
            "raw_ns": values,
            "median_ns": int(statistics.median(values)),
            "p90_ns": _percentile_90(values),
        }
        for variant, values in sorted(by_variant.items())
    }


def _measure(
    context: Mapping[str, Any], prepared: Any, *, preparation_ns: int
) -> tuple[dict[str, object], dict[str, object]]:
    started_ns = time.perf_counter_ns()
    cold = _execute_prepared(context, prepared)
    cold_execution_ns = time.perf_counter_ns() - started_ns
    cold_semantic = _production_semantic(cold)
    for _unused in range(TIMING_WARMUPS):
        _execute_prepared(context, prepared)

    samples: list[dict[str, object]] = []
    production_hashes: list[str] = []
    for group, order in enumerate(TIMING_ORDERS):
        for position, variant in enumerate(order):
            started_ns = time.perf_counter_ns()
            if variant == "audit":
                execution = _execute_audit(context)
                status = execution.trace.status
                unresolved = execution.trace.unresolved_clause_indices
                semantic_hash = execution.trace.stable_hash()
            else:
                execution = _execute_prepared(context, prepared)
                status = execution.trace.status
                unresolved = execution.trace.unresolved_clause_indices
                semantic_hash = execution.trace.stable_hash()
                production_hashes.append(semantic_hash)
            elapsed_ns = time.perf_counter_ns() - started_ns
            samples.append(
                {
                    "group": group,
                    "position": position,
                    "variant": variant,
                    "elapsed_ns": elapsed_ns,
                    "status": status,
                    "unresolved_clause_indices": list(unresolved),
                    "semantic_hash": semantic_hash,
                }
            )
    summary = _timing_summary(samples)
    audit_median = int(_mapping(summary["audit"], "audit timing")["median_ns"])
    production_median = int(
        _mapping(summary["prepared_production"], "production timing")["median_ns"]
    )
    timing = {
        "performance_claimed": False,
        "measurement_kind": (
            "fixed-ResNet CPU internal audit-overhead diagnostic; not competitor speedup"
        ),
        "groups": TIMING_GROUPS,
        "warmups": TIMING_WARMUPS,
        "orders": [list(order) for order in TIMING_ORDERS],
        "preparation_ns": preparation_ns,
        "cold_first_execution_ns": cold_execution_ns,
        "cold_prepare_plus_first_execution_ns": preparation_ns + cold_execution_ns,
        "prepared_payload_unique_bytes": _prepared_payload_bytes(prepared),
        "samples": samples,
        "summary": summary,
        "diagnostic_audit_to_warm_median_ratio": audit_median / production_median,
        "diagnostic_audit_median_to_cold_total_ratio": audit_median
        / (preparation_ns + cold_execution_ns),
    }
    if any(value != cold_semantic["trace_hash"] for value in production_hashes):
        raise ValueError("NRIR-16 production timing semantic identity drifted")
    return timing, cold_semantic


def build_evidence(args: argparse.Namespace) -> dict[str, object]:
    baseline = _load_baseline_reference(args.baseline_artifact_dir)
    context, prepared, preparation_ns = _prepare_context(args)
    timing, production = _measure(context, prepared, preparation_ns=preparation_ns)
    comparison = _compare_semantics(production, baseline)
    production_trace = _mapping(production["trace"], "production trace")
    gates = {
        "source_model_property_and_intermediates_are_digest_bound": True,
        "prepared_and_audit_clause_statuses_match": bool(
            comparison["statuses_same"] is True
            and production_trace["unresolved_clause_indices"] == [0, 2, 4]
        ),
        "prepared_and_audit_bounds_are_allclose": bool(
            comparison["lower_allclose"] is True
            and comparison["candidate_exact"] is True
        ),
        "production_omits_audit_hash_chain_and_selected_native_reexecution": bool(
            production_trace["audit_hash_chain_constructed"] is False
            and production_trace["selected_native_reexecution"] is False
        ),
        "three_group_warm_diagnostic_reduction_exceeds_ten_x": bool(
            cast(float, timing["diagnostic_audit_to_warm_median_ratio"])
            >= MIN_WARM_DIAGNOSTIC_REDUCTION
        ),
        "cold_preparation_and_first_execution_are_exposed": bool(
            cast(int, timing["preparation_ns"]) > 0
            and cast(int, timing["cold_first_execution_ns"]) > 0
            and cast(int, timing["prepared_payload_unique_bytes"]) > 0
        ),
    }
    if not all(gates.values()):
        raise ValueError(f"NRIR-16 gates failed: {gates}")
    evidence = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "ok",
        "performance_claimed": False,
        "property_status": "validated_reduced",
        "claim_boundary": (
            "fixed ResNet CPU prepared-production mechanism and three-group internal "
            "audit-overhead diagnosis; not competitor speedup, CUDA, multi-workload, "
            "branching, or complete verification evidence"
        ),
        "compiler_version": NATIVE_PREPARED_COMPLETE_QUERY_COMPILER_VERSION,
        "source": {
            **context["source"],
            "baseline_manifest_sha256": baseline["manifest_sha256"],
            "baseline_artifact_sha256": baseline["artifact_sha256"],
            "baseline_evidence_hash": baseline["evidence_hash"],
            "baseline_query_trace_hash": baseline["query_trace_hash"],
        },
        "protocol": {
            "query_id": QUERY_ID,
            "audit_query_id": AUDIT_QUERY_ID,
            "search_policy": SEARCH_POLICY.to_dict(),
            "optimizer_policy": EXTERNAL_ADAPTIVE_POLICY.to_dict(),
            "queue_config": QUEUE_CONFIG.to_dict(),
            "intermediate_bound_source": "external_verifier",
            "groups": TIMING_GROUPS,
            "warmups": TIMING_WARMUPS,
            "orders": [list(order) for order in TIMING_ORDERS],
            "minimum_warm_diagnostic_reduction": MIN_WARM_DIAGNOSTIC_REDUCTION,
        },
        "semantic": {
            "baseline_reference": baseline,
            "prepared_production": production,
            "comparison": comparison,
        },
        "timing": timing,
        "gates": gates,
        "limitations": [
            "only one fixed ResNet/VNNLIB pair is measured on CPU",
            "prepared capsules are exact-scope root-only and do not cover child split queues",
            "cold preparation and retained prepared tensor payload are material costs",
            "audit-versus-production ratio measures internal evidence overhead, not competitor speedup",
            "clauses 0, 2, and 4 remain unknown and complete verification is not claimed",
        ],
    }
    validate_evidence(evidence)
    return evidence


def validate_evidence(evidence: Mapping[str, Any]) -> None:
    if (
        evidence.get("schema_version") != EVIDENCE_SCHEMA_VERSION
        or evidence.get("status") != "ok"
        or evidence.get("performance_claimed") is not False
        or evidence.get("property_status") != "validated_reduced"
        or evidence.get("compiler_version")
        != NATIVE_PREPARED_COMPLETE_QUERY_COMPILER_VERSION
    ):
        raise ValueError("NRIR-16 evidence header differs")
    claim = str(evidence.get("claim_boundary", ""))
    for phrase in (
        "fixed ResNet CPU",
        "three-group",
        "not competitor speedup",
        "complete verification evidence",
    ):
        if phrase not in claim:
            raise ValueError("NRIR-16 claim boundary differs")
    source = _mapping(evidence.get("source"), "NRIR-16 source")
    expected_source = {
        "model_sha256": MODEL_SHA256,
        "vnnlib_sha256": VNNLIB_SHA256,
        "intermediate_bounds_sha256": INTERMEDIATE_BOUNDS_SHA256,
        "vnncomp_commit": VNNCOMP_COMMIT,
        "abcrown_commit": ABCROWN_COMMIT,
    }
    if any(source.get(name) != value for name, value in expected_source.items()) or any(
        not _sha256(source.get(name))
        for name in (
            "baseline_manifest_sha256",
            "baseline_artifact_sha256",
            "baseline_evidence_hash",
            "baseline_query_trace_hash",
        )
    ):
        raise ValueError("NRIR-16 source identity differs")
    protocol = _mapping(evidence.get("protocol"), "NRIR-16 protocol")
    if (
        protocol.get("query_id") != QUERY_ID
        or protocol.get("audit_query_id") != AUDIT_QUERY_ID
        or protocol.get("groups") != TIMING_GROUPS
        or protocol.get("warmups") != TIMING_WARMUPS
        or protocol.get("orders") != [list(order) for order in TIMING_ORDERS]
        or protocol.get("minimum_warm_diagnostic_reduction")
        != MIN_WARM_DIAGNOSTIC_REDUCTION
    ):
        raise ValueError("NRIR-16 protocol differs")
    semantic = _mapping(evidence.get("semantic"), "NRIR-16 semantic")
    production = _mapping(semantic.get("prepared_production"), "NRIR-16 production")
    trace = _mapping(production.get("trace"), "NRIR-16 production trace")
    rows = _list(production.get("rows"), "NRIR-16 rows")
    comparison = _mapping(semantic.get("comparison"), "NRIR-16 comparison")
    if (
        production.get("trace_hash") != canonical_hash(trace)
        or trace.get("status") != "unknown"
        or trace.get("unresolved_clause_indices") != [0, 2, 4]
        or trace.get("execution_mode") != "prepared_production_root_only"
        or trace.get("audit_hash_chain_constructed") is not False
        or trace.get("selected_native_reexecution") is not False
        or len(rows) != 9
        or comparison.get("statuses_same") is not True
        or comparison.get("lower_allclose") is not True
        or comparison.get("candidate_exact") is not True
        or float(comparison.get("lower_max_abs_diff", math.inf)) > NUMERIC_ATOL
    ):
        raise ValueError("NRIR-16 semantic boundary differs")
    timing = _mapping(evidence.get("timing"), "NRIR-16 timing")
    samples = _list(timing.get("samples"), "NRIR-16 samples")
    if (
        timing.get("performance_claimed") is not False
        or timing.get("groups") != TIMING_GROUPS
        or timing.get("warmups") != TIMING_WARMUPS
        or timing.get("orders") != [list(order) for order in TIMING_ORDERS]
        or len(samples) != TIMING_GROUPS * 2
        or not all(
            isinstance(timing.get(name), int) and int(timing[name]) > 0
            for name in (
                "preparation_ns",
                "cold_first_execution_ns",
                "cold_prepare_plus_first_execution_ns",
                "prepared_payload_unique_bytes",
            )
        )
        or float(timing.get("diagnostic_audit_to_warm_median_ratio", 0.0))
        < MIN_WARM_DIAGNOSTIC_REDUCTION
    ):
        raise ValueError("NRIR-16 timing boundary differs")
    variants: dict[str, int] = {"audit": 0, "prepared_production": 0}
    for sample in samples:
        variant = str(sample.get("variant"))
        if (
            variant not in variants
            or sample.get("group") not in range(TIMING_GROUPS)
            or not isinstance(sample.get("elapsed_ns"), int)
            or int(sample["elapsed_ns"]) <= 0
            or sample.get("status") != "unknown"
            or sample.get("unresolved_clause_indices") != [0, 2, 4]
            or not _sha256(sample.get("semantic_hash"))
        ):
            raise ValueError("NRIR-16 timing sample differs")
        variants[variant] += 1
    if variants != {"audit": 3, "prepared_production": 3} or timing.get(
        "summary"
    ) != _timing_summary(samples):
        raise ValueError("NRIR-16 timing summary differs")
    gates = _mapping(evidence.get("gates"), "NRIR-16 gates")
    if len(gates) != 6 or any(value is not True for value in gates.values()):
        raise ValueError("NRIR-16 gates differ")
    limitations = evidence.get("limitations")
    if not isinstance(limitations, list) or len(limitations) != 5:
        raise ValueError("NRIR-16 limitations differ")


def _generate(args: argparse.Namespace) -> None:
    evidence = build_evidence(args)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "ok",
        "evidence": evidence,
    }
    artifact_path = args.artifact_dir / ARTIFACT_FILE
    artifact_path.write_text(
        _canonical_json(artifact, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "ok",
        "performance_claimed": False,
        "files": {ARTIFACT_FILE: file_sha256(artifact_path)},
        "evidence_hash": canonical_hash(evidence),
    }
    (args.artifact_dir / MANIFEST_FILE).write_text(
        _canonical_json(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(_canonical_json({"status": "ok", "evidence_hash": manifest["evidence_hash"]}))


def _replay(args: argparse.Namespace) -> None:
    manifest_path = args.artifact_dir / MANIFEST_FILE
    artifact_path = args.artifact_dir / ARTIFACT_FILE
    manifest = _load_json(manifest_path)
    artifact = _load_json(artifact_path)
    if (
        manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION
        or manifest.get("status") != "ok"
        or manifest.get("performance_claimed") is not False
        or manifest.get("files") != {ARTIFACT_FILE: file_sha256(artifact_path)}
        or artifact.get("schema_version") != ARTIFACT_SCHEMA_VERSION
        or artifact.get("status") != "ok"
    ):
        raise ValueError("NRIR-16 artifact manifest/header differs")
    stored = _mapping(artifact.get("evidence"), "NRIR-16 stored evidence")
    validate_evidence(stored)
    if manifest.get("evidence_hash") != canonical_hash(stored):
        raise ValueError("NRIR-16 stored evidence hash differs")
    context, prepared, _preparation_ns = _prepare_context(args)
    actual = _production_semantic(_execute_prepared(context, prepared))
    semantic = _mapping(stored.get("semantic"), "NRIR-16 replay semantic")
    if semantic.get("prepared_production") != actual:
        raise ValueError("NRIR-16 semantic replay differs")
    print(_canonical_json({"status": "ok", "evidence_hash": manifest["evidence_hash"]}))


def main() -> None:
    args = _parse_args()
    if args.torch_threads < 1:
        raise ValueError("NRIR-16 torch thread count must be positive")
    torch.set_num_threads(args.torch_threads)
    if args.command == "generate":
        _generate(args)
    else:
        _replay(args)


if __name__ == "__main__":
    main()
