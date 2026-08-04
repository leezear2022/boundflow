#!/usr/bin/env python3
"""Run or replay NRIR-35 NRIR-31-floor plus packed-clause feasibility."""

# pylint: disable=too-many-locals,too-many-statements,too-many-boolean-expressions
# pylint: disable=missing-function-docstring,import-outside-toplevel,duplicate-code

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

from scripts.run_objective_ancestral_queue_artifact import _active_frontier
from scripts.run_typed_hard_clause_escalation_artifact import (
    _load_query_runtime,
    _policies,
    _public_workload,
    _resolve_workloads,
)

EVIDENCE_SCHEMA_VERSION = "boundflow.cross-clause-anytime-feasibility/v1"
MANIFEST_SCHEMA_VERSION = "boundflow.cross-clause-anytime-feasibility-manifest/v1"
ARTIFACT_DIR = Path(
    "artifacts/cross-clause-anytime-objective-evaluator/"
    "vnncomp21-resnet2b-property0-clause0-cpu-feasibility-v1"
)
TORCH_THREADS = 8
CLAUSE_INDEX = 0
BOUND_ATOL = 1e-5


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("generate", "replay"))
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--torch-threads", type=int, default=TORCH_THREADS)
    return parser.parse_args()


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
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical_json(value, indent=2) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _code_revision() -> str:
    root = _repo_root()
    paths = (
        "boundflow/ir/objective_hard_clause_escalation.py",
        "boundflow/runtime/native_objective_hard_clause_escalation.py",
        "boundflow/ir/objective_ancestral_sibling_pack.py",
        "boundflow/runtime/native_objective_ancestral_sibling_pack.py",
        "scripts/run_cross_clause_anytime_objective_feasibility.py",
    )
    return _canonical_hash({path: _file_sha256(root / path) for path in paths})


def _semantic_evidence(evidence: Mapping[str, Any]) -> dict[str, object]:
    return {
        "workload": evidence["workload"],
        "protocol": evidence["protocol"],
        "nrir31": evidence["nrir31"],
        "packed": evidence["packed"],
        "comparison": evidence["comparison"],
        "mechanism_gate_passed": evidence["mechanism_gate_passed"],
        "claim_boundary": evidence["claim_boundary"],
        "performance_claimed": False,
    }


def _gate(evidence: Mapping[str, Any]) -> bool:
    nrir31 = evidence["nrir31"]
    packed = evidence["packed"]
    comparison = evidence["comparison"]
    return bool(
        nrir31["completed_objective_clause_indices"] == list(range(9))
        and nrir31["final_unresolved_clause_indices"] == list(range(9))
        and nrir31["final_status"] == "unknown"
        and nrir31["fallback_reason"] == "none"
        and packed["source_elapsed_ns"] >= nrir31["elapsed_ns"]
        and packed["accepted_nodes"] >= 3
        and packed["accepted_nodes"] == 1 + 2 * packed["sibling_group_count"]
        and packed["all_groups_atomic_pairs"] is True
        and comparison["exact_nrir31_clause_source_bound"] is True
        and comparison["root_lower_max_abs_diff"] <= BOUND_ATOL
        and comparison["root_upper_max_abs_diff"] <= BOUND_ATOL
        and comparison["baseline_original_ordinals_preserved"] is True
        and comparison["feasibility_floor_verified_preserved"] is True
        and comparison["single_global_start_consumed"] is True
    )


def validate_evidence(evidence: Mapping[str, Any]) -> None:
    if (
        evidence.get("schema_version") != EVIDENCE_SCHEMA_VERSION
        or evidence.get("status") not in {"ok", "no_go"}
        or evidence.get("source", {}).get("native_code_revision") != _code_revision()
        or evidence.get("performance_claimed") is not False
        or evidence.get("evidence_hash")
        != _canonical_hash(_semantic_evidence(evidence))
    ):
        raise ValueError("NRIR-35 feasibility envelope differs")
    if (
        evidence.get("protocol", {}).get("whole_query_timeout_seconds") != 60
        or evidence.get("protocol", {}).get("packed_original_clause_index")
        != CLAUSE_INDEX
        or evidence.get("mechanism_gate_passed") != _gate(evidence)
        or evidence.get("status")
        != ("ok" if evidence.get("mechanism_gate_passed") else "no_go")
        or evidence.get("claim_boundary")
        != "mechanism_feasibility_only_no_property_or_performance_claim"
    ):
        raise ValueError("NRIR-35 feasibility gate differs")


def _generate(args: argparse.Namespace) -> None:
    import torch

    from boundflow.runtime.native_intermediate_refinement import (
        intermediate_bounds_hash,
        intermediate_refinement_semantic_trace_hash,
    )
    from boundflow.runtime.native_objective_ancestral_sibling_pack import (
        compile_native_objective_ancestral_sibling_pack_plan,
        execute_native_objective_ancestral_sibling_pack_queue,
    )
    from boundflow.runtime.native_objective_hard_clause_escalation import (
        compile_native_objective_hard_clause_escalation_program,
        execute_native_objective_hard_clause_escalation_program,
    )

    torch.set_num_threads(args.torch_threads)
    workloads = _resolve_workloads(args.benchmark_root.resolve())
    workload = next(
        item for item in workloads if item["workload_id"] == "cifar10_resnet:000"
    )
    _query, tensors, module, input_spec = _load_query_runtime(
        Path(str(workload["model"])),
        Path(str(workload["property"])),
        "cifar10_resnet:000",
    )
    search_policy, optimizer_policy = _policies()
    program = compile_native_objective_hard_clause_escalation_program(
        module,
        input_spec,
        linear_spec_C=tensors.linear_spec_c,
        thresholds=tensors.thresholds,
        plan_id="nrir35:cifar10_resnet:000:property0:nrir31-floor",
        search_policy=search_policy,
        optimizer_policy=optimizer_policy,
    )
    whole_started_ns = time.monotonic_ns()
    floor = execute_native_objective_hard_clause_escalation_program(
        program,
        module,
        input_spec,
        linear_spec_C=tensors.linear_spec_c,
        thresholds=tensors.thresholds,
        query_id="nrir35:cifar10_resnet:000:property0:nrir31-floor",
        search_policy=search_policy,
        optimizer_policy=optimizer_policy,
    )
    child = next(
        item
        for item in floor.clause_executions
        if item.original_clause_index == CLAUSE_INDEX
    )
    if not child.accepted_before_deadline or not child.query.clauses:
        raise ValueError("NRIR-35 feasibility lacks accepted NRIR-31 clause 0")
    objective = tensors.linear_spec_c[
        :, CLAUSE_INDEX : CLAUSE_INDEX + 1, :
    ].contiguous()
    threshold = tensors.thresholds[CLAUSE_INDEX : CLAUSE_INDEX + 1].contiguous()
    plan = compile_native_objective_ancestral_sibling_pack_plan(
        module,
        input_spec,
        linear_spec_C=objective,
        threshold=threshold,
        root_refinement=child.refinement,
        optimizer_policy=optimizer_policy,
        plan_id="nrir35:cifar10_resnet:000:property0:clause0:packed-plan",
    )
    packed = execute_native_objective_ancestral_sibling_pack_queue(
        plan,
        module,
        input_spec,
        linear_spec_C=objective,
        threshold=threshold,
        root_refinement=child.refinement,
        optimizer_policy=optimizer_policy,
        query_id="nrir35:cifar10_resnet:000:property0:clause0:packed",
        whole_query_started_ns=whole_started_ns,
    )
    floor_root = child.query.clauses[0].queue.trace.evaluations[0]
    packed_root = packed.queue.trace.evaluations[0]
    frontier = _active_frontier(packed.queue.trace)
    exact_source = (
        plan.root_refinement_plan_hash == child.refinement.program.plan.stable_hash()
        and plan.root_refinement_semantic_trace_hash
        == intermediate_refinement_semantic_trace_hash(child.refinement)
        and plan.root_intermediate_bounds_hash
        == intermediate_bounds_hash(child.refinement.relu_pre)
    )
    nrir31: dict[str, Any] = {
        "program_hash": program.plan.stable_hash(),
        "trace_hash": floor.trace.semantic_signature_hash,
        "elapsed_ns": floor.trace.elapsed_ns,
        "completed_objective_clause_indices": list(
            floor.trace.completed_objective_clause_indices
        ),
        "final_status": floor.trace.final_status,
        "final_verified_clause_indices": list(
            floor.trace.final_verified_clause_indices
        ),
        "final_unresolved_clause_indices": list(
            floor.trace.final_unresolved_clause_indices
        ),
        "fallback_reason": floor.trace.fallback_reason,
        "clause0_root_lower": floor_root.lower,
        "clause0_root_upper": floor_root.upper,
        "clause0_refinement_plan_hash": child.refinement.program.plan.stable_hash(),
        "clause0_refinement_trace_hash": (
            intermediate_refinement_semantic_trace_hash(child.refinement)
        ),
    }
    packed_summary: dict[str, Any] = {
        "plan_hash": plan.stable_hash(),
        "aggregate_trace_hash": packed.trace.semantic_signature_hash,
        "queue_trace_hash": packed.queue.trace.stable_hash(),
        "source_elapsed_ns": packed.trace.source_elapsed_ns,
        "queue_elapsed_ns": packed.trace.queue_elapsed_ns,
        "whole_elapsed_ns": packed.trace.whole_elapsed_ns,
        "accepted_nodes": frontier["evaluated_nodes"],
        "maximum_depth": frontier["maximum_depth"],
        "worst_active_lower": frontier["worst_active_lower"],
        "root_lower": packed_root.lower,
        "root_upper": packed_root.upper,
        "sibling_group_count": len(packed.sibling_groups),
        "all_groups_atomic_pairs": all(
            item.child_branch_values == (-1, 1) for item in packed.sibling_groups
        ),
        "fallback_reason": packed.trace.fallback_reason,
        "discarded_attempt_stage": packed.trace.discarded_attempt_stage,
    }
    comparison: dict[str, Any] = {
        "exact_nrir31_clause_source_bound": exact_source,
        "root_lower_max_abs_diff": abs(floor_root.lower - packed_root.lower),
        "root_upper_max_abs_diff": abs(floor_root.upper - packed_root.upper),
        "baseline_original_ordinals_preserved": (
            floor.trace.completed_objective_clause_indices == tuple(range(9))
        ),
        "feasibility_floor_verified_preserved": (
            floor.trace.final_verified_clause_indices
            == tuple(nrir31["final_verified_clause_indices"])
        ),
        "single_global_start_consumed": (
            packed.trace.source_elapsed_ns >= floor.trace.elapsed_ns
            and packed.trace.whole_elapsed_ns >= packed.trace.source_elapsed_ns
        ),
    }
    evidence: dict[str, Any] = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "no_go",
        "source": {"native_code_revision": _code_revision()},
        "workload": _public_workload(workload),
        "protocol": {
            "whole_query_timeout_seconds": 60,
            "original_clause_count": 9,
            "packed_original_clause_index": CLAUSE_INDEX,
            "child_cap": 128,
            "search_budget": {"max_nodes": 31, "max_depth": 4},
            "torch_threads": args.torch_threads,
            "execution_order": ["nrir31_floor", "packed_clause0"],
        },
        "nrir31": nrir31,
        "packed": packed_summary,
        "comparison": comparison,
        "mechanism_gate_passed": False,
        "claim_boundary": (
            "mechanism_feasibility_only_no_property_or_performance_claim"
        ),
        "performance_claimed": False,
    }
    evidence["mechanism_gate_passed"] = _gate(evidence)
    evidence["status"] = "ok" if evidence["mechanism_gate_passed"] else "no_go"
    evidence["evidence_hash"] = _canonical_hash(_semantic_evidence(evidence))
    validate_evidence(evidence)
    artifact_dir = args.artifact_dir.resolve()
    evidence_path = artifact_dir / "evidence.json"
    _write_json(evidence_path, evidence)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "files": {"evidence.json": _file_sha256(evidence_path)},
        "evidence_hash": evidence["evidence_hash"],
    }
    _write_json(artifact_dir / "manifest.json", manifest)
    print(
        _canonical_json(
            {
                "status": evidence["status"],
                "nrir31_seconds": nrir31["elapsed_ns"] / 1e9,
                "packed_source_seconds": packed_summary["source_elapsed_ns"] / 1e9,
                "packed_nodes": packed_summary["accepted_nodes"],
                "root_lower_diff": comparison["root_lower_max_abs_diff"],
                "evidence_hash": evidence["evidence_hash"],
            }
        )
    )


def _replay(args: argparse.Namespace) -> None:
    artifact_dir = args.artifact_dir.resolve()
    manifest = _load_json(artifact_dir / "manifest.json")
    evidence = _load_json(artifact_dir / "evidence.json")
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or manifest.get("files", {}).get("evidence.json")
        != _file_sha256(artifact_dir / "evidence.json")
        or manifest.get("evidence_hash") != evidence.get("evidence_hash")
    ):
        raise ValueError("NRIR-35 feasibility manifest differs")
    validate_evidence(evidence)
    workloads = _resolve_workloads(args.benchmark_root.resolve())
    workload = next(
        item for item in workloads if item["workload_id"] == "cifar10_resnet:000"
    )
    if evidence.get("workload") != _public_workload(workload):
        raise ValueError("NRIR-35 feasibility workload differs")
    print(
        _canonical_json(
            {
                "status": evidence["status"],
                "mechanism_gate_passed": evidence["mechanism_gate_passed"],
                "evidence_hash": evidence["evidence_hash"],
            }
        )
    )


def main() -> None:
    args = _parse_args()
    if args.command == "generate":
        _generate(args)
    else:
        _replay(args)


if __name__ == "__main__":
    main()
