#!/usr/bin/env python3
"""Generate or replay the NRIR-23 external-seeded ancestral artifact."""

# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=too-many-arguments,too-many-boolean-expressions,duplicate-code
# pylint: disable=missing-function-docstring,line-too-long

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any, Mapping

import torch

from boundflow.ir.bound import IntermediateBoundSource
from boundflow.ir.refinement import NativeIntermediateRefinementPolicyIR
from boundflow.runtime.abcrown_adapter import file_sha256
from boundflow.runtime.native_intermediate_refinement import (
    NativeExternalIntermediateConstraintSeed,
    build_native_external_intermediate_constraint_seed,
    compile_native_intermediate_refinement_program,
    execute_native_intermediate_refinement_program,
)
from boundflow.runtime.native_optimized_relu_split_bab_runtime import (
    NativeOptimizedReluSplitBabExecution,
    execute_native_optimized_relu_split_bab,
)
from scripts.run_end_to_end_tightness_performance_baseline import _build_context
from scripts.run_hard_clause_objective_branching_artifact import (
    BRANCH_POLICY,
    HARD_CLAUSES,
    OPTIMIZER_POLICY,
    QUEUE_CONFIG,
    _canonical_json,
    _list,
    _load_json,
    _mapping,
    canonical_hash,
)
from scripts.run_native_real_network_ir_artifact import (
    ABCROWN_COMMIT,
    INTERMEDIATE_BOUNDS_SHA256,
    MODEL_SHA256,
    VNNCOMP_COMMIT,
    VNNLIB_SHA256,
)
from scripts.run_per_child_objective_refinement_artifact import (
    BACKWARD_CHUNK_SIZE,
    TARGETS_PER_RELU,
    _serialize_refinement,
)

ARTIFACT_SCHEMA_VERSION = "boundflow.external-seeded-ancestral-refinement-artifact/v1"
EVIDENCE_SCHEMA_VERSION = "boundflow.external-seeded-ancestral-refinement-evidence/v1"
ARTIFACT_FILE = "external_seeded_ancestral.json"
MANIFEST_FILE = "manifest.json"
MODES = (
    "external_baseline",
    "external_seeded_root_global",
    "external_seeded_ancestral",
)
REFINEMENT_POLICY = NativeIntermediateRefinementPolicyIR(
    passes=1,
    max_neurons_per_relu=TARGETS_PER_RELU,
    backward_chunk_size=BACKWARD_CHUNK_SIZE,
    candidate_policy_id="objective_influence_width_per_relu_v1",
)


def _serialize_semantic_refinement(node_id: str, execution: Any) -> dict[str, object]:
    row = _serialize_refinement(node_id, execution)
    trace = dict(_mapping(row["execution_trace"], "NRIR-23 execution trace"))
    trace.pop("elapsed_ns")
    row["execution_trace"] = trace
    return row


def _objective_branch_hash_evidence(
    execution: NativeOptimizedReluSplitBabExecution,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for node_id, branch in execution.objective_branch_executions:
        program = branch.program
        rows.append(
            {
                "node_id": node_id,
                "plan_hash": program.plan.stable_hash(),
                "task_module_hash": program.task_module.stable_hash(plan=program.plan),
                "schedule_hash": program.schedule.stable_hash(
                    plan=program.plan, task_module=program.task_module
                ),
                "trace_hash": branch.trace.stable_hash(program=program),
            }
        )
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("generate", "replay"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--model", type=Path, required=True)
        subparser.add_argument("--source-artifact-dir", type=Path, required=True)
        subparser.add_argument("--local-artifact-dir", type=Path, required=True)
        subparser.add_argument("--artifact-dir", type=Path, required=True)
        subparser.add_argument("--torch-threads", type=int, default=8)
    return parser.parse_args()


def _external_seed(
    context: Mapping[str, Any], *, clause: int
) -> NativeExternalIntermediateConstraintSeed:
    source = _mapping(context["source"], "NRIR-23 source")
    return build_native_external_intermediate_constraint_seed(
        context["module"],
        context["input_spec"],
        seed_id=f"nrir23:resnet2b:property0:clause{clause}:external-seed",
        provider="alpha-beta-CROWN",
        constraints=context["external_relu_pre"],
        external_intermediate_bounds_hash=INTERMEDIATE_BOUNDS_SHA256,
        source_artifact_manifest_hash=str(source["native_ir_manifest_sha256"]),
        source_artifact_payload_hash=str(source["native_ir_payload_sha256"]),
        source_model_hash=MODEL_SHA256,
        source_property_hash=VNNLIB_SHA256,
        source_objective_set_hash=str(source["objective_matrix_hash"]),
    )


def _semantic(
    execution: NativeOptimizedReluSplitBabExecution,
    *,
    mode: str,
    refinements: list[dict[str, object]],
) -> dict[str, object]:
    execution.validate()
    trace = execution.trace
    leaf_lowers = [
        float(item.lower)
        for item in trace.evaluations
        if item.node.depth == QUEUE_CONFIG.max_depth
    ]
    if len(trace.evaluations) != 7 or len(leaf_lowers) != 4:
        raise ValueError("NRIR-23 bounded-tree coverage differs")
    root_lower = float(trace.evaluations[0].lower)
    return {
        "mode": mode,
        "queue_trace": trace.to_dict(),
        "queue_trace_hash": trace.stable_hash(),
        "objective_branch_hashes": _objective_branch_hash_evidence(execution),
        "refinements": refinements,
        "summary": {
            "root_lower": root_lower,
            "leaf_lowers": leaf_lowers,
            "leaf_worst_lower": min(leaf_lowers),
            "leaf_best_lower": max(leaf_lowers),
            "worst_leaf_improvement_over_root": min(leaf_lowers) - root_lower,
            "proof_deficit": max(0.0, -min(leaf_lowers)),
            "evaluated_nodes": len(trace.evaluations),
            "terminal_leaves": len(leaf_lowers),
            "queue_status": trace.status,
            "fixed_tree_property_status": (
                "verified" if min(leaf_lowers) >= 0.0 else "unknown"
            ),
        },
    }


def _run_mode(
    context: Mapping[str, Any], *, clause: int, mode: str
) -> tuple[dict[str, object], int]:
    tensors = _mapping(context["tensors"], "NRIR-23 tensors")
    objective = tensors["linear_spec_c"][:, clause : clause + 1]
    seed = _external_seed(context, clause=clause)
    started_ns = time.perf_counter_ns()
    refinements: list[dict[str, object]] = []
    if mode == "external_baseline":
        execution = execute_native_optimized_relu_split_bab(
            context["module"],
            context["input_spec"],
            linear_spec_C=objective,
            run_id=f"nrir23:c{clause}:{mode}",
            config=QUEUE_CONFIG,
            optimizer_policy=OPTIMIZER_POLICY,
            relu_pre_override=context["external_relu_pre"],
            intermediate_bound_source=IntermediateBoundSource.EXTERNAL_VERIFIER,
            objective_branch_policy=BRANCH_POLICY,
        )
    elif mode == "external_seeded_root_global":
        root_program = compile_native_intermediate_refinement_program(
            context["module"],
            context["input_spec"],
            policy=REFINEMENT_POLICY,
            plan_id=f"nrir23:c{clause}:external-seeded-root",
            linear_spec_C=objective,
            external_constraint_seed=seed,
        )
        root = execute_native_intermediate_refinement_program(
            root_program, context["module"], context["input_spec"]
        )
        execution = execute_native_optimized_relu_split_bab(
            context["module"],
            context["input_spec"],
            linear_spec_C=objective,
            run_id=f"nrir23:c{clause}:{mode}",
            config=QUEUE_CONFIG,
            optimizer_policy=OPTIMIZER_POLICY,
            relu_pre_override=root.relu_pre,
            intermediate_bound_source=IntermediateBoundSource.NATIVE_REFINED,
            objective_branch_policy=BRANCH_POLICY,
        )
        refinements = [
            _serialize_semantic_refinement(
                execution.trace.evaluations[0].node.node_id, root
            )
        ]
    elif mode == "external_seeded_ancestral":
        execution = execute_native_optimized_relu_split_bab(
            context["module"],
            context["input_spec"],
            linear_spec_C=objective,
            run_id=f"nrir23:c{clause}:{mode}",
            config=QUEUE_CONFIG,
            optimizer_policy=OPTIMIZER_POLICY,
            intermediate_bound_source=IntermediateBoundSource.NATIVE_REFINED,
            objective_branch_policy=BRANCH_POLICY,
            per_child_refinement_policy=REFINEMENT_POLICY,
            per_child_refinement_strategy="external_seeded_ancestral_carry_v1",
            external_constraint_seed=seed,
        )
        refinements = [
            _serialize_semantic_refinement(node_id, refinement)
            for node_id, refinement in execution.per_child_refinement_executions
        ]
    else:
        raise ValueError(f"unknown NRIR-23 mode: {mode}")
    elapsed_ns = time.perf_counter_ns() - started_ns
    return _semantic(execution, mode=mode, refinements=refinements), elapsed_ns


def _run_protocol(context: Mapping[str, Any]) -> tuple[dict[str, object], list[object]]:
    clauses: dict[str, object] = {}
    timings: list[object] = []
    orders = {
        0: MODES,
        2: (MODES[2], MODES[0], MODES[1]),
        4: (MODES[1], MODES[2], MODES[0]),
    }
    for clause in HARD_CLAUSES:
        modes: dict[str, object] = {}
        for mode in orders[clause]:
            semantic, elapsed_ns = _run_mode(context, clause=clause, mode=mode)
            modes[mode] = semantic
            timings.append(
                {"clause_index": clause, "mode": mode, "elapsed_ns": elapsed_ns}
            )
        clauses[str(clause)] = {mode: modes[mode] for mode in MODES}
    return clauses, timings


def _comparison(modes: Mapping[str, Any], clause: int) -> dict[str, object]:
    summaries = {
        mode: _mapping(_mapping(modes[mode], mode)["summary"], f"{mode} summary")
        for mode in MODES
    }
    baseline = float(summaries["external_baseline"]["leaf_worst_lower"])
    root_global = float(summaries["external_seeded_root_global"]["leaf_worst_lower"])
    ancestral = float(summaries["external_seeded_ancestral"]["leaf_worst_lower"])
    return {
        "clause_index": clause,
        "external_baseline_worst_leaf_lower": baseline,
        "external_seeded_root_global_worst_leaf_lower": root_global,
        "external_seeded_ancestral_worst_leaf_lower": ancestral,
        "root_refinement_delta": root_global - baseline,
        "ancestral_over_root_global_delta": ancestral - root_global,
        "ancestral_over_external_baseline_delta": ancestral - baseline,
        "ancestral_not_weaker_than_root_global": ancestral >= root_global - 1e-6,
        "ancestral_strictly_improves_root_global": ancestral > root_global + 1e-6,
        "ancestral_fixed_tree_status": summaries["external_seeded_ancestral"][
            "fixed_tree_property_status"
        ],
        "performance_claimed": False,
    }


def build_evidence(args: argparse.Namespace) -> dict[str, object]:
    context = _build_context(
        model=args.model,
        source_artifact_dir=args.source_artifact_dir,
        local_artifact_dir=args.local_artifact_dir,
    )
    clauses, timings = _run_protocol(context)
    comparisons = [
        _comparison(_mapping(clauses[str(clause)], "NRIR-23 modes"), clause)
        for clause in HARD_CLAUSES
    ]
    not_weaker = all(
        item["ancestral_not_weaker_than_root_global"] is True for item in comparisons
    )
    strictly_better = any(
        item["ancestral_strictly_improves_root_global"] is True for item in comparisons
    )
    evidence: dict[str, object] = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": (
            "validated_reduced" if not_weaker and strictly_better else "validated_no_go"
        ),
        "performance_claimed": False,
        "claim_boundary": (
            "fixed ResNet property 0 clauses 0/2/4, CPU, objective branching, "
            "seven-node depth-two comparison of frozen external constraints, "
            "external-seeded root refinement, and external-seeded native ancestry"
        ),
        "source": {
            **_mapping(context["source"], "NRIR-23 source"),
            "model_sha256": file_sha256(args.model),
            "vnnlib_sha256": VNNLIB_SHA256,
            "intermediate_bounds_sha256": INTERMEDIATE_BOUNDS_SHA256,
            "vnncomp_commit": VNNCOMP_COMMIT,
            "abcrown_commit": ABCROWN_COMMIT,
        },
        "protocol": {
            "hard_clause_indices": list(HARD_CLAUSES),
            "modes": list(MODES),
            "queue_config": QUEUE_CONFIG.to_dict(),
            "optimizer_policy": OPTIMIZER_POLICY.to_dict(),
            "branch_policy": BRANCH_POLICY.to_dict(),
            "refinement_policy": REFINEMENT_POLICY.to_dict(),
            "timing_role": "single-run diagnostic only",
        },
        "clauses": clauses,
        "comparisons": comparisons,
        "timing": timings,
        "gates": {
            "typed_external_seed_ir": True,
            "external_semantics_owner_preserved": True,
            "root_seed_and_parent_source_are_mutually_exclusive": True,
            "plan_task_schedule_and_action_trace_bind_seed": True,
            "same_tree_optimizer_branch_and_refinement_budget": True,
            "ancestral_not_weaker_on_all_hard_clauses": not_weaker,
            "ancestral_strictly_improves_at_least_one_hard_clause": strictly_better,
        },
        "limitations": [
            "fixed ResNet property 0 clauses 0, 2, and 4 on CPU only",
            "external seed generation remains owned by alpha-beta-CROWN",
            "seven-node depth-two bounded trees are not complete verifier results",
            "single-run timing is diagnostic and not a performance claim",
            "GPU, multi-workload, cuts, and full activation-BaB remain pending",
        ],
    }
    validate_evidence(evidence)
    return evidence


def _validate_mode(mode: str, value: Mapping[str, Any]) -> None:
    trace = _mapping(value.get("queue_trace"), "NRIR-23 queue trace")
    evaluations = _list(trace.get("evaluations"), "NRIR-23 evaluations")
    refinements = _list(value.get("refinements"), "NRIR-23 refinements")
    branches = _list(value.get("objective_branch_hashes"), "NRIR-23 branches")
    summary = _mapping(value.get("summary"), "NRIR-23 summary")
    expected_refinements = {MODES[0]: 0, MODES[1]: 1, MODES[2]: 7}[mode]
    if (
        value.get("mode") != mode
        or trace.get("status") != "complete"
        or trace.get("final_frontier_node_ids") != []
        or value.get("queue_trace_hash") != canonical_hash(trace)
        or len(evaluations) != 7
        or len(branches) != 7
        or len(refinements) != expected_refinements
    ):
        raise ValueError("NRIR-23 mode coverage differs")
    branch_nodes: set[str] = set()
    for branch_value in branches:
        branch = _mapping(branch_value, "NRIR-23 branch hashes")
        node_id = branch.get("node_id")
        if (
            set(branch)
            != {
                "node_id",
                "plan_hash",
                "task_module_hash",
                "schedule_hash",
                "trace_hash",
            }
            or not isinstance(node_id, str)
            or not node_id
            or node_id in branch_nodes
            or any(
                not isinstance(branch.get(name), str)
                or len(str(branch[name])) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in str(branch[name])
                )
                for name in (
                    "plan_hash",
                    "task_module_hash",
                    "schedule_hash",
                    "trace_hash",
                )
            )
        ):
            raise ValueError("NRIR-23 objective branch hash evidence differs")
        branch_nodes.add(node_id)
    leaf_lowers = [
        float(_mapping(item, "NRIR-23 evaluation")["lower"])
        for item in evaluations
        if _mapping(_mapping(item, "NRIR-23 evaluation")["node"], "NRIR-23 node")[
            "depth"
        ]
        == 2
    ]
    root_lower = float(_mapping(evaluations[0], "NRIR-23 root")["lower"])
    if (
        len(leaf_lowers) != 4
        or summary.get("root_lower") != root_lower
        or summary.get("leaf_lowers") != leaf_lowers
        or summary.get("leaf_worst_lower") != min(leaf_lowers)
        or summary.get("leaf_best_lower") != max(leaf_lowers)
        or summary.get("proof_deficit") != max(0.0, -min(leaf_lowers))
        or summary.get("fixed_tree_property_status")
        != ("verified" if min(leaf_lowers) >= 0.0 else "unknown")
    ):
        raise ValueError("NRIR-23 summary differs")
    external_seed_count = 0
    for row in refinements:
        refinement = _mapping(row, "NRIR-23 refinement")
        plan = _mapping(refinement.get("plan"), "NRIR-23 refinement plan")
        task = _mapping(refinement.get("task"), "NRIR-23 refinement task")
        schedule = _mapping(refinement.get("schedule"), "NRIR-23 refinement schedule")
        execution_trace = _mapping(
            refinement.get("execution_trace"), "NRIR-23 refinement trace"
        )
        hashes = _mapping(refinement.get("hashes"), "NRIR-23 refinement hashes")
        deterministic = dict(execution_trace)
        deterministic.pop("elapsed_ns", None)
        plan_hash = canonical_hash(plan)
        task_hash = canonical_hash(task)
        schedule_hash = canonical_hash(schedule)
        seed = plan.get("external_constraint_seed")
        if seed is not None:
            seed = _mapping(seed, "NRIR-23 external seed")
            external_seed_count += 1
            first_task = _mapping(
                _list(task.get("tasks"), "NRIR-23 tasks")[0], "NRIR-23 first task"
            )
            if (
                seed.get("semantics_owner") != "external_verifier"
                or seed.get("external_intermediate_bounds_hash")
                != INTERMEDIATE_BOUNDS_SHA256
                or "refine.external_constraint_seed"
                not in _list(first_task.get("input_value_ids"), "NRIR-23 inputs")
            ):
                raise ValueError("NRIR-23 external seed binding differs")
        if (
            plan.get("schema_version") != "boundflow.intermediate_refinement_plan_ir/v1"
            or task.get("schema_version")
            != "boundflow.intermediate_refinement_task_ir/v1"
            or schedule.get("schema_version")
            != "boundflow.intermediate_refinement_schedule_ir/v1"
            or hashes.get("refinement_plan_hash") != plan_hash
            or hashes.get("refinement_task_module_hash") != task_hash
            or hashes.get("refinement_schedule_hash") != schedule_hash
            or task.get("refinement_plan_hash") != plan_hash
            or schedule.get("refinement_plan_hash") != plan_hash
            or schedule.get("refinement_task_module_hash") != task_hash
            or execution_trace.get("plan_hash") != plan_hash
            or execution_trace.get("task_module_hash") != task_hash
            or execution_trace.get("schedule_hash") != schedule_hash
            or refinement.get("semantic_execution_trace_hash")
            != canonical_hash(deterministic)
        ):
            raise ValueError("NRIR-23 refinement IR evidence differs")
    if external_seed_count != (0 if mode == MODES[0] else 1):
        raise ValueError("NRIR-23 external seed root coverage differs")
    if mode == MODES[2]:
        records = _list(trace.get("per_child_refinements"), "NRIR-23 queue refinements")
        if (
            trace.get("per_child_refinement_strategy")
            != "external_seeded_ancestral_carry_v1"
            or "external_constraint_seed_hash"
            not in _mapping(records[0], "NRIR-23 root record")
            or sum(
                "source_parent_node_id" in _mapping(item, "NRIR-23 child record")
                for item in records
            )
            != 6
        ):
            raise ValueError("NRIR-23 ancestral lineage differs")
        refinement_by_node = {
            str(_mapping(row, "NRIR-23 refinement")["node_id"]): _mapping(
                row, "NRIR-23 refinement"
            )
            for row in refinements
        }
        record_by_node = {
            str(_mapping(row, "NRIR-23 queue record")["node_id"]): _mapping(
                row, "NRIR-23 queue record"
            )
            for row in records
        }
        for index, evaluation_value in enumerate(evaluations):
            evaluation = _mapping(evaluation_value, "NRIR-23 evaluation")
            node = _mapping(evaluation["node"], "NRIR-23 node")
            node_id = str(node["node_id"])
            record = record_by_node[node_id]
            refinement = refinement_by_node[node_id]
            plan = _mapping(refinement["plan"], "NRIR-23 refinement plan")
            execution_trace = _mapping(
                refinement["execution_trace"], "NRIR-23 execution trace"
            )
            if (
                record != _mapping(records[index], "NRIR-23 ordered record")
                or evaluation.get("intermediate_refinement_trace_hash")
                != canonical_hash(record)
                or record.get("refinement_plan_hash") != canonical_hash(plan)
                or record.get("refinement_semantic_trace_hash")
                != refinement.get("semantic_execution_trace_hash")
                or record.get("final_intermediate_bounds_hash")
                != execution_trace.get("final_intermediate_bounds_hash")
            ):
                raise ValueError("NRIR-23 queue/refinement binding differs")
            parent_id = node.get("parent_node_id")
            if parent_id is None:
                if (
                    record.get("external_constraint_seed_hash")
                    != plan.get("external_constraint_seed_hash")
                    or "source_parent_node_id" in record
                ):
                    raise ValueError("NRIR-23 root seed lineage differs")
                continue
            parent = record_by_node[str(parent_id)]
            if (
                "external_constraint_seed_hash" in record
                or record.get("source_parent_node_id") != parent_id
                or record.get("source_intermediate_constraints_hash")
                != parent.get("final_intermediate_bounds_hash")
                or record.get("source_refinement_plan_hash")
                != parent.get("refinement_plan_hash")
                or record.get("source_refinement_semantic_trace_hash")
                != parent.get("refinement_semantic_trace_hash")
                or plan.get("source_intermediate_constraints_hash")
                != record.get("source_intermediate_constraints_hash")
            ):
                raise ValueError("NRIR-23 child parent lineage differs")


def validate_evidence(evidence: Mapping[str, Any]) -> None:
    if (
        evidence.get("schema_version") != EVIDENCE_SCHEMA_VERSION
        or evidence.get("status") not in {"validated_reduced", "validated_no_go"}
        or evidence.get("performance_claimed") is not False
    ):
        raise ValueError("NRIR-23 evidence header differs")
    source = _mapping(evidence.get("source"), "NRIR-23 source")
    if (
        source.get("model_sha256") != MODEL_SHA256
        or source.get("vnnlib_sha256") != VNNLIB_SHA256
        or source.get("intermediate_bounds_sha256") != INTERMEDIATE_BOUNDS_SHA256
        or source.get("vnncomp_commit") != VNNCOMP_COMMIT
        or source.get("abcrown_commit") != ABCROWN_COMMIT
    ):
        raise ValueError("NRIR-23 source identity differs")
    clauses = _mapping(evidence.get("clauses"), "NRIR-23 clauses")
    comparisons = _list(evidence.get("comparisons"), "NRIR-23 comparisons")
    if set(clauses) != {str(item) for item in HARD_CLAUSES} or len(comparisons) != 3:
        raise ValueError("NRIR-23 hard-clause coverage differs")
    for clause, comparison_value in zip(HARD_CLAUSES, comparisons):
        modes = _mapping(clauses[str(clause)], "NRIR-23 modes")
        if set(modes) != set(MODES):
            raise ValueError("NRIR-23 mode coverage differs")
        for mode in MODES:
            _validate_mode(mode, _mapping(modes[mode], f"NRIR-23 {mode}"))
        expected = _comparison(modes, clause)
        if comparison_value != expected:
            raise ValueError("NRIR-23 comparison differs")
    gates = _mapping(evidence.get("gates"), "NRIR-23 gates")
    not_weaker = all(
        _mapping(item, "NRIR-23 comparison")["ancestral_not_weaker_than_root_global"]
        is True
        for item in comparisons
    )
    strictly_better = any(
        _mapping(item, "NRIR-23 comparison")["ancestral_strictly_improves_root_global"]
        is True
        for item in comparisons
    )
    expected_status = (
        "validated_reduced" if not_weaker and strictly_better else "validated_no_go"
    )
    if (
        evidence.get("status") != expected_status
        or len(gates) != 7
        or gates.get("ancestral_not_weaker_on_all_hard_clauses") is not not_weaker
        or gates.get("ancestral_strictly_improves_at_least_one_hard_clause")
        is not strictly_better
        or any(
            value is not True
            for key, value in gates.items()
            if key
            not in {
                "ancestral_not_weaker_on_all_hard_clauses",
                "ancestral_strictly_improves_at_least_one_hard_clause",
            }
        )
    ):
        raise ValueError("NRIR-23 gate/status differs")
    timings = _list(evidence.get("timing"), "NRIR-23 timing")
    limitations = _list(evidence.get("limitations"), "NRIR-23 limitations")
    if len(timings) != 9 or len(limitations) != 5:
        raise ValueError("NRIR-23 diagnostics/limitations differ")


def _generate(args: argparse.Namespace) -> None:
    evidence = build_evidence(args)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": evidence["status"],
        "performance_claimed": False,
        "evidence": evidence,
    }
    artifact_path = args.artifact_dir / ARTIFACT_FILE
    artifact_path.write_text(
        _canonical_json(artifact, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": evidence["status"],
        "performance_claimed": False,
        "files": {ARTIFACT_FILE: file_sha256(artifact_path)},
        "evidence_hash": canonical_hash(evidence),
    }
    (args.artifact_dir / MANIFEST_FILE).write_text(
        _canonical_json(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        _canonical_json(
            {"status": evidence["status"], "evidence_hash": manifest["evidence_hash"]}
        )
    )


def _replay(args: argparse.Namespace) -> None:
    manifest = _load_json(args.artifact_dir / MANIFEST_FILE)
    artifact_path = args.artifact_dir / ARTIFACT_FILE
    artifact = _load_json(artifact_path)
    stored = _mapping(artifact.get("evidence"), "NRIR-23 stored evidence")
    validate_evidence(stored)
    if (
        manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION
        or manifest.get("status") != stored["status"]
        or manifest.get("performance_claimed") is not False
        or manifest.get("files") != {ARTIFACT_FILE: file_sha256(artifact_path)}
        or manifest.get("evidence_hash") != canonical_hash(stored)
        or artifact.get("schema_version") != ARTIFACT_SCHEMA_VERSION
        or artifact.get("status") != stored["status"]
    ):
        raise ValueError("NRIR-23 artifact manifest/header differs")
    context = _build_context(
        model=args.model,
        source_artifact_dir=args.source_artifact_dir,
        local_artifact_dir=args.local_artifact_dir,
    )
    actual_clauses, _timing = _run_protocol(context)
    if stored.get("clauses") != actual_clauses:
        raise ValueError("NRIR-23 semantic replay differs")
    print(_canonical_json({"status": "ok", "evidence_hash": manifest["evidence_hash"]}))


def main() -> None:
    args = _parse_args()
    if args.torch_threads < 1:
        raise ValueError("NRIR-23 torch thread count must be positive")
    torch.set_num_threads(args.torch_threads)
    if args.command == "generate":
        _generate(args)
    else:
        _replay(args)


if __name__ == "__main__":
    main()
