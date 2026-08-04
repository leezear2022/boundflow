#!/usr/bin/env python3
"""Generate or replay the NRIR-25 dynamic refinement-budget artifact."""

# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=too-many-arguments,too-many-boolean-expressions,duplicate-code
# pylint: disable=missing-function-docstring,line-too-long,too-many-lines

from __future__ import annotations

import argparse
from copy import deepcopy
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping

import torch

from boundflow.ir.bound import IntermediateBoundSource
from boundflow.ir.refinement import (
    NativeIntermediateRefinementBudgetDecisionIR,
    NativeIntermediateRefinementBudgetPolicyIR,
)
from boundflow.runtime.abcrown_adapter import file_sha256
from boundflow.runtime.native_optimized_relu_split_bab_runtime import (
    NativeOptimizedReluSplitBabExecution,
    execute_native_optimized_relu_split_bab,
)
from scripts.run_end_to_end_tightness_performance_baseline import _build_context
from scripts.run_external_seeded_ancestral_refinement_artifact import (
    REFINEMENT_POLICY,
    _objective_branch_hash_evidence,
)
from scripts.run_external_seeded_depth_node_convergence_artifact import (
    FROZEN_SOURCE,
    _external_seed,
    _is_sha256,
    _node_projection,
    _queue_config,
    _source,
    _summary,
    _write_json_atomic,
)
from scripts.run_hard_clause_objective_branching_artifact import (
    BRANCH_POLICY,
    HARD_CLAUSES,
    OPTIMIZER_POLICY,
    _canonical_json,
    _list,
    _load_json,
    _mapping,
    canonical_hash,
)

ARTIFACT_SCHEMA_VERSION = "boundflow.dynamic-refinement-budget-artifact/v1"
EVIDENCE_SCHEMA_VERSION = "boundflow.dynamic-refinement-budget-evidence/v1"
SHARD_SCHEMA_VERSION = "boundflow.dynamic-refinement-budget-shard/v1"
ARTIFACT_FILE = "dynamic_refinement_budget.json"
MANIFEST_FILE = "manifest.json"
SHARD_DIR = "shards"
MODES = ("fixed16", "dynamic8_24")
MAX_NODES = 31
MAX_DEPTH = 4
COMPARISON_TOLERANCE = 1e-6
DYNAMIC_BUDGET_POLICY = NativeIntermediateRefinementBudgetPolicyIR(
    base_max_neurons_per_relu=16,
    high_max_neurons_per_relu=24,
    low_max_neurons_per_relu=8,
)
QUEUE_CONFIG = _queue_config(MAX_NODES, MAX_DEPTH)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("generate", "replay"):
        subparser = subparsers.add_parser(command)
        _add_common_arguments(subparser)
        subparser.add_argument("--force", action="store_true")
    worker = subparsers.add_parser("worker")
    _add_common_arguments(worker, artifact_dir=False)
    worker.add_argument("--clause", type=int, required=True)
    worker.add_argument("--mode", choices=MODES, required=True)
    worker.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _add_common_arguments(
    parser: argparse.ArgumentParser, *, artifact_dir: bool = True
) -> None:
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-artifact-dir", type=Path, required=True)
    parser.add_argument("--local-artifact-dir", type=Path, required=True)
    if artifact_dir:
        parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--torch-threads", type=int, default=8)


def _shard_name(clause: int, mode: str) -> str:
    return f"clause{clause}_{mode}.json"


def _run_id(clause: int, mode: str) -> str:
    return f"nrir25:c{clause}:{mode}"


def _program_projections(
    execution: NativeOptimizedReluSplitBabExecution,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for node_id, refinement in execution.per_child_refinement_executions:
        program = refinement.program
        hashes = program.hashes()
        rows.append(
            {
                "node_id": node_id,
                "policy": program.plan.policy.to_dict(),
                "selected_target_count": len(program.plan.targets),
                **hashes,
            }
        )
    return rows


def _build_shard(
    context: Mapping[str, Any], *, model: Path, clause: int, mode: str
) -> dict[str, object]:
    if clause not in HARD_CLAUSES or mode not in MODES:
        raise ValueError("NRIR-25 worker identity is outside the frozen matrix")
    tensors = _mapping(context["tensors"], "NRIR-25 tensors")
    objective = tensors["linear_spec_c"][:, clause : clause + 1]
    seed = _external_seed(context, clause)
    dynamic = DYNAMIC_BUDGET_POLICY if mode == "dynamic8_24" else None
    execution = execute_native_optimized_relu_split_bab(
        context["module"],
        context["input_spec"],
        linear_spec_C=objective,
        run_id=_run_id(clause, mode),
        config=QUEUE_CONFIG,
        optimizer_policy=OPTIMIZER_POLICY,
        intermediate_bound_source=IntermediateBoundSource.NATIVE_REFINED,
        objective_branch_policy=BRANCH_POLICY,
        per_child_refinement_policy=REFINEMENT_POLICY,
        per_child_refinement_budget_policy=dynamic,
        per_child_refinement_strategy="external_seeded_ancestral_carry_v1",
        external_constraint_seed=seed,
    )
    execution.validate()
    trace = execution.trace
    body: dict[str, object] = {
        "schema_version": SHARD_SCHEMA_VERSION,
        "clause_index": clause,
        "mode": mode,
        "performance_claimed": False,
        "source": _source(context, model),
        "protocol": {
            "queue_config": QUEUE_CONFIG.to_dict(),
            "optimizer_policy": OPTIMIZER_POLICY.to_dict(),
            "branch_policy": BRANCH_POLICY.to_dict(),
            "base_refinement_policy": REFINEMENT_POLICY.to_dict(),
            "dynamic_budget_policy": (None if dynamic is None else dynamic.to_dict()),
            "strategy": "external_seeded_ancestral_carry_v1",
        },
        "external_seed_ir": seed.ir.to_dict(),
        "external_seed_hash": seed.stable_hash(),
        "queue": {
            "run_id": trace.run_id,
            "status": trace.status,
            "termination_reason": trace.termination_reason,
            "root_input_lower_hash": trace.root_input_lower_hash,
            "root_input_upper_hash": trace.root_input_upper_hash,
            "objective_hash": trace.objective_hash,
            "queue_trace_hash": trace.stable_hash(),
            "final_frontier_node_ids": list(trace.final_frontier_node_ids),
            "max_queue_size": trace.max_queue_size,
            "evaluations": _node_projection(execution),
            "decisions": [item.to_dict() for item in trace.decisions],
            "refinements": [item.to_dict() for item in trace.per_child_refinements],
            "refinement_programs": _program_projections(execution),
            "objective_branch_hashes": _objective_branch_hash_evidence(execution),
        },
        "summary": _summary(execution),
    }
    shard = {**body, "semantic_hash": canonical_hash(body)}
    validate_shard(shard, expected_clause=clause, expected_mode=mode)
    return shard


def _decision_from_dict(
    value: Mapping[str, Any], *, policy: NativeIntermediateRefinementBudgetPolicyIR
) -> NativeIntermediateRefinementBudgetDecisionIR:
    decision = NativeIntermediateRefinementBudgetDecisionIR(
        decision_id=str(value.get("decision_id", "")),
        budget_policy_hash=str(value.get("budget_policy_hash", "")),
        group_id=str(value.get("group_id", "")),
        group_semantic_hash=str(value.get("group_semantic_hash", "")),
        group_size=int(value.get("group_size", 0)),
        group_base_cap_total=int(value.get("group_base_cap_total", 0)),
        group_assigned_cap_total=int(value.get("group_assigned_cap_total", 0)),
        node_id=str(value.get("node_id", "")),
        node_split_state_hash=str(value.get("node_split_state_hash", "")),
        node_depth=int(value.get("node_depth", -1)),
        assigned_max_neurons_per_relu=int(
            value.get("assigned_max_neurons_per_relu", 0)
        ),
        allocation_rank=str(value.get("allocation_rank", "")),
        parent_node_id=(
            None
            if value.get("parent_node_id") is None
            else str(value["parent_node_id"])
        ),
        parent_lower=(
            None if value.get("parent_lower") is None else float(value["parent_lower"])
        ),
        semantics_owner=str(value.get("semantics_owner", "")),
    )
    decision.validate(policy=policy)
    if decision.to_dict(policy=policy) != value:
        raise ValueError("NRIR-25 budget decision payload differs")
    return decision


def _validate_budget_records(
    records: list[Any], *, mode: str
) -> tuple[int, dict[str, int]]:
    if mode == "fixed16":
        if any(
            "budget_decision" in _mapping(value, "NRIR-25 fixed refinement")
            or "budget_decision_hash" in _mapping(value, "NRIR-25 fixed refinement")
            for value in records
        ):
            raise ValueError("NRIR-25 fixed mode declares dynamic budget")
        return len(records) * REFINEMENT_POLICY.max_neurons_per_relu, {
            "fixed": len(records)
        }
    groups: dict[str, list[NativeIntermediateRefinementBudgetDecisionIR]] = {}
    ranks: dict[str, int] = {}
    for value in records:
        record = _mapping(value, "NRIR-25 dynamic refinement")
        decision_value = _mapping(
            record.get("budget_decision"), "NRIR-25 budget decision"
        )
        decision = _decision_from_dict(decision_value, policy=DYNAMIC_BUDGET_POLICY)
        if record.get("budget_decision_hash") != decision.stable_hash(
            policy=DYNAMIC_BUDGET_POLICY
        ):
            raise ValueError("NRIR-25 budget decision hash differs")
        groups.setdefault(decision.group_id, []).append(decision)
        ranks[decision.allocation_rank] = ranks.get(decision.allocation_rank, 0) + 1
    total = 0
    for decisions in groups.values():
        first = decisions[0]
        semantics = {
            "group_id": first.group_id,
            "budget_policy_hash": first.budget_policy_hash,
            "base_cap_total": first.group_base_cap_total,
            "assigned_cap_total": first.group_assigned_cap_total,
            "nodes": [
                {
                    "node_id": item.node_id,
                    "node_split_state_hash": item.node_split_state_hash,
                    "node_depth": item.node_depth,
                    "parent_node_id": item.parent_node_id,
                    "parent_lower": item.parent_lower,
                    "assigned_max_neurons_per_relu": (
                        item.assigned_max_neurons_per_relu
                    ),
                    "allocation_rank": item.allocation_rank,
                }
                for item in decisions
            ],
        }
        assigned = sum(item.assigned_max_neurons_per_relu for item in decisions)
        if (
            len(decisions) != first.group_size
            or assigned != first.group_assigned_cap_total
            or assigned != first.group_base_cap_total
            or canonical_hash(semantics) != first.group_semantic_hash
        ):
            raise ValueError("NRIR-25 budget group conservation differs")
        total += assigned
    if total != len(records) * REFINEMENT_POLICY.max_neurons_per_relu:
        raise ValueError("NRIR-25 tree budget conservation differs")
    return total, ranks


def _validate_queue(
    queue: Mapping[str, Any],
    summary: Mapping[str, Any],
    *,
    clause: int,
    mode: str,
) -> tuple[int, int, dict[str, int]]:
    evaluations = _list(queue.get("evaluations"), "NRIR-25 evaluations")
    decisions = _list(queue.get("decisions"), "NRIR-25 decisions")
    refinements = _list(queue.get("refinements"), "NRIR-25 refinements")
    programs = _list(queue.get("refinement_programs"), "NRIR-25 programs")
    branches = _list(queue.get("objective_branch_hashes"), "NRIR-25 branches")
    run_id = _run_id(clause, mode)
    if (
        queue.get("run_id") != run_id
        or queue.get("status") != "complete"
        or queue.get("termination_reason") != "configured_bounded_tree_exhausted"
        or queue.get("final_frontier_node_ids") != []
        or any(
            not _is_sha256(queue.get(name))
            for name in (
                "root_input_lower_hash",
                "root_input_upper_hash",
                "objective_hash",
                "queue_trace_hash",
            )
        )
        or len(evaluations) != MAX_NODES
        or len(decisions) != MAX_NODES
        or len(refinements) != MAX_NODES
        or len(programs) != MAX_NODES
        or len(branches) != MAX_NODES
    ):
        raise ValueError("NRIR-25 queue header/coverage differs")
    evaluation_by_id: dict[str, Mapping[str, Any]] = {}
    lowers: dict[str, float] = {}
    for index, value in enumerate(evaluations):
        evaluation = _mapping(value, "NRIR-25 evaluation")
        node = _mapping(evaluation.get("node"), "NRIR-25 node")
        node_id = node.get("node_id")
        lower = evaluation.get("lower")
        upper = evaluation.get("upper")
        if (
            node_id != f"{run_id}:n{index:06d}"
            or node_id in evaluation_by_id
            or not isinstance(node.get("depth"), int)
            or int(node["depth"]) > MAX_DEPTH
            or not isinstance(lower, (int, float))
            or not isinstance(upper, (int, float))
            or not math.isfinite(float(lower))
            or not math.isfinite(float(upper))
            or float(lower) > float(upper)
            or not _is_sha256(node.get("split_state_hash"))
            or not _is_sha256(evaluation.get("intermediate_refinement_trace_hash"))
        ):
            raise ValueError("NRIR-25 evaluation identity differs")
        parent_id = node.get("parent_node_id")
        if index == 0:
            if parent_id is not None or node.get("depth") != 0:
                raise ValueError("NRIR-25 root lineage differs")
        else:
            parent = evaluation_by_id.get(str(parent_id))
            if (
                parent is None
                or int(node["depth"])
                != int(_mapping(parent["node"], "NRIR-25 parent")["depth"]) + 1
            ):
                raise ValueError("NRIR-25 parent lineage differs")
        evaluation_by_id[str(node_id)] = evaluation
        lowers[str(node_id)] = float(lower)
    refinement_by_id: dict[str, Mapping[str, Any]] = {}
    for value in refinements:
        record = _mapping(value, "NRIR-25 refinement")
        node_id = str(record.get("node_id"))
        if (
            node_id in refinement_by_id
            or node_id not in evaluation_by_id
            or canonical_hash(record)
            != evaluation_by_id[node_id].get("intermediate_refinement_trace_hash")
        ):
            raise ValueError("NRIR-25 refinement binding differs")
        parent_id = _mapping(evaluation_by_id[node_id]["node"], "NRIR-25 node").get(
            "parent_node_id"
        )
        if parent_id is None:
            if not _is_sha256(record.get("external_constraint_seed_hash")):
                raise ValueError("NRIR-25 root seed binding differs")
        else:
            parent = refinement_by_id.get(str(parent_id))
            if (
                parent is None
                or record.get("source_parent_node_id") != parent_id
                or record.get("source_intermediate_constraints_hash")
                != parent.get("final_intermediate_bounds_hash")
                or record.get("source_refinement_plan_hash")
                != parent.get("refinement_plan_hash")
                or record.get("source_refinement_semantic_trace_hash")
                != parent.get("refinement_semantic_trace_hash")
            ):
                raise ValueError("NRIR-25 ancestral lineage differs")
        refinement_by_id[node_id] = record
    program_by_id: dict[str, Mapping[str, Any]] = {}
    for value in programs:
        program = _mapping(value, "NRIR-25 refinement program")
        node_id = str(program.get("node_id"))
        refinement_record = refinement_by_id.get(node_id)
        if (
            refinement_record is None
            or node_id in program_by_id
            or program.get("refinement_plan_hash")
            != refinement_record.get("refinement_plan_hash")
            or program.get("refinement_task_module_hash")
            != refinement_record.get("refinement_task_module_hash")
            or program.get("refinement_schedule_hash")
            != refinement_record.get("refinement_schedule_hash")
            or program.get("selected_target_count")
            != refinement_record.get("selected_target_count")
        ):
            raise ValueError("NRIR-25 refinement program linkage differs")
        expected_policy = REFINEMENT_POLICY.to_dict()
        if mode == "dynamic8_24":
            budget = _mapping(
                refinement_record.get("budget_decision"), "NRIR-25 decision"
            )
            expected_policy = {
                **expected_policy,
                "max_neurons_per_relu": budget["assigned_max_neurons_per_relu"],
            }
        if program.get("policy") != expected_policy:
            raise ValueError("NRIR-25 refinement Plan policy differs")
        program_by_id[node_id] = program
    planned_cap, rank_counts = _validate_budget_records(refinements, mode=mode)
    branch_ids: set[str] = set()
    for value in branches:
        branch = _mapping(value, "NRIR-25 objective branch")
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
            or node_id in branch_ids
            or node_id not in evaluation_by_id
            or any(
                not _is_sha256(branch.get(name)) for name in set(branch) - {"node_id"}
            )
        ):
            raise ValueError("NRIR-25 objective branch linkage differs")
        branch_ids.add(node_id)
    decision_by_id: dict[str, Mapping[str, Any]] = {}
    expanded_children: set[str] = set()
    terminal_ids: list[str] = []
    for index, value in enumerate(decisions):
        decision = _mapping(value, "NRIR-25 queue decision")
        node_id = str(decision.get("node_id"))
        kind = decision.get("kind")
        if (
            decision.get("decision_index") != index
            or node_id in decision_by_id
            or node_id not in evaluation_by_id
            or kind not in {"expand", "prune", "terminal"}
        ):
            raise ValueError("NRIR-25 queue decision identity differs")
        decision_by_id[node_id] = decision
        if kind == "expand":
            children = _list(decision.get("child_node_ids"), "NRIR-25 children")
            if len(children) != 2 or any(
                _mapping(evaluation_by_id[str(child)]["node"], "NRIR-25 child").get(
                    "parent_node_id"
                )
                != node_id
                for child in children
            ):
                raise ValueError("NRIR-25 expansion lineage differs")
            expanded_children.update(str(child) for child in children)
        else:
            terminal_ids.append(node_id)
    root_id = f"{run_id}:n000000"
    if expanded_children != set(evaluation_by_id) - {root_id}:
        raise ValueError("NRIR-25 expanded child coverage differs")
    terminal_lowers = [lowers[node_id] for node_id in terminal_ids]
    all_proved = all(
        decision_by_id[node_id].get("kind") == "prune" and lowers[node_id] >= 0.0
        for node_id in terminal_ids
    )
    expected_summary = {
        "root_lower": lowers[root_id],
        "terminal_node_ids": terminal_ids,
        "terminal_lowers": terminal_lowers,
        "worst_terminal_lower": min(terminal_lowers),
        "best_terminal_lower": max(terminal_lowers),
        "proof_deficit": max(0.0, -min(terminal_lowers)),
        "evaluated_nodes": len(evaluations),
        "terminal_domains": len(terminal_ids),
        "max_evaluated_depth": max(
            int(_mapping(value["node"], "NRIR-25 node")["depth"])
            for value in evaluation_by_id.values()
        ),
        "bounded_tree_status": "verified" if all_proved else "unknown",
    }
    if summary != expected_summary:
        raise ValueError("NRIR-25 summary differs")
    selected_targets = sum(
        int(record["selected_target_count"]) for record in refinements
    )
    return planned_cap, selected_targets, rank_counts


def validate_shard(
    shard: Mapping[str, Any], *, expected_clause: int, expected_mode: str
) -> None:
    body = dict(shard)
    semantic_hash = body.pop("semantic_hash", None)
    expected_dynamic = (
        DYNAMIC_BUDGET_POLICY.to_dict() if expected_mode == "dynamic8_24" else None
    )
    if (
        shard.get("schema_version") != SHARD_SCHEMA_VERSION
        or shard.get("clause_index") != expected_clause
        or shard.get("mode") != expected_mode
        or shard.get("performance_claimed") is not False
        or semantic_hash != canonical_hash(body)
        or shard.get("source") != FROZEN_SOURCE
        or shard.get("protocol")
        != {
            "queue_config": QUEUE_CONFIG.to_dict(),
            "optimizer_policy": OPTIMIZER_POLICY.to_dict(),
            "branch_policy": BRANCH_POLICY.to_dict(),
            "base_refinement_policy": REFINEMENT_POLICY.to_dict(),
            "dynamic_budget_policy": expected_dynamic,
            "strategy": "external_seeded_ancestral_carry_v1",
        }
    ):
        raise ValueError("NRIR-25 shard header/protocol differs")
    seed = _mapping(shard.get("external_seed_ir"), "NRIR-25 seed")
    if (
        seed.get("semantics_owner") != "external_verifier"
        or seed.get("consumption") != "sound_constraint_intersection_only"
        or shard.get("external_seed_hash") != canonical_hash(seed)
    ):
        raise ValueError("NRIR-25 external seed binding differs")
    _validate_queue(
        _mapping(shard.get("queue"), "NRIR-25 queue"),
        _mapping(shard.get("summary"), "NRIR-25 summary"),
        clause=expected_clause,
        mode=expected_mode,
    )


def _mode_metrics(shard: Mapping[str, Any]) -> dict[str, object]:
    queue = _mapping(shard["queue"], "NRIR-25 queue")
    summary = _mapping(shard["summary"], "NRIR-25 summary")
    planned, selected, ranks = _validate_queue(
        queue,
        summary,
        clause=int(shard["clause_index"]),
        mode=str(shard["mode"]),
    )
    return {
        "worst_terminal_lower": summary["worst_terminal_lower"],
        "best_terminal_lower": summary["best_terminal_lower"],
        "proof_deficit": summary["proof_deficit"],
        "bounded_tree_status": summary["bounded_tree_status"],
        "evaluated_nodes": summary["evaluated_nodes"],
        "planned_target_cap_per_relu": planned,
        "actual_selected_target_count": selected,
        "allocation_rank_counts": ranks,
    }


def _comparison(
    fixed: Mapping[str, Any], dynamic: Mapping[str, Any], clause: int
) -> dict[str, object]:
    fixed_metrics = _mode_metrics(fixed)
    dynamic_metrics = _mode_metrics(dynamic)
    fixed_splits = {
        str(
            _mapping(_mapping(value, "NRIR-25 evaluation")["node"], "NRIR-25 node")[
                "split_state_hash"
            ]
        )
        for value in _list(
            _mapping(fixed["queue"], "NRIR-25 fixed queue")["evaluations"],
            "NRIR-25 fixed evaluations",
        )
    }
    dynamic_splits = {
        str(
            _mapping(_mapping(value, "NRIR-25 evaluation")["node"], "NRIR-25 node")[
                "split_state_hash"
            ]
        )
        for value in _list(
            _mapping(dynamic["queue"], "NRIR-25 dynamic queue")["evaluations"],
            "NRIR-25 dynamic evaluations",
        )
    }
    dynamic_lower = dynamic_metrics["worst_terminal_lower"]
    fixed_lower = fixed_metrics["worst_terminal_lower"]
    if not isinstance(dynamic_lower, (int, float)) or not isinstance(
        fixed_lower, (int, float)
    ):
        raise TypeError("NRIR-25 worst terminal lower must be numeric")
    delta = float(dynamic_lower) - float(fixed_lower)
    return {
        "clause_index": clause,
        "fixed16": fixed_metrics,
        "dynamic8_24": dynamic_metrics,
        "dynamic_worst_lower_delta": delta,
        "dynamic_not_weaker": delta >= -COMPARISON_TOLERANCE,
        "dynamic_strictly_better": delta > COMPARISON_TOLERANCE,
        "logical_domain_overlap": len(fixed_splits & dynamic_splits),
        "logical_domain_union": len(fixed_splits | dynamic_splits),
        "same_planned_target_cap": (
            fixed_metrics["planned_target_cap_per_relu"]
            == dynamic_metrics["planned_target_cap_per_relu"]
            == MAX_NODES * REFINEMENT_POLICY.max_neurons_per_relu
        ),
        "performance_claimed": False,
    }


def _assemble_evidence(shards: Mapping[str, Mapping[str, Any]]) -> dict[str, object]:
    expected = {_shard_name(clause, mode) for clause in HARD_CLAUSES for mode in MODES}
    if set(shards) != expected:
        raise ValueError("NRIR-25 shard matrix differs")
    comparisons: list[dict[str, object]] = []
    for clause in HARD_CLAUSES:
        for mode in MODES:
            validate_shard(
                shards[_shard_name(clause, mode)],
                expected_clause=clause,
                expected_mode=mode,
            )
        comparisons.append(
            _comparison(
                shards[_shard_name(clause, "fixed16")],
                shards[_shard_name(clause, "dynamic8_24")],
                clause,
            )
        )
    not_weaker = all(bool(item["dynamic_not_weaker"]) for item in comparisons)
    strict = any(bool(item["dynamic_strictly_better"]) for item in comparisons)
    conserved = all(bool(item["same_planned_target_cap"]) for item in comparisons)
    status = (
        "validated_reduced"
        if not_weaker and strict and conserved
        else "validated_no_go"
    )
    evidence: dict[str, object] = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": status,
        "performance_claimed": False,
        "claim_boundary": (
            "fixed ResNet property 0 clauses 0/2/4, CPU, 31-node depth-four "
            "fixed16 versus conserved parent-lower dynamic8_24 target caps"
        ),
        "source": deepcopy(FROZEN_SOURCE),
        "protocol": {
            "hard_clause_indices": list(HARD_CLAUSES),
            "modes": list(MODES),
            "queue_config": QUEUE_CONFIG.to_dict(),
            "optimizer_policy": OPTIMIZER_POLICY.to_dict(),
            "branch_policy": BRANCH_POLICY.to_dict(),
            "base_refinement_policy": REFINEMENT_POLICY.to_dict(),
            "dynamic_budget_policy": DYNAMIC_BUDGET_POLICY.to_dict(),
            "strategy": "external_seeded_ancestral_carry_v1",
            "execution_isolation": "fresh_python_process_per_clause_mode",
        },
        "shard_semantic_hashes": {
            name: shard["semantic_hash"] for name, shard in sorted(shards.items())
        },
        "comparisons": comparisons,
        "gates": {
            "all_six_units_present": True,
            "all_tree_planned_caps_conserved": conserved,
            "dynamic_not_weaker_on_all_hard_clauses": not_weaker,
            "dynamic_strictly_improves_at_least_one_hard_clause": strict,
            "any_fixed_clause_bounded_tree_closed": any(
                _mapping(item[mode], "NRIR-25 mode comparison")["bounded_tree_status"]
                == "verified"
                for item in comparisons
                for mode in MODES
            ),
        },
        "limitations": [
            "fixed ResNet property 0 clauses 0, 2, and 4 on CPU only",
            "external seed generation remains owned by alpha-beta-CROWN",
            "equal planned target caps are not equal latency or actual selected targets",
            "bounded trees do not imply complete property or verifier closure",
            "GPU, multi-workload, cuts, multi-pass, and performance remain pending",
        ],
    }
    return evidence


def validate_evidence(
    evidence: Mapping[str, Any], shards: Mapping[str, Mapping[str, Any]]
) -> None:
    if evidence != _assemble_evidence(shards):
        raise ValueError("NRIR-25 aggregate evidence differs")


def _worker(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.torch_threads)
    context = _build_context(
        model=args.model,
        source_artifact_dir=args.source_artifact_dir,
        local_artifact_dir=args.local_artifact_dir,
    )
    shard = _build_shard(context, model=args.model, clause=args.clause, mode=args.mode)
    _write_json_atomic(args.output, shard)
    print(
        _canonical_json(
            {
                "status": "ok",
                "shard": args.output.name,
                "semantic_hash": shard["semantic_hash"],
            }
        )
    )


def _spawn_worker(
    args: argparse.Namespace, *, clause: int, mode: str, output: Path
) -> float:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--model",
        str(args.model),
        "--source-artifact-dir",
        str(args.source_artifact_dir),
        "--local-artifact-dir",
        str(args.local_artifact_dir),
        "--torch-threads",
        str(args.torch_threads),
        "--clause",
        str(clause),
        "--mode",
        mode,
        "--output",
        str(output),
    ]
    started = time.monotonic()
    subprocess.run(command, check=True)
    return time.monotonic() - started


def _load_shards(artifact_dir: Path) -> dict[str, Mapping[str, Any]]:
    return {
        _shard_name(clause, mode): _load_json(
            artifact_dir / SHARD_DIR / _shard_name(clause, mode)
        )
        for clause in HARD_CLAUSES
        for mode in MODES
    }


def shard_is_reusable(path: Path, *, clause: int, mode: str) -> bool:
    try:
        shard = _load_json(path)
        validate_shard(shard, expected_clause=clause, expected_mode=mode)
    except (OSError, TypeError, ValueError):
        return False
    return True


def _write_artifact(
    artifact_dir: Path, shards: Mapping[str, Mapping[str, Any]]
) -> None:
    evidence = _assemble_evidence(shards)
    validate_evidence(evidence, shards)
    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": evidence["status"],
        "performance_claimed": False,
        "evidence": evidence,
    }
    artifact_path = artifact_dir / ARTIFACT_FILE
    _write_json_atomic(artifact_path, artifact)
    files = {ARTIFACT_FILE: file_sha256(artifact_path)}
    files.update(
        {
            f"{SHARD_DIR}/{name}": file_sha256(artifact_dir / SHARD_DIR / name)
            for name in sorted(shards)
        }
    )
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": evidence["status"],
        "performance_claimed": False,
        "files": files,
        "evidence_hash": canonical_hash(evidence),
    }
    _write_json_atomic(artifact_dir / MANIFEST_FILE, manifest)
    print(
        _canonical_json(
            {"status": evidence["status"], "evidence_hash": manifest["evidence_hash"]}
        )
    )


def _generate(args: argparse.Namespace) -> None:
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    for clause in HARD_CLAUSES:
        for mode in MODES:
            path = args.artifact_dir / SHARD_DIR / _shard_name(clause, mode)
            if not args.force and shard_is_reusable(path, clause=clause, mode=mode):
                print(_canonical_json({"status": "resume", "shard": path.name}))
                continue
            elapsed = _spawn_worker(args, clause=clause, mode=mode, output=path)
            print(
                _canonical_json(
                    {
                        "status": "generated",
                        "shard": path.name,
                        "elapsed_seconds_diagnostic": round(elapsed, 3),
                    }
                )
            )
    _write_artifact(args.artifact_dir, _load_shards(args.artifact_dir))


def _validate_artifact(
    artifact_dir: Path,
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    manifest = _load_json(artifact_dir / MANIFEST_FILE)
    artifact_path = artifact_dir / ARTIFACT_FILE
    artifact = _load_json(artifact_path)
    shards = _load_shards(artifact_dir)
    evidence = _mapping(artifact.get("evidence"), "NRIR-25 evidence")
    validate_evidence(evidence, shards)
    files = {ARTIFACT_FILE: file_sha256(artifact_path)}
    files.update(
        {
            f"{SHARD_DIR}/{name}": file_sha256(artifact_dir / SHARD_DIR / name)
            for name in sorted(shards)
        }
    )
    if (
        manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION
        or artifact.get("schema_version") != ARTIFACT_SCHEMA_VERSION
        or manifest.get("status") != evidence["status"]
        or artifact.get("status") != evidence["status"]
        or manifest.get("performance_claimed") is not False
        or artifact.get("performance_claimed") is not False
        or manifest.get("files") != files
        or manifest.get("evidence_hash") != canonical_hash(evidence)
    ):
        raise ValueError("NRIR-25 artifact manifest/header differs")
    return manifest, shards


def _replay(args: argparse.Namespace) -> None:
    manifest, stored = _validate_artifact(args.artifact_dir)
    with tempfile.TemporaryDirectory(prefix="boundflow-nrir25-replay-") as temp:
        temporary = Path(temp)
        for clause in HARD_CLAUSES:
            for mode in MODES:
                name = _shard_name(clause, mode)
                output = temporary / name
                _spawn_worker(args, clause=clause, mode=mode, output=output)
                if _load_json(output) != stored[name]:
                    raise ValueError(f"NRIR-25 semantic replay differs: {name}")
    print(_canonical_json({"status": "ok", "evidence_hash": manifest["evidence_hash"]}))


def main() -> None:
    args = _parse_args()
    if args.torch_threads < 1:
        raise ValueError("NRIR-25 torch thread count must be positive")
    if args.command == "worker":
        _worker(args)
    elif args.command == "generate":
        _generate(args)
    else:
        _replay(args)


if __name__ == "__main__":
    main()
