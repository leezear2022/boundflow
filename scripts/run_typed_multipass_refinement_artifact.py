#!/usr/bin/env python3
"""Generate or replay the NRIR-26 typed multi-pass refinement artifact."""

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
    NativeIntermediateRefinementMultiPassPolicyIR,
    NativeIntermediateRefinementPassDecisionIR,
    NativeIntermediateRefinementPolicyIR,
)
from boundflow.runtime.abcrown_adapter import file_sha256
from boundflow.runtime.native_optimized_relu_split_bab_runtime import (
    NativeOptimizedReluSplitBabExecution,
    execute_native_optimized_relu_split_bab,
)
from scripts.run_dynamic_ancestral_refinement_budget_artifact import (
    DYNAMIC_BUDGET_POLICY,
    _validate_budget_records,
)
from scripts.run_end_to_end_tightness_performance_baseline import _build_context
from scripts.run_external_seeded_ancestral_refinement_artifact import (
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

ARTIFACT_SCHEMA_VERSION = "boundflow.typed-multipass-refinement-artifact/v1"
EVIDENCE_SCHEMA_VERSION = "boundflow.typed-multipass-refinement-evidence/v1"
SHARD_SCHEMA_VERSION = "boundflow.typed-multipass-refinement-shard/v1"
ARTIFACT_FILE = "typed_multipass_refinement.json"
MANIFEST_FILE = "manifest.json"
SHARD_DIR = "shards"
MODES = ("single_pass_dynamic8_24", "split_two_pass_dynamic8_24")
MAX_NODES = 31
MAX_DEPTH = 4
COMPARISON_TOLERANCE = 1e-6
QUEUE_CONFIG = _queue_config(MAX_NODES, MAX_DEPTH)
SINGLE_PASS_POLICY = NativeIntermediateRefinementPolicyIR(
    passes=1,
    max_neurons_per_relu=16,
    backward_chunk_size=4,
    candidate_policy_id="objective_influence_width_per_relu_v1",
)
TWO_PASS_POLICY = NativeIntermediateRefinementPolicyIR(
    passes=2,
    max_neurons_per_relu=16,
    backward_chunk_size=4,
    candidate_policy_id="objective_influence_width_per_relu_v1",
)
MULTI_PASS_POLICY = NativeIntermediateRefinementMultiPassPolicyIR()


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
    return f"nrir26:c{clause}:{mode}"


def _mode_policy(
    mode: str,
) -> tuple[
    NativeIntermediateRefinementPolicyIR,
    NativeIntermediateRefinementMultiPassPolicyIR | None,
]:
    if mode == "single_pass_dynamic8_24":
        return SINGLE_PASS_POLICY, None
    if mode == "split_two_pass_dynamic8_24":
        return TWO_PASS_POLICY, MULTI_PASS_POLICY
    raise ValueError("NRIR-26 mode is outside the frozen matrix")


def _program_projections(
    execution: NativeOptimizedReluSplitBabExecution,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for node_id, refinement in execution.per_child_refinement_executions:
        program = refinement.program
        decisions = [
            trace.selection_decision
            for trace in refinement.trace.pass_traces
            if trace.selection_decision is not None
        ]
        rows.append(
            {
                "node_id": node_id,
                "policy": program.plan.policy.to_dict(),
                "multi_pass_policy": (
                    None
                    if program.plan.multi_pass_policy is None
                    else program.plan.multi_pass_policy.to_dict()
                ),
                "initial_selected_target_count": len(program.plan.targets),
                "actual_selected_target_count": (
                    len(program.plan.targets)
                    if not decisions
                    else sum(item.selected_target_count for item in decisions)
                ),
                "pass_decisions": [item.to_dict() for item in decisions],
                "pass_decision_hashes": [item.stable_hash() for item in decisions],
                **program.hashes(),
            }
        )
    return rows


def _build_shard(
    context: Mapping[str, Any], *, model: Path, clause: int, mode: str
) -> dict[str, object]:
    if clause not in HARD_CLAUSES or mode not in MODES:
        raise ValueError("NRIR-26 worker identity is outside the frozen matrix")
    tensors = _mapping(context["tensors"], "NRIR-26 tensors")
    objective = tensors["linear_spec_c"][:, clause : clause + 1]
    seed = _external_seed(context, clause)
    refinement_policy, multi_pass_policy = _mode_policy(mode)
    execution = execute_native_optimized_relu_split_bab(
        context["module"],
        context["input_spec"],
        linear_spec_C=objective,
        run_id=_run_id(clause, mode),
        config=QUEUE_CONFIG,
        optimizer_policy=OPTIMIZER_POLICY,
        intermediate_bound_source=IntermediateBoundSource.NATIVE_REFINED,
        objective_branch_policy=BRANCH_POLICY,
        per_child_refinement_policy=refinement_policy,
        per_child_refinement_budget_policy=DYNAMIC_BUDGET_POLICY,
        per_child_refinement_multi_pass_policy=multi_pass_policy,
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
            "refinement_policy": refinement_policy.to_dict(),
            "dynamic_budget_policy": DYNAMIC_BUDGET_POLICY.to_dict(),
            "multi_pass_policy": (
                None if multi_pass_policy is None else multi_pass_policy.to_dict()
            ),
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


def _pass_decision_from_dict(
    value: Mapping[str, Any],
) -> NativeIntermediateRefinementPassDecisionIR:
    decision = NativeIntermediateRefinementPassDecisionIR(
        plan_hash=str(value.get("plan_hash", "")),
        multi_pass_policy_hash=str(value.get("multi_pass_policy_hash", "")),
        pass_index=int(value.get("pass_index", -1)),
        total_target_cap_per_relu=int(value.get("total_target_cap_per_relu", 0)),
        pass_target_cap_per_relu=int(value.get("pass_target_cap_per_relu", 0)),
        input_bounds_hash=str(value.get("input_bounds_hash", "")),
        prior_target_ledger_hash=str(value.get("prior_target_ledger_hash", "")),
        selected_targets_hash=str(value.get("selected_targets_hash", "")),
        result_target_ledger_hash=str(value.get("result_target_ledger_hash", "")),
        prior_selected_target_count=int(value.get("prior_selected_target_count", -1)),
        selected_target_count=int(value.get("selected_target_count", -1)),
        cumulative_selected_target_count=int(
            value.get("cumulative_selected_target_count", -1)
        ),
        continuation=bool(value.get("continuation", False)),
        termination_reason=str(value.get("termination_reason", "")),
        semantics_owner=str(value.get("semantics_owner", "")),
    )
    decision.validate()
    if decision.to_dict() != value:
        raise ValueError("NRIR-26 pass decision payload differs")
    return decision


def _validate_multi_pass_record(
    record: Mapping[str, Any], program: Mapping[str, Any], *, mode: str
) -> tuple[int, int]:
    assigned = int(
        _mapping(record.get("budget_decision"), "NRIR-26 budget decision")[
            "assigned_max_neurons_per_relu"
        ]
    )
    expected_policy, expected_multi = _mode_policy(mode)
    expected_program_policy = {
        **expected_policy.to_dict(),
        "max_neurons_per_relu": assigned,
    }
    if (
        program.get("policy") != expected_program_policy
        or program.get("refinement_plan_hash") != record.get("refinement_plan_hash")
        or program.get("refinement_task_module_hash")
        != record.get("refinement_task_module_hash")
        or program.get("refinement_schedule_hash")
        != record.get("refinement_schedule_hash")
        or program.get("initial_selected_target_count")
        != record.get("selected_target_count")
    ):
        raise ValueError("NRIR-26 refinement program linkage differs")
    if expected_multi is None:
        forbidden = {
            "multi_pass_policy",
            "multi_pass_decisions",
            "multi_pass_decision_hashes",
        }
        if (
            any(key in record for key in forbidden)
            or program.get("multi_pass_policy") is not None
            or program.get("pass_decisions") != []
            or program.get("pass_decision_hashes") != []
            or program.get("actual_selected_target_count")
            != record.get("selected_target_count")
        ):
            raise ValueError("NRIR-26 single-pass payload declares multi-pass state")
        return int(record["selected_target_count"]), 0
    if (
        record.get("multi_pass_policy") != expected_multi.to_dict()
        or program.get("multi_pass_policy") != expected_multi.to_dict()
    ):
        raise ValueError("NRIR-26 multi-pass policy differs")
    values = _list(record.get("multi_pass_decisions"), "NRIR-26 pass decisions")
    hashes = _list(
        record.get("multi_pass_decision_hashes"), "NRIR-26 pass decision hashes"
    )
    decisions = [
        _pass_decision_from_dict(_mapping(value, "NRIR-26 pass decision"))
        for value in values
    ]
    if (
        len(decisions) != expected_multi.maximum_passes
        or hashes != [item.stable_hash() for item in decisions]
        or program.get("pass_decisions") != values
        or program.get("pass_decision_hashes") != hashes
    ):
        raise ValueError("NRIR-26 pass decision linkage differs")
    previous_hash = canonical_hash([])
    previous_count = 0
    for index, decision in enumerate(decisions):
        if (
            decision.plan_hash != record.get("refinement_plan_hash")
            or decision.multi_pass_policy_hash != expected_multi.stable_hash()
            or decision.pass_index != index
            or decision.total_target_cap_per_relu != assigned
            or decision.pass_target_cap_per_relu != assigned // 2
            or decision.prior_target_ledger_hash != previous_hash
            or decision.prior_selected_target_count != previous_count
        ):
            raise ValueError("NRIR-26 pass target ledger differs")
        previous_hash = decision.result_target_ledger_hash
        previous_count = decision.cumulative_selected_target_count
    if program.get("actual_selected_target_count") != previous_count:
        raise ValueError("NRIR-26 actual target total differs")
    return previous_count, sum(not item.continuation for item in decisions)


def _validate_queue(
    queue: Mapping[str, Any], summary: Mapping[str, Any], *, clause: int, mode: str
) -> tuple[int, int, int, dict[str, int]]:
    evaluations = _list(queue.get("evaluations"), "NRIR-26 evaluations")
    decisions = _list(queue.get("decisions"), "NRIR-26 queue decisions")
    refinements = _list(queue.get("refinements"), "NRIR-26 refinements")
    programs = _list(queue.get("refinement_programs"), "NRIR-26 programs")
    branches = _list(queue.get("objective_branch_hashes"), "NRIR-26 branches")
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
        or any(
            len(values) != MAX_NODES
            for values in (evaluations, decisions, refinements, programs, branches)
        )
    ):
        raise ValueError("NRIR-26 queue header/coverage differs")
    evaluation_by_id: dict[str, Mapping[str, Any]] = {}
    lowers: dict[str, float] = {}
    for index, value in enumerate(evaluations):
        evaluation = _mapping(value, "NRIR-26 evaluation")
        node = _mapping(evaluation.get("node"), "NRIR-26 node")
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
            raise ValueError("NRIR-26 evaluation identity differs")
        parent_id = node.get("parent_node_id")
        if index == 0:
            if parent_id is not None or node.get("depth") != 0:
                raise ValueError("NRIR-26 root lineage differs")
        else:
            parent = evaluation_by_id.get(str(parent_id))
            if (
                parent is None
                or int(node["depth"])
                != int(_mapping(parent["node"], "NRIR-26 parent")["depth"]) + 1
            ):
                raise ValueError("NRIR-26 parent lineage differs")
        evaluation_by_id[str(node_id)] = evaluation
        lowers[str(node_id)] = float(lower)
    refinement_by_id: dict[str, Mapping[str, Any]] = {}
    for value in refinements:
        record = _mapping(value, "NRIR-26 refinement")
        node_id = str(record.get("node_id"))
        if (
            node_id in refinement_by_id
            or node_id not in evaluation_by_id
            or canonical_hash(record)
            != evaluation_by_id[node_id].get("intermediate_refinement_trace_hash")
        ):
            raise ValueError("NRIR-26 refinement binding differs")
        parent_id = _mapping(evaluation_by_id[node_id]["node"], "NRIR-26 node").get(
            "parent_node_id"
        )
        if parent_id is None:
            if not _is_sha256(record.get("external_constraint_seed_hash")):
                raise ValueError("NRIR-26 root seed binding differs")
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
                raise ValueError("NRIR-26 ancestral lineage differs")
        refinement_by_id[node_id] = record
    program_by_id: dict[str, Mapping[str, Any]] = {}
    actual_targets = 0
    stopped_passes = 0
    for value in programs:
        program = _mapping(value, "NRIR-26 refinement program")
        node_id = str(program.get("node_id"))
        refinement_record = refinement_by_id.get(node_id)
        if refinement_record is None or node_id in program_by_id:
            raise ValueError("NRIR-26 refinement program coverage differs")
        selected, stopped = _validate_multi_pass_record(
            refinement_record, program, mode=mode
        )
        actual_targets += selected
        stopped_passes += stopped
        program_by_id[node_id] = program
    planned_cap, ranks = _validate_budget_records(refinements, mode="dynamic8_24")
    branch_ids = {
        str(_mapping(value, "NRIR-26 objective branch").get("node_id"))
        for value in branches
    }
    if branch_ids != set(evaluation_by_id):
        raise ValueError("NRIR-26 objective branch coverage differs")
    decision_by_id: dict[str, Mapping[str, Any]] = {}
    expanded_children: set[str] = set()
    terminal_ids: list[str] = []
    for index, value in enumerate(decisions):
        decision = _mapping(value, "NRIR-26 queue decision")
        node_id = str(decision.get("node_id"))
        kind = decision.get("kind")
        if (
            decision.get("decision_index") != index
            or node_id in decision_by_id
            or node_id not in evaluation_by_id
            or kind not in {"expand", "prune", "terminal"}
        ):
            raise ValueError("NRIR-26 queue decision identity differs")
        decision_by_id[node_id] = decision
        if kind == "expand":
            children = _list(decision.get("child_node_ids"), "NRIR-26 children")
            if len(children) != 2 or any(
                _mapping(evaluation_by_id[str(child)]["node"], "NRIR-26 child").get(
                    "parent_node_id"
                )
                != node_id
                for child in children
            ):
                raise ValueError("NRIR-26 expansion lineage differs")
            expanded_children.update(str(child) for child in children)
        else:
            terminal_ids.append(node_id)
    root_id = f"{run_id}:n000000"
    if expanded_children != set(evaluation_by_id) - {root_id}:
        raise ValueError("NRIR-26 expanded child coverage differs")
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
            int(_mapping(value["node"], "NRIR-26 node")["depth"])
            for value in evaluation_by_id.values()
        ),
        "bounded_tree_status": "verified" if all_proved else "unknown",
    }
    if summary != expected_summary:
        raise ValueError("NRIR-26 summary differs")
    return planned_cap, actual_targets, stopped_passes, ranks


def validate_shard(
    shard: Mapping[str, Any], *, expected_clause: int, expected_mode: str
) -> None:
    body = dict(shard)
    semantic_hash = body.pop("semantic_hash", None)
    refinement_policy, multi_pass_policy = _mode_policy(expected_mode)
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
            "refinement_policy": refinement_policy.to_dict(),
            "dynamic_budget_policy": DYNAMIC_BUDGET_POLICY.to_dict(),
            "multi_pass_policy": (
                None if multi_pass_policy is None else multi_pass_policy.to_dict()
            ),
            "strategy": "external_seeded_ancestral_carry_v1",
        }
    ):
        raise ValueError("NRIR-26 shard header/protocol differs")
    seed = _mapping(shard.get("external_seed_ir"), "NRIR-26 seed")
    if (
        seed.get("semantics_owner") != "external_verifier"
        or seed.get("consumption") != "sound_constraint_intersection_only"
        or shard.get("external_seed_hash") != canonical_hash(seed)
    ):
        raise ValueError("NRIR-26 external seed binding differs")
    _validate_queue(
        _mapping(shard.get("queue"), "NRIR-26 queue"),
        _mapping(shard.get("summary"), "NRIR-26 summary"),
        clause=expected_clause,
        mode=expected_mode,
    )


def _mode_metrics(shard: Mapping[str, Any]) -> dict[str, object]:
    summary = _mapping(shard["summary"], "NRIR-26 summary")
    planned, actual, stopped, ranks = _validate_queue(
        _mapping(shard["queue"], "NRIR-26 queue"),
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
        "planned_total_target_cap_per_relu": planned,
        "actual_selected_target_count": actual,
        "stopped_pass_count": stopped,
        "allocation_rank_counts": ranks,
    }


def _comparison(
    single: Mapping[str, Any], split: Mapping[str, Any], clause: int
) -> dict[str, object]:
    single_metrics = _mode_metrics(single)
    split_metrics = _mode_metrics(split)
    single_splits = {
        str(
            _mapping(_mapping(value, "NRIR-26 evaluation")["node"], "NRIR-26 node")[
                "split_state_hash"
            ]
        )
        for value in _list(
            _mapping(single["queue"], "NRIR-26 queue")["evaluations"],
            "NRIR-26 evaluations",
        )
    }
    split_splits = {
        str(
            _mapping(_mapping(value, "NRIR-26 evaluation")["node"], "NRIR-26 node")[
                "split_state_hash"
            ]
        )
        for value in _list(
            _mapping(split["queue"], "NRIR-26 queue")["evaluations"],
            "NRIR-26 evaluations",
        )
    }
    single_lower = single_metrics["worst_terminal_lower"]
    split_lower = split_metrics["worst_terminal_lower"]
    if not isinstance(single_lower, (int, float)) or not isinstance(
        split_lower, (int, float)
    ):
        raise TypeError("NRIR-26 worst terminal lower must be numeric")
    delta = float(split_lower) - float(single_lower)
    return {
        "clause_index": clause,
        "single_pass_dynamic8_24": single_metrics,
        "split_two_pass_dynamic8_24": split_metrics,
        "split_worst_lower_delta": delta,
        "split_not_weaker": delta >= -COMPARISON_TOLERANCE,
        "split_strictly_better": delta > COMPARISON_TOLERANCE,
        "logical_domain_overlap": len(single_splits & split_splits),
        "logical_domain_union": len(single_splits | split_splits),
        "same_planned_total_target_cap": (
            single_metrics["planned_total_target_cap_per_relu"]
            == split_metrics["planned_total_target_cap_per_relu"]
            == MAX_NODES * SINGLE_PASS_POLICY.max_neurons_per_relu
        ),
        "performance_claimed": False,
    }


def _assemble_evidence(shards: Mapping[str, Mapping[str, Any]]) -> dict[str, object]:
    expected = {_shard_name(clause, mode) for clause in HARD_CLAUSES for mode in MODES}
    if set(shards) != expected:
        raise ValueError("NRIR-26 shard matrix differs")
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
                shards[_shard_name(clause, MODES[0])],
                shards[_shard_name(clause, MODES[1])],
                clause,
            )
        )
    not_weaker = all(bool(item["split_not_weaker"]) for item in comparisons)
    strict = any(bool(item["split_strictly_better"]) for item in comparisons)
    conserved = all(bool(item["same_planned_total_target_cap"]) for item in comparisons)
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
            "single-pass versus disjoint split-two-pass at equal dynamic total caps"
        ),
        "source": deepcopy(FROZEN_SOURCE),
        "protocol": {
            "hard_clause_indices": list(HARD_CLAUSES),
            "modes": list(MODES),
            "queue_config": QUEUE_CONFIG.to_dict(),
            "optimizer_policy": OPTIMIZER_POLICY.to_dict(),
            "branch_policy": BRANCH_POLICY.to_dict(),
            "single_pass_policy": SINGLE_PASS_POLICY.to_dict(),
            "two_pass_policy": TWO_PASS_POLICY.to_dict(),
            "dynamic_budget_policy": DYNAMIC_BUDGET_POLICY.to_dict(),
            "multi_pass_policy": MULTI_PASS_POLICY.to_dict(),
            "strategy": "external_seeded_ancestral_carry_v1",
            "execution_isolation": "fresh_python_process_per_clause_mode",
        },
        "shard_semantic_hashes": {
            name: shard["semantic_hash"] for name, shard in sorted(shards.items())
        },
        "comparisons": comparisons,
        "gates": {
            "all_six_units_present": True,
            "all_tree_total_caps_conserved": conserved,
            "split_not_weaker_on_all_hard_clauses": not_weaker,
            "split_strictly_improves_at_least_one_hard_clause": strict,
            "any_bounded_tree_closed": any(
                _mapping(item[mode], "NRIR-26 mode comparison")["bounded_tree_status"]
                == "verified"
                for item in comparisons
                for mode in MODES
            ),
        },
        "limitations": [
            "fixed ResNet property 0 clauses 0, 2, and 4 on CPU only",
            "objective influence is node-initial while pass width is updated",
            "equal planned target caps are not equal latency",
            "bounded trees do not imply complete property or verifier closure",
            "GPU, multi-workload, cuts, performance, and adaptive pass counts remain pending",
        ],
    }
    return evidence


def validate_evidence(
    evidence: Mapping[str, Any], shards: Mapping[str, Mapping[str, Any]]
) -> None:
    if evidence != _assemble_evidence(shards):
        raise ValueError("NRIR-26 aggregate evidence differs")


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
    started = time.monotonic()
    subprocess.run(
        [
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
        ],
        check=True,
    )
    return time.monotonic() - started


def _load_shards(artifact_dir: Path) -> dict[str, Mapping[str, Any]]:
    shards: dict[str, Mapping[str, Any]] = {}
    for clause in HARD_CLAUSES:
        for mode in MODES:
            name = _shard_name(clause, mode)
            shards[name] = _mapping(
                _load_json(artifact_dir / SHARD_DIR / name), "NRIR-26 shard"
            )
    return shards


def shard_is_reusable(path: Path, *, clause: int, mode: str) -> bool:
    try:
        shard = _mapping(_load_json(path), "NRIR-26 checkpoint shard")
        validate_shard(shard, expected_clause=clause, expected_mode=mode)
    except (OSError, TypeError, ValueError):
        return False
    return True


def _write_artifact(artifact_dir: Path, shards: Mapping[str, Mapping[str, Any]]) -> str:
    evidence = _assemble_evidence(shards)
    validate_evidence(evidence, shards)
    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": evidence["status"],
        "performance_claimed": False,
        "evidence": evidence,
    }
    _write_json_atomic(artifact_dir / ARTIFACT_FILE, artifact)
    files = {
        ARTIFACT_FILE: file_sha256(artifact_dir / ARTIFACT_FILE),
        **{
            f"{SHARD_DIR}/{name}": file_sha256(artifact_dir / SHARD_DIR / name)
            for name in sorted(shards)
        },
    }
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": evidence["status"],
        "performance_claimed": False,
        "evidence_hash": canonical_hash(evidence),
        "files": files,
    }
    _write_json_atomic(artifact_dir / MANIFEST_FILE, manifest)
    return str(manifest["evidence_hash"])


def _generate(args: argparse.Namespace) -> None:
    shard_dir = args.artifact_dir / SHARD_DIR
    shard_dir.mkdir(parents=True, exist_ok=True)
    diagnostics: list[dict[str, object]] = []
    for clause in HARD_CLAUSES:
        for mode in MODES:
            name = _shard_name(clause, mode)
            output = shard_dir / name
            if not args.force and shard_is_reusable(output, clause=clause, mode=mode):
                diagnostics.append({"shard": name, "status": "reused"})
                continue
            elapsed = _spawn_worker(args, clause=clause, mode=mode, output=output)
            diagnostics.append(
                {
                    "shard": name,
                    "status": "generated",
                    "elapsed_seconds_diagnostic": round(elapsed, 3),
                }
            )
    shards = _load_shards(args.artifact_dir)
    evidence_hash = _write_artifact(args.artifact_dir, shards)
    for item in diagnostics:
        print(_canonical_json(item))
    artifact = _mapping(
        _load_json(args.artifact_dir / ARTIFACT_FILE), "NRIR-26 artifact"
    )
    print(
        _canonical_json({"status": artifact["status"], "evidence_hash": evidence_hash})
    )


def _validate_artifact(
    artifact_dir: Path,
) -> tuple[Mapping[str, Any], dict[str, Mapping[str, Any]]]:
    manifest = _mapping(_load_json(artifact_dir / MANIFEST_FILE), "NRIR-26 manifest")
    artifact = _mapping(_load_json(artifact_dir / ARTIFACT_FILE), "NRIR-26 artifact")
    shards = _load_shards(artifact_dir)
    evidence = _mapping(artifact.get("evidence"), "NRIR-26 evidence")
    validate_evidence(evidence, shards)
    files = {
        ARTIFACT_FILE: file_sha256(artifact_dir / ARTIFACT_FILE),
        **{
            f"{SHARD_DIR}/{name}": file_sha256(artifact_dir / SHARD_DIR / name)
            for name in sorted(shards)
        },
    }
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
        raise ValueError("NRIR-26 artifact manifest/header differs")
    return manifest, shards


def _replay(args: argparse.Namespace) -> None:
    manifest, stored = _validate_artifact(args.artifact_dir)
    with tempfile.TemporaryDirectory(prefix="boundflow-nrir26-replay-") as temp:
        temporary = Path(temp)
        for clause in HARD_CLAUSES:
            for mode in MODES:
                name = _shard_name(clause, mode)
                output = temporary / name
                _spawn_worker(args, clause=clause, mode=mode, output=output)
                if _load_json(output) != stored[name]:
                    raise ValueError(f"NRIR-26 semantic replay differs: {name}")
    print(_canonical_json({"status": "ok", "evidence_hash": manifest["evidence_hash"]}))


def main() -> None:
    args = _parse_args()
    if args.torch_threads < 1:
        raise ValueError("NRIR-26 torch thread count must be positive")
    if args.command == "worker":
        _worker(args)
    elif args.command == "generate":
        _generate(args)
    else:
        _replay(args)


if __name__ == "__main__":
    main()
