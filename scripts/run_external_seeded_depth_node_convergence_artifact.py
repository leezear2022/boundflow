#!/usr/bin/env python3
"""Generate or replay the NRIR-24 external-seeded convergence artifact."""

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
from typing import Any, Mapping, cast

import torch

from boundflow.ir.bound import IntermediateBoundSource
from boundflow.runtime.abcrown_adapter import file_sha256
from boundflow.runtime.native_intermediate_refinement import (
    build_native_external_intermediate_constraint_seed,
)
from boundflow.runtime.native_optimized_relu_split_bab_runtime import (
    NATIVE_REEXECUTION_ATOL,
    NativeOptimizedReluSplitBabExecution,
    execute_native_optimized_relu_split_bab,
)
from boundflow.runtime.native_relu_split_bab_runtime import NativeReluSplitBabConfig
from scripts.run_end_to_end_tightness_performance_baseline import _build_context
from scripts.run_external_seeded_ancestral_refinement_artifact import (
    REFINEMENT_POLICY,
    _objective_branch_hash_evidence,
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
from scripts.run_native_real_network_ir_artifact import (
    ABCROWN_COMMIT,
    INTERMEDIATE_BOUNDS_SHA256,
    MODEL_SHA256,
    VNNCOMP_COMMIT,
    VNNLIB_SHA256,
)

ARTIFACT_SCHEMA_VERSION = "boundflow.external-seeded-convergence-artifact/v1"
EVIDENCE_SCHEMA_VERSION = "boundflow.external-seeded-convergence-evidence/v1"
SHARD_SCHEMA_VERSION = "boundflow.external-seeded-convergence-shard/v1"
ARTIFACT_FILE = "external_seeded_convergence.json"
MANIFEST_FILE = "manifest.json"
SHARD_DIR = "shards"
BUDGETS = ((7, 2), (15, 3), (31, 4))
MONOTONIC_TOLERANCE = 1e-6
RUN_ID_TEMPLATE = "nrir24:c{clause}:external-seeded-ancestral"

FROZEN_SOURCE = {
    "abcrown_commit": ABCROWN_COMMIT,
    "external_oracle_lower_hash": (
        "f99f8b863e044328e2747aa372f0d25da18257979cb94d60f8391edbf0a761fa"
    ),
    "input_lower_hash": (
        "bed6da6e49f1123b54fe61247615cf0e86f1b4257face7b073428f257607bda6"
    ),
    "input_upper_hash": (
        "955d79bd0824185cad0be1e758e6e492c51aec6c9199f81359b2528b848ec9ab"
    ),
    "intermediate_bounds_sha256": INTERMEDIATE_BOUNDS_SHA256,
    "local_complete_query_artifact_sha256": (
        "9c490bf33edab555f14e4d4c354f6d6055c1fe627102473fc2e506e29fdbaaf8"
    ),
    "local_complete_query_manifest_sha256": (
        "d76595fba9318c7dc70fd79fb9ebd28594b83c59202426f33a45c58486a409b3"
    ),
    "model_sha256": MODEL_SHA256,
    "native_ir_artifact_schema": "boundflow.native-real-network-ir-artifact/v1",
    "native_ir_manifest_sha256": (
        "5d38aa4adcd7b8593b0ca3751106d3a254b72a2698bd445a8dde59d50eb5a676"
    ),
    "native_ir_payload_sha256": (
        "3ba949de66380c7d16dc44bb934eca8400182521e761c19eaa671cd8cf0eb557"
    ),
    "objective_matrix_hash": (
        "380f1c32321ea47967ad48221134d92d494630777eabe10ef31576a0b051e408"
    ),
    "vnncomp_commit": VNNCOMP_COMMIT,
    "vnnlib_sha256": VNNLIB_SHA256,
}


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
    worker.add_argument("--max-nodes", type=int, required=True)
    worker.add_argument("--max-depth", type=int, required=True)
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


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _budget_key(max_nodes: int, max_depth: int) -> str:
    return f"n{max_nodes}_d{max_depth}"


def _shard_name(clause: int, max_nodes: int, max_depth: int) -> str:
    return f"clause{clause}_{_budget_key(max_nodes, max_depth)}.json"


def _queue_config(max_nodes: int, max_depth: int) -> NativeReluSplitBabConfig:
    if (max_nodes, max_depth) not in BUDGETS:
        raise ValueError("NRIR-24 queue budget is outside the frozen matrix")
    return NativeReluSplitBabConfig(
        max_nodes=max_nodes,
        max_depth=max_depth,
        expansion_batch_size=2,
        max_eval_batch_size=4,
    )


def _source(context: Mapping[str, Any], model: Path) -> dict[str, object]:
    source = {
        **_mapping(context["source"], "NRIR-24 source"),
        "model_sha256": file_sha256(model),
        "vnnlib_sha256": VNNLIB_SHA256,
        "intermediate_bounds_sha256": INTERMEDIATE_BOUNDS_SHA256,
        "vnncomp_commit": VNNCOMP_COMMIT,
        "abcrown_commit": ABCROWN_COMMIT,
    }
    if source != FROZEN_SOURCE:
        raise ValueError("NRIR-24 frozen source identity differs")
    return source


def _external_seed(context: Mapping[str, Any], clause: int) -> Any:
    source = _mapping(context["source"], "NRIR-24 source")
    return build_native_external_intermediate_constraint_seed(
        context["module"],
        context["input_spec"],
        seed_id=f"nrir24:resnet2b:property0:clause{clause}:external-seed",
        provider="alpha-beta-CROWN",
        constraints=context["external_relu_pre"],
        external_intermediate_bounds_hash=INTERMEDIATE_BOUNDS_SHA256,
        source_artifact_manifest_hash=str(source["native_ir_manifest_sha256"]),
        source_artifact_payload_hash=str(source["native_ir_payload_sha256"]),
        source_model_hash=MODEL_SHA256,
        source_property_hash=VNNLIB_SHA256,
        source_objective_set_hash=str(source["objective_matrix_hash"]),
    )


def _node_projection(execution: NativeOptimizedReluSplitBabExecution) -> list[object]:
    return [item.to_dict() for item in execution.trace.evaluations]


def _summary(execution: NativeOptimizedReluSplitBabExecution) -> dict[str, object]:
    trace = execution.trace
    evaluation_by_id = {item.node.node_id: item for item in trace.evaluations}
    terminal_decisions = [item for item in trace.decisions if item.kind != "expand"]
    terminal_lowers = [
        float(evaluation_by_id[item.node_id].lower) for item in terminal_decisions
    ]
    if not terminal_lowers:
        raise ValueError("NRIR-24 queue has no terminal domains")
    all_proved = (
        trace.status == "complete"
        and not trace.final_frontier_node_ids
        and all(
            item.kind == "prune" and evaluation_by_id[item.node_id].lower >= 0.0
            for item in terminal_decisions
        )
    )
    return {
        "root_lower": float(trace.evaluations[0].lower),
        "terminal_node_ids": [item.node_id for item in terminal_decisions],
        "terminal_lowers": terminal_lowers,
        "worst_terminal_lower": min(terminal_lowers),
        "best_terminal_lower": max(terminal_lowers),
        "proof_deficit": max(0.0, -min(terminal_lowers)),
        "evaluated_nodes": len(trace.evaluations),
        "terminal_domains": len(terminal_decisions),
        "max_evaluated_depth": max(item.node.depth for item in trace.evaluations),
        "bounded_tree_status": "verified" if all_proved else "unknown",
    }


def _build_shard(
    context: Mapping[str, Any],
    *,
    model: Path,
    clause: int,
    max_nodes: int,
    max_depth: int,
) -> dict[str, object]:
    if clause not in HARD_CLAUSES:
        raise ValueError("NRIR-24 clause is outside the frozen hard-clause set")
    config = _queue_config(max_nodes, max_depth)
    tensors = _mapping(context["tensors"], "NRIR-24 tensors")
    objective = tensors["linear_spec_c"][:, clause : clause + 1]
    seed = _external_seed(context, clause)
    execution = execute_native_optimized_relu_split_bab(
        context["module"],
        context["input_spec"],
        linear_spec_C=objective,
        run_id=RUN_ID_TEMPLATE.format(clause=clause),
        config=config,
        optimizer_policy=OPTIMIZER_POLICY,
        intermediate_bound_source=IntermediateBoundSource.NATIVE_REFINED,
        objective_branch_policy=BRANCH_POLICY,
        per_child_refinement_policy=REFINEMENT_POLICY,
        per_child_refinement_strategy="external_seeded_ancestral_carry_v1",
        external_constraint_seed=seed,
    )
    execution.validate()
    trace = execution.trace
    body: dict[str, object] = {
        "schema_version": SHARD_SCHEMA_VERSION,
        "clause_index": clause,
        "budget_key": _budget_key(max_nodes, max_depth),
        "performance_claimed": False,
        "source": _source(context, model),
        "protocol": {
            "queue_config": config.to_dict(),
            "optimizer_policy": OPTIMIZER_POLICY.to_dict(),
            "branch_policy": BRANCH_POLICY.to_dict(),
            "refinement_policy": REFINEMENT_POLICY.to_dict(),
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
            "objective_branch_hashes": _objective_branch_hash_evidence(execution),
        },
        "summary": _summary(execution),
    }
    shard = {**body, "semantic_hash": canonical_hash(body)}
    validate_shard(
        shard,
        expected_clause=clause,
        expected_max_nodes=max_nodes,
        expected_max_depth=max_depth,
    )
    return shard


def _validate_seed(seed: Mapping[str, Any], clause: int) -> None:
    if (
        set(seed)
        != {
            "seed_id",
            "provider",
            "primal_graph_hash",
            "input_bounds_hash",
            "external_intermediate_bounds_hash",
            "bound_intermediate_constraints_hash",
            "source_artifact_manifest_hash",
            "source_artifact_payload_hash",
            "source_model_hash",
            "source_property_hash",
            "source_objective_set_hash",
            "consumption",
            "semantics_owner",
        }
        or seed.get("seed_id")
        != f"nrir24:resnet2b:property0:clause{clause}:external-seed"
        or seed.get("provider") != "alpha-beta-CROWN"
        or seed.get("semantics_owner") != "external_verifier"
        or seed.get("consumption") != "sound_constraint_intersection_only"
        or seed.get("external_intermediate_bounds_hash") != INTERMEDIATE_BOUNDS_SHA256
        or seed.get("source_artifact_manifest_hash")
        != FROZEN_SOURCE["native_ir_manifest_sha256"]
        or seed.get("source_artifact_payload_hash")
        != FROZEN_SOURCE["native_ir_payload_sha256"]
        or seed.get("source_model_hash") != MODEL_SHA256
        or seed.get("source_property_hash") != VNNLIB_SHA256
        or seed.get("source_objective_set_hash")
        != FROZEN_SOURCE["objective_matrix_hash"]
        or any(
            not _is_sha256(seed.get(name))
            for name in (
                "primal_graph_hash",
                "input_bounds_hash",
                "bound_intermediate_constraints_hash",
            )
        )
    ):
        raise ValueError("NRIR-24 external seed identity differs")


def _validate_queue(
    queue: Mapping[str, Any],
    summary: Mapping[str, Any],
    *,
    clause: int,
    max_nodes: int,
    max_depth: int,
) -> None:
    evaluations = _list(queue.get("evaluations"), "NRIR-24 evaluations")
    decisions = _list(queue.get("decisions"), "NRIR-24 decisions")
    refinements = _list(queue.get("refinements"), "NRIR-24 refinements")
    branches = _list(queue.get("objective_branch_hashes"), "NRIR-24 branches")
    if (
        queue.get("run_id") != RUN_ID_TEMPLATE.format(clause=clause)
        or queue.get("status") != "complete"
        or queue.get("termination_reason") != "configured_bounded_tree_exhausted"
        or queue.get("final_frontier_node_ids") != []
        or not all(
            _is_sha256(queue.get(name))
            for name in (
                "root_input_lower_hash",
                "root_input_upper_hash",
                "objective_hash",
                "queue_trace_hash",
            )
        )
        or not evaluations
        or len(evaluations) > max_nodes
        or len(decisions) != len(evaluations)
        or len(refinements) != len(evaluations)
        or len(branches) != len(evaluations)
    ):
        raise ValueError("NRIR-24 queue header/coverage differs")
    evaluation_by_id: dict[str, Mapping[str, Any]] = {}
    lowers: dict[str, float] = {}
    for index, value in enumerate(evaluations):
        evaluation = _mapping(value, "NRIR-24 evaluation")
        node = _mapping(evaluation.get("node"), "NRIR-24 node")
        node_id = node.get("node_id")
        parent_id = node.get("parent_node_id")
        lower = evaluation.get("lower")
        upper = evaluation.get("upper")
        if (
            not isinstance(node_id, str)
            or node_id in evaluation_by_id
            or node_id != f"{RUN_ID_TEMPLATE.format(clause=clause)}:n{index:06d}"
            or not isinstance(node.get("depth"), int)
            or int(node["depth"]) > max_depth
            or not isinstance(lower, (int, float))
            or not isinstance(upper, (int, float))
            or not math.isfinite(float(lower))
            or not math.isfinite(float(upper))
            or float(lower) > float(upper)
            or not _is_sha256(node.get("split_state_hash"))
            or not _is_sha256(evaluation.get("selected_state_hash"))
            or not _is_sha256(evaluation.get("intermediate_refinement_trace_hash"))
        ):
            raise ValueError("NRIR-24 evaluation identity differs")
        if index == 0:
            if parent_id is not None or node.get("depth") != 0:
                raise ValueError("NRIR-24 root lineage differs")
        else:
            parent = evaluation_by_id.get(str(parent_id))
            if (
                parent is None
                or int(node["depth"])
                != int(_mapping(parent["node"], "NRIR-24 parent node")["depth"]) + 1
            ):
                raise ValueError("NRIR-24 parent lineage differs")
        evaluation_by_id[node_id] = evaluation
        lowers[node_id] = float(lower)
    refinement_by_id: dict[str, Mapping[str, Any]] = {}
    for value in refinements:
        refinement = _mapping(value, "NRIR-24 refinement")
        node_id = refinement.get("node_id")
        if (
            not isinstance(node_id, str)
            or node_id in refinement_by_id
            or node_id not in evaluation_by_id
            or canonical_hash(refinement)
            != evaluation_by_id[node_id].get("intermediate_refinement_trace_hash")
            or not _is_sha256(refinement.get("refinement_plan_hash"))
            or not _is_sha256(refinement.get("refinement_semantic_trace_hash"))
            or not _is_sha256(refinement.get("final_intermediate_bounds_hash"))
        ):
            raise ValueError("NRIR-24 refinement binding differs")
        parent_id = _mapping(evaluation_by_id[node_id]["node"], "NRIR-24 node").get(
            "parent_node_id"
        )
        if parent_id is None:
            if (
                not _is_sha256(refinement.get("external_constraint_seed_hash"))
                or "source_parent_node_id" in refinement
            ):
                raise ValueError("NRIR-24 root seed binding differs")
        else:
            parent_refinement = refinement_by_id.get(str(parent_id))
            if (
                parent_refinement is None
                or "external_constraint_seed_hash" in refinement
                or refinement.get("source_parent_node_id") != parent_id
                or refinement.get("source_intermediate_constraints_hash")
                != parent_refinement.get("final_intermediate_bounds_hash")
                or refinement.get("source_refinement_plan_hash")
                != parent_refinement.get("refinement_plan_hash")
                or refinement.get("source_refinement_semantic_trace_hash")
                != parent_refinement.get("refinement_semantic_trace_hash")
            ):
                raise ValueError("NRIR-24 ancestral refinement lineage differs")
        refinement_by_id[node_id] = refinement
    branch_ids: set[str] = set()
    for value in branches:
        branch = _mapping(value, "NRIR-24 objective branch")
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
            raise ValueError("NRIR-24 objective branch binding differs")
        branch_ids.add(node_id)
    decision_by_id: dict[str, Mapping[str, Any]] = {}
    children: set[str] = set()
    terminal_ids: list[str] = []
    for index, value in enumerate(decisions):
        decision = _mapping(value, "NRIR-24 decision")
        node_id = decision.get("node_id")
        kind = decision.get("kind")
        if (
            decision.get("decision_index") != index
            or not isinstance(node_id, str)
            or node_id in decision_by_id
            or node_id not in evaluation_by_id
            or kind not in {"expand", "prune", "terminal"}
        ):
            raise ValueError("NRIR-24 decision identity differs")
        decision_by_id[node_id] = decision
        if kind == "expand":
            child_ids = _list(decision.get("child_node_ids"), "NRIR-24 children")
            branch = _mapping(decision.get("branch_candidate"), "NRIR-24 branch")
            if (
                len(child_ids) != 2
                or any(child_id not in evaluation_by_id for child_id in child_ids)
                or any(
                    _mapping(evaluation_by_id[child_id]["node"], "NRIR-24 child").get(
                        "parent_node_id"
                    )
                    != node_id
                    for child_id in child_ids
                )
                or branch != evaluation_by_id[node_id].get("branch_candidate")
            ):
                raise ValueError("NRIR-24 expansion lineage differs")
            children.update(str(child_id) for child_id in child_ids)
        else:
            terminal_ids.append(node_id)
    root_id = f"{RUN_ID_TEMPLATE.format(clause=clause)}:n000000"
    if children != set(evaluation_by_id) - {root_id}:
        raise ValueError("NRIR-24 expanded child coverage differs")
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
            int(_mapping(value["node"], "NRIR-24 node")["depth"])
            for value in evaluation_by_id.values()
        ),
        "bounded_tree_status": "verified" if all_proved else "unknown",
    }
    if not terminal_ids or summary != expected_summary:
        raise ValueError("NRIR-24 summary differs")


def validate_shard(
    shard: Mapping[str, Any],
    *,
    expected_clause: int,
    expected_max_nodes: int,
    expected_max_depth: int,
) -> None:
    body = dict(shard)
    semantic_hash = body.pop("semantic_hash", None)
    if (
        shard.get("schema_version") != SHARD_SCHEMA_VERSION
        or shard.get("clause_index") != expected_clause
        or shard.get("budget_key")
        != _budget_key(expected_max_nodes, expected_max_depth)
        or shard.get("performance_claimed") is not False
        or semantic_hash != canonical_hash(body)
        or shard.get("source") != FROZEN_SOURCE
    ):
        raise ValueError("NRIR-24 shard header/source differs")
    protocol = _mapping(shard.get("protocol"), "NRIR-24 protocol")
    if protocol != {
        "queue_config": _queue_config(expected_max_nodes, expected_max_depth).to_dict(),
        "optimizer_policy": OPTIMIZER_POLICY.to_dict(),
        "branch_policy": BRANCH_POLICY.to_dict(),
        "refinement_policy": REFINEMENT_POLICY.to_dict(),
        "strategy": "external_seeded_ancestral_carry_v1",
    }:
        raise ValueError("NRIR-24 protocol differs")
    seed = _mapping(shard.get("external_seed_ir"), "NRIR-24 seed")
    _validate_seed(seed, expected_clause)
    if shard.get("external_seed_hash") != canonical_hash(seed):
        raise ValueError("NRIR-24 seed hash differs")
    _validate_queue(
        _mapping(shard.get("queue"), "NRIR-24 queue"),
        _mapping(shard.get("summary"), "NRIR-24 summary"),
        clause=expected_clause,
        max_nodes=expected_max_nodes,
        max_depth=expected_max_depth,
    )


def _logical_domain_nesting(
    smaller: Mapping[str, Any], larger: Mapping[str, Any]
) -> dict[str, object]:
    small_queue = _mapping(smaller["queue"], "NRIR-24 smaller queue")
    large_queue = _mapping(larger["queue"], "NRIR-24 larger queue")
    small_values = [
        _mapping(value, "NRIR-24 smaller evaluation")
        for value in _list(small_queue["evaluations"], "NRIR-24 smaller evaluations")
    ]
    large_values = [
        _mapping(value, "NRIR-24 larger evaluation")
        for value in _list(large_queue["evaluations"], "NRIR-24 larger evaluations")
    ]
    small_by_id = {
        str(_mapping(value["node"], "NRIR-24 node")["node_id"]): value
        for value in small_values
    }
    large_by_id = {
        str(_mapping(value["node"], "NRIR-24 node")["node_id"]): value
        for value in large_values
    }
    small_by_split = {
        str(_mapping(value["node"], "NRIR-24 node")["split_state_hash"]): value
        for value in small_values
    }
    large_by_split = {
        str(_mapping(value["node"], "NRIR-24 node")["split_state_hash"]): value
        for value in large_values
    }

    def parent_split(
        value: Mapping[str, Any], by_id: Mapping[str, Mapping[str, Any]]
    ) -> str | None:
        node = _mapping(value["node"], "NRIR-24 node")
        parent_id = node.get("parent_node_id")
        if parent_id is None:
            return None
        parent = by_id[str(parent_id)]
        return str(_mapping(parent["node"], "NRIR-24 parent node")["split_state_hash"])

    def refinement_by_split(
        queue: Mapping[str, Any], by_id: Mapping[str, Mapping[str, Any]]
    ) -> dict[str, dict[str, object]]:
        result: dict[str, dict[str, object]] = {}
        for value in _list(queue["refinements"], "NRIR-24 refinements"):
            refinement = dict(_mapping(value, "NRIR-24 refinement"))
            node_id = str(refinement.pop("node_id"))
            refinement.pop("source_parent_node_id", None)
            split_hash = str(
                _mapping(by_id[node_id]["node"], "NRIR-24 node")["split_state_hash"]
            )
            result[split_hash] = refinement
        return result

    small_refinements = refinement_by_split(small_queue, small_by_id)
    large_refinements = refinement_by_split(large_queue, large_by_id)
    all_present = set(small_by_split) <= set(large_by_split)
    structure_matches = all_present
    branch_selection_matches = all_present
    refinement_semantics_match = all_present
    max_lower_diff = 0.0
    max_upper_diff = 0.0
    max_priority_diff = 0.0
    for split_hash, small in small_by_split.items():
        large = large_by_split.get(split_hash)
        if large is None:
            continue
        small_node = _mapping(small["node"], "NRIR-24 smaller node")
        large_node = _mapping(large["node"], "NRIR-24 larger node")
        structure_matches = structure_matches and (
            {
                "depth": small_node["depth"],
                "branch_relu_input": small_node["branch_relu_input"],
                "branch_neuron_index": small_node["branch_neuron_index"],
                "branch_value": small_node["branch_value"],
                "parent_split_state_hash": parent_split(small, small_by_id),
            }
            == {
                "depth": large_node["depth"],
                "branch_relu_input": large_node["branch_relu_input"],
                "branch_neuron_index": large_node["branch_neuron_index"],
                "branch_value": large_node["branch_value"],
                "parent_split_state_hash": parent_split(large, large_by_id),
            }
        )
        branch_selection_matches = branch_selection_matches and (
            small.get("branch_candidate") == large.get("branch_candidate")
        )
        refinement_semantics_match = refinement_semantics_match and (
            small_refinements[split_hash] == large_refinements[split_hash]
        )
        max_lower_diff = max(
            max_lower_diff, abs(float(small["lower"]) - float(large["lower"]))
        )
        max_upper_diff = max(
            max_upper_diff, abs(float(small["upper"]) - float(large["upper"]))
        )
        max_priority_diff = max(
            max_priority_diff,
            abs(float(small["priority"]) - float(large["priority"])),
        )
    numerical_semantics_match = (
        max(max_lower_diff, max_upper_diff, max_priority_diff)
        <= NATIVE_REEXECUTION_ATOL
    )
    passed = (
        all_present
        and structure_matches
        and branch_selection_matches
        and refinement_semantics_match
        and numerical_semantics_match
    )
    return {
        "smaller_domain_count": len(small_by_split),
        "larger_domain_count": len(large_by_split),
        "all_smaller_domains_present": all_present,
        "logical_lineage_matches": structure_matches,
        "selected_branch_matches": branch_selection_matches,
        "refinement_semantics_match": refinement_semantics_match,
        "max_common_lower_abs_diff": max_lower_diff,
        "max_common_upper_abs_diff": max_upper_diff,
        "max_common_priority_abs_diff": max_priority_diff,
        "numeric_tolerance": NATIVE_REEXECUTION_ATOL,
        "numerical_semantics_match": numerical_semantics_match,
        "passed": passed,
    }


def _curve(shards: list[Mapping[str, Any]], clause: int) -> dict[str, object]:
    points: list[dict[str, object]] = []
    previous: Mapping[str, Any] | None = None
    for shard, (max_nodes, max_depth) in zip(shards, BUDGETS):
        summary = _mapping(shard["summary"], "NRIR-24 summary")
        point: dict[str, object] = {
            "budget_key": _budget_key(max_nodes, max_depth),
            "max_nodes": max_nodes,
            "max_depth": max_depth,
            "evaluated_nodes": summary["evaluated_nodes"],
            "terminal_domains": summary["terminal_domains"],
            "worst_terminal_lower": summary["worst_terminal_lower"],
            "best_terminal_lower": summary["best_terminal_lower"],
            "proof_deficit": summary["proof_deficit"],
            "bounded_tree_status": summary["bounded_tree_status"],
        }
        if previous is None:
            point["worst_lower_delta_from_previous"] = None
            point["deficit_reduction_per_added_node"] = None
        else:
            previous_summary = _mapping(previous["summary"], "NRIR-24 previous")
            added_nodes = int(summary["evaluated_nodes"]) - int(
                previous_summary["evaluated_nodes"]
            )
            point["worst_lower_delta_from_previous"] = float(
                summary["worst_terminal_lower"]
            ) - float(previous_summary["worst_terminal_lower"])
            point["deficit_reduction_per_added_node"] = (
                float(previous_summary["proof_deficit"])
                - float(summary["proof_deficit"])
            ) / added_nodes
        points.append(point)
        previous = shard
    deltas = [
        cast(float, point["worst_lower_delta_from_previous"]) for point in points[1:]
    ]
    return {
        "clause_index": clause,
        "points": points,
        "monotonic_non_decreasing": all(
            delta >= -MONOTONIC_TOLERANCE for delta in deltas
        ),
        "strict_improvement_after_depth_two": any(
            delta > MONOTONIC_TOLERANCE for delta in deltas
        ),
        "depth_four_saturated": deltas[-1] <= MONOTONIC_TOLERANCE,
        "deepest_bounded_tree_status": points[-1]["bounded_tree_status"],
    }


def _assemble_evidence(shards: Mapping[str, Mapping[str, Any]]) -> dict[str, object]:
    expected_keys = {
        _shard_name(clause, max_nodes, max_depth)
        for clause in HARD_CLAUSES
        for max_nodes, max_depth in BUDGETS
    }
    if set(shards) != expected_keys:
        raise ValueError("NRIR-24 shard matrix differs")
    curves: list[dict[str, object]] = []
    logical_domain_nesting: dict[str, object] = {}
    for clause in HARD_CLAUSES:
        ordered: list[Mapping[str, Any]] = []
        for max_nodes, max_depth in BUDGETS:
            name = _shard_name(clause, max_nodes, max_depth)
            shard = shards[name]
            validate_shard(
                shard,
                expected_clause=clause,
                expected_max_nodes=max_nodes,
                expected_max_depth=max_depth,
            )
            ordered.append(shard)
        for smaller, larger in zip(ordered, ordered[1:]):
            key = f"clause{clause}:{smaller['budget_key']}->{larger['budget_key']}"
            logical_domain_nesting[key] = _logical_domain_nesting(smaller, larger)
        curves.append(_curve(ordered, clause))
    monotonic = all(bool(curve["monotonic_non_decreasing"]) for curve in curves)
    strict = any(bool(curve["strict_improvement_after_depth_two"]) for curve in curves)
    saturated = all(bool(curve["depth_four_saturated"]) for curve in curves)
    nesting = all(
        _mapping(value, "NRIR-24 nesting").get("passed") is True
        for value in logical_domain_nesting.values()
    )
    status = (
        "validated_reduced"
        if monotonic and strict and not saturated and nesting
        else "validated_no_go"
    )
    evidence: dict[str, object] = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": status,
        "performance_claimed": False,
        "claim_boundary": (
            "fixed ResNet property 0 clauses 0/2/4, CPU, external-seeded "
            "ancestral strategy, nested 7/15/31-node depth-2/3/4 bounded trees"
        ),
        "source": deepcopy(FROZEN_SOURCE),
        "protocol": {
            "hard_clause_indices": list(HARD_CLAUSES),
            "budgets": [
                _queue_config(max_nodes, max_depth).to_dict()
                for max_nodes, max_depth in BUDGETS
            ],
            "optimizer_policy": OPTIMIZER_POLICY.to_dict(),
            "branch_policy": BRANCH_POLICY.to_dict(),
            "refinement_policy": REFINEMENT_POLICY.to_dict(),
            "strategy": "external_seeded_ancestral_carry_v1",
            "execution_isolation": "fresh_python_process_per_clause_budget",
        },
        "shard_semantic_hashes": {
            name: shard["semantic_hash"] for name, shard in sorted(shards.items())
        },
        "logical_domain_nesting": logical_domain_nesting,
        "curves": curves,
        "gates": {
            "all_nine_units_present": True,
            "all_nested_logical_domains_match": nesting,
            "all_clauses_monotonic_non_decreasing": monotonic,
            "strict_improvement_after_depth_two": strict,
            "all_clauses_depth_four_saturated": saturated,
            "any_fixed_clause_bounded_tree_closed": any(
                curve["deepest_bounded_tree_status"] == "verified" for curve in curves
            ),
        },
        "limitations": [
            "fixed ResNet property 0 clauses 0, 2, and 4 on CPU only",
            "external seed generation remains owned by alpha-beta-CROWN",
            "bounded trees do not imply complete property or verifier closure",
            "single executions are convergence evidence, not a performance claim",
            "GPU, multi-workload, cuts, and full activation-BaB remain pending",
        ],
    }
    return evidence


def build_evidence(shards: Mapping[str, Mapping[str, Any]]) -> dict[str, object]:
    evidence = _assemble_evidence(shards)
    validate_evidence(evidence, shards)
    return evidence


def validate_evidence(
    evidence: Mapping[str, Any], shards: Mapping[str, Mapping[str, Any]]
) -> None:
    rebuilt = _assemble_evidence(shards)
    if evidence != rebuilt:
        raise ValueError("NRIR-24 aggregate evidence differs")


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(_canonical_json(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _worker(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.torch_threads)
    context = _build_context(
        model=args.model,
        source_artifact_dir=args.source_artifact_dir,
        local_artifact_dir=args.local_artifact_dir,
    )
    shard = _build_shard(
        context,
        model=args.model,
        clause=args.clause,
        max_nodes=args.max_nodes,
        max_depth=args.max_depth,
    )
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
    args: argparse.Namespace,
    *,
    clause: int,
    max_nodes: int,
    max_depth: int,
    output: Path,
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
        "--max-nodes",
        str(max_nodes),
        "--max-depth",
        str(max_depth),
        "--output",
        str(output),
    ]
    started = time.monotonic()
    subprocess.run(command, check=True)
    return time.monotonic() - started


def _load_shards(artifact_dir: Path) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for clause in HARD_CLAUSES:
        for max_nodes, max_depth in BUDGETS:
            name = _shard_name(clause, max_nodes, max_depth)
            result[name] = _load_json(artifact_dir / SHARD_DIR / name)
    return result


def shard_is_reusable(
    path: Path, *, clause: int, max_nodes: int, max_depth: int
) -> bool:
    try:
        shard = _load_json(path)
        validate_shard(
            shard,
            expected_clause=clause,
            expected_max_nodes=max_nodes,
            expected_max_depth=max_depth,
        )
    except (OSError, TypeError, ValueError):
        return False
    return True


def _write_artifact(
    artifact_dir: Path, shards: Mapping[str, Mapping[str, Any]]
) -> None:
    evidence = build_evidence(shards)
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
        for max_nodes, max_depth in BUDGETS:
            path = (
                args.artifact_dir
                / SHARD_DIR
                / _shard_name(clause, max_nodes, max_depth)
            )
            if not args.force and shard_is_reusable(
                path,
                clause=clause,
                max_nodes=max_nodes,
                max_depth=max_depth,
            ):
                print(_canonical_json({"status": "resume", "shard": path.name}))
                continue
            elapsed = _spawn_worker(
                args,
                clause=clause,
                max_nodes=max_nodes,
                max_depth=max_depth,
                output=path,
            )
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
    evidence = _mapping(artifact.get("evidence"), "NRIR-24 evidence")
    validate_evidence(evidence, shards)
    expected_files = {ARTIFACT_FILE: file_sha256(artifact_path)}
    expected_files.update(
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
        or manifest.get("files") != expected_files
        or manifest.get("evidence_hash") != canonical_hash(evidence)
    ):
        raise ValueError("NRIR-24 artifact manifest/header differs")
    return manifest, shards


def _replay(args: argparse.Namespace) -> None:
    manifest, stored = _validate_artifact(args.artifact_dir)
    with tempfile.TemporaryDirectory(prefix="boundflow-nrir24-replay-") as temp:
        temporary = Path(temp)
        for clause in HARD_CLAUSES:
            for max_nodes, max_depth in BUDGETS:
                name = _shard_name(clause, max_nodes, max_depth)
                output = temporary / name
                _spawn_worker(
                    args,
                    clause=clause,
                    max_nodes=max_nodes,
                    max_depth=max_depth,
                    output=output,
                )
                actual = _load_json(output)
                if actual != stored[name]:
                    raise ValueError(f"NRIR-24 semantic replay differs: {name}")
    print(_canonical_json({"status": "ok", "evidence_hash": manifest["evidence_hash"]}))


def main() -> None:
    args = _parse_args()
    if args.torch_threads < 1:
        raise ValueError("NRIR-24 torch thread count must be positive")
    if args.command == "worker":
        _worker(args)
    elif args.command == "generate":
        _generate(args)
    else:
        _replay(args)


if __name__ == "__main__":
    main()
