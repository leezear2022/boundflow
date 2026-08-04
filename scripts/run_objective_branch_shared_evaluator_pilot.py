#!/usr/bin/env python3
"""Generate or replay the preregistered NRIR-39 fixed-budget branch pilot."""

# pylint: disable=too-many-lines,too-many-locals,too-many-statements
# pylint: disable=too-many-boolean-expressions,missing-function-docstring
# pylint: disable=import-outside-toplevel,duplicate-code,too-many-branches

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping

from boundflow.ir.objective_branch_shared_evaluator import (
    NativeObjectiveBranchBindingIR,
    NativeObjectiveBranchSharedDecisionIR,
    NativeObjectiveBranchSharedTaskKind,
)
from boundflow.runtime.native_objective_branch_score import (
    NativeObjectiveBranchPolicy,
)
from scripts.run_shared_parametric_objective_evaluator_pilot import (
    _execute_floor,
    _source,
)
from scripts.run_typed_hard_clause_escalation_artifact import (
    _load_query_runtime,
    _policies,
    _public_workload,
    _resolve_workloads,
)

PILOT_SCHEMA_VERSION = "boundflow.objective-branch-shared-evaluator-pilot/v1"
MANIFEST_SCHEMA_VERSION = (
    "boundflow.objective-branch-shared-evaluator-pilot-manifest/v1"
)
ARTIFACT_DIR = Path(
    "artifacts/objective-branch-shared-evaluator/"
    "vnncomp21-resnet2b-property0-cpu-pilot-v1"
)
TORCH_THREADS = 8
EXPECTED_SELECTED = (2, 3)
EXPECTED_EVALUATIONS = 31
EXPECTED_ACTIVE = 16
EXPECTED_DEPTH_COUNTS = {"0": 1, "1": 2, "2": 4, "3": 8, "4": 16}
EXPECTED_CONTROL_WORST = {2: -37.57428741455078, 3: -35.90021514892578}


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


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


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
        "boundflow/ir/objective_branch_shared_evaluator.py",
        "boundflow/runtime/native_objective_branch_shared_evaluator.py",
        "boundflow/ir/branch.py",
        "boundflow/runtime/native_objective_branch_score.py",
        "boundflow/ir/shared_parametric_ancestral.py",
        "boundflow/runtime/native_shared_parametric_ancestral.py",
        "scripts/run_objective_branch_shared_evaluator_pilot.py",
    )
    return _canonical_hash({path: _file_sha256(root / path) for path in paths})


def _active_rows(summary: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    rows = summary.get("evaluations")
    if not isinstance(rows, list):
        raise TypeError("NRIR-39 evaluation rows differ")
    return tuple(row for row in rows if row.get("active") is True)


def _execution_summary(execution: Any) -> dict[str, object]:
    decisions = {item.node_id: item for item in execution.queue.trace.decisions}
    frontier = set(execution.queue.trace.final_frontier_node_ids)
    rows: list[dict[str, Any]] = []
    for evaluation in execution.queue.trace.evaluations:
        node_id = evaluation.node.node_id
        decision = decisions.get(node_id)
        active = node_id in frontier or (
            decision is not None and decision.kind == "terminal"
        )
        rows.append(
            {
                "node_id": node_id,
                "parent_node_id": evaluation.node.parent_node_id,
                "depth": evaluation.node.depth,
                "split_state_hash": evaluation.node.split_state_hash,
                "lower": evaluation.lower,
                "upper": evaluation.upper,
                "evaluation_hash": _canonical_hash(evaluation.to_dict()),
                "active": active,
                "decision_kind": None if decision is None else decision.kind,
                "decision_reason": None if decision is None else decision.reason,
                "branch_candidate": (
                    None
                    if evaluation.branch_candidate is None
                    else evaluation.branch_candidate.to_dict()
                ),
            }
        )
    active_lowers = [float(row["lower"]) for row in rows if row["active"]]
    depth_counts = {
        str(depth): sum(row["depth"] == depth for row in rows)
        for depth in sorted({int(row["depth"]) for row in rows})
    }
    return {
        "execution_hash": _canonical_hash(execution.to_dict()),
        "queue_trace_hash": execution.queue.trace.stable_hash(),
        "evaluation_count": len(rows),
        "active_count": len(active_lowers),
        "batch_count": len(execution.batch_commits),
        "depth_counts": depth_counts,
        "root_lower": rows[0]["lower"],
        "root_upper": rows[0]["upper"],
        "worst_active_lower": min(active_lowers),
        "median_active_lower": float(statistics.median(active_lowers)),
        "cache_outcomes": [
            item.cache_event.outcome for item in execution.compiler_batches
        ],
        "evaluations": rows,
    }


def _branch_evidence(execution: Any) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for node_id, branch in execution.shared_execution.queue.objective_branch_executions:
        trace = branch.trace.to_dict(program=branch.program)
        result.append(
            {
                "node_id": node_id,
                "trace": trace,
                "trace_hash": branch.trace.stable_hash(program=branch.program),
                "selected_branch": branch.branch.to_dict(),
            }
        )
    return result


def _semantic_clause(value: Mapping[str, Any]) -> dict[str, object]:
    return {
        key: value[key]
        for key in (
            "original_clause_index",
            "plan",
            "control",
            "candidate",
            "branch_bindings",
            "branch_evidence",
            "decision",
            "task_kinds",
            "task_ir_hash",
            "schedule_hash",
            "task_semantic_hash",
        )
    }


def _clause_payload(execution: Any, control: Any, ordinal: int) -> dict[str, Any]:
    plan = execution.plan
    bindings = [item.to_dict() for item in execution.branch_bindings]
    control_summary = _execution_summary(control)
    candidate_summary = _execution_summary(execution.shared_execution)
    decision = execution.decision.to_dict()
    task_kinds = [task.kind.value for task in execution.task_ir.tasks]
    task_semantic_hash = _canonical_hash(
        {
            "plan_hash": plan.stable_hash(),
            "candidate_execution_hash": candidate_summary["execution_hash"],
            "branch_binding_hashes": [
                item.stable_hash() for item in execution.branch_bindings
            ],
            "decision_hash": execution.decision.stable_hash(),
            "task_kinds": task_kinds,
        }
    )
    value: dict[str, Any] = {
        "original_clause_index": ordinal,
        "plan": {
            "plan_hash": plan.stable_hash(),
            "shared_plan_hash": plan.shared_plan.stable_hash(),
            "branch_policy_hash": plan.branch_policy_hash,
            "candidates_per_relu": plan.candidates_per_relu,
            "candidate_batch_size": plan.candidate_batch_size,
            "max_candidates": plan.max_candidates,
            "candidate_policy_id": plan.candidate_policy_id,
            "reduce_policy": plan.reduce_policy,
            "minimum_worst_active_lower_improvement": (
                plan.minimum_worst_active_lower_improvement
            ),
            "median_lower_tolerance": plan.median_lower_tolerance,
            "frozen_variables": list(plan.frozen_variables),
            "performance_claimed": plan.performance_claimed,
        },
        "control": control_summary,
        "candidate": candidate_summary,
        "branch_bindings": bindings,
        "branch_evidence": _branch_evidence(execution),
        "decision": decision,
        "task_kinds": task_kinds,
        "task_ir_hash": execution.task_ir.stable_hash(),
        "schedule_hash": execution.schedule.stable_hash(execution.task_ir),
        "task_semantic_hash": task_semantic_hash,
    }
    value["clause_hash"] = _canonical_hash(_semantic_clause(value))
    return value


def _validate_summary(value: Mapping[str, Any]) -> None:
    rows = value.get("evaluations")
    if not isinstance(rows, list) or len(rows) != EXPECTED_EVALUATIONS:
        raise ValueError("NRIR-39 evaluation coverage differs")
    node_ids = [row.get("node_id") for row in rows]
    active = _active_rows(value)
    lowers = [float(row["lower"]) for row in active]
    if (
        len(node_ids) != len(set(node_ids))
        or value.get("evaluation_count") != EXPECTED_EVALUATIONS
        or value.get("active_count") != EXPECTED_ACTIVE
        or value.get("batch_count") != 16
        or value.get("depth_counts") != EXPECTED_DEPTH_COUNTS
        or len(active) != EXPECTED_ACTIVE
        or any(row.get("depth") != 4 for row in active)
        or any(
            not _is_sha256(row.get("split_state_hash"))
            or not _is_sha256(row.get("evaluation_hash"))
            or not all(math.isfinite(float(row[key])) for key in ("lower", "upper"))
            or float(row["lower"]) > float(row["upper"])
            for row in rows
        )
        or float(value["root_lower"]) != float(rows[0]["lower"])
        or float(value["root_upper"]) != float(rows[0]["upper"])
        or float(value["worst_active_lower"]) != min(lowers)
        or float(value["median_active_lower"]) != float(statistics.median(lowers))
        or not _is_sha256(value.get("execution_hash"))
        or not _is_sha256(value.get("queue_trace_hash"))
        or len(value.get("cache_outcomes", [])) != 16
        or value.get("cache_outcomes", []).count("miss_compiled") > 1
        or any(
            item not in {"miss_compiled", "hit_exact_contract"}
            for item in value.get("cache_outcomes", [])
        )
    ):
        raise ValueError("NRIR-39 execution summary differs")


def _binding_from_dict(value: Mapping[str, Any]) -> NativeObjectiveBranchBindingIR:
    return NativeObjectiveBranchBindingIR(**value)


def _decision_from_dict(
    value: Mapping[str, Any],
) -> NativeObjectiveBranchSharedDecisionIR:
    return NativeObjectiveBranchSharedDecisionIR(**value)


def _validate_clause(value: Mapping[str, Any]) -> None:
    ordinal = value.get("original_clause_index")
    if ordinal not in EXPECTED_SELECTED:
        raise ValueError("NRIR-39 original clause differs")
    plan = value.get("plan")
    control = value.get("control")
    candidate = value.get("candidate")
    raw_bindings = value.get("branch_bindings")
    evidence = value.get("branch_evidence")
    if not isinstance(plan, dict):
        raise TypeError("NRIR-39 clause plan differs")
    if not isinstance(control, dict) or not isinstance(candidate, dict):
        raise TypeError("NRIR-39 clause objects differ")
    if not isinstance(raw_bindings, list) or not isinstance(evidence, list):
        raise TypeError("NRIR-39 branch evidence differs")
    policy = NativeObjectiveBranchPolicy()
    if (
        plan.get("branch_policy_hash") != policy.stable_hash()
        or plan.get("candidates_per_relu") != 8
        or plan.get("candidate_batch_size") != 64
        or plan.get("max_candidates") != 256
        or plan.get("candidate_policy_id") != "top_width_per_relu_v1"
        or plan.get("reduce_policy") != "maximize_worst_child_then_mean"
        or plan.get("minimum_worst_active_lower_improvement") != 1.0
        or plan.get("median_lower_tolerance") != 1e-5
        or plan.get("performance_claimed") is not False
        or not _is_sha256(plan.get("plan_hash"))
        or not _is_sha256(plan.get("shared_plan_hash"))
    ):
        raise ValueError("NRIR-39 frozen plan differs")
    _validate_summary(control)
    _validate_summary(candidate)
    if not math.isclose(
        float(control["worst_active_lower"]),
        EXPECTED_CONTROL_WORST[int(ordinal)],
        rel_tol=1e-5,
        abs_tol=1e-5,
    ):
        raise ValueError("NRIR-39 control no longer reproduces NRIR-37")
    bindings = tuple(_binding_from_dict(item) for item in raw_bindings)
    for binding in bindings:
        binding.validate()
    binding_by_id = {item.node_id: item for item in bindings}
    candidate_rows = {item["node_id"]: item for item in candidate["evaluations"]}
    expected_branch_ids = {
        node_id
        for node_id, row in candidate_rows.items()
        if row["branch_candidate"] is not None
    }
    if (
        len(binding_by_id) != len(bindings)
        or set(binding_by_id) != expected_branch_ids
        or len(evidence) != len(bindings)
    ):
        raise ValueError("NRIR-39 branch coverage differs")
    evidence_ids: set[str] = set()
    for item in evidence:
        node_id = item.get("node_id")
        trace = item.get("trace")
        selected = item.get("selected_branch")
        if (
            not isinstance(node_id, str)
            or node_id in evidence_ids
            or not isinstance(trace, dict)
            or not isinstance(selected, dict)
            or node_id not in binding_by_id
        ):
            raise ValueError("NRIR-39 branch trace identity differs")
        evidence_ids.add(node_id)
        binding = binding_by_id[node_id]
        scores = trace.get("scores")
        program_hashes = trace.get("program_hashes")
        selected_ordinal = trace.get("selected_candidate_ordinal")
        if not isinstance(scores, list) or not isinstance(program_hashes, dict):
            raise TypeError("NRIR-39 branch trace payload differs")
        expected_selected = min(
            scores,
            key=lambda score: (
                -float(score["worst_child_lower"]),
                -float(score["mean_child_lower"]),
                int(score["candidate_ordinal"]),
            ),
        )["candidate_ordinal"]
        if (
            item.get("trace_hash") != _canonical_hash(trace)
            or binding.branch_trace_hash != item.get("trace_hash")
            or binding.branch_plan_hash != program_hashes.get("branch_plan_hash")
            or binding.branch_task_hash != program_hashes.get("branch_task_module_hash")
            or binding.branch_schedule_hash
            != program_hashes.get("branch_schedule_hash")
            or trace.get("score_hash") != _canonical_hash(scores)
            or [score.get("candidate_ordinal") for score in scores]
            != list(range(len(scores)))
            or selected_ordinal != expected_selected
            or binding.selected_candidate_ordinal != selected_ordinal
            or binding.candidate_count != len(scores)
            or binding.selected_relu_input != selected.get("relu_input")
            or binding.selected_neuron_index != selected.get("neuron_index")
            or candidate_rows[node_id]["branch_candidate"] != selected
            or binding.evaluation_hash != candidate_rows[node_id]["evaluation_hash"]
        ):
            raise ValueError("NRIR-39 branch semantic binding differs")
    decision = _decision_from_dict(value["decision"])
    decision.validate()
    control_worst = float(control["worst_active_lower"])
    candidate_worst = float(candidate["worst_active_lower"])
    control_median = float(control["median_active_lower"])
    candidate_median = float(candidate["median_active_lower"])
    if (
        decision.plan_hash != plan["plan_hash"]
        or decision.control_execution_hash != control["execution_hash"]
        or decision.candidate_execution_hash != candidate["execution_hash"]
        or decision.control_active_count != EXPECTED_ACTIVE
        or decision.candidate_active_count != EXPECTED_ACTIVE
        or decision.branch_execution_count != len(bindings)
        or decision.control_root_lower != control["root_lower"]
        or decision.candidate_root_lower != candidate["root_lower"]
        or decision.control_worst_active_lower != control_worst
        or decision.candidate_worst_active_lower != candidate_worst
        or decision.worst_active_lower_improvement != candidate_worst - control_worst
        or decision.control_median_active_lower != control_median
        or decision.candidate_median_active_lower != candidate_median
        or decision.median_active_lower_delta != candidate_median - control_median
        or decision.structure_passed is not True
    ):
        raise ValueError("NRIR-39 decision evidence differs")
    task_kinds = [item.value for item in NativeObjectiveBranchSharedTaskKind]
    binding_hashes = [item.stable_hash() for item in bindings]
    expected_task_semantic = _canonical_hash(
        {
            "plan_hash": plan["plan_hash"],
            "candidate_execution_hash": candidate["execution_hash"],
            "branch_binding_hashes": binding_hashes,
            "decision_hash": decision.stable_hash(),
            "task_kinds": task_kinds,
        }
    )
    if (
        value.get("task_kinds") != task_kinds
        or value.get("task_semantic_hash") != expected_task_semantic
        or not _is_sha256(value.get("task_ir_hash"))
        or not _is_sha256(value.get("schedule_hash"))
        or value.get("clause_hash") != _canonical_hash(_semantic_clause(value))
    ):
        raise ValueError("NRIR-39 Task/Schedule artifact binding differs")


def _semantic_pilot(value: Mapping[str, Any]) -> dict[str, object]:
    return {
        key: value[key]
        for key in (
            "protocol",
            "workload",
            "source",
            "clauses",
            "all_candidate_gates_passed",
            "claim_boundary",
            "performance_claimed",
        )
    }


def validate_pilot(value: Mapping[str, Any]) -> None:
    protocol = value.get("protocol")
    clauses = value.get("clauses")
    if (
        value.get("schema_version") != PILOT_SCHEMA_VERSION
        or value.get("source", {}).get("native_code_revision") != _code_revision()
        or not isinstance(protocol, dict)
        or protocol.get("selected_original_clause_indices") != list(EXPECTED_SELECTED)
        or protocol.get("node_budget") != EXPECTED_EVALUATIONS
        or protocol.get("depth_budget") != 4
        or protocol.get("branch_policy") != NativeObjectiveBranchPolicy().to_dict()
        or protocol.get("single_changed_variable") != "branch_candidate_selection"
        or protocol.get("minimum_worst_active_lower_improvement") != 1.0
        or not isinstance(clauses, list)
        or len(clauses) != 2
        or value.get("claim_boundary") != "fixed_budget_objective_branch_selection_only"
        or value.get("performance_claimed") is not False
    ):
        raise ValueError("NRIR-39 pilot envelope differs")
    for clause in clauses:
        _validate_clause(clause)
    all_go = all(clause["decision"]["go"] for clause in clauses)
    if (
        [clause["original_clause_index"] for clause in clauses]
        != list(EXPECTED_SELECTED)
        or value.get("all_candidate_gates_passed") is not all_go
        or value.get("status") != ("validated-reduced" if all_go else "no_go")
        or value.get("pilot_hash") != _canonical_hash(_semantic_pilot(value))
    ):
        raise ValueError("NRIR-39 pilot decision differs")


def _generate(args: argparse.Namespace) -> None:
    import torch

    from boundflow.runtime.native_objective_branch_shared_evaluator import (
        compile_native_objective_branch_shared_plan,
        execute_native_objective_branch_shared_queue,
    )
    from boundflow.runtime.native_parametric_optimizer import (
        NativeParametricOptimizerTemplateCache,
    )
    from boundflow.runtime.native_shared_parametric_ancestral import (
        execute_native_shared_parametric_ancestral_queue,
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
    branch_policy = NativeObjectiveBranchPolicy()
    floor_program, floor, floor_decision = _execute_floor(
        module,
        input_spec,
        tensors.linear_spec_c,
        tensors.thresholds,
        search_policy,
        optimizer_policy,
        query_id="nrir39:cifar10_resnet:000:property0:pilot",
    )
    if floor_decision.selected_original_clause_indices != EXPECTED_SELECTED:
        raise RuntimeError("NRIR-39 floor priority differs")
    control_cache = NativeParametricOptimizerTemplateCache()
    candidate_cache = NativeParametricOptimizerTemplateCache()
    clauses: list[dict[str, Any]] = []
    for ordinal in EXPECTED_SELECTED:
        source = _source(floor, ordinal)
        objective = tensors.linear_spec_c[:, ordinal : ordinal + 1, :].contiguous()
        threshold = tensors.thresholds[ordinal : ordinal + 1].contiguous()
        plan = compile_native_objective_branch_shared_plan(
            module,
            input_spec,
            linear_spec_C=objective,
            threshold=threshold,
            root_refinement=source.refinement,
            optimizer_policy=optimizer_policy,
            branch_policy=branch_policy,
            plan_id=f"nrir39:cifar10_resnet:000:clause:{ordinal:04d}",
        )
        control = execute_native_shared_parametric_ancestral_queue(
            plan.shared_plan,
            module,
            input_spec,
            linear_spec_C=objective,
            threshold=threshold,
            root_refinement=source.refinement,
            optimizer_policy=optimizer_policy,
            compiler_cache=control_cache,
            query_id=f"nrir39:cifar10_resnet:000:clause:{ordinal:04d}:widest",
            clock_ns=lambda: 0,
        )
        candidate = execute_native_objective_branch_shared_queue(
            plan,
            module,
            input_spec,
            linear_spec_C=objective,
            threshold=threshold,
            root_refinement=source.refinement,
            optimizer_policy=optimizer_policy,
            branch_policy=branch_policy,
            compiler_cache=candidate_cache,
            control_execution=control,
            query_id=f"nrir39:cifar10_resnet:000:clause:{ordinal:04d}:objective",
            clock_ns=lambda: 0,
        )
        clauses.append(_clause_payload(candidate, control, ordinal))
    all_go = all(clause["decision"]["go"] for clause in clauses)
    pilot: dict[str, Any] = {
        "schema_version": PILOT_SCHEMA_VERSION,
        "status": "validated-reduced" if all_go else "no_go",
        "protocol": {
            "selected_original_clause_indices": list(EXPECTED_SELECTED),
            "node_budget": EXPECTED_EVALUATIONS,
            "depth_budget": 4,
            "active_nodes_per_clause": EXPECTED_ACTIVE,
            "optimizer_steps": optimizer_policy.steps,
            "child_refinement_cap": 128,
            "torch_threads": args.torch_threads,
            "branch_policy": branch_policy.to_dict(),
            "control_branch_mode": "widest_unsplit_ambiguous_relu",
            "candidate_branch_mode": "objective_bound_impact",
            "single_changed_variable": "branch_candidate_selection",
            "minimum_worst_active_lower_improvement": 1.0,
            "median_lower_tolerance": 1e-5,
            "logical_fixed_budget_clock": True,
        },
        "workload": _public_workload(workload),
        "source": {
            "native_code_revision": _code_revision(),
            "floor_plan_hash": floor_program.plan.stable_hash(),
            "floor_execution_hash": floor.trace.semantic_signature_hash,
            "selected_original_clause_indices": list(
                floor_decision.selected_original_clause_indices
            ),
        },
        "clauses": clauses,
        "all_candidate_gates_passed": all_go,
        "claim_boundary": "fixed_budget_objective_branch_selection_only",
        "performance_claimed": False,
    }
    pilot["pilot_hash"] = _canonical_hash(_semantic_pilot(pilot))
    validate_pilot(pilot)
    artifact_dir = args.artifact_dir.resolve()
    pilot_path = artifact_dir / "pilot.json"
    _write_json(pilot_path, pilot)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "files": {"pilot.json": _file_sha256(pilot_path)},
        "pilot_hash": _canonical_hash(pilot),
    }
    _write_json(artifact_dir / "manifest.json", manifest)
    print(
        _canonical_json(
            {
                "status": pilot["status"],
                "pilot_hash": pilot["pilot_hash"],
                "decisions": [
                    {
                        "original_clause_index": clause["original_clause_index"],
                        "go": clause["decision"]["go"],
                        "worst_improvement": clause["decision"][
                            "worst_active_lower_improvement"
                        ],
                        "median_delta": clause["decision"]["median_active_lower_delta"],
                    }
                    for clause in clauses
                ],
            }
        )
    )


def _replay(args: argparse.Namespace) -> None:
    artifact_dir = args.artifact_dir.resolve()
    pilot_path = artifact_dir / "pilot.json"
    pilot = _load_json(pilot_path)
    manifest = _load_json(artifact_dir / "manifest.json")
    workloads = _resolve_workloads(args.benchmark_root.resolve())
    workload = next(
        item for item in workloads if item["workload_id"] == "cifar10_resnet:000"
    )
    validate_pilot(pilot)
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or manifest.get("files") != {"pilot.json": _file_sha256(pilot_path)}
        or manifest.get("pilot_hash") != _canonical_hash(pilot)
        or pilot.get("workload") != _public_workload(workload)
    ):
        raise ValueError("NRIR-39 manifest/workload differs")
    print(
        _canonical_json(
            {
                "status": pilot["status"],
                "pilot_hash": pilot["pilot_hash"],
                "performance_claimed": False,
            }
        )
    )


def main() -> None:
    args = _parse_args()
    if args.torch_threads < 1:
        raise ValueError("torch thread count must be positive")
    if args.command == "generate":
        _generate(args)
    else:
        _replay(args)


if __name__ == "__main__":
    main()
