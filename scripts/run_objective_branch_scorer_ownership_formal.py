#!/usr/bin/env python3
"""Generate or replay NRIR-42 scorer-ownership Phase-A evidence."""

# pylint: disable=too-many-lines,too-many-locals,too-many-statements
# pylint: disable=too-many-branches,too-many-boolean-expressions
# pylint: disable=missing-function-docstring,import-outside-toplevel
# pylint: disable=protected-access,duplicate-code
# pylint: disable=too-many-arguments,cell-var-from-loop

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

from boundflow.ir.objective_branch_scorer_ownership_evidence import (
    NativeObjectiveBranchScorerClauseMetricIR,
    NativeObjectiveBranchScorerOwnershipDecisionIR,
    NativeObjectiveBranchScorerOwnershipPlanIR,
    NativeObjectiveBranchScorerOwnershipRowIR,
    NativeObjectiveBranchScorerParityIR,
    lower_native_objective_branch_scorer_ownership_schedule,
)
from boundflow.ir.branch import (
    NativeObjectiveBranchCandidateIR,
    NativeObjectiveBranchPlanIR,
    ObjectiveBranchTaskKind,
)
from boundflow.ir.objective_branch_scorer_ownership import (
    NativeObjectiveBranchScorerScheduleAction,
    NativeObjectiveBranchScorerScheduleIR,
    NativeObjectiveBranchScorerTaskIRModule,
    NativeObjectiveBranchScorerTaskIRUnit,
    NativeValidatedBranchProgramCapsuleIR,
)
from scripts.run_objective_branch_production_cost_attribution import (
    validate_attribution,
)
from scripts.run_typed_hard_clause_escalation_artifact import (
    _load_query_runtime,
    _policies,
    _public_workload,
    _resolve_workloads,
)

FORMAL_SCHEMA_VERSION = "boundflow.objective-branch-scorer-ownership-formal/v1"
WORKER_SCHEMA_VERSION = "boundflow.objective-branch-scorer-ownership-worker/v1"
MANIFEST_SCHEMA_VERSION = "boundflow.objective-branch-scorer-ownership-manifest/v1"
ARTIFACT_DIR = Path(
    "artifacts/objective-branch-scorer-ownership/"
    "vnncomp21-resnet2b-property0-three-repeat-cpu-v1"
)
SOURCE_ATTRIBUTION = Path(
    "artifacts/objective-branch-production-cost-attribution/"
    "vnncomp21-resnet2b-property0-three-repeat-cpu-v1/formal.json"
)
EXPECTED_CLAUSES = (2, 3)
REPEAT_COUNT = 3
TORCH_THREADS = 8
WORKER_TIMEOUT_SECONDS = 300


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("generate", "replay"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--benchmark-root", type=Path, required=True)
        subparser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
        subparser.add_argument("--torch-threads", type=int, default=TORCH_THREADS)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--model", type=Path, required=True)
    worker.add_argument("--property", type=Path, required=True)
    worker.add_argument("--result-json", type=Path, required=True)
    worker.add_argument("--repeat-index", type=int, required=True)
    worker.add_argument("--torch-threads", type=int, required=True)
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


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical_json(value, indent=2) + "\n", encoding="utf-8")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _code_revision() -> str:
    root = _repo_root()
    paths = (
        "boundflow/ir/objective_branch_scorer_ownership.py",
        "boundflow/ir/objective_branch_scorer_ownership_evidence.py",
        "boundflow/runtime/native_prevalidated_objective_branch_score.py",
        "boundflow/runtime/native_prevalidated_objective_branch_shared_evaluator.py",
        "boundflow/runtime/native_prevalidated_objective_branch_shared_production_queue.py",
        "scripts/run_objective_branch_scorer_ownership_formal.py",
    )
    return _canonical_hash({path: _file_sha256(root / path) for path in paths})


def _source_attribution() -> dict[str, Any]:
    value = _load_json(_repo_root() / SOURCE_ATTRIBUTION)
    validate_attribution(value)
    if value.get("decision", {}).get("next_route") != "optimize_scorer_ownership":
        raise ValueError("NRIR-42 source route differs")
    return value


def _plan() -> NativeObjectiveBranchScorerOwnershipPlanIR:
    source = _source_attribution()
    plan = NativeObjectiveBranchScorerOwnershipPlanIR(
        plan_id="nrir42:cifar10_resnet:000:property0:phase-a",
        source_cost_formal_hash=source["formal_hash"],
    )
    plan.validate()
    return plan


def _plan_from_dict(
    value: Mapping[str, Any],
) -> NativeObjectiveBranchScorerOwnershipPlanIR:
    return NativeObjectiveBranchScorerOwnershipPlanIR(
        plan_id=value["plan_id"],
        source_cost_formal_hash=value["source_cost_formal_hash"],
        clause_ordinals=tuple(value["clause_ordinals"]),
        paired_orders=tuple(tuple(item) for item in value["paired_orders"]),
        required_nodes=value["required_nodes"],
        required_sibling_groups=value["required_sibling_groups"],
        historical_enumerations_per_clause=value["historical_enumerations_per_clause"],
        prevalidated_compile_enumerations_per_clause=value[
            "prevalidated_compile_enumerations_per_clause"
        ],
        prevalidated_execute_enumerations_per_clause=value[
            "prevalidated_execute_enumerations_per_clause"
        ],
        maximum_queue_median_ratio=value["maximum_queue_median_ratio"],
        torch_threads=value["torch_threads"],
        semantics_owner=value["semantics_owner"],
        performance_claimed=value["performance_claimed"],
        schema_version=value["schema_version"],
    )


def _row_from_dict(
    value: Mapping[str, Any],
) -> NativeObjectiveBranchScorerOwnershipRowIR:
    return NativeObjectiveBranchScorerOwnershipRowIR(**value)


def _parity_from_dict(value: Mapping[str, Any]) -> NativeObjectiveBranchScorerParityIR:
    return NativeObjectiveBranchScorerParityIR(**value)


def _metric_from_dict(
    value: Mapping[str, Any],
) -> NativeObjectiveBranchScorerClauseMetricIR:
    return NativeObjectiveBranchScorerClauseMetricIR(**value)


def _decision_from_dict(
    value: Mapping[str, Any],
) -> NativeObjectiveBranchScorerOwnershipDecisionIR:
    return NativeObjectiveBranchScorerOwnershipDecisionIR(
        plan_hash=value["plan_hash"],
        clause_metrics=tuple(
            _metric_from_dict(item) for item in value["clause_metrics"]
        ),
        parity_passed=value["parity_passed"],
        enumeration_ownership_passed=value["enumeration_ownership_passed"],
        internal_cost_passed=value["internal_cost_passed"],
        phase_a_go=value["phase_a_go"],
        next_route=value["next_route"],
        reason=value["reason"],
        performance_claimed=value["performance_claimed"],
        schema_version=value["schema_version"],
    )


def _validate_serialized_capsule(
    value: Mapping[str, Any], branch: Mapping[str, Any]
) -> None:
    plan_value = value["plan"]
    plan = NativeObjectiveBranchPlanIR(
        plan_id=plan_value["plan_id"],
        objective_hash=plan_value["objective_hash"],
        split_state_hash=plan_value["split_state_hash"],
        selected_state_hash=plan_value["selected_state_hash"],
        state_scope_hash=plan_value["state_scope_hash"],
        policy_hash=plan_value["policy_hash"],
        candidate_policy_id=plan_value["candidate_policy_id"],
        candidates_per_relu=plan_value["candidates_per_relu"],
        candidate_batch_size=plan_value["candidate_batch_size"],
        max_candidates=plan_value["max_candidates"],
        intermediate_bound_source=plan_value["intermediate_bound_source"],
        candidates=tuple(
            NativeObjectiveBranchCandidateIR(**candidate)
            for candidate in plan_value["candidates"]
        ),
        reduce_policy=plan_value["reduce_policy"],
        schema_version=plan_value["schema_version"],
    )
    task_value = value["task_module"]
    task_module = NativeObjectiveBranchScorerTaskIRModule(
        module_id=task_value["module_id"],
        branch_plan_hash=task_value["branch_plan_hash"],
        tasks=tuple(
            NativeObjectiveBranchScorerTaskIRUnit(
                task_id=task["task_id"],
                kind=ObjectiveBranchTaskKind(task["kind"]),
                input_value_ids=tuple(task["input_value_ids"]),
                output_value_ids=tuple(task["output_value_ids"]),
                dependency_task_ids=tuple(task["dependency_task_ids"]),
                semantics_owner=task["semantics_owner"],
            )
            for task in task_value["tasks"]
        ),
        output_task_id=task_value["output_task_id"],
        schema_version=task_value["schema_version"],
    )
    schedule_value = value["schedule"]
    schedule = NativeObjectiveBranchScorerScheduleIR(
        schedule_id=schedule_value["schedule_id"],
        branch_plan_hash=schedule_value["branch_plan_hash"],
        branch_task_module_hash=schedule_value["branch_task_module_hash"],
        actions=tuple(
            NativeObjectiveBranchScorerScheduleAction(
                action_id=action["action_id"],
                sequence=action["sequence"],
                task_id=action["task_id"],
                input_value_ids=tuple(action["input_value_ids"]),
                output_value_ids=tuple(action["output_value_ids"]),
            )
            for action in schedule_value["actions"]
        ),
        selected_candidate_value_id=schedule_value["selected_candidate_value_id"],
        schema_version=schedule_value["schema_version"],
    )
    capsule = NativeValidatedBranchProgramCapsuleIR(**value["capsule"])
    capsule.validate(plan=plan, task_module=task_module, schedule=schedule)
    if (
        plan_value.get("performance_claimed") is not False
        or plan_value["candidates"] != branch["candidates"]
        or value["candidate_table_hash"] != capsule.candidate_table_hash
        or capsule.candidate_count != len(branch["candidates"])
    ):
        raise ValueError("NRIR-42 serialized capsule semantic binding differs")


def _queue_semantics(execution: Any) -> dict[str, object]:
    trace = execution.queue.trace
    evaluations = []
    for item in trace.evaluations:
        row = item.to_dict()
        row.pop("batch_trace_hash")
        evaluations.append(row)
    return {
        "status": trace.status,
        "termination_reason": trace.termination_reason,
        "config": trace.config.to_dict(),
        "optimizer_policy": trace.optimizer_policy.to_dict(),
        "intermediate_bound_source": trace.intermediate_bound_source.value,
        "root_input_lower_hash": trace.root_input_lower_hash,
        "root_input_upper_hash": trace.root_input_upper_hash,
        "objective_hash": trace.objective_hash,
        "evaluations": evaluations,
        "decisions": [item.to_dict() for item in trace.decisions],
        "final_frontier_node_ids": list(trace.final_frontier_node_ids),
        "max_queue_size": trace.max_queue_size,
        "compiler_semantics": [
            {
                "template_hash": item.template_hash,
                "task_hash": item.task_hash,
                "schedule_hash": item.schedule_hash,
                "cache_outcome": item.cache_event.outcome,
                "instance_hash": item.instance_ir.stable_hash(),
            }
            for item in execution.compiler_batches
        ],
    }


def _branch_semantics(execution: Any) -> list[dict[str, object]]:
    result = []
    for node_id, branch in execution.queue.objective_branch_executions:
        result.append(
            {
                "node_id": node_id,
                "branch": branch.branch.to_dict(),
                "candidates": [
                    candidate.to_dict() for candidate in branch.program.plan.candidates
                ],
                "scores": [score.to_dict() for score in branch.trace.scores],
                "child_lower_hash": branch.trace.child_lower_hash,
                "score_hash": branch.trace.score_hash,
                "selected_candidate_ordinal": (branch.trace.selected_candidate_ordinal),
            }
        )
    return result


def _state_semantics(execution: Any) -> list[dict[str, object]]:
    return [
        {"node_id": node_id, "state": state.to_dict()}
        for node_id, state in execution.queue.selected_states
    ]


def _refinement_semantics(execution: Any) -> list[dict[str, object]]:
    return [item.semantic_dict() for item in execution.node_refinements]


def _capsule_table(execution: Any, *, mode: str) -> list[dict[str, object]]:
    if mode == "historical":
        return [
            {
                "node_id": node_id,
                "candidate_table_hash": _canonical_hash(
                    [item.to_dict() for item in branch.program.plan.candidates]
                ),
                "capsule": None,
            }
            for node_id, branch in execution.queue.objective_branch_executions
        ]
    result = []
    for node_id, branch in execution.queue.objective_branch_executions:
        program = branch.program
        result.append(
            {
                "node_id": node_id,
                "candidate_table_hash": program.capsule.candidate_table_hash,
                "capsule": program.capsule.to_dict(
                    plan=program.plan,
                    task_module=program.task_module,
                    schedule=program.schedule,
                ),
                "plan": program.plan.to_dict(),
                "task_module": program.task_module.to_dict(plan=program.plan),
                "schedule": program.schedule.to_dict(
                    plan=program.plan, task_module=program.task_module
                ),
            }
        )
    return result


def _execute_mode(
    *,
    mode: str,
    plan: Any,
    module: Any,
    input_spec: Any,
    objective: Any,
    threshold: Any,
    refinement: Any,
    optimizer_policy: Any,
    branch_policy: Any,
    query_id: str,
) -> tuple[Any, int]:
    import boundflow.runtime.native_objective_branch_score as historical_scorer
    import boundflow.runtime.native_prevalidated_objective_branch_score as new_scorer
    from boundflow.runtime.native_objective_branch_shared_production_queue import (
        execute_native_objective_branch_shared_production_queue,
    )
    from boundflow.runtime.native_parametric_optimizer import (
        NativeParametricOptimizerTemplateCache,
    )
    from boundflow.runtime.native_prevalidated_objective_branch_shared_production_queue import (
        execute_native_prevalidated_objective_branch_shared_production_queue,
    )

    scorer = historical_scorer if mode == "historical" else new_scorer
    original = getattr(scorer, "_enumerate_candidates")
    calls = 0

    def counted(*args: Any, **kwargs: Any):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    setattr(scorer, "_enumerate_candidates", counted)
    execute = (
        execute_native_objective_branch_shared_production_queue
        if mode == "historical"
        else execute_native_prevalidated_objective_branch_shared_production_queue
    )
    try:
        execution = execute(
            plan,
            module,
            input_spec,
            linear_spec_C=objective,
            threshold=threshold,
            root_refinement=refinement,
            optimizer_policy=optimizer_policy,
            branch_policy=branch_policy,
            compiler_cache=NativeParametricOptimizerTemplateCache(),
            query_id=query_id,
        )
    finally:
        setattr(scorer, "_enumerate_candidates", original)
    return execution, calls


def _run_worker(args: argparse.Namespace) -> None:
    import torch

    from boundflow.runtime.native_objective_branch_score import (
        NativeObjectiveBranchPolicy,
    )
    from boundflow.runtime.native_objective_branch_shared_evaluator import (
        compile_native_objective_branch_shared_plan,
    )
    from scripts.run_objective_branch_shared_evaluator_pilot import (
        _execute_floor,
        _source,
    )

    torch.set_num_threads(args.torch_threads)
    plan = _plan()
    _query, tensors, module, input_spec = _load_query_runtime(
        args.model.resolve(), args.property.resolve(), "cifar10_resnet:000"
    )
    search_policy, optimizer_policy = _policies()
    branch_policy = NativeObjectiveBranchPolicy()
    _floor_program, floor, floor_decision = _execute_floor(
        module,
        input_spec,
        tensors.linear_spec_c,
        tensors.thresholds,
        search_policy,
        optimizer_policy,
        query_id=f"nrir42:repeat{args.repeat_index}:floor",
    )
    if floor_decision.selected_original_clause_indices != EXPECTED_CLAUSES:
        raise RuntimeError("NRIR-42 floor selection differs")
    raw_runs: list[dict[str, Any]] = []
    parities: list[dict[str, Any]] = []
    for ordinal in plan.clause_ordinals:
        source = _source(floor, ordinal)
        objective = tensors.linear_spec_c[:, ordinal : ordinal + 1, :].contiguous()
        threshold = tensors.thresholds[ordinal : ordinal + 1].contiguous()
        composite = compile_native_objective_branch_shared_plan(
            module,
            input_spec,
            linear_spec_C=objective,
            threshold=threshold,
            root_refinement=source.refinement,
            optimizer_policy=optimizer_policy,
            branch_policy=branch_policy,
            plan_id=f"nrir42:r{args.repeat_index}:clause:{ordinal}",
        )
        by_mode: dict[
            str, tuple[NativeObjectiveBranchScorerOwnershipRowIR, dict[str, Any]]
        ] = {}
        for position, mode in enumerate(plan.paired_orders[args.repeat_index]):
            execution, calls = _execute_mode(
                mode=mode,
                plan=composite,
                module=module,
                input_spec=input_spec,
                objective=objective,
                threshold=threshold,
                refinement=source.refinement,
                optimizer_policy=optimizer_policy,
                branch_policy=branch_policy,
                query_id=f"nrir42:r{args.repeat_index}:c{ordinal}",
            )
            queue = _queue_semantics(execution)
            branches = _branch_semantics(execution)
            states = _state_semantics(execution)
            refinements = _refinement_semantics(execution)
            capsules = _capsule_table(execution, mode=mode)
            row = NativeObjectiveBranchScorerOwnershipRowIR(
                plan_hash=plan.stable_hash(),
                repeat_index=args.repeat_index,
                original_clause_index=ordinal,
                mode=mode,
                order_position=position,
                queue_elapsed_ns=execution.trace.queue_elapsed_ns,
                whole_elapsed_ns=execution.trace.whole_elapsed_ns,
                accepted_nodes=len(execution.queue.trace.evaluations),
                sibling_group_count=len(execution.batch_commits) - 1,
                branch_execution_count=len(execution.queue.objective_branch_executions),
                enumeration_call_count=calls,
                compile_enumeration_count=(31 if mode == "prevalidated" else 0),
                execute_enumeration_count=0,
                queue_semantic_hash=_canonical_hash(queue),
                branch_semantic_hash=_canonical_hash(branches),
                capsule_table_hash=_canonical_hash(capsules),
            )
            row.validate()
            raw: dict[str, Any] = {
                "row": row.to_dict(),
                "row_hash": row.stable_hash(),
                "queue_semantics": queue,
                "branch_semantics": branches,
                "selected_states": states,
                "refinements": refinements,
                "capsules": capsules,
            }
            raw["raw_hash"] = _canonical_hash(raw)
            raw_runs.append(raw)
            by_mode[mode] = (row, raw)
        historical_row, historical = by_mode["historical"]
        candidate_row, candidate = by_mode["prevalidated"]
        exact = all(
            historical[key] == candidate[key]
            for key in (
                "queue_semantics",
                "branch_semantics",
                "selected_states",
                "refinements",
            )
        )
        if not exact:
            raise ValueError("NRIR-42 old/new production semantics differ")
        parity = NativeObjectiveBranchScorerParityIR(
            plan_hash=plan.stable_hash(),
            repeat_index=args.repeat_index,
            original_clause_index=ordinal,
            historical_row_hash=historical_row.stable_hash(),
            prevalidated_row_hash=candidate_row.stable_hash(),
            queue_semantic_hash=historical_row.queue_semantic_hash,
            branch_semantic_hash=historical_row.branch_semantic_hash,
            selected_state_hash=_canonical_hash(historical["selected_states"]),
            refinement_semantic_hash=_canonical_hash(historical["refinements"]),
            exact=exact,
        )
        parity.validate()
        parities.append(parity.to_dict())
    worker = {
        "schema_version": WORKER_SCHEMA_VERSION,
        "repeat_index": args.repeat_index,
        "plan_hash": plan.stable_hash(),
        "paired_order": list(plan.paired_orders[args.repeat_index]),
        "selected_original_clause_indices": list(EXPECTED_CLAUSES),
        "raw_runs": raw_runs,
        "parities": parities,
        "performance_claimed": False,
    }
    worker["worker_hash"] = _canonical_hash(worker)
    _write_json(args.result_json, worker)
    print(
        _canonical_json(
            {
                "repeat_index": args.repeat_index,
                "queue_elapsed_ns": {
                    f"c{raw['row']['original_clause_index']}:{raw['row']['mode']}": raw[
                        "row"
                    ]["queue_elapsed_ns"]
                    for raw in raw_runs
                },
                "worker_hash": worker["worker_hash"],
            }
        )
    )


def _validate_worker(
    value: Mapping[str, Any], plan: NativeObjectiveBranchScorerOwnershipPlanIR
) -> None:
    repeat = value.get("repeat_index")
    if (
        value.get("schema_version") != WORKER_SCHEMA_VERSION
        or repeat not in {0, 1, 2}
        or value.get("plan_hash") != plan.stable_hash()
        or value.get("paired_order") != list(plan.paired_orders[repeat])
        or value.get("selected_original_clause_indices") != list(EXPECTED_CLAUSES)
        or value.get("performance_claimed") is not False
    ):
        raise ValueError("NRIR-42 worker header differs")
    raw_runs = value.get("raw_runs")
    parities = value.get("parities")
    if not isinstance(raw_runs, list) or len(raw_runs) != 4:
        raise ValueError("NRIR-42 worker run coverage differs")
    by_key: dict[
        tuple[int, str],
        tuple[NativeObjectiveBranchScorerOwnershipRowIR, Mapping[str, Any]],
    ] = {}
    for raw in raw_runs:
        if not isinstance(raw, dict):
            raise TypeError("NRIR-42 raw run differs")
        expected_raw_hash = _canonical_hash(
            {key: item for key, item in raw.items() if key != "raw_hash"}
        )
        if raw.get("raw_hash") != expected_raw_hash:
            raise ValueError("NRIR-42 raw run hash differs")
        row = _row_from_dict(raw["row"])
        row.validate()
        if (
            row.stable_hash() != raw.get("row_hash")
            or row.plan_hash != plan.stable_hash()
            or row.repeat_index != repeat
            or row.queue_semantic_hash != _canonical_hash(raw["queue_semantics"])
            or row.branch_semantic_hash != _canonical_hash(raw["branch_semantics"])
            or row.capsule_table_hash != _canonical_hash(raw["capsules"])
        ):
            raise ValueError("NRIR-42 raw typed binding differs")
        branch_rows = raw["branch_semantics"]
        if len(branch_rows) != 31 or len(raw["capsules"]) != 31:
            raise ValueError("NRIR-42 branch/capsule coverage differs")
        for branch in branch_rows:
            scores = branch["scores"]
            selected = branch["selected_candidate_ordinal"]
            candidates = branch["candidates"]
            expected = min(
                scores,
                key=lambda score: (
                    -score["worst_child_lower"],
                    -score["mean_child_lower"],
                    candidates[score["candidate_ordinal"]]["relu_input"],
                    candidates[score["candidate_ordinal"]]["neuron_index"],
                ),
            )["candidate_ordinal"]
            for candidate in candidates:
                if (
                    not math.isfinite(candidate["lower"])
                    or not math.isfinite(candidate["upper"])
                    or not candidate["lower"] < 0.0 < candidate["upper"]
                    or not math.isclose(
                        candidate["width"],
                        candidate["upper"] - candidate["lower"],
                        rel_tol=1e-6,
                        abs_tol=1e-6,
                    )
                ):
                    raise ValueError("NRIR-42 candidate semantic replay differs")
            for score in scores:
                inactive = score["inactive_lower"]
                active = score["active_lower"]
                if (
                    not all(
                        math.isfinite(number)
                        for number in (
                            inactive,
                            active,
                            score["worst_child_lower"],
                            score["mean_child_lower"],
                        )
                    )
                    or not math.isclose(
                        score["worst_child_lower"],
                        min(inactive, active),
                        rel_tol=1e-6,
                        abs_tol=1e-6,
                    )
                    or not math.isclose(
                        score["mean_child_lower"],
                        (inactive + active) / 2.0,
                        rel_tol=1e-6,
                        abs_tol=1e-6,
                    )
                ):
                    raise ValueError("NRIR-42 score semantic replay differs")
            selected_candidate = candidates[selected]
            if (
                selected != expected
                or branch["score_hash"] != _canonical_hash(scores)
                or branch["branch"]["relu_input"] != selected_candidate["relu_input"]
                or branch["branch"]["neuron_index"]
                != selected_candidate["neuron_index"]
                or branch["branch"]["lower"] != selected_candidate["lower"]
                or branch["branch"]["upper"] != selected_candidate["upper"]
                or branch["branch"]["width"] != selected_candidate["width"]
            ):
                raise ValueError("NRIR-42 branch semantic replay differs")
        branch_by_id = {branch["node_id"]: branch for branch in branch_rows}
        for capsule in raw["capsules"]:
            branch = branch_by_id.get(capsule["node_id"])
            if branch is None:
                raise ValueError("NRIR-42 capsule node coverage differs")
            if row.mode == "historical":
                if capsule["capsule"] is not None or capsule[
                    "candidate_table_hash"
                ] != _canonical_hash(branch["candidates"]):
                    raise ValueError("NRIR-42 historical table replay differs")
            else:
                _validate_serialized_capsule(capsule, branch)
        by_key[(row.original_clause_index, row.mode)] = (row, raw)
    if set(by_key) != {
        (ordinal, mode)
        for ordinal in plan.clause_ordinals
        for mode in ("historical", "prevalidated")
    }:
        raise ValueError("NRIR-42 worker key coverage differs")
    if not isinstance(parities, list) or len(parities) != 2:
        raise ValueError("NRIR-42 parity coverage differs")
    for value_parity in parities:
        parity = _parity_from_dict(value_parity)
        parity.validate()
        historical_row, historical = by_key[
            (parity.original_clause_index, "historical")
        ]
        candidate_row, candidate = by_key[
            (parity.original_clause_index, "prevalidated")
        ]
        if (
            historical_row.stable_hash() != parity.historical_row_hash
            or candidate_row.stable_hash() != parity.prevalidated_row_hash
            or any(
                historical[key] != candidate[key]
                for key in (
                    "queue_semantics",
                    "branch_semantics",
                    "selected_states",
                    "refinements",
                )
            )
            or parity.queue_semantic_hash != historical_row.queue_semantic_hash
            or parity.branch_semantic_hash != historical_row.branch_semantic_hash
            or parity.selected_state_hash
            != _canonical_hash(historical["selected_states"])
            or parity.refinement_semantic_hash
            != _canonical_hash(historical["refinements"])
        ):
            raise ValueError("NRIR-42 exact parity replay differs")
    expected_hash = _canonical_hash(
        {key: item for key, item in value.items() if key != "worker_hash"}
    )
    if value.get("worker_hash") != expected_hash:
        raise ValueError("NRIR-42 worker hash differs")


def _median_mad(values: list[int]) -> tuple[int, int]:
    median = int(statistics.median(values))
    mad = int(statistics.median(abs(value - median) for value in values))
    return median, mad


def _derive_decision(
    plan: NativeObjectiveBranchScorerOwnershipPlanIR,
    workers: Sequence[Mapping[str, Any]],
) -> NativeObjectiveBranchScorerOwnershipDecisionIR:
    metrics = []
    for ordinal in plan.clause_ordinals:
        by_mode: dict[str, list[int]] = {
            "historical": [],
            "prevalidated": [],
        }
        for worker in workers:
            for raw in worker["raw_runs"]:
                row = _row_from_dict(raw["row"])
                if row.original_clause_index == ordinal:
                    by_mode[row.mode].append(row.queue_elapsed_ns)
        historical_median, historical_mad = _median_mad(by_mode["historical"])
        candidate_median, candidate_mad = _median_mad(by_mode["prevalidated"])
        improvement = historical_median - candidate_median
        metric = NativeObjectiveBranchScorerClauseMetricIR(
            original_clause_index=ordinal,
            historical_queue_median_ns=historical_median,
            prevalidated_queue_median_ns=candidate_median,
            historical_queue_mad_ns=historical_mad,
            prevalidated_queue_mad_ns=candidate_mad,
            median_ratio=candidate_median / historical_median,
            median_improvement_ns=improvement,
            ratio_passed=(candidate_median / historical_median) <= 0.75,
            mad_passed=improvement > max(historical_mad, candidate_mad),
        )
        metric.validate()
        metrics.append(metric)
    cost = all(item.ratio_passed and item.mad_passed for item in metrics)
    decision = NativeObjectiveBranchScorerOwnershipDecisionIR(
        plan_hash=plan.stable_hash(),
        clause_metrics=tuple(metrics),
        parity_passed=True,
        enumeration_ownership_passed=True,
        internal_cost_passed=cost,
        phase_a_go=cost,
        next_route="run_phase_b_global_60s" if cost else "close_nrir42_no_go",
        reason=(
            "exact_parity_31_compile_0_execute_and_cost_gate_passed"
            if cost
            else "exact_parity_and_ownership_passed_but_cost_gate_failed"
        ),
    )
    decision.validate()
    return decision


def _formal_semantic(value: Mapping[str, Any]) -> dict[str, object]:
    return {
        key: value[key]
        for key in (
            "schema_version",
            "source",
            "plan",
            "plan_hash",
            "workers",
            "paired_rows",
            "parities",
            "capsule_table_hash",
            "decision",
            "decision_hash",
            "task_ir",
            "task_ir_hash",
            "schedule",
            "schedule_hash",
            "performance_claimed",
        )
    }


def validate_formal(value: Mapping[str, Any]) -> None:
    plan = _plan_from_dict(value["plan"])
    plan.validate()
    source = _source_attribution()
    workers = value.get("workers")
    if not isinstance(workers, list) or len(workers) != REPEAT_COUNT:
        raise ValueError("NRIR-42 formal worker coverage differs")
    for worker in workers:
        _validate_worker(worker, plan)
    rows = [raw["row"] for worker in workers for raw in worker["raw_runs"]]
    parities = [item for worker in workers for item in worker["parities"]]
    capsules = [
        item
        for worker in workers
        for raw in worker["raw_runs"]
        if raw["row"]["mode"] == "prevalidated"
        for item in raw["capsules"]
    ]
    expected_decision = _derive_decision(plan, workers)
    decision = _decision_from_dict(value["decision"])
    decision.validate()
    capsule_table_hash = _canonical_hash(capsules)
    task_ir, schedule = lower_native_objective_branch_scorer_ownership_schedule(
        plan,
        capsule_table_hash=capsule_table_hash,
        paired_rows_hash=_canonical_hash(rows),
        parity_rows_hash=_canonical_hash(parities),
        metrics_hash=_canonical_hash(
            [item.to_dict() for item in expected_decision.clause_metrics]
        ),
        decision_hash=expected_decision.stable_hash(),
    )
    if (
        value.get("schema_version") != FORMAL_SCHEMA_VERSION
        or value.get("source", {}).get("source_cost_formal_hash")
        != source["formal_hash"]
        or value.get("source", {}).get("native_code_revision") != _code_revision()
        or plan.source_cost_formal_hash != source["formal_hash"]
        or value.get("plan_hash") != plan.stable_hash()
        or value.get("paired_rows") != rows
        or value.get("parities") != parities
        or value.get("capsule_table_hash") != capsule_table_hash
        or decision != expected_decision
        or value.get("decision_hash") != decision.stable_hash()
        or value.get("task_ir") != task_ir.to_dict()
        or value.get("task_ir_hash") != task_ir.stable_hash()
        or value.get("schedule") != schedule.to_dict(task_ir)
        or value.get("schedule_hash") != schedule.stable_hash(task_ir)
        or value.get("performance_claimed") is not False
        or value.get("formal_hash") != _canonical_hash(_formal_semantic(value))
    ):
        raise ValueError("NRIR-42 formal derived evidence differs")


def _worker_command(
    workload: Mapping[str, object], result: Path, repeat: int, threads: int
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--model",
        str(workload["model"]),
        "--property",
        str(workload["property"]),
        "--result-json",
        str(result),
        "--repeat-index",
        str(repeat),
        "--torch-threads",
        str(threads),
    ]


def _run_subprocess(command: list[str], log_path: Path) -> None:
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = "0"
    completed = subprocess.run(
        command,
        cwd=_repo_root(),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=WORKER_TIMEOUT_SECONDS,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"NRIR-42 worker failed with {completed.returncode}: "
            f"{completed.stdout[-12000:]}"
        )
    print(completed.stdout.strip())


def _generate(args: argparse.Namespace) -> None:
    workloads = _resolve_workloads(args.benchmark_root.resolve())
    workload = next(
        item for item in workloads if item["workload_id"] == "cifar10_resnet:000"
    )
    plan = _plan()
    artifact_dir = args.artifact_dir.resolve()
    workers: list[dict[str, Any]] = []
    files: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="boundflow-nrir42-") as temporary:
        temporary_root = Path(temporary)
        for repeat in range(REPEAT_COUNT):
            result = temporary_root / f"repeat-{repeat}.json"
            log = artifact_dir / "logs" / f"repeat-{repeat}.log"
            _run_subprocess(
                _worker_command(workload, result, repeat, args.torch_threads), log
            )
            worker = _load_json(result)
            _validate_worker(worker, plan)
            shard = artifact_dir / "shards" / f"repeat-{repeat}.json"
            _write_json(shard, worker)
            workers.append(worker)
            files[str(log.relative_to(artifact_dir))] = _file_sha256(log)
            files[str(shard.relative_to(artifact_dir))] = _file_sha256(shard)
    decision = _derive_decision(plan, workers)
    rows = [raw["row"] for worker in workers for raw in worker["raw_runs"]]
    parities = [item for worker in workers for item in worker["parities"]]
    capsules = [
        item
        for worker in workers
        for raw in worker["raw_runs"]
        if raw["row"]["mode"] == "prevalidated"
        for item in raw["capsules"]
    ]
    capsule_table_hash = _canonical_hash(capsules)
    task_ir, schedule = lower_native_objective_branch_scorer_ownership_schedule(
        plan,
        capsule_table_hash=capsule_table_hash,
        paired_rows_hash=_canonical_hash(rows),
        parity_rows_hash=_canonical_hash(parities),
        metrics_hash=_canonical_hash(
            [item.to_dict() for item in decision.clause_metrics]
        ),
        decision_hash=decision.stable_hash(),
    )
    formal: dict[str, Any] = {
        "schema_version": FORMAL_SCHEMA_VERSION,
        "source": {
            "workload": _public_workload(workload),
            "source_cost_formal_hash": plan.source_cost_formal_hash,
            "native_code_revision": _code_revision(),
        },
        "plan": plan.to_dict(),
        "plan_hash": plan.stable_hash(),
        "workers": workers,
        "paired_rows": rows,
        "parities": parities,
        "capsule_table_hash": capsule_table_hash,
        "decision": decision.to_dict(),
        "decision_hash": decision.stable_hash(),
        "task_ir": task_ir.to_dict(),
        "task_ir_hash": task_ir.stable_hash(),
        "schedule": schedule.to_dict(task_ir),
        "schedule_hash": schedule.stable_hash(task_ir),
        "performance_claimed": False,
    }
    formal["formal_hash"] = _canonical_hash(_formal_semantic(formal))
    validate_formal(formal)
    formal_path = artifact_dir / "formal.json"
    _write_json(formal_path, formal)
    files["formal.json"] = _file_sha256(formal_path)
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source": formal["source"],
        "plan_hash": plan.stable_hash(),
        "formal_hash": formal["formal_hash"],
        "files": dict(sorted(files.items())),
        "performance_claimed": False,
    }
    manifest["manifest_hash"] = _canonical_hash(manifest)
    _write_json(artifact_dir / "manifest.json", manifest)
    print(
        _canonical_json(
            {
                "status": "generated",
                "formal_hash": formal["formal_hash"],
                "decision": decision.to_dict(),
            }
        )
    )


def _replay(args: argparse.Namespace) -> None:
    artifact_dir = args.artifact_dir.resolve()
    manifest = _load_json(artifact_dir / "manifest.json")
    expected_manifest_hash = _canonical_hash(
        {key: item for key, item in manifest.items() if key != "manifest_hash"}
    )
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or manifest.get("manifest_hash") != expected_manifest_hash
        or manifest.get("performance_claimed") is not False
    ):
        raise ValueError("NRIR-42 manifest differs")
    for relative, digest in manifest["files"].items():
        if _file_sha256(artifact_dir / relative) != digest:
            raise ValueError(f"NRIR-42 artifact digest differs: {relative}")
    workloads = _resolve_workloads(args.benchmark_root.resolve())
    workload = next(
        item for item in workloads if item["workload_id"] == "cifar10_resnet:000"
    )
    formal = _load_json(artifact_dir / "formal.json")
    if formal.get("source", {}).get("workload") != _public_workload(workload):
        raise ValueError("NRIR-42 replay workload differs")
    validate_formal(formal)
    if (
        manifest.get("formal_hash") != formal["formal_hash"]
        or manifest.get("plan_hash") != formal["plan_hash"]
        or manifest.get("source") != formal["source"]
    ):
        raise ValueError("NRIR-42 manifest/formal binding differs")
    print(
        _canonical_json(
            {
                "status": "replayed",
                "formal_hash": formal["formal_hash"],
                "phase_a_go": formal["decision"]["phase_a_go"],
                "next_route": formal["decision"]["next_route"],
            }
        )
    )


def main() -> None:
    args = _parse_args()
    if args.command == "worker":
        _run_worker(args)
    elif args.command == "generate":
        _generate(args)
    elif args.command == "replay":
        _replay(args)
    else:
        raise AssertionError("unreachable NRIR-42 command")


if __name__ == "__main__":
    main()
