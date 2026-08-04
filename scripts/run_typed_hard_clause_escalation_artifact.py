#!/usr/bin/env python3
"""Generate or replay the NRIR-30 typed hard-clause escalation artifact."""

# pylint: disable=too-many-lines,too-many-locals,too-many-statements
# pylint: disable=too-many-branches,too-many-arguments,too-many-boolean-expressions
# pylint: disable=missing-function-docstring,import-outside-toplevel,duplicate-code
# pylint: disable=protected-access,import-error,line-too-long

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Optional, Sequence

from scripts.run_multiworkload_competitor_e2e_artifact import (
    VNNCOMP_COMMIT,
    WORKLOAD_ROWS,
    _csv_selection,
    _git_revision,
    _onnx_inventory,
)
from scripts.run_wall_clock_parametric_bab_scaling_artifact import (
    _compiler_evidence,
    _validate_compiler,
)

ARTIFACT_SCHEMA_VERSION = "boundflow.typed-hard-clause-escalation-artifact/v1"
EVIDENCE_SCHEMA_VERSION = "boundflow.typed-hard-clause-escalation-evidence/v1"
WORKER_SCHEMA_VERSION = "boundflow.typed-hard-clause-escalation-worker/v1"
ARTIFACT_DIR = Path(
    "artifacts/typed-hard-clause-escalation/vnncomp21-three-topology-cpu-v1"
)
PREDECESSOR_DIR = Path(
    "artifacts/wall-clock-parametric-bab-scaling/vnncomp21-three-topology-cpu-v1"
)
PREDECESSOR_EVIDENCE_HASH = (
    "e01d35c0afa8501f3d02ffaaa4eeaf609c444ed497c1a2d2efff4e97b3520214"
)
MANIFEST_FILE = "manifest.json"
EVIDENCE_FILE = "evidence.json"
REPEATS = 3
TORCH_THREADS = 8
TIMEOUT_SECONDS = 60
SEARCH_STEPS = 4
OPTIMIZER_STEPS = 5


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("generate", "replay"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--benchmark-root", type=Path, required=True)
        subparser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
        subparser.add_argument("--torch-threads", type=int, default=TORCH_THREADS)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--workload-id", required=True)
    worker.add_argument("--model", type=Path, required=True)
    worker.add_argument("--property", type=Path, required=True)
    worker.add_argument("--result-json", type=Path, required=True)
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


def canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
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


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    return value


def _list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be an array")
    return value


def _sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _code_revision() -> str:
    root = _repo_root()
    paths = (
        "boundflow/ir/hard_clause_escalation.py",
        "boundflow/runtime/native_hard_clause_escalation.py",
        "boundflow/ir/refinement.py",
        "boundflow/runtime/native_intermediate_refinement.py",
        "boundflow/ir/parametric_optimizer.py",
        "boundflow/runtime/native_parametric_optimizer.py",
        "boundflow/runtime/native_parametric_production_verifier.py",
        "boundflow/runtime/native_parametric_production_complete_query.py",
        "scripts/run_typed_hard_clause_escalation_artifact.py",
    )
    return canonical_hash({path: file_sha256(root / path) for path in paths})


def _resolve_workloads(benchmark_root: Path) -> list[dict[str, object]]:
    from boundflow.frontends.vnnlib import import_vnnlib_box_query

    if _git_revision(benchmark_root) != VNNCOMP_COMMIT:
        raise ValueError("NRIR-30 VNN-COMP commit differs")
    resolved = []
    for definition in WORKLOAD_ROWS:
        csv_path, model_path, property_path = _csv_selection(benchmark_root, definition)
        workload_id = str(definition["workload_id"])
        query = import_vnnlib_box_query(property_path, query_id=workload_id)
        input_shape, output_dim, ops = _onnx_inventory(model_path)
        if (
            len(query.input_names) != math.prod(input_shape[1:])
            or len(query.output_names) != output_dim
        ):
            raise ValueError("NRIR-30 ONNX/VNNLIB dimensions differ")
        resolved.append(
            {
                "workload_id": workload_id,
                "category": str(definition["category"]),
                "csv_ordinal": int(definition["csv_ordinal"]),
                "csv_relative_path": str(definition["csv"]),
                "model_relative_path": str(definition["model"]),
                "property_relative_path": str(definition["property"]),
                "csv_sha256": file_sha256(csv_path),
                "model_sha256": file_sha256(model_path),
                "property_sha256": file_sha256(property_path),
                "query_ir_hash": query.stable_hash(),
                "model_input_shape": list(input_shape),
                "model_output_dim": output_dim,
                "onnx_ops": list(ops),
                "model": model_path,
                "property": property_path,
            }
        )
    return resolved


def _public_workload(workload: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in workload.items()
        if key not in {"model", "property"}
    }


def _load_query_runtime(model: Path, property_path: Path, workload_id: str):
    from boundflow.frontends.onnx.frontend import import_onnx
    from boundflow.frontends.vnnlib import (
        import_vnnlib_box_query,
        materialize_vnnlib_box_query,
    )
    from boundflow.planner import plan_interval_ibp_v0
    from boundflow.runtime.task_executor import InputSpec

    query = import_vnnlib_box_query(property_path, query_id=workload_id)
    input_shape, output_dim, _ops = _onnx_inventory(model)
    if len(query.output_names) != output_dim:
        raise ValueError("NRIR-30 worker output dimension differs")
    tensors = materialize_vnnlib_box_query(query, input_shape=input_shape[1:])
    primal = import_onnx(str(model), do_shape_infer=True, normalize=True)
    module = plan_interval_ibp_v0(primal)
    input_spec = InputSpec.box(
        value_name=primal.graph.inputs[0],
        lower=tensors.input_lower,
        upper=tensors.input_upper,
    )
    return query, tensors, module, input_spec


def _policies():
    from boundflow.runtime.native_alpha_beta_optimization_state import (
        NativeAlphaBetaOptimizerPolicy,
    )
    from boundflow.runtime.native_candidate_search import (
        NativeProjectedGradientSearchPolicy,
    )

    return (
        NativeProjectedGradientSearchPolicy(steps=SEARCH_STEPS, step_size=0.002),
        NativeAlphaBetaOptimizerPolicy(
            steps=OPTIMIZER_STEPS,
            lr=0.1,
            alpha_initialization_mode="adaptive",
        ),
    )


def _query_semantics(
    execution: Any, *, original_ordinals: Optional[Sequence[int]] = None
) -> dict[str, object]:
    ordinal_map = (
        tuple(range(len(execution.trace.thresholds)))
        if original_ordinals is None
        else tuple(original_ordinals)
    )
    clauses = []
    semantic_clauses = []
    for clause in execution.clauses:
        queue = clause.queue.trace
        projected = clause.trace.clause_index
        original = ordinal_map[projected]
        semantic = {
            "projected_clause_index": projected,
            "original_clause_index": original,
            "status": clause.trace.status,
            "root_lower": queue.evaluations[0].lower,
            "root_upper": queue.evaluations[0].upper,
            "queue_status": queue.status,
            "evaluated_nodes": len(queue.evaluations),
            "domains": [
                {
                    "split_state_hash": item.node.split_state_hash,
                    "depth": item.node.depth,
                    "lower": item.lower,
                    "upper": item.upper,
                }
                for item in queue.evaluations
            ],
        }
        clauses.append(
            {
                **semantic,
                "queue_trace_hash": queue.stable_hash(),
                "verdict_trace_hash": clause.verdict.trace.stable_hash(queue),
            }
        )
        semantic_clauses.append(semantic)
    completed_original = [
        ordinal_map[item.clause_index] for item in execution.trace.completed_clauses
    ]
    unresolved_original = [
        ordinal_map[index] for index in execution.trace.unresolved_clause_indices
    ]
    pending_original = [
        ordinal_map[index] for index in execution.trace.pending_clause_indices
    ]
    verified_original = [
        ordinal_map[item.trace.clause_index]
        for item in execution.clauses
        if item.trace.status == "verified"
    ]
    semantic = {
        "solver_status": execution.trace.status,
        "completed_original_clause_indices": completed_original,
        "verified_original_clause_indices": verified_original,
        "unresolved_original_clause_indices": unresolved_original,
        "pending_original_clause_indices": pending_original,
        "clauses": semantic_clauses,
    }
    return {
        **semantic,
        "clauses": clauses,
        "query_trace_hash": execution.trace.stable_hash(),
        "semantic_signature_hash": canonical_hash(semantic),
        "compiler": _compiler_evidence(execution),
    }


def _refinement_evidence(execution: Any) -> Optional[dict[str, object]]:
    from boundflow.runtime.native_intermediate_refinement import (
        intermediate_refinement_semantic_trace_hash,
    )

    if execution.refinement is None or execution.refinement_program is None:
        return None
    program = execution.refinement_program
    refinement = execution.refinement
    return {
        "plan": program.plan.to_dict(),
        "plan_hash": program.plan.stable_hash(),
        "task_ir": program.task_module.to_dict(plan=program.plan),
        "task_ir_hash": program.task_module.stable_hash(plan=program.plan),
        "schedule": program.schedule.to_dict(
            plan=program.plan, task_module=program.task_module
        ),
        "schedule_hash": program.schedule.stable_hash(
            plan=program.plan, task_module=program.task_module
        ),
        "execution_trace": refinement.trace.to_dict(),
        "execution_trace_hash": refinement.trace.stable_hash(),
        "semantic_trace_hash": intermediate_refinement_semantic_trace_hash(refinement),
    }


def _worker(args: argparse.Namespace) -> None:
    import torch

    from boundflow.runtime.native_hard_clause_escalation import (
        compile_native_hard_clause_escalation_program,
        execute_native_hard_clause_escalation_program,
    )

    torch.set_num_threads(args.torch_threads)
    started_ns = time.perf_counter_ns()
    query, tensors, module, input_spec = _load_query_runtime(
        args.model, args.property, args.workload_id
    )
    search_policy, optimizer_policy = _policies()
    program = compile_native_hard_clause_escalation_program(
        module,
        input_spec,
        linear_spec_C=tensors.linear_spec_c,
        thresholds=tensors.thresholds,
        plan_id=f"nrir30:{args.workload_id}",
        search_policy=search_policy,
        optimizer_policy=optimizer_policy,
    )
    setup_ns = time.perf_counter_ns() - started_ns
    execute_started_ns = time.perf_counter_ns()
    execution = execute_native_hard_clause_escalation_program(
        program,
        module,
        input_spec,
        linear_spec_C=tensors.linear_spec_c,
        thresholds=tensors.thresholds,
        query_id=f"nrir30:{query.query_id}",
        search_policy=search_policy,
        optimizer_policy=optimizer_policy,
    )
    execution_ns = time.perf_counter_ns() - execute_started_ns
    baseline = _query_semantics(execution.baseline)
    original_ordinals = execution.trace.decision.escalated_clause_indices
    escalation = (
        None
        if execution.escalation is None
        else _query_semantics(execution.escalation, original_ordinals=original_ordinals)
    )
    refinement = _refinement_evidence(execution)
    semantic = {
        "program_hashes": {
            "plan": program.plan.stable_hash(),
            "task_ir": program.task_ir.stable_hash(),
            "schedule": program.schedule.stable_hash(program.task_ir),
        },
        "control": execution.trace.semantic_dict(),
        "baseline_signature_hash": baseline["semantic_signature_hash"],
        "refinement_semantic_trace_hash": (
            None if refinement is None else refinement["semantic_trace_hash"]
        ),
        "escalation_signature_hash": (
            None if escalation is None else escalation["semantic_signature_hash"]
        ),
    }
    result = {
        "schema_version": WORKER_SCHEMA_VERSION,
        "workload_id": args.workload_id,
        "setup_ns": setup_ns,
        "execution_ns": execution_ns,
        "worker_elapsed_ns": time.perf_counter_ns() - started_ns,
        "program": program.to_dict(),
        "control_trace": execution.trace.to_dict(),
        "baseline": baseline,
        "refinement": refinement,
        "escalation": escalation,
        "semantic_signature_hash": canonical_hash(semantic),
        "performance_claimed": False,
    }
    _validate_worker_result(result)
    _write_json(args.result_json, result)
    print(
        _canonical_json(
            {
                "status": "ok",
                "workload_id": args.workload_id,
                "baseline_verified": execution.trace.decision.baseline_verified_clause_indices,
                "escalated": execution.trace.decision.escalated_clause_indices,
                "final_status": execution.trace.final_status,
                "final_verified": execution.trace.final_verified_clause_indices,
                "fallback": execution.trace.fallback_reason,
                "execution_ns": execution_ns,
            }
        )
    )


def _semantic_trace_projection(trace: Mapping[str, Any]) -> dict[str, object]:
    return {
        key: trace[key]
        for key in (
            "decision_semantics",
            "escalation_original_clause_indices",
            "escalation_completed_original_clause_indices",
            "escalation_verified_original_clause_indices",
            "escalation_pending_original_clause_indices",
            "final_status",
            "final_verified_clause_indices",
            "final_unresolved_clause_indices",
            "final_unsafe_clause_index",
            "fallback_reason",
        )
    }


def _semantic_query_projection(query: Mapping[str, Any]) -> dict[str, object]:
    clauses = _list(query.get("clauses"), "NRIR-30 query clauses")
    return {
        "solver_status": query["solver_status"],
        "completed_original_clause_indices": query["completed_original_clause_indices"],
        "verified_original_clause_indices": query["verified_original_clause_indices"],
        "unresolved_original_clause_indices": query[
            "unresolved_original_clause_indices"
        ],
        "pending_original_clause_indices": query["pending_original_clause_indices"],
        "clauses": [
            {
                key: _mapping(value, "NRIR-30 clause")[key]
                for key in (
                    "projected_clause_index",
                    "original_clause_index",
                    "status",
                    "root_lower",
                    "root_upper",
                    "queue_status",
                    "evaluated_nodes",
                    "domains",
                )
            }
            for value in clauses
        ],
    }


def _validate_program(program: Mapping[str, Any]) -> None:
    plan = _mapping(program.get("plan"), "NRIR-30 Plan")
    task_ir = _mapping(program.get("task_ir"), "NRIR-30 Task IR")
    schedule = _mapping(program.get("schedule"), "NRIR-30 Schedule IR")
    if (
        canonical_hash(plan) != program.get("plan_hash")
        or canonical_hash(task_ir) != program.get("task_ir_hash")
        or canonical_hash(schedule) != program.get("schedule_hash")
        or task_ir.get("plan_hash") != program.get("plan_hash")
        or schedule.get("plan_hash") != program.get("plan_hash")
        or schedule.get("task_ir_hash") != program.get("task_ir_hash")
        or plan.get("whole_query_timeout_ns") != TIMEOUT_SECONDS * 1_000_000_000
        or _mapping(plan.get("baseline_budget"), "baseline budget").get("max_nodes")
        != 7
        or _mapping(plan.get("baseline_budget"), "baseline budget").get("max_depth")
        != 2
        or _mapping(plan.get("escalation_budget"), "escalation budget").get("max_nodes")
        != 31
        or _mapping(plan.get("escalation_budget"), "escalation budget").get("max_depth")
        != 4
        or _mapping(plan.get("refinement_policy"), "refinement policy").get(
            "max_neurons_per_relu"
        )
        != 128
        or _mapping(plan.get("refinement_policy"), "refinement policy").get(
            "backward_chunk_size"
        )
        != 32
    ):
        raise ValueError("NRIR-30 source-to-program IR differs")
    tasks = _list(task_ir.get("tasks"), "NRIR-30 tasks")
    actions = _list(schedule.get("actions"), "NRIR-30 schedule actions")
    if len(tasks) != 8 or len(actions) != 8:
        raise ValueError("NRIR-30 Task/Schedule coverage differs")
    for sequence, (task_value, action_value) in enumerate(zip(tasks, actions)):
        task = _mapping(task_value, "NRIR-30 task")
        action = _mapping(action_value, "NRIR-30 schedule action")
        if (
            action.get("sequence") != sequence
            or action.get("task_id") != task.get("task_id")
            or action.get("kind") != task.get("kind")
            or action.get("guard") != task.get("guard")
        ):
            raise ValueError("NRIR-30 Schedule/Task linkage differs")


def _validate_query(query: Mapping[str, Any]) -> None:
    clauses = _list(query.get("clauses"), "NRIR-30 clauses")
    projection = _semantic_query_projection(query)
    if (
        query.get("solver_status") not in {"verified", "unsafe", "unknown"}
        or canonical_hash(projection) != query.get("semantic_signature_hash")
        or not _sha256(query.get("query_trace_hash"))
        or not clauses
    ):
        raise ValueError("NRIR-30 query semantic digest differs")
    original_indices = []
    for value in clauses:
        clause = _mapping(value, "NRIR-30 clause")
        domains = _list(clause.get("domains"), "NRIR-30 domains")
        original_indices.append(int(clause["original_clause_index"]))
        if (
            clause.get("status") not in {"verified", "unsafe", "unknown"}
            or clause.get("evaluated_nodes") != len(domains)
            or not domains
            or any(
                not _sha256(_mapping(item, "NRIR-30 domain").get("split_state_hash"))
                for item in domains
            )
            or not _sha256(clause.get("queue_trace_hash"))
            or not _sha256(clause.get("verdict_trace_hash"))
        ):
            raise ValueError("NRIR-30 clause/domain evidence differs")
    if len(original_indices) != len(set(original_indices)):
        raise ValueError("NRIR-30 original clause ordinal repeats")
    _validate_compiler(_mapping(query.get("compiler"), "NRIR-30 compiler"))


def _validate_refinement(refinement: Mapping[str, Any]) -> None:
    plan = _mapping(refinement.get("plan"), "NRIR-30 refinement Plan")
    task_ir = _mapping(refinement.get("task_ir"), "NRIR-30 refinement Task")
    schedule = _mapping(refinement.get("schedule"), "NRIR-30 refinement Schedule")
    trace = _mapping(refinement.get("execution_trace"), "NRIR-30 refinement trace")
    semantic_trace = dict(trace)
    semantic_trace.pop("elapsed_ns", None)
    if (
        canonical_hash(plan) != refinement.get("plan_hash")
        or canonical_hash(task_ir) != refinement.get("task_ir_hash")
        or canonical_hash(schedule) != refinement.get("schedule_hash")
        or canonical_hash(trace) != refinement.get("execution_trace_hash")
        or canonical_hash(semantic_trace) != refinement.get("semantic_trace_hash")
    ):
        raise ValueError("NRIR-30 refinement IR/trace digest differs")


def _validate_worker_result(result: Mapping[str, Any]) -> None:
    program = _mapping(result.get("program"), "NRIR-30 program")
    trace = _mapping(result.get("control_trace"), "NRIR-30 control trace")
    baseline = _mapping(result.get("baseline"), "NRIR-30 baseline")
    refinement_value = result.get("refinement")
    escalation_value = result.get("escalation")
    if (
        result.get("schema_version") != WORKER_SCHEMA_VERSION
        or result.get("performance_claimed") is not False
        or not result.get("workload_id")
        or any(
            not isinstance(result.get(name), int) or int(result[name]) < 0
            for name in ("setup_ns", "execution_ns", "worker_elapsed_ns")
        )
        or not _sha256(result.get("semantic_signature_hash"))
    ):
        raise ValueError("NRIR-30 worker header differs")
    _validate_program(program)
    _validate_query(baseline)
    if refinement_value is None:
        if escalation_value is not None:
            raise ValueError("NRIR-30 escalation lacks refinement")
    else:
        _validate_refinement(_mapping(refinement_value, "NRIR-30 refinement"))
    if escalation_value is not None:
        _validate_query(_mapping(escalation_value, "NRIR-30 escalation"))
    decision = _mapping(trace.get("decision"), "NRIR-30 decision")
    actions = _list(trace.get("actions"), "NRIR-30 actions")
    trace_semantic = _semantic_trace_projection(trace)
    baseline_pending = decision.get("baseline_pending_clause_indices", [])
    baseline_unsafe = decision.get("baseline_unsafe_clause_index")
    expected_escalated = (
        []
        if baseline_pending or baseline_unsafe is not None
        else baseline.get("unresolved_original_clause_indices")
    )
    clause_count = int(_mapping(program.get("plan"), "NRIR-30 Plan")["clause_count"])
    final_verified = set(trace.get("final_verified_clause_indices", []))
    final_unresolved = set(trace.get("final_unresolved_clause_indices", []))
    escalated = set(decision.get("escalated_clause_indices", []))
    escalation_verified = set(
        trace.get("escalation_verified_original_clause_indices", [])
    )
    if (
        canonical_hash(decision) != trace.get("decision_hash")
        or canonical_hash(trace_semantic) != trace.get("semantic_signature_hash")
        or trace.get("plan_hash") != program.get("plan_hash")
        or trace.get("task_ir_hash") != program.get("task_ir_hash")
        or trace.get("schedule_hash") != program.get("schedule_hash")
        or len(actions) != 8
        or decision.get("escalated_clause_indices") != expected_escalated
        or not set(decision.get("baseline_verified_clause_indices", []))
        <= final_verified
        or not escalation_verified <= escalated
        or not escalation_verified <= final_verified
        or final_verified & final_unresolved
        or not final_verified | final_unresolved <= set(range(clause_count))
        or (
            trace.get("final_status") == "verified"
            and final_verified != set(range(clause_count))
        )
        or (trace.get("final_status") == "unknown" and not final_unresolved)
        or (
            trace.get("fallback_reason") == "none"
            and int(trace.get("elapsed_ns", -1)) > int(trace.get("deadline_ns", -1))
        )
    ):
        raise ValueError("NRIR-30 control/admission evidence differs")
    for sequence, value in enumerate(actions):
        action = _mapping(value, "NRIR-30 action")
        if (
            action.get("sequence") != sequence
            or not _sha256(action.get("output_hash"))
            or not isinstance(action.get("elapsed_ns"), int)
        ):
            raise ValueError("NRIR-30 action trace differs")
    executed = [
        bool(_mapping(value, "NRIR-30 action")["executed"]) for value in actions
    ]
    if (
        executed[:2] != [True, True]
        or executed[6:] != [True, True]
        or (not escalated and executed[2:6] != [False, False, False, False])
        or (escalated and executed[2:5] != [True, True, True])
    ):
        raise ValueError("NRIR-30 guarded Schedule execution differs")
    if escalation_value is not None:
        escalation = _mapping(escalation_value, "NRIR-30 escalation")
        if escalation.get("completed_original_clause_indices") != trace.get(
            "escalation_completed_original_clause_indices"
        ):
            raise ValueError("NRIR-30 projected/original clause mapping differs")
    semantic = {
        "program_hashes": {
            "plan": program["plan_hash"],
            "task_ir": program["task_ir_hash"],
            "schedule": program["schedule_hash"],
        },
        "control": trace_semantic,
        "baseline_signature_hash": baseline["semantic_signature_hash"],
        "refinement_semantic_trace_hash": (
            None
            if refinement_value is None
            else _mapping(refinement_value, "NRIR-30 refinement")["semantic_trace_hash"]
        ),
        "escalation_signature_hash": (
            None
            if escalation_value is None
            else _mapping(escalation_value, "NRIR-30 escalation")[
                "semantic_signature_hash"
            ]
        ),
    }
    if canonical_hash(semantic) != result.get("semantic_signature_hash"):
        raise ValueError("NRIR-30 worker semantic signature differs")


def _worker_command(
    *, workload: Mapping[str, object], result_path: Path, torch_threads: int
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--workload-id",
        str(workload["workload_id"]),
        "--model",
        str(workload["model"]),
        "--property",
        str(workload["property"]),
        "--result-json",
        str(result_path),
        "--torch-threads",
        str(torch_threads),
    ]


def _run_worker(
    *, workload: Mapping[str, object], result_path: Path, torch_threads: int
) -> tuple[dict[str, Any], str, int]:
    started_ns = time.perf_counter_ns()
    completed = subprocess.run(
        _worker_command(
            workload=workload,
            result_path=result_path,
            torch_threads=torch_threads,
        ),
        cwd=_repo_root(),
        text=True,
        capture_output=True,
        timeout=TIMEOUT_SECONDS + 180,
        check=False,
    )
    elapsed_ns = time.perf_counter_ns() - started_ns
    log = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise RuntimeError(
            f"NRIR-30 {workload['workload_id']} worker failed with "
            f"{completed.returncode}: {log}"
        )
    result = _load_json(result_path)
    _validate_worker_result(result)
    return result, log, elapsed_ns


def _predecessor_evidence() -> dict[str, Any]:
    root = _repo_root()
    manifest = _load_json(root / PREDECESSOR_DIR / MANIFEST_FILE)
    evidence = _load_json(root / PREDECESSOR_DIR / EVIDENCE_FILE)
    if (
        manifest.get("evidence_hash") != PREDECESSOR_EVIDENCE_HASH
        or canonical_hash(evidence) != PREDECESSOR_EVIDENCE_HASH
    ):
        raise ValueError("NRIR-30 predecessor artifact differs")
    return evidence


def _validate_predecessor_alignment(records: Sequence[Mapping[str, Any]]) -> None:
    predecessor = _predecessor_evidence()
    predecessor_records = _list(predecessor.get("records"), "NRIR-29 records")
    for record in records:
        workload_id = record["workload_id"]
        baseline = _mapping(
            _mapping(record["result"], "NRIR-30 result")["baseline"],
            "NRIR-30 baseline",
        )
        reference_record = next(
            _mapping(value, "NRIR-29 record")
            for value in predecessor_records
            if _mapping(value, "NRIR-29 record").get("workload_id") == workload_id
            and _mapping(value, "NRIR-29 record").get("budget_id") == "n7d2"
        )
        reference = _mapping(reference_record["result"], "NRIR-29 result")
        reference_clauses = _list(reference["clauses"], "NRIR-29 clauses")
        baseline_clauses = _list(baseline["clauses"], "NRIR-30 clauses")
        if (
            baseline.get("completed_original_clause_indices")
            != reference.get("completed_clause_indices")
            or baseline.get("verified_original_clause_indices")
            != reference.get("verified_clause_indices")
            or baseline.get("unresolved_original_clause_indices")
            != reference.get("unresolved_clause_indices")
            or baseline.get("pending_original_clause_indices")
            != reference.get("pending_clause_indices")
            or len(baseline_clauses) != len(reference_clauses)
        ):
            raise ValueError("NRIR-30 baseline/predecessor accounting differs")
        for baseline_value, reference_value in zip(baseline_clauses, reference_clauses):
            left = _mapping(baseline_value, "NRIR-30 baseline clause")
            right = _mapping(reference_value, "NRIR-29 baseline clause")
            if (
                left.get("status") != right.get("status")
                or left.get("evaluated_nodes") != right.get("evaluated_nodes")
                or abs(float(left["root_lower"]) - float(right["root_lower"])) > 1e-5
                or abs(float(left["root_upper"]) - float(right["root_upper"])) > 1e-5
            ):
                raise ValueError("NRIR-30 baseline/predecessor semantics differ")


def _p90(values: list[int]) -> int:
    return sorted(values)[-1]


def _summaries(records: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    summaries: dict[str, object] = {}
    workload_ids = sorted({str(record["workload_id"]) for record in records})
    for workload_id in workload_ids:
        selected = [
            record for record in records if record["workload_id"] == workload_id
        ]
        execution_ns = [
            int(_mapping(record["result"], "NRIR-30 result")["execution_ns"])
            for record in selected
        ]
        e2e_ns = [int(record["e2e_elapsed_ns"]) for record in selected]
        traces = [
            _mapping(
                _mapping(record["result"], "NRIR-30 result")["control_trace"],
                "NRIR-30 trace",
            )
            for record in selected
        ]
        summaries[workload_id] = {
            "raw_execution_ns": execution_ns,
            "median_execution_ns": int(statistics.median(execution_ns)),
            "p90_execution_ns": _p90(execution_ns),
            "raw_e2e_ns": e2e_ns,
            "median_e2e_ns": int(statistics.median(e2e_ns)),
            "p90_e2e_ns": _p90(e2e_ns),
            "baseline_verified_clause_indices": [
                trace["decision"]["baseline_verified_clause_indices"]
                for trace in traces
            ],
            "escalated_clause_indices": [
                trace["decision"]["escalated_clause_indices"] for trace in traces
            ],
            "final_statuses": [trace["final_status"] for trace in traces],
            "final_verified_clause_indices": [
                trace["final_verified_clause_indices"] for trace in traces
            ],
            "fallback_reasons": [trace["fallback_reason"] for trace in traces],
            "performance_claimed": False,
        }
    return summaries


def _property_status(summaries: Mapping[str, Any]) -> str:
    strict_full_closure = False
    for value in summaries.values():
        summary = _mapping(value, "NRIR-30 summary")
        baseline_sets = summary["baseline_verified_clause_indices"]
        final_sets = summary["final_verified_clause_indices"]
        statuses = summary["final_statuses"]
        if all(status == "verified" for status in statuses) and all(
            len(final) > len(baseline)
            for baseline, final in zip(baseline_sets, final_sets)
        ):
            strict_full_closure = True
    return "validated_reduced" if strict_full_closure else "validated_no_go"


def validate_evidence_structure(evidence: Mapping[str, Any]) -> None:
    if (
        evidence.get("schema_version") != EVIDENCE_SCHEMA_VERSION
        or evidence.get("status") != "ok"
        or evidence.get("property_status")
        not in {"validated_reduced", "validated_no_go"}
        or evidence.get("performance_claimed") is not False
    ):
        raise ValueError("NRIR-30 evidence header differs")
    records = [
        dict(_mapping(value, "NRIR-30 record"))
        for value in _list(evidence.get("records"), "NRIR-30 records")
    ]
    if len(records) != len(WORKLOAD_ROWS) * REPEATS:
        raise ValueError("NRIR-30 record count differs")
    for record in records:
        _validate_worker_result(_mapping(record.get("result"), "NRIR-30 result"))
        if (
            record.get("repeat_index") not in range(REPEATS)
            or not isinstance(record.get("e2e_elapsed_ns"), int)
            or int(record["e2e_elapsed_ns"]) <= 0
            or not _sha256(record.get("log_sha256"))
        ):
            raise ValueError("NRIR-30 record identity differs")
    for workload_id in sorted({str(record["workload_id"]) for record in records}):
        selected = [
            _mapping(record["result"], "NRIR-30 repeat result")
            for record in records
            if record["workload_id"] == workload_id
        ]
        if (
            len(selected) != REPEATS
            or len({str(result["semantic_signature_hash"]) for result in selected}) != 1
        ):
            raise ValueError("NRIR-30 repeated semantics differ")
    _validate_predecessor_alignment(records)
    summaries = _mapping(evidence.get("summaries"), "NRIR-30 summaries")
    recomputed = _summaries(records)
    if dict(summaries) != recomputed:
        raise ValueError("NRIR-30 summary replay differs")
    if evidence.get("property_status") != _property_status(recomputed):
        raise ValueError("NRIR-30 property closure gate differs")


def _build_evidence(
    args: argparse.Namespace,
) -> tuple[dict[str, object], dict[str, str]]:
    workloads = _resolve_workloads(args.benchmark_root)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    (args.artifact_dir / "logs").mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    files: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="boundflow-nrir30-") as temporary:
        temporary_root = Path(temporary)
        for workload in workloads:
            workload_id = str(workload["workload_id"])
            for repeat_index in range(REPEATS):
                stem = f"{workload_id.replace(':', '-')}-r{repeat_index}"
                result_path = temporary_root / f"{stem}.json"
                result, log, elapsed_ns = _run_worker(
                    workload=workload,
                    result_path=result_path,
                    torch_threads=args.torch_threads,
                )
                log_name = f"logs/{stem}.log"
                log_path = args.artifact_dir / log_name
                log_path.write_text(log, encoding="utf-8")
                files[log_name] = file_sha256(log_path)
                records.append(
                    {
                        "workload_id": workload_id,
                        "repeat_index": repeat_index,
                        "e2e_elapsed_ns": elapsed_ns,
                        "log_path": log_name,
                        "log_sha256": files[log_name],
                        "result": result,
                    }
                )
                trace = _mapping(result["control_trace"], "NRIR-30 trace")
                print(
                    _canonical_json(
                        {
                            "workload_id": workload_id,
                            "repeat_index": repeat_index,
                            "baseline_verified": trace["decision"][
                                "baseline_verified_clause_indices"
                            ],
                            "escalated": trace["decision"]["escalated_clause_indices"],
                            "final_status": trace["final_status"],
                            "final_verified": trace["final_verified_clause_indices"],
                            "fallback": trace["fallback_reason"],
                            "execution_ns": result["execution_ns"],
                        }
                    ),
                    flush=True,
                )
    summaries = _summaries(records)
    evidence = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "ok",
        "property_status": _property_status(summaries),
        "performance_claimed": False,
        "claim_boundary": "typed unresolved-clause staged control and fixed-whole-deadline property coverage on three real VNN-COMP CPU workloads; no speedup, GPU, competitor, full-suite, or ASPLOS-ready claim",
        "source": {
            "vnncomp_commit": VNNCOMP_COMMIT,
            "native_code_revision": _code_revision(),
            "predecessor_artifact_evidence_hash": PREDECESSOR_EVIDENCE_HASH,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device": "cpu",
            "torch_threads": args.torch_threads,
            "cuda_executed": False,
        },
        "protocol": {
            "repeats": REPEATS,
            "whole_query_timeout_seconds": TIMEOUT_SECONDS,
            "baseline_budget": {"max_nodes": 7, "max_depth": 2},
            "escalation_budget": {"max_nodes": 31, "max_depth": 4},
            "refinement": {
                "passes": 1,
                "max_neurons_per_relu": 128,
                "backward_chunk_size": 32,
                "candidate_policy_id": "top_ambiguous_width_per_relu_v1",
            },
            "admission": "exact_baseline_unresolved_original_clause_ordinals",
            "fallback": "preserve_baseline_verdicts",
        },
        "workloads": [_public_workload(item) for item in workloads],
        "records": records,
        "summaries": summaries,
        "limitations": [
            "The result evaluates typed staged control and property coverage, not cross-stage or external-verifier speedup.",
            "The refinement policy is one shared top-width selected-CROWN pass; objective-specific, ancestral, and external-seeded refinement are not used.",
            "The 60-second whole-query deadline is cooperative at stage and clause boundaries; over-deadline escalation evidence is discarded in favor of baseline verdicts.",
            "Only three selected CPU workloads are covered; no GPU, complete benchmark suite, competitor superiority, or ASPLOS readiness is claimed.",
        ],
    }
    validate_evidence_structure(evidence)
    return evidence, files


def _generate(args: argparse.Namespace) -> None:
    evidence, files = _build_evidence(args)
    evidence_path = args.artifact_dir / EVIDENCE_FILE
    _write_json(evidence_path, evidence)
    files[EVIDENCE_FILE] = file_sha256(evidence_path)
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "ok",
        "property_status": evidence["property_status"],
        "performance_claimed": False,
        "files": dict(sorted(files.items())),
        "evidence_hash": canonical_hash(evidence),
    }
    _write_json(args.artifact_dir / MANIFEST_FILE, manifest)
    print(
        _canonical_json(
            {
                "status": "ok",
                "property_status": manifest["property_status"],
                "evidence_hash": manifest["evidence_hash"],
            }
        )
    )


def _expected_program_payload(workload: Mapping[str, object]) -> dict[str, object]:
    from boundflow.runtime.native_hard_clause_escalation import (
        compile_native_hard_clause_escalation_program,
    )

    _query, tensors, module, input_spec = _load_query_runtime(
        Path(str(workload["model"])),
        Path(str(workload["property"])),
        str(workload["workload_id"]),
    )
    search_policy, optimizer_policy = _policies()
    program = compile_native_hard_clause_escalation_program(
        module,
        input_spec,
        linear_spec_C=tensors.linear_spec_c,
        thresholds=tensors.thresholds,
        plan_id=f"nrir30:{workload['workload_id']}",
        search_policy=search_policy,
        optimizer_policy=optimizer_policy,
    )
    return program.to_dict()


def _replay(args: argparse.Namespace) -> None:
    manifest = _load_json(args.artifact_dir / MANIFEST_FILE)
    evidence = _load_json(args.artifact_dir / EVIDENCE_FILE)
    files = _mapping(manifest.get("files"), "NRIR-30 manifest files")
    actual_files = {
        str(path.relative_to(args.artifact_dir)): file_sha256(path)
        for path in sorted(args.artifact_dir.rglob("*"))
        if path.is_file() and path.name != MANIFEST_FILE
    }
    if (
        manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION
        or manifest.get("status") != "ok"
        or manifest.get("performance_claimed") is not False
        or manifest.get("property_status") != evidence.get("property_status")
        or dict(files) != actual_files
        or manifest.get("evidence_hash") != canonical_hash(evidence)
    ):
        raise ValueError("NRIR-30 artifact manifest differs")
    source = _mapping(evidence.get("source"), "NRIR-30 source")
    if (
        source.get("vnncomp_commit") != VNNCOMP_COMMIT
        or source.get("native_code_revision") != _code_revision()
        or source.get("predecessor_artifact_evidence_hash") != PREDECESSOR_EVIDENCE_HASH
    ):
        raise ValueError("NRIR-30 source revision differs")
    workloads = _resolve_workloads(args.benchmark_root)
    if evidence.get("workloads") != [_public_workload(item) for item in workloads]:
        raise ValueError("NRIR-30 workload source replay differs")
    expected_programs = {
        str(item["workload_id"]): _expected_program_payload(item) for item in workloads
    }
    for record_value in _list(evidence.get("records"), "NRIR-30 records"):
        record = _mapping(record_value, "NRIR-30 record")
        result = _mapping(record["result"], "NRIR-30 result")
        if result.get("program") != expected_programs[str(record["workload_id"])]:
            raise ValueError("NRIR-30 source-to-program replay differs")
        if (
            file_sha256(args.artifact_dir / str(record["log_path"]))
            != record["log_sha256"]
        ):
            raise ValueError("NRIR-30 log digest differs")
    validate_evidence_structure(evidence)
    print(
        _canonical_json(
            {
                "status": "ok",
                "property_status": evidence["property_status"],
                "evidence_hash": manifest["evidence_hash"],
            }
        )
    )


def main() -> None:
    args = _parse_args()
    if args.torch_threads < 1:
        raise ValueError("NRIR-30 torch thread count must be positive")
    if args.command == "worker":
        _worker(args)
    elif args.command == "generate":
        _generate(args)
    else:
        _replay(args)


if __name__ == "__main__":
    main()
