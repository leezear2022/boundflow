#!/usr/bin/env python3
"""Generate or replay the NRIR-29 fixed-wall-clock BaB scaling artifact."""

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
from typing import Any, Mapping, Sequence

from scripts.run_multiworkload_competitor_e2e_artifact import (
    VNNCOMP_COMMIT,
    WORKLOAD_ROWS,
    _csv_selection,
    _git_revision,
    _onnx_inventory,
)

ARTIFACT_SCHEMA_VERSION = "boundflow.wall-clock-bab-scaling-artifact/v1"
EVIDENCE_SCHEMA_VERSION = "boundflow.wall-clock-bab-scaling-evidence/v1"
WORKER_SCHEMA_VERSION = "boundflow.wall-clock-bab-scaling-worker/v1"
ARTIFACT_DIR = Path(
    "artifacts/wall-clock-parametric-bab-scaling/vnncomp21-three-topology-cpu-v1"
)
MANIFEST_FILE = "manifest.json"
EVIDENCE_FILE = "evidence.json"
PLAN_ID = "nrir29-wall-clock-parametric-bab-scaling-v1"
REPEATS = 3
TORCH_THREADS = 8
TIMEOUT_SECONDS = 60
ALPHA_STEPS = 5
SEARCH_STEPS = 4
EXPANSION_BATCH_SIZE = 2
MAX_EVAL_BATCH_SIZE = 4
COMMON_DOMAIN_LOWER_TOLERANCE = 1e-5
BUDGET_SPECS = (
    ("n7d2", 7, 2),
    ("n31d4", 31, 4),
    ("n127d6", 127, 6),
)


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
    worker.add_argument("--budget-id", required=True)
    worker.add_argument("--max-nodes", type=int, required=True)
    worker.add_argument("--max-depth", type=int, required=True)
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
        "boundflow/ir/workload.py",
        "boundflow/ir/search_scaling.py",
        "boundflow/ir/parametric_optimizer.py",
        "boundflow/runtime/native_parametric_optimizer.py",
        "boundflow/runtime/native_parametric_production_verifier.py",
        "boundflow/runtime/native_parametric_production_complete_query.py",
        "scripts/run_wall_clock_parametric_bab_scaling_artifact.py",
    )
    return canonical_hash({path: file_sha256(root / path) for path in paths})


def _build_experiment_ir(
    benchmark_root: Path, torch_threads: int
) -> tuple[Any, Any, Any, list[dict[str, object]]]:
    from boundflow.frontends.vnnlib import import_vnnlib_box_query
    from boundflow.ir.search_scaling import (
        NativeBabSearchBudgetIR,
        NativeBabSearchScalingPlanIR,
        compile_search_scaling_schedule_ir,
        compile_search_scaling_task_ir,
    )
    from boundflow.ir.workload import VerificationWorkloadSourceIR

    if _git_revision(benchmark_root) != VNNCOMP_COMMIT:
        raise ValueError("NRIR-29 VNN-COMP commit differs")
    sources = []
    resolved: list[dict[str, object]] = []
    for definition in WORKLOAD_ROWS:
        csv_path, model_path, property_path = _csv_selection(benchmark_root, definition)
        workload_id = str(definition["workload_id"])
        query = import_vnnlib_box_query(property_path, query_id=workload_id)
        input_shape, output_dim, ops = _onnx_inventory(model_path)
        if (
            len(query.input_names) != math.prod(input_shape[1:])
            or len(query.output_names) != output_dim
        ):
            raise ValueError("NRIR-29 ONNX/VNNLIB dimensions differ")
        ordinal = definition["csv_ordinal"]
        if not isinstance(ordinal, int):
            raise TypeError("NRIR-29 CSV ordinal must be an integer")
        source = VerificationWorkloadSourceIR(
            workload_id=workload_id,
            category=str(definition["category"]),
            csv_ordinal=ordinal,
            csv_relative_path=str(definition["csv"]),
            model_relative_path=str(definition["model"]),
            property_relative_path=str(definition["property"]),
            csv_sha256=file_sha256(csv_path),
            model_sha256=file_sha256(model_path),
            property_sha256=file_sha256(property_path),
            query_ir_hash=query.stable_hash(),
            model_input_shape=input_shape,
            model_output_dim=output_dim,
            onnx_ops=ops,
        )
        source.validate()
        sources.append(source)
        resolved.append(
            {
                "workload_id": workload_id,
                "source": source,
                "model": model_path,
                "property": property_path,
            }
        )
    budgets = tuple(
        NativeBabSearchBudgetIR(budget_id, max_nodes, max_depth)
        for budget_id, max_nodes, max_depth in BUDGET_SPECS
    )
    plan = NativeBabSearchScalingPlanIR(
        plan_id=PLAN_ID,
        benchmark_commit=VNNCOMP_COMMIT,
        native_code_revision=_code_revision(),
        workloads=tuple(sources),
        budgets=budgets,
        repeats=REPEATS,
        timeout_seconds=TIMEOUT_SECONDS,
        torch_threads=torch_threads,
        optimizer_steps=ALPHA_STEPS,
        search_steps=SEARCH_STEPS,
        expansion_batch_size=EXPANSION_BATCH_SIZE,
        max_eval_batch_size=MAX_EVAL_BATCH_SIZE,
    )
    task_ir = compile_search_scaling_task_ir(plan)
    schedule = compile_search_scaling_schedule_ir(plan, task_ir)
    return plan, task_ir, schedule, resolved


def _compiler_evidence(execution: Any) -> dict[str, object]:
    instances = []
    for clause in execution.clauses:
        for batch in clause.queue.compiler_batches:
            instances.append(
                {
                    "batch_id": batch.batch_id,
                    "cache_event_hash": batch.cache_event.stable_hash(),
                    "instance": batch.instance_ir.to_dict(),
                    "instance_hash": batch.instance_ir.stable_hash(),
                    "template_hash": batch.template_hash,
                    "task_hash": batch.task_hash,
                    "schedule_hash": batch.schedule_hash,
                }
            )
    return {
        "cache": execution.compiler_cache_trace.to_dict(),
        "instances": instances,
    }


def _leaf_records(
    evaluation_by_id: Mapping[str, Any], node_ids: Sequence[str]
) -> list[dict[str, object]]:
    return [
        {
            "node_id": node_id,
            "split_state_hash": evaluation_by_id[node_id].node.split_state_hash,
            "lower": evaluation_by_id[node_id].lower,
            "upper": evaluation_by_id[node_id].upper,
        }
        for node_id in node_ids
    ]


def _semantic_query(execution: Any) -> dict[str, object]:
    clauses = []
    semantic_clauses = []
    for clause in execution.clauses:
        queue = clause.queue.trace
        evaluation_by_id = {item.node.node_id: item for item in queue.evaluations}
        domains = [
            {
                "node_id": item.node.node_id,
                "parent_node_id": item.node.parent_node_id,
                "split_state_hash": item.node.split_state_hash,
                "depth": item.node.depth,
                "lower": item.lower,
                "upper": item.upper,
                "selected_state_hash": item.selected_state_hash,
            }
            for item in queue.evaluations
        ]
        pruned_ids = clause.verdict.trace.sound_pruned_leaf_node_ids
        unresolved_ids = clause.verdict.trace.unresolved_leaf_node_ids

        pruned = _leaf_records(evaluation_by_id, pruned_ids)
        unresolved = _leaf_records(evaluation_by_id, unresolved_ids)
        semantic = {
            "clause_index": clause.trace.clause_index,
            "status": clause.trace.status,
            "queue_status": queue.status,
            "termination_reason": queue.termination_reason,
            "evaluated_nodes": len(queue.evaluations),
            "root_lower": queue.evaluations[0].lower,
            "root_upper": queue.evaluations[0].upper,
            "domains": domains,
            "sound_pruned_leaves": pruned,
            "unresolved_leaves": unresolved,
            "worst_unresolved_lower": (
                None
                if not unresolved
                else min(float(item["lower"]) for item in unresolved)
            ),
        }
        clauses.append(
            {
                **semantic,
                "logical_queue_signature_hash": canonical_hash(
                    queue.logical_queue_signature()
                ),
                "queue_trace_hash": queue.stable_hash(),
                "verdict_trace_hash": clause.verdict.trace.stable_hash(queue),
            }
        )
        semantic_clauses.append(semantic)
    semantic_query = {
        "solver_status": execution.trace.status,
        "completed_clause_count": len(execution.clauses),
        "completed_clause_indices": [
            item.clause_index for item in execution.trace.completed_clauses
        ],
        "unresolved_clause_indices": list(execution.trace.unresolved_clause_indices),
        "pending_clause_indices": list(execution.trace.pending_clause_indices),
        "verified_clause_indices": [
            item.trace.clause_index
            for item in execution.clauses
            if item.trace.status == "verified"
        ],
        "clauses": semantic_clauses,
    }
    return {
        **semantic_query,
        "clauses": clauses,
        "query_trace_hash": execution.trace.stable_hash(),
        "semantic_signature_hash": canonical_hash(semantic_query),
    }


def _worker(args: argparse.Namespace) -> None:
    import torch

    from boundflow.frontends.onnx.frontend import import_onnx
    from boundflow.frontends.vnnlib import (
        import_vnnlib_box_query,
        materialize_vnnlib_box_query,
    )
    from boundflow.planner import plan_interval_ibp_v0
    from boundflow.runtime.complete_verifier_query import CompleteVerifierQueryPolicy
    from boundflow.runtime.native_alpha_beta_optimization_state import (
        NativeAlphaBetaOptimizerPolicy,
    )
    from boundflow.runtime.native_candidate_search import (
        NativeProjectedGradientSearchPolicy,
    )
    from boundflow.runtime.native_parametric_production_complete_query import (
        execute_native_parametric_production_complete_verifier_query,
    )
    from boundflow.runtime.native_relu_split_bab_runtime import NativeReluSplitBabConfig
    from boundflow.runtime.task_executor import InputSpec

    expected_budget = next(
        (
            (max_nodes, max_depth)
            for budget_id, max_nodes, max_depth in BUDGET_SPECS
            if budget_id == args.budget_id
        ),
        None,
    )
    if expected_budget != (args.max_nodes, args.max_depth):
        raise ValueError("NRIR-29 worker budget differs from registered IR")
    torch.set_num_threads(args.torch_threads)
    started_ns = time.perf_counter_ns()
    query = import_vnnlib_box_query(args.property, query_id=args.workload_id)
    input_shape, output_dim, _ops = _onnx_inventory(args.model)
    if len(query.output_names) != output_dim:
        raise ValueError("NRIR-29 worker output dimension differs")
    tensors = materialize_vnnlib_box_query(query, input_shape=input_shape[1:])
    program = import_onnx(str(args.model), do_shape_infer=True, normalize=True)
    module = plan_interval_ibp_v0(program)
    input_spec = InputSpec.box(
        value_name=program.graph.inputs[0],
        lower=tensors.input_lower,
        upper=tensors.input_upper,
    )
    setup_ns = time.perf_counter_ns() - started_ns
    query_policy = CompleteVerifierQueryPolicy(timeout_ns=TIMEOUT_SECONDS * 10**9)
    search_policy = NativeProjectedGradientSearchPolicy(
        steps=SEARCH_STEPS, step_size=0.002
    )
    queue_config = NativeReluSplitBabConfig(
        max_nodes=args.max_nodes,
        max_depth=args.max_depth,
        expansion_batch_size=EXPANSION_BATCH_SIZE,
        max_eval_batch_size=MAX_EVAL_BATCH_SIZE,
    )
    optimizer_policy = NativeAlphaBetaOptimizerPolicy(
        steps=ALPHA_STEPS,
        lr=0.1,
        alpha_initialization_mode="adaptive",
    )
    execute_started_ns = time.perf_counter_ns()
    execution = execute_native_parametric_production_complete_verifier_query(
        module,
        input_spec,
        linear_spec_C=tensors.linear_spec_c,
        thresholds=tensors.thresholds,
        query_id=f"{query.query_id}:full",
        query_policy=query_policy,
        search_policy=search_policy,
        queue_config=queue_config,
        optimizer_policy=optimizer_policy,
    )
    execution_ns = time.perf_counter_ns() - execute_started_ns
    execution.validate_against(module, input_spec, linear_spec_C=tensors.linear_spec_c)
    result = {
        "schema_version": WORKER_SCHEMA_VERSION,
        "workload_id": args.workload_id,
        "budget": {
            "budget_id": args.budget_id,
            "max_nodes": args.max_nodes,
            "max_depth": args.max_depth,
        },
        "setup_ns": setup_ns,
        "execution_ns": execution_ns,
        "worker_elapsed_ns": time.perf_counter_ns() - started_ns,
        "execution_mode": "production_parametric_template_instance_v2",
        "performance_claimed": False,
        "compiler": _compiler_evidence(execution),
        **_semantic_query(execution),
    }
    _write_json(args.result_json, result)
    print(
        _canonical_json(
            {
                "status": "ok",
                "workload_id": args.workload_id,
                "budget_id": args.budget_id,
                "solver_status": execution.trace.status,
                "completed": len(execution.clauses),
            }
        )
    )


def _worker_command(
    *,
    workload: Mapping[str, object],
    budget: Mapping[str, object],
    result_path: Path,
    torch_threads: int,
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
        "--budget-id",
        str(budget["budget_id"]),
        "--max-nodes",
        str(budget["max_nodes"]),
        "--max-depth",
        str(budget["max_depth"]),
        "--result-json",
        str(result_path),
        "--torch-threads",
        str(torch_threads),
    ]


def _run_worker(
    *,
    workload: Mapping[str, object],
    budget: Mapping[str, object],
    result_path: Path,
    torch_threads: int,
) -> tuple[dict[str, Any], str, int]:
    started_ns = time.perf_counter_ns()
    completed = subprocess.run(
        _worker_command(
            workload=workload,
            budget=budget,
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
            f"NRIR-29 {workload['workload_id']} {budget['budget_id']} worker "
            f"failed with {completed.returncode}: {log}"
        )
    return _load_json(result_path), log, elapsed_ns


def _validate_compiler(compiler: Mapping[str, Any]) -> None:
    cache = _mapping(compiler.get("cache"), "NRIR-29 compiler cache")
    templates = _list(cache.get("templates"), "NRIR-29 compiler templates")
    events = _list(cache.get("events"), "NRIR-29 compiler events")
    instances = _list(compiler.get("instances"), "NRIR-29 compiler instances")
    if (
        len(templates) != 1
        or cache.get("template_count") != 1
        or cache.get("instance_count") != len(events)
        or cache.get("miss_count") != 1
        or cache.get("hit_count") != len(events) - 1
        or len(events) != len(instances)
    ):
        raise ValueError("NRIR-29 compiler cache accounting differs")
    template = _mapping(templates[0], "NRIR-29 compiler template")
    template_ir = _mapping(template.get("template"), "NRIR-29 template IR")
    task_ir = _mapping(template.get("task_ir"), "NRIR-29 template Task IR")
    schedule = _mapping(template.get("schedule"), "NRIR-29 template Schedule IR")
    if (
        canonical_hash(template_ir) != template.get("template_hash")
        or canonical_hash(task_ir) != template.get("task_hash")
        or canonical_hash(schedule) != template.get("schedule_hash")
        or template_ir.get("cache_key")
        != canonical_hash(
            {
                key: value
                for key, value in template_ir.items()
                if key not in {"template_id", "cache_key", "performance_claimed"}
            }
        )
    ):
        raise ValueError("NRIR-29 compiler source-to-IR digest differs")
    template_hash = template["template_hash"]
    cache_key = template_ir["cache_key"]
    for index, (event_value, instance_value) in enumerate(zip(events, instances)):
        event = _mapping(event_value, "NRIR-29 cache event")
        item = _mapping(instance_value, "NRIR-29 instance item")
        instance = _mapping(item.get("instance"), "NRIR-29 instance IR")
        if (
            event.get("event_index") != index
            or item.get("cache_event_hash") != canonical_hash(event)
            or item.get("instance_hash") != canonical_hash(instance)
            or item.get("batch_id") != event.get("batch_id")
            or instance.get("instance_id")
            != f"{item.get('batch_id')}:optimizer-instance"
            or event.get("template_hash") != template_hash
            or instance.get("template_hash") != template_hash
            or event.get("cache_key") != cache_key
            or instance.get("cache_key") != cache_key
            or item.get("template_hash") != template_hash
            or item.get("task_hash") != template.get("task_hash")
            or item.get("schedule_hash") != template.get("schedule_hash")
            or event.get("outcome")
            != ("miss_compiled" if index == 0 else "hit_exact_contract")
        ):
            raise ValueError("NRIR-29 compiler instance linkage differs")


def _validate_worker_result(result: Mapping[str, Any]) -> None:
    budget = _mapping(result.get("budget"), "NRIR-29 result budget")
    clauses = _list(result.get("clauses"), "NRIR-29 result clauses")
    expected_budget = next(
        (
            (budget_id, max_nodes, max_depth)
            for budget_id, max_nodes, max_depth in BUDGET_SPECS
            if budget_id == budget.get("budget_id")
        ),
        None,
    )
    if (
        result.get("schema_version") != WORKER_SCHEMA_VERSION
        or result.get("execution_mode") != "production_parametric_template_instance_v2"
        or result.get("performance_claimed") is not False
        or expected_budget
        != (budget.get("budget_id"), budget.get("max_nodes"), budget.get("max_depth"))
        or result.get("solver_status") not in {"verified", "unsafe", "unknown"}
        or not isinstance(result.get("completed_clause_count"), int)
        or not clauses
        or len(clauses) != result.get("completed_clause_count")
        or any(
            not isinstance(result.get(name), int) or int(result[name]) < 0
            for name in ("setup_ns", "execution_ns", "worker_elapsed_ns")
        )
        or not _sha256(result.get("semantic_signature_hash"))
    ):
        raise ValueError("NRIR-29 worker result header differs")
    semantic_clauses = []
    for expected_index, value in enumerate(clauses):
        clause = _mapping(value, "NRIR-29 clause")
        domains = _list(clause.get("domains"), "NRIR-29 domains")
        pruned = _list(clause.get("sound_pruned_leaves"), "NRIR-29 pruned leaves")
        unresolved = _list(clause.get("unresolved_leaves"), "NRIR-29 unresolved leaves")
        split_hashes = [
            _mapping(item, "NRIR-29 domain").get("split_state_hash") for item in domains
        ]
        if (
            clause.get("clause_index") != expected_index
            or clause.get("status") not in {"verified", "unsafe", "unknown"}
            or clause.get("queue_status") not in {"complete", "budget_exhausted"}
            or clause.get("evaluated_nodes") != len(domains)
            or not domains
            or len(split_hashes) != len(set(split_hashes))
            or any(not _sha256(value) for value in split_hashes)
            or any(
                _mapping(item, "NRIR-29 domain").get("depth", -1)
                > int(budget["max_depth"])
                for item in domains
            )
            or len(domains) > int(budget["max_nodes"])
            or (clause.get("status") == "verified" and unresolved)
            or (clause.get("status") == "unknown" and not unresolved)
            or not _sha256(clause.get("logical_queue_signature_hash"))
            or not _sha256(clause.get("queue_trace_hash"))
            or not _sha256(clause.get("verdict_trace_hash"))
        ):
            raise ValueError("NRIR-29 clause/domain accounting differs")
        leaf_hashes = {
            _mapping(item, "NRIR-29 leaf").get("split_state_hash")
            for item in (*pruned, *unresolved)
        }
        if not leaf_hashes <= set(split_hashes):
            raise ValueError("NRIR-29 leaf/domain identity differs")
        semantic_clauses.append(
            {
                key: clause[key]
                for key in (
                    "clause_index",
                    "status",
                    "queue_status",
                    "termination_reason",
                    "evaluated_nodes",
                    "root_lower",
                    "root_upper",
                    "domains",
                    "sound_pruned_leaves",
                    "unresolved_leaves",
                    "worst_unresolved_lower",
                )
            }
        )
    semantic = {
        "solver_status": result["solver_status"],
        "completed_clause_count": result["completed_clause_count"],
        "completed_clause_indices": result["completed_clause_indices"],
        "unresolved_clause_indices": result["unresolved_clause_indices"],
        "pending_clause_indices": result["pending_clause_indices"],
        "verified_clause_indices": result["verified_clause_indices"],
        "clauses": semantic_clauses,
    }
    if canonical_hash(semantic) != result.get("semantic_signature_hash"):
        raise ValueError("NRIR-29 worker semantic digest differs")
    _validate_compiler(_mapping(result.get("compiler"), "NRIR-29 compiler"))


def _result_domains(
    result: Mapping[str, Any], clause_index: int
) -> dict[str, Mapping[str, Any]]:
    clauses = _list(result.get("clauses"), "NRIR-29 clauses")
    clause = _mapping(clauses[clause_index], "NRIR-29 clause")
    return {
        str(domain["split_state_hash"]): domain
        for value in _list(clause.get("domains"), "NRIR-29 domains")
        for domain in (_mapping(value, "NRIR-29 domain"),)
    }


def validate_scaling_group(results: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    """Validate one repeat's three budgets and return its closure projection."""

    by_budget = {
        str(_mapping(result["budget"], "NRIR-29 budget")["budget_id"]): result
        for result in results
    }
    budget_ids = tuple(item[0] for item in BUDGET_SPECS)
    if tuple(by_budget) != budget_ids:
        raise ValueError("NRIR-29 scaling group budget order differs")
    ordered = [by_budget[budget_id] for budget_id in budget_ids]
    for result in ordered:
        _validate_worker_result(result)
    completed = [
        set(int(value) for value in result["completed_clause_indices"])
        for result in ordered
    ]
    verified = [
        set(int(value) for value in result["verified_clause_indices"])
        for result in ordered
    ]
    no_completed_regression = all(
        left <= right for left, right in zip(completed, completed[1:])
    )
    no_verified_regression = all(
        left <= right for left, right in zip(verified, verified[1:])
    )
    domain_nested = True
    common_lower_max_diff = 0.0
    for left, right in zip(ordered, ordered[1:]):
        common_clauses = min(
            int(left["completed_clause_count"]),
            int(right["completed_clause_count"]),
        )
        for clause_index in range(common_clauses):
            left_domains = _result_domains(left, clause_index)
            right_domains = _result_domains(right, clause_index)
            if not set(left_domains) <= set(right_domains):
                domain_nested = False
                continue
            for split_hash, left_domain in left_domains.items():
                right_domain = right_domains[split_hash]
                difference = abs(
                    float(left_domain["lower"]) - float(right_domain["lower"])
                )
                common_lower_max_diff = max(common_lower_max_diff, difference)
    if common_lower_max_diff > COMMON_DOMAIN_LOWER_TOLERANCE:
        domain_nested = False
    strict_verified_increase = len(verified[-1]) > len(verified[0])
    all_completed = all(
        int(result["completed_clause_count"]) == 9
        and result["pending_clause_indices"] == []
        for result in ordered
    )
    return {
        "all_completed": all_completed,
        "no_completed_regression": no_completed_regression,
        "no_verified_regression": no_verified_regression,
        "strict_verified_increase": strict_verified_increase,
        "domain_nested": domain_nested,
        "common_domain_lower_max_diff": common_lower_max_diff,
        "verified_clause_indices": [sorted(value) for value in verified],
        "completed_clause_counts": [len(value) for value in completed],
    }


def _p90(values: list[int]) -> int:
    return sorted(values)[-1]


def _summaries(records: list[dict[str, Any]]) -> dict[str, object]:
    summaries: dict[str, object] = {}
    for workload_id in sorted({str(item["workload_id"]) for item in records}):
        budgets: dict[str, object] = {}
        for budget_id, _max_nodes, _max_depth in BUDGET_SPECS:
            selected = [
                item
                for item in records
                if item["workload_id"] == workload_id and item["budget_id"] == budget_id
            ]
            e2e = [int(item["e2e_elapsed_ns"]) for item in selected]
            execution = [int(item["result"]["execution_ns"]) for item in selected]
            node_totals = [
                sum(
                    int(clause["evaluated_nodes"])
                    for clause in item["result"]["clauses"]
                )
                for item in selected
            ]
            budgets[budget_id] = {
                "raw_e2e_ns": e2e,
                "median_e2e_ns": int(statistics.median(e2e)),
                "p90_e2e_ns": _p90(e2e),
                "raw_execution_ns": execution,
                "median_execution_ns": int(statistics.median(execution)),
                "p90_execution_ns": _p90(execution),
                "raw_evaluated_nodes": node_totals,
                "median_evaluated_nodes": int(statistics.median(node_totals)),
                "solver_statuses": [
                    item["result"]["solver_status"] for item in selected
                ],
                "completed_clause_counts": [
                    item["result"]["completed_clause_count"] for item in selected
                ],
                "verified_clause_indices": [
                    item["result"]["verified_clause_indices"] for item in selected
                ],
                "performance_claimed": True,
            }
        groups = []
        for repeat_index in range(REPEATS):
            group_records = [
                item
                for item in records
                if item["workload_id"] == workload_id
                and item["repeat_index"] == repeat_index
            ]
            group_records.sort(
                key=lambda item: tuple(value[0] for value in BUDGET_SPECS).index(
                    str(item["budget_id"])
                )
            )
            groups.append(
                validate_scaling_group(
                    [
                        _mapping(item["result"], "NRIR-29 group result")
                        for item in group_records
                    ]
                )
            )
        summaries[workload_id] = {
            "budgets": budgets,
            "repeat_gates": groups,
            "all_completed": all(bool(item["all_completed"]) for item in groups),
            "domain_nested": all(bool(item["domain_nested"]) for item in groups),
            "no_verified_regression": all(
                bool(item["no_verified_regression"]) for item in groups
            ),
            "strict_verified_increase": all(
                bool(item["strict_verified_increase"]) for item in groups
            ),
            "performance_claimed": True,
        }
    return summaries


def _property_status(summaries: Mapping[str, Any]) -> str:
    workload_values = [
        _mapping(value, "NRIR-29 workload summary") for value in summaries.values()
    ]
    no_regression = all(
        bool(value["all_completed"])
        and bool(value["domain_nested"])
        and bool(value["no_verified_regression"])
        for value in workload_values
    )
    strict_increase = any(
        bool(value["strict_verified_increase"]) for value in workload_values
    )
    return (
        "validated_reduced" if no_regression and strict_increase else "validated_no_go"
    )


def _experiment_ir_dict(plan: Any, task_ir: Any, schedule: Any) -> dict[str, object]:
    return {
        "plan": plan.to_dict(),
        "plan_hash": plan.stable_hash(),
        "task_ir": task_ir.to_dict(),
        "task_ir_hash": task_ir.stable_hash(plan),
        "schedule": schedule.to_dict(),
        "schedule_hash": schedule.stable_hash(plan, task_ir),
    }


def validate_evidence_structure(evidence: Mapping[str, Any]) -> None:
    if (
        evidence.get("schema_version") != EVIDENCE_SCHEMA_VERSION
        or evidence.get("status") != "ok"
        or evidence.get("property_status")
        not in {"validated_reduced", "validated_no_go"}
        or evidence.get("performance_claimed") is not True
    ):
        raise ValueError("NRIR-29 evidence header differs")
    records = [
        dict(_mapping(value, "NRIR-29 record"))
        for value in _list(evidence.get("records"), "NRIR-29 records")
    ]
    if len(records) != len(WORKLOAD_ROWS) * len(BUDGET_SPECS) * REPEATS:
        raise ValueError("NRIR-29 record count differs")
    experiment_ir = _mapping(evidence.get("experiment_ir"), "NRIR-29 experiment IR")
    plan_payload = _mapping(experiment_ir.get("plan"), "NRIR-29 Plan IR")
    task_payload = _mapping(experiment_ir.get("task_ir"), "NRIR-29 Task IR")
    schedule_payload = _mapping(experiment_ir.get("schedule"), "NRIR-29 Schedule IR")
    if (
        canonical_hash(plan_payload) != experiment_ir.get("plan_hash")
        or canonical_hash(task_payload) != experiment_ir.get("task_ir_hash")
        or canonical_hash(schedule_payload) != experiment_ir.get("schedule_hash")
    ):
        raise ValueError("NRIR-29 experiment IR digest differs")
    expected_task_ids = _list(
        schedule_payload.get("ordered_task_ids"), "NRIR-29 ordered tasks"
    )
    if len(expected_task_ids) != len(records):
        raise ValueError("NRIR-29 Schedule/record coverage differs")
    for expected_task_id, record in zip(expected_task_ids, records):
        result = _mapping(record.get("result"), "NRIR-29 result")
        _validate_worker_result(result)
        if (
            record.get("task_id") != expected_task_id
            or record.get("task_hash")
            != canonical_hash(
                next(
                    item
                    for item in _list(task_payload.get("tasks"), "NRIR-29 tasks")
                    if _mapping(item, "NRIR-29 task").get("task_id") == expected_task_id
                )
            )
            or not isinstance(record.get("e2e_elapsed_ns"), int)
            or int(record["e2e_elapsed_ns"]) <= 0
            or not _sha256(record.get("log_sha256"))
            or record.get("workload_id") != result.get("workload_id")
            or record.get("budget_id")
            != _mapping(result.get("budget"), "NRIR-29 budget").get("budget_id")
        ):
            raise ValueError("NRIR-29 Task/record binding differs")
    for workload_id in sorted({str(item["workload_id"]) for item in records}):
        for budget_id, _max_nodes, _max_depth in BUDGET_SPECS:
            selected = [
                _mapping(item["result"], "NRIR-29 repeat result")
                for item in records
                if item["workload_id"] == workload_id and item["budget_id"] == budget_id
            ]
            if (
                len(selected) != REPEATS
                or len({str(item["semantic_signature_hash"]) for item in selected}) != 1
            ):
                raise ValueError("NRIR-29 repeated semantics differ")
    summaries = _mapping(evidence.get("summaries"), "NRIR-29 summaries")
    recomputed = _summaries(records)
    if dict(summaries) != recomputed:
        raise ValueError("NRIR-29 summary replay differs")
    if evidence.get("property_status") != _property_status(recomputed):
        raise ValueError("NRIR-29 closure gate differs")


def _build_evidence(
    args: argparse.Namespace,
) -> tuple[dict[str, object], dict[str, str]]:
    plan, task_ir, schedule, workloads = _build_experiment_ir(
        args.benchmark_root, args.torch_threads
    )
    workload_by_id = {str(item["workload_id"]): item for item in workloads}
    budget_by_id = {budget.budget_id: budget.to_dict() for budget in plan.budgets}
    task_by_id = {item.task_id: item for item in task_ir.tasks}
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    (args.artifact_dir / "logs").mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    files: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="boundflow-nrir29-") as temporary:
        temporary_root = Path(temporary)
        for task_id in schedule.ordered_task_ids:
            task = task_by_id[task_id]
            workload = workload_by_id[task.workload_id]
            budget = budget_by_id[task.budget_id]
            stem = (
                f"{task.workload_id.replace(':', '-')}-r{task.repeat_index}-"
                f"{task.budget_id}"
            )
            result_path = temporary_root / f"{stem}.json"
            result, log, elapsed_ns = _run_worker(
                workload=workload,
                budget=budget,
                result_path=result_path,
                torch_threads=args.torch_threads,
            )
            _validate_worker_result(result)
            log_name = f"logs/{stem}.log"
            log_path = args.artifact_dir / log_name
            log_path.write_text(log, encoding="utf-8")
            files[log_name] = file_sha256(log_path)
            records.append(
                {
                    "task_id": task_id,
                    "task_hash": task.stable_hash(),
                    "workload_id": task.workload_id,
                    "repeat_index": task.repeat_index,
                    "group_order_index": task.group_order_index,
                    "budget_id": task.budget_id,
                    "e2e_elapsed_ns": elapsed_ns,
                    "log_path": log_name,
                    "log_sha256": files[log_name],
                    "result": result,
                }
            )
            print(
                _canonical_json(
                    {
                        "workload_id": task.workload_id,
                        "repeat_index": task.repeat_index,
                        "budget_id": task.budget_id,
                        "execution_ns": result["execution_ns"],
                        "completed": result["completed_clause_count"],
                        "verified": result["verified_clause_indices"],
                    }
                ),
                flush=True,
            )
    summaries = _summaries(records)
    evidence = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "ok",
        "property_status": _property_status(summaries),
        "performance_claimed": True,
        "claim_boundary": "three real VNN-COMP CPU workloads; same parametric algorithm and 60-second query deadline; repeated search-coverage/resource scaling only; no cross-budget speedup, competitor, GPU, full-property, or ASPLOS-ready claim",
        "source": {
            "vnncomp_commit": VNNCOMP_COMMIT,
            "native_code_revision": _code_revision(),
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
            "budgets": [budget.to_dict() for budget in plan.budgets],
            "budget_orders": [list(item) for item in schedule.budget_orders],
            "timing_boundary": plan.timing_boundary,
            "timeout_seconds": TIMEOUT_SECONDS,
            "search_steps": SEARCH_STEPS,
            "optimizer_steps": ALPHA_STEPS,
            "domain_nesting_identity": "clause_index_and_split_state_hash",
            "common_domain_lower_tolerance": COMMON_DOMAIN_LOWER_TOLERANCE,
        },
        "experiment_ir": _experiment_ir_dict(plan, task_ir, schedule),
        "records": records,
        "summaries": summaries,
        "limitations": [
            "The evidence compares search coverage across different budgets and does not report cross-budget speedup.",
            "All measurements are CPU-only on three selected VNN-COMP 2021 topology workloads; no GPU or external-verifier claim is made.",
            "Unknown outcomes remain valid and no complete-property or benchmark-suite closure is claimed.",
            "The 60-second deadline is checked between clauses; a final in-flight clause may complete after the deadline before the next check.",
            "The result does not by itself establish ASPLOS readiness or end-to-end competitor superiority.",
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
        "performance_claimed": True,
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


def _replay(args: argparse.Namespace) -> None:
    manifest = _load_json(args.artifact_dir / MANIFEST_FILE)
    evidence = _load_json(args.artifact_dir / EVIDENCE_FILE)
    files = _mapping(manifest.get("files"), "NRIR-29 manifest files")
    actual_files = {
        str(path.relative_to(args.artifact_dir)): file_sha256(path)
        for path in sorted(args.artifact_dir.rglob("*"))
        if path.is_file() and path.name != MANIFEST_FILE
    }
    if (
        manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION
        or manifest.get("status") != "ok"
        or manifest.get("performance_claimed") is not True
        or manifest.get("property_status") != evidence.get("property_status")
        or dict(files) != actual_files
        or manifest.get("evidence_hash") != canonical_hash(evidence)
    ):
        raise ValueError("NRIR-29 artifact manifest differs")
    source = _mapping(evidence.get("source"), "NRIR-29 source")
    if (
        source.get("vnncomp_commit") != VNNCOMP_COMMIT
        or source.get("native_code_revision") != _code_revision()
    ):
        raise ValueError("NRIR-29 source revision differs")
    plan, task_ir, schedule, _workloads = _build_experiment_ir(
        args.benchmark_root, args.torch_threads
    )
    if evidence.get("experiment_ir") != _experiment_ir_dict(plan, task_ir, schedule):
        raise ValueError("NRIR-29 source-to-experiment IR replay differs")
    validate_evidence_structure(evidence)
    for record_value in _list(evidence["records"], "NRIR-29 records"):
        record = _mapping(record_value, "NRIR-29 record")
        if (
            file_sha256(args.artifact_dir / str(record["log_path"]))
            != record["log_sha256"]
        ):
            raise ValueError("NRIR-29 log digest differs")
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
        raise ValueError("NRIR-29 torch thread count must be positive")
    if args.command == "worker":
        _worker(args)
    elif args.command == "generate":
        _generate(args)
    else:
        _replay(args)


if __name__ == "__main__":
    main()
