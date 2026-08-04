#!/usr/bin/env python3
"""Generate or replay the NRIR-19 native intermediate refinement artifact."""

# pylint: disable=too-many-lines,too-many-locals,too-many-statements
# pylint: disable=too-many-branches,too-many-arguments,duplicate-code
# pylint: disable=too-many-boolean-expressions,import-outside-toplevel,import-error
# pylint: disable=missing-function-docstring,line-too-long,protected-access
# pylint: disable=wrong-import-position

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, cast

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_multiworkload_competitor_e2e_artifact import (
    VNNCOMP_COMMIT,
    WORKLOAD_ROWS,
    _csv_selection,
    _onnx_inventory,
    canonical_hash,
    file_sha256,
)

ARTIFACT_SCHEMA_VERSION = "boundflow.native-intermediate-refinement-artifact/v1"
EVIDENCE_SCHEMA_VERSION = "boundflow.native-intermediate-refinement-evidence/v1"
WORKER_RESULT_SCHEMA_VERSION = "boundflow.native-intermediate-refinement-worker/v1"
MANIFEST_FILE = "manifest.json"
EVIDENCE_FILE = "evidence.json"
QUERY_TIMEOUT_SECONDS = 60
TORCH_THREADS = 8
ALPHA_STEPS = 5
SEARCH_STEPS = 4
MAX_NODES = 7
MAX_DEPTH = 2
REFINEMENT_POLICIES = {
    "mnistfc:000": {
        "passes": 1,
        "max_neurons_per_relu": 128,
        "backward_chunk_size": 32,
    },
    "cifar10_resnet:000": {
        "passes": 1,
        "max_neurons_per_relu": 16,
        "backward_chunk_size": 8,
    },
    "oval21:000": {
        "passes": 1,
        "max_neurons_per_relu": 128,
        "backward_chunk_size": 32,
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("generate", "replay"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--benchmark-root", type=Path, required=True)
        subparser.add_argument("--artifact-dir", type=Path, required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--mode", choices=("baseline", "native_refined"), required=True)
    worker.add_argument("--workload-id", required=True)
    worker.add_argument("--model", type=Path, required=True)
    worker.add_argument("--property", type=Path, required=True)
    worker.add_argument("--input-shape", type=int, nargs="+", required=True)
    worker.add_argument("--result-json", type=Path, required=True)
    return parser.parse_args()


def _canonical_json(value: object, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
        allow_nan=False,
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical_json(value, indent=2) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a list")
    return value


def _sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _git_revision(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def _repo_root() -> Path:
    return REPO_ROOT


def _native_code_revision() -> str:
    root = _repo_root()
    paths = (
        "boundflow/ir/bound.py",
        "boundflow/ir/refinement.py",
        "boundflow/runtime/crown_ibp.py",
        "boundflow/runtime/native_intermediate_refinement.py",
        "boundflow/runtime/native_alpha_beta_optimizer_schedule.py",
        "boundflow/runtime/native_optimized_relu_split_bab_runtime.py",
        "boundflow/runtime/complete_verifier_query.py",
        "scripts/run_native_intermediate_refinement_artifact.py",
    )
    return canonical_hash({path: file_sha256(root / path) for path in paths})


def _resolved_workloads(benchmark_root: Path) -> list[dict[str, object]]:
    if _git_revision(benchmark_root) != VNNCOMP_COMMIT:
        raise ValueError("NRIR-19 VNN-COMP source revision differs")
    from boundflow.frontends.vnnlib import import_vnnlib_box_query

    workloads: list[dict[str, object]] = []
    for definition in WORKLOAD_ROWS:
        csv_path, model_path, property_path = _csv_selection(benchmark_root, definition)
        input_shape, output_dim, onnx_ops = _onnx_inventory(model_path)
        workload_id = str(definition["workload_id"])
        query = import_vnnlib_box_query(property_path, query_id=workload_id)
        if (
            len(query.input_names) != int(math.prod(input_shape[1:]))
            or len(query.output_names) != output_dim
        ):
            raise ValueError("NRIR-19 ONNX/VNNLIB dimensions differ")
        workloads.append(
            {
                "workload_id": workload_id,
                "category": str(definition["category"]),
                "csv_path": csv_path,
                "model_path": model_path,
                "property_path": property_path,
                "input_shape": input_shape,
                "output_dim": output_dim,
                "onnx_ops": onnx_ops,
                "query_hash": query.stable_hash(),
                "csv_sha256": file_sha256(csv_path),
                "model_sha256": file_sha256(model_path),
                "property_sha256": file_sha256(property_path),
            }
        )
    return workloads


def _source_payload(workloads: list[dict[str, object]]) -> dict[str, object]:
    return {
        "vnncomp_commit": VNNCOMP_COMMIT,
        "native_code_revision": _native_code_revision(),
        "workloads": [
            {
                "workload_id": workload["workload_id"],
                "category": workload["category"],
                "csv_sha256": workload["csv_sha256"],
                "model_sha256": workload["model_sha256"],
                "property_sha256": workload["property_sha256"],
                "query_hash": workload["query_hash"],
                "input_shape": list(cast(tuple[int, ...], workload["input_shape"])),
                "output_dim": workload["output_dim"],
                "onnx_ops": list(cast(tuple[str, ...], workload["onnx_ops"])),
            }
            for workload in workloads
        ],
    }


def _policy_payload() -> dict[str, object]:
    return {
        "device": "cpu",
        "torch_threads": TORCH_THREADS,
        "query_timeout_seconds": QUERY_TIMEOUT_SECONDS,
        "alpha_steps": ALPHA_STEPS,
        "search_steps": SEARCH_STEPS,
        "max_nodes": MAX_NODES,
        "max_depth": MAX_DEPTH,
        "expansion_batch_size": 2,
        "max_eval_batch_size": 4,
        "baseline_intermediate_bound_source": "local_forward",
        "refined_intermediate_bound_source": "native_refined",
        "refinement_by_workload": REFINEMENT_POLICIES,
        "performance_claimed": False,
    }


def _worker_command(
    *,
    mode: str,
    workload: Mapping[str, object],
    result_path: Path,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--mode",
        mode,
        "--workload-id",
        str(workload["workload_id"]),
        "--model",
        str(workload["model_path"]),
        "--property",
        str(workload["property_path"]),
        "--input-shape",
        *(str(value) for value in cast(tuple[int, ...], workload["input_shape"])[1:]),
        "--result-json",
        str(result_path),
    ]


def _run_worker(
    *,
    mode: str,
    workload: Mapping[str, object],
    result_path: Path,
) -> tuple[dict[str, object], str, int, int]:
    started_ns = time.perf_counter_ns()
    completed = subprocess.run(
        _worker_command(mode=mode, workload=workload, result_path=result_path),
        cwd=_repo_root(),
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=QUERY_TIMEOUT_SECONDS + 45,
        check=False,
    )
    elapsed_ns = time.perf_counter_ns() - started_ns
    if completed.returncode != 0 or not result_path.is_file():
        raise RuntimeError(
            f"NRIR-19 {workload['workload_id']} {mode} worker failed "
            f"with {completed.returncode}: {completed.stdout[-8000:]}"
        )
    result = _load_json(result_path)
    return result, completed.stdout, completed.returncode, elapsed_ns


def _query_result(execution: Any) -> dict[str, object]:
    clauses = []
    for clause in execution.clauses:
        queue = clause.queue.trace
        root = queue.evaluations[0]
        clauses.append(
            {
                "clause_index": clause.trace.clause_index,
                "status": clause.trace.status,
                "root_lower": root.lower,
                "root_upper": root.upper,
                "evaluated_nodes": len(queue.evaluations),
                "final_frontier_nodes": len(queue.final_frontier_node_ids),
                "queue_trace_hash": queue.stable_hash(),
                "clause_trace_hash": canonical_hash(clause.trace.to_dict()),
            }
        )
    return {
        "solver_status": execution.trace.status,
        "query_trace_hash": execution.trace.stable_hash(),
        "completed_clause_count": len(execution.clauses),
        "unresolved_clause_indices": list(execution.trace.unresolved_clause_indices),
        "pending_clause_indices": list(execution.trace.pending_clause_indices),
        "clauses": clauses,
        "evaluated_node_count": sum(
            len(clause.queue.trace.evaluations) for clause in execution.clauses
        ),
    }


def _worker(args: argparse.Namespace) -> None:
    import torch

    from boundflow.frontends.onnx.frontend import import_onnx
    from boundflow.frontends.vnnlib import (
        import_vnnlib_box_query,
        materialize_vnnlib_box_query,
    )
    from boundflow.ir.bound import IntermediateBoundSource
    from boundflow.ir.refinement import NativeIntermediateRefinementPolicyIR
    from boundflow.planner import plan_interval_ibp_v0
    from boundflow.runtime.complete_verifier_query import (
        CompleteVerifierQueryPolicy,
        execute_complete_verifier_query,
    )
    from boundflow.runtime.native_alpha_beta_optimization_state import (
        NativeAlphaBetaOptimizerPolicy,
    )
    from boundflow.runtime.native_candidate_search import (
        NativeProjectedGradientSearchPolicy,
    )
    from boundflow.runtime.native_intermediate_refinement import (
        compile_native_intermediate_refinement_program,
        execute_native_intermediate_refinement_program,
    )
    from boundflow.runtime.native_relu_split_bab_runtime import (
        NativeReluSplitBabConfig,
    )
    from boundflow.runtime.task_executor import InputSpec

    torch.set_num_threads(TORCH_THREADS)
    started_ns = time.perf_counter_ns()
    query = import_vnnlib_box_query(args.property, query_id=args.workload_id)
    tensors = materialize_vnnlib_box_query(query, input_shape=tuple(args.input_shape))
    program = import_onnx(str(args.model), do_shape_infer=True, normalize=True)
    module = plan_interval_ibp_v0(program)
    input_spec = InputSpec.box(
        value_name=program.graph.inputs[0],
        lower=tensors.input_lower,
        upper=tensors.input_upper,
    )
    setup_ns = time.perf_counter_ns() - started_ns
    refinement_payload: dict[str, object] | None = None
    relu_pre_override = None
    source = IntermediateBoundSource.LOCAL_FORWARD
    refinement_ns = 0
    if args.mode == "native_refined":
        policy_values = REFINEMENT_POLICIES[args.workload_id]
        refinement_started_ns = time.perf_counter_ns()
        refinement_program = compile_native_intermediate_refinement_program(
            module,
            input_spec,
            policy=NativeIntermediateRefinementPolicyIR(
                passes=policy_values["passes"],
                max_neurons_per_relu=policy_values["max_neurons_per_relu"],
                backward_chunk_size=policy_values["backward_chunk_size"],
            ),
            plan_id=f"nrir19:{args.workload_id}",
        )
        refinement_execution = execute_native_intermediate_refinement_program(
            refinement_program, module, input_spec
        )
        refinement_ns = time.perf_counter_ns() - refinement_started_ns
        refinement_payload = {
            "plan": refinement_program.plan.to_dict(),
            "task": refinement_program.task_module.to_dict(
                plan=refinement_program.plan
            ),
            "schedule": refinement_program.schedule.to_dict(
                plan=refinement_program.plan,
                task_module=refinement_program.task_module,
            ),
            "hashes": refinement_program.hashes(),
            "execution_trace": refinement_execution.trace.to_dict(),
            "execution_trace_hash": refinement_execution.trace.stable_hash(),
        }
        relu_pre_override = refinement_execution.relu_pre
        source = IntermediateBoundSource.NATIVE_REFINED
        del refinement_execution, refinement_program
        gc.collect()
    query_started_ns = time.perf_counter_ns()
    execution = execute_complete_verifier_query(
        module,
        input_spec,
        linear_spec_C=tensors.linear_spec_c,
        thresholds=tensors.thresholds,
        query_id=f"{query.query_id}:{args.mode}",
        query_policy=CompleteVerifierQueryPolicy(
            timeout_ns=QUERY_TIMEOUT_SECONDS * 1_000_000_000
        ),
        search_policy=NativeProjectedGradientSearchPolicy(
            steps=SEARCH_STEPS, step_size=0.002
        ),
        queue_config=NativeReluSplitBabConfig(
            max_nodes=MAX_NODES,
            max_depth=MAX_DEPTH,
            expansion_batch_size=2,
            max_eval_batch_size=4,
        ),
        optimizer_policy=NativeAlphaBetaOptimizerPolicy(
            steps=ALPHA_STEPS,
            lr=0.1,
            alpha_initialization_mode="adaptive",
        ),
        relu_pre_override=relu_pre_override,
        intermediate_bound_source=source,
    )
    query_ns = time.perf_counter_ns() - query_started_ns
    result = {
        "schema_version": WORKER_RESULT_SCHEMA_VERSION,
        "workload_id": args.workload_id,
        "mode": args.mode,
        "execution_state": "completed",
        "query_ir_hash": query.stable_hash(),
        "setup_ns": setup_ns,
        "refinement_ns": refinement_ns,
        "query_ns": query_ns,
        "worker_elapsed_ns": time.perf_counter_ns() - started_ns,
        "intermediate_bound_source": source.value,
        "refinement": refinement_payload,
        "query": _query_result(execution),
        "performance_claimed": False,
    }
    _write_json(args.result_json, result)
    print(
        _canonical_json(
            {
                "status": "ok",
                "workload_id": args.workload_id,
                "mode": args.mode,
                "solver_status": execution.trace.status,
            }
        )
    )


def _comparison(
    baseline: Mapping[str, object], refined: Mapping[str, object]
) -> dict[str, object]:
    baseline_query = _mapping(baseline["query"], "baseline query")
    refined_query = _mapping(refined["query"], "refined query")
    baseline_clauses = {
        int(clause["clause_index"]): clause
        for clause in _list(baseline_query["clauses"], "baseline clauses")
    }
    refined_clauses = {
        int(clause["clause_index"]): clause
        for clause in _list(refined_query["clauses"], "refined clauses")
    }
    shared = sorted(set(baseline_clauses) & set(refined_clauses))
    root_deltas = [
        {
            "clause_index": index,
            "baseline_root_lower": baseline_clauses[index]["root_lower"],
            "refined_root_lower": refined_clauses[index]["root_lower"],
            "root_lower_delta": float(refined_clauses[index]["root_lower"])
            - float(baseline_clauses[index]["root_lower"]),
        }
        for index in shared
    ]
    closed = [
        index
        for index in shared
        if baseline_clauses[index]["status"] == "unknown"
        and refined_clauses[index]["status"] == "verified"
    ]
    return {
        "baseline_status": baseline_query["solver_status"],
        "refined_status": refined_query["solver_status"],
        "closed_clause_indices": closed,
        "baseline_unresolved_clause_indices": baseline_query[
            "unresolved_clause_indices"
        ],
        "refined_unresolved_clause_indices": refined_query["unresolved_clause_indices"],
        "baseline_pending_clause_indices": baseline_query["pending_clause_indices"],
        "refined_pending_clause_indices": refined_query["pending_clause_indices"],
        "baseline_evaluated_node_count": baseline_query["evaluated_node_count"],
        "refined_evaluated_node_count": refined_query["evaluated_node_count"],
        "root_lower_deltas": root_deltas,
        "performance_claimed": False,
    }


def _build_evidence(
    benchmark_root: Path, artifact_dir: Path
) -> tuple[dict[str, object], dict[str, str]]:
    workloads = _resolved_workloads(benchmark_root)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    files: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="boundflow-nrir19-") as temporary:
        temp_root = Path(temporary)
        for workload in workloads:
            for mode in ("baseline", "native_refined"):
                result_path = temp_root / f"{workload['workload_id']}-{mode}.json"
                result, log, returncode, e2e_ns = _run_worker(
                    mode=mode,
                    workload=workload,
                    result_path=result_path,
                )
                log_path = (
                    artifact_dir
                    / "logs"
                    / f"{str(workload['workload_id']).replace(':', '-')}-{mode}.log"
                )
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text(log, encoding="utf-8")
                relative_log = str(log_path.relative_to(artifact_dir))
                files[relative_log] = file_sha256(log_path)
                records.append(
                    {
                        "workload_id": workload["workload_id"],
                        "mode": mode,
                        "process_returncode": returncode,
                        "e2e_elapsed_ns": e2e_ns,
                        "log_path": relative_log,
                        "log_sha256": files[relative_log],
                        "result": result,
                    }
                )
    by_identity = {
        (record["workload_id"], record["mode"]): record["result"] for record in records
    }
    comparisons = [
        {
            "workload_id": workload["workload_id"],
            **_comparison(
                _mapping(
                    by_identity[(workload["workload_id"], "baseline")],
                    "baseline result",
                ),
                _mapping(
                    by_identity[(workload["workload_id"], "native_refined")],
                    "refined result",
                ),
            ),
        }
        for workload in workloads
    ]
    evidence = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "ok",
        "claim_boundary": (
            "native selected-CROWN intermediate refinement on three real VNN-COMP "
            "CPU workloads; same-policy baseline/refined comparison; diagnostic "
            "timings only; no speedup claim; ASPLOS-ready remains evidence-dependent"
        ),
        "performance_claimed": False,
        "source": _source_payload(workloads),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device": "cpu",
            "torch_threads": TORCH_THREADS,
        },
        "policy": _policy_payload(),
        "records": records,
        "comparisons": comparisons,
        "limitations": [
            "CPU timings are one fresh-process observation per workload/mode and are not performance claims.",
            "Refinement is root-global and reused soundly in child domains; per-child recomputation is not implemented.",
            "The fixed 7-node/depth-2 complete-verifier budget may still return unknown or hit the query deadline.",
            "Target count is workload-bounded; unselected ambiguous neurons retain sound forward bounds.",
            "CUDA execution and timing remain pending because the generating host has no usable driver/device.",
        ],
    }
    return evidence, files


def _validate_worker_result(result: Mapping[str, Any]) -> None:
    mode = result.get("mode")
    query = _mapping(result.get("query"), "NRIR-19 worker query")
    refinement = result.get("refinement")
    if (
        result.get("schema_version") != WORKER_RESULT_SCHEMA_VERSION
        or mode not in {"baseline", "native_refined"}
        or result.get("execution_state") != "completed"
        or not _sha256(result.get("query_ir_hash"))
        or result.get("performance_claimed") is not False
        or query.get("solver_status") not in {"verified", "unsafe", "unknown"}
        or not _sha256(query.get("query_trace_hash"))
        or not isinstance(query.get("clauses"), list)
        or not isinstance(query.get("evaluated_node_count"), int)
        or int(query["evaluated_node_count"]) < 1
    ):
        raise ValueError("NRIR-19 worker result differs")
    if mode == "baseline":
        if (
            refinement is not None
            or result.get("intermediate_bound_source") != "local_forward"
            or result.get("refinement_ns") != 0
        ):
            raise ValueError("NRIR-19 baseline result differs")
    else:
        payload = _mapping(refinement, "NRIR-19 refinement payload")
        hashes = _mapping(payload.get("hashes"), "NRIR-19 refinement hashes")
        plan = _mapping(payload.get("plan"), "NRIR-19 refinement Plan")
        task = _mapping(payload.get("task"), "NRIR-19 refinement Task")
        schedule = _mapping(payload.get("schedule"), "NRIR-19 refinement Schedule")
        trace = _mapping(payload.get("execution_trace"), "NRIR-19 refinement trace")
        if (
            result.get("intermediate_bound_source") != "native_refined"
            or not isinstance(result.get("refinement_ns"), int)
            or int(result["refinement_ns"]) <= 0
            or hashes.get("refinement_plan_hash") != canonical_hash(plan)
            or hashes.get("refinement_task_module_hash") != canonical_hash(task)
            or hashes.get("refinement_schedule_hash") != canonical_hash(schedule)
            or task.get("refinement_plan_hash") != hashes.get("refinement_plan_hash")
            or schedule.get("refinement_plan_hash")
            != hashes.get("refinement_plan_hash")
            or schedule.get("refinement_task_module_hash")
            != hashes.get("refinement_task_module_hash")
            or trace.get("plan_hash") != hashes.get("refinement_plan_hash")
            or trace.get("task_module_hash")
            != hashes.get("refinement_task_module_hash")
            or trace.get("schedule_hash") != hashes.get("refinement_schedule_hash")
            or payload.get("execution_trace_hash") != canonical_hash(trace)
        ):
            raise ValueError("NRIR-19 refinement IR/trace linkage differs")


def validate_evidence_structure(evidence: Mapping[str, Any]) -> None:
    if (
        evidence.get("schema_version") != EVIDENCE_SCHEMA_VERSION
        or evidence.get("status") != "ok"
        or evidence.get("performance_claimed") is not False
    ):
        raise ValueError("NRIR-19 evidence header differs")
    claim = str(evidence.get("claim_boundary", ""))
    for phrase in ("three real VNN-COMP", "no speedup", "ASPLOS-ready"):
        if phrase not in claim:
            raise ValueError("NRIR-19 claim boundary differs")
    source = _mapping(evidence.get("source"), "NRIR-19 source")
    if (
        source.get("vnncomp_commit") != VNNCOMP_COMMIT
        or not _sha256(source.get("native_code_revision"))
        or len(_list(source.get("workloads"), "NRIR-19 workloads")) != 3
    ):
        raise ValueError("NRIR-19 source identity differs")
    records = _list(evidence.get("records"), "NRIR-19 records")
    identities: set[tuple[str, str]] = set()
    for record in records:
        item = _mapping(record, "NRIR-19 record")
        identity = (str(item.get("workload_id")), str(item.get("mode")))
        result = _mapping(item.get("result"), "NRIR-19 result")
        _validate_worker_result(result)
        if (
            identity in identities
            or result.get("workload_id") != identity[0]
            or result.get("mode") != identity[1]
            or item.get("process_returncode") != 0
            or not isinstance(item.get("e2e_elapsed_ns"), int)
            or int(item["e2e_elapsed_ns"]) <= 0
            or not str(item.get("log_path", "")).startswith("logs/")
            or not _sha256(item.get("log_sha256"))
        ):
            raise ValueError("NRIR-19 execution record differs")
        identities.add(identity)
    expected = {
        (str(workload["workload_id"]), mode)
        for workload in WORKLOAD_ROWS
        for mode in ("baseline", "native_refined")
    }
    if identities != expected:
        raise ValueError("NRIR-19 workload/mode coverage differs")
    comparisons = _list(evidence.get("comparisons"), "NRIR-19 comparisons")
    if (
        len(comparisons) != 3
        or {item["workload_id"] for item in comparisons}
        != {str(workload["workload_id"]) for workload in WORKLOAD_ROWS}
        or any(item.get("performance_claimed") is not False for item in comparisons)
    ):
        raise ValueError("NRIR-19 comparison coverage differs")
    limitations = evidence.get("limitations")
    if not isinstance(limitations, list) or len(limitations) != 5:
        raise ValueError("NRIR-19 limitation ledger differs")


def _generate(args: argparse.Namespace) -> None:
    evidence, files = _build_evidence(args.benchmark_root, args.artifact_dir)
    validate_evidence_structure(evidence)
    evidence_path = args.artifact_dir / EVIDENCE_FILE
    _write_json(evidence_path, evidence)
    files[EVIDENCE_FILE] = file_sha256(evidence_path)
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "ok",
        "performance_claimed": False,
        "files": dict(sorted(files.items())),
        "evidence_hash": canonical_hash(evidence),
    }
    _write_json(args.artifact_dir / MANIFEST_FILE, manifest)
    print(_canonical_json({"status": "ok", "evidence_hash": manifest["evidence_hash"]}))


def _recompile_refinement_ir(
    benchmark_root: Path, evidence: Mapping[str, Any]
) -> dict[str, dict[str, str]]:
    import torch

    from boundflow.frontends.onnx.frontend import import_onnx
    from boundflow.frontends.vnnlib import (
        import_vnnlib_box_query,
        materialize_vnnlib_box_query,
    )
    from boundflow.ir.refinement import NativeIntermediateRefinementPolicyIR
    from boundflow.planner import plan_interval_ibp_v0
    from boundflow.runtime.native_intermediate_refinement import (
        compile_native_intermediate_refinement_program,
    )
    from boundflow.runtime.task_executor import InputSpec

    torch.set_num_threads(TORCH_THREADS)
    workloads = _resolved_workloads(benchmark_root)
    expected_by_workload = {
        str(record["workload_id"]): _mapping(
            _mapping(record["result"], "result")["refinement"], "refinement"
        )
        for record in _list(evidence["records"], "records")
        if record["mode"] == "native_refined"
    }
    actual: dict[str, dict[str, str]] = {}
    for workload in workloads:
        workload_id = str(workload["workload_id"])
        query = import_vnnlib_box_query(
            cast(Path, workload["property_path"]), query_id=workload_id
        )
        input_shape = cast(tuple[int, ...], workload["input_shape"])
        tensors = materialize_vnnlib_box_query(query, input_shape=input_shape[1:])
        primal = import_onnx(
            str(workload["model_path"]), do_shape_infer=True, normalize=True
        )
        module = plan_interval_ibp_v0(primal)
        input_spec = InputSpec.box(
            value_name=primal.graph.inputs[0],
            lower=tensors.input_lower,
            upper=tensors.input_upper,
        )
        program = compile_native_intermediate_refinement_program(
            module,
            input_spec,
            policy=NativeIntermediateRefinementPolicyIR(
                passes=REFINEMENT_POLICIES[workload_id]["passes"],
                max_neurons_per_relu=REFINEMENT_POLICIES[workload_id][
                    "max_neurons_per_relu"
                ],
                backward_chunk_size=REFINEMENT_POLICIES[workload_id][
                    "backward_chunk_size"
                ],
            ),
            plan_id=f"nrir19:{workload_id}",
        )
        expected = expected_by_workload[workload_id]
        if (
            program.plan.to_dict() != expected["plan"]
            or program.task_module.to_dict(plan=program.plan) != expected["task"]
            or program.schedule.to_dict(
                plan=program.plan, task_module=program.task_module
            )
            != expected["schedule"]
        ):
            raise ValueError("NRIR-19 source-to-refinement IR replay differs")
        actual[workload_id] = program.hashes()
    return actual


def _replay(args: argparse.Namespace) -> None:
    manifest = _load_json(args.artifact_dir / MANIFEST_FILE)
    evidence = _load_json(args.artifact_dir / EVIDENCE_FILE)
    actual_files = {
        str(path.relative_to(args.artifact_dir)): file_sha256(path)
        for path in sorted(args.artifact_dir.rglob("*"))
        if path.is_file() and path.name != MANIFEST_FILE
    }
    if (
        manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION
        or manifest.get("status") != "ok"
        or manifest.get("performance_claimed") is not False
        or manifest.get("files") != actual_files
        or manifest.get("evidence_hash") != canonical_hash(evidence)
    ):
        raise ValueError("NRIR-19 artifact manifest differs")
    validate_evidence_structure(evidence)
    if (
        _mapping(evidence["source"], "source").get("native_code_revision")
        != _native_code_revision()
    ):
        raise ValueError("NRIR-19 native code revision differs")
    hashes = _recompile_refinement_ir(args.benchmark_root, evidence)
    print(
        _canonical_json(
            {
                "status": "ok",
                "evidence_hash": manifest["evidence_hash"],
                "recompiled_refinement_ir_hashes": hashes,
            }
        )
    )


def main() -> None:
    args = _parse_args()
    if args.command == "generate":
        _generate(args)
    elif args.command == "replay":
        _replay(args)
    else:
        _worker(args)


if __name__ == "__main__":
    main()
