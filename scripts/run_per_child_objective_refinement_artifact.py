#!/usr/bin/env python3
"""Generate or replay the NRIR-21 per-child refinement artifact."""

# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=too-many-boolean-expressions,duplicate-code
# pylint: disable=import-outside-toplevel,import-error,wrong-import-position
# pylint: disable=missing-function-docstring,line-too-long,protected-access

from __future__ import annotations

import argparse
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
from scripts.run_native_intermediate_refinement_artifact import (
    _canonical_json,
    _git_revision,
    _load_json,
    _mapping,
    _sha256,
    _write_json,
)

ARTIFACT_SCHEMA_VERSION = "boundflow.per-child-objective-refinement-artifact/v1"
EVIDENCE_SCHEMA_VERSION = "boundflow.per-child-objective-refinement-evidence/v1"
WORKER_SCHEMA_VERSION = "boundflow.per-child-objective-refinement-worker/v1"
MANIFEST_FILE = "manifest.json"
EVIDENCE_FILE = "evidence.json"
WORKLOAD_ID = "cifar10_resnet:000"
CLAUSE_INDICES = (0, 1)
MODES = ("root_global", "per_child")
PER_CHILD_STRATEGIES = (
    "independent_exact_split_v1",
    "ancestral_constraint_carry_v1",
)
TORCH_THREADS = 8
WORKER_TIMEOUT_SECONDS = 180
ALPHA_STEPS = 5
MAX_NODES = 7
MAX_DEPTH = 2
TARGETS_PER_RELU = 16
BACKWARD_CHUNK_SIZE = 8


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("generate", "replay"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--benchmark-root", type=Path, required=True)
        subparser.add_argument("--artifact-dir", type=Path, required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--mode", choices=MODES, required=True)
    worker.add_argument("--clause-index", type=int, required=True)
    worker.add_argument("--model", type=Path, required=True)
    worker.add_argument("--property", type=Path, required=True)
    worker.add_argument("--input-shape", type=int, nargs="+", required=True)
    worker.add_argument("--result-json", type=Path, required=True)
    worker.add_argument(
        "--per-child-strategy",
        choices=PER_CHILD_STRATEGIES,
        default="independent_exact_split_v1",
    )
    return parser.parse_args()


def _native_code_revision() -> str:
    paths = (
        "boundflow/ir/refinement.py",
        "boundflow/runtime/crown_ibp.py",
        "boundflow/runtime/native_intermediate_refinement.py",
        "boundflow/runtime/native_alpha_beta_optimization_state.py",
        "boundflow/runtime/native_optimized_relu_split_bab_runtime.py",
        "scripts/run_per_child_objective_refinement_artifact.py",
    )
    return canonical_hash({path: file_sha256(REPO_ROOT / path) for path in paths})


def _resolved_source(benchmark_root: Path) -> dict[str, object]:
    if _git_revision(benchmark_root) != VNNCOMP_COMMIT:
        raise ValueError("NRIR-21 VNN-COMP source revision differs")
    from boundflow.frontends.vnnlib import import_vnnlib_box_query

    definition = next(
        item for item in WORKLOAD_ROWS if item["workload_id"] == WORKLOAD_ID
    )
    csv_path, model_path, property_path = _csv_selection(benchmark_root, definition)
    input_shape, output_dim, onnx_ops = _onnx_inventory(model_path)
    query = import_vnnlib_box_query(property_path, query_id=WORKLOAD_ID)
    return {
        "vnncomp_commit": VNNCOMP_COMMIT,
        "native_code_revision": _native_code_revision(),
        "workload_id": WORKLOAD_ID,
        "csv_path": csv_path,
        "model_path": model_path,
        "property_path": property_path,
        "csv_sha256": file_sha256(csv_path),
        "model_sha256": file_sha256(model_path),
        "property_sha256": file_sha256(property_path),
        "query_hash": query.stable_hash(),
        "input_shape": input_shape,
        "output_dim": output_dim,
        "onnx_ops": onnx_ops,
    }


def _source_payload(source: Mapping[str, object]) -> dict[str, object]:
    return {
        key: (list(value) if isinstance(value, tuple) else value)
        for key, value in source.items()
        if key not in {"csv_path", "model_path", "property_path"}
    }


def _policy_payload() -> dict[str, object]:
    return {
        "device": "cpu",
        "torch_threads": TORCH_THREADS,
        "worker_timeout_seconds": WORKER_TIMEOUT_SECONDS,
        "alpha_steps": ALPHA_STEPS,
        "max_nodes": MAX_NODES,
        "max_depth": MAX_DEPTH,
        "expansion_batch_size": 2,
        "max_eval_batch_size": 4,
        "targets_per_relu": TARGETS_PER_RELU,
        "backward_chunk_size": BACKWARD_CHUNK_SIZE,
        "candidate_policy_id": "objective_influence_width_per_relu_v1",
        "selection_score": "ambiguous_width*max(abs(A_u),abs(A_l))",
        "comparison": "same-tree-budget-root-global-vs-per-child",
        "performance_claimed": False,
    }


def _worker_command(
    *,
    mode: str,
    clause_index: int,
    source: Mapping[str, object],
    result: Path,
    per_child_strategy: str = "independent_exact_split_v1",
) -> list[str]:
    input_shape = cast(tuple[int, ...], source["input_shape"])
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--mode",
        mode,
        "--clause-index",
        str(clause_index),
        "--model",
        str(source["model_path"]),
        "--property",
        str(source["property_path"]),
        "--input-shape",
        *(str(value) for value in input_shape[1:]),
        "--result-json",
        str(result),
        "--per-child-strategy",
        per_child_strategy,
    ]


def _run_worker(
    *,
    mode: str,
    clause_index: int,
    source: Mapping[str, object],
    result: Path,
    per_child_strategy: str = "independent_exact_split_v1",
) -> tuple[dict[str, Any], str, int, int]:
    started_ns = time.perf_counter_ns()
    completed = subprocess.run(
        _worker_command(
            mode=mode,
            clause_index=clause_index,
            source=source,
            result=result,
            per_child_strategy=per_child_strategy,
        ),
        cwd=REPO_ROOT,
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=WORKER_TIMEOUT_SECONDS,
        check=False,
    )
    elapsed_ns = time.perf_counter_ns() - started_ns
    if completed.returncode != 0 or not result.is_file():
        raise RuntimeError(
            f"NRIR-21 clause {clause_index} {mode} worker failed with "
            f"{completed.returncode}: {completed.stdout[-8000:]}"
        )
    return _load_json(result), completed.stdout, completed.returncode, elapsed_ns


def _refinement_policy() -> Any:
    from boundflow.ir.refinement import NativeIntermediateRefinementPolicyIR

    return NativeIntermediateRefinementPolicyIR(
        passes=1,
        max_neurons_per_relu=TARGETS_PER_RELU,
        backward_chunk_size=BACKWARD_CHUNK_SIZE,
        candidate_policy_id="objective_influence_width_per_relu_v1",
    )


def _serialize_refinement(node_id: str, execution: Any) -> dict[str, object]:
    program = execution.program
    trace = execution.trace.to_dict()
    deterministic_trace = dict(trace)
    deterministic_trace.pop("elapsed_ns")
    return {
        "node_id": node_id,
        "plan": program.plan.to_dict(),
        "task": program.task_module.to_dict(plan=program.plan),
        "schedule": program.schedule.to_dict(
            plan=program.plan, task_module=program.task_module
        ),
        "hashes": program.hashes(),
        "execution_trace": trace,
        "semantic_execution_trace_hash": canonical_hash(deterministic_trace),
    }


def _worker(args: argparse.Namespace) -> None:
    import torch

    from boundflow.frontends.onnx.frontend import import_onnx
    from boundflow.frontends.vnnlib import (
        import_vnnlib_box_query,
        materialize_vnnlib_box_query,
    )
    from boundflow.ir.bound import IntermediateBoundSource
    from boundflow.planner import plan_interval_ibp_v0
    from boundflow.runtime.native_alpha_beta_optimization_state import (
        NativeAlphaBetaOptimizerPolicy,
    )
    from boundflow.runtime.native_intermediate_refinement import (
        compile_native_intermediate_refinement_program,
        execute_native_intermediate_refinement_program,
    )
    from boundflow.runtime.native_optimized_relu_split_bab_runtime import (
        execute_native_optimized_relu_split_bab,
    )
    from boundflow.runtime.native_relu_split_bab_runtime import (
        NativeReluSplitBabConfig,
    )
    from boundflow.runtime.task_executor import InputSpec

    if args.clause_index not in CLAUSE_INDICES:
        raise ValueError("NRIR-21 clause index is outside the frozen set")
    torch.set_num_threads(TORCH_THREADS)
    started_ns = time.perf_counter_ns()
    query = import_vnnlib_box_query(args.property, query_id=WORKLOAD_ID)
    tensors = materialize_vnnlib_box_query(query, input_shape=tuple(args.input_shape))
    primal = import_onnx(str(args.model), do_shape_infer=True, normalize=True)
    module = plan_interval_ibp_v0(primal)
    input_spec = InputSpec.box(
        value_name=primal.graph.inputs[0],
        lower=tensors.input_lower,
        upper=tensors.input_upper,
    )
    objective = tensors.linear_spec_c[:, args.clause_index, :].contiguous()
    config = NativeReluSplitBabConfig(
        max_nodes=MAX_NODES,
        max_depth=MAX_DEPTH,
        expansion_batch_size=2,
        max_eval_batch_size=4,
        threshold=float(tensors.thresholds[args.clause_index]),
    )
    optimizer_policy = NativeAlphaBetaOptimizerPolicy(
        steps=ALPHA_STEPS,
        lr=0.1,
        alpha_initialization_mode="adaptive",
    )
    queue_started_ns = time.perf_counter_ns()
    if args.mode == "root_global":
        root_program = compile_native_intermediate_refinement_program(
            module,
            input_spec,
            policy=_refinement_policy(),
            plan_id=f"nrir21:{WORKLOAD_ID}:clause:{args.clause_index:04d}:root",
            linear_spec_C=objective,
        )
        root_execution = execute_native_intermediate_refinement_program(
            root_program, module, input_spec
        )
        queue = execute_native_optimized_relu_split_bab(
            module,
            input_spec,
            linear_spec_C=objective,
            run_id=f"nrir21:{args.clause_index:04d}",
            config=config,
            optimizer_policy=optimizer_policy,
            relu_pre_override=root_execution.relu_pre,
            intermediate_bound_source=IntermediateBoundSource.NATIVE_REFINED,
        )
        refinements = [
            _serialize_refinement(
                queue.trace.evaluations[0].node.node_id, root_execution
            )
        ]
    else:
        queue = execute_native_optimized_relu_split_bab(
            module,
            input_spec,
            linear_spec_C=objective,
            run_id=f"nrir21:{args.clause_index:04d}",
            config=config,
            optimizer_policy=optimizer_policy,
            intermediate_bound_source=IntermediateBoundSource.NATIVE_REFINED,
            per_child_refinement_policy=_refinement_policy(),
            per_child_refinement_strategy=args.per_child_strategy,
        )
        refinements = [
            _serialize_refinement(node_id, execution)
            for node_id, execution in queue.per_child_refinement_executions
        ]
    queue_ns = time.perf_counter_ns() - queue_started_ns
    trace = queue.trace
    leaves = [
        evaluation
        for evaluation in trace.evaluations
        if evaluation.node.depth == MAX_DEPTH
    ]
    if len(leaves) != 4:
        raise ValueError("NRIR-21 fixed tree did not produce four depth-limit leaves")
    result = {
        "schema_version": WORKER_SCHEMA_VERSION,
        "workload_id": WORKLOAD_ID,
        "clause_index": args.clause_index,
        "mode": args.mode,
        "execution_state": "completed",
        "threshold": float(tensors.thresholds[args.clause_index]),
        "queue_trace": trace.to_dict(),
        "queue_trace_hash": trace.stable_hash(),
        "refinement_programs": refinements,
        "root_lower": trace.evaluations[0].lower,
        "root_upper": trace.evaluations[0].upper,
        "leaf_lowers": [item.lower for item in leaves],
        "leaf_uppers": [item.upper for item in leaves],
        "leaf_split_state_hashes": [item.node.split_state_hash for item in leaves],
        "worst_leaf_lower": min(item.lower for item in leaves),
        "best_leaf_lower": max(item.lower for item in leaves),
        "queue_ns": queue_ns,
        "worker_elapsed_ns": time.perf_counter_ns() - started_ns,
        "performance_claimed": False,
    }
    _write_json(args.result_json, result)
    print(
        _canonical_json(
            {
                "status": "ok",
                "clause_index": args.clause_index,
                "mode": args.mode,
                "root_lower": result["root_lower"],
                "worst_leaf_lower": result["worst_leaf_lower"],
            }
        )
    )


def _comparison(
    clause_index: int,
    root_global: Mapping[str, Any],
    per_child: Mapping[str, Any],
) -> dict[str, object]:
    root_delta = float(per_child["root_lower"]) - float(root_global["root_lower"])
    worst_delta = float(per_child["worst_leaf_lower"]) - float(
        root_global["worst_leaf_lower"]
    )
    return {
        "clause_index": clause_index,
        "root_global_root_lower": root_global["root_lower"],
        "per_child_root_lower": per_child["root_lower"],
        "root_lower_delta": root_delta,
        "root_global_worst_leaf_lower": root_global["worst_leaf_lower"],
        "per_child_worst_leaf_lower": per_child["worst_leaf_lower"],
        "worst_leaf_lower_delta": worst_delta,
        "root_global_leaf_lowers": root_global["leaf_lowers"],
        "per_child_leaf_lowers": per_child["leaf_lowers"],
        "strict_worst_frontier_improvement": worst_delta > 0.0,
        "root_bound_same": abs(root_delta) <= 1e-5,
        "performance_claimed": False,
    }


def _build_evidence(
    benchmark_root: Path, artifact_dir: Path
) -> tuple[dict[str, object], dict[str, str]]:
    source = _resolved_source(benchmark_root)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    files: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="boundflow-nrir21-") as temporary:
        temp_root = Path(temporary)
        for clause_index in CLAUSE_INDICES:
            for mode in MODES:
                result_path = temp_root / f"clause-{clause_index}-{mode}.json"
                result, log, returncode, elapsed_ns = _run_worker(
                    mode=mode,
                    clause_index=clause_index,
                    source=source,
                    result=result_path,
                )
                log_path = artifact_dir / "logs" / f"clause-{clause_index}-{mode}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text(log, encoding="utf-8")
                relative_log = str(log_path.relative_to(artifact_dir))
                files[relative_log] = file_sha256(log_path)
                records.append(
                    {
                        "clause_index": clause_index,
                        "mode": mode,
                        "process_returncode": returncode,
                        "e2e_elapsed_ns": elapsed_ns,
                        "log_path": relative_log,
                        "log_sha256": files[relative_log],
                        "result": result,
                    }
                )
    by_identity = {
        (cast(int, record["clause_index"]), str(record["mode"])): cast(
            Mapping[str, Any], record["result"]
        )
        for record in records
    }
    comparisons = [
        _comparison(
            clause_index,
            by_identity[(clause_index, "root_global")],
            by_identity[(clause_index, "per_child")],
        )
        for clause_index in CLAUSE_INDICES
    ]
    all_improved = all(
        bool(item["strict_worst_frontier_improvement"]) for item in comparisons
    )
    evidence: dict[str, object] = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "validated_reduced" if all_improved else "validated_no_go",
        "claim_boundary": (
            "same-policy, same-seven-node/depth-two CPU tightness comparison of "
            "root-global reuse versus exact-split per-child refinement on two "
            "fixed VNN-COMP ResNet clauses; no speedup or property-closure claim"
        ),
        "source": _source_payload(source),
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
            "The fixed experiment covers two clauses, one model, seven nodes, and depth two; it is not a complete property result.",
            "Per-child refinement is executed serially before packed optimizer evaluation; no latency or speedup claim is made.",
            "The comparison uses depth-limit leaves as the bounded frontier; it does not claim unbounded BaB convergence.",
            "The objective-directed shortlist may not dominate root-global reuse on every workload or clause.",
            "CUDA, repeated timing, competitor parity, and ASPLOS-ready claims remain pending.",
        ],
        "performance_claimed": False,
    }
    return evidence, files


def _deterministic_trace(trace: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(trace)
    result.pop("elapsed_ns", None)
    return result


def _validate_refinement_program(program: Mapping[str, Any]) -> None:
    plan = _mapping(program.get("plan"), "NRIR-21 Plan")
    task = _mapping(program.get("task"), "NRIR-21 Task")
    schedule = _mapping(program.get("schedule"), "NRIR-21 Schedule")
    hashes = _mapping(program.get("hashes"), "NRIR-21 hashes")
    trace = _mapping(program.get("execution_trace"), "NRIR-21 execution trace")
    if (
        not str(program.get("node_id", ""))
        or hashes.get("refinement_plan_hash") != canonical_hash(plan)
        or hashes.get("refinement_task_module_hash") != canonical_hash(task)
        or hashes.get("refinement_schedule_hash") != canonical_hash(schedule)
        or task.get("refinement_plan_hash") != hashes.get("refinement_plan_hash")
        or schedule.get("refinement_plan_hash") != hashes.get("refinement_plan_hash")
        or schedule.get("refinement_task_module_hash")
        != hashes.get("refinement_task_module_hash")
        or trace.get("plan_hash") != hashes.get("refinement_plan_hash")
        or trace.get("task_module_hash") != hashes.get("refinement_task_module_hash")
        or trace.get("schedule_hash") != hashes.get("refinement_schedule_hash")
        or program.get("semantic_execution_trace_hash")
        != canonical_hash(_deterministic_trace(trace))
    ):
        raise ValueError("NRIR-21 refinement IR/trace linkage differs")


def _validate_worker_result(result: Mapping[str, Any]) -> None:
    mode = result.get("mode")
    queue = _mapping(result.get("queue_trace"), "NRIR-21 queue trace")
    evaluations = queue.get("evaluations")
    refinements = result.get("refinement_programs")
    if (
        result.get("schema_version") != WORKER_SCHEMA_VERSION
        or result.get("workload_id") != WORKLOAD_ID
        or result.get("clause_index") not in CLAUSE_INDICES
        or mode not in MODES
        or result.get("execution_state") != "completed"
        or result.get("performance_claimed") is not False
        or queue.get("performance_claimed") is not False
        or result.get("queue_trace_hash") != canonical_hash(queue)
        or not isinstance(evaluations, list)
        or len(evaluations) != MAX_NODES
        or not isinstance(refinements, list)
        or len(result.get("leaf_lowers", [])) != 4
        or len(result.get("leaf_uppers", [])) != 4
        or len(result.get("leaf_split_state_hashes", [])) != 4
        or float(result["worst_leaf_lower"]) != min(result["leaf_lowers"])
        or float(result["best_leaf_lower"]) != max(result["leaf_lowers"])
    ):
        raise ValueError("NRIR-21 worker result differs")
    for program in refinements:
        program_item = _mapping(program, "NRIR-21 refinement program")
        _validate_refinement_program(program_item)
        plan = _mapping(program_item.get("plan"), "NRIR-21 Plan")
        plan_policy = _mapping(plan.get("policy"), "NRIR-21 refinement policy")
        if (
            plan.get("objective_hash") != queue.get("objective_hash")
            or plan_policy.get("candidate_policy_id")
            != "objective_influence_width_per_relu_v1"
            or len(cast(list[object], plan.get("targets"))) != 96
        ):
            raise ValueError("NRIR-21 refinement objective/policy binding differs")
    if mode == "root_global":
        if (
            len(refinements) != 1
            or "per_child_refinement_policy" in queue
            or any("intermediate_refinement_trace_hash" in item for item in evaluations)
        ):
            raise ValueError("NRIR-21 root-global trace boundary differs")
        return
    queue_records = queue.get("per_child_refinements")
    if (
        len(refinements) != MAX_NODES
        or not isinstance(queue_records, list)
        or len(queue_records) != MAX_NODES
        or _mapping(queue.get("per_child_refinement_policy"), "policy").get(
            "candidate_policy_id"
        )
        != "objective_influence_width_per_relu_v1"
    ):
        raise ValueError("NRIR-21 per-child trace coverage differs")
    by_node = {str(item["node_id"]): item for item in refinements}
    if len(by_node) != MAX_NODES:
        raise ValueError("NRIR-21 per-child refinement node IDs repeat")
    for evaluation, record in zip(evaluations, queue_records):
        evaluation_item = _mapping(evaluation, "NRIR-21 evaluation")
        node = _mapping(evaluation_item.get("node"), "NRIR-21 node")
        record_item = _mapping(record, "NRIR-21 refinement record")
        program = _mapping(by_node.get(str(node.get("node_id"))), "NRIR-21 program")
        plan = _mapping(program.get("plan"), "NRIR-21 Plan")
        hashes = _mapping(program.get("hashes"), "NRIR-21 hashes")
        trace = _mapping(program.get("execution_trace"), "NRIR-21 trace")
        if (
            record_item.get("node_id") != node.get("node_id")
            or record_item.get("node_split_state_hash") != node.get("split_state_hash")
            or plan.get("split_state_hash") != node.get("split_state_hash")
            or record_item.get("refinement_plan_hash")
            != hashes.get("refinement_plan_hash")
            or record_item.get("refinement_task_module_hash")
            != hashes.get("refinement_task_module_hash")
            or record_item.get("refinement_schedule_hash")
            != hashes.get("refinement_schedule_hash")
            or record_item.get("initial_intermediate_bounds_hash")
            != plan.get("initial_intermediate_bounds_hash")
            or record_item.get("final_intermediate_bounds_hash")
            != trace.get("final_intermediate_bounds_hash")
            or record_item.get("refinement_semantic_trace_hash")
            != program.get("semantic_execution_trace_hash")
            or record_item.get("selected_target_count")
            != len(cast(list[object], plan.get("targets")))
            or evaluation_item.get("intermediate_refinement_trace_hash")
            != canonical_hash(record_item)
            or record_item.get("parent_refinement_consumed_as_exact") is not False
        ):
            raise ValueError("NRIR-21 node/refinement lineage differs")


def validate_evidence_structure(evidence: Mapping[str, Any]) -> None:
    if (
        evidence.get("schema_version") != EVIDENCE_SCHEMA_VERSION
        or evidence.get("status") not in {"validated_reduced", "validated_no_go"}
        or evidence.get("performance_claimed") is not False
        or "no speedup" not in str(evidence.get("claim_boundary"))
    ):
        raise ValueError("NRIR-21 evidence header differs")
    source = _mapping(evidence.get("source"), "NRIR-21 source")
    if (
        source.get("vnncomp_commit") != VNNCOMP_COMMIT
        or source.get("workload_id") != WORKLOAD_ID
        or not _sha256(source.get("native_code_revision"))
        or not _sha256(source.get("model_sha256"))
        or not _sha256(source.get("property_sha256"))
    ):
        raise ValueError("NRIR-21 source identity differs")
    records = evidence.get("records")
    if not isinstance(records, list):
        raise TypeError("NRIR-21 records must be a list")
    identities: set[tuple[int, str]] = set()
    by_identity: dict[tuple[int, str], Mapping[str, Any]] = {}
    for record in records:
        item = _mapping(record, "NRIR-21 record")
        identity = (int(item["clause_index"]), str(item["mode"]))
        result = _mapping(item.get("result"), "NRIR-21 result")
        _validate_worker_result(result)
        if (
            identity in identities
            or identity != (int(result["clause_index"]), str(result["mode"]))
            or item.get("process_returncode") != 0
            or not isinstance(item.get("e2e_elapsed_ns"), int)
            or int(item["e2e_elapsed_ns"]) <= 0
            or not str(item.get("log_path", "")).startswith("logs/")
            or not _sha256(item.get("log_sha256"))
        ):
            raise ValueError("NRIR-21 execution record differs")
        identities.add(identity)
        by_identity[identity] = result
    expected = {
        (clause_index, mode) for clause_index in CLAUSE_INDICES for mode in MODES
    }
    if identities != expected:
        raise ValueError("NRIR-21 record coverage differs")
    comparisons = evidence.get("comparisons")
    if not isinstance(comparisons, list) or len(comparisons) != len(CLAUSE_INDICES):
        raise ValueError("NRIR-21 comparison coverage differs")
    improved: list[bool] = []
    for comparison in comparisons:
        item = _mapping(comparison, "NRIR-21 comparison")
        clause = int(item["clause_index"])
        root = by_identity[(clause, "root_global")]
        child = by_identity[(clause, "per_child")]
        root_delta = float(child["root_lower"]) - float(root["root_lower"])
        worst_delta = float(child["worst_leaf_lower"]) - float(root["worst_leaf_lower"])
        if (
            abs(float(item["root_lower_delta"]) - root_delta) > 1e-7
            or abs(float(item["worst_leaf_lower_delta"]) - worst_delta) > 1e-7
            or bool(item["root_bound_same"]) != (abs(root_delta) <= 1e-5)
            or bool(item["strict_worst_frontier_improvement"]) != (worst_delta > 0.0)
            or item.get("performance_claimed") is not False
        ):
            raise ValueError("NRIR-21 tightness comparison differs")
        improved.append(worst_delta > 0.0)
    expected_status = "validated_reduced" if all(improved) else "validated_no_go"
    if evidence.get("status") != expected_status:
        raise ValueError("NRIR-21 closure status differs")
    limitations = evidence.get("limitations")
    if not isinstance(limitations, list) or len(limitations) != 5:
        raise ValueError("NRIR-21 limitation ledger differs")


def _generate(args: argparse.Namespace) -> None:
    evidence, files = _build_evidence(args.benchmark_root, args.artifact_dir)
    validate_evidence_structure(evidence)
    evidence_path = args.artifact_dir / EVIDENCE_FILE
    _write_json(evidence_path, evidence)
    files[EVIDENCE_FILE] = file_sha256(evidence_path)
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": evidence["status"],
        "performance_claimed": False,
        "files": dict(sorted(files.items())),
        "evidence_hash": canonical_hash(evidence),
    }
    _write_json(args.artifact_dir / MANIFEST_FILE, manifest)
    print(
        _canonical_json(
            {
                "status": evidence["status"],
                "evidence_hash": manifest["evidence_hash"],
            }
        )
    )


def _without_diagnostic_timing(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            key: _without_diagnostic_timing(item)
            for key, item in value.items()
            if key
            not in {
                "elapsed_ns",
                "queue_ns",
                "worker_elapsed_ns",
                "e2e_elapsed_ns",
            }
        }
    if isinstance(value, list):
        return [_without_diagnostic_timing(item) for item in value]
    return value


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
        or manifest.get("status") != evidence.get("status")
        or manifest.get("performance_claimed") is not False
        or manifest.get("files") != actual_files
        or manifest.get("evidence_hash") != canonical_hash(evidence)
    ):
        raise ValueError("NRIR-21 artifact manifest differs")
    validate_evidence_structure(evidence)
    if evidence.get("source") != _source_payload(_resolved_source(args.benchmark_root)):
        raise ValueError("NRIR-21 source replay differs")
    source = _resolved_source(args.benchmark_root)
    stored = {
        (int(record["clause_index"]), str(record["mode"])): _mapping(
            record["result"], "NRIR-21 stored result"
        )
        for record in cast(list[Mapping[str, Any]], evidence["records"])
    }
    replayed_hashes: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="boundflow-nrir21-replay-") as temporary:
        temp_root = Path(temporary)
        for clause_index in CLAUSE_INDICES:
            for mode in MODES:
                result_path = temp_root / f"clause-{clause_index}-{mode}.json"
                fresh, _log, _returncode, _elapsed = _run_worker(
                    mode=mode,
                    clause_index=clause_index,
                    source=source,
                    result=result_path,
                )
                _validate_worker_result(fresh)
                if _without_diagnostic_timing(fresh) != _without_diagnostic_timing(
                    stored[(clause_index, mode)]
                ):
                    raise ValueError("NRIR-21 source-to-IR semantic replay differs")
                replayed_hashes[f"clause-{clause_index}-{mode}"] = canonical_hash(
                    _without_diagnostic_timing(fresh)
                )
    print(
        _canonical_json(
            {
                "status": "replayed",
                "closure": evidence["status"],
                "evidence_hash": manifest["evidence_hash"],
                "semantic_result_hashes": replayed_hashes,
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
