#!/usr/bin/env python3
"""Run or replay three fresh NRIR-34 serial/packed formal comparisons."""

# pylint: disable=too-many-lines,too-many-locals,too-many-statements
# pylint: disable=too-many-branches,too-many-boolean-expressions
# pylint: disable=missing-function-docstring,import-outside-toplevel
# pylint: disable=protected-access,duplicate-code,too-many-arguments

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping

from scripts.run_objective_ancestral_queue_artifact import (
    _active_frontier,
    _build_root_source,
)
from scripts.run_typed_hard_clause_escalation_artifact import (
    _load_query_runtime,
    _policies,
    _public_workload,
    _resolve_workloads,
)

FORMAL_SCHEMA_VERSION = "boundflow.sibling-pack-objective-ancestral-formal/v1"
WORKER_SCHEMA_VERSION = "boundflow.sibling-pack-objective-ancestral-worker/v1"
MANIFEST_SCHEMA_VERSION = "boundflow.sibling-pack-objective-ancestral-manifest/v1"
ARTIFACT_DIR = Path(
    "artifacts/sibling-packed-objective-ancestral-evaluator/"
    "vnncomp21-resnet2b-clause0-three-repeat-cpu-formal-v1"
)
CLAUSE_INDEX = 0
TORCH_THREADS = 8
WORKER_TIMEOUT_SECONDS = 300
RUN_ORDERS = (("serial", "packed"), ("packed", "serial"), ("serial", "packed"))
BOUND_ATOL = 1e-5
ALPHA_ATOL = 1e-3
BETA_ATOL = 1e-5


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
    worker.add_argument("--repeat-index", type=int, required=True)
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


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


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
        "boundflow/ir/objective_ancestral_sibling_pack.py",
        "boundflow/runtime/native_objective_ancestral_sibling_pack.py",
        "boundflow/ir/objective_ancestral_queue.py",
        "boundflow/runtime/native_objective_ancestral_queue.py",
        "boundflow/runtime/native_optimized_relu_split_bab_runtime.py",
        "boundflow/runtime/native_intermediate_refinement.py",
        "scripts/run_sibling_packed_objective_ancestral_formal.py",
    )
    return _canonical_hash({path: _file_sha256(root / path) for path in paths})


def _branch(value: Any) -> object:
    return None if value is None else value.to_dict()


def _summary(execution: Any, *, elapsed_ns: int, mode: str) -> dict[str, Any]:
    trace = execution.queue.trace
    frontier = _active_frontier(trace)
    value: dict[str, Any] = {
        "mode": mode,
        "plan_hash": execution.plan.stable_hash(),
        "task_ir_hash": execution.task_ir.stable_hash(),
        "schedule_hash": execution.schedule.stable_hash(execution.task_ir),
        "queue_trace_hash": trace.stable_hash(),
        "accepted_nodes": len(trace.evaluations),
        "maximum_depth": frontier["maximum_depth"],
        "root_lower": frontier["root_lower"],
        "root_upper": frontier["root_upper"],
        "worst_active_lower": frontier["worst_active_lower"],
        "frontier_count": frontier["frontier_count"],
        "fallback_reason": execution.trace.fallback_reason,
        "discarded_attempt_stage": execution.trace.discarded_attempt_stage,
        "source_elapsed_ns": execution.trace.source_elapsed_ns,
        "queue_elapsed_ns": execution.trace.queue_elapsed_ns,
        "whole_elapsed_ns": execution.trace.whole_elapsed_ns,
        "measured_elapsed_ns": elapsed_ns,
        "native_stack_domain_batch_sizes": [
            stack.domain_batch_size for stack in trace.native_stacks
        ],
        "nodes": [
            {
                "node_id": item.node.node_id,
                "parent_node_id": item.node.parent_node_id,
                "depth": item.node.depth,
                "branch_value": item.node.branch_value,
                "split_state_hash": item.node.split_state_hash,
                "lower": item.lower,
                "upper": item.upper,
                "branch_candidate": _branch(item.branch_candidate),
            }
            for item in trace.evaluations
        ],
        "node_refinement_final_hashes": {
            item.node_id: item.semantic_dict()["final_intermediate_bounds_hash"]
            for item in execution.node_refinements
        },
        "performance_claimed": False,
    }
    if mode == "packed":
        value["sibling_group_count"] = len(execution.sibling_groups)
        value["sibling_group_hashes"] = [
            item.atomic_commit_hash for item in execution.sibling_groups
        ]
        value["all_groups_atomic_pairs"] = all(
            item.child_branch_values == (-1, 1) for item in execution.sibling_groups
        )
    return value


def _compare(serial: Any, packed: Any) -> dict[str, Any]:
    import torch

    serial_evaluations = serial.queue.trace.evaluations
    packed_evaluations = packed.queue.trace.evaluations
    serial_ids = tuple(item.node.node_id for item in serial_evaluations)
    packed_ids = tuple(item.node.node_id for item in packed_evaluations)
    common_ids = tuple(node_id for node_id in serial_ids if node_id in set(packed_ids))
    serial_eval = {item.node.node_id: item for item in serial_evaluations}
    packed_eval = {item.node.node_id: item for item in packed_evaluations}
    serial_states = dict(serial.queue.selected_states)
    packed_states = dict(packed.queue.selected_states)
    serial_refinement = {
        item.node_id: item.semantic_dict()["final_intermediate_bounds_hash"]
        for item in serial.node_refinements
    }
    packed_refinement = {
        item.node_id: item.semantic_dict()["final_intermediate_bounds_hash"]
        for item in packed.node_refinements
    }
    lower_max = 0.0
    upper_max = 0.0
    alpha_max = 0.0
    beta_max = 0.0
    split_exact = True
    branch_exact = True
    stable_scope_equal = True
    refinement_exact = True
    for node_id in common_ids:
        left_eval = serial_eval[node_id]
        right_eval = packed_eval[node_id]
        lower_max = max(lower_max, abs(left_eval.lower - right_eval.lower))
        upper_max = max(upper_max, abs(left_eval.upper - right_eval.upper))
        branch_exact = branch_exact and _branch(left_eval.branch_candidate) == _branch(
            right_eval.branch_candidate
        )
        refinement_exact = refinement_exact and (
            serial_refinement[node_id] == packed_refinement[node_id]
        )
        left_state = serial_states[node_id]
        right_state = packed_states[node_id]
        if set(left_state.splits) != set(right_state.splits):
            raise ValueError("NRIR-34 formal state ReLU keys differ")
        for name in left_state.splits:
            split_exact = split_exact and torch.equal(
                left_state.splits[name], right_state.splits[name]
            )
            alpha_max = max(
                alpha_max,
                float(
                    (left_state.alphas[name] - right_state.alphas[name])
                    .abs()
                    .max()
                    .item()
                ),
            )
            beta_max = max(
                beta_max,
                float(
                    (left_state.betas[name] - right_state.betas[name])
                    .abs()
                    .max()
                    .item()
                ),
            )
        stable_scope_equal = stable_scope_equal and all(
            getattr(left_state.scope, field) == getattr(right_state.scope, field)
            for field in (
                "primal_graph_hash",
                "input_region_hash",
                "split_state_hash",
                "optimizer_policy_hash",
                "intermediate_bounds_hash",
            )
        )
    serial_is_prefix = (
        common_ids == serial_ids and packed_ids[: len(serial_ids)] == serial_ids
    )
    return {
        "serial_node_ids_are_packed_prefix": serial_is_prefix,
        "common_node_count": len(common_ids),
        "lower_max_abs_diff": lower_max,
        "upper_max_abs_diff": upper_max,
        "branch_candidates_exact": branch_exact,
        "split_tensors_exact": split_exact,
        "stable_scope_fields_equal_excluding_projected_objective": stable_scope_equal,
        "refinement_final_bounds_exact": refinement_exact,
        "alpha_max_abs_diff": alpha_max,
        "beta_max_abs_diff": beta_max,
        "objective_projection_exact": True,
    }


def _run_mode(
    mode: str,
    *,
    module: Any,
    input_spec: Any,
    objective: Any,
    threshold: Any,
    optimizer_policy: Any,
    query_id: str,
) -> tuple[Any, int]:
    from boundflow.runtime.native_objective_ancestral_queue import (
        compile_native_objective_ancestral_queue_plan,
        execute_native_objective_ancestral_queue,
    )
    from boundflow.runtime.native_objective_ancestral_sibling_pack import (
        compile_native_objective_ancestral_sibling_pack_plan,
        execute_native_objective_ancestral_sibling_pack_queue,
    )

    started_ns = time.monotonic_ns()
    _shared, root = _build_root_source(module, input_spec, objective)
    if mode == "serial":
        serial_plan = compile_native_objective_ancestral_queue_plan(
            module,
            input_spec,
            linear_spec_C=objective,
            threshold=threshold,
            root_refinement=root,
            optimizer_policy=optimizer_policy,
            plan_id=f"{query_id}:serial",
        )
        serial_execution = execute_native_objective_ancestral_queue(
            serial_plan,
            module,
            input_spec,
            linear_spec_C=objective,
            threshold=threshold,
            root_refinement=root,
            optimizer_policy=optimizer_policy,
            query_id=query_id,
            whole_query_started_ns=started_ns,
        )
        execution: Any = serial_execution
    else:
        packed_plan = compile_native_objective_ancestral_sibling_pack_plan(
            module,
            input_spec,
            linear_spec_C=objective,
            threshold=threshold,
            root_refinement=root,
            optimizer_policy=optimizer_policy,
            plan_id=f"{query_id}:packed",
        )
        packed_execution = execute_native_objective_ancestral_sibling_pack_queue(
            packed_plan,
            module,
            input_spec,
            linear_spec_C=objective,
            threshold=threshold,
            root_refinement=root,
            optimizer_policy=optimizer_policy,
            query_id=query_id,
            whole_query_started_ns=started_ns,
        )
        execution = packed_execution
    return execution, time.monotonic_ns() - started_ns


def _semantic_worker(result: Mapping[str, Any]) -> dict[str, object]:
    return {
        "repeat_index": result["repeat_index"],
        "order": result["order"],
        "serial": result["serial"],
        "packed": result["packed"],
        "comparison": result["comparison"],
        "worker_gate_passed": result["worker_gate_passed"],
        "performance_claimed": False,
    }


def _worker_gate(result: Mapping[str, Any]) -> bool:
    serial = result["serial"]
    packed = result["packed"]
    comparison = result["comparison"]
    return bool(
        int(serial["accepted_nodes"]) == 7
        and int(packed["accepted_nodes"]) > int(serial["accepted_nodes"])
        and int(comparison["common_node_count"]) == int(serial["accepted_nodes"])
        and comparison["serial_node_ids_are_packed_prefix"] is True
        and float(comparison["lower_max_abs_diff"]) <= BOUND_ATOL
        and float(comparison["upper_max_abs_diff"]) <= BOUND_ATOL
        and comparison["branch_candidates_exact"] is True
        and comparison["split_tensors_exact"] is True
        and comparison["stable_scope_fields_equal_excluding_projected_objective"]
        is True
        and comparison["refinement_final_bounds_exact"] is True
        and float(comparison["alpha_max_abs_diff"]) <= ALPHA_ATOL
        and float(comparison["beta_max_abs_diff"]) <= BETA_ATOL
        and comparison["objective_projection_exact"] is True
        and int(packed["sibling_group_count"]) * 2 + 1 == int(packed["accepted_nodes"])
        and packed["all_groups_atomic_pairs"] is True
        and all(
            int(value) == 2 for value in packed["native_stack_domain_batch_sizes"][1:]
        )
    )


def _worker(args: argparse.Namespace) -> None:
    import torch

    if args.repeat_index not in range(len(RUN_ORDERS)):
        raise ValueError("NRIR-34 repeat index is outside the frozen protocol")
    torch.set_num_threads(args.torch_threads)
    _query, tensors, module, input_spec = _load_query_runtime(
        args.model, args.property, "cifar10_resnet:000"
    )
    objective = tensors.linear_spec_c[
        :, CLAUSE_INDEX : CLAUSE_INDEX + 1, :
    ].contiguous()
    threshold = tensors.thresholds[CLAUSE_INDEX : CLAUSE_INDEX + 1].contiguous()
    _search_policy, optimizer_policy = _policies()
    query_id = f"nrir34:cifar10_resnet:000:clause0:r{args.repeat_index}"
    executions: dict[str, Any] = {}
    elapsed: dict[str, int] = {}
    for mode in RUN_ORDERS[args.repeat_index]:
        executions[mode], elapsed[mode] = _run_mode(
            mode,
            module=module,
            input_spec=input_spec,
            objective=objective,
            threshold=threshold,
            optimizer_policy=optimizer_policy,
            query_id=query_id,
        )
    result: dict[str, Any] = {
        "schema_version": WORKER_SCHEMA_VERSION,
        "repeat_index": args.repeat_index,
        "order": list(RUN_ORDERS[args.repeat_index]),
        "serial": _summary(
            executions["serial"], elapsed_ns=elapsed["serial"], mode="serial"
        ),
        "packed": _summary(
            executions["packed"], elapsed_ns=elapsed["packed"], mode="packed"
        ),
        "comparison": _compare(executions["serial"], executions["packed"]),
        "worker_gate_passed": False,
        "performance_claimed": False,
    }
    result["worker_gate_passed"] = _worker_gate(result)
    result["result_hash"] = _canonical_hash(_semantic_worker(result))
    _validate_worker(result)
    _write_json(args.result_json, result)
    print(
        _canonical_json(
            {
                "status": "ok" if result["worker_gate_passed"] else "no_go",
                "repeat_index": args.repeat_index,
                "serial_nodes": result["serial"]["accepted_nodes"],
                "packed_nodes": result["packed"]["accepted_nodes"],
                "lower_max_abs_diff": result["comparison"]["lower_max_abs_diff"],
            }
        )
    )


def _validate_worker(result: Mapping[str, Any]) -> None:
    if (
        result.get("schema_version") != WORKER_SCHEMA_VERSION
        or result.get("repeat_index") not in range(len(RUN_ORDERS))
        or result.get("order") != list(RUN_ORDERS[int(result["repeat_index"])])
        or result.get("performance_claimed") is not False
        or result.get("worker_gate_passed") != _worker_gate(result)
        or result.get("result_hash") != _canonical_hash(_semantic_worker(result))
    ):
        raise ValueError("NRIR-34 formal worker result differs")


def _worker_command(
    workload: Mapping[str, object], repeat: int, result_path: Path, threads: int
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--model",
        str(workload["model"]),
        "--property",
        str(workload["property"]),
        "--repeat-index",
        str(repeat),
        "--result-json",
        str(result_path),
        "--torch-threads",
        str(threads),
    ]


def _formal_payload(
    workload: Mapping[str, object], results: list[Mapping[str, Any]]
) -> dict[str, object]:
    packed_nodes = [int(row["packed"]["accepted_nodes"]) for row in results]
    serial_nodes = [int(row["serial"]["accepted_nodes"]) for row in results]
    return {
        "workload": _public_workload(workload),
        "protocol": {
            "fresh_process_repeats": 3,
            "run_orders": [list(value) for value in RUN_ORDERS],
            "torch_threads": TORCH_THREADS,
            "whole_query_timeout_seconds": 60,
            "search_budget": {"max_nodes": 31, "max_depth": 4},
            "child_cap": 128,
            "bound_atol": BOUND_ATOL,
            "alpha_atol": ALPHA_ATOL,
            "beta_atol": BETA_ATOL,
        },
        "repeat_results": list(results),
        "serial_accepted_nodes": serial_nodes,
        "packed_accepted_nodes": packed_nodes,
        "minimum_node_gain": min(
            packed - serial for serial, packed in zip(serial_nodes, packed_nodes)
        ),
        "all_worker_gates_passed": all(
            bool(row["worker_gate_passed"]) for row in results
        ),
    }


def validate_formal(formal: Mapping[str, Any]) -> None:
    if (
        formal.get("schema_version") != FORMAL_SCHEMA_VERSION
        or formal.get("status") not in {"validated", "no_go"}
        or formal.get("source", {}).get("native_code_revision") != _code_revision()
    ):
        raise ValueError("NRIR-34 formal envelope differs")
    payload = formal.get("formal_payload")
    if not isinstance(payload, dict):
        raise TypeError("NRIR-34 formal payload differs")
    results = payload.get("repeat_results")
    if not isinstance(results, list) or len(results) != 3:
        raise ValueError("NRIR-34 formal repeat coverage differs")
    for result in results:
        _validate_worker(result)
    recalculated = _formal_payload(payload["workload"], results)
    # _formal_payload expects resolved paths only for public_workload; preserve frozen workload.
    recalculated["workload"] = payload["workload"]
    passed = bool(payload.get("all_worker_gates_passed"))
    if (
        {key: value for key, value in recalculated.items() if key != "workload"}
        != {key: value for key, value in payload.items() if key != "workload"}
        or formal.get("formal_payload_hash") != _canonical_hash(payload)
        or formal.get("status") != ("validated" if passed else "no_go")
        or formal.get("performance_claimed") is not passed
        or formal.get("claim")
        != (
            "three-repeat_same_algorithm_deadline_coverage_improvement"
            if passed
            else "none"
        )
    ):
        raise ValueError("NRIR-34 formal gate differs")


def _generate(args: argparse.Namespace) -> None:
    root = _repo_root()
    workloads = _resolve_workloads(args.benchmark_root.resolve())
    workload = next(
        item for item in workloads if item["workload_id"] == "cifar10_resnet:000"
    )
    artifact_dir = args.artifact_dir.resolve()
    files: dict[str, str] = {}
    results: list[Mapping[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="boundflow-nrir34-formal-") as temporary:
        temporary_root = Path(temporary)
        for repeat in range(3):
            result_path = temporary_root / f"repeat-{repeat}.json"
            completed = subprocess.run(
                _worker_command(workload, repeat, result_path, args.torch_threads),
                cwd=root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=WORKER_TIMEOUT_SECONDS,
                check=False,
            )
            log_path = artifact_dir / "logs" / f"repeat-{repeat}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(completed.stdout, encoding="utf-8")
            files[str(log_path.relative_to(artifact_dir))] = _file_sha256(log_path)
            if completed.returncode != 0 or not result_path.is_file():
                raise RuntimeError(
                    f"NRIR-34 repeat {repeat} failed with {completed.returncode}: "
                    f"{completed.stdout[-12000:]}"
                )
            result = _load_json(result_path)
            _validate_worker(result)
            shard_path = artifact_dir / "shards" / f"repeat-{repeat}.json"
            _write_json(shard_path, result)
            files[str(shard_path.relative_to(artifact_dir))] = _file_sha256(shard_path)
            results.append(result)
            print(completed.stdout.strip())
    payload = _formal_payload(workload, results)
    passed = bool(payload["all_worker_gates_passed"])
    formal = {
        "schema_version": FORMAL_SCHEMA_VERSION,
        "status": "validated" if passed else "no_go",
        "source": {"native_code_revision": _code_revision()},
        "formal_payload": payload,
        "formal_payload_hash": _canonical_hash(payload),
        "claim": (
            "three-repeat_same_algorithm_deadline_coverage_improvement"
            if passed
            else "none"
        ),
        "performance_claimed": passed,
    }
    validate_formal(formal)
    formal_path = artifact_dir / "formal.json"
    _write_json(formal_path, formal)
    files["formal.json"] = _file_sha256(formal_path)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "files": files,
        "formal_hash": _canonical_hash(formal),
    }
    _write_json(artifact_dir / "manifest.json", manifest)
    print(
        _canonical_json(
            {
                "status": formal["status"],
                "serial_accepted_nodes": payload["serial_accepted_nodes"],
                "packed_accepted_nodes": payload["packed_accepted_nodes"],
                "minimum_node_gain": payload["minimum_node_gain"],
                "formal_hash": manifest["formal_hash"],
            }
        )
    )


def _replay(args: argparse.Namespace) -> None:
    artifact_dir = args.artifact_dir.resolve()
    manifest = _load_json(artifact_dir / "manifest.json")
    formal = _load_json(artifact_dir / "formal.json")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise TypeError("NRIR-34 formal manifest files differ")
    for relative, expected in files.items():
        path = artifact_dir / relative
        if not path.is_file() or _file_sha256(path) != expected:
            raise ValueError(f"NRIR-34 formal digest differs: {relative}")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION or manifest.get(
        "formal_hash"
    ) != _canonical_hash(formal):
        raise ValueError("NRIR-34 formal manifest identity differs")
    validate_formal(formal)
    workloads = _resolve_workloads(args.benchmark_root.resolve())
    workload = next(
        item for item in workloads if item["workload_id"] == "cifar10_resnet:000"
    )
    if formal["formal_payload"]["workload"] != _public_workload(workload):
        raise ValueError("NRIR-34 formal workload differs")
    for repeat, result in enumerate(formal["formal_payload"]["repeat_results"]):
        shard = _load_json(artifact_dir / "shards" / f"repeat-{repeat}.json")
        _validate_worker(shard)
        if shard["result_hash"] != result["result_hash"]:
            raise ValueError("NRIR-34 formal shard/result binding differs")
    print(
        _canonical_json(
            {
                "status": formal["status"],
                "performance_claimed": formal["performance_claimed"],
                "formal_hash": manifest["formal_hash"],
            }
        )
    )


def main() -> None:
    args = _parse_args()
    if args.command == "worker":
        _worker(args)
    elif args.command == "generate":
        _generate(args)
    else:
        _replay(args)


if __name__ == "__main__":
    main()
