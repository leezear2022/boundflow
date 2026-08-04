#!/usr/bin/env python3
"""Run or replay three fresh NRIR-40 objective-branch whole queries."""

# pylint: disable=too-many-lines,too-many-locals,too-many-statements
# pylint: disable=too-many-boolean-expressions,missing-function-docstring
# pylint: disable=import-outside-toplevel,duplicate-code,too-many-branches

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

from boundflow.ir.objective_branch_shared_evaluator import (
    NativeObjectiveBranchBindingIR,
)
from boundflow.runtime.native_intermediate_refinement import (
    intermediate_bounds_hash,
    intermediate_refinement_semantic_trace_hash,
)
from boundflow.runtime.native_objective_branch_score import (
    NativeObjectiveBranchPolicy,
)
from boundflow.runtime.native_objective_branch_shared_evaluator import (
    _branch_bindings,
)
from scripts.run_objective_ancestral_queue_artifact import _active_frontier
from scripts.run_typed_hard_clause_escalation_artifact import (
    _load_query_runtime,
    _policies,
    _public_workload,
    _resolve_workloads,
)

FORMAL_SCHEMA_VERSION = "boundflow.objective-branch-whole-query-formal/v1"
WORKER_SCHEMA_VERSION = "boundflow.objective-branch-whole-query-worker/v1"
MANIFEST_SCHEMA_VERSION = "boundflow.objective-branch-whole-query-manifest/v1"
ARTIFACT_DIR = Path(
    "artifacts/objective-branch-whole-query/"
    "vnncomp21-resnet2b-property0-three-repeat-cpu-formal-v1"
)
TORCH_THREADS = 8
WORKER_TIMEOUT_SECONDS = 180
REPEAT_COUNT = 3
CLAUSE_COUNT = 9
SELECTED_COUNT = 2
WHOLE_QUERY_TIMEOUT_NS = 60_000_000_000
MAX_COOPERATIVE_ELAPSED_NS = 70_000_000_000
EXPECTED_SELECTED = [2, 3]
EXPECTED_WIDEST_WORST = [-37.57428741455078, -35.90021514892578]


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
        "boundflow/runtime/native_objective_branch_shared_production_queue.py",
        "boundflow/runtime/native_objective_branch_shared_multi_clause_anytime.py",
        "boundflow/ir/objective_branch_shared_evaluator.py",
        "boundflow/runtime/native_objective_branch_shared_evaluator.py",
        "boundflow/ir/multi_clause_anytime.py",
        "boundflow/runtime/native_multi_clause_anytime.py",
        "scripts/run_objective_branch_whole_query_formal.py",
    )
    return _canonical_hash({path: _file_sha256(root / path) for path in paths})


def _semantic_worker(value: Mapping[str, Any]) -> dict[str, object]:
    return {
        key: value[key]
        for key in (
            "repeat_index",
            "protocol",
            "program",
            "floor",
            "decision",
            "slices",
            "cache",
            "aggregate",
            "runtime_trace",
            "checks",
            "correctness_gate_passed",
            "production_gate_passed",
            "claim_boundary",
            "performance_claimed",
        )
    }


def _binding_from_dict(value: Mapping[str, Any]) -> NativeObjectiveBranchBindingIR:
    return NativeObjectiveBranchBindingIR(**value)


def _correctness_gate(value: Mapping[str, Any]) -> bool:
    floor = value["floor"]
    decision = value["decision"]
    slices = value["slices"]
    cache = value["cache"]
    aggregate = value["aggregate"]
    checks = value["checks"]
    candidates = decision["candidates"]
    expected_rank = [
        item["original_clause_index"]
        for item in sorted(
            candidates,
            key=lambda item: (
                -item["root_lower_margin"],
                item["original_clause_index"],
            ),
        )
    ]
    if (
        floor["completed_original_clause_indices"] != list(range(CLAUSE_COUNT))
        or floor["status"] != "unknown"
        or floor["unresolved_original_clause_indices"] != list(range(CLAUSE_COUNT))
        or len(candidates) != CLAUSE_COUNT
        or decision["ranked_original_clause_indices"] != expected_rank
        or decision["selected_original_clause_indices"]
        != expected_rank[:SELECTED_COUNT]
        != EXPECTED_SELECTED
        or len(slices) != SELECTED_COUNT
        or aggregate["original_clause_indices"] != list(range(CLAUSE_COUNT))
        or aggregate["final_status"] not in {"verified", "unsafe", "unknown"}
        or not all(checks.values())
    ):
        return False
    previous_finished = 0
    committed_event_count = 0
    for position, item in enumerate(slices):
        slice_ir = item["slice"]
        remaining = max(
            0, WHOLE_QUERY_TIMEOUT_NS - slice_ir["dispatch_started_elapsed_ns"]
        )
        remaining_count = SELECTED_COUNT - position
        allocation = remaining // remaining_count
        cutoff = min(
            WHOLE_QUERY_TIMEOUT_NS,
            slice_ir["dispatch_started_elapsed_ns"] + allocation,
        )
        if (
            slice_ir["priority_position"] != position
            or slice_ir["original_clause_index"] != EXPECTED_SELECTED[position]
            or slice_ir["dispatch_started_elapsed_ns"] < previous_finished
            or slice_ir["remaining_before_ns"] != remaining
            or slice_ir["remaining_selected_count"] != remaining_count
            or slice_ir["allocated_slice_ns"] != allocation
            or slice_ir["slice_cutoff_elapsed_ns"] != cutoff
            or slice_ir["finished_elapsed_ns"] < slice_ir["dispatch_started_elapsed_ns"]
            or slice_ir["accepted_nodes"] != 1 + 2 * slice_ir["sibling_group_count"]
            and slice_ir["accepted_nodes"] != 0
        ):
            return False
        previous_finished = slice_ir["finished_elapsed_ns"]
        if item["packed"] is None:
            if (
                slice_ir["accepted_nodes"] != 0
                or item["branch_bindings"]
                or item["compiler_batch_count"] != 0
            ):
                return False
            continue
        packed = item["packed"]
        bindings = tuple(_binding_from_dict(row) for row in item["branch_bindings"])
        for binding in bindings:
            binding.validate()
        expected_branch_ids = set(packed["branch_candidate_node_ids"])
        if (
            slice_ir["accepted_nodes"] != packed["evaluation_count"]
            or item["batch_commit_count"] != 1 + slice_ir["sibling_group_count"]
            or item["compiler_batch_count"] != item["batch_commit_count"]
            or len({binding.node_id for binding in bindings}) != len(bindings)
            or {binding.node_id for binding in bindings} != expected_branch_ids
            or packed["branch_execution_count"] != len(bindings)
            or packed["objective_branch_policy_hash"]
            != NativeObjectiveBranchPolicy().stable_hash()
            or not all(
                _is_sha256(item[key])
                for key in (
                    "plan_hash",
                    "task_ir_hash",
                    "schedule_hash",
                    "queue_trace_hash",
                    "execution_trace_hash",
                    "verdict_trace_hash",
                )
            )
        ):
            return False
        committed_event_count += item["compiler_batch_count"]
    events = cache["events"]
    return bool(
        cache["template_count"] == (1 if events else 0)
        and cache["miss_count"] == (1 if events else 0)
        and cache["hit_count"] == len(events) - cache["miss_count"]
        and [item["event_index"] for item in events] == list(range(len(events)))
        and (not events or events[0]["outcome"] == "miss_compiled")
        and all(item["outcome"] == "hit_exact_contract" for item in events[1:])
        and committed_event_count <= len(events)
    )


def _production_gate(value: Mapping[str, Any]) -> bool:
    if not _correctness_gate(value):
        return False
    slices = value["slices"]
    return bool(
        value["runtime_trace"]["elapsed_ns"] <= MAX_COOPERATIVE_ELAPSED_NS
        and all(
            item["packed"] is not None
            and item["slice"]["accepted_nodes"] == 31
            and item["slice"]["sibling_group_count"] == 15
            and item["packed"]["worst_active_lower"] - EXPECTED_WIDEST_WORST[position]
            >= 1.0
            and item["packed"]["fallback_reason"] == "none"
            and item["packed"]["discarded_attempt_stage"] is None
            for position, item in enumerate(slices)
        )
    )


def _validate_worker(value: Mapping[str, Any]) -> None:
    protocol = value.get("protocol", {})
    if (
        value.get("schema_version") != WORKER_SCHEMA_VERSION
        or value.get("source", {}).get("native_code_revision") != _code_revision()
        or value.get("repeat_index") not in range(REPEAT_COUNT)
        or protocol.get("whole_query_timeout_seconds") != 60
        or protocol.get("original_clause_count") != CLAUSE_COUNT
        or protocol.get("selected_clause_count") != SELECTED_COUNT
        or protocol.get("branch_policy") != NativeObjectiveBranchPolicy().to_dict()
        or protocol.get("scoring_inside_global_deadline") is not True
        or value.get("performance_claimed") is not False
        or value.get("claim_boundary")
        != "objective_branch_global_deadline_production_admission_only"
        or value.get("correctness_gate_passed") != _correctness_gate(value)
        or value.get("production_gate_passed") != _production_gate(value)
        or value.get("result_hash") != _canonical_hash(_semantic_worker(value))
    ):
        raise ValueError("NRIR-40 worker result differs")


def _worker(args: argparse.Namespace) -> None:  # pylint: disable=too-many-statements
    import torch

    from boundflow.runtime.native_multi_clause_anytime import (
        compile_native_multi_clause_anytime_program,
    )
    from boundflow.runtime.native_objective_branch_shared_multi_clause_anytime import (
        execute_native_objective_branch_shared_multi_clause_anytime_program,
    )

    torch.set_num_threads(args.torch_threads)
    _query, tensors, module, input_spec = _load_query_runtime(
        args.model.resolve(), args.property.resolve(), "cifar10_resnet:000"
    )
    search_policy, optimizer_policy = _policies()
    branch_policy = NativeObjectiveBranchPolicy()
    query_id = f"nrir40:cifar10_resnet:000:property0:repeat{args.repeat_index}"
    program = compile_native_multi_clause_anytime_program(
        module,
        input_spec,
        linear_spec_C=tensors.linear_spec_c,
        thresholds=tensors.thresholds,
        plan_id=query_id,
        search_policy=search_policy,
        optimizer_policy=optimizer_policy,
    )
    measured_started_ns = time.monotonic_ns()
    execution = execute_native_objective_branch_shared_multi_clause_anytime_program(
        program,
        module,
        input_spec,
        linear_spec_C=tensors.linear_spec_c,
        thresholds=tensors.thresholds,
        query_id=query_id,
        search_policy=search_policy,
        optimizer_policy=optimizer_policy,
        branch_policy=branch_policy,
    )
    measured_elapsed_ns = time.monotonic_ns() - measured_started_ns
    sources = []
    for child in execution.floor.clause_executions:
        if child.accepted_before_deadline and child.query.clauses:
            sources.append(
                {
                    "original_clause_index": child.original_clause_index,
                    "plan_hash": child.refinement.program.plan.stable_hash(),
                    "semantic_trace_hash": (
                        intermediate_refinement_semantic_trace_hash(child.refinement)
                    ),
                    "final_intermediate_bounds_hash": intermediate_bounds_hash(
                        child.refinement.relu_pre
                    ),
                }
            )
    slice_rows: list[dict[str, object]] = []
    for item in execution.packed_executions:
        packed = item.packed
        packed_summary = None
        bindings: list[dict[str, object]] = []
        plan_hash = task_hash = schedule_hash = queue_hash = trace_hash = (
            verdict_hash
        ) = None
        batch_count = compiler_count = 0
        if packed is not None and item.verdict is not None:
            frontier = _active_frontier(packed.queue.trace)
            typed_bindings = _branch_bindings(packed)
            bindings = [binding.to_dict() for binding in typed_bindings]
            branch_ids = [
                evaluation.node.node_id
                for evaluation in packed.queue.trace.evaluations
                if evaluation.branch_candidate is not None
            ]
            packed_summary = {
                "evaluation_count": len(packed.queue.trace.evaluations),
                "maximum_depth": frontier["maximum_depth"],
                "worst_active_lower": frontier["worst_active_lower"],
                "branch_execution_count": len(packed.queue.objective_branch_executions),
                "branch_candidate_node_ids": branch_ids,
                "objective_branch_policy_hash": branch_policy.stable_hash(),
                "fallback_reason": packed.trace.fallback_reason,
                "discarded_attempt_stage": packed.trace.discarded_attempt_stage,
            }
            plan_hash = packed.plan.stable_hash()
            task_hash = packed.task_ir.stable_hash()
            schedule_hash = packed.schedule.stable_hash(packed.task_ir)
            queue_hash = packed.queue.trace.stable_hash()
            trace_hash = packed.trace.semantic_signature_hash
            verdict_hash = item.verdict.trace.stable_hash(packed.queue.trace)
            batch_count = len(packed.batch_commits)
            compiler_count = len(packed.compiler_batches)
        slice_rows.append(
            {
                "slice": item.slice_ir.to_dict(program.plan, execution.decision),
                "packed": packed_summary,
                "branch_bindings": bindings,
                "plan_hash": plan_hash,
                "task_ir_hash": task_hash,
                "schedule_hash": schedule_hash,
                "queue_trace_hash": queue_hash,
                "execution_trace_hash": trace_hash,
                "verdict_trace_hash": verdict_hash,
                "batch_commit_count": batch_count,
                "compiler_batch_count": compiler_count,
            }
        )
    slices = tuple(item.slice_ir for item in execution.packed_executions)
    result: dict[str, Any] = {
        "schema_version": WORKER_SCHEMA_VERSION,
        "source": {"native_code_revision": _code_revision()},
        "repeat_index": args.repeat_index,
        "protocol": {
            "whole_query_timeout_seconds": 60,
            "maximum_cooperative_elapsed_seconds": 70,
            "original_clause_count": CLAUSE_COUNT,
            "selected_clause_count": SELECTED_COUNT,
            "search_budget": {"max_nodes": 31, "max_depth": 4},
            "child_cap": 128,
            "torch_threads": args.torch_threads,
            "branch_policy": branch_policy.to_dict(),
            "scoring_inside_global_deadline": True,
        },
        "program": {
            "plan_hash": program.plan.stable_hash(),
            "task_ir_hash": program.task_ir.stable_hash(),
            "schedule_hash": program.schedule.stable_hash(program.task_ir),
        },
        "floor": {
            "trace_hash": execution.floor.trace.semantic_signature_hash,
            "elapsed_ns": execution.floor.trace.elapsed_ns,
            "completed_original_clause_indices": list(
                execution.floor.trace.completed_objective_clause_indices
            ),
            "status": execution.floor.trace.final_status,
            "unresolved_original_clause_indices": list(
                execution.floor.trace.final_unresolved_clause_indices
            ),
            "sources": sources,
        },
        "decision": execution.decision.to_dict(program.plan),
        "slices": slice_rows,
        "cache": {
            "template_count": len(execution.template_hashes),
            "template_hashes": list(execution.template_hashes),
            "events": [item.to_dict() for item in execution.cache_events],
            "miss_count": sum(
                item.outcome == "miss_compiled" for item in execution.cache_events
            ),
            "hit_count": sum(
                item.outcome == "hit_exact_contract" for item in execution.cache_events
            ),
        },
        "aggregate": execution.aggregate.to_dict(
            program.plan, execution.decision, slices
        ),
        "runtime_trace": {
            "trace_hash": execution.trace.semantic_signature_hash,
            "elapsed_ns": execution.trace.elapsed_ns,
            "measured_elapsed_ns": measured_elapsed_ns,
            "fallback_reasons": list(execution.trace.fallback_reasons),
            "actions": [item.to_dict() for item in execution.trace.actions],
        },
        "checks": {
            "all_runtime_validators_passed": True,
            "rank_recomputed_from_floor": tuple(
                item.original_clause_index
                for item in sorted(
                    execution.decision.candidates,
                    key=lambda item: (
                        -item.root_lower_margin,
                        item.original_clause_index,
                    ),
                )
            )
            == execution.decision.ranked_original_clause_indices,
            "exact_floor_sources": len(sources) == CLAUSE_COUNT,
            "dynamic_allocations_recomputed": all(
                item.slice_ir.remaining_before_ns
                == max(
                    0,
                    WHOLE_QUERY_TIMEOUT_NS - item.slice_ir.dispatch_started_elapsed_ns,
                )
                and item.slice_ir.allocated_slice_ns
                == item.slice_ir.remaining_before_ns
                // item.slice_ir.remaining_selected_count
                for item in execution.packed_executions
            ),
            "atomic_sibling_accounting": all(
                item.slice_ir.accepted_nodes
                in (0, 1 + 2 * item.slice_ir.sibling_group_count)
                for item in execution.packed_executions
            ),
            "objective_branch_not_erased": all(
                item.packed is None
                or item.packed.queue.objective_branch_policy == branch_policy
                for item in execution.packed_executions
            ),
        },
        "correctness_gate_passed": False,
        "production_gate_passed": False,
        "claim_boundary": (
            "objective_branch_global_deadline_production_admission_only"
        ),
        "performance_claimed": False,
    }
    result["correctness_gate_passed"] = _correctness_gate(result)
    result["production_gate_passed"] = _production_gate(result)
    result["result_hash"] = _canonical_hash(_semantic_worker(result))
    _validate_worker(result)
    _write_json(args.result_json.resolve(), result)
    print(
        _canonical_json(
            {
                "repeat": args.repeat_index,
                "correctness": result["correctness_gate_passed"],
                "production": result["production_gate_passed"],
                "floor_seconds": execution.floor.trace.elapsed_ns / 1e9,
                "whole_seconds": execution.trace.elapsed_ns / 1e9,
                "selected": list(execution.decision.selected_original_clause_indices),
                "accepted_nodes": [
                    item.slice_ir.accepted_nodes for item in execution.packed_executions
                ],
                "result_hash": result["result_hash"],
            }
        )
    )


def _formal_payload(
    workload: Mapping[str, object], results: list[Mapping[str, Any]], threads: int
) -> dict[str, object]:
    return {
        "workload": _public_workload(workload),
        "protocol": {
            "fresh_process_repeats": REPEAT_COUNT,
            "torch_threads": threads,
            "whole_query_timeout_seconds": 60,
            "maximum_cooperative_elapsed_seconds": 70,
        },
        "repeat_results": list(results),
        "floor_elapsed_ns": [int(row["floor"]["elapsed_ns"]) for row in results],
        "whole_elapsed_ns": [
            int(row["runtime_trace"]["elapsed_ns"]) for row in results
        ],
        "selected_original_clause_indices": [
            row["decision"]["selected_original_clause_indices"] for row in results
        ],
        "accepted_nodes": [
            [int(item["slice"]["accepted_nodes"]) for item in row["slices"]]
            for row in results
        ],
        "worst_active_lowers": [
            [
                None if item["packed"] is None else item["packed"]["worst_active_lower"]
                for item in row["slices"]
            ]
            for row in results
        ],
        "branch_execution_counts": [
            [
                (
                    0
                    if item["packed"] is None
                    else item["packed"]["branch_execution_count"]
                )
                for item in row["slices"]
            ]
            for row in results
        ],
        "cache_miss_counts": [int(row["cache"]["miss_count"]) for row in results],
        "all_correctness_gates_passed": all(
            bool(row["correctness_gate_passed"]) for row in results
        ),
        "all_production_gates_passed": all(
            bool(row["production_gate_passed"]) for row in results
        ),
    }


def validate_formal(value: Mapping[str, Any]) -> None:
    payload = value.get("formal_payload")
    if (
        value.get("schema_version") != FORMAL_SCHEMA_VERSION
        or value.get("status") not in {"validated-reduced", "no_go"}
        or value.get("source", {}).get("native_code_revision") != _code_revision()
        or value.get("performance_claimed") is not False
        or value.get("claim") != "objective_branch_global_deadline_production_admission"
        or not isinstance(payload, dict)
    ):
        raise ValueError("NRIR-40 formal envelope differs")
    results = payload.get("repeat_results")
    if not isinstance(results, list) or len(results) != REPEAT_COUNT:
        raise ValueError("NRIR-40 formal repeat coverage differs")
    for result in results:
        _validate_worker(result)
    workload = payload["workload"]
    threads = int(payload["protocol"]["torch_threads"])
    expected = _formal_payload(workload, results, threads)
    expected["workload"] = workload
    production = bool(payload["all_production_gates_passed"])
    if (
        expected != payload
        or payload.get("all_correctness_gates_passed") is not True
        or value.get("formal_payload_hash") != _canonical_hash(payload)
        or value.get("status") != ("validated-reduced" if production else "no_go")
    ):
        raise ValueError("NRIR-40 formal gate differs")


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


def _generate(args: argparse.Namespace) -> None:
    root = _repo_root()
    workloads = _resolve_workloads(args.benchmark_root.resolve())
    workload = next(
        item for item in workloads if item["workload_id"] == "cifar10_resnet:000"
    )
    artifact_dir = args.artifact_dir.resolve()
    files: dict[str, str] = {}
    results: list[Mapping[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="boundflow-nrir40-formal-") as temporary:
        temporary_root = Path(temporary)
        for repeat in range(REPEAT_COUNT):
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
                    f"NRIR-40 repeat {repeat} failed with {completed.returncode}: "
                    f"{completed.stdout[-12000:]}"
                )
            result = _load_json(result_path)
            _validate_worker(result)
            shard_path = artifact_dir / "shards" / f"repeat-{repeat}.json"
            _write_json(shard_path, result)
            files[str(shard_path.relative_to(artifact_dir))] = _file_sha256(shard_path)
            results.append(result)
            print(completed.stdout.strip())
    payload = _formal_payload(workload, results, args.torch_threads)
    passed = bool(payload["all_production_gates_passed"])
    formal = {
        "schema_version": FORMAL_SCHEMA_VERSION,
        "status": "validated-reduced" if passed else "no_go",
        "source": {"native_code_revision": _code_revision()},
        "claim": "objective_branch_global_deadline_production_admission",
        "formal_payload": payload,
        "formal_payload_hash": _canonical_hash(payload),
        "performance_claimed": False,
    }
    validate_formal(formal)
    formal_path = artifact_dir / "formal.json"
    _write_json(formal_path, formal)
    files["formal.json"] = _file_sha256(formal_path)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "files": files,
        "formal_hash": _canonical_hash(formal),
        "performance_claimed": False,
    }
    _write_json(artifact_dir / "manifest.json", manifest)
    whole_elapsed_ns = payload["whole_elapsed_ns"]
    if not isinstance(whole_elapsed_ns, list) or not all(
        isinstance(value, int) for value in whole_elapsed_ns
    ):
        raise TypeError("NRIR-40 whole elapsed vector differs")
    print(
        _canonical_json(
            {
                "status": formal["status"],
                "formal_payload_hash": formal["formal_payload_hash"],
                "accepted_nodes": payload["accepted_nodes"],
                "whole_seconds": [value / 1e9 for value in whole_elapsed_ns],
            }
        )
    )


def _replay(args: argparse.Namespace) -> None:
    artifact_dir = args.artifact_dir.resolve()
    formal = _load_json(artifact_dir / "formal.json")
    manifest = _load_json(artifact_dir / "manifest.json")
    workloads = _resolve_workloads(args.benchmark_root.resolve())
    workload = next(
        item for item in workloads if item["workload_id"] == "cifar10_resnet:000"
    )
    validate_formal(formal)
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise TypeError("NRIR-40 manifest files differ")
    repeat_results = formal["formal_payload"]["repeat_results"]
    shard_results = [
        _load_json(artifact_dir / "shards" / f"repeat-{index}.json")
        for index in range(REPEAT_COUNT)
    ]
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or any(
            _file_sha256(artifact_dir / path) != digest
            for path, digest in files.items()
        )
        or set(files)
        != {
            "formal.json",
            *(f"logs/repeat-{index}.log" for index in range(REPEAT_COUNT)),
            *(f"shards/repeat-{index}.json" for index in range(REPEAT_COUNT)),
        }
        or manifest.get("formal_hash") != _canonical_hash(formal)
        or manifest.get("performance_claimed") is not False
        or formal["formal_payload"]["workload"] != _public_workload(workload)
        or shard_results != repeat_results
    ):
        raise ValueError("NRIR-40 manifest/workload differs")
    print(
        _canonical_json(
            {
                "status": formal["status"],
                "formal_payload_hash": formal["formal_payload_hash"],
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
    elif args.command == "replay":
        _replay(args)
    else:
        _worker(args)


if __name__ == "__main__":
    main()
