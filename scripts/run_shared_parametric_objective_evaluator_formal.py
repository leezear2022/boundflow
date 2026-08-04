#!/usr/bin/env python3
"""Generate or replay three fresh NRIR-37 shared-evaluator runs."""

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

from scripts.run_objective_ancestral_queue_artifact import _active_frontier
from scripts.run_typed_hard_clause_escalation_artifact import (
    _load_query_runtime,
    _policies,
    _public_workload,
    _resolve_workloads,
)
from boundflow.runtime.native_intermediate_refinement import (
    intermediate_bounds_hash,
    intermediate_refinement_semantic_trace_hash,
)

FORMAL_SCHEMA_VERSION = "boundflow.shared-parametric-objective-evaluator-formal/v1"
WORKER_SCHEMA_VERSION = "boundflow.shared-parametric-objective-evaluator-worker/v1"
MANIFEST_SCHEMA_VERSION = "boundflow.shared-parametric-objective-evaluator-manifest/v1"
ARTIFACT_DIR = Path(
    "artifacts/shared-parametric-objective-evaluator/"
    "vnncomp21-resnet2b-property0-three-repeat-cpu-formal-v1"
)
TORCH_THREADS = 8
WORKER_TIMEOUT_SECONDS = 180
REPEAT_COUNT = 3
CLAUSE_COUNT = 9
SELECTED_COUNT = 2
WHOLE_QUERY_TIMEOUT_NS = 60_000_000_000
EXPECTED_SELECTED = [2, 3]


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
        "boundflow/ir/shared_parametric_ancestral.py",
        "boundflow/runtime/native_shared_parametric_ancestral.py",
        "boundflow/runtime/native_shared_parametric_multi_clause_anytime.py",
        "boundflow/ir/parametric_optimizer.py",
        "boundflow/runtime/native_parametric_optimizer.py",
        "boundflow/ir/multi_clause_anytime.py",
        "boundflow/runtime/native_multi_clause_anytime.py",
        "scripts/run_shared_parametric_objective_evaluator_formal.py",
    )
    return _canonical_hash({path: _file_sha256(root / path) for path in paths})


def _semantic_worker(result: Mapping[str, Any]) -> dict[str, object]:
    return {
        "repeat_index": result["repeat_index"],
        "protocol": result["protocol"],
        "program": result["program"],
        "floor": result["floor"],
        "decision": result["decision"],
        "slices": result["slices"],
        "cache": result["cache"],
        "aggregate": result["aggregate"],
        "runtime_trace": result["runtime_trace"],
        "checks": result["checks"],
        "worker_gate_passed": result["worker_gate_passed"],
        "claim_boundary": result["claim_boundary"],
        "performance_claimed": False,
    }


def _worker_gate(result: Mapping[str, Any]) -> bool:
    floor = result["floor"]
    decision = result["decision"]
    slices = result["slices"]
    cache = result["cache"]
    aggregate = result["aggregate"]
    runtime_trace = result["runtime_trace"]
    checks = result["checks"]
    candidates = decision["candidates"]
    sources = floor["sources"]
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
    exact_sources = all(
        item["root_lower_margin"] == item["root_lower"] - item["threshold"]
        and any(
            source["original_clause_index"] == item["original_clause_index"]
            and source["plan_hash"] == item["root_refinement_plan_hash"]
            and source["semantic_trace_hash"]
            == item["root_refinement_semantic_trace_hash"]
            and source["final_intermediate_bounds_hash"]
            == item["root_final_intermediate_bounds_hash"]
            for source in sources
        )
        for item in candidates
    )
    slices_valid = len(slices) == SELECTED_COUNT
    previous_finished = 0
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
        compiler = item["compiler"]
        slices_valid = bool(
            slices_valid
            and slice_ir["priority_position"] == position
            and slice_ir["original_clause_index"] == EXPECTED_SELECTED[position]
            and slice_ir["dispatch_started_elapsed_ns"] >= previous_finished
            and slice_ir["remaining_before_ns"] == remaining
            and slice_ir["remaining_selected_count"] == remaining_count
            and slice_ir["allocated_slice_ns"] == allocation
            and slice_ir["slice_cutoff_elapsed_ns"] == cutoff
            and slice_ir["finished_elapsed_ns"]
            >= slice_ir["dispatch_started_elapsed_ns"]
            and slice_ir["accepted_nodes"] >= 3
            and slice_ir["accepted_nodes"] == 1 + 2 * slice_ir["sibling_group_count"]
            and slice_ir["packed_status"] in {"verified", "unsafe", "unknown"}
            and item["batch_commit_count"] == 1 + slice_ir["sibling_group_count"]
            and item["compiler_batch_count"] == item["batch_commit_count"]
            and compiler["all_same_template"] is True
            and compiler["selected_native_reexecution"] is False
            and all(
                isinstance(item[key], str) and len(item[key]) == 64
                for key in (
                    "plan_hash",
                    "task_ir_hash",
                    "schedule_hash",
                    "queue_trace_hash",
                    "execution_trace_hash",
                    "verdict_trace_hash",
                )
            )
        )
        previous_finished = slice_ir["finished_elapsed_ns"]
    events = cache["events"]
    return bool(
        floor["completed_original_clause_indices"] == list(range(CLAUSE_COUNT))
        and floor["status"] == "unknown"
        and floor["unresolved_original_clause_indices"] == list(range(CLAUSE_COUNT))
        and len(candidates) == CLAUSE_COUNT
        and len(sources) == CLAUSE_COUNT
        and exact_sources
        and decision["ranked_original_clause_indices"] == expected_rank
        and decision["selected_original_clause_indices"]
        == expected_rank[:SELECTED_COUNT]
        == EXPECTED_SELECTED
        and slices_valid
        and cache["template_count"] == 1
        and cache["miss_count"] == 1
        and cache["hit_count"] == len(events) - 1
        and [item["event_index"] for item in events] == list(range(len(events)))
        and events[0]["outcome"] == "miss_compiled"
        and all(item["outcome"] == "hit_exact_contract" for item in events[1:])
        and aggregate["original_clause_indices"] == list(range(CLAUSE_COUNT))
        and len(runtime_trace["actions"]) == 8
        and [item["sequence"] for item in runtime_trace["actions"]] == list(range(8))
        and all(item["executed"] for item in runtime_trace["actions"])
        and checks["all_runtime_validators_passed"] is True
        and checks["rank_recomputed_from_floor"] is True
        and checks["exact_floor_sources"] is True
        and checks["single_query_cache_owner"] is True
        and checks["no_selected_native_reexecution"] is True
        and checks["dynamic_allocations_recomputed"] is True
        and checks["atomic_sibling_accounting"] is True
    )


def _validate_worker(result: Mapping[str, Any]) -> None:
    protocol = result.get("protocol", {})
    if (
        result.get("schema_version") != WORKER_SCHEMA_VERSION
        or result.get("source", {}).get("native_code_revision") != _code_revision()
        or result.get("repeat_index") not in range(REPEAT_COUNT)
        or protocol.get("whole_query_timeout_seconds") != 60
        or protocol.get("original_clause_count") != CLAUSE_COUNT
        or protocol.get("selected_clause_count") != SELECTED_COUNT
        or result.get("performance_claimed") is not False
        or result.get("claim_boundary")
        != "shared_compiler_reuse_and_fixed_deadline_coverage_only"
        or result.get("worker_gate_passed") != _worker_gate(result)
        or result.get("result_hash") != _canonical_hash(_semantic_worker(result))
    ):
        raise ValueError("NRIR-37 formal worker result differs")


def _worker(args: argparse.Namespace) -> None:  # pylint: disable=too-many-statements
    import torch

    from boundflow.runtime.native_multi_clause_anytime import (
        compile_native_multi_clause_anytime_program,
    )
    from boundflow.runtime.native_shared_parametric_multi_clause_anytime import (
        execute_native_shared_parametric_multi_clause_anytime_program,
    )

    torch.set_num_threads(args.torch_threads)
    _query, tensors, module, input_spec = _load_query_runtime(
        args.model.resolve(), args.property.resolve(), "cifar10_resnet:000"
    )
    search_policy, optimizer_policy = _policies()
    query_id = f"nrir37:cifar10_resnet:000:property0:repeat{args.repeat_index}"
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
    execution = execute_native_shared_parametric_multi_clause_anytime_program(
        program,
        module,
        input_spec,
        linear_spec_C=tensors.linear_spec_c,
        thresholds=tensors.thresholds,
        query_id=query_id,
        search_policy=search_policy,
        optimizer_policy=optimizer_policy,
    )
    measured_elapsed_ns = time.monotonic_ns() - measured_started_ns
    source_rows: list[dict[str, object]] = []
    for child in execution.floor.clause_executions:
        if not child.accepted_before_deadline or not child.query.clauses:
            continue
        source_rows.append(
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
        assert packed is not None and item.verdict is not None
        frontier = _active_frontier(packed.queue.trace)
        compiler_templates = tuple(
            batch.template_hash for batch in packed.compiler_batches
        )
        slice_rows.append(
            {
                "slice": item.slice_ir.to_dict(program.plan, execution.decision),
                "plan_hash": packed.plan.stable_hash(),
                "task_ir_hash": packed.task_ir.stable_hash(),
                "schedule_hash": packed.schedule.stable_hash(packed.task_ir),
                "queue_trace_hash": packed.queue.trace.stable_hash(),
                "execution_trace_hash": packed.trace.semantic_signature_hash,
                "verdict_trace_hash": item.verdict.trace.stable_hash(
                    packed.queue.trace
                ),
                "verdict_reason": item.verdict.trace.reason,
                "batch_commit_count": len(packed.batch_commits),
                "compiler_batch_count": len(packed.compiler_batches),
                "maximum_depth": frontier["maximum_depth"],
                "worst_active_lower": frontier["worst_active_lower"],
                "fallback_reason": packed.trace.fallback_reason,
                "discarded_attempt_stage": packed.trace.discarded_attempt_stage,
                "compiler": {
                    "cache_outcomes": list(packed.trace.cache_outcomes),
                    "template_hashes": list(compiler_templates),
                    "all_same_template": len(set(compiler_templates)) == 1,
                    "selected_native_reexecution": (
                        packed.trace.selected_native_reexecution
                    ),
                },
            }
        )
    cache_events = [item.to_dict() for item in execution.cache_events]
    slices = tuple(item.slice_ir for item in execution.packed_executions)
    aggregate = execution.aggregate.to_dict(program.plan, execution.decision, slices)
    result: dict[str, Any] = {
        "schema_version": WORKER_SCHEMA_VERSION,
        "source": {"native_code_revision": _code_revision()},
        "repeat_index": args.repeat_index,
        "protocol": {
            "whole_query_timeout_seconds": 60,
            "original_clause_count": CLAUSE_COUNT,
            "selected_clause_count": SELECTED_COUNT,
            "priority": "root_lower_margin_desc_ordinal_asc",
            "allocation": "dynamic_equal_remaining_selected_v1",
            "floor": "nrir31_objective_hard_clause_escalation",
            "evaluator": "nrir37_shared_parametric_ancestral",
            "search_budget": {"max_nodes": 31, "max_depth": 4},
            "child_cap": 128,
            "torch_threads": args.torch_threads,
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
            "verified_original_clause_indices": list(
                execution.floor.trace.final_verified_clause_indices
            ),
            "unresolved_original_clause_indices": list(
                execution.floor.trace.final_unresolved_clause_indices
            ),
            "unsafe_original_clause_index": (
                execution.floor.trace.final_unsafe_clause_index
            ),
            "sources": source_rows,
        },
        "decision": execution.decision.to_dict(program.plan),
        "slices": slice_rows,
        "cache": {
            "scope": "one_query_cross_batch_cross_clause_v1",
            "template_count": len(execution.template_hashes),
            "template_hashes": list(execution.template_hashes),
            "events": cache_events,
            "miss_count": sum(
                item.outcome == "miss_compiled" for item in execution.cache_events
            ),
            "hit_count": sum(
                item.outcome == "hit_exact_contract" for item in execution.cache_events
            ),
        },
        "aggregate": aggregate,
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
            "exact_floor_sources": all(
                any(
                    source["original_clause_index"] == candidate.original_clause_index
                    and source["plan_hash"] == candidate.root_refinement_plan_hash
                    and source["semantic_trace_hash"]
                    == candidate.root_refinement_semantic_trace_hash
                    and source["final_intermediate_bounds_hash"]
                    == candidate.root_final_intermediate_bounds_hash
                    for source in source_rows
                )
                for candidate in execution.decision.candidates
            ),
            "single_query_cache_owner": bool(
                len(execution.template_hashes) == 1
                and sum(
                    item.outcome == "miss_compiled" for item in execution.cache_events
                )
                == 1
            ),
            "no_selected_native_reexecution": all(
                item.packed is not None
                and item.packed.trace.selected_native_reexecution is False
                for item in execution.packed_executions
            ),
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
                == 1 + 2 * item.slice_ir.sibling_group_count
                for item in execution.packed_executions
            ),
        },
        "worker_gate_passed": False,
        "claim_boundary": ("shared_compiler_reuse_and_fixed_deadline_coverage_only"),
        "performance_claimed": False,
    }
    result["worker_gate_passed"] = _worker_gate(result)
    result["result_hash"] = _canonical_hash(_semantic_worker(result))
    _validate_worker(result)
    _write_json(args.result_json.resolve(), result)
    print(
        _canonical_json(
            {
                "repeat": args.repeat_index,
                "status": "ok" if result["worker_gate_passed"] else "no_go",
                "floor_seconds": execution.floor.trace.elapsed_ns / 1e9,
                "whole_seconds": execution.trace.elapsed_ns / 1e9,
                "selected": list(execution.decision.selected_original_clause_indices),
                "packed_nodes": [
                    item.slice_ir.accepted_nodes for item in execution.packed_executions
                ],
                "cache_events": len(execution.cache_events),
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
            "original_clause_count": CLAUSE_COUNT,
            "selected_clause_count": SELECTED_COUNT,
        },
        "repeat_results": list(results),
        "floor_elapsed_ns": [int(row["floor"]["elapsed_ns"]) for row in results],
        "whole_elapsed_ns": [
            int(row["runtime_trace"]["elapsed_ns"]) for row in results
        ],
        "selected_original_clause_indices": [
            row["decision"]["selected_original_clause_indices"] for row in results
        ],
        "packed_accepted_nodes": [
            [int(item["slice"]["accepted_nodes"]) for item in row["slices"]]
            for row in results
        ],
        "cache_miss_counts": [int(row["cache"]["miss_count"]) for row in results],
        "all_worker_gates_passed": all(
            bool(row["worker_gate_passed"]) for row in results
        ),
    }


def validate_formal(formal: Mapping[str, Any]) -> None:
    if (
        formal.get("schema_version") != FORMAL_SCHEMA_VERSION
        or formal.get("status") not in {"validated-reduced", "no_go"}
        or formal.get("source", {}).get("native_code_revision") != _code_revision()
        or formal.get("performance_claimed") is not False
        or formal.get("claim")
        != "shared_parametric_compiler_reuse_and_fixed_deadline_coverage"
    ):
        raise ValueError("NRIR-37 formal envelope differs")
    payload = formal.get("formal_payload")
    if not isinstance(payload, dict):
        raise TypeError("NRIR-37 formal payload differs")
    results = payload.get("repeat_results")
    if not isinstance(results, list) or len(results) != REPEAT_COUNT:
        raise ValueError("NRIR-37 formal repeat coverage differs")
    for result in results:
        _validate_worker(result)
    workload = payload["workload"]
    threads = int(payload["protocol"]["torch_threads"])
    recalculated = _formal_payload(workload, results, threads)
    recalculated["workload"] = workload
    passed = bool(payload.get("all_worker_gates_passed"))
    if (
        recalculated != payload
        or formal.get("formal_payload_hash") != _canonical_hash(payload)
        or formal.get("status") != ("validated-reduced" if passed else "no_go")
    ):
        raise ValueError("NRIR-37 formal gate differs")


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
    with tempfile.TemporaryDirectory(prefix="boundflow-nrir37-formal-") as temporary:
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
                    f"NRIR-37 repeat {repeat} failed with {completed.returncode}: "
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
    passed = bool(payload["all_worker_gates_passed"])
    formal = {
        "schema_version": FORMAL_SCHEMA_VERSION,
        "status": "validated-reduced" if passed else "no_go",
        "source": {"native_code_revision": _code_revision()},
        "formal_payload": payload,
        "formal_payload_hash": _canonical_hash(payload),
        "claim": "shared_parametric_compiler_reuse_and_fixed_deadline_coverage",
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
    }
    _write_json(artifact_dir / "manifest.json", manifest)
    print(
        _canonical_json(
            {
                "status": formal["status"],
                "floor_seconds": [
                    int(row["floor"]["elapsed_ns"]) / 1e9 for row in results
                ],
                "whole_seconds": [
                    int(row["runtime_trace"]["elapsed_ns"]) / 1e9 for row in results
                ],
                "selected": payload["selected_original_clause_indices"],
                "packed_nodes": payload["packed_accepted_nodes"],
                "cache_miss_counts": payload["cache_miss_counts"],
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
        raise TypeError("NRIR-37 formal manifest files differ")
    for relative, expected in files.items():
        path = artifact_dir / relative
        if not path.is_file() or _file_sha256(path) != expected:
            raise ValueError(f"NRIR-37 formal digest differs: {relative}")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION or manifest.get(
        "formal_hash"
    ) != _canonical_hash(formal):
        raise ValueError("NRIR-37 formal manifest identity differs")
    validate_formal(formal)
    workloads = _resolve_workloads(args.benchmark_root.resolve())
    workload = next(
        item for item in workloads if item["workload_id"] == "cifar10_resnet:000"
    )
    if formal["formal_payload"]["workload"] != _public_workload(workload):
        raise ValueError("NRIR-37 formal workload differs")
    for repeat, result in enumerate(formal["formal_payload"]["repeat_results"]):
        shard = _load_json(artifact_dir / "shards" / f"repeat-{repeat}.json")
        _validate_worker(shard)
        if shard["result_hash"] != result["result_hash"]:
            raise ValueError("NRIR-37 formal shard/result binding differs")
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
