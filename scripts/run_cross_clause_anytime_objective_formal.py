#!/usr/bin/env python3
"""Run or replay three fresh NRIR-35 cross-clause anytime evaluations."""

# pylint: disable=too-many-lines,too-many-locals,too-many-statements
# pylint: disable=too-many-boolean-expressions,missing-function-docstring
# pylint: disable=import-outside-toplevel,duplicate-code

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

FORMAL_SCHEMA_VERSION = "boundflow.cross-clause-anytime-formal/v1"
WORKER_SCHEMA_VERSION = "boundflow.cross-clause-anytime-worker/v1"
MANIFEST_SCHEMA_VERSION = "boundflow.cross-clause-anytime-manifest/v1"
ARTIFACT_DIR = Path(
    "artifacts/cross-clause-anytime-objective-evaluator/"
    "vnncomp21-resnet2b-property0-three-repeat-cpu-formal-v1"
)
TORCH_THREADS = 8
WORKER_TIMEOUT_SECONDS = 180
REPEAT_COUNT = 3
CLAUSE_COUNT = 9
PACKED_CLAUSE_INDEX = 0


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
        "boundflow/ir/cross_clause_anytime.py",
        "boundflow/runtime/native_cross_clause_anytime.py",
        "boundflow/ir/objective_hard_clause_escalation.py",
        "boundflow/runtime/native_objective_hard_clause_escalation.py",
        "boundflow/ir/objective_ancestral_sibling_pack.py",
        "boundflow/runtime/native_objective_ancestral_sibling_pack.py",
        "boundflow/runtime/native_property_verdict.py",
        "scripts/run_cross_clause_anytime_objective_formal.py",
    )
    return _canonical_hash({path: _file_sha256(root / path) for path in paths})


def _semantic_worker(result: Mapping[str, Any]) -> dict[str, object]:
    return {
        "repeat_index": result["repeat_index"],
        "protocol": result["protocol"],
        "program": result["program"],
        "floor": result["floor"],
        "decision": result["decision"],
        "packed": result["packed"],
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
    packed = result["packed"]
    aggregate = result["aggregate"]
    runtime_trace = result["runtime_trace"]
    checks = result["checks"]
    source = floor["packed_clause_source"]
    actions = runtime_trace["actions"]
    packed_status = None if packed is None else packed["verdict_status"]
    expected_verified = list(floor["verified_original_clause_indices"])
    expected_unresolved = list(floor["unresolved_original_clause_indices"])
    expected_unsafe = floor["unsafe_original_clause_index"]
    expected_status = floor["status"]
    if packed_status == "verified":
        expected_verified = sorted({*expected_verified, PACKED_CLAUSE_INDEX})
        expected_unresolved = [
            item for item in expected_unresolved if item != PACKED_CLAUSE_INDEX
        ]
        expected_unsafe = None
        expected_status = (
            "verified" if len(expected_verified) == CLAUSE_COUNT else "unknown"
        )
    elif packed_status == "unsafe":
        expected_status = "unsafe"
        expected_unsafe = PACKED_CLAUSE_INDEX
    return bool(
        floor["completed_original_clause_indices"] == list(range(CLAUSE_COUNT))
        and floor["status"] == "unknown"
        and floor["unresolved_original_clause_indices"] == list(range(CLAUSE_COUNT))
        and floor["unsafe_original_clause_index"] is None
        and decision["admitted"] is True
        and decision["admitted_original_clause_index"] == PACKED_CLAUSE_INDEX
        and decision["root_refinement_plan_hash"] == source["plan_hash"]
        and decision["root_refinement_semantic_trace_hash"]
        == source["semantic_trace_hash"]
        and decision["root_final_intermediate_bounds_hash"]
        == source["final_intermediate_bounds_hash"]
        and packed is not None
        and packed["accepted_nodes"] >= 3
        and packed["accepted_nodes"] == 1 + 2 * packed["sibling_group_count"]
        and packed["all_groups_atomic_pairs"] is True
        and packed["source_elapsed_ns"] >= floor["elapsed_ns"]
        and packed["whole_elapsed_ns"] >= packed["source_elapsed_ns"]
        and aggregate["original_clause_indices"] == list(range(CLAUSE_COUNT))
        and set(floor["verified_original_clause_indices"])
        <= set(aggregate["final_verified_clause_indices"])
        and aggregate["floor_verified_clause_indices"]
        == floor["verified_original_clause_indices"]
        and aggregate["floor_unresolved_clause_indices"]
        == floor["unresolved_original_clause_indices"]
        and aggregate["final_status"] == expected_status
        and aggregate["final_verified_clause_indices"] == expected_verified
        and aggregate["final_unresolved_clause_indices"] == expected_unresolved
        and aggregate["final_unsafe_clause_index"] == expected_unsafe
        and len(actions) == 6
        and [action["sequence"] for action in actions] == list(range(6))
        and [action["kind"] for action in actions]
        == [
            "execute_floor",
            "decide_anytime",
            "compile_packed_plan",
            "execute_packed_queue",
            "aggregate_original_ordinals",
            "emit_result",
        ]
        and all(action["executed"] for action in actions)
        and checks["exact_floor_clause_source"] is True
        and checks["single_global_start_consumed"] is True
        and checks["floor_accounting_preserved"] is True
        and checks["six_stage_schedule_executed"] is True
        and checks["all_runtime_validators_passed"] is True
    )


def _validate_worker(result: Mapping[str, Any]) -> None:
    if (
        result.get("schema_version") != WORKER_SCHEMA_VERSION
        or result.get("source", {}).get("native_code_revision") != _code_revision()
        or result.get("repeat_index") not in range(REPEAT_COUNT)
        or result.get("protocol", {}).get("whole_query_timeout_seconds") != 60
        or result.get("protocol", {}).get("original_clause_count") != CLAUSE_COUNT
        or result.get("protocol", {}).get("packed_original_clause_index")
        != PACKED_CLAUSE_INDEX
        or result.get("performance_claimed") is not False
        or result.get("claim_boundary")
        != "cross_clause_preservation_and_additive_work_no_performance_claim"
        or result.get("worker_gate_passed") != _worker_gate(result)
        or result.get("result_hash") != _canonical_hash(_semantic_worker(result))
    ):
        raise ValueError("NRIR-35 formal worker result differs")


def _worker(args: argparse.Namespace) -> None:
    import torch

    from boundflow.runtime.native_cross_clause_anytime import (
        compile_native_cross_clause_anytime_program,
        execute_native_cross_clause_anytime_program,
    )
    from boundflow.runtime.native_intermediate_refinement import (
        intermediate_bounds_hash,
        intermediate_refinement_semantic_trace_hash,
    )

    torch.set_num_threads(args.torch_threads)
    _query, tensors, module, input_spec = _load_query_runtime(
        args.model.resolve(),
        args.property.resolve(),
        "cifar10_resnet:000",
    )
    search_policy, optimizer_policy = _policies()
    program = compile_native_cross_clause_anytime_program(
        module,
        input_spec,
        linear_spec_C=tensors.linear_spec_c,
        thresholds=tensors.thresholds,
        plan_id=f"nrir35:cifar10_resnet:000:property0:repeat{args.repeat_index}",
        search_policy=search_policy,
        optimizer_policy=optimizer_policy,
    )
    measured_started_ns = time.monotonic_ns()
    execution = execute_native_cross_clause_anytime_program(
        program,
        module,
        input_spec,
        linear_spec_C=tensors.linear_spec_c,
        thresholds=tensors.thresholds,
        query_id=f"nrir35:cifar10_resnet:000:property0:repeat{args.repeat_index}",
        search_policy=search_policy,
        optimizer_policy=optimizer_policy,
    )
    measured_elapsed_ns = time.monotonic_ns() - measured_started_ns
    floor = execution.floor
    child = next(
        item
        for item in floor.clause_executions
        if item.original_clause_index == PACKED_CLAUSE_INDEX
    )
    packed = execution.packed
    packed_summary: dict[str, Any] | None = None
    if packed is not None and execution.packed_verdict is not None:
        frontier = _active_frontier(packed.queue.trace)
        packed_summary = {
            "plan_hash": packed.plan.stable_hash(),
            "task_ir_hash": packed.task_ir.stable_hash(),
            "schedule_hash": packed.schedule.stable_hash(packed.task_ir),
            "aggregate_trace_hash": packed.trace.semantic_signature_hash,
            "queue_trace_hash": packed.queue.trace.stable_hash(),
            "verdict_status": execution.packed_verdict.trace.status,
            "verdict_reason": execution.packed_verdict.trace.reason,
            "accepted_nodes": len(packed.queue.trace.evaluations),
            "maximum_depth": frontier["maximum_depth"],
            "worst_active_lower": frontier["worst_active_lower"],
            "sibling_group_count": len(packed.sibling_groups),
            "all_groups_atomic_pairs": all(
                item.child_branch_values == (-1, 1) for item in packed.sibling_groups
            ),
            "source_elapsed_ns": packed.trace.source_elapsed_ns,
            "queue_elapsed_ns": packed.trace.queue_elapsed_ns,
            "whole_elapsed_ns": packed.trace.whole_elapsed_ns,
            "fallback_reason": packed.trace.fallback_reason,
            "discarded_attempt_stage": packed.trace.discarded_attempt_stage,
        }
    exact_source = (
        execution.decision.root_refinement_plan_hash
        == child.refinement.program.plan.stable_hash()
        and execution.decision.root_refinement_semantic_trace_hash
        == intermediate_refinement_semantic_trace_hash(child.refinement)
        and execution.decision.root_final_intermediate_bounds_hash
        == intermediate_bounds_hash(child.refinement.relu_pre)
    )
    aggregate = execution.aggregate.to_dict(program.plan, execution.decision)
    result: dict[str, Any] = {
        "schema_version": WORKER_SCHEMA_VERSION,
        "source": {"native_code_revision": _code_revision()},
        "repeat_index": args.repeat_index,
        "protocol": {
            "whole_query_timeout_seconds": 60,
            "original_clause_count": CLAUSE_COUNT,
            "packed_original_clause_index": PACKED_CLAUSE_INDEX,
            "floor": "nrir31_objective_hard_clause_escalation",
            "anytime": "nrir34_sibling_packed_clause0",
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
            "trace_hash": floor.trace.semantic_signature_hash,
            "elapsed_ns": floor.trace.elapsed_ns,
            "completed_original_clause_indices": list(
                floor.trace.completed_objective_clause_indices
            ),
            "status": floor.trace.final_status,
            "verified_original_clause_indices": list(
                floor.trace.final_verified_clause_indices
            ),
            "unresolved_original_clause_indices": list(
                floor.trace.final_unresolved_clause_indices
            ),
            "unsafe_original_clause_index": floor.trace.final_unsafe_clause_index,
            "fallback_reason": floor.trace.fallback_reason,
            "packed_clause_source": {
                "plan_hash": child.refinement.program.plan.stable_hash(),
                "semantic_trace_hash": (
                    intermediate_refinement_semantic_trace_hash(child.refinement)
                ),
                "final_intermediate_bounds_hash": intermediate_bounds_hash(
                    child.refinement.relu_pre
                ),
            },
        },
        "decision": execution.decision.to_dict(program.plan),
        "packed": packed_summary,
        "aggregate": aggregate,
        "runtime_trace": {
            "trace_hash": execution.trace.semantic_signature_hash,
            "elapsed_ns": execution.trace.elapsed_ns,
            "measured_elapsed_ns": measured_elapsed_ns,
            "fallback_reason": execution.trace.fallback_reason,
            "actions": [action.to_dict() for action in execution.trace.actions],
        },
        "checks": {
            "exact_floor_clause_source": exact_source,
            "single_global_start_consumed": bool(
                packed is not None
                and packed.trace.source_elapsed_ns >= floor.trace.elapsed_ns
                and packed.trace.whole_elapsed_ns >= packed.trace.source_elapsed_ns
            ),
            "floor_accounting_preserved": bool(
                set(floor.trace.final_verified_clause_indices)
                <= set(execution.aggregate.final_verified_clause_indices)
                and execution.aggregate.original_clause_indices
                == tuple(range(CLAUSE_COUNT))
            ),
            "six_stage_schedule_executed": bool(
                len(execution.trace.actions) == 6
                and tuple(action.sequence for action in execution.trace.actions)
                == tuple(range(6))
            ),
            "all_runtime_validators_passed": True,
        },
        "worker_gate_passed": False,
        "claim_boundary": (
            "cross_clause_preservation_and_additive_work_no_performance_claim"
        ),
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
                "floor_seconds": floor.trace.elapsed_ns / 1e9,
                "whole_seconds": execution.trace.elapsed_ns / 1e9,
                "packed_nodes": (
                    None if packed_summary is None else packed_summary["accepted_nodes"]
                ),
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
            "packed_original_clause_index": PACKED_CLAUSE_INDEX,
        },
        "repeat_results": list(results),
        "floor_elapsed_ns": [int(row["floor"]["elapsed_ns"]) for row in results],
        "whole_elapsed_ns": [
            int(row["runtime_trace"]["elapsed_ns"]) for row in results
        ],
        "packed_accepted_nodes": [
            int(row["packed"]["accepted_nodes"]) for row in results
        ],
        "all_original_ordinals_preserved": all(
            row["aggregate"]["original_clause_indices"] == list(range(CLAUSE_COUNT))
            for row in results
        ),
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
        or formal.get("claim") != "cross_clause_preservation_and_additive_work_only"
    ):
        raise ValueError("NRIR-35 formal envelope differs")
    payload = formal.get("formal_payload")
    if not isinstance(payload, dict):
        raise TypeError("NRIR-35 formal payload differs")
    results = payload.get("repeat_results")
    if not isinstance(results, list) or len(results) != REPEAT_COUNT:
        raise ValueError("NRIR-35 formal repeat coverage differs")
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
        raise ValueError("NRIR-35 formal gate differs")


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
    with tempfile.TemporaryDirectory(prefix="boundflow-nrir35-formal-") as temporary:
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
                    f"NRIR-35 repeat {repeat} failed with {completed.returncode}: "
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
        "claim": "cross_clause_preservation_and_additive_work_only",
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
                "packed_nodes": payload["packed_accepted_nodes"],
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
        raise TypeError("NRIR-35 formal manifest files differ")
    for relative, expected in files.items():
        path = artifact_dir / relative
        if not path.is_file() or _file_sha256(path) != expected:
            raise ValueError(f"NRIR-35 formal digest differs: {relative}")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION or manifest.get(
        "formal_hash"
    ) != _canonical_hash(formal):
        raise ValueError("NRIR-35 formal manifest identity differs")
    validate_formal(formal)
    workloads = _resolve_workloads(args.benchmark_root.resolve())
    workload = next(
        item for item in workloads if item["workload_id"] == "cifar10_resnet:000"
    )
    if formal["formal_payload"]["workload"] != _public_workload(workload):
        raise ValueError("NRIR-35 formal workload differs")
    for repeat, result in enumerate(formal["formal_payload"]["repeat_results"]):
        shard = _load_json(artifact_dir / "shards" / f"repeat-{repeat}.json")
        _validate_worker(shard)
        if shard["result_hash"] != result["result_hash"]:
            raise ValueError("NRIR-35 formal shard/result binding differs")
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
