#!/usr/bin/env python3
"""Generate or replay the NRIR-32 objective-ancestral queue artifact."""

# pylint: disable=too-many-lines,too-many-locals,too-many-statements
# pylint: disable=too-many-branches,too-many-boolean-expressions
# pylint: disable=missing-function-docstring,import-outside-toplevel
# pylint: disable=protected-access,duplicate-code

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

from scripts.run_typed_hard_clause_escalation_artifact import (
    _load_query_runtime,
    _policies,
    _public_workload,
    _resolve_workloads,
)

ARTIFACT_SCHEMA_VERSION = "boundflow.objective-ancestral-queue-artifact/v1"
EVIDENCE_SCHEMA_VERSION = "boundflow.objective-ancestral-queue-evidence/v1"
WORKER_SCHEMA_VERSION = "boundflow.objective-ancestral-queue-worker/v1"
ARTIFACT_DIR = Path(
    "artifacts/objective-ancestral-hard-clause-escalation/"
    "vnncomp21-resnet2b-clause0-cpu-v1"
)
FEASIBILITY_PILOT = Path(
    "artifacts/objective-ancestral-hard-clause-escalation/"
    "vnncomp21-resnet2b-clause0-first-child-cpu-pilot-v1/pilot.json"
)
MANIFEST_FILE = "manifest.json"
EVIDENCE_FILE = "evidence.json"
CLAUSE_INDEX = 0
REPEATS = 3
TORCH_THREADS = 8
WORKER_TIMEOUT_SECONDS = 240


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
        "boundflow/ir/objective_ancestral_queue.py",
        "boundflow/runtime/native_objective_ancestral_queue.py",
        "boundflow/ir/refinement.py",
        "boundflow/runtime/native_intermediate_refinement.py",
        "boundflow/runtime/native_optimized_relu_split_bab_runtime.py",
        "boundflow/runtime/native_parametric_production_complete_query.py",
        "scripts/run_objective_ancestral_queue_artifact.py",
    )
    return _canonical_hash({path: _file_sha256(root / path) for path in paths})


def _sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _active_frontier(queue_trace: Any) -> dict[str, Any]:
    evaluations = {item.node.node_id: item for item in queue_trace.evaluations}
    terminal = tuple(
        decision.node_id
        for decision in queue_trace.decisions
        if decision.kind == "terminal"
    )
    active = tuple(queue_trace.final_frontier_node_ids) + terminal
    if not active:
        active = tuple(
            item.node.node_id
            for item in queue_trace.evaluations
            if item.lower < queue_trace.config.threshold
        )
    active_rows = tuple(
        {
            "node_id": node_id,
            "split_state_hash": evaluations[node_id].node.split_state_hash,
            "depth": evaluations[node_id].node.depth,
            "lower": evaluations[node_id].lower,
            "upper": evaluations[node_id].upper,
        }
        for node_id in active
    )
    return {
        "evaluated_nodes": len(queue_trace.evaluations),
        "decision_count": len(queue_trace.decisions),
        "frontier_count": len(queue_trace.final_frontier_node_ids),
        "maximum_depth": max(item.node.depth for item in queue_trace.evaluations),
        "root_lower": queue_trace.evaluations[0].lower,
        "root_upper": queue_trace.evaluations[0].upper,
        "active_domains": list(active_rows),
        "worst_active_lower": min(row["lower"] for row in active_rows),
        "minimum_active_upper": min(row["upper"] for row in active_rows),
    }


def _build_root_source(module: Any, input_spec: Any, objective: Any):
    from boundflow.ir.refinement import NativeIntermediateRefinementPolicyIR
    from boundflow.runtime.native_intermediate_refinement import (
        compile_native_intermediate_refinement_program,
        execute_native_intermediate_refinement_program,
    )

    shared_program = compile_native_intermediate_refinement_program(
        module,
        input_spec,
        policy=NativeIntermediateRefinementPolicyIR(
            passes=1, max_neurons_per_relu=128, backward_chunk_size=32
        ),
        plan_id="nrir32:cifar10_resnet:000:clause0:shared",
    )
    shared = execute_native_intermediate_refinement_program(
        shared_program, module, input_spec
    )
    root_program = compile_native_intermediate_refinement_program(
        module,
        input_spec,
        policy=NativeIntermediateRefinementPolicyIR(
            passes=1,
            max_neurons_per_relu=128,
            backward_chunk_size=32,
            candidate_policy_id="objective_influence_width_per_relu_v1",
        ),
        plan_id="nrir32:cifar10_resnet:000:clause0:objective-root",
        linear_spec_C=objective,
        source_refinement_execution=shared,
    )
    root = execute_native_intermediate_refinement_program(
        root_program, module, input_spec
    )
    return shared, root


def _worker(args: argparse.Namespace) -> None:
    import torch

    from boundflow.ir.bound import IntermediateBoundSource
    from boundflow.runtime.complete_verifier_query import CompleteVerifierQueryPolicy
    from boundflow.runtime.native_intermediate_refinement import (
        intermediate_refinement_semantic_trace_hash,
    )
    from boundflow.runtime.native_objective_ancestral_queue import (
        compile_native_objective_ancestral_queue_plan,
        execute_native_objective_ancestral_queue,
    )
    from boundflow.runtime.native_parametric_production_complete_query import (
        execute_native_parametric_production_complete_verifier_query,
    )
    from boundflow.runtime.native_relu_split_bab_runtime import NativeReluSplitBabConfig

    torch.set_num_threads(args.torch_threads)
    _query, tensors, module, input_spec = _load_query_runtime(
        args.model, args.property, "cifar10_resnet:000"
    )
    objective = tensors.linear_spec_c[
        :, CLAUSE_INDEX : CLAUSE_INDEX + 1, :
    ].contiguous()
    threshold = tensors.thresholds[CLAUSE_INDEX : CLAUSE_INDEX + 1].contiguous()
    search_policy, optimizer_policy = _policies()
    whole_started_ns = time.monotonic_ns()
    shared, root = _build_root_source(module, input_spec, objective)
    plan = compile_native_objective_ancestral_queue_plan(
        module,
        input_spec,
        linear_spec_C=objective,
        threshold=threshold,
        root_refinement=root,
        optimizer_policy=optimizer_policy,
        plan_id="nrir32:cifar10_resnet:000:clause0:queue",
    )
    ancestral = execute_native_objective_ancestral_queue(
        plan,
        module,
        input_spec,
        linear_spec_C=objective,
        threshold=threshold,
        root_refinement=root,
        optimizer_policy=optimizer_policy,
        query_id="nrir32:cifar10_resnet:000:clause0:queue",
        whole_query_started_ns=whole_started_ns,
    )
    baseline = execute_native_parametric_production_complete_verifier_query(
        module,
        input_spec,
        linear_spec_C=objective,
        thresholds=threshold,
        query_id="nrir32:cifar10_resnet:000:clause0:root-global",
        query_policy=CompleteVerifierQueryPolicy(timeout_ns=60_000_000_000),
        search_policy=search_policy,
        queue_config=NativeReluSplitBabConfig(
            max_nodes=31,
            max_depth=4,
            expansion_batch_size=1,
            max_eval_batch_size=1,
        ),
        optimizer_policy=optimizer_policy,
        relu_pre_override=root.relu_pre,
        intermediate_bound_source=IntermediateBoundSource.NATIVE_REFINED,
    )
    if len(baseline.clauses) != 1:
        raise ValueError("NRIR-32 root-global baseline did not complete one clause")
    baseline_queue = baseline.clauses[0].queue.trace
    ancestral_frontier = _active_frontier(ancestral.queue.trace)
    baseline_frontier = _active_frontier(baseline_queue)
    root_delta = float(ancestral_frontier["root_lower"]) - float(
        baseline_frontier["root_lower"]
    )
    worst_delta = float(ancestral_frontier["worst_active_lower"]) - float(
        baseline_frontier["worst_active_lower"]
    )
    gate = {
        "root_lower_delta": root_delta,
        "root_parity_1e_5": abs(root_delta) <= 1e-5,
        "baseline_worst_active_lower": baseline_frontier["worst_active_lower"],
        "ancestral_worst_active_lower": ancestral_frontier["worst_active_lower"],
        "worst_active_lower_delta": worst_delta,
        "strict_frontier_improvement_gt_1e_4": worst_delta > 1e-4,
        "ancestral_accepted_nodes": ancestral_frontier["evaluated_nodes"],
        "ancestral_deadline_fallback": ancestral.trace.fallback_reason,
        "pilot_gate_passed": abs(root_delta) <= 1e-5 and worst_delta > 1e-4,
    }
    semantic = {
        "plan_hash": plan.stable_hash(),
        "ancestral_semantic_signature_hash": ancestral.trace.semantic_signature_hash,
        "baseline_queue_trace_hash": baseline_queue.stable_hash(),
        "ancestral_frontier": ancestral_frontier,
        "baseline_frontier": baseline_frontier,
        "gate": gate,
    }
    result = {
        "schema_version": WORKER_SCHEMA_VERSION,
        "workload_id": "cifar10_resnet:000",
        "clause_index": CLAUSE_INDEX,
        "root_source": {
            "shared_program_hashes": shared.program.hashes(),
            "shared_semantic_trace_hash": (
                intermediate_refinement_semantic_trace_hash(shared)
            ),
            "objective_program_hashes": root.program.hashes(),
            "objective_semantic_trace_hash": (
                intermediate_refinement_semantic_trace_hash(root)
            ),
        },
        "ancestral": ancestral.to_dict(),
        "ancestral_frontier": ancestral_frontier,
        "root_global_baseline": {
            "query_trace_hash": baseline.trace.stable_hash(),
            "queue_trace": baseline_queue.to_dict(),
            "queue_trace_hash": baseline_queue.stable_hash(),
            "frontier": baseline_frontier,
        },
        "gate": gate,
        "semantic_signature_hash": _canonical_hash(semantic),
        "worker_elapsed_ns": time.monotonic_ns() - whole_started_ns,
        "performance_claimed": False,
    }
    _write_json(args.result_json, result)
    print(_canonical_json({"status": "ok", **gate}))


def _validate_result(result: Mapping[str, Any]) -> None:
    if (
        result.get("schema_version") != WORKER_SCHEMA_VERSION
        or result.get("workload_id") != "cifar10_resnet:000"
        or result.get("clause_index") != CLAUSE_INDEX
        or result.get("performance_claimed") is not False
    ):
        raise ValueError("NRIR-32 worker envelope differs")
    ancestral = result.get("ancestral")
    gate = result.get("gate")
    root_source = result.get("root_source")
    baseline = result.get("root_global_baseline")
    if not all(
        isinstance(value, dict) for value in (ancestral, gate, root_source, baseline)
    ):
        raise TypeError("NRIR-32 worker structure differs")
    assert isinstance(ancestral, dict)
    assert isinstance(gate, dict)
    assert isinstance(root_source, dict)
    plan = ancestral.get("plan")
    task_ir = ancestral.get("task_ir")
    schedule = ancestral.get("schedule")
    trace = ancestral.get("trace")
    node_refinements = ancestral.get("node_refinements")
    if not all(
        isinstance(value, dict) for value in (plan, task_ir, schedule, trace)
    ) or not isinstance(node_refinements, list):
        raise TypeError("NRIR-32 typed program differs")
    assert isinstance(plan, dict)
    assert isinstance(task_ir, dict)
    assert isinstance(schedule, dict)
    assert isinstance(trace, dict)
    assert isinstance(baseline, dict)
    if (
        plan.get("search_budget", {}).get("max_nodes") != 31
        or plan.get("search_budget", {}).get("max_depth") != 4
        or plan.get("whole_query_timeout_ns") != 60_000_000_000
        or plan.get("child_refinement_policy", {}).get("max_neurons_per_relu") != 128
        or len(task_ir.get("tasks", [])) != len(schedule.get("actions", []))
        or len(node_refinements) != int(result["ancestral_frontier"]["evaluated_nodes"])
        or trace.get("fallback_reason") != "deadline_preserve_accepted_frontier"
        or gate.get("root_parity_1e_5") is not True
        or gate.get("strict_frontier_improvement_gt_1e_4") is not True
        or gate.get("pilot_gate_passed") is not True
        or float(gate.get("worst_active_lower_delta", 0.0)) <= 1e-4
    ):
        raise ValueError("NRIR-32 frozen gate differs")
    refinements = {item["node_id"]: item for item in node_refinements}
    if len(refinements) != len(node_refinements):
        raise ValueError("NRIR-32 refinement node IDs repeat")
    for item in node_refinements[1:]:
        parent = refinements.get(item["parent_node_id"])
        if (
            parent is None
            or item.get("source_intermediate_constraints_hash")
            != parent.get("final_intermediate_bounds_hash")
            or item.get("source_refinement_plan_hash")
            != parent.get("program_hashes", {}).get("refinement_plan_hash")
            or item.get("source_refinement_semantic_trace_hash")
            != parent.get("semantic_trace_hash")
            or item.get("source_consumption") != "sound_constraint_only"
        ):
            raise ValueError("NRIR-32 serialized parent lineage differs")
    semantic = {
        "plan_hash": ancestral["plan_hash"],
        "ancestral_semantic_signature_hash": trace["semantic_signature_hash"],
        "baseline_queue_trace_hash": baseline["queue_trace_hash"],
        "ancestral_frontier": result["ancestral_frontier"],
        "baseline_frontier": baseline["frontier"],
        "gate": gate,
    }
    if result.get("semantic_signature_hash") != _canonical_hash(semantic):
        raise ValueError("NRIR-32 worker semantic signature differs")


def _summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    deltas = [
        float(record["result"]["gate"]["worst_active_lower_delta"])
        for record in records
    ]
    accepted = [
        int(record["result"]["gate"]["ancestral_accepted_nodes"]) for record in records
    ]
    elapsed = [int(record["result"]["worker_elapsed_ns"]) for record in records]
    return {
        "root_global_worst_active_lower": [
            record["result"]["gate"]["baseline_worst_active_lower"]
            for record in records
        ],
        "ancestral_worst_active_lower": [
            record["result"]["gate"]["ancestral_worst_active_lower"]
            for record in records
        ],
        "worst_active_lower_deltas": deltas,
        "minimum_worst_active_lower_delta": min(deltas),
        "ancestral_accepted_nodes": accepted,
        "semantic_signature_hashes": [
            record["result"]["semantic_signature_hash"] for record in records
        ],
        "committed_queue_trace_hashes": [
            record["result"]["ancestral"]["trace"]["queue_trace_hash"]
            for record in records
        ],
        "committed_task_ir_hashes": [
            record["result"]["ancestral"]["task_ir_hash"] for record in records
        ],
        "committed_node_refinement_hashes": [
            _canonical_hash(record["result"]["ancestral"]["node_refinements"])
            for record in records
        ],
        "discarded_attempt_stages": [
            record["result"]["ancestral"]["trace"]["discarded_attempt_stage"]
            for record in records
        ],
        "median_worker_elapsed_ns": int(statistics.median(elapsed)),
        "performance_claimed": False,
    }


def validate_evidence_structure(evidence: Mapping[str, Any]) -> None:
    if (
        evidence.get("schema_version") != EVIDENCE_SCHEMA_VERSION
        or evidence.get("status") != "ok"
        or evidence.get("property_status") != "validated-reduced"
        or evidence.get("performance_claimed") is not False
    ):
        raise ValueError("NRIR-32 evidence envelope differs")
    records = evidence.get("records")
    source = evidence.get("source")
    protocol = evidence.get("protocol")
    if (
        not isinstance(records, list)
        or not isinstance(source, dict)
        or not isinstance(protocol, dict)
    ):
        raise TypeError("NRIR-32 evidence structure differs")
    if (
        len(records) != REPEATS
        or protocol.get("repeats") != REPEATS
        or protocol.get("workloads") != 1
        or protocol.get("clauses") != [CLAUSE_INDEX]
        or not _sha256(source.get("native_code_revision"))
        or not _sha256(source.get("feasibility_pilot_sha256"))
    ):
        raise ValueError("NRIR-32 source/protocol differs")
    for index, record in enumerate(records):
        if record.get("repeat_index") != index:
            raise ValueError("NRIR-32 repeat order differs")
        _validate_result(record["result"])
    summary = _summary(records)
    if evidence.get("summary") != summary:
        raise ValueError("NRIR-32 summary differs")
    if (
        float(summary["minimum_worst_active_lower_delta"]) <= 1e-4
        or len(set(summary["ancestral_accepted_nodes"])) != 1
        or len(set(summary["committed_queue_trace_hashes"])) != 1
        or len(set(summary["committed_task_ir_hashes"])) != 1
        or len(set(summary["committed_node_refinement_hashes"])) != 1
    ):
        raise ValueError("NRIR-32 repeated tightness gate differs")


def _worker_command(
    workload: Mapping[str, object], result: Path, threads: int
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
    records = []
    files: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="boundflow-nrir32-") as temporary:
        temporary_root = Path(temporary)
        for repeat_index in range(REPEATS):
            result_path = temporary_root / f"repeat-{repeat_index}.json"
            completed = subprocess.run(
                _worker_command(workload, result_path, args.torch_threads),
                cwd=root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=WORKER_TIMEOUT_SECONDS,
                check=False,
            )
            log_path = artifact_dir / "logs" / f"repeat-{repeat_index}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(completed.stdout, encoding="utf-8")
            relative_log = str(log_path.relative_to(artifact_dir))
            files[relative_log] = _file_sha256(log_path)
            if completed.returncode != 0 or not result_path.is_file():
                raise RuntimeError(
                    f"NRIR-32 repeat {repeat_index} failed with {completed.returncode}: "
                    f"{completed.stdout[-8000:]}"
                )
            result = _load_json(result_path)
            _validate_result(result)
            shard_path = artifact_dir / "shards" / f"repeat-{repeat_index}.json"
            _write_json(shard_path, result)
            relative_shard = str(shard_path.relative_to(artifact_dir))
            files[relative_shard] = _file_sha256(shard_path)
            records.append(
                {
                    "repeat_index": repeat_index,
                    "log_path": relative_log,
                    "log_sha256": files[relative_log],
                    "shard_path": relative_shard,
                    "shard_sha256": files[relative_shard],
                    "result": result,
                }
            )
            print(
                _canonical_json(
                    {
                        "repeat_index": repeat_index,
                        "accepted_nodes": result["gate"]["ancestral_accepted_nodes"],
                        "worst_delta": result["gate"]["worst_active_lower_delta"],
                        "semantic_signature_hash": result["semantic_signature_hash"],
                        "discarded_attempt_stage": result["ancestral"]["trace"][
                            "discarded_attempt_stage"
                        ],
                    }
                )
            )
    evidence = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "ok",
        "property_status": "validated-reduced",
        "source": {
            "native_code_revision": _code_revision(),
            "feasibility_pilot_sha256": _file_sha256(root / FEASIBILITY_PILOT),
        },
        "protocol": {
            "workloads": 1,
            "clauses": [CLAUSE_INDEX],
            "repeats": REPEATS,
            "torch_threads": args.torch_threads,
            "whole_query_timeout_seconds": 60,
            "search_budget": {"max_nodes": 31, "max_depth": 4},
            "child_refinement": "objective_influence_width_per_relu_v1:128:32:1",
            "performance_claimed": False,
        },
        "workload": _public_workload(workload),
        "records": records,
        "summary": _summary(records),
        "claim_boundary": (
            "typed objective-root ancestral propagation improves accepted-frontier "
            "lower bounds under cooperative whole-deadline execution"
        ),
        "limitations": [
            "single ResNet2B property and clause",
            "no new verified or unsafe property closure",
            "cooperative deadline can finish after 60 seconds while discarding late work",
            "ancestral path uses serial audit evaluator; timing is diagnostic only",
            "no GPU, competitor, complete suite, or ASPLOS-ready claim",
        ],
        "performance_claimed": False,
    }
    validate_evidence_structure(evidence)
    evidence_path = artifact_dir / EVIDENCE_FILE
    _write_json(evidence_path, evidence)
    files[EVIDENCE_FILE] = _file_sha256(evidence_path)
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "files": files,
        "evidence_hash": _canonical_hash(evidence),
    }
    _write_json(artifact_dir / MANIFEST_FILE, manifest)
    print(_canonical_json({"status": "ok", "evidence_hash": manifest["evidence_hash"]}))


def _replay(args: argparse.Namespace) -> None:
    root = _repo_root()
    artifact_dir = args.artifact_dir.resolve()
    manifest = _load_json(artifact_dir / MANIFEST_FILE)
    evidence = _load_json(artifact_dir / EVIDENCE_FILE)
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise TypeError("NRIR-32 manifest files differ")
    for relative, expected in files.items():
        path = artifact_dir / relative
        if not path.is_file() or _file_sha256(path) != expected:
            raise ValueError(f"NRIR-32 artifact digest differs: {relative}")
    if (
        manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION
        or manifest.get("evidence_hash") != _canonical_hash(evidence)
        or evidence.get("source", {}).get("native_code_revision") != _code_revision()
        or evidence.get("source", {}).get("feasibility_pilot_sha256")
        != _file_sha256(root / FEASIBILITY_PILOT)
    ):
        raise ValueError("NRIR-32 replay source identity differs")
    validate_evidence_structure(evidence)
    workloads = _resolve_workloads(args.benchmark_root.resolve())
    workload = next(
        item for item in workloads if item["workload_id"] == "cifar10_resnet:000"
    )
    if evidence.get("workload") != _public_workload(workload):
        raise ValueError("NRIR-32 replay workload identity differs")
    import torch

    from boundflow.runtime.native_objective_ancestral_queue import (
        compile_native_objective_ancestral_queue_plan,
    )

    torch.set_num_threads(args.torch_threads)
    _query, tensors, module, input_spec = _load_query_runtime(
        Path(str(workload["model"])),
        Path(str(workload["property"])),
        "cifar10_resnet:000",
    )
    objective = tensors.linear_spec_c[
        :, CLAUSE_INDEX : CLAUSE_INDEX + 1, :
    ].contiguous()
    threshold = tensors.thresholds[CLAUSE_INDEX : CLAUSE_INDEX + 1].contiguous()
    _search, optimizer = _policies()
    _shared, root_refinement = _build_root_source(module, input_spec, objective)
    plan = compile_native_objective_ancestral_queue_plan(
        module,
        input_spec,
        linear_spec_C=objective,
        threshold=threshold,
        root_refinement=root_refinement,
        optimizer_policy=optimizer,
        plan_id="nrir32:cifar10_resnet:000:clause0:queue",
    )
    for record in evidence["records"]:
        if record["result"]["ancestral"]["plan"] != plan.to_dict():
            raise ValueError("NRIR-32 replay recompiled Plan differs")
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
    if args.command == "worker":
        _worker(args)
    elif args.command == "generate":
        _generate(args)
    else:
        _replay(args)


if __name__ == "__main__":
    main()
