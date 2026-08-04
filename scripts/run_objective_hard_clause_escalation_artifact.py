#!/usr/bin/env python3
"""Generate or replay the NRIR-31 objective hard-clause artifact."""

# pylint: disable=too-many-lines,too-many-locals,too-many-statements
# pylint: disable=too-many-branches,missing-function-docstring
# pylint: disable=import-outside-toplevel,duplicate-code,protected-access
# pylint: disable=too-many-boolean-expressions

from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

from scripts.run_objective_hard_clause_escalation_pilot import (
    PREDECESSOR_EVIDENCE,
    TORCH_THREADS,
    WORKER_SCHEMA_VERSION,
    _canonical_hash,
    _canonical_json,
    _comparison,
    _file_sha256,
    _load_json,
    _predecessor_rows,
    _repo_root,
    _worker,
    _write_json,
)
from scripts.run_typed_hard_clause_escalation_artifact import (
    _load_query_runtime,
    _policies,
    _public_workload,
    _resolve_workloads,
)

ARTIFACT_SCHEMA_VERSION = "boundflow.objective-hard-clause-escalation-artifact/v1"
EVIDENCE_SCHEMA_VERSION = "boundflow.objective-hard-clause-escalation-evidence/v1"
ARTIFACT_DIR = Path(
    "artifacts/objective-hard-clause-escalation/vnncomp21-three-topology-cpu-v1"
)
PILOT_EVIDENCE = Path(
    "artifacts/objective-hard-clause-escalation/vnncomp21-three-topology-cpu-pilot-v1/pilot.json"
)
MANIFEST_FILE = "manifest.json"
EVIDENCE_FILE = "evidence.json"
REPEATS = 3
WORKER_TIMEOUT_SECONDS = 180


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


def _code_revision() -> str:
    root = _repo_root()
    paths = (
        "boundflow/ir/objective_hard_clause_escalation.py",
        "boundflow/runtime/native_objective_hard_clause_escalation.py",
        "boundflow/ir/hard_clause_escalation.py",
        "boundflow/runtime/native_hard_clause_escalation.py",
        "boundflow/ir/refinement.py",
        "boundflow/runtime/native_intermediate_refinement.py",
        "scripts/run_objective_hard_clause_escalation_pilot.py",
        "scripts/run_objective_hard_clause_escalation_artifact.py",
    )
    return _canonical_hash({path: _file_sha256(root / path) for path in paths})


def _sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_result(result: Mapping[str, Any]) -> None:
    if (
        result.get("schema_version") != WORKER_SCHEMA_VERSION
        or result.get("performance_claimed") is not False
        or int(result.get("execution_ns", -1)) < 0
    ):
        raise ValueError("NRIR-31 worker envelope differs")
    program = result.get("program")
    trace = result.get("control_trace")
    clauses = result.get("clauses")
    if (
        not isinstance(program, dict)
        or not isinstance(trace, dict)
        or not isinstance(clauses, list)
    ):
        raise TypeError("NRIR-31 typed worker payload differs")
    plan = program.get("plan")
    task_ir = program.get("task_ir")
    schedule = program.get("schedule")
    base = program.get("base_program")
    if not all(isinstance(value, dict) for value in (plan, task_ir, schedule, base)):
        raise TypeError("NRIR-31 program stack differs")
    assert isinstance(plan, dict)
    assert isinstance(task_ir, dict)
    assert isinstance(schedule, dict)
    assert isinstance(base, dict)
    base_plan = base.get("plan")
    policy = plan.get("objective_refinement_policy")
    if not isinstance(base_plan, dict) or not isinstance(policy, dict):
        raise TypeError("NRIR-31 policy stack differs")
    clause_count = int(plan.get("clause_count", -1))
    if (
        clause_count < 1
        or base_plan.get("whole_query_timeout_ns") != 60_000_000_000
        or base_plan.get("baseline_budget", {}).get("max_nodes") != 7
        or base_plan.get("baseline_budget", {}).get("max_depth") != 2
        or base_plan.get("escalation_budget", {}).get("max_nodes") != 31
        or base_plan.get("escalation_budget", {}).get("max_depth") != 4
        or base_plan.get("refinement_policy", {}).get("candidate_policy_id")
        != "top_ambiguous_width_per_relu_v1"
        or policy.get("candidate_policy_id") != "objective_influence_width_per_relu_v1"
        or policy.get("max_neurons_per_relu") != 128
        or policy.get("backward_chunk_size") != 32
        or len(task_ir.get("tasks", [])) != 4 + 3 * clause_count + 2
        or len(schedule.get("actions", [])) != 4 + 3 * clause_count + 2
        or trace.get("deadline_ns") != 60_000_000_000
        or trace.get("performance_claimed") is not False
        or len(trace.get("actions", [])) != 4 + 3 * clause_count + 2
    ):
        raise ValueError("NRIR-31 frozen Plan/Task/Schedule policy differs")
    decision = trace.get("decision")
    if not isinstance(decision, dict):
        raise TypeError("NRIR-31 decision differs")
    admitted = tuple(decision.get("escalated_clause_indices", []))
    unresolved = tuple(decision.get("baseline_unresolved_clause_indices", []))
    if admitted != unresolved:
        raise ValueError("NRIR-31 admission is not exact unresolved ordinals")
    shared_hash = result.get("shared_refinement_semantic_trace_hash")
    if admitted and not _sha256(shared_hash):
        raise ValueError("NRIR-31 shared source trace is missing")
    child_ordinals = []
    for clause in clauses:
        if not isinstance(clause, dict):
            raise TypeError("NRIR-31 clause row differs")
        ordinal = int(clause.get("original_clause_index", -1))
        child_ordinals.append(ordinal)
        if (
            ordinal not in admitted
            or clause.get("source_refinement_semantic_trace_hash") != shared_hash
            or not _sha256(clause.get("objective_hash"))
            or not _sha256(clause.get("refinement_plan_hash"))
            or not _sha256(clause.get("query_trace_hash"))
            or clause.get("accepted_before_deadline") is not True
            or clause.get("root_lower") is None
            or clause.get("root_upper") is None
        ):
            raise ValueError("NRIR-31 objective child provenance differs")
    if tuple(child_ordinals) != admitted:
        raise ValueError("NRIR-31 objective child ordinal coverage differs")
    final_verified = set(trace.get("final_verified_clause_indices", []))
    baseline_verified = set(decision.get("baseline_verified_clause_indices", []))
    if (
        not baseline_verified <= final_verified
        or trace.get("fallback_reason") != "none"
    ):
        raise ValueError("NRIR-31 aggregate non-regression differs")


def _summaries(records: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    summaries: dict[str, object] = {}
    for workload_id in sorted({str(record["workload_id"]) for record in records}):
        rows = [record for record in records if record["workload_id"] == workload_id]
        execution_ns = sorted(int(record["result"]["execution_ns"]) for record in rows)
        clause_ordinals = tuple(
            int(clause["original_clause_index"])
            for clause in rows[0]["result"]["clauses"]
        )
        root_deltas = {
            str(ordinal): [
                next(
                    float(item["root_lower_delta"])
                    for item in row["comparison"]["root_comparisons"]
                    if int(item["original_clause_index"]) == ordinal
                )
                for row in rows
            ]
            for ordinal in clause_ordinals
        }
        summaries[workload_id] = {
            "baseline_verified_clause_indices": [
                row["result"]["control_trace"]["decision"][
                    "baseline_verified_clause_indices"
                ]
                for row in rows
            ],
            "admitted_clause_indices": [
                row["result"]["control_trace"]["decision"]["escalated_clause_indices"]
                for row in rows
            ],
            "final_verified_clause_indices": [
                row["result"]["control_trace"]["final_verified_clause_indices"]
                for row in rows
            ],
            "fallback_reasons": [
                row["result"]["control_trace"]["fallback_reason"] for row in rows
            ],
            "root_lower_deltas": root_deltas,
            "minimum_root_lower_delta": min(
                delta for values in root_deltas.values() for delta in values
            ),
            "maximum_root_lower_delta": max(
                delta for values in root_deltas.values() for delta in values
            ),
            "median_execution_ns": int(statistics.median(execution_ns)),
            "raw_execution_ns": execution_ns,
            "performance_claimed": False,
        }
    return summaries


def validate_evidence_structure(evidence: Mapping[str, Any]) -> None:
    if (
        evidence.get("schema_version") != EVIDENCE_SCHEMA_VERSION
        or evidence.get("status") != "ok"
        or evidence.get("property_status") != "validated-reduced"
        or evidence.get("performance_claimed") is not False
    ):
        raise ValueError("NRIR-31 evidence envelope differs")
    protocol = evidence.get("protocol")
    records = evidence.get("records")
    source = evidence.get("source")
    if (
        not isinstance(protocol, dict)
        or not isinstance(records, list)
        or not isinstance(source, dict)
    ):
        raise TypeError("NRIR-31 evidence structure differs")
    if (
        protocol.get("workloads") != 3
        or protocol.get("repeats") != REPEATS
        or len(records) != 3 * REPEATS
        or not _sha256(source.get("native_code_revision"))
        or not _sha256(source.get("predecessor_evidence_sha256"))
        or not _sha256(source.get("pilot_evidence_sha256"))
    ):
        raise ValueError("NRIR-31 protocol/source differs")
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("NRIR-31 record differs")
        _validate_result(record["result"])
        comparison = record.get("comparison")
        if (
            not isinstance(comparison, dict)
            or comparison.get("final_verified_superset") is not True
            or comparison.get("all_common_roots_non_regressing") is not True
        ):
            raise ValueError("NRIR-31 comparison non-regression differs")
    summaries = _summaries(records)
    if evidence.get("summaries") != summaries:
        raise ValueError("NRIR-31 summaries differ")
    resnet = summaries["cifar10_resnet:000"]
    if (
        not isinstance(resnet, dict)
        or float(resnet["minimum_root_lower_delta"]) <= 1e-4
    ):
        raise ValueError("NRIR-31 ResNet strict-tightness gate differs")


def _worker_command(
    workload: Mapping[str, object], result_path: Path, torch_threads: int
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


def _generate(args: argparse.Namespace) -> None:
    root = _repo_root()
    artifact_dir = args.artifact_dir.resolve()
    workloads = _resolve_workloads(args.benchmark_root.resolve())
    predecessor_summaries, predecessor_roots = _predecessor_rows()
    records: list[dict[str, Any]] = []
    files: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="boundflow-nrir31-") as temporary:
        temporary_root = Path(temporary)
        for workload in workloads:
            workload_id = str(workload["workload_id"])
            for repeat_index in range(REPEATS):
                result_path = temporary_root / (
                    f"{workload_id.replace(':', '-')}-{repeat_index}.json"
                )
                started_ns = time.perf_counter_ns()
                completed = subprocess.run(
                    _worker_command(workload, result_path, args.torch_threads),
                    cwd=root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=WORKER_TIMEOUT_SECONDS,
                    check=False,
                )
                e2e_ns = time.perf_counter_ns() - started_ns
                log_path = (
                    artifact_dir
                    / "logs"
                    / (f"{workload_id.replace(':', '-')}-r{repeat_index}.log")
                )
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text(completed.stdout, encoding="utf-8")
                relative_log = str(log_path.relative_to(artifact_dir))
                files[relative_log] = _file_sha256(log_path)
                if completed.returncode != 0 or not result_path.is_file():
                    raise RuntimeError(
                        f"NRIR-31 {workload_id} r{repeat_index} failed with "
                        f"{completed.returncode}: {completed.stdout[-8000:]}"
                    )
                result = _load_json(result_path)
                _validate_result(result)
                comparison = _comparison(
                    result,
                    predecessor_summaries[workload_id],
                    predecessor_roots[workload_id],
                )
                records.append(
                    {
                        "workload_id": workload_id,
                        "repeat_index": repeat_index,
                        "e2e_ns": e2e_ns,
                        "log_path": relative_log,
                        "log_sha256": files[relative_log],
                        "result": result,
                        "comparison": comparison,
                    }
                )
                print(
                    _canonical_json(
                        {
                            "workload_id": workload_id,
                            "repeat_index": repeat_index,
                            "final_verified": comparison[
                                "nrir31_final_verified_clause_indices"
                            ],
                            "minimum_root_delta": min(
                                item["root_lower_delta"]
                                for item in comparison["root_comparisons"]
                            ),
                        }
                    )
                )
    summaries = _summaries(records)
    evidence: dict[str, Any] = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "ok",
        "property_status": "validated-reduced",
        "source": {
            "native_code_revision": _code_revision(),
            "predecessor_evidence_sha256": _file_sha256(root / PREDECESSOR_EVIDENCE),
            "pilot_evidence_sha256": _file_sha256(root / PILOT_EVIDENCE),
        },
        "protocol": {
            "workloads": 3,
            "repeats": REPEATS,
            "torch_threads": args.torch_threads,
            "whole_query_timeout_seconds": 60,
            "baseline_budget": {"max_nodes": 7, "max_depth": 2},
            "objective_budget": {"max_nodes": 31, "max_depth": 4},
            "shared_refinement": "top_ambiguous_width_per_relu_v1:128:32:1",
            "objective_refinement": "objective_influence_width_per_relu_v1:128:32:1",
        },
        "workloads": [_public_workload(workload) for workload in workloads],
        "records": records,
        "summaries": summaries,
        "claim_boundary": (
            "objective-directed refinement provides repeatable root-bound "
            "tightening while preserving NRIR-30 property coverage"
        ),
        "limitations": [
            "no new clause closure on the frozen three-workload set",
            "no speedup or performance claim",
            "CPU-only fixed public instances",
            "no external-verifier or complete-suite comparison",
            "not an ASPLOS-ready end-to-end evaluation",
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
    print(
        _canonical_json(
            {
                "status": "ok",
                "property_status": "validated-reduced",
                "records": len(records),
                "evidence_hash": manifest["evidence_hash"],
            }
        )
    )


def _replay(args: argparse.Namespace) -> None:
    root = _repo_root()
    artifact_dir = args.artifact_dir.resolve()
    manifest = _load_json(artifact_dir / MANIFEST_FILE)
    evidence = _load_json(artifact_dir / EVIDENCE_FILE)
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise TypeError("NRIR-31 manifest files differ")
    for relative, expected in files.items():
        path = artifact_dir / relative
        if not path.is_file() or _file_sha256(path) != expected:
            raise ValueError(f"NRIR-31 artifact digest differs: {relative}")
    if (
        manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION
        or manifest.get("evidence_hash") != _canonical_hash(evidence)
        or evidence.get("source", {}).get("native_code_revision") != _code_revision()
        or evidence.get("source", {}).get("predecessor_evidence_sha256")
        != _file_sha256(root / PREDECESSOR_EVIDENCE)
        or evidence.get("source", {}).get("pilot_evidence_sha256")
        != _file_sha256(root / PILOT_EVIDENCE)
    ):
        raise ValueError("NRIR-31 replay source identity differs")
    validate_evidence_structure(evidence)
    workloads = _resolve_workloads(args.benchmark_root.resolve())
    if evidence.get("workloads") != [
        _public_workload(workload) for workload in workloads
    ]:
        raise ValueError("NRIR-31 replay workload identity differs")
    from boundflow.runtime.native_objective_hard_clause_escalation import (
        compile_native_objective_hard_clause_escalation_program,
    )

    expected_programs: dict[str, dict[str, object]] = {}
    for workload in workloads:
        workload_id = str(workload["workload_id"])
        _query, tensors, module, input_spec = _load_query_runtime(
            Path(str(workload["model"])),
            Path(str(workload["property"])),
            workload_id,
        )
        search_policy, optimizer_policy = _policies()
        program = compile_native_objective_hard_clause_escalation_program(
            module,
            input_spec,
            linear_spec_C=tensors.linear_spec_c,
            thresholds=tensors.thresholds,
            plan_id=f"nrir31:{workload_id}",
            search_policy=search_policy,
            optimizer_policy=optimizer_policy,
        )
        expected_programs[workload_id] = program.to_dict()
    for record in evidence["records"]:
        if record["result"]["program"] != expected_programs[record["workload_id"]]:
            raise ValueError("NRIR-31 replay recompiled program differs")
    print(
        _canonical_json(
            {
                "status": "ok",
                "property_status": evidence["property_status"],
                "records": len(evidence["records"]),
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
