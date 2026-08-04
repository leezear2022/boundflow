#!/usr/bin/env python3
"""Run the NRIR-31 three-topology objective-escalation pilot gate."""

# pylint: disable=too-many-locals,too-many-statements,missing-function-docstring
# pylint: disable=import-outside-toplevel,duplicate-code,protected-access

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

from scripts.run_typed_hard_clause_escalation_artifact import (
    _load_query_runtime,
    _policies,
    _public_workload,
    _resolve_workloads,
)

SCHEMA_VERSION = "boundflow.objective-hard-clause-escalation-pilot/v1"
WORKER_SCHEMA_VERSION = "boundflow.objective-hard-clause-escalation-pilot-worker/v1"
ARTIFACT_DIR = Path(
    "artifacts/objective-hard-clause-escalation/"
    "vnncomp21-three-topology-cpu-pilot-v1"
)
PREDECESSOR_EVIDENCE = Path(
    "artifacts/typed-hard-clause-escalation/"
    "vnncomp21-three-topology-cpu-v1/evidence.json"
)
TORCH_THREADS = 8
WORKER_TIMEOUT_SECONDS = 180


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
        "boundflow/ir/objective_hard_clause_escalation.py",
        "boundflow/runtime/native_objective_hard_clause_escalation.py",
        "boundflow/ir/hard_clause_escalation.py",
        "boundflow/runtime/native_hard_clause_escalation.py",
        "boundflow/ir/refinement.py",
        "boundflow/runtime/native_intermediate_refinement.py",
        "scripts/run_objective_hard_clause_escalation_pilot.py",
    )
    return _canonical_hash({path: _file_sha256(root / path) for path in paths})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--benchmark-root", type=Path, required=True)
    run.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    run.add_argument("--torch-threads", type=int, default=TORCH_THREADS)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--workload-id", required=True)
    worker.add_argument("--model", type=Path, required=True)
    worker.add_argument("--property", type=Path, required=True)
    worker.add_argument("--result-json", type=Path, required=True)
    worker.add_argument("--torch-threads", type=int, required=True)
    return parser.parse_args()


def _worker(args: argparse.Namespace) -> None:
    import torch

    from boundflow.runtime.native_intermediate_refinement import (
        intermediate_refinement_semantic_trace_hash,
    )
    from boundflow.runtime.native_objective_hard_clause_escalation import (
        compile_native_objective_hard_clause_escalation_program,
        execute_native_objective_hard_clause_escalation_program,
    )

    torch.set_num_threads(args.torch_threads)
    started_ns = time.perf_counter_ns()
    query, tensors, module, input_spec = _load_query_runtime(
        args.model, args.property, args.workload_id
    )
    search_policy, optimizer_policy = _policies()
    program = compile_native_objective_hard_clause_escalation_program(
        module,
        input_spec,
        linear_spec_C=tensors.linear_spec_c,
        thresholds=tensors.thresholds,
        plan_id=f"nrir31:{args.workload_id}",
        search_policy=search_policy,
        optimizer_policy=optimizer_policy,
    )
    setup_ns = time.perf_counter_ns() - started_ns
    execute_started_ns = time.perf_counter_ns()
    execution = execute_native_objective_hard_clause_escalation_program(
        program,
        module,
        input_spec,
        linear_spec_C=tensors.linear_spec_c,
        thresholds=tensors.thresholds,
        query_id=f"nrir31:{query.query_id}",
        search_policy=search_policy,
        optimizer_policy=optimizer_policy,
    )
    execution_ns = time.perf_counter_ns() - execute_started_ns
    shared = execution.shared_refinement
    child_rows = []
    for child, trace in zip(execution.clause_executions, execution.trace.clause_traces):
        child_rows.append(
            {
                **trace.to_dict(),
                "refinement_program_hashes": child.refinement_program.hashes(),
                "query_status": child.query.trace.status,
                "query_reason": child.query.trace.reason,
                "evaluated_nodes": (
                    0
                    if not child.query.clauses
                    else len(child.query.clauses[0].queue.trace.evaluations)
                ),
            }
        )
    result = {
        "schema_version": WORKER_SCHEMA_VERSION,
        "workload_id": args.workload_id,
        "setup_ns": setup_ns,
        "execution_ns": execution_ns,
        "worker_elapsed_ns": time.perf_counter_ns() - started_ns,
        "program": program.to_dict(),
        "control_trace": execution.trace.to_dict(),
        "baseline_query_trace_hash": execution.baseline.trace.stable_hash(),
        "shared_refinement_semantic_trace_hash": (
            None
            if shared is None
            else intermediate_refinement_semantic_trace_hash(shared)
        ),
        "clauses": child_rows,
        "performance_claimed": False,
    }
    _write_json(args.result_json, result)
    print(
        _canonical_json(
            {
                "status": "ok",
                "workload_id": args.workload_id,
                "baseline_verified": list(
                    execution.trace.decision.baseline_verified_clause_indices
                ),
                "admitted": list(execution.trace.decision.escalated_clause_indices),
                "completed": list(execution.trace.completed_objective_clause_indices),
                "final_verified": list(execution.trace.final_verified_clause_indices),
                "fallback": execution.trace.fallback_reason,
                "execution_ns": execution_ns,
            }
        )
    )


def _predecessor_rows() -> tuple[dict[str, Any], dict[str, dict[int, float]]]:
    evidence = _load_json(_repo_root() / PREDECESSOR_EVIDENCE)
    summaries = evidence["summaries"]
    if not isinstance(summaries, dict):
        raise TypeError("NRIR-30 summaries are invalid")
    roots: dict[str, dict[int, float]] = {}
    for record in evidence["records"]:
        if record["repeat_index"] != 0:
            continue
        workload_id = str(record["workload_id"])
        escalation = record["result"]["escalation"]
        roots[workload_id] = (
            {}
            if escalation is None
            else {
                int(clause["original_clause_index"]): float(clause["root_lower"])
                for clause in escalation["clauses"]
            }
        )
    return summaries, roots


def _comparison(
    result: dict[str, Any],
    predecessor_summary: dict[str, Any],
    predecessor_roots: Mapping[int, float],
) -> dict[str, Any]:
    trace = result["control_trace"]
    old_verified = set(predecessor_summary["final_verified_clause_indices"][0])
    new_verified = set(trace["final_verified_clause_indices"])
    root_rows = []
    for clause in result["clauses"]:
        ordinal = int(clause["original_clause_index"])
        if ordinal not in predecessor_roots or clause["root_lower"] is None:
            continue
        delta = float(clause["root_lower"]) - predecessor_roots[ordinal]
        root_rows.append(
            {
                "original_clause_index": ordinal,
                "nrir30_shared_root_lower": predecessor_roots[ordinal],
                "nrir31_objective_root_lower": clause["root_lower"],
                "root_lower_delta": delta,
                "non_regression_1e_5": delta >= -1e-5,
                "strict_improvement_gt_1e_4": delta > 1e-4,
            }
        )
    return {
        "nrir30_final_verified_clause_indices": sorted(old_verified),
        "nrir31_final_verified_clause_indices": sorted(new_verified),
        "final_verified_superset": old_verified <= new_verified,
        "strict_new_verified": bool(new_verified - old_verified),
        "root_comparisons": root_rows,
        "all_common_roots_non_regressing": bool(root_rows)
        and all(row["non_regression_1e_5"] for row in root_rows),
        "any_root_strict_improvement_gt_1e_4": any(
            row["strict_improvement_gt_1e_4"] for row in root_rows
        ),
    }


def _run(args: argparse.Namespace) -> None:
    root = _repo_root()
    artifact_dir = args.artifact_dir.resolve()
    workloads = _resolve_workloads(args.benchmark_root.resolve())
    predecessor_summaries, predecessor_roots = _predecessor_rows()
    records: list[dict[str, Any]] = []
    files: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="boundflow-nrir31-pilot-") as temporary:
        temporary_root = Path(temporary)
        for workload in workloads:
            workload_id = str(workload["workload_id"])
            result_path = temporary_root / f"{workload_id.replace(':', '-')}.json"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "worker",
                "--workload-id",
                workload_id,
                "--model",
                str(workload["model"]),
                "--property",
                str(workload["property"]),
                "--result-json",
                str(result_path),
                "--torch-threads",
                str(args.torch_threads),
            ]
            started_ns = time.perf_counter_ns()
            completed = subprocess.run(
                command,
                cwd=root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=WORKER_TIMEOUT_SECONDS,
                check=False,
            )
            e2e_ns = time.perf_counter_ns() - started_ns
            log_path = artifact_dir / "logs" / f"{workload_id.replace(':', '-')}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(completed.stdout, encoding="utf-8")
            relative_log = str(log_path.relative_to(artifact_dir))
            files[relative_log] = _file_sha256(log_path)
            if completed.returncode != 0 or not result_path.is_file():
                raise RuntimeError(
                    f"NRIR-31 {workload_id} worker failed with "
                    f"{completed.returncode}: {completed.stdout[-8000:]}"
                )
            result = _load_json(result_path)
            comparison = _comparison(
                result,
                predecessor_summaries[workload_id],
                predecessor_roots[workload_id],
            )
            records.append(
                {
                    "workload_id": workload_id,
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
                        "final_verified": comparison[
                            "nrir31_final_verified_clause_indices"
                        ],
                        "strict_new_verified": comparison["strict_new_verified"],
                        "root_deltas": [
                            row["root_lower_delta"]
                            for row in comparison["root_comparisons"]
                        ],
                    }
                )
            )
    strict_new_verified = any(
        record["comparison"]["strict_new_verified"] for record in records
    )
    resnet = next(
        record for record in records if record["workload_id"] == "cifar10_resnet:000"
    )
    resnet_tightness_gate = bool(
        resnet["comparison"]["all_common_roots_non_regressing"]
        and resnet["comparison"]["any_root_strict_improvement_gt_1e_4"]
    )
    all_final_supersets = all(
        record["comparison"]["final_verified_superset"] for record in records
    )
    pilot_gate_passed = all_final_supersets and (
        strict_new_verified or resnet_tightness_gate
    )
    evidence: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "source": {
            "native_code_revision": _code_revision(),
            "predecessor_evidence_sha256": _file_sha256(root / PREDECESSOR_EVIDENCE),
        },
        "protocol": {
            "workloads": 3,
            "repeats": 1,
            "torch_threads": args.torch_threads,
            "whole_query_timeout_seconds": 60,
            "baseline_budget": {"max_nodes": 7, "max_depth": 2},
            "objective_budget": {"max_nodes": 31, "max_depth": 4},
            "shared_refinement": "top_ambiguous_width_per_relu_v1:128:32:1",
            "objective_refinement": "objective_influence_width_per_relu_v1:128:32:1",
        },
        "workloads": [_public_workload(workload) for workload in workloads],
        "records": records,
        "gate": {
            "all_final_verified_supersets": all_final_supersets,
            "strict_new_verified": strict_new_verified,
            "resnet_tightness_gate": resnet_tightness_gate,
            "pilot_gate_passed": pilot_gate_passed,
            "next": (
                "run_three_fresh_repeats"
                if pilot_gate_passed
                else "close_validated_no_go_without_formal_repeats"
            ),
        },
        "performance_claimed": False,
    }
    evidence_path = artifact_dir / "pilot.json"
    _write_json(evidence_path, evidence)
    files["pilot.json"] = _file_sha256(evidence_path)
    manifest = {
        "schema_version": "boundflow.objective-hard-clause-escalation-pilot-manifest/v1",
        "files": files,
        "evidence_hash": _canonical_hash(evidence),
    }
    _write_json(artifact_dir / "manifest.json", manifest)
    gate = evidence["gate"]
    if not isinstance(gate, dict):
        raise TypeError("NRIR-31 pilot gate is invalid")
    print(_canonical_json({"status": "ok", **gate}))


def main() -> None:
    args = _parse_args()
    if args.command == "worker":
        _worker(args)
    else:
        _run(args)


if __name__ == "__main__":
    main()
