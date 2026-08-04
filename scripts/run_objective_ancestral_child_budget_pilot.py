#!/usr/bin/env python3
"""Run or replay the NRIR-33 five-cap fresh-process calibration pilot."""

# pylint: disable=too-many-lines,too-many-locals,too-many-statements
# pylint: disable=too-many-branches,too-many-boolean-expressions
# pylint: disable=missing-function-docstring,import-outside-toplevel
# pylint: disable=protected-access,duplicate-code

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

PILOT_SCHEMA_VERSION = "boundflow.objective-ancestral-child-budget-pilot/v1"
WORKER_SCHEMA_VERSION = "boundflow.objective-ancestral-child-budget-worker/v1"
MANIFEST_SCHEMA_VERSION = "boundflow.objective-ancestral-child-budget-manifest/v1"
ARTIFACT_DIR = Path(
    "artifacts/objective-ancestral-child-budget-pareto/"
    "vnncomp21-resnet2b-clause0-five-cap-cpu-pilot-v1"
)
CLAUSE_INDEX = 0
TORCH_THREADS = 8
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
    worker.add_argument("--model", type=Path, required=True)
    worker.add_argument("--property", type=Path, required=True)
    worker.add_argument("--cap", type=int, required=True)
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
        "boundflow/ir/objective_ancestral_child_budget.py",
        "boundflow/runtime/native_objective_ancestral_child_budget.py",
        "boundflow/ir/objective_ancestral_queue.py",
        "boundflow/runtime/native_objective_ancestral_queue.py",
        "boundflow/ir/refinement.py",
        "boundflow/runtime/native_intermediate_refinement.py",
        "scripts/run_objective_ancestral_child_budget_pilot.py",
    )
    return _canonical_hash({path: _file_sha256(root / path) for path in paths})


def _semantic_result(result: Mapping[str, Any]) -> dict[str, object]:
    return {
        "cap": result["cap"],
        "plan_hash": result["plan_hash"],
        "queue_trace_hash": result["queue_trace_hash"],
        "task_ir_hash": result["task_ir_hash"],
        "schedule_hash": result["schedule_hash"],
        "node_refinement_hash": result["node_refinement_hash"],
        "root_lower": result["root_lower"],
        "worst_active_lower": result["worst_active_lower"],
        "accepted_nodes": result["accepted_nodes"],
        "maximum_depth": result["maximum_depth"],
        "fallback_reason": result["fallback_reason"],
        "discarded_attempt_stage": result["discarded_attempt_stage"],
        "root_global_root_lower": result["root_global_root_lower"],
        "root_global_worst_active_lower": result["root_global_worst_active_lower"],
        "root_global_queue_trace_hash": result["root_global_queue_trace_hash"],
        "lineage_valid": result["lineage_valid"],
        "performance_claimed": False,
    }


def _worker(args: argparse.Namespace) -> None:
    import torch

    from boundflow.ir.bound import IntermediateBoundSource
    from boundflow.ir.objective_ancestral_child_budget import (
        NativeObjectiveAncestralChildBudgetPolicyIR,
    )
    from boundflow.runtime.complete_verifier_query import CompleteVerifierQueryPolicy
    from boundflow.runtime.native_objective_ancestral_child_budget import (
        compile_native_objective_ancestral_child_budget_plan,
        execute_native_objective_ancestral_child_budget_queue,
    )
    from boundflow.runtime.native_parametric_production_complete_query import (
        execute_native_parametric_production_complete_verifier_query,
    )
    from boundflow.runtime.native_relu_split_bab_runtime import NativeReluSplitBabConfig

    torch.set_num_threads(args.torch_threads)
    policy = NativeObjectiveAncestralChildBudgetPolicyIR()
    policy.validate()
    if args.cap not in policy.candidate_caps:
        raise ValueError("NRIR-33 worker cap is outside the frozen policy")
    _query, tensors, module, input_spec = _load_query_runtime(
        args.model, args.property, "cifar10_resnet:000"
    )
    objective = tensors.linear_spec_c[
        :, CLAUSE_INDEX : CLAUSE_INDEX + 1, :
    ].contiguous()
    threshold = tensors.thresholds[CLAUSE_INDEX : CLAUSE_INDEX + 1].contiguous()
    search_policy, optimizer_policy = _policies()
    whole_started_ns = time.monotonic_ns()
    _shared, root = _build_root_source(module, input_spec, objective)
    plan = compile_native_objective_ancestral_child_budget_plan(
        module,
        input_spec,
        linear_spec_C=objective,
        threshold=threshold,
        root_refinement=root,
        optimizer_policy=optimizer_policy,
        plan_id=f"nrir33:cifar10_resnet:000:clause0:cap{args.cap}",
        selected_cap=args.cap,
        budget_policy=policy,
    )
    execution = execute_native_objective_ancestral_child_budget_queue(
        plan,
        module,
        input_spec,
        linear_spec_C=objective,
        threshold=threshold,
        root_refinement=root,
        optimizer_policy=optimizer_policy,
        query_id=f"nrir33:cifar10_resnet:000:clause0:cap{args.cap}",
        whole_query_started_ns=whole_started_ns,
    )
    baseline = execute_native_parametric_production_complete_verifier_query(
        module,
        input_spec,
        linear_spec_C=objective,
        thresholds=threshold,
        query_id="nrir33:cifar10_resnet:000:clause0:root-global",
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
        raise ValueError("NRIR-33 root-global baseline clause coverage differs")
    ancestral_frontier = _active_frontier(execution.queue_execution.queue.trace)
    baseline_trace = baseline.clauses[0].queue.trace
    baseline_frontier = _active_frontier(baseline_trace)
    queue_execution = execution.queue_execution
    result: dict[str, Any] = {
        "schema_version": WORKER_SCHEMA_VERSION,
        "cap": args.cap,
        "plan": plan.to_dict(),
        "plan_hash": plan.stable_hash(),
        "queue_trace_hash": queue_execution.trace.queue_trace_hash,
        "task_ir_hash": queue_execution.trace.task_ir_hash,
        "schedule_hash": queue_execution.trace.schedule_hash,
        "node_refinement_hash": _canonical_hash(
            [item.semantic_dict() for item in queue_execution.node_refinements]
        ),
        "root_lower": ancestral_frontier["root_lower"],
        "worst_active_lower": ancestral_frontier["worst_active_lower"],
        "accepted_nodes": ancestral_frontier["evaluated_nodes"],
        "maximum_depth": ancestral_frontier["maximum_depth"],
        "frontier": ancestral_frontier,
        "fallback_reason": queue_execution.trace.fallback_reason,
        "discarded_attempt_stage": queue_execution.trace.discarded_attempt_stage,
        "root_global_root_lower": baseline_frontier["root_lower"],
        "root_global_worst_active_lower": baseline_frontier["worst_active_lower"],
        "root_global_queue_trace_hash": baseline_trace.stable_hash(),
        "lineage_valid": True,
        "worker_elapsed_ns": time.monotonic_ns() - whole_started_ns,
        "performance_claimed": False,
    }
    result["result_hash"] = _canonical_hash(_semantic_result(result))
    _write_json(args.result_json, result)
    print(
        _canonical_json(
            {
                "status": "ok",
                "cap": args.cap,
                "accepted_nodes": result["accepted_nodes"],
                "worst_active_lower": result["worst_active_lower"],
                "fallback_reason": result["fallback_reason"],
            }
        )
    )


def _validate_result(result: Mapping[str, Any]) -> None:
    from boundflow.ir.objective_ancestral_child_budget import (
        NativeObjectiveAncestralChildBudgetPolicyIR,
    )

    policy = NativeObjectiveAncestralChildBudgetPolicyIR()
    if (
        result.get("schema_version") != WORKER_SCHEMA_VERSION
        or result.get("cap") not in policy.candidate_caps
        or result.get("lineage_valid") is not True
        or result.get("performance_claimed") is not False
        or result.get("result_hash") != _canonical_hash(_semantic_result(result))
        or abs(
            float(result.get("root_lower", 0.0))
            - float(result.get("root_global_root_lower", 1.0))
        )
        > policy.root_parity_tolerance
        or int(result.get("accepted_nodes", 0)) < 1
    ):
        raise ValueError("NRIR-33 worker result differs")
    plan = result.get("plan")
    if (
        not isinstance(plan, dict)
        or plan.get("child_budget_policy") != policy.to_dict()
        or plan.get("child_budget_decision", {}).get("selected_cap") != result["cap"]
        or plan.get("child_budget_decision", {}).get("selection_mode")
        != "calibration_candidate"
        or plan.get("child_refinement_policy", {}).get("max_neurons_per_relu")
        != result["cap"]
    ):
        raise ValueError("NRIR-33 worker Plan/cap binding differs")


def _calibration_payload(
    workload: Mapping[str, object], results: list[Mapping[str, Any]]
) -> dict[str, object]:
    from boundflow.ir.objective_ancestral_child_budget import (
        NativeObjectiveAncestralChildBudgetPolicyIR,
    )

    policy = NativeObjectiveAncestralChildBudgetPolicyIR()
    ordered = sorted(results, key=lambda row: int(row["cap"]))
    baseline_values = {float(row["root_global_worst_active_lower"]) for row in ordered}
    baseline_root_values = {float(row["root_global_root_lower"]) for row in ordered}
    if len(baseline_values) != 1 or len(baseline_root_values) != 1:
        raise ValueError("NRIR-33 root-global baseline drifted across workers")
    return {
        "policy": policy.to_dict(),
        "workload": _public_workload(workload),
        "root_global_root_lower": next(iter(baseline_root_values)),
        "root_global_worst_active_lower": next(iter(baseline_values)),
        "candidate_results": [
            {
                "cap": row["cap"],
                "root_lower": row["root_lower"],
                "worst_active_lower": row["worst_active_lower"],
                "accepted_nodes": row["accepted_nodes"],
                "maximum_depth": row["maximum_depth"],
                "fallback_reason": row["fallback_reason"],
                "discarded_attempt_stage": row["discarded_attempt_stage"],
                "lineage_valid": row["lineage_valid"],
                "result_hash": row["result_hash"],
                "worker_elapsed_ns": row["worker_elapsed_ns"],
                "performance_claimed": False,
            }
            for row in ordered
        ],
        "performance_claimed": False,
    }


def _compile_decision(payload: Mapping[str, Any]):
    from boundflow.ir.objective_ancestral_child_budget import (
        NativeObjectiveAncestralChildBudgetCalibrationIR,
        NativeObjectiveAncestralChildBudgetPolicyIR,
        compile_frozen_child_budget_decision,
    )

    policy = NativeObjectiveAncestralChildBudgetPolicyIR()
    rows = tuple(
        NativeObjectiveAncestralChildBudgetCalibrationIR(
            cap=int(row["cap"]),
            root_lower=float(row["root_lower"]),
            worst_active_lower=float(row["worst_active_lower"]),
            accepted_nodes=int(row["accepted_nodes"]),
            lineage_valid=bool(row["lineage_valid"]),
            result_hash=str(row["result_hash"]),
        )
        for row in payload["candidate_results"]
    )
    return compile_frozen_child_budget_decision(
        policy,
        calibration_evidence_hash=_canonical_hash(payload),
        root_global_worst_active_lower=float(payload["root_global_worst_active_lower"]),
        calibration_rows=rows,
    )


def validate_pilot(pilot: Mapping[str, Any]) -> None:
    if (
        pilot.get("schema_version") != PILOT_SCHEMA_VERSION
        or pilot.get("status") != "ok"
        or pilot.get("performance_claimed") is not False
        or pilot.get("source", {}).get("native_code_revision") != _code_revision()
    ):
        raise ValueError("NRIR-33 pilot envelope differs")
    payload = pilot.get("calibration_payload")
    if not isinstance(payload, dict):
        raise TypeError("NRIR-33 calibration payload differs")
    decision = _compile_decision(payload)
    if (
        pilot.get("calibration_evidence_hash") != _canonical_hash(payload)
        or pilot.get("decision") != decision.to_dict()
        or pilot.get("decision_hash")
        != decision.stable_hash(
            __import__(
                "boundflow.ir.objective_ancestral_child_budget",
                fromlist=["NativeObjectiveAncestralChildBudgetPolicyIR"],
            ).NativeObjectiveAncestralChildBudgetPolicyIR()
        )
        or pilot.get("selected_cap") != decision.selected_cap
    ):
        raise ValueError("NRIR-33 frozen pilot decision differs")


def _worker_command(
    workload: Mapping[str, object], cap: int, result_path: Path, threads: int
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--model",
        str(workload["model"]),
        "--property",
        str(workload["property"]),
        "--cap",
        str(cap),
        "--result-json",
        str(result_path),
        "--torch-threads",
        str(threads),
    ]


def _generate(args: argparse.Namespace) -> None:
    from boundflow.ir.objective_ancestral_child_budget import (
        NativeObjectiveAncestralChildBudgetPolicyIR,
    )

    root = _repo_root()
    workloads = _resolve_workloads(args.benchmark_root.resolve())
    workload = next(
        item for item in workloads if item["workload_id"] == "cifar10_resnet:000"
    )
    policy = NativeObjectiveAncestralChildBudgetPolicyIR()
    artifact_dir = args.artifact_dir.resolve()
    results: list[Mapping[str, Any]] = []
    files: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="boundflow-nrir33-pilot-") as temporary:
        temporary_root = Path(temporary)
        for order_index, cap in enumerate(policy.pilot_order):
            result_path = temporary_root / f"cap-{cap}.json"
            completed = subprocess.run(
                _worker_command(workload, cap, result_path, args.torch_threads),
                cwd=root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=WORKER_TIMEOUT_SECONDS,
                check=False,
            )
            log_path = artifact_dir / "logs" / f"order-{order_index}-cap-{cap}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(completed.stdout, encoding="utf-8")
            relative_log = str(log_path.relative_to(artifact_dir))
            files[relative_log] = _file_sha256(log_path)
            if completed.returncode != 0 or not result_path.is_file():
                raise RuntimeError(
                    f"NRIR-33 cap {cap} failed with {completed.returncode}: "
                    f"{completed.stdout[-8000:]}"
                )
            result = _load_json(result_path)
            _validate_result(result)
            shard_path = artifact_dir / "shards" / f"cap-{cap}.json"
            _write_json(shard_path, result)
            relative_shard = str(shard_path.relative_to(artifact_dir))
            files[relative_shard] = _file_sha256(shard_path)
            results.append(result)
            print(
                _canonical_json(
                    {
                        "order_index": order_index,
                        "cap": cap,
                        "accepted_nodes": result["accepted_nodes"],
                        "worst_active_lower": result["worst_active_lower"],
                        "whole_seconds": float(result["worker_elapsed_ns"]) / 1e9,
                    }
                )
            )
    payload = _calibration_payload(workload, results)
    decision = _compile_decision(payload)
    pilot = {
        "schema_version": PILOT_SCHEMA_VERSION,
        "status": "ok",
        "source": {"native_code_revision": _code_revision()},
        "protocol": {
            "fresh_process_per_cap": True,
            "torch_threads": args.torch_threads,
            "whole_query_timeout_seconds": 60,
            "search_budget": {"max_nodes": 31, "max_depth": 4},
            "pilot_order": list(policy.pilot_order),
            "formal_repeats": 3,
        },
        "calibration_payload": payload,
        "calibration_evidence_hash": _canonical_hash(payload),
        "decision": decision.to_dict(),
        "decision_hash": decision.stable_hash(policy),
        "selected_cap": decision.selected_cap,
        "claim_boundary": "calibration_only_no_formal_performance_or_property_claim",
        "performance_claimed": False,
    }
    validate_pilot(pilot)
    pilot_path = artifact_dir / "pilot.json"
    _write_json(pilot_path, pilot)
    files["pilot.json"] = _file_sha256(pilot_path)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "files": files,
        "pilot_hash": _canonical_hash(pilot),
    }
    _write_json(artifact_dir / "manifest.json", manifest)
    print(
        _canonical_json(
            {
                "status": "ok",
                "selected_cap": decision.selected_cap,
                "selected_gain_retention": decision.selected_gain_retention,
                "pilot_hash": manifest["pilot_hash"],
            }
        )
    )


def _replay(args: argparse.Namespace) -> None:
    artifact_dir = args.artifact_dir.resolve()
    manifest = _load_json(artifact_dir / "manifest.json")
    pilot = _load_json(artifact_dir / "pilot.json")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise TypeError("NRIR-33 manifest files differ")
    for relative, expected in files.items():
        path = artifact_dir / relative
        if not path.is_file() or _file_sha256(path) != expected:
            raise ValueError(f"NRIR-33 pilot digest differs: {relative}")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION or manifest.get(
        "pilot_hash"
    ) != _canonical_hash(pilot):
        raise ValueError("NRIR-33 manifest identity differs")
    validate_pilot(pilot)
    workloads = _resolve_workloads(args.benchmark_root.resolve())
    workload = next(
        item for item in workloads if item["workload_id"] == "cifar10_resnet:000"
    )
    if pilot["calibration_payload"]["workload"] != _public_workload(workload):
        raise ValueError("NRIR-33 pilot workload identity differs")
    for row in pilot["calibration_payload"]["candidate_results"]:
        shard = _load_json(artifact_dir / "shards" / f"cap-{row['cap']}.json")
        _validate_result(shard)
        if shard["result_hash"] != row["result_hash"]:
            raise ValueError("NRIR-33 pilot shard/result binding differs")
    print(
        _canonical_json(
            {
                "status": "ok",
                "selected_cap": pilot["selected_cap"],
                "pilot_hash": manifest["pilot_hash"],
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
