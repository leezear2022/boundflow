#!/usr/bin/env python3
"""Generate or replay the NRIR-27 production prepared verifier artifact."""

# pylint: disable=too-many-lines,too-many-locals,too-many-statements
# pylint: disable=too-many-branches,too-many-arguments,too-many-boolean-expressions
# pylint: disable=missing-function-docstring,import-outside-toplevel,duplicate-code
# pylint: disable=protected-access,import-error,line-too-long

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping

from scripts.run_multiworkload_competitor_e2e_artifact import (
    ABCROWN_COMMIT,
    VNNCOMP_COMMIT,
    WORKLOAD_ROWS,
    _csv_selection,
    _git_revision,
    _onnx_inventory,
)

ARTIFACT_SCHEMA_VERSION = "boundflow.production-prepared-verifier-artifact/v1"
EVIDENCE_SCHEMA_VERSION = "boundflow.production-prepared-verifier-evidence/v1"
WORKER_SCHEMA_VERSION = "boundflow.production-prepared-verifier-worker/v1"
PLAN_ID = "vnncomp21-three-topology-production-cpu-v1"
REPEATS = 3
TORCH_THREADS = 8
TIMEOUT_SECONDS = 60
ALPHA_STEPS = 5
SEARCH_STEPS = 4
MAX_NODES = 7
ARTIFACT_DIR = Path(
    "artifacts/production-prepared-verifier/vnncomp21-three-topology-cpu-v1"
)
COMPETITOR_REFERENCE_DIR = Path(
    "artifacts/multiworkload-competitor-e2e/vnncomp21-three-topology-cpu-v1"
)
MANIFEST_FILE = "manifest.json"
EVIDENCE_FILE = "evidence.json"
CLAUSE_MODES = ("clause_audit", "clause_production")
FULL_MODE = "full_production"
MODE_ORDERS = (
    ("clause_audit", "clause_production", "full_production"),
    ("full_production", "clause_production", "clause_audit"),
    ("clause_production", "clause_audit", "full_production"),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("generate", "replay"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--benchmark-root", type=Path, required=True)
        subparser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
        subparser.add_argument("--torch-threads", type=int, default=TORCH_THREADS)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--mode", choices=(*CLAUSE_MODES, FULL_MODE), required=True)
    worker.add_argument("--workload-id", required=True)
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


def canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical_json(value, indent=2) + "\n", encoding="utf-8")


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


def _integer(value: object, label: str) -> int:
    if not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    return value


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _code_revision() -> str:
    root = _repo_root()
    paths = (
        "boundflow/ir/workload.py",
        "boundflow/ir/production_verifier.py",
        "boundflow/runtime/native_alpha_beta_optimizer_schedule.py",
        "boundflow/runtime/native_production_verifier.py",
        "boundflow/runtime/native_production_complete_query.py",
        "boundflow/runtime/native_property_verdict.py",
        "boundflow/runtime/complete_verifier_query.py",
        "scripts/run_production_prepared_verifier_artifact.py",
    )
    return canonical_hash({path: file_sha256(root / path) for path in paths})


def _build_suite_ir(
    benchmark_root: Path, torch_threads: int
) -> tuple[Any, Any, Any, list[dict[str, object]]]:
    from boundflow.frontends.vnnlib import import_vnnlib_box_query
    from boundflow.ir.workload import (
        MultiWorkloadPlanIR,
        VerificationWorkloadSourceIR,
        VerifierBackendKind,
        VerifierExecutionPolicyIR,
        compile_multiworkload_schedule_ir,
        compile_multiworkload_task_ir,
    )

    if _git_revision(benchmark_root) != VNNCOMP_COMMIT:
        raise ValueError("NRIR-27 VNN-COMP commit differs")
    sources = []
    resolved: list[dict[str, object]] = []
    for definition in WORKLOAD_ROWS:
        csv_path, model_path, property_path = _csv_selection(benchmark_root, definition)
        workload_id = str(definition["workload_id"])
        query = import_vnnlib_box_query(property_path, query_id=workload_id)
        input_shape, output_dim, ops = _onnx_inventory(model_path)
        if (
            len(query.input_names) != math.prod(input_shape[1:])
            or len(query.output_names) != output_dim
        ):
            raise ValueError("NRIR-27 ONNX/VNNLIB dimensions differ")
        ordinal = definition["csv_ordinal"]
        if not isinstance(ordinal, int):
            raise TypeError("NRIR-27 CSV ordinal must be an integer")
        source = VerificationWorkloadSourceIR(
            workload_id=workload_id,
            category=str(definition["category"]),
            csv_ordinal=ordinal,
            csv_relative_path=str(definition["csv"]),
            model_relative_path=str(definition["model"]),
            property_relative_path=str(definition["property"]),
            csv_sha256=file_sha256(csv_path),
            model_sha256=file_sha256(model_path),
            property_sha256=file_sha256(property_path),
            query_ir_hash=query.stable_hash(),
            model_input_shape=input_shape,
            model_output_dim=output_dim,
            onnx_ops=ops,
        )
        source.validate()
        sources.append(source)
        resolved.append(
            {
                "workload_id": workload_id,
                "model": model_path,
                "property": property_path,
            }
        )
    revision = _code_revision()
    audit = VerifierExecutionPolicyIR(
        backend=VerifierBackendKind.BOUNDFLOW_NATIVE,
        implementation_id="boundflow-audit-complete-query-v1",
        implementation_revision=revision,
        device="cpu",
        torch_threads=torch_threads,
        timeout_seconds=TIMEOUT_SECONDS,
        alpha_steps=ALPHA_STEPS,
        beta_steps=ALPHA_STEPS,
        search_steps=SEARCH_STEPS,
        max_nodes=MAX_NODES,
        attack_policy="native_projected_gradient",
        complete_verifier="bounded_relu_bab",
    )
    production = VerifierExecutionPolicyIR(
        backend=VerifierBackendKind.BOUNDFLOW_PRODUCTION,
        implementation_id="boundflow-production-complete-query-v1",
        implementation_revision=revision,
        device="cpu",
        torch_threads=torch_threads,
        timeout_seconds=TIMEOUT_SECONDS,
        alpha_steps=ALPHA_STEPS,
        beta_steps=ALPHA_STEPS,
        search_steps=SEARCH_STEPS,
        max_nodes=MAX_NODES,
        attack_policy="native_projected_gradient",
        complete_verifier="production_prepared_bounded_relu_bab",
    )
    plan = MultiWorkloadPlanIR(
        plan_id=PLAN_ID,
        benchmark_commit=VNNCOMP_COMMIT,
        workloads=tuple(sources),
        policies=(audit, production),
        claim_boundary="cpu_audit_production_repeated_internal_speedup",
    )
    task_ir = compile_multiworkload_task_ir(plan)
    schedule = compile_multiworkload_schedule_ir(plan, task_ir)
    return plan, task_ir, schedule, resolved


def _batch_ir_record(batch: Any) -> dict[str, object]:
    return {
        "plan": batch.plan.to_dict(),
        "plan_hash": batch.plan.stable_hash(),
        "task": batch.task_ir.to_dict(),
        "task_hash": batch.task_ir.stable_hash(),
        "schedule": batch.schedule.to_dict(),
        "schedule_hash": batch.schedule.stable_hash(
            plan=batch.plan, task_ir=batch.task_ir
        ),
        "action_elapsed_ns": [action.elapsed_ns for action in batch.actions],
        "audit_hash_chain_constructed": batch.audit_hash_chain_constructed,
        "selected_native_reexecution": batch.selected_native_reexecution,
    }


def _query_result(mode: str, execution: Any) -> dict[str, object]:
    clauses = []
    semantic_clauses = []
    for clause in execution.clauses:
        queue = clause.queue.trace
        root = queue.evaluations[0]
        selected_state_hashes = [item.selected_state_hash for item in queue.evaluations]
        signature_hash = canonical_hash(queue.logical_queue_signature())
        row: dict[str, object] = {
            "clause_index": clause.trace.clause_index,
            "status": clause.trace.status,
            "root_lower": root.lower,
            "root_upper": root.upper,
            "queue_status": queue.status,
            "evaluated_nodes": len(queue.evaluations),
            "logical_queue_signature_hash": signature_hash,
            "selected_state_hashes": selected_state_hashes,
            "queue_trace_hash": queue.stable_hash(),
        }
        if mode in {"clause_production", "full_production"}:
            row["production_batch_ir"] = [
                _batch_ir_record(batch) for batch in queue.batches
            ]
            row["audit_hash_chain_constructed"] = queue.audit_hash_chain_constructed
            row["selected_native_reexecution"] = queue.selected_native_reexecution
        clauses.append(row)
        semantic_clauses.append(
            {
                "clause_index": clause.trace.clause_index,
                "status": clause.trace.status,
                "queue_status": queue.status,
                "evaluated_nodes": len(queue.evaluations),
                "logical_queue_signature_hash": signature_hash,
                "selected_state_hashes": selected_state_hashes,
            }
        )
    semantic_signature = {
        "solver_status": execution.trace.status,
        "completed_clause_count": len(execution.clauses),
        "unresolved_clause_indices": list(execution.trace.unresolved_clause_indices),
        "pending_clause_indices": list(execution.trace.pending_clause_indices),
        "clauses": semantic_clauses,
    }
    return {
        **semantic_signature,
        "query_trace_hash": execution.trace.stable_hash(),
        "clauses": clauses,
        "semantic_signature_hash": canonical_hash(semantic_signature),
    }


def _worker(args: argparse.Namespace) -> None:
    import torch

    from boundflow.frontends.onnx.frontend import import_onnx
    from boundflow.frontends.vnnlib import (
        import_vnnlib_box_query,
        materialize_vnnlib_box_query,
    )
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
    from boundflow.runtime.native_production_complete_query import (
        execute_native_production_complete_verifier_query,
    )
    from boundflow.runtime.native_relu_split_bab_runtime import (
        NativeReluSplitBabConfig,
    )
    from boundflow.runtime.task_executor import InputSpec

    torch.set_num_threads(args.torch_threads)
    started_ns = time.perf_counter_ns()
    query = import_vnnlib_box_query(args.property, query_id=args.workload_id)
    input_shape, output_dim, _ops = _onnx_inventory(args.model)
    if len(query.output_names) != output_dim:
        raise ValueError("NRIR-27 worker output dimension differs")
    tensors = materialize_vnnlib_box_query(query, input_shape=input_shape[1:])
    program = import_onnx(str(args.model), do_shape_infer=True, normalize=True)
    module = plan_interval_ibp_v0(program)
    input_spec = InputSpec.box(
        value_name=program.graph.inputs[0],
        lower=tensors.input_lower,
        upper=tensors.input_upper,
    )
    setup_ns = time.perf_counter_ns() - started_ns
    clause_only = args.mode in CLAUSE_MODES
    objectives = (
        tensors.linear_spec_c[:, :1, :].contiguous()
        if clause_only
        else tensors.linear_spec_c
    )
    thresholds = (
        tensors.thresholds[:1].contiguous() if clause_only else tensors.thresholds
    )
    query_policy = CompleteVerifierQueryPolicy(
        timeout_ns=(None if clause_only else TIMEOUT_SECONDS * 1_000_000_000)
    )
    query_id = f"{query.query_id}:{'clause0' if clause_only else 'full'}"
    search_policy = NativeProjectedGradientSearchPolicy(
        steps=SEARCH_STEPS, step_size=0.002
    )
    queue_config = NativeReluSplitBabConfig(
        max_nodes=MAX_NODES,
        max_depth=2,
        expansion_batch_size=2,
        max_eval_batch_size=4,
    )
    optimizer_policy = NativeAlphaBetaOptimizerPolicy(
        steps=ALPHA_STEPS,
        lr=0.1,
        alpha_initialization_mode="adaptive",
    )
    execute_started_ns = time.perf_counter_ns()
    execution: Any
    if args.mode == "clause_audit":
        execution = execute_complete_verifier_query(
            module,
            input_spec,
            linear_spec_C=objectives,
            thresholds=thresholds,
            query_id=query_id,
            query_policy=query_policy,
            search_policy=search_policy,
            queue_config=queue_config,
            optimizer_policy=optimizer_policy,
        )
    else:
        execution = execute_native_production_complete_verifier_query(
            module,
            input_spec,
            linear_spec_C=objectives,
            thresholds=thresholds,
            query_id=query_id,
            query_policy=query_policy,
            search_policy=search_policy,
            queue_config=queue_config,
            optimizer_policy=optimizer_policy,
        )
    execution_ns = time.perf_counter_ns() - execute_started_ns
    result = {
        "schema_version": WORKER_SCHEMA_VERSION,
        "workload_id": args.workload_id,
        "mode": args.mode,
        "setup_ns": setup_ns,
        "execution_ns": execution_ns,
        "worker_elapsed_ns": time.perf_counter_ns() - started_ns,
        "execution_mode": execution.trace.execution_mode,
        "performance_claimed": False,
        **_query_result(args.mode, execution),
    }
    _write_json(args.result_json, result)
    print(
        _canonical_json(
            {
                "status": "ok",
                "mode": args.mode,
                "solver_status": execution.trace.status,
            }
        )
    )


def _worker_command(
    *,
    mode: str,
    workload: Mapping[str, object],
    result_path: Path,
    torch_threads: int,
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
        str(workload["model"]),
        "--property",
        str(workload["property"]),
        "--result-json",
        str(result_path),
        "--torch-threads",
        str(torch_threads),
    ]


def _run_worker(
    *,
    mode: str,
    workload: Mapping[str, object],
    result_path: Path,
    torch_threads: int,
) -> tuple[dict[str, Any], str, int]:
    started_ns = time.perf_counter_ns()
    completed = subprocess.run(
        _worker_command(
            mode=mode,
            workload=workload,
            result_path=result_path,
            torch_threads=torch_threads,
        ),
        cwd=_repo_root(),
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=TIMEOUT_SECONDS + 90,
        check=False,
    )
    elapsed_ns = time.perf_counter_ns() - started_ns
    if completed.returncode != 0 or not result_path.is_file():
        raise RuntimeError(
            f"NRIR-27 {mode} worker failed with {completed.returncode}: "
            f"{completed.stdout[-4000:]}"
        )
    return _load_json(result_path), completed.stdout, elapsed_ns


def _p90(values: list[int]) -> int:
    if len(values) < 3:
        raise ValueError("NRIR-27 performance summary requires at least 3 samples")
    return sorted(values)[math.ceil(0.9 * len(values)) - 1]


def _competitor_reference() -> tuple[dict[str, object], str]:
    root = _repo_root() / COMPETITOR_REFERENCE_DIR
    manifest = _load_json(root / "manifest.json")
    evidence = _load_json(root / "evidence.json")
    rows = {}
    for record in _list(evidence.get("records"), "competitor records"):
        item = _mapping(record, "competitor record")
        if item.get("backend") == "external_abcrown":
            rows[str(item["workload_id"])] = {
                "e2e_elapsed_ns": int(item["e2e_elapsed_ns"]),
                "solver_status": str(
                    _mapping(item["result"], "competitor result")["solver_status"]
                ),
            }
    if len(rows) != 3 or evidence.get("performance_claimed") is not False:
        raise ValueError("NRIR-27 competitor reference differs")
    return (
        {
            "source_artifact": str(COMPETITOR_REFERENCE_DIR),
            "source_manifest_sha256": file_sha256(root / "manifest.json"),
            "source_evidence_hash": manifest["evidence_hash"],
            "abcrown_commit": ABCROWN_COMMIT,
            "single_observation_only": True,
            "performance_claimed": False,
            "records": rows,
        },
        file_sha256(root / "manifest.json"),
    )


def _summaries(
    records: list[dict[str, object]], competitor: Mapping[str, Any]
) -> dict[str, object]:
    summaries: dict[str, object] = {}
    competitor_records = _mapping(competitor["records"], "competitor records")
    for definition in WORKLOAD_ROWS:
        workload_id = str(definition["workload_id"])
        rows = [row for row in records if row["workload_id"] == workload_id]
        clause_by_mode = {
            mode: [
                _integer(row["e2e_elapsed_ns"], "record E2E elapsed")
                for row in rows
                if row["mode"] == mode
            ]
            for mode in CLAUSE_MODES
        }
        full = [
            _integer(row["e2e_elapsed_ns"], "record E2E elapsed")
            for row in rows
            if row["mode"] == FULL_MODE
        ]
        audit_median = int(statistics.median(clause_by_mode["clause_audit"]))
        production_median = int(statistics.median(clause_by_mode["clause_production"]))
        full_median = int(statistics.median(full))
        competitor_row = _mapping(competitor_records[workload_id], "competitor row")
        competitor_ns = int(competitor_row["e2e_elapsed_ns"])
        summaries[workload_id] = {
            "clause0": {
                "audit_raw_e2e_ns": clause_by_mode["clause_audit"],
                "production_raw_e2e_ns": clause_by_mode["clause_production"],
                "audit_median_e2e_ns": audit_median,
                "production_median_e2e_ns": production_median,
                "audit_p90_e2e_ns": _p90(clause_by_mode["clause_audit"]),
                "production_p90_e2e_ns": _p90(clause_by_mode["clause_production"]),
                "internal_speedup": audit_median / production_median,
                "semantic_parity": True,
                "performance_claimed": True,
            },
            "full_production": {
                "raw_e2e_ns": full,
                "median_e2e_ns": full_median,
                "p90_e2e_ns": _p90(full),
                "solver_statuses": [
                    _mapping(row["result"], "full result")["solver_status"]
                    for row in rows
                    if row["mode"] == FULL_MODE
                ],
                "competitor_single_reference_e2e_ns": competitor_ns,
                "production_to_competitor_single_reference_ratio": (
                    full_median / competitor_ns
                ),
                "competitor_ratio_diagnostic_only": True,
                "performance_claimed": False,
            },
        }
    return summaries


def _validate_worker_result(result: Mapping[str, Any]) -> None:
    mode = result.get("mode")
    clauses = _list(result.get("clauses"), "NRIR-27 clauses")
    if (
        result.get("schema_version") != WORKER_SCHEMA_VERSION
        or mode not in {*CLAUSE_MODES, FULL_MODE}
        or not result.get("workload_id")
        or result.get("execution_mode")
        != ("audit_validation" if mode == "clause_audit" else "production_prepared")
        or result.get("performance_claimed") is not False
        or any(
            not isinstance(result.get(name), int) or int(result[name]) <= 0
            for name in ("setup_ns", "execution_ns", "worker_elapsed_ns")
        )
        or not _sha256(result.get("query_trace_hash"))
        or not _sha256(result.get("semantic_signature_hash"))
        or not clauses
        or (mode in CLAUSE_MODES and len(clauses) != 1)
    ):
        raise ValueError("NRIR-27 worker result differs")
    for clause in clauses:
        item = _mapping(clause, "NRIR-27 clause")
        if (
            not _sha256(item.get("logical_queue_signature_hash"))
            or not _sha256(item.get("queue_trace_hash"))
            or not _list(item.get("selected_state_hashes"), "selected states")
        ):
            raise ValueError("NRIR-27 clause result differs")
        if mode != "clause_audit":
            if (
                item.get("audit_hash_chain_constructed") is not False
                or item.get("selected_native_reexecution") is not False
                or not _list(item.get("production_batch_ir"), "production batch IR")
            ):
                raise ValueError("NRIR-27 production disclosure differs")
            for batch in item["production_batch_ir"]:
                batch_ir = _mapping(batch, "production batch")
                if (
                    canonical_hash(batch_ir["plan"]) != batch_ir.get("plan_hash")
                    or canonical_hash(batch_ir["task"]) != batch_ir.get("task_hash")
                    or canonical_hash(batch_ir["schedule"])
                    != batch_ir.get("schedule_hash")
                    or batch_ir.get("audit_hash_chain_constructed") is not False
                    or batch_ir.get("selected_native_reexecution") is not False
                    or len(_list(batch_ir.get("action_elapsed_ns"), "action timing"))
                    != 4
                ):
                    raise ValueError("NRIR-27 production batch IR differs")


def validate_evidence_structure(evidence: Mapping[str, Any]) -> None:
    if (
        evidence.get("schema_version") != EVIDENCE_SCHEMA_VERSION
        or evidence.get("status") != "ok"
        or evidence.get("property_status") != "validated_reduced"
        or evidence.get("performance_claimed") is not True
    ):
        raise ValueError("NRIR-27 evidence header differs")
    claim = str(evidence.get("claim_boundary", ""))
    for phrase in (
        "same-algorithm clause-0",
        "no competitor speedup",
        "CPU",
        "no GPU",
    ):
        if phrase not in claim:
            raise ValueError("NRIR-27 claim boundary differs")
    source = _mapping(evidence.get("source"), "NRIR-27 source")
    if (
        source.get("vnncomp_commit") != VNNCOMP_COMMIT
        or source.get("abcrown_reference_commit") != ABCROWN_COMMIT
        or not _sha256(source.get("native_code_revision"))
    ):
        raise ValueError("NRIR-27 source identity differs")
    ir = _mapping(evidence.get("ir"), "NRIR-27 IR")
    if (
        canonical_hash(ir.get("plan")) != ir.get("plan_hash")
        or canonical_hash(ir.get("task")) != ir.get("task_hash")
        or canonical_hash(ir.get("schedule")) != ir.get("schedule_hash")
        or len(_list(_mapping(ir["task"], "task")["tasks"], "tasks")) != 21
        or len(
            _list(
                _mapping(ir["schedule"], "schedule")["fresh_process_task_ids"],
                "fresh tasks",
            )
        )
        != 6
    ):
        raise ValueError("NRIR-27 suite IR differs")
    records = _list(evidence.get("records"), "NRIR-27 records")
    identities: set[tuple[str, int, str]] = set()
    for row in records:
        record = _mapping(row, "NRIR-27 record")
        identity = (
            str(record.get("workload_id")),
            int(record.get("group_index", -1)),
            str(record.get("mode")),
        )
        result = _mapping(record.get("result"), "NRIR-27 result")
        _validate_worker_result(result)
        if (
            identity in identities
            or result.get("workload_id") != identity[0]
            or result.get("mode") != identity[2]
            or identity[1] not in range(REPEATS)
            or identity[2] not in {*CLAUSE_MODES, FULL_MODE}
            or not isinstance(record.get("e2e_elapsed_ns"), int)
            or int(record["e2e_elapsed_ns"]) <= 0
            or not _sha256(record.get("log_sha256"))
        ):
            raise ValueError("NRIR-27 record identity differs")
        identities.add(identity)
    expected = {
        (str(definition["workload_id"]), group, mode)
        for definition in WORKLOAD_ROWS
        for group in range(REPEATS)
        for mode in (*CLAUSE_MODES, FULL_MODE)
    }
    if identities != expected:
        raise ValueError("NRIR-27 execution coverage differs")
    for definition in WORKLOAD_ROWS:
        workload_id = str(definition["workload_id"])
        for group in range(REPEATS):
            pair = {
                str(row["mode"]): _mapping(row["result"], "pair result")
                for row in records
                if row["workload_id"] == workload_id
                and row["group_index"] == group
                and row["mode"] in CLAUSE_MODES
            }
            if (
                set(pair) != set(CLAUSE_MODES)
                or pair["clause_audit"]["semantic_signature_hash"]
                != pair["clause_production"]["semantic_signature_hash"]
            ):
                raise ValueError("NRIR-27 audit/production semantic parity differs")
            audit_clause = _mapping(
                _list(pair["clause_audit"]["clauses"], "audit clauses")[0],
                "audit clause",
            )
            production_clause = _mapping(
                _list(pair["clause_production"]["clauses"], "production clauses")[0],
                "production clause",
            )
            if any(
                abs(float(audit_clause[name]) - float(production_clause[name])) > 2e-3
                for name in ("root_lower", "root_upper")
            ):
                raise ValueError("NRIR-27 audit/production numeric parity differs")
    competitor = _mapping(evidence.get("competitor_reference"), "competitor")
    summaries = _mapping(evidence.get("summaries"), "NRIR-27 summaries")
    if dict(summaries) != _summaries([dict(row) for row in records], competitor):
        raise ValueError("NRIR-27 summary replay differs")
    if any(
        float(
            _mapping(_mapping(summary, "summary")["clause0"], "clause")[
                "internal_speedup"
            ]
        )
        <= 1.0
        for summary in summaries.values()
    ):
        raise ValueError("NRIR-27 internal performance gate failed")


def _build_evidence(
    args: argparse.Namespace,
) -> tuple[dict[str, object], dict[str, str]]:
    plan, task_ir, schedule, workloads = _build_suite_ir(
        args.benchmark_root, args.torch_threads
    )
    competitor, _competitor_manifest_sha = _competitor_reference()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = args.artifact_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    files: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="boundflow-nrir27-") as temporary:
        temporary_root = Path(temporary)
        for workload in workloads:
            workload_id = str(workload["workload_id"])
            for group_index, order in enumerate(MODE_ORDERS):
                for order_index, mode in enumerate(order):
                    result_path = (
                        temporary_root
                        / f"{workload_id.replace(':', '-')}-{group_index}-{mode}.json"
                    )
                    result, log, elapsed_ns = _run_worker(
                        mode=mode,
                        workload=workload,
                        result_path=result_path,
                        torch_threads=args.torch_threads,
                    )
                    _validate_worker_result(result)
                    log_name = f"logs/{workload_id.replace(':', '-')}-g{group_index}-{mode}.log"
                    log_path = args.artifact_dir / log_name
                    log_path.write_text(log, encoding="utf-8")
                    files[log_name] = file_sha256(log_path)
                    records.append(
                        {
                            "workload_id": workload_id,
                            "group_index": group_index,
                            "order_index": order_index,
                            "mode": mode,
                            "e2e_elapsed_ns": elapsed_ns,
                            "log_path": log_name,
                            "log_sha256": files[log_name],
                            "result": result,
                        }
                    )
                    print(
                        _canonical_json(
                            {
                                "workload_id": workload_id,
                                "group_index": group_index,
                                "mode": mode,
                                "e2e_elapsed_ns": elapsed_ns,
                                "solver_status": result["solver_status"],
                            }
                        ),
                        flush=True,
                    )
    summaries = _summaries(records, competitor)
    evidence = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "ok",
        "property_status": "validated_reduced",
        "performance_claimed": True,
        "claim_boundary": "three real VNN-COMP CPU workloads; repeated same-algorithm clause-0 audit-to-production internal speedup only; no competitor speedup, no GPU, no ASPLOS-ready claim",
        "source": {
            "vnncomp_commit": VNNCOMP_COMMIT,
            "abcrown_reference_commit": ABCROWN_COMMIT,
            "native_code_revision": _code_revision(),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device": "cpu",
            "torch_threads": args.torch_threads,
            "cuda_executed": False,
        },
        "protocol": {
            "repeats": REPEATS,
            "orders": [list(order) for order in MODE_ORDERS],
            "clause_comparison": "clause_0_same_search_optimizer_queue_budget",
            "full_query": "production_only_cooperative_60s_deadline",
            "timing_boundary": "fresh_process_start_to_structured_result",
        },
        "ir": {
            "plan": plan.to_dict(),
            "plan_hash": plan.stable_hash(),
            "task": task_ir.to_dict(),
            "task_hash": task_ir.stable_hash(),
            "schedule": schedule.to_dict(),
            "schedule_hash": schedule.stable_hash(task_ir),
        },
        "records": records,
        "summaries": summaries,
        "competitor_reference": competitor,
        "limitations": [
            "The performance claim is internal audit-validation overhead removal on identical clause-0 algorithms only.",
            "Full-query production timings are repeated, but the alpha-beta-CROWN reference remains one historical fresh-process observation with different complete search.",
            "The generating host has no visible CUDA driver; no GPU, allocator, OOM, or Pareto claim is made.",
            "BoundFlow remains bounded to 7 nodes/depth 2 and may return unknown.",
            "Production dynamic batches still compile once per new split scope; cross-property/model compilation reuse is not claimed.",
        ],
    }
    validate_evidence_structure(evidence)
    return evidence, files


def _generate(args: argparse.Namespace) -> None:
    evidence, files = _build_evidence(args)
    evidence_path = args.artifact_dir / EVIDENCE_FILE
    _write_json(evidence_path, evidence)
    files[EVIDENCE_FILE] = file_sha256(evidence_path)
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "ok",
        "performance_claimed": True,
        "files": dict(sorted(files.items())),
        "evidence_hash": canonical_hash(evidence),
    }
    _write_json(args.artifact_dir / MANIFEST_FILE, manifest)
    print(_canonical_json({"status": "ok", "evidence_hash": manifest["evidence_hash"]}))


def _replay(args: argparse.Namespace) -> None:
    manifest = _load_json(args.artifact_dir / MANIFEST_FILE)
    evidence = _load_json(args.artifact_dir / EVIDENCE_FILE)
    files = _mapping(manifest.get("files"), "NRIR-27 manifest files")
    actual_files = {
        str(path.relative_to(args.artifact_dir)): file_sha256(path)
        for path in sorted(args.artifact_dir.rglob("*"))
        if path.is_file() and path.name != MANIFEST_FILE
    }
    if (
        manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION
        or manifest.get("status") != "ok"
        or manifest.get("performance_claimed") is not True
        or dict(files) != actual_files
        or manifest.get("evidence_hash") != canonical_hash(evidence)
    ):
        raise ValueError("NRIR-27 artifact manifest differs")
    validate_evidence_structure(evidence)
    plan, task_ir, schedule, _workloads = _build_suite_ir(
        args.benchmark_root,
        int(_mapping(evidence["environment"], "environment")["torch_threads"]),
    )
    expected_ir = {
        "plan": plan.to_dict(),
        "plan_hash": plan.stable_hash(),
        "task": task_ir.to_dict(),
        "task_hash": task_ir.stable_hash(),
        "schedule": schedule.to_dict(),
        "schedule_hash": schedule.stable_hash(task_ir),
    }
    if dict(_mapping(evidence["ir"], "stored IR")) != expected_ir:
        raise ValueError("NRIR-27 source-to-IR replay differs")
    competitor, _manifest_sha = _competitor_reference()
    if dict(_mapping(evidence["competitor_reference"], "competitor")) != competitor:
        raise ValueError("NRIR-27 competitor reference replay differs")
    for record in _list(evidence["records"], "records"):
        item = _mapping(record, "record")
        if file_sha256(args.artifact_dir / str(item["log_path"])) != item["log_sha256"]:
            raise ValueError("NRIR-27 log digest differs")
    print(_canonical_json({"status": "ok", "evidence_hash": manifest["evidence_hash"]}))


def main() -> None:
    args = _parse_args()
    if args.torch_threads < 1:
        raise ValueError("NRIR-27 torch thread count must be positive")
    if args.command == "worker":
        _worker(args)
    elif args.command == "generate":
        _generate(args)
    else:
        _replay(args)


if __name__ == "__main__":
    main()
