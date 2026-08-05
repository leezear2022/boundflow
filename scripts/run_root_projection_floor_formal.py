#!/usr/bin/env python3
"""Generate or replay NRIR44 Phase-A root-projection-floor evidence."""

# pylint: disable=too-many-lines,too-many-locals,too-many-statements
# pylint: disable=too-many-branches,too-many-boolean-expressions
# pylint: disable=missing-function-docstring,import-outside-toplevel,duplicate-code
# pylint: disable=protected-access,too-many-arguments

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from boundflow.ir.root_projection_floor import (
    NativeRootProjectionClauseOwnerIR,
    NativeRootProjectionClauseTraceIR,
    NativeRootProjectionFloorInstanceIR,
    NativeRootProjectionFloorPlanIR,
    NativeRootProjectionFloorScheduleAction,
    NativeRootProjectionFloorScheduleIR,
    NativeRootProjectionFloorTaskIRModule,
    NativeRootProjectionFloorTaskIRUnit,
    NativeRootProjectionFloorTaskKind,
    NativeRootProjectionFloorTraceIR,
)
from scripts.run_typed_hard_clause_escalation_artifact import (
    _load_query_runtime,
    _policies,
    _public_workload,
    _resolve_workloads,
)

ARTIFACT_DIR = Path(
    "artifacts/root-projection-floor/"
    "vnncomp21-resnet2b-property0-three-repeat-cpu-phase-a-v1"
)
SOURCE_NRIR42 = Path(
    "artifacts/objective-branch-scorer-ownership/"
    "vnncomp21-resnet2b-property0-three-repeat-cpu-v1/formal.json"
)
EXPECTED_SOURCE_NRIR42_HASH = (
    "0d310c2ffc96844648a83f9921bc7f353ec8425986bccb36f75e6d1cd2b25b58"
)
BASE_REVISION = "d9d76da"
SCHEMA_VERSION = "boundflow.root-projection-floor-formal/v1"
WORKER_SCHEMA_VERSION = "boundflow.root-projection-floor-worker/v1"
MANIFEST_SCHEMA_VERSION = "boundflow.root-projection-floor-manifest/v1"
REPEAT_COUNT = 3
TORCH_THREADS = 8
WORKER_TIMEOUT_SECONDS = 180
PAIRED_ORDERS = (
    ("nrir42", "projected"),
    ("projected", "nrir42"),
    ("nrir42", "projected"),
)
EXPECTED_RANK = (2, 3, 4, 5, 0, 8, 6, 7, 1)
EXPECTED_SELECTED = (2, 3)
FULL_EVALUATIONS = 279
PROJECTED_EVALUATIONS = 9
MAX_PROJECTED_REPEAT_NS = 11_000_000_000
MAX_MEDIAN_RATIO = 0.50


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("generate", "replay"):
        child = subparsers.add_parser(command)
        child.add_argument("--benchmark-root", type=Path, required=True)
        child.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
        child.add_argument("--torch-threads", type=int, default=TORCH_THREADS)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--model", type=Path, required=True)
    worker.add_argument("--property", type=Path, required=True)
    worker.add_argument("--result-json", type=Path, required=True)
    worker.add_argument("--repeat-index", type=int, required=True)
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


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical_json(value, indent=2) + "\n", encoding="utf-8")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _code_revision() -> str:
    root = _repo_root()
    paths = (
        "boundflow/ir/root_projection_floor.py",
        "boundflow/runtime/native_root_projection_floor.py",
        "boundflow/runtime/native_objective_hard_clause_escalation.py",
        "boundflow/runtime/native_multi_clause_anytime.py",
        "scripts/run_root_projection_floor_formal.py",
    )
    return _canonical_hash({path: _file_sha256(root / path) for path in paths})


def _source_hash() -> str:
    value = _load_json(_repo_root() / SOURCE_NRIR42).get("formal_hash")
    if value != EXPECTED_SOURCE_NRIR42_HASH:
        raise ValueError("NRIR44 frozen NRIR42 source differs")
    return value


def _branch(value: Any) -> object:
    return None if value is None else value.to_dict()


def _queue_semantics(query: Any) -> list[dict[str, object]]:
    result = []
    for clause in query.clauses:
        queue = clause.queue.trace
        result.append(
            {
                "status": clause.trace.status,
                "evaluations": [
                    {
                        "depth": item.node.depth,
                        "branch_relu_input": item.node.branch_relu_input,
                        "branch_neuron_index": item.node.branch_neuron_index,
                        "branch_value": item.node.branch_value,
                        "lower": item.lower,
                        "upper": item.upper,
                        "priority": item.priority,
                        "branch_candidate": _branch(item.branch_candidate),
                    }
                    for item in queue.evaluations
                ],
                "decisions": [
                    {
                        "kind": item.kind,
                        "reason": item.reason,
                        "branch_candidate": _branch(item.branch_candidate),
                    }
                    for item in queue.decisions
                ],
            }
        )
    return result


def _root_semantics(floor: Any) -> list[dict[str, object]]:
    result = []
    for child in floor.clause_executions:
        clause = child.query.clauses[0]
        root = clause.queue.trace.evaluations[0]
        result.append(
            {
                "original_clause_index": child.original_clause_index,
                "status": clause.trace.status,
                "root_lower": root.lower,
                "root_upper": root.upper,
                "branch_candidate": _branch(root.branch_candidate),
            }
        )
    return result


def _semantic_summary(floor: Any, decision: Any) -> dict[str, object]:
    from boundflow.runtime.native_intermediate_refinement import (
        intermediate_bounds_hash,
    )

    shared = floor.shared_refinement
    if shared is None:
        raise ValueError("NRIR44 real floor lacks shared refinement")
    return {
        "baseline": _queue_semantics(floor.baseline),
        "shared_refinement_hash": intermediate_bounds_hash(shared.relu_pre),
        "objective_refinement_hashes": [
            intermediate_bounds_hash(child.refinement.relu_pre)
            for child in floor.clause_executions
        ],
        "roots": _root_semantics(floor),
        "completed_original_clause_indices": list(
            floor.trace.completed_objective_clause_indices
        ),
        "final_status": floor.trace.final_status,
        "ranked_original_clause_indices": list(decision.ranked_original_clause_indices),
        "selected_original_clause_indices": list(
            decision.selected_original_clause_indices
        ),
    }


def _projection_payload(execution: Any) -> dict[str, object]:
    program = execution.program
    trace = execution.projection_trace
    return {
        "plan": program.plan.to_dict(),
        "instance": program.instance.to_dict(plan=program.plan),
        "task_module": program.task_ir.to_dict(
            plan=program.plan, instance=program.instance
        ),
        "schedule": program.schedule.to_dict(
            plan=program.plan,
            instance=program.instance,
            task_module=program.task_ir,
        ),
        "trace": trace.to_dict(
            plan=program.plan,
            instance=program.instance,
            task_module=program.task_ir,
            schedule=program.schedule,
        ),
        "trace_hash": trace.stable_hash(
            plan=program.plan,
            instance=program.instance,
            task_module=program.task_ir,
            schedule=program.schedule,
        ),
    }


def _validate_projection_payload(value: Mapping[str, Any]) -> None:
    plan_value = value["plan"]
    plan = NativeRootProjectionFloorPlanIR(
        plan_id=plan_value["plan_id"],
        source_plan_hash=plan_value["source_plan_hash"],
        source_task_ir_hash=plan_value["source_task_ir_hash"],
        source_schedule_hash=plan_value["source_schedule_hash"],
        clause_count=plan_value["clause_count"],
        consumed_result_fields=tuple(plan_value["consumed_result_fields"]),
        full_max_nodes=plan_value["full_max_nodes"],
        full_max_depth=plan_value["full_max_depth"],
        projected_max_nodes=plan_value["projected_max_nodes"],
        projected_max_depth=plan_value["projected_max_depth"],
        soundness_mode=plan_value["soundness_mode"],
        semantics_owner=plan_value["semantics_owner"],
        performance_claimed=plan_value["performance_claimed"],
        schema_version=plan_value["schema_version"],
    )
    instance_value = value["instance"]
    instance = NativeRootProjectionFloorInstanceIR(
        instance_id=instance_value["instance_id"],
        plan_hash=instance_value["plan_hash"],
        objective_matrix_hash=instance_value["objective_matrix_hash"],
        thresholds_hash=instance_value["thresholds_hash"],
        clause_owners=tuple(
            NativeRootProjectionClauseOwnerIR(**item)
            for item in instance_value["clause_owners"]
        ),
        semantic_token=instance_value["semantic_token"],
        schema_version=instance_value["schema_version"],
    )
    task_value = value["task_module"]
    task_module = NativeRootProjectionFloorTaskIRModule(
        module_id=task_value["module_id"],
        plan_hash=task_value["plan_hash"],
        instance_hash=task_value["instance_hash"],
        tasks=tuple(
            NativeRootProjectionFloorTaskIRUnit(
                task_id=item["task_id"],
                kind=NativeRootProjectionFloorTaskKind(item["kind"]),
                dependency_task_ids=tuple(item["dependency_task_ids"]),
                input_value_ids=tuple(item["input_value_ids"]),
                output_value_ids=tuple(item["output_value_ids"]),
            )
            for item in task_value["tasks"]
        ),
        output_task_id=task_value["output_task_id"],
        schema_version=task_value["schema_version"],
    )
    schedule_value = value["schedule"]
    schedule = NativeRootProjectionFloorScheduleIR(
        schedule_id=schedule_value["schedule_id"],
        plan_hash=schedule_value["plan_hash"],
        instance_hash=schedule_value["instance_hash"],
        task_module_hash=schedule_value["task_module_hash"],
        actions=tuple(
            NativeRootProjectionFloorScheduleAction(**item)
            for item in schedule_value["actions"]
        ),
        full_evaluation_budget=schedule_value["full_evaluation_budget"],
        projected_evaluation_budget=schedule_value["projected_evaluation_budget"],
        schema_version=schedule_value["schema_version"],
    )
    trace_value = value["trace"]
    trace = NativeRootProjectionFloorTraceIR(
        plan_hash=trace_value["plan_hash"],
        instance_hash=trace_value["instance_hash"],
        task_module_hash=trace_value["task_module_hash"],
        schedule_hash=trace_value["schedule_hash"],
        source_floor_trace_hash=trace_value["source_floor_trace_hash"],
        baseline_trace_hash=trace_value["baseline_trace_hash"],
        shared_refinement_trace_hash=trace_value["shared_refinement_trace_hash"],
        clause_traces=tuple(
            NativeRootProjectionClauseTraceIR(**item)
            for item in trace_value["clause_traces"]
        ),
        completed_original_clause_indices=tuple(
            trace_value["completed_original_clause_indices"]
        ),
        final_status=trace_value["final_status"],
        performance_claimed=trace_value["performance_claimed"],
        schema_version=trace_value["schema_version"],
    )
    trace.validate(
        plan=plan, instance=instance, task_module=task_module, schedule=schedule
    )
    if value["trace_hash"] != trace.stable_hash(
        plan=plan, instance=instance, task_module=task_module, schedule=schedule
    ):
        raise ValueError("NRIR44 projection Trace hash differs")


def _run_worker(args: argparse.Namespace) -> None:
    import torch

    from boundflow.runtime.native_multi_clause_anytime import (
        _decision_from_floor,
        compile_native_multi_clause_anytime_program,
    )
    from boundflow.runtime.native_objective_hard_clause_escalation import (
        execute_native_objective_hard_clause_escalation_program,
    )
    from boundflow.runtime.native_root_projection_floor import (
        execute_native_root_projection_floor_program,
        lower_native_root_projection_floor_program,
    )

    if not 0 <= args.repeat_index < REPEAT_COUNT:
        raise ValueError("NRIR44 repeat index differs")
    torch.set_num_threads(args.torch_threads)
    _source_hash()
    _query, tensors, module, input_spec = _load_query_runtime(
        args.model.resolve(), args.property.resolve(), "cifar10_resnet:000"
    )
    search_policy, optimizer_policy = _policies()
    anytime = compile_native_multi_clause_anytime_program(
        module,
        input_spec,
        linear_spec_C=tensors.linear_spec_c,
        thresholds=tensors.thresholds,
        plan_id=f"nrir44:repeat:{args.repeat_index}",
        search_policy=search_policy,
        optimizer_policy=optimizer_policy,
    )
    projected_program = lower_native_root_projection_floor_program(
        anytime.floor_program,
        linear_spec_C=tensors.linear_spec_c,
        thresholds=tensors.thresholds,
        plan_id=f"nrir44:repeat:{args.repeat_index}:projection",
    )
    rows: list[dict[str, Any]] = []
    for position, mode in enumerate(PAIRED_ORDERS[args.repeat_index]):
        started_ns = time.monotonic_ns()
        if mode == "nrir42":
            floor = execute_native_objective_hard_clause_escalation_program(
                anytime.floor_program,
                module,
                input_spec,
                linear_spec_C=tensors.linear_spec_c,
                thresholds=tensors.thresholds,
                query_id=f"nrir44:r{args.repeat_index}:nrir42",
                search_policy=search_policy,
                optimizer_policy=optimizer_policy,
            )
            projection = None
        else:
            projected = execute_native_root_projection_floor_program(
                projected_program,
                module,
                input_spec,
                linear_spec_C=tensors.linear_spec_c,
                thresholds=tensors.thresholds,
                query_id=f"nrir44:r{args.repeat_index}:projected",
                search_policy=search_policy,
                optimizer_policy=optimizer_policy,
            )
            floor = projected.source_execution
            projection = _projection_payload(projected)
        elapsed_ns = time.monotonic_ns() - started_ns
        decision = _decision_from_floor(anytime.plan, floor, tensors.thresholds)
        evaluation_count = sum(
            len(child.query.clauses[0].queue.trace.evaluations)
            for child in floor.clause_executions
        )
        row: dict[str, Any] = {
            "repeat_index": args.repeat_index,
            "mode": mode,
            "order_position": position,
            "elapsed_ns": elapsed_ns,
            "objective_evaluation_count": evaluation_count,
            "semantics": _semantic_summary(floor, decision),
            "projection": projection,
            "performance_claimed": False,
        }
        row["row_hash"] = _canonical_hash(row)
        rows.append(row)
    if rows[0]["semantics"] != rows[1]["semantics"]:
        raise ValueError("NRIR44 projected/full floor semantics differ")
    worker: dict[str, Any] = {
        "schema_version": WORKER_SCHEMA_VERSION,
        "repeat_index": args.repeat_index,
        "paired_order": list(PAIRED_ORDERS[args.repeat_index]),
        "rows": rows,
        "performance_claimed": False,
    }
    worker["worker_hash"] = _canonical_hash(worker)
    validate_worker(worker, repeat_index=args.repeat_index)
    _write_json(args.result_json, worker)
    print(
        _canonical_json(
            {
                "repeat_index": args.repeat_index,
                "elapsed_ns": {row["mode"]: row["elapsed_ns"] for row in rows},
                "worker_hash": worker["worker_hash"],
            }
        ),
        flush=True,
    )


def validate_worker(value: Mapping[str, Any], *, repeat_index: int) -> None:
    rows = value.get("rows")
    if (
        value.get("schema_version") != WORKER_SCHEMA_VERSION
        or value.get("repeat_index") != repeat_index
        or value.get("paired_order") != list(PAIRED_ORDERS[repeat_index])
        or not isinstance(rows, list)
        or len(rows) != 2
        or value.get("performance_claimed") is not False
    ):
        raise ValueError("NRIR44 worker envelope differs")
    by_mode: dict[str, Mapping[str, Any]] = {}
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError("NRIR44 row must be an object")
        semantic = {key: item for key, item in row.items() if key != "row_hash"}
        mode = row.get("mode")
        expected_evaluations = (
            FULL_EVALUATIONS if mode == "nrir42" else PROJECTED_EVALUATIONS
        )
        summary = row.get("semantics")
        if (
            row.get("row_hash") != _canonical_hash(semantic)
            or row.get("repeat_index") != repeat_index
            or row.get("order_position") != position
            or mode not in {"nrir42", "projected"}
            or not isinstance(row.get("elapsed_ns"), int)
            or row["elapsed_ns"] <= 0
            or row.get("objective_evaluation_count") != expected_evaluations
            or not isinstance(summary, dict)
            or summary.get("completed_original_clause_indices") != list(range(9))
            or summary.get("final_status") != "unknown"
            or summary.get("ranked_original_clause_indices") != list(EXPECTED_RANK)
            or summary.get("selected_original_clause_indices")
            != list(EXPECTED_SELECTED)
            or [item["status"] for item in summary.get("roots", [])] != ["unknown"] * 9
            or row.get("performance_claimed") is not False
        ):
            raise ValueError("NRIR44 evidence row differs")
        projection = row.get("projection")
        if mode == "projected":
            if not isinstance(projection, dict):
                raise TypeError("NRIR44 projected row lacks typed projection")
            _validate_projection_payload(projection)
        elif projection is not None:
            raise ValueError("NRIR44 control row unexpectedly owns projection")
        by_mode[mode] = row
    if (
        set(by_mode) != {"nrir42", "projected"}
        or by_mode["nrir42"]["semantics"] != by_mode["projected"]["semantics"]
    ):
        raise ValueError("NRIR44 projected/full semantic parity differs")
    semantic_worker = {key: item for key, item in value.items() if key != "worker_hash"}
    if value.get("worker_hash") != _canonical_hash(semantic_worker):
        raise ValueError("NRIR44 worker hash differs")


def _median_mad(values: list[int]) -> tuple[int, int]:
    median = int(statistics.median(values))
    mad = int(statistics.median(abs(value - median) for value in values))
    return median, mad


def _build_formal(
    workers: Sequence[Mapping[str, Any]], *, workload: Mapping[str, Any]
) -> dict[str, Any]:
    rows = [row for worker in workers for row in worker["rows"]]
    controls = [row["elapsed_ns"] for row in rows if row["mode"] == "nrir42"]
    candidates = [row["elapsed_ns"] for row in rows if row["mode"] == "projected"]
    control_median, control_mad = _median_mad(controls)
    candidate_median, candidate_mad = _median_mad(candidates)
    ratio = candidate_median / control_median
    improvement = control_median - candidate_median
    parity_passed = all(
        worker["rows"][0]["semantics"] == worker["rows"][1]["semantics"]
        for worker in workers
    )
    budget_gate_passed = all(
        row["objective_evaluation_count"]
        == (FULL_EVALUATIONS if row["mode"] == "nrir42" else PROJECTED_EVALUATIONS)
        for row in rows
    )
    repeat_ceiling_passed = all(
        value <= MAX_PROJECTED_REPEAT_NS for value in candidates
    )
    improvement_exceeds_pooled_mad = improvement > max(control_mad, candidate_mad)
    timing_gate_passed = (
        repeat_ceiling_passed
        and ratio <= MAX_MEDIAN_RATIO
        and improvement_exceeds_pooled_mad
    )
    phase_a_go = parity_passed and budget_gate_passed and timing_gate_passed
    decision = {
        "parity_passed": parity_passed,
        "budget_gate_passed": budget_gate_passed,
        "repeat_ceiling_passed": repeat_ceiling_passed,
        "timing_gate_passed": timing_gate_passed,
        "phase_a_go": phase_a_go,
        "next_route": "run_nrir44_phase_b" if phase_a_go else "stop_root_projection",
        "reason": (
            "exact semantics and preregistered Phase-A cost gates passed"
            if phase_a_go
            else "NRIR44 Phase-A gate failed; Phase B is gated off"
        ),
    }
    metrics = {
        "nrir42_elapsed_ns": controls,
        "projected_elapsed_ns": candidates,
        "nrir42_median_ns": control_median,
        "projected_median_ns": candidate_median,
        "nrir42_mad_ns": control_mad,
        "projected_mad_ns": candidate_mad,
        "projected_to_nrir42_median_ratio": ratio,
        "median_improvement_ns": improvement,
        "improvement_exceeds_pooled_mad": improvement_exceeds_pooled_mad,
        "maximum_projected_repeat_ns": max(candidates),
    }
    formal: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "base_revision": BASE_REVISION,
            "native_code_revision": _code_revision(),
            "source_nrir42_hash": _source_hash(),
            "workload": _public_workload(workload),
        },
        "contract": {
            "repeat_count": REPEAT_COUNT,
            "paired_orders": [list(item) for item in PAIRED_ORDERS],
            "expected_rank": list(EXPECTED_RANK),
            "expected_selected": list(EXPECTED_SELECTED),
            "full_evaluations": FULL_EVALUATIONS,
            "projected_evaluations": PROJECTED_EVALUATIONS,
            "maximum_projected_repeat_ns": MAX_PROJECTED_REPEAT_NS,
            "maximum_median_ratio": MAX_MEDIAN_RATIO,
        },
        "workers": [
            {
                "repeat_index": worker["repeat_index"],
                "worker_hash": worker["worker_hash"],
            }
            for worker in workers
        ],
        "metrics": metrics,
        "decision": decision,
        "decision_hash": _canonical_hash(decision),
        "status": "validated-reduced" if phase_a_go else "validated-no-go",
        "performance_claimed": False,
    }
    formal["formal_hash"] = _canonical_hash(formal)
    return formal


def validate_formal(value: Mapping[str, Any]) -> None:
    contract = value.get("contract", {})
    metrics = value.get("metrics", {})
    decision = value.get("decision", {})
    semantic = {key: item for key, item in value.items() if key != "formal_hash"}
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("source", {}).get("base_revision") != BASE_REVISION
        or value.get("source", {}).get("native_code_revision") != _code_revision()
        or value.get("source", {}).get("source_nrir42_hash") != _source_hash()
        or contract.get("repeat_count") != REPEAT_COUNT
        or contract.get("paired_orders") != [list(item) for item in PAIRED_ORDERS]
        or contract.get("expected_rank") != list(EXPECTED_RANK)
        or contract.get("expected_selected") != list(EXPECTED_SELECTED)
        or contract.get("full_evaluations") != FULL_EVALUATIONS
        or contract.get("projected_evaluations") != PROJECTED_EVALUATIONS
        or contract.get("maximum_projected_repeat_ns") != MAX_PROJECTED_REPEAT_NS
        or contract.get("maximum_median_ratio") != MAX_MEDIAN_RATIO
        or len(value.get("workers", [])) != REPEAT_COUNT
        or len(metrics.get("nrir42_elapsed_ns", [])) != REPEAT_COUNT
        or len(metrics.get("projected_elapsed_ns", [])) != REPEAT_COUNT
        or decision.get("phase_a_go")
        != (
            decision.get("parity_passed")
            and decision.get("budget_gate_passed")
            and decision.get("timing_gate_passed")
        )
        or value.get("decision_hash") != _canonical_hash(decision)
        or value.get("status")
        != ("validated-reduced" if decision.get("phase_a_go") else "validated-no-go")
        or value.get("performance_claimed") is not False
        or value.get("formal_hash") != _canonical_hash(semantic)
    ):
        raise ValueError("NRIR44 formal envelope differs")


def _run_subprocess(command: list[str], log_path: Path) -> None:
    completed = subprocess.run(
        command,
        cwd=_repo_root(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=WORKER_TIMEOUT_SECONDS,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"NRIR44 worker failed with exit {completed.returncode}: {log_path}"
        )


def _manifest(artifact_dir: Path, formal: Mapping[str, Any]) -> dict[str, Any]:
    files = {
        str(path.relative_to(artifact_dir)): _file_sha256(path)
        for path in sorted(artifact_dir.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    value: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "formal_hash": formal["formal_hash"],
        "files": files,
        "performance_claimed": False,
    }
    value["manifest_hash"] = _canonical_hash(value)
    return value


def _workload(benchmark_root: Path) -> Mapping[str, Any]:
    return next(
        item
        for item in _resolve_workloads(benchmark_root.resolve())
        if item["workload_id"] == "cifar10_resnet:000"
    )


def _generate(args: argparse.Namespace) -> None:
    if args.torch_threads != TORCH_THREADS:
        raise ValueError("NRIR44 torch thread count differs")
    workload = _workload(args.benchmark_root)
    artifact_dir = args.artifact_dir.resolve()
    workers: list[dict[str, Any]] = []
    for repeat_index in range(REPEAT_COUNT):
        shard = artifact_dir / "shards" / f"repeat-{repeat_index}.json"
        log = artifact_dir / "logs" / f"repeat-{repeat_index}.log"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "worker",
            "--model",
            str(workload["model"]),
            "--property",
            str(workload["property"]),
            "--result-json",
            str(shard),
            "--repeat-index",
            str(repeat_index),
            "--torch-threads",
            str(args.torch_threads),
        ]
        _run_subprocess(command, log)
        worker = _load_json(shard)
        validate_worker(worker, repeat_index=repeat_index)
        workers.append(worker)
    formal = _build_formal(workers, workload=workload)
    validate_formal(formal)
    _write_json(artifact_dir / "formal.json", formal)
    _write_json(artifact_dir / "manifest.json", _manifest(artifact_dir, formal))
    print(
        _canonical_json(
            {
                "status": formal["status"],
                "formal_hash": formal["formal_hash"],
                "metrics": formal["metrics"],
                "next_route": formal["decision"]["next_route"],
            }
        )
    )


def _replay(args: argparse.Namespace) -> None:
    if args.torch_threads != TORCH_THREADS:
        raise ValueError("NRIR44 torch thread count differs")
    workload = _workload(args.benchmark_root)
    artifact_dir = args.artifact_dir.resolve()
    manifest = _load_json(artifact_dir / "manifest.json")
    semantic_manifest = {
        key: item for key, item in manifest.items() if key != "manifest_hash"
    }
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or manifest.get("performance_claimed") is not False
        or manifest.get("manifest_hash") != _canonical_hash(semantic_manifest)
    ):
        raise ValueError("NRIR44 manifest envelope differs")
    for relative, expected in manifest["files"].items():
        if _file_sha256(artifact_dir / relative) != expected:
            raise ValueError(f"NRIR44 artifact digest differs: {relative}")
    workers = []
    for repeat_index in range(REPEAT_COUNT):
        worker = _load_json(artifact_dir / "shards" / f"repeat-{repeat_index}.json")
        validate_worker(worker, repeat_index=repeat_index)
        workers.append(worker)
    rebuilt = _build_formal(workers, workload=workload)
    formal = _load_json(artifact_dir / "formal.json")
    validate_formal(formal)
    if rebuilt != formal or manifest["formal_hash"] != formal["formal_hash"]:
        raise ValueError("NRIR44 formal replay differs")
    print(
        _canonical_json(
            {
                "status": formal["status"],
                "formal_hash": formal["formal_hash"],
                "decision_hash": formal["decision_hash"],
                "next_route": formal["decision"]["next_route"],
                "replay": "passed",
            }
        )
    )


def main() -> None:
    args = _parse_args()
    if args.command == "worker":
        _run_worker(args)
    elif args.command == "generate":
        _generate(args)
    else:
        _replay(args)


if __name__ == "__main__":
    main()
