#!/usr/bin/env python3
"""Generate or replay NRIR45 prepared-refinement Phase-A evidence."""

# pylint: disable=too-many-lines,too-many-locals,too-many-statements
# pylint: disable=too-many-branches,too-many-boolean-expressions
# pylint: disable=missing-function-docstring,import-outside-toplevel,protected-access
# pylint: disable=duplicate-code,cell-var-from-loop,too-many-arguments

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from boundflow.ir.prepared_intermediate_refinement import (
    NativePreparedIntermediateRefinementCapsuleIR,
    NativePreparedRefinementExecutionTraceIR,
    NativePreparedRefinementScheduleAction,
    NativePreparedRefinementScheduleIR,
    NativePreparedRefinementTaskIRModule,
    NativePreparedRefinementTaskIRUnit,
    NativePreparedRefinementTaskKind,
)
from scripts.run_objective_branch_scorer_ownership_formal import (
    _branch_semantics,
    _queue_semantics,
    _refinement_semantics,
    _state_semantics,
)
from scripts.run_typed_hard_clause_escalation_artifact import (
    _load_query_runtime,
    _policies,
    _public_workload,
    _resolve_workloads,
)

FORMAL_SCHEMA_VERSION = "boundflow.prepared-refinement-formal/v1"
WORKER_SCHEMA_VERSION = "boundflow.prepared-refinement-worker/v1"
MANIFEST_SCHEMA_VERSION = "boundflow.prepared-refinement-manifest/v1"
ARTIFACT_DIR = Path(
    "artifacts/prepared-intermediate-refinement/"
    "vnncomp21-resnet2b-property0-three-repeat-cpu-phase-a-v1"
)
SOURCE_NRIR44 = Path(
    "artifacts/root-projection-floor/"
    "vnncomp21-resnet2b-property0-three-repeat-cpu-phase-b-v1/formal.json"
)
EXPECTED_SOURCE_NRIR44_HASH = (
    "2f22d44fe9f57f233c8a853b66f67f404b03a087d097451e10f663ee257272d9"
)
EXPECTED_CLAUSES = (2, 3)
EXPECTED_WORST_ACTIVE_LOWERS = {
    2: -35.53092575073242,
    3: -30.258447647094727,
}
PAIRED_ORDERS = (
    ("control", "prepared"),
    ("prepared", "control"),
    ("control", "prepared"),
)
REPEAT_COUNT = 3
TORCH_THREADS = 8
WORKER_TIMEOUT_SECONDS = 240
REQUIRED_NODES = 31
REQUIRED_SIBLING_GROUPS = 15
MAXIMUM_QUEUE_MEDIAN_RATIO = 0.80


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
        "boundflow/ir/prepared_intermediate_refinement.py",
        "boundflow/runtime/native_prepared_intermediate_refinement.py",
        "boundflow/runtime/native_prepared_per_child_refinement.py",
        "boundflow/runtime/native_prepared_shared_parametric_ancestral.py",
        "boundflow/runtime/native_prepared_objective_branch_shared_production_queue.py",
        "scripts/run_prepared_intermediate_refinement_formal.py",
    )
    return _canonical_hash({path: _file_sha256(root / path) for path in paths})


def _source_hash() -> str:
    value = _load_json(_repo_root() / SOURCE_NRIR44)
    if value.get("formal_payload_hash") != EXPECTED_SOURCE_NRIR44_HASH:
        raise ValueError("NRIR45 frozen NRIR44 source differs")
    return value["formal_payload_hash"]


def _prepared_payload(node_refinement: Any) -> dict[str, Any]:
    from boundflow.runtime.native_prepared_intermediate_refinement import (
        NativePreparedIntermediateRefinementExecution,
        NativePreparedIntermediateRefinementProgram,
    )

    execution = node_refinement.execution
    program = node_refinement.program
    if not isinstance(
        program, NativePreparedIntermediateRefinementProgram
    ) or not isinstance(execution, NativePreparedIntermediateRefinementExecution):
        raise TypeError("NRIR45 prepared node ownership differs")
    payload: dict[str, Any] = {
        "node_id": node_refinement.node_id,
        "source_hashes": program.hashes(),
        "source_targets": [target.to_dict() for target in program.plan.targets],
        "capsule": program.capsule.to_dict(),
        "task_module": program.prepared_task_module.to_dict(capsule=program.capsule),
        "schedule": program.prepared_schedule.to_dict(
            capsule=program.capsule, task_module=program.prepared_task_module
        ),
        "trace": execution.prepared_trace.to_dict(
            capsule=program.capsule,
            task_module=program.prepared_task_module,
            schedule=program.prepared_schedule,
        ),
        "performance_claimed": False,
    }
    payload["payload_hash"] = _canonical_hash(payload)
    return payload


def _validate_prepared_payload(value: Mapping[str, Any]) -> None:
    capsule = NativePreparedIntermediateRefinementCapsuleIR(**value["capsule"])
    task_value = value["task_module"]
    task_module = NativePreparedRefinementTaskIRModule(
        module_id=task_value["module_id"],
        capsule_hash=task_value["capsule_hash"],
        tasks=tuple(
            NativePreparedRefinementTaskIRUnit(
                task_id=item["task_id"],
                kind=NativePreparedRefinementTaskKind(item["kind"]),
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
    schedule = NativePreparedRefinementScheduleIR(
        schedule_id=schedule_value["schedule_id"],
        capsule_hash=schedule_value["capsule_hash"],
        task_module_hash=schedule_value["task_module_hash"],
        actions=tuple(
            NativePreparedRefinementScheduleAction(**item)
            for item in schedule_value["actions"]
        ),
        full_validation_launches=schedule_value["full_validation_launches"],
        runtime_target_selection_launches=schedule_value[
            "runtime_target_selection_launches"
        ],
        schema_version=schedule_value["schema_version"],
    )
    trace = NativePreparedRefinementExecutionTraceIR(**value["trace"])
    trace.validate(capsule=capsule, task_module=task_module, schedule=schedule)
    source_hashes = value["source_hashes"]
    semantic = {key: item for key, item in value.items() if key != "payload_hash"}
    if (
        capsule.source_plan_hash != source_hashes["refinement_plan_hash"]
        or capsule.source_task_module_hash
        != source_hashes["refinement_task_module_hash"]
        or capsule.source_schedule_hash != source_hashes["refinement_schedule_hash"]
        or capsule.target_table_hash != _canonical_hash(value["source_targets"])
        or value.get("performance_claimed") is not False
        or value.get("payload_hash") != _canonical_hash(semantic)
    ):
        raise ValueError("NRIR45 prepared payload binding differs")


def _semantic_tables(execution: Any) -> dict[str, Any]:
    return {
        "queue": _queue_semantics(execution),
        "branches": _branch_semantics(execution),
        "states": _state_semantics(execution),
        "refinements": _refinement_semantics(execution),
    }


def _execute_mode(
    *,
    mode: str,
    plan: Any,
    module: Any,
    input_spec: Any,
    objective: Any,
    threshold: Any,
    refinement: Any,
    optimizer_policy: Any,
    branch_policy: Any,
    query_id: str,
) -> tuple[Any, dict[str, int], int]:
    import boundflow.runtime.native_intermediate_refinement as refinement_runtime
    from boundflow.runtime.native_parametric_optimizer import (
        NativeParametricOptimizerTemplateCache,
    )
    from boundflow.runtime.native_prepared_intermediate_refinement import (
        NativePreparedIntermediateRefinementProgram,
    )
    from boundflow.runtime.native_prepared_objective_branch_shared_production_queue import (
        execute_native_prepared_objective_branch_shared_production_queue,
    )
    from boundflow.runtime.native_prevalidated_objective_branch_shared_production_queue import (
        execute_native_prevalidated_objective_branch_shared_production_queue,
    )

    counts = {
        "target_selection": 0,
        "full_program_validation": 0,
        "full_program_hash": 0,
        "prepared_fast_validation": 0,
    }
    original_select = refinement_runtime._select_targets
    original_validate = refinement_runtime.NativeIntermediateRefinementProgram.validate
    original_hashes = refinement_runtime.NativeIntermediateRefinementProgram.hashes
    original_prepared_validate = NativePreparedIntermediateRefinementProgram.validate

    def counted_select(*args: Any, **kwargs: Any):
        counts["target_selection"] += 1
        return original_select(*args, **kwargs)

    def counted_validate(self: Any, *args: Any, **kwargs: Any):
        counts["full_program_validation"] += 1
        return original_validate(self, *args, **kwargs)

    def counted_hashes(self: Any):
        counts["full_program_hash"] += 1
        return original_hashes(self)

    def counted_prepared_validate(self: Any, *args: Any, **kwargs: Any):
        counts["prepared_fast_validation"] += 1
        return original_prepared_validate(self, *args, **kwargs)

    refinement_runtime._select_targets = counted_select
    refinement_runtime.NativeIntermediateRefinementProgram.validate = counted_validate
    refinement_runtime.NativeIntermediateRefinementProgram.hashes = counted_hashes
    setattr(
        NativePreparedIntermediateRefinementProgram,
        "validate",
        counted_prepared_validate,
    )
    execute = (
        execute_native_prevalidated_objective_branch_shared_production_queue
        if mode == "control"
        else execute_native_prepared_objective_branch_shared_production_queue
    )
    started_ns = time.monotonic_ns()
    try:
        execution = execute(
            plan,
            module,
            input_spec,
            linear_spec_C=objective,
            threshold=threshold,
            root_refinement=refinement,
            optimizer_policy=optimizer_policy,
            branch_policy=branch_policy,
            compiler_cache=NativeParametricOptimizerTemplateCache(),
            query_id=query_id,
        )
    finally:
        elapsed_ns = time.monotonic_ns() - started_ns
        refinement_runtime._select_targets = original_select
        refinement_runtime.NativeIntermediateRefinementProgram.validate = (
            original_validate
        )
        refinement_runtime.NativeIntermediateRefinementProgram.hashes = original_hashes
        setattr(
            NativePreparedIntermediateRefinementProgram,
            "validate",
            original_prepared_validate,
        )
    return execution, counts, elapsed_ns


def _run_worker(args: argparse.Namespace) -> None:
    import torch

    from boundflow.runtime.native_objective_branch_score import (
        NativeObjectiveBranchPolicy,
    )
    from boundflow.runtime.native_objective_branch_shared_evaluator import (
        compile_native_objective_branch_shared_plan,
    )
    from boundflow.runtime.native_prepared_intermediate_refinement import (
        NativePreparedIntermediateRefinementExecution,
        validate_native_prepared_intermediate_refinement_full,
    )
    from boundflow.runtime.native_relu_split_bab_runtime import _repeat_box_input_spec
    from scripts.run_objective_ancestral_queue_artifact import _active_frontier
    from scripts.run_objective_branch_shared_evaluator_pilot import (
        _execute_floor,
        _source,
    )

    if args.repeat_index not in range(REPEAT_COUNT):
        raise ValueError("NRIR45 repeat index differs")
    torch.set_num_threads(args.torch_threads)
    _source_hash()
    _query, tensors, module, input_spec = _load_query_runtime(
        args.model.resolve(), args.property.resolve(), "cifar10_resnet:000"
    )
    search_policy, optimizer_policy = _policies()
    branch_policy = NativeObjectiveBranchPolicy()
    _floor_program, floor, decision = _execute_floor(
        module,
        input_spec,
        tensors.linear_spec_c,
        tensors.thresholds,
        search_policy,
        optimizer_policy,
        query_id=f"nrir45:repeat{args.repeat_index}:floor",
    )
    if decision.selected_original_clause_indices != EXPECTED_CLAUSES:
        raise ValueError("NRIR45 selected clauses differ")
    rows: list[dict[str, Any]] = []
    parities: list[dict[str, Any]] = []
    for ordinal in EXPECTED_CLAUSES:
        source = _source(floor, ordinal)
        objective = tensors.linear_spec_c[:, ordinal : ordinal + 1, :].contiguous()
        threshold = tensors.thresholds[ordinal : ordinal + 1].contiguous()
        query_id = f"nrir45:r{args.repeat_index}:c{ordinal}"
        plan = compile_native_objective_branch_shared_plan(
            module,
            input_spec,
            linear_spec_C=objective,
            threshold=threshold,
            root_refinement=source.refinement,
            optimizer_policy=optimizer_policy,
            branch_policy=branch_policy,
            plan_id=query_id,
        )
        by_mode: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        for position, mode in enumerate(PAIRED_ORDERS[args.repeat_index]):
            execution, counts, measured_ns = _execute_mode(
                mode=mode,
                plan=plan,
                module=module,
                input_spec=input_spec,
                objective=objective,
                threshold=threshold,
                refinement=source.refinement,
                optimizer_policy=optimizer_policy,
                branch_policy=branch_policy,
                query_id=query_id,
            )
            semantics = _semantic_tables(execution)
            frontier = _active_frontier(execution.queue.trace)
            prepared_nodes = [
                item
                for item in execution.node_refinements
                if isinstance(
                    item.execution, NativePreparedIntermediateRefinementExecution
                )
            ]
            full_replay_count = 0
            representative = None
            capsule_summary_hash = None
            if mode == "prepared":
                single_input = _repeat_box_input_spec(input_spec, count=1)
                for item in prepared_nodes:
                    validate_native_prepared_intermediate_refinement_full(
                        item.execution, module, single_input
                    )
                full_replay_count = len(prepared_nodes)
                payloads = [_prepared_payload(item) for item in prepared_nodes]
                representative = payloads[0]
                capsule_summary_hash = _canonical_hash(
                    [
                        {
                            "node_id": item["node_id"],
                            "capsule_hash": _canonical_hash(item["capsule"]),
                            "trace_hash": _canonical_hash(item["trace"]),
                            "payload_hash": item["payload_hash"],
                        }
                        for item in payloads
                    ]
                )
            row: dict[str, Any] = {
                "repeat_index": args.repeat_index,
                "original_clause_index": ordinal,
                "mode": mode,
                "order_position": position,
                "query_id": query_id,
                "queue_elapsed_ns": execution.trace.queue_elapsed_ns,
                "measured_elapsed_ns": measured_ns,
                "accepted_nodes": len(execution.queue.trace.evaluations),
                "sibling_group_count": len(execution.batch_commits) - 1,
                "worst_active_lower": frontier["worst_active_lower"],
                "call_counts": counts,
                "prepared_capsule_count": len(prepared_nodes),
                "full_replay_count": full_replay_count,
                "capsule_summary_hash": capsule_summary_hash,
                "representative_prepared_payload": representative,
                "semantic_hashes": {
                    key: _canonical_hash(value) for key, value in semantics.items()
                },
                "performance_claimed": False,
            }
            row["row_hash"] = _canonical_hash(row)
            rows.append(row)
            by_mode[mode] = (row, semantics)
        control_row, control_semantics = by_mode["control"]
        prepared_row, prepared_semantics = by_mode["prepared"]
        parity = {
            "repeat_index": args.repeat_index,
            "original_clause_index": ordinal,
            "control_row_hash": control_row["row_hash"],
            "prepared_row_hash": prepared_row["row_hash"],
            "exact": control_semantics == prepared_semantics,
        }
        parity["parity_hash"] = _canonical_hash(parity)
        if not parity["exact"]:
            raise ValueError("NRIR45 control/prepared semantics differ")
        parities.append(parity)
    worker: dict[str, Any] = {
        "schema_version": WORKER_SCHEMA_VERSION,
        "source": {
            "native_code_revision": _code_revision(),
            "source_nrir44_hash": _source_hash(),
        },
        "repeat_index": args.repeat_index,
        "paired_order": list(PAIRED_ORDERS[args.repeat_index]),
        "selected_original_clause_indices": list(EXPECTED_CLAUSES),
        "rows": rows,
        "parities": parities,
        "performance_claimed": False,
    }
    worker["worker_hash"] = _canonical_hash(worker)
    validate_worker(worker, repeat_index=args.repeat_index)
    _write_json(args.result_json.resolve(), worker)
    print(
        _canonical_json(
            {
                "repeat_index": args.repeat_index,
                "queue_seconds": {
                    f"c{row['original_clause_index']}:{row['mode']}": row[
                        "queue_elapsed_ns"
                    ]
                    / 1e9
                    for row in rows
                },
                "worker_hash": worker["worker_hash"],
            }
        ),
        flush=True,
    )


def validate_worker(value: Mapping[str, Any], *, repeat_index: int) -> None:
    rows = value.get("rows")
    parities = value.get("parities")
    if (
        value.get("schema_version") != WORKER_SCHEMA_VERSION
        or value.get("source", {}).get("native_code_revision") != _code_revision()
        or value.get("source", {}).get("source_nrir44_hash") != _source_hash()
        or value.get("repeat_index") != repeat_index
        or value.get("paired_order") != list(PAIRED_ORDERS[repeat_index])
        or value.get("selected_original_clause_indices") != list(EXPECTED_CLAUSES)
        or not isinstance(rows, list)
        or len(rows) != 4
        or not isinstance(parities, list)
        or len(parities) != 2
        or value.get("performance_claimed") is not False
    ):
        raise ValueError("NRIR45 worker envelope differs")
    by_key: dict[tuple[int, str], Mapping[str, Any]] = {}
    for row in rows:
        semantic = {key: item for key, item in row.items() if key != "row_hash"}
        ordinal = row.get("original_clause_index")
        mode = row.get("mode")
        counts = row.get("call_counts", {})
        if (
            ordinal not in EXPECTED_CLAUSES
            or mode not in {"control", "prepared"}
            or row.get("accepted_nodes") != REQUIRED_NODES
            or row.get("sibling_group_count") != REQUIRED_SIBLING_GROUPS
            or row.get("worst_active_lower") != EXPECTED_WORST_ACTIVE_LOWERS[ordinal]
            or row.get("queue_elapsed_ns", 0) <= 0
            or row.get("measured_elapsed_ns", 0) <= 0
            or row.get("performance_claimed") is not False
            or row.get("row_hash") != _canonical_hash(semantic)
            or not all(isinstance(item, int) and item >= 0 for item in counts.values())
        ):
            raise ValueError("NRIR45 row differs")
        if mode == "prepared":
            payload = row.get("representative_prepared_payload")
            if (
                row.get("prepared_capsule_count") != 30
                or row.get("full_replay_count") != 30
                or not isinstance(row.get("capsule_summary_hash"), str)
                or not isinstance(payload, dict)
            ):
                raise ValueError("NRIR45 prepared ownership coverage differs")
            _validate_prepared_payload(payload)
        elif any(
            row.get(key) not in {0, None}
            for key in (
                "prepared_capsule_count",
                "full_replay_count",
                "capsule_summary_hash",
                "representative_prepared_payload",
            )
        ):
            raise ValueError("NRIR45 control unexpectedly owns prepared evidence")
        by_key[(ordinal, mode)] = row
    if set(by_key) != {
        (ordinal, mode)
        for ordinal in EXPECTED_CLAUSES
        for mode in ("control", "prepared")
    }:
        raise ValueError("NRIR45 row coverage differs")
    for parity in parities:
        semantic = {key: item for key, item in parity.items() if key != "parity_hash"}
        key = parity["original_clause_index"]
        if (
            parity.get("repeat_index") != repeat_index
            or key not in EXPECTED_CLAUSES
            or parity.get("control_row_hash") != by_key[(key, "control")]["row_hash"]
            or parity.get("prepared_row_hash") != by_key[(key, "prepared")]["row_hash"]
            or parity.get("exact") is not True
            or parity.get("parity_hash") != _canonical_hash(semantic)
        ):
            raise ValueError("NRIR45 parity differs")
    semantic_worker = {key: item for key, item in value.items() if key != "worker_hash"}
    if value.get("worker_hash") != _canonical_hash(semantic_worker):
        raise ValueError("NRIR45 worker hash differs")


def _median_mad(values: list[int]) -> tuple[int, int]:
    median = int(statistics.median(values))
    mad = int(statistics.median(abs(value - median) for value in values))
    return median, mad


def _build_formal(
    workers: Sequence[Mapping[str, Any]], *, workload: Mapping[str, Any]
) -> dict[str, Any]:
    metrics: list[dict[str, Any]] = []
    for ordinal in EXPECTED_CLAUSES:
        controls = [
            row["queue_elapsed_ns"]
            for worker in workers
            for row in worker["rows"]
            if row["original_clause_index"] == ordinal and row["mode"] == "control"
        ]
        prepared = [
            row["queue_elapsed_ns"]
            for worker in workers
            for row in worker["rows"]
            if row["original_clause_index"] == ordinal and row["mode"] == "prepared"
        ]
        control_median, control_mad = _median_mad(controls)
        prepared_median, prepared_mad = _median_mad(prepared)
        improvement = control_median - prepared_median
        metrics.append(
            {
                "original_clause_index": ordinal,
                "control_queue_elapsed_ns": controls,
                "prepared_queue_elapsed_ns": prepared,
                "control_median_ns": control_median,
                "prepared_median_ns": prepared_median,
                "control_mad_ns": control_mad,
                "prepared_mad_ns": prepared_mad,
                "prepared_to_control_median_ratio": (prepared_median / control_median),
                "median_improvement_ns": improvement,
                "improvement_exceeds_pooled_mad": improvement
                > max(control_mad, prepared_mad),
            }
        )
    parity_passed = all(
        parity["exact"] for worker in workers for parity in worker["parities"]
    )
    ownership_passed = all(
        prepared["call_counts"]["target_selection"]
        < control["call_counts"]["target_selection"]
        and prepared["call_counts"]["full_program_validation"]
        < control["call_counts"]["full_program_validation"]
        and prepared["call_counts"]["full_program_hash"]
        < control["call_counts"]["full_program_hash"]
        and prepared["prepared_capsule_count"] == 30
        and prepared["full_replay_count"] == 30
        for worker in workers
        for ordinal in EXPECTED_CLAUSES
        for control in [
            next(
                row
                for row in worker["rows"]
                if row["original_clause_index"] == ordinal and row["mode"] == "control"
            )
        ]
        for prepared in [
            next(
                row
                for row in worker["rows"]
                if row["original_clause_index"] == ordinal and row["mode"] == "prepared"
            )
        ]
    )
    timing_passed = all(
        item["prepared_to_control_median_ratio"] <= MAXIMUM_QUEUE_MEDIAN_RATIO
        and item["improvement_exceeds_pooled_mad"]
        for item in metrics
    )
    phase_a_go = parity_passed and ownership_passed and timing_passed
    decision = {
        "parity_passed": parity_passed,
        "ownership_passed": ownership_passed,
        "timing_passed": timing_passed,
        "phase_a_go": phase_a_go,
        "next_route": "run_nrir45_phase_b" if phase_a_go else "stop_nrir45",
        "reason": (
            "exact semantics, prepared ownership, and timing gates passed"
            if phase_a_go
            else "NRIR45 Phase-A gate failed; Phase B is gated off"
        ),
    }
    formal: dict[str, Any] = {
        "schema_version": FORMAL_SCHEMA_VERSION,
        "source": {
            "base_revision": "b6eb697",
            "native_code_revision": _code_revision(),
            "source_nrir44_hash": _source_hash(),
            "workload": _public_workload(workload),
        },
        "contract": {
            "repeat_count": REPEAT_COUNT,
            "paired_orders": [list(item) for item in PAIRED_ORDERS],
            "clauses": list(EXPECTED_CLAUSES),
            "required_nodes": REQUIRED_NODES,
            "required_sibling_groups": REQUIRED_SIBLING_GROUPS,
            "maximum_queue_median_ratio": MAXIMUM_QUEUE_MEDIAN_RATIO,
            "torch_threads": TORCH_THREADS,
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
    decision = value.get("decision", {})
    semantic = {key: item for key, item in value.items() if key != "formal_hash"}
    if (
        value.get("schema_version") != FORMAL_SCHEMA_VERSION
        or value.get("source", {}).get("base_revision") != "b6eb697"
        or value.get("source", {}).get("native_code_revision") != _code_revision()
        or value.get("source", {}).get("source_nrir44_hash") != _source_hash()
        or contract.get("repeat_count") != REPEAT_COUNT
        or contract.get("paired_orders") != [list(item) for item in PAIRED_ORDERS]
        or contract.get("clauses") != list(EXPECTED_CLAUSES)
        or contract.get("required_nodes") != REQUIRED_NODES
        or contract.get("required_sibling_groups") != REQUIRED_SIBLING_GROUPS
        or contract.get("maximum_queue_median_ratio") != MAXIMUM_QUEUE_MEDIAN_RATIO
        or contract.get("torch_threads") != TORCH_THREADS
        or len(value.get("workers", [])) != REPEAT_COUNT
        or len(value.get("metrics", [])) != len(EXPECTED_CLAUSES)
        or decision.get("phase_a_go")
        != (
            decision.get("parity_passed")
            and decision.get("ownership_passed")
            and decision.get("timing_passed")
        )
        or value.get("decision_hash") != _canonical_hash(decision)
        or value.get("status")
        != ("validated-reduced" if decision.get("phase_a_go") else "validated-no-go")
        or value.get("performance_claimed") is not False
        or value.get("formal_hash") != _canonical_hash(semantic)
    ):
        raise ValueError("NRIR45 formal envelope differs")


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
            f"NRIR45 worker failed with exit {completed.returncode}: {log_path}"
        )


def _generate(args: argparse.Namespace) -> None:
    if args.torch_threads != TORCH_THREADS:
        raise ValueError("NRIR45 torch thread count differs")
    workload = _workload(args.benchmark_root)
    artifact_dir = args.artifact_dir.resolve()
    workers = []
    for repeat_index in range(REPEAT_COUNT):
        shard = artifact_dir / "shards" / f"repeat-{repeat_index}.json"
        log = artifact_dir / "logs" / f"repeat-{repeat_index}.log"
        _run_subprocess(
            [
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
            ],
            log,
        )
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
                "decision": formal["decision"],
                "metrics": formal["metrics"],
            }
        )
    )


def _replay(args: argparse.Namespace) -> None:
    artifact_dir = args.artifact_dir.resolve()
    manifest = _load_json(artifact_dir / "manifest.json")
    files = manifest.get("files", {})
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or manifest.get("performance_claimed") is not False
        or manifest.get("manifest_hash")
        != _canonical_hash(
            {key: item for key, item in manifest.items() if key != "manifest_hash"}
        )
        or any(
            _file_sha256(artifact_dir / path) != digest
            for path, digest in files.items()
        )
    ):
        raise ValueError("NRIR45 manifest differs")
    workers = []
    for repeat_index in range(REPEAT_COUNT):
        worker = _load_json(artifact_dir / "shards" / f"repeat-{repeat_index}.json")
        validate_worker(worker, repeat_index=repeat_index)
        workers.append(worker)
    formal = _load_json(artifact_dir / "formal.json")
    validate_formal(formal)
    if manifest.get("formal_hash") != formal["formal_hash"]:
        raise ValueError("NRIR45 manifest/formal differs")
    rebuilt = _build_formal(workers, workload=_workload(args.benchmark_root))
    if rebuilt != formal:
        raise ValueError("NRIR45 formal replay differs")
    representative = next(
        row["representative_prepared_payload"]
        for worker in workers
        for row in worker["rows"]
        if row["mode"] == "prepared"
    )
    tampered = copy.deepcopy(representative)
    tampered["source_targets"][0]["initial_lower"] += 0.125
    tampered["payload_hash"] = _canonical_hash(
        {key: item for key, item in tampered.items() if key != "payload_hash"}
    )
    try:
        _validate_prepared_payload(tampered)
    except ValueError:
        tamper_rejected = True
    else:
        tamper_rejected = False
    if not tamper_rejected:
        raise ValueError("NRIR45 synchronized outer-rehash tamper was accepted")
    print(
        _canonical_json(
            {
                "status": "replay-passed",
                "formal_hash": formal["formal_hash"],
                "typed_payloads_replayed": REPEAT_COUNT * len(EXPECTED_CLAUSES),
                "outer_rehash_tamper_rejected": True,
            }
        )
    )


def main() -> None:
    args = _parse_args()
    if args.command == "worker":
        _run_worker(args)
    elif args.command == "generate":
        _generate(args)
    elif args.command == "replay":
        _replay(args)
    else:
        raise AssertionError("unreachable NRIR45 command")


if __name__ == "__main__":
    main()
