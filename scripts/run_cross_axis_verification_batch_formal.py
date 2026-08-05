#!/usr/bin/env python3
"""Generate or replay NRIR-43 Phase-A cross-axis batch evidence."""

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
from typing import Any, Mapping, Sequence, cast

from boundflow.ir.cross_axis_verification_batch import (
    NativeCrossAxisVerificationBatchInstanceIR,
    NativeCrossAxisVerificationBatchPlanIR,
    NativeCrossAxisVerificationBatchScheduleAction,
    NativeCrossAxisVerificationBatchScheduleIR,
    NativeCrossAxisVerificationBatchSegmentIR,
    NativeCrossAxisVerificationBatchTaskIRModule,
    NativeCrossAxisVerificationBatchTaskIRUnit,
    NativeCrossAxisVerificationBatchTaskKind,
    NativeCrossAxisVerificationBatchTraceIR,
)
from boundflow.ir.cross_axis_verification_batch_evidence import (
    NativeCrossAxisVerificationBatchClauseMetricIR,
    NativeCrossAxisVerificationBatchDecisionIR,
    NativeCrossAxisVerificationBatchEvidencePlanIR,
    NativeCrossAxisVerificationBatchParityIR,
    NativeCrossAxisVerificationBatchRowIR,
)
from scripts.run_objective_branch_scorer_ownership_formal import (
    _branch_semantics,
    _capsule_table,
    _queue_semantics,
    _refinement_semantics,
    _state_semantics,
    _validate_serialized_capsule,
)
from scripts.run_typed_hard_clause_escalation_artifact import (
    _load_query_runtime,
    _policies,
    _public_workload,
    _resolve_workloads,
)

ARTIFACT_DIR = Path(
    "artifacts/cross-axis-verification-batch/"
    "vnncomp21-resnet2b-property0-three-repeat-cpu-phase-a-v1"
)
SOURCE_PHASE_A = Path(
    "artifacts/objective-branch-scorer-ownership/"
    "vnncomp21-resnet2b-property0-three-repeat-cpu-v1/formal.json"
)
SOURCE_PHASE_B = Path(
    "artifacts/objective-branch-scorer-ownership-global/"
    "vnncomp21-resnet2b-property0-three-repeat-cpu-formal-v1/formal.json"
)
EXPECTED_PHASE_A_HASH = (
    "0d310c2ffc96844648a83f9921bc7f353ec8425986bccb36f75e6d1cd2b25b58"
)
EXPECTED_PHASE_B_HASH = (
    "7274e834b3bf08a9e138fa3284b70222620cf3c571395331e1a87ed5fee7d759"
)
SCHEMA_VERSION = "boundflow.cross-axis-verification-batch-formal/v1"
WORKER_SCHEMA_VERSION = "boundflow.cross-axis-verification-batch-worker/v1"
MANIFEST_SCHEMA_VERSION = "boundflow.cross-axis-verification-batch-manifest/v1"
REPEAT_COUNT = 3
TORCH_THREADS = 8
WORKER_TIMEOUT_SECONDS = 300


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
        "boundflow/ir/cross_axis_verification_batch.py",
        "boundflow/ir/cross_axis_verification_batch_evidence.py",
        "boundflow/runtime/native_cross_axis_prevalidated_objective_branch.py",
        "boundflow/runtime/native_cross_axis_prevalidated_objective_branch_shared_evaluator.py",
        "boundflow/runtime/native_cross_axis_objective_branch_shared_production_queue.py",
        "boundflow/runtime/native_prevalidated_objective_branch_shared_production_queue.py",
        "scripts/run_cross_axis_verification_batch_formal.py",
    )
    return _canonical_hash({path: _file_sha256(root / path) for path in paths})


def _source_hash(path: Path, key: str) -> str:
    value = _load_json(_repo_root() / path)
    result = value.get(key)
    if not isinstance(result, str):
        raise ValueError(f"NRIR-43 source hash missing: {path}")
    return result


def _plan() -> NativeCrossAxisVerificationBatchEvidencePlanIR:
    phase_a = _source_hash(SOURCE_PHASE_A, "formal_hash")
    phase_b = _source_hash(SOURCE_PHASE_B, "formal_payload_hash")
    if phase_a != EXPECTED_PHASE_A_HASH or phase_b != EXPECTED_PHASE_B_HASH:
        raise ValueError("NRIR-43 frozen source differs")
    plan = NativeCrossAxisVerificationBatchEvidencePlanIR(
        plan_id="nrir43:cifar10_resnet:000:property0:phase-a",
        source_phase_a_hash=phase_a,
        source_phase_b_hash=phase_b,
    )
    plan.validate()
    return plan


def _plan_from_dict(
    value: Mapping[str, Any],
) -> NativeCrossAxisVerificationBatchEvidencePlanIR:
    return NativeCrossAxisVerificationBatchEvidencePlanIR(
        plan_id=value["plan_id"],
        source_phase_a_hash=value["source_phase_a_hash"],
        source_phase_b_hash=value["source_phase_b_hash"],
        clause_ordinals=tuple(value["clause_ordinals"]),
        paired_orders=tuple(tuple(item) for item in value["paired_orders"]),
        required_nodes=value["required_nodes"],
        required_sibling_groups=value["required_sibling_groups"],
        nrir42_scorer_launches_per_clause=value["nrir42_scorer_launches_per_clause"],
        cross_axis_scorer_launches_per_clause=value[
            "cross_axis_scorer_launches_per_clause"
        ],
        maximum_queue_median_ratio=value["maximum_queue_median_ratio"],
        torch_threads=value["torch_threads"],
        performance_claimed=value["performance_claimed"],
        schema_version=value["schema_version"],
    )


def _row_from_dict(value: Mapping[str, Any]) -> NativeCrossAxisVerificationBatchRowIR:
    return NativeCrossAxisVerificationBatchRowIR(
        plan_hash=value["plan_hash"],
        repeat_index=value["repeat_index"],
        original_clause_index=value["original_clause_index"],
        mode=value["mode"],
        order_position=value["order_position"],
        queue_elapsed_ns=value["queue_elapsed_ns"],
        accepted_nodes=value["accepted_nodes"],
        sibling_group_count=value["sibling_group_count"],
        scorer_launch_count=value["scorer_launch_count"],
        scorer_node_widths=tuple(value["scorer_node_widths"]),
        scorer_child_domain_widths=tuple(value["scorer_child_domain_widths"]),
        queue_semantic_hash=value["queue_semantic_hash"],
        branch_semantic_hash=value["branch_semantic_hash"],
        state_semantic_hash=value["state_semantic_hash"],
        refinement_semantic_hash=value["refinement_semantic_hash"],
        cross_batch_trace_hashes=tuple(value["cross_batch_trace_hashes"]),
        performance_claimed=value["performance_claimed"],
        schema_version=value["schema_version"],
    )


def _parity_from_dict(
    value: Mapping[str, Any],
) -> NativeCrossAxisVerificationBatchParityIR:
    return NativeCrossAxisVerificationBatchParityIR(**value)


def _metric_from_dict(
    value: Mapping[str, Any],
) -> NativeCrossAxisVerificationBatchClauseMetricIR:
    return NativeCrossAxisVerificationBatchClauseMetricIR(**value)


def _decision_from_dict(
    value: Mapping[str, Any],
) -> NativeCrossAxisVerificationBatchDecisionIR:
    return NativeCrossAxisVerificationBatchDecisionIR(
        plan_hash=value["plan_hash"],
        clause_metrics=tuple(
            _metric_from_dict(item) for item in value["clause_metrics"]
        ),
        parity_passed=value["parity_passed"],
        launch_gate_passed=value["launch_gate_passed"],
        timing_gate_passed=value["timing_gate_passed"],
        phase_a_go=value["phase_a_go"],
        next_route=value["next_route"],
        reason=value["reason"],
        performance_claimed=value["performance_claimed"],
    )


def _cross_batch_payload(value: Any) -> dict[str, object]:
    plan = value.program.plan
    instance = value.program.instance
    task_module = value.program.task_module
    schedule = value.program.schedule
    trace = value.trace
    return {
        "plan": plan.to_dict(),
        "instance": instance.to_dict(plan=plan),
        "task_module": task_module.to_dict(plan=plan, instance=instance),
        "schedule": schedule.to_dict(
            plan=plan, instance=instance, task_module=task_module
        ),
        "trace": trace.to_dict(
            plan=plan,
            instance=instance,
            task_module=task_module,
            schedule=schedule,
        ),
        "trace_hash": trace.stable_hash(
            plan=plan,
            instance=instance,
            task_module=task_module,
            schedule=schedule,
        ),
    }


def _validate_cross_batch(value: Mapping[str, Any]) -> None:
    plan_value = value["plan"]
    plan = NativeCrossAxisVerificationBatchPlanIR(
        plan_id=plan_value["plan_id"],
        optimizer_policy_hash=plan_value["optimizer_policy_hash"],
        branch_policy_hash=plan_value["branch_policy_hash"],
        segments=tuple(
            NativeCrossAxisVerificationBatchSegmentIR(**item)
            for item in plan_value["segments"]
        ),
        clause_count=plan_value["clause_count"],
        node_count=plan_value["node_count"],
        candidate_count=plan_value["candidate_count"],
        child_domain_count=plan_value["child_domain_count"],
        max_child_domains=plan_value["max_child_domains"],
        semantics_owner=plan_value["semantics_owner"],
        performance_claimed=plan_value["performance_claimed"],
        schema_version=plan_value["schema_version"],
    )
    instance_value = value["instance"]
    instance = NativeCrossAxisVerificationBatchInstanceIR(
        instance_id=instance_value["instance_id"],
        plan_hash=instance_value["plan_hash"],
        segment_hashes=tuple(instance_value["segment_hashes"]),
        ready_set_hash=instance_value["ready_set_hash"],
        child_domain_count=instance_value["child_domain_count"],
        semantic_token=instance_value["semantic_token"],
        schema_version=instance_value["schema_version"],
    )
    task_value = value["task_module"]
    task_module = NativeCrossAxisVerificationBatchTaskIRModule(
        module_id=task_value["module_id"],
        plan_hash=task_value["plan_hash"],
        instance_hash=task_value["instance_hash"],
        tasks=tuple(
            NativeCrossAxisVerificationBatchTaskIRUnit(
                task_id=item["task_id"],
                kind=NativeCrossAxisVerificationBatchTaskKind(item["kind"]),
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
    schedule = NativeCrossAxisVerificationBatchScheduleIR(
        schedule_id=schedule_value["schedule_id"],
        plan_hash=schedule_value["plan_hash"],
        instance_hash=schedule_value["instance_hash"],
        task_module_hash=schedule_value["task_module_hash"],
        actions=tuple(
            NativeCrossAxisVerificationBatchScheduleAction(**item)
            for item in schedule_value["actions"]
        ),
        lower_launch_count=schedule_value["lower_launch_count"],
        schema_version=schedule_value["schema_version"],
    )
    trace_value = value["trace"]
    trace = NativeCrossAxisVerificationBatchTraceIR(
        plan_hash=trace_value["plan_hash"],
        instance_hash=trace_value["instance_hash"],
        task_module_hash=trace_value["task_module_hash"],
        schedule_hash=trace_value["schedule_hash"],
        batch_child_lower_hash=trace_value["batch_child_lower_hash"],
        segment_child_lower_hashes=tuple(trace_value["segment_child_lower_hashes"]),
        segment_score_hashes=tuple(trace_value["segment_score_hashes"]),
        selected_candidate_ordinals=tuple(trace_value["selected_candidate_ordinals"]),
        lower_launch_count=trace_value["lower_launch_count"],
        performance_claimed=trace_value["performance_claimed"],
        schema_version=trace_value["schema_version"],
    )
    trace.validate(
        plan=plan,
        instance=instance,
        task_module=task_module,
        schedule=schedule,
    )
    if value["trace_hash"] != trace.stable_hash(
        plan=plan,
        instance=instance,
        task_module=task_module,
        schedule=schedule,
    ):
        raise ValueError("NRIR-43 cross batch trace hash differs")


def _execute_mode(
    *,
    mode: str,
    clause_ordinal: int,
    plan: Any,
    module: Any,
    input_spec: Any,
    objective: Any,
    threshold: Any,
    refinement: Any,
    optimizer_policy: Any,
    branch_policy: Any,
    query_id: str,
) -> Any:
    from boundflow.runtime.native_cross_axis_objective_branch_shared_production_queue import (
        execute_native_cross_axis_objective_branch_shared_production_queue,
    )
    from boundflow.runtime.native_parametric_optimizer import (
        NativeParametricOptimizerTemplateCache,
    )
    from boundflow.runtime.native_prevalidated_objective_branch_shared_production_queue import (
        execute_native_prevalidated_objective_branch_shared_production_queue,
    )

    common = {
        "linear_spec_C": objective,
        "threshold": threshold,
        "root_refinement": refinement,
        "optimizer_policy": optimizer_policy,
        "branch_policy": branch_policy,
        "compiler_cache": NativeParametricOptimizerTemplateCache(),
        "query_id": query_id,
    }
    if mode == "nrir42":
        return execute_native_prevalidated_objective_branch_shared_production_queue(
            plan, module, input_spec, **common
        )
    if mode == "cross_axis":
        return execute_native_cross_axis_objective_branch_shared_production_queue(
            plan,
            module,
            input_spec,
            clause_ordinal=clause_ordinal,
            **common,
        )
    raise ValueError("NRIR-43 execution mode differs")


def _run_worker(args: argparse.Namespace) -> None:
    import torch

    from boundflow.runtime.native_objective_branch_score import (
        NativeObjectiveBranchPolicy,
    )
    from boundflow.runtime.native_objective_branch_shared_evaluator import (
        compile_native_objective_branch_shared_plan,
    )
    from scripts.run_objective_branch_shared_evaluator_pilot import (
        _execute_floor,
        _source,
    )

    torch.set_num_threads(args.torch_threads)
    evidence_plan = _plan()
    _query, tensors, module, input_spec = _load_query_runtime(
        args.model.resolve(), args.property.resolve(), "cifar10_resnet:000"
    )
    search_policy, optimizer_policy = _policies()
    branch_policy = NativeObjectiveBranchPolicy()
    _floor_program, floor, floor_decision = _execute_floor(
        module,
        input_spec,
        tensors.linear_spec_c,
        tensors.thresholds,
        search_policy,
        optimizer_policy,
        query_id=f"nrir43:repeat{args.repeat_index}:floor",
    )
    if floor_decision.selected_original_clause_indices != evidence_plan.clause_ordinals:
        raise RuntimeError("NRIR-43 floor selection differs")
    raw_runs: list[dict[str, Any]] = []
    parities: list[dict[str, Any]] = []
    for ordinal in evidence_plan.clause_ordinals:
        source = _source(floor, ordinal)
        objective = tensors.linear_spec_c[:, ordinal : ordinal + 1, :].contiguous()
        threshold = tensors.thresholds[ordinal : ordinal + 1].contiguous()
        composite = compile_native_objective_branch_shared_plan(
            module,
            input_spec,
            linear_spec_C=objective,
            threshold=threshold,
            root_refinement=source.refinement,
            optimizer_policy=optimizer_policy,
            branch_policy=branch_policy,
            plan_id=f"nrir43:r{args.repeat_index}:clause:{ordinal}",
        )
        by_mode: dict[str, dict[str, Any]] = {}
        for position, mode in enumerate(evidence_plan.paired_orders[args.repeat_index]):
            execution = _execute_mode(
                mode=mode,
                clause_ordinal=ordinal,
                plan=composite,
                module=module,
                input_spec=input_spec,
                objective=objective,
                threshold=threshold,
                refinement=source.refinement,
                optimizer_policy=optimizer_policy,
                branch_policy=branch_policy,
                query_id=f"nrir43:r{args.repeat_index}:c{ordinal}",
            )
            queue = _queue_semantics(execution)
            branches = _branch_semantics(execution)
            states = _state_semantics(execution)
            refinements = _refinement_semantics(execution)
            capsules = _capsule_table(execution, mode="prevalidated")
            if mode == "cross_axis":
                cross_batches = [
                    _cross_batch_payload(item) for item in execution.scorer_batches
                ]
                node_widths = tuple(
                    item.program.plan.node_count for item in execution.scorer_batches
                )
                child_widths = tuple(
                    item.program.plan.child_domain_count
                    for item in execution.scorer_batches
                )
            else:
                cross_batches = []
                node_widths = (1,) * len(branches)
                child_widths = tuple(
                    2 * len(cast(list[object], item["candidates"])) for item in branches
                )
            row = NativeCrossAxisVerificationBatchRowIR(
                plan_hash=evidence_plan.stable_hash(),
                repeat_index=args.repeat_index,
                original_clause_index=ordinal,
                mode=mode,
                order_position=position,
                queue_elapsed_ns=execution.trace.queue_elapsed_ns,
                accepted_nodes=len(execution.queue.trace.evaluations),
                sibling_group_count=len(execution.batch_commits) - 1,
                scorer_launch_count=len(node_widths),
                scorer_node_widths=node_widths,
                scorer_child_domain_widths=child_widths,
                queue_semantic_hash=_canonical_hash(queue),
                branch_semantic_hash=_canonical_hash(branches),
                state_semantic_hash=_canonical_hash(states),
                refinement_semantic_hash=_canonical_hash(refinements),
                cross_batch_trace_hashes=tuple(
                    cast(str, item["trace_hash"]) for item in cross_batches
                ),
            )
            row.validate(plan=evidence_plan)
            raw: dict[str, Any] = {
                "row": row.to_dict(plan=evidence_plan),
                "row_hash": row.stable_hash(plan=evidence_plan),
                "queue_semantics": queue,
                "branch_semantics": branches,
                "selected_states": states,
                "refinements": refinements,
                "capsules": capsules,
                "cross_batches": cross_batches,
            }
            raw["raw_hash"] = _canonical_hash(raw)
            raw_runs.append(raw)
            by_mode[mode] = raw
        control = by_mode["nrir42"]
        candidate = by_mode["cross_axis"]
        parity = NativeCrossAxisVerificationBatchParityIR(
            plan_hash=evidence_plan.stable_hash(),
            repeat_index=args.repeat_index,
            original_clause_index=ordinal,
            queue_exact=control["queue_semantics"] == candidate["queue_semantics"],
            branch_exact=control["branch_semantics"] == candidate["branch_semantics"],
            state_exact=control["selected_states"] == candidate["selected_states"],
            refinement_exact=control["refinements"] == candidate["refinements"],
            all_exact=all(
                control[key] == candidate[key]
                for key in (
                    "queue_semantics",
                    "branch_semantics",
                    "selected_states",
                    "refinements",
                )
            ),
            nrir42_raw_hash=control["raw_hash"],
            cross_axis_raw_hash=candidate["raw_hash"],
        )
        parity.validate(plan=evidence_plan)
        if not parity.all_exact:
            raise ValueError("NRIR-43 cross-axis semantics differ")
        parities.append(parity.to_dict())
    worker: dict[str, Any] = {
        "schema_version": WORKER_SCHEMA_VERSION,
        "repeat_index": args.repeat_index,
        "plan_hash": evidence_plan.stable_hash(),
        "floor": {
            "selected_original_clause_indices": list(
                floor_decision.selected_original_clause_indices
            ),
            "semantic_signature_hash": floor.trace.semantic_signature_hash,
            "elapsed_ns": floor.trace.elapsed_ns,
        },
        "raw_runs": raw_runs,
        "parities": parities,
        "performance_claimed": False,
    }
    worker["worker_hash"] = _canonical_hash(worker)
    validate_worker(worker, plan=evidence_plan, repeat_index=args.repeat_index)
    _write_json(args.result_json, worker)
    print(
        _canonical_json(
            {
                "repeat_index": args.repeat_index,
                "queue_elapsed_ns": {
                    f"c{raw['row']['original_clause_index']}:{raw['row']['mode']}": raw[
                        "row"
                    ]["queue_elapsed_ns"]
                    for raw in raw_runs
                },
                "worker_hash": worker["worker_hash"],
            }
        ),
        flush=True,
    )


def validate_worker(
    value: Mapping[str, Any],
    *,
    plan: NativeCrossAxisVerificationBatchEvidencePlanIR,
    repeat_index: int,
) -> None:
    raw_runs = value.get("raw_runs")
    parities = value.get("parities")
    floor = value.get("floor")
    if (
        value.get("schema_version") != WORKER_SCHEMA_VERSION
        or value.get("repeat_index") != repeat_index
        or value.get("plan_hash") != plan.stable_hash()
        or not isinstance(raw_runs, list)
        or len(raw_runs) != 4
        or not isinstance(parities, list)
        or len(parities) != 2
        or not isinstance(floor, dict)
        or floor.get("selected_original_clause_indices") != list(plan.clause_ordinals)
        or value.get("performance_claimed") is not False
    ):
        raise ValueError("NRIR-43 worker envelope differs")
    by_key: dict[tuple[int, str], Mapping[str, Any]] = {}
    for raw in raw_runs:
        if not isinstance(raw, dict):
            raise TypeError("NRIR-43 raw run must be an object")
        semantic = {key: item for key, item in raw.items() if key != "raw_hash"}
        if raw.get("raw_hash") != _canonical_hash(semantic):
            raise ValueError("NRIR-43 raw hash differs")
        row = _row_from_dict(raw["row"])
        row.validate(plan=plan)
        if (
            raw.get("row_hash") != row.stable_hash(plan=plan)
            or row.repeat_index != repeat_index
            or row.queue_semantic_hash != _canonical_hash(raw["queue_semantics"])
            or row.branch_semantic_hash != _canonical_hash(raw["branch_semantics"])
            or row.state_semantic_hash != _canonical_hash(raw["selected_states"])
            or row.refinement_semantic_hash != _canonical_hash(raw["refinements"])
        ):
            raise ValueError("NRIR-43 row semantic binding differs")
        branches = raw["branch_semantics"]
        capsules = raw["capsules"]
        if len(branches) != plan.required_nodes or len(capsules) != len(branches):
            raise ValueError("NRIR-43 capsule coverage differs")
        for capsule, branch in zip(capsules, branches):
            _validate_serialized_capsule(capsule, branch)
        batches = raw["cross_batches"]
        if row.mode == "cross_axis":
            for batch in batches:
                _validate_cross_batch(batch)
            if row.cross_batch_trace_hashes != tuple(
                item["trace_hash"] for item in batches
            ):
                raise ValueError("NRIR-43 batch trace ownership differs")
        elif batches:
            raise ValueError("NRIR-43 baseline unexpectedly has cross batches")
        by_key[(row.original_clause_index, row.mode)] = raw
    for ordinal, parity_value in zip(plan.clause_ordinals, parities):
        parity = _parity_from_dict(parity_value)
        parity.validate(plan=plan)
        control = by_key[(ordinal, "nrir42")]
        candidate = by_key[(ordinal, "cross_axis")]
        exacts = tuple(
            control[left] == candidate[left]
            for left in (
                "queue_semantics",
                "branch_semantics",
                "selected_states",
                "refinements",
            )
        )
        if (
            exacts
            != (
                parity.queue_exact,
                parity.branch_exact,
                parity.state_exact,
                parity.refinement_exact,
            )
            or parity.nrir42_raw_hash != control["raw_hash"]
            or parity.cross_axis_raw_hash != candidate["raw_hash"]
        ):
            raise ValueError("NRIR-43 parity evidence differs")
    semantic_worker = {key: item for key, item in value.items() if key != "worker_hash"}
    if value.get("worker_hash") != _canonical_hash(semantic_worker):
        raise ValueError("NRIR-43 worker hash differs")


def _median_mad(values: list[int]) -> tuple[int, int]:
    median = int(statistics.median(values))
    mad = int(statistics.median(abs(value - median) for value in values))
    return median, mad


def _build_formal(
    workers: Sequence[Mapping[str, Any]],
    *,
    plan: NativeCrossAxisVerificationBatchEvidencePlanIR,
    workload: Mapping[str, Any],
) -> dict[str, Any]:
    rows = [raw["row"] for worker in workers for raw in worker["raw_runs"]]
    parities = [item for worker in workers for item in worker["parities"]]
    metrics: list[NativeCrossAxisVerificationBatchClauseMetricIR] = []
    for ordinal in plan.clause_ordinals:
        control = [
            row["queue_elapsed_ns"]
            for row in rows
            if row["original_clause_index"] == ordinal and row["mode"] == "nrir42"
        ]
        candidate = [
            row["queue_elapsed_ns"]
            for row in rows
            if row["original_clause_index"] == ordinal and row["mode"] == "cross_axis"
        ]
        control_median, control_mad = _median_mad(control)
        candidate_median, candidate_mad = _median_mad(candidate)
        ratio = candidate_median / control_median
        improvement = control_median - candidate_median
        exceeds_mad = improvement > max(control_mad, candidate_mad)
        metric = NativeCrossAxisVerificationBatchClauseMetricIR(
            original_clause_index=ordinal,
            nrir42_median_ns=control_median,
            cross_axis_median_ns=candidate_median,
            nrir42_mad_ns=control_mad,
            cross_axis_mad_ns=candidate_mad,
            median_ratio=ratio,
            median_improvement_ns=improvement,
            improvement_exceeds_pooled_mad=exceeds_mad,
            timing_gate_passed=(
                ratio <= plan.maximum_queue_median_ratio and exceeds_mad
            ),
        )
        metric.validate(plan=plan)
        metrics.append(metric)
    parity_passed = all(item["all_exact"] for item in parities)
    launch_gate_passed = all(
        row["scorer_launch_count"]
        == (
            plan.nrir42_scorer_launches_per_clause
            if row["mode"] == "nrir42"
            else plan.cross_axis_scorer_launches_per_clause
        )
        for row in rows
    )
    timing_gate_passed = all(item.timing_gate_passed for item in metrics)
    phase_a_go = parity_passed and launch_gate_passed and timing_gate_passed
    decision = NativeCrossAxisVerificationBatchDecisionIR(
        plan_hash=plan.stable_hash(),
        clause_metrics=tuple(metrics),
        parity_passed=parity_passed,
        launch_gate_passed=launch_gate_passed,
        timing_gate_passed=timing_gate_passed,
        phase_a_go=phase_a_go,
        next_route=("run_phase_b" if phase_a_go else "stop_cross_axis_batching"),
        reason=(
            "exact semantics, launch, and paired timing gates passed"
            if phase_a_go
            else "Phase-A cross-axis batching gate failed; Phase B is gated off"
        ),
    )
    decision.validate(plan=plan)
    formal: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "native_code_revision": _code_revision(),
            "source_phase_a_hash": plan.source_phase_a_hash,
            "source_phase_b_hash": plan.source_phase_b_hash,
            "workload": _public_workload(workload),
        },
        "plan": plan.to_dict(),
        "plan_hash": plan.stable_hash(),
        "rows": rows,
        "parities": parities,
        "metrics": [item.to_dict() for item in metrics],
        "decision": decision.to_dict(),
        "decision_hash": decision.stable_hash(),
        "workers": [
            {
                "repeat_index": worker["repeat_index"],
                "worker_hash": worker["worker_hash"],
            }
            for worker in workers
        ],
        "status": "validated-reduced" if phase_a_go else "validated-no-go",
        "performance_claimed": False,
    }
    formal["formal_hash"] = _canonical_hash(formal)
    return formal


def validate_formal(value: Mapping[str, Any]) -> None:
    plan = _plan_from_dict(value["plan"])
    plan.validate()
    rows = value.get("rows")
    parities = value.get("parities")
    metrics = value.get("metrics")
    if (
        not isinstance(rows, list)
        or not isinstance(parities, list)
        or not isinstance(metrics, list)
    ):
        raise TypeError("NRIR-43 formal evidence collections differ")
    decision = _decision_from_dict(value["decision"])
    for row_value in rows:
        _row_from_dict(row_value).validate(plan=plan)
    for parity_value in parities:
        _parity_from_dict(parity_value).validate(plan=plan)
    for metric_value in metrics:
        _metric_from_dict(metric_value).validate(plan=plan)
    decision.validate(plan=plan)
    semantic = {key: item for key, item in value.items() if key != "formal_hash"}
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("plan_hash") != plan.stable_hash()
        or value.get("source", {}).get("native_code_revision") != _code_revision()
        or len(rows) != 12
        or len(parities) != 6
        or len(metrics) != 2
        or value.get("decision_hash") != decision.stable_hash()
        or value.get("status")
        != ("validated-reduced" if decision.phase_a_go else "validated-no-go")
        or value.get("performance_claimed") is not False
        or value.get("formal_hash") != _canonical_hash(semantic)
    ):
        raise ValueError("NRIR-43 formal envelope differs")


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
            f"NRIR-43 worker failed with exit {completed.returncode}: {log_path}"
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


def _generate(args: argparse.Namespace) -> None:
    if args.torch_threads != TORCH_THREADS:
        raise ValueError("NRIR-43 torch thread count differs")
    plan = _plan()
    workloads = _resolve_workloads(args.benchmark_root.resolve())
    workload = next(
        item for item in workloads if item["workload_id"] == "cifar10_resnet:000"
    )
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
        validate_worker(worker, plan=plan, repeat_index=repeat_index)
        workers.append(worker)
    formal = _build_formal(workers, plan=plan, workload=workload)
    validate_formal(formal)
    _write_json(artifact_dir / "formal.json", formal)
    manifest = _manifest(artifact_dir, formal)
    _write_json(artifact_dir / "manifest.json", manifest)
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
        raise ValueError("NRIR-43 torch thread count differs")
    plan = _plan()
    workloads = _resolve_workloads(args.benchmark_root.resolve())
    workload = next(
        item for item in workloads if item["workload_id"] == "cifar10_resnet:000"
    )
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
        raise ValueError("NRIR-43 manifest envelope differs")
    for relative, expected in manifest["files"].items():
        if _file_sha256(artifact_dir / relative) != expected:
            raise ValueError(f"NRIR-43 artifact digest differs: {relative}")
    workers: list[dict[str, Any]] = []
    for repeat_index in range(REPEAT_COUNT):
        worker = _load_json(artifact_dir / "shards" / f"repeat-{repeat_index}.json")
        validate_worker(worker, plan=plan, repeat_index=repeat_index)
        workers.append(worker)
    rebuilt = _build_formal(workers, plan=plan, workload=workload)
    formal = _load_json(artifact_dir / "formal.json")
    validate_formal(formal)
    if rebuilt != formal or manifest["formal_hash"] != formal["formal_hash"]:
        raise ValueError("NRIR-43 formal replay differs")
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
