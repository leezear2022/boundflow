#!/usr/bin/env python3
"""Generate or replay NRIR-41 objective-branch production cost attribution."""

# pylint: disable=too-many-lines,too-many-locals,too-many-statements
# pylint: disable=too-many-branches,too-many-boolean-expressions
# pylint: disable=missing-function-docstring,import-outside-toplevel,duplicate-code

from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
from pathlib import Path
import pstats
import statistics
import subprocess
import sys
import tempfile
from typing import Any, Mapping

from boundflow.ir.objective_branch_cost_attribution import (
    NativeObjectiveBranchCostAttributionPlanIR,
    NativeObjectiveBranchCostDecisionIR,
    NativeObjectiveBranchPrefixAttributionIR,
    NativeObjectiveBranchProfilePhaseIR,
    NativeObjectiveBranchWallAttributionIR,
    lower_native_objective_branch_cost_schedule,
)
from boundflow.runtime.native_objective_branch_cost_attribution import (
    compile_native_objective_branch_cost_plan,
    derive_native_objective_branch_cost_decision,
    native_objective_branch_profile_rows,
    native_objective_branch_wall_row,
    reconstruct_native_objective_branch_prefixes,
)
from boundflow.runtime.native_objective_branch_score import (
    NativeObjectiveBranchPolicy,
)
from scripts.run_objective_branch_shared_evaluator_pilot import (
    _execute_floor,
    _source,
    validate_pilot,
)
from scripts.run_objective_branch_whole_query_formal import (
    validate_formal as validate_nrir40_formal,
)
from scripts.run_typed_hard_clause_escalation_artifact import (
    _load_query_runtime,
    _policies,
    _public_workload,
    _resolve_workloads,
)

FORMAL_SCHEMA_VERSION = "boundflow.objective-branch-cost-attribution-formal/v1"
WORKER_SCHEMA_VERSION = "boundflow.objective-branch-cost-attribution-worker/v1"
PROFILE_SCHEMA_VERSION = "boundflow.objective-branch-cost-attribution-profile/v1"
MANIFEST_SCHEMA_VERSION = "boundflow.objective-branch-cost-attribution-manifest/v1"
ARTIFACT_DIR = Path(
    "artifacts/objective-branch-production-cost-attribution/"
    "vnncomp21-resnet2b-property0-three-repeat-cpu-v1"
)
SOURCE_PILOT = Path(
    "artifacts/objective-branch-shared-evaluator/"
    "vnncomp21-resnet2b-property0-cpu-pilot-v1/pilot.json"
)
SOURCE_FORMAL = Path(
    "artifacts/objective-branch-whole-query/"
    "vnncomp21-resnet2b-property0-three-repeat-cpu-formal-v1/formal.json"
)
TORCH_THREADS = 8
REPEAT_COUNT = 3
WORKER_TIMEOUT_SECONDS = 240
EXPECTED_CLAUSES = (2, 3)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("generate", "replay"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--benchmark-root", type=Path, required=True)
        subparser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
        subparser.add_argument("--torch-threads", type=int, default=TORCH_THREADS)
    for command in ("worker", "profile-worker"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--model", type=Path, required=True)
        subparser.add_argument("--property", type=Path, required=True)
        subparser.add_argument("--result-json", type=Path, required=True)
        subparser.add_argument("--torch-threads", type=int, required=True)
        if command == "worker":
            subparser.add_argument("--repeat-index", type=int, required=True)
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
        "boundflow/ir/objective_branch_cost_attribution.py",
        "boundflow/runtime/native_objective_branch_cost_attribution.py",
        "scripts/run_objective_branch_production_cost_attribution.py",
    )
    return _canonical_hash({path: _file_sha256(root / path) for path in paths})


def _source_artifacts() -> tuple[dict[str, Any], dict[str, Any]]:
    root = _repo_root()
    pilot = _load_json(root / SOURCE_PILOT)
    formal = _load_json(root / SOURCE_FORMAL)
    validate_pilot(pilot)
    validate_nrir40_formal(formal)
    return pilot, formal


def _plan() -> NativeObjectiveBranchCostAttributionPlanIR:
    pilot, formal = _source_artifacts()
    return compile_native_objective_branch_cost_plan(
        plan_id="nrir41:cifar10_resnet:000:property0",
        source_pilot_hash=pilot["pilot_hash"],
        source_formal_hash=formal["formal_payload_hash"],
    )


def _plan_from_dict(
    value: Mapping[str, Any],
) -> NativeObjectiveBranchCostAttributionPlanIR:
    return NativeObjectiveBranchCostAttributionPlanIR(
        plan_id=value["plan_id"],
        source_pilot_hash=value["source_pilot_hash"],
        source_formal_hash=value["source_formal_hash"],
        clause_ordinals=tuple(value["clause_ordinals"]),
        prefix_node_counts=tuple(value["prefix_node_counts"]),
        paired_orders=tuple(tuple(item) for item in value["paired_orders"]),
        required_nodes=value["required_nodes"],
        required_sibling_groups=value["required_sibling_groups"],
        minimum_frontier_improvement=value["minimum_frontier_improvement"],
        minimum_queue_ratio=value["minimum_queue_ratio"],
        minimum_branch_program_share=value["minimum_branch_program_share"],
        torch_threads=value["torch_threads"],
        candidate_policy_id=value["candidate_policy_id"],
        control_policy_id=value["control_policy_id"],
        semantics_owner=value["semantics_owner"],
        performance_claimed=value["performance_claimed"],
        schema_version=value["schema_version"],
    )


def _prefix_from_dict(
    value: Mapping[str, Any],
) -> NativeObjectiveBranchPrefixAttributionIR:
    return NativeObjectiveBranchPrefixAttributionIR(
        plan_hash=value["plan_hash"],
        original_clause_index=value["original_clause_index"],
        policy_id=value["policy_id"],
        accepted_nodes=value["accepted_nodes"],
        active_node_ids=tuple(value["active_node_ids"]),
        active_evaluation_hashes=tuple(value["active_evaluation_hashes"]),
        active_count=value["active_count"],
        worst_active_lower=value["worst_active_lower"],
        median_active_lower=value["median_active_lower"],
        source_rows_hash=value["source_rows_hash"],
    )


def _wall_from_dict(value: Mapping[str, Any]) -> NativeObjectiveBranchWallAttributionIR:
    return NativeObjectiveBranchWallAttributionIR(**value)


def _profile_from_dict(
    value: Mapping[str, Any],
) -> NativeObjectiveBranchProfilePhaseIR:
    return NativeObjectiveBranchProfilePhaseIR(**value)


def _decision_from_dict(
    value: Mapping[str, Any],
) -> NativeObjectiveBranchCostDecisionIR:
    return NativeObjectiveBranchCostDecisionIR(
        plan_hash=value["plan_hash"],
        frontier_improvements=tuple(value["frontier_improvements"].items()),
        queue_ratios=tuple(value["queue_ratios"].items()),
        branch_program_shares=tuple(value["branch_program_shares"].items()),
        frontier_order_retained=value["frontier_order_retained"],
        scoring_cost_dominant=value["scoring_cost_dominant"],
        next_route=value["next_route"],
        reason=value["reason"],
    )


def _expected_final(
    prefixes: tuple[NativeObjectiveBranchPrefixAttributionIR, ...],
) -> dict[tuple[int, str], NativeObjectiveBranchPrefixAttributionIR]:
    return {
        (item.original_clause_index, item.policy_id): item
        for item in prefixes
        if item.accepted_nodes == 31
    }


def _execute_pair(args: argparse.Namespace) -> None:
    import torch

    from boundflow.runtime.native_objective_branch_shared_evaluator import (
        compile_native_objective_branch_shared_plan,
    )
    from boundflow.runtime.native_objective_branch_shared_production_queue import (
        execute_native_objective_branch_shared_production_queue,
    )
    from boundflow.runtime.native_parametric_optimizer import (
        NativeParametricOptimizerTemplateCache,
    )
    from boundflow.runtime.native_shared_parametric_ancestral import (
        execute_native_shared_parametric_ancestral_queue,
    )

    torch.set_num_threads(args.torch_threads)
    plan = _plan()
    pilot, _formal = _source_artifacts()
    prefixes = reconstruct_native_objective_branch_prefixes(plan, pilot["clauses"])
    expected = _expected_final(prefixes)
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
        query_id=f"nrir41:repeat{args.repeat_index}:floor",
    )
    if floor_decision.selected_original_clause_indices != EXPECTED_CLAUSES:
        raise RuntimeError("NRIR-41 floor selection differs")
    rows: list[NativeObjectiveBranchWallAttributionIR] = []
    order = plan.paired_orders[args.repeat_index]
    for ordinal in plan.clause_ordinals:
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
            plan_id=f"nrir41:repeat{args.repeat_index}:clause:{ordinal}",
        )
        for position, mode in enumerate(order):
            cache = NativeParametricOptimizerTemplateCache()
            if mode == "widest":
                execution = execute_native_shared_parametric_ancestral_queue(
                    composite.shared_plan,
                    module,
                    input_spec,
                    linear_spec_C=objective,
                    threshold=threshold,
                    root_refinement=source.refinement,
                    optimizer_policy=optimizer_policy,
                    compiler_cache=cache,
                    query_id=f"nrir41:r{args.repeat_index}:c{ordinal}:widest",
                )
                policy_id = plan.control_policy_id
                branch_count = 0
            else:
                execution = execute_native_objective_branch_shared_production_queue(
                    composite,
                    module,
                    input_spec,
                    linear_spec_C=objective,
                    threshold=threshold,
                    root_refinement=source.refinement,
                    optimizer_policy=optimizer_policy,
                    branch_policy=branch_policy,
                    compiler_cache=cache,
                    query_id=f"nrir41:r{args.repeat_index}:c{ordinal}:objective",
                )
                policy_id = plan.candidate_policy_id
                branch_count = len(execution.queue.objective_branch_executions)
            row = native_objective_branch_wall_row(
                plan,
                execution,
                repeat_index=args.repeat_index,
                original_clause_index=ordinal,
                policy_id=policy_id,
                order_position=position,
                branch_execution_count=branch_count,
            )
            frozen = expected[(ordinal, policy_id)]
            if (
                row.root_lower != execution.queue.trace.evaluations[0].lower
                or abs(row.worst_active_lower - frozen.worst_active_lower) > 1e-5
                or abs(row.median_active_lower - frozen.median_active_lower) > 1e-5
            ):
                raise ValueError("NRIR-41 paired execution differs from frozen source")
            rows.append(row)
    payload = {
        "schema_version": WORKER_SCHEMA_VERSION,
        "repeat_index": args.repeat_index,
        "paired_order": list(order),
        "floor_trace_hash": floor.trace.semantic_signature_hash,
        "wall_rows": [item.to_dict() for item in rows],
        "performance_claimed": False,
    }
    payload["worker_hash"] = _canonical_hash(payload)
    _validate_worker(payload, plan)
    _write_json(args.result_json.resolve(), payload)
    print(
        _canonical_json(
            {
                "repeat": args.repeat_index,
                "order": list(order),
                "queue_seconds": [item.queue_elapsed_ns / 1e9 for item in rows],
                "worker_hash": payload["worker_hash"],
            }
        )
    )


def _execute_profile(args: argparse.Namespace) -> None:
    import torch

    from boundflow.runtime.native_objective_branch_shared_evaluator import (
        compile_native_objective_branch_shared_plan,
    )
    from boundflow.runtime.native_objective_branch_shared_production_queue import (
        execute_native_objective_branch_shared_production_queue,
    )
    from boundflow.runtime.native_parametric_optimizer import (
        NativeParametricOptimizerTemplateCache,
    )

    torch.set_num_threads(args.torch_threads)
    plan = _plan()
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
        query_id="nrir41:profile:floor",
    )
    if floor_decision.selected_original_clause_indices != EXPECTED_CLAUSES:
        raise RuntimeError("NRIR-41 profile floor selection differs")
    profiles: list[NativeObjectiveBranchProfilePhaseIR] = []
    for ordinal in plan.clause_ordinals:
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
            plan_id=f"nrir41:profile:clause:{ordinal}",
        )
        profiler = cProfile.Profile()
        profiler.enable()
        execution = execute_native_objective_branch_shared_production_queue(
            composite,
            module,
            input_spec,
            linear_spec_C=objective,
            threshold=threshold,
            root_refinement=source.refinement,
            optimizer_policy=optimizer_policy,
            branch_policy=branch_policy,
            compiler_cache=NativeParametricOptimizerTemplateCache(),
            query_id=f"nrir41:profile:clause:{ordinal}:objective",
        )
        profiler.disable()
        if len(execution.queue.trace.evaluations) != plan.required_nodes:
            raise RuntimeError("NRIR-41 profiled queue coverage differs")
        stats: Any = getattr(pstats.Stats(profiler), "stats")
        profiles.extend(
            native_objective_branch_profile_rows(
                plan,
                original_clause_index=ordinal,
                stats=stats,
                profile_queue_elapsed_ns=execution.trace.queue_elapsed_ns,
            )
        )
    payload = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "profile_rows": [item.to_dict() for item in profiles],
        "performance_claimed": False,
    }
    payload["profile_hash"] = _canonical_hash(payload)
    _validate_profile(payload, plan)
    _write_json(args.result_json.resolve(), payload)
    print(
        _canonical_json(
            {
                "branch_program_shares": {
                    f"clause_{item.original_clause_index}": item.cumulative_ns
                    / item.profile_queue_elapsed_ns
                    for item in profiles
                    if item.phase_id == "branch_program"
                },
                "profile_hash": payload["profile_hash"],
            }
        )
    )


def _validate_worker(
    value: Mapping[str, Any], plan: NativeObjectiveBranchCostAttributionPlanIR
) -> None:
    rows = value.get("wall_rows")
    if not isinstance(rows, list):
        raise TypeError("NRIR-41 worker wall rows differ")
    typed = tuple(_wall_from_dict(item) for item in rows)
    repeat = value.get("repeat_index")
    if not isinstance(repeat, int) or repeat not in range(REPEAT_COUNT):
        raise ValueError("NRIR-41 worker repeat differs")
    if (
        value.get("schema_version") != WORKER_SCHEMA_VERSION
        or value.get("paired_order") != list(plan.paired_orders[repeat])
        or len(typed) != 4
        or {(item.original_clause_index, item.policy_id) for item in typed}
        != {
            (ordinal, policy)
            for ordinal in plan.clause_ordinals
            for policy in (plan.control_policy_id, plan.candidate_policy_id)
        }
        or any(item.repeat_index != repeat for item in typed)
        or value.get("performance_claimed") is not False
    ):
        raise ValueError("NRIR-41 worker envelope differs")
    semantic = {key: value[key] for key in value if key != "worker_hash"}
    if value.get("worker_hash") != _canonical_hash(semantic):
        raise ValueError("NRIR-41 worker hash differs")


def _validate_profile(
    value: Mapping[str, Any], plan: NativeObjectiveBranchCostAttributionPlanIR
) -> None:
    rows = value.get("profile_rows")
    if not isinstance(rows, list):
        raise TypeError("NRIR-41 profile rows differ")
    typed = tuple(_profile_from_dict(item) for item in rows)
    if (
        value.get("schema_version") != PROFILE_SCHEMA_VERSION
        or len(typed) != 8
        or {(item.original_clause_index, item.phase_id) for item in typed}
        != {
            (ordinal, phase)
            for ordinal in plan.clause_ordinals
            for phase in (
                "branch_program",
                "enumerate_candidates",
                "materialize_children",
                "evaluate_child_bounds",
            )
        }
        or value.get("performance_claimed") is not False
    ):
        raise ValueError("NRIR-41 profile envelope differs")
    semantic = {key: value[key] for key in value if key != "profile_hash"}
    if value.get("profile_hash") != _canonical_hash(semantic):
        raise ValueError("NRIR-41 profile hash differs")


def _formal_semantic(value: Mapping[str, Any]) -> dict[str, object]:
    return {
        key: value[key]
        for key in (
            "protocol",
            "workload",
            "source",
            "plan",
            "prefix_rows",
            "workers",
            "profile",
            "wall_summary",
            "decision",
            "task_ir",
            "schedule",
            "claim_boundary",
            "performance_claimed",
        )
    }


def _wall_summary(
    walls: tuple[NativeObjectiveBranchWallAttributionIR, ...],
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for ordinal in EXPECTED_CLAUSES:
        for policy_id, label in (
            ("widest_unsplit_ambiguous_relu", "widest"),
            ("objective_bound_impact", "objective"),
        ):
            values = [
                item.queue_elapsed_ns
                for item in walls
                if item.original_clause_index == ordinal and item.policy_id == policy_id
            ]
            if len(values) != REPEAT_COUNT:
                raise ValueError("NRIR-41 wall summary coverage differs")
            median_ns = int(statistics.median(values))
            mad_ns = int(statistics.median(abs(value - median_ns) for value in values))
            result[f"clause_{ordinal}_{label}"] = {
                "median_queue_elapsed_ns": median_ns,
                "mad_queue_elapsed_ns": mad_ns,
            }
    return result


def validate_attribution(value: Mapping[str, Any]) -> None:
    plan_raw = value.get("plan")
    if not isinstance(plan_raw, dict):
        raise TypeError("NRIR-41 formal Plan differs")
    plan = _plan_from_dict(plan_raw)
    plan.validate()
    workers = value.get("workers")
    profile_raw = value.get("profile")
    prefix_raw = value.get("prefix_rows")
    if (
        value.get("schema_version") != FORMAL_SCHEMA_VERSION
        or value.get("status") != "validated-reduced"
        or value.get("source", {}).get("native_code_revision") != _code_revision()
        or not isinstance(workers, list)
        or len(workers) != REPEAT_COUNT
        or not isinstance(profile_raw, dict)
        or not isinstance(prefix_raw, list)
        or value.get("claim_boundary") != "internal_causal_attribution_only"
        or value.get("performance_claimed") is not False
    ):
        raise ValueError("NRIR-41 formal envelope differs")
    for worker in workers:
        _validate_worker(worker, plan)
    _validate_profile(profile_raw, plan)
    pilot, formal40 = _source_artifacts()
    if (
        plan != _plan()
        or value["source"]["source_pilot_hash"] != pilot["pilot_hash"]
        or value["source"]["source_formal_hash"] != formal40["formal_payload_hash"]
    ):
        raise ValueError("NRIR-41 frozen source identity differs")
    prefixes = tuple(_prefix_from_dict(item) for item in prefix_raw)
    expected_prefixes = reconstruct_native_objective_branch_prefixes(
        plan, pilot["clauses"]
    )
    walls = tuple(
        _wall_from_dict(item) for worker in workers for item in worker["wall_rows"]
    )
    profiles = tuple(_profile_from_dict(item) for item in profile_raw["profile_rows"])
    decision = derive_native_objective_branch_cost_decision(
        plan, prefixes, walls, profiles
    )
    task_ir, schedule = lower_native_objective_branch_cost_schedule(
        plan, prefixes, walls, profiles, decision
    )
    if (
        prefixes != expected_prefixes
        or value.get("wall_summary") != _wall_summary(walls)
        or value.get("decision") != decision.to_dict()
        or value.get("task_ir") != task_ir.to_dict()
        or value.get("schedule") != schedule.to_dict(task_ir)
        or value.get("formal_hash") != _canonical_hash(_formal_semantic(value))
    ):
        raise ValueError("NRIR-41 formal derived evidence differs")


def _subprocess_command(
    command: str,
    workload: Mapping[str, object],
    result_path: Path,
    threads: int,
    repeat: int | None = None,
) -> list[str]:
    result = [
        sys.executable,
        str(Path(__file__).resolve()),
        command,
        "--model",
        str(workload["model"]),
        "--property",
        str(workload["property"]),
        "--result-json",
        str(result_path),
        "--torch-threads",
        str(threads),
    ]
    if repeat is not None:
        result.extend(("--repeat-index", str(repeat)))
    return result


def _run_subprocess(command: list[str], log_path: Path) -> None:
    completed = subprocess.run(
        command,
        cwd=_repo_root(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=WORKER_TIMEOUT_SECONDS,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"NRIR-41 subprocess failed with {completed.returncode}: "
            f"{completed.stdout[-12000:]}"
        )
    print(completed.stdout.strip())


def _generate(args: argparse.Namespace) -> None:
    workloads = _resolve_workloads(args.benchmark_root.resolve())
    workload = next(
        item for item in workloads if item["workload_id"] == "cifar10_resnet:000"
    )
    plan = _plan()
    pilot, formal40 = _source_artifacts()
    prefixes = reconstruct_native_objective_branch_prefixes(plan, pilot["clauses"])
    artifact_dir = args.artifact_dir.resolve()
    files: dict[str, str] = {}
    workers: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="boundflow-nrir41-") as temporary:
        temporary_root = Path(temporary)
        for repeat in range(REPEAT_COUNT):
            result_path = temporary_root / f"repeat-{repeat}.json"
            log_path = artifact_dir / "logs" / f"repeat-{repeat}.log"
            _run_subprocess(
                _subprocess_command(
                    "worker", workload, result_path, args.torch_threads, repeat
                ),
                log_path,
            )
            worker = _load_json(result_path)
            _validate_worker(worker, plan)
            shard_path = artifact_dir / "shards" / f"repeat-{repeat}.json"
            _write_json(shard_path, worker)
            workers.append(worker)
            files[str(log_path.relative_to(artifact_dir))] = _file_sha256(log_path)
            files[str(shard_path.relative_to(artifact_dir))] = _file_sha256(shard_path)
        profile_path = temporary_root / "profile.json"
        profile_log = artifact_dir / "logs/profile.log"
        _run_subprocess(
            _subprocess_command(
                "profile-worker", workload, profile_path, args.torch_threads
            ),
            profile_log,
        )
        profile = _load_json(profile_path)
        _validate_profile(profile, plan)
        frozen_profile_path = artifact_dir / "profile.json"
        _write_json(frozen_profile_path, profile)
        files[str(profile_log.relative_to(artifact_dir))] = _file_sha256(profile_log)
        files[str(frozen_profile_path.relative_to(artifact_dir))] = _file_sha256(
            frozen_profile_path
        )
    walls = tuple(
        _wall_from_dict(item) for worker in workers for item in worker["wall_rows"]
    )
    profiles = tuple(_profile_from_dict(item) for item in profile["profile_rows"])
    decision = derive_native_objective_branch_cost_decision(
        plan, prefixes, walls, profiles
    )
    task_ir, schedule = lower_native_objective_branch_cost_schedule(
        plan, prefixes, walls, profiles, decision
    )
    formal: dict[str, Any] = {
        "schema_version": FORMAL_SCHEMA_VERSION,
        "status": "validated-reduced",
        "protocol": {
            "fresh_paired_repeats": REPEAT_COUNT,
            "paired_orders": [list(value) for value in plan.paired_orders],
            "profiled_runs_excluded_from_wall_medians": True,
            "torch_threads": args.torch_threads,
        },
        "workload": _public_workload(workload),
        "source": {
            "native_code_revision": _code_revision(),
            "source_pilot_hash": pilot["pilot_hash"],
            "source_formal_hash": formal40["formal_payload_hash"],
        },
        "plan": plan.to_dict(),
        "prefix_rows": [item.to_dict() for item in prefixes],
        "workers": workers,
        "profile": profile,
        "wall_summary": _wall_summary(walls),
        "decision": decision.to_dict(),
        "task_ir": task_ir.to_dict(),
        "schedule": schedule.to_dict(task_ir),
        "claim_boundary": "internal_causal_attribution_only",
        "performance_claimed": False,
    }
    formal["formal_hash"] = _canonical_hash(_formal_semantic(formal))
    validate_attribution(formal)
    formal_path = artifact_dir / "formal.json"
    _write_json(formal_path, formal)
    files["formal.json"] = _file_sha256(formal_path)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "files": files,
        "formal_hash": _canonical_hash(formal),
        "performance_claimed": False,
    }
    _write_json(artifact_dir / "manifest.json", manifest)
    print(
        _canonical_json(
            {
                "formal_hash": formal["formal_hash"],
                "frontier_retained": decision.frontier_order_retained,
                "scoring_cost_dominant": decision.scoring_cost_dominant,
                "queue_ratios": dict(decision.queue_ratios),
                "branch_program_shares": dict(decision.branch_program_shares),
                "next_route": decision.next_route,
            }
        )
    )


def _replay(args: argparse.Namespace) -> None:
    artifact_dir = args.artifact_dir.resolve()
    formal = _load_json(artifact_dir / "formal.json")
    manifest = _load_json(artifact_dir / "manifest.json")
    validate_attribution(formal)
    workloads = _resolve_workloads(args.benchmark_root.resolve())
    workload = next(
        item for item in workloads if item["workload_id"] == "cifar10_resnet:000"
    )
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise TypeError("NRIR-41 manifest files differ")
    expected_files = {
        "formal.json",
        "profile.json",
        "logs/profile.log",
        *(f"logs/repeat-{index}.log" for index in range(REPEAT_COUNT)),
        *(f"shards/repeat-{index}.json" for index in range(REPEAT_COUNT)),
    }
    shards = [
        _load_json(artifact_dir / "shards" / f"repeat-{index}.json")
        for index in range(REPEAT_COUNT)
    ]
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or set(files) != expected_files
        or any(
            _file_sha256(artifact_dir / path) != digest
            for path, digest in files.items()
        )
        or manifest.get("formal_hash") != _canonical_hash(formal)
        or manifest.get("performance_claimed") is not False
        or formal["workload"] != _public_workload(workload)
        or shards != formal["workers"]
        or _load_json(artifact_dir / "profile.json") != formal["profile"]
    ):
        raise ValueError("NRIR-41 manifest/workload differs")
    print(
        _canonical_json(
            {
                "formal_hash": formal["formal_hash"],
                "next_route": formal["decision"]["next_route"],
                "performance_claimed": False,
            }
        )
    )


def main() -> None:
    args = _parse_args()
    if args.torch_threads < 1:
        raise ValueError("torch thread count must be positive")
    if args.command == "generate":
        _generate(args)
    elif args.command == "replay":
        _replay(args)
    elif args.command == "worker":
        _execute_pair(args)
    else:
        _execute_profile(args)


if __name__ == "__main__":
    main()
