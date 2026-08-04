#!/usr/bin/env python3
"""Generate or replay the preregistered NRIR-38 exact-frontier pilot."""

# pylint: disable=too-many-lines,too-many-locals,too-many-statements
# pylint: disable=too-many-boolean-expressions,missing-function-docstring
# pylint: disable=import-outside-toplevel,duplicate-code,too-many-branches

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any, Mapping

from boundflow.ir.frontier_tightness_attribution import (
    NativeFrontierCandidateNodeIR,
    NativeFrontierNodeAttributionIR,
    NativeFrontierTightnessAttributionPlanIR,
    NativeFrontierTightnessDecisionIR,
    lower_native_frontier_tightness_attribution_schedule,
)
from boundflow.runtime.native_frontier_tightness_attribution import (
    _decision,
    compile_native_frontier_tightness_attribution_plan,
    execute_native_frontier_tightness_attribution,
)
from scripts.run_typed_hard_clause_escalation_artifact import (
    _load_query_runtime,
    _policies,
    _public_workload,
    _resolve_workloads,
)

PILOT_SCHEMA_VERSION = "boundflow.full-frontier-tightness-attribution-pilot/v1"
MANIFEST_SCHEMA_VERSION = (
    "boundflow.full-frontier-tightness-attribution-pilot-manifest/v1"
)
ARTIFACT_DIR = Path(
    "artifacts/full-frontier-tightness-attribution/"
    "vnncomp21-resnet2b-property0-cpu-pilot-v1"
)
TORCH_THREADS = 8
EXPECTED_SELECTED = (2, 3)
EXPECTED_EVALUATIONS = 31
EXPECTED_ACTIVE_NODES = 16
EXPECTED_ACTIVE_DEPTH = 4


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("generate", "replay"))
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--torch-threads", type=int, default=TORCH_THREADS)
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
        "boundflow/ir/frontier_tightness_attribution.py",
        "boundflow/runtime/native_frontier_tightness_attribution.py",
        "boundflow/ir/shared_parametric_ancestral.py",
        "boundflow/runtime/native_shared_parametric_ancestral.py",
        "boundflow/runtime/native_shared_parametric_multi_clause_anytime.py",
        "scripts/run_full_frontier_tightness_attribution_pilot.py",
    )
    return _canonical_hash({path: _file_sha256(root / path) for path in paths})


def _plan_from_dict(
    value: Mapping[str, Any],
) -> NativeFrontierTightnessAttributionPlanIR:
    active = value["active_node_split_hashes"]
    if not isinstance(active, dict):
        raise TypeError("NRIR-38 active node map differs")
    return NativeFrontierTightnessAttributionPlanIR(
        plan_id=value["plan_id"],
        source_execution_hash=value["source_execution_hash"],
        source_plan_hash=value["source_plan_hash"],
        source_queue_trace_hash=value["source_queue_trace_hash"],
        objective_hash=value["objective_hash"],
        threshold_hash=value["threshold_hash"],
        original_clause_index=value["original_clause_index"],
        active_node_split_hashes=tuple(active.items()),
        baseline_optimizer_policy_hash=value["baseline_optimizer_policy_hash"],
        candidate_optimizer_policy_hash=value["candidate_optimizer_policy_hash"],
        baseline_optimizer_steps=value["baseline_optimizer_steps"],
        candidate_optimizer_steps=value["candidate_optimizer_steps"],
        required_active_depth=value["required_active_depth"],
        required_active_nodes=value["required_active_nodes"],
        lower_delta_tolerance=value["lower_delta_tolerance"],
        minimum_worst_lower_improvement=value["minimum_worst_lower_improvement"],
        minimum_improved_nodes=value["minimum_improved_nodes"],
        candidate_id=value["candidate_id"],
        frozen_variables=tuple(value["frozen_variables"]),
        semantics_owner=value["semantics_owner"],
        performance_claimed=value["performance_claimed"],
        schema_version=value["schema_version"],
    )


def _node_from_dict(value: Mapping[str, Any]) -> NativeFrontierNodeAttributionIR:
    return NativeFrontierNodeAttributionIR(
        node_id=value["node_id"],
        parent_node_id=value.get("parent_node_id"),
        split_state_hash=value["split_state_hash"],
        evaluation_hash=value["evaluation_hash"],
        depth=value["depth"],
        active=value["active"],
        lower=value["lower"],
        upper=value["upper"],
        proof_deficit=value["proof_deficit"],
        parent_lower_gain=value.get("parent_lower_gain"),
        refinement_plan_hash=value["refinement_plan_hash"],
        refinement_semantic_trace_hash=value["refinement_semantic_trace_hash"],
        final_intermediate_bounds_hash=value["final_intermediate_bounds_hash"],
        selected_target_count=value["selected_target_count"],
        tightened_neuron_count=value["tightened_neuron_count"],
        width_reduction_sum=value["width_reduction_sum"],
        initial_ambiguous_count=value["initial_ambiguous_count"],
        final_ambiguous_count=value["final_ambiguous_count"],
        alpha_count=value["alpha_count"],
        alpha_boundary_count=value["alpha_boundary_count"],
        alpha_interior_count=value["alpha_interior_count"],
        beta_count=value["beta_count"],
        beta_positive_count=value["beta_positive_count"],
    )


def _candidate_from_dict(value: Mapping[str, Any]) -> NativeFrontierCandidateNodeIR:
    return NativeFrontierCandidateNodeIR(**value)


def _decision_from_dict(value: Mapping[str, Any]) -> NativeFrontierTightnessDecisionIR:
    return NativeFrontierTightnessDecisionIR(**value)


def _depth_summary(
    rows: tuple[NativeFrontierNodeAttributionIR, ...],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for depth in sorted({row.depth for row in rows}):
        selected = tuple(row for row in rows if row.depth == depth)
        lowers = [row.lower for row in selected]
        parent_gains = [
            row.parent_lower_gain
            for row in selected
            if row.parent_lower_gain is not None
        ]
        alpha_count = sum(row.alpha_count for row in selected)
        result.append(
            {
                "depth": depth,
                "evaluated_node_count": len(selected),
                "active_node_count": sum(row.active for row in selected),
                "minimum_lower": min(lowers),
                "median_lower": float(statistics.median(lowers)),
                "maximum_lower": max(lowers),
                "mean_parent_lower_gain": (
                    None if not parent_gains else sum(parent_gains) / len(parent_gains)
                ),
                "selected_target_count": sum(
                    row.selected_target_count for row in selected
                ),
                "tightened_neuron_count": sum(
                    row.tightened_neuron_count for row in selected
                ),
                "width_reduction_sum": sum(row.width_reduction_sum for row in selected),
                "alpha_interior_fraction": (
                    0.0
                    if alpha_count == 0
                    else sum(row.alpha_interior_count for row in selected) / alpha_count
                ),
                "beta_positive_count": sum(row.beta_positive_count for row in selected),
            }
        )
    return result


def _clause_payload(execution: Any) -> dict[str, Any]:
    rows = execution.node_rows
    candidates = execution.candidate_rows
    payload = {
        "original_clause_index": execution.plan.original_clause_index,
        "plan": execution.plan.to_dict(),
        "plan_hash": execution.plan.stable_hash(),
        "task_ir_hash": execution.task_ir.stable_hash(),
        "schedule_hash": execution.schedule.stable_hash(execution.task_ir),
        "task_kinds": [task.kind.value for task in execution.task_ir.tasks],
        "node_rows": [row.to_dict() for row in rows],
        "candidate_rows": [row.to_dict() for row in candidates],
        "decision": execution.decision.to_dict(),
        "depth_summary": _depth_summary(rows),
        "baseline_batch_hashes": [
            item.stable_hash() for item in execution.baseline_batches
        ],
        "baseline_compiler_hashes": [
            item.stable_hash() for item in execution.baseline_compilers
        ],
        "baseline_cache_outcomes": [
            item.cache_event.outcome for item in execution.baseline_compilers
        ],
        "candidate_batch_hashes": [
            item.stable_hash() for item in execution.candidate_batches
        ],
        "candidate_compiler_hashes": [
            item.stable_hash() for item in execution.candidate_compilers
        ],
        "candidate_cache_outcomes": [
            item.cache_event.outcome for item in execution.candidate_compilers
        ],
    }
    payload["execution_hash"] = _canonical_hash(_semantic_clause(payload))
    return payload


def _semantic_clause(value: Mapping[str, Any]) -> dict[str, object]:
    return {
        key: value[key]
        for key in (
            "original_clause_index",
            "plan",
            "plan_hash",
            "task_ir_hash",
            "schedule_hash",
            "task_kinds",
            "node_rows",
            "candidate_rows",
            "decision",
            "depth_summary",
            "baseline_batch_hashes",
            "baseline_compiler_hashes",
            "baseline_cache_outcomes",
            "candidate_batch_hashes",
            "candidate_compiler_hashes",
            "candidate_cache_outcomes",
        )
    }


def _validate_clause(value: Mapping[str, Any]) -> None:
    plan = _plan_from_dict(value["plan"])
    plan.validate()
    rows = tuple(_node_from_dict(row) for row in value["node_rows"])
    candidates = tuple(_candidate_from_dict(row) for row in value["candidate_rows"])
    decision = _decision_from_dict(value["decision"])
    for node_row in rows:
        node_row.validate()
    for candidate_row in candidates:
        candidate_row.validate()
    decision.validate()
    active_rows = tuple(row for row in rows if row.active)
    active_ids = tuple(node_id for node_id, _split in plan.active_node_split_hashes)
    parent_by_id = {row.node_id: row.parent_node_id for row in rows}
    batch_indices = [row.sibling_batch_index for row in candidates]
    task_ir, schedule = lower_native_frontier_tightness_attribution_schedule(
        plan, rows, candidates, decision
    )
    expected_cache = ["miss_compiled", *("hit_exact_contract" for _ in range(7))]
    if (
        value["original_clause_index"] != plan.original_clause_index
        or value["plan_hash"] != plan.stable_hash()
        or value["task_ir_hash"] != task_ir.stable_hash()
        or value["schedule_hash"] != schedule.stable_hash(task_ir)
        or value["task_kinds"] != [task.kind.value for task in task_ir.tasks]
        or len(rows) != EXPECTED_EVALUATIONS
        or len({row.node_id for row in rows}) != EXPECTED_EVALUATIONS
        or len(active_rows) != EXPECTED_ACTIVE_NODES
        or tuple(sorted(row.node_id for row in active_rows)) != active_ids
        or any(row.depth != EXPECTED_ACTIVE_DEPTH for row in active_rows)
        or any(
            row.parent_node_id is not None and row.parent_node_id not in parent_by_id
            for row in rows
        )
        or len(candidates) != EXPECTED_ACTIVE_NODES
        or tuple(row.node_id for row in candidates) != active_ids
        or batch_indices != [index for index in range(8) for _ in range(2)]
        or decision != _decision(plan, candidates, source_coverage_passed=True)
        or value["depth_summary"] != _depth_summary(rows)
        or len(value["baseline_batch_hashes"]) != 8
        or len(value["baseline_compiler_hashes"]) != 8
        or len(value["candidate_batch_hashes"]) != 8
        or len(value["candidate_compiler_hashes"]) != 8
        or any(
            not isinstance(item, str) or len(item) != 64
            for key in (
                "baseline_batch_hashes",
                "baseline_compiler_hashes",
                "candidate_batch_hashes",
                "candidate_compiler_hashes",
            )
            for item in value[key]
        )
        or value["baseline_cache_outcomes"] != expected_cache
        or value["candidate_cache_outcomes"] != expected_cache
        or value["execution_hash"] != _canonical_hash(_semantic_clause(value))
    ):
        raise ValueError("NRIR-38 clause attribution differs")


def _semantic_pilot(pilot: Mapping[str, Any]) -> dict[str, object]:
    return {
        "protocol": pilot["protocol"],
        "workload": pilot["workload"],
        "source": pilot["source"],
        "clauses": pilot["clauses"],
        "all_candidate_gates_passed": pilot["all_candidate_gates_passed"],
        "claim_boundary": pilot["claim_boundary"],
        "performance_claimed": False,
    }


def validate_pilot(pilot: Mapping[str, Any]) -> None:
    if (
        pilot.get("schema_version") != PILOT_SCHEMA_VERSION
        or pilot.get("source", {}).get("native_code_revision") != _code_revision()
        or pilot.get("performance_claimed") is not False
        or pilot.get("claim_boundary")
        != "exact_frontier_optimizer_steps_tightness_only"
    ):
        raise ValueError("NRIR-38 pilot envelope differs")
    protocol = pilot.get("protocol", {})
    clauses = pilot.get("clauses")
    if (
        protocol.get("selected_original_clause_indices") != list(EXPECTED_SELECTED)
        or protocol.get("baseline_optimizer_steps") != 5
        or protocol.get("candidate_optimizer_steps") != 15
        or protocol.get("required_active_nodes_per_clause") != EXPECTED_ACTIVE_NODES
        or protocol.get("required_active_depth") != EXPECTED_ACTIVE_DEPTH
        or not isinstance(clauses, list)
        or len(clauses) != 2
    ):
        raise ValueError("NRIR-38 pilot protocol differs")
    for clause in clauses:
        _validate_clause(clause)
    all_go = all(clause["decision"]["go"] for clause in clauses)
    if (
        [clause["original_clause_index"] for clause in clauses]
        != list(EXPECTED_SELECTED)
        or pilot.get("all_candidate_gates_passed") is not all_go
        or pilot.get("status") != ("validated-reduced" if all_go else "no_go")
        or pilot.get("pilot_hash") != _canonical_hash(_semantic_pilot(pilot))
    ):
        raise ValueError("NRIR-38 pilot decision differs")


def _generate(args: argparse.Namespace) -> None:
    import torch

    from boundflow.runtime.native_multi_clause_anytime import (
        compile_native_multi_clause_anytime_program,
    )
    from boundflow.runtime.native_shared_parametric_multi_clause_anytime import (
        execute_native_shared_parametric_multi_clause_anytime_program,
    )

    torch.set_num_threads(args.torch_threads)
    workloads = _resolve_workloads(args.benchmark_root.resolve())
    workload = next(
        item for item in workloads if item["workload_id"] == "cifar10_resnet:000"
    )
    _query, tensors, module, input_spec = _load_query_runtime(
        Path(str(workload["model"])),
        Path(str(workload["property"])),
        "cifar10_resnet:000",
    )
    search_policy, baseline_policy = _policies()
    candidate_policy = replace(baseline_policy, steps=15)
    query_id = "nrir38:cifar10_resnet:000:property0:pilot"
    program = compile_native_multi_clause_anytime_program(
        module,
        input_spec,
        linear_spec_C=tensors.linear_spec_c,
        thresholds=tensors.thresholds,
        plan_id=query_id,
        search_policy=search_policy,
        optimizer_policy=baseline_policy,
    )
    source = execute_native_shared_parametric_multi_clause_anytime_program(
        program,
        module,
        input_spec,
        linear_spec_C=tensors.linear_spec_c,
        thresholds=tensors.thresholds,
        query_id=query_id,
        search_policy=search_policy,
        optimizer_policy=baseline_policy,
    )
    if source.decision.selected_original_clause_indices != EXPECTED_SELECTED or any(
        item.packed is None
        or len(item.packed.queue.trace.evaluations) != EXPECTED_EVALUATIONS
        for item in source.packed_executions
    ):
        raise RuntimeError("NRIR-38 source did not reach the frozen full frontier")
    clauses: list[dict[str, Any]] = []
    for item in source.packed_executions:
        packed = item.packed
        assert packed is not None
        ordinal = item.slice_ir.original_clause_index
        objective = tensors.linear_spec_c[:, ordinal : ordinal + 1, :].contiguous()
        threshold = tensors.thresholds[ordinal : ordinal + 1].contiguous()
        attribution_plan = compile_native_frontier_tightness_attribution_plan(
            packed,
            linear_spec_C=objective,
            threshold=threshold,
            original_clause_index=ordinal,
            baseline_optimizer_policy=baseline_policy,
            candidate_optimizer_policy=candidate_policy,
            plan_id=f"{query_id}:clause:{ordinal:04d}:frontier-attribution",
        )
        attribution = execute_native_frontier_tightness_attribution(
            attribution_plan,
            packed,
            module,
            input_spec,
            linear_spec_C=objective,
            threshold=threshold,
            original_clause_index=ordinal,
            baseline_optimizer_policy=baseline_policy,
            candidate_optimizer_policy=candidate_policy,
        )
        clauses.append(_clause_payload(attribution))
    all_go = all(clause["decision"]["go"] for clause in clauses)
    pilot: dict[str, Any] = {
        "schema_version": PILOT_SCHEMA_VERSION,
        "status": "validated-reduced" if all_go else "no_go",
        "protocol": {
            "selected_original_clause_indices": list(EXPECTED_SELECTED),
            "source_evaluations_per_clause": EXPECTED_EVALUATIONS,
            "required_active_nodes_per_clause": EXPECTED_ACTIVE_NODES,
            "required_active_depth": EXPECTED_ACTIVE_DEPTH,
            "baseline_optimizer_steps": 5,
            "candidate_optimizer_steps": 15,
            "torch_threads": args.torch_threads,
            "replay_allclose": {"atol": 1e-5, "rtol": 1e-5},
            "candidate_gate": {
                "minimum_node_delta": -1e-5,
                "minimum_worst_lower_improvement": 1.0,
                "minimum_improved_nodes_per_clause": 12,
            },
        },
        "workload": _public_workload(workload),
        "source": {
            "native_code_revision": _code_revision(),
            "program_plan_hash": program.plan.stable_hash(),
            "execution_trace_hash": source.trace.semantic_signature_hash,
            "selected_original_clause_indices": list(
                source.decision.selected_original_clause_indices
            ),
            "accepted_nodes": [
                item.slice_ir.accepted_nodes for item in source.packed_executions
            ],
            "template_count": len(source.template_hashes),
            "cache_miss_count": sum(
                event.outcome == "miss_compiled" for event in source.cache_events
            ),
        },
        "clauses": clauses,
        "all_candidate_gates_passed": all_go,
        "claim_boundary": "exact_frontier_optimizer_steps_tightness_only",
        "performance_claimed": False,
    }
    pilot["pilot_hash"] = _canonical_hash(_semantic_pilot(pilot))
    validate_pilot(pilot)
    pilot_path = args.artifact_dir.resolve() / "pilot.json"
    _write_json(pilot_path, pilot)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "files": {"pilot.json": _file_sha256(pilot_path)},
        "pilot_hash": _canonical_hash(pilot),
    }
    _write_json(args.artifact_dir.resolve() / "manifest.json", manifest)
    print(
        _canonical_json(
            {
                "status": pilot["status"],
                "pilot_hash": pilot["pilot_hash"],
                "decisions": [
                    {
                        "original_clause_index": clause["original_clause_index"],
                        "go": clause["decision"]["go"],
                        "worst_improvement": clause["decision"][
                            "worst_active_lower_improvement"
                        ],
                        "improved_nodes": clause["decision"]["improved_node_count"],
                    }
                    for clause in clauses
                ],
            }
        )
    )


def _replay(args: argparse.Namespace) -> None:
    artifact_dir = args.artifact_dir.resolve()
    manifest = _load_json(artifact_dir / "manifest.json")
    pilot_path = artifact_dir / "pilot.json"
    pilot = _load_json(pilot_path)
    workloads = _resolve_workloads(args.benchmark_root.resolve())
    workload = next(
        item for item in workloads if item["workload_id"] == "cifar10_resnet:000"
    )
    validate_pilot(pilot)
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or manifest.get("files") != {"pilot.json": _file_sha256(pilot_path)}
        or manifest.get("pilot_hash") != _canonical_hash(pilot)
        or pilot.get("workload") != _public_workload(workload)
    ):
        raise ValueError("NRIR-38 pilot manifest/workload differs")
    print(
        _canonical_json(
            {
                "status": pilot["status"],
                "pilot_hash": pilot["pilot_hash"],
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
    else:
        _replay(args)


if __name__ == "__main__":
    main()
