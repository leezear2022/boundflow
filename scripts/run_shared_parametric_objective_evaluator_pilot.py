#!/usr/bin/env python3
"""Run or replay the NRIR-37 parity and top-2 coverage pilot."""

# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=too-many-boolean-expressions,missing-function-docstring
# pylint: disable=import-outside-toplevel,protected-access,duplicate-code
# pylint: disable=too-many-arguments,too-many-positional-arguments
# pylint: disable=too-few-public-methods

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

from scripts.run_typed_hard_clause_escalation_artifact import (
    _load_query_runtime,
    _policies,
    _public_workload,
    _resolve_workloads,
)

PILOT_SCHEMA_VERSION = "boundflow.shared-parametric-objective-evaluator-pilot/v1"
MANIFEST_SCHEMA_VERSION = (
    "boundflow.shared-parametric-objective-evaluator-pilot-manifest/v1"
)
ARTIFACT_DIR = Path(
    "artifacts/shared-parametric-objective-evaluator/"
    "vnncomp21-resnet2b-property0-cpu-pilot-v1"
)
TORCH_THREADS = 8
WHOLE_QUERY_TIMEOUT_NS = 60_000_000_000
EXPECTED_SELECTED = (2, 3)


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
        "boundflow/ir/shared_parametric_ancestral.py",
        "boundflow/runtime/native_shared_parametric_ancestral.py",
        "boundflow/ir/parametric_optimizer.py",
        "boundflow/runtime/native_parametric_optimizer.py",
        "boundflow/ir/multi_clause_anytime.py",
        "boundflow/runtime/native_multi_clause_anytime.py",
        "scripts/run_shared_parametric_objective_evaluator_pilot.py",
    )
    return _canonical_hash({path: _file_sha256(root / path) for path in paths})


class _OnePairClock:
    """Allow root and one pair, then expose the fixed deadline."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> int:
        self.calls += 1
        return 0 if self.calls <= 6 else WHOLE_QUERY_TIMEOUT_NS


def _execute_floor(
    module: Any,
    input_spec: Any,
    objectives: Any,
    thresholds: Any,
    search_policy: Any,
    optimizer_policy: Any,
    *,
    query_id: str,
    global_started_ns: int | None = None,
):
    from boundflow.runtime.native_multi_clause_anytime import (
        _decision_from_floor,
        compile_native_multi_clause_anytime_program,
    )
    from boundflow.runtime.native_objective_hard_clause_escalation import (
        execute_native_objective_hard_clause_escalation_program,
    )

    program = compile_native_multi_clause_anytime_program(
        module,
        input_spec,
        linear_spec_C=objectives,
        thresholds=thresholds,
        plan_id=query_id,
        search_policy=search_policy,
        optimizer_policy=optimizer_policy,
    )
    first = True

    def floor_clock() -> int:
        nonlocal first
        if first and global_started_ns is not None:
            first = False
            return global_started_ns
        first = False
        return time.monotonic_ns()

    floor = execute_native_objective_hard_clause_escalation_program(
        program.floor_program,
        module,
        input_spec,
        linear_spec_C=objectives,
        thresholds=thresholds,
        query_id=f"{query_id}:floor-nrir31",
        search_policy=search_policy,
        optimizer_policy=optimizer_policy,
        clock_ns=floor_clock,
    )
    decision = _decision_from_floor(program.plan, floor, thresholds)
    return program, floor, decision


def _source(floor: Any, ordinal: int):
    from boundflow.runtime.native_multi_clause_anytime import _accepted_child

    child = _accepted_child(floor, ordinal)
    if child is None:
        raise ValueError("NRIR-37 selected floor source is absent")
    return child


def _state_parity(left: Any, right: Any) -> dict[str, object]:
    import torch

    left_states = left.queue.state_map()
    right_states = right.queue.state_map()
    if tuple(left_states) != tuple(right_states):
        raise ValueError("NRIR-37 parity node IDs differ")
    alpha_max = 0.0
    beta_max = 0.0
    split_exact = True
    for node_id in left_states:
        left_state = left_states[node_id]
        right_state = right_states[node_id]
        if set(left_state.splits) != set(right_state.splits):
            raise ValueError("NRIR-37 parity ReLU state keys differ")
        for name in left_state.splits:
            split_exact = split_exact and torch.equal(
                left_state.splits[name], right_state.splits[name]
            )
            alpha_max = max(
                alpha_max,
                float(
                    (left_state.alphas[name] - right_state.alphas[name])
                    .abs()
                    .max()
                    .item()
                ),
            )
            beta_max = max(
                beta_max,
                float(
                    (left_state.betas[name] - right_state.betas[name])
                    .abs()
                    .max()
                    .item()
                ),
            )
    return {
        "split_exact": split_exact,
        "alpha_max_abs_diff": alpha_max,
        "beta_max_abs_diff": beta_max,
        "alpha_within_1_1e_4": alpha_max <= 1.1e-4,
        "beta_within_1e_7": beta_max <= 1e-7,
    }


def _parity_pilot(
    module: Any,
    input_spec: Any,
    objectives: Any,
    thresholds: Any,
    search_policy: Any,
    optimizer_policy: Any,
) -> dict[str, object]:
    import torch

    from boundflow.runtime.native_objective_ancestral_sibling_pack import (
        compile_native_objective_ancestral_sibling_pack_plan,
        execute_native_objective_ancestral_sibling_pack_queue,
    )
    from boundflow.runtime.native_parametric_optimizer import (
        NativeParametricOptimizerTemplateCache,
    )
    from boundflow.runtime.native_shared_parametric_ancestral import (
        compile_native_shared_parametric_ancestral_plan,
        execute_native_shared_parametric_ancestral_queue,
    )

    _program, floor, decision = _execute_floor(
        module,
        input_spec,
        objectives,
        thresholds,
        search_policy,
        optimizer_policy,
        query_id="nrir37-parity",
    )
    if decision.selected_original_clause_indices != EXPECTED_SELECTED:
        raise ValueError("NRIR-37 parity floor rank differs")
    ordinal = EXPECTED_SELECTED[0]
    source = _source(floor, ordinal)
    objective = objectives[:, ordinal : ordinal + 1, :].contiguous()
    threshold = thresholds[ordinal : ordinal + 1].contiguous()
    query_id = f"nrir37-parity:clause:{ordinal:04d}"
    audit_plan = compile_native_objective_ancestral_sibling_pack_plan(
        module,
        input_spec,
        linear_spec_C=objective,
        threshold=threshold,
        root_refinement=source.refinement,
        optimizer_policy=optimizer_policy,
        plan_id=query_id,
    )
    audit_started_ns = time.perf_counter_ns()
    audit = execute_native_objective_ancestral_sibling_pack_queue(
        audit_plan,
        module,
        input_spec,
        linear_spec_C=objective,
        threshold=threshold,
        root_refinement=source.refinement,
        optimizer_policy=optimizer_policy,
        query_id=query_id,
        whole_query_started_ns=0,
        clock_ns=_OnePairClock(),
    )
    audit_elapsed_ns = time.perf_counter_ns() - audit_started_ns
    shared_plan = compile_native_shared_parametric_ancestral_plan(
        module,
        input_spec,
        linear_spec_C=objective,
        threshold=threshold,
        root_refinement=source.refinement,
        optimizer_policy=optimizer_policy,
        plan_id=query_id,
    )
    cache = NativeParametricOptimizerTemplateCache()
    shared_started_ns = time.perf_counter_ns()
    shared = execute_native_shared_parametric_ancestral_queue(
        shared_plan,
        module,
        input_spec,
        linear_spec_C=objective,
        threshold=threshold,
        root_refinement=source.refinement,
        optimizer_policy=optimizer_policy,
        compiler_cache=cache,
        query_id=query_id,
        whole_query_started_ns=0,
        clock_ns=_OnePairClock(),
    )
    shared_elapsed_ns = time.perf_counter_ns() - shared_started_ns
    audit_evaluations = audit.queue.trace.evaluations
    shared_evaluations = shared.queue.trace.evaluations
    if len(audit_evaluations) != 3 or len(shared_evaluations) != 3:
        raise ValueError("NRIR-37 parity evaluation coverage differs")
    audit_lower = torch.tensor([item.lower for item in audit_evaluations])
    shared_lower = torch.tensor([item.lower for item in shared_evaluations])
    audit_upper = torch.tensor([item.upper for item in audit_evaluations])
    shared_upper = torch.tensor([item.upper for item in shared_evaluations])
    lower_max = float((audit_lower - shared_lower).abs().max().item())
    upper_max = float((audit_upper - shared_upper).abs().max().item())
    bounds_allclose = bool(
        torch.allclose(audit_lower, shared_lower, atol=1e-5, rtol=1e-5)
        and torch.allclose(audit_upper, shared_upper, atol=1e-5, rtol=1e-5)
    )
    branches_exact = all(
        left.branch_candidate == right.branch_candidate
        for left, right in zip(audit_evaluations, shared_evaluations)
    )
    audit_refinements = tuple(
        item.execution.trace.final_intermediate_bounds_hash
        for item in audit.node_refinements
    )
    shared_refinements = tuple(
        item.execution.trace.final_intermediate_bounds_hash
        for item in shared.node_refinements
    )
    state = _state_parity(audit, shared)
    passed = bool(
        len(audit_evaluations) == len(shared_evaluations) == 3
        and bounds_allclose
        and branches_exact
        and audit_refinements == shared_refinements
        and state["split_exact"]
        and state["alpha_within_1_1e_4"]
        and state["beta_within_1e_7"]
        and [item.cache_event.outcome for item in shared.compiler_batches]
        == ["miss_compiled", "hit_exact_contract"]
        and shared.trace.selected_native_reexecution is False
    )
    return {
        "original_clause_index": ordinal,
        "audit_elapsed_ns": audit_elapsed_ns,
        "shared_elapsed_ns": shared_elapsed_ns,
        "node_count": len(shared_evaluations),
        "lower_max_abs_diff": lower_max,
        "upper_max_abs_diff": upper_max,
        "bounds_allclose_atol_1e_5_rtol_1e_5": bounds_allclose,
        "branches_exact": branches_exact,
        "refinement_final_bounds_hashes_exact": (
            audit_refinements == shared_refinements
        ),
        "state": state,
        "cache_outcomes": [
            item.cache_event.outcome for item in shared.compiler_batches
        ],
        "template_hashes": [item.template_hash for item in shared.compiler_batches],
        "selected_native_reexecution": shared.trace.selected_native_reexecution,
        "passed": passed,
    }


def _coverage_pilot(
    module: Any,
    input_spec: Any,
    objectives: Any,
    thresholds: Any,
    search_policy: Any,
    optimizer_policy: Any,
) -> dict[str, object]:
    from boundflow.runtime.native_multi_clause_anytime import _OneShotSliceClock
    from boundflow.runtime.native_parametric_optimizer import (
        NativeParametricOptimizerTemplateCache,
    )
    from boundflow.runtime.native_property_verdict import derive_native_property_verdict
    from boundflow.runtime.native_shared_parametric_ancestral import (
        compile_native_shared_parametric_ancestral_plan,
        execute_native_shared_parametric_ancestral_queue,
    )

    started_ns = time.monotonic_ns()
    _program, floor, decision = _execute_floor(
        module,
        input_spec,
        objectives,
        thresholds,
        search_policy,
        optimizer_policy,
        query_id="nrir37-coverage",
        global_started_ns=started_ns,
    )
    global_deadline_ns = started_ns + WHOLE_QUERY_TIMEOUT_NS
    cache = NativeParametricOptimizerTemplateCache()
    rows: list[dict[str, Any]] = []
    selected = decision.selected_original_clause_indices
    for position, ordinal in enumerate(selected):
        source = _source(floor, ordinal)
        dispatch_ns = time.monotonic_ns()
        remaining_ns = max(0, global_deadline_ns - dispatch_ns)
        remaining_count = len(selected) - position
        allocated_ns = remaining_ns // remaining_count
        cutoff_ns = min(global_deadline_ns, dispatch_ns + allocated_ns)
        objective = objectives[:, ordinal : ordinal + 1, :].contiguous()
        threshold = thresholds[ordinal : ordinal + 1].contiguous()
        query_id = f"nrir37-coverage:clause:{ordinal:04d}:priority:{position}"
        plan = compile_native_shared_parametric_ancestral_plan(
            module,
            input_spec,
            linear_spec_C=objective,
            threshold=threshold,
            root_refinement=source.refinement,
            optimizer_policy=optimizer_policy,
            plan_id=query_id,
        )
        slice_clock = _OneShotSliceClock(
            time.monotonic_ns,
            cutoff_ns=cutoff_ns,
            global_deadline_ns=global_deadline_ns,
        )
        execution = execute_native_shared_parametric_ancestral_queue(
            plan,
            module,
            input_spec,
            linear_spec_C=objective,
            threshold=threshold,
            root_refinement=source.refinement,
            optimizer_policy=optimizer_policy,
            compiler_cache=cache,
            query_id=query_id,
            whole_query_started_ns=started_ns,
            clock_ns=slice_clock,
        )
        root_id = execution.queue.trace.evaluations[0].node.node_id
        verdict = derive_native_property_verdict(
            module,
            input_spec,
            linear_spec_C=objective,
            queue_execution=execution.queue,
            candidate_counterexamples=(
                (root_id, source.query.clauses[0].search.best_input),
            ),
        )
        rows.append(
            {
                "priority_position": position,
                "original_clause_index": ordinal,
                "dispatch_started_elapsed_ns": dispatch_ns - started_ns,
                "remaining_before_ns": remaining_ns,
                "remaining_selected_count": remaining_count,
                "allocated_slice_ns": allocated_ns,
                "slice_cutoff_elapsed_ns": cutoff_ns - started_ns,
                "accepted_nodes": len(execution.queue.trace.evaluations),
                "sibling_group_count": len(execution.batch_commits) - 1,
                "cache_outcomes": [
                    item.cache_event.outcome for item in execution.compiler_batches
                ],
                "template_hashes": [
                    item.template_hash for item in execution.compiler_batches
                ],
                "discarded_attempt_stage": execution.trace.discarded_attempt_stage,
                "cutoff_signaled": slice_clock.signaled,
                "verdict": verdict.trace.status,
                "selected_native_reexecution": (
                    execution.trace.selected_native_reexecution
                ),
                "plan_hash": plan.stable_hash(),
                "queue_trace_hash": execution.queue.trace.stable_hash(),
                "execution_trace_hash": execution.trace.semantic_signature_hash,
            }
        )
    finished_ns = time.monotonic_ns()
    cache.validate()
    all_events = cache.events
    passed = bool(
        selected == EXPECTED_SELECTED
        and len(rows) == 2
        and all(int(item["accepted_nodes"]) >= 3 for item in rows)
        and all(
            int(item["accepted_nodes"]) == 1 + 2 * int(item["sibling_group_count"])
            for item in rows
        )
        and sum(item.outcome == "miss_compiled" for item in all_events) == 1
        and len(cache.templates) == 1
        and all(item["selected_native_reexecution"] is False for item in rows)
    )
    return {
        "floor_elapsed_ns": floor.trace.elapsed_ns,
        "whole_elapsed_ns": finished_ns - started_ns,
        "ranked_original_clause_indices": list(decision.ranked_original_clause_indices),
        "selected_original_clause_indices": list(selected),
        "slices": rows,
        "cache_event_count": len(all_events),
        "cache_miss_count": sum(item.outcome == "miss_compiled" for item in all_events),
        "cache_hit_count": sum(
            item.outcome == "hit_exact_contract" for item in all_events
        ),
        "template_count": len(cache.templates),
        "passed": passed,
    }


def _semantic_payload(pilot: Mapping[str, Any]) -> dict[str, object]:
    return {
        "source": pilot["source"],
        "workload": pilot["workload"],
        "protocol": pilot["protocol"],
        "parity": pilot["parity"],
        "coverage": pilot["coverage"],
        "status": pilot["status"],
        "claim_boundary": pilot["claim_boundary"],
        "performance_claimed": False,
    }


def validate_pilot(pilot: Mapping[str, Any]) -> None:
    parity = pilot.get("parity")
    coverage = pilot.get("coverage")
    if not isinstance(parity, dict) or not isinstance(coverage, dict):
        raise TypeError("NRIR-37 pilot structure differs")
    passed = bool(parity.get("passed") and coverage.get("passed"))
    if (
        pilot.get("schema_version") != PILOT_SCHEMA_VERSION
        or pilot.get("source", {}).get("native_code_revision") != _code_revision()
        or pilot.get("protocol", {}).get("selected_clause_count") != 2
        or pilot.get("protocol", {}).get("whole_query_timeout_seconds") != 60
        or pilot.get("performance_claimed") is not False
        or pilot.get("claim_boundary")
        != "same_algorithm_shared_compiler_and_fixed_deadline_coverage_only"
        or pilot.get("status") != ("ok" if passed else "no_go")
        or pilot.get("pilot_hash") != _canonical_hash(_semantic_payload(pilot))
    ):
        raise ValueError("NRIR-37 pilot envelope differs")


def _generate(args: argparse.Namespace) -> None:
    import torch

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
    search_policy, optimizer_policy = _policies()
    parity = _parity_pilot(
        module,
        input_spec,
        tensors.linear_spec_c,
        tensors.thresholds,
        search_policy,
        optimizer_policy,
    )
    coverage: dict[str, Any]
    if parity["passed"]:
        coverage = _coverage_pilot(
            module,
            input_spec,
            tensors.linear_spec_c,
            tensors.thresholds,
            search_policy,
            optimizer_policy,
        )
    else:
        coverage = {"passed": False, "reason": "parity_gate_failed"}
    passed = bool(parity["passed"] and coverage["passed"])
    pilot: dict[str, object] = {
        "schema_version": PILOT_SCHEMA_VERSION,
        "source": {"native_code_revision": _code_revision()},
        "workload": _public_workload(workload),
        "protocol": {
            "torch_threads": args.torch_threads,
            "whole_query_timeout_seconds": 60,
            "selected_clause_count": 2,
            "priority": "root_lower_margin_desc_ordinal_asc",
            "allocation": "dynamic_equal_remaining_selected_v1",
            "search_budget": {"max_nodes": 31, "max_depth": 4},
            "child_refinement_cap": 128,
            "shared_cache_scope": "one_query_cross_batch_cross_clause_v1",
        },
        "parity": parity,
        "coverage": coverage,
        "status": "ok" if passed else "no_go",
        "claim_boundary": (
            "same_algorithm_shared_compiler_and_fixed_deadline_coverage_only"
        ),
        "performance_claimed": False,
    }
    pilot["pilot_hash"] = _canonical_hash(_semantic_payload(pilot))
    validate_pilot(pilot)
    artifact_dir = args.artifact_dir.resolve()
    pilot_path = artifact_dir / "pilot.json"
    _write_json(pilot_path, pilot)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "files": {"pilot.json": _file_sha256(pilot_path)},
        "pilot_hash": _canonical_hash(pilot),
    }
    _write_json(artifact_dir / "manifest.json", manifest)
    coverage_slices = coverage.get("slices", [])
    if not isinstance(coverage_slices, list):
        raise TypeError("NRIR-37 coverage slices differ")
    print(
        _canonical_json(
            {
                "status": pilot["status"],
                "parity": parity["passed"],
                "selected": coverage.get("selected_original_clause_indices", []),
                "packed_nodes": [item["accepted_nodes"] for item in coverage_slices],
                "cache_miss_count": coverage.get("cache_miss_count"),
                "pilot_hash": pilot["pilot_hash"],
            }
        )
    )


def _replay(args: argparse.Namespace) -> None:
    artifact_dir = args.artifact_dir.resolve()
    pilot_path = artifact_dir / "pilot.json"
    pilot = _load_json(pilot_path)
    manifest = _load_json(artifact_dir / "manifest.json")
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or manifest.get("files", {}).get("pilot.json") != _file_sha256(pilot_path)
        or manifest.get("pilot_hash") != _canonical_hash(pilot)
    ):
        raise ValueError("NRIR-37 pilot manifest differs")
    validate_pilot(pilot)
    workloads = _resolve_workloads(args.benchmark_root.resolve())
    workload = next(
        item for item in workloads if item["workload_id"] == "cifar10_resnet:000"
    )
    if pilot["workload"] != _public_workload(workload):
        raise ValueError("NRIR-37 pilot workload differs")
    print(
        _canonical_json(
            {
                "status": pilot["status"],
                "performance_claimed": pilot["performance_claimed"],
                "pilot_hash": pilot["pilot_hash"],
            }
        )
    )


def main() -> None:
    args = _parse_args()
    if args.command == "generate":
        _generate(args)
    else:
        _replay(args)


if __name__ == "__main__":
    main()
