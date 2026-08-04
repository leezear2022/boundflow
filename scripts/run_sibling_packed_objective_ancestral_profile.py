#!/usr/bin/env python3
"""Profile serial versus packed NRIR-34 first-sibling evaluation."""

# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=too-many-arguments,too-many-boolean-expressions
# pylint: disable=missing-function-docstring,import-outside-toplevel
# pylint: disable=protected-access,duplicate-code

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
from functools import wraps
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Iterator, Mapping

from scripts.run_objective_ancestral_queue_artifact import _build_root_source
from scripts.run_typed_hard_clause_escalation_artifact import (
    _load_query_runtime,
    _policies,
    _public_workload,
    _resolve_workloads,
)

PROFILE_SCHEMA_VERSION = "boundflow.sibling-packed-objective-ancestral-profile/v1"
MANIFEST_SCHEMA_VERSION = (
    "boundflow.sibling-packed-objective-ancestral-profile-manifest/v1"
)
ARTIFACT_DIR = Path(
    "artifacts/sibling-packed-objective-ancestral-evaluator/"
    "vnncomp21-resnet2b-clause0-first-pair-cpu-profile-v1"
)
CLAUSE_INDEX = 0
TORCH_THREADS = 8
PHASE_FUNCTIONS = {
    "refinement_compile": "compile_native_intermediate_refinement_program",
    "refinement_execute": "execute_native_intermediate_refinement_program",
    "optimizer_compile": "compile_native_alpha_beta_optimizer_program",
    "optimizer_execute": "execute_native_alpha_beta_optimizer_program",
    "selected_native_compile": "compile_native_alpha_beta_state_query",
    "selected_native_execute": "execute_native_alpha_beta_state_query",
}


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
        "boundflow/runtime/native_objective_ancestral_queue.py",
        "boundflow/runtime/native_optimized_relu_split_bab_runtime.py",
        "boundflow/runtime/native_intermediate_refinement.py",
        "scripts/run_sibling_packed_objective_ancestral_profile.py",
    )
    return _canonical_hash({path: _file_sha256(root / path) for path in paths})


@contextmanager
def _profile_phases(runtime_module: Any) -> Iterator[dict[str, dict[str, int]]]:
    phases = {name: {"calls": 0, "elapsed_ns": 0} for name in PHASE_FUNCTIONS}
    originals: dict[str, Callable[..., Any]] = {}
    for phase, function_name in PHASE_FUNCTIONS.items():
        original = getattr(runtime_module, function_name)
        originals[function_name] = original

        def make_wrapper(
            target: Callable[..., Any], phase_name: str
        ) -> Callable[..., Any]:
            @wraps(target)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                started_ns = time.perf_counter_ns()
                try:
                    return target(*args, **kwargs)
                finally:
                    phases[phase_name]["calls"] += 1
                    phases[phase_name]["elapsed_ns"] += (
                        time.perf_counter_ns() - started_ns
                    )

            return wrapper

        setattr(runtime_module, function_name, make_wrapper(original, phase))
    try:
        yield phases
    finally:
        for function_name, original in originals.items():
            setattr(runtime_module, function_name, original)


def _root_runtime(module: Any, input_spec: Any):
    import torch

    from boundflow.frontends.plain_crown_bound_ir import relu_split_state_hash
    from boundflow.ir.bound import IntermediateBoundSource
    from boundflow.runtime.crown_ibp import _forward_ibp_trace_mlp
    from boundflow.runtime import native_optimized_relu_split_bab_runtime as optimized
    from boundflow.runtime.native_relu_split_bab_runtime import (
        NativeReluSplitBabConfig,
        NativeReluSplitBabNode,
    )

    _root_interval, root_pre = _forward_ibp_trace_mlp(module, input_spec)
    root_splits = tuple(
        (
            name,
            torch.zeros(
                tuple(int(dim) for dim in pre.lower.shape[1:]),
                dtype=torch.int8,
                device=pre.lower.device,
            ),
        )
        for name, pre in sorted(root_pre.items())
    )
    root_mapping = {name: value.unsqueeze(0) for name, value in root_splits}
    root = optimized._RuntimeNode(
        node=NativeReluSplitBabNode(
            node_id="nrir34:cifar10_resnet:000:clause0:n000000",
            parent_node_id=None,
            depth=0,
            branch_relu_input=None,
            branch_neuron_index=None,
            branch_value=0,
            split_state_hash=relu_split_state_hash(root_mapping),
        ),
        split_state=root_splits,
    )
    config = NativeReluSplitBabConfig(
        max_nodes=31,
        max_depth=4,
        expansion_batch_size=1,
        max_eval_batch_size=2,
        threshold=1e6,
    )
    return root, config, IntermediateBoundSource.NATIVE_REFINED


def _evaluate_root(
    module: Any,
    input_spec: Any,
    *,
    objective: Any,
    optimizer: Any,
    root: Any,
    root_refinement: Any,
    config: Any,
    source: Any,
):
    from boundflow.runtime import native_optimized_relu_split_bab_runtime as optimized

    evaluated, _stack, branches, refinements, records = (
        optimized._evaluate_optimized_node_batch(
            module,
            input_spec,
            objective=objective,
            nodes=(root,),
            batch_id="nrir34:root",
            config=config,
            policy=optimizer,
            parent_by_id={},
            relu_pre_override=root_refinement.relu_pre,
            intermediate_bound_source=source,
            objective_branch_policy=None,
            refine_external_constraints=False,
            per_child_refinement_policy=None,
            per_child_refinement_budget_policy=None,
            per_child_refinement_multi_pass_policy=None,
            per_child_refinement_strategy="independent_exact_split_v1",
            external_constraint_seed=None,
        )
    )
    if len(evaluated) != 1 or branches or refinements or records:
        raise ValueError("NRIR-34 root evaluator coverage differs")
    return replace(evaluated[0], refinement_execution=root_refinement)


def _children(root_evaluated: Any):
    from boundflow.runtime import native_optimized_relu_split_bab_runtime as optimized

    branch = root_evaluated.evaluation.branch_candidate
    if branch is None:
        raise ValueError("NRIR-34 root lacks a branch candidate")
    return tuple(
        optimized._make_child_runtime_node(
            root_evaluated.runtime_node,
            child_id=f"nrir34:cifar10_resnet:000:clause0:n{index + 1:06d}",
            branch=branch,
            branch_value=branch_value,
        )
        for index, branch_value in enumerate((-1, 1))
    )


def _serial_children(
    module: Any,
    input_spec: Any,
    *,
    objective: Any,
    optimizer_policy: Any,
    root_evaluated: Any,
    children: tuple[Any, ...],
    config: Any,
    source: Any,
) -> tuple[tuple[Any, ...], tuple[Any, ...], dict[str, dict[str, int]], int]:
    from boundflow.ir.refinement import NativeIntermediateRefinementPolicyIR
    from boundflow.runtime import native_optimized_relu_split_bab_runtime as optimized

    policy = NativeIntermediateRefinementPolicyIR(
        passes=1,
        max_neurons_per_relu=128,
        backward_chunk_size=32,
        candidate_policy_id="objective_influence_width_per_relu_v1",
    )
    parent_by_id = {root_evaluated.runtime_node.node.node_id: root_evaluated}
    evaluations = []
    executions = []
    started_ns = time.perf_counter_ns()
    with _profile_phases(optimized) as phases:
        for index, child in enumerate(children):
            program = optimized.compile_native_intermediate_refinement_program(
                module,
                input_spec,
                policy=policy,
                plan_id=f"per-child-refinement:{child.node.split_state_hash}",
                relu_split_state=optimized._node_split_mapping(child),
                linear_spec_C=objective,
                source_refinement_execution=root_evaluated.refinement_execution,
            )
            refinement = optimized.execute_native_intermediate_refinement_program(
                program, module, input_spec
            )
            evaluated, _stack, branches, refinements, records = (
                optimized._evaluate_optimized_node_batch(
                    module,
                    input_spec,
                    objective=objective,
                    nodes=(child,),
                    batch_id=f"nrir34:serial:{index}",
                    config=config,
                    policy=optimizer_policy,
                    parent_by_id=parent_by_id,
                    relu_pre_override=refinement.relu_pre,
                    intermediate_bound_source=source,
                    objective_branch_policy=None,
                    refine_external_constraints=False,
                    per_child_refinement_policy=None,
                    per_child_refinement_budget_policy=None,
                    per_child_refinement_multi_pass_policy=None,
                    per_child_refinement_strategy="independent_exact_split_v1",
                    external_constraint_seed=None,
                )
            )
            if len(evaluated) != 1 or branches or refinements or records:
                raise ValueError("NRIR-34 serial child coverage differs")
            evaluations.append(evaluated[0])
            executions.append(refinement)
    return (
        tuple(evaluations),
        tuple(executions),
        phases,
        time.perf_counter_ns() - started_ns,
    )


def _packed_children(
    module: Any,
    input_spec: Any,
    *,
    objective: Any,
    optimizer_policy: Any,
    root_evaluated: Any,
    children: tuple[Any, ...],
    config: Any,
    source: Any,
) -> tuple[tuple[Any, ...], tuple[Any, ...], dict[str, dict[str, int]], int]:
    from boundflow.ir.refinement import NativeIntermediateRefinementPolicyIR
    from boundflow.runtime import native_optimized_relu_split_bab_runtime as optimized

    policy = NativeIntermediateRefinementPolicyIR(
        passes=1,
        max_neurons_per_relu=128,
        backward_chunk_size=32,
        candidate_policy_id="objective_influence_width_per_relu_v1",
    )
    parent_by_id = {root_evaluated.runtime_node.node.node_id: root_evaluated}
    started_ns = time.perf_counter_ns()
    with _profile_phases(optimized) as phases:
        evaluated, _stack, branches, refinements, records = (
            optimized._evaluate_optimized_node_batch(
                module,
                input_spec,
                objective=objective,
                nodes=children,
                batch_id="nrir34:packed",
                config=config,
                policy=optimizer_policy,
                parent_by_id=parent_by_id,
                relu_pre_override=None,
                intermediate_bound_source=source,
                objective_branch_policy=None,
                refine_external_constraints=False,
                per_child_refinement_policy=policy,
                per_child_refinement_budget_policy=None,
                per_child_refinement_multi_pass_policy=None,
                per_child_refinement_strategy="ancestral_constraint_carry_v1",
                external_constraint_seed=None,
            )
        )
    if len(evaluated) != 2 or branches or len(refinements) != 2 or len(records) != 2:
        raise ValueError("NRIR-34 packed child coverage differs")
    return (
        evaluated,
        tuple(execution for _node_id, execution in refinements),
        phases,
        time.perf_counter_ns() - started_ns,
    )


def _state_comparison(left: tuple[Any, ...], right: tuple[Any, ...]) -> dict[str, Any]:
    import torch

    alpha_max = 0.0
    beta_max = 0.0
    split_exact = True
    stable_scope_fields_equal = True
    for serial, packed in zip(left, right):
        left_state = serial.selected_state
        right_state = packed.selected_state
        if set(left_state.splits) != set(right_state.splits):
            raise ValueError("NRIR-34 state ReLU keys differ")
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
        stable_scope_fields_equal = stable_scope_fields_equal and all(
            getattr(left_state.scope, field) == getattr(right_state.scope, field)
            for field in (
                "primal_graph_hash",
                "input_region_hash",
                "objective_hash",
                "split_state_hash",
                "optimizer_policy_hash",
                "intermediate_bounds_hash",
            )
        )
    return {
        "split_tensors_exact": split_exact,
        "stable_scope_fields_equal": stable_scope_fields_equal,
        "alpha_max_abs_diff": alpha_max,
        "beta_max_abs_diff": beta_max,
    }


def _semantics(
    evaluated: tuple[Any, ...], refinements: tuple[Any, ...]
) -> dict[str, Any]:
    from boundflow.runtime.native_intermediate_refinement import (
        intermediate_refinement_semantic_trace_hash,
    )

    return {
        "nodes": [
            {
                "node": item.evaluation.node.to_dict(),
                "lower": item.evaluation.lower,
                "upper": item.evaluation.upper,
                "branch_candidate": (
                    None
                    if item.evaluation.branch_candidate is None
                    else item.evaluation.branch_candidate.to_dict()
                ),
                "selected_state_hash": item.selected_state.stable_hash(),
            }
            for item in evaluated
        ],
        "refinements": [
            {
                "plan_hash": execution.program.plan.stable_hash(),
                "semantic_trace_hash": intermediate_refinement_semantic_trace_hash(
                    execution
                ),
                "source_plan_hash": execution.program.plan.source_refinement_plan_hash,
                "source_semantic_trace_hash": (
                    execution.program.plan.source_refinement_semantic_trace_hash
                ),
                "source_constraints_hash": (
                    execution.program.plan.source_intermediate_constraints_hash
                ),
            }
            for execution in refinements
        ],
    }


def _generate(args: argparse.Namespace) -> None:
    import torch

    torch.set_num_threads(args.torch_threads)
    workloads = _resolve_workloads(args.benchmark_root.resolve())
    workload = next(
        item for item in workloads if item["workload_id"] == "cifar10_resnet:000"
    )
    whole_started_ns = time.perf_counter_ns()
    _query, tensors, module, input_spec = _load_query_runtime(
        Path(str(workload["model"])),
        Path(str(workload["property"])),
        "cifar10_resnet:000",
    )
    objective = tensors.linear_spec_c[
        :, CLAUSE_INDEX : CLAUSE_INDEX + 1, :
    ].contiguous()
    evaluator_objective = objective[0].contiguous()
    _search_policy, optimizer_policy = _policies()
    source_started_ns = time.perf_counter_ns()
    _shared, root_refinement = _build_root_source(module, input_spec, objective)
    source_elapsed_ns = time.perf_counter_ns() - source_started_ns
    root, config, source = _root_runtime(module, input_spec)
    root_started_ns = time.perf_counter_ns()
    root_evaluated = _evaluate_root(
        module,
        input_spec,
        objective=evaluator_objective,
        optimizer=optimizer_policy,
        root=root,
        root_refinement=root_refinement,
        config=config,
        source=source,
    )
    root_elapsed_ns = time.perf_counter_ns() - root_started_ns
    children = _children(root_evaluated)
    serial, serial_refinements, serial_phases, serial_elapsed_ns = _serial_children(
        module,
        input_spec,
        objective=evaluator_objective,
        optimizer_policy=optimizer_policy,
        root_evaluated=root_evaluated,
        children=children,
        config=config,
        source=source,
    )
    packed, packed_refinements, packed_phases, packed_elapsed_ns = _packed_children(
        module,
        input_spec,
        objective=evaluator_objective,
        optimizer_policy=optimizer_policy,
        root_evaluated=root_evaluated,
        children=children,
        config=config,
        source=source,
    )
    lower_diff = max(
        abs(left.evaluation.lower - right.evaluation.lower)
        for left, right in zip(serial, packed)
    )
    upper_diff = max(
        abs(left.evaluation.upper - right.evaluation.upper)
        for left, right in zip(serial, packed)
    )
    refinement_equal = _semantics(serial, serial_refinements)["refinements"] == (
        _semantics(packed, packed_refinements)["refinements"]
    )
    state = _state_comparison(serial, packed)
    gate = {
        "lower_max_abs_diff": lower_diff,
        "upper_max_abs_diff": upper_diff,
        "bounds_parity_1e_5": lower_diff <= 1e-5 and upper_diff <= 1e-5,
        "split_tensors_exact": state["split_tensors_exact"],
        "stable_scope_fields_equal": state["stable_scope_fields_equal"],
        "alpha_max_abs_diff": state["alpha_max_abs_diff"],
        "beta_max_abs_diff": state["beta_max_abs_diff"],
        "refinement_semantics_equal": refinement_equal,
        "serial_optimizer_groups": serial_phases["optimizer_execute"]["calls"],
        "packed_optimizer_groups": packed_phases["optimizer_execute"]["calls"],
        "serial_native_groups": serial_phases["selected_native_execute"]["calls"],
        "packed_native_groups": packed_phases["selected_native_execute"]["calls"],
        "packed_child_elapsed_strictly_lower": packed_elapsed_ns < serial_elapsed_ns,
        "mechanism_gate_passed": (
            lower_diff <= 1e-5
            and upper_diff <= 1e-5
            and state["split_tensors_exact"] is True
            and state["stable_scope_fields_equal"] is True
            and refinement_equal
            and packed_elapsed_ns < serial_elapsed_ns
            and serial_phases["optimizer_execute"]["calls"] == 2
            and packed_phases["optimizer_execute"]["calls"] == 1
            and serial_phases["selected_native_execute"]["calls"] == 2
            and packed_phases["selected_native_execute"]["calls"] == 1
        ),
    }
    profile = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "status": "ok" if gate["mechanism_gate_passed"] else "no_go",
        "source": {"native_code_revision": _code_revision()},
        "workload": _public_workload(workload),
        "protocol": {
            "clause_index": CLAUSE_INDEX,
            "child_cap": 128,
            "optimizer_steps": optimizer_policy.steps,
            "torch_threads": args.torch_threads,
            "order": ["serial", "packed"],
            "objective_projection": "drop_singleton_domain_axis_v1",
            "root_source_objective_shape": list(objective.shape),
            "evaluator_objective_shape": list(evaluator_objective.shape),
            "performance_claimed": False,
        },
        "root": {
            "lower": root_evaluated.evaluation.lower,
            "upper": root_evaluated.evaluation.upper,
            "branch": root_evaluated.evaluation.branch_candidate.to_dict(),
            "source_elapsed_ns": source_elapsed_ns,
            "evaluation_elapsed_ns": root_elapsed_ns,
        },
        "serial": {
            "child_elapsed_ns": serial_elapsed_ns,
            "phases": serial_phases,
            "semantics": _semantics(serial, serial_refinements),
        },
        "packed": {
            "child_elapsed_ns": packed_elapsed_ns,
            "phases": packed_phases,
            "semantics": _semantics(packed, packed_refinements),
        },
        "comparison": gate,
        "whole_elapsed_ns": time.perf_counter_ns() - whole_started_ns,
        "performance_claimed": False,
    }
    validate_profile(profile)
    artifact_dir = args.artifact_dir.resolve()
    profile_path = artifact_dir / "profile.json"
    _write_json(profile_path, profile)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "files": {"profile.json": _file_sha256(profile_path)},
        "profile_hash": _canonical_hash(profile),
    }
    _write_json(artifact_dir / "manifest.json", manifest)
    print(
        _canonical_json(
            {
                "status": profile["status"],
                "serial_child_seconds": serial_elapsed_ns / 1e9,
                "packed_child_seconds": packed_elapsed_ns / 1e9,
                "diagnostic_speedup": serial_elapsed_ns / packed_elapsed_ns,
                "profile_hash": manifest["profile_hash"],
            }
        )
    )


def validate_profile(profile: Mapping[str, Any]) -> None:
    if (
        profile.get("schema_version") != PROFILE_SCHEMA_VERSION
        or profile.get("status") not in {"ok", "no_go"}
        or profile.get("performance_claimed") is not False
        or profile.get("source", {}).get("native_code_revision") != _code_revision()
        or profile.get("protocol", {}).get("performance_claimed") is not False
    ):
        raise ValueError("NRIR-34 profile envelope differs")
    serial = profile.get("serial")
    packed = profile.get("packed")
    comparison = profile.get("comparison")
    if not all(isinstance(value, dict) for value in (serial, packed, comparison)):
        raise TypeError("NRIR-34 profile structure differs")
    assert isinstance(serial, dict)
    assert isinstance(packed, dict)
    assert isinstance(comparison, dict)
    serial_nodes = serial.get("semantics", {}).get("nodes", [])
    packed_nodes = packed.get("semantics", {}).get("nodes", [])
    recalculated_lower = max(
        abs(float(left["lower"]) - float(right["lower"]))
        for left, right in zip(serial_nodes, packed_nodes)
    )
    recalculated_upper = max(
        abs(float(left["upper"]) - float(right["upper"]))
        for left, right in zip(serial_nodes, packed_nodes)
    )
    expected_gate = (
        recalculated_lower <= 1e-5
        and recalculated_upper <= 1e-5
        and comparison.get("split_tensors_exact") is True
        and comparison.get("stable_scope_fields_equal") is True
        and comparison.get("refinement_semantics_equal") is True
        and int(packed["child_elapsed_ns"]) < int(serial["child_elapsed_ns"])
        and comparison.get("serial_optimizer_groups") == 2
        and comparison.get("packed_optimizer_groups") == 1
        and comparison.get("serial_native_groups") == 2
        and comparison.get("packed_native_groups") == 1
    )
    if (
        len(serial_nodes) != 2
        or len(packed_nodes) != 2
        or comparison.get("lower_max_abs_diff") != recalculated_lower
        or comparison.get("upper_max_abs_diff") != recalculated_upper
        or comparison.get("bounds_parity_1e_5")
        != (recalculated_lower <= 1e-5 and recalculated_upper <= 1e-5)
        or comparison.get("packed_child_elapsed_strictly_lower")
        != (int(packed["child_elapsed_ns"]) < int(serial["child_elapsed_ns"]))
        or comparison.get("mechanism_gate_passed") != expected_gate
        or profile.get("status") != ("ok" if expected_gate else "no_go")
    ):
        raise ValueError("NRIR-34 profile gate differs")


def _replay(args: argparse.Namespace) -> None:
    artifact_dir = args.artifact_dir.resolve()
    profile = _load_json(artifact_dir / "profile.json")
    manifest = _load_json(artifact_dir / "manifest.json")
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or manifest.get("files", {}).get("profile.json")
        != _file_sha256(artifact_dir / "profile.json")
        or manifest.get("profile_hash") != _canonical_hash(profile)
    ):
        raise ValueError("NRIR-34 profile manifest differs")
    validate_profile(profile)
    workloads = _resolve_workloads(args.benchmark_root.resolve())
    workload = next(
        item for item in workloads if item["workload_id"] == "cifar10_resnet:000"
    )
    if profile.get("workload") != _public_workload(workload):
        raise ValueError("NRIR-34 profile workload differs")
    print(
        _canonical_json(
            {
                "status": "ok",
                "mechanism_gate_passed": profile["comparison"]["mechanism_gate_passed"],
                "profile_hash": manifest["profile_hash"],
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
