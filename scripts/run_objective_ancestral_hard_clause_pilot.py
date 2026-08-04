#!/usr/bin/env python3
"""Run the NRIR-32 typed root-to-first-child feasibility pilot."""

# pylint: disable=too-many-locals,too-many-statements,missing-function-docstring
# pylint: disable=import-outside-toplevel,protected-access,duplicate-code
# pylint: disable=too-many-arguments

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
    _resolve_workloads,
)

SCHEMA_VERSION = "boundflow.objective-ancestral-hard-clause-pilot/v1"
ARTIFACT_DIR = Path(
    "artifacts/objective-ancestral-hard-clause-escalation/"
    "vnncomp21-resnet2b-clause0-first-child-cpu-pilot-v1"
)
CLAUSE_INDEX = 0
TORCH_THREADS = 8


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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


def _source_revision() -> str:
    root = Path(__file__).resolve().parents[1]
    paths = (
        "boundflow/ir/refinement.py",
        "boundflow/runtime/native_intermediate_refinement.py",
        "boundflow/runtime/native_optimized_relu_split_bab_runtime.py",
        "scripts/run_objective_ancestral_hard_clause_pilot.py",
    )
    return _canonical_hash({path: _file_sha256(root / path) for path in paths})


def _evaluation_payload(value: Any) -> dict[str, object]:
    evaluation = value.evaluation
    return {
        "node": evaluation.node.to_dict(),
        "lower": evaluation.lower,
        "upper": evaluation.upper,
        "selected_state_hash": evaluation.selected_state_hash,
        "optimizer_ir_hashes": dict(evaluation.optimizer_ir_hashes),
        "optimizer_execution_trace_hash": evaluation.optimizer_execution_trace_hash,
        "native_ir_hashes": dict(evaluation.native_ir_hashes),
        "branch_candidate": (
            None
            if evaluation.branch_candidate is None
            else evaluation.branch_candidate.to_dict()
        ),
    }


def _evaluate_serial(
    module: Any,
    input_spec: Any,
    *,
    objective: Any,
    node: Any,
    batch_id: str,
    config: Any,
    optimizer_policy: Any,
    parent_by_id: Mapping[str, Any],
    relu_pre: Mapping[str, Any],
) -> Any:
    from boundflow.ir.bound import IntermediateBoundSource
    from boundflow.runtime.native_optimized_relu_split_bab_runtime import (
        _evaluate_optimized_node_batch,
    )

    evaluated, _stack, _branches, _refinements, _records = (
        _evaluate_optimized_node_batch(
            module,
            input_spec,
            objective=objective,
            nodes=(node,),
            batch_id=batch_id,
            config=config,
            policy=optimizer_policy,
            parent_by_id=parent_by_id,
            relu_pre_override=relu_pre,
            intermediate_bound_source=IntermediateBoundSource.NATIVE_REFINED,
            objective_branch_policy=None,
            refine_external_constraints=False,
            per_child_refinement_policy=None,
            per_child_refinement_budget_policy=None,
            per_child_refinement_multi_pass_policy=None,
            per_child_refinement_strategy="independent_exact_split_v1",
            external_constraint_seed=None,
        )
    )
    if len(evaluated) != 1:
        raise ValueError("NRIR-32 serial evaluator coverage differs")
    return evaluated[0]


def _run(args: argparse.Namespace) -> None:
    import torch

    from boundflow.frontends.plain_crown_bound_ir import relu_split_state_hash
    from boundflow.ir.refinement import NativeIntermediateRefinementPolicyIR
    from boundflow.runtime.crown_ibp import _forward_ibp_trace_mlp
    from boundflow.runtime.native_intermediate_refinement import (
        compile_native_intermediate_refinement_program,
        execute_native_intermediate_refinement_program,
        intermediate_bounds_hash,
        intermediate_refinement_semantic_trace_hash,
    )
    from boundflow.runtime.native_optimized_relu_split_bab_runtime import (
        _RuntimeNode,
        _make_child_runtime_node,
        _node_split_mapping,
    )
    from boundflow.runtime.native_relu_split_bab_runtime import (
        NativeReluSplitBabConfig,
        NativeReluSplitBabNode,
    )

    torch.set_num_threads(args.torch_threads)
    workloads = _resolve_workloads(args.benchmark_root.resolve())
    workload = next(
        item for item in workloads if item["workload_id"] == "cifar10_resnet:000"
    )
    _query, tensors, module, input_spec = _load_query_runtime(
        Path(str(workload["model"])),
        Path(str(workload["property"])),
        str(workload["workload_id"]),
    )
    objective = tensors.linear_spec_c[
        :, CLAUSE_INDEX : CLAUSE_INDEX + 1, :
    ].contiguous()
    threshold = float(tensors.thresholds[CLAUSE_INDEX].item())
    _search_policy, optimizer_policy = _policies()
    shared_policy = NativeIntermediateRefinementPolicyIR(
        passes=1, max_neurons_per_relu=128, backward_chunk_size=32
    )
    objective_policy = NativeIntermediateRefinementPolicyIR(
        passes=1,
        max_neurons_per_relu=128,
        backward_chunk_size=32,
        candidate_policy_id="objective_influence_width_per_relu_v1",
    )

    started_ns = time.perf_counter_ns()
    shared_program = compile_native_intermediate_refinement_program(
        module,
        input_spec,
        policy=shared_policy,
        plan_id="nrir32:resnet:clause0:shared",
    )
    shared = execute_native_intermediate_refinement_program(
        shared_program, module, input_spec
    )
    root_program = compile_native_intermediate_refinement_program(
        module,
        input_spec,
        policy=objective_policy,
        plan_id="nrir32:resnet:clause0:objective-root",
        linear_spec_C=objective,
        source_refinement_execution=shared,
    )
    root_refinement = execute_native_intermediate_refinement_program(
        root_program, module, input_spec
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
    root_node = _RuntimeNode(
        node=NativeReluSplitBabNode(
            node_id="nrir32:resnet:clause0:n000000",
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
        max_nodes=7,
        max_depth=2,
        expansion_batch_size=1,
        max_eval_batch_size=1,
        threshold=threshold,
    )
    root_evaluated = _evaluate_serial(
        module,
        input_spec,
        objective=objective,
        node=root_node,
        batch_id="nrir32:root",
        config=config,
        optimizer_policy=optimizer_policy,
        parent_by_id={},
        relu_pre=root_refinement.relu_pre,
    )
    branch = root_evaluated.evaluation.branch_candidate
    if branch is None:
        raise ValueError("NRIR-32 root lacks a branch candidate")
    children = tuple(
        _make_child_runtime_node(
            root_evaluated.runtime_node,
            child_id=f"nrir32:resnet:clause0:n{index + 1:06d}",
            branch=branch,
            branch_value=branch_value,
        )
        for index, branch_value in enumerate((-1, 1))
    )
    parent_by_id = {root_evaluated.runtime_node.node.node_id: root_evaluated}

    rows = []
    for child in children:
        branch_value = child.node.branch_value
        root_global = _evaluate_serial(
            module,
            input_spec,
            objective=objective,
            node=child,
            batch_id=f"nrir32:root-global:{branch_value}",
            config=config,
            optimizer_policy=optimizer_policy,
            parent_by_id=parent_by_id,
            relu_pre=root_refinement.relu_pre,
        )
        child_program = compile_native_intermediate_refinement_program(
            module,
            input_spec,
            policy=objective_policy,
            plan_id=f"nrir32:resnet:clause0:child:{branch_value}",
            relu_split_state=_node_split_mapping(child),
            linear_spec_C=objective,
            source_refinement_execution=root_refinement,
        )
        child_refinement = execute_native_intermediate_refinement_program(
            child_program, module, input_spec
        )
        ancestral = _evaluate_serial(
            module,
            input_spec,
            objective=objective,
            node=child,
            batch_id=f"nrir32:ancestral:{branch_value}",
            config=config,
            optimizer_policy=optimizer_policy,
            parent_by_id=parent_by_id,
            relu_pre=child_refinement.relu_pre,
        )
        if (
            child_program.plan.source_intermediate_constraints_hash
            != intermediate_bounds_hash(root_refinement.relu_pre)
            or child_program.plan.source_refinement_plan_hash
            != root_program.plan.stable_hash()
            or child_program.plan.source_refinement_semantic_trace_hash
            != intermediate_refinement_semantic_trace_hash(root_refinement)
        ):
            raise ValueError("NRIR-32 root/child typed lineage differs")
        delta = ancestral.evaluation.lower - root_global.evaluation.lower
        rows.append(
            {
                "branch_value": branch_value,
                "child_node_id": child.node.node_id,
                "child_split_state_hash": child.node.split_state_hash,
                "root_global": _evaluation_payload(root_global),
                "ancestral": _evaluation_payload(ancestral),
                "lower_delta": delta,
                "non_regression_1e_5": delta >= -1e-5,
                "child_refinement": {
                    "program_hashes": child_program.hashes(),
                    "semantic_trace_hash": intermediate_refinement_semantic_trace_hash(
                        child_refinement
                    ),
                    "source_intermediate_constraints_hash": (
                        child_program.plan.source_intermediate_constraints_hash
                    ),
                    "source_refinement_plan_hash": (
                        child_program.plan.source_refinement_plan_hash
                    ),
                    "source_refinement_semantic_trace_hash": (
                        child_program.plan.source_refinement_semantic_trace_hash
                    ),
                    "source_consumption": "sound_constraint_only",
                },
            }
        )

    root_global_worst = min(float(row["root_global"]["lower"]) for row in rows)
    ancestral_worst = min(float(row["ancestral"]["lower"]) for row in rows)
    worst_delta = ancestral_worst - root_global_worst
    gate = {
        "root_exact_source": True,
        "all_children_non_regressing": all(row["non_regression_1e_5"] for row in rows),
        "root_global_worst_child_lower": root_global_worst,
        "ancestral_worst_child_lower": ancestral_worst,
        "worst_child_lower_delta": worst_delta,
        "strict_worst_child_improvement_gt_1e_4": worst_delta > 1e-4,
    }
    gate["pilot_gate_passed"] = bool(
        gate["root_exact_source"]
        and gate["all_children_non_regressing"]
        and gate["strict_worst_child_improvement_gt_1e_4"]
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "workload_id": workload["workload_id"],
        "clause_index": CLAUSE_INDEX,
        "protocol": {
            "torch_threads": args.torch_threads,
            "optimizer_steps": optimizer_policy.steps,
            "shared_policy": shared_policy.to_dict(),
            "objective_policy": objective_policy.to_dict(),
            "search_budget_context": {"max_nodes": 7, "max_depth": 2},
            "child_evaluation_batch_size": 1,
            "performance_claimed": False,
        },
        "source": {
            "code_revision": _source_revision(),
            "model_sha256": workload["model_sha256"],
            "property_sha256": workload["property_sha256"],
        },
        "shared": {
            "program_hashes": shared_program.hashes(),
            "semantic_trace_hash": intermediate_refinement_semantic_trace_hash(shared),
        },
        "objective_root": {
            "program_hashes": root_program.hashes(),
            "semantic_trace_hash": intermediate_refinement_semantic_trace_hash(
                root_refinement
            ),
            "evaluation": _evaluation_payload(root_evaluated),
            "branch": branch.to_dict(),
        },
        "children": rows,
        "gate": gate,
        "elapsed_ns": time.perf_counter_ns() - started_ns,
        "performance_claimed": False,
    }
    artifact_dir = args.artifact_dir.resolve()
    evidence_path = artifact_dir / "pilot.json"
    _write_json(evidence_path, result)
    manifest = {
        "schema_version": "boundflow.objective-ancestral-hard-clause-pilot-manifest/v1",
        "files": {"pilot.json": _file_sha256(evidence_path)},
        "evidence_hash": _canonical_hash(result),
    }
    _write_json(artifact_dir / "manifest.json", manifest)
    print(_canonical_json({"status": "ok", **gate}))


def main() -> None:
    _run(_parse_args())


if __name__ == "__main__":
    main()
