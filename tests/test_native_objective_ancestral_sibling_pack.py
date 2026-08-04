"""Typed atomic sibling-packed objective-ancestral queue tests."""

# pylint: disable=missing-function-docstring,redefined-outer-name,duplicate-code

from dataclasses import replace

import pytest
import torch

from boundflow.ir.objective_ancestral_sibling_pack import (
    NativeObjectiveAncestralSiblingPackTaskKind,
)
from boundflow.ir.refinement import NativeIntermediateRefinementPolicyIR
from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime.native_alpha_beta_optimization_state import (
    NativeAlphaBetaOptimizerPolicy,
)
from boundflow.runtime.complete_verifier_query import CompleteVerifierQueryPolicy
from boundflow.runtime.native_candidate_search import (
    NativeProjectedGradientSearchPolicy,
)
from boundflow.runtime.native_intermediate_refinement import (
    compile_native_intermediate_refinement_program,
    execute_native_intermediate_refinement_program,
)
from boundflow.runtime.native_objective_ancestral_sibling_pack import (
    compile_native_objective_ancestral_sibling_pack_plan,
    execute_native_objective_ancestral_sibling_pack_queue,
)
from boundflow.runtime.native_objective_ancestral_sibling_pack_complete_query import (
    execute_native_objective_ancestral_sibling_pack_complete_query,
)
from boundflow.runtime.native_relu_split_bab_runtime import NativeReluSplitBabConfig
from boundflow.runtime.task_executor import InputSpec


def _fixture():
    module = BFTaskModule(
        tasks=[
            BoundTask(
                task_id="sibling-pack-toy",
                kind=TaskKind.INTERVAL_IBP,
                ops=[
                    TaskOp("linear", "linear1", ["input", "W1", "b1"], ["h1"]),
                    TaskOp("relu", "relu1", ["h1"], ["r1"]),
                    TaskOp("linear", "linear2", ["r1", "W2", "b2"], ["out"]),
                ],
                input_values=["input"],
                output_values=["out"],
            )
        ],
        entry_task_id="sibling-pack-toy",
        bindings={
            "params": {
                "W1": torch.tensor(
                    [
                        [1.0, -0.5],
                        [-0.25, 0.75],
                        [0.6, 0.8],
                        [-0.7, -0.4],
                        [0.9, 0.3],
                        [-0.4, 0.9],
                        [0.5, -0.8],
                        [-0.9, 0.6],
                    ]
                ),
                "b1": torch.tensor([0.1, -0.2, 0.0, 0.15, -0.1, 0.05, 0.0, 0.2]),
                "W2": torch.tensor(
                    [
                        [0.75, -1.0, 0.4, -0.3, 0.2, 0.5, -0.7, 0.6],
                        [-0.5, 0.25, -0.6, 0.9, -0.4, 0.3, 0.8, -0.2],
                    ]
                ),
                "b2": torch.tensor([0.15, -0.1]),
            }
        },
    )
    spec = InputSpec.box(
        value_name="input",
        lower=torch.tensor([[-0.3, -0.6]]),
        upper=torch.tensor([[0.7, 0.4]]),
    )
    objective = torch.tensor([[[1.0, -1.0]]])
    threshold = torch.tensor([-0.1])
    shared_program = compile_native_intermediate_refinement_program(
        module,
        spec,
        policy=NativeIntermediateRefinementPolicyIR(
            passes=1, max_neurons_per_relu=8, backward_chunk_size=8
        ),
        plan_id="sibling-pack-toy:shared",
    )
    shared = execute_native_intermediate_refinement_program(
        shared_program, module, spec
    )
    root_program = compile_native_intermediate_refinement_program(
        module,
        spec,
        policy=NativeIntermediateRefinementPolicyIR(
            passes=1,
            max_neurons_per_relu=8,
            backward_chunk_size=8,
            candidate_policy_id="objective_influence_width_per_relu_v1",
        ),
        plan_id="sibling-pack-toy:root",
        linear_spec_C=objective,
        source_refinement_execution=shared,
    )
    root = execute_native_intermediate_refinement_program(root_program, module, spec)
    optimizer = NativeAlphaBetaOptimizerPolicy(steps=1, lr=0.1)
    plan = compile_native_objective_ancestral_sibling_pack_plan(
        module,
        spec,
        linear_spec_C=objective,
        threshold=threshold,
        root_refinement=root,
        optimizer_policy=optimizer,
        plan_id="sibling-pack-toy",
    )
    execution = execute_native_objective_ancestral_sibling_pack_queue(
        plan,
        module,
        spec,
        linear_spec_C=objective,
        threshold=threshold,
        root_refinement=root,
        optimizer_policy=optimizer,
        query_id="sibling-pack-toy",
    )
    return module, spec, objective, threshold, root, optimizer, execution


@pytest.fixture(scope="module")
def execution_bundle():
    return _fixture()


def test_sibling_pack_lowers_projection_and_atomic_groups(execution_bundle) -> None:
    _module, _spec, _objective, _threshold, _root, _optimizer, execution = (
        execution_bundle
    )
    kinds = tuple(task.kind for task in execution.task_ir.tasks)

    assert kinds[:3] == (
        NativeObjectiveAncestralSiblingPackTaskKind.ADMIT_ROOT_SOURCE,
        NativeObjectiveAncestralSiblingPackTaskKind.PROJECT_OBJECTIVE,
        NativeObjectiveAncestralSiblingPackTaskKind.EVALUATE_ROOT,
    )
    assert kinds[-1] == NativeObjectiveAncestralSiblingPackTaskKind.EMIT_RESULT
    assert execution.sibling_groups
    assert len(execution.node_refinements) == len(execution.queue.trace.evaluations)
    assert len(execution.node_refinements) == 1 + 2 * len(execution.sibling_groups)
    assert all(
        stack.domain_batch_size == 2
        for stack in execution.queue.trace.native_stacks[1:]
    )
    assert execution.trace.performance_claimed is False


def test_sibling_pack_binds_each_group_to_one_parent_and_two_branches(
    execution_bundle,
) -> None:
    _module, _spec, _objective, _threshold, _root, _optimizer, execution = (
        execution_bundle
    )
    refinements = {item.node_id: item for item in execution.node_refinements}

    for group in execution.sibling_groups:
        assert group.child_branch_values == (-1, 1)
        assert all(
            refinements[node_id].parent_node_id == group.parent_node_id
            for node_id in group.child_node_ids
        )
        assert group.atomic_commit_hash != "0" * 64
        group.validate()


def test_sibling_pack_emit_depends_on_complete_committed_groups(
    execution_bundle,
) -> None:
    _module, _spec, _objective, _threshold, _root, _optimizer, execution = (
        execution_bundle
    )
    emit = execution.task_ir.tasks[-1]
    expected = tuple(
        task.task_id
        for task in execution.task_ir.tasks[:-1]
        if task.kind
        in {
            NativeObjectiveAncestralSiblingPackTaskKind.EVALUATE_ROOT,
            NativeObjectiveAncestralSiblingPackTaskKind.TRANSITION_QUEUE,
            NativeObjectiveAncestralSiblingPackTaskKind.EXECUTE_PACKED_EVALUATOR,
        }
    )
    assert emit.dependency_task_ids == expected

    tampered = replace(emit, dependency_task_ids=emit.dependency_task_ids[:-1])
    with pytest.raises(ValueError, match="emit dependencies differ"):
        replace(
            execution.task_ir, tasks=(*execution.task_ir.tasks[:-1], tampered)
        ).validate()


def test_sibling_pack_rejects_projection_tamper(execution_bundle) -> None:
    module, spec, objective, threshold, root, optimizer, execution = execution_bundle
    tampered = replace(execution.plan, evaluator_objective_hash="0" * 64)

    with pytest.raises(ValueError, match="plan/query binding differs"):
        execute_native_objective_ancestral_sibling_pack_queue(
            tampered,
            module,
            spec,
            linear_spec_C=objective,
            threshold=threshold,
            root_refinement=root,
            optimizer_policy=optimizer,
            query_id="sibling-pack-tampered",
        )


def test_sibling_pack_rejects_group_atomic_hash_tamper(execution_bundle) -> None:
    _module, _spec, _objective, _threshold, _root, _optimizer, execution = (
        execution_bundle
    )
    group = execution.sibling_groups[0]

    with pytest.raises(ValueError, match="group execution is invalid"):
        replace(group, atomic_commit_hash="0" * 64).validate()


def test_sibling_pack_complete_query_preserves_all_pending_on_initial_deadline(
    execution_bundle,
) -> None:
    module, spec, objective, _threshold, _root, optimizer, _execution = execution_bundle
    objectives = torch.cat((objective, -objective), dim=1)
    clock_values = iter((0, 60_000_000_000, 60_000_000_000))

    execution = execute_native_objective_ancestral_sibling_pack_complete_query(
        module,
        spec,
        linear_spec_C=objectives,
        thresholds=torch.tensor([-0.1, -0.2]),
        query_id="sibling-pack-complete-timeout",
        query_policy=CompleteVerifierQueryPolicy(timeout_ns=60_000_000_000),
        search_policy=NativeProjectedGradientSearchPolicy(steps=1, step_size=0.01),
        queue_config=NativeReluSplitBabConfig(
            max_nodes=31,
            max_depth=4,
            expansion_batch_size=1,
            max_eval_batch_size=2,
        ),
        optimizer_policy=optimizer,
        clock_ns=lambda: next(clock_values),
    )

    assert execution.trace.status == "unknown"
    assert execution.trace.reason == "query_deadline_exhausted"
    assert execution.trace.pending_clause_indices == (0, 1)
    assert not execution.trace.completed_clauses
