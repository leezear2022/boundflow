"""Typed objective-root ancestral queue integration tests."""

from dataclasses import replace

import pytest
import torch

from boundflow.ir.objective_ancestral_queue import (
    ObjectiveAncestralQueueTaskKind,
)
from boundflow.ir.refinement import NativeIntermediateRefinementPolicyIR
from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime.native_alpha_beta_optimization_state import (
    NativeAlphaBetaOptimizerPolicy,
)
from boundflow.runtime.native_intermediate_refinement import (
    compile_native_intermediate_refinement_program,
    execute_native_intermediate_refinement_program,
)
from boundflow.runtime.native_objective_ancestral_queue import (
    compile_native_objective_ancestral_queue_plan,
    execute_native_objective_ancestral_queue,
)
from boundflow.runtime.task_executor import InputSpec


def _fixture() -> tuple[BFTaskModule, InputSpec, torch.Tensor, torch.Tensor]:
    module = BFTaskModule(
        tasks=[
            BoundTask(
                task_id="objective-ancestral-toy",
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
        entry_task_id="objective-ancestral-toy",
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
    return module, spec, torch.tensor([[[1.0, -1.0]]]), torch.tensor([-0.1])


def _root_refinement(module: BFTaskModule, spec: InputSpec, objective: torch.Tensor):
    shared_program = compile_native_intermediate_refinement_program(
        module,
        spec,
        policy=NativeIntermediateRefinementPolicyIR(
            passes=1, max_neurons_per_relu=128, backward_chunk_size=32
        ),
        plan_id="objective-ancestral-toy:shared",
    )
    shared = execute_native_intermediate_refinement_program(
        shared_program, module, spec
    )
    root_program = compile_native_intermediate_refinement_program(
        module,
        spec,
        policy=NativeIntermediateRefinementPolicyIR(
            passes=1,
            max_neurons_per_relu=128,
            backward_chunk_size=32,
            candidate_policy_id="objective_influence_width_per_relu_v1",
        ),
        plan_id="objective-ancestral-toy:root",
        linear_spec_C=objective,
        source_refinement_execution=shared,
    )
    return execute_native_intermediate_refinement_program(root_program, module, spec)


def _execute():
    module, spec, objective, threshold = _fixture()
    optimizer = NativeAlphaBetaOptimizerPolicy(steps=1, lr=0.1)
    root = _root_refinement(module, spec, objective)
    plan = compile_native_objective_ancestral_queue_plan(
        module,
        spec,
        linear_spec_C=objective,
        threshold=threshold,
        root_refinement=root,
        optimizer_policy=optimizer,
        plan_id="objective-ancestral-toy",
    )
    execution = execute_native_objective_ancestral_queue(
        plan,
        module,
        spec,
        linear_spec_C=objective,
        threshold=threshold,
        root_refinement=root,
        optimizer_policy=optimizer,
        query_id="objective-ancestral-toy",
    )
    execution.validate_against(
        module,
        spec,
        linear_spec_C=objective,
        threshold=threshold,
        root_refinement=root,
        optimizer_policy=optimizer,
    )
    return module, spec, objective, threshold, optimizer, root, execution


@pytest.fixture(scope="module")
def execution_bundle():
    return _execute()


def test_objective_ancestral_queue_lowers_committed_dynamic_schedule(
    execution_bundle,
) -> None:
    _module, _spec, _objective, _threshold, _optimizer, _root, execution = (
        execution_bundle
    )

    kinds = tuple(task.kind for task in execution.task_ir.tasks)
    assert kinds[:2] == (
        ObjectiveAncestralQueueTaskKind.ADMIT_ROOT_SOURCE,
        ObjectiveAncestralQueueTaskKind.EVALUATE_ROOT,
    )
    assert kinds[-1] == ObjectiveAncestralQueueTaskKind.EMIT_RESULT
    assert len(execution.node_refinements) == len(execution.queue.trace.evaluations)
    assert execution.trace.performance_claimed is False
    execution.schedule.validate_against(execution.task_ir)


def test_objective_ancestral_queue_emit_depends_on_complete_committed_proof(
    execution_bundle,
) -> None:
    _module, _spec, _objective, _threshold, _optimizer, _root, execution = (
        execution_bundle
    )
    emit = execution.task_ir.tasks[-1]
    expected = tuple(
        task.task_id
        for task in execution.task_ir.tasks[:-1]
        if task.kind
        in {
            ObjectiveAncestralQueueTaskKind.EVALUATE_ROOT,
            ObjectiveAncestralQueueTaskKind.EVALUATE_CHILD,
            ObjectiveAncestralQueueTaskKind.TRANSITION_QUEUE,
        }
    )
    assert emit.dependency_task_ids == expected

    tampered_emit = replace(emit, dependency_task_ids=emit.dependency_task_ids[:-1])
    tampered_module = replace(
        execution.task_ir,
        tasks=(*execution.task_ir.tasks[:-1], tampered_emit),
    )
    with pytest.raises(ValueError, match="emit dependencies differ"):
        tampered_module.validate()


def test_objective_ancestral_queue_binds_every_child_to_exact_parent(
    execution_bundle,
) -> None:
    _module, _spec, _objective, _threshold, _optimizer, _root, execution = (
        execution_bundle
    )
    refinements = {item.node_id: item for item in execution.node_refinements}

    assert len(refinements) > 1
    for refinement in execution.node_refinements[1:]:
        parent = refinements[refinement.parent_node_id]
        assert refinement.program.plan.source_refinement_plan_hash == (
            parent.program.plan.stable_hash()
        )
        assert refinement.program.plan.source_refinement_semantic_trace_hash is not None
        assert (
            refinement.semantic_dict()["source_consumption"] == "sound_constraint_only"
        )


def test_objective_ancestral_queue_rejects_plan_source_tamper(execution_bundle) -> None:
    module, spec, objective, threshold, optimizer, root, execution = execution_bundle
    tampered = replace(execution.plan, root_refinement_plan_hash="0" * 64)

    with pytest.raises(ValueError, match="plan/query binding differs"):
        execute_native_objective_ancestral_queue(
            tampered,
            module,
            spec,
            linear_spec_C=objective,
            threshold=threshold,
            root_refinement=root,
            optimizer_policy=optimizer,
            query_id="objective-ancestral-tampered",
        )


def test_objective_ancestral_queue_rejects_aggregate_signature_tamper(
    execution_bundle,
) -> None:
    _module, _spec, _objective, _threshold, _optimizer, _root, execution = (
        execution_bundle
    )
    tampered = replace(execution.trace, fallback_reason="invented")

    with pytest.raises(ValueError, match="aggregate trace differs"):
        tampered.validate_against(
            execution.plan, execution.task_ir, execution.schedule, execution.queue
        )
