"""Compile-owned objective-branch scorer capsule and parity tests."""

# pylint: disable=missing-function-docstring,import-outside-toplevel
# pylint: disable=protected-access,no-member,too-many-locals,duplicate-code

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from boundflow.domains.interval import IntervalState
from boundflow.ir.bound import IntermediateBoundSource
from boundflow.ir.objective_branch_shared_evaluator import (
    NativeObjectiveBranchSharedPlanIR,
)
from boundflow.ir.refinement import NativeIntermediateRefinementPolicyIR
from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime.native_alpha_beta_optimization_state import (
    NativeAlphaBetaOptimizerPolicy,
)
from boundflow.runtime.native_alpha_beta_optimizer_schedule import (
    compile_native_alpha_beta_optimizer_program,
    execute_native_alpha_beta_optimizer_program,
)
from boundflow.runtime.native_objective_branch_score import (
    NativeObjectiveBranchPolicy,
    compile_native_objective_branch_program,
    execute_native_objective_branch_program,
)
from boundflow.runtime.native_objective_branch_shared_evaluator import (
    compile_native_objective_branch_shared_plan,
)
from boundflow.runtime.native_objective_branch_shared_production_queue import (
    execute_native_objective_branch_shared_production_queue,
)
from boundflow.runtime.native_parametric_optimizer import (
    NativeParametricOptimizerTemplateCache,
)
from boundflow.runtime.native_prevalidated_objective_branch_score import (
    compile_native_prevalidated_objective_branch_program,
    execute_native_prevalidated_objective_branch_program,
)
from boundflow.runtime.native_prevalidated_objective_branch_shared_production_queue import (
    execute_native_prevalidated_objective_branch_shared_production_queue,
)
from boundflow.runtime.native_intermediate_refinement import (
    compile_native_intermediate_refinement_program,
    execute_native_intermediate_refinement_program,
)
from boundflow.runtime.task_executor import InputSpec


def _module() -> BFTaskModule:
    return BFTaskModule(
        tasks=[
            BoundTask(
                task_id="scorer-ownership-toy",
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
        entry_task_id="scorer-ownership-toy",
        bindings={
            "params": {
                "W1": torch.tensor([[1.0, -0.5], [-0.25, 0.75], [0.5, 0.5]]),
                "b1": torch.tensor([0.0, -0.1, 0.05]),
                "W2": torch.tensor([[0.75, -1.0, 0.5]]),
                "b2": torch.tensor([0.1]),
            }
        },
    )


def _inputs():
    module = _module()
    spec = InputSpec.box(
        value_name="input",
        lower=torch.tensor([[-0.5, -0.4]]),
        upper=torch.tensor([[0.6, 0.7]]),
    )
    objective = torch.tensor([[1.0]])
    optimizer_policy = NativeAlphaBetaOptimizerPolicy(
        steps=1,
        lr=0.1,
        alpha_initialization_mode="adaptive",
    )
    split = {"h1": torch.zeros((1, 3), dtype=torch.int8)}
    optimizer = compile_native_alpha_beta_optimizer_program(
        module,
        spec,
        linear_spec_C=objective,
        relu_split_state=split,
        policy=optimizer_policy,
        program_id="scorer-ownership-toy:optimizer",
    )
    selected = execute_native_alpha_beta_optimizer_program(
        optimizer, module, spec, linear_spec_C=objective
    )
    branch_policy = NativeObjectiveBranchPolicy(
        candidates_per_relu=3,
        candidate_batch_size=2,
    )
    return (
        module,
        spec,
        objective,
        optimizer.relu_pre,
        selected.state,
        optimizer_policy,
        branch_policy,
    )


def _compile_new():
    values = _inputs()
    program = compile_native_prevalidated_objective_branch_program(
        values[0],
        values[1],
        linear_spec_C=values[2],
        relu_pre=values[3],
        selected_state=values[4],
        optimizer_policy=values[5],
        branch_policy=values[6],
        intermediate_bound_source=IntermediateBoundSource.LOCAL_FORWARD,
        plan_id="scorer-ownership-toy:new",
    )
    return (*values, program)


def test_candidate_enumeration_occurs_once_at_compile_and_never_at_execute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import boundflow.runtime.native_prevalidated_objective_branch_score as scorer

    original = scorer._enumerate_candidates
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(scorer, "_enumerate_candidates", counted)
    values = _inputs()
    program = compile_native_prevalidated_objective_branch_program(
        values[0],
        values[1],
        linear_spec_C=values[2],
        relu_pre=values[3],
        selected_state=values[4],
        optimizer_policy=values[5],
        branch_policy=values[6],
        intermediate_bound_source=IntermediateBoundSource.LOCAL_FORWARD,
        plan_id="scorer-ownership-toy:counted",
    )
    assert calls == 1
    assert program.capsule.compile_enumeration_count == 1
    assert program.capsule.execute_enumeration_count == 0

    def forbidden(*_args, **_kwargs):
        raise AssertionError("execute regenerated compile-owned candidates")

    monkeypatch.setattr(scorer, "_enumerate_candidates", forbidden)
    execution = execute_native_prevalidated_objective_branch_program(
        program, node_id="toy:n0"
    )
    execution.validate()
    execution.trace.to_dict(program=execution.program)


def test_prevalidated_scorer_is_exactly_equal_to_historical_scorer() -> None:
    (
        module,
        spec,
        objective,
        relu_pre,
        selected_state,
        optimizer_policy,
        branch_policy,
        candidate,
    ) = _compile_new()
    control = compile_native_objective_branch_program(
        module,
        spec,
        linear_spec_C=objective,
        relu_pre=relu_pre,
        selected_state=selected_state,
        optimizer_policy=optimizer_policy,
        branch_policy=branch_policy,
        intermediate_bound_source=IntermediateBoundSource.LOCAL_FORWARD,
        plan_id="scorer-ownership-toy:control",
    )
    control_execution = execute_native_objective_branch_program(
        control, node_id="toy:n0"
    )
    candidate_execution = execute_native_prevalidated_objective_branch_program(
        candidate, node_id="toy:n0"
    )

    assert candidate.plan.candidates == control.plan.candidates
    assert candidate_execution.branch == control_execution.branch
    assert candidate_execution.trace.scores == control_execution.trace.scores
    assert (
        candidate_execution.trace.child_lower_hash
        == control_execution.trace.child_lower_hash
    )
    assert (
        candidate_execution.trace.selected_candidate_ordinal
        == control_execution.trace.selected_candidate_ordinal
    )


def test_capsule_task_and_tensor_tamper_fail_closed() -> None:
    *_values, program = _compile_new()
    with pytest.raises(ValueError, match="capsule"):
        replace(
            program,
            capsule=replace(program.capsule, candidate_count=2),
        ).validate()
    tasks = list(program.task_module.tasks)
    tasks[0] = replace(tasks[0], input_value_ids=("branch.relu_pre",))
    with pytest.raises(ValueError, match="candidate ownership"):
        replace(
            program,
            task_module=replace(program.task_module, tasks=tuple(tasks)),
        ).validate()
    relu_pre = dict(program.relu_pre)
    original = relu_pre["h1"]
    relu_pre["h1"] = IntervalState(
        lower=original.lower - 0.01,
        upper=original.upper,
    )
    with pytest.raises(ValueError, match="program identity"):
        replace(program, relu_pre=relu_pre).validate()


def test_capsule_and_trace_are_stable_and_performance_neutral() -> None:
    *_values, program = _compile_new()
    left = execute_native_prevalidated_objective_branch_program(
        program, node_id="toy:n0"
    )
    right = execute_native_prevalidated_objective_branch_program(
        program, node_id="toy:n0"
    )

    assert program.task_module.tasks[0].input_value_ids == ("branch.plan.candidates",)
    assert program.capsule.candidate_source == "plan_owned_immutable"
    assert program.capsule.performance_claimed is False
    assert left.trace.stable_hash(program=left.program) == right.trace.stable_hash(
        program=right.program
    )


def _production_module() -> BFTaskModule:
    module = _module()
    module.bindings["params"] = {
        "W1": torch.tensor(
            [
                [1.0, -0.5],
                [-0.25, 0.75],
                [0.6, 0.8],
                [-0.7, -0.4],
                [0.9, 0.3],
                [-0.4, 0.9],
                [0.5, -0.8],
                [-0.9, 0.2],
            ]
        ),
        "b1": torch.zeros(8),
        "W2": torch.tensor(
            [
                [0.75, -1.0, 0.4, -0.3, 0.5, -0.6, 0.8, -0.2],
                [-0.5, 0.25, -0.6, 0.9, -0.7, 0.4, -0.3, 0.65],
            ]
        ),
        "b2": torch.tensor([0.15, -0.1]),
    }
    return module


def _production_plan() -> tuple[
    BFTaskModule,
    InputSpec,
    torch.Tensor,
    torch.Tensor,
    object,
    NativeAlphaBetaOptimizerPolicy,
    NativeObjectiveBranchPolicy,
    NativeObjectiveBranchSharedPlanIR,
]:
    module = _production_module()
    spec = InputSpec.box(
        value_name="input",
        lower=torch.tensor([[-0.3, -0.6]]),
        upper=torch.tensor([[0.7, 0.4]]),
    )
    objective = torch.tensor([[[1.0, -1.0]]])
    threshold = torch.tensor([1e6])
    shared_program = compile_native_intermediate_refinement_program(
        module,
        spec,
        policy=NativeIntermediateRefinementPolicyIR(
            passes=1, max_neurons_per_relu=4, backward_chunk_size=4
        ),
        plan_id="scorer-ownership:shared",
    )
    shared = execute_native_intermediate_refinement_program(
        shared_program, module, spec
    )
    root_program = compile_native_intermediate_refinement_program(
        module,
        spec,
        policy=NativeIntermediateRefinementPolicyIR(
            passes=1,
            max_neurons_per_relu=4,
            backward_chunk_size=4,
            candidate_policy_id="objective_influence_width_per_relu_v1",
        ),
        plan_id="scorer-ownership:root",
        linear_spec_C=objective,
        source_refinement_execution=shared,
    )
    root = execute_native_intermediate_refinement_program(root_program, module, spec)
    optimizer_policy = NativeAlphaBetaOptimizerPolicy(steps=1, lr=0.1)
    branch_policy = NativeObjectiveBranchPolicy()
    plan = compile_native_objective_branch_shared_plan(
        module,
        spec,
        linear_spec_C=objective,
        threshold=threshold,
        root_refinement=root,
        optimizer_policy=optimizer_policy,
        branch_policy=branch_policy,
        plan_id="scorer-ownership:production",
    )
    return (
        module,
        spec,
        objective,
        threshold,
        root,
        optimizer_policy,
        branch_policy,
        plan,
    )


def test_production_queue_has_exact_parity_and_one_enumeration_per_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import boundflow.runtime.native_prevalidated_objective_branch_score as scorer

    (
        module,
        spec,
        objective,
        threshold,
        root,
        optimizer_policy,
        branch_policy,
        plan,
    ) = _production_plan()
    common = {
        "linear_spec_C": objective,
        "threshold": threshold,
        "root_refinement": root,
        "optimizer_policy": optimizer_policy,
        "branch_policy": branch_policy,
        "query_id": "scorer-ownership:production",
        "clock_ns": lambda: 0,
    }
    control = execute_native_objective_branch_shared_production_queue(
        plan,
        module,
        spec,
        compiler_cache=NativeParametricOptimizerTemplateCache(),
        **common,
    )
    original = scorer._enumerate_candidates
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(scorer, "_enumerate_candidates", counted)
    candidate = execute_native_prevalidated_objective_branch_shared_production_queue(
        plan,
        module,
        spec,
        compiler_cache=NativeParametricOptimizerTemplateCache(),
        **common,
    )
    control_branches = dict(control.queue.objective_branch_executions)
    candidate_branches = dict(candidate.queue.objective_branch_executions)

    assert len(candidate.queue.trace.evaluations) == 31
    assert len(candidate_branches) == calls == 31

    def evaluation_semantics(item):
        value = item.to_dict()
        value.pop("batch_trace_hash")
        return value

    assert tuple(
        evaluation_semantics(item) for item in candidate.queue.trace.evaluations
    ) == tuple(evaluation_semantics(item) for item in control.queue.trace.evaluations)
    assert tuple(item.to_dict() for item in candidate.queue.trace.decisions) == tuple(
        item.to_dict() for item in control.queue.trace.decisions
    )
    assert [state.stable_hash() for _, state in candidate.queue.selected_states] == [
        state.stable_hash() for _, state in control.queue.selected_states
    ]
    assert [item.semantic_dict() for item in candidate.node_refinements] == [
        item.semantic_dict() for item in control.node_refinements
    ]
    for node_id, execution in candidate_branches.items():
        historical = control_branches[node_id]
        assert execution.branch == historical.branch
        assert execution.trace.scores == historical.trace.scores
        assert execution.trace.child_lower_hash == historical.trace.child_lower_hash
        assert execution.program.capsule.compile_enumeration_count == 1
        assert execution.program.capsule.execute_enumeration_count == 0
