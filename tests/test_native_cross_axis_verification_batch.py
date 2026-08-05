"""Typed cross-axis scorer batch ownership and parity tests."""

# pylint: disable=missing-function-docstring,protected-access,duplicate-code
# pylint: disable=import-outside-toplevel,too-many-locals

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from boundflow.ir.bound import IntermediateBoundSource
from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime.native_alpha_beta_optimization_state import (
    NativeAlphaBetaOptimizerPolicy,
)
from boundflow.runtime.native_alpha_beta_optimizer_schedule import (
    compile_native_alpha_beta_optimizer_program,
    execute_native_alpha_beta_optimizer_program,
)
from boundflow.runtime.native_cross_axis_prevalidated_objective_branch import (
    NativeCrossAxisObjectiveBranchBinding,
    compile_native_cross_axis_prevalidated_objective_branch_batch,
    execute_native_cross_axis_prevalidated_objective_branch_batch,
)
from boundflow.runtime.native_objective_branch_score import (
    NativeObjectiveBranchPolicy,
)
from boundflow.runtime.native_prevalidated_objective_branch_score import (
    compile_native_prevalidated_objective_branch_program,
    execute_native_prevalidated_objective_branch_program,
)
from boundflow.runtime.task_executor import InputSpec


def _module() -> BFTaskModule:
    return BFTaskModule(
        tasks=[
            BoundTask(
                task_id="cross-axis-toy",
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
        entry_task_id="cross-axis-toy",
        bindings={
            "params": {
                "W1": torch.tensor([[1.0, -0.5], [-0.25, 0.75], [0.5, 0.5]]),
                "b1": torch.tensor([0.0, -0.1, 0.05]),
                "W2": torch.tensor([[0.75, -1.0, 0.5]]),
                "b2": torch.tensor([0.1]),
            }
        },
    )


def _programs():
    module = _module()
    input_spec = InputSpec.box(
        value_name="input",
        lower=torch.tensor([[-0.5, -0.4]]),
        upper=torch.tensor([[0.6, 0.7]]),
    )
    optimizer_policy = NativeAlphaBetaOptimizerPolicy(
        steps=1,
        lr=0.1,
        alpha_initialization_mode="adaptive",
    )
    branch_policy = NativeObjectiveBranchPolicy(
        candidates_per_relu=3,
        candidate_batch_size=4,
    )
    programs = []
    for ordinal, objective in enumerate(
        (torch.tensor([[1.0]]), torch.tensor([[-1.0]]))
    ):
        optimizer = compile_native_alpha_beta_optimizer_program(
            module,
            input_spec,
            linear_spec_C=objective,
            relu_split_state={"h1": torch.zeros((1, 3), dtype=torch.int8)},
            policy=optimizer_policy,
            program_id=f"cross-axis-toy:optimizer:{ordinal}",
        )
        selected = execute_native_alpha_beta_optimizer_program(
            optimizer,
            module,
            input_spec,
            linear_spec_C=objective,
        )
        programs.append(
            compile_native_prevalidated_objective_branch_program(
                module,
                input_spec,
                linear_spec_C=objective,
                relu_pre=optimizer.relu_pre,
                selected_state=selected.state,
                optimizer_policy=optimizer_policy,
                branch_policy=branch_policy,
                intermediate_bound_source=IntermediateBoundSource.LOCAL_FORWARD,
                plan_id=f"cross-axis-toy:branch:{ordinal}",
            )
        )
    return tuple(programs)


def _batch():
    programs = _programs()
    bindings = tuple(
        NativeCrossAxisObjectiveBranchBinding(
            clause_ordinal=ordinal,
            node_id=f"toy:c{ordinal}:n0",
            program=program,
        )
        for ordinal, program in enumerate(programs)
    )
    return programs, compile_native_cross_axis_prevalidated_objective_branch_batch(
        bindings,
        batch_id="cross-axis-toy:batch",
        max_child_domains=16,
    )


def test_cross_axis_batch_is_exactly_equal_to_two_serial_programs() -> None:
    programs, batch = _batch()
    control = tuple(
        execute_native_prevalidated_objective_branch_program(
            program, node_id=f"toy:c{ordinal}:n0"
        )
        for ordinal, program in enumerate(programs)
    )
    candidate = execute_native_cross_axis_prevalidated_objective_branch_batch(batch)

    assert batch.plan.clause_count == 2
    assert batch.plan.node_count == 2
    assert batch.plan.candidate_count == 6
    assert batch.plan.child_domain_count == 12
    assert candidate.trace.lower_launch_count == 1
    assert tuple(item.branch for item in candidate.executions) == tuple(
        item.branch for item in control
    )
    assert tuple(item.trace.scores for item in candidate.executions) == tuple(
        item.trace.scores for item in control
    )
    assert candidate.trace.segment_child_lower_hashes == tuple(
        item.trace.child_lower_hash for item in control
    )


def test_cross_axis_batch_physically_launches_lower_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import boundflow.runtime.native_cross_axis_prevalidated_objective_branch as runtime

    _programs_value, batch = _batch()
    original = runtime._evaluate_state
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime, "_evaluate_state", counted)
    execute_native_cross_axis_prevalidated_objective_branch_batch(batch)
    assert calls == 1


def test_cross_axis_segment_and_instance_tamper_fail_closed() -> None:
    _programs_value, batch = _batch()
    segments = list(batch.plan.segments)
    segments[1] = replace(segments[1], child_domain_offset=1)
    with pytest.raises(ValueError, match="segments are not packed"):
        replace(batch, plan=replace(batch.plan, segments=tuple(segments))).validate()

    with pytest.raises(ValueError, match="Instance IR differs"):
        replace(
            batch,
            instance=replace(batch.instance, semantic_token="0" * 64),
        ).validate()

    capsule_segments = list(batch.plan.segments)
    capsule_segments[0] = replace(capsule_segments[0], capsule_hash="0" * 64)
    with pytest.raises(
        ValueError, match="Instance IR differs|Task module is invalid|owner differs"
    ):
        replace(
            batch,
            plan=replace(batch.plan, segments=tuple(capsule_segments)),
        ).validate()

    with pytest.raises(ValueError, match="segments are not packed"):
        replace(
            batch,
            plan=replace(batch.plan, segments=tuple(reversed(batch.plan.segments))),
        ).validate()


def test_cross_axis_owner_and_capacity_fail_closed() -> None:
    programs = _programs()
    repeated_owner = (
        NativeCrossAxisObjectiveBranchBinding(0, "toy:n0", programs[0]),
        NativeCrossAxisObjectiveBranchBinding(1, "toy:n0", programs[1]),
    )
    with pytest.raises(ValueError, match="node identity repeats"):
        compile_native_cross_axis_prevalidated_objective_branch_batch(
            repeated_owner, batch_id="cross-axis-toy:repeated"
        )

    bindings = tuple(
        NativeCrossAxisObjectiveBranchBinding(index, f"toy:n{index}", program)
        for index, program in enumerate(programs)
    )
    with pytest.raises(ValueError, match="Plan IR is invalid"):
        compile_native_cross_axis_prevalidated_objective_branch_batch(
            bindings,
            batch_id="cross-axis-toy:too-small",
            max_child_domains=8,
        )


def test_cross_axis_production_queue_batches_each_sibling_pair_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import boundflow.runtime.native_cross_axis_prevalidated_objective_branch as runtime
    from boundflow.runtime.native_cross_axis_objective_branch_shared_production_queue import (
        execute_native_cross_axis_objective_branch_shared_production_queue,
    )
    from boundflow.runtime.native_parametric_optimizer import (
        NativeParametricOptimizerTemplateCache,
    )
    from boundflow.runtime.native_prevalidated_objective_branch_shared_production_queue import (
        execute_native_prevalidated_objective_branch_shared_production_queue,
    )
    from tests.test_native_objective_branch_scorer_ownership import _production_plan

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
        "query_id": "cross-axis:production",
        "clock_ns": lambda: 0,
    }
    control = execute_native_prevalidated_objective_branch_shared_production_queue(
        plan,
        module,
        spec,
        compiler_cache=NativeParametricOptimizerTemplateCache(),
        **common,
    )
    original = runtime._evaluate_state
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime, "_evaluate_state", counted)
    candidate = execute_native_cross_axis_objective_branch_shared_production_queue(
        plan,
        module,
        spec,
        compiler_cache=NativeParametricOptimizerTemplateCache(),
        clause_ordinal=2,
        **common,
    )

    def evaluation_semantics(item):
        value = item.to_dict()
        value.pop("batch_trace_hash")
        return value

    assert len(candidate.queue.trace.evaluations) == 31
    assert len(candidate.scorer_batches) == calls == 16
    assert [item.program.plan.node_count for item in candidate.scorer_batches] == [
        1,
        *([2] * 15),
    ]
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
    control_branches = dict(control.queue.objective_branch_executions)
    for node_id, execution in candidate.queue.objective_branch_executions:
        historical = control_branches[node_id]
        assert execution.branch == historical.branch
        assert execution.trace.scores == historical.trace.scores
        assert execution.trace.child_lower_hash == historical.trace.child_lower_hash
