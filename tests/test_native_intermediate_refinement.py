"""Contract, soundness, and execution tests for native refinement IR."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from boundflow.domains.interval import IntervalState
from boundflow.ir.refinement import (
    ExternalIntermediateConstraintSeedIR,
    IntermediateRefinementTaskKind,
    NativeIntermediateRefinementMultiPassPolicyIR,
    NativeIntermediateRefinementPolicyIR,
)
from boundflow.ir.bound import IntermediateBoundSource, ReluRelaxationAttrs
from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime.native_alpha_beta_optimization_state import (
    NativeAlphaBetaOptimizerPolicy,
)
from boundflow.runtime.native_alpha_beta_optimizer_schedule import (
    compile_native_alpha_beta_optimizer_program,
    execute_native_alpha_beta_optimizer_program,
)
from boundflow.runtime.crown_ibp import _forward_ibp_trace_mlp
from boundflow.runtime.native_intermediate_refinement import (
    NativeExternalIntermediateConstraintSeed,
    NativeIntermediateRefinementProgram,
    compile_native_intermediate_refinement_program,
    execute_native_intermediate_refinement_program,
    intermediate_bounds_hash,
)
from boundflow.runtime.task_executor import InputSpec


def _dependency_mlp() -> BFTaskModule:
    return BFTaskModule(
        tasks=[
            BoundTask(
                task_id="refinement-test",
                kind=TaskKind.INTERVAL_IBP,
                ops=[
                    TaskOp("linear", "mirror", ["input", "W1", "b1"], ["h1"]),
                    TaskOp("relu", "relu1", ["h1"], ["r1"]),
                    TaskOp("linear", "join", ["r1", "W2", "b2"], ["h2"]),
                    TaskOp("relu", "relu2", ["h2"], ["r2"]),
                    TaskOp("linear", "output", ["r2", "W3", "b3"], ["out"]),
                ],
                input_values=["input"],
                output_values=["out"],
            )
        ],
        entry_task_id="refinement-test",
        bindings={
            "params": {
                "W1": torch.tensor([[1.0], [-1.0]]),
                "b1": torch.zeros(2),
                "W2": torch.tensor([[1.0, 1.0]]),
                "b2": torch.tensor([-1.0]),
                "W3": torch.ones(1, 1),
                "b3": torch.zeros(1),
            }
        },
    )


def _input_spec() -> InputSpec:
    return InputSpec.box(
        value_name="input",
        lower=torch.tensor([[-1.0]]),
        upper=torch.tensor([[1.0]]),
    )


def _competing_width_influence_mlp() -> BFTaskModule:
    return BFTaskModule(
        tasks=[
            BoundTask(
                task_id="objective-refinement-test",
                kind=TaskKind.INTERVAL_IBP,
                ops=[
                    TaskOp("linear", "scaled", ["input", "W1", "b1"], ["h1"]),
                    TaskOp("relu", "relu1", ["h1"], ["r1"]),
                    TaskOp("linear", "output", ["r1", "W2", "b2"], ["out"]),
                ],
                input_values=["input"],
                output_values=["out"],
            )
        ],
        entry_task_id="objective-refinement-test",
        bindings={
            "params": {
                "W1": torch.tensor([[2.0, 0.0], [0.0, 1.0]]),
                "b1": torch.zeros(2),
                "W2": torch.tensor([[0.01, 1.0]]),
                "b2": torch.zeros(1),
            }
        },
    )


def _two_dimensional_input_spec() -> InputSpec:
    return InputSpec.box(
        value_name="input",
        lower=torch.tensor([[-1.0, -1.0]]),
        upper=torch.tensor([[1.0, 1.0]]),
    )


def _policy(*, passes: int = 1) -> NativeIntermediateRefinementPolicyIR:
    return NativeIntermediateRefinementPolicyIR(
        passes=passes,
        max_neurons_per_relu=2,
        backward_chunk_size=1,
    )


def _external_seed(
    module: BFTaskModule, spec: InputSpec
) -> NativeExternalIntermediateConstraintSeed:
    source_program = compile_native_intermediate_refinement_program(
        module,
        spec,
        policy=_policy(),
        plan_id="external-seed-source-scope",
    )
    source = execute_native_intermediate_refinement_program(
        source_program, module, spec
    )
    return NativeExternalIntermediateConstraintSeed(
        ir=ExternalIntermediateConstraintSeedIR(
            seed_id="external-seed:test",
            provider="test-external-verifier",
            primal_graph_hash=source_program.plan.primal_graph_hash,
            input_bounds_hash=source_program.plan.input_bounds_hash,
            external_intermediate_bounds_hash="a" * 64,
            bound_intermediate_constraints_hash=intermediate_bounds_hash(
                source.relu_pre
            ),
            source_artifact_manifest_hash="b" * 64,
            source_artifact_payload_hash="c" * 64,
            source_model_hash="d" * 64,
            source_property_hash="e" * 64,
            source_objective_set_hash="f" * 64,
        ),
        constraints=source.relu_pre,
    )


def test_refinement_ir_is_unrolled_and_cross_linked() -> None:
    module = _dependency_mlp()
    spec = _input_spec()
    program = compile_native_intermediate_refinement_program(
        module,
        spec,
        policy=_policy(passes=2),
        plan_id="test-refinement",
    )

    assert len(program.plan.targets) == 3
    assert len(program.task_module.tasks) == 10
    assert len(program.schedule.actions) == 10
    assert [task.kind for task in program.task_module.tasks].count(
        IntermediateRefinementTaskKind.BACKWARD_SELECTED
    ) == 2
    assert program.hashes() == program.hashes()
    program.validate(module, spec)


def test_typed_multi_pass_reselects_disjoint_targets_and_is_schedule_visible() -> None:
    module = _dependency_mlp()
    spec = _input_spec()
    multi_pass = NativeIntermediateRefinementMultiPassPolicyIR()
    program = compile_native_intermediate_refinement_program(
        module,
        spec,
        policy=_policy(passes=2),
        plan_id="test-typed-multi-pass",
        multi_pass_policy=multi_pass,
    )
    execution = execute_native_intermediate_refinement_program(program, module, spec)

    assert program.plan.multi_pass_policy == multi_pass
    assert len(program.task_module.tasks) == len(program.schedule.actions) == 12
    assert [
        task.pass_index
        for task in program.task_module.tasks
        if task.kind == IntermediateRefinementTaskKind.SELECT_TARGETS
    ] == [0, 1]
    decisions = [trace.selection_decision for trace in execution.trace.pass_traces]
    assert all(decision is not None for decision in decisions)
    assert [
        decision.pass_target_cap_per_relu for decision in decisions if decision
    ] == [
        1,
        1,
    ]
    assert decisions[0].prior_selected_target_count == 0
    assert (
        decisions[1].prior_selected_target_count == decisions[0].selected_target_count
    )
    assert decisions[1].prior_target_ledger_hash == (
        decisions[0].result_target_ledger_hash
    )
    assert decisions[1].cumulative_selected_target_count > (
        decisions[0].cumulative_selected_target_count
    )
    execution.validate(module, spec)


def test_typed_multi_pass_stops_when_no_unseen_target_and_tamper_fails_closed() -> None:
    module = _competing_width_influence_mlp()
    spec = _two_dimensional_input_spec()
    multi_pass = NativeIntermediateRefinementMultiPassPolicyIR()
    policy = NativeIntermediateRefinementPolicyIR(
        passes=2,
        max_neurons_per_relu=4,
        backward_chunk_size=2,
    )
    program = compile_native_intermediate_refinement_program(
        module,
        spec,
        policy=policy,
        plan_id="test-typed-multi-pass-stop",
        multi_pass_policy=multi_pass,
    )
    execution = execute_native_intermediate_refinement_program(program, module, spec)
    stopped = execution.trace.pass_traces[1]
    decision = stopped.selection_decision

    assert decision is not None
    assert decision.continuation is False
    assert decision.termination_reason == "no_unseen_eligible_targets"
    assert decision.selected_target_count == 0
    assert stopped.input_bounds_hash == stopped.output_bounds_hash

    changed_decision = replace(decision, result_target_ledger_hash="f" * 64)
    changed_passes = list(execution.trace.pass_traces)
    changed_passes[1] = replace(stopped, selection_decision=changed_decision)
    with pytest.raises(ValueError, match="selection lineage differs"):
        replace(execution.trace, pass_traces=tuple(changed_passes)).validate(
            program=program
        )


def test_selected_intermediate_crown_tightens_and_is_sound() -> None:
    module = _dependency_mlp()
    spec = _input_spec()
    _initial_env, initial_pre = _forward_ibp_trace_mlp(module, spec)
    assert initial_pre["h2"].lower.item() == pytest.approx(-1.0)
    assert initial_pre["h2"].upper.item() == pytest.approx(1.0)

    program = compile_native_intermediate_refinement_program(
        module,
        spec,
        policy=_policy(),
        plan_id="test-refinement",
    )
    execution = execute_native_intermediate_refinement_program(program, module, spec)

    assert execution.relu_pre["h2"].lower.item() == pytest.approx(-1.0)
    assert execution.relu_pre["h2"].upper.item() == pytest.approx(0.0)
    assert execution.trace.pass_traces[0].tightened_neuron_count >= 1
    assert execution.trace.pass_traces[0].upper_improvement_max == pytest.approx(1.0)
    assert len(execution.trace.action_traces) == len(program.schedule.actions)

    for x_value in torch.linspace(-1.0, 1.0, steps=17):
        h2 = torch.abs(x_value) - 1.0
        assert execution.relu_pre["h2"].lower.item() <= h2.item()
        assert h2.item() <= execution.relu_pre["h2"].upper.item() + 1e-7


def test_ancestral_source_execution_is_first_class_and_monotonic() -> None:
    module = _dependency_mlp()
    spec = _input_spec()
    root_program = compile_native_intermediate_refinement_program(
        module,
        spec,
        policy=_policy(),
        plan_id="ancestral-root",
    )
    root = execute_native_intermediate_refinement_program(root_program, module, spec)
    child_split = {
        name: value.detach().clone() for name, value in root_program.split_state.items()
    }
    child_split["h1"][0, 0] = 1
    child_program = compile_native_intermediate_refinement_program(
        module,
        spec,
        policy=_policy(),
        plan_id="ancestral-child",
        relu_split_state=child_split,
        source_refinement_execution=root,
    )

    assert child_program.plan.source_intermediate_constraints_hash == (
        intermediate_bounds_hash(root.relu_pre)
    )
    assert child_program.plan.source_refinement_plan_hash == (
        root.program.plan.stable_hash()
    )
    assert child_program.plan.source_refinement_semantic_trace_hash is not None
    assert child_program.task_module.tasks[0].input_value_ids == (
        "refine.module",
        "refine.input",
        "refine.split_state",
        "refine.source_intermediate_constraints",
    )
    _local_env, local_pre = _forward_ibp_trace_mlp(
        module, spec, relu_split_state=child_split
    )
    for name, local in local_pre.items():
        initial = child_program.initial_relu_pre[name]
        assert bool((initial.lower >= local.lower).all())
        assert bool((initial.upper <= local.upper).all())

    child = execute_native_intermediate_refinement_program(child_program, module, spec)
    first_action_inputs = dict(child.trace.action_traces[0].input_hashes)
    assert first_action_inputs["refine.source_intermediate_constraints"] == (
        child_program.plan.source_intermediate_constraints_hash
    )
    for name, initial in child_program.initial_relu_pre.items():
        final = child.relu_pre[name]
        assert bool((final.lower >= initial.lower).all())
        assert bool((final.upper <= initial.upper).all())


def test_ancestral_source_constraint_tampering_fails_closed() -> None:
    module = _dependency_mlp()
    spec = _input_spec()
    root_program = compile_native_intermediate_refinement_program(
        module,
        spec,
        policy=_policy(),
        plan_id="ancestral-tamper-root",
    )
    root = execute_native_intermediate_refinement_program(root_program, module, spec)
    child = compile_native_intermediate_refinement_program(
        module,
        spec,
        policy=_policy(),
        plan_id="ancestral-tamper-child",
        source_refinement_execution=root,
    )
    assert child.source_intermediate_constraints is not None
    changed_constraints = dict(child.source_intermediate_constraints)
    h2 = changed_constraints["h2"]
    changed_constraints["h2"] = IntervalState(
        lower=h2.lower,
        upper=h2.upper - 0.1,
    )
    with pytest.raises(ValueError, match="source constraints differ"):
        replace(child, source_intermediate_constraints=changed_constraints).validate(
            module, spec
        )

    with pytest.raises(ValueError, match="program identity|source execution"):
        compile_native_intermediate_refinement_program(
            _competing_width_influence_mlp(),
            _two_dimensional_input_spec(),
            policy=_policy(),
            plan_id="ancestral-wrong-source-module",
            source_refinement_execution=root,
        )


def test_external_constraint_seed_is_typed_lowered_and_trace_bound() -> None:
    module = _dependency_mlp()
    spec = _input_spec()
    seed = _external_seed(module, spec)
    program = compile_native_intermediate_refinement_program(
        module,
        spec,
        policy=_policy(),
        plan_id="external-seeded-refinement",
        external_constraint_seed=seed,
    )

    assert program.plan.external_constraint_seed == seed.ir
    assert program.plan.source_intermediate_constraints_hash is None
    assert program.task_module.tasks[0].input_value_ids == (
        "refine.module",
        "refine.input",
        "refine.split_state",
        "refine.external_constraint_seed",
    )
    execution = execute_native_intermediate_refinement_program(program, module, spec)
    assert (
        dict(execution.trace.action_traces[0].input_hashes)[
            "refine.external_constraint_seed"
        ]
        == seed.stable_hash()
    )
    assert (
        execution.program.plan.to_dict()["external_constraint_seed"]["semantics_owner"]
        == "external_verifier"
    )


def test_external_constraint_seed_tamper_scope_and_conflict_fail_closed() -> None:
    module = _dependency_mlp()
    spec = _input_spec()
    seed = _external_seed(module, spec)
    changed_ir = replace(seed.ir, source_artifact_payload_hash="0" * 64)
    changed_seed = replace(seed, ir=changed_ir)
    program = compile_native_intermediate_refinement_program(
        module,
        spec,
        policy=_policy(),
        plan_id="external-seed-tamper",
        external_constraint_seed=changed_seed,
    )
    assert (
        program.plan.stable_hash()
        != compile_native_intermediate_refinement_program(
            module,
            spec,
            policy=_policy(),
            plan_id="external-seed-tamper",
            external_constraint_seed=seed,
        ).plan.stable_hash()
    )

    invalid_content_ir = replace(seed.ir, bound_intermediate_constraints_hash="0" * 64)
    with pytest.raises(ValueError, match="seed content differs"):
        compile_native_intermediate_refinement_program(
            module,
            spec,
            policy=_policy(),
            plan_id="external-seed-content-tamper",
            external_constraint_seed=replace(seed, ir=invalid_content_ir),
        )

    widened = dict(seed.constraints)
    first_name = next(iter(widened))
    first = widened[first_name]
    widened[first_name] = IntervalState(
        lower=first.lower - 0.1,
        upper=first.upper,
    )
    widened_seed = NativeExternalIntermediateConstraintSeed(
        ir=replace(
            seed.ir,
            bound_intermediate_constraints_hash=intermediate_bounds_hash(widened),
        ),
        constraints=widened,
    )
    with pytest.raises(ValueError, match="external intermediate constraint seed"):
        widened_seed.validate(module, spec)

    source_program = compile_native_intermediate_refinement_program(
        module, spec, policy=_policy(), plan_id="external-seed-conflict-source"
    )
    source = execute_native_intermediate_refinement_program(
        source_program, module, spec
    )
    with pytest.raises(ValueError, match="sources conflict"):
        compile_native_intermediate_refinement_program(
            module,
            spec,
            policy=_policy(),
            plan_id="external-seed-conflict",
            source_refinement_execution=source,
            external_constraint_seed=seed,
        )


def test_refinement_rejects_source_and_schedule_tamper() -> None:
    module = _dependency_mlp()
    spec = _input_spec()
    program = compile_native_intermediate_refinement_program(
        module,
        spec,
        policy=_policy(),
        plan_id="test-refinement",
    )

    changed_source = NativeIntermediateRefinementProgram(
        plan=replace(program.plan, input_bounds_hash="0" * 64),
        task_module=program.task_module,
        schedule=program.schedule,
        initial_interval_env=program.initial_interval_env,
        initial_relu_pre=program.initial_relu_pre,
        split_state=program.split_state,
    )
    with pytest.raises(ValueError, match="Task IR|Schedule IR|program identity"):
        changed_source.validate(module, spec)

    action = program.schedule.actions[3]
    changed_action = replace(action, task_id="tampered.task")
    changed_schedule = replace(
        program.schedule,
        actions=(
            *program.schedule.actions[:3],
            changed_action,
            *program.schedule.actions[4:],
        ),
    )
    with pytest.raises(ValueError, match="Schedule/Task linkage"):
        changed_schedule.validate(plan=program.plan, task_module=program.task_module)


def test_native_refined_provenance_reaches_optimizer_and_bound_ir() -> None:
    module = _dependency_mlp()
    spec = _input_spec()
    refinement = compile_native_intermediate_refinement_program(
        module,
        spec,
        policy=_policy(),
        plan_id="test-refinement",
    )
    execution = execute_native_intermediate_refinement_program(refinement, module, spec)
    splits = {
        name: torch.zeros_like(value.lower, dtype=torch.int8)
        for name, value in execution.relu_pre.items()
    }
    optimizer = compile_native_alpha_beta_optimizer_program(
        module,
        spec,
        linear_spec_C=torch.ones(1, 1),
        relu_split_state=splits,
        policy=NativeAlphaBetaOptimizerPolicy(steps=0, lr=0.1),
        program_id="native-refined-optimizer",
        relu_pre_override=execution.relu_pre,
        intermediate_bound_source=IntermediateBoundSource.NATIVE_REFINED,
    )
    result = execute_native_alpha_beta_optimizer_program(
        optimizer,
        module,
        spec,
        linear_spec_C=torch.ones(1, 1),
    )

    assert result.bounds.lower.shape == (1, 1)
    relu_attrs = [
        op.attrs
        for op in optimizer.source_compilation.source_bound_module.graph.ops
        if isinstance(op.attrs, ReluRelaxationAttrs)
    ]
    assert relu_attrs
    assert all(
        attrs.intermediate_bound_source == IntermediateBoundSource.NATIVE_REFINED
        for attrs in relu_attrs
    )


def test_objective_directed_policy_changes_target_and_has_exact_ir_dependency() -> None:
    module = _competing_width_influence_mlp()
    spec = _two_dimensional_input_spec()
    width_program = compile_native_intermediate_refinement_program(
        module,
        spec,
        policy=NativeIntermediateRefinementPolicyIR(
            passes=1,
            max_neurons_per_relu=1,
            backward_chunk_size=1,
        ),
        plan_id="width-directed",
    )
    objective_program = compile_native_intermediate_refinement_program(
        module,
        spec,
        policy=NativeIntermediateRefinementPolicyIR(
            passes=1,
            max_neurons_per_relu=1,
            backward_chunk_size=1,
            candidate_policy_id="objective_influence_width_per_relu_v1",
        ),
        plan_id="objective-directed",
        linear_spec_C=torch.ones(1, 1),
    )

    assert width_program.plan.targets[0].neuron_index == 0
    target = objective_program.plan.targets[0]
    assert target.neuron_index == 1
    assert target.objective_influence == pytest.approx(1.0)
    assert target.selection_score == pytest.approx(2.0)
    assert objective_program.plan.objective_hash is not None
    assert objective_program.task_module.tasks[2].input_value_ids == (
        "refine.bounds.p0",
        "refine.candidates",
        "refine.policy",
        "refine.objective_influence",
    )
    execution = execute_native_intermediate_refinement_program(
        objective_program, module, spec
    )
    assert len(execution.trace.action_traces) == len(objective_program.schedule.actions)


def test_objective_directed_policy_fails_closed_on_admission_and_tamper() -> None:
    module = _competing_width_influence_mlp()
    spec = _two_dimensional_input_spec()
    objective_policy = NativeIntermediateRefinementPolicyIR(
        passes=1,
        max_neurons_per_relu=1,
        backward_chunk_size=1,
        candidate_policy_id="objective_influence_width_per_relu_v1",
    )
    with pytest.raises(ValueError, match="policy/objective admission"):
        compile_native_intermediate_refinement_program(
            module,
            spec,
            policy=objective_policy,
            plan_id="missing-objective",
        )
    with pytest.raises(ValueError, match="policy/objective admission"):
        compile_native_intermediate_refinement_program(
            module,
            spec,
            policy=NativeIntermediateRefinementPolicyIR(
                passes=1,
                max_neurons_per_relu=1,
                backward_chunk_size=1,
            ),
            plan_id="unexpected-objective",
            linear_spec_C=torch.ones(1, 1),
        )
    with pytest.raises(ValueError, match="one finite scalar clause"):
        compile_native_intermediate_refinement_program(
            module,
            spec,
            policy=objective_policy,
            plan_id="multi-clause-objective",
            linear_spec_C=torch.ones(2, 1),
        )

    program = compile_native_intermediate_refinement_program(
        module,
        spec,
        policy=objective_policy,
        plan_id="tampered-objective",
        linear_spec_C=torch.ones(1, 1),
    )
    changed = replace(program, objective=torch.full((1, 1), 2.0))
    with pytest.raises(ValueError, match="objective hash differs"):
        changed.validate(module, spec)


def test_refinement_policy_rejects_unbounded_or_inconsistent_cost() -> None:
    with pytest.raises(ValueError, match="policy IR"):
        NativeIntermediateRefinementPolicyIR(
            passes=0,
            max_neurons_per_relu=2,
            backward_chunk_size=1,
        ).validate()
    with pytest.raises(ValueError, match="policy IR"):
        NativeIntermediateRefinementPolicyIR(
            passes=1,
            max_neurons_per_relu=2,
            backward_chunk_size=3,
        ).validate()
