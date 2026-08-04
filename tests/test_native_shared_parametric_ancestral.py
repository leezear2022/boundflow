"""Shared-parametric ancestral queue contract and tamper tests."""

# pylint: disable=missing-function-docstring,redefined-outer-name,duplicate-code

from dataclasses import replace

import pytest
import torch

from boundflow.ir.refinement import NativeIntermediateRefinementPolicyIR
from boundflow.ir.shared_parametric_ancestral import (
    NativeSharedParametricAncestralBatchIR,
    NativeSharedParametricAncestralTaskKind,
    _canonical_hash,
    lower_native_shared_parametric_ancestral_schedule,
)
from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime.native_alpha_beta_optimization_state import (
    NativeAlphaBetaOptimizerPolicy,
)
from boundflow.runtime.native_candidate_search import (
    NativeProjectedGradientSearchPolicy,
)
from boundflow.runtime.native_intermediate_refinement import (
    compile_native_intermediate_refinement_program,
    execute_native_intermediate_refinement_program,
)
from boundflow.runtime.native_parametric_optimizer import (
    NativeParametricOptimizerTemplateCache,
)
from boundflow.runtime.native_multi_clause_anytime import (
    compile_native_multi_clause_anytime_program,
)
from boundflow.runtime.native_shared_parametric_multi_clause_anytime import (
    execute_native_shared_parametric_multi_clause_anytime_program,
)
from boundflow.runtime.native_shared_parametric_ancestral import (
    compile_native_shared_parametric_ancestral_plan,
    execute_native_shared_parametric_ancestral_queue,
)
from boundflow.runtime.task_executor import InputSpec


def _module() -> BFTaskModule:
    return BFTaskModule(
        tasks=[
            BoundTask(
                task_id="shared-parametric-ancestral-toy",
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
        entry_task_id="shared-parametric-ancestral-toy",
        bindings={
            "params": {
                "W1": torch.tensor(
                    [
                        [1.0, -0.5],
                        [-0.25, 0.75],
                        [0.6, 0.8],
                        [-0.7, -0.4],
                    ]
                ),
                "b1": torch.tensor([0.1, -0.2, 0.0, 0.15]),
                "W2": torch.tensor(
                    [
                        [0.75, -1.0, 0.4, -0.3],
                        [-0.5, 0.25, -0.6, 0.9],
                    ]
                ),
                "b2": torch.tensor([0.15, -0.1]),
            }
        },
    )


def _spec() -> InputSpec:
    return InputSpec.box(
        value_name="input",
        lower=torch.tensor([[-0.3, -0.6]]),
        upper=torch.tensor([[0.7, 0.4]]),
    )


def _root_refinement(
    module: BFTaskModule, spec: InputSpec, objective: torch.Tensor, *, suffix: str
):
    shared_program = compile_native_intermediate_refinement_program(
        module,
        spec,
        policy=NativeIntermediateRefinementPolicyIR(
            passes=1, max_neurons_per_relu=4, backward_chunk_size=4
        ),
        plan_id=f"shared-parametric-ancestral:{suffix}:shared",
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
        plan_id=f"shared-parametric-ancestral:{suffix}:root",
        linear_spec_C=objective,
        source_refinement_execution=shared,
    )
    return execute_native_intermediate_refinement_program(root_program, module, spec)


def _clock_for_one_pair():
    values = iter((0, 0, 0, 0, 0, 0, 60_000_000_000, 60_000_000_000))
    return lambda: next(values)


def _clock_for_root_only():
    values = iter((0, 0, 0, 60_000_000_000, 60_000_000_000))
    return lambda: next(values)


@pytest.fixture(scope="module")
def execution_bundle():
    module = _module()
    spec = _spec()
    objective = torch.tensor([[[1.0, -1.0]]])
    threshold = torch.tensor([1e6])
    root = _root_refinement(module, spec, objective, suffix="primary")
    policy = NativeAlphaBetaOptimizerPolicy(steps=1, lr=0.1)
    plan = compile_native_shared_parametric_ancestral_plan(
        module,
        spec,
        linear_spec_C=objective,
        threshold=threshold,
        root_refinement=root,
        optimizer_policy=policy,
        plan_id="shared-parametric-ancestral-primary",
    )
    cache = NativeParametricOptimizerTemplateCache()
    execution = execute_native_shared_parametric_ancestral_queue(
        plan,
        module,
        spec,
        linear_spec_C=objective,
        threshold=threshold,
        root_refinement=root,
        optimizer_policy=policy,
        compiler_cache=cache,
        query_id="shared-parametric-ancestral-primary",
        clock_ns=_clock_for_one_pair(),
    )
    return module, spec, objective, threshold, root, policy, cache, execution


def test_shared_parametric_ancestral_compiles_once_and_commits_pair(
    execution_bundle,
) -> None:
    (
        _module_value,
        _spec_value,
        _objective,
        _threshold,
        _root,
        _policy,
        cache,
        execution,
    ) = execution_bundle

    assert len(execution.queue.trace.evaluations) == 3
    assert len(execution.batch_commits) == len(execution.compiler_batches) == 2
    assert [item.cache_event.outcome for item in execution.compiler_batches] == [
        "miss_compiled",
        "hit_exact_contract",
    ]
    assert len(cache.templates) == 1
    assert execution.batch_commits[0].commit_kind == "root"
    assert execution.batch_commits[1].commit_kind == "atomic_sibling_pair"
    assert len(execution.batch_commits[1].node_ids) == 2
    assert execution.trace.selected_native_reexecution is False
    assert execution.queue.trace.selected_native_reexecution is False
    assert execution.trace.discarded_attempt_stage is None


def test_shared_parametric_ancestral_lowers_first_class_tasks(
    execution_bundle,
) -> None:
    *_unused, execution = execution_bundle
    kinds = tuple(item.kind for item in execution.task_ir.tasks)

    assert kinds[0] == NativeSharedParametricAncestralTaskKind.ADMIT_QUERY
    assert kinds.count(NativeSharedParametricAncestralTaskKind.ACQUIRE_TEMPLATE) == 2
    assert kinds.count(NativeSharedParametricAncestralTaskKind.INSTANTIATE_BATCH) == 2
    assert kinds.count(NativeSharedParametricAncestralTaskKind.EXECUTE_BATCH) == 2
    assert kinds.count(NativeSharedParametricAncestralTaskKind.COMMIT_ROOT) == 1
    assert kinds.count(NativeSharedParametricAncestralTaskKind.COMMIT_SIBLING_PAIR) == 1
    assert kinds[-1] == NativeSharedParametricAncestralTaskKind.EMIT_RESULT


def test_shared_parametric_ancestral_cache_tamper_fails_closed(
    execution_bundle,
) -> None:
    module, spec, objective, threshold, root, policy, _cache, execution = (
        execution_bundle
    )
    compiler = execution.compiler_batches[1]
    tampered = replace(
        compiler,
        cache_event=replace(compiler.cache_event, cache_key="0" * 64),
    )

    with pytest.raises(ValueError, match="batch trace differs"):
        replace(
            execution,
            compiler_batches=(execution.compiler_batches[0], tampered),
        ).validate_against(
            module,
            spec,
            linear_spec_C=objective,
            threshold=threshold,
            root_refinement=root,
            optimizer_policy=policy,
        )


def test_shared_parametric_ancestral_partial_pair_fails_closed(
    execution_bundle,
) -> None:
    *_unused, execution = execution_bundle
    pair = execution.batch_commits[1]

    with pytest.raises(ValueError, match="Batch IR is invalid"):
        replace(
            pair,
            node_ids=pair.node_ids[:1],
            node_split_state_hashes=pair.node_split_state_hashes[:1],
            refinement_semantic_trace_hashes=(
                pair.refinement_semantic_trace_hashes[:1]
            ),
            evaluation_hashes=pair.evaluation_hashes[:1],
        ).validate()


def test_shared_parametric_ancestral_task_batch_binding_fails_closed(
    execution_bundle,
) -> None:
    module, spec, objective, threshold, root, policy, _cache, execution = (
        execution_bundle
    )
    task_ir = replace(
        execution.task_ir,
        batch_hashes=(execution.task_ir.batch_hashes[0], "0" * 64),
    )
    schedule = replace(execution.schedule, task_ir_hash=task_ir.stable_hash())

    with pytest.raises(ValueError, match="Task/Batch commit binding differs"):
        replace(execution, task_ir=task_ir, schedule=schedule).validate_against(
            module,
            spec,
            linear_spec_C=objective,
            threshold=threshold,
            root_refinement=root,
            optimizer_policy=policy,
        )


def test_shared_parametric_ancestral_instance_tamper_fails_closed(
    execution_bundle,
) -> None:
    module, spec, objective, threshold, root, policy, _cache, execution = (
        execution_bundle
    )
    compiler = execution.compiler_batches[1]
    tampered = replace(
        compiler,
        instance_ir=replace(compiler.instance_ir, objective_hash="0" * 64),
    )

    with pytest.raises(ValueError, match="Plan binding differs|Batch binding differs"):
        replace(
            execution,
            compiler_batches=(execution.compiler_batches[0], tampered),
        ).validate_against(
            module,
            spec,
            linear_spec_C=objective,
            threshold=threshold,
            root_refinement=root,
            optimizer_policy=policy,
        )


# pylint: disable-next=too-many-locals
def test_shared_parametric_ancestral_recompile_event_fails_closed(
    execution_bundle,
) -> None:
    module, spec, objective, threshold, root, policy, _cache, execution = (
        execution_bundle
    )
    compiler = execution.compiler_batches[1]
    tampered = replace(
        compiler,
        cache_event=replace(
            compiler.cache_event,
            outcome="miss_compiled",
            compile_elapsed_ns=1,
        ),
    )
    commit = execution.batch_commits[1]
    values = {
        name: value
        for name, value in commit.__dict__.items()
        if name not in {"atomic_commit_hash", "schema_version"}
    }
    values["compiler_batch_trace_hash"] = tampered.stable_hash()
    values["cache_event_hash"] = tampered.cache_event.stable_hash()
    tampered_commit = NativeSharedParametricAncestralBatchIR.committed(**values)
    tasks = tuple(
        (
            replace(
                task,
                output_hash=_canonical_hash(tampered_commit.to_dict()),
            )
            if task.kind == NativeSharedParametricAncestralTaskKind.COMMIT_SIBLING_PAIR
            else task
        )
        for task in execution.task_ir.tasks
    )
    task_ir = replace(
        execution.task_ir,
        tasks=tasks,
        batch_hashes=(
            execution.task_ir.batch_hashes[0],
            tampered_commit.stable_hash(),
        ),
    )
    schedule = replace(execution.schedule, task_ir_hash=task_ir.stable_hash())

    with pytest.raises(ValueError, match="recompiled after first batch"):
        replace(
            execution,
            task_ir=task_ir,
            schedule=schedule,
            compiler_batches=(execution.compiler_batches[0], tampered),
            batch_commits=(execution.batch_commits[0], tampered_commit),
        ).validate_against(
            module,
            spec,
            linear_spec_C=objective,
            threshold=threshold,
            root_refinement=root,
            optimizer_policy=policy,
        )


def test_shared_parametric_ancestral_ordinal_tamper_fails_closed(
    execution_bundle,
) -> None:
    *_unused, execution = execution_bundle
    pair = execution.batch_commits[1]
    values = {
        name: value
        for name, value in pair.__dict__.items()
        if name not in {"atomic_commit_hash", "schema_version"}
    }
    values["batch_index"] = 2
    tampered = NativeSharedParametricAncestralBatchIR.committed(**values)

    with pytest.raises(ValueError, match="Batch order differs"):
        lower_native_shared_parametric_ancestral_schedule(
            execution.plan,
            execution.task_ir.tasks,
            (execution.batch_commits[0], tampered),
        )


def test_shared_parametric_ancestral_bound_drift_fails_closed(
    execution_bundle,
) -> None:
    module, spec, objective, threshold, root, policy, _cache, execution = (
        execution_bundle
    )
    evaluations = list(execution.queue.trace.evaluations)
    evaluations[1] = replace(evaluations[1], upper=evaluations[1].upper + 1.0)
    queue = replace(
        execution.queue,
        trace=replace(execution.queue.trace, evaluations=tuple(evaluations)),
    )

    with pytest.raises(ValueError, match="Batch binding differs"):
        replace(execution, queue=queue).validate_against(
            module,
            spec,
            linear_spec_C=objective,
            threshold=threshold,
            root_refinement=root,
            optimizer_policy=policy,
        )


def test_shared_parametric_ancestral_source_lineage_tamper_fails_closed(
    execution_bundle,
) -> None:
    module, spec, objective, threshold, root, policy, _cache, execution = (
        execution_bundle
    )
    child = execution.node_refinements[1]
    plan = replace(
        child.program.plan,
        source_intermediate_constraints_hash="0" * 64,
    )
    program = replace(child.program, plan=plan)
    tampered = replace(
        child,
        program=program,
        execution=replace(child.execution, program=program),
    )

    with pytest.raises(ValueError):
        replace(
            execution,
            node_refinements=(
                execution.node_refinements[0],
                tampered,
                execution.node_refinements[2],
            ),
        ).validate_against(
            module,
            spec,
            linear_spec_C=objective,
            threshold=threshold,
            root_refinement=root,
            optimizer_policy=policy,
        )


def test_shared_parametric_ancestral_native_reexecution_flag_fails_closed(
    execution_bundle,
) -> None:
    *_unused, execution = execution_bundle

    with pytest.raises(ValueError, match="queue trace header is invalid"):
        replace(execution.queue.trace, selected_native_reexecution=True).validate()


def test_shared_parametric_ancestral_plan_contract_tamper_fails_closed(
    execution_bundle,
) -> None:
    *_unused, execution = execution_bundle

    with pytest.raises(ValueError, match="Plan IR is invalid"):
        replace(
            execution.plan,
            template_contract_excludes=("objective_content",),
        ).validate()


def test_shared_parametric_ancestral_preserves_external_deadline() -> None:
    module = _module()
    spec = _spec()
    objective = torch.tensor([[[1.0, -1.0]]])
    threshold = torch.tensor([1e6])
    root = _root_refinement(module, spec, objective, suffix="deadline")
    policy = NativeAlphaBetaOptimizerPolicy(steps=1, lr=0.1)
    plan = compile_native_shared_parametric_ancestral_plan(
        module,
        spec,
        linear_spec_C=objective,
        threshold=threshold,
        root_refinement=root,
        optimizer_policy=policy,
        plan_id="shared-parametric-ancestral-deadline",
    )
    clock_values = iter((59_000_000_000, 61_000_000_000))

    with pytest.raises(RuntimeError, match="expired at root"):
        execute_native_shared_parametric_ancestral_queue(
            plan,
            module,
            spec,
            linear_spec_C=objective,
            threshold=threshold,
            root_refinement=root,
            optimizer_policy=policy,
            compiler_cache=NativeParametricOptimizerTemplateCache(),
            query_id="shared-parametric-ancestral-deadline",
            whole_query_started_ns=0,
            clock_ns=lambda: next(clock_values),
        )


def test_shared_parametric_ancestral_cache_reuses_across_objectives() -> None:
    module = _module()
    spec = _spec()
    threshold = torch.tensor([1e6])
    policy = NativeAlphaBetaOptimizerPolicy(steps=1, lr=0.1)
    cache = NativeParametricOptimizerTemplateCache()
    outcomes: list[str] = []
    for index, objective in enumerate(
        (torch.tensor([[[1.0, -1.0]]]), torch.tensor([[[-1.0, 1.0]]]))
    ):
        root = _root_refinement(module, spec, objective, suffix=str(index))
        plan = compile_native_shared_parametric_ancestral_plan(
            module,
            spec,
            linear_spec_C=objective,
            threshold=threshold,
            root_refinement=root,
            optimizer_policy=policy,
            plan_id=f"shared-parametric-cross-objective-{index}",
        )
        execution = execute_native_shared_parametric_ancestral_queue(
            plan,
            module,
            spec,
            linear_spec_C=objective,
            threshold=threshold,
            root_refinement=root,
            optimizer_policy=policy,
            compiler_cache=cache,
            query_id=f"shared-parametric-cross-objective-{index}",
            clock_ns=_clock_for_root_only(),
        )
        outcomes.append(execution.compiler_batches[0].cache_event.outcome)

    cache.validate()
    assert outcomes == ["miss_compiled", "hit_exact_contract"]
    assert len(cache.templates) == 1


def test_shared_parametric_multi_clause_preserves_floor_only_result() -> None:
    module = _module()
    spec = _spec()
    objectives = torch.tensor([[[1.0, -1.0]]]).repeat(1, 9, 1)
    thresholds = torch.full((9,), -1e6)
    search = NativeProjectedGradientSearchPolicy(steps=1, step_size=0.01)
    optimizer = NativeAlphaBetaOptimizerPolicy(steps=1, lr=0.1)
    program = compile_native_multi_clause_anytime_program(
        module,
        spec,
        linear_spec_C=objectives,
        thresholds=thresholds,
        plan_id="shared-parametric-multi-clause-floor-only",
        search_policy=search,
        optimizer_policy=optimizer,
    )
    execution = execute_native_shared_parametric_multi_clause_anytime_program(
        program,
        module,
        spec,
        linear_spec_C=objectives,
        thresholds=thresholds,
        query_id="shared-parametric-multi-clause-floor-only",
        search_policy=search,
        optimizer_policy=optimizer,
    )

    assert execution.floor.trace.final_status == "verified"
    assert not execution.packed_executions
    assert not execution.cache_events
    assert not execution.template_hashes
    assert execution.aggregate.final_status == "verified"
    assert execution.trace.performance_claimed is False
