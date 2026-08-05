"""Contract and fail-closed tests for prepared refinement ownership."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from boundflow.ir.refinement import NativeIntermediateRefinementPolicyIR
from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime.native_intermediate_refinement import (
    NativeIntermediateRefinementProgram,
)
from boundflow.runtime.native_prepared_intermediate_refinement import (
    NativePreparedIntermediateRefinementExecution,
    NativePreparedIntermediateRefinementProgram,
    compile_native_prepared_intermediate_refinement_program,
    execute_native_prepared_intermediate_refinement_program,
    validate_native_prepared_intermediate_refinement_full,
)
from boundflow.runtime.task_executor import InputSpec


def _module() -> BFTaskModule:
    return BFTaskModule(
        tasks=[
            BoundTask(
                task_id="prepared-refinement-test",
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
        entry_task_id="prepared-refinement-test",
        bindings={
            "params": {
                "W1": torch.tensor([[1.0], [-1.0]]),
                "b1": torch.zeros(2),
                "W2": torch.tensor([[1.0, 1.0]]),
                "b2": torch.zeros(1),
            }
        },
    )


def _spec() -> InputSpec:
    return InputSpec.box(
        value_name="input",
        lower=torch.tensor([[-1.0]]),
        upper=torch.tensor([[1.0]]),
    )


def _policy() -> NativeIntermediateRefinementPolicyIR:
    return NativeIntermediateRefinementPolicyIR(
        passes=1,
        max_neurons_per_relu=2,
        backward_chunk_size=2,
    )


def test_prepared_capsule_owns_one_full_validation_and_one_runtime_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    spec = _spec()
    calls = 0
    frozen_validate = NativeIntermediateRefinementProgram.validate

    def counted_validate(
        self: NativeIntermediateRefinementProgram,
        candidate_module: BFTaskModule,
        candidate_spec: InputSpec,
    ) -> None:
        nonlocal calls
        calls += 1
        frozen_validate(self, candidate_module, candidate_spec)

    monkeypatch.setattr(
        NativeIntermediateRefinementProgram, "validate", counted_validate
    )
    program = compile_native_prepared_intermediate_refinement_program(
        module,
        spec,
        policy=_policy(),
        plan_id="prepared-refinement",
    )
    assert calls == 1
    assert isinstance(program, NativePreparedIntermediateRefinementProgram)
    assert program.capsule.full_validation_count == 1
    assert program.prepared_schedule.full_validation_launches == 1
    assert program.prepared_schedule.runtime_target_selection_launches == 1

    execution = execute_native_prepared_intermediate_refinement_program(
        program, module, spec
    )
    assert calls == 1
    assert isinstance(execution, NativePreparedIntermediateRefinementExecution)
    assert execution.prepared_trace.full_validation_count == 1
    assert execution.prepared_trace.runtime_target_selection_count == 1
    execution.validate(module, spec)
    assert calls == 1

    validate_native_prepared_intermediate_refinement_full(execution, module, spec)
    assert calls == 2


def test_prepared_ancestral_source_is_consumed_without_revalidation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    spec = _spec()
    calls = 0
    frozen_validate = NativeIntermediateRefinementProgram.validate

    def counted_validate(
        self: NativeIntermediateRefinementProgram,
        candidate_module: BFTaskModule,
        candidate_spec: InputSpec,
    ) -> None:
        nonlocal calls
        calls += 1
        frozen_validate(self, candidate_module, candidate_spec)

    monkeypatch.setattr(
        NativeIntermediateRefinementProgram, "validate", counted_validate
    )
    root_program = compile_native_prepared_intermediate_refinement_program(
        module, spec, policy=_policy(), plan_id="prepared-root"
    )
    root = execute_native_prepared_intermediate_refinement_program(
        root_program, module, spec
    )
    assert calls == 1
    child_program = compile_native_prepared_intermediate_refinement_program(
        module,
        spec,
        policy=_policy(),
        plan_id="prepared-child",
        source_refinement_execution=root,
    )
    assert calls == 2
    child = execute_native_prepared_intermediate_refinement_program(
        child_program, module, spec
    )
    assert calls == 2
    assert (
        child.program.plan.source_refinement_plan_hash
        == root.program.plan.stable_hash()
    )


def test_prepared_runtime_tensor_mutation_fails_closed() -> None:
    module = _module()
    spec = _spec()
    program = compile_native_prepared_intermediate_refinement_program(
        module, spec, policy=_policy(), plan_id="prepared-mutation"
    )
    execution = execute_native_prepared_intermediate_refinement_program(
        program, module, spec
    )
    execution.relu_pre["h1"].lower.add_(0.1)
    with pytest.raises(ValueError, match="runtime identity differs"):
        execution.validate(module, spec)


def test_prepared_capsule_and_runtime_scope_tampering_fail_closed() -> None:
    module = _module()
    spec = _spec()
    program = compile_native_prepared_intermediate_refinement_program(
        module, spec, policy=_policy(), plan_id="prepared-tamper"
    )
    with pytest.raises(ValueError, match="runtime identity differs"):
        replace(
            program,
            capsule=replace(program.capsule, target_table_hash="f" * 64),
        ).validate(module, spec)
    with pytest.raises(ValueError, match="runtime identity differs"):
        program.validate(
            module,
            InputSpec.box(
                value_name="input",
                lower=torch.tensor([[-2.0]]),
                upper=torch.tensor([[2.0]]),
            ),
        )
