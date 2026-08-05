"""Contract tests for single-pass exact target admission ownership."""

# pylint: disable=missing-function-docstring,protected-access

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
import torch

from boundflow.ir.refinement import (
    NativeIntermediateRefinementPolicyIR,
    lower_native_intermediate_refinement_ir,
)
from boundflow.ir.target_admission import lower_native_target_admission_ir
from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime import native_intermediate_refinement as legacy
from boundflow.runtime import native_target_admission as admission
from boundflow.runtime.native_prepared_intermediate_refinement import (
    NativeSinglePassPreparedIntermediateRefinementProgram,
    compile_native_prepared_intermediate_refinement_program,
    compile_native_single_pass_prepared_intermediate_refinement_program,
    execute_native_prepared_intermediate_refinement_program,
    validate_native_prepared_intermediate_refinement_full,
)
from boundflow.runtime.native_target_admission import (
    NativeSinglePassTargetAdmissionProgram,
    _build_target_admission_receipt,
    compile_native_single_pass_target_admission_program,
    validate_native_target_admission_binding,
)
from boundflow.runtime.task_executor import InputSpec


def _module() -> BFTaskModule:
    return BFTaskModule(
        tasks=[
            BoundTask(
                task_id="single-pass-admission-test",
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
        entry_task_id="single-pass-admission-test",
        bindings={
            "params": {
                "W1": torch.tensor([[1.0], [-1.0]]),
                "b1": torch.zeros(2),
                "W2": torch.tensor([[1.0, 1.0]]),
                "b2": torch.zeros(1),
            }
        },
    )


def _spec(radius: float = 1.0) -> InputSpec:
    return InputSpec.box(
        value_name="input",
        lower=torch.tensor([[-radius]]),
        upper=torch.tensor([[radius]]),
    )


def _policy() -> NativeIntermediateRefinementPolicyIR:
    return NativeIntermediateRefinementPolicyIR(
        passes=1,
        max_neurons_per_relu=2,
        backward_chunk_size=2,
    )


def test_single_pass_compile_selects_once_and_full_replay_reselects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    spec = _spec()
    calls = 0
    frozen_select = legacy._select_targets

    def counted_select(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return frozen_select(*args, **kwargs)

    monkeypatch.setattr(legacy, "_select_targets", counted_select)
    monkeypatch.setattr(admission, "_select_targets", counted_select)
    program = compile_native_single_pass_target_admission_program(
        module, spec, policy=_policy(), plan_id="single-pass"
    )
    assert isinstance(program, NativeSinglePassTargetAdmissionProgram)
    assert calls == 1
    assert program.target_admission_receipt.selection_count == 1
    assert program.target_admission_schedule.production_selection_launches == 1
    assert program.target_admission_schedule.full_replay_selection_launches == 0

    program.validate(module, spec)
    assert calls == 1
    program.validate_full(module, spec)
    assert calls == 2


def test_single_pass_prepared_capsule_binds_receipt_and_keeps_v1_compatible() -> None:
    module = _module()
    spec = _spec()
    legacy_program = compile_native_prepared_intermediate_refinement_program(
        module, spec, policy=_policy(), plan_id="legacy-prepared"
    )
    assert legacy_program.capsule.schema_version.endswith("/v1")
    assert legacy_program.capsule.target_admission_receipt_hash is None

    program = compile_native_single_pass_prepared_intermediate_refinement_program(
        module, spec, policy=_policy(), plan_id="single-pass-prepared"
    )
    assert isinstance(program, NativeSinglePassPreparedIntermediateRefinementProgram)
    assert program.capsule.schema_version.endswith("/v2")
    assert (
        program.capsule.target_admission_receipt_hash
        == program.target_admission_receipt.stable_hash()
    )
    program.validate(module, spec)


def test_synchronized_wrong_target_receipt_fails_closed() -> None:
    module = _module()
    spec = _spec()
    program = compile_native_single_pass_target_admission_program(
        module, spec, policy=_policy(), plan_id="receipt-tamper"
    )
    receipt = replace(
        program.target_admission_receipt,
        target_table_hash="f" * 64,
        admission_receipt_hash="0" * 64,
    )
    receipt = replace(receipt, admission_receipt_hash=receipt.expected_receipt_hash())
    task_module, schedule = lower_native_target_admission_ir(
        source_plan_hash=program.plan.stable_hash(), receipt=receipt
    )
    with pytest.raises(ValueError, match="receipt differs"):
        validate_native_target_admission_binding(
            program,
            receipt=receipt,
            task_module=task_module,
            schedule=schedule,
        )


def test_cross_program_receipt_and_input_scope_fail_closed() -> None:
    module = _module()
    first = compile_native_single_pass_target_admission_program(
        module, _spec(), policy=_policy(), plan_id="first"
    )
    second = compile_native_single_pass_target_admission_program(
        module, _spec(2.0), policy=_policy(), plan_id="second"
    )
    with pytest.raises(ValueError, match="receipt differs"):
        validate_native_target_admission_binding(
            second,
            receipt=first.target_admission_receipt,
            task_module=first.target_admission_task_module,
            schedule=first.target_admission_schedule,
        )
    with pytest.raises(ValueError, match="identity differs"):
        first.validate(module, _spec(2.0))


def test_full_replay_rejects_semantically_reordered_targets() -> None:
    module = _module()
    spec = _spec()
    program = compile_native_single_pass_target_admission_program(
        module, spec, policy=_policy(), plan_id="reordered"
    )
    reordered = tuple(
        replace(target, ordinal=index)
        for index, target in enumerate(reversed(program.plan.targets))
    )
    plan = replace(program.plan, targets=reordered)
    task_module, schedule = lower_native_intermediate_refinement_ir(plan)
    synchronized = replace(
        program,
        plan=plan,
        task_module=task_module,
        schedule=schedule,
    )
    receipt = _build_target_admission_receipt(synchronized)
    admission_tasks, admission_schedule = lower_native_target_admission_ir(
        source_plan_hash=plan.stable_hash(), receipt=receipt
    )
    synchronized = replace(
        synchronized,
        target_admission_receipt=receipt,
        target_admission_task_module=admission_tasks,
        target_admission_schedule=admission_schedule,
    )
    synchronized.validate(module, spec)
    with pytest.raises(ValueError, match="target selection differs"):
        synchronized.validate_full(module, spec)


def test_prepared_full_replay_reselects_without_weakening_fast_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    spec = _spec()
    calls = 0
    frozen_select = legacy._select_targets

    def counted_select(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return frozen_select(*args, **kwargs)

    monkeypatch.setattr(legacy, "_select_targets", counted_select)
    monkeypatch.setattr(admission, "_select_targets", counted_select)
    program = compile_native_single_pass_prepared_intermediate_refinement_program(
        module, spec, policy=_policy(), plan_id="prepared-full"
    )
    assert calls == 1
    execution = execute_native_prepared_intermediate_refinement_program(
        program, module, spec
    )
    assert calls == 2  # one compile selection plus one runtime semantic selection
    program.validate(module, spec)
    assert calls == 2
    validate_native_prepared_intermediate_refinement_full(execution, module, spec)
    assert calls == 3  # explicit full replay performs the second compile-time selection
