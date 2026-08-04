"""First-class Plan/Task/Schedule tests for NRIR-29 search scaling."""

from dataclasses import replace

import pytest

from boundflow.ir.search_scaling import (
    NativeBabSearchBudgetIR,
    NativeBabSearchScalingPlanIR,
    compile_search_scaling_schedule_ir,
    compile_search_scaling_task_ir,
)
from boundflow.ir.workload import VerificationWorkloadSourceIR


def _source(index: int) -> VerificationWorkloadSourceIR:
    digest = f"{index + 1:064x}"
    return VerificationWorkloadSourceIR(
        workload_id=f"workload-{index}",
        category=f"category-{index}",
        csv_ordinal=index,
        csv_relative_path=f"category-{index}/instances.csv",
        model_relative_path=f"category-{index}/model.onnx",
        property_relative_path=f"category-{index}/property.vnnlib",
        csv_sha256=digest,
        model_sha256=digest,
        property_sha256=digest,
        query_ir_hash=digest,
        model_input_shape=(1, 2),
        model_output_dim=2,
        onnx_ops=("Gemm", "Relu"),
    )


def _plan() -> NativeBabSearchScalingPlanIR:
    return NativeBabSearchScalingPlanIR(
        plan_id="search-scaling-test",
        benchmark_commit="a" * 40,
        native_code_revision="b" * 64,
        workloads=tuple(_source(index) for index in range(3)),
        budgets=(
            NativeBabSearchBudgetIR("n7d2", 7, 2),
            NativeBabSearchBudgetIR("n31d4", 31, 4),
            NativeBabSearchBudgetIR("n127d6", 127, 6),
        ),
        repeats=3,
        timeout_seconds=60,
        torch_threads=8,
        optimizer_steps=5,
        search_steps=4,
        expansion_batch_size=2,
        max_eval_batch_size=4,
    )


def test_search_scaling_compiles_exact_rotated_fresh_process_schedule() -> None:
    plan = _plan()
    task_ir = compile_search_scaling_task_ir(plan)
    schedule = compile_search_scaling_schedule_ir(plan, task_ir)

    assert len(task_ir.tasks) == 27
    assert schedule.budget_orders == (
        ("n7d2", "n31d4", "n127d6"),
        ("n31d4", "n127d6", "n7d2"),
        ("n127d6", "n7d2", "n31d4"),
    )
    assert schedule.ordered_task_ids == schedule.fresh_process_task_ids
    assert [task.budget_id for task in task_ir.tasks[:9]] == [
        "n7d2",
        "n31d4",
        "n127d6",
        "n31d4",
        "n127d6",
        "n7d2",
        "n127d6",
        "n7d2",
        "n31d4",
    ]


def test_search_scaling_plan_rejects_post_registration_budget_change() -> None:
    plan = _plan()
    tampered = replace(
        plan,
        budgets=(
            NativeBabSearchBudgetIR("n7d2", 7, 2),
            NativeBabSearchBudgetIR("n31d4", 31, 4),
            NativeBabSearchBudgetIR("n255d7", 255, 7),
        ),
    )

    with pytest.raises(ValueError, match="Plan IR is invalid"):
        tampered.validate()


def test_search_scaling_task_hash_fails_closed_on_budget_rebinding() -> None:
    plan = _plan()
    task_ir = compile_search_scaling_task_ir(plan)
    tampered = replace(
        task_ir,
        tasks=(replace(task_ir.tasks[0], budget_hash="f" * 64), *task_ir.tasks[1:]),
    )

    with pytest.raises(ValueError, match="Task/Plan binding differs"):
        tampered.validate_against(plan)
