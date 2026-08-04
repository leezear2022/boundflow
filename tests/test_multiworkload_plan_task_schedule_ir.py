"""Tests for multi-workload verifier Plan/Task/Schedule IR."""

# pylint: disable=missing-function-docstring,duplicate-code

from __future__ import annotations

from dataclasses import replace

import pytest

from boundflow.ir.workload import (
    MultiWorkloadPlanIR,
    MultiWorkloadTaskKind,
    VerificationWorkloadSourceIR,
    VerifierBackendKind,
    VerifierExecutionPolicyIR,
    compile_multiworkload_schedule_ir,
    compile_multiworkload_task_ir,
)


def _sha(character: str) -> str:
    return character * 64


def _workload(ordinal: int, category: str) -> VerificationWorkloadSourceIR:
    return VerificationWorkloadSourceIR(
        workload_id=f"{category}:000",
        category=category,
        csv_ordinal=ordinal,
        csv_relative_path=f"benchmarks/{category}/{category}_instances.csv",
        model_relative_path=f"benchmarks/{category}/model.onnx",
        property_relative_path=f"benchmarks/{category}/property.vnnlib",
        csv_sha256=_sha("a"),
        model_sha256=_sha(str(ordinal + 1)),
        property_sha256=_sha(chr(ord("d") + ordinal)),
        query_ir_hash=_sha(chr(ord("7") + ordinal)),
        model_input_shape=(784,) if category == "mnistfc" else (3, 32, 32),
        model_output_dim=10,
        onnx_ops=("Flatten", "Gemm", "Relu"),
    )


def _plan() -> MultiWorkloadPlanIR:
    native = VerifierExecutionPolicyIR(
        backend=VerifierBackendKind.BOUNDFLOW_NATIVE,
        implementation_id="boundflow-native-complete-query-v1",
        implementation_revision=_sha("b"),
        device="cpu",
        torch_threads=8,
        timeout_seconds=60,
        alpha_steps=5,
        beta_steps=5,
        search_steps=4,
        max_nodes=7,
        attack_policy="native_projected_gradient",
        complete_verifier="bounded_relu_bab",
    )
    competitor = VerifierExecutionPolicyIR(
        backend=VerifierBackendKind.EXTERNAL_ABCROWN,
        implementation_id="alpha-beta-CROWN",
        implementation_revision=_sha("e"),
        device="cpu",
        torch_threads=8,
        timeout_seconds=60,
        alpha_steps=25,
        beta_steps=10,
        search_steps=0,
        max_nodes=0,
        attack_policy="skip",
        complete_verifier="bab",
    )
    return MultiWorkloadPlanIR(
        plan_id="vnncomp21-three-topology-cpu-v1",
        benchmark_commit=_sha("c"),
        workloads=(
            _workload(0, "mnistfc"),
            _workload(0, "cifar10_resnet"),
            _workload(0, "oval21"),
        ),
        policies=(native, competitor),
    )


def _production_plan() -> MultiWorkloadPlanIR:
    baseline = _plan()
    production = VerifierExecutionPolicyIR(
        backend=VerifierBackendKind.BOUNDFLOW_PRODUCTION,
        implementation_id="boundflow-production-complete-query-v1",
        implementation_revision=_sha("f"),
        device="cpu",
        torch_threads=8,
        timeout_seconds=60,
        alpha_steps=5,
        beta_steps=5,
        search_steps=4,
        max_nodes=7,
        attack_policy="native_projected_gradient",
        complete_verifier="production_prepared_bounded_relu_bab",
    )
    return replace(
        baseline,
        plan_id="vnncomp21-three-topology-production-cpu-v1",
        policies=(baseline.policies[0], production, baseline.policies[1]),
        claim_boundary=(
            "cpu_audit_production_repeated_diagnostic_no_competitor_speedup"
        ),
    )


def test_multiworkload_plan_compiles_complete_task_and_schedule_ir() -> None:
    plan = _plan()
    task_ir = compile_multiworkload_task_ir(plan)
    schedule = compile_multiworkload_schedule_ir(plan, task_ir)

    assert len(task_ir.tasks) == 21
    assert tuple(task.kind for task in task_ir.tasks[:7]) == (
        MultiWorkloadTaskKind.ACQUIRE_SOURCES,
        MultiWorkloadTaskKind.PARSE_QUERY,
        MultiWorkloadTaskKind.IMPORT_ONNX,
        MultiWorkloadTaskKind.COMPILE_NATIVE,
        MultiWorkloadTaskKind.EXECUTE_NATIVE,
        MultiWorkloadTaskKind.EXECUTE_COMPETITOR,
        MultiWorkloadTaskKind.EMIT_RESULT,
    )
    assert len(schedule.fresh_process_task_ids) == 6
    assert schedule.ordered_task_ids == tuple(task.task_id for task in task_ir.tasks)
    assert plan.stable_hash() == task_ir.plan_hash == schedule.plan_hash
    assert task_ir.stable_hash() == schedule.task_ir_hash
    assert schedule.stable_hash(task_ir) == schedule.stable_hash(task_ir)


def test_multiworkload_ir_rejects_source_policy_and_schedule_tamper() -> None:
    plan = _plan()
    task_ir = compile_multiworkload_task_ir(plan)
    schedule = compile_multiworkload_schedule_ir(plan, task_ir)

    with pytest.raises(ValueError, match="source IR is invalid"):
        replace(plan.workloads[0], model_relative_path="../model.onnx").validate()
    with pytest.raises(ValueError, match="external alpha-beta-CROWN"):
        replace(plan.policies[1], attack_policy="middle").validate()
    with pytest.raises(ValueError, match="Schedule IR differs"):
        replace(schedule, task_ir_hash=_sha("0")).validate_against(task_ir)
    with pytest.raises(ValueError, match="fresh-process coverage differs"):
        replace(
            schedule, fresh_process_task_ids=schedule.fresh_process_task_ids[:-1]
        ).validate_against(task_ir)


def test_multiworkload_production_plan_adds_typed_execution_path() -> None:
    plan = _production_plan()
    task_ir = compile_multiworkload_task_ir(plan)
    schedule = compile_multiworkload_schedule_ir(plan, task_ir)

    assert len(task_ir.tasks) == 24
    assert tuple(task.kind for task in task_ir.tasks[:8]) == (
        MultiWorkloadTaskKind.ACQUIRE_SOURCES,
        MultiWorkloadTaskKind.PARSE_QUERY,
        MultiWorkloadTaskKind.IMPORT_ONNX,
        MultiWorkloadTaskKind.COMPILE_NATIVE,
        MultiWorkloadTaskKind.EXECUTE_NATIVE,
        MultiWorkloadTaskKind.EXECUTE_PRODUCTION,
        MultiWorkloadTaskKind.EXECUTE_COMPETITOR,
        MultiWorkloadTaskKind.EMIT_RESULT,
    )
    assert len(schedule.fresh_process_task_ids) == 9
    emit = task_ir.tasks[7]
    assert emit.dependency_task_ids == (
        "mnistfc:000:execute-native",
        "mnistfc:000:execute-production",
        "mnistfc:000:execute-abcrown",
    )
    schedule.validate_against(task_ir)

    with pytest.raises(ValueError, match="BoundFlow execution policy differs"):
        replace(plan.policies[1], complete_verifier="bounded_relu_bab").validate()
    with pytest.raises(ValueError, match="claim boundary differs"):
        replace(plan, claim_boundary="cpu_diagnostic_no_speedup").validate()

    internal = replace(
        plan,
        policies=plan.policies[:2],
        claim_boundary="cpu_audit_production_repeated_internal_speedup",
    )
    internal_tasks = compile_multiworkload_task_ir(internal)
    internal_schedule = compile_multiworkload_schedule_ir(internal, internal_tasks)
    assert len(internal_tasks.tasks) == 21
    assert len(internal_schedule.fresh_process_task_ids) == 6
    assert internal_tasks.tasks[6].dependency_task_ids == (
        "mnistfc:000:execute-native",
        "mnistfc:000:execute-production",
    )


def test_multiworkload_task_ir_rejects_dependency_reordering() -> None:
    plan = _plan()
    task_ir = compile_multiworkload_task_ir(plan)
    tampered_tasks = list(task_ir.tasks)
    tampered_tasks[1], tampered_tasks[3] = tampered_tasks[3], tampered_tasks[1]

    with pytest.raises(ValueError, match="dependency order differs"):
        replace(task_ir, tasks=tuple(tampered_tasks)).validate()
