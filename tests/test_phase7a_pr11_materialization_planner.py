import json

import pytest
import torch

from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.planner.materialization import (
    BoundMethod,
    MaterializationAction,
    MaterializationContext,
    MaterializationObservation,
    MaterializationPlanRecord,
    MaterializationPlannerOptions,
    MaterializationPolicy,
    OperatorTreeSummary,
    OptimizationStage,
    PLAN_SCHEMA_VERSION,
    TargetProfile,
    plan_materialization,
    plan_materialization_oracle,
)
from boundflow.planner.materialization_cost_model import (
    COST_MODEL_SCHEMA_VERSION,
    MaterializationCalibrationSample,
    fit_materialization_cost_model,
)
from boundflow.runtime.crown_ibp import (
    MaterializationReplanRequired,
    _relu_backward_mode,
    plan_crown_materialization,
    run_crown_ibp_mlp,
)
from boundflow.runtime.alpha_beta_crown import run_alpha_beta_crown_mlp
from boundflow.runtime.alpha_crown import run_alpha_crown_mlp
from boundflow.runtime.materialization import trace_materializations
from boundflow.runtime.task_executor import InputSpec


def _context(
    *,
    method: BoundMethod = BoundMethod.CROWN,
    requires_grad: bool = False,
    alpha_enabled: bool = False,
    beta_enabled: bool = False,
    dense_bytes: int = 500,
    structured_bytes: int = 200,
    memory_budget_bytes: int = 1_000,
    dense_latency_ms: float | None = None,
    structured_latency_ms: float | None = None,
    target: TargetProfile = TargetProfile(),
) -> MaterializationContext:
    return MaterializationContext(
        bound_method=method,
        requires_grad=requires_grad,
        optimization_stage=(
            OptimizationStage.ALPHA_OPTIMIZE
            if requires_grad
            else OptimizationStage.INFERENCE
        ),
        alpha_enabled=alpha_enabled,
        beta_enabled=beta_enabled,
        split_state_present=beta_enabled,
        batch_size=8,
        spec_size=16,
        domain_batch_size=8,
        operator_summary=OperatorTreeSummary(
            dense_a_bytes=dense_bytes,
            structured_base_bytes=structured_bytes,
            scale_bytes=0,
            dense_latency_ms=dense_latency_ms,
            structured_latency_ms=structured_latency_ms,
        ),
        memory_budget_bytes=memory_budget_bytes,
        available_memory_bytes=memory_budget_bytes,
        safety_margin=0.9,
        target=target,
    )


def _relu_mlp() -> BFTaskModule:
    torch.manual_seed(41)
    task = BoundTask(
        task_id="t0",
        kind=TaskKind.INTERVAL_IBP,
        ops=[
            TaskOp(
                op_type="linear",
                name="linear1",
                inputs=["input", "W1", "b1"],
                outputs=["h1"],
            ),
            TaskOp(op_type="relu", name="relu1", inputs=["h1"], outputs=["r1"]),
            TaskOp(
                op_type="linear",
                name="linear2",
                inputs=["r1", "W2", "b2"],
                outputs=["out"],
            ),
        ],
        input_values=["input"],
        output_values=["out"],
    )
    return BFTaskModule(
        tasks=[task],
        entry_task_id="t0",
        bindings={
            "params": {
                "W1": torch.randn(6, 4, dtype=torch.float64),
                "b1": torch.randn(6, dtype=torch.float64),
                "W2": torch.randn(3, 6, dtype=torch.float64),
                "b2": torch.randn(3, dtype=torch.float64),
            }
        },
    )


def test_global_planner_prefers_dense_when_both_fit_and_latency_is_unknown() -> None:
    plan = plan_materialization(_context())

    assert plan.action == MaterializationAction.DENSE
    assert plan.recommended_domain_batch_size == 8
    assert all(candidate.feasible for candidate in plan.candidates)
    json.dumps(plan.to_dict(), sort_keys=True)


def test_plan_record_schema_v1_freezes_context_and_decision_keys() -> None:
    context = _context()
    plan = plan_materialization(context)
    record = MaterializationPlanRecord(context=context, plan=plan).to_dict()

    assert record["schema_version"] == PLAN_SCHEMA_VERSION
    assert set(record) == {"schema_version", "context", "plan"}
    assert set(record["context"]) == {
        "bound_method",
        "requires_grad",
        "optimization_stage",
        "alpha_enabled",
        "beta_enabled",
        "split_state_present",
        "batch_size",
        "spec_size",
        "domain_batch_size",
        "operator_summary",
        "memory_budget_bytes",
        "available_memory_bytes",
        "safety_margin",
        "expected_query_reuse",
        "target",
    }
    assert set(record["plan"]) == {
        "schema_version",
        "policy",
        "action",
        "safe_memory_budget_bytes",
        "recommended_domain_batch_size",
        "reason",
        "candidates",
    }
    assert all(
        set(candidate)
        == {
            "action",
            "capability_legal",
            "memory_feasible",
            "predicted_peak_bytes",
            "predicted_latency_ms",
            "reasons",
        }
        for candidate in record["plan"]["candidates"]
    )
    json.dumps(record, sort_keys=True)


def test_global_planner_uses_structured_as_plain_crown_memory_escape() -> None:
    plan = plan_materialization(_context(dense_bytes=1_000, structured_bytes=200))

    assert plan.action == MaterializationAction.STRUCTURED
    assert plan.safe_memory_budget_bytes == 900
    assert plan.candidates[0].memory_feasible is False
    assert plan.candidates[1].feasible is True


def test_global_planner_uses_measured_latency_after_feasibility() -> None:
    plan = plan_materialization(
        _context(
            dense_latency_ms=8.0,
            structured_latency_ms=3.0,
            target=TargetProfile(supports_structured_latency_selection=True),
        )
    )

    assert plan.action == MaterializationAction.STRUCTURED


def test_global_planner_keeps_dense_when_structured_latency_is_not_validated() -> None:
    plan = plan_materialization(
        _context(dense_latency_ms=8.0, structured_latency_ms=3.0)
    )

    assert plan.action == MaterializationAction.DENSE


@pytest.mark.parametrize(
    ("method", "alpha_enabled", "beta_enabled"),
    [
        (BoundMethod.ALPHA_CROWN, True, False),
        (BoundMethod.ALPHA_BETA_CROWN, True, True),
    ],
)
def test_optimized_bound_structured_is_filtered_by_backend_capability(
    method: BoundMethod, alpha_enabled: bool, beta_enabled: bool
) -> None:
    plan = plan_materialization(
        _context(
            method=method,
            requires_grad=True,
            alpha_enabled=alpha_enabled,
            beta_enabled=beta_enabled,
            dense_bytes=500,
            structured_bytes=100,
            dense_latency_ms=10.0,
            structured_latency_ms=1.0,
        )
    )

    assert plan.action == MaterializationAction.DENSE
    structured = plan.candidates[1]
    assert structured.capability_legal is False
    assert "target_lacks_structured_autograd" in structured.reasons
    assert "target_lacks_optimized_bound_structured" in structured.reasons


def test_planner_reduces_batch_when_no_legal_materialization_fits() -> None:
    plan = plan_materialization(
        _context(dense_bytes=1_000, structured_bytes=500, memory_budget_bytes=400)
    )

    assert plan.action == MaterializationAction.REDUCE_BATCH
    assert 1 <= plan.recommended_domain_batch_size < 8
    assert plan.reason == "no_materialization_action_fits_safe_budget"


def test_baseline_policy_cannot_bypass_capability_filter() -> None:
    plan = plan_materialization(
        _context(
            method=BoundMethod.ALPHA_CROWN,
            requires_grad=True,
            alpha_enabled=True,
        ),
        policy=MaterializationPolicy.ALWAYS_STRUCTURED,
    )

    assert plan.action == MaterializationAction.REDUCE_BATCH
    assert plan.candidates[1].capability_legal is False


def test_per_case_oracle_selects_fastest_observed_feasible_action() -> None:
    context = _context(memory_budget_bytes=1_000)
    plan = plan_materialization_oracle(
        context,
        (
            MaterializationObservation(
                action=MaterializationAction.DENSE,
                status="ok",
                peak_bytes=700,
                latency_ms=8.0,
            ),
            MaterializationObservation(
                action=MaterializationAction.STRUCTURED,
                status="ok",
                peak_bytes=600,
                latency_ms=3.0,
            ),
        ),
    )

    assert plan.policy == MaterializationPolicy.ORACLE
    assert plan.action == MaterializationAction.STRUCTURED
    assert plan.reason == "oracle_fastest_observed_structured"


def test_per_case_oracle_preserves_capability_and_oom_failures() -> None:
    context = _context(
        method=BoundMethod.ALPHA_CROWN,
        requires_grad=True,
        alpha_enabled=True,
        memory_budget_bytes=1_000,
    )
    plan = plan_materialization_oracle(
        context,
        (
            MaterializationObservation(
                action=MaterializationAction.DENSE,
                status="oom",
                peak_bytes=None,
                latency_ms=None,
            ),
            MaterializationObservation(
                action=MaterializationAction.STRUCTURED,
                status="ok",
                peak_bytes=100,
                latency_ms=1.0,
            ),
        ),
    )

    assert plan.action == MaterializationAction.REDUCE_BATCH
    assert plan.candidates[1].capability_legal is False
    assert "target_lacks_structured_autograd" in plan.candidates[1].reasons


def test_explainable_cost_model_fits_calibration_only_and_predicts_new_context() -> (
    None
):
    samples = []
    for index in range(1, 9):
        context = _context(
            dense_bytes=200 * index,
            structured_bytes=120 * index,
            memory_budget_bytes=100_000,
        )
        samples.extend(
            [
                MaterializationCalibrationSample(
                    context=context,
                    observation=MaterializationObservation(
                        action=MaterializationAction.DENSE,
                        status="ok",
                        peak_bytes=400 * index + 1_000,
                        latency_ms=2.0 * index + 1.0,
                    ),
                ),
                MaterializationCalibrationSample(
                    context=context,
                    observation=MaterializationObservation(
                        action=MaterializationAction.STRUCTURED,
                        status="ok",
                        peak_bytes=240 * index + 1_200,
                        latency_ms=5.0 * index + 2.0,
                    ),
                ),
            ]
        )
    model = fit_materialization_cost_model(samples, ridge=1e-4)
    heldout = _context(
        dense_bytes=1_900,
        structured_bytes=1_100,
        memory_budget_bytes=100_000,
    )
    predicted = model.predict(heldout)

    assert model.schema_version == COST_MODEL_SCHEMA_VERSION
    assert [action.training_samples for action in model.actions] == [8, 8]
    assert predicted is not heldout
    assert predicted.bound_method == heldout.bound_method
    assert predicted.operator_summary.dense_a_bytes > 0
    assert predicted.operator_summary.structured_base_bytes > 0
    assert predicted.operator_summary.dense_latency_ms is not None
    assert predicted.operator_summary.structured_latency_ms is not None
    json.dumps(model.to_dict(), sort_keys=True)


def test_cost_model_rejects_missing_successful_action_calibration() -> None:
    context = _context()
    samples = [
        MaterializationCalibrationSample(
            context=context,
            observation=MaterializationObservation(
                action=MaterializationAction.DENSE,
                status="ok",
                peak_bytes=500,
                latency_ms=1.0,
            ),
        ),
        MaterializationCalibrationSample(
            context=context,
            observation=MaterializationObservation(
                action=MaterializationAction.STRUCTURED,
                status="oom",
                peak_bytes=None,
                latency_ms=None,
            ),
        ),
    ]

    with pytest.raises(ValueError, match="structured"):
        fit_materialization_cost_model(samples)


def test_explicit_plan_overrides_legacy_relu_mode_and_preserves_bounds() -> None:
    module = _relu_mlp()
    spec = InputSpec.linf(
        value_name="input",
        center=torch.randn(2, 4, dtype=torch.float64),
        eps=0.1,
    )
    linear_spec = torch.randn(2, 5, 3, dtype=torch.float64)
    dense_plan = plan_materialization(_context())
    structured_plan = plan_materialization(
        _context(dense_bytes=1_000, structured_bytes=200)
    )

    with _relu_backward_mode("structured"):
        with trace_materializations() as dense_trace:
            dense = run_crown_ibp_mlp(
                module,
                spec,
                linear_spec_C=linear_spec,
                materialization_plan=dense_plan,
            )
    with _relu_backward_mode("dense"):
        with trace_materializations() as structured_trace:
            structured = run_crown_ibp_mlp(
                module,
                spec,
                linear_spec_C=linear_spec,
                materialization_plan=structured_plan,
            )

    assert torch.allclose(structured.lower, dense.lower, atol=1e-10, rtol=1e-10)
    assert torch.allclose(structured.upper, dense.upper, atol=1e-10, rtol=1e-10)
    assert dense_trace.summary()["by_lifetime_class"]["persistent"]["count"] > 0
    assert "persistent" not in structured_trace.summary()["by_lifetime_class"]


def test_real_crown_query_builds_shape_derived_context_and_plan() -> None:
    module = _relu_mlp()
    spec = InputSpec.linf(
        value_name="input",
        center=torch.randn(2, 4, dtype=torch.float64),
        eps=0.1,
    )
    linear_spec = torch.randn(2, 5, 3, dtype=torch.float64)

    context, plan = plan_crown_materialization(
        module,
        spec,
        linear_spec_C=linear_spec,
        options=MaterializationPlannerOptions(memory_budget_bytes=10_000),
    )

    assert context.bound_method == BoundMethod.CROWN
    assert context.batch_size == 2
    assert context.domain_batch_size == 2
    assert context.spec_size == 5
    assert context.operator_summary.dense_a_bytes == 960
    assert context.operator_summary.structured_base_bytes == 480
    assert context.operator_summary.scale_bytes == 384
    assert context.operator_summary.temporary_bytes == 960
    assert plan.action == MaterializationAction.DENSE


def test_reduce_batch_plan_is_returned_to_host_runtime() -> None:
    module = _relu_mlp()
    spec = InputSpec.linf(
        value_name="input",
        center=torch.randn(2, 4, dtype=torch.float64),
        eps=0.1,
    )
    plan = plan_materialization(
        _context(dense_bytes=1_000, structured_bytes=500, memory_budget_bytes=400)
    )

    with pytest.raises(MaterializationReplanRequired) as error:
        run_crown_ibp_mlp(module, spec, materialization_plan=plan)
    assert error.value.plan == plan
    assert "recommended_domain_batch_size=" in str(error.value)


@pytest.mark.parametrize("runner", [run_alpha_crown_mlp, run_alpha_beta_crown_mlp])
def test_optimized_bound_runtime_rejects_structured_plan(runner) -> None:
    module = _relu_mlp()
    spec = InputSpec.linf(
        value_name="input",
        center=torch.randn(1, 4, dtype=torch.float64),
        eps=0.1,
    )
    structured_plan = plan_materialization(
        _context(dense_bytes=1_000, structured_bytes=200)
    )
    assert structured_plan.action == MaterializationAction.STRUCTURED

    with pytest.raises(ValueError, match="structured optimized bounds"):
        runner(module, spec, steps=1, materialization_plan=structured_plan)


def test_alpha_runtime_executes_explicit_dense_plan() -> None:
    module = _relu_mlp()
    spec = InputSpec.linf(
        value_name="input",
        center=torch.randn(1, 4, dtype=torch.float64),
        eps=0.1,
    )
    dense_plan = plan_materialization(_context())

    bounds, _state, _stats = run_alpha_crown_mlp(
        module,
        spec,
        steps=1,
        materialization_plan=dense_plan,
    )

    assert torch.isfinite(bounds.lower).all()
    assert torch.isfinite(bounds.upper).all()
