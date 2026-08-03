"""Contracts for backend-aware PR-12 execution candidates."""

import pytest

from boundflow.planner.execution_candidate import (
    BACKEND_CANDIDATE_SCHEMA_VERSION,
    BackendVariant,
    ExecutionCandidate,
    ExecutionContext,
    OperatorFamily,
    PlacementKind,
    capability_rejections,
    fused_tir_linear_v1_capability,
)
from boundflow.planner.materialization import BoundMethod, OptimizationStage


def _context(**updates: object) -> ExecutionContext:
    values: dict[str, object] = {
        "bound_method": BoundMethod.CROWN,
        "requires_grad": False,
        "optimization_stage": OptimizationStage.INFERENCE,
        "alpha_enabled": False,
        "beta_enabled": False,
        "split_state_present": False,
        "operator_family": OperatorFamily.LINEAR,
        "device": "cuda",
        "dtype": "float32",
        "layout": "contiguous",
        "static_shape": True,
    }
    values.update(updates)
    return ExecutionContext(**values)  # type: ignore[arg-type]


def test_fused_tir_v1_accepts_only_legal_plain_crown_context() -> None:
    capability = fused_tir_linear_v1_capability()

    assert capability_rejections(_context(), capability) == ()
    assert "conv2d_unsupported" in capability_rejections(
        _context(operator_family=OperatorFamily.CONV2D, layout="nchw"), capability
    )


@pytest.mark.parametrize(
    ("updates", "reason"),
    [
        ({"bound_method": BoundMethod.ALPHA_CROWN}, "bound_method_unsupported"),
        ({"requires_grad": True}, "requires_grad_unsupported"),
        ({"alpha_enabled": True}, "alpha_unsupported"),
        ({"beta_enabled": True}, "beta_unsupported"),
        ({"split_state_present": True}, "split_state_unsupported"),
        ({"dtype": "float16"}, "dtype_unsupported"),
        ({"device": "cpu"}, "device_unsupported"),
        ({"static_shape": False}, "dynamic_shape_unsupported"),
        (
            {"optimization_stage": OptimizationStage.TRAINING},
            "optimization_stage_unsupported",
        ),
    ],
)
def test_fused_tir_v1_reports_explicit_fallback_reason(
    updates: dict[str, object], reason: str
) -> None:
    assert reason in capability_rejections(
        _context(**updates), fused_tir_linear_v1_capability()
    )


def test_candidate_dump_keeps_placement_and_backend_independent() -> None:
    candidate = ExecutionCandidate(
        placement=PlacementKind.STRUCTURED,
        backend=BackendVariant.TVM_FUSED_TIR,
        domain_batch_size=3,
        spec_batch_size=17,
        materialization_points=(),
        capability_id=fused_tir_linear_v1_capability().capability_id,
        schedule_id="linear_two_kernel_v0",
        reason="avoid_scaled_a_global_write",
    )

    payload = candidate.to_dict()
    assert payload["schema_version"] == BACKEND_CANDIDATE_SCHEMA_VERSION
    assert payload["placement"] == "structured"
    assert payload["backend"] == "tvm_fused_tir"
