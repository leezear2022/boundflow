"""S3 host-policy optimizer loop over the S2 canonical direct VJP."""

# mypy: disable-error-code=import-untyped
# pylint: disable=too-many-instance-attributes,too-many-locals
# pylint: disable=too-many-boolean-expressions,protected-access
# pylint: disable=missing-function-docstring,too-many-statements
# pylint: disable=duplicate-code

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math

import torch

from boundflow.runtime.asplos27_s2_crown_pipeline import (
    PreparedS2CrownProgramV1,
)
from boundflow.runtime.fsg4_b3_terminal_optimizer_schedule import (
    NativeTerminalOptimizerScheduleV1,
)
from boundflow.runtime.r3_structured_owner_custom_backward import (
    R31FullRegionPlanV1,
)

S3_EXECUTION_RECEIPT_SCHEMA = "boundflow.asplos27-s3-optimizer-execution/v1"


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _clone_cpu(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu().contiguous().clone()


@dataclass(frozen=True)
class S3OptimizerStepCaptureV1:
    """Untimed semantic state for one frozen optimizer evaluation."""

    evaluation_ordinal: int
    update_after: bool
    alpha_learning_rate: float
    alpha_before: torch.Tensor
    lower: torch.Tensor
    gradient: torch.Tensor
    alpha_after: torch.Tensor
    optimizer_step: float
    optimizer_exp_avg: torch.Tensor
    optimizer_exp_avg_sq: torch.Tensor

    def validate(self) -> None:
        alpha_shape = (2, 1, 6, 86)
        if (
            self.evaluation_ordinal not in range(10)
            or self.update_after != (self.evaluation_ordinal < 9)
            or not math.isfinite(self.alpha_learning_rate)
            or self.alpha_learning_rate <= 0.0
            or tuple(self.alpha_before.shape) != alpha_shape
            or tuple(self.lower.shape) != (6, 1)
            or tuple(self.gradient.shape) != alpha_shape
            or tuple(self.alpha_after.shape) != alpha_shape
            or tuple(self.optimizer_exp_avg.shape) != alpha_shape
            or tuple(self.optimizer_exp_avg_sq.shape) != alpha_shape
            or self.optimizer_step != float(min(self.evaluation_ordinal + 1, 9))
            or any(
                value.device.type != "cpu" or value.dtype != torch.float32
                for value in (
                    self.alpha_before,
                    self.lower,
                    self.gradient,
                    self.alpha_after,
                    self.optimizer_exp_avg,
                    self.optimizer_exp_avg_sq,
                )
            )
        ):
            raise ValueError("S3 optimizer step capture differs")


@dataclass(frozen=True)
class S3OptimizerExecutionReceiptV1:
    """Aggregate execution evidence for one complete 10/9 local wrapper."""

    production_plan_hash: str
    trace_hash: str
    optimizer_schedule_hash: str
    s2_receipt_hash: str
    evaluation_count: int
    optimizer_mutation_count: int
    scheduler_mutation_count: int
    custom_forward_count: int
    custom_backward_count: int
    forward_graph_replay_count: int
    selected_graph_replay_count: int
    selected_vm_invocation_count: int
    selected_output_copy_count: int
    warm_dlpack_view_count: int
    host_policy_cut_count: int
    autograd_function_count: int
    executor_registry_count: int
    fallback_count: int
    eager_candidate_count: int
    native_shadow_count: int
    saved_dense_a_count: int
    saved_autograd_history: bool
    schema_version: str = S3_EXECUTION_RECEIPT_SCHEMA
    performance_claimed: bool = False

    def validate(self) -> None:
        hashes = (
            self.production_plan_hash,
            self.trace_hash,
            self.optimizer_schedule_hash,
            self.s2_receipt_hash,
        )
        if (
            self.schema_version != S3_EXECUTION_RECEIPT_SCHEMA
            or any(
                len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in hashes
            )
            or self.evaluation_count != 10
            or self.optimizer_mutation_count != 9
            or self.scheduler_mutation_count != 9
            or self.custom_forward_count != 10
            or self.custom_backward_count != 10
            or self.forward_graph_replay_count != 10
            or self.selected_graph_replay_count != 0
            or self.selected_vm_invocation_count != 10
            or self.selected_output_copy_count != 10
            or self.warm_dlpack_view_count != 0
            or self.host_policy_cut_count != 10
            or self.autograd_function_count
            or self.executor_registry_count
            or self.fallback_count
            or self.eager_candidate_count
            or self.native_shadow_count
            or self.saved_dense_a_count
            or self.saved_autograd_history
            or self.performance_claimed
        ):
            raise ValueError("S3 optimizer execution receipt differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        result = asdict(self)
        result["receipt_hash"] = _canonical_hash(result)
        return result


@dataclass(frozen=True)
class S3OptimizerResultV1:
    """Terminal state plus optional untimed per-step semantic capture."""

    terminal_lower: torch.Tensor
    terminal_alpha: torch.Tensor
    steps: tuple[S3OptimizerStepCaptureV1, ...]
    receipt: S3OptimizerExecutionReceiptV1

    def validate(self, *, capture: bool) -> None:
        self.receipt.validate()
        if (
            tuple(self.terminal_lower.shape) != (6, 1)
            or tuple(self.terminal_alpha.shape) != (2, 1, 6, 86)
            or len(self.steps) != (10 if capture else 0)
        ):
            raise ValueError("S3 optimizer result differs")
        for ordinal, step in enumerate(self.steps):
            step.validate()
            if step.evaluation_ordinal != ordinal:
                raise ValueError("S3 optimizer capture order differs")


def _optimizer_state(
    optimizer: torch.optim.Optimizer, alpha: torch.Tensor
) -> tuple[float, torch.Tensor, torch.Tensor]:
    state = optimizer.state.get(alpha)
    if not state:
        raise RuntimeError("S3 optimizer state is absent")
    raw_step = state.get("step")
    exp_avg = state.get("exp_avg")
    exp_avg_sq = state.get("exp_avg_sq")
    if not torch.is_tensor(exp_avg) or not torch.is_tensor(exp_avg_sq):
        raise TypeError("S3 optimizer moments differ")
    step = float(raw_step.item() if torch.is_tensor(raw_step) else raw_step)
    return step, exp_avg, exp_avg_sq


def execute_asplos27_s3_optimizer_v1(
    plan: R31FullRegionPlanV1,
    tensors: tuple[torch.Tensor, ...],
    schedule: NativeTerminalOptimizerScheduleV1,
    candidate: PreparedS2CrownProgramV1,
    *,
    capture: bool = False,
) -> S3OptimizerResultV1:
    """Execute ten direct VJPs with a host-owned frozen Adam policy cut."""

    plan.validate()
    schedule.validate()
    if (
        candidate.tensors is not tensors
        or candidate.plan.stable_hash() != plan.stable_hash()
        or len(schedule.actions) != 10
    ):
        raise ValueError("S3 optimizer candidate binding differs")
    alpha = tensors[plan.p_alpha_input_ordinal]
    if not alpha.is_leaf or not alpha.requires_grad:
        raise ValueError("S3 optimizer alpha owner differs")

    optimizer = torch.optim.Adam([alpha], lr=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    terminal: torch.Tensor | None = None
    rows: list[S3OptimizerStepCaptureV1] = []
    custom_forward_count = 0
    custom_backward_count = 0
    forward_graph_replay_count = 0
    selected_graph_replay_count = 0
    selected_vm_invocation_count = 0
    selected_output_copy_count = 0
    candidate.begin_sample()

    for action in schedule.actions:
        ordinal = action.evaluation_ordinal
        learning_rate = float(optimizer.param_groups[0]["lr"])
        if not math.isclose(
            learning_rate,
            action.alpha_learning_rate,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("S3 optimizer learning rate differs")
        alpha_before = _clone_cpu(alpha) if capture else None
        candidate.begin_evaluation(ordinal)
        lower = candidate.forward()
        gradient = candidate.backward(candidate.upstream_gradient)
        if (
            candidate.custom_forward_count != 1
            or candidate.custom_backward_count != 1
            or candidate.s2_forward_graph_replay_count != 1
            or candidate.s2_selected_vm_invocation_count != 1
            or candidate.s2_selected_output_copy_count != 1
        ):
            raise RuntimeError("S3 optimizer per-step execution differs")
        custom_forward_count += 1
        custom_backward_count += 1
        forward_graph_replay_count += 1
        selected_vm_invocation_count += 1
        selected_output_copy_count += 1

        lower_capture = _clone_cpu(lower) if capture else None
        gradient_capture = _clone_cpu(gradient) if capture else None
        if action.update_after:
            optimizer.zero_grad(set_to_none=True)
            alpha.grad = gradient.detach()
            optimizer.step()
            with torch.no_grad():
                alpha.clamp_(0.0, 1.0)
            scheduler.step()
        else:
            terminal = lower
        if capture:
            step, exp_avg, exp_avg_sq = _optimizer_state(optimizer, alpha)
            row = S3OptimizerStepCaptureV1(
                evaluation_ordinal=ordinal,
                update_after=action.update_after,
                alpha_learning_rate=learning_rate,
                alpha_before=alpha_before,  # type: ignore[arg-type]
                lower=lower_capture,  # type: ignore[arg-type]
                gradient=gradient_capture,  # type: ignore[arg-type]
                alpha_after=_clone_cpu(alpha),
                optimizer_step=step,
                optimizer_exp_avg=_clone_cpu(exp_avg),
                optimizer_exp_avg_sq=_clone_cpu(exp_avg_sq),
            )
            row.validate()
            rows.append(row)

    if terminal is None:
        raise RuntimeError("S3 optimizer terminal lower is absent")
    s2_receipt = candidate.execution_receipt().to_dict()
    receipt = S3OptimizerExecutionReceiptV1(
        production_plan_hash=plan.stable_hash(),
        trace_hash=candidate.trace.stable_hash(),
        optimizer_schedule_hash=schedule.stable_hash(),
        s2_receipt_hash=str(s2_receipt["receipt_hash"]),
        evaluation_count=10,
        optimizer_mutation_count=9,
        scheduler_mutation_count=9,
        custom_forward_count=custom_forward_count,
        custom_backward_count=custom_backward_count,
        forward_graph_replay_count=forward_graph_replay_count,
        selected_graph_replay_count=selected_graph_replay_count,
        selected_vm_invocation_count=selected_vm_invocation_count,
        selected_output_copy_count=selected_output_copy_count,
        warm_dlpack_view_count=0,
        host_policy_cut_count=10,
        autograd_function_count=0,
        executor_registry_count=0,
        fallback_count=0,
        eager_candidate_count=0,
        native_shadow_count=0,
        saved_dense_a_count=0,
        saved_autograd_history=False,
    )
    result = S3OptimizerResultV1(
        terminal_lower=terminal.detach(),
        terminal_alpha=alpha.detach(),
        steps=tuple(rows),
        receipt=receipt,
    )
    result.validate(capture=capture)
    return result


__all__ = [
    "execute_asplos27_s3_optimizer_v1",
    "S3OptimizerExecutionReceiptV1",
    "S3OptimizerResultV1",
    "S3OptimizerStepCaptureV1",
]
