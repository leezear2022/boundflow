"""R3-2A P-anchor 10/9 optimizer trajectory correctness runtime."""

# pylint: disable=too-many-instance-attributes,too-many-locals,too-many-statements
# pylint: disable=too-many-boolean-expressions,protected-access
# pylint: disable=missing-function-docstring,too-many-arguments

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import json
import math

import torch

from ..ir.r3_bounded_arena import R31BBoundedArenaTraceV1
from .fsg4_b3_terminal_optimizer_schedule import (
    NativeTerminalOptimizerScheduleV1,
)
from .r3_compiled_p_alpha_vjp import (
    execute_r31b2_compiled_custom_backward_v1,
    R31B2CompiledReceiptV1,
)
from .r3_structured_owner_custom_backward import (
    execute_r31_native_oracle_v1,
    R31FullRegionPlanV1,
)
from .rvir_v4_production_state import production_tensor_sha256

R32A_TRAJECTORY_SCHEMA = "boundflow.r3-2a-p-optimizer-trajectory/v1"
R32A_REBIND_SCHEMA = "boundflow.r3-2a-dynamic-rebind/v1"


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _is_hash(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


class R32AExecutionMode(str, Enum):
    """Closed native/candidate worker vocabulary."""

    NATIVE = "native"
    CANDIDATE = "candidate"


@dataclass(frozen=True)
class R32ADynamicRebindReceiptV1:
    """One mutable-P-alpha instance identity over a fixed static recurrence."""

    base_plan_hash: str
    base_trace_hash: str
    trajectory_id: str
    evaluation_ordinal: int
    updates_before: int
    p_alpha_name: str
    p_alpha_content_sha256: str
    p_alpha_version: int
    immutable_content_hash: str
    rebound_plan_hash: str
    rebound_trace_hash: str
    alpha_learning_rate: float
    schema_version: str = R32A_REBIND_SCHEMA

    def validate(self) -> None:
        if (
            self.schema_version != R32A_REBIND_SCHEMA
            or not all(
                _is_hash(value)
                for value in (
                    self.base_plan_hash,
                    self.base_trace_hash,
                    self.trajectory_id,
                    self.p_alpha_content_sha256,
                    self.immutable_content_hash,
                    self.rebound_plan_hash,
                    self.rebound_trace_hash,
                )
            )
            or self.evaluation_ordinal < 0
            or self.updates_before != self.evaluation_ordinal
            or self.p_alpha_name != "relu/25/alpha"
            or self.p_alpha_version < 0
            or not math.isfinite(self.alpha_learning_rate)
            or self.alpha_learning_rate <= 0.0
        ):
            raise ValueError("R3-2A dynamic rebind receipt differs")

    def metadata(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "base_plan_hash": self.base_plan_hash,
            "base_trace_hash": self.base_trace_hash,
            "trajectory_id": self.trajectory_id,
            "evaluation_ordinal": self.evaluation_ordinal,
            "updates_before": self.updates_before,
            "p_alpha_name": self.p_alpha_name,
            "p_alpha_content_sha256": self.p_alpha_content_sha256,
            "p_alpha_version": self.p_alpha_version,
            "immutable_content_hash": self.immutable_content_hash,
            "rebound_plan_hash": self.rebound_plan_hash,
            "rebound_trace_hash": self.rebound_trace_hash,
            "alpha_learning_rate": self.alpha_learning_rate,
        }


def rebind_r32a_dynamic_instance_v1(
    base_plan: R31FullRegionPlanV1,
    base_trace: R31BBoundedArenaTraceV1,
    tensors: tuple[torch.Tensor, ...],
    *,
    trajectory_id: str,
    evaluation_ordinal: int,
    alpha_learning_rate: float,
) -> tuple[R31FullRegionPlanV1, R31BBoundedArenaTraceV1, R32ADynamicRebindReceiptV1]:
    """Rebind only mutable P-alpha while preserving every static/copy-in field."""

    base_plan.validate()
    base_trace.validate()
    if (
        base_trace.production_plan_hash != base_plan.stable_hash()
        or len(tensors) != len(base_plan.tensor_specs)
        or not _is_hash(trajectory_id)
        or evaluation_ordinal < 0
        or not math.isfinite(alpha_learning_rate)
        or alpha_learning_rate <= 0.0
    ):
        raise ValueError("R3-2A dynamic rebind admission differs")
    p_ordinal = base_plan.p_alpha_input_ordinal
    for ordinal, (spec, tensor) in enumerate(zip(base_plan.tensor_specs, tensors)):
        if tuple(tensor.shape) != spec.shape or str(tensor.dtype) != spec.dtype:
            raise ValueError(f"R3-2A runtime tensor schema differs: {spec.name}")
        if (
            ordinal != p_ordinal
            and production_tensor_sha256(tensor) != spec.content_sha256
        ):
            raise ValueError(f"R3-2A immutable tensor drifted: {spec.name}")
    p_alpha = tensors[p_ordinal]
    p_hash = production_tensor_sha256(p_alpha)
    rebound_specs = tuple(
        replace(spec, content_sha256=p_hash) if ordinal == p_ordinal else spec
        for ordinal, spec in enumerate(base_plan.tensor_specs)
    )
    rebound_plan = replace(base_plan, tensor_specs=rebound_specs)
    rebound_plan.validate()
    rebound_trace = replace(base_trace, production_plan_hash=rebound_plan.stable_hash())
    rebound_trace.validate()
    immutable_hash = _canonical_hash(
        [
            {
                "name": spec.name,
                "content_sha256": production_tensor_sha256(tensor),
                "version": tensor._version,
            }
            for ordinal, (spec, tensor) in enumerate(
                zip(base_plan.tensor_specs, tensors)
            )
            if ordinal != p_ordinal
        ]
    )
    receipt = R32ADynamicRebindReceiptV1(
        base_plan_hash=base_plan.stable_hash(),
        base_trace_hash=base_trace.stable_hash(),
        trajectory_id=trajectory_id,
        evaluation_ordinal=evaluation_ordinal,
        updates_before=evaluation_ordinal,
        p_alpha_name=base_plan.tensor_specs[p_ordinal].name,
        p_alpha_content_sha256=p_hash,
        p_alpha_version=p_alpha._version,
        immutable_content_hash=immutable_hash,
        rebound_plan_hash=rebound_plan.stable_hash(),
        rebound_trace_hash=rebound_trace.stable_hash(),
        alpha_learning_rate=alpha_learning_rate,
    )
    receipt.validate()
    return rebound_plan, rebound_trace, receipt


@dataclass(frozen=True)
class R32AOptimizerStateV1:
    """Canonical Adam state after an optional mutation."""

    initialized: bool
    step: float
    exp_avg: torch.Tensor
    exp_avg_sq: torch.Tensor

    def validate(self, *, alpha_shape: tuple[int, ...]) -> None:
        if self.initialized:
            if (
                self.step < 1.0
                or tuple(self.exp_avg.shape) != alpha_shape
                or tuple(self.exp_avg_sq.shape) != alpha_shape
                or not bool(torch.isfinite(self.exp_avg).all().item())
                or not bool(torch.isfinite(self.exp_avg_sq).all().item())
            ):
                raise ValueError("R3-2A initialized Adam state differs")
        elif self.step != 0.0 or self.exp_avg.numel() or self.exp_avg_sq.numel():
            raise ValueError("R3-2A empty Adam state differs")

    def metadata(self) -> dict[str, object]:
        return {
            "initialized": self.initialized,
            "step": self.step,
            "exp_avg_sha256": production_tensor_sha256(self.exp_avg),
            "exp_avg_sq_sha256": production_tensor_sha256(self.exp_avg_sq),
            "exp_avg_shape": list(self.exp_avg.shape),
            "exp_avg_sq_shape": list(self.exp_avg_sq.shape),
        }


def _optimizer_state(
    optimizer: torch.optim.Optimizer, alpha: torch.Tensor
) -> R32AOptimizerStateV1:
    raw = optimizer.state.get(alpha)
    if not raw:
        empty = torch.empty(0, dtype=alpha.dtype)
        state = R32AOptimizerStateV1(False, 0.0, empty, empty.clone())
    else:
        raw_step = raw.get("step")
        step = float(raw_step.item() if torch.is_tensor(raw_step) else raw_step)
        exp_avg = raw.get("exp_avg")
        exp_avg_sq = raw.get("exp_avg_sq")
        if not torch.is_tensor(exp_avg) or not torch.is_tensor(exp_avg_sq):
            raise TypeError("R3-2A Adam moment state differs")
        state = R32AOptimizerStateV1(
            True,
            step,
            exp_avg.detach().cpu().contiguous().clone(),
            exp_avg_sq.detach().cpu().contiguous().clone(),
        )
    state.validate(alpha_shape=tuple(alpha.shape))
    return state


@dataclass(frozen=True)
class R32ATrajectoryStepV1:
    """One evaluation and its optional optimizer mutation evidence."""

    evaluation_ordinal: int
    update_after: bool
    alpha_learning_rate: float
    alpha_before: torch.Tensor
    lower: torch.Tensor
    gradient: torch.Tensor
    alpha_after: torch.Tensor
    optimizer_after: R32AOptimizerStateV1
    rebind: R32ADynamicRebindReceiptV1
    compiled_receipt: R31B2CompiledReceiptV1 | None
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    allocated_after_bytes: int
    reserved_after_bytes: int

    def validate(self, *, mode: R32AExecutionMode) -> None:
        shape = (2, 1, 6, 86)
        self.rebind.validate()
        self.optimizer_after.validate(alpha_shape=shape)
        if (
            self.evaluation_ordinal != self.rebind.evaluation_ordinal
            or self.update_after != (self.evaluation_ordinal < 9)
            or not math.isclose(
                self.alpha_learning_rate,
                0.01 * 0.98**self.evaluation_ordinal,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or tuple(self.alpha_before.shape) != shape
            or tuple(self.alpha_after.shape) != shape
            or tuple(self.gradient.shape) != shape
            or tuple(self.lower.shape) != (6, 1)
            or min(
                self.peak_allocated_bytes,
                self.peak_reserved_bytes,
                self.allocated_after_bytes,
                self.reserved_after_bytes,
            )
            < 0
            or self.peak_allocated_bytes < self.allocated_after_bytes
            or self.peak_reserved_bytes < self.reserved_after_bytes
            or not all(
                bool(torch.isfinite(value).all().item())
                for value in (
                    self.alpha_before,
                    self.alpha_after,
                    self.gradient,
                    self.lower,
                )
            )
            or not self.optimizer_after.initialized
            or self.optimizer_after.step != float(min(self.evaluation_ordinal + 1, 9))
        ):
            raise ValueError("R3-2A trajectory step differs")
        if mode == R32AExecutionMode.CANDIDATE:
            if self.compiled_receipt is None:
                raise ValueError("R3-2A candidate compiled receipt is absent")
            self.compiled_receipt.validate()
            if (
                self.compiled_receipt.production_plan_hash
                != self.rebind.rebound_plan_hash
            ):
                raise ValueError("R3-2A candidate plan receipt differs")
        elif self.compiled_receipt is not None:
            raise ValueError("R3-2A native compiled receipt is present")

    def metadata(self) -> dict[str, object]:
        return {
            "evaluation_ordinal": self.evaluation_ordinal,
            "update_after": self.update_after,
            "alpha_learning_rate": self.alpha_learning_rate,
            "alpha_before_sha256": production_tensor_sha256(self.alpha_before),
            "lower_sha256": production_tensor_sha256(self.lower),
            "gradient_sha256": production_tensor_sha256(self.gradient),
            "alpha_after_sha256": production_tensor_sha256(self.alpha_after),
            "optimizer_after": self.optimizer_after.metadata(),
            "rebind": self.rebind.metadata(),
            "compiled_receipt": (
                self.compiled_receipt.__dict__
                if self.compiled_receipt is not None
                else None
            ),
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "peak_reserved_bytes": self.peak_reserved_bytes,
            "allocated_after_bytes": self.allocated_after_bytes,
            "reserved_after_bytes": self.reserved_after_bytes,
        }


@dataclass(frozen=True)
class R32ATrajectoryResultV1:
    """Complete correctness-only 10/9 P-anchor trajectory."""

    mode: R32AExecutionMode
    trajectory_id: str
    base_plan_hash: str
    base_trace_hash: str
    immutable_content_hash: str
    initial_alpha: torch.Tensor
    terminal_alpha: torch.Tensor
    steps: tuple[R32ATrajectoryStepV1, ...]
    initial_immutable_versions: tuple[int, ...]
    terminal_immutable_versions: tuple[int, ...]
    optimizer_mutation_count: int
    scheduler_mutation_count: int
    fallback_count: int
    eager_candidate_count: int
    native_shadow_count: int
    timing_recorded: bool = False
    performance_claimed: bool = False
    schema_version: str = R32A_TRAJECTORY_SCHEMA

    @property
    def peak_allocated_bytes(self) -> int:
        return max(step.peak_allocated_bytes for step in self.steps)

    @property
    def peak_reserved_bytes(self) -> int:
        return max(step.peak_reserved_bytes for step in self.steps)

    def validate(self) -> None:
        if (
            self.schema_version != R32A_TRAJECTORY_SCHEMA
            or not all(
                _is_hash(value)
                for value in (
                    self.trajectory_id,
                    self.base_plan_hash,
                    self.base_trace_hash,
                    self.immutable_content_hash,
                )
            )
            or tuple(self.initial_alpha.shape) != (2, 1, 6, 86)
            or tuple(self.terminal_alpha.shape) != (2, 1, 6, 86)
            or len(self.steps) != 10
            or tuple(step.evaluation_ordinal for step in self.steps) != tuple(range(10))
            or self.optimizer_mutation_count != 9
            or self.scheduler_mutation_count != 9
            or self.initial_immutable_versions != self.terminal_immutable_versions
            or self.fallback_count
            or self.eager_candidate_count
            or self.native_shadow_count
            or self.timing_recorded
            or self.performance_claimed
        ):
            raise ValueError("R3-2A trajectory result differs")
        previous = self.initial_alpha
        for step in self.steps:
            step.validate(mode=self.mode)
            if not torch.equal(step.alpha_before, previous):
                raise ValueError("R3-2A alpha mutation lineage differs")
            previous = step.alpha_after
        if not torch.equal(previous, self.terminal_alpha):
            raise ValueError("R3-2A terminal alpha lineage differs")
        immutable_hashes = {step.rebind.immutable_content_hash for step in self.steps}
        if immutable_hashes != {self.immutable_content_hash}:
            raise ValueError("R3-2A immutable content lineage differs")

    def metadata(self) -> dict[str, object]:
        self.validate()
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "mode": self.mode.value,
            "trajectory_id": self.trajectory_id,
            "base_plan_hash": self.base_plan_hash,
            "base_trace_hash": self.base_trace_hash,
            "immutable_content_hash": self.immutable_content_hash,
            "initial_alpha_sha256": production_tensor_sha256(self.initial_alpha),
            "terminal_alpha_sha256": production_tensor_sha256(self.terminal_alpha),
            "steps": [step.metadata() for step in self.steps],
            "initial_immutable_versions": list(self.initial_immutable_versions),
            "terminal_immutable_versions": list(self.terminal_immutable_versions),
            "optimizer_mutation_count": self.optimizer_mutation_count,
            "scheduler_mutation_count": self.scheduler_mutation_count,
            "fallback_count": self.fallback_count,
            "eager_candidate_count": self.eager_candidate_count,
            "native_shadow_count": self.native_shadow_count,
            "timing_recorded": self.timing_recorded,
            "performance_claimed": self.performance_claimed,
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "peak_reserved_bytes": self.peak_reserved_bytes,
        }
        payload["trajectory_hash"] = _canonical_hash(payload)
        return payload


def execute_r32a_optimizer_trajectory_v1(
    base_plan: R31FullRegionPlanV1,
    base_trace: R31BBoundedArenaTraceV1,
    tensors: tuple[torch.Tensor, ...],
    *,
    schedule: NativeTerminalOptimizerScheduleV1,
    mode: R32AExecutionMode,
) -> R32ATrajectoryResultV1:
    """Execute a correctness-capturing P-only 10/9 native or compiled path."""

    base_plan.validate()
    base_trace.validate()
    schedule.validate()
    if schedule.evaluation_count != 10 or schedule.update_count != 9:
        raise ValueError("R3-2A optimizer schedule differs")
    p_ordinal = base_plan.p_alpha_input_ordinal
    alpha = tensors[p_ordinal]
    if not alpha.is_leaf or not alpha.requires_grad:
        raise ValueError("R3-2A P-alpha owner differs")
    immutable_ordinals = tuple(
        ordinal for ordinal in range(len(tensors)) if ordinal != p_ordinal
    )
    initial_versions = tuple(
        tensors[ordinal]._version for ordinal in immutable_ordinals
    )
    trajectory_id = _canonical_hash(
        {
            "base_plan_hash": base_plan.stable_hash(),
            "base_trace_hash": base_trace.stable_hash(),
            "initial_alpha": production_tensor_sha256(alpha),
            "schedule_hash": schedule.stable_hash(),
            "mode_independent": True,
        }
    )
    optimizer = torch.optim.Adam([alpha], lr=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    initial_alpha = alpha.detach().cpu().contiguous().clone()
    rows: list[R32ATrajectoryStepV1] = []
    stream = (
        torch.cuda.Stream(device=alpha.device)
        if mode == R32AExecutionMode.CANDIDATE
        else None
    )
    for action in schedule.actions:
        lr = float(optimizer.param_groups[0]["lr"])
        if not math.isclose(lr, action.alpha_learning_rate, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("R3-2A runtime learning rate differs")
        rebound_plan, rebound_trace, rebind = rebind_r32a_dynamic_instance_v1(
            base_plan,
            base_trace,
            tensors,
            trajectory_id=trajectory_id,
            evaluation_ordinal=action.evaluation_ordinal,
            alpha_learning_rate=lr,
        )
        alpha_before = alpha.detach().cpu().contiguous().clone()
        compiled_receipt: R31B2CompiledReceiptV1 | None = None
        if mode == R32AExecutionMode.NATIVE:
            lower, gradient = execute_r31_native_oracle_v1(rebound_plan, tensors)
        else:
            assert stream is not None
            with torch.cuda.stream(stream):
                compiled = execute_r31b2_compiled_custom_backward_v1(
                    rebound_plan, rebound_trace, tensors
                )
            stream.synchronize()
            lower = compiled.final_lower
            gradient = compiled.compressed_alpha_gradient
            compiled_receipt = compiled.receipt
        lower_cpu = lower.detach().cpu().contiguous().clone()
        gradient_cpu = gradient.detach().cpu().contiguous().clone()
        if action.update_after:
            optimizer.zero_grad(set_to_none=True)
            alpha.grad = gradient.detach().clone()
            optimizer.step()
            with torch.no_grad():
                alpha.clamp_(0.0, 1.0)
            scheduler.step()
        peak_allocated = torch.cuda.max_memory_allocated(alpha.device)
        peak_reserved = torch.cuda.max_memory_reserved(alpha.device)
        allocated_after = torch.cuda.memory_allocated(alpha.device)
        reserved_after = torch.cuda.memory_reserved(alpha.device)
        alpha_after = alpha.detach().cpu().contiguous().clone()
        rows.append(
            R32ATrajectoryStepV1(
                evaluation_ordinal=action.evaluation_ordinal,
                update_after=action.update_after,
                alpha_learning_rate=lr,
                alpha_before=alpha_before,
                lower=lower_cpu,
                gradient=gradient_cpu,
                alpha_after=alpha_after,
                optimizer_after=_optimizer_state(optimizer, alpha),
                rebind=rebind,
                compiled_receipt=compiled_receipt,
                peak_allocated_bytes=peak_allocated,
                peak_reserved_bytes=peak_reserved,
                allocated_after_bytes=allocated_after,
                reserved_after_bytes=reserved_after,
            )
        )
    terminal_versions = tuple(
        tensors[ordinal]._version for ordinal in immutable_ordinals
    )
    result = R32ATrajectoryResultV1(
        mode=mode,
        trajectory_id=trajectory_id,
        base_plan_hash=base_plan.stable_hash(),
        base_trace_hash=base_trace.stable_hash(),
        immutable_content_hash=rows[0].rebind.immutable_content_hash,
        initial_alpha=initial_alpha,
        terminal_alpha=alpha.detach().cpu().contiguous().clone(),
        steps=tuple(rows),
        initial_immutable_versions=initial_versions,
        terminal_immutable_versions=terminal_versions,
        optimizer_mutation_count=9,
        scheduler_mutation_count=9,
        fallback_count=0,
        eager_candidate_count=0,
        native_shadow_count=0,
    )
    result.validate()
    return result


__all__ = [
    "execute_r32a_optimizer_trajectory_v1",
    "rebind_r32a_dynamic_instance_v1",
    "R32ADynamicRebindReceiptV1",
    "R32AExecutionMode",
    "R32AOptimizerStateV1",
    "R32ATrajectoryResultV1",
    "R32ATrajectoryStepV1",
]
