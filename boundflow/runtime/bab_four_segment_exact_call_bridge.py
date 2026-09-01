"""RVIR exact-call bridge for the four-segment activation-BaB TIR owner."""

# mypy: disable-error-code=import-untyped
# pylint: disable=import-error,protected-access,too-many-arguments
# pylint: disable=too-many-positional-arguments,too-many-locals
# pylint: disable=too-many-instance-attributes,missing-function-docstring
# pylint: disable=too-many-boolean-expressions,duplicate-code

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import time
from typing import Mapping, cast

import torch

from boundflow.domains.interval import IntervalState
from boundflow.frontends.plain_crown_bound_ir import (
    plain_crown_primal_graph_hash,
    relu_split_state_hash,
)
from boundflow.ir.task import BFTaskModule
from boundflow.runtime.asplos27_s4_exact_call_bridge import (
    _dense_terminal_state,
    _runtime_sources,
    PreparedS4ExactCallRegionV1,
)
from boundflow.runtime.bab_four_segment_optimizer import (
    PreparedBabFourSegmentOptimizerV1,
)
from boundflow.runtime.crown_ibp import _forward_ibp_trace_mlp
from boundflow.runtime.fsg4_b3_prepared_core import CorePlanInstanceV1
from boundflow.runtime.fsg4_b3_terminal_optimizer_schedule import (
    NativeOptimizerForwardTraceV1,
    NativeTerminalOptimizerResultV1,
    NativeTerminalOptimizerScheduleV1,
)
from boundflow.runtime.fsg4_b4a_terminal_lower_adjoint_handoff import (
    _lineage,
    _topology_hash,
    NativeTerminalLowerAdjointHandoffV1,
    NativeTerminalOptimizerHandoffResultB4A,
)
from boundflow.runtime.native_alpha_beta_optimization_state import (
    NativeAlphaBetaOptimizationState,
)
from boundflow.runtime.rvir_v4_optimizer_mutation import ProductionMutationPolicyV4
from boundflow.runtime.rvir_v4_pre_state_initializer import ProductionReluTopologyV4
from boundflow.runtime.task_executor import InputSpec

BAB_FOUR_SEGMENT_EXACT_CALL_SCHEMA = "boundflow.bab-four-segment-exact-call/v1"


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


@dataclass(frozen=True)
class BabFourSegmentExactCallReceiptV1:
    """Live solver seam identity and execution inventory."""

    exact_call_id: str
    assets_hash: str
    production_plan_hash: str
    source_state_hash: str
    terminal_state_hash: str
    rebind_ns: int
    optimizer_ns: int
    handoff_ns: int
    evaluation_count: int
    mutation_count: int
    compiled_segment_count: int
    compiled_forward_launch_count: int
    compiled_backward_launch_count: int
    provider_callback_count: int = 0
    fallback_count: int = 0
    compile_inside_exact_call_count: int = 0
    performance_claimed: bool = False
    schema_version: str = BAB_FOUR_SEGMENT_EXACT_CALL_SCHEMA

    def validate(self) -> None:
        if (
            self.schema_version != BAB_FOUR_SEGMENT_EXACT_CALL_SCHEMA
            or not self.exact_call_id
            or any(
                not _is_sha256(value)
                for value in (
                    self.assets_hash,
                    self.production_plan_hash,
                    self.source_state_hash,
                    self.terminal_state_hash,
                )
            )
            or min(self.rebind_ns, self.optimizer_ns, self.handoff_ns) <= 0
            or self.evaluation_count != 10
            or self.mutation_count != 9
            or self.compiled_segment_count != 4
            or self.compiled_forward_launch_count != 76
            or self.compiled_backward_launch_count != 36
            or self.provider_callback_count
            or self.fallback_count
            or self.compile_inside_exact_call_count
            or self.performance_claimed
        ):
            raise ValueError("activation-BaB exact-call receipt differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        payload = asdict(self)
        payload["receipt_hash"] = _canonical_hash(payload)
        return payload


@dataclass(frozen=True)
class BabFourSegmentExactCallExecutionV1:
    """Existing B4-A handoff ABI plus four-segment execution evidence."""

    handoff_result: NativeTerminalOptimizerHandoffResultB4A
    receipt: BabFourSegmentExactCallReceiptV1


def execute_bab_four_segment_exact_call_handoff_v1(  # pylint: disable=too-many-statements
    *,
    module: BFTaskModule,
    live_sources: dict[str, torch.Tensor],
    exact_call_id: str,
    input_spec: InputSpec,
    linear_spec_C: torch.Tensor,
    relu_pre: Mapping[str, object],
    initial_state: NativeAlphaBetaOptimizationState,
    mutation_policy: ProductionMutationPolicyV4,
    schedule: NativeTerminalOptimizerScheduleV1,
    topology: tuple[ProductionReluTopologyV4, ...],
    stream: torch.cuda.Stream,
    prevalidated_plan: CorePlanInstanceV1,
    prepared_region: PreparedS4ExactCallRegionV1,
    optimizer: PreparedBabFourSegmentOptimizerV1,
) -> BabFourSegmentExactCallExecutionV1:
    """Replace the live exact-call bound producer while retaining host ownership."""

    schedule.validate()
    mutation_policy.validate()
    initial_state.validate()
    native_policy = mutation_policy.to_native_policy()
    if (
        prepared_region.used
        or prepared_region.exact_call_id != exact_call_id
        or optimizer.region is not prepared_region
        or prepared_region.stream is not stream
        or prevalidated_plan.initial_state is not initial_state
        or prevalidated_plan.mutation_policy_hash != mutation_policy.stable_hash()
        or schedule.evaluation_count != 10
        or schedule.update_count != 9
        or not math.isclose(native_policy.lr, 0.01)
        or not math.isclose(native_policy.effective_beta_lr, 0.05)
        or not math.isclose(mutation_policy.controls.lr_decay, 0.98)
        or tuple(sorted(relu_pre)) != tuple(sorted(initial_state.splits))
    ):
        raise ValueError("activation-BaB prepared exact-call instance differs")
    rebind_started = time.perf_counter_ns()
    typed_relu_pre = cast(Mapping[str, IntervalState], relu_pre)
    sources = _runtime_sources(
        prepared_region.plan,
        module,
        input_spec,
        linear_spec_C,
        typed_relu_pre,
        live_sources,
    )
    prepared_region.executor.rebind_prevalidated_inputs(sources)
    optimizer.rebind(live_sources)
    prepared_region.used = True
    rebind_ns = time.perf_counter_ns() - rebind_started
    optimizer_started = time.perf_counter_ns()
    run = optimizer.run(stream)
    optimizer_ns = time.perf_counter_ns() - optimizer_started
    handoff_started = time.perf_counter_ns()
    plan = prepared_region.plan
    terminal_state = _dense_terminal_state(plan, initial_state, run.terminal_parameters)
    interval_env, local_pre = _forward_ibp_trace_mlp(
        module, input_spec, relu_split_state=terminal_state.splits
    )
    forward_trace = NativeOptimizerForwardTraceV1(
        scope_hash=terminal_state.scope.stable_hash(),
        primal_graph_hash=plain_crown_primal_graph_hash(module),
        split_state_hash=relu_split_state_hash(terminal_state.splits),
        interval_by_value=tuple(sorted(interval_env.items())),
        local_relu_pre_by_input=tuple(sorted(local_pre.items())),
    )
    optimizer_result = NativeTerminalOptimizerResultV1(
        source_state_hash=initial_state.stable_hash(),
        mutation_policy_hash=mutation_policy.stable_hash(),
        schedule_hash=schedule.stable_hash(),
        terminal_lower=run.terminal_lower,
        terminal_state=terminal_state,
        forward_trace=forward_trace,
    )
    adjoints = {
        layout.native_preactivation: value
        for layout, value in zip(plan.relu_layouts, run.terminal_las)
    }
    topology_by_native = {item.native_preactivation: item for item in topology}
    typed_relu_dict = cast(dict[str, IntervalState], dict(relu_pre.items()))
    handoff = NativeTerminalLowerAdjointHandoffV1(
        source_state_hash=initial_state.stable_hash(),
        mutation_policy_hash=mutation_policy.stable_hash(),
        schedule_hash=schedule.stable_hash(),
        scope_hash=terminal_state.scope.stable_hash(),
        primal_graph_hash=plain_crown_primal_graph_hash(module),
        split_state_hash=relu_split_state_hash(terminal_state.splits),
        topology_hash=_topology_hash(topology),
        lower=run.terminal_lower,
        lower_adjoint_by_native_preactivation=tuple(sorted(adjoints.items())),
        lineage_by_native_preactivation=tuple(
            (
                name,
                _lineage(
                    module=module,
                    topology=topology_by_native[name],
                    preactivation=typed_relu_dict[name],
                    coefficient=adjoints[name],
                ),
            )
            for name in sorted(adjoints)
        ),
    )
    handoff_result = NativeTerminalOptimizerHandoffResultB4A(
        optimizer_result=optimizer_result,
        handoff=handoff,
    )
    handoff_result.validate(
        module=module,
        relu_pre=typed_relu_dict,
        topology=topology,
        schedule=schedule,
        mutation_policy=mutation_policy,
    )
    handoff_ns = time.perf_counter_ns() - handoff_started
    receipt = BabFourSegmentExactCallReceiptV1(
        exact_call_id=exact_call_id,
        assets_hash=_canonical_hash(
            {
                "base_assets_hash": prepared_region.assets_hash,
                "four_segment_assets_hash": optimizer.compiled_identity_hash,
            }
        ),
        production_plan_hash=plan.stable_hash(),
        source_state_hash=initial_state.stable_hash(),
        terminal_state_hash=terminal_state.stable_hash(),
        rebind_ns=rebind_ns,
        optimizer_ns=optimizer_ns,
        handoff_ns=handoff_ns,
        evaluation_count=run.evaluation_count,
        mutation_count=run.mutation_count,
        compiled_segment_count=run.compiled_segment_count,
        compiled_forward_launch_count=run.compiled_forward_launch_count,
        compiled_backward_launch_count=run.compiled_backward_launch_count,
        fallback_count=run.fallback_count,
    )
    receipt.validate()
    prepared_region.evaluator.close()
    prepared_region.buffers.close()
    return BabFourSegmentExactCallExecutionV1(handoff_result, receipt)


__all__ = [
    "BabFourSegmentExactCallExecutionV1",
    "BabFourSegmentExactCallReceiptV1",
    "execute_bab_four_segment_exact_call_handoff_v1",
]
