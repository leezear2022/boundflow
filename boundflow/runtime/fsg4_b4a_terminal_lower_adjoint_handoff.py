"""B4-A terminal lower/lA producer, one-shot handoff, and typed assembly."""

# pylint: disable=too-many-arguments,too-many-locals,too-many-statements
# pylint: disable=too-many-instance-attributes,too-many-boolean-expressions
# pylint: disable=duplicate-code,protected-access,missing-function-docstring

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping

import torch

from ..domains.interval import IntervalState
from ..frontends.plain_crown_bound_ir import (
    plain_crown_primal_graph_hash,
    relu_split_state_hash,
)
from ..ir.task import BFTaskModule, TaskOp
from .alpha_beta_crown import BetaState, _beta_to_relu_pre_add_coeff
from .crown_ibp import (
    _forward_ibp_trace_mlp,
    run_crown_ibp_mlp_from_forward_trace,
    run_crown_ibp_mlp_with_relu_lower_coefficients_from_forward_trace,
)
from .fsg4_b3_prepared_core import CorePlanInstanceV1
from .fsg4_b3_terminal_optimizer_schedule import (
    _expected_scope,
    NativeOptimizerForwardTraceV1,
    NativeTerminalOptimizerResultV1,
    NativeTerminalOptimizerScheduleV1,
)
from .native_alpha_beta_optimization_state import NativeAlphaBetaOptimizationState
from .rvir_v4_native_backward_export import NativeBackwardExportV4
from .rvir_v4_optimizer_mutation import ProductionMutationPolicyV4
from .rvir_v4_pre_state_initializer import ProductionReluTopologyV4
from .rvir_v4_production_state import production_tensor_sha256
from .task_executor import InputSpec

B4A_TERMINAL_HANDOFF_SCHEMA = "boundflow.fsg4-b4a-terminal-lower-adjoint-handoff/v1"
B4A_TERMINAL_ASSEMBLY_SCHEMA = "boundflow.fsg4-b4a-terminal-export-assembly/v1"


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _layout(value: torch.Tensor) -> str:
    if value.layout != torch.strided or not value.is_contiguous():
        raise ValueError("FSG4/B4-A handoff requires contiguous strided tensors")
    return "contiguous-strided"


def _tensor_identity(value: torch.Tensor) -> dict[str, object]:
    return {
        "shape": [int(dimension) for dimension in value.shape],
        "dtype": str(value.dtype),
        "device": str(value.device),
        "layout": _layout(value),
        "strides": [int(stride) for stride in value.stride()],
        "content_sha256": production_tensor_sha256(value),
    }


def _topology_hash(topology: tuple[ProductionReluTopologyV4, ...]) -> str:
    for item in topology:
        item.validate()
    return _canonical_hash([item.to_dict() for item in topology])


def _producer(module: BFTaskModule, value_name: str) -> tuple[int, TaskOp]:
    matches = [
        (ordinal, op)
        for ordinal, op in enumerate(module.get_entry_task().ops)
        if value_name in op.outputs
    ]
    if len(matches) != 1:
        raise ValueError("FSG4/B4-A producer operator identity differs")
    return matches[0]


@dataclass(frozen=True)
class TerminalLowerAdjointLineageV1:
    """Correlation-parent operator lineage for one native terminal lA."""

    native_preactivation: str
    provider_activation: str
    provider_preactivation: str
    producer_op_ordinal: int
    producer_op_name: str
    producer_op_type: str
    producer_output: str
    preactivation_shape: tuple[int, ...]
    coefficient_shape: tuple[int, ...]
    dtype: str
    device: str
    preactivation_layout: str
    coefficient_layout: str
    preactivation_strides: tuple[int, ...]
    coefficient_strides: tuple[int, ...]
    shape_source: str = "correlation-parent-boundflow-operator"
    kernel_shape_inferred: bool = False

    def validate(
        self,
        *,
        module: BFTaskModule,
        topology: ProductionReluTopologyV4,
        preactivation: IntervalState,
        coefficient: torch.Tensor,
    ) -> None:
        topology.validate()
        ordinal, producer = _producer(module, topology.native_preactivation)
        expected_coefficient_shape = (
            int(preactivation.lower.shape[0]),
            1,
            *(int(dimension) for dimension in preactivation.lower.shape[1:]),
        )
        if (
            self.native_preactivation != topology.native_preactivation
            or self.provider_activation != topology.provider_activation
            or self.provider_preactivation != topology.provider_preactivation
            or self.producer_op_ordinal != ordinal
            or self.producer_op_name != producer.name
            or self.producer_op_type != producer.op_type
            or self.producer_output != topology.native_preactivation
            or self.producer_output not in producer.outputs
            or self.preactivation_shape != tuple(preactivation.lower.shape)
            or self.coefficient_shape != tuple(coefficient.shape)
            or self.coefficient_shape != expected_coefficient_shape
            or self.dtype != str(preactivation.lower.dtype)
            or self.dtype != str(coefficient.dtype)
            or self.device != str(preactivation.lower.device)
            or self.device != str(coefficient.device)
            or self.preactivation_layout != _layout(preactivation.lower)
            or self.coefficient_layout != _layout(coefficient)
            or self.preactivation_strides != tuple(preactivation.lower.stride())
            or self.coefficient_strides != tuple(coefficient.stride())
            or self.shape_source != "correlation-parent-boundflow-operator"
            or self.kernel_shape_inferred is not False
            or not producer.name
        ):
            raise ValueError("FSG4/B4-A lower-adjoint operator lineage differs")

    def metadata(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "native_preactivation": self.native_preactivation,
            "provider_activation": self.provider_activation,
            "provider_preactivation": self.provider_preactivation,
            "producer_op_ordinal": self.producer_op_ordinal,
            "producer_op_name": self.producer_op_name,
            "producer_op_type": self.producer_op_type,
            "producer_output": self.producer_output,
            "preactivation_shape": list(self.preactivation_shape),
            "coefficient_shape": list(self.coefficient_shape),
            "dtype": self.dtype,
            "device": self.device,
            "preactivation_layout": self.preactivation_layout,
            "coefficient_layout": self.coefficient_layout,
            "preactivation_strides": list(self.preactivation_strides),
            "coefficient_strides": list(self.coefficient_strides),
            "coefficient_sha256": None,
            "shape_source": self.shape_source,
            "kernel_shape_inferred": self.kernel_shape_inferred,
        }
        payload["lineage_hash"] = _canonical_hash(payload)
        return payload


@dataclass(frozen=True)
class NativeTerminalLowerAdjointHandoffV1:
    """Immutable terminal lower/lA payload tied to exact production state."""

    source_state_hash: str
    mutation_policy_hash: str
    schedule_hash: str
    scope_hash: str
    primal_graph_hash: str
    split_state_hash: str
    topology_hash: str
    lower: torch.Tensor
    lower_adjoint_by_native_preactivation: tuple[tuple[str, torch.Tensor], ...]
    lineage_by_native_preactivation: tuple[
        tuple[str, TerminalLowerAdjointLineageV1], ...
    ]
    terminal_lower_adjoint_handoff_count: int = 1
    provider_core_callback_count: int = 0
    provider_compute_bounds_callback_count: int = 0
    provider_update_bounds_callback_count: int = 0
    fallback_dispatch_count: int = 0
    schema_version: str = B4A_TERMINAL_HANDOFF_SCHEMA

    @property
    def lower_adjoints(self) -> dict[str, torch.Tensor]:
        return dict(self.lower_adjoint_by_native_preactivation)

    @property
    def lineages(self) -> dict[str, TerminalLowerAdjointLineageV1]:
        return dict(self.lineage_by_native_preactivation)

    def validate(
        self,
        *,
        module: BFTaskModule,
        relu_pre: Mapping[str, IntervalState],
        terminal_state: NativeAlphaBetaOptimizationState,
        topology: tuple[ProductionReluTopologyV4, ...],
        forward_trace: NativeOptimizerForwardTraceV1,
        schedule: NativeTerminalOptimizerScheduleV1,
        mutation_policy: ProductionMutationPolicyV4,
    ) -> None:
        schedule.validate()
        mutation_policy.validate()
        lower_adjoints = self.lower_adjoints
        lineages = self.lineages
        native_names = {item.native_preactivation for item in topology}
        if (
            self.schema_version != B4A_TERMINAL_HANDOFF_SCHEMA
            or not _is_sha256(self.source_state_hash)
            or self.mutation_policy_hash != mutation_policy.stable_hash()
            or self.schedule_hash != schedule.stable_hash()
            or self.scope_hash != terminal_state.scope.stable_hash()
            or self.primal_graph_hash != plain_crown_primal_graph_hash(module)
            or self.split_state_hash != forward_trace.split_state_hash
            or self.topology_hash != _topology_hash(topology)
            or set(relu_pre) != native_names
            or set(lower_adjoints) != native_names
            or len(lower_adjoints) != len(self.lower_adjoint_by_native_preactivation)
            or set(lineages) != native_names
            or len(lineages) != len(self.lineage_by_native_preactivation)
            or tuple(self.lower.shape) != (6, 1)
            or _layout(self.lower) != "contiguous-strided"
            or self.terminal_lower_adjoint_handoff_count != 1
            or self.provider_core_callback_count != 0
            or self.provider_compute_bounds_callback_count != 0
            or self.provider_update_bounds_callback_count != 0
            or self.fallback_dispatch_count != 0
        ):
            raise ValueError("FSG4/B4-A terminal lower-adjoint handoff differs")
        topology_by_native = {item.native_preactivation: item for item in topology}
        for name in sorted(native_names):
            lineages[name].validate(
                module=module,
                topology=topology_by_native[name],
                preactivation=relu_pre[name],
                coefficient=lower_adjoints[name],
            )

    def runtime_metadata(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "source_state_hash": self.source_state_hash,
            "mutation_policy_hash": self.mutation_policy_hash,
            "schedule_hash": self.schedule_hash,
            "scope_hash": self.scope_hash,
            "primal_graph_hash": self.primal_graph_hash,
            "split_state_hash": self.split_state_hash,
            "topology_hash": self.topology_hash,
            "lower_schema": {
                "shape": list(self.lower.shape),
                "dtype": str(self.lower.dtype),
                "device": str(self.lower.device),
                "layout": _layout(self.lower),
            },
            "lower_adjoints": {
                name: {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "device": str(value.device),
                    "layout": _layout(value),
                }
                for name, value in self.lower_adjoint_by_native_preactivation
            },
            "lineages": {
                name: lineage.metadata()
                for name, lineage in self.lineage_by_native_preactivation
            },
            "terminal_lower_adjoint_handoff_count": (
                self.terminal_lower_adjoint_handoff_count
            ),
            "provider_core_callback_count": self.provider_core_callback_count,
            "provider_compute_bounds_callback_count": (
                self.provider_compute_bounds_callback_count
            ),
            "provider_update_bounds_callback_count": (
                self.provider_update_bounds_callback_count
            ),
            "fallback_dispatch_count": self.fallback_dispatch_count,
            "performance_claimed": False,
        }
        payload["runtime_handoff_hash"] = _canonical_hash(payload)
        return payload

    def metadata(
        self,
        *,
        terminal_state: NativeAlphaBetaOptimizationState,
        forward_trace: NativeOptimizerForwardTraceV1,
    ) -> dict[str, object]:
        """Build content-bound evidence after timed execution has ended."""

        payload = self.runtime_metadata()
        payload["terminal_state_hash"] = terminal_state.stable_hash()
        payload["forward_trace"] = forward_trace.metadata()
        payload["lower"] = _tensor_identity(self.lower)
        payload["lower_adjoints"] = {
            name: _tensor_identity(value)
            for name, value in self.lower_adjoint_by_native_preactivation
        }
        payload["lineages"] = {
            name: {
                **lineage.metadata(),
                "coefficient_sha256": production_tensor_sha256(
                    self.lower_adjoints[name]
                ),
            }
            for name, lineage in self.lineage_by_native_preactivation
        }
        payload["handoff_hash"] = _canonical_hash(payload)
        return payload

    def stable_hash(self) -> str:
        return str(self.runtime_metadata()["runtime_handoff_hash"])


@dataclass(frozen=True)
class NativeTerminalOptimizerHandoffResultB4A:
    """B3-compatible terminal result plus one B4-A lower-adjoint payload."""

    optimizer_result: NativeTerminalOptimizerResultV1
    handoff: NativeTerminalLowerAdjointHandoffV1
    optimizer_evaluation_count: int = 10
    optimizer_update_count: int = 9
    terminal_lower_adjoint_handoff_count: int = 1
    terminal_export_crown_rerun_count: int = 0

    def validate(
        self,
        *,
        module: BFTaskModule,
        relu_pre: Mapping[str, IntervalState],
        topology: tuple[ProductionReluTopologyV4, ...],
        schedule: NativeTerminalOptimizerScheduleV1,
        mutation_policy: ProductionMutationPolicyV4,
    ) -> None:
        self.optimizer_result.validate(module=module, schedule=schedule)
        self.handoff.validate(
            module=module,
            relu_pre=relu_pre,
            terminal_state=self.optimizer_result.terminal_state,
            topology=topology,
            forward_trace=self.optimizer_result.forward_trace,
            schedule=schedule,
            mutation_policy=mutation_policy,
        )
        if (
            self.optimizer_evaluation_count != 10
            or self.optimizer_update_count != 9
            or self.terminal_lower_adjoint_handoff_count != 1
            or self.terminal_export_crown_rerun_count != 0
            or self.optimizer_result.terminal_lower.data_ptr()
            != self.handoff.lower.data_ptr()
        ):
            raise ValueError("FSG4/B4-A terminal optimizer handoff result differs")

    def metadata(self) -> dict[str, object]:
        return {
            "optimizer_evaluation_count": self.optimizer_evaluation_count,
            "optimizer_update_count": self.optimizer_update_count,
            "terminal_lower_adjoint_handoff_count": (
                self.terminal_lower_adjoint_handoff_count
            ),
            "terminal_export_crown_rerun_count": (
                self.terminal_export_crown_rerun_count
            ),
            "runtime_handoff_hash": self.handoff.stable_hash(),
            "performance_claimed": False,
        }


class NativeTerminalLowerAdjointLeaseV1:
    """One-shot runtime lease preventing duplicate terminal handoff consumption."""

    def __init__(self, handoff: NativeTerminalLowerAdjointHandoffV1) -> None:
        # pylint: disable-next=unidiomatic-typecheck
        if type(handoff) is not NativeTerminalLowerAdjointHandoffV1:
            raise TypeError("FSG4/B4-A terminal lower-adjoint lease payload differs")
        self._handoff = handoff
        self._consumed = False

    @property
    def consumed(self) -> bool:
        return self._consumed

    def consume(
        self,
        *,
        module: BFTaskModule,
        relu_pre: Mapping[str, IntervalState],
        terminal_state: NativeAlphaBetaOptimizationState,
        topology: tuple[ProductionReluTopologyV4, ...],
        forward_trace: NativeOptimizerForwardTraceV1,
        schedule: NativeTerminalOptimizerScheduleV1,
        mutation_policy: ProductionMutationPolicyV4,
    ) -> NativeTerminalLowerAdjointHandoffV1:
        if self._consumed:
            raise ValueError(
                "FSG4/B4-A terminal lower-adjoint handoff already consumed"
            )
        self._handoff.validate(
            module=module,
            relu_pre=relu_pre,
            terminal_state=terminal_state,
            topology=topology,
            forward_trace=forward_trace,
            schedule=schedule,
            mutation_policy=mutation_policy,
        )
        self._consumed = True
        return self._handoff


@dataclass(frozen=True)
class NativeBackwardAssemblyReceiptB4A:
    """Typed no-CROWN terminal export assembly receipt."""

    export: NativeBackwardExportV4
    handoff_hash: str
    terminal_lower_adjoint_handoff_count: int = 1
    terminal_export_crown_rerun_count: int = 0
    provider_core_callback_count: int = 0
    provider_compute_bounds_callback_count: int = 0
    provider_update_bounds_callback_count: int = 0
    fallback_dispatch_count: int = 0
    schema_version: str = B4A_TERMINAL_ASSEMBLY_SCHEMA

    def validate(self, *, handoff: NativeTerminalLowerAdjointHandoffV1) -> None:
        self.export.validate()
        if (
            self.schema_version != B4A_TERMINAL_ASSEMBLY_SCHEMA
            or self.handoff_hash != handoff.stable_hash()
            or self.terminal_lower_adjoint_handoff_count != 1
            or self.terminal_export_crown_rerun_count != 0
            or self.provider_core_callback_count != 0
            or self.provider_compute_bounds_callback_count != 0
            or self.provider_update_bounds_callback_count != 0
            or self.fallback_dispatch_count != 0
        ):
            raise ValueError("FSG4/B4-A terminal export assembly receipt differs")

    def metadata(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "handoff_hash": self.handoff_hash,
            "export_schema_version": self.export.schema_version,
            "terminal_lower_adjoint_handoff_count": (
                self.terminal_lower_adjoint_handoff_count
            ),
            "terminal_export_crown_rerun_count": (
                self.terminal_export_crown_rerun_count
            ),
            "provider_core_callback_count": self.provider_core_callback_count,
            "provider_compute_bounds_callback_count": (
                self.provider_compute_bounds_callback_count
            ),
            "provider_update_bounds_callback_count": (
                self.provider_update_bounds_callback_count
            ),
            "fallback_dispatch_count": self.fallback_dispatch_count,
            "performance_claimed": False,
        }
        payload["assembly_hash"] = _canonical_hash(payload)
        return payload


def _lineage(
    *,
    module: BFTaskModule,
    topology: ProductionReluTopologyV4,
    preactivation: IntervalState,
    coefficient: torch.Tensor,
) -> TerminalLowerAdjointLineageV1:
    ordinal, producer = _producer(module, topology.native_preactivation)
    lineage = TerminalLowerAdjointLineageV1(
        native_preactivation=topology.native_preactivation,
        provider_activation=topology.provider_activation,
        provider_preactivation=topology.provider_preactivation,
        producer_op_ordinal=ordinal,
        producer_op_name=producer.name,
        producer_op_type=producer.op_type,
        producer_output=topology.native_preactivation,
        preactivation_shape=tuple(preactivation.lower.shape),
        coefficient_shape=tuple(coefficient.shape),
        dtype=str(coefficient.dtype),
        device=str(coefficient.device),
        preactivation_layout=_layout(preactivation.lower),
        coefficient_layout=_layout(coefficient),
        preactivation_strides=tuple(preactivation.lower.stride()),
        coefficient_strides=tuple(coefficient.stride()),
    )
    lineage.validate(
        module=module,
        topology=topology,
        preactivation=preactivation,
        coefficient=coefficient,
    )
    return lineage


def execute_terminal_optimizer_with_lower_adjoint_handoff_v1(
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    linear_spec_C: torch.Tensor,
    relu_pre: Mapping[str, IntervalState],
    initial_state: NativeAlphaBetaOptimizationState,
    mutation_policy: ProductionMutationPolicyV4,
    schedule: NativeTerminalOptimizerScheduleV1,
    topology: tuple[ProductionReluTopologyV4, ...],
    prevalidated_plan: CorePlanInstanceV1 | None = None,
) -> NativeTerminalOptimizerHandoffResultB4A:
    """Execute 10/9 and capture terminal lower/lA in evaluation 9 itself."""

    schedule.validate()
    mutation_policy.validate()
    initial_state.validate()
    if len(topology) != 6:
        raise ValueError("FSG4/B4-A terminal optimizer topology inventory differs")
    expected_scope = _expected_scope(
        module,
        input_spec,
        linear_spec_C=linear_spec_C,
        relu_pre=relu_pre,
        initial_state=initial_state,
        mutation_policy=mutation_policy,
        prevalidated_plan=prevalidated_plan,
    )
    native_names = {item.native_preactivation for item in topology}
    if (
        initial_state.scope != expected_scope
        or set(relu_pre) != set(initial_state.splits)
        or set(relu_pre) != native_names
        or schedule.evaluation_count != mutation_policy.evaluation_count
        or schedule.update_count != mutation_policy.update_count
    ):
        raise ValueError("FSG4/B4-A terminal optimizer admission differs")
    interval_env, local_pre = _forward_ibp_trace_mlp(
        module, input_spec, relu_split_state=initial_state.splits
    )
    alphas = {
        name: value.detach().clone().requires_grad_(True)
        for name, value in sorted(initial_state.alphas.items())
    }
    betas = {
        name: value.detach().clone().requires_grad_(True)
        for name, value in sorted(initial_state.betas.items())
    }
    native_policy = mutation_policy.to_native_policy()
    optimizer = torch.optim.Adam(
        (
            {"params": list(alphas.values()), "lr": native_policy.lr},
            {"params": list(betas.values()), "lr": native_policy.effective_beta_lr},
        )
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=mutation_policy.controls.lr_decay
    )
    terminal_lower: torch.Tensor | None = None
    terminal_lower_adjoints: dict[str, torch.Tensor] | None = None
    for action in schedule.actions:
        if not math.isclose(
            float(optimizer.param_groups[0]["lr"]),
            action.alpha_learning_rate,
            rel_tol=0.0,
            abs_tol=1e-15,
        ) or not math.isclose(
            float(optimizer.param_groups[1]["lr"]),
            action.beta_learning_rate,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("FSG4/B4-A runtime learning rate differs from schedule")
        relu_pre_add = _beta_to_relu_pre_add_coeff(
            BetaState(betas),
            relu_pre=dict(relu_pre),
            relu_split_state=initial_state.splits,
        )
        if action.update_after:
            bounds = run_crown_ibp_mlp_from_forward_trace(
                module,
                input_spec,
                interval_env=interval_env,
                relu_pre=dict(relu_pre),
                linear_spec_C=linear_spec_C,
                relu_alpha=alphas,
                relu_pre_add_coeff_l=relu_pre_add,
            )
            optimizer.zero_grad(set_to_none=True)
            (-bounds.lower.sum()).backward()
            optimizer.step()
            with torch.no_grad():
                for value in alphas.values():
                    value.clamp_(0.0, 1.0)
                for value in betas.values():
                    value.clamp_(min=0.0)
            scheduler.step()
        else:
            bounds, coefficients = (
                run_crown_ibp_mlp_with_relu_lower_coefficients_from_forward_trace(
                    module,
                    input_spec,
                    interval_env=interval_env,
                    relu_pre=dict(relu_pre),
                    linear_spec_C=linear_spec_C,
                    relu_alpha=alphas,
                    relu_pre_add_coeff_l=relu_pre_add,
                )
            )
            terminal_lower = bounds.lower.detach().contiguous().clone()
            terminal_lower_adjoints = {
                name: value.detach().contiguous().clone()
                for name, value in sorted(coefficients.items())
            }
    if terminal_lower is None or terminal_lower_adjoints is None:
        raise ValueError("FSG4/B4-A terminal optimizer produced no terminal handoff")
    terminal_state = NativeAlphaBetaOptimizationState(
        scope=initial_state.scope,
        split_by_relu_input=initial_state.split_by_relu_input,
        alpha_by_relu_input=tuple(
            (name, value.detach().contiguous().clone())
            for name, value in sorted(alphas.items())
        ),
        beta_by_relu_input=tuple(
            (name, value.detach().contiguous().clone())
            for name, value in sorted(betas.items())
        ),
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
        terminal_lower=terminal_lower,
        terminal_state=terminal_state,
        forward_trace=forward_trace,
    )
    topology_by_native = {item.native_preactivation: item for item in topology}
    handoff = NativeTerminalLowerAdjointHandoffV1(
        source_state_hash=initial_state.stable_hash(),
        mutation_policy_hash=mutation_policy.stable_hash(),
        schedule_hash=schedule.stable_hash(),
        scope_hash=terminal_state.scope.stable_hash(),
        primal_graph_hash=plain_crown_primal_graph_hash(module),
        split_state_hash=relu_split_state_hash(terminal_state.splits),
        topology_hash=_topology_hash(topology),
        lower=terminal_lower,
        lower_adjoint_by_native_preactivation=tuple(
            sorted(terminal_lower_adjoints.items())
        ),
        lineage_by_native_preactivation=tuple(
            (
                name,
                _lineage(
                    module=module,
                    topology=topology_by_native[name],
                    preactivation=relu_pre[name],
                    coefficient=terminal_lower_adjoints[name],
                ),
            )
            for name in sorted(terminal_lower_adjoints)
        ),
    )
    result = NativeTerminalOptimizerHandoffResultB4A(
        optimizer_result=optimizer_result,
        handoff=handoff,
    )
    result.validate(
        module=module,
        relu_pre=relu_pre,
        topology=topology,
        schedule=schedule,
        mutation_policy=mutation_policy,
    )
    return result


def assemble_native_backward_from_terminal_handoff_v1(
    *,
    module: BFTaskModule,
    relu_pre: Mapping[str, IntervalState],
    terminal_state: NativeAlphaBetaOptimizationState,
    topology: tuple[ProductionReluTopologyV4, ...],
    forward_trace: NativeOptimizerForwardTraceV1,
    schedule: NativeTerminalOptimizerScheduleV1,
    mutation_policy: ProductionMutationPolicyV4,
    handoff_lease: NativeTerminalLowerAdjointLeaseV1,
) -> NativeBackwardAssemblyReceiptB4A:
    """Assemble the existing native export without invoking any CROWN runner."""

    # pylint: disable-next=unidiomatic-typecheck
    if type(handoff_lease) is not NativeTerminalLowerAdjointLeaseV1:
        raise TypeError("FSG4/B4-A terminal lower-adjoint handoff lease differs")
    handoff = handoff_lease.consume(
        module=module,
        relu_pre=relu_pre,
        terminal_state=terminal_state,
        topology=topology,
        forward_trace=forward_trace,
        schedule=schedule,
        mutation_policy=mutation_policy,
    )
    native_l_as = handoff.lower_adjoints
    export = NativeBackwardExportV4(
        lower=handoff.lower.detach().contiguous().clone(),
        l_a_by_provider_activation=tuple(
            sorted(
                (
                    item.provider_activation,
                    native_l_as[item.native_preactivation]
                    .detach()
                    .contiguous()
                    .clone(),
                )
                for item in topology
            )
        ),
        intermediate_by_provider_preactivation=tuple(
            sorted(
                (
                    item.provider_preactivation,
                    IntervalState(
                        lower=relu_pre[item.native_preactivation]
                        .lower.detach()
                        .contiguous()
                        .clone(),
                        upper=relu_pre[item.native_preactivation]
                        .upper.detach()
                        .contiguous()
                        .clone(),
                    ),
                )
                for item in topology
            )
        ),
        intermediate_source="shared-pre-result-external-bounds",
    )
    receipt = NativeBackwardAssemblyReceiptB4A(
        export=export,
        handoff_hash=handoff.stable_hash(),
    )
    receipt.validate(handoff=handoff)
    return receipt


__all__ = [
    "assemble_native_backward_from_terminal_handoff_v1",
    "B4A_TERMINAL_ASSEMBLY_SCHEMA",
    "B4A_TERMINAL_HANDOFF_SCHEMA",
    "execute_terminal_optimizer_with_lower_adjoint_handoff_v1",
    "NativeBackwardAssemblyReceiptB4A",
    "NativeTerminalLowerAdjointHandoffV1",
    "NativeTerminalLowerAdjointLeaseV1",
    "NativeTerminalOptimizerHandoffResultB4A",
    "TerminalLowerAdjointLineageV1",
]
