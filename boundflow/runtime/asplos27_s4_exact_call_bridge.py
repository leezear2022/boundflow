"""RVIR exact-call bridge for the compiled S4 all-state optimizer.

This module replaces only the optimizer/lower-adjoint producer inside the
existing RVIR live-return path.  The host solver, backward export, KFSB,
atomic commit, postprocessing, and queue ownership remain unchanged.
"""

# mypy: disable-error-code=import-untyped
# pylint: disable=import-error,import-outside-toplevel,protected-access
# pylint: disable=too-many-arguments,too-many-locals,too-many-statements
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-instance-attributes,duplicate-code
# pylint: disable=missing-function-docstring,too-many-boolean-expressions
# pylint: disable=non-parent-init-called,super-init-not-called

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import time
from typing import cast, Mapping

import torch

from boundflow.domains.interval import IntervalState
from boundflow.backends.tvm.asplos27_s4_compact_coefficient import (
    CompiledS4CompactCoefficientV1,
    compile_s4_compact_coefficient_v1,
)
from boundflow.backends.tvm.asplos27_s4_compressed_gradient import (
    CompiledS4CompressedGradientV1,
    compile_s4_compressed_gradient_v1,
)
from boundflow.backends.tvm.asplos27_s4_six_site_value import (
    CompiledS4SelectorPackV1,
    CompiledS4SixSiteValueV1,
    compile_s4_selector_pack_v1,
    compile_s4_six_site_value_v1,
)
from boundflow.backends.tvm.r3_d1c_wrapper_schedule import (
    CompiledR3D1CWrapperScheduleV1,
    compile_r3d1c_wrapper_schedule_v1,
)
from boundflow.frontends.plain_crown_bound_ir import (
    plain_crown_primal_graph_hash,
    relu_split_state_hash,
)
from boundflow.ir.task import BFTaskModule
from boundflow.ir.r3_bounded_arena import R31BBoundedArenaTraceV1
from boundflow.runtime.asplos27_s4_all_state_evaluator import (
    PreparedS4AllStateEvaluatorV1,
)
from boundflow.runtime.asplos27_s4_mutable_state_admission import (
    prepare_s4_mutable_state_admission_v1,
)
from boundflow.runtime.asplos27_s4_optimizer_driver import (
    execute_s4_optimizer_v1,
)
from boundflow.runtime.asplos27_s4_ordered_buffer_abi import (
    prepare_s4_mutable_buffers_v1,
    PreparedS4MutableBuffersV1,
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
from boundflow.runtime.r3_bounded_arena_trace_compiler import (
    compile_r31b_bounded_arena_trace_v1,
)
from boundflow.runtime.r3_compiled_p_alpha_vjp import R31B2_MODULE_CACHE
from boundflow.runtime.r3_d2b_staged_backward import (
    PreparedR3D2BStagedBackwardCandidateV1,
)
from boundflow.runtime.r3_full_lower_forward_tir import R31B1_MODULE_CACHE
from boundflow.runtime.r3_optimizer_trajectory_timing import (
    PreparedR32BTimingCandidateV1,
)
from boundflow.runtime.r3_structured_owner_custom_backward import (
    bind_r31_runtime_inputs_v1,
    compile_r31_full_region_plan_v1,
    R31FullRegionPlanV1,
)
from boundflow.runtime.rvir_v4_optimizer_mutation import ProductionMutationPolicyV4
from boundflow.runtime.rvir_v4_pre_state_initializer import (
    ProductionNativePreStateV4,
    ProductionReluTopologyV4,
)
from boundflow.runtime.rvir_v4_production_state import ProductionStateSnapshotV4
from boundflow.runtime.task_executor import InputSpec

S4_EXACT_CALL_SCHEMA = "boundflow.asplos27-s4-exact-call/v1"
S4_SITE_ORDER = ("17", "19", "23", "25", "28", "31")


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
class S4ExactCallCompiledAssetsV1:
    """Query-independent modules compiled before host timing begins."""

    compute_capability: str
    d1c: CompiledR3D1CWrapperScheduleV1
    compact: CompiledS4CompactCoefficientV1
    selector: CompiledS4SelectorPackV1
    value: CompiledS4SixSiteValueV1
    gradient: CompiledS4CompressedGradientV1

    def validate(self) -> None:
        if (
            not _is_sha256(self.d1c.scheduled_tir_hash)
            or not _is_sha256(self.d1c.device_source_hash)
            or not self.d1c.exported_symbols
            or self.d1c.threads_per_block <= 0
            or not self.d1c.reduction_kind
            or self.d1c.vector_width <= 0
            or not self.d1c.tvm_version
        ):
            raise ValueError("S4 exact-call D1C compiled identity differs")
        self.compact.validate()
        self.selector.validate()
        self.value.validate()
        self.gradient.validate()
        if (
            not self.compute_capability.startswith("sm_")
            or self.d1c.global_workspace_bytes
            or self.compact.global_workspace_bytes
            or self.gradient.global_workspace_bytes
        ):
            raise ValueError("S4 exact-call compiled assets differ")

    def identity(self) -> dict[str, object]:
        self.validate()
        return {
            "compute_capability": self.compute_capability,
            "d1c_schedule_hash": self.d1c.scheduled_tir_hash,
            "compact_schedule_hash": self.compact.scheduled_tir_hash,
            "selector_schedule_hash": self.selector.scheduled_tir_hash,
            "value_lowered_relax_hash": self.value.lowered_relax_ir_hash,
            "gradient_schedule_hash": self.gradient.scheduled_tir_hash,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.identity())


def compile_s4_exact_call_assets_v1(
    *, device: torch.device | str
) -> S4ExactCallCompiledAssetsV1:
    """Compile all static S4 modules and warm the inherited B1/B2 caches."""

    target = torch.device(device)
    if target.type != "cuda":
        raise ValueError("S4 exact-call compile requires CUDA")
    ordinal = target.index if target.index is not None else torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(ordinal)
    capability = f"sm_{major}{minor}"
    # These two modules are inherited by the prepared R3 coefficient owner.
    R31B1_MODULE_CACHE.get(capability)
    R31B2_MODULE_CACHE.get(capability)
    result = S4ExactCallCompiledAssetsV1(
        compute_capability=capability,
        d1c=compile_r3d1c_wrapper_schedule_v1(),
        compact=compile_s4_compact_coefficient_v1(compute_capability=capability),
        selector=compile_s4_selector_pack_v1(device_index=ordinal),
        value=compile_s4_six_site_value_v1(device_index=ordinal),
        gradient=compile_s4_compressed_gradient_v1(compute_capability=capability),
    )
    result.validate()
    return result


class _PreparedS4R3ExecutorV1(PreparedR3D2BStagedBackwardCandidateV1):
    """D2B owner whose D1C module was compiled outside the exact call."""

    def __init__(
        self,
        plan: R31FullRegionPlanV1,
        trace: object,
        tensors: tuple[torch.Tensor, ...],
        compiled: CompiledR3D1CWrapperScheduleV1,
    ) -> None:
        import tvm

        PreparedR32BTimingCandidateV1.__init__(self, plan, trace, tensors)
        self.d1c_compiled = compiled
        self._residual11_scratch = self.forward_executor.scratch_1[6144:12288]
        self._residual6_scratch = self.forward_executor.scratch_0[12288:18432]
        self._register_view(tvm, self._residual11_scratch)
        self._register_view(tvm, self._residual6_scratch)
        self.d1c_launch_count = 0
        self.d1c_bias_inplace_alias_count = 0
        self.d2b_backward_staged_launch_count = 0
        self.d2b_backward_bias_inplace_alias_count = 0

    def rebind_prevalidated_inputs(
        self, sources: tuple[torch.Tensor, ...]
    ) -> tuple[int, int]:
        """Copy one validated dynamic instance into persistent device storage."""

        if len(sources) != len(self.tensors):
            raise ValueError("S4 exact-call persistent input count differs")
        copied_bytes = 0
        with torch.no_grad():
            for source, target, spec in zip(
                sources, self.tensors, self.plan.tensor_specs
            ):
                if (
                    tuple(source.shape) != spec.shape
                    or str(source.dtype) != spec.dtype
                    or source.device != target.device
                    or not source.is_contiguous()
                ):
                    raise ValueError(
                        f"S4 exact-call persistent input differs: {spec.name}"
                    )
                if source.data_ptr() != target.data_ptr():
                    target.copy_(source)
                    copied_bytes += source.numel() * source.element_size()
        self._tensor_versions = tuple(value._version for value in self.tensors)
        return len(sources), copied_bytes


@dataclass
class PreparedS4ExactCallRegionV1:
    """One-shot persistent executor whose static work precedes query timing."""

    exact_call_id: str
    plan: R31FullRegionPlanV1
    trace: R31BBoundedArenaTraceV1
    executor: _PreparedS4R3ExecutorV1
    buffers: PreparedS4MutableBuffersV1
    evaluator: PreparedS4AllStateEvaluatorV1
    assets: S4ExactCallCompiledAssetsV1
    assets_hash: str
    stream: torch.cuda.Stream
    used: bool = False

    def validate(self) -> None:
        if (
            not self.exact_call_id
            or not _is_sha256(self.assets_hash)
            or self.assets_hash != self.assets.stable_hash()
            or self.executor.plan is not self.plan
            or self.evaluator.executor is not self.executor
            or self.evaluator.buffers is not self.buffers
            or self.evaluator.stream is not self.stream
            or self.used
        ):
            raise ValueError("S4 prepared exact-call region differs")


def _runtime_sources(
    plan: R31FullRegionPlanV1,
    module: BFTaskModule,
    input_spec: InputSpec,
    linear_spec_c: torch.Tensor,
    relu_pre: Mapping[str, IntervalState],
    live_sources: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    lower, upper = input_spec.perturbation.bounding_box(input_spec.center)
    params = module.bindings.get("params")
    if not isinstance(params, Mapping):
        raise ValueError("S4 exact-call persistent parameter binding differs")
    values: list[torch.Tensor] = [lower, upper, linear_spec_c]
    values.extend(params[name] for name in plan.parameter_names)
    for layout in plan.relu_layouts:
        interval = relu_pre.get(layout.native_preactivation)
        alpha = live_sources.get(layout.alpha_path)
        beta = live_sources.get(layout.beta_path)
        if interval is None or alpha is None or beta is None:
            raise ValueError("S4 exact-call persistent state coverage differs")
        values.extend((interval.lower, interval.upper, alpha, beta))
    return tuple(value.contiguous() for value in values)


def _rebind_compact_parameters(
    region: PreparedS4ExactCallRegionV1,
    live_sources: Mapping[str, torch.Tensor],
) -> tuple[int, int]:
    resources = region.buffers._resources
    if resources is None or len(resources._parameters) != 7:
        raise ValueError("S4 exact-call persistent parameter owner differs")
    copied = 0
    active_beta = 0
    with torch.no_grad():
        for ordinal, layout in enumerate(region.plan.relu_layouts):
            source = live_sources[layout.alpha_path][0, 0].contiguous()
            target = resources._parameters[ordinal]
            if source.shape != target.shape or source.device != target.device:
                raise ValueError("S4 exact-call compact alpha binding differs")
            target.copy_(source)
            copied += source.numel() * source.element_size()
            if any(layout.beta_locations):
                beta = live_sources[layout.beta_path].contiguous()
                beta_target = resources._parameters[6]
                if beta.shape != beta_target.shape or beta.device != beta_target.device:
                    raise ValueError("S4 exact-call compact beta binding differs")
                beta_target.copy_(beta)
                copied += beta.numel() * beta.element_size()
                active_beta += 1
    if active_beta != 1:
        raise ValueError("S4 exact-call compact beta owner differs")
    return 7, copied


def prepare_s4_exact_call_region_v1(
    *,
    program: object,
    module: BFTaskModule,
    snapshot: ProductionStateSnapshotV4,
    mapping: ProductionNativePreStateV4,
    live_sources: dict[str, torch.Tensor],
    exact_call_id: str,
    topology: tuple[ProductionReluTopologyV4, ...],
    stream: torch.cuda.Stream,
    assets: S4ExactCallCompiledAssetsV1,
) -> PreparedS4ExactCallRegionV1:
    """Prepare a one-shot S4 executor and all persistent views before query."""

    assets.validate()
    plan = compile_r31_full_region_plan_v1(module, snapshot, mapping, topology)
    trace = compile_r31b_bounded_arena_trace_v1(program, module, plan)
    tensors = bind_r31_runtime_inputs_v1(
        plan, module, snapshot, device=stream.device, dtype=torch.float32
    )
    executor = _PreparedS4R3ExecutorV1(plan, trace, tensors, assets.d1c)
    admission = prepare_s4_mutable_state_admission_v1(
        snapshot, topology, plan, live_sources, exact_call_id=exact_call_id
    )
    buffers = prepare_s4_mutable_buffers_v1(
        admission, live_sources, exact_call_id=exact_call_id
    )
    evaluator = PreparedS4AllStateEvaluatorV1(
        executor,
        buffers,
        exact_call_id=exact_call_id,
        stream=stream,
        compiled_compact=assets.compact,
        compiled_selector=assets.selector,
        compiled_value=assets.value,
        compiled_gradient=assets.gradient,
    )
    region = PreparedS4ExactCallRegionV1(
        exact_call_id=exact_call_id,
        plan=plan,
        trace=trace,
        executor=executor,
        buffers=buffers,
        evaluator=evaluator,
        assets=assets,
        assets_hash=assets.stable_hash(),
        stream=stream,
    )
    region.validate()
    return region


def _dense_terminal_state(
    plan: R31FullRegionPlanV1,
    initial: NativeAlphaBetaOptimizationState,
    parameters: tuple[torch.Tensor, ...],
) -> NativeAlphaBetaOptimizationState:
    if (
        len(parameters) != 7
        or tuple(layout.native_preactivation for layout in plan.relu_layouts)
        != S4_SITE_ORDER
    ):
        raise ValueError("S4 exact-call compact parameter order differs")
    initial_alphas = initial.alphas
    initial_betas = initial.betas
    alphas: dict[str, torch.Tensor] = {}
    betas: dict[str, torch.Tensor] = {}
    active_beta_count = 0
    for ordinal, layout in enumerate(plan.relu_layouts):
        name = layout.native_preactivation
        reference_alpha = initial_alphas[name]
        reference_beta = initial_betas[name]
        compact_alpha = parameters[ordinal]
        # S4 owns only the compressed production slice.  Values outside that
        # slice remain host-solver state and must survive the exact call.
        flat_alpha = reference_alpha.clone().reshape(plan.domain_count, -1)
        alpha_index = (
            torch.tensor(
                layout.alpha_flat_indices,
                device=compact_alpha.device,
                dtype=torch.int64,
            )
            .reshape(1, -1)
            .expand(plan.domain_count, -1)
        )
        if tuple(compact_alpha.shape) != (
            plan.domain_count,
            len(layout.alpha_flat_indices),
        ):
            raise ValueError("S4 exact-call compact alpha shape differs")
        alphas[name] = flat_alpha.scatter(1, alpha_index, compact_alpha).reshape_as(
            reference_alpha
        )
        flat_beta = reference_beta.clone().reshape(plan.domain_count, -1)
        if any(layout.beta_locations):
            beta_locations = torch.tensor(
                layout.beta_locations,
                device=parameters[6].device,
                dtype=torch.int64,
            )
            if tuple(parameters[6].shape) != tuple(beta_locations.shape):
                raise ValueError("S4 exact-call compact beta shape differs")
            flat_beta = flat_beta.scatter(1, beta_locations, parameters[6])
            active_beta_count += 1
        betas[name] = flat_beta.reshape_as(reference_beta)
    if active_beta_count != 1:
        raise ValueError("S4 exact-call active beta owner differs")
    terminal = NativeAlphaBetaOptimizationState(
        scope=initial.scope,
        split_by_relu_input=initial.split_by_relu_input,
        alpha_by_relu_input=tuple(sorted(alphas.items())),
        beta_by_relu_input=tuple(sorted(betas.items())),
    )
    terminal.validate()
    return terminal


@dataclass(frozen=True)
class S4ExactCallExecutionReceiptV1:
    """Exact-call ownership and local phase attribution for one live run."""

    exact_call_id: str
    assets_hash: str
    production_plan_hash: str
    trace_hash: str
    admission_hash: str
    buffer_receipt_hash: str
    source_state_hash: str
    terminal_state_hash: str
    setup_ns: int
    ir_prepare_ns: int
    executor_prepare_ns: int
    state_prepare_ns: int
    evaluator_prepare_ns: int
    optimizer_ns: int
    handoff_ns: int
    evaluation_count: int
    mutation_count: int
    value_graph_submission_count: int
    compact_coefficient_launch_count: int
    provider_callback_count: int = 0
    fallback_count: int = 0
    compile_inside_exact_call_count: int = 0
    performance_claimed: bool = False
    schema_version: str = S4_EXACT_CALL_SCHEMA

    def validate(self) -> None:
        hashes = (
            self.assets_hash,
            self.production_plan_hash,
            self.trace_hash,
            self.admission_hash,
            self.buffer_receipt_hash,
            self.source_state_hash,
            self.terminal_state_hash,
        )
        if (
            self.schema_version != S4_EXACT_CALL_SCHEMA
            or not self.exact_call_id
            or any(not _is_sha256(value) for value in hashes)
            or min(
                self.setup_ns,
                self.ir_prepare_ns,
                self.executor_prepare_ns,
                self.state_prepare_ns,
                self.evaluator_prepare_ns,
                self.optimizer_ns,
                self.handoff_ns,
            )
            <= 0
            or self.evaluation_count != 10
            or self.mutation_count != 9
            or self.value_graph_submission_count != 10
            or self.compact_coefficient_launch_count != 180
            or self.provider_callback_count
            or self.fallback_count
            or self.compile_inside_exact_call_count
            or self.performance_claimed
        ):
            raise ValueError("S4 exact-call execution receipt differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        payload = asdict(self)
        payload["receipt_hash"] = _canonical_hash(payload)
        return payload


@dataclass(frozen=True)
class S4ExactCallExecutionV1:
    """B4-A-compatible handoff plus S4 exact-call evidence."""

    handoff_result: NativeTerminalOptimizerHandoffResultB4A
    receipt: S4ExactCallExecutionReceiptV1


def execute_s4_exact_call_handoff_v1(  # pylint: disable=too-many-arguments
    *,
    program: object,
    module: BFTaskModule,
    snapshot: ProductionStateSnapshotV4,
    mapping: ProductionNativePreStateV4,
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
    assets: S4ExactCallCompiledAssetsV1,
    prevalidated_plan: CorePlanInstanceV1 | None = None,
    prepared_region: PreparedS4ExactCallRegionV1 | None = None,
) -> S4ExactCallExecutionV1:
    """Execute S4 at the existing RVIR optimizer/handoff seam."""

    if prepared_region is None:
        assets.validate()
    elif (
        prepared_region.used
        or prepared_region.exact_call_id != exact_call_id
        or prepared_region.assets is not assets
        or prepared_region.stream is not stream
    ):
        raise ValueError("S4 prepared exact-call activation differs")
    schedule.validate()
    mutation_policy.validate()
    initial_state.validate()
    if prepared_region is None:
        snapshot.validate()
        mapping.validate()
        if prevalidated_plan is not None and (
            prevalidated_plan.snapshot_hash != snapshot.stable_hash()
            or prevalidated_plan.mapping_hash != mapping.stable_hash()
            or prevalidated_plan.mutation_policy_hash != mutation_policy.stable_hash()
            or prevalidated_plan.initial_state.stable_hash()
            != initial_state.stable_hash()
        ):
            raise ValueError("S4 exact-call prevalidated plan differs")
    elif (
        prevalidated_plan is None
        or prevalidated_plan.initial_state is not initial_state
        or prevalidated_plan.mutation_policy_hash != mutation_policy.stable_hash()
        or mapping.identity.topology_hash != _topology_hash(topology)
    ):
        raise ValueError("S4 prepared exact-call instance differs")
    native_policy = mutation_policy.to_native_policy()
    if (
        schedule.evaluation_count != 10
        or schedule.update_count != 9
        or not math.isclose(native_policy.lr, 0.01)
        or not math.isclose(native_policy.effective_beta_lr, 0.05)
        or not math.isclose(mutation_policy.controls.lr_decay, 0.98)
        or tuple(sorted(relu_pre)) != tuple(sorted(initial_state.splits))
    ):
        raise ValueError("S4 exact-call optimizer policy differs")
    setup_started = time.perf_counter_ns()
    if prepared_region is None:
        ir_started = time.perf_counter_ns()
        plan = compile_r31_full_region_plan_v1(module, snapshot, mapping, topology)
        trace = compile_r31b_bounded_arena_trace_v1(program, module, plan)
        ir_prepare_ns = time.perf_counter_ns() - ir_started
        executor_started = time.perf_counter_ns()
        tensors = bind_r31_runtime_inputs_v1(
            plan,
            module,
            snapshot,
            device=linear_spec_C.device,
            dtype=linear_spec_C.dtype,
        )
        executor = _PreparedS4R3ExecutorV1(plan, trace, tensors, assets.d1c)
        executor_prepare_ns = time.perf_counter_ns() - executor_started
        state_started = time.perf_counter_ns()
        admission = prepare_s4_mutable_state_admission_v1(
            snapshot, topology, plan, live_sources, exact_call_id=exact_call_id
        )
        buffers = prepare_s4_mutable_buffers_v1(
            admission, live_sources, exact_call_id=exact_call_id
        )
        state_prepare_ns = time.perf_counter_ns() - state_started
        evaluator_started = time.perf_counter_ns()
        evaluator = PreparedS4AllStateEvaluatorV1(
            executor,
            buffers,
            exact_call_id=exact_call_id,
            stream=stream,
            compiled_compact=assets.compact,
            compiled_selector=assets.selector,
            compiled_value=assets.value,
            compiled_gradient=assets.gradient,
        )
        evaluator_prepare_ns = time.perf_counter_ns() - evaluator_started
        admission_hash = admission.receipt.admission_hash
        buffer_receipt_hash = buffers.receipt.stable_hash()
        assets_hash = assets.stable_hash()
    else:
        assert prevalidated_plan is not None
        ir_started = time.perf_counter_ns()
        plan = prepared_region.plan
        trace = prepared_region.trace
        if (
            plain_crown_primal_graph_hash(module) != plan.primal_graph_hash
            or tuple(layout.native_preactivation for layout in plan.relu_layouts)
            != S4_SITE_ORDER
        ):
            raise ValueError("S4 prepared exact-call region structure differs")
        ir_prepare_ns = time.perf_counter_ns() - ir_started
        executor_started = time.perf_counter_ns()
        typed_relu_pre = cast(Mapping[str, IntervalState], relu_pre)
        sources = _runtime_sources(
            plan,
            module,
            input_spec,
            linear_spec_C,
            typed_relu_pre,
            live_sources,
        )
        prepared_region.executor.rebind_prevalidated_inputs(sources)
        executor = prepared_region.executor
        executor_prepare_ns = time.perf_counter_ns() - executor_started
        state_started = time.perf_counter_ns()
        _rebind_compact_parameters(prepared_region, live_sources)
        state_prepare_ns = time.perf_counter_ns() - state_started
        evaluator_started = time.perf_counter_ns()
        buffers = prepared_region.buffers
        evaluator = prepared_region.evaluator
        if evaluator._next_ordinal or evaluator._closed or evaluator._terminal_complete:
            raise ValueError("S4 prepared exact-call evaluator is not fresh")
        evaluator_prepare_ns = time.perf_counter_ns() - evaluator_started
        admission_hash = _canonical_hash(
            {
                "exact_call_id": exact_call_id,
                "core_instance_hash": prevalidated_plan.instance_hash,
                "snapshot_hash": prevalidated_plan.snapshot_hash,
                "mapping_hash": prevalidated_plan.mapping_hash,
                "region_template_hash": plan.stable_hash(),
            }
        )
        buffer_receipt_hash = buffers.receipt.stable_hash()
        assets_hash = prepared_region.assets_hash
        prepared_region.used = True
    setup_ns = time.perf_counter_ns() - setup_started
    optimizer_started = time.perf_counter_ns()
    run = execute_s4_optimizer_v1(evaluator)
    optimizer_ns = time.perf_counter_ns() - optimizer_started
    handoff_started = time.perf_counter_ns()
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
    terminal_lower = run.terminal_lower.reshape(6, 1).contiguous()
    optimizer_result = NativeTerminalOptimizerResultV1(
        source_state_hash=initial_state.stable_hash(),
        mutation_policy_hash=mutation_policy.stable_hash(),
        schedule_hash=schedule.stable_hash(),
        terminal_lower=terminal_lower,
        terminal_state=terminal_state,
        forward_trace=forward_trace,
    )
    adjoints = {
        layout.native_preactivation: value.contiguous()
        for layout, value in zip(plan.relu_layouts, run.terminal_las)
    }
    topology_by_native = {item.native_preactivation: item for item in topology}
    typed_relu_pre = cast(dict[str, IntervalState], dict(relu_pre.items()))
    handoff = NativeTerminalLowerAdjointHandoffV1(
        source_state_hash=initial_state.stable_hash(),
        mutation_policy_hash=mutation_policy.stable_hash(),
        schedule_hash=schedule.stable_hash(),
        scope_hash=terminal_state.scope.stable_hash(),
        primal_graph_hash=plain_crown_primal_graph_hash(module),
        split_state_hash=relu_split_state_hash(terminal_state.splits),
        topology_hash=_topology_hash(topology),
        lower=terminal_lower,
        lower_adjoint_by_native_preactivation=tuple(sorted(adjoints.items())),
        lineage_by_native_preactivation=tuple(
            (
                name,
                _lineage(
                    module=module,
                    topology=topology_by_native[name],
                    preactivation=typed_relu_pre[name],  # type: ignore[arg-type]
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
        relu_pre=typed_relu_pre,  # type: ignore[arg-type]
        topology=topology,
        schedule=schedule,
        mutation_policy=mutation_policy,
    )
    handoff_ns = time.perf_counter_ns() - handoff_started
    receipt = S4ExactCallExecutionReceiptV1(
        exact_call_id=exact_call_id,
        assets_hash=assets_hash,
        production_plan_hash=plan.stable_hash(),
        trace_hash=trace.stable_hash(),
        admission_hash=admission_hash,
        buffer_receipt_hash=buffer_receipt_hash,
        source_state_hash=initial_state.stable_hash(),
        terminal_state_hash=terminal_state.stable_hash(),
        setup_ns=setup_ns,
        ir_prepare_ns=ir_prepare_ns,
        executor_prepare_ns=executor_prepare_ns,
        state_prepare_ns=state_prepare_ns,
        evaluator_prepare_ns=evaluator_prepare_ns,
        optimizer_ns=optimizer_ns,
        handoff_ns=handoff_ns,
        evaluation_count=run.evaluation_count,
        mutation_count=run.optimizer_mutation_count,
        value_graph_submission_count=run.value_graph_submission_count,
        compact_coefficient_launch_count=run.compact_coefficient_launch_count,
    )
    receipt.validate()
    evaluator.close()
    buffers.close()
    return S4ExactCallExecutionV1(handoff_result=handoff_result, receipt=receipt)


__all__ = [
    "compile_s4_exact_call_assets_v1",
    "execute_s4_exact_call_handoff_v1",
    "prepare_s4_exact_call_region_v1",
    "PreparedS4ExactCallRegionV1",
    "S4ExactCallCompiledAssetsV1",
    "S4ExactCallExecutionReceiptV1",
    "S4ExactCallExecutionV1",
]
