"""Query-local parametric optimizer compiler, cache, binder, and executor."""

# pylint: disable=too-many-arguments,too-many-locals,too-many-statements
# pylint: disable=too-many-instance-attributes,too-many-boolean-expressions
# pylint: disable=protected-access,missing-function-docstring,duplicate-code
# pylint: disable=invalid-name,too-many-branches

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Mapping, Optional

import torch

from ..domains.interval import IntervalState
from ..frontends.plain_crown_bound_ir import (
    plain_crown_primal_graph_hash,
    tensor_content_hash,
)
from ..ir.bound import IntermediateBoundSource
from ..ir.optimizer import OptimizerTaskKind
from ..ir.parametric_optimizer import (
    NativeParametricOptimizerCacheEventIR,
    NativeParametricOptimizerInstanceIR,
    NativeParametricOptimizerScheduleIR,
    NativeParametricOptimizerTaskIRModule,
    NativeParametricOptimizerTemplateIR,
    lower_native_parametric_optimizer_template_ir,
)
from ..ir.task import BFTaskModule
from .native_alpha_beta_optimization_state import (
    NativeAlphaBetaOptimizationState,
    NativeAlphaBetaOptimizerPolicy,
    NativeWarmStartDecision,
    build_native_alpha_beta_scope,
    classify_native_alpha_beta_warm_start,
)
from .native_alpha_beta_optimizer_schedule import (
    NativeProductionOptimizerResult,
    _evaluate_state,
    _initial_alpha_state,
    _metric_by_domain,
    _optimizer_intermediate_semantics,
    _select_improved_state_slices,
)
from .relu_shape_utils import relu_input_shapes
from .task_executor import InputSpec


def _tensor_contract(tensor: torch.Tensor) -> tuple[tuple[int, ...], str, str]:
    return (
        tuple(int(dimension) for dimension in tensor.shape),
        str(tensor.dtype),
        str(tensor.device),
    )


def _input_contract(input_spec: InputSpec) -> tuple[tuple[int, ...], str, str]:
    shape, dtype, device = _tensor_contract(input_spec.center)
    if len(shape) < 2:
        raise ValueError("parametric optimizer input must have a batch dimension")
    return shape[1:], dtype, device


def _relu_layout(
    relu_pre: Mapping[str, IntervalState],
) -> tuple[tuple[str, tuple[int, ...], str, str], ...]:
    layout: list[tuple[str, tuple[int, ...], str, str]] = []
    for name, value in sorted(relu_pre.items()):
        if (
            value.lower.shape != value.upper.shape
            or value.lower.dtype != value.upper.dtype
            or value.lower.device != value.upper.device
            or value.lower.dim() < 2
        ):
            raise ValueError("parametric optimizer ReLU contract differs")
        layout.append(
            (
                name,
                tuple(int(dimension) for dimension in value.lower.shape[1:]),
                str(value.lower.dtype),
                str(value.lower.device),
            )
        )
    if not layout:
        raise ValueError("parametric optimizer requires at least one ReLU")
    return tuple(layout)


@dataclass(frozen=True)
class NativeParametricOptimizerTemplate:
    """Fully validated static compiler product retained by a query-local cache."""

    ir: NativeParametricOptimizerTemplateIR
    task_ir: NativeParametricOptimizerTaskIRModule
    schedule: NativeParametricOptimizerScheduleIR
    module: BFTaskModule
    policy: NativeAlphaBetaOptimizerPolicy
    template_hash: str
    task_hash: str
    schedule_hash: str

    def validate(self) -> None:
        self.schedule.validate(template=self.ir, task_module=self.task_ir)
        self.module.validate()
        self.policy.validate()
        if (
            self.ir.primal_graph_hash != plain_crown_primal_graph_hash(self.module)
            or self.ir.optimizer_policy_hash != self.policy.stable_hash()
            or self.ir.steps != self.policy.steps
            or self.ir.objective != self.policy.objective
            or self.ir.spec_reduce != self.policy.spec_reduce
            or self.template_hash != self.ir.stable_hash()
            or self.task_hash != self.task_ir.stable_hash(template=self.ir)
            or self.schedule_hash
            != self.schedule.stable_hash(template=self.ir, task_module=self.task_ir)
        ):
            raise ValueError("parametric optimizer template runtime differs")

    def require_contract(
        self,
        module: BFTaskModule,
        input_spec: InputSpec,
        *,
        linear_spec_C: torch.Tensor,
        relu_pre: Mapping[str, IntervalState],
        policy: NativeAlphaBetaOptimizerPolicy,
        intermediate_bound_source: IntermediateBoundSource,
        refine_external_constraints: bool,
    ) -> None:
        input_shape, input_dtype, input_device = _input_contract(input_spec)
        objective_shape, objective_dtype, objective_device = _tensor_contract(
            linear_spec_C
        )
        if (
            module is not self.module
            or policy != self.policy
            or input_spec.value_name != self.ir.input_value_name
            or input_shape != self.ir.input_nonbatch_shape
            or input_dtype != self.ir.input_dtype
            or input_device != self.ir.input_device
            or objective_shape != self.ir.objective_shape
            or objective_dtype != self.ir.objective_dtype
            or objective_device != self.ir.objective_device
            or _relu_layout(relu_pre) != self.ir.relu_state_layout
            or intermediate_bound_source.value != self.ir.intermediate_bound_source
            or refine_external_constraints != self.ir.refine_external_constraints
        ):
            raise ValueError("parametric optimizer template contract differs")

    def hashes(self) -> dict[str, str]:
        return {
            "optimizer_plan_hash": self.template_hash,
            "optimizer_task_module_hash": self.task_hash,
            "optimizer_schedule_hash": self.schedule_hash,
        }


@dataclass(frozen=True)
class NativeParametricOptimizerInstance:
    """Runtime tensors and exact dynamic identity bound to one template."""

    ir: NativeParametricOptimizerInstanceIR
    template: NativeParametricOptimizerTemplate
    input_spec: InputSpec
    linear_spec_C: torch.Tensor
    initial_state: NativeAlphaBetaOptimizationState
    interval_env: Mapping[str, IntervalState]
    relu_pre: Mapping[str, IntervalState]
    warm_start_decision: Optional[NativeWarmStartDecision]
    intermediate_bound_source: IntermediateBoundSource

    @property
    def policy(self) -> NativeAlphaBetaOptimizerPolicy:
        return self.template.policy

    def validate(self) -> None:
        self.ir.validate()
        self.initial_state.validate()
        warm_kind = (
            "none"
            if self.warm_start_decision is None
            else self.warm_start_decision.kind
        )
        if self.warm_start_decision is not None:
            self.warm_start_decision.validate()
        if (
            self.ir.template_hash != self.template.template_hash
            or self.ir.cache_key != self.template.ir.cache_key()
            or self.ir.batch_size != int(self.input_spec.center.shape[0])
            or self.ir.input_region_hash != self.initial_state.scope.input_region_hash
            or self.ir.objective_hash != tensor_content_hash(self.linear_spec_C)
            or self.ir.intermediate_bounds_hash
            != self.initial_state.scope.intermediate_bounds_hash
            or self.ir.split_state_hash != self.initial_state.scope.split_state_hash
            or self.ir.state_scope_hash != self.initial_state.scope.stable_hash()
            or self.ir.initial_state_hash != self.initial_state.stable_hash()
            or self.ir.warm_start_kind != warm_kind
        ):
            raise ValueError("parametric optimizer instance runtime differs")

    def require_exact_runtime(
        self,
        module: BFTaskModule,
        input_spec: InputSpec,
        *,
        linear_spec_C: torch.Tensor,
        intermediate_bound_source: IntermediateBoundSource,
    ) -> None:
        if (
            module is not self.template.module
            or input_spec is not self.input_spec
            or linear_spec_C is not self.linear_spec_C
            or intermediate_bound_source != self.intermediate_bound_source
        ):
            raise ValueError("parametric optimizer exact runtime binding differs")


def compile_native_parametric_optimizer_template(
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    linear_spec_C: torch.Tensor,
    relu_pre: Mapping[str, IntervalState],
    policy: NativeAlphaBetaOptimizerPolicy,
    intermediate_bound_source: IntermediateBoundSource,
    refine_external_constraints: bool,
    template_id: str,
) -> NativeParametricOptimizerTemplate:
    """Compile the static graph/tensor/policy contract exactly once."""

    if not template_id:
        raise ValueError("parametric optimizer template ID must be non-empty")
    module.validate()
    policy.validate()
    if not isinstance(intermediate_bound_source, IntermediateBoundSource):
        raise TypeError("parametric optimizer intermediate-bound source is invalid")
    if not isinstance(refine_external_constraints, bool):
        raise TypeError("parametric optimizer external refinement flag is invalid")
    input_shape, input_dtype, input_device = _input_contract(input_spec)
    objective_shape, objective_dtype, objective_device = _tensor_contract(linear_spec_C)
    if not torch.is_floating_point(linear_spec_C) or not bool(
        torch.isfinite(linear_spec_C).all()
    ):
        raise ValueError("parametric optimizer objective contract is invalid")
    ir = NativeParametricOptimizerTemplateIR(
        template_id=template_id,
        primal_graph_hash=plain_crown_primal_graph_hash(module),
        input_value_name=input_spec.value_name,
        input_nonbatch_shape=input_shape,
        input_dtype=input_dtype,
        input_device=input_device,
        objective_shape=objective_shape,
        objective_dtype=objective_dtype,
        objective_device=objective_device,
        relu_state_layout=_relu_layout(relu_pre),
        optimizer_policy_hash=policy.stable_hash(),
        steps=policy.steps,
        objective=policy.objective,
        spec_reduce=policy.spec_reduce,
        intermediate_bound_source=intermediate_bound_source.value,
        refine_external_constraints=refine_external_constraints,
    )
    task_ir, schedule = lower_native_parametric_optimizer_template_ir(ir)
    template = NativeParametricOptimizerTemplate(
        ir=ir,
        task_ir=task_ir,
        schedule=schedule,
        module=module,
        policy=policy,
        template_hash=ir.stable_hash(),
        task_hash=task_ir.stable_hash(template=ir),
        schedule_hash=schedule.stable_hash(template=ir, task_module=task_ir),
    )
    template.validate()
    return template


@dataclass
class NativeParametricOptimizerTemplateCache:
    """Single-owner query-local exact-contract template cache."""

    _templates: dict[str, NativeParametricOptimizerTemplate] = field(
        default_factory=dict
    )
    _events: list[NativeParametricOptimizerCacheEventIR] = field(default_factory=list)

    @property
    def events(self) -> tuple[NativeParametricOptimizerCacheEventIR, ...]:
        return tuple(self._events)

    @property
    def templates(self) -> tuple[NativeParametricOptimizerTemplate, ...]:
        return tuple(self._templates[key] for key in sorted(self._templates))

    def acquire(
        self,
        module: BFTaskModule,
        input_spec: InputSpec,
        *,
        linear_spec_C: torch.Tensor,
        relu_pre: Mapping[str, IntervalState],
        policy: NativeAlphaBetaOptimizerPolicy,
        intermediate_bound_source: IntermediateBoundSource,
        refine_external_constraints: bool,
        template_id: str,
        batch_id: str,
    ) -> tuple[
        NativeParametricOptimizerTemplate,
        NativeParametricOptimizerCacheEventIR,
    ]:
        for cache_key, template in self._templates.items():
            if template.module is not module:
                continue
            template.require_contract(
                module,
                input_spec,
                linear_spec_C=linear_spec_C,
                relu_pre=relu_pre,
                policy=policy,
                intermediate_bound_source=intermediate_bound_source,
                refine_external_constraints=refine_external_constraints,
            )
            event = NativeParametricOptimizerCacheEventIR(
                event_index=len(self._events),
                batch_id=batch_id,
                cache_key=cache_key,
                template_hash=template.template_hash,
                outcome="hit_exact_contract",
                compile_elapsed_ns=0,
            )
            event.validate()
            self._events.append(event)
            return template, event

        started_ns = time.perf_counter_ns()
        template = compile_native_parametric_optimizer_template(
            module,
            input_spec,
            linear_spec_C=linear_spec_C,
            relu_pre=relu_pre,
            policy=policy,
            intermediate_bound_source=intermediate_bound_source,
            refine_external_constraints=refine_external_constraints,
            template_id=template_id,
        )
        elapsed_ns = time.perf_counter_ns() - started_ns
        cache_key = template.ir.cache_key()
        existing = self._templates.get(cache_key)
        if existing is not None:
            raise ValueError("parametric optimizer cache key collision")
        self._templates[cache_key] = template
        event = NativeParametricOptimizerCacheEventIR(
            event_index=len(self._events),
            batch_id=batch_id,
            cache_key=cache_key,
            template_hash=template.template_hash,
            outcome="miss_compiled",
            compile_elapsed_ns=elapsed_ns,
        )
        event.validate()
        self._events.append(event)
        return template, event

    def validate(self) -> None:
        for cache_key, template in self._templates.items():
            if cache_key != template.ir.cache_key():
                raise ValueError("parametric optimizer cache key differs")
        if tuple(event.event_index for event in self._events) != tuple(
            range(len(self._events))
        ):
            raise ValueError("parametric optimizer cache event order differs")
        for event in self._events:
            event.validate()
            target = self._templates.get(event.cache_key)
            if target is None or event.template_hash != target.template_hash:
                raise ValueError("parametric optimizer cache event target differs")
        misses = tuple(
            event for event in self._events if event.outcome == "miss_compiled"
        )
        if len(misses) != len(self._templates):
            raise ValueError("parametric optimizer cache miss accounting differs")


@dataclass(frozen=True)
class NativeParametricOptimizerTemplateTrace:
    """Serializable static compiler product emitted once per cache entry."""

    ir: NativeParametricOptimizerTemplateIR
    task_ir: NativeParametricOptimizerTaskIRModule
    schedule: NativeParametricOptimizerScheduleIR
    template_hash: str
    task_hash: str
    schedule_hash: str

    @classmethod
    def from_template(
        cls, template: NativeParametricOptimizerTemplate
    ) -> "NativeParametricOptimizerTemplateTrace":
        return cls(
            ir=template.ir,
            task_ir=template.task_ir,
            schedule=template.schedule,
            template_hash=template.template_hash,
            task_hash=template.task_hash,
            schedule_hash=template.schedule_hash,
        )

    def validate(self) -> None:
        self.schedule.validate(template=self.ir, task_module=self.task_ir)
        if (
            self.template_hash != self.ir.stable_hash()
            or self.task_hash != self.task_ir.stable_hash(template=self.ir)
            or self.schedule_hash
            != self.schedule.stable_hash(template=self.ir, task_module=self.task_ir)
        ):
            raise ValueError("parametric optimizer template trace differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "template": self.ir.to_dict(),
            "template_hash": self.template_hash,
            "task_ir": self.task_ir.to_dict(template=self.ir),
            "task_hash": self.task_hash,
            "schedule": self.schedule.to_dict(
                template=self.ir, task_module=self.task_ir
            ),
            "schedule_hash": self.schedule_hash,
        }


@dataclass(frozen=True)
class NativeParametricOptimizerCacheTrace:
    """Immutable query-level template definitions and cache decisions."""

    templates: tuple[NativeParametricOptimizerTemplateTrace, ...]
    events: tuple[NativeParametricOptimizerCacheEventIR, ...]

    @classmethod
    def from_cache(
        cls, cache: NativeParametricOptimizerTemplateCache
    ) -> "NativeParametricOptimizerCacheTrace":
        cache.validate()
        trace = cls(
            templates=tuple(
                NativeParametricOptimizerTemplateTrace.from_template(template)
                for template in cache.templates
            ),
            events=cache.events,
        )
        trace.validate()
        return trace

    def validate(self) -> None:
        if not self.templates or not self.events:
            raise ValueError("parametric optimizer cache trace is empty")
        by_key: dict[str, NativeParametricOptimizerTemplateTrace] = {}
        for template in self.templates:
            template.validate()
            cache_key = template.ir.cache_key()
            if cache_key in by_key:
                raise ValueError("parametric optimizer cache trace key repeats")
            by_key[cache_key] = template
        if tuple(event.event_index for event in self.events) != tuple(
            range(len(self.events))
        ):
            raise ValueError("parametric optimizer cache trace order differs")
        for event in self.events:
            event.validate()
            target = by_key.get(event.cache_key)
            if target is None or target.template_hash != event.template_hash:
                raise ValueError("parametric optimizer cache trace target differs")
        if sum(event.outcome == "miss_compiled" for event in self.events) != len(
            self.templates
        ):
            raise ValueError("parametric optimizer cache trace misses differ")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "templates": [template.to_dict() for template in self.templates],
            "events": [event.to_dict() for event in self.events],
            "template_count": len(self.templates),
            "instance_count": len(self.events),
            "miss_count": sum(
                event.outcome == "miss_compiled" for event in self.events
            ),
            "hit_count": sum(
                event.outcome == "hit_exact_contract" for event in self.events
            ),
            "compile_elapsed_ns": sum(
                event.compile_elapsed_ns for event in self.events
            ),
        }


def instantiate_native_parametric_optimizer(
    template: NativeParametricOptimizerTemplate,
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    linear_spec_C: torch.Tensor,
    relu_split_state: Mapping[str, torch.Tensor],
    instance_id: str,
    warm_start: Optional[NativeAlphaBetaOptimizationState] = None,
    relu_pre_override: Optional[Mapping[str, IntervalState]] = None,
    intermediate_bound_source: IntermediateBoundSource = (
        IntermediateBoundSource.LOCAL_FORWARD
    ),
    refine_external_constraints: bool = False,
) -> NativeParametricOptimizerInstance:
    """Bind exact dynamic tensors without rebuilding replay-grade source IR."""

    if not instance_id:
        raise ValueError("parametric optimizer instance ID must be non-empty")
    interval_env, relu_pre = _optimizer_intermediate_semantics(
        module,
        input_spec,
        relu_split_state=relu_split_state,
        relu_pre_override=relu_pre_override,
        intermediate_bound_source=intermediate_bound_source,
        refine_external_constraints=refine_external_constraints,
    )
    template.require_contract(
        module,
        input_spec,
        linear_spec_C=linear_spec_C,
        relu_pre=relu_pre,
        policy=template.policy,
        intermediate_bound_source=intermediate_bound_source,
        refine_external_constraints=refine_external_constraints,
    )
    scope = build_native_alpha_beta_scope(
        module,
        input_spec,
        linear_spec_C=linear_spec_C,
        relu_pre=relu_pre,
        relu_split_state=relu_split_state,
        policy=template.policy,
    )
    warm_decision: Optional[NativeWarmStartDecision] = None
    if warm_start is None:
        alpha = _initial_alpha_state(relu_pre, policy=template.policy)
        batch = int(input_spec.center.shape[0])
        beta = {
            name: torch.full(
                (batch, *shape),
                template.policy.beta_init,
                device=input_spec.center.device,
                dtype=input_spec.center.dtype,
            )
            for name, shape in relu_input_shapes(dict(relu_pre)).items()
        }
    else:
        warm_decision = classify_native_alpha_beta_warm_start(
            warm_start,
            target_scope=scope,
            target_split_state=relu_split_state,
        )
        if warm_decision.kind == "rejected":
            raise ValueError(
                f"parametric optimizer warm start rejected: {warm_decision.reason}"
            )
        alpha = warm_start.alphas
        beta = warm_start.betas
    initial_state = NativeAlphaBetaOptimizationState(
        scope=scope,
        split_by_relu_input=tuple(
            (name, value.detach().contiguous().clone())
            for name, value in sorted(relu_split_state.items())
        ),
        alpha_by_relu_input=tuple(
            (name, value.detach().contiguous().clone())
            for name, value in sorted(alpha.items())
        ),
        beta_by_relu_input=tuple(
            (name, value.detach().contiguous().clone())
            for name, value in sorted(beta.items())
        ),
    )
    initial_state.validate()
    warm_kind = "none" if warm_decision is None else warm_decision.kind
    instance_ir = NativeParametricOptimizerInstanceIR(
        instance_id=instance_id,
        template_hash=template.template_hash,
        cache_key=template.ir.cache_key(),
        batch_size=int(input_spec.center.shape[0]),
        input_region_hash=scope.input_region_hash,
        objective_hash=tensor_content_hash(linear_spec_C),
        intermediate_bounds_hash=scope.intermediate_bounds_hash,
        split_state_hash=scope.split_state_hash,
        state_scope_hash=scope.stable_hash(),
        initial_state_hash=initial_state.stable_hash(),
        warm_start_kind=warm_kind,
    )
    instance = NativeParametricOptimizerInstance(
        ir=instance_ir,
        template=template,
        input_spec=input_spec,
        linear_spec_C=linear_spec_C,
        initial_state=initial_state,
        interval_env=interval_env,
        relu_pre=relu_pre,
        warm_start_decision=warm_decision,
        intermediate_bound_source=intermediate_bound_source,
    )
    instance.validate()
    return instance


def execute_native_parametric_optimizer(
    instance: NativeParametricOptimizerInstance,
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    linear_spec_C: torch.Tensor,
    intermediate_bound_source: IntermediateBoundSource,
) -> NativeProductionOptimizerResult:
    """Execute one exact instance through the reusable optimizer Schedule."""

    instance.require_exact_runtime(
        module,
        input_spec,
        linear_spec_C=linear_spec_C,
        intermediate_bound_source=intermediate_bound_source,
    )
    alpha = {
        name: value.detach().clone().requires_grad_(True)
        for name, value in instance.initial_state.alphas.items()
    }
    beta = {
        name: value.detach().clone().requires_grad_(True)
        for name, value in instance.initial_state.betas.items()
    }
    optimizer = torch.optim.Adam(
        [*alpha.values(), *beta.values()], lr=instance.policy.lr
    )
    bounds_by_iteration: dict[int, IntervalState] = {}
    metric_by_iteration: dict[int, torch.Tensor] = {}
    best_metric: Optional[torch.Tensor] = None
    best_bounds: Optional[IntervalState] = None
    best_alpha: Optional[dict[str, torch.Tensor]] = None
    best_beta: Optional[dict[str, torch.Tensor]] = None
    best_iteration: Optional[torch.Tensor] = None

    for action, task in zip(
        instance.template.schedule.actions, instance.template.task_ir.tasks
    ):
        if action.task_id != task.task_id:
            raise ValueError("parametric optimizer Schedule/Task identity differs")
        iteration = task.iteration
        if task.kind == OptimizerTaskKind.EVALUATE_BOUND:
            assert iteration is not None
            bounds_by_iteration[iteration] = _evaluate_state(
                module,
                input_spec,
                linear_spec_C,
                alpha,
                beta,
                interval_env=instance.interval_env,
                relu_pre=instance.relu_pre,
                relu_split_state=instance.initial_state.splits,
                objective=instance.policy.objective,
            )
        elif task.kind == OptimizerTaskKind.REDUCE_METRIC:
            assert iteration is not None
            bounds = bounds_by_iteration[iteration]
            metric = _metric_by_domain(
                bounds,
                objective=instance.policy.objective,
                spec_reduce=instance.policy.spec_reduce,
                soft_tau=instance.policy.soft_tau,
            )
            metric_by_iteration[iteration] = metric
            detached = metric.detach()
            if best_metric is None:
                improve = torch.ones_like(detached, dtype=torch.bool)
                best_metric = detached.clone()
                best_bounds = IntervalState(
                    lower=bounds.lower.detach().clone(),
                    upper=bounds.upper.detach().clone(),
                )
                best_alpha = {
                    name: value.detach().clone() for name, value in alpha.items()
                }
                best_beta = {
                    name: value.detach().clone() for name, value in beta.items()
                }
                best_iteration = torch.full(
                    detached.shape,
                    iteration,
                    dtype=torch.int64,
                    device=detached.device,
                )
            else:
                improve = detached > best_metric
                if bool(improve.any().item()):
                    best_metric = torch.where(improve, detached, best_metric)
                    assert best_bounds is not None
                    best_bounds = IntervalState(
                        lower=torch.where(
                            improve.unsqueeze(1),
                            bounds.lower.detach(),
                            best_bounds.lower,
                        ),
                        upper=torch.where(
                            improve.unsqueeze(1),
                            bounds.upper.detach(),
                            best_bounds.upper,
                        ),
                    )
                    assert best_alpha is not None and best_beta is not None
                    best_alpha = _select_improved_state_slices(
                        alpha, best_alpha, improve
                    )
                    best_beta = _select_improved_state_slices(beta, best_beta, improve)
                    assert best_iteration is not None
                    best_iteration = torch.where(
                        improve,
                        torch.full_like(best_iteration, iteration),
                        best_iteration,
                    )
        elif task.kind == OptimizerTaskKind.BACKWARD:
            assert iteration is not None
            optimizer.zero_grad(set_to_none=True)
            (-metric_by_iteration[iteration].sum()).backward()
        elif task.kind == OptimizerTaskKind.ADAM_UPDATE:
            optimizer.step()
        elif task.kind == OptimizerTaskKind.PROJECT_STATE:
            with torch.no_grad():
                for value in alpha.values():
                    value.clamp_(0.0, 1.0)
                for value in beta.values():
                    value.clamp_(0.0)
        elif task.kind == OptimizerTaskKind.SELECT_BEST:
            if best_bounds is None or best_alpha is None or best_beta is None:
                raise ValueError("parametric optimizer selected before evaluation")
        else:
            raise AssertionError("unreachable parametric optimizer task kind")

    assert best_bounds is not None
    assert best_alpha is not None and best_beta is not None
    assert best_iteration is not None
    selected_state = NativeAlphaBetaOptimizationState(
        scope=instance.initial_state.scope,
        split_by_relu_input=tuple(
            (name, value.detach().contiguous().clone())
            for name, value in sorted(instance.initial_state.splits.items())
        ),
        alpha_by_relu_input=tuple(
            (name, value.detach().contiguous().clone())
            for name, value in sorted(best_alpha.items())
        ),
        beta_by_relu_input=tuple(
            (name, value.detach().contiguous().clone())
            for name, value in sorted(best_beta.items())
        ),
    )
    selected_state.validate()
    result = NativeProductionOptimizerResult(
        bounds=best_bounds,
        state=selected_state,
        best_iteration_by_domain=tuple(
            int(value) for value in best_iteration.detach().cpu()
        ),
    )
    if (
        result.bounds.lower.shape != result.bounds.upper.shape
        or not bool(torch.isfinite(result.bounds.lower).all())
        or not bool(torch.isfinite(result.bounds.upper).all())
        or int(result.bounds.lower.shape[0]) != len(result.best_iteration_by_domain)
    ):
        raise ValueError("parametric optimizer produced invalid bounds")
    return result


__all__ = [
    "NativeParametricOptimizerCacheTrace",
    "NativeParametricOptimizerInstance",
    "NativeParametricOptimizerTemplate",
    "NativeParametricOptimizerTemplateCache",
    "NativeParametricOptimizerTemplateTrace",
    "compile_native_parametric_optimizer_template",
    "execute_native_parametric_optimizer",
    "instantiate_native_parametric_optimizer",
]
