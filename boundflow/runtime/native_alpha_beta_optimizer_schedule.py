"""Schedule-driven native alpha/beta optimizer execution and trace."""

# pylint: disable=too-many-arguments,too-many-locals,too-many-statements
# pylint: disable=too-many-instance-attributes,too-many-branches
# pylint: disable=missing-function-docstring,protected-access,invalid-name
# pylint: disable=too-many-boolean-expressions

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping, Optional

import torch

from ..domains.interval import IntervalState
from ..frontends.plain_crown_bound_ir import tensor_content_hash
from ..ir.optimizer import (
    NativeOptimizerPlanIR,
    NativeOptimizerScheduleIR,
    NativeOptimizerTaskIRModule,
    OptimizerTaskKind,
    lower_native_optimizer_ir,
)
from ..ir.task import BFTaskModule
from .alpha_beta_crown import BetaState, _beta_to_relu_pre_add_coeff
from .alpha_crown import AlphaObjective, SpecReduce
from .crown_ibp import (
    _forward_ibp_trace_mlp,
    run_crown_ibp_mlp_from_forward_trace,
)
from .native_alpha_beta_optimization_state import (
    NativeAlphaBetaOptimizationResult,
    NativeAlphaBetaOptimizationState,
    NativeAlphaBetaOptimizerPolicy,
    NativeWarmStartDecision,
    build_native_alpha_beta_scope,
    classify_native_alpha_beta_warm_start,
    compile_native_alpha_beta_state_query,
)
from .native_verifier_ir_integration import (
    NativePlainCrownRepresentationCompilation,
)
from .relu_shape_utils import relu_input_shapes
from .task_executor import InputSpec

NATIVE_OPTIMIZER_EXECUTION_TRACE_SCHEMA_VERSION = (
    "boundflow.native-alpha-beta-optimizer-execution/v1"
)


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class NativeOptimizerProgram:
    """Cross-linked optimizer IR and its exact initial dynamic state."""

    plan: NativeOptimizerPlanIR
    task_module: NativeOptimizerTaskIRModule
    schedule: NativeOptimizerScheduleIR
    source_compilation: NativePlainCrownRepresentationCompilation
    initial_state: NativeAlphaBetaOptimizationState
    interval_env: Mapping[str, IntervalState]
    relu_pre: Mapping[str, IntervalState]
    policy: NativeAlphaBetaOptimizerPolicy
    warm_start_decision: Optional[NativeWarmStartDecision]

    def validate(self) -> None:
        self.schedule.validate(plan=self.plan, task_module=self.task_module)
        self.source_compilation.validate()
        self.initial_state.validate()
        self.policy.validate()
        if (
            self.plan.source_ir_hashes
            != tuple(sorted(self.source_compilation.hashes().items()))
            or self.plan.initial_state_hash != self.initial_state.stable_hash()
            or self.plan.state_scope_hash != self.initial_state.scope.stable_hash()
            or self.plan.optimizer_policy_hash != self.policy.stable_hash()
            or self.plan.steps != self.policy.steps
            or self.plan.relu_state_keys != tuple(sorted(self.initial_state.splits))
            or self.plan.warm_start_kind
            != (
                "none"
                if self.warm_start_decision is None
                else self.warm_start_decision.kind
            )
        ):
            raise ValueError("native optimizer program cross-layer identity differs")
        if self.warm_start_decision is not None:
            self.warm_start_decision.validate()

    def hashes(self) -> dict[str, str]:
        self.validate()
        return {
            "optimizer_plan_hash": self.plan.stable_hash(),
            "optimizer_task_module_hash": self.task_module.stable_hash(plan=self.plan),
            "optimizer_schedule_hash": self.schedule.stable_hash(
                plan=self.plan, task_module=self.task_module
            ),
        }


@dataclass(frozen=True)
class NativeOptimizerActionTrace:
    """One executed optimizer Schedule action and its runtime evidence."""

    sequence: int
    action_id: str
    task_id: str
    kind: OptimizerTaskKind
    iteration: Optional[int]
    input_hashes: tuple[tuple[str, str], ...]
    output_hashes: tuple[tuple[str, str], ...]
    beta_gradient_l1: Optional[float] = None
    alpha_gradient_l1: Optional[float] = None
    projection_applied: bool = False

    def validate(self) -> None:
        if (
            self.sequence < 0
            or not self.action_id
            or not self.task_id
            or not self.input_hashes
            or not self.output_hashes
            or len(dict(self.input_hashes)) != len(self.input_hashes)
            or len(dict(self.output_hashes)) != len(self.output_hashes)
            or any(
                not _is_sha256(value)
                for _name, value in (*self.input_hashes, *self.output_hashes)
            )
        ):
            raise ValueError("native optimizer action trace is invalid")
        if self.kind == OptimizerTaskKind.BACKWARD:
            if (
                self.beta_gradient_l1 is None
                or self.alpha_gradient_l1 is None
                or self.beta_gradient_l1 < 0.0
                or self.alpha_gradient_l1 < 0.0
            ):
                raise ValueError("optimizer backward trace lacks gradient evidence")
        elif self.beta_gradient_l1 is not None or self.alpha_gradient_l1 is not None:
            raise ValueError("non-backward optimizer action declares gradients")
        if self.projection_applied != (self.kind == OptimizerTaskKind.PROJECT_STATE):
            raise ValueError("optimizer projection trace differs from task kind")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "sequence": self.sequence,
            "action_id": self.action_id,
            "task_id": self.task_id,
            "kind": self.kind.value,
            "iteration": self.iteration,
            "input_hashes": dict(self.input_hashes),
            "output_hashes": dict(self.output_hashes),
            "beta_gradient_l1": self.beta_gradient_l1,
            "alpha_gradient_l1": self.alpha_gradient_l1,
            "projection_applied": self.projection_applied,
        }


@dataclass(frozen=True)
class NativeOptimizerEvaluationTrace:
    """One evaluated state, bounds, and per-domain objective metric."""

    iteration: int
    state_hash: str
    lower_hash: str
    upper_hash: str
    metric_hash: str
    metric_values: tuple[float, ...]

    def validate(self) -> None:
        if (
            self.iteration < 0
            or any(
                not _is_sha256(value)
                for value in (
                    self.state_hash,
                    self.lower_hash,
                    self.upper_hash,
                    self.metric_hash,
                )
            )
            or not self.metric_values
            or any(
                not torch.isfinite(torch.tensor(value)).item()
                for value in self.metric_values
            )
        ):
            raise ValueError("native optimizer evaluation trace is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "iteration": self.iteration,
            "state_hash": self.state_hash,
            "lower_hash": self.lower_hash,
            "upper_hash": self.upper_hash,
            "metric_hash": self.metric_hash,
            "metric_values": list(self.metric_values),
        }


@dataclass(frozen=True)
class NativeOptimizerExecutionTrace:
    """Replay-grade proof that optimizer Schedule actions drove execution."""

    plan_hash: str
    task_module_hash: str
    schedule_hash: str
    initial_state_hash: str
    selected_state_hash: str
    actions: tuple[NativeOptimizerActionTrace, ...]
    evaluations: tuple[NativeOptimizerEvaluationTrace, ...]
    best_iteration_by_domain: tuple[int, ...]
    performance_claimed: bool = False
    schema_version: str = NATIVE_OPTIMIZER_EXECUTION_TRACE_SCHEMA_VERSION

    def validate(self, *, program: NativeOptimizerProgram) -> None:
        program.validate()
        hashes = program.hashes()
        if (
            self.schema_version != NATIVE_OPTIMIZER_EXECUTION_TRACE_SCHEMA_VERSION
            or self.plan_hash != hashes["optimizer_plan_hash"]
            or self.task_module_hash != hashes["optimizer_task_module_hash"]
            or self.schedule_hash != hashes["optimizer_schedule_hash"]
            or self.initial_state_hash != program.initial_state.stable_hash()
            or not _is_sha256(self.selected_state_hash)
            or len(self.actions) != len(program.schedule.actions)
            or len(self.evaluations) != program.plan.steps + 1
            or not self.best_iteration_by_domain
            or self.performance_claimed is not False
        ):
            raise ValueError("native optimizer execution trace header differs")
        runtime_hashes = {"optimizer.state.s000": program.initial_state.stable_hash()}
        evaluation_by_iteration = {
            evaluation.iteration: evaluation for evaluation in self.evaluations
        }
        for sequence, (trace, action, task) in enumerate(
            zip(self.actions, program.schedule.actions, program.task_module.tasks)
        ):
            trace.validate()
            if (
                trace.sequence != sequence
                or trace.action_id != action.action_id
                or trace.task_id != task.task_id
                or trace.kind != task.kind
                or trace.iteration != task.iteration
                or tuple(name for name, _hash in trace.input_hashes)
                != task.input_value_ids
                or tuple(name for name, _hash in trace.output_hashes)
                != task.output_value_ids
            ):
                raise ValueError("optimizer execution trace action linkage differs")
            if trace.input_hashes != tuple(
                (value_id, runtime_hashes[value_id])
                for value_id in task.input_value_ids
            ):
                raise ValueError("optimizer execution trace input hash chain differs")
            for value_id, value_hash in trace.output_hashes:
                if value_id in runtime_hashes:
                    raise ValueError("optimizer execution trace redefines a value")
                runtime_hashes[value_id] = value_hash
            if task.kind == OptimizerTaskKind.EVALUATE_BOUND:
                assert task.iteration is not None
                evaluation = evaluation_by_iteration.get(task.iteration)
                if evaluation is None or trace.output_hashes[0][1] != _canonical_hash(
                    {
                        "lower": evaluation.lower_hash,
                        "upper": evaluation.upper_hash,
                    }
                ):
                    raise ValueError("optimizer evaluation bound hash differs")
                if trace.input_hashes[0][1] != evaluation.state_hash:
                    raise ValueError("optimizer evaluation state hash differs")
            elif task.kind == OptimizerTaskKind.REDUCE_METRIC:
                assert task.iteration is not None
                evaluation = evaluation_by_iteration.get(task.iteration)
                if evaluation is None or trace.output_hashes[0][1] != (
                    evaluation.metric_hash
                ):
                    raise ValueError("optimizer reduction metric hash differs")
            elif task.kind == OptimizerTaskKind.SELECT_BEST:
                if trace.output_hashes[0][1] != self.selected_state_hash:
                    raise ValueError("optimizer select-best state hash differs")
        for iteration, evaluation in enumerate(self.evaluations):
            evaluation.validate()
            if evaluation.iteration != iteration:
                raise ValueError("optimizer evaluation trace order differs")
        domain_count = len(self.evaluations[0].metric_values)
        if len(self.best_iteration_by_domain) != domain_count or any(
            len(evaluation.metric_values) != domain_count
            for evaluation in self.evaluations
        ):
            raise ValueError("optimizer execution trace domain count differs")
        if any(
            iteration < 0 or iteration >= len(self.evaluations)
            for iteration in self.best_iteration_by_domain
        ):
            raise ValueError("optimizer best iteration is not evaluated")
        selected_hashes = {
            self.evaluations[iteration].state_hash
            for iteration in self.best_iteration_by_domain
        }
        if (
            len(selected_hashes) == 1
            and self.selected_state_hash not in selected_hashes
        ):
            raise ValueError("optimizer selected state is not an evaluated candidate")

    def to_dict(self, *, program: NativeOptimizerProgram) -> dict[str, object]:
        self.validate(program=program)
        return {
            "schema_version": self.schema_version,
            "plan_hash": self.plan_hash,
            "task_module_hash": self.task_module_hash,
            "schedule_hash": self.schedule_hash,
            "initial_state_hash": self.initial_state_hash,
            "selected_state_hash": self.selected_state_hash,
            "actions": [action.to_dict() for action in self.actions],
            "evaluations": [item.to_dict() for item in self.evaluations],
            "best_iteration_by_domain": list(self.best_iteration_by_domain),
            "performance_claimed": self.performance_claimed,
        }

    def stable_hash(self, *, program: NativeOptimizerProgram) -> str:
        return _canonical_hash(self.to_dict(program=program))


@dataclass(frozen=True)
class NativeScheduledOptimizerResult:
    """Selected bounds/state and exact optimizer control trace."""

    bounds: IntervalState
    state: NativeAlphaBetaOptimizationState
    trace: NativeOptimizerExecutionTrace

    def validate(self, *, program: NativeOptimizerProgram) -> None:
        self.state.validate()
        self.trace.validate(program=program)
        if self.state.stable_hash() != self.trace.selected_state_hash:
            raise ValueError("scheduled optimizer result/trace state differs")
        if self.bounds.lower.shape != self.bounds.upper.shape:
            raise ValueError("scheduled optimizer result bounds shape differs")


def _metric_by_domain(
    bounds: IntervalState,
    *,
    objective: AlphaObjective,
    spec_reduce: SpecReduce,
    soft_tau: float,
) -> torch.Tensor:
    batch_size = int(bounds.lower.shape[0])

    def reduce(x: torch.Tensor, *, direction: str) -> torch.Tensor:
        if x.dim() != 2:
            return x.mean().expand(batch_size)
        if spec_reduce == "mean":
            return x.mean(dim=1)
        if spec_reduce == "min":
            return x.min(dim=1).values if direction == "min" else x.max(dim=1).values
        if spec_reduce == "softmin":
            if direction == "min":
                return -soft_tau * torch.logsumexp(-x / soft_tau, dim=1)
            return soft_tau * torch.logsumexp(x / soft_tau, dim=1)
        raise AssertionError("unreachable spec reduction")

    if objective == "lower":
        return reduce(bounds.lower, direction="min")
    if objective == "upper":
        return -reduce(bounds.upper, direction="max")
    if objective == "gap":
        return -reduce(bounds.upper - bounds.lower, direction="max")
    if objective == "both":
        return reduce(bounds.lower, direction="min") - reduce(
            bounds.upper, direction="max"
        )
    raise AssertionError("unreachable optimizer objective")


def _state_from_tensors(
    program: NativeOptimizerProgram,
    alpha: Mapping[str, torch.Tensor],
    beta: Mapping[str, torch.Tensor],
) -> NativeAlphaBetaOptimizationState:
    state = NativeAlphaBetaOptimizationState(
        scope=program.initial_state.scope,
        split_by_relu_input=tuple(
            (name, value.detach().contiguous().clone())
            for name, value in sorted(program.initial_state.splits.items())
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
    state.validate()
    return state


def _select_improved_state_slices(
    current: Mapping[str, torch.Tensor],
    previous: Mapping[str, torch.Tensor],
    improve: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {
        name: torch.where(
            improve.view(int(improve.shape[0]), *([1] * (value.dim() - 1))),
            value.detach(),
            previous[name],
        )
        for name, value in current.items()
    }


def _evaluate_state(
    module: BFTaskModule,
    input_spec: InputSpec,
    linear_spec_C: torch.Tensor,
    alpha: Mapping[str, torch.Tensor],
    beta: Mapping[str, torch.Tensor],
    *,
    interval_env: Mapping[str, IntervalState],
    relu_pre: Mapping[str, IntervalState],
    relu_split_state: Mapping[str, torch.Tensor],
    objective: AlphaObjective,
) -> IntervalState:
    relu_pre_add = _beta_to_relu_pre_add_coeff(
        BetaState(beta_by_relu_input=dict(beta)),
        relu_pre=dict(relu_pre),
        relu_split_state=dict(relu_split_state),
    )
    return run_crown_ibp_mlp_from_forward_trace(
        module,
        input_spec,
        interval_env=dict(interval_env),
        relu_pre=dict(relu_pre),
        linear_spec_C=linear_spec_C,
        relu_alpha=dict(alpha),
        relu_pre_add_coeff_l=(relu_pre_add if objective == "lower" else None),
    )


def compile_native_alpha_beta_optimizer_program(
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    linear_spec_C: torch.Tensor,
    relu_split_state: Mapping[str, torch.Tensor],
    policy: NativeAlphaBetaOptimizerPolicy,
    program_id: str,
    warm_start: Optional[NativeAlphaBetaOptimizationState] = None,
) -> NativeOptimizerProgram:
    """Compile fixed-step optimizer control around one NRIR-10 source stack."""

    if not program_id:
        raise ValueError("native optimizer program ID must be non-empty")
    policy.validate()
    interval_env, relu_pre = _forward_ibp_trace_mlp(
        module, input_spec, relu_split_state=dict(relu_split_state)
    )
    scope = build_native_alpha_beta_scope(
        module,
        input_spec,
        linear_spec_C=linear_spec_C,
        relu_pre=relu_pre,
        relu_split_state=relu_split_state,
        policy=policy,
    )
    warm_decision: Optional[NativeWarmStartDecision] = None
    shapes = relu_input_shapes(dict(relu_pre))
    batch = int(input_spec.center.shape[0])
    if warm_start is not None:
        warm_decision = classify_native_alpha_beta_warm_start(
            warm_start,
            target_scope=scope,
            target_split_state=relu_split_state,
        )
        if warm_decision.kind == "rejected":
            raise ValueError(
                f"native optimizer warm start rejected: {warm_decision.reason}"
            )
        alpha = warm_start.alphas
        beta = warm_start.betas
    else:
        alpha = {
            name: torch.full(
                (batch, *shape),
                policy.alpha_init,
                device=input_spec.center.device,
                dtype=input_spec.center.dtype,
            )
            for name, shape in shapes.items()
        }
        beta = {
            name: torch.full(
                (batch, *shape),
                policy.beta_init,
                device=input_spec.center.device,
                dtype=input_spec.center.dtype,
            )
            for name, shape in shapes.items()
        }
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

    initial_bounds = _evaluate_state(
        module,
        input_spec,
        linear_spec_C,
        initial_state.alphas,
        initial_state.betas,
        interval_env=interval_env,
        relu_pre=relu_pre,
        relu_split_state=initial_state.splits,
        objective=policy.objective,
    )
    initial_result = NativeAlphaBetaOptimizationResult(
        bounds=IntervalState(
            lower=initial_bounds.lower.detach().clone(),
            upper=initial_bounds.upper.detach().clone(),
        ),
        state=initial_state,
        interval_env=interval_env,
        relu_pre=relu_pre,
        warm_start_decision=warm_decision,
    )
    source_compilation = compile_native_alpha_beta_state_query(
        module,
        input_spec,
        linear_spec_C=linear_spec_C,
        optimization=initial_result,
        query_id=f"{program_id}:source",
    )
    plan = NativeOptimizerPlanIR(
        plan_id=program_id,
        source_ir_hashes=tuple(sorted(source_compilation.hashes().items())),
        initial_state_hash=initial_state.stable_hash(),
        state_scope_hash=scope.stable_hash(),
        optimizer_policy_hash=policy.stable_hash(),
        steps=policy.steps,
        relu_state_keys=tuple(sorted(initial_state.splits)),
        warm_start_kind=("none" if warm_decision is None else warm_decision.kind),
        objective=policy.objective,
        spec_reduce=policy.spec_reduce,
    )
    task_module, schedule = lower_native_optimizer_ir(plan)
    program = NativeOptimizerProgram(
        plan=plan,
        task_module=task_module,
        schedule=schedule,
        source_compilation=source_compilation,
        initial_state=initial_state,
        interval_env=interval_env,
        relu_pre=relu_pre,
        policy=policy,
        warm_start_decision=warm_decision,
    )
    program.validate()
    return program


def execute_native_alpha_beta_optimizer_program(
    program: NativeOptimizerProgram,
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    linear_spec_C: torch.Tensor,
) -> NativeScheduledOptimizerResult:
    """Execute every optimizer Schedule action in exact Task order."""

    program.validate()
    runtime_scope = build_native_alpha_beta_scope(
        module,
        input_spec,
        linear_spec_C=linear_spec_C,
        relu_pre=program.relu_pre,
        relu_split_state=program.initial_state.splits,
        policy=program.policy,
    )
    if runtime_scope != program.initial_state.scope:
        raise ValueError("native optimizer runtime semantic scope differs")
    alpha = {
        name: value.detach().clone().requires_grad_(True)
        for name, value in program.initial_state.alphas.items()
    }
    beta = {
        name: value.detach().clone().requires_grad_(True)
        for name, value in program.initial_state.betas.items()
    }
    parameters = [*alpha.values(), *beta.values()]
    optimizer = torch.optim.Adam(parameters, lr=program.policy.lr)
    runtime_hashes: dict[str, str] = {
        "optimizer.state.s000": program.initial_state.stable_hash()
    }
    bounds_by_iteration: dict[int, IntervalState] = {}
    metric_by_iteration: dict[int, torch.Tensor] = {}
    state_by_iteration: dict[int, NativeAlphaBetaOptimizationState] = {}
    action_traces: list[NativeOptimizerActionTrace] = []
    evaluation_traces: list[NativeOptimizerEvaluationTrace] = []
    best_metric: Optional[torch.Tensor] = None
    best_bounds: Optional[IntervalState] = None
    best_alpha: Optional[dict[str, torch.Tensor]] = None
    best_beta: Optional[dict[str, torch.Tensor]] = None
    best_iteration: Optional[torch.Tensor] = None

    for action, task in zip(program.schedule.actions, program.task_module.tasks):
        input_hashes = tuple(
            (value_id, runtime_hashes[value_id]) for value_id in task.input_value_ids
        )
        beta_gradient_l1: Optional[float] = None
        alpha_gradient_l1: Optional[float] = None
        iteration = task.iteration
        if task.kind == OptimizerTaskKind.EVALUATE_BOUND:
            assert iteration is not None
            state = _state_from_tensors(program, alpha, beta)
            state_by_iteration[iteration] = state
            bounds = _evaluate_state(
                module,
                input_spec,
                linear_spec_C,
                alpha,
                beta,
                interval_env=program.interval_env,
                relu_pre=program.relu_pre,
                relu_split_state=program.initial_state.splits,
                objective=program.policy.objective,
            )
            bounds_by_iteration[iteration] = bounds
            output_hash = _canonical_hash(
                {
                    "lower": tensor_content_hash(bounds.lower),
                    "upper": tensor_content_hash(bounds.upper),
                }
            )
            runtime_hashes[task.output_value_ids[0]] = output_hash
        elif task.kind == OptimizerTaskKind.REDUCE_METRIC:
            assert iteration is not None
            bounds = bounds_by_iteration[iteration]
            metric = _metric_by_domain(
                bounds,
                objective=program.policy.objective,
                spec_reduce=program.policy.spec_reduce,
                soft_tau=program.policy.soft_tau,
            )
            metric_by_iteration[iteration] = metric
            metric_hash = tensor_content_hash(metric)
            runtime_hashes[task.output_value_ids[0]] = metric_hash
            state = state_by_iteration[iteration]
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
            evaluation_traces.append(
                NativeOptimizerEvaluationTrace(
                    iteration=iteration,
                    state_hash=state.stable_hash(),
                    lower_hash=tensor_content_hash(bounds.lower.detach()),
                    upper_hash=tensor_content_hash(bounds.upper.detach()),
                    metric_hash=metric_hash,
                    metric_values=tuple(float(value) for value in detached.cpu()),
                )
            )
        elif task.kind == OptimizerTaskKind.BACKWARD:
            assert iteration is not None
            optimizer.zero_grad(set_to_none=True)
            loss = -metric_by_iteration[iteration].sum()
            loss.backward()
            alpha_gradient_l1 = sum(
                0.0 if value.grad is None else float(value.grad.abs().sum().item())
                for value in alpha.values()
            )
            beta_gradient_l1 = sum(
                0.0 if value.grad is None else float(value.grad.abs().sum().item())
                for value in beta.values()
            )
            runtime_hashes[task.output_value_ids[0]] = _canonical_hash(
                {
                    "alpha_gradient_l1": alpha_gradient_l1,
                    "beta_gradient_l1": beta_gradient_l1,
                    "iteration": iteration,
                }
            )
        elif task.kind == OptimizerTaskKind.ADAM_UPDATE:
            optimizer.step()
            runtime_hashes[task.output_value_ids[0]] = _canonical_hash(
                {
                    "alpha": {
                        name: tensor_content_hash(value.detach())
                        for name, value in sorted(alpha.items())
                    },
                    "beta": {
                        name: tensor_content_hash(value.detach())
                        for name, value in sorted(beta.items())
                    },
                    "projected": False,
                }
            )
        elif task.kind == OptimizerTaskKind.PROJECT_STATE:
            with torch.no_grad():
                for value in alpha.values():
                    value.clamp_(0.0, 1.0)
                for value in beta.values():
                    value.clamp_(0.0)
            projected = _state_from_tensors(program, alpha, beta)
            runtime_hashes[task.output_value_ids[0]] = projected.stable_hash()
        elif task.kind == OptimizerTaskKind.SELECT_BEST:
            assert best_bounds is not None
            assert best_alpha is not None and best_beta is not None
            selected = _state_from_tensors(program, best_alpha, best_beta)
            runtime_hashes[task.output_value_ids[0]] = selected.stable_hash()
        else:
            raise AssertionError("unreachable optimizer task kind")
        output_hashes = tuple(
            (value_id, runtime_hashes[value_id]) for value_id in task.output_value_ids
        )
        action_traces.append(
            NativeOptimizerActionTrace(
                sequence=action.sequence,
                action_id=action.action_id,
                task_id=task.task_id,
                kind=task.kind,
                iteration=task.iteration,
                input_hashes=input_hashes,
                output_hashes=output_hashes,
                beta_gradient_l1=beta_gradient_l1,
                alpha_gradient_l1=alpha_gradient_l1,
                projection_applied=task.kind == OptimizerTaskKind.PROJECT_STATE,
            )
        )

    assert best_bounds is not None
    assert best_alpha is not None and best_beta is not None
    assert best_iteration is not None
    selected_state = _state_from_tensors(program, best_alpha, best_beta)
    hashes = program.hashes()
    trace = NativeOptimizerExecutionTrace(
        plan_hash=hashes["optimizer_plan_hash"],
        task_module_hash=hashes["optimizer_task_module_hash"],
        schedule_hash=hashes["optimizer_schedule_hash"],
        initial_state_hash=program.initial_state.stable_hash(),
        selected_state_hash=selected_state.stable_hash(),
        actions=tuple(action_traces),
        evaluations=tuple(evaluation_traces),
        best_iteration_by_domain=tuple(
            int(value) for value in best_iteration.detach().cpu()
        ),
    )
    result = NativeScheduledOptimizerResult(
        bounds=best_bounds,
        state=selected_state,
        trace=trace,
    )
    result.validate(program=program)
    return result
