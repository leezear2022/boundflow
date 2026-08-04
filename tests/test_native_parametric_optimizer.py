"""Parametric optimizer compiler/cache/runtime parity and rejection tests."""

# pylint: disable=missing-function-docstring,duplicate-code

import pytest
import torch

from boundflow.ir.bound import IntermediateBoundSource
from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime.crown_ibp import _forward_ibp_trace_mlp
from boundflow.runtime.native_alpha_beta_optimization_state import (
    NativeAlphaBetaOptimizerPolicy,
)
from boundflow.runtime.native_alpha_beta_optimizer_schedule import (
    NativePreparedOptimizerProgram,
    compile_native_alpha_beta_optimizer_program,
    execute_prepared_native_alpha_beta_optimizer_program,
)
from boundflow.runtime.native_parametric_optimizer import (
    NativeParametricOptimizerTemplateCache,
    execute_native_parametric_optimizer,
    instantiate_native_parametric_optimizer,
)
from boundflow.runtime.task_executor import InputSpec


def _module() -> BFTaskModule:
    return BFTaskModule(
        tasks=[
            BoundTask(
                task_id="parametric-optimizer-toy",
                kind=TaskKind.INTERVAL_IBP,
                ops=[
                    TaskOp("linear", "linear1", ["input", "W1", "b1"], ["h1"]),
                    TaskOp("relu", "relu1", ["h1"], ["r1"]),
                    TaskOp("linear", "linear2", ["r1", "W2", "b2"], ["out"]),
                ],
                input_values=["input"],
                output_values=["out"],
            )
        ],
        entry_task_id="parametric-optimizer-toy",
        bindings={
            "params": {
                "W1": torch.tensor([[1.0, -0.5], [0.25, 1.0]]),
                "b1": torch.tensor([0.0, 0.1]),
                "W2": torch.tensor([[1.0, -0.75], [-0.25, 0.5]]),
                "b2": torch.tensor([0.0, -0.1]),
            }
        },
    )


def _spec() -> InputSpec:
    return InputSpec.box(
        value_name="input",
        lower=torch.tensor([[-1.0, -0.5]]),
        upper=torch.tensor([[1.0, 0.75]]),
    )


def _policy() -> NativeAlphaBetaOptimizerPolicy:
    return NativeAlphaBetaOptimizerPolicy(steps=2, lr=0.1)


def _template_and_instance():
    module = _module()
    spec = _spec()
    objective = torch.tensor([[1.0, -1.0]])
    _interval, relu_pre = _forward_ibp_trace_mlp(module, spec)
    split = {
        name: torch.zeros_like(value.lower, dtype=torch.int8)
        for name, value in relu_pre.items()
    }
    cache = NativeParametricOptimizerTemplateCache()
    template, event = cache.acquire(
        module,
        spec,
        linear_spec_C=objective,
        relu_pre=relu_pre,
        policy=_policy(),
        intermediate_bound_source=IntermediateBoundSource.LOCAL_FORWARD,
        refine_external_constraints=False,
        template_id="parametric-template",
        batch_id="batch-0",
    )
    instance = instantiate_native_parametric_optimizer(
        template,
        module,
        spec,
        linear_spec_C=objective,
        relu_split_state=split,
        instance_id="instance-0",
    )
    return module, spec, objective, split, relu_pre, cache, template, event, instance


def test_parametric_optimizer_matches_prepared_v1_exactly() -> None:
    module, spec, objective, split, _pre, _cache, template, event, instance = (
        _template_and_instance()
    )
    parametric = execute_native_parametric_optimizer(
        instance,
        module,
        spec,
        linear_spec_C=objective,
        intermediate_bound_source=IntermediateBoundSource.LOCAL_FORWARD,
    )
    baseline_program = compile_native_alpha_beta_optimizer_program(
        module,
        spec,
        linear_spec_C=objective,
        relu_split_state=split,
        policy=_policy(),
        program_id="baseline",
    )
    prepared = NativePreparedOptimizerProgram.prepare(
        baseline_program,
        module,
        spec,
        linear_spec_C=objective,
        intermediate_bound_source=IntermediateBoundSource.LOCAL_FORWARD,
    )
    baseline = execute_prepared_native_alpha_beta_optimizer_program(
        prepared,
        module,
        spec,
        linear_spec_C=objective,
        intermediate_bound_source=IntermediateBoundSource.LOCAL_FORWARD,
    )

    assert event.outcome == "miss_compiled"
    assert torch.equal(parametric.bounds.lower, baseline.bounds.lower)
    assert torch.equal(parametric.bounds.upper, baseline.bounds.upper)
    assert parametric.state.stable_hash() == baseline.state.stable_hash()
    assert template.hashes().keys() == {
        "optimizer_plan_hash",
        "optimizer_task_module_hash",
        "optimizer_schedule_hash",
    }


def test_parametric_optimizer_cache_hits_across_objective_content() -> None:
    module, spec, _objective, _split, relu_pre, cache, template, _event, _instance = (
        _template_and_instance()
    )
    second, event = cache.acquire(
        module,
        spec,
        linear_spec_C=torch.tensor([[-1.0, 1.0]]),
        relu_pre=relu_pre,
        policy=_policy(),
        intermediate_bound_source=IntermediateBoundSource.LOCAL_FORWARD,
        refine_external_constraints=False,
        template_id="ignored-on-hit",
        batch_id="batch-1",
    )

    cache.validate()
    assert second is template
    assert event.outcome == "hit_exact_contract"
    assert event.compile_elapsed_ns == 0
    assert len(cache.templates) == 1
    assert [item.outcome for item in cache.events] == [
        "miss_compiled",
        "hit_exact_contract",
    ]


def test_parametric_optimizer_cache_rejects_contract_drift() -> None:
    module, spec, _objective, _split, relu_pre, cache, _template, _event, _instance = (
        _template_and_instance()
    )
    with pytest.raises(ValueError, match="template contract differs"):
        cache.acquire(
            module,
            spec,
            linear_spec_C=torch.tensor([[[1.0, -1.0]]]),
            relu_pre=relu_pre,
            policy=_policy(),
            intermediate_bound_source=IntermediateBoundSource.LOCAL_FORWARD,
            refine_external_constraints=False,
            template_id="contract-drift",
            batch_id="batch-1",
        )


def test_parametric_optimizer_instance_rejects_runtime_tensor_rebinding() -> None:
    module, spec, objective, _split, _pre, _cache, _template, _event, instance = (
        _template_and_instance()
    )
    with pytest.raises(ValueError, match="exact runtime binding differs"):
        execute_native_parametric_optimizer(
            instance,
            module,
            spec,
            linear_spec_C=objective.clone(),
            intermediate_bound_source=IntermediateBoundSource.LOCAL_FORWARD,
        )
