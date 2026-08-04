"""Parametric production queue parity, cache, and tamper tests."""

# pylint: disable=missing-function-docstring,duplicate-code

from dataclasses import replace

import pytest
import torch

from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime.native_alpha_beta_optimization_state import (
    NativeAlphaBetaOptimizerPolicy,
)
from boundflow.runtime.native_parametric_optimizer import (
    NativeParametricOptimizerTemplateCache,
)
from boundflow.runtime.native_parametric_production_verifier import (
    execute_native_parametric_production_relu_split_bab,
)
from boundflow.runtime.native_production_verifier import (
    execute_native_production_relu_split_bab,
)
from boundflow.runtime.native_relu_split_bab_runtime import NativeReluSplitBabConfig
from boundflow.runtime.task_executor import InputSpec


def _module() -> BFTaskModule:
    return BFTaskModule(
        tasks=[
            BoundTask(
                task_id="parametric-production-toy",
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
        entry_task_id="parametric-production-toy",
        bindings={
            "params": {
                "W1": torch.tensor(
                    [[1.0, -0.5, 0.25], [-0.25, 0.75, 1.0], [0.5, 0.5, -1.0]]
                ),
                "b1": torch.tensor([0.1, -0.2, 0.05]),
                "W2": torch.tensor([[0.75, -1.0, 0.5], [-0.5, 0.25, 1.25]]),
                "b2": torch.tensor([0.15, -0.1]),
            }
        },
    )


def _spec() -> InputSpec:
    return InputSpec.box(
        value_name="input",
        lower=torch.tensor([[-0.3, -0.6, -0.1]]),
        upper=torch.tensor([[0.7, 0.4, 0.9]]),
    )


def _policy() -> NativeAlphaBetaOptimizerPolicy:
    return NativeAlphaBetaOptimizerPolicy(steps=1, lr=0.1)


def _config() -> NativeReluSplitBabConfig:
    return NativeReluSplitBabConfig(
        max_nodes=7,
        max_depth=2,
        expansion_batch_size=2,
        max_eval_batch_size=4,
        threshold=1e6,
    )


def test_parametric_production_queue_matches_v1_with_one_template() -> None:
    module = _module()
    spec = _spec()
    objective = torch.tensor([[1.0, -1.0]])
    baseline = execute_native_production_relu_split_bab(
        module,
        spec,
        linear_spec_C=objective,
        run_id="parametric-production-parity",
        config=_config(),
        optimizer_policy=_policy(),
    )
    cache = NativeParametricOptimizerTemplateCache()
    parametric = execute_native_parametric_production_relu_split_bab(
        module,
        spec,
        linear_spec_C=objective,
        run_id="parametric-production-parity",
        config=_config(),
        optimizer_policy=_policy(),
        compiler_cache=cache,
    )

    cache.validate()
    assert parametric.trace.logical_queue_signature() == (
        baseline.trace.logical_queue_signature()
    )
    assert [
        (item.lower, item.upper, item.selected_state_hash)
        for item in parametric.trace.evaluations
    ] == [
        (item.lower, item.upper, item.selected_state_hash)
        for item in baseline.trace.evaluations
    ]
    assert len(cache.templates) == 1
    assert len(cache.events) == len(parametric.trace.batches) == 3
    assert [event.outcome for event in cache.events] == [
        "miss_compiled",
        "hit_exact_contract",
        "hit_exact_contract",
    ]


def test_parametric_production_cache_reuses_template_across_objectives() -> None:
    module = _module()
    spec = _spec()
    cache = NativeParametricOptimizerTemplateCache()
    for index, objective in enumerate(
        (torch.tensor([[1.0, -1.0]]), torch.tensor([[-1.0, 1.0]]))
    ):
        execute_native_parametric_production_relu_split_bab(
            module,
            spec,
            linear_spec_C=objective,
            run_id=f"parametric-production-objective-{index}",
            config=_config(),
            optimizer_policy=_policy(),
            compiler_cache=cache,
        )

    cache.validate()
    assert len(cache.templates) == 1
    assert len(cache.events) == 6
    assert sum(event.outcome == "miss_compiled" for event in cache.events) == 1
    assert sum(event.outcome == "hit_exact_contract" for event in cache.events) == 5


def test_parametric_production_compiler_trace_tamper_fails_closed() -> None:
    execution = execute_native_parametric_production_relu_split_bab(
        _module(),
        _spec(),
        linear_spec_C=torch.tensor([[1.0, -1.0]]),
        run_id="parametric-production-tamper",
        config=_config(),
        optimizer_policy=_policy(),
        compiler_cache=NativeParametricOptimizerTemplateCache(),
    )
    first = execution.compiler_batches[0]
    tampered = replace(
        first,
        cache_event=replace(first.cache_event, cache_key="f" * 64),
    )

    with pytest.raises(ValueError, match="batch trace differs"):
        replace(
            execution,
            compiler_batches=(tampered, *execution.compiler_batches[1:]),
        ).validate()
