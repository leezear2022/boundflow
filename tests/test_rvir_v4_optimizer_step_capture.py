"""Scoped production-call capture tests for RVIR-v4 V4-2B."""

# pylint: disable=missing-function-docstring,protected-access
# pylint: disable=too-few-public-methods,duplicate-code

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from boundflow.runtime.rvir_v4_optimizer_mutation import (
    production_optimizer_step_trace_from_payload_v4,
    production_optimizer_step_trace_to_payload_v4,
)
from boundflow.runtime.rvir_v4_production_state import ProductionOptimizerPolicyV4
from boundflow.runtime.rvir_v4_production_state import (
    capture_module_alpha_beta_state_v4,
)
from scripts import run_rvir_v4_optimizer_step_artifact as artifact_runner
from scripts import run_rvir_v4_production_state_capture as capture_runner


def _loss_reduction_sum(value: torch.Tensor) -> torch.Tensor:
    return value.sum(dim=-1)


class _SparseBeta:
    def __init__(self, *, optimized: bool) -> None:
        self.val = torch.nn.Parameter(torch.zeros((6, 1)), requires_grad=optimized)
        self.loc = torch.zeros((6, 1), dtype=torch.long)
        self.sign = torch.ones((6, 1))


class _Node:
    def __init__(self, ordinal: int) -> None:
        self.name = f"layer-{ordinal}"
        self.alpha = {
            "start": torch.nn.Parameter(torch.full((2, 1, 6, 1), ordinal / 100.0))
        }
        self.sparse_betas = {0: _SparseBeta(optimized=ordinal == 0)}


class _FakeBoundedModule:
    def __init__(self) -> None:
        self._nodes = [_Node(ordinal) for ordinal in range(6)]
        self.cut_used = False
        self.bound_opts = {
            "optimize_bound_args": {
                "optimizer": "adam",
                "lr_decay": 0.98,
                "keep_best": True,
                "loss_reduction_func": _loss_reduction_sum,
                "early_stop_patience": 10,
                "start_save_best": 0.5,
                "use_float64_in_last_iteration": False,
                "pruning_in_iteration": True,
                "pruning_in_iteration_threshold": 0.2,
                "max_time": 60.0,
                "enable_alpha_crown": True,
                "enable_beta_crown": True,
                "init_alpha": False,
                "use_shared_alpha": False,
                "apply_output_constraints_to": [],
                "directly_optimize": [],
                "tighten_input_bounds": False,
            }
        }

    def nodes(self) -> list[_Node]:
        return self._nodes

    def compute_bounds(self, *_args: Any, **kwargs: Any) -> tuple[torch.Tensor, None]:
        if kwargs.get("method") != "CROWN-optimized":
            value = sum(node.alpha["start"].sum() for node in self._nodes)
            value = value + self._nodes[0].sparse_betas[0].val.sum()
            return value.expand(6, 1), None

        optimizer = torch.optim.Adam(
            (
                {
                    "params": [node.alpha["start"] for node in self._nodes],
                    "lr": 0.01,
                },
                {"params": [self._nodes[0].sparse_betas[0].val], "lr": 0.05},
            )
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.98)
        result: tuple[torch.Tensor, None] | None = None
        for ordinal in range(10):
            result = self.compute_bounds(
                method="backward", bound_lower=True, bound_upper=False
            )
            if ordinal < 9:
                optimizer.zero_grad(set_to_none=True)
                (-result[0].sum()).backward()
                optimizer.step()
            scheduler.step()
        assert result is not None
        return result


def _production_policy() -> ProductionOptimizerPolicyV4:
    return ProductionOptimizerPolicyV4(
        iteration=10,
        alpha_learning_rate=0.01,
        beta_learning_rate=0.05,
        bound_lower=True,
        bound_upper=False,
        fix_intermediate_bounds=True,
        deterministic=False,
        stop_criterion_id="auto_LiRPA.utils.stop_criterion_batch_any.lambda",
    )


def _worker_capture(observer: Any) -> dict[str, object]:
    calls = list(observer.calls)
    calls.extend(
        {
            "call_id": call_id,
            "parent_call_id": None,
            "depth": 0,
            "core_id": None,
            "phase": "initial_crown" if call_id < 23 else "alpha_optimize",
            "bound_lower": True,
            "bound_upper": False,
        }
        for call_id in range(11, 24)
    )
    return {
        "schema_version": capture_runner.OPTIMIZER_WORKER_SCHEMA_VERSION,
        "source": {
            "abcrown_commit": capture_runner.ABCROWN_COMMIT,
            "auto_lirpa_commit": capture_runner.AUTO_LIRPA_COMMIT,
            "vnncomp_commit": capture_runner.VNNCOMP_COMMIT,
            "model_relative_path": capture_runner.MODEL_RELATIVE_PATH,
            "property_relative_path": capture_runner.PROPERTY_RELATIVE_PATH,
            "model_sha256": capture_runner.MODEL_SHA256,
            "property_sha256": capture_runner.PROPERTY_SHA256,
        },
        "protocol": {
            "device": "cuda",
            "seed": 100,
            "max_iterations": 1,
            "batch_size": 64,
            "alpha_steps": 5,
            "beta_steps": 10,
            "property_cache": "cold_isolated_copy",
            "performance_claimed": False,
        },
        "solver_result": {
            "status": "unknown",
            "success": False,
            "visited_domains": [6],
        },
        "calls": calls,
        "cores": [
            {
                "core_id": 0,
                "pre_snapshot": {"optimizer_policy": _production_policy().to_dict()},
            }
        ],
        "optimizer_step_traces": observer.optimizer_step_traces,
        "performance_claimed": False,
    }


def test_scoped_capture_observes_real_ten_evaluation_nine_adam_step_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        capture_runner, "_phase_from_stack", lambda _method: ("beta_split", "test")
    )
    observer = capture_runner._CaptureObserver(
        torch, SimpleNamespace(Config={}), capture_optimizer_steps=True
    )
    observer._active_core_id = 0
    observer._core_policies[0] = _production_policy()
    module = _FakeBoundedModule()

    with (
        observer.instrument_compute(_FakeBoundedModule),
        observer.instrument_adam(),
    ):
        module.compute_bounds(
            method="CROWN-optimized", bound_lower=True, bound_upper=False
        )
    observer._finalize_optimizer_trace(0)

    assert len(observer.calls) == 11
    assert len(observer.optimizer_step_traces) == 1
    replayed = production_optimizer_step_trace_from_payload_v4(
        observer.optimizer_step_traces[0]
    )
    assert [step.updates_before for step in replayed.steps] == list(range(10))
    assert [step.optimizer_step_ordinal for step in replayed.steps] == [
        *range(9),
        None,
    ]
    assert sum(step.update_after for step in replayed.steps) == 9
    assert all(
        len(step.state_tensors) == 24 and tuple(step.lower.shape) == (6, 1)
        for step in replayed.steps
    )
    assert replayed.steps[4].alpha_learning_rate == pytest.approx(0.01 * 0.98**4)
    assert replayed.steps[4].beta_learning_rate == pytest.approx(0.05 * 0.98**4)

    cuda_trace = replace(
        replayed,
        steps=tuple(
            replace(
                step,
                state_tensors=tuple(
                    replace(tensor, source_device="cuda:0")
                    for tensor in step.state_tensors
                ),
            )
            for step in replayed.steps
        ),
    )
    observer.optimizer_step_traces = [
        production_optimizer_step_trace_to_payload_v4(cuda_trace)
    ]
    trace_projection, summary = artifact_runner.validate_worker_capture(
        _worker_capture(observer)
    )
    assert trace_projection == cuda_trace.metadata()
    assert summary["status"] == "validated-step-capture-schema"
    assert summary["evaluation_count"] == 10
    assert summary["update_count"] == 9
    assert summary["state_tensor_counts"] == [24] * 10
    assert summary["all_state_sources_cuda"] is True
    assert summary["adjacent_mutable_change_counts"] == [7] * 9

    forged_steps = list(cuda_trace.steps)
    forged_tensors = list(forged_steps[0].state_tensors)
    forged_tensors[0] = replace(forged_tensors[0], source_device="cuda-forged")
    forged_steps[0] = replace(forged_steps[0], state_tensors=tuple(forged_tensors))
    observer.optimizer_step_traces = [
        production_optimizer_step_trace_to_payload_v4(
            replace(cuda_trace, steps=tuple(forged_steps))
        )
    ]
    with pytest.raises(ValueError, match="not CUDA production state"):
        artifact_runner.validate_worker_capture(_worker_capture(observer))
    observer.optimizer_step_traces = [
        production_optimizer_step_trace_to_payload_v4(cuda_trace)
    ]

    tampered = _worker_capture(observer)
    tampered_source = tampered["source"]
    assert isinstance(tampered_source, dict)
    tampered_source["abcrown_commit"] = "0" * 40
    with pytest.raises(ValueError, match="source identity"):
        artifact_runner.validate_worker_capture(tampered)


def test_scoped_capture_rejects_unexpected_optimizer_parameter_groups() -> None:
    observer = capture_runner._CaptureObserver(
        torch, SimpleNamespace(Config={}), capture_optimizer_steps=True
    )
    observer._active_core_id = 0
    observer._core_policies[0] = _production_policy()
    module = _FakeBoundedModule()
    observer._capture_mutation_policy(module)
    observer._active_adam = SimpleNamespace(param_groups=[{"lr": 0.01}])
    state = capture_module_alpha_beta_state_v4(module.nodes(), require_beta=True)

    with pytest.raises(ValueError, match="parameter-group count"):
        observer._begin_optimizer_evaluation(
            call_id=1, parent_call_id=0, state_tensors=state
        )
