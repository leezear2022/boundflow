"""PR-13C same-solver query-runtime integration tests."""

# pylint: disable=duplicate-code

from dataclasses import replace

import pytest
import torch

from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime import bab_query_runtime as runtime_module
from boundflow.runtime.bab import BabConfig, solve_bab_mlp
from boundflow.runtime.bab_query import (
    BoundQueryRequest,
    BoundQueryResult,
    FixedBabQueryRecorder,
    compare_query_results,
)
from boundflow.runtime.bab_query_runtime import (
    SameSolverQueryRuntime,
    SameSolverRuntimeConfig,
)
from boundflow.runtime.task_executor import InputSpec


def _make_module() -> BFTaskModule:
    task = BoundTask(
        task_id="t0",
        kind=TaskKind.INTERVAL_IBP,
        ops=[
            TaskOp(
                op_type="linear",
                name="linear1",
                inputs=["input", "W1", "b1"],
                outputs=["h1"],
            ),
            TaskOp(op_type="relu", name="relu1", inputs=["h1"], outputs=["r1"]),
            TaskOp(
                op_type="linear",
                name="linear2",
                inputs=["r1", "W2", "b2"],
                outputs=["h2"],
            ),
            TaskOp(op_type="relu", name="relu2", inputs=["h2"], outputs=["r2"]),
            TaskOp(
                op_type="linear",
                name="linear3",
                inputs=["r2", "W3", "b3"],
                outputs=["out"],
            ),
        ],
        input_values=["input"],
        output_values=["out"],
    )
    params = {
        "W1": torch.tensor([[1.0], [-1.0]], dtype=torch.float32),
        "b1": torch.tensor([0.1, -0.1], dtype=torch.float32),
        "W2": torch.tensor([[1.0, -0.5], [-0.25, 1.0]], dtype=torch.float32),
        "b2": torch.tensor([-0.2, 0.15], dtype=torch.float32),
        "W3": torch.tensor([[1.0, -0.75]], dtype=torch.float32),
        "b3": torch.tensor([-0.1], dtype=torch.float32),
    }
    return BFTaskModule(tasks=[task], entry_task_id="t0", bindings={"params": params})


def _module_on(device: torch.device) -> BFTaskModule:
    """Move the deterministic test module without mutating shared tensors."""

    module = _make_module()
    params = module.bindings["params"]
    assert isinstance(params, dict)
    return BFTaskModule(
        tasks=module.tasks,
        entry_task_id=module.entry_task_id,
        bindings={
            "params": {
                name: value.to(device)
                for name, value in params.items()
                if torch.is_tensor(value)
            }
        },
    )


def _config(*, node_batch_size: int = 4) -> BabConfig:
    return BabConfig(
        max_nodes=9,
        oracle="alpha_beta",
        node_batch_size=node_batch_size,
        enable_node_eval_cache=False,
        alpha_steps=3,
        alpha_init=0.5,
        beta_init=0.0,
        threshold=0.35,
        tol=1e-8,
    )


def _runtime(*, max_batch_size: int = 4) -> SameSolverQueryRuntime:
    return SameSolverQueryRuntime(
        SameSolverRuntimeConfig(
            max_batch_size=max_batch_size,
            memory_budget_bytes=1 << 30,
            allow_legacy_alpha_beta=True,
        )
    )


def test_pr13c_same_solver_runtime_preserves_search_and_results() -> None:
    """Only the bound-call executor changes; search counters and trace stay equal."""

    module = _make_module()
    spec = InputSpec.linf(
        value_name="input",
        center=torch.tensor([[0.0]], dtype=torch.float32),
        eps=1.0,
    )
    baseline_recorder = FixedBabQueryRecorder()
    runtime_recorder = FixedBabQueryRecorder()
    baseline = solve_bab_mlp(
        module,
        spec,
        config=_config(),
        query_recorder=baseline_recorder,
    )
    runtime = _runtime()
    adapted = solve_bab_mlp(
        module,
        spec,
        config=_config(),
        query_recorder=runtime_recorder,
        query_runtime=runtime,
    )

    assert adapted.status == baseline.status
    assert adapted.nodes_visited == baseline.nodes_visited
    assert adapted.nodes_evaluated == baseline.nodes_evaluated
    assert adapted.nodes_expanded == baseline.nodes_expanded
    assert adapted.max_queue == baseline.max_queue
    assert adapted.batch_rounds == baseline.batch_rounds
    assert adapted.avg_batch_fill_rate == pytest.approx(baseline.avg_batch_fill_rate)
    assert adapted.best_lower == pytest.approx(baseline.best_lower, abs=2e-4)
    assert adapted.best_upper == pytest.approx(baseline.best_upper, abs=2e-4)

    baseline_recorder.validate_complete()
    runtime_recorder.validate_complete()
    assert [entry.query.query_id for entry in runtime_recorder.entries] == [
        entry.query.query_id for entry in baseline_recorder.entries
    ]
    for expected, actual in zip(baseline_recorder.entries, runtime_recorder.entries):
        assert expected.result is not None and actual.result is not None
        assert compare_query_results(
            expected.query.query_id, expected.result, actual.result
        ).passed
    audit = runtime.audit()
    assert audit["submitted_queries"] == len(runtime_recorder.entries)
    assert audit["completed_queries"] == len(runtime_recorder.entries)
    assert audit["query_loss"] == 0
    assert audit["dispatch_plan_cache_misses"] == 1
    dispatch_hits = audit["dispatch_plan_cache_hits"]
    assert isinstance(dispatch_hits, int) and dispatch_hits >= 1
    assert audit["compiled_plan_cache_applicable"] is False
    assert audit["pr12_planner_dispatches"] == 0


def test_pr13c_alpha_serial_runtime_preserves_solver_status() -> None:
    """The adapter also retains the original alpha-only serial path semantics."""

    module = _make_module()
    spec = InputSpec.linf(
        value_name="input",
        center=torch.tensor([[0.0]], dtype=torch.float32),
        eps=1.0,
    )
    config = replace(
        _config(node_batch_size=1),
        oracle="alpha",
        use_1d_linf_input_restriction_patch=True,
    )
    baseline = solve_bab_mlp(module, spec, config=config)
    adapted = solve_bab_mlp(
        module,
        spec,
        config=config,
        query_runtime=_runtime(max_batch_size=1),
    )
    assert adapted.status == baseline.status
    assert adapted.nodes_visited == baseline.nodes_visited
    assert adapted.nodes_evaluated == baseline.nodes_evaluated
    assert adapted.nodes_expanded == baseline.nodes_expanded
    assert adapted.best_lower == pytest.approx(baseline.best_lower, abs=2e-4)
    assert adapted.best_upper == pytest.approx(baseline.best_upper, abs=2e-4)


def test_pr13c_invalid_capability_never_invokes_dense_alpha_beta_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A forged plain-CROWN capability is rejected before physical execution."""

    module = _make_module()
    spec = InputSpec.linf(
        value_name="input",
        center=torch.tensor([[0.0]], dtype=torch.float32),
        eps=1.0,
    )
    recorder = FixedBabQueryRecorder()
    solve_bab_mlp(
        module,
        spec,
        config=replace(_config(node_batch_size=1), max_nodes=1),
        query_recorder=recorder,
    )
    entry = recorder.entries[0]
    forged_key = replace(
        entry.query.compatibility_key,
        backend_capability_class="plain_crown_fused",
    )
    forged_query = replace(entry.query, compatibility_key=forged_key)
    calls = 0

    def forbidden_executor(_module, _batch):
        nonlocal calls
        calls += 1
        raise AssertionError("physical alpha-beta executor must not run")

    monkeypatch.setattr(
        "boundflow.runtime.bab_query_runtime.execute_alpha_beta_query_batch",
        forbidden_executor,
    )
    runtime = _runtime(max_batch_size=1)
    with pytest.raises(ValueError, match="unsupported capability"):
        runtime.execute(
            module,
            [BoundQueryRequest(forged_query, entry.payload)],
        )
    assert calls == 0
    assert runtime.audit()["dispatch_plan_cache_entries"] == 0


def test_pr13c_solver_fails_closed_on_rejected_runtime_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A runtime failure must abort, never drop a node and report it proven."""

    module = _make_module()
    spec = InputSpec.linf(
        value_name="input",
        center=torch.tensor([[0.0]], dtype=torch.float32),
        eps=1.0,
    )
    runtime = _runtime(max_batch_size=1)

    def rejected(_module, requests, **_kwargs):
        request = requests[0]
        return [
            (
                request.query.query_id,
                BoundQueryResult(
                    status="rejected",
                    lower=None,
                    upper=None,
                    branch=None,
                    alpha_state_version=None,
                    beta_state_version=None,
                ),
            )
        ]

    monkeypatch.setattr(runtime, "execute", rejected)
    with pytest.raises(RuntimeError, match="rejected node"):
        solve_bab_mlp(
            module,
            spec,
            config=replace(_config(node_batch_size=1), max_nodes=1),
            query_runtime=runtime,
        )


def test_pr13c_solver_fails_closed_on_reordered_runtime_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The solver defensively rejects a runtime that violates query order."""

    module = _make_module()
    spec = InputSpec.linf(
        value_name="input",
        center=torch.tensor([[0.0]], dtype=torch.float32),
        eps=1.0,
    )
    runtime = _runtime(max_batch_size=4)
    original_execute = runtime.execute

    def reordered(module_arg, requests, **kwargs):
        outputs = original_execute(module_arg, requests, **kwargs)
        return list(reversed(outputs)) if len(outputs) > 1 else outputs

    monkeypatch.setattr(runtime, "execute", reordered)
    with pytest.raises(RuntimeError, match="reordered batch results"):
        solve_bab_mlp(
            module,
            spec,
            config=_config(node_batch_size=4),
            query_runtime=runtime,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
# pylint: disable-next=too-many-locals
def test_pr13c_runtime_obeys_non_default_torch_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execute and consume results on one custom stream without global sync."""

    device = torch.device("cuda")
    module = _module_on(device)
    spec = InputSpec.linf(
        value_name="input",
        center=torch.tensor([[0.0]], dtype=torch.float32, device=device),
        eps=1.0,
    )
    recorder = FixedBabQueryRecorder()
    solve_bab_mlp(
        module,
        spec,
        config=replace(_config(node_batch_size=1), max_nodes=1, alpha_steps=1),
        query_recorder=recorder,
    )
    entry = recorder.entries[0]
    request = BoundQueryRequest(entry.query, entry.payload)
    runtime = _runtime(max_batch_size=1)
    custom = torch.cuda.Stream(device=device)
    default = torch.cuda.current_stream(device)
    custom.wait_stream(default)
    observed_streams: list[int] = []

    original_executor = runtime_module.execute_alpha_beta_query_batch

    def observed_executor(module_arg, batch_arg):
        observed_streams.append(torch.cuda.current_stream(device).cuda_stream)
        return original_executor(module_arg, batch_arg)

    monkeypatch.setattr(
        runtime_module, "execute_alpha_beta_query_batch", observed_executor
    )
    done = torch.cuda.Event()
    with torch.cuda.stream(custom):
        [(query_id, result)] = runtime.execute(module, [request])
        assert result.lower is not None and result.upper is not None
        checksum = (result.lower + result.upper).sum()
        done.record(custom)
    done.synchronize()

    assert query_id == request.query.query_id
    assert observed_streams == [custom.cuda_stream]
    assert torch.isfinite(checksum).item()
