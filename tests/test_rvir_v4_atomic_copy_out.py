"""Real ResNet atomic terminal copy-out tests for RVIR-v4 V4-2E."""

# pylint: disable=missing-function-docstring,too-many-locals

from dataclasses import dataclass, replace
from pathlib import Path

import pytest
import torch

import boundflow.runtime.rvir_v4_atomic_copy_out as copy_out_module
from boundflow.frontends.onnx.frontend import import_onnx
from boundflow.planner import plan_interval_ibp_v0
from boundflow.runtime.native_alpha_beta_optimization_state import (
    build_native_alpha_beta_scope,
    NativeAlphaBetaOptimizationState,
)
from boundflow.runtime.rvir_v4_atomic_copy_out import (
    commit_rvir_v4_atomic_copy_out,
    commit_rvir_v4_live_atomic_copy_out,
    ProductionAtomicCopyOutV4,
    stage_rvir_v4_atomic_copy_out,
    stage_rvir_v4_live_atomic_copy_out,
)
from boundflow.runtime.rvir_v4_native_optimizer import (
    execute_rvir_v4_native_optimizer_trace,
    NativeProductionOptimizerTraceV4,
)
from boundflow.runtime.rvir_v4_optimizer_mutation import (
    production_optimizer_step_trace_from_payload_v4,
)
from boundflow.runtime.rvir_v4_pre_state_initializer import (
    initialize_rvir_v4_native_pre_state,
)
from boundflow.runtime.rvir_v4_production_state import (
    production_snapshot_from_payload_v4,
    ProductionStateSnapshotV4,
    ProductionTensorOwnership,
    ProductionTensorRole,
)
from boundflow.runtime.task_executor import InputSpec
from scripts.run_rvir_v4_pre_state_artifact import TOPOLOGY

ROOT = Path(__file__).resolve().parents[1]
CAPTURE = (
    ROOT
    / "artifacts/rvir-v4-native-optimizer/resnet2b-core-step-parity-v1/source_capture.pt"
)
MODEL = ROOT.parent / "vnncomp2021/benchmarks/cifar10_resnet/onnx/resnet_2b.onnx"


@dataclass(frozen=True)
class _Fixture:
    pre: ProductionStateSnapshotV4
    post: ProductionStateSnapshotV4
    native: NativeProductionOptimizerTraceV4
    terminal: NativeAlphaBetaOptimizationState
    staged: ProductionAtomicCopyOutV4


def _stage() -> _Fixture:
    capture = torch.load(CAPTURE, map_location="cpu", weights_only=True)
    core = capture["cores"][0]
    pre = production_snapshot_from_payload_v4(core["pre_snapshot"])
    post = production_snapshot_from_payload_v4(core["post_snapshot"])
    production = production_optimizer_step_trace_from_payload_v4(
        capture["optimizer_step_traces"][0]
    )
    program = import_onnx(str(MODEL), do_shape_infer=True, normalize=True)
    module = plan_interval_ibp_v0(program)
    lower = next(
        t.value for t in pre.tensors if t.role == ProductionTensorRole.INPUT_LOWER
    )
    upper = next(
        t.value for t in pre.tensors if t.role == ProductionTensorRole.INPUT_UPPER
    )
    objective = next(
        t.value for t in pre.tensors if t.role == ProductionTensorRole.LINEAR_SPEC
    )
    spec = InputSpec.box(value_name=program.graph.inputs[0], lower=lower, upper=upper)
    mapping = initialize_rvir_v4_native_pre_state(pre, TOPOLOGY)
    policy = production.mutation_policy.to_native_policy()
    scope = build_native_alpha_beta_scope(
        module,
        spec,
        linear_spec_C=objective,
        relu_pre=mapping.relu_pre,
        relu_split_state=mapping.splits,
        policy=policy,
    )
    initial = mapping.to_native_state(scope)
    native = execute_rvir_v4_native_optimizer_trace(
        module,
        spec,
        linear_spec_C=objective,
        relu_pre=mapping.relu_pre,
        initial_state=initial,
        mutation_policy=production.mutation_policy,
    )
    terminal = replace(
        initial,
        alpha_by_relu_input=native.steps[-1].alpha_by_relu_input,
        beta_by_relu_input=native.steps[-1].beta_by_relu_input,
    )
    staged = stage_rvir_v4_atomic_copy_out(
        pre=pre,
        terminal_state=terminal,
        topology=TOPOLOGY,
        expected_post=post,
        terminal_lower=native.steps[-1].lower,
        expected_lower=production.steps[-1].lower,
        candidate_snapshot_id="core:000000:native-candidate",
    )
    return _Fixture(pre, post, native, terminal, staged)


def test_terminal_state_stages_all_twelve_paths_and_commits_atomically() -> None:
    fixture = _stage()
    live = {
        path: tensor.value.clone()
        for path, tensor in fixture.pre.tensor_map().items()
        if tensor.ownership == ProductionTensorOwnership.MUTABLE_COPY_OUT
    }

    receipt = commit_rvir_v4_atomic_copy_out(
        fixture.staged, pre=fixture.pre, live_targets=live
    )

    assert len(fixture.staged.path_receipts) == 12
    assert receipt["committed_path_count"] == 12
    assert receipt["atomic_commit"] is True
    assert receipt["provider_callback_count"] == 0
    assert receipt["fallback_dispatch_count"] == 0
    for path, value in live.items():
        torch.testing.assert_close(
            value, fixture.staged.candidate_snapshot.tensor_map()[path].value
        )


def test_invalid_terminal_state_fails_before_any_live_write() -> None:
    fixture = _stage()
    live = {
        path: tensor.value.clone()
        for path, tensor in fixture.pre.tensor_map().items()
        if tensor.ownership == ProductionTensorOwnership.MUTABLE_COPY_OUT
    }
    before = {path: value.clone() for path, value in live.items()}
    alpha = dict(fixture.native.steps[-1].alpha_by_relu_input)
    first = sorted(alpha)[0]
    alpha[first] = alpha[first].clone()
    alpha[first].reshape(-1)[0] = float("nan")
    # Staging rejects before commit is callable; existing live targets remain untouched.
    with pytest.raises(ValueError, match="native alpha/beta tensor contract differs"):
        terminal = replace(
            fixture.terminal,
            alpha_by_relu_input=tuple(sorted(alpha.items())),
        )
        stage_rvir_v4_atomic_copy_out(
            pre=fixture.pre,
            terminal_state=terminal,
            topology=TOPOLOGY,
            expected_post=fixture.post,
            terminal_lower=fixture.native.steps[-1].lower,
            expected_lower=fixture.native.steps[-1].lower,
            candidate_snapshot_id="bad",
        )
    for path in live:
        torch.testing.assert_close(live[path], before[path])


def test_stale_live_target_rejects_without_partial_commit() -> None:
    fixture = _stage()
    live = {
        path: tensor.value.clone()
        for path, tensor in fixture.pre.tensor_map().items()
        if tensor.ownership == ProductionTensorOwnership.MUTABLE_COPY_OUT
    }
    first = sorted(live)[0]
    live[first].reshape(-1)[0] += 0.5
    before = {path: value.clone() for path, value in live.items()}

    with pytest.raises(ValueError, match="live target differs"):
        commit_rvir_v4_atomic_copy_out(
            fixture.staged, pre=fixture.pre, live_targets=live
        )
    for path in live:
        torch.testing.assert_close(live[path], before[path])


def test_runtime_copy_failure_rolls_back_already_written_paths(
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    fixture = _stage()
    live = {
        path: tensor.value.clone()
        for path, tensor in fixture.pre.tensor_map().items()
        if tensor.ownership == ProductionTensorOwnership.MUTABLE_COPY_OUT
    }
    before = {path: value.clone() for path, value in live.items()}
    original = copy_out_module._copy_value  # pylint: disable=protected-access
    calls = 0

    def fail_fifth(target: torch.Tensor, source: torch.Tensor) -> None:
        nonlocal calls
        calls += 1
        if calls == 5:
            raise RuntimeError("injected copy failure")
        original(target, source)

    monkeypatch.setattr(copy_out_module, "_copy_value", fail_fifth)
    with pytest.raises(RuntimeError, match="injected copy failure"):
        commit_rvir_v4_atomic_copy_out(
            fixture.staged, pre=fixture.pre, live_targets=live
        )
    for path in live:
        torch.testing.assert_close(live[path], before[path])


def _live_host_packets(
    fixture: _Fixture,
) -> tuple[dict[str, object], dict[str, object]]:
    thresholds = next(
        tensor.value
        for tensor in fixture.pre.tensors
        if tensor.role == ProductionTensorRole.DECISION_THRESHOLD
    )
    history = [[[], [], [], [], []] for _ in range(6)]
    host: dict[str, object] = {
        "history": history,
        "depths": [1, 1, 1, 1, 1, 1],
        "thresholds": thresholds.clone(),
        "discard_after_core": torch.arange(6),
    }
    candidate: dict[str, object] = {
        "history": history,
        "depths": [1, 1, 1, 1, 1, 1],
        "thresholds": thresholds.clone(),
    }
    return host, candidate


def test_live_candidate_has_no_expected_post_dependency_and_commits_host() -> None:
    fixture = _stage()
    host, candidate_host = _live_host_packets(fixture)
    staged = stage_rvir_v4_live_atomic_copy_out(
        pre=fixture.pre,
        terminal_state=fixture.terminal,
        topology=TOPOLOGY,
        terminal_lower=fixture.native.steps[-1].lower,
        host_packet=host,
        host_packet_candidate=candidate_host,
        candidate_snapshot_id="core:000000:live-candidate",
    )
    live = {
        path: tensor.value.clone()
        for path, tensor in fixture.pre.tensor_map().items()
        if tensor.ownership == ProductionTensorOwnership.MUTABLE_COPY_OUT
    }

    receipt = commit_rvir_v4_live_atomic_copy_out(
        staged,
        pre=fixture.pre,
        live_targets=live,
        host_packet=host,
    )

    assert len(staged.path_receipts) == 12
    assert sum(row.changed for row in staged.path_receipts) == 7
    assert receipt["committed_path_count"] == 12
    assert receipt["changed_path_count"] == 7
    assert receipt["atomic_live_and_host_commit"] is True
    assert set(host) == {"depths", "history", "thresholds"}
    assert host["depths"] == candidate_host["depths"]
    assert host["history"] == candidate_host["history"]
    torch.testing.assert_close(host["thresholds"], candidate_host["thresholds"])
    for path, value in live.items():
        torch.testing.assert_close(
            value, staged.candidate_snapshot.tensor_map()[path].value
        )


def test_live_host_failure_rolls_back_all_tensors_and_host(
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    fixture = _stage()
    host, candidate_host = _live_host_packets(fixture)
    staged = stage_rvir_v4_live_atomic_copy_out(
        pre=fixture.pre,
        terminal_state=fixture.terminal,
        topology=TOPOLOGY,
        terminal_lower=fixture.native.steps[-1].lower,
        host_packet=host,
        host_packet_candidate=candidate_host,
        candidate_snapshot_id="core:000000:live-candidate",
    )
    live = {
        path: tensor.value.clone()
        for path, tensor in fixture.pre.tensor_map().items()
        if tensor.ownership == ProductionTensorOwnership.MUTABLE_COPY_OUT
    }
    tensor_before = {path: value.clone() for path, value in live.items()}
    host_before = dict(host)
    original = copy_out_module._replace_host_packet  # pylint: disable=protected-access
    calls = 0

    def fail_first_host_write(target, source) -> None:  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        if calls == 1:
            target.clear()
            raise RuntimeError("injected host packet failure")
        original(target, source)

    monkeypatch.setattr(copy_out_module, "_replace_host_packet", fail_first_host_write)
    with pytest.raises(RuntimeError, match="injected host packet failure"):
        commit_rvir_v4_live_atomic_copy_out(
            staged,
            pre=fixture.pre,
            live_targets=live,
            host_packet=host,
        )
    assert set(host) == set(host_before)
    assert host["depths"] == host_before["depths"]
    assert host["history"] == host_before["history"]
    torch.testing.assert_close(host["thresholds"], host_before["thresholds"])
    torch.testing.assert_close(
        host["discard_after_core"], host_before["discard_after_core"]
    )
    for path in live:
        torch.testing.assert_close(live[path], tensor_before[path])
