"""GPU transaction gates for the FSG4/B3-C atomic commit plan."""

# pylint: disable=missing-function-docstring,protected-access,too-many-locals

from dataclasses import replace

import pytest
import torch

import boundflow.runtime.fsg4_b3_device_atomic_commit as device_atomic
from boundflow.runtime.fsg4_b3_device_atomic_commit import (
    audit_device_atomic_transaction_v1,
    commit_device_atomic_transaction_v1,
    compile_device_atomic_commit_plan_v1,
    DeviceAtomicPathSpecV1,
    stage_device_atomic_transaction_v1,
)
from boundflow.runtime.rvir_v4_production_state import (
    ProductionTensorOwnership,
)
from scripts.run_rvir_v4_pre_state_artifact import TOPOLOGY
from tests.test_rvir_v4_atomic_copy_out import _live_host_packets, _stage

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _terminal_cuda(terminal):  # type: ignore[no-untyped-def]
    return replace(
        terminal,
        split_by_relu_input=tuple(
            (name, value.to("cuda")) for name, value in terminal.split_by_relu_input
        ),
        alpha_by_relu_input=tuple(
            (name, value.to("cuda")) for name, value in terminal.alpha_by_relu_input
        ),
        beta_by_relu_input=tuple(
            (name, value.to("cuda")) for name, value in terminal.beta_by_relu_input
        ),
    )


def _plan_and_targets(fixture):  # type: ignore[no-untyped-def]
    mutable = tuple(
        tensor
        for tensor in fixture.pre.tensors
        if tensor.ownership == ProductionTensorOwnership.MUTABLE_COPY_OUT
    )
    targets = {
        tensor.semantic_path: tensor.value.to("cuda").clone() for tensor in mutable
    }
    plan = compile_device_atomic_commit_plan_v1(
        plan_id="unit-resnet2b-device-commit",
        prepared_template_hash="a" * 64,
        paths=tuple(
            DeviceAtomicPathSpecV1(
                semantic_path=tensor.semantic_path,
                role=tensor.role,
                shape=tuple(tensor.value.shape),
                dtype=str(tensor.value.dtype),
                device="cuda:0",
                alias_group=f"target:{ordinal:02d}",
                rollback_ordinal=ordinal,
            )
            for ordinal, tensor in enumerate(mutable)
        ),
    )
    return plan, targets


def _transaction():  # type: ignore[no-untyped-def]
    fixture = _stage()
    terminal = _terminal_cuda(fixture.terminal)
    plan, targets = _plan_and_targets(fixture)
    host, host_candidate = _live_host_packets(fixture)
    lower = fixture.native.steps[-1].lower.to("cuda")
    transaction = stage_device_atomic_transaction_v1(
        plan=plan,
        core_instance_hash="b" * 64,
        pre_snapshot_hash=fixture.pre.stable_hash(),
        pre_snapshot=fixture.pre,
        live_targets=targets,
        terminal_state=terminal,
        topology=TOPOLOGY,
        terminal_lower=lower,
        host_packet=host,
        host_packet_candidate=host_candidate,
    )
    return fixture, plan, targets, host, host_candidate, transaction


def test_device_candidates_commit_and_audit_outside_headline() -> None:
    fixture, plan, targets, host, _host_candidate, transaction = _transaction()

    assert len(transaction.candidates) == 12
    assert all(
        value.device.type == "cuda" for value in transaction.candidate_values.values()
    )
    receipt = commit_device_atomic_transaction_v1(
        transaction, live_targets=targets, host_packet=host
    )
    torch.cuda.synchronize()
    audit = audit_device_atomic_transaction_v1(
        transaction,
        receipt=receipt,
        pre_snapshot=fixture.pre,
        live_targets=targets,
    )

    assert receipt["plan_hash"] == plan.stable_hash()
    assert receipt["candidate_d2h_copy_count"] == 0
    assert receipt["committed_path_count"] == 12
    assert receipt["device_rollback_backup_count"] == 12
    assert receipt["content_audit_pending"] is True
    assert audit["content_audit_complete"] is True
    assert audit["headline_timing_excluded"] is True
    assert audit["commit_hash"] == receipt["commit_hash"]
    assert len(audit["path_digests"]) == 12


def test_discarded_provider_host_object_is_versioned_by_inventory_only() -> None:
    fixture = _stage()
    terminal = _terminal_cuda(fixture.terminal)
    plan, targets = _plan_and_targets(fixture)
    host, host_candidate = _live_host_packets(fixture)
    host["provider_branching_object"] = object()

    transaction = stage_device_atomic_transaction_v1(
        plan=plan,
        core_instance_hash="b" * 64,
        pre_snapshot_hash=fixture.pre.stable_hash(),
        pre_snapshot=fixture.pre,
        live_targets=targets,
        terminal_state=terminal,
        topology=TOPOLOGY,
        terminal_lower=fixture.native.steps[-1].lower.to("cuda"),
        host_packet=host,
        host_packet_candidate=host_candidate,
    )
    receipt = commit_device_atomic_transaction_v1(
        transaction, live_targets=targets, host_packet=host
    )

    assert receipt["committed_path_count"] == 12
    assert "provider_branching_object" not in host


def test_empty_beta_targets_remain_distinct_and_alias_tamper_rejects() -> None:
    fixture = _stage()
    terminal = _terminal_cuda(fixture.terminal)
    plan, targets = _plan_and_targets(fixture)
    empty = [path for path, value in targets.items() if value.numel() == 0]
    assert len(empty) == 5
    assert all(targets[path].data_ptr() == 0 for path in empty)
    targets[empty[1]] = targets[empty[0]]
    host, host_candidate = _live_host_packets(fixture)

    with pytest.raises(ValueError, match="alias contract"):
        stage_device_atomic_transaction_v1(
            plan=plan,
            core_instance_hash="b" * 64,
            pre_snapshot_hash=fixture.pre.stable_hash(),
            pre_snapshot=fixture.pre,
            live_targets=targets,
            terminal_state=terminal,
            topology=TOPOLOGY,
            terminal_lower=fixture.native.steps[-1].lower.to("cuda"),
            host_packet=host,
            host_packet_candidate=host_candidate,
        )


def test_stale_tensor_version_rejects_before_any_commit() -> None:
    _fixture, _plan, targets, host, _host_candidate, transaction = _transaction()
    path = transaction.plan.rollback_order[0]
    with torch.no_grad():
        targets[path].add_(0.25)
    before = {name: value.clone() for name, value in targets.items()}

    with pytest.raises(ValueError, match="version is stale"):
        commit_device_atomic_transaction_v1(
            transaction, live_targets=targets, host_packet=host
        )
    for name, value in targets.items():
        torch.testing.assert_close(value, before[name])


def test_nan_terminal_state_rejects_before_staging() -> None:
    fixture = _stage()
    terminal = _terminal_cuda(fixture.terminal)
    alpha = dict(terminal.alpha_by_relu_input)
    first = sorted(alpha)[0]
    alpha[first] = alpha[first].clone()
    alpha[first].reshape(-1)[0] = float("nan")
    terminal = replace(terminal, alpha_by_relu_input=tuple(sorted(alpha.items())))
    plan, targets = _plan_and_targets(fixture)
    before = {name: value.clone() for name, value in targets.items()}
    host, host_candidate = _live_host_packets(fixture)

    with pytest.raises(ValueError, match="native alpha/beta tensor contract"):
        stage_device_atomic_transaction_v1(
            plan=plan,
            core_instance_hash="b" * 64,
            pre_snapshot_hash=fixture.pre.stable_hash(),
            pre_snapshot=fixture.pre,
            live_targets=targets,
            terminal_state=terminal,
            topology=TOPOLOGY,
            terminal_lower=fixture.native.steps[-1].lower.to("cuda"),
            host_packet=host,
            host_packet_candidate=host_candidate,
        )
    for name, value in targets.items():
        torch.testing.assert_close(value, before[name])


def test_mid_commit_failure_restores_all_device_tensors(
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    _fixture, _plan, targets, host, _host_candidate, transaction = _transaction()
    before = {name: value.clone() for name, value in targets.items()}
    host_before = dict(host)
    original = device_atomic._copy_device_value
    calls = 0

    def fail_fifth(target: torch.Tensor, source: torch.Tensor) -> None:
        nonlocal calls
        calls += 1
        if calls == 5:
            raise RuntimeError("injected device copy failure")
        original(target, source)

    monkeypatch.setattr(device_atomic, "_copy_device_value", fail_fifth)
    with pytest.raises(RuntimeError, match="injected device copy failure"):
        commit_device_atomic_transaction_v1(
            transaction, live_targets=targets, host_packet=host
        )
    assert set(host) == set(host_before)
    for name, value in targets.items():
        torch.testing.assert_close(value, before[name])


def test_host_failure_restores_all_device_tensors_and_host(
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    _fixture, _plan, targets, host, _host_candidate, transaction = _transaction()
    before = {name: value.clone() for name, value in targets.items()}
    host_before = dict(host)
    original = device_atomic._replace_host_packet
    calls = 0

    def fail_first(target, source) -> None:  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        if calls == 1:
            target.clear()
            raise RuntimeError("injected host commit failure")
        original(target, source)

    monkeypatch.setattr(device_atomic, "_replace_host_packet", fail_first)
    with pytest.raises(RuntimeError, match="injected host commit failure"):
        commit_device_atomic_transaction_v1(
            transaction, live_targets=targets, host_packet=host
        )
    assert set(host) == set(host_before)
    assert host["depths"] == host_before["depths"]
    assert host["history"] == host_before["history"]
    for name, value in targets.items():
        torch.testing.assert_close(value, before[name])


def test_outer_resigned_receipt_tamper_rejects_post_query_audit() -> None:
    fixture, _plan, targets, host, _host_candidate, transaction = _transaction()
    receipt = commit_device_atomic_transaction_v1(
        transaction, live_targets=targets, host_packet=host
    )
    tampered = dict(receipt)
    tampered["candidate_d2h_copy_count"] = 1
    payload = dict(tampered)
    payload.pop("commit_hash")
    tampered["commit_hash"] = device_atomic._canonical_hash(payload)

    with pytest.raises(ValueError, match="audit receipt binding"):
        audit_device_atomic_transaction_v1(
            transaction,
            receipt=tampered,
            pre_snapshot=fixture.pre,
            live_targets=targets,
        )
