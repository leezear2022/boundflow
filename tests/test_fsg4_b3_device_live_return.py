"""Provider packet integration for the FSG4/B3-C device transaction."""

# pylint: disable=missing-function-docstring,protected-access,too-many-locals
# pylint: disable=duplicate-code

from dataclasses import replace

import pytest
import torch

from boundflow.runtime.fsg4_b3_device_atomic_commit import (
    audit_device_atomic_transaction_v1,
    stage_device_atomic_transaction_v1,
)
from boundflow.runtime.fsg4_b3_device_live_return import (
    assemble_device_live_core_return_v1,
    commit_device_live_core_return_v1,
)
from boundflow.runtime.rvir_v4_live_return import live_targets_from_pre_result_v4
from scripts import run_rvir_v4_native_backward_export_artifact as backward_runner
from scripts import run_rvir_v4_native_kfsb_artifact as kfsb_runner
from tests.test_fsg4_b3_device_atomic_commit import (
    _plan_and_targets,
    _terminal_cuda,
)
from tests.test_rvir_v4_atomic_copy_out import _stage
from tests.test_rvir_v4_live_return import (
    BACKWARD_ARTIFACT,
    KFSB_ARTIFACT,
    TRUTH,
    _factories,
    _pre_result,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _move_pre_result_to_cuda(pre_result) -> None:  # type: ignore[no-untyped-def]
    for values in pre_result.alphas_by_layer._data.values():
        for name, value in tuple(values.items()):
            values[name] = value.to("cuda")
    for values in pre_result.betas_by_layer._data.values():
        for sparse in values:
            sparse.val = sparse.val.to("cuda")
    pre_result.c = pre_result.c.to("cuda")
    pre_result.lb_last = pre_result.lb_last.to("cuda")
    pre_result.ub_last = pre_result.ub_last.to("cuda")
    pre_result.d_dict["thresholds"] = pre_result.d_dict["thresholds"].to("cuda")


def _integrated_transaction():  # type: ignore[no-untyped-def]
    fixture = _stage()
    truth = kfsb_runner._load_torch(TRUTH)
    backward = backward_runner._export_from_payload(
        backward_runner._load_torch(BACKWARD_ARTIFACT)
    )
    backward = replace(backward, lower=backward.lower.to("cuda"))
    kfsb = kfsb_runner._evaluation_from_payload(kfsb_runner._load_torch(KFSB_ARTIFACT))
    pre_result = _pre_result(fixture, truth)
    _move_pre_result_to_cuda(pre_result)
    plan, _unused = _plan_and_targets(fixture)
    targets = live_targets_from_pre_result_v4(pre_result, kfsb_runner._topology())
    host_candidate = {
        "history": pre_result.d_dict["history"],
        "depths": list(pre_result.d_dict["depths"]),
        "thresholds": pre_result.d_dict["thresholds"],
    }
    transaction = stage_device_atomic_transaction_v1(
        plan=plan,
        core_instance_hash="b" * 64,
        pre_snapshot_hash=fixture.pre.stable_hash(),
        pre_snapshot=fixture.pre,
        live_targets=targets,
        terminal_state=_terminal_cuda(fixture.terminal),
        topology=kfsb_runner._topology(),
        terminal_lower=backward.lower,
        host_packet=pre_result.d_dict,
        host_packet_candidate=host_candidate,
    )
    return fixture, pre_result, backward, kfsb, transaction, targets


def test_device_live_return_commits_without_headline_content_digest() -> None:
    fixture, pre_result, backward, kfsb, transaction, targets = (
        _integrated_transaction()
    )
    assembly = assemble_device_live_core_return_v1(
        pre_result=pre_result,
        transaction=transaction,
        backward_export=backward,
        kfsb_evaluation=kfsb,
        topology=kfsb_runner._topology(),
        factories=_factories(),
    )
    metadata = assembly.metadata()
    core_result, receipt = commit_device_live_core_return_v1(
        assembly, host_packet=pre_result.d_dict
    )

    assert metadata["headline_content_digest_count"] == 0
    assert metadata["candidate_device_resident"] is True
    assert receipt["candidate_d2h_copy_count"] == 0
    assert receipt["committed_path_count"] == 12
    assert set(pre_result.d_dict) == {"depths", "history", "thresholds"}
    assert all(value.device.type == "cuda" for value in targets.values())
    assert all(
        value.device.type == "cuda"
        for values in core_result.working_alpha._data.values()
        for value in values.values()
    )
    torch.cuda.synchronize()
    audit = audit_device_atomic_transaction_v1(
        transaction,
        receipt=receipt,
        pre_snapshot=fixture.pre,
        live_targets=targets,
    )
    assert audit["commit_hash"] == receipt["commit_hash"]
    assert audit["headline_timing_excluded"] is True


def test_device_live_return_rejects_terminal_lower_identity_drift() -> None:
    _fixture, pre_result, backward, kfsb, transaction, _targets = (
        _integrated_transaction()
    )
    drifted = replace(backward, lower=backward.lower + 0.25)

    with pytest.raises(ValueError, match="live return source differs"):
        assemble_device_live_core_return_v1(
            pre_result=pre_result,
            transaction=transaction,
            backward_export=drifted,
            kfsb_evaluation=kfsb,
            topology=kfsb_runner._topology(),
            factories=_factories(),
        )
