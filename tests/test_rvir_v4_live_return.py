"""Provider-neutral contracts for RVIR-v4 V4-3D live return assembly."""

# pylint: disable=missing-function-docstring,too-many-locals,protected-access

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from boundflow.runtime.rvir_v4_atomic_copy_out import (
    stage_rvir_v4_live_atomic_copy_out,
)
from boundflow.runtime.rvir_v4_live_return import (
    assemble_rvir_v4_live_core_return,
    commit_rvir_v4_live_core_return,
    ProviderReturnFactoriesV4,
)
from boundflow.runtime.rvir_v4_production_state import (
    ProductionTensorOwnership,
    ProductionTensorRole,
)
from scripts import run_rvir_v4_native_backward_export_artifact as backward_runner
from scripts import run_rvir_v4_native_kfsb_artifact as kfsb_runner
from tests.test_rvir_v4_atomic_copy_out import _stage

BACKWARD_ARTIFACT = Path(
    "artifacts/rvir-v4-native-backward-export/resnet2b-core-v1/export.pt"
)
KFSB_ARTIFACT = Path("artifacts/rvir-v4-native-kfsb/resnet2b-core-v1/evaluation.pt")
TRUTH = Path("artifacts/rvir-v4-whole-core-truth/resnet2b-core-v1/truth.pt")


@dataclass
class _Data:
    _data: dict[str, Any]


@dataclass
class _Interm:
    lower_bound: torch.Tensor
    upper_bound: torch.Tensor


class _WorkingInterm(dict):
    pass


@dataclass
class _Branching:
    branching_decision: list[list[int]]
    branching_points: torch.Tensor | None
    split_depth: int
    batch_size: int


@dataclass
class _BatchedLA:
    _data: dict[str, torch.Tensor]
    is_emptied: bool


@dataclass
class _Clip:
    _data: dict[str, torch.Tensor]
    split_depth: int
    batch_size: int
    topk_objective: int


def _factories() -> ProviderReturnFactoriesV4:
    return ProviderReturnFactoriesV4(
        update_bound_core_return=SimpleNamespace,
        alpha_value_data=_Data,
        working_interm_bounds_info=_WorkingInterm,
        interm_bounds_info=_Interm,
        batched_l_a=_BatchedLA,
        branching_decisions=_Branching,
        sub_domain_clip_decisions=_Clip,
    )


def _encode(value: str) -> str:
    return value.replace("%", "%25").replace("/", "%2F")


def _one_role(fixture, role: ProductionTensorRole) -> torch.Tensor:  # type: ignore[no-untyped-def]
    values = [tensor.value for tensor in fixture.pre.tensors if tensor.role == role]
    assert len(values) == 1
    return values[0]


def _pre_result(fixture, truth: dict[str, Any]) -> SimpleNamespace:  # type: ignore[no-untyped-def]
    pre_map = fixture.pre.tensor_map()
    alphas: dict[str, dict[str, torch.Tensor]] = {}
    betas: dict[str, list[SimpleNamespace]] = {}
    for link in kfsb_runner._topology():
        alpha_path = (
            f"alpha/{_encode(link.provider_activation)}/"
            f"{_encode(link.provider_start_node)}"
        )
        beta_path = f"beta/{_encode(link.provider_preactivation)}/0/value"
        alphas[link.provider_activation] = {
            link.provider_start_node: pre_map[alpha_path].value.clone()
        }
        betas[link.provider_preactivation] = [
            SimpleNamespace(val=pre_map[beta_path].value.clone())
        ]
    fields = truth["whole_core_truths"][0]["fields"]
    history = [[[], [], [], [], []] for _ in range(6)]
    thresholds = _one_role(fixture, ProductionTensorRole.DECISION_THRESHOLD).clone()
    d_dict: dict[str, object] = {
        "history": history,
        "depths": [1, 1, 1, 1, 1, 1],
        "thresholds": thresholds,
        "discard_after_core": torch.arange(6),
    }
    return SimpleNamespace(
        alphas_by_layer=_Data(alphas),
        betas_by_layer=_Data(betas),
        nums_effective_beta_per_domain=[{"/input-28": 1} for _ in range(6)],
        d_dict=d_dict,
        c=_one_role(fixture, ProductionTensorRole.LINEAR_SPEC).clone(),
        lb_last=fields["lb_last"]["value"].clone(),
        ub_last=fields["ub_last"]["value"].clone(),
        new_x_Ls=None,
        new_x_Us=None,
        x_Ls=None,
        x_Us=None,
    )


def test_live_return_assembles_and_commits_without_truth_input() -> None:
    fixture = _stage()
    truth = kfsb_runner._load_torch(TRUTH)
    backward = backward_runner._export_from_payload(
        backward_runner._load_torch(BACKWARD_ARTIFACT)
    )
    kfsb = kfsb_runner._evaluation_from_payload(kfsb_runner._load_torch(KFSB_ARTIFACT))
    pre_result = _pre_result(fixture, truth)
    host_candidate = {
        "history": pre_result.d_dict["history"],
        "depths": list(pre_result.d_dict["depths"]),
        "thresholds": pre_result.d_dict["thresholds"],
    }
    staged = stage_rvir_v4_live_atomic_copy_out(
        pre=fixture.pre,
        terminal_state=fixture.terminal,
        topology=kfsb_runner._topology(),
        terminal_lower=backward.lower,
        host_packet=pre_result.d_dict,
        host_packet_candidate=host_candidate,
        candidate_snapshot_id="core:000000:live-return",
    )

    assembly = assemble_rvir_v4_live_core_return(
        pre_result=pre_result,
        pre_snapshot=fixture.pre,
        staged_copy_out=staged,
        backward_export=backward,
        kfsb_evaluation=kfsb,
        topology=kfsb_runner._topology(),
        factories=_factories(),
    )
    core_result, receipt = commit_rvir_v4_live_core_return(
        assembly,
        pre_snapshot=fixture.pre,
        host_packet=pre_result.d_dict,
    )

    assert receipt["atomic_live_and_host_commit"] is True
    assert receipt["committed_path_count"] == 12
    assert receipt["changed_path_count"] == 7
    assert receipt["provider_core_callback_count"] == 0
    assert receipt["provider_compute_bounds_callback_count"] == 0
    assert receipt["provider_update_bounds_callback_count"] == 0
    assert receipt["fallback_dispatch_count"] == 0
    assert set(pre_result.d_dict) == {"depths", "history", "thresholds"}
    assert core_result.branching_decision.branching_decision == [
        [5, 27],
        [5, 32],
        [5, 90],
        [5, 90],
        [5, 32],
        [5, 90],
    ]
    torch.testing.assert_close(core_result.lb, backward.lower)
    assert bool(torch.isinf(core_result.ub).all())
    assert core_result.batched_lA.is_emptied is True
    assert assembly.metadata()["live_return_assembly_admitted"] is True
    assert assembly.metadata()["five_fresh_correctness_admitted"] is False
    assert assembly.metadata()["performance_claimed"] is False


def test_live_return_rejects_verified_domain_before_commit() -> None:
    fixture = _stage()
    truth = kfsb_runner._load_torch(TRUTH)
    backward = backward_runner._export_from_payload(
        backward_runner._load_torch(BACKWARD_ARTIFACT)
    )
    kfsb = kfsb_runner._evaluation_from_payload(kfsb_runner._load_torch(KFSB_ARTIFACT))
    pre_result = _pre_result(fixture, truth)
    pre_result.d_dict["thresholds"] = backward.lower.clone()
    pre_result.d_dict["thresholds"][0, 0] = backward.lower[0, 0] - 1.0
    host_candidate = {
        "history": pre_result.d_dict["history"],
        "depths": list(pre_result.d_dict["depths"]),
        "thresholds": pre_result.d_dict["thresholds"],
    }
    staged = stage_rvir_v4_live_atomic_copy_out(
        pre=fixture.pre,
        terminal_state=fixture.terminal,
        topology=kfsb_runner._topology(),
        terminal_lower=backward.lower,
        host_packet=pre_result.d_dict,
        host_packet_candidate=host_candidate,
        candidate_snapshot_id="core:000000:live-return",
    )
    before = {
        path: value.clone()
        for path, value in (
            (path, tensor.value)
            for path, tensor in fixture.pre.tensor_map().items()
            if tensor.ownership == ProductionTensorOwnership.MUTABLE_COPY_OUT
        )
    }

    with pytest.raises(ValueError, match="requires all six domains unverified"):
        assemble_rvir_v4_live_core_return(
            pre_result=pre_result,
            pre_snapshot=fixture.pre,
            staged_copy_out=staged,
            backward_export=backward,
            kfsb_evaluation=kfsb,
            topology=kfsb_runner._topology(),
            factories=_factories(),
        )
    for path, value in before.items():
        torch.testing.assert_close(fixture.pre.tensor_map()[path].value, value)
