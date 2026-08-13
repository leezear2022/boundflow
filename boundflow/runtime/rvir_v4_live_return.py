"""Transactional RVIR-v4 live provider return assembly."""

# pylint: disable=too-many-arguments,too-many-locals,too-many-statements
# pylint: disable=too-many-boolean-expressions,missing-function-docstring
# pylint: disable=too-many-instance-attributes
# pylint: disable=import-outside-toplevel

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
from typing import Callable, Mapping, MutableMapping

import torch

from .rvir_v4_atomic_copy_out import (
    commit_rvir_v4_live_atomic_copy_out,
    ProductionLiveAtomicCopyOutV4,
)
from .rvir_v4_native_backward_export import NativeBackwardExportV4
from .rvir_v4_native_kfsb import NativeKfsbEvaluationV4
from .rvir_v4_pre_state_initializer import ProductionReluTopologyV4
from .rvir_v4_production_state import (
    production_tensor_sha256,
    ProductionStateSnapshotV4,
)

RVIR_V4_LIVE_RETURN_SCHEMA = "boundflow.rvir-v4-live-return/v1"


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _encode(value: str) -> str:
    return value.replace("%", "%25").replace("/", "%2F")


@dataclass(frozen=True)
class ProviderReturnFactoriesV4:
    """Late-bound αβ-CROWN constructors used only inside its live environment."""

    update_bound_core_return: Callable[..., object]
    alpha_value_data: Callable[..., object]
    working_interm_bounds_info: Callable[..., object]
    interm_bounds_info: Callable[..., object]
    batched_l_a: Callable[..., object]
    branching_decisions: Callable[..., object]
    sub_domain_clip_decisions: Callable[..., object]


def load_provider_return_factories_v4() -> ProviderReturnFactoriesV4:
    """Load pinned provider types without making BoundFlow import depend on them."""

    try:
        from activation_split.return_types import (  # type: ignore[import-not-found]
            UpdateBoundCoreReturn,
        )
        from domain_clipper import (  # type: ignore[import-not-found]
            SubDomainClipDecisions,
        )
        from heuristics.decision_types import (  # type: ignore[import-not-found]
            BranchingDecisions,
        )
        from state.alpha import AlphaValueData  # type: ignore[import-not-found]
        from state.intermediate_bounds import (  # type: ignore[import-not-found]
            IntermBoundsInfo,
            WorkingIntermBoundsInfo,
        )
        from state.lA import BatchedlA  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError(
            "RVIR-v4 live return requires the pinned alpha-beta-CROWN environment"
        ) from error
    return ProviderReturnFactoriesV4(
        update_bound_core_return=UpdateBoundCoreReturn,
        alpha_value_data=AlphaValueData,
        working_interm_bounds_info=WorkingIntermBoundsInfo,
        interm_bounds_info=IntermBoundsInfo,
        batched_l_a=BatchedlA,
        branching_decisions=BranchingDecisions,
        sub_domain_clip_decisions=SubDomainClipDecisions,
    )


@dataclass(frozen=True)
class LiveCoreReturnAssemblyV4:
    """Fully staged provider return plus exact live mutation targets."""

    core_result: object
    staged_copy_out: ProductionLiveAtomicCopyOutV4
    live_target_by_path: tuple[tuple[str, torch.Tensor], ...]
    final_lower_sha256: str
    final_decision: tuple[tuple[int, int], ...]
    provider_core_callback_count: int = 0
    provider_compute_bounds_callback_count: int = 0
    provider_update_bounds_callback_count: int = 0
    fallback_dispatch_count: int = 0
    schema_version: str = RVIR_V4_LIVE_RETURN_SCHEMA

    @property
    def live_targets(self) -> dict[str, torch.Tensor]:
        return dict(self.live_target_by_path)

    def validate(self) -> None:
        self.staged_copy_out.validate()
        targets = self.live_targets
        result = self.core_result
        required = (
            "lb",
            "ub",
            "working_alpha",
            "working_beta",
            "working_interm_bounds",
            "batched_lA",
            "branching_decision",
            "sub_domain_clip_decisions",
            "n_verified",
            "n_splits",
            "history",
            "depths",
            "thresholds",
        )
        if (
            self.schema_version != RVIR_V4_LIVE_RETURN_SCHEMA
            or len(targets) != len(self.live_target_by_path)
            or len(targets) != 12
            or set(targets)
            != {receipt.semantic_path for receipt in self.staged_copy_out.path_receipts}
            or any(not hasattr(result, name) for name in required)
            or tuple(getattr(result, "lb").shape) != (6, 1)
            or production_tensor_sha256(getattr(result, "lb"))
            != self.final_lower_sha256
            or not bool(torch.isinf(getattr(result, "ub")).all())
            or int(getattr(result, "n_verified")) != 0
            or int(getattr(result, "n_splits")) != 6
            or tuple(self.final_decision)
            != tuple(
                (int(layer), int(neuron))
                for layer, neuron in getattr(
                    getattr(result, "branching_decision"), "branching_decision"
                )
            )
            or int(getattr(getattr(result, "branching_decision"), "split_depth")) != 1
            or int(getattr(getattr(result, "branching_decision"), "batch_size")) != 6
            or self.provider_core_callback_count != 0
            or self.provider_compute_bounds_callback_count != 0
            or self.provider_update_bounds_callback_count != 0
            or self.fallback_dispatch_count != 0
        ):
            raise ValueError("RVIR-v4 live core return assembly differs")
        pre_map = self.staged_copy_out.candidate_snapshot.tensor_map()
        for path, live in targets.items():
            if (
                not torch.is_tensor(live)
                or live.shape != pre_map[path].value.shape
                or live.dtype != pre_map[path].value.dtype
            ):
                raise ValueError("RVIR-v4 live core return target differs")

    def metadata(self) -> dict[str, object]:
        self.validate()
        result = self.core_result
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "live_copy_out_hash": self.staged_copy_out.metadata()["live_copy_out_hash"],
            "live_target_paths": sorted(self.live_targets),
            "final_lower_sha256": self.final_lower_sha256,
            "final_decision": [list(decision) for decision in self.final_decision],
            "lb_final_max": float(getattr(result, "lb_final_max")),
            "lb_final_min": float(getattr(result, "lb_final_min")),
            "n_verified": int(getattr(result, "n_verified")),
            "n_splits": int(getattr(result, "n_splits")),
            "provider_core_callback_count": self.provider_core_callback_count,
            "provider_compute_bounds_callback_count": (
                self.provider_compute_bounds_callback_count
            ),
            "provider_update_bounds_callback_count": (
                self.provider_update_bounds_callback_count
            ),
            "fallback_dispatch_count": self.fallback_dispatch_count,
            "live_return_assembly_admitted": True,
            "five_fresh_correctness_admitted": False,
            "b2_same_solver_timing_admitted": False,
            "performance_claimed": False,
        }
        payload["assembly_hash"] = _canonical_hash(payload)
        return payload


def _raw_data(value: object, label: str) -> MutableMapping[str, object]:
    raw = getattr(value, "_data", None)
    if not isinstance(raw, MutableMapping):
        raise TypeError(f"RVIR-v4 live {label} provider data differs")
    return raw


def live_targets_from_pre_result_v4(
    pre_result: object,
    topology: tuple[ProductionReluTopologyV4, ...],
) -> dict[str, torch.Tensor]:
    """Resolve the twelve actual provider-owned α and SparseBeta tensors."""

    alpha_data = _raw_data(getattr(pre_result, "alphas_by_layer"), "alpha")
    beta_data = _raw_data(getattr(pre_result, "betas_by_layer"), "beta")
    targets: dict[str, torch.Tensor] = {}
    for link in topology:
        alpha_path = (
            f"alpha/{_encode(link.provider_activation)}/"
            f"{_encode(link.provider_start_node)}"
        )
        beta_path = f"beta/{_encode(link.provider_preactivation)}/0/value"
        activation = alpha_data.get(link.provider_activation)
        sparse_betas = beta_data.get(link.provider_preactivation)
        if not isinstance(activation, Mapping) or not isinstance(sparse_betas, list):
            raise ValueError("RVIR-v4 live provider state topology differs")
        alpha = activation.get(link.provider_start_node)
        sparse = sparse_betas[0] if len(sparse_betas) == 1 else None
        beta = getattr(sparse, "val", None)
        if not torch.is_tensor(alpha) or not torch.is_tensor(beta):
            raise TypeError("RVIR-v4 live provider tensor target differs")
        targets[alpha_path] = alpha
        targets[beta_path] = beta
    if len(targets) != 12:
        raise ValueError("RVIR-v4 live provider target inventory differs")
    return targets


def _working_alpha(
    staged: ProductionLiveAtomicCopyOutV4,
    topology: tuple[ProductionReluTopologyV4, ...],
    *,
    device: torch.device,
    factories: ProviderReturnFactoriesV4,
) -> object:
    candidate = staged.candidate_snapshot.tensor_map()
    data: dict[str, dict[str, torch.Tensor]] = {}
    for link in topology:
        path = (
            f"alpha/{_encode(link.provider_activation)}/"
            f"{_encode(link.provider_start_node)}"
        )
        data[link.provider_activation] = {
            link.provider_start_node: candidate[path].value.to(device=device)
        }
    return factories.alpha_value_data(_data=data)


def _working_beta(
    pre_result: object,
    staged: ProductionLiveAtomicCopyOutV4,
    topology: tuple[ProductionReluTopologyV4, ...],
) -> object:
    working = copy.deepcopy(getattr(pre_result, "betas_by_layer"))
    raw = _raw_data(working, "working beta")
    candidate = staged.candidate_snapshot.tensor_map()
    for link in topology:
        path = f"beta/{_encode(link.provider_preactivation)}/0/value"
        sparse_betas = raw.get(link.provider_preactivation)
        sparse = (
            sparse_betas[0]
            if isinstance(sparse_betas, list) and len(sparse_betas) == 1
            else None
        )
        if sparse is None:
            raise TypeError("RVIR-v4 live working beta object differs")
        live = getattr(sparse, "val", None)
        if not torch.is_tensor(live):
            raise TypeError("RVIR-v4 live working beta tensor differs")
        sparse.val = candidate[path].value.to(device=live.device, dtype=live.dtype)
    return working


def _working_intermediates(
    export: NativeBackwardExportV4,
    factories: ProviderReturnFactoriesV4,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> object:
    return factories.working_interm_bounds_info(
        {
            name: factories.interm_bounds_info(
                lower_bound=interval.lower.to(device=device, dtype=dtype),
                upper_bound=interval.upper.to(device=device, dtype=dtype),
            )
            for name, interval in export.intermediate_by_provider_preactivation
        }
    )


def assemble_rvir_v4_live_core_return(
    *,
    pre_result: object,
    pre_snapshot: ProductionStateSnapshotV4,
    staged_copy_out: ProductionLiveAtomicCopyOutV4,
    backward_export: NativeBackwardExportV4,
    kfsb_evaluation: NativeKfsbEvaluationV4,
    topology: tuple[ProductionReluTopologyV4, ...],
    factories: ProviderReturnFactoriesV4 | None = None,
) -> LiveCoreReturnAssemblyV4:
    """Assemble the exact fixed production provider packet without a provider bound call."""

    pre_snapshot.validate()
    staged_copy_out.validate()
    backward_export.validate()
    kfsb_evaluation.validate()
    provider = factories or load_provider_return_factories_v4()
    if (
        staged_copy_out.pre_snapshot_hash != pre_snapshot.stable_hash()
        or len(topology) != 6
        or staged_copy_out.terminal_lower_sha256
        != production_tensor_sha256(backward_export.lower)
    ):
        raise ValueError("RVIR-v4 live return source identity differs")
    d = getattr(pre_result, "d_dict", None)
    c = getattr(pre_result, "c", None)
    lb_last = getattr(pre_result, "lb_last", None)
    ub_last = getattr(pre_result, "ub_last", None)
    if (
        not isinstance(d, MutableMapping)
        or not torch.is_tensor(c)
        or not torch.is_tensor(lb_last)
        or not torch.is_tensor(ub_last)
        or tuple(c.shape) != (6, 1, 10)
        or tuple(lb_last.shape) != (6, 1)
        or tuple(ub_last.shape) != (6, 1)
        or getattr(pre_result, "x_Ls", None) is not None
        or getattr(pre_result, "x_Us", None) is not None
    ):
        raise ValueError("RVIR-v4 live return pre-result schema differs")
    thresholds = d.get("thresholds")
    history = d.get("history")
    depths_raw = d.get("depths")
    if (
        not torch.is_tensor(thresholds)
        or tuple(thresholds.shape) != (6, 1)
        or not isinstance(history, list)
        or len(history) != 6
        or not isinstance(depths_raw, list)
        or len(depths_raw) != 6
    ):
        raise ValueError("RVIR-v4 live return host packet schema differs")
    lb = backward_export.lower.to(device=c.device, dtype=c.dtype)
    if not bool(torch.all(lb <= thresholds.to(device=lb.device))):
        raise ValueError("RVIR-v4 live return v1 requires all six domains unverified")
    ub = torch.full_like(lb, fill_value=torch.inf)
    depths = torch.as_tensor(depths_raw, dtype=torch.int32)
    host_candidate = dict(staged_copy_out.host_packet_candidate)
    host_thresholds = host_candidate.get("thresholds")
    if (
        host_candidate.get("history") is not history
        or host_candidate.get("depths") != depths.tolist()
        or not torch.is_tensor(host_thresholds)
        or not torch.equal(host_thresholds, thresholds)
    ):
        raise ValueError("RVIR-v4 live return host candidate differs")
    decision = provider.branching_decisions(
        branching_decision=[list(value) for value in kfsb_evaluation.final_decision],
        branching_points=None,
        split_depth=1,
        batch_size=6,
    )
    core_result = provider.update_bound_core_return(
        lb=lb,
        ub=ub,
        lb_last=lb_last,
        ub_last=ub_last,
        nums_effective_beta_per_domain=getattr(
            pre_result, "nums_effective_beta_per_domain"
        ),
        input_split_idx=None,
        primal_x=None,
        x_Ls=None,
        x_Us=None,
        new_x_Ls=getattr(pre_result, "new_x_Ls"),
        new_x_Us=getattr(pre_result, "new_x_Us"),
        c=c,
        working_beta=_working_beta(pre_result, staged_copy_out, topology),
        working_alpha=_working_alpha(
            staged_copy_out,
            topology,
            device=lb.device,
            factories=provider,
        ),
        working_interm_bounds=_working_intermediates(
            backward_export,
            provider,
            device=lb.device,
            dtype=lb.dtype,
        ),
        batched_lA=provider.batched_l_a({}, is_emptied=True),
        branching_decision=decision,
        sub_domain_clip_decisions=provider.sub_domain_clip_decisions({}, 0, 0, 0),
        decision_thresh=thresholds,
        lb_final_max=float(lb.max().item()),
        lb_final_min=float(lb.min().item()),
        n_verified=0,
        n_splits=6,
        new_split_history=[{} for _ in range(6)],
        history=history,
        depths=depths,
        thresholds=thresholds,
    )
    assembly = LiveCoreReturnAssemblyV4(
        core_result=core_result,
        staged_copy_out=staged_copy_out,
        live_target_by_path=tuple(
            sorted(live_targets_from_pre_result_v4(pre_result, topology).items())
        ),
        final_lower_sha256=production_tensor_sha256(lb),
        final_decision=kfsb_evaluation.final_decision,
    )
    assembly.validate()
    return assembly


def commit_rvir_v4_live_core_return(
    assembly: LiveCoreReturnAssemblyV4,
    *,
    pre_snapshot: ProductionStateSnapshotV4,
    host_packet: MutableMapping[str, object],
) -> tuple[object, dict[str, object]]:
    """Commit live state and host packet only after complete provider assembly."""

    assembly.validate()
    receipt = commit_rvir_v4_live_atomic_copy_out(
        assembly.staged_copy_out,
        pre=pre_snapshot,
        live_targets=assembly.live_targets,
        host_packet=host_packet,
    )
    result = {
        **receipt,
        "assembly_hash": assembly.metadata()["assembly_hash"],
        "provider_core_callback_count": 0,
        "provider_compute_bounds_callback_count": 0,
        "provider_update_bounds_callback_count": 0,
        "fallback_dispatch_count": 0,
        "performance_claimed": False,
    }
    result["live_return_commit_hash"] = _canonical_hash(result)
    return assembly.core_result, result


__all__ = [
    "assemble_rvir_v4_live_core_return",
    "commit_rvir_v4_live_core_return",
    "LiveCoreReturnAssemblyV4",
    "live_targets_from_pre_result_v4",
    "load_provider_return_factories_v4",
    "ProviderReturnFactoriesV4",
]
