"""Provider return assembly for the FSG4/B3-C device transaction."""

# pylint: disable=too-many-arguments,too-many-locals,too-many-statements
# pylint: disable=too-many-boolean-expressions,missing-function-docstring
# pylint: disable=too-many-instance-attributes,protected-access

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
from typing import cast, MutableMapping

import torch

from .fsg4_b3_device_atomic_commit import (
    commit_device_atomic_transaction_v1,
    DeviceAtomicTransactionV1,
    validate_device_atomic_targets_v1,
)
from .rvir_v4_live_return import (
    live_targets_from_pre_result_v4,
    load_provider_return_factories_v4,
    ProviderReturnFactoriesV4,
)
from .rvir_v4_native_backward_export import NativeBackwardExportV4
from .rvir_v4_native_kfsb import NativeKfsbEvaluationV4
from .rvir_v4_pre_state_initializer import ProductionReluTopologyV4

FSG4_B3_DEVICE_LIVE_RETURN_SCHEMA = "boundflow.fsg4-b3-device-live-return/v1"


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _encode(value: str) -> str:
    return value.replace("%", "%25").replace("/", "%2F")


def _raw_data(value: object, label: str) -> MutableMapping[str, object]:
    raw = getattr(value, "_data", None)
    if not isinstance(raw, MutableMapping):
        raise TypeError(f"FSG4/B3-C live {label} provider data differs")
    return raw


def _working_alpha(
    transaction: DeviceAtomicTransactionV1,
    topology: tuple[ProductionReluTopologyV4, ...],
    *,
    factories: ProviderReturnFactoriesV4,
) -> object:
    candidate = transaction.candidate_values
    data: dict[str, dict[str, torch.Tensor]] = {}
    for link in topology:
        path = (
            f"alpha/{_encode(link.provider_activation)}/"
            f"{_encode(link.provider_start_node)}"
        )
        data[link.provider_activation] = {link.provider_start_node: candidate[path]}
    return factories.alpha_value_data(_data=data)


def _working_beta(
    pre_result: object,
    transaction: DeviceAtomicTransactionV1,
    topology: tuple[ProductionReluTopologyV4, ...],
) -> object:
    working = copy.deepcopy(getattr(pre_result, "betas_by_layer"))
    raw = _raw_data(working, "working beta")
    candidate = transaction.candidate_values
    for link in topology:
        path = f"beta/{_encode(link.provider_preactivation)}/0/value"
        sparse_betas = raw.get(link.provider_preactivation)
        sparse = (
            sparse_betas[0]
            if isinstance(sparse_betas, list) and len(sparse_betas) == 1
            else None
        )
        if sparse is None or not torch.is_tensor(getattr(sparse, "val", None)):
            raise TypeError("FSG4/B3-C live working beta differs")
        sparse.val = candidate[path]
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


@dataclass(frozen=True)
class DeviceLiveCoreReturnAssemblyV1:
    """Hash-free headline assembly bound to one device transaction."""

    core_result: object
    transaction: DeviceAtomicTransactionV1
    live_target_by_path: tuple[tuple[str, torch.Tensor], ...]
    final_decision: tuple[tuple[int, int], ...]
    provider_core_callback_count: int = 0
    provider_compute_bounds_callback_count: int = 0
    provider_update_bounds_callback_count: int = 0
    fallback_dispatch_count: int = 0
    schema_version: str = FSG4_B3_DEVICE_LIVE_RETURN_SCHEMA

    @property
    def live_targets(self) -> dict[str, torch.Tensor]:
        return dict(self.live_target_by_path)

    def validate(self) -> None:
        self.transaction.validate()
        validate_device_atomic_targets_v1(self.transaction.plan, self.live_targets)
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
        lower = getattr(result, "lb", None)
        if not torch.is_tensor(lower):
            raise ValueError("FSG4/B3-C live return lower differs")
        lower_tensor = cast(torch.Tensor, lower)
        if (
            self.schema_version != FSG4_B3_DEVICE_LIVE_RETURN_SCHEMA
            or len(self.live_targets) != len(self.live_target_by_path)
            or len(self.live_targets) != 12
            or any(not hasattr(result, name) for name in required)
            or tuple(lower_tensor.shape) != (6, 1)
            or lower_tensor.device != self.transaction.terminal_lower.device
            or lower_tensor.dtype != self.transaction.terminal_lower.dtype
            or not torch.equal(lower_tensor, self.transaction.terminal_lower)
            or not bool(torch.isinf(getattr(result, "ub")).all().item())
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
            raise ValueError("FSG4/B3-C live return assembly differs")

    def metadata(self) -> dict[str, object]:
        self.validate()
        result = self.core_result
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "plan_hash": self.transaction.plan.stable_hash(),
            "transaction_version": self.transaction.transaction_version,
            "live_target_paths": sorted(self.live_targets),
            "final_lower_metadata": {
                "shape": list(getattr(result, "lb").shape),
                "dtype": str(getattr(result, "lb").dtype),
                "device": str(getattr(result, "lb").device),
            },
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
            "candidate_device_resident": True,
            "headline_content_digest_count": 0,
            "live_return_assembly_admitted": True,
            "five_fresh_correctness_admitted": False,
            "b2_same_solver_timing_admitted": False,
            "performance_claimed": False,
        }
        payload["assembly_hash"] = _canonical_hash(payload)
        return payload


def assemble_device_live_core_return_v1(
    *,
    pre_result: object,
    transaction: DeviceAtomicTransactionV1,
    backward_export: NativeBackwardExportV4,
    kfsb_evaluation: NativeKfsbEvaluationV4,
    topology: tuple[ProductionReluTopologyV4, ...],
    factories: ProviderReturnFactoriesV4 | None = None,
) -> DeviceLiveCoreReturnAssemblyV1:
    """Build the provider packet without a GPU content hash or CPU candidate."""

    transaction.validate()
    backward_export.validate()
    kfsb_evaluation.validate()
    provider = factories or load_provider_return_factories_v4()
    if len(topology) != 6 or not torch.equal(
        transaction.terminal_lower, backward_export.lower
    ):
        raise ValueError("FSG4/B3-C live return source differs")
    d = getattr(pre_result, "d_dict", None)
    c = getattr(pre_result, "c", None)
    lb_last = getattr(pre_result, "lb_last", None)
    ub_last = getattr(pre_result, "ub_last", None)
    if (
        not torch.is_tensor(c)
        or not torch.is_tensor(lb_last)
        or not torch.is_tensor(ub_last)
    ):
        raise ValueError("FSG4/B3-C live return tensor pre-result differs")
    c_tensor = cast(torch.Tensor, c)
    lb_last_tensor = cast(torch.Tensor, lb_last)
    ub_last_tensor = cast(torch.Tensor, ub_last)
    if (
        not isinstance(d, MutableMapping)
        or tuple(c_tensor.shape) != (6, 1, 10)
        or tuple(lb_last_tensor.shape) != (6, 1)
        or tuple(ub_last_tensor.shape) != (6, 1)
        or getattr(pre_result, "x_Ls", None) is not None
        or getattr(pre_result, "x_Us", None) is not None
    ):
        raise ValueError("FSG4/B3-C live return pre-result differs")
    thresholds = d.get("thresholds")
    history = d.get("history")
    depths_raw = d.get("depths")
    if (
        not torch.is_tensor(thresholds)
        or not isinstance(history, list)
        or len(history) != 6
        or not isinstance(depths_raw, list)
        or len(depths_raw) != 6
    ):
        raise ValueError("FSG4/B3-C live return host packet differs")
    thresholds_tensor = cast(torch.Tensor, thresholds)
    if tuple(thresholds_tensor.shape) != (6, 1):
        raise ValueError("FSG4/B3-C live return threshold shape differs")
    lb = backward_export.lower.to(device=c_tensor.device, dtype=c_tensor.dtype)
    if not bool(torch.all(lb <= thresholds_tensor.to(device=lb.device)).item()):
        raise ValueError("FSG4/B3-C live return requires all domains unverified")
    ub = torch.full_like(lb, fill_value=torch.inf)
    depths = torch.as_tensor(depths_raw, dtype=torch.int32)
    host_candidate = transaction.host_candidate_map
    host_thresholds = host_candidate.get("thresholds")
    if not torch.is_tensor(host_thresholds):
        raise ValueError("FSG4/B3-C live return host threshold differs")
    host_threshold_tensor = cast(torch.Tensor, host_thresholds)
    if (
        host_candidate.get("history") is not history
        or host_candidate.get("depths") != depths.tolist()
        or not torch.equal(host_threshold_tensor, thresholds_tensor)
    ):
        raise ValueError("FSG4/B3-C live return host candidate differs")
    decision = provider.branching_decisions(
        branching_decision=[list(value) for value in kfsb_evaluation.final_decision],
        branching_points=None,
        split_depth=1,
        batch_size=6,
    )
    core_result = provider.update_bound_core_return(
        lb=lb,
        ub=ub,
        lb_last=lb_last_tensor,
        ub_last=ub_last_tensor,
        nums_effective_beta_per_domain=getattr(
            pre_result, "nums_effective_beta_per_domain"
        ),
        input_split_idx=None,
        primal_x=None,
        x_Ls=None,
        x_Us=None,
        new_x_Ls=getattr(pre_result, "new_x_Ls"),
        new_x_Us=getattr(pre_result, "new_x_Us"),
        c=c_tensor,
        working_beta=_working_beta(pre_result, transaction, topology),
        working_alpha=_working_alpha(transaction, topology, factories=provider),
        working_interm_bounds=_working_intermediates(
            backward_export, provider, device=lb.device, dtype=lb.dtype
        ),
        batched_lA=provider.batched_l_a({}, is_emptied=True),
        branching_decision=decision,
        sub_domain_clip_decisions=provider.sub_domain_clip_decisions({}, 0, 0, 0),
        decision_thresh=thresholds_tensor,
        lb_final_max=float(lb.max().item()),
        lb_final_min=float(lb.min().item()),
        n_verified=0,
        n_splits=6,
        new_split_history=[{} for _ in range(6)],
        history=history,
        depths=depths,
        thresholds=thresholds_tensor,
    )
    assembly = DeviceLiveCoreReturnAssemblyV1(
        core_result=core_result,
        transaction=transaction,
        live_target_by_path=tuple(
            sorted(live_targets_from_pre_result_v4(pre_result, topology).items())
        ),
        final_decision=kfsb_evaluation.final_decision,
    )
    assembly.validate()
    return assembly


def commit_device_live_core_return_v1(
    assembly: DeviceLiveCoreReturnAssemblyV1,
    *,
    host_packet: MutableMapping[str, object],
) -> tuple[object, dict[str, object]]:
    """Commit the device transaction only after complete provider assembly."""

    assembly.validate()
    receipt = commit_device_atomic_transaction_v1(
        assembly.transaction,
        live_targets=assembly.live_targets,
        host_packet=host_packet,
    )
    receipt.pop("commit_hash")
    receipt.update(
        {
            "assembly_hash": assembly.metadata()["assembly_hash"],
            "provider_core_callback_count": 0,
            "provider_compute_bounds_callback_count": 0,
            "provider_update_bounds_callback_count": 0,
            "fallback_dispatch_count": 0,
            "performance_claimed": False,
        }
    )
    receipt["commit_hash"] = _canonical_hash(receipt)
    return assembly.core_result, receipt


__all__ = [
    "assemble_device_live_core_return_v1",
    "commit_device_live_core_return_v1",
    "DeviceLiveCoreReturnAssemblyV1",
    "FSG4_B3_DEVICE_LIVE_RETURN_SCHEMA",
]
