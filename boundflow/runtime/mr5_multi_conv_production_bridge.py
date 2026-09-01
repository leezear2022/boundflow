"""Fail-closed three-site production bridge for MR5 correctness."""

# pylint: disable=too-many-instance-attributes,too-many-arguments
# pylint: disable=too-many-locals,too-many-boolean-expressions
# pylint: disable=missing-function-docstring,import-error

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from boundflow.backends.tvm.mr5_generalized_crown_conv import (
    MR5GeneralizedConvSignatureV1,
)
from boundflow.runtime.mr5_generalized_crown_conv import (
    MR5GeneralizedConvModuleCacheV1,
    MR5GeneralizedConvTensorsV1,
    execute_mr5_generalized_conv_v1,
)

MR5_TARGET_START = "/49"
MR5_SITE_ORDER = ("C2", "C1", "C0")
MR5_SITE_NODES = {
    "C0": ("/input-4", "/input"),
    "C1": ("/input-12", "/input-8"),
    "C2": ("/input-24", "/input-20"),
}


def mr5_frozen_signatures(
    compute_capability: str,
) -> dict[str, MR5GeneralizedConvSignatureV1]:
    """Build the only three signatures admitted by the MR4 census."""

    return {
        "C0": MR5GeneralizedConvSignatureV1(
            site_id="C0",
            input_channels=3,
            output_channels=8,
            input_height=32,
            input_width=32,
            output_height=16,
            output_width=16,
            stride=(2, 2),
            padding=(1, 1),
            output_padding=(1, 1),
            compute_capability=compute_capability,
        ),
        "C1": MR5GeneralizedConvSignatureV1(
            site_id="C1",
            input_channels=8,
            output_channels=16,
            input_height=16,
            input_width=16,
            output_height=8,
            output_width=8,
            stride=(2, 2),
            padding=(1, 1),
            output_padding=(1, 1),
            compute_capability=compute_capability,
        ),
        "C2": MR5GeneralizedConvSignatureV1(
            site_id="C2",
            input_channels=16,
            output_channels=16,
            input_height=8,
            input_width=8,
            output_height=8,
            output_width=8,
            stride=(1, 1),
            padding=(1, 1),
            output_padding=(0, 0),
            compute_capability=compute_capability,
        ),
    }


@dataclass(frozen=True)
class MR5MultiConvBridgeReceiptV1:
    """Three-site lifecycle and compiler identity for one outer exact call."""

    evaluation_count: int
    site_order_count: int
    forward_launches: tuple[tuple[str, int], ...]
    backward_launches: tuple[tuple[str, int], ...]
    beta_tensor_count: tuple[tuple[str, int], ...]
    beta_numel: tuple[tuple[str, int], ...]
    handoff_content_count: tuple[tuple[str, int], ...]
    handoff_pointer_count: tuple[tuple[str, int], ...]
    cache_miss_count: tuple[tuple[str, int], ...]
    cache_hit_count: tuple[tuple[str, int], ...]
    signature_hashes: tuple[tuple[str, str], ...]
    module_receipts: tuple[tuple[str, tuple[tuple[str, object], ...]], ...]
    pending_site_count: int
    fallback_count: int
    eager_count: int
    native_shadow_count: int
    timing_recorded: bool
    performance_claimed: bool

    def validate(self) -> None:
        forward = dict(self.forward_launches)
        backward = dict(self.backward_launches)
        beta_count = dict(self.beta_tensor_count)
        beta_numel = dict(self.beta_numel)
        handoff = dict(self.handoff_content_count)
        pointers = dict(self.handoff_pointer_count)
        misses = dict(self.cache_miss_count)
        hits = dict(self.cache_hit_count)
        hashes = dict(self.signature_hashes)
        receipt_sites = {site for site, _receipt in self.module_receipts}
        if (
            self.evaluation_count != 10
            or self.site_order_count != 30
            or any(forward.get(site) != 10 for site in MR5_SITE_ORDER)
            or any(backward.get(site) != 9 for site in MR5_SITE_ORDER)
            or any(beta_count.get(site) != 10 for site in MR5_SITE_ORDER)
            or any(beta_numel.get(site) != 0 for site in MR5_SITE_ORDER)
            or any(handoff.get(site) != 10 for site in MR5_SITE_ORDER)
            or any(pointers.get(site) not in range(11) for site in MR5_SITE_ORDER)
            or any(misses.get(site) != 1 for site in MR5_SITE_ORDER)
            or any(hits.get(site) != 9 for site in MR5_SITE_ORDER)
            or any(len(hashes.get(site, "")) != 64 for site in MR5_SITE_ORDER)
            or receipt_sites != set(MR5_SITE_ORDER)
            or len(self.module_receipts) != 3
            or self.pending_site_count != 0
            or self.fallback_count != 0
            or self.eager_count != 0
            or self.native_shadow_count != 0
            or self.timing_recorded
            or self.performance_claimed
        ):
            raise ValueError("MR5 multi-Conv bridge receipt differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "evaluation_count": self.evaluation_count,
            "site_order_count": self.site_order_count,
            "forward_launches": dict(self.forward_launches),
            "backward_launches": dict(self.backward_launches),
            "beta_tensor_count": dict(self.beta_tensor_count),
            "beta_numel": dict(self.beta_numel),
            "handoff_content_count": dict(self.handoff_content_count),
            "handoff_pointer_count": dict(self.handoff_pointer_count),
            "cache_miss_count": dict(self.cache_miss_count),
            "cache_hit_count": dict(self.cache_hit_count),
            "signature_hashes": dict(self.signature_hashes),
            "module_receipts": {
                site: dict(receipt) for site, receipt in self.module_receipts
            },
            "pending_site_count": self.pending_site_count,
            "fallback_count": self.fallback_count,
            "eager_count": self.eager_count,
            "native_shadow_count": self.native_shadow_count,
            "timing_recorded": self.timing_recorded,
            "performance_claimed": self.performance_claimed,
        }


def _content_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return tuple(left.shape) == tuple(right.shape) and torch.equal(left, right)


class MR5MultiConvProductionBridgeV1:
    """Replace C2→C1→C0 lower paths while provider owns optimizer and state."""

    def __init__(self, *, cache: MR5GeneralizedConvModuleCacheV1 | None = None) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("MR5 multi-Conv bridge requires CUDA")
        major, minor = torch.cuda.get_device_capability()
        self.signatures = mr5_frozen_signatures(f"sm_{major}{minor}")
        for signature in self.signatures.values():
            signature.validate()
        self.cache = cache or MR5GeneralizedConvModuleCacheV1()
        self.current_evaluation: int | None = None
        self.expected_site_ordinal = 0
        self.pending: dict[str, dict[str, torch.Tensor]] = {}
        self.evaluation_count = 0
        self.site_order_count = 0
        self.forward = {site: 0 for site in MR5_SITE_ORDER}
        self.backward = {site: 0 for site in MR5_SITE_ORDER}
        self.beta_count = {site: 0 for site in MR5_SITE_ORDER}
        self.beta_numel = {site: 0 for site in MR5_SITE_ORDER}
        self.handoff_content = {site: 0 for site in MR5_SITE_ORDER}
        self.handoff_pointer = {site: 0 for site in MR5_SITE_ORDER}
        self.cache_miss = {site: 0 for site in MR5_SITE_ORDER}
        self.cache_hit = {site: 0 for site in MR5_SITE_ORDER}
        self.module_receipts: dict[str, dict[str, object]] = {}
        self.fallback_count = 0
        self.eager_count = 0
        self.native_shadow_count = 0

    def _reject(self, reason: str) -> None:
        self.fallback_count += 1
        raise ValueError(reason)

    def begin_evaluation(self, ordinal: int) -> None:
        if (
            self.current_evaluation is not None
            or self.pending
            or ordinal != self.evaluation_count
            or ordinal not in range(10)
        ):
            self._reject("MR5 multi-Conv evaluation order differs")
        self.current_evaluation = ordinal
        self.expected_site_ordinal = 0

    def _expect_site(self, site: str) -> None:
        if (
            self.current_evaluation is None
            or self.expected_site_ordinal >= len(MR5_SITE_ORDER)
            or MR5_SITE_ORDER[self.expected_site_ordinal] != site
        ):
            self._reject("MR5 multi-Conv site order differs")

    def route_relu(
        self,
        site: str,
        relu: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
        *,
        beta_tensors: tuple[torch.Tensor, ...],
    ) -> Any:
        self._expect_site(site)
        signature = self.signatures[site]
        incoming = args[0] if args else None
        preactivation = args[2] if len(args) >= 3 else None
        alpha = getattr(relu, "alpha", {}).get(MR5_TARGET_START)
        indices = getattr(relu, "alpha_indices", None)
        provider_shape = (
            signature.spec_count,
            signature.domain_count,
            signature.output_channels,
            signature.output_height,
            signature.output_width,
        )
        if (
            str(getattr(kwargs.get("start_node"), "name", "")) != MR5_TARGET_START
            or not torch.is_tensor(incoming)
            or tuple(incoming.shape) != provider_shape
            or len(args) < 3
            or args[1] is not None
            or preactivation is None
            or tuple(preactivation.lower.shape) != signature.relaxation_shape
            or tuple(preactivation.upper.shape) != signature.relaxation_shape
            or not torch.is_tensor(alpha)
            or not alpha.requires_grad
            or not isinstance(indices, list)
            or len(indices) != 3
            or not all(torch.is_tensor(item) for item in indices)
            or len(beta_tensors) != 1
            or tuple(beta_tensors[0].shape) != (6, 0)
            or beta_tensors[0].numel() != 0
            or site in self.pending
        ):
            self._reject(f"MR5 {site} ReLU admission differs")
        assert torch.is_tensor(incoming)
        assert torch.is_tensor(alpha)
        assert preactivation is not None
        full_alpha = relu.reconstruct_full_alpha(alpha[0], provider_shape, indices)[
            0
        ].contiguous()
        expects_gradient = (
            self.current_evaluation is not None and self.current_evaluation < 9
        )
        if (
            tuple(full_alpha.shape) != signature.relaxation_shape
            or torch.is_grad_enabled() != expects_gradient
            or full_alpha.requires_grad != expects_gradient
        ):
            self._reject(f"MR5 {site} alpha reconstruction differs")
        handoff = incoming.detach()
        self.pending[site] = {
            "incoming": incoming.permute(1, 0, 2, 3, 4).contiguous(),
            "lower": preactivation.lower.contiguous(),
            "upper": preactivation.upper.contiguous(),
            "alpha": full_alpha,
            "handoff": handoff,
        }
        self.beta_count[site] += 1
        self.beta_numel[site] += int(beta_tensors[0].numel())
        bridge_bias = torch.zeros(
            (signature.spec_count, signature.domain_count),
            dtype=torch.float32,
            device=incoming.device,
        )
        return [(handoff, None)], bridge_bias, 0

    def _backward_completed(self, site: str) -> None:
        if site not in self.backward:
            self._reject("MR5 multi-Conv backward site differs")
        self.backward[site] += 1

    def route_conv(self, site: str, args: tuple[Any, ...]) -> Any:
        self._expect_site(site)
        if len(args) < 5 or site not in self.pending:
            self._reject(f"MR5 {site} Conv admission differs")
        signature = self.signatures[site]
        pending = self.pending.pop(site)
        provider_input = args[0]
        weight = args[3].lower
        operator_bias = args[4].lower
        if (
            not torch.is_tensor(provider_input)
            or not _content_equal(provider_input, pending["handoff"])
            or tuple(weight.shape) != signature.weight_shape
            or tuple(operator_bias.shape) != (signature.output_channels,)
        ):
            self._reject(f"MR5 {site} ReLU-to-Conv handoff differs")
        self.handoff_content[site] += 1
        self.handoff_pointer[site] += int(
            provider_input.data_ptr() == pending["handoff"].data_ptr()
        )
        incoming_bias = torch.zeros(
            (signature.domain_count, signature.spec_count),
            dtype=torch.float32,
            device=provider_input.device,
        )
        tensors = MR5GeneralizedConvTensorsV1(
            incoming=pending["incoming"],
            lower=pending["lower"],
            upper=pending["upper"],
            alpha=pending["alpha"],
            incoming_bias=incoming_bias,
            weight=weight.contiguous(),
            operator_bias=operator_bias.contiguous(),
        )
        result_a, result_bias, executor = execute_mr5_generalized_conv_v1(
            signature,
            tensors,
            self.cache,
            backward_observer=self._backward_completed,
        )
        self.forward[site] += executor.forward_launch_count
        self.cache_miss[site] += int(executor.cache_event == "miss")
        self.cache_hit[site] += int(executor.cache_event == "hit")
        receipt = executor.module_receipt.to_dict()
        existing = self.module_receipts.get(site)
        if existing is not None and existing != receipt:
            self._reject(f"MR5 {site} module receipt drifted")
        self.module_receipts[site] = receipt
        routed_a = result_a.permute(1, 0, 2, 3, 4).contiguous()
        routed_bias = result_bias.transpose(0, 1).contiguous()
        if not bool(torch.isfinite(routed_a).all()) or not bool(
            torch.isfinite(routed_bias).all()
        ):
            self._reject(f"MR5 {site} candidate output is nonfinite")
        self.site_order_count += 1
        self.expected_site_ordinal += 1
        if self.expected_site_ordinal == len(MR5_SITE_ORDER):
            if self.pending:
                self._reject("MR5 multi-Conv pending site leaked")
            self.evaluation_count += 1
            self.current_evaluation = None
        return (
            [(routed_a, None), (None, None), (None, None)],
            routed_bias,
            0,
        )

    def receipt(self) -> MR5MultiConvBridgeReceiptV1:
        receipt = MR5MultiConvBridgeReceiptV1(
            evaluation_count=self.evaluation_count,
            site_order_count=self.site_order_count,
            forward_launches=tuple(sorted(self.forward.items())),
            backward_launches=tuple(sorted(self.backward.items())),
            beta_tensor_count=tuple(sorted(self.beta_count.items())),
            beta_numel=tuple(sorted(self.beta_numel.items())),
            handoff_content_count=tuple(sorted(self.handoff_content.items())),
            handoff_pointer_count=tuple(sorted(self.handoff_pointer.items())),
            cache_miss_count=tuple(sorted(self.cache_miss.items())),
            cache_hit_count=tuple(sorted(self.cache_hit.items())),
            signature_hashes=tuple(
                sorted(
                    (site, signature.stable_hash())
                    for site, signature in self.signatures.items()
                )
            ),
            module_receipts=tuple(
                sorted(
                    (site, tuple(sorted(values.items())))
                    for site, values in self.module_receipts.items()
                )
            ),
            pending_site_count=len(self.pending),
            fallback_count=self.fallback_count,
            eager_count=self.eager_count,
            native_shadow_count=self.native_shadow_count,
            timing_recorded=False,
            performance_claimed=False,
        )
        receipt.validate()
        return receipt

    def timing_receipt(self) -> dict[str, object]:
        """Return the prewarmed 30/27 lifecycle without correctness cache policy."""

        if (
            self.evaluation_count != 10
            or self.site_order_count != 30
            or any(self.forward[site] != 10 for site in MR5_SITE_ORDER)
            or any(self.backward[site] != 9 for site in MR5_SITE_ORDER)
            or any(self.beta_count[site] != 10 for site in MR5_SITE_ORDER)
            or any(self.beta_numel[site] != 0 for site in MR5_SITE_ORDER)
            or any(self.handoff_content[site] != 10 for site in MR5_SITE_ORDER)
            or any(self.cache_miss[site] != 0 for site in MR5_SITE_ORDER)
            or any(self.cache_hit[site] != 10 for site in MR5_SITE_ORDER)
            or set(self.module_receipts) != set(MR5_SITE_ORDER)
            or self.pending
            or self.fallback_count
            or self.eager_count
            or self.native_shadow_count
        ):
            raise ValueError("MR5 prewarmed timing receipt differs")
        return {
            "evaluation_count": self.evaluation_count,
            "site_order_count": self.site_order_count,
            "forward_launches": dict(self.forward),
            "backward_launches": dict(self.backward),
            "beta_tensor_count": dict(self.beta_count),
            "beta_numel": dict(self.beta_numel),
            "handoff_content_count": dict(self.handoff_content),
            "handoff_pointer_count": dict(self.handoff_pointer),
            "cache_miss_count": dict(self.cache_miss),
            "cache_hit_count": dict(self.cache_hit),
            "signature_hashes": {
                site: signature.stable_hash()
                for site, signature in self.signatures.items()
            },
            "module_receipts": self.module_receipts,
            "pending_site_count": len(self.pending),
            "fallback_count": self.fallback_count,
            "eager_count": self.eager_count,
            "native_shadow_count": self.native_shadow_count,
            "prewarmed_before_outer": True,
            "timing_recorded": True,
            "performance_claimed": False,
        }


__all__ = [
    "MR5MultiConvBridgeReceiptV1",
    "MR5MultiConvProductionBridgeV1",
    "MR5_SITE_NODES",
    "MR5_SITE_ORDER",
    "MR5_TARGET_START",
    "mr5_frozen_signatures",
]
