"""Sparse-Patches start adapter for the compiled dual-lane CROWN chain."""

# mypy: disable-error-code=import-untyped
# pylint: disable=too-many-instance-attributes,too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=missing-function-docstring

from __future__ import annotations

from dataclasses import dataclass

import torch

from boundflow.backends.tvm.root_crown_input_domain import (
    RootCrownInputDomainTemplateV1,
)
from boundflow.backends.tvm.root_crown_projection import RootCrownProjectionTemplateV1
from boundflow.backends.tvm.root_crown_residual import RootCrownResidualTemplateV1
from boundflow.backends.tvm.root_crown_sparse_patches_seed import (
    RootCrownSparsePatchesSeedTemplateV1,
)
from boundflow.runtime.root_crown_intermediate_dual_lane_tir import (
    RootCrownIntermediateDualLaneTensorsV1,
    RootCrownIntermediateDualLaneTIRExecutorV1,
)
from boundflow.runtime.root_crown_sparse_patches_seed_tir import (
    RootCrownSparsePatchesSeedTIRExecutorV1,
)


@dataclass(frozen=True)
class RootCrownSparsePatchesDualLaneTensorsV1:
    """Dynamic state downstream of the sparse `/44` start carrier."""

    patches: torch.Tensor
    unstable_idx: tuple[torch.Tensor, ...]
    residual_main_weight: torch.Tensor
    residual_main_bias: torch.Tensor
    residual_inner_lower: torch.Tensor
    residual_inner_upper: torch.Tensor
    residual_inner_alpha: torch.Tensor
    residual_inner_weight: torch.Tensor
    residual_inner_bias: torch.Tensor
    projection_entry_lower: torch.Tensor
    projection_entry_upper: torch.Tensor
    projection_entry_alpha: torch.Tensor
    projection_outer_weight: torch.Tensor
    projection_outer_bias: torch.Tensor
    projection_inner_lower: torch.Tensor
    projection_inner_upper: torch.Tensor
    projection_inner_alpha: torch.Tensor
    projection_inner_weight: torch.Tensor
    projection_inner_bias: torch.Tensor
    projection_skip_weight: torch.Tensor
    projection_skip_bias: torch.Tensor
    input_lower: torch.Tensor
    input_upper: torch.Tensor
    input_alpha: torch.Tensor
    input_weight: torch.Tensor
    input_bias: torch.Tensor
    input_center: torch.Tensor
    input_radius: torch.Tensor


class RootCrownSparsePatchesDualLaneTIRExecutorV1:
    """Lower sparse identity Patches, then run both CROWN polarities."""

    def __init__(
        self,
        residual_template: RootCrownResidualTemplateV1,
        projection_template: RootCrownProjectionTemplateV1,
        input_template: RootCrownInputDomainTemplateV1,
    ) -> None:
        self.crown = RootCrownIntermediateDualLaneTIRExecutorV1(
            residual_template, projection_template, input_template
        )
        self.seed = RootCrownSparsePatchesSeedTIRExecutorV1(
            RootCrownSparsePatchesSeedTemplateV1(
                spec_count=residual_template.spec_count,
                domain_count=residual_template.domain_count,
                channels=residual_template.channels,
                height=residual_template.height,
                width=residual_template.width,
                compute_capability=residual_template.compute_capability,
                thread_extent=residual_template.thread_extent,
            )
        )
        self.residual_template = residual_template
        self.projection_template = projection_template
        self.input_template = input_template
        self._zero_bias: torch.Tensor | None = None
        self._identity_lower: torch.Tensor | None = None
        self._identity_upper: torch.Tensor | None = None
        self._unused_entry_alpha: torch.Tensor | None = None
        self.prepare_count = 0
        self.call_count = 0
        self.performance_claimed = False

    def prepare(self) -> None:
        if self.prepare_count:
            raise RuntimeError(
                "root sparse Patches dual-lane executor already prepared"
            )
        self.seed.prepare()
        self.crown.prepare()
        template = self.residual_template
        device = torch.device("cuda")
        self._zero_bias = torch.zeros(
            (template.spec_count, template.domain_count),
            dtype=torch.float32,
            device=device,
        )
        # The existing residual primitive begins with a ReLU.  `/44` starts
        # immediately after that ReLU, so a stable nonnegative interval makes
        # this first transform the exact identity with zero intercept.
        self._identity_lower = torch.zeros(
            template.bound_shape, dtype=torch.float32, device=device
        )
        self._identity_upper = torch.zeros_like(self._identity_lower)
        self._unused_entry_alpha = torch.zeros(
            (
                2,
                template.spec_count,
                template.domain_count,
                template.entry_alpha_count,
            ),
            dtype=torch.float32,
            device=device,
        )
        self.prepare_count = 1

    def execute(
        self, tensors: RootCrownSparsePatchesDualLaneTensorsV1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self.prepare_count != 1
            or self._zero_bias is None
            or self._identity_lower is None
            or self._identity_upper is None
            or self._unused_entry_alpha is None
        ):
            raise RuntimeError("root sparse Patches dual-lane executor not prepared")
        dense_seed = self.seed.execute(tensors.patches, tensors.unstable_idx)
        candidate = self.crown.execute(
            RootCrownIntermediateDualLaneTensorsV1(
                dense_seed,
                dense_seed,
                self._zero_bias,
                self._zero_bias,
                self._identity_lower,
                self._identity_upper,
                self._unused_entry_alpha,
                tensors.residual_main_weight,
                tensors.residual_main_bias,
                tensors.residual_inner_lower,
                tensors.residual_inner_upper,
                tensors.residual_inner_alpha,
                tensors.residual_inner_weight,
                tensors.residual_inner_bias,
                tensors.projection_entry_lower,
                tensors.projection_entry_upper,
                tensors.projection_entry_alpha,
                tensors.projection_outer_weight,
                tensors.projection_outer_bias,
                tensors.projection_inner_lower,
                tensors.projection_inner_upper,
                tensors.projection_inner_alpha,
                tensors.projection_inner_weight,
                tensors.projection_inner_bias,
                tensors.projection_skip_weight,
                tensors.projection_skip_bias,
                tensors.input_lower,
                tensors.input_upper,
                tensors.input_alpha,
                tensors.input_weight,
                tensors.input_bias,
                tensors.input_center,
                tensors.input_radius,
            )
        )
        self.call_count += 1
        return candidate

    def receipt(self) -> dict[str, object]:
        return {
            "schema_version": "boundflow.root-sparse-patches-dual-lane-tir/v1",
            "call_count": self.call_count,
            "spec_count": self.residual_template.spec_count,
            "sparse_seed": self.seed.receipt(),
            "crown": self.crown.receipt(),
            "entry_relu_policy": "stable-zero-identity-after-start",
            "single_rematerializing_owner": False,
            "performance_claimed": False,
        }


__all__ = [
    "RootCrownSparsePatchesDualLaneTensorsV1",
    "RootCrownSparsePatchesDualLaneTIRExecutorV1",
]
