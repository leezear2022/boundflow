"""Prepared four-segment TIR optimizer for one production RVIR exact call."""

# mypy: disable-error-code=import-untyped
# pylint: disable=import-error,protected-access,too-many-instance-attributes
# pylint: disable=too-many-locals,too-many-statements,missing-function-docstring
# pylint: disable=too-many-boolean-expressions,duplicate-code

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping

import torch

from boundflow.backends.tvm.bab_input_domain import BabInputDomainTemplateV1
from boundflow.backends.tvm.bab_terminal_linear import BabTerminalLinearTemplateV1
from boundflow.backends.tvm.root_crown_projection import RootCrownProjectionTemplateV1
from boundflow.backends.tvm.root_crown_residual import RootCrownResidualTemplateV1
from boundflow.runtime.asplos27_s4_gradient_emitters import S4_BETA_SIGN_V1
from boundflow.runtime.bab_full_region_owner import (
    BabFullRegionDynamicV1,
    BabFullRegionStaticV1,
    BabInputStaticV1,
    BabProjectionStaticV1,
    BabResidualStaticV1,
    PreparedBabFullRegionOwnerV1,
)
from boundflow.runtime.bab_input_domain_tir import BabInputDomainTIRExecutorV1
from boundflow.runtime.bab_terminal_tir import BabTerminalTIRExecutorV1
from boundflow.runtime.root_crown_projection_tir import (
    RootCrownProjectionTIRExecutorV1,
)
from boundflow.runtime.root_crown_residual_tir import RootCrownResidualTIRExecutorV1


def _coordinates(
    flat_indices: tuple[int, ...], shape: tuple[int, ...]
) -> tuple[tuple[int, int, int], ...]:
    if len(shape) != 3:
        raise ValueError("activation-BaB live convolution shape differs")
    channels, height, width = shape
    result = tuple(
        (index // (height * width), (index // width) % height, index % width)
        for index in flat_indices
    )
    if any(channel >= channels for channel, _row, _column in result):
        raise ValueError("activation-BaB live coordinate differs")
    return result


@dataclass(frozen=True)
class BabFourSegmentRunResultV1:
    """Terminal compact state, lower adjoints, and honest launch inventory."""

    terminal_lower: torch.Tensor
    terminal_parameters: tuple[torch.Tensor, ...]
    terminal_las: tuple[torch.Tensor, ...]
    learning_rates: tuple[tuple[float, float], ...]
    evaluation_count: int
    mutation_count: int
    compiled_segment_count: int
    compiled_forward_launch_count: int
    compiled_backward_launch_count: int
    fallback_count: int
    performance_claimed: bool = False

    def validate(self) -> None:
        if (
            tuple(self.terminal_lower.shape) != (6, 1)
            or len(self.terminal_parameters) != 7
            or len(self.terminal_las) != 6
            or len(self.learning_rates) != 10
            or self.evaluation_count != 10
            or self.mutation_count != 9
            or self.compiled_segment_count != 4
            or self.compiled_forward_launch_count != 76
            or self.compiled_backward_launch_count != 36
            or self.fallback_count
            or self.performance_claimed
            or not all(
                bool(torch.isfinite(value).all().item())
                for value in (
                    self.terminal_lower,
                    *self.terminal_parameters,
                    *self.terminal_las,
                )
            )
        ):
            raise ValueError("activation-BaB four-segment run differs")


class PreparedBabFourSegmentOptimizerV1:
    """Compile once, rebind one exact call, then execute the frozen 10/9 policy."""

    def __init__(self, region: Any) -> None:
        self.region = region
        plan = region.plan
        layouts = {layout.native_preactivation: layout for layout in plan.relu_layouts}
        if tuple(layouts) != ("17", "19", "23", "25", "28", "31"):
            raise ValueError("activation-BaB live topology differs")
        tensor = region.executor._tensor
        capability = region.assets.compute_capability
        layout17 = layouts["17"]
        layout19 = layouts["19"]
        layout23 = layouts["23"]
        layout25 = layouts["25"]
        layout28 = layouts["28"]
        layout31 = layouts["31"]
        self._layouts = layouts
        self._center = torch.empty_like(tensor("input/lower"))
        self._radius = torch.empty_like(tensor("input/lower"))
        self._terminal_incoming = torch.empty(
            (1, plan.domain_count, 100), device=self._center.device, dtype=torch.float32
        )
        self._initial_bias = torch.empty(
            (plan.domain_count, 1), device=self._center.device, dtype=torch.float32
        )
        self._alphas = {
            name: torch.empty_like(tensor(f"relu/{name}/alpha"), requires_grad=True)
            for name in layouts
        }
        self._beta = torch.empty(
            (plan.domain_count, 1),
            device=self._center.device,
            dtype=torch.float32,
            requires_grad=True,
        )
        terminal_coordinates = tuple(
            int(value) for value in layout31.alpha_flat_indices
        )
        c17 = _coordinates(layout17.alpha_flat_indices, layout17.feature_shape)
        c19 = _coordinates(layout19.alpha_flat_indices, layout19.feature_shape)
        c23 = _coordinates(layout23.alpha_flat_indices, layout23.feature_shape)
        c25 = _coordinates(layout25.alpha_flat_indices, layout25.feature_shape)
        c28 = _coordinates(layout28.alpha_flat_indices, layout28.feature_shape)
        locations = torch.tensor(
            layout31.beta_locations,
            device=self._center.device,
            dtype=torch.int64,
        ).contiguous()
        signs = torch.tensor(
            S4_BETA_SIGN_V1,
            device=self._center.device,
            dtype=torch.float32,
        ).reshape(plan.domain_count, 1)
        static = BabFullRegionStaticV1(
            terminal_lower=tensor("relu/31/lower"),
            terminal_upper=tensor("relu/31/upper"),
            terminal_coordinates=terminal_coordinates,
            terminal_weight=tensor("param/linear1.weight"),
            terminal_bias=tensor("param/linear1.bias"),
            beta_locations=locations,
            beta_signs=signs,
            residual=BabResidualStaticV1(
                entry_lower=tensor("relu/28/lower"),
                entry_upper=tensor("relu/28/upper"),
                entry_coordinates=c28,
                main_weight=tensor("param/layer1.1.conv2.weight"),
                main_bias=tensor("param/layer1.1.conv2.bias"),
                inner_lower=tensor("relu/25/lower"),
                inner_upper=tensor("relu/25/upper"),
                inner_coordinates=c25,
                inner_weight=tensor("param/layer1.1.conv1.weight"),
                inner_bias=tensor("param/layer1.1.conv1.bias"),
            ),
            projection=BabProjectionStaticV1(
                entry_lower=tensor("relu/23/lower"),
                entry_upper=tensor("relu/23/upper"),
                entry_coordinates=c23,
                outer_weight=tensor("param/layer1.0.conv2.weight"),
                outer_bias=tensor("param/layer1.0.conv2.bias"),
                inner_lower=tensor("relu/19/lower"),
                inner_upper=tensor("relu/19/upper"),
                inner_coordinates=c19,
                inner_weight=tensor("param/layer1.0.conv1.weight"),
                inner_bias=tensor("param/layer1.0.conv1.bias"),
                skip_weight=tensor("param/layer1.0.shortcut.0.weight"),
                skip_bias=tensor("param/layer1.0.shortcut.0.bias"),
            ),
            input_domain=BabInputStaticV1(
                lower=tensor("relu/17/lower"),
                upper=tensor("relu/17/upper"),
                coordinates=c17,
                weight=tensor("param/conv1.weight"),
                bias=tensor("param/conv1.bias"),
                center=self._center,
                radius=self._radius,
            ),
        )
        terminal = BabTerminalTIRExecutorV1(
            BabTerminalLinearTemplateV1(
                spec_count=1,
                domain_count=plan.domain_count,
                current_features=100,
                previous_features=1024,
                alpha_feature_indices=terminal_coordinates,
                beta_count=1,
                compute_capability=capability,
            )
        )
        residual = RootCrownResidualTIRExecutorV1(
            RootCrownResidualTemplateV1(
                spec_count=1,
                domain_count=plan.domain_count,
                channels=16,
                height=8,
                width=8,
                entry_alpha_coordinates=c28,
                inner_alpha_coordinates=c25,
                compute_capability=capability,
            )
        )
        projection = RootCrownProjectionTIRExecutorV1(
            RootCrownProjectionTemplateV1(
                spec_count=1,
                domain_count=plan.domain_count,
                output_channels=16,
                output_height=8,
                output_width=8,
                input_channels=8,
                input_height=16,
                input_width=16,
                entry_alpha_coordinates=c23,
                inner_alpha_coordinates=c19,
                compute_capability=capability,
            )
        )
        input_domain = BabInputDomainTIRExecutorV1(
            BabInputDomainTemplateV1(
                spec_count=1,
                domain_count=plan.domain_count,
                output_channels=8,
                output_height=16,
                output_width=16,
                input_channels=3,
                input_height=32,
                input_width=32,
                alpha_coordinates=c17,
                compute_capability=capability,
                forward_symbol="boundflow_bab_input_domain_forward_v1",
                backward_symbol="boundflow_bab_input_domain_backward_v1",
            )
        )
        self.owner = PreparedBabFullRegionOwnerV1(
            static,
            terminal_executor=terminal,
            residual_executor=residual,
            projection_executor=projection,
            input_executor=input_domain,
        )
        self.compiled_identity_hash = hashlib.sha256(
            json.dumps(
                {
                    "terminal_template": terminal.template.stable_hash(),
                    "terminal_schedule": terminal.compiled.scheduled_tir_hash,
                    "residual_template": residual.template.stable_hash(),
                    "residual_schedule": residual.compiled.scheduled_tir_hash,
                    "projection_template": projection.template.stable_hash(),
                    "projection_schedule": projection.compiled.scheduled_tir_hash,
                    "input_template": input_domain.template.stable_hash(),
                    "input_schedule": input_domain.compiled.scheduled_tir_hash,
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()

    def rebind(self, live_sources: Mapping[str, torch.Tensor]) -> None:
        tensor = self.region.executor._tensor
        with torch.no_grad():
            lower = tensor("input/lower")
            upper = tensor("input/upper")
            self._center.copy_((lower + upper) * 0.5)
            self._radius.copy_((upper - lower) * 0.5)
            objective = tensor("objective")
            terminal = objective.matmul(tensor("param/linear2.weight"))
            self._terminal_incoming.copy_(terminal.transpose(0, 1))
            self._initial_bias.copy_(
                (objective * tensor("param/linear2.bias")).sum(dim=-1)
            )
            for name, layout in self._layouts.items():
                source = live_sources[layout.alpha_path]
                target = self._alphas[name]
                if source.shape != target.shape:
                    raise ValueError("activation-BaB live alpha shape differs")
                target.copy_(source)
            active = [
                live_sources[layout.beta_path]
                for layout in self._layouts.values()
                if any(layout.beta_locations)
            ]
            if len(active) != 1 or active[0].shape != self._beta.shape:
                raise ValueError("activation-BaB live beta owner differs")
            self._beta.copy_(active[0])

    def _dynamic(self) -> BabFullRegionDynamicV1:
        return BabFullRegionDynamicV1(
            terminal_incoming=self._terminal_incoming,
            terminal_alpha=self._alphas["31"],
            residual_entry_alpha=self._alphas["28"],
            residual_inner_alpha=self._alphas["25"],
            projection_entry_alpha=self._alphas["23"],
            projection_inner_alpha=self._alphas["19"],
            input_alpha=self._alphas["17"],
            beta=self._beta,
        )

    def run(self, stream: torch.cuda.Stream) -> BabFourSegmentRunResultV1:
        parameters = [
            self._alphas["31"],
            self._alphas["28"],
            self._alphas["25"],
            self._alphas["23"],
            self._alphas["19"],
            self._alphas["17"],
            self._beta,
        ]
        optimizer = torch.optim.Adam(
            (
                {"params": parameters[:6], "lr": 0.01},
                {"params": [parameters[6]], "lr": 0.05},
            ),
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
        learning_rates: list[tuple[float, float]] = []
        terminal_lower: torch.Tensor | None = None
        terminal_las: tuple[torch.Tensor, ...] | None = None
        with torch.cuda.stream(stream):
            for ordinal in range(10):
                alpha_lr = float(optimizer.param_groups[0]["lr"])
                beta_lr = float(optimizer.param_groups[1]["lr"])
                if not math.isclose(
                    alpha_lr, 0.01 * 0.98**ordinal
                ) or not math.isclose(beta_lr, 0.05 * 0.98**ordinal):
                    raise ValueError("activation-BaB learning-rate sequence differs")
                learning_rates.append((alpha_lr, beta_lr))
                lower = self.owner.evaluate(self._dynamic()) + self._initial_bias
                if ordinal < 9:
                    gradients = torch.autograd.grad(-lower.sum(), parameters)
                    optimizer.zero_grad(set_to_none=True)
                    for parameter, gradient in zip(parameters, gradients):
                        parameter.grad = gradient
                    optimizer.step()
                    with torch.no_grad():
                        for parameter in parameters[:6]:
                            parameter.clamp_(0.0, 1.0)
                        parameters[6].clamp_(min=0.0)
                else:
                    terminal_lower = lower.detach().clone()
                    trace = self.owner.last_trace
                    if trace is None:
                        raise ValueError("activation-BaB terminal trace is absent")
                    by_name = dict(trace.site_lower_adjoints_spec_first)
                    terminal_las = tuple(
                        by_name[name]
                        .transpose(0, 1)
                        .detach()
                        .clone(memory_format=torch.contiguous_format)
                        for name in ("17", "19", "23", "25", "28", "31")
                    )
                scheduler.step()
        stream.synchronize()
        if terminal_lower is None or terminal_las is None:
            raise ValueError("activation-BaB terminal result is absent")
        compact = tuple(
            self._alphas[name][0, 0].detach().clone()
            for name in ("17", "19", "23", "25", "28", "31")
        ) + (self._beta.detach().clone(),)
        receipt = self.owner.receipt()
        result = BabFourSegmentRunResultV1(
            terminal_lower=terminal_lower,
            terminal_parameters=compact,
            terminal_las=terminal_las,
            learning_rates=tuple(learning_rates),
            evaluation_count=10,
            mutation_count=9,
            compiled_segment_count=receipt.compiled_segment_count,
            compiled_forward_launch_count=receipt.compiled_forward_launch_count,
            compiled_backward_launch_count=receipt.compiled_backward_launch_count,
            fallback_count=receipt.fallback_count,
        )
        result.validate()
        return result


__all__ = [
    "BabFourSegmentRunResultV1",
    "PreparedBabFourSegmentOptimizerV1",
]
