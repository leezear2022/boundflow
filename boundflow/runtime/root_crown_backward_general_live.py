"""Direct production bridge for one complete root CROWN backward transaction."""

# mypy: disable-error-code=import-untyped
# pylint: disable=protected-access,too-many-locals,too-many-statements
# pylint: disable=too-many-boolean-expressions,too-many-instance-attributes

from __future__ import annotations

from contextlib import contextmanager
import inspect
from functools import wraps
from typing import Any, Iterator, Mapping, cast

import torch

from boundflow.runtime.root_crown_expanded_suffix_tir import (
    RootCrownExpandedSuffixTensorsV1,
)
from boundflow.runtime.root_crown_full_pipeline_tir import (
    RootCrownFullPipelineTIRExecutorV1,
    execute_root_crown_full_pipeline_tir_v1,
)
from boundflow.runtime.root_crown_input_capture import (
    ROOT_INPUT_CONV,
    ROOT_INPUT_NODE,
    ROOT_INPUT_RELU,
)
from boundflow.runtime.root_crown_input_domain_live import (
    RootCrownInputDomainLiveBridgeV1,
)
from boundflow.runtime.root_crown_projection_capture import (
    ROOT_PROJECTION_ADD,
    ROOT_PROJECTION_ENTRY_RELU,
    ROOT_PROJECTION_INNER_RELU,
    ROOT_PROJECTION_MAIN_INNER_CONV,
    ROOT_PROJECTION_MAIN_OUTER_CONV,
    ROOT_PROJECTION_SKIP_CONV,
)
from boundflow.runtime.root_crown_projection_live import (
    RootCrownProjectionLiveBridgeV1,
)
from boundflow.runtime.root_crown_projection_tir import (
    RootCrownProjectionTensorsV1,
)
from boundflow.runtime.root_crown_residual_capture import (
    ROOT_RESIDUAL_ADD,
    ROOT_RESIDUAL_ENTRY_RELU,
    ROOT_RESIDUAL_INNER_CONV,
    ROOT_RESIDUAL_INNER_RELU,
    ROOT_RESIDUAL_MAIN_CONV,
)
from boundflow.runtime.root_crown_residual_tir import RootCrownResidualTensorsV1
from boundflow.runtime.root_crown_suffix_live import RootCrownSuffixLiveBridgeV1
from boundflow.runtime.root_crown_suffix_tir import RootCrownSuffixTensorsV1
from boundflow.runtime.root_crown_terminal_capture import (
    ROOT_START_NODE,
    ROOT_TERMINAL_LINEAR,
    ROOT_TERMINAL_RELU,
)
from boundflow.runtime.root_crown_terminal_tir import RootCrownTerminalTensorsV1


def _tensor(value: Any, name: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"root direct transaction tensor differs: {name}")
    return cast(torch.Tensor, value)


class RootCrownBackwardGeneralLiveBridgeV1:
    """Replace the deque traversal while preserving public compute_bounds setup."""

    def __init__(
        self,
        executor: RootCrownFullPipelineTIRExecutorV1,
        suffix: RootCrownSuffixLiveBridgeV1,
        projection: RootCrownProjectionLiveBridgeV1,
        input_domain: RootCrownInputDomainLiveBridgeV1,
    ) -> None:
        self.executor = executor
        self.suffix = suffix
        self.projection = projection
        self.input_domain = input_domain
        self.call_count = 0
        self.fallback_count = 0

    @contextmanager
    def install(self, bounded_module_type: type[Any]) -> Iterator[None]:
        """Install only at the five admitted final-node backward calls."""

        original = bounded_module_type.backward_general
        signature = inspect.signature(original)

        @wraps(original)
        def direct(instance: Any, *args: Any, **kwargs: Any) -> Any:
            if not self.suffix._active:
                return original(instance, *args, **kwargs)
            bound = signature.bind(instance, *args, **kwargs)
            bound.apply_defaults()
            values = bound.arguments
            bound_node = values.get("bound_node")
            if str(getattr(bound_node, "name", "")) != ROOT_START_NODE:
                return original(instance, *args, **kwargs)
            try:
                return self._execute(instance, values)
            except Exception:
                self.fallback_count += 1
                raise

        bounded_module_type.backward_general = direct
        try:
            yield
        finally:
            bounded_module_type.backward_general = original

    def _execute(
        self, instance: Any, values: Mapping[str, Any]
    ) -> tuple[torch.Tensor, None]:
        specification = _tensor(values.get("C"), "specification")
        bound_node = values.get("bound_node")
        if bound_node is None:
            raise TypeError("root direct bound node differs")
        output_constraints = values.get("apply_output_constraints_to")
        mismatches = {
            "count": self.call_count >= 5,
            "lower": values.get("bound_lower") is not True,
            "upper": values.get("bound_upper") is not False,
            "average": bool(values.get("average_A")),
            "need_a": bool(values.get("need_A_only")),
            "start": values.get("start_backpropagation_at_node") is not None,
            "unstable": values.get("unstable_idx") is not None,
            "mask": values.get("update_mask") is not None,
            "initial_as": values.get("initial_As") is not None,
            "initial_lb": values.get("initial_lb") is not None,
            "initial_ub": values.get("initial_ub") is not None,
            "return_a": bool(getattr(instance, "return_A", False)),
            "invprop": bool(getattr(instance, "invprop_enabled")()),
            "output_constraint": bool(
                bound_node.are_output_constraints_activated_for_layer(
                    output_constraints
                )
            ),
            "cut": bool(getattr(instance, "cut_used", False)),
            "fixed_intermediate": not bool(
                getattr(instance, "bound_opts", {})
                .get("optimize_bound_args", {})
                .get("fix_interm_bounds", False)
            ),
        }
        failed = sorted(name for name, mismatch in mismatches.items() if mismatch)
        if failed:
            raise ValueError("root direct transaction differs: " + ",".join(failed))

        nodes = {str(getattr(node, "name", "")): node for node in instance.nodes()}
        required = {
            ROOT_START_NODE,
            ROOT_TERMINAL_RELU,
            ROOT_TERMINAL_LINEAR,
            ROOT_RESIDUAL_ENTRY_RELU,
            ROOT_RESIDUAL_ADD,
            ROOT_RESIDUAL_MAIN_CONV,
            ROOT_RESIDUAL_INNER_RELU,
            ROOT_RESIDUAL_INNER_CONV,
            ROOT_PROJECTION_ENTRY_RELU,
            ROOT_PROJECTION_ADD,
            ROOT_PROJECTION_MAIN_OUTER_CONV,
            ROOT_PROJECTION_INNER_RELU,
            ROOT_PROJECTION_MAIN_INNER_CONV,
            ROOT_PROJECTION_SKIP_CONV,
            ROOT_INPUT_RELU,
            ROOT_INPUT_CONV,
            ROOT_INPUT_NODE,
        }
        if not required <= nodes.keys():
            raise ValueError("root direct transaction topology differs")

        for node in instance.nodes():
            node.lA = node.uA = None
        bound_node.lA = specification
        bound_node.uA = None

        final_weight = _tensor(bound_node.inputs[1].lower, "final weight")
        final_bias = _tensor(bound_node.inputs[2].lower, "final bias")
        terminal_incoming = torch.matmul(specification, final_weight).transpose(0, 1)

        terminal_relu = nodes[ROOT_TERMINAL_RELU]
        terminal_linear = nodes[ROOT_TERMINAL_LINEAR]
        residual_entry = nodes[ROOT_RESIDUAL_ENTRY_RELU]
        residual_add = nodes[ROOT_RESIDUAL_ADD]
        residual_main = nodes[ROOT_RESIDUAL_MAIN_CONV]
        residual_inner = nodes[ROOT_RESIDUAL_INNER_RELU]
        residual_inner_conv = nodes[ROOT_RESIDUAL_INNER_CONV]
        projection_entry = nodes[ROOT_PROJECTION_ENTRY_RELU]
        projection_add = nodes[ROOT_PROJECTION_ADD]
        projection_outer = nodes[ROOT_PROJECTION_MAIN_OUTER_CONV]
        projection_inner = nodes[ROOT_PROJECTION_INNER_RELU]
        projection_inner_conv = nodes[ROOT_PROJECTION_MAIN_INNER_CONV]
        projection_skip = nodes[ROOT_PROJECTION_SKIP_CONV]
        input_relu = nodes[ROOT_INPUT_RELU]
        input_conv = nodes[ROOT_INPUT_CONV]
        input_node = nodes[ROOT_INPUT_NODE]

        terminal_lower = _tensor(terminal_relu.inputs[0].lower, "terminal lower")
        terminal_upper = _tensor(terminal_relu.inputs[0].upper, "terminal upper")
        residual_entry_lower = _tensor(residual_add.lower, "residual entry lower")
        residual_entry_upper = _tensor(residual_add.upper, "residual entry upper")
        residual_inner_lower = _tensor(
            residual_inner.inputs[0].lower, "residual inner lower"
        )
        residual_inner_upper = _tensor(
            residual_inner.inputs[0].upper, "residual inner upper"
        )
        projection_entry_lower = _tensor(projection_add.lower, "projection entry lower")
        projection_entry_upper = _tensor(projection_add.upper, "projection entry upper")
        projection_inner_lower = _tensor(
            projection_inner.inputs[0].lower, "projection inner lower"
        )
        projection_inner_upper = _tensor(
            projection_inner.inputs[0].upper, "projection inner upper"
        )
        input_lower = _tensor(input_conv.lower, "input lower")
        input_upper = _tensor(input_conv.upper, "input upper")

        self.suffix._admit_static(terminal_relu, residual_entry, residual_inner)
        terminal = RootCrownTerminalTensorsV1(
            terminal_incoming,
            terminal_lower,
            terminal_upper,
            _tensor(terminal_relu.alpha.get(ROOT_START_NODE), "terminal alpha"),
            _tensor(terminal_linear.inputs[1].lower, "terminal weight"),
            _tensor(terminal_linear.inputs[2].lower, "terminal bias"),
        )
        self.suffix._set_relu_state(
            terminal_relu,
            terminal.preactivation_lower,
            terminal.preactivation_upper,
            terminal_incoming,
            "terminal",
        )
        terminal_linear._start = ROOT_START_NODE
        terminal_output, _ = self.executor.stage_terminal(terminal)

        residual_incoming = terminal_output.view(
            self.suffix.residual_template.coefficient_shape
        )
        residual = RootCrownResidualTensorsV1(
            residual_incoming,
            residual_entry_lower,
            residual_entry_upper,
            _tensor(residual_entry.alpha.get(ROOT_START_NODE), "residual entry alpha"),
            _tensor(residual_main.inputs[1].lower, "residual main weight"),
            _tensor(residual_main.inputs[2].lower, "residual main bias"),
            residual_inner_lower,
            residual_inner_upper,
            _tensor(residual_inner.alpha.get(ROOT_START_NODE), "residual inner alpha"),
            _tensor(residual_inner_conv.inputs[1].lower, "residual inner weight"),
            _tensor(residual_inner_conv.inputs[2].lower, "residual inner bias"),
        )
        self.suffix._set_relu_state(
            residual_entry,
            residual.entry_lower,
            residual.entry_upper,
            residual_incoming,
            "entry",
        )
        suffix_tensors = RootCrownSuffixTensorsV1(terminal, residual)
        residual_output, _ = self.executor.stage_residual(suffix_tensors)
        self.suffix._set_relu_state(
            residual_inner,
            residual.inner_lower,
            residual.inner_upper,
            self.executor.last_residual_main_a,
            "inner",
        )

        self.projection._admit_static(projection_entry, projection_inner)
        projection = RootCrownProjectionTensorsV1(
            residual_output,
            projection_entry_lower,
            projection_entry_upper,
            _tensor(
                projection_entry.alpha.get(ROOT_START_NODE),
                "projection entry alpha",
            ),
            _tensor(projection_outer.inputs[1].lower, "projection outer weight"),
            _tensor(projection_outer.inputs[2].lower, "projection outer bias"),
            projection_inner_lower,
            projection_inner_upper,
            _tensor(
                projection_inner.alpha.get(ROOT_START_NODE),
                "projection inner alpha",
            ),
            _tensor(
                projection_inner_conv.inputs[1].lower,
                "projection inner weight",
            ),
            _tensor(
                projection_inner_conv.inputs[2].lower,
                "projection inner bias",
            ),
            _tensor(projection_skip.inputs[1].lower, "projection skip weight"),
            _tensor(projection_skip.inputs[2].lower, "projection skip bias"),
        )
        self.projection._set_relu_state(
            projection_entry,
            projection.entry_lower,
            projection.entry_upper,
            residual_output,
            inner=False,
        )
        expanded = RootCrownExpandedSuffixTensorsV1(suffix_tensors, projection)
        projection_output, _ = self.executor.stage_projection(expanded)
        self.projection._set_relu_state(
            projection_inner,
            projection.inner_lower,
            projection.inner_upper,
            self.executor.last_projection_outer_a,
            inner=True,
        )

        self.input_domain._admit_static(input_relu, input_node)
        input_alpha = _tensor(input_relu.alpha.get(ROOT_START_NODE), "input alpha")
        self.input_domain._set_relu_state(
            input_relu, input_lower, input_upper, projection_output
        )
        center = self.input_domain._input_center
        radius = self.input_domain._input_radius
        if center is None or radius is None:
            raise RuntimeError("root direct input domain state differs")
        concrete, pipeline_bias = execute_root_crown_full_pipeline_tir_v1(
            expanded,
            input_lower,
            input_upper,
            input_alpha,
            _tensor(input_conv.inputs[1].lower, "input weight"),
            _tensor(input_conv.inputs[2].lower, "input bias"),
            center,
            radius,
            self.executor,
        )
        lower = concrete + pipeline_bias.transpose(0, 1)
        lower = lower + torch.matmul(specification, final_bias)

        # Preserve the optimizer-visible ReLU state and the existing receipts.
        self.suffix._last_suffix = suffix_tensors
        self.suffix.terminal_relu_count += 1
        self.suffix.terminal_linear_count += 1
        self.suffix.residual_entry_count += 1
        self.suffix.residual_add_count += 1
        self.projection._last_tensors = projection
        self.projection.entry_replacement_count += 1
        self.projection.add_replacement_count += 1
        self.projection.skip_carrier_count += 1
        self.input_domain._last_pending = (
            projection_output,
            input_lower,
            input_upper,
            input_alpha,
        )
        self.input_domain.relu_replacement_count += 1
        self.input_domain.conv_replacement_count += 1
        self.input_domain.concretize_replacement_count += 1
        self.input_domain.deferred_dense_a_count += 1
        for root in instance.roots():
            root.lb = root.ub = None
        self.call_count += 1
        return lower, None

    def validate(self) -> None:
        """Require exact five-forward/four-backward activation without fallback."""

        if self.call_count != 5 or self.fallback_count != 0:
            raise ValueError("root direct backward activation count differs")

    def receipt(self) -> dict[str, object]:
        """Return the exact replacement seam and activation counters."""

        self.validate()
        return {
            "schema_version": "boundflow.root-crown-backward-general-live/v1",
            "replacement_seam": "BoundedModule.backward_general:/49",
            "call_count": self.call_count,
            "native_deque_traversal_count": 0,
            "fallback_count": self.fallback_count,
            "performance_claimed": False,
        }


__all__ = ["RootCrownBackwardGeneralLiveBridgeV1"]
