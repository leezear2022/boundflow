"""Production-seam shadow bridge for the /input-28 intermediate CROWN path."""

# mypy: disable-error-code=import-untyped
# pylint: disable=protected-access,too-many-locals,too-many-statements
# pylint: disable=too-many-arguments
# pylint: disable=too-many-boolean-expressions,too-many-instance-attributes

from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
import inspect
from typing import Any, Iterator, Mapping, cast

import torch

from boundflow.runtime.root_crown_input_domain_live import (
    RootCrownInputDomainLiveBridgeV1,
)
from boundflow.runtime.root_crown_intermediate_dual_lane_tir import (
    RootCrownIntermediateDualLaneTensorsV1,
    RootCrownIntermediateDualLaneTIRExecutorV1,
)

INTERMEDIATE_START_NODE = "/input-28"


def _tensor(value: Any, name: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"root intermediate live tensor differs: {name}")
    return cast(torch.Tensor, value)


class RootCrownIntermediateLiveBridgeV1:
    """Execute the compiled dual-lane candidate beside the native transaction."""

    def __init__(
        self,
        executor: RootCrownIntermediateDualLaneTIRExecutorV1,
        input_bridge: RootCrownInputDomainLiveBridgeV1,
        *,
        replace_output: bool = False,
        execute_native: bool = True,
        suffix_bridge: Any | None = None,
        projection_bridge: Any | None = None,
    ) -> None:
        self.executor = executor
        self.input_bridge = input_bridge
        self.replace_output = replace_output
        self.execute_native = execute_native
        self.suffix_bridge = suffix_bridge
        self.projection_bridge = projection_bridge
        self.call_count = 0
        self.fallback_count = 0
        self.lower_max_abs_diff = 0.0
        self.upper_max_abs_diff = 0.0
        self.lower_sign_exact = True
        self.upper_sign_exact = True
        self._last_lower_seed: torch.Tensor | None = None

    @staticmethod
    def _selected_alpha(
        relu: Any,
        lower_a: torch.Tensor,
        upper_a: torch.Tensor,
        unstable_idx: torch.Tensor,
        start_node: Any,
    ) -> torch.Tensor:
        selected, _lookup = relu.select_alpha_by_idx(
            lower_a, upper_a, unstable_idx, start_node
        )
        return _tensor(selected, f"{relu.name} selected alpha").contiguous()

    @contextmanager
    def install(self, bounded_module_type: type[Any]) -> Iterator[None]:
        """Patch only the admitted /input-28 intermediate backward call."""

        original = bounded_module_type.backward_general
        signature = inspect.signature(original)

        @wraps(original)
        def wrapped(instance: Any, *args: Any, **kwargs: Any) -> Any:
            bound = signature.bind(instance, *args, **kwargs)
            bound.apply_defaults()
            values = bound.arguments
            bound_node = values.get("bound_node")
            specification = values.get("C")
            unstable_idx = values.get("unstable_idx")
            eligible = (
                str(getattr(bound_node, "name", "")) == INTERMEDIATE_START_NODE
                and type(specification).__name__ == "OneHotC"
                and getattr(specification, "shape", (None, None))[1]
                == self.executor.residual_template.spec_count
                and torch.is_tensor(unstable_idx)
                and cast(torch.Tensor, unstable_idx).ndim == 1
                and cast(torch.Tensor, unstable_idx).numel()
                == self.executor.residual_template.spec_count
                and values.get("bound_lower") is True
                and values.get("bound_upper") is True
            )
            if not eligible:
                return original(instance, *args, **kwargs)
            native = (
                original(instance, *args, **kwargs) if self.execute_native else None
            )
            try:
                candidate = self._execute(instance, values)
                if native is not None:
                    self._compare(native, candidate)
                else:
                    self._publish_state(instance)
            except Exception:
                self.fallback_count += 1
                raise
            self.call_count += 1
            return candidate if self.replace_output else native

        bounded_module_type.backward_general = wrapped
        try:
            yield
        finally:
            bounded_module_type.backward_general = original

    def _linear_seed(
        self,
        instance: Any,
        bound_node: Any,
        specification: Any,
        unstable_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        processed, _batch, _output_dim, _output_shape = instance._preprocess_C(
            specification, bound_node
        )
        result = bound_node.bound_backward(
            processed,
            processed,
            *bound_node.inputs,
            start_node=bound_node,
            unstable_idx=unstable_idx,
        )
        try:
            lower_a = _tensor(result[0][0][0], "linear lower seed")
            upper_a = _tensor(result[0][0][1], "linear upper seed")
            lower_bias = _tensor(result[1], "linear lower bias")
            upper_bias = _tensor(result[2], "linear upper bias")
        except (IndexError, TypeError) as error:
            raise ValueError("root intermediate linear result differs") from error
        shape = self.executor.residual_template.coefficient_shape
        return (
            lower_a.reshape(shape).contiguous(),
            upper_a.reshape(shape).contiguous(),
            lower_bias.contiguous(),
            upper_bias.contiguous(),
        )

    def _execute(
        self, instance: Any, values: Mapping[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bound_node = values.get("bound_node")
        specification = values.get("C")
        unstable_idx = values.get("unstable_idx")
        mismatches = {
            "lower": values.get("bound_lower") is not True,
            "upper": values.get("bound_upper") is not True,
            "average": bool(values.get("average_A")),
            "need_a": bool(values.get("need_A_only")),
            "start": values.get("start_backpropagation_at_node") is not None,
            "unstable": not torch.is_tensor(unstable_idx),
            "unstable_rank": torch.is_tensor(unstable_idx)
            and cast(torch.Tensor, unstable_idx).ndim != 1,
            "spec_kind": type(specification).__name__ != "OneHotC",
            "spec_count": getattr(specification, "shape", (None, None))[1]
            != self.executor.residual_template.spec_count,
        }
        failed = sorted(name for name, mismatch in mismatches.items() if mismatch)
        if failed or bound_node is None or not torch.is_tensor(unstable_idx):
            raise ValueError(
                "root intermediate live transaction differs: " + ",".join(failed)
            )
        lower_seed, upper_seed, lower_bias, upper_bias = self._linear_seed(
            instance, bound_node, specification, unstable_idx
        )
        nodes = {str(getattr(node, "name", "")): node for node in instance.nodes()}
        required = {
            "/45",
            "/44",
            "/43",
            "/input-24",
            "/input-20",
            "/input-16",
            "/39",
            "/37",
            "/input-12",
            "/input-8",
            "/38",
            "/input-4",
            "/input",
            "/input-1",
        }
        if not required <= nodes.keys():
            raise ValueError("root intermediate live topology differs")

        residual_entry = nodes["/45"]
        residual_inner = nodes["/input-24"]
        projection_entry = nodes["/input-16"]
        projection_inner = nodes["/input-12"]
        input_relu = nodes["/input-4"]
        self.input_bridge._admit_static(input_relu, nodes["/input-1"])
        center = self.input_bridge._input_center
        radius = self.input_bridge._input_radius
        if center is None or radius is None:
            raise RuntimeError("root intermediate input domain state differs")
        tensors = RootCrownIntermediateDualLaneTensorsV1(
            lower_seed,
            upper_seed,
            lower_bias,
            upper_bias,
            _tensor(nodes["/44"].lower, "residual entry lower"),
            _tensor(nodes["/44"].upper, "residual entry upper"),
            self._selected_alpha(
                residual_entry, lower_seed, upper_seed, unstable_idx, bound_node
            ),
            _tensor(nodes["/43"].inputs[1].lower, "residual main weight"),
            _tensor(nodes["/43"].inputs[2].lower, "residual main bias"),
            _tensor(nodes["/input-20"].lower, "residual inner lower"),
            _tensor(nodes["/input-20"].upper, "residual inner upper"),
            self._selected_alpha(
                residual_inner, lower_seed, upper_seed, unstable_idx, bound_node
            ),
            _tensor(nodes["/input-20"].inputs[1].lower, "residual inner weight"),
            _tensor(nodes["/input-20"].inputs[2].lower, "residual inner bias"),
            _tensor(nodes["/39"].lower, "projection entry lower"),
            _tensor(nodes["/39"].upper, "projection entry upper"),
            self._selected_alpha(
                projection_entry, lower_seed, upper_seed, unstable_idx, bound_node
            ),
            _tensor(nodes["/37"].inputs[1].lower, "projection outer weight"),
            _tensor(nodes["/37"].inputs[2].lower, "projection outer bias"),
            _tensor(nodes["/input-8"].lower, "projection inner lower"),
            _tensor(nodes["/input-8"].upper, "projection inner upper"),
            self._selected_alpha(
                projection_inner, lower_seed, upper_seed, unstable_idx, bound_node
            ),
            _tensor(nodes["/input-8"].inputs[1].lower, "projection inner weight"),
            _tensor(nodes["/input-8"].inputs[2].lower, "projection inner bias"),
            _tensor(nodes["/38"].inputs[1].lower, "projection skip weight"),
            _tensor(nodes["/38"].inputs[2].lower, "projection skip bias"),
            _tensor(nodes["/input"].lower, "input lower"),
            _tensor(nodes["/input"].upper, "input upper"),
            self._selected_alpha(
                input_relu, lower_seed, upper_seed, unstable_idx, bound_node
            ),
            _tensor(nodes["/input"].inputs[1].lower, "input weight"),
            _tensor(nodes["/input"].inputs[2].lower, "input bias"),
            center,
            radius,
        )
        self._last_lower_seed = lower_seed
        return self.executor.execute(tensors)

    def _publish_state(self, instance: Any) -> None:
        """Restore optimizer-visible ReLU state without native deque traversal."""

        if self.suffix_bridge is None or self.projection_bridge is None:
            raise RuntimeError("root intermediate state publisher differs")
        nodes = {str(getattr(node, "name", "")): node for node in instance.nodes()}
        for node in nodes.values():
            node.lA = node.uA = None
        lower = self.executor.lower
        residual_output = lower.residual._output_a
        projection_output = lower.projection._output_a
        if (
            residual_output is None
            or projection_output is None
            or self._last_lower_seed is None
        ):
            raise RuntimeError("root intermediate compiled state differs")
        self.suffix_bridge._set_relu_state(
            nodes["/45"],
            _tensor(nodes["/44"].lower, "residual entry lower"),
            _tensor(nodes["/44"].upper, "residual entry upper"),
            self._last_lower_seed,
            "entry",
        )
        self.suffix_bridge._set_relu_state(
            nodes["/input-24"],
            _tensor(nodes["/input-20"].lower, "residual inner lower"),
            _tensor(nodes["/input-20"].upper, "residual inner upper"),
            lower.residual.last_main_a,
            "inner",
        )
        self.projection_bridge._set_relu_state(
            nodes["/input-16"],
            _tensor(nodes["/39"].lower, "projection entry lower"),
            _tensor(nodes["/39"].upper, "projection entry upper"),
            residual_output,
            inner=False,
        )
        self.projection_bridge._set_relu_state(
            nodes["/input-12"],
            _tensor(nodes["/input-8"].lower, "projection inner lower"),
            _tensor(nodes["/input-8"].upper, "projection inner upper"),
            lower.projection.last_outer_a,
            inner=True,
        )
        self.input_bridge._set_relu_state(
            nodes["/input-4"],
            _tensor(nodes["/input"].lower, "input lower"),
            _tensor(nodes["/input"].upper, "input upper"),
            projection_output,
        )

    def _compare(
        self, native: Any, candidate: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Accumulate exact native-versus-compiled forward evidence."""

        if not isinstance(native, tuple) or len(native) < 2:
            raise ValueError("root intermediate native result differs")
        native_lower = _tensor(native[0], "native lower")
        native_upper = _tensor(native[1], "native upper")
        candidate_lower, candidate_upper = candidate
        lower_diff = float(
            (native_lower - candidate_lower).abs().max().detach().cpu().item()
        )
        upper_diff = float(
            (native_upper - candidate_upper).abs().max().detach().cpu().item()
        )
        self.lower_max_abs_diff = max(self.lower_max_abs_diff, lower_diff)
        self.upper_max_abs_diff = max(self.upper_max_abs_diff, upper_diff)
        self.lower_sign_exact = self.lower_sign_exact and bool(
            torch.equal(native_lower >= 0, candidate_lower >= 0)
        )
        self.upper_sign_exact = self.upper_sign_exact and bool(
            torch.equal(native_upper >= 0, candidate_upper >= 0)
        )

    def receipt(self) -> dict[str, object]:
        """Describe activation, parity, and the still-diagnostic owner boundary."""

        return {
            "schema_version": "boundflow.root-intermediate-live/v1",
            "replacement_seam": "BoundedModule.backward_general:/input-28",
            "mode": (
                "direct"
                if not self.execute_native
                else "replace" if self.replace_output else "shadow"
            ),
            "call_count": self.call_count,
            "fallback_count": self.fallback_count,
            "native_execution_count": self.call_count if self.execute_native else 0,
            "lower_max_abs_diff": self.lower_max_abs_diff,
            "upper_max_abs_diff": self.upper_max_abs_diff,
            "lower_sign_exact": self.lower_sign_exact,
            "upper_sign_exact": self.upper_sign_exact,
            "executor": self.executor.receipt(),
            "performance_claimed": False,
        }


__all__ = ["RootCrownIntermediateLiveBridgeV1"]
