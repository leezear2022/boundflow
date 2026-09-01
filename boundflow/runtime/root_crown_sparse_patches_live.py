"""Production bridge for the `/44` sparse-Patches CROWN transaction."""

# mypy: disable-error-code=import-untyped
# pylint: disable=protected-access,too-many-locals,too-many-statements
# pylint: disable=too-many-arguments,too-many-boolean-expressions
# pylint: disable=too-many-instance-attributes
# pylint: disable=missing-function-docstring

from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
import inspect
from typing import Any, Iterator, Mapping, cast

import torch

from boundflow.runtime.root_crown_input_domain_live import (
    RootCrownInputDomainLiveBridgeV1,
)
from boundflow.runtime.root_crown_sparse_patches_dual_lane_tir import (
    RootCrownSparsePatchesDualLaneTensorsV1,
    RootCrownSparsePatchesDualLaneTIRExecutorV1,
)
from boundflow.runtime.root_crown_sparse_patches_seed_tir import patches_payload_v1

SPARSE_PATCHES_START_NODE = "/44"


def _tensor(value: Any, name: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"root sparse Patches live tensor differs: {name}")
    return cast(torch.Tensor, value)


def _unstable_tuple(value: Any) -> tuple[torch.Tensor, ...]:
    if (
        not isinstance(value, tuple)
        or len(value) != 3
        or not all(torch.is_tensor(item) for item in value)
    ):
        raise TypeError("root sparse Patches unstable index differs")
    return cast(tuple[torch.Tensor, ...], value)


class RootCrownSparsePatchesLiveBridgeV1:
    """Replace exact `/44` Patches calls with specialized compiled regions."""

    def __init__(
        self,
        executors: Mapping[int, RootCrownSparsePatchesDualLaneTIRExecutorV1],
        input_bridge: RootCrownInputDomainLiveBridgeV1,
        *,
        replace_output: bool = False,
        execute_native: bool = True,
        suffix_bridge: Any | None = None,
        projection_bridge: Any | None = None,
    ) -> None:
        self.executors = dict(executors)
        if not self.executors or set(self.executors) != {
            executor.residual_template.spec_count
            for executor in self.executors.values()
        }:
            raise ValueError("root sparse Patches executor registry differs")
        self.input_bridge = input_bridge
        self.replace_output = replace_output
        self.execute_native = execute_native
        self.suffix_bridge = suffix_bridge
        self.projection_bridge = projection_bridge
        self.call_count = 0
        self.call_count_by_spec = {spec: 0 for spec in self.executors}
        self.fallback_count = 0
        self.lower_max_abs_diff = 0.0
        self.upper_max_abs_diff = 0.0
        self.lower_sign_exact = True
        self.upper_sign_exact = True
        self._last_executor: RootCrownSparsePatchesDualLaneTIRExecutorV1 | None = None

    @staticmethod
    def _selected_alpha(
        relu: Any,
        carrier: torch.Tensor,
        unstable_idx: tuple[torch.Tensor, ...],
        start_node: Any,
    ) -> torch.Tensor:
        selected, _lookup = relu.select_alpha_by_idx(
            carrier, carrier, unstable_idx, start_node
        )
        return _tensor(selected, f"{relu.name} selected alpha").contiguous()

    @contextmanager
    def install(self, bounded_module_type: type[Any]) -> Iterator[None]:
        original = bounded_module_type.backward_general
        signature = inspect.signature(original)

        @wraps(original)
        def wrapped(instance: Any, *args: Any, **kwargs: Any) -> Any:
            bound = signature.bind(instance, *args, **kwargs)
            bound.apply_defaults()
            values = bound.arguments
            node = values.get("bound_node")
            specification = values.get("C")
            unstable = values.get("unstable_idx")
            count = (
                int(unstable[0].numel())
                if isinstance(unstable, tuple)
                and len(unstable) == 3
                and torch.is_tensor(unstable[0])
                else -1
            )
            eligible = (
                str(getattr(node, "name", "")) == SPARSE_PATCHES_START_NODE
                and type(specification).__name__ == "Patches"
                and count in self.executors
                and values.get("bound_lower") is True
                and values.get("bound_upper") is True
            )
            if not eligible:
                return original(instance, *args, **kwargs)
            native = (
                original(instance, *args, **kwargs) if self.execute_native else None
            )
            try:
                candidate = self._execute(instance, values, count)
                if native is not None:
                    self._compare(native, candidate)
                else:
                    self._publish_state(instance)
            except Exception:
                self.fallback_count += 1
                raise
            self.call_count += 1
            self.call_count_by_spec[count] += 1
            return candidate if self.replace_output else native

        bounded_module_type.backward_general = wrapped
        try:
            yield
        finally:
            bounded_module_type.backward_general = original

    def _admit_patches(
        self,
        specification: Any,
        unstable_idx: tuple[torch.Tensor, ...],
        executor: RootCrownSparsePatchesDualLaneTIRExecutorV1,
    ) -> torch.Tensor:
        patches = patches_payload_v1(specification)
        if (
            getattr(specification, "unstable_idx", None) is not unstable_idx
            or tuple(getattr(specification, "shape", ()))
            != executor.seed.template.patches_shape
            or int(getattr(specification, "stride", -1)) != 1
            or getattr(specification, "padding", None) != 0
            or getattr(specification, "inserted_zeros", None) != 0
            or getattr(specification, "output_padding", None) != 0
            or tuple(getattr(specification, "output_shape", ())) != (1, 16, 8, 8)
        ):
            raise ValueError("root sparse Patches live carrier differs")
        return patches.contiguous()

    def _execute(
        self, instance: Any, values: Mapping[str, Any], count: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        specification = values.get("C")
        bound_node = values.get("bound_node")
        unstable_idx = _unstable_tuple(values.get("unstable_idx"))
        mismatches = {
            "lower": values.get("bound_lower") is not True,
            "upper": values.get("bound_upper") is not True,
            "average": bool(values.get("average_A")),
            "need_a": bool(values.get("need_A_only")),
            "start": values.get("start_backpropagation_at_node") is not None,
            "spec_kind": type(specification).__name__ != "Patches",
            "spec_count": count not in self.executors,
        }
        failed = sorted(name for name, mismatch in mismatches.items() if mismatch)
        if failed or bound_node is None:
            raise ValueError(
                "root sparse Patches live transaction differs: " + ",".join(failed)
            )
        executor = self.executors[count]
        patches = self._admit_patches(specification, unstable_idx, executor)
        nodes = {str(getattr(node, "name", "")): node for node in instance.nodes()}
        required = {
            "/44",
            "/43",
            "/input-24",
            "/input-20",
            "/input-16",
            "/39",
            "/37",
            "/38",
            "/input-12",
            "/input-8",
            "/input-4",
            "/input",
            "/input-1",
        }
        if not required <= nodes.keys():
            raise ValueError("root sparse Patches live topology differs")
        residual_inner = nodes["/input-24"]
        projection_entry = nodes["/input-16"]
        projection_inner = nodes["/input-12"]
        input_relu = nodes["/input-4"]
        self.input_bridge._admit_static(input_relu, nodes["/input-1"])
        center = self.input_bridge._input_center
        radius = self.input_bridge._input_radius
        if center is None or radius is None:
            raise RuntimeError("root sparse Patches input-domain state differs")
        carrier = executor.seed.selection_carrier
        tensors = RootCrownSparsePatchesDualLaneTensorsV1(
            patches,
            unstable_idx,
            _tensor(nodes["/43"].inputs[1].lower, "residual main weight"),
            _tensor(nodes["/43"].inputs[2].lower, "residual main bias"),
            _tensor(nodes["/input-20"].lower, "residual inner lower"),
            _tensor(nodes["/input-20"].upper, "residual inner upper"),
            self._selected_alpha(residual_inner, carrier, unstable_idx, bound_node),
            _tensor(nodes["/input-20"].inputs[1].lower, "residual inner weight"),
            _tensor(nodes["/input-20"].inputs[2].lower, "residual inner bias"),
            _tensor(nodes["/39"].lower, "projection entry lower"),
            _tensor(nodes["/39"].upper, "projection entry upper"),
            self._selected_alpha(projection_entry, carrier, unstable_idx, bound_node),
            _tensor(nodes["/37"].inputs[1].lower, "projection outer weight"),
            _tensor(nodes["/37"].inputs[2].lower, "projection outer bias"),
            _tensor(nodes["/input-8"].lower, "projection inner lower"),
            _tensor(nodes["/input-8"].upper, "projection inner upper"),
            self._selected_alpha(projection_inner, carrier, unstable_idx, bound_node),
            _tensor(nodes["/input-8"].inputs[1].lower, "projection inner weight"),
            _tensor(nodes["/input-8"].inputs[2].lower, "projection inner bias"),
            _tensor(nodes["/38"].inputs[1].lower, "projection skip weight"),
            _tensor(nodes["/38"].inputs[2].lower, "projection skip bias"),
            _tensor(nodes["/input"].lower, "input lower"),
            _tensor(nodes["/input"].upper, "input upper"),
            self._selected_alpha(input_relu, carrier, unstable_idx, bound_node),
            _tensor(nodes["/input"].inputs[1].lower, "input weight"),
            _tensor(nodes["/input"].inputs[2].lower, "input bias"),
            center,
            radius,
        )
        self._last_executor = executor
        return executor.execute(tensors)

    def _publish_state(self, instance: Any) -> None:
        if (
            self.suffix_bridge is None
            or self.projection_bridge is None
            or self._last_executor is None
        ):
            raise RuntimeError("root sparse Patches state publisher differs")
        nodes = {str(getattr(node, "name", "")): node for node in instance.nodes()}
        for node in nodes.values():
            node.lA = node.uA = None
        lower = self._last_executor.crown.lower
        residual_output = lower.residual._output_a
        projection_output = lower.projection._output_a
        if residual_output is None or projection_output is None:
            raise RuntimeError("root sparse Patches compiled state differs")
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
        if not isinstance(native, tuple) or len(native) < 2:
            raise ValueError("root sparse Patches native result differs")
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
        return {
            "schema_version": "boundflow.root-sparse-patches-live/v1",
            "replacement_seam": "BoundedModule.backward_general:/44",
            "mode": (
                "direct"
                if not self.execute_native
                else "replace" if self.replace_output else "shadow"
            ),
            "call_count": self.call_count,
            "call_count_by_spec": {
                str(key): value
                for key, value in sorted(self.call_count_by_spec.items())
            },
            "fallback_count": self.fallback_count,
            "native_execution_count": self.call_count if self.execute_native else 0,
            "lower_max_abs_diff": self.lower_max_abs_diff,
            "upper_max_abs_diff": self.upper_max_abs_diff,
            "lower_sign_exact": self.lower_sign_exact,
            "upper_sign_exact": self.upper_sign_exact,
            "executors": {
                str(key): value.receipt()
                for key, value in sorted(self.executors.items())
            },
            "performance_claimed": False,
        }


__all__ = ["RootCrownSparsePatchesLiveBridgeV1"]
