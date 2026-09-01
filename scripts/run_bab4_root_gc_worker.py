#!/usr/bin/env python3
"""Run symmetric GC control or cumulative root-CROWN plus BAB4 execution."""

# mypy: disable-error-code=import-untyped
# pylint: disable=import-error,wrong-import-position,import-outside-toplevel
# pylint: disable=too-many-locals,too-many-statements,protected-access,too-many-lines
# pylint: disable=too-many-boolean-expressions

from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
from dataclasses import replace
from functools import wraps
import hashlib
import inspect
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time
from types import MethodType
from typing import Any, Callable, cast, Iterator

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from boundflow.backends.tvm.root_crown_terminal_linear import (  # noqa: E402
    RootCrownTerminalLinearTemplateV1,
)
from boundflow.runtime.root_crown_full_pipeline_tir import (  # noqa: E402
    RootCrownFullPipelineTIRExecutorV1,
)
from boundflow.runtime.root_crown_backward_general_live import (  # noqa: E402
    RootCrownBackwardGeneralLiveBridgeV1,
)
from boundflow.runtime.root_crown_input_domain_live import (  # noqa: E402
    RootCrownInputDomainLiveBridgeV1,
)
from boundflow.runtime.root_crown_intermediate_dual_lane_tir import (  # noqa: E402
    RootCrownIntermediateDualLaneTIRExecutorV1,
)
from boundflow.runtime.root_crown_intermediate_live import (  # noqa: E402
    RootCrownIntermediateLiveBridgeV1,
)
from boundflow.runtime.root_crown_projection_live import (  # noqa: E402
    RootCrownProjectionLiveBridgeV1,
)
from boundflow.runtime.root_crown_suffix_live import (  # noqa: E402
    RootCrownSuffixLiveBridgeV1,
)
from scripts import run_asplos27_s4_same_solver_worker as s4_worker  # noqa: E402
from scripts import run_fsg4_b3_counter_diagnostic as diagnostic  # noqa: E402
from scripts.probe_root_crown_input_domain_tir import (  # noqa: E402
    _template as input_template,
)
from scripts.run_root_crown_expanded_live_worker import (  # noqa: E402
    DEFAULT_INPUT_CAPTURE,
    DEFAULT_PROJECTION_CAPTURE,
    DEFAULT_RESIDUAL_CAPTURE,
    _projection_template,
)
from scripts.run_root_crown_residual_live_worker import (  # noqa: E402
    _template as residual_template,
)
from scripts.run_root_crown_terminal_live_worker import FEATURE_INDICES  # noqa: E402

SCHEMA = "boundflow.bab4-root-gc-worker/v1"
CONTROL = "B4-A-GC"
CANDIDATE = "BAB4-GC-ROOT"


class _RootSegmentAttributionV1:
    """Measure host submission and same-stream CUDA spans outside formal timing."""

    def __init__(self) -> None:
        self._records: dict[str, list[tuple[int, Any, Any]]] = {}

    def _instrument_method(self, owner: Any, name: str, label: str) -> None:
        import torch

        original = cast(Callable[..., Any], getattr(owner, name))

        @wraps(original)
        def measured(*args: Any, **kwargs: Any) -> Any:
            device = torch.cuda.current_device()
            stream = torch.cuda.current_stream(device)
            start = torch.cuda.Event(enable_timing=True)
            finish = torch.cuda.Event(enable_timing=True)
            started_ns = time.perf_counter_ns()
            start.record(stream)
            try:
                return original(*args, **kwargs)
            finally:
                finish.record(stream)
                host_ns = time.perf_counter_ns() - started_ns
                self._records.setdefault(label, []).append((host_ns, start, finish))

        setattr(owner, name, measured)

    def install(
        self,
        executor: RootCrownFullPipelineTIRExecutorV1,
        suffix_bridge: RootCrownSuffixLiveBridgeV1,
        projection_bridge: RootCrownProjectionLiveBridgeV1,
        input_bridge: RootCrownInputDomainLiveBridgeV1,
    ) -> None:
        """Instrument cumulative boundaries, TIR modules, and admission setup."""

        for name in ("stage_terminal", "stage_residual", "stage_projection", "consume"):
            self._instrument_method(executor, name, f"pipeline.{name}")
        self._instrument_method(executor, "backward", "pipeline.backward")
        for label, executor_owner in (
            ("terminal", executor.expanded.suffix.terminal),
            ("residual", executor.expanded.suffix.residual),
            ("projection", executor.expanded.projection),
            ("input_domain", executor.input_domain),
        ):
            self._instrument_method(executor_owner, "forward", f"{label}.forward")
            self._instrument_method(executor_owner, "backward", f"{label}.backward")
        for label, bridge_owner in (
            ("suffix", suffix_bridge),
            ("projection", projection_bridge),
            ("input_domain", input_bridge),
        ):
            self._instrument_method(
                bridge_owner, "_admit_static", f"{label}.admit_static"
            )
            self._instrument_method(
                bridge_owner, "_set_relu_state", f"{label}.set_relu_state"
            )

    def receipt(self) -> dict[str, object]:
        """Synchronize after the query and serialize all diagnostic intervals."""

        import torch

        torch.cuda.synchronize()
        segments: dict[str, object] = {}
        event_pair_count = 0
        for label, records in sorted(self._records.items()):
            host_values = [record[0] for record in records]
            cuda_values = [
                round(float(record[1].elapsed_time(record[2])) * 1_000_000)
                for record in records
            ]
            event_pair_count += len(records)
            segments[label] = {
                "count": len(records),
                "host_total_ns": sum(host_values),
                "host_median_ns": round(statistics.median(host_values)),
                "cuda_total_ns": sum(cuda_values),
                "cuda_median_ns": round(statistics.median(cuda_values)),
            }
        return {
            "schema_version": "boundflow.root-segment-attribution/v1",
            "diagnostic_only": True,
            "included_in_performance_claim": False,
            "event_pair_count": event_pair_count,
            "segments": segments,
        }


def _tensor_digest(tensor: Any) -> str:
    """Hash one diagnostic tensor without retaining a solver graph."""

    payload = tensor.detach().contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _coefficient_summary(value: Any) -> dict[str, object] | None:
    """Describe Tensor/Patches/OneHot coefficient carriers without payload copies."""

    if value is None:
        return None
    summary: dict[str, object] = {"kind": type(value).__name__}
    shape = getattr(value, "shape", None)
    if shape is not None:
        summary["shape"] = list(shape)
    patches = getattr(value, "patches", None)
    if patches is not None:
        summary["patches_shape"] = list(patches.shape)
    for name in ("stride", "padding", "output_shape", "unstable_idx"):
        field = getattr(value, name, None)
        if field is None:
            continue
        if hasattr(field, "shape"):
            summary[f"{name}_shape"] = list(field.shape)
        elif isinstance(field, (bool, int, float, str)):
            summary[name] = field
        elif isinstance(field, (tuple, list)):
            summary[name] = repr(field)
    return summary


class _RootComputeTransactionCaptureV1:
    """Capture the five optimized compute_bounds calls at their public seam."""

    def __init__(
        self,
        executor: RootCrownFullPipelineTIRExecutorV1,
        suffix_bridge: RootCrownSuffixLiveBridgeV1,
    ) -> None:
        self.executor = executor
        self.suffix_bridge = suffix_bridge
        self.rows: list[dict[str, object]] = []
        self.backward_rows: list[dict[str, object]] = []
        self.intermediate_backward_rows: list[dict[str, object]] = []
        self.intermediate_node_rows: list[dict[str, object]] = []

    def _run_intermediate_backward(
        self,
        instance: Any,
        original_backward: Callable[..., Any],
        bound_node: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """Capture the exact node sequence and coefficient carriers for one start."""

        import torch

        originals = {node: node.bound_backward for node in instance.nodes()}
        transaction = len(self.intermediate_backward_rows)

        def replacement(
            node: Any, original_node: Callable[..., Any]
        ) -> Callable[..., Any]:
            def observed(_self: Any, *node_args: Any, **node_kwargs: Any) -> Any:
                result = original_node(*node_args, **node_kwargs)
                outputs: list[object] = []
                if isinstance(result, tuple) and result and isinstance(result[0], list):
                    for pair in result[0]:
                        if isinstance(pair, tuple) and len(pair) == 2:
                            outputs.append(
                                [
                                    _coefficient_summary(pair[0]),
                                    _coefficient_summary(pair[1]),
                                ]
                            )
                start_name = str(getattr(bound_node, "name", ""))
                selected_alpha = getattr(node, "alpha", {}).get(start_name)
                alpha_indices = getattr(node, "alpha_indices", None)
                serialized_indices = None
                if (
                    isinstance(alpha_indices, tuple)
                    and len(alpha_indices) == 3
                    and all(torch.is_tensor(value) for value in alpha_indices)
                ):
                    serialized_indices = [
                        cast(Any, value).detach().cpu().tolist()
                        for value in alpha_indices
                    ]
                self.intermediate_node_rows.append(
                    {
                        "transaction": transaction,
                        "ordinal": len(self.intermediate_node_rows),
                        "start_node_name": start_name,
                        "node_name": str(getattr(node, "name", "")),
                        "node_kind": type(node).__name__,
                        "incoming_lower": _coefficient_summary(
                            node_args[0] if node_args else None
                        ),
                        "incoming_upper": _coefficient_summary(
                            node_args[1] if len(node_args) > 1 else None
                        ),
                        "outputs": outputs,
                        "selected_alpha_shape": (
                            list(selected_alpha.shape)
                            if torch.is_tensor(selected_alpha)
                            else None
                        ),
                        "selected_alpha_digest": (
                            _tensor_digest(selected_alpha)
                            if torch.is_tensor(selected_alpha)
                            else None
                        ),
                        "alpha_indices": serialized_indices,
                        "lower_bias_shape": (
                            list(result[1].shape)
                            if isinstance(result, tuple)
                            and len(result) > 1
                            and hasattr(result[1], "shape")
                            else None
                        ),
                        "upper_bias_shape": (
                            list(result[2].shape)
                            if isinstance(result, tuple)
                            and len(result) > 2
                            and hasattr(result[2], "shape")
                            else None
                        ),
                    }
                )
                return result

            return observed

        for node, original_node in originals.items():
            node.bound_backward = MethodType(replacement(node, original_node), node)
        try:
            return original_backward(instance, *args, **kwargs)
        finally:
            for node, original_node in originals.items():
                node.bound_backward = original_node

    @contextmanager
    def install(self, bounded_module_type: type[Any]) -> Iterator[None]:
        """Patch only compute_bounds and capture calls owned by the root bridge."""

        import torch

        original = bounded_module_type.compute_bounds
        signature = inspect.signature(original)
        original_backward = bounded_module_type.backward_general
        backward_signature = inspect.signature(original_backward)

        @wraps(original)
        def captured(instance: Any, *args: Any, **kwargs: Any) -> Any:
            if not self.suffix_bridge._active:
                return original(instance, *args, **kwargs)
            bound = signature.bind(instance, *args, **kwargs)
            bound.apply_defaults()
            parameters = bound.arguments
            specification = parameters.get("C")
            if not torch.is_tensor(specification):
                raise TypeError("root compute transaction specification differs")
            result = original(instance, *args, **kwargs)
            nodes = {str(getattr(node, "name", "")): node for node in instance.nodes()}
            final = nodes.get("/49")
            suffix = self.suffix_bridge._last_suffix
            concrete = self.executor.input_domain._concrete_lower
            pipeline_bias = self.executor._final_bias
            if (
                final is None
                or len(getattr(final, "inputs", ())) != 3
                or suffix is None
                or concrete is None
                or pipeline_bias is None
                or not isinstance(result, tuple)
                or not result
                or not torch.is_tensor(result[0])
            ):
                raise ValueError("root compute transaction state differs")
            final_weight = final.inputs[1].lower
            final_bias = final.inputs[2].lower
            if not torch.is_tensor(final_weight) or not torch.is_tensor(final_bias):
                raise TypeError("root compute transaction final affine differs")
            terminal_seed = torch.matmul(specification, final_weight).transpose(0, 1)
            captured_seed = suffix.terminal.incoming_lower_a
            final_bias_term = torch.matmul(specification, final_bias)
            direct_lower = concrete + pipeline_bias.transpose(0, 1) + final_bias_term
            returned_lower = result[0]
            seed_diff = float(
                (terminal_seed - captured_seed).abs().max().detach().cpu().item()
            )
            lower_diff = float(
                (direct_lower - returned_lower).abs().max().detach().cpu().item()
            )
            self.rows.append(
                {
                    "ordinal": len(self.rows),
                    "method": str(parameters.get("method")),
                    "bound_lower": bool(parameters.get("bound_lower")),
                    "bound_upper": bool(parameters.get("bound_upper")),
                    "return_A": bool(parameters.get("return_A")),
                    "average_A": bool(parameters.get("average_A")),
                    "need_A_only": bool(parameters.get("need_A_only")),
                    "final_node_name": parameters.get("final_node_name"),
                    "specification_shape": list(specification.shape),
                    "specification_digest": _tensor_digest(specification),
                    "terminal_seed_shape": list(terminal_seed.shape),
                    "terminal_seed_max_abs_diff": seed_diff,
                    "returned_lower_shape": list(returned_lower.shape),
                    "direct_lower_max_abs_diff": lower_diff,
                    "returned_lower_requires_grad": bool(returned_lower.requires_grad),
                    "returned_upper_is_none": result[1] is None,
                    "result_arity": len(result),
                }
            )
            return result

        @wraps(original_backward)
        def captured_backward(instance: Any, *args: Any, **kwargs: Any) -> Any:
            if not self.suffix_bridge._active:
                return original_backward(instance, *args, **kwargs)
            bound = backward_signature.bind(instance, *args, **kwargs)
            bound.apply_defaults()
            parameters = bound.arguments
            specification = parameters.get("C")
            bound_node = parameters.get("bound_node")
            # ``check_prior_bounds`` may issue nested backward traversals for
            # intermediate nodes while the outer root transaction is active.
            # They remain native work and are not the replacement seam.  The
            # root transaction itself is the unique traversal from /49.
            if str(getattr(bound_node, "name", "")) != "/49":
                result = self._run_intermediate_backward(
                    instance,
                    original_backward,
                    bound_node,
                    args,
                    kwargs,
                )
                if not isinstance(result, tuple) or len(result) < 2:
                    raise ValueError("root intermediate backward result differs")
                unstable = parameters.get("unstable_idx")
                if isinstance(unstable, tuple):
                    unstable_count = int(unstable[0].numel())
                elif torch.is_tensor(unstable):
                    unstable_count = int(unstable.numel())
                elif unstable is None:
                    unstable_count = None
                else:
                    unstable_count = len(unstable)
                self.intermediate_backward_rows.append(
                    {
                        "ordinal": len(self.intermediate_backward_rows),
                        "bound_node_name": str(getattr(bound_node, "name", "")),
                        "bound_node_kind": type(bound_node).__name__,
                        "specification_kind": type(specification).__name__,
                        "specification_shape": (
                            list(cast(Any, specification).shape)
                            if hasattr(specification, "shape")
                            else None
                        ),
                        "bound_lower": bool(parameters.get("bound_lower")),
                        "bound_upper": bool(parameters.get("bound_upper")),
                        "unstable_count": unstable_count,
                        "returned_lower_shape": (
                            list(result[0].shape)
                            if torch.is_tensor(result[0])
                            else None
                        ),
                        "returned_upper_shape": (
                            list(result[1].shape)
                            if torch.is_tensor(result[1])
                            else None
                        ),
                        "returned_lower_requires_grad": (
                            bool(result[0].requires_grad)
                            if torch.is_tensor(result[0])
                            else False
                        ),
                        "returned_upper_requires_grad": (
                            bool(result[1].requires_grad)
                            if torch.is_tensor(result[1])
                            else False
                        ),
                    }
                )
                return result
            if not torch.is_tensor(specification):
                raise TypeError("root backward transaction specification differs")
            result = original_backward(instance, *args, **kwargs)
            if (
                not isinstance(result, tuple)
                or len(result) != 2
                or not torch.is_tensor(result[0])
            ):
                raise ValueError("root backward transaction result differs")
            self.backward_rows.append(
                {
                    "ordinal": len(self.backward_rows),
                    "bound_node_name": str(getattr(bound_node, "name", "")),
                    "start_node_name": str(
                        getattr(
                            parameters.get("start_backpropagation_at_node"),
                            "name",
                            "",
                        )
                    ),
                    "bound_lower": bool(parameters.get("bound_lower")),
                    "bound_upper": bool(parameters.get("bound_upper")),
                    "average_A": bool(parameters.get("average_A")),
                    "need_A_only": bool(parameters.get("need_A_only")),
                    "unstable_idx_is_none": parameters.get("unstable_idx") is None,
                    "update_mask_is_none": parameters.get("update_mask") is None,
                    "output_constraints_is_none": (
                        parameters.get("apply_output_constraints_to") is None
                    ),
                    "initial_As_is_none": parameters.get("initial_As") is None,
                    "initial_lb_is_none": parameters.get("initial_lb") is None,
                    "initial_ub_is_none": parameters.get("initial_ub") is None,
                    "specification_shape": list(specification.shape),
                    "specification_digest": _tensor_digest(specification),
                    "returned_lower_shape": list(result[0].shape),
                    "returned_upper_is_none": result[1] is None,
                }
            )
            return result

        bounded_module_type.compute_bounds = captured
        bounded_module_type.backward_general = captured_backward
        try:
            yield
        finally:
            bounded_module_type.backward_general = original_backward
            bounded_module_type.compute_bounds = original

    def receipt(self) -> dict[str, object]:
        """Return diagnostic evidence without upgrading a performance claim."""

        if (
            len(self.rows) != 5
            or len(self.backward_rows) != 5
            or [row["ordinal"] for row in self.rows] != list(range(5))
            or [row["ordinal"] for row in self.backward_rows] != list(range(5))
        ):
            raise ValueError("root compute transaction capture count differs")
        return {
            "schema_version": "boundflow.root-compute-transaction-capture/v1",
            "evaluation_count": len(self.rows),
            "rows": self.rows,
            "backward_general_count": len(self.backward_rows),
            "backward_general_rows": self.backward_rows,
            "intermediate_backward_general_count": len(self.intermediate_backward_rows),
            "intermediate_backward_general_rows": self.intermediate_backward_rows,
            "intermediate_node_call_count": len(self.intermediate_node_rows),
            "intermediate_node_rows": self.intermediate_node_rows,
            "terminal_seed_max_abs_diff": max(
                cast(float, row["terminal_seed_max_abs_diff"]) for row in self.rows
            ),
            "direct_lower_max_abs_diff": max(
                cast(float, row["direct_lower_max_abs_diff"]) for row in self.rows
            ),
            "diagnostic_only": True,
            "included_in_performance_claim": False,
            "performance_claimed": False,
        }


class _RootPriorBoundsAttributionV1:
    """Attribute repeated intermediate-bound construction by production node."""

    def __init__(self, suffix_bridge: RootCrownSuffixLiveBridgeV1) -> None:
        self.suffix_bridge = suffix_bridge
        self.rows: list[dict[str, object]] = []
        self._events: list[tuple[Any, Any]] = []

    @contextmanager
    def install(self, bounded_module_type: type[Any]) -> Iterator[None]:
        """Observe only intermediate bounds inside the admitted root transaction."""

        import torch

        original = bounded_module_type.compute_intermediate_bounds
        signature = inspect.signature(original)

        @wraps(original)
        def captured(instance: Any, *args: Any, **kwargs: Any) -> Any:
            if not self.suffix_bridge._active:
                return original(instance, *args, **kwargs)
            bound = signature.bind(instance, *args, **kwargs)
            bound.apply_defaults()
            node = bound.arguments.get("node")
            stream = torch.cuda.current_stream()
            started = torch.cuda.Event(enable_timing=True)
            finished = torch.cuda.Event(enable_timing=True)
            host_started_ns = time.perf_counter_ns()
            started.record(stream)
            result = original(instance, *args, **kwargs)
            finished.record(stream)
            host_ns = time.perf_counter_ns() - host_started_ns
            lower = getattr(node, "lower", None)
            upper = getattr(node, "upper", None)
            self.rows.append(
                {
                    "ordinal": len(self.rows),
                    "node_name": str(getattr(node, "name", "")),
                    "node_kind": type(node).__name__,
                    "prior_checked": bool(bound.arguments.get("prior_checked")),
                    "host_ns": host_ns,
                    "lower_shape": (
                        list(lower.shape) if torch.is_tensor(lower) else None
                    ),
                    "upper_shape": (
                        list(upper.shape) if torch.is_tensor(upper) else None
                    ),
                    "lower_requires_grad": (
                        bool(lower.requires_grad) if torch.is_tensor(lower) else False
                    ),
                    "upper_requires_grad": (
                        bool(upper.requires_grad) if torch.is_tensor(upper) else False
                    ),
                    "lower_digest": (
                        _tensor_digest(lower) if torch.is_tensor(lower) else None
                    ),
                    "upper_digest": (
                        _tensor_digest(upper) if torch.is_tensor(upper) else None
                    ),
                }
            )
            self._events.append((started, finished))
            return result

        bounded_module_type.compute_intermediate_bounds = captured
        try:
            yield
        finally:
            bounded_module_type.compute_intermediate_bounds = original

    def receipt(self) -> dict[str, object]:
        """Synchronize events and aggregate repeated value/gradient ownership."""

        import torch

        torch.cuda.synchronize()
        by_node: dict[str, dict[str, object]] = {}
        for row, (started, finished) in zip(self.rows, self._events):
            row["cuda_ns"] = round(float(started.elapsed_time(finished)) * 1_000_000)
            name = cast(str, row["node_name"])
            aggregate = by_node.setdefault(
                name,
                {
                    "node_kind": row["node_kind"],
                    "call_count": 0,
                    "host_total_ns": 0,
                    "cuda_total_ns": 0,
                    "lower_digests": set(),
                    "upper_digests": set(),
                    "lower_requires_grad": False,
                    "upper_requires_grad": False,
                },
            )
            aggregate["call_count"] = cast(int, aggregate["call_count"]) + 1
            aggregate["host_total_ns"] = cast(int, aggregate["host_total_ns"]) + cast(
                int, row["host_ns"]
            )
            aggregate["cuda_total_ns"] = cast(int, aggregate["cuda_total_ns"]) + cast(
                int, row["cuda_ns"]
            )
            cast(set[object], aggregate["lower_digests"]).add(row["lower_digest"])
            cast(set[object], aggregate["upper_digests"]).add(row["upper_digest"])
            aggregate["lower_requires_grad"] = bool(
                aggregate["lower_requires_grad"] or row["lower_requires_grad"]
            )
            aggregate["upper_requires_grad"] = bool(
                aggregate["upper_requires_grad"] or row["upper_requires_grad"]
            )
        serializable: dict[str, object] = {}
        for name, aggregate in sorted(by_node.items()):
            item = dict(aggregate)
            item["lower_distinct_digest_count"] = len(
                cast(set[object], item.pop("lower_digests"))
            )
            item["upper_distinct_digest_count"] = len(
                cast(set[object], item.pop("upper_digests"))
            )
            serializable[name] = item
        return {
            "schema_version": "boundflow.root-prior-bounds-attribution/v1",
            "call_count": len(self.rows),
            "rows": self.rows,
            "by_node": serializable,
            "diagnostic_only": True,
            "included_in_performance_claim": False,
            "performance_claimed": False,
        }


def _prepare_root_pipeline(
    args: argparse.Namespace,
) -> tuple[Any, Any, Any, Any, Any, int]:
    import torch

    residual = residual_template(args.residual_capture)
    terminal = RootCrownTerminalLinearTemplateV1(
        spec_count=residual.spec_count,
        domain_count=residual.domain_count,
        current_features=100,
        previous_features=(residual.channels * residual.height * residual.width),
        alpha_feature_indices=FEATURE_INDICES,
        compute_capability=residual.compute_capability,
        thread_extent=128,
    )
    projection = _projection_template(args.projection_capture)
    input_payload = torch.load(
        args.input_capture, map_location="cpu", weights_only=True
    )
    evaluations = cast(list[dict[str, Any]], input_payload.get("evaluations"))
    if (
        input_payload.get("schema_version") != "boundflow.root-crown-input-tensors/v1"
        or len(evaluations) != 5
    ):
        raise ValueError("BAB4 cumulative root input capture differs")
    started_ns = time.perf_counter_ns()
    root_input_template = input_template(evaluations[0])
    executor = RootCrownFullPipelineTIRExecutorV1(
        terminal, residual, projection, root_input_template
    )
    suffix_bridge = RootCrownSuffixLiveBridgeV1(terminal, residual, executor)
    projection_bridge = RootCrownProjectionLiveBridgeV1(projection, executor)
    input_bridge = RootCrownInputDomainLiveBridgeV1(executor.input_template, executor)
    executor.prepare()
    intermediate_executor = None
    if (
        args.shadow_root_intermediate
        or args.replace_root_intermediate
        or args.direct_root_intermediate
    ):
        intermediate_executor = RootCrownIntermediateDualLaneTIRExecutorV1(
            replace(residual, spec_count=27),
            replace(projection, spec_count=27),
            replace(root_input_template, spec_count=27),
        )
        intermediate_executor.prepare()
    prepare_ns = time.perf_counter_ns() - started_ns
    return (
        executor,
        suffix_bridge,
        projection_bridge,
        input_bridge,
        intermediate_executor,
        prepare_ns,
    )


def _worker(args: argparse.Namespace) -> None:
    if args.configuration not in (CONTROL, CANDIDATE):
        raise ValueError("BAB4 cumulative root configuration differs")
    sys.path.insert(0, str(args.abcrown_root / "complete_verifier"))
    sys.path.insert(0, str(args.abcrown_root))
    from auto_LiRPA import BoundedModule
    from boundflow.runtime import prepared_root_optimizer_warmup as warmup_module

    candidate = args.configuration == CANDIDATE
    warm_suffix = warm_projection = warm_input_domain = None
    suffix = projection = input_domain = None
    warm_executor: RootCrownFullPipelineTIRExecutorV1 | None = None
    root_warm_pipeline_prepare_ns = 0
    root_query_pipeline_prepare_ns = 0
    root_exact_warmup_reset_ns = 0
    root_segment_observer: _RootSegmentAttributionV1 | None = None
    root_compute_capture: _RootComputeTransactionCaptureV1 | None = None
    root_direct_bridge: RootCrownBackwardGeneralLiveBridgeV1 | None = None
    root_intermediate_bridge: RootCrownIntermediateLiveBridgeV1 | None = None
    root_prior_attribution: _RootPriorBoundsAttributionV1 | None = None
    warm_intermediate_executor = None
    if candidate:
        (
            warm_executor,
            warm_suffix,
            warm_projection,
            warm_input_domain,
            warm_intermediate_executor,
            root_warm_pipeline_prepare_ns,
        ) = _prepare_root_pipeline(args)

    with tempfile.TemporaryDirectory(prefix="boundflow-bab4-root-gc-") as raw:
        base_result = Path(raw) / "worker.json"
        base_args = argparse.Namespace(
            configuration="BAB4-GC" if candidate else "B4-A-GC",
            mode="control",
            run_id=args.run_id,
            block_index=args.block_index,
            sequence_position=args.sequence_position,
            benchmark_root=args.benchmark_root,
            abcrown_root=args.abcrown_root,
            model=args.model,
            property=args.property,
            result=base_result,
            attribute_root_incomplete=True,
            attribute_complete_prelude=True,
        )
        root_query_install_count = 0
        root_warmup_receipts: dict[str, object] | None = None
        with ExitStack() as stack:
            if candidate:
                assert warm_suffix is not None
                assert warm_projection is not None
                assert warm_input_domain is not None
                original_warmup = warmup_module.prepare_root_optimizer_warmup_v1

                def warm_then_install(*warm_args: Any, **warm_kwargs: Any) -> Any:
                    nonlocal root_query_install_count, root_warmup_receipts
                    nonlocal root_query_pipeline_prepare_ns
                    nonlocal root_exact_warmup_reset_ns
                    nonlocal suffix, projection, input_domain
                    nonlocal root_segment_observer
                    nonlocal root_compute_capture
                    nonlocal root_direct_bridge
                    nonlocal root_intermediate_bridge
                    nonlocal root_prior_attribution
                    if root_query_install_count != 0:
                        raise RuntimeError("BAB4 cumulative root install count differs")
                    with ExitStack() as warm_stack:
                        warm_stack.enter_context(warm_suffix.install(BoundedModule))
                        warm_stack.enter_context(warm_projection.install(BoundedModule))
                        warm_stack.enter_context(
                            warm_input_domain.install(BoundedModule)
                        )
                        receipt = original_warmup(*warm_args, **warm_kwargs)
                        root_warmup_receipts = {
                            "suffix": warm_suffix.receipt(),
                            "projection": warm_projection.receipt(),
                            "input_domain": warm_input_domain.receipt(),
                        }
                    if warm_executor is None:
                        raise RuntimeError("BAB4 cumulative root warm executor absent")
                    reset_started_ns = time.perf_counter_ns()
                    warm_suffix.reset_after_exact_warmup_v1()
                    warm_projection.reset_after_exact_warmup_v1()
                    warm_input_domain.reset_after_exact_warmup_v1()
                    warm_executor.reset_after_exact_warmup_v1()
                    root_exact_warmup_reset_ns = (
                        time.perf_counter_ns() - reset_started_ns
                    )
                    suffix = warm_suffix
                    projection = warm_projection
                    input_domain = warm_input_domain
                    if args.attribute_root_segments:
                        root_segment_observer = _RootSegmentAttributionV1()
                        root_segment_observer.install(
                            warm_executor, suffix, projection, input_domain
                        )
                    if args.capture_root_compute_transaction:
                        root_compute_capture = _RootComputeTransactionCaptureV1(
                            warm_executor, suffix
                        )
                        stack.enter_context(root_compute_capture.install(BoundedModule))
                    if args.direct_root_backward:
                        root_direct_bridge = RootCrownBackwardGeneralLiveBridgeV1(
                            warm_executor, suffix, projection, input_domain
                        )
                        stack.enter_context(root_direct_bridge.install(BoundedModule))
                    if (
                        args.shadow_root_intermediate
                        or args.replace_root_intermediate
                        or args.direct_root_intermediate
                    ):
                        if warm_intermediate_executor is None:
                            raise RuntimeError("BAB4 root intermediate executor absent")
                        root_intermediate_bridge = RootCrownIntermediateLiveBridgeV1(
                            warm_intermediate_executor,
                            input_domain,
                            replace_output=(
                                args.replace_root_intermediate
                                or args.direct_root_intermediate
                            ),
                            execute_native=not args.direct_root_intermediate,
                            suffix_bridge=suffix,
                            projection_bridge=projection,
                        )
                        stack.enter_context(
                            root_intermediate_bridge.install(BoundedModule)
                        )
                    if args.attribute_root_prior_bounds:
                        root_prior_attribution = _RootPriorBoundsAttributionV1(suffix)
                        stack.enter_context(
                            root_prior_attribution.install(BoundedModule)
                        )
                    stack.enter_context(suffix.install(BoundedModule))
                    stack.enter_context(projection.install(BoundedModule))
                    stack.enter_context(input_domain.install(BoundedModule))
                    root_query_install_count += 1
                    return receipt

                stack.enter_context(
                    diagnostic._patch_attribute(
                        warmup_module,
                        "prepare_root_optimizer_warmup_v1",
                        warm_then_install,
                    )
                )
            s4_worker._worker(base_args)
            if candidate:
                assert suffix is not None
                assert projection is not None
                assert input_domain is not None
                root_receipts = {
                    "suffix": suffix.receipt(),
                    "projection": projection.receipt(),
                    "input_domain": input_domain.receipt(),
                    "backward_general": (
                        root_direct_bridge.receipt()
                        if root_direct_bridge is not None
                        else None
                    ),
                    "intermediate": (
                        root_intermediate_bridge.receipt()
                        if root_intermediate_bridge is not None
                        else None
                    ),
                }
            else:
                root_receipts = None
        base = json.loads(base_result.read_text(encoding="utf-8"))

    root_segment_attribution = (
        root_segment_observer.receipt() if root_segment_observer is not None else None
    )
    root_compute_transaction_capture = (
        root_compute_capture.receipt() if root_compute_capture is not None else None
    )
    root_prior_bounds_attribution = (
        root_prior_attribution.receipt() if root_prior_attribution is not None else None
    )

    if root_query_install_count != int(candidate):
        raise ValueError("BAB4 cumulative root query installation differs")
    if base.get("performance_claimed") is not False:
        raise ValueError("BAB4 cumulative root base claim differs")
    payload = dict(base)
    payload.update(
        {
            "schema_version": SCHEMA,
            "configuration": args.configuration,
            "base_configuration": "BAB4-GC" if candidate else "B4-A-GC",
            "root_warm_pipeline_prepare_ns": root_warm_pipeline_prepare_ns,
            "root_query_pipeline_prepare_ns": root_query_pipeline_prepare_ns,
            "root_exact_warmup_reset_ns": root_exact_warmup_reset_ns,
            "root_total_prepare_ns": (
                root_warm_pipeline_prepare_ns + root_query_pipeline_prepare_ns
            ),
            "root_pipeline_prepare_excluded_from_query": True,
            "root_prepared_runtime_reused_after_exact_warmup": bool(candidate),
            "root_query_install_count": root_query_install_count,
            "root_warmup_receipts": root_warmup_receipts,
            "root_receipts": root_receipts,
            "root_segment_attribution": root_segment_attribution,
            "root_segment_attribution_enabled": bool(
                candidate and args.attribute_root_segments
            ),
            "root_compute_transaction_capture": root_compute_transaction_capture,
            "root_compute_transaction_capture_enabled": bool(
                candidate and args.capture_root_compute_transaction
            ),
            "root_direct_backward_enabled": bool(
                candidate and args.direct_root_backward
            ),
            "root_intermediate_shadow_enabled": bool(
                candidate and args.shadow_root_intermediate
            ),
            "root_intermediate_replace_enabled": bool(
                candidate and args.replace_root_intermediate
            ),
            "root_intermediate_direct_enabled": bool(
                candidate and args.direct_root_intermediate
            ),
            "root_prior_bounds_attribution": root_prior_bounds_attribution,
            "root_prior_bounds_attribution_enabled": bool(
                candidate and args.attribute_root_prior_bounds
            ),
            "performance_claimed": False,
        }
    )
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", choices=(CONTROL, CANDIDATE), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--block-index", type=int, required=True)
    parser.add_argument("--sequence-position", type=int, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--abcrown-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--property", type=Path, required=True)
    parser.add_argument(
        "--residual-capture", type=Path, default=DEFAULT_RESIDUAL_CAPTURE
    )
    parser.add_argument(
        "--projection-capture", type=Path, default=DEFAULT_PROJECTION_CAPTURE
    )
    parser.add_argument("--input-capture", type=Path, default=DEFAULT_INPUT_CAPTURE)
    parser.add_argument("--attribute-root-segments", action="store_true")
    parser.add_argument("--capture-root-compute-transaction", action="store_true")
    parser.add_argument("--direct-root-backward", action="store_true")
    parser.add_argument("--shadow-root-intermediate", action="store_true")
    parser.add_argument("--replace-root-intermediate", action="store_true")
    parser.add_argument("--direct-root-intermediate", action="store_true")
    parser.add_argument("--attribute-root-prior-bounds", action="store_true")
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    intermediate_modes = sum(
        bool(value)
        for value in (
            args.shadow_root_intermediate,
            args.replace_root_intermediate,
            args.direct_root_intermediate,
        )
    )
    if intermediate_modes > 1:
        parser.error("root intermediate modes are exclusive")
    return args


def main() -> None:
    """Run one cumulative same-solver worker."""

    args = _parse_args()
    for name in (
        "benchmark_root",
        "abcrown_root",
        "model",
        "property",
        "residual_capture",
        "projection_capture",
        "input_capture",
        "result",
    ):
        setattr(args, name, getattr(args, name).resolve())
    _worker(args)


if __name__ == "__main__":
    main()
