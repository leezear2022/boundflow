#!/usr/bin/env python3
"""Run one no-timing production CROWN Conv-site census worker."""

# pylint: disable=too-many-lines,too-many-locals,too-many-statements
# pylint: disable=too-many-branches,import-outside-toplevel,protected-access
# pylint: disable=missing-function-docstring,line-too-long,import-error
# pylint: disable=wrong-import-position,too-many-instance-attributes
# pylint: disable=too-few-public-methods,too-many-boolean-expressions

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boundflow.runtime.mr3_provider_hook_feasibility import (  # noqa: E402
    ABCROWN_COMMIT,
    AUTO_LIRPA_COMMIT,
    VNNCOMP_COMMIT,
    canonical_hash,
)
from scripts.run_mr3_provider_hook_feasibility import (  # noqa: E402
    MODEL_SHA256,
    PROPERTY_SHA256,
    _extract_lower_a,
    _extract_lower_bias,
    _git,
    _module_state,
    _phase_from_stack,
    _result_state,
    _sha256,
    _target_alpha,
    _visited_domains,
    _walk_tensor_values,
)

WORKER_SCHEMA = "boundflow.mr4-production-conv-site-census-worker/v1"
TARGET_START = "/49"
FROZEN_EDGES = (
    ("C0", "/input-4", "/input"),
    ("C1", "/input-12", "/input-8"),
    ("C2", "/input-24", "/input-20"),
)


def _tensor_meta(value: Any, torch_module: Any) -> dict[str, object]:
    if not torch_module.is_tensor(value):
        raise TypeError(f"MR4 expected tensor, got {type(value)!r}")
    return {
        "shape": [int(dimension) for dimension in value.shape],
        "stride": [int(dimension) for dimension in value.stride()],
        "dtype": str(value.dtype),
        "device": str(value.device),
        "requires_grad": bool(value.requires_grad),
        "numel": int(value.numel()),
        "element_size": int(value.element_size()),
        "contiguous": bool(value.is_contiguous()),
    }


def _finite(value: Any, torch_module: Any) -> bool:
    return bool(torch_module.isfinite(value).all().item())


def _beta_tensors(relu: Any, conv: Any, torch_module: Any) -> list[Any]:
    tensors = []
    seen: set[int] = set()
    for owner in (relu, conv):
        for attribute in ("sparse_betas", "beta", "split_beta"):
            for tensor in _walk_tensor_values(
                getattr(owner, attribute, None), torch_module
            ):
                identity = id(tensor)
                if identity not in seen:
                    seen.add(identity)
                    tensors.append(tensor)
    return tensors


class _SiteCensus:
    def __init__(self, torch_module: Any) -> None:
        self.torch = torch_module
        self.current_evaluation: int | None = None
        self.topology: list[dict[str, object]] = []
        self.rows: list[dict[str, object]] = []
        self.pending: dict[str, dict[str, Any]] = {}
        self.relu_calls = {site: 0 for site, _, _ in FROZEN_EDGES}
        self.conv_calls = {site: 0 for site, _, _ in FROZEN_EDGES}
        self.unexpected_target_start_calls = 0
        self.device_before = int(torch_module.cuda.current_device())
        self.stream_before = int(torch_module.cuda.current_stream().cuda_stream)
        self.device_after: int | None = None
        self.stream_after: int | None = None

    def _topology(self, instance: Any) -> dict[str, Any]:
        nodes = {str(getattr(node, "name", "")): node for node in instance.nodes()}
        direct_edges = []
        for node in instance.nodes():
            inputs = getattr(node, "inputs", ())
            if type(node).__name__ == "BoundRelu" and inputs:
                predecessor = inputs[0]
                if type(predecessor).__name__ == "BoundConv":
                    direct_edges.append(
                        {
                            "relu_name": str(getattr(node, "name", "")),
                            "relu_class": type(node).__name__,
                            "conv_name": str(getattr(predecessor, "name", "")),
                            "conv_class": type(predecessor).__name__,
                        }
                    )
        return {"nodes": nodes, "direct_edges": direct_edges}

    def _relu_wrapper(self, site: str, relu: Any, conv: Any, original: Any):
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            start = str(getattr(kwargs.get("start_node"), "name", ""))
            if self.current_evaluation is None:
                return result
            if start != TARGET_START:
                self.unexpected_target_start_calls += 1
                return result
            if site in self.pending or len(args) < 3:
                raise ValueError(f"MR4 {site} ReLU observation differs")
            incoming = args[0]
            preactivation = args[2]
            alpha = getattr(relu, "alpha", {}).get(TARGET_START)
            indices = getattr(relu, "alpha_indices", None)
            if (
                not self.torch.is_tensor(incoming)
                or args[1] is not None
                or not self.torch.is_tensor(alpha)
                or not isinstance(indices, list)
                or not all(self.torch.is_tensor(item) for item in indices)
            ):
                raise ValueError(f"MR4 {site} ReLU ABI differs")
            assert alpha is not None
            with self.torch.no_grad():
                reconstructed = relu.reconstruct_full_alpha(
                    alpha[0].detach(), tuple(incoming.shape), indices
                )[0]
            betas = _beta_tensors(relu, conv, self.torch)
            relu_output = _extract_lower_a(result, self.torch)
            lower = preactivation.lower
            upper = preactivation.upper
            row: dict[str, Any] = {
                "site": site,
                "evaluation_ordinal": self.current_evaluation,
                "grad_enabled": bool(self.torch.is_grad_enabled()),
                "start_node": start,
                "relu_name": str(getattr(relu, "name", "")),
                "conv_name": str(getattr(conv, "name", "")),
                "lower_only": True,
                "incoming_lower_a": _tensor_meta(incoming, self.torch),
                "preactivation_lower": _tensor_meta(lower, self.torch),
                "preactivation_upper": _tensor_meta(upper, self.torch),
                "bounds_finite": _finite(lower, self.torch)
                and _finite(upper, self.torch),
                "lower_le_upper": bool((lower <= upper).all().item()),
                "compressed_alpha": _tensor_meta(alpha, self.torch),
                "alpha_feature_index_shapes": [
                    [int(dimension) for dimension in item.shape] for item in indices
                ],
                "reconstructed_full_alpha": _tensor_meta(reconstructed, self.torch),
                "beta_tensor_count": len(betas),
                "beta_shapes": [
                    [int(dimension) for dimension in tensor.shape] for tensor in betas
                ],
                "beta_numel": sum(int(tensor.numel()) for tensor in betas),
                "relu_output_lower_a": _tensor_meta(relu_output, self.torch),
                "relu_lower_bias": _tensor_meta(
                    _extract_lower_bias(result, self.torch), self.torch
                ),
                "_relu_output": relu_output,
            }
            self.pending[site] = row
            self.relu_calls[site] += 1
            return result

        return wrapped

    def _conv_wrapper(self, site: str, original: Any):
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if self.current_evaluation is None:
                return original(*args, **kwargs)
            if site not in self.pending or len(args) < 5:
                raise ValueError(f"MR4 {site} Conv observation differs")
            pending = self.pending.pop(site)
            incoming = args[0]
            weight = args[3].lower
            bias = args[4].lower
            relu_output = pending.pop("_relu_output")
            result = original(*args, **kwargs)
            output_a = _extract_lower_a(result, self.torch)
            output_bias = _extract_lower_bias(result, self.torch)
            if not all(
                self.torch.is_tensor(value)
                for value in (incoming, weight, bias, output_a, output_bias)
            ):
                raise TypeError(f"MR4 {site} Conv tensor ABI differs")
            handoff_content = bool(self.torch.equal(incoming, relu_output))
            handoff_pointer = int(incoming.data_ptr()) == int(relu_output.data_ptr())
            weight_shape = tuple(int(dimension) for dimension in weight.shape)
            if len(weight_shape) != 4:
                raise ValueError(f"MR4 {site} Conv weight rank differs")
            mac_units = (
                int(output_a.numel())
                * weight_shape[0]
                * weight_shape[2]
                * weight_shape[3]
            )
            full_alpha = pending["reconstructed_full_alpha"]
            incoming_meta = pending["incoming_lower_a"]
            candidate_materialization_bytes = (
                int(incoming_meta["numel"])
                + int(full_alpha["numel"])
                + int(output_a.numel())
                + int(output_bias.numel())
            ) * int(incoming.element_size())
            pending.update(
                {
                    "conv_input_lower_a": _tensor_meta(incoming, self.torch),
                    "relu_conv_handoff_content_exact": handoff_content,
                    "relu_conv_handoff_pointer_exact": handoff_pointer,
                    "conv_weight": _tensor_meta(weight, self.torch),
                    "conv_bias": _tensor_meta(bias, self.torch),
                    "conv_output_lower_a": _tensor_meta(output_a, self.torch),
                    "conv_lower_bias": _tensor_meta(output_bias, self.torch),
                    "forward_mac_units": mac_units,
                    "candidate_minimum_materialization_bytes": candidate_materialization_bytes,
                }
            )
            self.rows.append(pending)
            self.conv_calls[site] += 1
            return result

        return wrapped

    @contextmanager
    def install(self, instance: Any) -> Iterator[None]:
        topology = self._topology(instance)
        nodes = topology["nodes"]
        self.topology = topology["direct_edges"]
        originals: list[tuple[Any, Any]] = []
        for site, relu_name, conv_name in FROZEN_EDGES:
            relu = nodes.get(relu_name)
            conv = nodes.get(conv_name)
            if (
                relu is None
                or conv is None
                or not getattr(relu, "inputs", ())
                or relu.inputs[0] is not conv
            ):
                raise ValueError(f"MR4 {site} topology differs")
            original_relu = relu.bound_backward
            original_conv = conv.bound_backward
            originals.extend(((relu, original_relu), (conv, original_conv)))
            relu.bound_backward = self._relu_wrapper(site, relu, conv, original_relu)
            conv.bound_backward = self._conv_wrapper(site, original_conv)
        try:
            yield
        finally:
            for node, original in originals:
                node.bound_backward = original
            self.device_after = int(self.torch.cuda.current_device())
            self.stream_after = int(self.torch.cuda.current_stream().cuda_stream)
            if self.pending:
                raise ValueError("MR4 census retained partial site observations")

    def receipt(self) -> dict[str, object]:
        return {
            "target_start": TARGET_START,
            "topology": self.topology,
            "rows": self.rows,
            "counters": {
                "row_count": len(self.rows),
                "relu_calls": self.relu_calls,
                "conv_calls": self.conv_calls,
                "unexpected_target_start_calls": self.unexpected_target_start_calls,
                "replacement_count": 0,
                "timing_observation_count": 0,
            },
            "device_before": self.device_before,
            "device_after": self.device_after,
            "stream_before": self.stream_before,
            "stream_after": self.stream_after,
        }


class _ExactCallTracker:
    def __init__(self, torch_module: Any) -> None:
        self.torch = torch_module
        self.stack: list[int] = []
        self.active_outer = False
        self.outer_count = 0
        self.inner_count = 0
        self.census: _SiteCensus | None = None
        self.outer_result_state: list[dict[str, object]] | None = None
        self.final_target_alpha_state: dict[str, object] | None = None
        self.final_module_state: list[dict[str, object]] | None = None

    @contextmanager
    def install(self, bounded_module: Any) -> Iterator[None]:
        original = bounded_module.compute_bounds

        def wrapped(instance: Any, *args: Any, **kwargs: Any) -> Any:
            method = str(kwargs.get("method", "backward"))
            phase = _phase_from_stack(method)
            is_outer = (
                not self.stack
                and phase == "beta_split"
                and "optimized" in method.lower()
            )
            is_inner = (
                self.active_outer
                and len(self.stack) == 1
                and phase == "beta_split"
                and method.lower() == "backward"
            )
            if is_outer:
                if self.outer_count:
                    raise ValueError("MR4 outer exact call repeated")
                self.outer_count += 1
                self.active_outer = True
                self.census = _SiteCensus(self.torch)
            call_id = len(self.stack)
            self.stack.append(call_id)
            if is_inner:
                if self.census is None:
                    raise ValueError("MR4 census is absent")
                self.census.current_evaluation = self.inner_count
                self.inner_count += 1
            hook = (
                self.census.install(instance)
                if is_outer and self.census is not None
                else nullcontext()
            )
            try:
                with hook:
                    result = original(instance, *args, **kwargs)
                if is_outer:
                    self.outer_result_state = _result_state(result, self.torch)
                    self.final_target_alpha_state = _result_state(
                        _target_alpha(instance, self.torch), self.torch
                    )[0]
                    self.final_module_state = _module_state(instance, self.torch)
                return result
            finally:
                if is_inner and self.census is not None:
                    self.census.current_evaluation = None
                if self.stack.pop() != call_id:
                    raise RuntimeError("MR4 compute_bounds stack differs")
                if is_outer:
                    self.active_outer = False

        bounded_module.compute_bounds = wrapped
        try:
            yield
        finally:
            bounded_module.compute_bounds = original


def _worker(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(args.abcrown_root / "complete_verifier"))
    sys.path.insert(0, str(args.abcrown_root))
    import torch

    from abcrown import ABCrownSolver, ConfigBuilder, IOConstraints  # type: ignore[import-not-found]
    from auto_LiRPA import BoundedModule  # type: ignore[import-untyped]

    if not torch.cuda.is_available():
        raise RuntimeError("MR4 census requires CUDA")
    tracker = _ExactCallTracker(torch)
    with tempfile.TemporaryDirectory(
        prefix="boundflow-mr4-census-property-"
    ) as workspace:
        isolated_property = Path(workspace) / args.property.name
        shutil.copy2(args.property, isolated_property)
        config = (
            ConfigBuilder.from_defaults()
            .set("general/device", "cuda")
            .set("general/seed", 100)
            .set("general/reset_seed_after_precompile", True)
            .set("general/complete_verifier", "bab")
            .set("attack/pgd_order", "skip")
            .set("bab/timeout", 60)
            .set("bab/max_iterations", 1)
            .set("solver/batch_size", 64)
            .set("solver/auto_enlarge_batch_size", False)
            .set("solver/alpha-crown/iteration", 5)
            .set("solver/beta-crown/iteration", 10)
        )
        with tracker.install(BoundedModule):
            solver = ABCrownSolver(str(args.model), config=config)
            result = solver.verify(
                constraints=IOConstraints(vnnlib_path=str(isolated_property))
            )
    if (
        tracker.outer_count != 1
        or tracker.inner_count != 10
        or tracker.census is None
        or tracker.outer_result_state is None
        or tracker.final_target_alpha_state is None
        or tracker.final_module_state is None
    ):
        raise ValueError("MR4 census worker did not close")
    payload: dict[str, object] = {
        "schema_version": WORKER_SCHEMA,
        "run_index": args.run_index,
        "source": {
            "abcrown_commit": _git(args.abcrown_root, "rev-parse", "HEAD"),
            "auto_lirpa_commit": _git(
                args.abcrown_root / "auto_LiRPA", "rev-parse", "HEAD"
            ),
            "vnncomp_commit": _git(args.benchmark_root, "rev-parse", "HEAD"),
            "model_sha256": _sha256(args.model),
            "property_sha256": _sha256(args.property),
        },
        "protocol": {
            "device": "cuda",
            "seed": 100,
            "max_iterations": 1,
            "batch_size": 64,
            "alpha_steps": 5,
            "beta_steps": 10,
            "property_cache": "cold_isolated_copy",
            "candidate_executed": False,
        },
        "solver_result": {
            "status": str(result.status),
            "success": bool(result.success),
            "visited_domains": _visited_domains(result),
        },
        "outer_exact_call_count": tracker.outer_count,
        "inner_evaluation_count": tracker.inner_count,
        "census": tracker.census.receipt(),
        "outer_result_state": tracker.outer_result_state,
        "final_target_alpha_state": tracker.final_target_alpha_state,
        "final_module_state": tracker.final_module_state,
        "timing_recorded": False,
        "performance_claimed": False,
    }
    payload["worker_hash"] = canonical_hash(payload)
    args.result_json.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"run_index": args.run_index, "status": str(result.status)}, sort_keys=True
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--abcrown-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--property", type=Path, required=True)
    parser.add_argument("--run-index", type=int, choices=range(5), required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    args = parser.parse_args()
    if (
        _git(args.abcrown_root, "rev-parse", "HEAD") != ABCROWN_COMMIT
        or _git(args.abcrown_root / "auto_LiRPA", "rev-parse", "HEAD")
        != AUTO_LIRPA_COMMIT
        or _git(args.benchmark_root, "rev-parse", "HEAD") != VNNCOMP_COMMIT
        or _sha256(args.model) != MODEL_SHA256
        or _sha256(args.property) != PROPERTY_SHA256
    ):
        raise ValueError("MR4 census frozen input differs")
    _worker(args)


if __name__ == "__main__":
    main()
