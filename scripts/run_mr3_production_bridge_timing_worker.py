#!/usr/bin/env python3
"""Run one lean provider or bridge production exact-call timing worker."""

# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=import-outside-toplevel,protected-access,import-error
# pylint: disable=missing-function-docstring,wrong-import-position
# pylint: disable=too-many-instance-attributes,too-few-public-methods
# pylint: disable=line-too-long,too-many-boolean-expressions

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boundflow.backends.tvm.cibc_dense_exact_conv import (  # noqa: E402
    compile_cibc_dense_exact_conv_tir_v3,
)
from boundflow.runtime.fsg4_b4b3_cibc_dense_tir import (  # noqa: E402
    execute_cibc_dense_exact_tir_v3,
)
from boundflow.runtime.mr3_production_p_anchor_bridge import (  # noqa: E402
    MR3ProductionPAnchorBridgeV1,
    TARGET_CONV,
    TARGET_RELU,
    TARGET_START,
)
from boundflow.runtime.mr3_provider_hook_feasibility import (  # noqa: E402
    ABCROWN_COMMIT,
    AUTO_LIRPA_COMMIT,
    VNNCOMP_COMMIT,
    canonical_hash,
)
from scripts.run_mr3_provider_hook_feasibility import (  # noqa: E402
    MODEL_SHA256,
    PROPERTY_SHA256,
    _git,
    _module_state,
    _phase_from_stack,
    _result_state,
    _sha256,
    _target_alpha,
    _visited_domains,
    _walk_tensor_values,
)

WORKER_SCHEMA = "boundflow.mr3-production-bridge-timing-worker/v1"


def _gpu_snapshot() -> dict[str, object]:
    fields = (
        "name",
        "driver_version",
        "temperature.gpu",
        "power.draw",
        "clocks.current.graphics",
        "clocks.current.memory",
        "power.limit",
    )
    result = subprocess.run(
        (
            "nvidia-smi",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
            "--id=0",
        ),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    values = [item.strip() for item in result.stdout.strip().split(",")]
    if len(values) != len(fields):
        raise ValueError("MR3 timing GPU snapshot differs")
    return dict(zip(fields, values))


def _warm_candidate(torch_module: Any):
    major, minor = torch_module.cuda.get_device_capability()
    compiled = compile_cibc_dense_exact_conv_tir_v3(
        compute_capability=f"sm_{major}{minor}"
    )
    device = torch_module.device("cuda:0")
    incoming = torch_module.zeros(
        (6, 1, 16, 8, 8), dtype=torch_module.float32, device=device, requires_grad=True
    )
    lower = -torch_module.ones((6, 16, 8, 8), dtype=torch_module.float32, device=device)
    upper = torch_module.ones_like(lower)
    alpha = torch_module.full(
        (6, 16, 8, 8),
        0.5,
        dtype=torch_module.float32,
        device=device,
        requires_grad=True,
    )
    incoming_bias = torch_module.zeros(
        (6, 1), dtype=torch_module.float32, device=device
    )
    weight = torch_module.zeros(
        (16, 16, 3, 3), dtype=torch_module.float32, device=device
    )
    bias = torch_module.zeros((16,), dtype=torch_module.float32, device=device)
    output_a, output_bias, executor = execute_cibc_dense_exact_tir_v3(
        incoming_lower_a=incoming,
        preactivation_lower=lower,
        preactivation_upper=upper,
        native_alpha=alpha,
        incoming_lower_bias=incoming_bias,
        operator_weight=weight,
        operator_bias=bias,
        compiled=compiled,
    )
    torch_module.autograd.backward(
        (output_a, output_bias),
        (torch_module.ones_like(output_a), torch_module.ones_like(output_bias)),
    )
    torch_module.cuda.synchronize()
    if (
        executor.forward_launch_count != 1
        or executor.backward_launch_count != 1
        or executor.fallback_count != 0
        or executor.eager_count != 0
    ):
        raise ValueError("MR3 timing candidate warm receipt differs")
    return compiled


class _TimingTracker:
    def __init__(self, torch_module: Any, *, mode: str) -> None:
        self.torch = torch_module
        self.mode = mode
        self.stack: list[int] = []
        self.active_outer = False
        self.current_evaluation: int | None = None
        self.outer_count = 0
        self.inner_count = 0
        self.bridge: MR3ProductionPAnchorBridgeV1 | None = None
        self.bridge_receipt: dict[str, object] | None = None
        self.compiled = _warm_candidate(torch_module) if mode == "bridge" else None
        self.start_event = torch_module.cuda.Event(enable_timing=True)
        self.end_event = torch_module.cuda.Event(enable_timing=True)
        self.measurement: dict[str, object] | None = None
        self.outer_state: list[dict[str, object]] | None = None
        self.final_alpha_state: dict[str, object] | None = None
        self.final_module_state: list[dict[str, object]] | None = None

    @contextmanager
    def _node_bridge(self, instance: Any) -> Iterator[None]:
        nodes = {str(getattr(node, "name", "")): node for node in instance.nodes()}
        relu = nodes.get(TARGET_RELU)
        conv = nodes.get(TARGET_CONV)
        if (
            relu is None
            or conv is None
            or not getattr(relu, "inputs", ())
            or relu.inputs[0] is not conv
            or self.compiled is None
        ):
            raise ValueError("MR3 timing bridge topology differs")
        self.bridge = MR3ProductionPAnchorBridgeV1(compiled=self.compiled)
        original_relu = relu.bound_backward
        original_conv = conv.bound_backward

        def relu_wrapped(*args: Any, **kwargs: Any) -> Any:
            start = str(getattr(kwargs.get("start_node"), "name", ""))
            if self.current_evaluation is None or start != TARGET_START:
                return original_relu(*args, **kwargs)
            beta_tensors = []
            for owner in (relu, conv):
                for attribute in ("sparse_betas", "beta", "split_beta"):
                    beta_tensors.extend(
                        _walk_tensor_values(getattr(owner, attribute, None), self.torch)
                    )
            assert self.bridge is not None
            self.bridge.begin_evaluation(self.current_evaluation)
            return self.bridge.route_relu(
                relu,
                args,
                kwargs,
                beta_tensors=tuple(beta_tensors),
            )

        def conv_wrapped(*args: Any, **kwargs: Any) -> Any:
            if self.current_evaluation is None:
                return original_conv(*args, **kwargs)
            assert self.bridge is not None
            return self.bridge.route_conv(args)

        relu.bound_backward = relu_wrapped
        conv.bound_backward = conv_wrapped
        try:
            yield
        finally:
            relu.bound_backward = original_relu
            conv.bound_backward = original_conv

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
                    raise ValueError("MR3 timing outer call repeated")
                self.outer_count += 1
                self.active_outer = True
            call_id = len(self.stack)
            self.stack.append(call_id)
            if is_inner:
                self.current_evaluation = self.inner_count
                self.inner_count += 1
            node_context = (
                self._node_bridge(instance)
                if is_outer and self.mode == "bridge"
                else nullcontext()
            )
            try:
                with node_context:
                    if is_outer:
                        self.torch.cuda.synchronize()
                        self.torch.cuda.reset_peak_memory_stats()
                        base_allocated = int(self.torch.cuda.memory_allocated())
                        base_reserved = int(self.torch.cuda.memory_reserved())
                        device_before = int(self.torch.cuda.current_device())
                        stream_before = int(
                            self.torch.cuda.current_stream().cuda_stream
                        )
                        self.start_event.record()
                        host_start = time.perf_counter_ns()
                    result = original(instance, *args, **kwargs)
                    if is_outer:
                        self.end_event.record()
                        self.torch.cuda.synchronize()
                        host_end = time.perf_counter_ns()
                        self.measurement = {
                            "host_ns": host_end - host_start,
                            "cuda_event_ms": float(
                                self.start_event.elapsed_time(self.end_event)
                            ),
                            "device_before": device_before,
                            "device_after": int(self.torch.cuda.current_device()),
                            "stream_before": stream_before,
                            "stream_after": int(
                                self.torch.cuda.current_stream().cuda_stream
                            ),
                            "base_allocated_bytes": base_allocated,
                            "base_reserved_bytes": base_reserved,
                            "peak_allocated_bytes": int(
                                self.torch.cuda.max_memory_allocated()
                            ),
                            "peak_reserved_bytes": int(
                                self.torch.cuda.max_memory_reserved()
                            ),
                        }
                        self.outer_state = _result_state(result, self.torch)
                        self.final_alpha_state = _result_state(
                            _target_alpha(instance, self.torch), self.torch
                        )[0]
                        self.final_module_state = _module_state(instance, self.torch)
                        if self.bridge is not None:
                            self.bridge_receipt = self.bridge.receipt().to_dict()
                return result
            finally:
                if is_inner:
                    self.current_evaluation = None
                if self.stack.pop() != call_id:
                    raise RuntimeError("MR3 timing call stack differs")
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
        raise RuntimeError("MR3 timing requires CUDA")
    before = _gpu_snapshot()
    tracker = _TimingTracker(torch, mode=args.mode)
    with tempfile.TemporaryDirectory(
        prefix="boundflow-mr3-timing-property-"
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
    after = _gpu_snapshot()
    if (
        tracker.outer_count != 1
        or tracker.inner_count != 10
        or tracker.measurement is None
        or tracker.outer_state is None
        or tracker.final_alpha_state is None
        or tracker.final_module_state is None
        or (args.mode == "bridge" and tracker.bridge_receipt is None)
    ):
        raise ValueError("MR3 timing worker did not close")
    payload: dict[str, object] = {
        "schema_version": WORKER_SCHEMA,
        "mode": args.mode,
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
            "formal_observation_enabled": False,
            "compile_timed": False,
            "dummy_module_warm_timed": False,
        },
        "gpu_before": before,
        "gpu_after": after,
        "device_before": tracker.measurement["device_before"],
        "device_after": tracker.measurement["device_after"],
        "stream_before": tracker.measurement["stream_before"],
        "stream_after": tracker.measurement["stream_after"],
        "solver_result": {
            "status": str(result.status),
            "success": bool(result.success),
            "visited_domains": _visited_domains(result),
        },
        "outer_result_state": tracker.outer_state,
        "final_target_alpha_state": tracker.final_alpha_state,
        "final_module_state": tracker.final_module_state,
        "measurement": tracker.measurement,
        "bridge_receipt": tracker.bridge_receipt,
        "timing_recorded": True,
        "performance_claimed": False,
    }
    payload["worker_hash"] = canonical_hash(payload)
    args.result_json.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"mode": args.mode, "status": str(result.status)}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--abcrown-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--property", type=Path, required=True)
    parser.add_argument("--mode", choices=("provider", "bridge"), required=True)
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
        raise ValueError("MR3 timing frozen input differs")
    _worker(args)


if __name__ == "__main__":
    main()
