#!/usr/bin/env python3
"""Run one unprofiled or profiled MR7 diagnostic-bridge attribution worker."""

# pylint: disable=protected-access,import-error,wrong-import-position
# pylint: disable=missing-function-docstring,too-many-locals,duplicate-code
# pylint: disable=too-few-public-methods,super-init-not-called
# pylint: disable=too-many-statements,too-many-branches
# pylint: disable=too-many-boolean-expressions,import-outside-toplevel
# mypy: disable-error-code=import-untyped

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from functools import wraps
import json
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping, cast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boundflow.runtime.mr3_provider_hook_feasibility import (  # noqa: E402
    ABCROWN_COMMIT,
    AUTO_LIRPA_COMMIT,
    VNNCOMP_COMMIT,
    canonical_hash,
)
from boundflow.runtime import (  # noqa: E402
    mr5_generalized_crown_conv as runtime_module,
)
from boundflow.runtime import (  # noqa: E402
    mr5_multi_conv_production_bridge as bridge_module,
)
from boundflow.runtime.mr5_generalized_crown_conv import (  # noqa: E402
    MR5_BACKWARD_SYMBOL,
    MR5_FORWARD_SYMBOL,
    MR5GeneralizedConvTensorsV1,
)
from boundflow.runtime.mr7_launch_materialization_attribution import (  # noqa: E402
    MARKER_PREFIX,
    MR7HostLedger,
    WORKER_SCHEMA,
    extract_device_events,
    extract_device_marker_totals,
)
from scripts import run_mr5_multi_conv_timing_worker as mr5_worker  # noqa: E402
from scripts.run_mr3_provider_hook_feasibility import (  # noqa: E402
    MODEL_SHA256,
    PROPERTY_SHA256,
    _git,
    _sha256,
)
from scripts.run_mr6_guard_attribution_worker import (  # noqa: E402
    _GuardLedger,
    _guard_policy,
)

KINDS = ("control", "profile")
_BaseTimingBridge = mr5_worker._TimingBridge


def _marker_context(name: str, enabled: bool):
    if not enabled:
        return nullcontext()
    from torch.profiler import record_function

    return record_function(f"{MARKER_PREFIX}{name}")


@contextmanager
def _instrument_executor(
    ledger: MR7HostLedger, *, profile_enabled: bool
) -> Iterator[dict[str, int]]:
    executor = runtime_module.MR5GeneralizedConvExecutorV1
    original_launch = executor._launch
    original_forward = executor.forward
    original_backward = executor.backward
    counts: dict[str, int] = {}

    @wraps(original_launch)
    def launch_wrapped(self: Any, symbol: str, sources: Any, outputs: Any):
        direction = (
            "forward"
            if symbol == MR5_FORWARD_SYMBOL
            else "backward" if symbol == MR5_BACKWARD_SYMBOL else "unknown"
        )
        if direction == "unknown":
            raise ValueError("MR7 launch symbol differs")
        site = str(self.signature.site_id)
        key = f"{direction}.{site}"
        ordinal = counts.get(key, 0)
        counts[key] = ordinal + 1
        with _marker_context(f"{direction}.{site}.{ordinal:02d}", profile_enabled):
            with ledger.span("ffi_dlpack_stream"):
                return original_launch(self, symbol, sources, outputs)

    @wraps(original_forward)
    def forward_wrapped(self: Any):
        with ledger.span("layout_materialization"):
            return original_forward(self)

    @wraps(original_backward)
    def backward_wrapped(self: Any, result_a_gradient: Any, result_bias_gradient: Any):
        with ledger.span("layout_materialization"):
            return original_backward(self, result_a_gradient, result_bias_gradient)

    setattr(executor, "_launch", launch_wrapped)
    setattr(executor, "forward", forward_wrapped)
    setattr(executor, "backward", backward_wrapped)
    try:
        yield counts
    finally:
        setattr(executor, "_launch", original_launch)
        setattr(executor, "forward", original_forward)
        setattr(executor, "backward", original_backward)


class _AttributedBridge(_BaseTimingBridge):
    ledger: MR7HostLedger

    def __init__(self, *, cache: Any) -> None:
        super().__init__(cache=cache)
        if not hasattr(type(self), "ledger"):
            raise RuntimeError("MR7 bridge ledger absent")

    def route_relu(
        self,
        site: str,
        relu: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
        *,
        beta_tensors: tuple[Any, ...],
    ) -> Any:
        import torch

        with self.ledger.span("admission_handoff"):
            self._expect_site(site)
            signature = self.signatures[site]
            incoming = args[0] if args else None
            preactivation = args[2] if len(args) >= 3 else None
            alpha = getattr(relu, "alpha", {}).get(bridge_module.MR5_TARGET_START)
            indices = getattr(relu, "alpha_indices", None)
            provider_shape = (
                signature.spec_count,
                signature.domain_count,
                signature.output_channels,
                signature.output_height,
                signature.output_width,
            )
            if (
                str(getattr(kwargs.get("start_node"), "name", ""))
                != bridge_module.MR5_TARGET_START
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
                self._reject(f"MR7 {site} ReLU admission differs")
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
                self._reject(f"MR7 {site} alpha reconstruction differs")
            handoff = incoming.detach()
        with self.ledger.span("layout_materialization"):
            self.pending[site] = {
                "incoming": incoming.permute(1, 0, 2, 3, 4).contiguous(),
                "lower": preactivation.lower.contiguous(),
                "upper": preactivation.upper.contiguous(),
                "alpha": full_alpha,
                "handoff": handoff,
            }
            bridge_bias = torch.zeros(
                (signature.spec_count, signature.domain_count),
                dtype=torch.float32,
                device=incoming.device,
            )
        with self.ledger.span("admission_handoff"):
            self.beta_count[site] += 1
            self.beta_numel[site] += int(beta_tensors[0].numel())
        return [(handoff, None)], bridge_bias, 0

    def route_conv(self, site: str, args: tuple[Any, ...]) -> Any:
        import torch

        with self.ledger.span("admission_handoff"):
            self._expect_site(site)
            if len(args) < 5 or site not in self.pending:
                self._reject(f"MR7 {site} Conv admission differs")
            signature = self.signatures[site]
            pending = self.pending.pop(site)
            provider_input = args[0]
            weight = args[3].lower
            operator_bias = args[4].lower
            if (
                not torch.is_tensor(provider_input)
                or not bridge_module._content_equal(provider_input, pending["handoff"])
                or tuple(weight.shape) != signature.weight_shape
                or tuple(operator_bias.shape) != (signature.output_channels,)
            ):
                self._reject(f"MR7 {site} ReLU-to-Conv handoff differs")
            self.handoff_content[site] += 1
            self.handoff_pointer[site] += int(
                provider_input.data_ptr() == pending["handoff"].data_ptr()
            )
        with self.ledger.span("layout_materialization"):
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
        with self.ledger.span("admission_handoff"):
            result_a, result_bias, executor = (
                runtime_module.execute_mr5_generalized_conv_v1(
                    signature,
                    tensors,
                    self.cache,
                    backward_observer=self._backward_completed,
                )
            )
            self.forward[site] += executor.forward_launch_count
            self.cache_miss[site] += int(executor.cache_event == "miss")
            self.cache_hit[site] += int(executor.cache_event == "hit")
            receipt = executor.module_receipt.to_dict()
            existing = self.module_receipts.get(site)
            if existing is not None and existing != receipt:
                self._reject(f"MR7 {site} module receipt drifted")
            self.module_receipts[site] = receipt
        with self.ledger.span("post_output_guard"):
            routed_a = result_a.permute(1, 0, 2, 3, 4).contiguous()
            routed_bias = result_bias.transpose(0, 1).contiguous()
            if not bool(torch.isfinite(routed_a).all()) or not bool(
                torch.isfinite(routed_bias).all()
            ):
                self._reject(f"MR7 {site} candidate output is nonfinite")
        with self.ledger.span("admission_handoff"):
            self.site_order_count += 1
            self.expected_site_ordinal += 1
            if self.expected_site_ordinal == len(bridge_module.MR5_SITE_ORDER):
                if self.pending:
                    self._reject("MR7 multi-Conv pending site leaked")
                self.evaluation_count += 1
                self.current_evaluation = None
        return (
            [(routed_a, None), (None, None), (None, None)],
            routed_bias,
            0,
        )


def _configured_tracker(kind: str, sink: list[Any]):
    profile_enabled = kind == "profile"

    class _MR7Tracker(mr5_worker._TimingTracker):
        def __init__(self, torch_module: Any, *, mode: str) -> None:
            super().__init__(torch_module, mode="bridge")
            self.host_ledger = MR7HostLedger()
            self.guard_ledger = _GuardLedger("diagnostic")
            self.launch_marker_counts: dict[str, int] = {}
            sink.append(self)

        @contextmanager
        def install(self, bounded_module: Any) -> Iterator[None]:
            _AttributedBridge.ledger = self.host_ledger
            original_bridge = mr5_worker._TimingBridge
            setattr(mr5_worker, "_TimingBridge", _AttributedBridge)
            try:
                with _guard_policy(self.guard_ledger):
                    with _instrument_executor(
                        self.host_ledger, profile_enabled=profile_enabled
                    ) as counts:
                        with super().install(bounded_module):
                            yield
                        self.launch_marker_counts = dict(counts)
            finally:
                setattr(mr5_worker, "_TimingBridge", original_bridge)

    return _MR7Tracker


def _run(args: argparse.Namespace) -> dict[str, object]:
    import torch

    trackers: list[Any] = []
    setattr(
        mr5_worker.legacy, "_TimingTracker", _configured_tracker(args.kind, trackers)
    )
    mr5_worker.legacy.WORKER_SCHEMA = mr5_worker.WORKER_SCHEMA
    base_path = args.result_json.with_suffix(".base.json")
    base_args = argparse.Namespace(
        benchmark_root=args.benchmark_root,
        abcrown_root=args.abcrown_root,
        model=args.model,
        property=args.property,
        mode="bridge",
        result_json=base_path,
    )
    profiler = None
    if args.kind == "profile":
        from torch.profiler import profile, ProfilerActivity

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            acc_events=True,
        ) as profiler:
            mr5_worker.legacy._worker(base_args)
        torch.cuda.synchronize()
    else:
        mr5_worker.legacy._worker(base_args)
    if len(trackers) != 1:
        raise ValueError("MR7 tracker lifecycle differs")
    tracker = trackers[0]
    base_worker = json.loads(base_path.read_text(encoding="utf-8"))
    base_path.unlink()
    guard_receipt = tracker.guard_ledger.to_dict(site_evaluations=30)
    if guard_receipt["synchronizing_guards_executed"] != 60:
        raise ValueError("MR7 diagnostic guard count differs")
    measurement = base_worker.get("measurement")
    if not isinstance(measurement, dict):
        raise ValueError("MR7 inherited measurement absent")
    host_receipt = tracker.host_ledger.receipt(outer_ns=int(measurement["host_ns"]))
    profiler_events = () if profiler is None else profiler.events()
    events = () if profiler is None else extract_device_events(profiler_events)
    marker_totals = (
        {} if profiler is None else extract_device_marker_totals(profiler_events)
    )
    event_payload = [event.to_dict() for event in events]
    payload: dict[str, object] = {
        "schema_version": WORKER_SCHEMA,
        "kind": args.kind,
        "base_worker": base_worker,
        "guard_receipt": guard_receipt,
        "host_receipt": host_receipt,
        "launch_marker_counts": tracker.launch_marker_counts,
        "device_events": event_payload,
        "device_marker_totals": marker_totals,
        "device_event_hash": canonical_hash(event_payload),
        "timing_recorded": True,
        "production_admitted": False,
        "performance_claimed": False,
    }
    payload["worker_hash"] = canonical_hash(payload)
    args.result_json.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--abcrown-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--property", type=Path, required=True)
    parser.add_argument("--kind", choices=KINDS, required=True)
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
        raise ValueError("MR7 attribution frozen input differs")
    result = _run(args)
    print(
        json.dumps(
            {
                "kind": result["kind"],
                "host_receipt": result["host_receipt"],
                "device_event_count": len(cast(list[object], result["device_events"])),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
