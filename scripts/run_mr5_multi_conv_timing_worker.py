#!/usr/bin/env python3
"""Run one lean provider or prewarmed MR5 multi-Conv exact-call timing worker."""

# pylint: disable=protected-access,import-error,wrong-import-position
# pylint: disable=missing-function-docstring,too-many-locals
# pylint: disable=too-many-instance-attributes,duplicate-code
# pylint: disable=too-few-public-methods,super-init-not-called
# mypy: disable-error-code=import-untyped

from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
import sys
from typing import Any, Iterator, cast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boundflow.runtime.mr3_provider_hook_feasibility import (  # noqa: E402
    ABCROWN_COMMIT,
    AUTO_LIRPA_COMMIT,
    VNNCOMP_COMMIT,
    canonical_hash,
)
from boundflow.runtime.mr5_generalized_crown_conv import (  # noqa: E402
    MR5GeneralizedConvModuleCacheV1,
    MR5GeneralizedConvTensorsV1,
    execute_mr5_generalized_conv_v1,
)
from boundflow.runtime.mr5_multi_conv_production_bridge import (  # noqa: E402
    MR5MultiConvProductionBridgeV1,
    MR5_SITE_NODES,
    MR5_SITE_ORDER,
    MR5_TARGET_START,
    mr5_frozen_signatures,
)
from scripts import run_mr3_production_bridge_timing_worker as legacy  # noqa: E402
from scripts.run_mr3_provider_hook_feasibility import (  # noqa: E402
    MODEL_SHA256,
    PROPERTY_SHA256,
    _git,
    _sha256,
    _walk_tensor_values,
)

WORKER_SCHEMA = "boundflow.mr5-multi-conv-timing-worker/v1"


class _ReceiptProjection:
    def __init__(self, value: dict[str, object]) -> None:
        self.value = value

    def to_dict(self) -> dict[str, object]:
        return self.value


class _TimingBridge(MR5MultiConvProductionBridgeV1):
    def receipt(self):  # type: ignore[override]
        return _ReceiptProjection(self.timing_receipt())


def _warm_candidate(torch_module: Any):
    major, minor = torch_module.cuda.get_device_capability()
    signatures = mr5_frozen_signatures(f"sm_{major}{minor}")
    cache = MR5GeneralizedConvModuleCacheV1()
    receipts: dict[str, object] = {}
    device = torch_module.device("cuda:0")
    for site in MR5_SITE_ORDER:
        signature = signatures[site]
        incoming = torch_module.zeros(
            signature.incoming_shape,
            dtype=torch_module.float32,
            device=device,
            requires_grad=True,
        )
        lower = -torch_module.ones(
            signature.relaxation_shape, dtype=torch_module.float32, device=device
        )
        upper = torch_module.ones_like(lower)
        alpha = torch_module.full(
            signature.relaxation_shape,
            0.5,
            dtype=torch_module.float32,
            device=device,
            requires_grad=True,
        )
        tensors = MR5GeneralizedConvTensorsV1(
            incoming=incoming,
            lower=lower,
            upper=upper,
            alpha=alpha,
            incoming_bias=torch_module.zeros(
                (6, 1), dtype=torch_module.float32, device=device
            ),
            weight=torch_module.zeros(
                signature.weight_shape, dtype=torch_module.float32, device=device
            ),
            operator_bias=torch_module.zeros(
                (signature.output_channels,),
                dtype=torch_module.float32,
                device=device,
            ),
        )
        output_a, output_bias, executor = execute_mr5_generalized_conv_v1(
            signature, tensors, cache
        )
        torch_module.autograd.backward(
            (output_a, output_bias),
            (torch_module.ones_like(output_a), torch_module.ones_like(output_bias)),
        )
        if (
            executor.cache_event != "miss"
            or executor.forward_launch_count != 1
            or executor.backward_launch_count != 1
            or executor.fallback_count != 0
            or executor.eager_count != 0
        ):
            raise ValueError(f"MR5 {site} timing warm receipt differs")
        receipts[site] = executor.module_receipt.to_dict()
    torch_module.cuda.synchronize()
    receipt: dict[str, object] = {
        "site_order": list(MR5_SITE_ORDER),
        "module_receipts": receipts,
        "dummy_forward_launch_count": 3,
        "dummy_backward_launch_count": 3,
        "dummy_fallback_count": 0,
        "dummy_eager_count": 0,
    }
    receipt["receipt_hash"] = canonical_hash(receipt)
    return cache, receipt


class _TimingTracker(legacy._TimingTracker):
    """Reuse the approved full-outer measurement bracket with three-site routing."""

    def __init__(self, torch_module: Any, *, mode: str) -> None:
        self.torch = torch_module
        self.mode = mode
        self.stack: list[int] = []
        self.active_outer = False
        self.current_evaluation: int | None = None
        self.outer_count = 0
        self.inner_count = 0
        self.bridge = None
        self.bridge_receipt: dict[str, object] | None = None
        self.compiled = None
        self.cache = None
        self.candidate_module_receipt: dict[str, object] | None = None
        if mode == "bridge":
            self.cache, self.candidate_module_receipt = _warm_candidate(torch_module)
        self.start_event = torch_module.cuda.Event(enable_timing=True)
        self.end_event = torch_module.cuda.Event(enable_timing=True)
        self.measurement: dict[str, object] | None = None
        self.outer_state: list[dict[str, object]] | None = None
        self.final_alpha_state: dict[str, object] | None = None
        self.final_module_state: list[dict[str, object]] | None = None

    @contextmanager
    def _node_bridge(self, instance: Any) -> Iterator[None]:
        if self.cache is None:
            raise ValueError("MR5 timing prewarmed cache is absent")
        nodes = {str(getattr(node, "name", "")): node for node in instance.nodes()}
        site_nodes: dict[str, tuple[Any, Any]] = {}
        originals: dict[str, tuple[Any, Any]] = {}
        for site, (relu_name, conv_name) in MR5_SITE_NODES.items():
            relu = nodes.get(relu_name)
            conv = nodes.get(conv_name)
            if (
                relu is None
                or conv is None
                or not getattr(relu, "inputs", ())
                or relu.inputs[0] is not conv
            ):
                raise ValueError(f"MR5 {site} timing topology differs")
            site_nodes[site] = (relu, conv)
            originals[site] = (relu.bound_backward, conv.bound_backward)
        self.bridge = cast(Any, _TimingBridge(cache=self.cache))

        def relu_wrapper(site: str, relu: Any, conv: Any, original: Any):
            def wrapped(*args: Any, **kwargs: Any) -> Any:
                start = str(getattr(kwargs.get("start_node"), "name", ""))
                if self.current_evaluation is None or start != MR5_TARGET_START:
                    return original(*args, **kwargs)
                bridge = cast(_TimingBridge, self.bridge)
                if site == MR5_SITE_ORDER[0]:
                    bridge.begin_evaluation(self.current_evaluation)
                beta_tensors = []
                seen: set[int] = set()
                for owner in (relu, conv):
                    for attribute in ("sparse_betas", "beta", "split_beta"):
                        for tensor in _walk_tensor_values(
                            getattr(owner, attribute, None), self.torch
                        ):
                            if id(tensor) not in seen:
                                seen.add(id(tensor))
                                beta_tensors.append(tensor)
                return bridge.route_relu(
                    site,
                    relu,
                    args,
                    kwargs,
                    beta_tensors=tuple(beta_tensors),
                )

            return wrapped

        def conv_wrapper(site: str, original: Any):
            def wrapped(*args: Any, **kwargs: Any) -> Any:
                if self.current_evaluation is None:
                    return original(*args, **kwargs)
                return cast(_TimingBridge, self.bridge).route_conv(site, args)

            return wrapped

        for site, (relu, conv) in site_nodes.items():
            relu.bound_backward = relu_wrapper(site, relu, conv, originals[site][0])
            conv.bound_backward = conv_wrapper(site, originals[site][1])
        try:
            yield
        finally:
            for site, (relu, conv) in site_nodes.items():
                relu.bound_backward, conv.bound_backward = originals[site]


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
        raise ValueError("MR5 timing frozen input differs")
    legacy.WORKER_SCHEMA = WORKER_SCHEMA
    setattr(legacy, "_TimingTracker", _TimingTracker)
    legacy._worker(args)


if __name__ == "__main__":
    main()
