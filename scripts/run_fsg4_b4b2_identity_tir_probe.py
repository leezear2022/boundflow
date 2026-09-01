"""Run the B4-B2 B2-0 identity-TIR ABI probe on the active CUDA device."""

# pylint: disable=import-error

from __future__ import annotations

import json
from pathlib import Path

import torch

from boundflow.runtime.fsg4_b4b1_pytorch_reference import (
    build_b4b1_differentiable_lower_instance_v1,
    build_b4b1_differentiable_lower_ir_v1,
)
from boundflow.runtime.fsg4_b4b1_reference_capture import (
    production_differentiable_reference_capture_from_payload_v1,
)
from boundflow.runtime.fsg4_b4b2_identity_tir import (
    DifferentiableLowerIdentityModuleCache,
    build_b4b2_identity_instance_v1,
    build_b4b2_identity_schedule_v1,
    build_b4b2_identity_template_v1,
    run_b4b2_identity_tir_probe_v1,
)

CAPTURE = Path("artifacts/fsg4-b4b1-reference-five-fresh/resnet2b-prop0-v1/run_00.pt")


def run_probe() -> dict[str, object]:
    """Execute cold and warm differentiable launches on one explicit stream."""

    if not torch.cuda.is_available():
        raise RuntimeError("B4-B2 identity TIR probe requires CUDA")
    payload = torch.load(CAPTURE, map_location="cpu", weights_only=False)
    capture = production_differentiable_reference_capture_from_payload_v1(
        payload["captures"][0]
    )
    lower_ir = build_b4b1_differentiable_lower_ir_v1(capture)
    lower_instance = build_b4b1_differentiable_lower_instance_v1(capture, lower_ir)
    major, minor = torch.cuda.get_device_capability()
    template = build_b4b2_identity_template_v1(
        lower_ir, tensor_numel=257, compute_capability=f"sm_{major}{minor}"
    )
    schedule = build_b4b2_identity_schedule_v1(template)
    cache = DifferentiableLowerIdentityModuleCache()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        source = torch.linspace(-1.0, 1.0, 257, device="cuda").requires_grad_(True)
        upstream = torch.linspace(0.5, 1.5, 257, device="cuda")
        instance = build_b4b2_identity_instance_v1(
            template, lower_ir, lower_instance, source, upstream
        )
        cold = run_b4b2_identity_tir_probe_v1(
            template, instance, schedule, source, upstream, cache=cache
        )
        warm = run_b4b2_identity_tir_probe_v1(
            template, instance, schedule, source, upstream, cache=cache
        )
    stream.synchronize()
    torch.testing.assert_close(cold.output, source, rtol=0, atol=0)
    torch.testing.assert_close(cold.input_gradient, upstream, rtol=0, atol=0)
    return {
        "status": "probe-passed",
        "device": torch.cuda.get_device_name(),
        "compute_capability": template.compute_capability,
        "template_hash": template.stable_hash(),
        "instance_hash": instance.stable_hash(template),
        "schedule_hash": schedule.stable_hash(template),
        "module_receipt_hash": cold.module_receipt.stable_hash(template, schedule),
        "cold_cache_event": cold.launch_receipt.cache_event,
        "warm_cache_event": warm.launch_receipt.cache_event,
        "stream_id": cold.launch_receipt.stream_id,
        "forward_launch_count": cold.launch_receipt.forward_launch_count,
        "backward_launch_count": cold.launch_receipt.backward_launch_count,
        "fallback_count": cold.launch_receipt.fallback_count,
        "eager_backward_count": cold.launch_receipt.eager_backward_count,
        "zero_copy_exact": all(
            (
                cold.launch_receipt.input_roundtrip_ptr_exact,
                cold.launch_receipt.output_roundtrip_ptr_exact,
                cold.launch_receipt.upstream_gradient_roundtrip_ptr_exact,
                cold.launch_receipt.input_gradient_roundtrip_ptr_exact,
            )
        ),
        "output_aliases_input": cold.launch_receipt.output_aliases_input,
        "input_gradient_aliases_upstream": (
            cold.launch_receipt.input_gradient_aliases_upstream
        ),
        "first_order_gradient_exact": True,
        "enabled_by_default": template.enabled_by_default,
        "performance_claimed": cold.launch_receipt.performance_claimed,
    }


def main() -> None:
    """Print one canonical probe summary."""

    print(json.dumps(run_probe(), sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
