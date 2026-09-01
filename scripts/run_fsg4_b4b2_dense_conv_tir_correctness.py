"""Run the B4-B2 B2-3 five-fresh P-anchor dense Conv correctness gate."""

# pylint: disable=import-error

from __future__ import annotations

import json
from pathlib import Path

import torch

from boundflow.runtime.fsg4_b4b1_pytorch_reference import (
    build_b4b1_differentiable_lower_ir_v1,
)
from boundflow.runtime.fsg4_b4b1_reference_capture import (
    production_differentiable_reference_capture_from_payload_v1,
)
from boundflow.runtime.fsg4_b4b2_dense_conv_tir import (
    DifferentiableLowerDenseConvModuleCache,
    build_b4b2_dense_conv_schedule_v1,
    build_b4b2_dense_conv_template_v1,
    run_b4b2_dense_conv_tir_v1,
)

ARTIFACT = Path("artifacts/fsg4-b4b1-reference-five-fresh/resnet2b-prop0-v1")


def _capture(run_ordinal: int):
    payload = torch.load(
        ARTIFACT / f"run_{run_ordinal:02d}.pt",
        map_location="cpu",
        weights_only=False,
    )
    return production_differentiable_reference_capture_from_payload_v1(
        payload["captures"][1]
    )


def run_gate() -> dict[str, object]:
    """Execute five P captures through one beta-free Conv module cache."""

    if not torch.cuda.is_available():
        raise RuntimeError("B4-B2 dense Conv correctness requires CUDA")
    cache = DifferentiableLowerDenseConvModuleCache()
    rows: list[dict[str, object]] = []
    result = None
    metric_count = 0
    maximum_absolute_difference = 0.0
    allclose = True
    sign_exact = True
    for run_ordinal in range(5):
        candidate = run_b4b2_dense_conv_tir_v1(
            _capture(run_ordinal), fresh_run_ordinal=run_ordinal, cache=cache
        )
        rows.append(
            {
                "run_ordinal": run_ordinal,
                "cache_event": candidate.launch_receipt.cache_event,
                "maximum_absolute_difference": max(
                    metric.maximum_absolute_difference for metric in candidate.metrics
                ),
                "allclose": all(metric.allclose for metric in candidate.metrics),
                "sign_exact": all(metric.sign_exact for metric in candidate.metrics),
                "forward_launch_count": candidate.launch_receipt.forward_launch_count,
                "backward_launch_count": candidate.launch_receipt.backward_launch_count,
                "fallback_count": candidate.launch_receipt.fallback_count,
                "eager_backward_count": candidate.launch_receipt.eager_backward_count,
                "beta_gradient_present": (
                    candidate.launch_receipt.beta_gradient_present
                ),
                "metric_hashes": {
                    metric.name: metric.candidate_hash for metric in candidate.metrics
                },
            }
        )
        metric_count += len(candidate.metrics)
        maximum_absolute_difference = max(
            maximum_absolute_difference,
            *(metric.maximum_absolute_difference for metric in candidate.metrics),
        )
        allclose = allclose and all(metric.allclose for metric in candidate.metrics)
        sign_exact = sign_exact and all(
            metric.sign_exact for metric in candidate.metrics
        )
        result = candidate
    assert result is not None
    lower_ir = build_b4b1_differentiable_lower_ir_v1(_capture(0))
    major, minor = torch.cuda.get_device_capability()
    template = build_b4b2_dense_conv_template_v1(
        lower_ir, compute_capability=f"sm_{major}{minor}"
    )
    schedule = build_b4b2_dense_conv_schedule_v1(template)
    return {
        "status": "validated-b2-3-p-anchor-dense-conv-correctness",
        "run_count": len(rows),
        "metric_count": metric_count,
        "element_count": 5 * (6144 + 6144 + 6 + 6144),
        "maximum_absolute_difference": maximum_absolute_difference,
        "allclose": allclose,
        "sign_exact": sign_exact,
        "template_hash": template.stable_hash(),
        "schedule_hash": schedule.stable_hash(template),
        "module_receipt_hash": result.module_receipt.stable_hash(template, schedule),
        "device": torch.cuda.get_device_name(),
        "compute_capability": template.compute_capability,
        "rows": rows,
        "observed_workspace_inventory": [
            {"name": name, "shape": list(shape)}
            for name, shape in result.module_receipt.observed_workspace_inventory
        ],
        "structural_workspace_check": (
            result.module_receipt.structural_workspace_check
        ),
        "beta_gradient_present": False,
        "performance_claimed": False,
    }


def main() -> None:
    """Print one canonical five-fresh dense Conv correctness summary."""

    print(json.dumps(run_gate(), sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
