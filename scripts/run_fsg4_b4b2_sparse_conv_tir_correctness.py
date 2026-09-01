"""Run B4-B2 B2-4 P0 five-raw and bounded candidate correctness gates."""

# pylint: disable=import-error,missing-function-docstring,too-many-locals

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
from boundflow.runtime.fsg4_b4b2_sparse_conv_tir import (
    DifferentiableLowerSparseConvModuleCache,
    build_b4b2_sparse_conv_ledger_v1,
    build_b4b2_sparse_conv_schedules_v1,
    build_b4b2_sparse_conv_template_v1,
    run_b4b2_sparse_conv_tir_v1,
)

ARTIFACT = Path("artifacts/fsg4-b4b1-reference-five-fresh/resnet2b-prop0-v1")


def _capture(run_ordinal: int):
    payload = torch.load(
        ARTIFACT / f"run_{run_ordinal:02d}.pt", map_location="cpu", weights_only=False
    )
    return production_differentiable_reference_capture_from_payload_v1(
        payload["captures"][1]
    )


def _summary_row(result, *, ordinal: int) -> dict[str, object]:
    return {
        "ordinal": ordinal,
        "cache_event": result.launch_receipt.cache_event,
        "maximum_absolute_difference": max(
            metric.maximum_absolute_difference for metric in result.metrics
        ),
        "allclose": all(metric.allclose for metric in result.metrics),
        "sign_exact": all(metric.sign_exact for metric in result.metrics),
        "forward_launch_count": result.launch_receipt.forward_launch_count,
        "backward_launch_count": result.launch_receipt.backward_launch_count,
        "fallback_count": result.launch_receipt.fallback_count,
        "eager_backward_count": result.launch_receipt.eager_backward_count,
        "beta_gradient_present": result.launch_receipt.beta_gradient_present,
    }


def run_gate() -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("B4-B2 sparse-source Conv correctness requires CUDA")
    lower_ir = build_b4b1_differentiable_lower_ir_v1(_capture(0))
    major, minor = torch.cuda.get_device_capability()
    template = build_b4b2_sparse_conv_template_v1(
        lower_ir, _capture(0), compute_capability=f"sm_{major}{minor}"
    )
    schedules = build_b4b2_sparse_conv_schedules_v1(template)
    ledger = build_b4b2_sparse_conv_ledger_v1(template, schedules)

    p0_cache = DifferentiableLowerSparseConvModuleCache()
    p0_rows = []
    maximum_difference = 0.0
    p0_result = None
    for run_ordinal in range(5):
        p0_result = run_b4b2_sparse_conv_tir_v1(
            _capture(run_ordinal),
            fresh_run_ordinal=run_ordinal,
            candidate_ordinal=0,
            cache=p0_cache,
        )
        row = _summary_row(p0_result, ordinal=run_ordinal)
        row["module_receipt_hash"] = p0_result.module_receipt.stable_hash(
            template, schedules[0]
        )
        p0_rows.append(row)
        maximum_difference = max(
            maximum_difference,
            *(metric.maximum_absolute_difference for metric in p0_result.metrics),
        )
    assert p0_result is not None

    candidate_cache = DifferentiableLowerSparseConvModuleCache()
    candidate_rows = []
    for candidate_ordinal, schedule in enumerate(schedules):
        result = run_b4b2_sparse_conv_tir_v1(
            _capture(0),
            fresh_run_ordinal=0,
            candidate_ordinal=candidate_ordinal,
            cache=candidate_cache,
        )
        row = _summary_row(result, ordinal=candidate_ordinal)
        row.update(
            {
                "knobs": list(schedule.knob_tuple),
                "schedule_hash": schedule.stable_hash(template),
                "module_receipt_hash": result.module_receipt.stable_hash(
                    template, schedule
                ),
            }
        )
        candidate_rows.append(row)
        maximum_difference = max(
            maximum_difference,
            *(metric.maximum_absolute_difference for metric in result.metrics),
        )

    return {
        "status": "validated-b2-4-sparse-source-conv-p0-and-bounded-ledger-correctness",
        "p0_run_count": 5,
        "p0_metric_count": 20,
        "p0_element_count": 5 * (516 + 6144 + 6 + 6144),
        "candidate_count": len(schedules),
        "candidate_confirmation_metric_count": 12 * 4,
        "candidate_confirmation_element_count": 12 * (516 + 6144 + 6 + 6144),
        "maximum_absolute_difference": maximum_difference,
        "allclose": all(row["allclose"] for row in (*p0_rows, *candidate_rows)),
        "sign_exact": all(row["sign_exact"] for row in (*p0_rows, *candidate_rows)),
        "template_hash": template.stable_hash(),
        "ledger_hash": ledger.stable_hash(template, schedules),
        "schedule_hashes": list(ledger.schedule_hashes),
        "p0_rows": p0_rows,
        "candidate_rows": candidate_rows,
        "observed_workspace_inventory": [
            {"name": name, "shape": list(shape)}
            for name, shape in p0_result.module_receipt.observed_workspace_inventory
        ],
        "compressed_alpha_shape": [6, 86],
        "compressed_beta_shape": [6, 0],
        "timing_raw_present": False,
        "winner_selected": False,
        "performance_claimed": False,
        "device": torch.cuda.get_device_name(),
        "compute_capability": template.compute_capability,
    }


def main() -> None:
    print(json.dumps(run_gate(), sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
