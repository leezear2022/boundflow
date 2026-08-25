#!/usr/bin/env python3
"""Run one fresh R3-3 S-anchor active-beta correctness worker."""

# pylint: disable=wrong-import-position,too-many-locals,missing-function-docstring
# pylint: disable=protected-access

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boundflow.runtime import fsg4_b4b2_sparse_linear_tir as sparse_linear
from boundflow.runtime.fsg4_b4b1_pytorch_reference import (
    build_b4b1_differentiable_lower_instance_v1,
    build_b4b1_differentiable_lower_ir_v1,
    run_b4b1_pytorch_reference_v1,
)
from boundflow.runtime.fsg4_b4b1_reference_capture import (
    production_differentiable_reference_capture_from_payload_v1,
)
from boundflow.runtime.rvir_v4_production_state import production_tensor_sha256
from scripts.run_fsg4_b4b1_pytorch_reference_artifact import (
    _reference_execution_policy,
)

CAPTURE_ARTIFACT = ROOT / "artifacts/fsg4-b4b1-reference-five-fresh/resnet2b-prop0-v1"
SCHEMA = "boundflow.r3-3-active-beta-worker/v1"


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _capture(run_ordinal: int, anchor_ordinal: int):  # type: ignore[no-untyped-def]
    path = CAPTURE_ARTIFACT / f"run_{run_ordinal:02d}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return path, production_differentiable_reference_capture_from_payload_v1(
        payload["captures"][anchor_ordinal]
    )


def _run(run_ordinal: int) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("R3-3 worker requires CUDA")
    capture_path, capture = _capture(run_ordinal, 0)
    lower_ir = build_b4b1_differentiable_lower_ir_v1(capture)
    lower_instance = build_b4b1_differentiable_lower_instance_v1(capture, lower_ir)
    major, minor = torch.cuda.get_device_capability()
    template = sparse_linear.build_b4b2_sparse_linear_template_v1(
        lower_ir, capture, compute_capability=f"sm_{major}{minor}"
    )
    schedule = sparse_linear.build_b4b2_sparse_linear_schedule_v1(template)
    tensors = sparse_linear.build_b4b2_sparse_linear_tensors_v1(
        capture, template, device=torch.device("cuda:0")
    )
    instance = sparse_linear.build_b4b2_sparse_linear_instance_v1(
        template,
        lower_ir,
        lower_instance,
        capture,
        tensors,
        fresh_run_ordinal=run_ordinal,
    )
    with _reference_execution_policy():
        result = sparse_linear.run_b4b2_sparse_linear_tir_v1(
            capture,
            fresh_run_ordinal=run_ordinal,
            cache=sparse_linear.DifferentiableLowerSparseLinearModuleCache(),
        )
        reference = run_b4b1_pytorch_reference_v1(capture, lower_ir, lower_instance)
    _path, empty_capture = _capture(run_ordinal, 1)
    empty_ir = build_b4b1_differentiable_lower_ir_v1(empty_capture)
    empty_rejected = False
    try:
        sparse_linear.build_b4b2_sparse_linear_template_v1(
            empty_ir, empty_capture, compute_capability="sm_89"
        )
    except ValueError as caught:
        empty_rejected = "S-anchor differs" in str(caught)
    if not empty_rejected:
        raise RuntimeError("R3-3 empty-beta specialization was accepted")
    if reference.native_beta_gradient is None:
        raise RuntimeError("R3-3 native beta gradient is absent")
    reference_alpha, reference_beta, unowned_zero = (
        sparse_linear._reference_compressed_gradients(reference, template)
    )
    outputs = {
        "output_lower_a": result.output_lower_a.detach().cpu().contiguous(),
        "output_bias": result.output_bias.detach().cpu().contiguous(),
        "compressed_alpha_gradient": result.compressed_alpha_gradient.detach()
        .cpu()
        .contiguous(),
        "compressed_beta_gradient": result.compressed_beta_gradient.detach()
        .cpu()
        .contiguous(),
    }
    references = {
        "output_lower_a": reference.output_lower_a.detach().cpu().contiguous(),
        "output_bias": reference.output_bias.detach().cpu().contiguous(),
        "compressed_alpha_gradient": reference_alpha.detach().cpu().contiguous(),
        "compressed_beta_gradient": reference_beta.detach().cpu().contiguous(),
    }
    native_gradients = {
        "native_alpha_gradient": reference.native_alpha_gradient.detach()
        .cpu()
        .contiguous(),
        "native_beta_gradient": reference.native_beta_gradient.detach()
        .cpu()
        .contiguous(),
    }
    payload: dict[str, object] = {
        "schema_version": SCHEMA,
        "run_ordinal": run_ordinal,
        "capture_sha256": _file_hash(capture_path),
        "template_hash": template.stable_hash(),
        "schedule_hash": schedule.stable_hash(template),
        "module_receipt_hash": result.module_receipt.stable_hash(template, schedule),
        "template": template.to_dict(),
        "schedule": schedule.to_dict(template),
        "instance": instance.to_dict(template),
        "metrics": [metric.to_dict() for metric in result.metrics],
        "outputs": outputs,
        "references": references,
        "native_gradients": native_gradients,
        "output_hashes": {
            name: production_tensor_sha256(value) for name, value in outputs.items()
        },
        "reference_hashes": {
            name: production_tensor_sha256(value) for name, value in references.items()
        },
        "native_gradient_hashes": {
            name: production_tensor_sha256(value)
            for name, value in native_gradients.items()
        },
        "projection_receipt": result.projection_receipt.to_dict(template, instance),
        "module_receipt": result.module_receipt.to_dict(template, schedule),
        "launch_receipt": result.launch_receipt.to_dict(
            template,
            instance,
            schedule,
            result.module_receipt,
            result.projection_receipt,
        ),
        "alpha_feature_indices": list(template.alpha_feature_indices),
        "beta_locations": list(template.beta_locations),
        "beta_signs": list(template.beta_signs),
        "empty_beta_specialization_rejected": empty_rejected,
        "unowned_native_zero_exact": unowned_zero,
        "beta_nonzero_count": int(
            torch.count_nonzero(outputs["compressed_beta_gradient"]).item()
        ),
        "timing_recorded": False,
        "performance_claimed": False,
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-ordinal", type=int, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    if args.run_ordinal not in range(5):
        raise ValueError("R3-3 run ordinal differs")
    payload = _run(args.run_ordinal)
    torch.save(payload, args.result)
    print(
        f"R3-3 run={args.run_ordinal} beta_nonzero={payload['beta_nonzero_count']} "
        "performance_claimed=false",
        flush=True,
    )


if __name__ == "__main__":
    main()
