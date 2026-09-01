#!/usr/bin/env python3
"""Compile the S4 exact-call tensor-free AOT plan template."""

# pylint: disable=wrong-import-position,import-error

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from boundflow.frontends.onnx.frontend import import_onnx
from boundflow.planner import plan_interval_ibp_v0
from boundflow.runtime.asplos27_s4_exact_call_plan_template import (
    compile_s4_exact_call_plan_template_v1,
)
from boundflow.runtime.fsg4_b3_prepared_core import prepare_core_template_v1
from boundflow.runtime.rvir_v4_pre_state_initializer import (
    initialize_rvir_v4_native_pre_state,
)
from boundflow.runtime.rvir_v4_production_state import (
    production_snapshot_from_payload_v4,
    ProductionTensorOwnership,
    ProductionTensorRole,
)
from scripts.run_rvir_v4_pre_state_artifact import TOPOLOGY


def _one(snapshot, role: ProductionTensorRole) -> torch.Tensor:
    values = [item.value for item in snapshot.tensors if item.role == role]
    if len(values) != 1:
        raise ValueError(f"S4 AOT generator requires one {role.value}")
    return values[0]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-capture",
        type=Path,
        default=REPOSITORY_ROOT
        / "artifacts/rvir-v4-pre-state/resnet2b-core-pre-state-v1/source_capture.pt",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=REPOSITORY_ROOT.parent
        / "vnncomp2021/benchmarks/cifar10_resnet/onnx/resnet_2b.onnx",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT
        / "artifacts/asplos27-s4-exact-call-plan/resnet2b-prop0-v1/plan_template.json",
    )
    return parser.parse_args()


def main() -> None:
    """Compile and persist the canonical production-shape template."""

    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("S4 AOT generator requires CUDA")
    device = torch.device("cuda", torch.cuda.current_device())
    raw = torch.load(
        args.source_capture.resolve(), map_location="cpu", weights_only=True
    )
    snapshot = production_snapshot_from_payload_v4(raw["cores"][0]["pre_snapshot"])
    mapping = initialize_rvir_v4_native_pre_state(snapshot, TOPOLOGY)
    program = import_onnx(
        str(args.model.resolve()), do_shape_infer=True, normalize=True
    )
    module = plan_interval_ibp_v0(program)
    mutable_paths = tuple(
        sorted(
            item.semantic_path
            for item in snapshot.tensors
            if item.ownership == ProductionTensorOwnership.MUTABLE_COPY_OUT
        )
    )
    lower = _one(snapshot, ProductionTensorRole.INPUT_LOWER)
    objective = _one(snapshot, ProductionTensorRole.LINEAR_SPEC)
    core_template = prepare_core_template_v1(
        template_id="resnet2b-prop0-b4-a",
        program=program,
        module=module,
        topology=TOPOLOGY,
        device=device,
        dtype=torch.float32,
        input_shape=lower.shape,
        objective_shape=objective.shape,
        mutable_paths=mutable_paths,
    )
    major, minor = torch.cuda.get_device_capability(device)
    plan_template = compile_s4_exact_call_plan_template_v1(
        template_id="resnet2b-prop0-s4-exact-call",
        core_template=core_template,
        snapshot=snapshot,
        mapping=mapping,
        topology=TOPOLOGY,
        compute_capability=f"sm_{major}{minor}",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(plan_template.to_dict(), sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "template_hash": plan_template.stable_hash(),
                "core_template_hash": core_template.stable_hash(),
                "source_capture_runtime_dependency": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
