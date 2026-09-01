"""Real ResNet frozen-state evaluation for RVIR-v4."""

# pylint: disable=missing-function-docstring

from pathlib import Path

import torch

from boundflow.frontends.onnx.frontend import import_onnx
from boundflow.planner import plan_interval_ibp_v0
from boundflow.runtime.rvir_v4_frozen_state import (
    ProductionReluTopologyV4,
    evaluate_rvir_v4_frozen_state,
)
from boundflow.runtime.rvir_v4_production_state import (
    production_snapshot_from_payload_v4,
)

ROOT = Path(__file__).resolve().parents[1]
CAPTURE = (
    ROOT / "artifacts/rvir-v4-production-state/resnet2b-core-capture-v2/capture.pt"
)
MODEL = ROOT.parent / "vnncomp2021/benchmarks/cifar10_resnet/onnx/resnet_2b.onnx"
TOPOLOGY = tuple(
    ProductionReluTopologyV4(*values, provider_start_node="/49")
    for values in (
        ("/input-4", "/input", "17"),
        ("/input-12", "/input-8", "19"),
        ("/input-16", "/39", "23"),
        ("/input-24", "/input-20", "25"),
        ("/45", "/44", "28"),
        ("/48", "/input-28", "31"),
    )
)


def test_real_resnet_frozen_state_matches_production_core_lower() -> None:
    payload = torch.load(CAPTURE, map_location="cpu", weights_only=True)
    core = payload["cores"][0]
    pre = production_snapshot_from_payload_v4(core["pre_snapshot"])
    post = production_snapshot_from_payload_v4(core["post_snapshot"])
    program = import_onnx(str(MODEL), do_shape_infer=True, normalize=True)
    module = plan_interval_ibp_v0(program)

    result = evaluate_rvir_v4_frozen_state(
        module=module,
        input_value_name=program.graph.inputs[0],
        pre=pre,
        post=post,
        topology=TOPOLOGY,
        query_id="rvir-v4-v4-1-resnet2b-core-000000",
        expected_lower=core["lower"],
    )

    difference = (result.lower - core["lower"]).abs()
    assert float(difference.max()) <= 2e-4
    assert torch.equal(result.lower >= 0, core["lower"] >= 0)
    assert len(result.ir_hashes) == 10
