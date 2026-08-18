#!/usr/bin/env python3
"""Run one fresh B4-B0 evaluation-zero dual-anchor CUDA capture."""

# pylint: disable=wrong-import-position,too-many-locals,missing-function-docstring

from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path
import sys
from typing import Any, cast

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from boundflow.frontends.onnx.frontend import import_onnx
from boundflow.planner import plan_interval_ibp_v0
from boundflow.runtime.fsg4_b3_prepared_core import (
    instantiate_core_plan_v1,
    prepare_core_template_v1,
)
from boundflow.runtime.fsg4_b3_terminal_optimizer_schedule import (
    compile_terminal_optimizer_schedule_v1,
    execute_terminal_optimizer_schedule_v1,
)
from boundflow.runtime.fsg4_b4b_production_region_capture import (
    B4BRegionLiveObserverV1,
    build_production_differentiable_region_lineage_v1,
    capture_production_differentiable_region_v1,
    production_differentiable_region_capture_to_payload_v1,
)
from boundflow.runtime.rvir_v4_optimizer_mutation import (
    production_optimizer_step_trace_from_payload_v4,
)
from boundflow.runtime.rvir_v4_pre_state_initializer import (
    initialize_rvir_v4_native_pre_state,
)
from boundflow.runtime.rvir_v4_production_state import (
    ProductionTensorOwnership,
    ProductionTensorRole,
    production_snapshot_from_payload_v4,
)
from boundflow.runtime.task_executor import InputSpec
from scripts.run_rvir_v4_pre_state_artifact import TOPOLOGY

WORKER_SCHEMA = "boundflow.fsg4-b4b0-five-fresh-worker/v1"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _move(value: object, *, device: torch.device) -> object:
    if torch.is_tensor(value):
        dtype = torch.float32 if value.is_floating_point() else value.dtype
        return value.to(device=device, dtype=dtype)
    if isinstance(value, dict):
        return {key: _move(item, device=device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move(item, device=device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move(item, device=device) for item in value)
    return value


def _one(snapshot, role: ProductionTensorRole) -> torch.Tensor:
    values = [item.value for item in snapshot.tensors if item.role == role]
    if len(values) != 1:
        raise ValueError(f"FSG4/B4-B0 worker requires one {role.value}")
    return values[0]


def _run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("FSG4/B4-B0 worker requires CUDA")
    device = torch.device("cuda:0")
    raw = torch.load(args.source_capture, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict):
        raise TypeError("FSG4/B4-B0 source capture root differs")
    snapshot = production_snapshot_from_payload_v4(raw["cores"][0]["pre_snapshot"])
    production = production_optimizer_step_trace_from_payload_v4(
        raw["optimizer_step_traces"][0]
    )
    cpu_mapping = initialize_rvir_v4_native_pre_state(snapshot, TOPOLOGY)
    mapping = cpu_mapping.to(device=device, dtype=torch.float32)
    program = import_onnx(str(args.model), do_shape_infer=True, normalize=True)
    module = plan_interval_ibp_v0(program)
    module.bindings = copy.deepcopy(
        cast(dict[str, Any], _move(module.bindings, device=device))
    )
    lower = _one(snapshot, ProductionTensorRole.INPUT_LOWER).to(device)
    upper = _one(snapshot, ProductionTensorRole.INPUT_UPPER).to(device)
    objective = _one(snapshot, ProductionTensorRole.LINEAR_SPEC).to(device)
    mutable_paths = tuple(
        sorted(
            item.semantic_path
            for item in snapshot.tensors
            if item.ownership == ProductionTensorOwnership.MUTABLE_COPY_OUT
        )
    )
    template = prepare_core_template_v1(
        template_id="resnet2b-prop0-b4-b0-five-fresh",
        program=program,
        module=module,
        topology=TOPOLOGY,
        device=str(device),
        dtype=torch.float32,
        input_shape=lower.shape,
        objective_shape=objective.shape,
        mutable_paths=mutable_paths,
    )
    spec = InputSpec.box(value_name=program.graph.inputs[0], lower=lower, upper=upper)
    instance = instantiate_core_plan_v1(
        template=template,
        topology=TOPOLOGY,
        snapshot=snapshot,
        mapping=mapping,
        input_spec=spec,
        linear_spec_C=objective,
        mutation_policy=production.mutation_policy,
    )
    initial = instance.initial_state
    observer = B4BRegionLiveObserverV1()
    schedule = compile_terminal_optimizer_schedule_v1()
    result = execute_terminal_optimizer_schedule_v1(
        module,
        spec,
        linear_spec_C=objective,
        relu_pre=mapping.relu_pre,
        initial_state=initial,
        mutation_policy=production.mutation_policy,
        schedule=schedule,
        prevalidated_plan=instance,
        b4b_region_observer=observer,
    )
    production_by_path = {
        item.semantic_path: item.value.to(device)
        for item in snapshot.tensors
        if item.semantic_path
        in {
            anchor.production_alpha_path
            for anchor in (observation.anchor for observation in observer.observations)
        }
        | {
            anchor.production_beta_path
            for anchor in (observation.anchor for observation in observer.observations)
        }
    }
    captures = []
    for observation in observer.observations:
        anchor = observation.anchor
        values = {
            "incoming_lower_a": observation.incoming_lower_a,
            "preactivation_lower": observation.preactivation_lower,
            "preactivation_upper": observation.preactivation_upper,
            "production_alpha": production_by_path[anchor.production_alpha_path],
            "native_alpha": observation.native_alpha,
            "production_beta": production_by_path[anchor.production_beta_path],
            "native_beta": observation.native_beta,
            "operator_weight": observation.operator_weight,
            "output_lower_a": observation.output_lower_a,
            "output_bias": observation.output_bias,
            "loss_seed": observation.loss_seed,
        }
        if observation.relu_pre_add_coeff_l is not None:
            values["relu_pre_add_coeff_l"] = observation.relu_pre_add_coeff_l
        gradients = {"native_alpha": observation.native_alpha_gradient}
        if observation.native_beta_gradient is not None:
            gradients["native_beta"] = observation.native_beta_gradient
        if observation.incoming_lower_a_gradient is not None:
            gradients["incoming_lower_a"] = observation.incoming_lower_a_gradient
        capture = capture_production_differentiable_region_v1(
            source_state_hash=initial.stable_hash(),
            primal_graph_hash=initial.scope.primal_graph_hash,
            split_state_hash=initial.scope.split_state_hash,
            topology_hash=mapping.identity.topology_hash,
            anchor=anchor,
            production_lineage=build_production_differentiable_region_lineage_v1(
                snapshot, cpu_mapping, anchor
            ),
            values=values,
            gradients=gradients,
            operator_attributes=dict(observation.operator_attributes),
        )
        captures.append(production_differentiable_region_capture_to_payload_v1(capture))
    torch.cuda.synchronize(device)
    properties = torch.cuda.get_device_properties(device)
    payload: dict[str, object] = {
        "schema_version": WORKER_SCHEMA,
        "run_index": args.run_index,
        "source_capture_sha256": _file_sha256(args.source_capture),
        "model_sha256": _file_sha256(args.model),
        "source_state_hash": initial.stable_hash(),
        "terminal_state_hash": result.terminal_state.stable_hash(),
        "schedule_hash": schedule.stable_hash(),
        "evaluation_count": result.evaluation_count,
        "update_count": result.update_count,
        "captures": captures,
        "environment": {
            "torch_version": str(torch.__version__),
            "cuda_runtime": torch.version.cuda,
            "gpu_name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "device_index": device.index,
        },
        "performance_claimed": False,
    }
    torch.save(payload, args.result)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-capture", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run-index", type=int, required=True)
    parser.add_argument("--result", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.source_capture = args.source_capture.resolve()
    args.model = args.model.resolve()
    args.result = args.result.resolve()
    if args.run_index < 0:
        raise ValueError("FSG4/B4-B0 run index differs")
    payload = _run(args)
    captures = cast(list[object], payload["captures"])
    print(
        f"B4-B0 run={payload['run_index']} captures={len(captures)} "
        "performance_claimed=false",
        flush=True,
    )


if __name__ == "__main__":
    main()
