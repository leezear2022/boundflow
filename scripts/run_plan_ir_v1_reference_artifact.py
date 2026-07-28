"""Generate or independently replay the deterministic Plan IR v1 smoke artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch

from boundflow.frontends.plain_crown_bound_ir import build_plain_crown_bound_ir
from boundflow.ir.bound import BoundMethodKind, BoundRepresentation
from boundflow.ir.plan import (
    BackendCapabilitySpec,
    BackendKind,
    HardwareProfile,
    PlanCost,
    PlanProvenance,
    WorkloadProfile,
)
from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.planner.plan_ir_artifact import (
    verify_plan_selection_artifact,
    write_plan_selection_artifact,
)
from boundflow.planner.plan_ir_builder import (
    BackendEvidence,
    BatchEvidence,
    ReferencePlanEvidence,
    RegionEvidence,
    RepresentationEvidence,
    StorageEvidence,
    build_reference_plan_template,
)
from boundflow.planner.plan_ir_selector import select_plan_instance
from boundflow.runtime.crown_ibp import _forward_ibp_trace_mlp
from boundflow.runtime.task_executor import InputSpec


def build_reference_smoke_inputs():
    """Reconstruct the exact typed Bound IR and PlanTemplate used by the CLI."""

    task_module = BFTaskModule(
        tasks=[
            BoundTask(
                task_id="plan-ir-reference-smoke",
                kind=TaskKind.INTERVAL_IBP,
                ops=[
                    TaskOp(
                        "linear",
                        "linear1",
                        ["input", "weight", "bias"],
                        ["output"],
                    )
                ],
                input_values=["input"],
                output_values=["output"],
            )
        ],
        entry_task_id="plan-ir-reference-smoke",
        bindings={
            "params": {
                "weight": torch.tensor([[1.0, -0.5], [0.25, 0.75]]),
                "bias": torch.tensor([0.1, -0.2]),
            }
        },
    )
    input_spec = InputSpec.linf(
        value_name="input",
        center=torch.zeros(2, 2),
        eps=0.1,
    )
    interval_env, relu_pre = _forward_ibp_trace_mlp(task_module, input_spec)
    bound_module = build_plain_crown_bound_ir(
        task_module,
        input_spec,
        interval_env=interval_env,
        relu_pre=relu_pre,
    ).module
    capability = BackendCapabilitySpec(
        capability_id="reference-dense-cpu-v1",
        backend=BackendKind.REFERENCE,
        supported_methods=(BoundMethodKind.CROWN,),
        supported_op_kinds=tuple(
            dict.fromkeys(op.kind for op in bound_module.graph.ops)
        ),
        supported_representations=(BoundRepresentation.DENSE,),
        supported_dtypes=("float32",),
        supported_devices=("cpu",),
        supports_grad=False,
        supports_alpha=False,
        supports_beta=False,
        supports_split_state=False,
        static_shape_only=True,
    )
    hardware = HardwareProfile(
        profile_id="reference-cpu-smoke-v1",
        device="cpu",
        total_memory_bytes=1 << 30,
        supported_dtypes=("float32",),
        backend_capability_ids=(capability.capability_id,),
        alignment_bytes=16,
    )
    workload = WorkloadProfile(
        profile_id="plain-crown-reference-smoke-v1",
        method=BoundMethodKind.CROWN,
        requires_grad=False,
        alpha_enabled=False,
        beta_enabled=False,
        split_state_present=False,
        static_shapes=True,
        domain_batch_size=2,
        spec_batch_size=2,
        sample_batch_size=1,
        dtype="float32",
        device="cpu",
        numeric_policy="float32_dense_reference",
    )
    cost = PlanCost(
        predicted_latency_ms=0.1,
        predicted_peak_bytes=0,
        compile_cost_ms=0.0,
        setup_cost_ms=0.0,
        confidence=1.0,
        risk_tags=("reference_smoke_only",),
    )
    region_evidence = tuple(
        RegionEvidence(op.op_id, (op.op_id,), cost) for op in bound_module.graph.ops
    )
    representation_evidence = tuple(
        RepresentationEvidence(
            evidence_id=f"dense:{op.op_id}",
            region_evidence_id=op.op_id,
            representation=BoundRepresentation.DENSE,
            required_transition_evidence_ids=(),
            cost=cost,
        )
        for op in bound_module.graph.ops
    )
    backend_evidence = tuple(
        BackendEvidence(
            evidence_id=f"reference:{op.op_id}",
            region_evidence_id=op.op_id,
            representation_evidence_id=f"dense:{op.op_id}",
            capability_id=capability.capability_id,
            cost=cost,
        )
        for op in bound_module.graph.ops
    )
    evidence = ReferencePlanEvidence(
        evidence_set_id="plan-ir-reference-smoke-v1",
        regions=region_evidence,
        transitions=(),
        representations=representation_evidence,
        backends=backend_evidence,
        batches=(
            BatchEvidence(
                evidence_id="full",
                domain_batch_size=2,
                spec_batch_size=2,
                sample_batch_size=1,
                estimated_payload_bytes=32,
                cost=cost,
            ),
        ),
        storage=(
            StorageEvidence(
                evidence_id="dense",
                compatible_batch_evidence_ids=("full",),
                compatible_representation_evidence_ids=tuple(
                    item.evidence_id for item in representation_evidence
                ),
                value_layout_overrides=(),
                arena_id="cpu-main",
                cost=cost,
            ),
        ),
        provenance=(PlanProvenance("artifact_scope", "reference_contract_smoke"),),
    )
    template = build_reference_plan_template(
        bound_module,
        hardware=hardware,
        workload=workload,
        capabilities=(capability,),
        evidence=evidence,
    )
    return bound_module, template


def main(argv: Sequence[str] | None = None) -> int:
    """Generate a new artifact or verify one in a fresh process."""

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--out-dir", type=Path, required=True)
    generate.add_argument("--available-memory-bytes", type=int, default=1 << 30)
    generate.add_argument("--memory-budget-bytes", type=int, default=1 << 30)
    replay = subparsers.add_parser("replay")
    replay.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    bound_module, template = build_reference_smoke_inputs()
    if args.command == "generate":
        instance = select_plan_instance(
            template,
            bound_module=bound_module,
            query_bucket_id="reference-smoke",
            available_memory_bytes=args.available_memory_bytes,
            memory_budget_bytes=args.memory_budget_bytes,
        )
        manifest = write_plan_selection_artifact(
            args.out_dir,
            bound_module=bound_module,
            template=template,
            instance=instance,
        )
        result = {
            "status": "generated",
            "manifest": str(manifest),
            "bound_module_hash": bound_module.stable_hash(),
            "plan_template_hash": template.stable_hash(bound_module=bound_module),
            "plan_instance_hash": instance.stable_hash(
                template=template,
                bound_module=bound_module,
            ),
        }
    else:
        instance = verify_plan_selection_artifact(
            args.artifact_dir,
            bound_module=bound_module,
            template=template,
        )
        result = {
            "status": "replayed",
            "bound_module_hash": bound_module.stable_hash(),
            "plan_template_hash": template.stable_hash(bound_module=bound_module),
            "plan_instance_hash": instance.stable_hash(
                template=template,
                bound_module=bound_module,
            ),
        }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
