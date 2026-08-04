"""Typed NRIR-33 child-budget selection and artifact tests."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from boundflow.ir.objective_ancestral_child_budget import (
    CHILD_BUDGET_CANDIDATE_CAPS,
    NativeObjectiveAncestralChildBudgetCalibrationIR,
    NativeObjectiveAncestralChildBudgetPolicyIR,
    compile_frozen_child_budget_decision,
)
from boundflow.ir.refinement import NativeIntermediateRefinementPolicyIR
from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime.native_alpha_beta_optimization_state import (
    NativeAlphaBetaOptimizerPolicy,
)
from boundflow.runtime.native_intermediate_refinement import (
    compile_native_intermediate_refinement_program,
    execute_native_intermediate_refinement_program,
)
from boundflow.runtime.native_objective_ancestral_child_budget import (
    compile_native_objective_ancestral_child_budget_plan,
)
from boundflow.runtime.task_executor import InputSpec
from scripts.run_objective_ancestral_child_budget_pilot import (
    ARTIFACT_DIR,
    validate_pilot,
)


def _rows() -> tuple[NativeObjectiveAncestralChildBudgetCalibrationIR, ...]:
    worst = {8: -115.0, 16: -110.0, 32: -105.0, 64: -102.0, 128: -100.0}
    return tuple(
        NativeObjectiveAncestralChildBudgetCalibrationIR(
            cap=cap,
            root_lower=-204.0,
            worst_active_lower=worst[cap],
            accepted_nodes=7,
            lineage_valid=True,
            result_hash=f"{cap:064x}",
        )
        for cap in CHILD_BUDGET_CANDIDATE_CAPS
    )


def test_child_budget_selects_smallest_cap_retaining_ninety_percent() -> None:
    policy = NativeObjectiveAncestralChildBudgetPolicyIR()
    decision = compile_frozen_child_budget_decision(
        policy,
        calibration_evidence_hash="f" * 64,
        root_global_worst_active_lower=-200.0,
        calibration_rows=_rows(),
    )

    assert decision.selected_cap == 16
    assert decision.reference_gain == 100.0
    assert decision.selected_gain_retention == 0.90


def test_child_budget_rejects_synchronized_winner_tamper() -> None:
    policy = NativeObjectiveAncestralChildBudgetPolicyIR()
    decision = compile_frozen_child_budget_decision(
        policy,
        calibration_evidence_hash="f" * 64,
        root_global_worst_active_lower=-200.0,
        calibration_rows=_rows(),
    )

    with pytest.raises(ValueError, match="winner differs"):
        replace(decision, selected_cap=32).validate_against(policy)


def _toy_root():
    module = BFTaskModule(
        tasks=[
            BoundTask(
                task_id="nrir33-toy",
                kind=TaskKind.INTERVAL_IBP,
                ops=[
                    TaskOp("linear", "linear1", ["input", "W1", "b1"], ["h1"]),
                    TaskOp("relu", "relu1", ["h1"], ["r1"]),
                    TaskOp("linear", "linear2", ["r1", "W2", "b2"], ["out"]),
                ],
                input_values=["input"],
                output_values=["out"],
            )
        ],
        entry_task_id="nrir33-toy",
        bindings={
            "params": {
                "W1": torch.tensor([[1.0, -0.5], [-0.25, 0.75]]),
                "b1": torch.tensor([0.1, -0.2]),
                "W2": torch.tensor([[0.75, -1.0], [-0.5, 0.25]]),
                "b2": torch.tensor([0.15, -0.1]),
            }
        },
    )
    spec = InputSpec.box(
        value_name="input",
        lower=torch.tensor([[-0.3, -0.6]]),
        upper=torch.tensor([[0.7, 0.4]]),
    )
    objective = torch.tensor([[[1.0, -1.0]]])
    shared_program = compile_native_intermediate_refinement_program(
        module,
        spec,
        policy=NativeIntermediateRefinementPolicyIR(
            passes=1, max_neurons_per_relu=2, backward_chunk_size=2
        ),
        plan_id="nrir33-toy:shared",
    )
    shared = execute_native_intermediate_refinement_program(
        shared_program, module, spec
    )
    root_program = compile_native_intermediate_refinement_program(
        module,
        spec,
        policy=NativeIntermediateRefinementPolicyIR(
            passes=1,
            max_neurons_per_relu=2,
            backward_chunk_size=2,
            candidate_policy_id="objective_influence_width_per_relu_v1",
        ),
        plan_id="nrir33-toy:root",
        linear_spec_C=objective,
        source_refinement_execution=shared,
    )
    root = execute_native_intermediate_refinement_program(root_program, module, spec)
    return module, spec, objective, torch.tensor([-0.1]), root


def test_child_budget_plan_binds_cap_without_changing_nrir32_engine() -> None:
    module, spec, objective, threshold, root = _toy_root()
    optimizer = NativeAlphaBetaOptimizerPolicy(steps=1, lr=0.1)
    plan = compile_native_objective_ancestral_child_budget_plan(
        module,
        spec,
        linear_spec_C=objective,
        threshold=threshold,
        root_refinement=root,
        optimizer_policy=optimizer,
        plan_id="nrir33-toy:cap16",
        selected_cap=16,
    )

    assert plan.child_budget_decision.selected_cap == 16
    assert plan.child_refinement_policy.max_neurons_per_relu == 16
    assert plan.to_dict()["semantics_owner"] == (
        "boundflow_native_objective_ancestral_child_budget"
    )


def _pilot() -> dict[str, object]:
    path = Path(__file__).resolve().parents[1] / ARTIFACT_DIR / "pilot.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_frozen_child_budget_pilot_replays_as_cap_only_no_go() -> None:
    pilot = _pilot()
    validate_pilot(pilot)

    assert pilot["selected_cap"] == 128
    payload = pilot["calibration_payload"]
    assert isinstance(payload, dict)
    rows = payload["candidate_results"]
    assert [row["accepted_nodes"] for row in rows] == [7, 7, 7, 7, 7]


def test_frozen_child_budget_pilot_rejects_decision_tamper() -> None:
    pilot = deepcopy(_pilot())
    decision = pilot["decision"]
    assert isinstance(decision, dict)
    decision["selected_cap"] = 64
    pilot["selected_cap"] = 64

    with pytest.raises(ValueError, match="frozen pilot decision differs"):
        validate_pilot(pilot)
