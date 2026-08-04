"""Fail-closed artifact tests for NRIR-30 staged escalation evidence."""

import copy
from dataclasses import replace

import pytest
import torch

from boundflow.runtime.native_hard_clause_escalation import (
    NativeHardClauseEscalationExecution,
    NativeHardClauseEscalationTrace,
    compile_native_hard_clause_escalation_program,
)
from scripts.run_typed_hard_clause_escalation_artifact import (
    _validate_program,
    canonical_hash,
)
from tests.test_native_hard_clause_escalation import (
    _execute,
    _fixture,
    _policies,
)


def _program_payload() -> dict[str, object]:
    module, spec = _fixture()
    search_policy, optimizer_policy = _policies()
    program = compile_native_hard_clause_escalation_program(
        module,
        spec,
        linear_spec_C=torch.tensor([[[1.0, -1.0]]]),
        thresholds=torch.tensor([-0.1]),
        plan_id="hard-clause-artifact-toy",
        search_policy=search_policy,
        optimizer_policy=optimizer_policy,
    )
    return program.to_dict()


def test_escalation_artifact_recomputes_plan_task_schedule_links() -> None:
    payload = _program_payload()
    _validate_program(payload)

    tampered = copy.deepcopy(payload)
    tampered["plan"]["escalation_budget"]["max_nodes"] = 63
    tampered["plan_hash"] = canonical_hash(tampered["plan"])
    with pytest.raises(ValueError, match="program IR differs"):
        _validate_program(tampered)


def test_escalation_runtime_rejects_synced_aggregate_upgrade() -> None:
    execution = _execute(-0.1)
    trace = execution.trace
    tampered = replace(
        trace,
        final_status="verified",
        final_verified_clause_indices=(0,),
        final_unresolved_clause_indices=(),
        semantic_signature_hash="",
    )
    tampered = NativeHardClauseEscalationTrace(
        **{
            **tampered.__dict__,
            "semantic_signature_hash": canonical_hash(tampered.semantic_dict()),
        }
    )
    forged = NativeHardClauseEscalationExecution(
        program=execution.program,
        baseline=execution.baseline,
        refinement_program=execution.refinement_program,
        refinement=execution.refinement,
        escalation=execution.escalation,
        trace=tampered,
    )
    module, spec = _fixture()
    search_policy, optimizer_policy = _policies()

    with pytest.raises(ValueError, match="child/aggregate semantics differ"):
        forged.validate_against(
            module,
            spec,
            linear_spec_C=torch.tensor([[[1.0, -1.0]]]),
            thresholds=torch.tensor([-0.1]),
            search_policy=search_policy,
            optimizer_policy=optimizer_policy,
        )
