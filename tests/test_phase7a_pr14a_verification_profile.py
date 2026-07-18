"""PR-14A coverage profile and external αβ-CROWN adapter contracts."""

# pylint: disable=too-few-public-methods

from dataclasses import replace
import json

import torch

from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.planner.materialization import BoundMethod, OptimizationStage
from boundflow.runtime.abcrown_adapter import ABCrownBoundQueryProfiler
from boundflow.runtime.bab_query import make_bound_query
from boundflow.runtime.task_executor import InputSpec
from boundflow.runtime.verification_profile import (
    VerificationCoverageReport,
    VerificationQueryProfile,
    module_layer_pattern,
)


def _mlp_module() -> BFTaskModule:
    return BFTaskModule(
        tasks=[
            BoundTask(
                task_id="t0",
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
        entry_task_id="t0",
        bindings={
            "params": {
                "W1": torch.ones((4, 2)),
                "b1": torch.zeros(4),
                "W2": torch.ones((2, 4)),
                "b2": torch.zeros(2),
            }
        },
    )


def _alpha_beta_query(module: BFTaskModule):
    query, _ = make_bound_query(
        module=module,
        query_id="q0",
        parent_query_id=None,
        sequence_number=0,
        example_idx=0,
        input_spec=InputSpec.linf(
            value_name="input", center=torch.zeros((3, 2)), eps=0.1
        ),
        linear_spec_c=torch.ones((3, 5, 2)),
        split_by_relu_input={"h1": torch.zeros((3, 4), dtype=torch.int8)},
        warm_alpha_by_relu_input={"h1": torch.full((3, 4), 0.5)},
        warm_beta_by_relu_input={"h1": torch.zeros((3, 4))},
        bound_method=BoundMethod.ALPHA_BETA_CROWN,
        execution_options={
            "alpha_steps": 1,
            "alpha_lr": 0.1,
            "alpha_init": 0.5,
            "beta_init": 0.0,
            "objective": "lower",
            "spec_reduce": "mean",
            "soft_tau": 1.0,
            "lb_weight": 1.0,
            "ub_weight": 1.0,
        },
    )
    return query


def test_profile_is_derived_from_bound_query_and_rejects_alpha_beta() -> None:
    """Real αβ/split state must be visible and rejected by current fused TIR."""

    module = _mlp_module()
    query = _alpha_beta_query(module)

    profile = VerificationQueryProfile.from_bound_query(
        query,
        solver_phase="activation_bab_bound",
        layer_pattern=module_layer_pattern(module),
    )

    assert profile.query_id == query.query_id
    assert profile.bound_method == "alpha-beta-CROWN"
    assert profile.domain_size == 3
    assert profile.spec_size == 5
    assert profile.alpha_enabled and profile.beta_enabled and profile.split_state
    assert not profile.backend_eligible
    assert {
        "requires_grad_unsupported",
        "alpha_unsupported",
        "beta_unsupported",
        "split_state_unsupported",
        "optimization_stage_unsupported",
    }.issubset(profile.reason_if_not)


def test_plain_crown_final_bound_has_a_legal_linear_candidate() -> None:
    """A CUDA plain-CROWN final-bound region must expose the legal capability."""

    module = _mlp_module()
    alpha_beta = _alpha_beta_query(module)
    compatibility = replace(
        alpha_beta.compatibility_key,
        bound_method=BoundMethod.CROWN.value,
        optimization_stage=OptimizationStage.FINAL_BOUND.value,
        requires_grad=False,
        split_tensor_shapes=(),
        device="cuda",
    )
    query = replace(
        alpha_beta,
        bound_method=BoundMethod.CROWN,
        optimization_stage=OptimizationStage.FINAL_BOUND,
        requires_grad=False,
        split_signature="empty",
        device="cuda",
        compatibility_key=compatibility,
        execution_options={"split_state_present": False},
    )

    profile = VerificationQueryProfile.from_bound_query(
        query,
        solver_phase="final_bound",
        layer_pattern=module_layer_pattern(module),
    )

    assert profile.backend_eligible
    assert not profile.reason_if_not
    assert profile.eligible_capability_ids == (
        "tvm_fused_tir_linear_plain_crown_fp32_static_v1",
    )


def test_coverage_report_keeps_all_rejection_reasons() -> None:
    """Coverage aggregation must not erase unsupported-query evidence."""

    module = _mlp_module()
    rejected = VerificationQueryProfile.from_bound_query(
        _alpha_beta_query(module),
        solver_phase="activation_bab_bound",
        layer_pattern=module_layer_pattern(module),
    )
    report = VerificationCoverageReport.from_profiles([rejected])

    assert report.total_queries == 1
    assert report.eligible_queries == 0
    assert report.eligible_percent == 0.0
    assert report.rejection_reasons["beta_unsupported"] == 1


class BoundLinear:
    """Fake external node carrying the same class-name convention as auto_LiRPA."""

    alpha = {"out": torch.ones((2, 1, 1))}


class BoundRelu:
    """Fake external ReLU state with a split beta tensor."""

    sparse_beta = torch.ones((2, 1))


class FakeBoundedModule:
    """Small stand-in for the external BoundedModule observer boundary."""

    def __init__(self) -> None:
        self.bound_opts = {"optimize_bound_args": {"enable_beta_crown": True}}

    def nodes(self):
        """Return a verifier-style operator sequence."""

        return [BoundLinear(), BoundRelu(), BoundLinear()]

    def compute_bounds(  # pylint: disable=invalid-name,unused-argument
        self, x=None, C=None, method="backward", **_kwargs
    ):
        """Return a deterministic sentinel result for observer transparency."""

        tensor = x[0]
        return tensor.sum() + C.sum(), None


def test_external_adapter_is_observational_reversible_and_writes_artifacts(
    tmp_path,
) -> None:
    """The adapter must preserve results, restore methods, and close artifacts."""

    original = FakeBoundedModule.compute_bounds
    profiler = ABCrownBoundQueryProfiler(
        model_structure_hash="model-hash",
        weight_version="weight-hash",
        phase_resolver=lambda: "activation_bab_bound",
    )
    module = FakeBoundedModule()
    x = torch.ones((2, 4))
    spec = torch.ones((2, 3, 2))

    with profiler.instrument(FakeBoundedModule):
        actual, _ = module.compute_bounds(x=(x,), C=spec, method="CROWN-Optimized")

    assert actual.item() == x.sum().item() + spec.sum().item()
    assert FakeBoundedModule.compute_bounds is original
    assert len(profiler.queries) == len(profiler.profiles) == 1
    query = profiler.queries[0]
    profile = profiler.profiles[0]
    assert query.bound_method == BoundMethod.ALPHA_BETA_CROWN
    assert query.execution_options["solver_phase"] == "activation_bab_bound"
    assert profile.layer_pattern == ("linear", "relu", "linear")
    assert profile.domain_size == 2 and profile.spec_size == 3

    profiler.write_artifacts(tmp_path)
    coverage = json.loads((tmp_path / "coverage.json").read_text())
    assert coverage["total_queries"] == 1
    assert (tmp_path / "queries.jsonl").read_text().count("\n") == 1
    assert (tmp_path / "profiles.jsonl").read_text().count("\n") == 1
