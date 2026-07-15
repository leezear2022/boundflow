"""PR-13A query identity, compatibility, and state-validity tests."""

# pylint: disable=duplicate-code

from dataclasses import replace

import pytest
import torch

from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.planner.materialization import BoundMethod
from boundflow.runtime.bab_query import (
    BoundQueryRequest,
    ReuseClass,
    StateArtifactKind,
    StateValidityManager,
    build_query_batch,
    compatibility_groups,
    make_bound_query,
)
from boundflow.runtime.task_executor import InputSpec


def _make_module() -> BFTaskModule:
    task = BoundTask(
        task_id="t0",
        kind=TaskKind.INTERVAL_IBP,
        ops=[
            TaskOp(
                op_type="linear",
                name="linear1",
                inputs=["input", "W1", "b1"],
                outputs=["h1"],
            ),
            TaskOp(op_type="relu", name="relu1", inputs=["h1"], outputs=["r1"]),
            TaskOp(
                op_type="linear",
                name="linear2",
                inputs=["r1", "W2", "b2"],
                outputs=["out"],
            ),
        ],
        input_values=["input"],
        output_values=["out"],
    )
    params = {
        "W1": torch.tensor([[1.0], [-1.0]], dtype=torch.float32),
        "b1": torch.tensor([0.0, 0.0], dtype=torch.float32),
        "W2": torch.tensor([[1.0, -1.0]], dtype=torch.float32),
        "b2": torch.tensor([0.0], dtype=torch.float32),
    }
    return BFTaskModule(tasks=[task], entry_task_id="t0", bindings={"params": params})


def _options() -> dict[str, object]:
    return {
        "alpha_steps": 0,
        "alpha_lr": 0.2,
        "alpha_init": 0.5,
        "beta_init": 0.0,
        "objective": "lower",
        "spec_reduce": "mean",
        "soft_tau": 1.0,
        "lb_weight": 1.0,
        "ub_weight": 1.0,
    }


def _make_query(
    module: BFTaskModule,
    *,
    query_id: str,
    parent_query_id: str | None,
    sequence_number: int,
    split: torch.Tensor,
):
    return make_bound_query(
        module=module,
        query_id=query_id,
        parent_query_id=parent_query_id,
        sequence_number=sequence_number,
        example_idx=0,
        input_spec=InputSpec.linf(
            value_name="input",
            center=torch.tensor([[0.0]], dtype=torch.float32),
            eps=1.0,
        ),
        linear_spec_c=None,
        split_by_relu_input={"h1": split},
        warm_alpha_by_relu_input={"h1": torch.full((2,), 0.5)},
        warm_beta_by_relu_input={"h1": torch.zeros(2)},
        bound_method=BoundMethod.ALPHA_BETA_CROWN,
        execution_options=_options(),
    )


def test_pr13a_query_identity_is_deterministic_and_dense_capability_only() -> None:
    """The same logical call must serialize identically and remain dense-only."""

    module = _make_module()
    args = {
        "query_id": "bab-e0-n0",
        "parent_query_id": None,
        "sequence_number": 0,
        "split": torch.zeros(2, dtype=torch.int8),
    }
    query1, payload1 = _make_query(module, **args)
    query2, payload2 = _make_query(module, **args)

    assert query1.canonical_json() == query2.canonical_json()
    assert query1.compatibility_key == query2.compatibility_key
    assert query1.compatibility_key.backend_capability_class == "alpha_beta_dense_split"
    assert payload1 is not payload2
    assert (
        payload1.input_spec.center.data_ptr() != payload2.input_spec.center.data_ptr()
    )


def test_pr13a_state_validity_for_parent_child_is_explicit() -> None:
    """Parent state may only cross a split under artifact-specific rules."""

    module = _make_module()
    parent, _ = _make_query(
        module,
        query_id="bab-e0-n0",
        parent_query_id=None,
        sequence_number=0,
        split=torch.zeros(2, dtype=torch.int8),
    )
    child, _ = _make_query(
        module,
        query_id="bab-e0-n1",
        parent_query_id=parent.query_id,
        sequence_number=1,
        split=torch.tensor([-1, 0], dtype=torch.int8),
    )
    manager = StateValidityManager()

    assert (
        manager.classify(parent, child, StateArtifactKind.ALPHA_STATE)
        == ReuseClass.WARM_START_ONLY
    )
    assert (
        manager.classify(parent, child, StateArtifactKind.INTERMEDIATE_BOUNDS)
        == ReuseClass.WARM_START_ONLY
    )
    assert (
        manager.classify(parent, child, StateArtifactKind.BETA_STATE)
        == ReuseClass.INVALIDATE
    )
    assert (
        manager.classify(parent, child, StateArtifactKind.FINAL_BOUNDS)
        == ReuseClass.INVALIDATE
    )
    assert (
        manager.classify(parent, child, StateArtifactKind.COMPILED_MODULE)
        == ReuseClass.EXACT_REUSE
    )

    changed_weights = replace(child, weight_version="different")
    assert (
        manager.classify(parent, changed_weights, StateArtifactKind.COMPILED_MODULE)
        == ReuseClass.INVALIDATE
    )


def test_pr13a_compatibility_grouping_uses_full_key() -> None:
    """A numeric-policy difference must prevent otherwise compatible batching."""

    module = _make_module()
    root, _ = _make_query(
        module,
        query_id="bab-e0-n0",
        parent_query_id=None,
        sequence_number=0,
        split=torch.zeros(2, dtype=torch.int8),
    )
    same_shape, _ = _make_query(
        module,
        query_id="bab-e0-n1",
        parent_query_id=root.query_id,
        sequence_number=1,
        split=torch.tensor([-1, 0], dtype=torch.int8),
    )
    different_key = replace(
        same_shape,
        query_id="bab-e0-n2",
        sequence_number=2,
        compatibility_key=replace(
            same_shape.compatibility_key,
            numeric_policy="fp32_relaxed",
        ),
    )

    groups = compatibility_groups([root, same_shape, different_key])
    assert sorted(len(group) for group in groups.values()) == [1, 2]

    _, root_payload = _make_query(
        module,
        query_id="bab-e0-n0",
        parent_query_id=None,
        sequence_number=0,
        split=torch.zeros(2, dtype=torch.int8),
    )
    _, different_payload = _make_query(
        module,
        query_id="bab-e0-n2",
        parent_query_id=root.query_id,
        sequence_number=2,
        split=torch.tensor([1, 0], dtype=torch.int8),
    )
    with pytest.raises(ValueError, match="incompatible query"):
        build_query_batch(
            [
                BoundQueryRequest(root, root_payload),
                BoundQueryRequest(different_key, different_payload),
            ],
            estimated_peak_bytes=1024,
            memory_budget_bytes=2048,
        )
