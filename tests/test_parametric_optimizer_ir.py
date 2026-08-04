from dataclasses import replace

import pytest

from boundflow.ir.parametric_optimizer import (
    NativeParametricOptimizerInstanceIR,
    NativeParametricOptimizerTemplateIR,
    lower_native_parametric_optimizer_template_ir,
)


def _template() -> NativeParametricOptimizerTemplateIR:
    return NativeParametricOptimizerTemplateIR(
        template_id="query:parametric-template:0000",
        primal_graph_hash="1" * 64,
        input_value_name="input",
        input_nonbatch_shape=(2,),
        input_dtype="torch.float32",
        input_device="cpu",
        objective_shape=(1, 1),
        objective_dtype="torch.float32",
        objective_device="cpu",
        relu_state_layout=(("relu0.pre", (3,), "torch.float32", "cpu"),),
        optimizer_policy_hash="2" * 64,
        steps=2,
        objective="lower",
        spec_reduce="mean",
        intermediate_bound_source="local_forward",
        refine_external_constraints=False,
    )


def test_parametric_optimizer_template_lowers_reusable_task_schedule() -> None:
    template = _template()
    task_ir, schedule = lower_native_parametric_optimizer_template_ir(template)

    template.validate()
    task_ir.validate(template=template)
    schedule.validate(template=template, task_module=task_ir)
    assert len(task_ir.tasks) == 13
    assert len(schedule.actions) == 13
    assert (
        template.cache_key()
        == replace(template, template_id="other-query-template").cache_key()
    )
    assert (
        template.stable_hash()
        != replace(template, template_id="other-query-template").stable_hash()
    )


def test_parametric_optimizer_instance_binds_dynamic_content() -> None:
    template = _template()
    instance = NativeParametricOptimizerInstanceIR(
        instance_id="query:clause:0:eval:0:optimizer",
        template_hash=template.stable_hash(),
        cache_key=template.cache_key(),
        batch_size=2,
        input_region_hash="3" * 64,
        objective_hash="4" * 64,
        intermediate_bounds_hash="5" * 64,
        split_state_hash="6" * 64,
        state_scope_hash="7" * 64,
        initial_state_hash="8" * 64,
        warm_start_kind="monotonic_split_refinement",
    )

    instance.validate()
    assert instance.to_dict()["batch_size"] == 2


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("input_nonbatch_shape", ()),
        (
            "relu_state_layout",
            (
                ("relu0.pre", (3,), "torch.float32", "cpu"),
                ("relu0.pre", (3,), "torch.float32", "cpu"),
            ),
        ),
        ("intermediate_bound_source", "unknown"),
        ("refine_external_constraints", True),
    ),
)
def test_parametric_optimizer_template_rejects_invalid_contract(
    field: str, value: object
) -> None:
    with pytest.raises(ValueError, match="template IR is invalid"):
        replace(_template(), **{field: value}).validate()


def test_parametric_optimizer_schedule_rejects_template_tamper() -> None:
    template = _template()
    task_ir, schedule = lower_native_parametric_optimizer_template_ir(template)

    with pytest.raises(ValueError):
        schedule.validate(template=replace(template, steps=3), task_module=task_ir)
