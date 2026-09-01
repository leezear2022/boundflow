"""FSG4/B3-A static template and dynamic core-plan contracts."""

# pylint: disable=missing-function-docstring,redefined-outer-name
# pylint: disable=import-outside-toplevel,duplicate-code

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from boundflow.frontends.onnx.frontend import import_onnx
from boundflow.planner import plan_interval_ibp_v0
from boundflow.runtime.fsg4_b3_prepared_core import (
    instantiate_core_plan_v1,
    PreparedCoreTemplateCache,
    prepare_core_template_v1,
)
from boundflow.runtime.rvir_v4_native_optimizer import (
    execute_rvir_v4_native_optimizer_trace,
)
from boundflow.runtime.rvir_v4_optimizer_mutation import (
    production_optimizer_step_trace_from_payload_v4,
)
from boundflow.runtime.rvir_v4_pre_state_initializer import (
    initialize_rvir_v4_native_pre_state,
)
from boundflow.runtime.rvir_v4_production_state import (
    production_snapshot_from_payload_v4,
    ProductionTensorOwnership,
    ProductionTensorRole,
)
from boundflow.runtime.task_executor import InputSpec
from scripts.run_rvir_v4_pre_state_artifact import TOPOLOGY

ROOT = Path(__file__).resolve().parents[1]
CAPTURE = (
    ROOT / "artifacts/rvir-v4-pre-state/resnet2b-core-pre-state-v1/source_capture.pt"
)
MODEL = ROOT.parent / "vnncomp2021/benchmarks/cifar10_resnet/onnx/resnet_2b.onnx"


@pytest.fixture(scope="module")
def prepared_case():
    capture = torch.load(CAPTURE, map_location="cpu", weights_only=True)
    snapshot = production_snapshot_from_payload_v4(capture["cores"][0]["pre_snapshot"])
    production = production_optimizer_step_trace_from_payload_v4(
        capture["optimizer_step_traces"][0]
    )
    program = import_onnx(str(MODEL), do_shape_infer=True, normalize=True)
    module = plan_interval_ibp_v0(program)
    tensor_map = snapshot.tensor_map()

    def one(role):
        values = [item.value for item in tensor_map.values() if item.role == role]
        assert len(values) == 1
        return values[0]

    lower = one(ProductionTensorRole.INPUT_LOWER)
    upper = one(ProductionTensorRole.INPUT_UPPER)
    objective = one(ProductionTensorRole.LINEAR_SPEC)
    mutable_paths = tuple(
        sorted(
            item.semantic_path
            for item in snapshot.tensors
            if item.ownership == ProductionTensorOwnership.MUTABLE_COPY_OUT
        )
    )
    template = prepare_core_template_v1(
        template_id="resnet2b-prop0-b3-a",
        program=program,
        module=module,
        topology=TOPOLOGY,
        device="cpu",
        dtype=torch.float32,
        input_shape=lower.shape,
        objective_shape=objective.shape,
        mutable_paths=mutable_paths,
    )
    mapping = initialize_rvir_v4_native_pre_state(snapshot, TOPOLOGY)
    spec = InputSpec.box(value_name=program.graph.inputs[0], lower=lower, upper=upper)
    return {
        "snapshot": snapshot,
        "production": production,
        "program": program,
        "module": module,
        "lower": lower,
        "upper": upper,
        "objective": objective,
        "mutable_paths": mutable_paths,
        "template": template,
        "mapping": mapping,
        "spec": spec,
    }


def test_prepared_template_cache_miss_then_exact_hit(prepared_case) -> None:
    cache = PreparedCoreTemplateCache()
    template = prepared_case["template"]
    template_hash = cache.insert(template)

    assert cache.compile_count == 1
    assert cache.hit_count == 0
    assert cache.resolve(template_hash, topology=TOPOLOGY) is template
    assert cache.compile_count == 1
    assert cache.hit_count == 1
    with pytest.raises(ValueError, match="already cached"):
        cache.insert(template)
    with pytest.raises(KeyError, match="cache miss"):
        cache.resolve("f" * 64, topology=TOPOLOGY)


def test_dynamic_instance_binds_once_and_optimizer_skips_scope_rebuild(
    prepared_case, monkeypatch
) -> None:
    from boundflow.runtime import rvir_v4_native_optimizer as optimizer

    instance = instantiate_core_plan_v1(
        template=prepared_case["template"],
        topology=TOPOLOGY,
        snapshot=prepared_case["snapshot"],
        mapping=prepared_case["mapping"],
        input_spec=prepared_case["spec"],
        linear_spec_C=prepared_case["objective"],
        mutation_policy=prepared_case["production"].mutation_policy,
    )

    def forbidden_scope_rebuild(*_args, **_kwargs):
        raise AssertionError("optimizer rebuilt the prevalidated scope")

    monkeypatch.setattr(
        optimizer, "build_native_alpha_beta_scope", forbidden_scope_rebuild
    )
    trace = execute_rvir_v4_native_optimizer_trace(
        prepared_case["module"],
        prepared_case["spec"],
        linear_spec_C=prepared_case["objective"],
        relu_pre=prepared_case["mapping"].relu_pre,
        initial_state=instance.initial_state,
        mutation_policy=prepared_case["production"].mutation_policy,
        prevalidated_plan=instance,
    )

    assert len(trace.steps) == 10
    assert sum(step.update_after for step in trace.steps) == 9
    assert trace.scope_hash == instance.scope.stable_hash()


def test_prepared_template_rejects_topology_device_dtype_and_inventory_drift(
    prepared_case,
) -> None:
    template = prepared_case["template"]
    bad_topology = (replace(TOPOLOGY[0], provider_activation="/wrong"),) + TOPOLOGY[1:]
    with pytest.raises(ValueError, match="prepared topology differs"):
        template.validate(topology=bad_topology)
    with pytest.raises(ValueError, match="binding placement differs"):
        replace(template, device="cuda:0").validate()
    with pytest.raises(ValueError, match="binding placement differs"):
        replace(template, dtype="torch.float64").validate()
    with pytest.raises(ValueError, match="mutable inventory differs"):
        instantiate_core_plan_v1(
            template=replace(template, mutable_paths=template.mutable_paths[:-1]),
            topology=TOPOLOGY,
            snapshot=prepared_case["snapshot"],
            mapping=prepared_case["mapping"],
            input_spec=prepared_case["spec"],
            linear_spec_C=prepared_case["objective"],
            mutation_policy=prepared_case["production"].mutation_policy,
        )


def test_prepared_template_rejects_stale_module_parameter(prepared_case) -> None:
    template = prepared_case["template"]
    params = template.module.bindings["params"]
    name = sorted(params)[0]
    original = params[name]
    params[name] = original + 1.0
    try:
        with pytest.raises(ValueError, match="module graph is stale"):
            template.validate()
    finally:
        params[name] = original
    template.validate()


def test_optimizer_rejects_prevalidated_plan_for_other_state(prepared_case) -> None:
    instance = instantiate_core_plan_v1(
        template=prepared_case["template"],
        topology=TOPOLOGY,
        snapshot=prepared_case["snapshot"],
        mapping=prepared_case["mapping"],
        input_spec=prepared_case["spec"],
        linear_spec_C=prepared_case["objective"],
        mutation_policy=prepared_case["production"].mutation_policy,
    )
    changed_alpha = tuple(instance.initial_state.alpha_by_relu_input)
    name, value = changed_alpha[0]
    bad_state = replace(
        instance.initial_state,
        alpha_by_relu_input=((name, value * 0.99),) + changed_alpha[1:],
    )
    with pytest.raises(ValueError, match="prevalidated state differs"):
        execute_rvir_v4_native_optimizer_trace(
            prepared_case["module"],
            prepared_case["spec"],
            linear_spec_C=prepared_case["objective"],
            relu_pre=prepared_case["mapping"].relu_pre,
            initial_state=bad_state,
            mutation_policy=prepared_case["production"].mutation_policy,
            prevalidated_plan=instance,
        )
