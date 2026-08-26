"""GC0-0 generic verification-graph schema and direct rejection tests."""

# pylint: disable=missing-function-docstring,too-many-locals,too-many-statements
# pylint: disable=too-many-arguments

from __future__ import annotations

import ast
from dataclasses import replace
import json
from pathlib import Path
from typing import Callable, cast

import pytest

from boundflow.ir.verification_graph import (
    GC01_ANALYSIS_REJECTION_REASONS,
    GC0_DIRECT_REJECTION_REASONS,
    LegalityResultV1,
    VerificationAxisRole,
    VerificationEffectAccess,
    VerificationEffectKind,
    VerificationEffectTokenV1,
    VerificationFallbackPolicy,
    VerificationFinitePolicy,
    VerificationGraphModuleV1,
    VerificationGraphValidationError,
    VerificationOpKind,
    VerificationOpV1,
    VerificationPolarity,
    VerificationProgramV1,
    VerificationRegionV1,
    VerificationRejectionReason,
    VerificationRepresentation,
    VerificationStorageClass,
    VerificationVJPContractV1,
    VerificationValueRole,
    VerificationValueV1,
    build_gc0_rule_registry_v1,
    freeze_verification_attributes,
)


def _strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = 1
    result = []
    for dimension in reversed(shape):
        result.append(stride)
        stride *= max(dimension, 1)
    return tuple(reversed(result))


def _tensor(
    value_id: str,
    *,
    role: VerificationValueRole,
    shape: tuple[int, ...],
    axes: tuple[VerificationAxisRole, ...],
    polarity: VerificationPolarity = VerificationPolarity.LOWER,
    representation: VerificationRepresentation = VerificationRepresentation.DENSE,
    storage: VerificationStorageClass = VerificationStorageClass.ARENA_SCRATCH,
    producer: str | None = None,
    consumers: tuple[str, ...] = (),
    requires_grad: bool = False,
    state_version: str | None = None,
    present: bool = True,
    dtype: str = "float32",
) -> VerificationValueV1:
    return VerificationValueV1(
        value_id=value_id,
        role=role,
        shape=shape,
        dtype=dtype,
        device_kind="cuda",
        layout=(
            "opaque-compressed"
            if representation
            in {
                VerificationRepresentation.COMPRESSED_INDEXED,
                VerificationRepresentation.SPARSE_LOCATION,
            }
            else "contiguous-strided"
        ),
        strides=_strides(shape),
        axis_roles=axes,
        polarity=polarity,
        representation=representation,
        requires_grad=requires_grad,
        state_version=state_version,
        lineage_id="lineage:fixture",
        storage_class=storage,
        alias_set=None,
        producer_op_id=producer,
        consumer_op_ids=consumers,
        external_use_count=0,
        present=present,
        finite_policy=(
            VerificationFinitePolicy.INTEGER_EXACT
            if dtype.startswith("int")
            else VerificationFinitePolicy.FINITE_REQUIRED
        ),
    )


def _token(
    value_id: str,
    *,
    role: VerificationValueRole,
    producer: str,
    consumers: tuple[str, ...] = (),
    state_version: str,
) -> VerificationValueV1:
    return VerificationValueV1(
        value_id=value_id,
        role=role,
        shape=(),
        dtype="token",
        device_kind="host",
        layout="token",
        strides=(),
        axis_roles=(),
        polarity=VerificationPolarity.NONE,
        representation=VerificationRepresentation.TOKEN,
        requires_grad=False,
        state_version=state_version,
        lineage_id="lineage:fixture",
        storage_class=VerificationStorageClass.HOST_STATUS,
        alias_set=None,
        producer_op_id=producer,
        consumer_op_ids=consumers,
        external_use_count=0,
        present=True,
        finite_policy=VerificationFinitePolicy.TOKEN,
    )


def _schema_fixture(
    *, beta_width: int, affine_kind: VerificationOpKind, affine_count: int
) -> VerificationGraphModuleV1:  # pylint: disable=too-many-locals
    assert affine_kind in {
        VerificationOpKind.LINEAR_RIGHT,
        VerificationOpKind.CONV2D_RIGHT,
    }
    domain, spec, feature = 6, 1, 8
    registry = build_gc0_rule_registry_v1()
    alpha_attrs = freeze_verification_attributes(
        {
            "direction_index": 0,
            "feature_index_value_id": "value.alpha.index",
            "spec_index": 0,
            "start_node_key": "start:fixture",
            "state_version": "alpha:v0",
        }
    )
    beta_attributes: dict[str, object] = {"active": beta_width > 0}
    if beta_width:
        beta_attributes.update(
            {
                "history_value_id": "value.beta.history",
                "location_value_id": "value.beta.location",
                "sign_value_id": "value.beta.sign",
            }
        )
    beta_attrs = freeze_verification_attributes(beta_attributes)
    op_ids = ["op.alpha", "op.beta", "op.relu"]
    values = [
        _tensor(
            "value.coefficient.input",
            role=VerificationValueRole.COEFFICIENT,
            shape=(domain, spec, feature),
            axes=(
                VerificationAxisRole.DOMAIN,
                VerificationAxisRole.SPEC,
                VerificationAxisRole.FEATURE,
            ),
            storage=VerificationStorageClass.EXTERNAL_BORROWED,
            consumers=("op.relu",),
        ),
        _tensor(
            "value.alpha.state",
            role=VerificationValueRole.ALPHA,
            shape=(2, spec, domain, feature),
            axes=(
                VerificationAxisRole.DIRECTION,
                VerificationAxisRole.SPEC,
                VerificationAxisRole.DOMAIN,
                VerificationAxisRole.FEATURE,
            ),
            representation=VerificationRepresentation.COMPRESSED_INDEXED,
            storage=VerificationStorageClass.EXTERNAL_BORROWED,
            consumers=("op.alpha",),
            requires_grad=True,
            state_version="alpha:v0",
        ),
        _tensor(
            "value.alpha.index",
            role=VerificationValueRole.INDEX,
            shape=(feature,),
            axes=(VerificationAxisRole.FEATURE,),
            storage=VerificationStorageClass.EXTERNAL_BORROWED,
            consumers=("op.alpha", "op.vjp"),
            dtype="int64",
        ),
        _tensor(
            "value.alpha.expanded",
            role=VerificationValueRole.ALPHA,
            shape=(domain, spec, feature),
            axes=(
                VerificationAxisRole.DOMAIN,
                VerificationAxisRole.SPEC,
                VerificationAxisRole.FEATURE,
            ),
            producer="op.alpha",
            consumers=("op.relu",),
            state_version="alpha:v0",
        ),
        _tensor(
            "value.beta.state",
            role=VerificationValueRole.BETA,
            shape=(domain, beta_width),
            axes=(VerificationAxisRole.DOMAIN, VerificationAxisRole.BETA_SLOT),
            representation=VerificationRepresentation.SPARSE_LOCATION,
            storage=VerificationStorageClass.EXTERNAL_BORROWED,
            consumers=("op.beta",),
            requires_grad=beta_width > 0,
            state_version="beta:v0",
            present=beta_width > 0,
        ),
        _tensor(
            "value.beta.contribution",
            role=VerificationValueRole.COEFFICIENT,
            shape=(domain, spec, feature),
            axes=(
                VerificationAxisRole.DOMAIN,
                VerificationAxisRole.SPEC,
                VerificationAxisRole.FEATURE,
            ),
            producer="op.beta",
            consumers=("op.relu",),
        ),
        _tensor(
            "value.relu.output",
            role=VerificationValueRole.COEFFICIENT,
            shape=(domain, spec, feature),
            axes=(
                VerificationAxisRole.DOMAIN,
                VerificationAxisRole.SPEC,
                VerificationAxisRole.FEATURE,
            ),
            producer="op.relu",
            consumers=("op.affine.0",),
        ),
        _tensor(
            "value.weight",
            role=VerificationValueRole.PARAMETER,
            shape=(feature, feature),
            axes=(
                VerificationAxisRole.OUTPUT_CHANNEL,
                VerificationAxisRole.INPUT_CHANNEL,
            ),
            polarity=VerificationPolarity.NONE,
            storage=VerificationStorageClass.PARAMETER_RESIDENT,
            consumers=tuple(f"op.affine.{index}" for index in range(affine_count)),
        ),
        _tensor(
            "value.bias.parameter",
            role=VerificationValueRole.PARAMETER,
            shape=(feature,),
            axes=(VerificationAxisRole.OUTPUT_CHANNEL,),
            polarity=VerificationPolarity.NONE,
            storage=VerificationStorageClass.PARAMETER_RESIDENT,
            consumers=tuple(f"op.affine.{index}" for index in range(affine_count)),
        ),
        _tensor(
            "value.adjoint",
            role=VerificationValueRole.INCOMING_ADJOINT,
            shape=(domain, spec),
            axes=(VerificationAxisRole.DOMAIN, VerificationAxisRole.SPEC),
            storage=VerificationStorageClass.EXTERNAL_BORROWED,
            consumers=("op.vjp",),
        ),
        _tensor(
            "value.alpha.gradient",
            role=VerificationValueRole.GRADIENT,
            shape=(2, spec, domain, feature),
            axes=(
                VerificationAxisRole.DIRECTION,
                VerificationAxisRole.SPEC,
                VerificationAxisRole.DOMAIN,
                VerificationAxisRole.FEATURE,
            ),
            representation=VerificationRepresentation.COMPRESSED_INDEXED,
            storage=VerificationStorageClass.ARENA_PERSISTENT,
            producer="op.vjp",
            requires_grad=False,
        ),
    ]
    beta_external = ["value.beta.state"]
    beta_owners: tuple[str, ...] = ()
    if beta_width:
        for suffix, role in (
            ("location", VerificationValueRole.INDEX),
            ("sign", VerificationValueRole.SPLIT),
            ("history", VerificationValueRole.HISTORY),
        ):
            value_id = f"value.beta.{suffix}"
            values.append(
                _tensor(
                    value_id,
                    role=role,
                    shape=(domain, beta_width),
                    axes=(VerificationAxisRole.DOMAIN, VerificationAxisRole.BETA_SLOT),
                    polarity=VerificationPolarity.NONE,
                    representation=VerificationRepresentation.SPARSE_LOCATION,
                    storage=VerificationStorageClass.EXTERNAL_BORROWED,
                    consumers=("op.beta",),
                    state_version=(
                        "split:v0"
                        if role
                        in {VerificationValueRole.SPLIT, VerificationValueRole.HISTORY}
                        else None
                    ),
                    dtype="int64",
                )
            )
            beta_external.append(value_id)
        values.append(
            _tensor(
                "value.beta.gradient",
                role=VerificationValueRole.GRADIENT,
                shape=(domain, beta_width),
                axes=(VerificationAxisRole.DOMAIN, VerificationAxisRole.BETA_SLOT),
                representation=VerificationRepresentation.SPARSE_LOCATION,
                storage=VerificationStorageClass.ARENA_PERSISTENT,
                producer="op.vjp",
            )
        )
        beta_owners = ("value.beta.gradient",)
    ops = [
        VerificationOpV1(
            op_id="op.alpha",
            op_kind=VerificationOpKind.COMPRESSED_ALPHA_GATHER,
            semantic_version="1",
            input_value_ids=("value.alpha.state", "value.alpha.index"),
            output_value_ids=("value.alpha.expanded",),
            parameter_value_ids=(),
            effect_read_ids=("effect.alpha.read",),
            effect_write_ids=(),
            attributes=alpha_attrs,
            bound_direction=VerificationPolarity.LOWER,
            numeric_policy_id="numeric:float32-strict",
            vjp_contract_id=None,
            source_op_ids=("source:relu",),
        ),
        VerificationOpV1(
            op_id="op.beta",
            op_kind=VerificationOpKind.SPARSE_BETA_INJECT,
            semantic_version="1",
            input_value_ids=tuple(beta_external),
            output_value_ids=("value.beta.contribution",),
            parameter_value_ids=(),
            effect_read_ids=("effect.beta.read",),
            effect_write_ids=(),
            attributes=beta_attrs,
            bound_direction=VerificationPolarity.LOWER,
            numeric_policy_id="numeric:float32-strict",
            vjp_contract_id=None,
            source_op_ids=("source:split",),
        ),
        VerificationOpV1(
            op_id="op.relu",
            op_kind=VerificationOpKind.RELU_RELAXATION,
            semantic_version="1",
            input_value_ids=(
                "value.coefficient.input",
                "value.alpha.expanded",
                "value.beta.contribution",
            ),
            output_value_ids=("value.relu.output",),
            parameter_value_ids=(),
            effect_read_ids=(),
            effect_write_ids=(),
            attributes=freeze_verification_attributes(
                {"endpoint_policy": "zero-selects-alpha-v1"}
            ),
            bound_direction=VerificationPolarity.LOWER,
            numeric_policy_id="numeric:float32-strict",
            vjp_contract_id=None,
            source_op_ids=("source:relu",),
        ),
    ]
    previous = "value.relu.output"
    for index in range(affine_count):
        op_id = f"op.affine.{index}"
        output = f"value.affine.{index}.output"
        next_consumer = (
            f"op.affine.{index + 1}" if index + 1 < affine_count else "op.vjp"
        )
        values.append(
            _tensor(
                output,
                role=VerificationValueRole.COEFFICIENT,
                shape=(domain, spec, feature),
                axes=(
                    VerificationAxisRole.DOMAIN,
                    VerificationAxisRole.SPEC,
                    VerificationAxisRole.FEATURE,
                ),
                producer=op_id,
                consumers=(next_consumer,),
            )
        )
        ops.append(
            VerificationOpV1(
                op_id=op_id,
                op_kind=affine_kind,
                semantic_version="1",
                input_value_ids=(previous,),
                output_value_ids=(output,),
                parameter_value_ids=("value.weight", "value.bias.parameter"),
                effect_read_ids=(),
                effect_write_ids=(),
                attributes=freeze_verification_attributes({"operator_ordinal": index}),
                bound_direction=VerificationPolarity.LOWER,
                numeric_policy_id="numeric:float32-strict",
                vjp_contract_id=None,
                source_op_ids=(f"source:affine:{index}",),
            )
        )
        previous = output
    vjp_outputs = ("value.alpha.gradient", *beta_owners)
    ops.extend(
        (
            VerificationOpV1(
                op_id="op.vjp",
                op_kind=VerificationOpKind.MINIMAL_STATE_VJP,
                semantic_version="1",
                input_value_ids=(previous, "value.adjoint", "value.alpha.index"),
                output_value_ids=vjp_outputs,
                parameter_value_ids=(),
                effect_read_ids=(),
                effect_write_ids=(),
                attributes=freeze_verification_attributes({"first_order_only": True}),
                bound_direction=VerificationPolarity.LOWER,
                numeric_policy_id="numeric:float32-strict",
                vjp_contract_id="vjp:fixture",
                source_op_ids=(),
            ),
            VerificationOpV1(
                op_id="op.status",
                op_kind=VerificationOpKind.COMPACT_STATUS,
                semantic_version="1",
                input_value_ids=(previous,),
                output_value_ids=("value.status",),
                parameter_value_ids=(),
                effect_read_ids=("effect.queue.read",),
                effect_write_ids=(),
                attributes=freeze_verification_attributes({"compact": True}),
                bound_direction=VerificationPolarity.NONE,
                numeric_policy_id="numeric:float32-strict",
                vjp_contract_id=None,
                source_op_ids=(),
            ),
            VerificationOpV1(
                op_id="op.commit",
                op_kind=VerificationOpKind.COARSE_COMMIT,
                semantic_version="1",
                input_value_ids=("value.status",),
                output_value_ids=("value.commit",),
                parameter_value_ids=(),
                effect_read_ids=(),
                effect_write_ids=("effect.commit.write",),
                attributes=freeze_verification_attributes(
                    {"evaluation_count": 10, "mutation_count": 9}
                ),
                bound_direction=VerificationPolarity.NONE,
                numeric_policy_id="numeric:float32-strict",
                vjp_contract_id=None,
                source_op_ids=(),
            ),
        )
    )
    op_ids.extend(f"op.affine.{index}" for index in range(affine_count))
    op_ids.extend(("op.vjp", "op.status", "op.commit"))
    values.extend(
        (
            VerificationValueV1(
                value_id="value.status",
                role=VerificationValueRole.STATUS,
                shape=(),
                dtype="int64",
                device_kind="host",
                layout="scalar",
                strides=(),
                axis_roles=(),
                polarity=VerificationPolarity.NONE,
                representation=VerificationRepresentation.SCALAR,
                requires_grad=False,
                state_version=None,
                lineage_id="lineage:fixture",
                storage_class=VerificationStorageClass.HOST_STATUS,
                alias_set=None,
                producer_op_id="op.status",
                consumer_op_ids=("op.commit",),
                external_use_count=0,
                present=True,
                finite_policy=VerificationFinitePolicy.INTEGER_EXACT,
            ),
            _token(
                "value.commit",
                role=VerificationValueRole.COMMIT_TOKEN,
                producer="op.commit",
                state_version="commit:v1",
            ),
        )
    )
    effects = (
        VerificationEffectTokenV1(
            "effect.alpha.read",
            VerificationEffectKind.ALPHA_STATE,
            "resource.alpha",
            "alpha:v0",
            "alpha:v0",
            VerificationEffectAccess.READ,
            0,
        ),
        VerificationEffectTokenV1(
            "effect.beta.read",
            VerificationEffectKind.BETA_STATE,
            "resource.beta",
            "beta:v0",
            "beta:v0",
            VerificationEffectAccess.READ,
            1,
        ),
        VerificationEffectTokenV1(
            "effect.queue.read",
            VerificationEffectKind.QUEUE_STATE,
            "resource.queue",
            "queue:v0",
            "queue:v0",
            VerificationEffectAccess.EXTERNAL_BOUNDARY,
            2,
        ),
        VerificationEffectTokenV1(
            "effect.commit.write",
            VerificationEffectKind.COMMIT_STATE,
            "resource.commit",
            "commit:v0",
            "commit:v1",
            VerificationEffectAccess.WRITE,
            3,
        ),
    )
    vjp = VerificationVJPContractV1(
        contract_id="vjp:fixture",
        primal_input_value_ids=("value.relu.output",),
        primal_output_value_ids=(previous,),
        incoming_adjoint_value_ids=("value.adjoint",),
        alpha_gradient_owner_value_ids=("value.alpha.gradient",),
        beta_gradient_owner_value_ids=beta_owners,
        compressed_output_layouts=("compressed-indexed",)
        + (("sparse-location",) if beta_width else ()),
        saved_value_ids=("value.alpha.index",),
        recomputed_value_ids=("value.relu.output",),
        endpoint_policy="zero-selects-alpha-v1",
    )
    external_values = (
        "value.coefficient.input",
        "value.alpha.state",
        "value.alpha.index",
        *beta_external,
        "value.weight",
        "value.bias.parameter",
        "value.adjoint",
    )
    region = VerificationRegionV1(
        region_id="region:fixture",
        op_ids=tuple(op_ids),
        input_value_ids=external_values,
        output_value_ids=(previous, *vjp_outputs, "value.status", "value.commit"),
        parameter_value_ids=("value.weight", "value.bias.parameter"),
        external_use_ids=(),
        effect_input_ids=(
            "effect.alpha.read",
            "effect.beta.read",
            "effect.queue.read",
        ),
        effect_output_ids=("effect.commit.write",),
        saved_state_ids=("value.alpha.index",),
        gradient_owner_ids=vjp_outputs,
        entry_op_ids=("op.alpha", "op.beta", "op.relu"),
        exit_op_ids=("op.commit",),
        postdominator_witness="witness:commit-postdominates",
        closed_world=True,
        fallback_policy=VerificationFallbackPolicy.REJECT_BEFORE_LAUNCH,
    )
    program = VerificationProgramV1(
        program_id="program:fixture",
        source_graph_hash="1" * 64,
        parameter_schema_hash="2" * 64,
        numeric_policy_id="numeric:float32-strict",
        target_contract_id="target:cuda-generic",
        region_ids=(region.region_id,),
        entry_region_ids=(region.region_id,),
        external_value_ids=external_values,
        external_effect_ids=region.effect_input_ids,
        rule_registry_hash=registry.stable_hash(),
    )
    module = VerificationGraphModuleV1(
        module_id="module:fixture",
        program=program,
        regions=(region,),
        values=tuple(values),
        ops=tuple(ops),
        effects=effects,
        vjp_contracts=(vjp,),
        rule_registry=registry,
    )
    module.validate()
    return module


@pytest.mark.parametrize(
    ("beta_width", "affine_kind", "affine_count"),
    (
        (0, VerificationOpKind.CONV2D_RIGHT, 1),
        (1, VerificationOpKind.LINEAR_RIGHT, 1),
        (0, VerificationOpKind.CONV2D_RIGHT, 3),
    ),
)
def test_three_signature_schemas_round_trip_without_execution(
    beta_width: int, affine_kind: VerificationOpKind, affine_count: int
) -> None:
    module = _schema_fixture(
        beta_width=beta_width,
        affine_kind=affine_kind,
        affine_count=affine_count,
    )
    encoded = module.canonical_json()
    restored = VerificationGraphModuleV1.from_canonical_json(encoded)
    assert restored == module
    assert restored.stable_hash() == module.stable_hash()
    assert restored.timing_recorded is False
    assert restored.performance_claimed is False
    assert restored.rule_registry.execution_enabled is False
    leaf_objects = (
        *restored.regions,
        *restored.values,
        *restored.ops,
        *restored.effects,
        *restored.vjp_contracts,
        *restored.rule_registry.rules,
    )
    assert all(len(item.stable_hash()) == 64 for item in leaf_objects)
    assert all(
        json.loads(item.canonical_json()) == item.to_dict() for item in leaf_objects
    )
    assert json.loads(restored.program.canonical_json()) == restored.program.to_dict()
    assert (
        json.loads(restored.rule_registry.canonical_json())
        == restored.rule_registry.to_dict()
    )
    commit = next(
        op for op in restored.ops if op.op_kind == VerificationOpKind.COARSE_COMMIT
    )
    assert commit.attribute_map["evaluation_count"] == 10
    assert commit.attribute_map["mutation_count"] == 9
    beta = next(
        value for value in restored.values if value.role == VerificationValueRole.BETA
    )
    assert beta.present == (beta_width > 0)
    assert beta.shape[-1] == beta_width


def test_schema_source_has_no_frozen_model_or_site_identity() -> None:
    source = Path("boundflow/ir/verification_graph.py").read_text(encoding="utf-8")
    for forbidden in (
        "ResNet2B",
        '"/49"',
        "25/Conv_8",
        "31/Gemm_14",
        '"C0"',
        '"C1"',
        '"C2"',
    ):
        assert forbidden not in source


def test_schema_source_has_no_backend_runtime_or_timing_dependency() -> None:
    source = Path("boundflow/ir/verification_graph.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imports.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert not any(
        name == "torch"
        or name.startswith("torch.")
        or name == "tvm"
        or name.startswith("tvm.")
        or "runtime" in name
        for name in imports
    )
    for forbidden in ("perf_counter", "cuda.Event", "torch.compile"):
        assert forbidden not in source


def test_rejection_vocabulary_is_complete_and_stage_partitioned() -> None:
    assert len(tuple(VerificationRejectionReason)) == 22
    assert set(GC0_DIRECT_REJECTION_REASONS).isdisjoint(GC01_ANALYSIS_REJECTION_REASONS)
    assert set(GC0_DIRECT_REJECTION_REASONS) | set(
        GC01_ANALYSIS_REJECTION_REASONS
    ) == set(VerificationRejectionReason)


def _assert_reason(
    call: Callable[[], object], reason: VerificationRejectionReason
) -> None:
    with pytest.raises(VerificationGraphValidationError) as error:
        call()
    assert error.value.reason == reason


def test_gc0_direct_value_rejections_are_fail_closed() -> None:
    module = _schema_fixture(
        beta_width=0, affine_kind=VerificationOpKind.CONV2D_RIGHT, affine_count=1
    )
    coefficient = next(
        value for value in module.values if value.value_id == "value.coefficient.input"
    )
    alpha = next(
        value for value in module.values if value.value_id == "value.alpha.state"
    )
    beta = next(
        value for value in module.values if value.value_id == "value.beta.state"
    )
    _assert_reason(
        lambda: replace(coefficient, shape=(None, 1, 8)).validate(),
        VerificationRejectionReason.DYNAMIC_SHAPE_UNBOUND,
    )
    _assert_reason(
        lambda: replace(coefficient, dtype="complex64").validate(),
        VerificationRejectionReason.DTYPE_OR_DEVICE_MISMATCH,
    )
    _assert_reason(
        lambda: replace(coefficient, layout="mystery").validate(),
        VerificationRejectionReason.LAYOUT_NOT_NORMALIZABLE,
    )
    _assert_reason(
        lambda: replace(alpha, state_version=None).validate(),
        VerificationRejectionReason.STATE_VERSION_MISMATCH,
    )
    _assert_reason(
        lambda: replace(beta, present=True).validate(),
        VerificationRejectionReason.BETA_ACTIVE_EMPTY_MISMATCH,
    )


def test_gc0_direct_op_rejections_are_fail_closed() -> None:
    module = _schema_fixture(
        beta_width=1, affine_kind=VerificationOpKind.LINEAR_RIGHT, affine_count=1
    )
    alpha = next(
        op
        for op in module.ops
        if op.op_kind == VerificationOpKind.COMPRESSED_ALPHA_GATHER
    )
    beta = next(
        op for op in module.ops if op.op_kind == VerificationOpKind.SPARSE_BETA_INJECT
    )
    relu = next(
        op for op in module.ops if op.op_kind == VerificationOpKind.RELU_RELAXATION
    )
    _assert_reason(
        lambda: replace(alpha, op_kind=cast(VerificationOpKind, "unknown")).validate(),
        VerificationRejectionReason.UNSUPPORTED_OP_KIND,
    )
    alpha_no_start = dict(alpha.attribute_map)
    alpha_no_start.pop("start_node_key")
    _assert_reason(
        lambda: replace(
            alpha, attributes=freeze_verification_attributes(alpha_no_start)
        ).validate(),
        VerificationRejectionReason.ALPHA_START_NODE_MISMATCH,
    )
    alpha_bad_index = dict(alpha.attribute_map)
    alpha_bad_index["direction_index"] = -1
    _assert_reason(
        lambda: replace(
            alpha, attributes=freeze_verification_attributes(alpha_bad_index)
        ).validate(),
        VerificationRejectionReason.ALPHA_INDEX_OR_DIRECTION_MISMATCH,
    )
    beta_bad_active = dict(beta.attribute_map)
    beta_bad_active["active"] = "yes"
    _assert_reason(
        lambda: replace(
            beta, attributes=freeze_verification_attributes(beta_bad_active)
        ).validate(),
        VerificationRejectionReason.BETA_ACTIVE_EMPTY_MISMATCH,
    )
    beta_missing_history = dict(beta.attribute_map)
    beta_missing_history.pop("history_value_id")
    _assert_reason(
        lambda: replace(
            beta, attributes=freeze_verification_attributes(beta_missing_history)
        ).validate(),
        VerificationRejectionReason.BETA_LOCATION_SIGN_HISTORY_MISMATCH,
    )
    _assert_reason(
        lambda: replace(alpha, bound_direction=VerificationPolarity.NONE).validate(),
        VerificationRejectionReason.BOUND_POLARITY_MISMATCH,
    )
    _assert_reason(
        lambda: replace(relu, attributes=()).validate(),
        VerificationRejectionReason.ENDPOINT_POLICY_MISMATCH,
    )


def test_gc0_direct_vjp_region_and_identity_rejections_are_fail_closed() -> None:
    module = _schema_fixture(
        beta_width=0, affine_kind=VerificationOpKind.CONV2D_RIGHT, affine_count=1
    )
    vjp = module.vjp_contracts[0]
    region = module.regions[0]
    _assert_reason(
        lambda: replace(
            vjp,
            alpha_gradient_owner_value_ids=(),
            compressed_output_layouts=(),
        ).validate(),
        VerificationRejectionReason.VJP_OWNER_OR_SAVED_STATE_MISMATCH,
    )
    _assert_reason(
        lambda: replace(vjp, higher_order_policy="allow").validate(),
        VerificationRejectionReason.HIGHER_ORDER_GRAD_UNSUPPORTED,
    )
    _assert_reason(
        lambda: replace(vjp, dense_a_escape_policy="allow").validate(),
        VerificationRejectionReason.DENSE_A_ESCAPE,
    )
    _assert_reason(
        lambda: replace(
            region, fallback_policy=cast(VerificationFallbackPolicy, "runtime")
        ).validate(),
        VerificationRejectionReason.RUNTIME_FALLBACK_REQUIRED,
    )
    _assert_reason(
        lambda: replace(
            module,
            program=replace(module.program, rule_registry_hash="0" * 64),
        ).validate(),
        VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
    )


def test_legality_result_schema_requires_stable_fail_closed_evidence() -> None:
    rejected = LegalityResultV1(
        admitted=False,
        region_id="region:fixture",
        ordered_op_ids=(),
        boundary_input_ids=(),
        boundary_output_ids=(),
        external_use_witnesses=(),
        effect_order_witnesses=(),
        alias_witnesses=(),
        dense_escape_witnesses=(),
        vjp_witnesses=(),
        rejection_reasons=(VerificationRejectionReason.UNSUPPORTED_OP_KIND,),
    )
    rejected.validate()
    assert len(rejected.stable_hash()) == 64
    _assert_reason(
        lambda: replace(rejected, rejection_reasons=()).validate(),
        VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
    )
    _assert_reason(
        lambda: replace(rejected, admitted=True, rejection_reasons=()).validate(),
        VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
    )


def test_canonical_json_rejects_format_and_fully_resigned_identity_tamper() -> None:
    module = _schema_fixture(
        beta_width=0, affine_kind=VerificationOpKind.CONV2D_RIGHT, affine_count=3
    )
    encoded = module.canonical_json()
    _assert_reason(
        lambda: VerificationGraphModuleV1.from_canonical_json(
            encoded.replace(",", ", ", 1)
        ),
        VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
    )
    tampered = replace(module, performance_claimed=True)
    _assert_reason(
        tampered.validate,
        VerificationRejectionReason.RUNTIME_FALLBACK_REQUIRED,
    )
