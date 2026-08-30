"""S4-1A ordered compressed mutable-buffer ownership gates."""

# pylint: disable=missing-function-docstring,missing-class-docstring
# pylint: disable=protected-access,too-many-locals,redefined-outer-name
# pylint: disable=unnecessary-lambda-assignment
# pylint: disable=unidiomatic-typecheck

from __future__ import annotations

import copy
from dataclasses import dataclass, fields, replace
import gc
import json
from pathlib import Path
import pickle
import weakref

import pytest
import torch

from boundflow.frontends.onnx.frontend import import_onnx
from boundflow.ir.verification_graph import VerificationRejectionReason
from boundflow.planner import plan_interval_ibp_v0
from boundflow.runtime import asplos27_s4_ordered_buffer_abi as buffer_module
from boundflow.runtime.asplos27_s4_mutable_state_admission import (
    prepare_s4_mutable_state_admission_v1,
)
from boundflow.runtime.asplos27_s4_ordered_buffer_abi import (
    PreparedS4MutableBuffersV1,
    S4MutableBufferPreparationError,
    S4_MUTABLE_BUFFER_CONSTRUCTION_HASH_V1,
    prepare_s4_mutable_buffers_v1,
)
from boundflow.runtime.r3_structured_owner_custom_backward import (
    R31FullRegionPlanV1,
    compile_r31_full_region_plan_v1,
)
from boundflow.runtime.rvir_v4_pre_state_initializer import (
    initialize_rvir_v4_native_pre_state,
)
from boundflow.runtime.rvir_v4_production_state import (
    ProductionStateSnapshotV4,
    production_snapshot_from_payload_v4,
    production_tensor_sha256,
)
from scripts.run_rvir_v4_pre_state_artifact import TOPOLOGY

ROOT = Path(__file__).resolve().parents[1]
CAPTURE = (
    ROOT / "artifacts/rvir-v4-pre-state/resnet2b-core-pre-state-v1/source_capture.pt"
)
MODEL = Path(
    "/home/lee/Codes/vnncomp2021/benchmarks/cifar10_resnet/onnx/resnet_2b.onnx"
)
CALL_ID = "asplos27-s4-1a-call-0001"
EXPECTED_CONSTRUCTION_HASH = (
    "8ad25c2abf1eb98c3b1097bf7acb46aba227f7e94f0c7c03169f39e8da409a9d"
)


@dataclass(frozen=True)
class _FormalFixture:
    snapshot: ProductionStateSnapshotV4
    plan: R31FullRegionPlanV1


@pytest.fixture(scope="module")
def formal_fixture() -> _FormalFixture:
    if not MODEL.is_file() or not CAPTURE.is_file():
        pytest.skip("S4-1A frozen ResNet2B fixture is unavailable")
    raw = torch.load(CAPTURE, map_location="cpu", weights_only=True)
    snapshot = production_snapshot_from_payload_v4(raw["cores"][0]["pre_snapshot"])
    mapping = initialize_rvir_v4_native_pre_state(snapshot, TOPOLOGY)
    program = import_onnx(str(MODEL), do_shape_infer=True, normalize=True)
    module = plan_interval_ibp_v0(program)
    plan = compile_r31_full_region_plan_v1(module, snapshot, mapping, TOPOLOGY)
    return _FormalFixture(snapshot=snapshot, plan=plan)


def _live_sources(fixture: _FormalFixture) -> dict[str, torch.Tensor]:
    if not torch.cuda.is_available():
        pytest.skip("S4-1A CUDA fixture is unavailable")
    tensor_map = fixture.snapshot.tensor_map()
    return {
        path: tensor_map[path].value.to("cuda:0").detach().clone().requires_grad_(True)
        for layout in fixture.plan.relu_layouts
        for path in (layout.alpha_path, layout.beta_path)
    }


def _admit(
    fixture: _FormalFixture,
    live: dict[str, torch.Tensor],
    *,
    call_id: str = CALL_ID,
):
    return prepare_s4_mutable_state_admission_v1(
        fixture.snapshot,
        TOPOLOGY,
        fixture.plan,
        live,
        exact_call_id=call_id,
    )


def _prepare(fixture: _FormalFixture):
    live = _live_sources(fixture)
    admission = _admit(fixture, live)
    prepared = prepare_s4_mutable_buffers_v1(admission, live, exact_call_id=CALL_ID)
    return live, admission, prepared


def _assert_error(detail: str, reason: VerificationRejectionReason, operation) -> None:
    with pytest.raises(S4MutableBufferPreparationError) as caught:
        operation()
    assert caught.value.detail_code == detail
    assert caught.value.verification_reason == reason
    assert caught.value.__context__ is None


def test_construction_model_hash_is_recomputed_exactly() -> None:
    assert S4_MUTABLE_BUFFER_CONSTRUCTION_HASH_V1 == EXPECTED_CONSTRUCTION_HASH


def test_positive_order_counts_content_views_and_claim_boundary(formal_fixture) -> None:
    live, _, prepared = _prepare(formal_fixture)
    assert type(prepared) is PreparedS4MutableBuffersV1
    receipt = prepared.receipt
    receipt.validate()
    assert receipt.parameter_count == receipt.gradient_count == 7
    assert (
        receipt.parameter_elements,
        receipt.gradient_elements,
        receipt.parameter_bytes,
        receipt.gradient_bytes,
    ) == (4254, 4254, 17016, 17016)
    assert (
        receipt.candidate_storage_count,
        receipt.candidate_logical_bytes,
        receipt.base_dlpack_view_count,
        receipt.empty_beta_token_count,
    ) == (16, 34080, 16, 5)
    assert (
        receipt.s4_1a_d2h_copy_count,
        receipt.s4_1a_d2h_bytes,
        receipt.cumulative_d2h_copy_count,
        receipt.cumulative_d2h_bytes,
        receipt.parameter_d2d_copy_count,
        receipt.parameter_d2d_bytes,
    ) == (32, 85056, 56, 153072, 7, 17016)
    assert not any(
        (
            receipt.provider_mapping_stability_validated,
            receipt.process_global_exclusivity_validated,
            receipt.crown_numeric_semantics_validated,
            receipt.optimizer_trajectory_validated,
            receipt.timing_recorded,
            receipt.performance_claimed,
        )
    )
    resources = prepared._resources
    assert resources is not None
    assert len(resources._parameters) == len(resources._gradients) == 7
    assert len(resources._views) == len(resources._private_view_keys) == 16
    assert len(set(resources._private_view_keys)) == 16
    buffers = resources.buffers()
    assert len({_storage_identity(item) for item in buffers}) == 16
    assert all(type(item) is torch.Tensor for item in buffers)
    assert all(
        item.dtype == torch.float32 and item.device.type == "cuda" for item in buffers
    )
    assert all(item.is_contiguous() and item.storage_offset() == 0 for item in buffers)
    assert all(item.is_leaf and item.requires_grad for item in resources._parameters)
    assert all(not item.requires_grad and item.is_leaf for item in resources._gradients)
    assert resources._lower is not None and resources._upstream is not None
    assert not resources._lower.requires_grad
    assert not resources._upstream.requires_grad
    assert torch.equal(resources._upstream, torch.full_like(resources._upstream, -1.0))
    for ordinal, (parameter, layout) in enumerate(
        zip(resources._parameters[:6], formal_fixture.plan.relu_layouts)
    ):
        assert production_tensor_sha256(parameter) == production_tensor_sha256(
            live[layout.alpha_path][0, 0]
        ), ordinal
    active = [
        item
        for item in formal_fixture.plan.relu_layouts
        if live[item.beta_path].numel()
    ]
    assert len(active) == 1
    assert production_tensor_sha256(
        resources._parameters[6]
    ) == production_tensor_sha256(live[active[0].beta_path])
    assert (
        json.loads(json.dumps(receipt.to_dict()))["receipt_hash"]
        == receipt.receipt_hash
    )
    prepared.close()
    prepared.close()
    assert prepared._state == "CLOSED" and prepared._resources is None


def _storage_identity(value: torch.Tensor) -> tuple[int, int, int]:
    storage = value.untyped_storage()
    return int(storage._cdata), int(storage.data_ptr()), int(storage.nbytes())


def _unsafe_receipt_payload(receipt) -> dict[str, object]:
    """Serialize a tampered frozen receipt without invoking its validators."""

    payload: dict[str, object] = {}
    for item in fields(receipt):
        if item.name == "receipt_hash":
            continue
        value = object.__getattribute__(receipt, item.name)
        if item.name in {"buffer_descriptors", "empty_beta_tokens"}:
            payload[item.name] = [
                {
                    nested.name: buffer_module._canonical_field(
                        object.__getattribute__(entry, nested.name)
                    )
                    for nested in fields(entry)
                }
                for entry in value
            ]
        else:
            payload[item.name] = buffer_module._canonical_field(value)
    return payload


def test_empty_beta_tokens_have_no_physical_resource(formal_fixture) -> None:
    _, _, prepared = _prepare(formal_fixture)
    tokens = prepared.receipt.empty_beta_tokens
    assert tuple(item.slot_ordinal for item in tokens) == tuple(range(5))
    assert all(item.shape == (6, 0) for item in tokens)
    assert all(not item.physical_buffer_present for item in tokens)
    assert all(not item.physical_view_present for item in tokens)
    assert all(item.optimizer_ordinal == -1 for item in tokens)
    prepared.close()


@pytest.mark.parametrize("target", ["prepared", "resource_owner", "ticket"])
@pytest.mark.parametrize("mode", ["copy", "deepcopy", "pickle"])
def test_ticket_resource_and_prepared_owners_are_nonserializable(
    formal_fixture, target, mode
) -> None:
    _, _, prepared = _prepare(formal_fixture)
    resources = prepared._resources
    assert resources is not None
    value = {
        "prepared": prepared,
        "resource_owner": resources,
        "ticket": resources._ticket,
    }[target]
    operation = {
        "copy": lambda: copy.copy(value),
        "deepcopy": lambda: copy.deepcopy(value),
        "pickle": lambda: pickle.dumps(value),
    }[mode]
    _assert_error(
        "BUFFER_PREPARE_SERIALIZATION_FORBIDDEN",
        VerificationRejectionReason.UNSAFE_ALIAS_OR_LIFETIME,
        operation,
    )
    prepared.close()


def test_prepare_is_one_shot_even_after_close(formal_fixture) -> None:
    live, admission, prepared = _prepare(formal_fixture)
    prepared.close()
    _assert_error(
        "BUFFER_PREPARE_ALREADY_ATTEMPTED",
        VerificationRejectionReason.UNSAFE_ALIAS_OR_LIFETIME,
        lambda: prepare_s4_mutable_buffers_v1(admission, live, exact_call_id=CALL_ID),
    )


def test_exact_call_mismatch_closes_admission_without_retry(formal_fixture) -> None:
    live = _live_sources(formal_fixture)
    admission = _admit(formal_fixture, live)
    _assert_error(
        "BUFFER_PREPARE_EXACT_CALL_MISMATCH",
        VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
        lambda: prepare_s4_mutable_buffers_v1(
            admission, live, exact_call_id="different-call"
        ),
    )
    assert admission._state == "FAILED_CLOSED"


def test_source_identity_drift_is_rejected_before_candidate_allocation(
    formal_fixture,
) -> None:
    live = _live_sources(formal_fixture)
    admission = _admit(formal_fixture, live)
    path = formal_fixture.plan.relu_layouts[0].alpha_path
    live[path] = live[path].detach().clone().requires_grad_(True)
    _assert_error(
        "BUFFER_PREPARE_SOURCE_IDENTITY_MISMATCH",
        VerificationRejectionReason.STATE_VERSION_MISMATCH,
        lambda: prepare_s4_mutable_buffers_v1(admission, live, exact_call_id=CALL_ID),
    )


def test_full_private_view_key_distinguishes_same_pointer_shape_stride() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    value = torch.arange(4, dtype=torch.float32, device="cuda:0").reshape(2, 2)
    transposed = value.T
    assert value.data_ptr() == transposed.data_ptr()
    assert tuple(value.shape) == tuple(transposed.shape)
    assert (value.data_ptr(), tuple(value.shape)) == (
        transposed.data_ptr(),
        tuple(transposed.shape),
    )
    key = buffer_module._view_key(0, value)
    assert key[6] == (2, 1)
    _assert_error(
        "BASE_DLPACK_VIEW_KEY_MISMATCH",
        VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
        lambda: buffer_module._view_key(0, transposed),
    )


@pytest.mark.parametrize(
    ("helper", "detail"),
    [
        ("_clone_parameter", "BUFFER_PREPARE_MANIFEST_MISMATCH"),
        ("_empty_buffer", "BUFFER_PREPARE_MANIFEST_MISMATCH"),
        ("_full_upstream", "BUFFER_PREPARE_RESOURCE_CONTEXT_RETAINED"),
        ("_create_dlpack_view", "BASE_DLPACK_VIEW_COUNT_MISMATCH"),
        ("_roundtrip_dlpack", "BASE_DLPACK_VIEW_KEY_MISMATCH"),
        ("_build_receipt", "BUFFER_PREPARE_VALIDATION_COPY_ACCOUNTING_MISMATCH"),
        ("_adopt_prepared", "BUFFER_PREPARE_ADOPTION_OWNER_MISMATCH"),
    ],
)
def test_isolated_faults_close_ticket_without_exception_context(
    formal_fixture, monkeypatch, helper, detail
) -> None:
    live = _live_sources(formal_fixture)
    admission = _admit(formal_fixture, live)

    def fail(*_args, **_kwargs):
        raise RuntimeError(f"injected:{helper}")

    monkeypatch.setattr(buffer_module, helper, fail)
    _assert_error(
        detail,
        buffer_module._REASON_BY_DETAIL[detail],
        lambda: prepare_s4_mutable_buffers_v1(admission, live, exact_call_id=CALL_ID),
    )
    assert admission._state == "TRANSFERRED"


def test_retained_stable_error_does_not_retain_candidate_tensors(
    formal_fixture, monkeypatch
) -> None:
    live = _live_sources(formal_fixture)
    admission = _admit(formal_fixture, live)
    candidate_refs: list[weakref.ReferenceType[torch.Tensor]] = []

    def fail_adoption(_receipt, owner):
        candidate_refs.extend(weakref.ref(item) for item in owner.buffers())
        raise RuntimeError("retained-adoption-fault")

    monkeypatch.setattr(buffer_module, "_adopt_prepared", fail_adoption)
    with pytest.raises(S4MutableBufferPreparationError) as caught:
        prepare_s4_mutable_buffers_v1(admission, live, exact_call_id=CALL_ID)
    stable_error = caught.value
    assert stable_error.__context__ is None
    assert len(candidate_refs) == 16
    gc.collect()
    torch.cuda.synchronize()
    assert all(reference() is None for reference in candidate_refs)


_RECEIPT_TAMPERS = (
    ("parameter_count", 6),
    ("gradient_count", 6),
    ("empty_beta_token_count", 4),
    ("candidate_storage_count", 15),
    ("base_dlpack_view_count", 15),
    ("parameter_elements", 4253),
    ("parameter_bytes", 17012),
    ("gradient_elements", 4253),
    ("gradient_bytes", 17012),
    ("candidate_logical_bytes", 34076),
    ("leased_source_tensor_count", 11),
    ("leased_source_elements", 8501),
    ("leased_source_bytes", 34004),
    ("source_d2h_copy_count", 23),
    ("source_d2h_bytes", 68012),
    ("initialized_candidate_d2h_copy_count", 7),
    ("initialized_candidate_d2h_bytes", 17036),
    ("s4_1a_d2h_copy_count", 31),
    ("s4_1a_d2h_bytes", 85052),
    ("prior_s4_0_d2h_copy_count", 23),
    ("prior_s4_0_d2h_bytes", 68012),
    ("cumulative_d2h_copy_count", 55),
    ("cumulative_d2h_bytes", 153068),
    ("parameter_d2d_copy_count", 6),
    ("parameter_d2d_bytes", 17012),
    ("warm_dlpack_view_count", 1),
    ("full_alpha_device_copy_count", 1),
    ("dense_alpha_materialization_count", 1),
    ("dense_beta_materialization_count", 1),
    ("prepare_retry_count", 1),
    ("prepare_fallback_count", 1),
    ("empty_cache_call_count", 1),
    ("provider_mapping_stability_validated", True),
    ("process_global_exclusivity_validated", True),
    ("crown_numeric_semantics_validated", True),
    ("optimizer_trajectory_validated", True),
    ("timing_recorded", True),
    ("performance_claimed", True),
    ("device", "cpu"),
    ("dtype", "torch.float64"),
)


@pytest.mark.parametrize(("field", "value"), _RECEIPT_TAMPERS)
def test_resigned_receipt_accounting_and_claim_tamper_is_rejected(
    formal_fixture, field, value
) -> None:
    _, _, prepared = _prepare(formal_fixture)
    receipt = prepared.receipt
    draft = replace(receipt, **{field: value}, receipt_hash="")
    resigned = replace(
        draft,
        receipt_hash=buffer_module._canonical_hash(draft._payload_without_hash()),
    )
    _assert_error(
        "BUFFER_PREPARE_VALIDATION_COPY_ACCOUNTING_MISMATCH",
        VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
        resigned.validate,
    )
    prepared.close()


_DESCRIPTOR_TAMPERS = (
    ("buffer_ordinal", 99),
    ("semantic_role", "optimizer_state"),
    ("slot_ordinal_or_minus_one", 99),
    ("shape", (1,)),
    ("stride", (99,)),
    ("storage_offset", 1),
    ("dtype", "torch.float64"),
    ("device", "cpu"),
    ("element_count", 1),
    ("logical_bytes", 1),
    ("requires_grad", False),
    ("is_leaf", False),
    ("contiguous", False),
    ("initialized_at_prepare", False),
    ("initial_content_hash_or_none", None),
    ("view_ordinal", 99),
)


@pytest.mark.parametrize(("field", "value"), _DESCRIPTOR_TAMPERS)
def test_resigned_descriptor_manifest_tamper_is_rejected(
    formal_fixture, field, value
) -> None:
    _, _, prepared = _prepare(formal_fixture)
    receipt = prepared.receipt
    changed = replace(receipt.buffer_descriptors[0], **{field: value})
    descriptors = (changed, *receipt.buffer_descriptors[1:])
    draft = replace(receipt, buffer_descriptors=descriptors, receipt_hash="")
    resigned = replace(
        draft,
        receipt_hash=buffer_module._canonical_hash(_unsafe_receipt_payload(draft)),
    )
    with pytest.raises(S4MutableBufferPreparationError) as caught:
        resigned.validate()
    assert caught.value.detail_code in {
        "BUFFER_PREPARE_MANIFEST_MISMATCH",
        "BUFFER_PREPARE_VALIDATION_COPY_ACCOUNTING_MISMATCH",
    }
    prepared.close()
