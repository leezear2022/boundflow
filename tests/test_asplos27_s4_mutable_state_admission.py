"""S4-0 mutable-state admission, receipt, and live-lease gates."""

# pylint: disable=missing-function-docstring,missing-class-docstring
# pylint: disable=protected-access,too-many-locals,redefined-outer-name
# pylint: disable=broad-exception-caught
# pylint: disable=unnecessary-lambda-assignment

from __future__ import annotations

from collections import defaultdict
import copy
from dataclasses import dataclass, replace
import gc
import hashlib
import json
from pathlib import Path
import pickle
import threading
from types import SimpleNamespace
import weakref

import pytest
import torch

from boundflow.frontends.onnx.frontend import import_onnx
from boundflow.ir.verification_graph import VerificationRejectionReason
from boundflow.planner import plan_interval_ibp_v0
from boundflow.runtime.asplos27_s4_mutable_state_admission import (
    S4MutableStateAdmissionError,
    _prepare_s4_mutable_state_admission_v1 as _prepare_with_failure_hook,
    extract_s4_live_mutable_sources_v1,
    prepare_s4_mutable_state_admission_v1,
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
)
from scripts.run_rvir_v4_pre_state_artifact import TOPOLOGY

ROOT = Path(__file__).resolve().parents[1]
CAPTURE = (
    ROOT / "artifacts/rvir-v4-pre-state/resnet2b-core-pre-state-v1/source_capture.pt"
)
MODEL = Path(
    "/home/lee/Codes/vnncomp2021/benchmarks/cifar10_resnet/onnx/resnet_2b.onnx"
)
CALL_ID = "asplos27-s4-formal-call-0001"


@dataclass(frozen=True)
class _FormalFixture:
    snapshot: ProductionStateSnapshotV4
    plan: R31FullRegionPlanV1


@pytest.fixture(scope="module")
def formal_fixture() -> _FormalFixture:
    if not MODEL.is_file() or not CAPTURE.is_file():
        pytest.skip("S4-0 frozen ResNet2B fixture is unavailable")
    raw = torch.load(CAPTURE, map_location="cpu", weights_only=True)
    snapshot = production_snapshot_from_payload_v4(raw["cores"][0]["pre_snapshot"])
    mapping = initialize_rvir_v4_native_pre_state(snapshot, TOPOLOGY)
    program = import_onnx(str(MODEL), do_shape_infer=True, normalize=True)
    module = plan_interval_ibp_v0(program)
    plan = compile_r31_full_region_plan_v1(module, snapshot, mapping, TOPOLOGY)
    return _FormalFixture(snapshot=snapshot, plan=plan)


def _live_sources(fixture: _FormalFixture) -> dict[str, torch.Tensor]:
    if not torch.cuda.is_available():
        pytest.skip("S4-0 CUDA fixture is unavailable")
    tensor_map = fixture.snapshot.tensor_map()
    return {
        path: tensor_map[path].value.to("cuda:0").detach().clone().requires_grad_(True)
        for layout in fixture.plan.relu_layouts
        for path in (layout.alpha_path, layout.beta_path)
    }


def _prepare(
    fixture: _FormalFixture,
    live: dict[str, torch.Tensor] | None = None,
    *,
    call_id: str = CALL_ID,
    topology=TOPOLOGY,
    snapshot: ProductionStateSnapshotV4 | None = None,
):
    sources = _live_sources(fixture) if live is None else live
    prepared = prepare_s4_mutable_state_admission_v1(
        fixture.snapshot if snapshot is None else snapshot,
        topology,
        fixture.plan,
        sources,
        exact_call_id=call_id,
    )
    return sources, prepared


def _assert_error(
    expected_detail: str,
    expected_reason: VerificationRejectionReason,
    function,
) -> None:
    with pytest.raises(S4MutableStateAdmissionError) as caught:
        function()
    assert caught.value.detail_code == expected_detail
    assert caught.value.verification_reason == expected_reason


def test_formal_admission_counts_policy_and_claim_boundary(formal_fixture) -> None:
    live, prepared = _prepare(formal_fixture)
    receipt = prepared.receipt
    receipt.validate()
    assert len(receipt.slots) == receipt.alpha_source_count == 6
    assert (
        receipt.alpha_stored_element_count,
        receipt.alpha_active_element_count,
        receipt.alpha_preserved_element_count,
    ) == (8_496, 4_248, 4_248)
    assert (
        receipt.beta_slot_count,
        receipt.active_beta_slot_count,
        receipt.active_beta_element_count,
    ) == (6, 1, 6)
    assert (
        receipt.live_tensor_count,
        receipt.live_element_count_per_pass,
        receipt.live_bytes_per_pass,
    ) == (12, 8_502, 34_008)
    assert (
        receipt.live_content_capture_pass_count,
        receipt.device_to_host_validation_copy_count,
        receipt.device_to_host_validation_bytes,
    ) == (2, 24, 68_016)
    assert formal_fixture.snapshot.optimizer_policy.deterministic is False
    assert receipt.candidate_kernel_launch_count == 0
    assert receipt.candidate_cuda_allocation_count == 0
    assert not receipt.dense_materialization_observed
    assert not receipt.timing_recorded
    assert not receipt.performance_claimed
    assert not receipt.process_global_query_exclusivity_validated
    adoption = prepared.begin_buffer_prepare(live, exact_call_id=CALL_ID)
    assert adoption.receipt.admission_hash == receipt.admission_hash
    adoption._lease.close()


def test_provider_source_readiness_is_recorded_not_rewritten(formal_fixture) -> None:
    live = _live_sources(formal_fixture)
    for layout in formal_fixture.plan.relu_layouts:
        live[layout.alpha_path].requires_grad_(False)
    _, prepared = _prepare(formal_fixture, live)
    assert all(not slot.alpha_live_requires_grad for slot in prepared.receipt.slots)
    assert all(slot.alpha_live_is_leaf for slot in prepared.receipt.slots)
    assert all(slot.beta_live_requires_grad for slot in prepared.receipt.slots)
    assert all(slot.beta_live_is_leaf for slot in prepared.receipt.slots)
    prepared.close()


def test_provider_source_readiness_drift_fails_closed(formal_fixture) -> None:
    live = _live_sources(formal_fixture)
    alpha_path = formal_fixture.plan.relu_layouts[0].alpha_path
    live[alpha_path].requires_grad_(False)
    _, prepared = _prepare(formal_fixture, live)
    live[alpha_path].requires_grad_(True)
    _assert_error(
        "LIVE_SOURCE_READINESS_MISMATCH",
        VerificationRejectionReason.DTYPE_OR_DEVICE_MISMATCH,
        lambda: prepared.begin_buffer_prepare(live, exact_call_id=CALL_ID),
    )


def test_receipt_is_canonical_tensor_free_and_json_serializable(formal_fixture) -> None:
    _, prepared = _prepare(formal_fixture)
    payload = prepared.receipt.to_dict()
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    assert "Tensor" not in encoded
    assert "cuda_stream" not in encoded
    assert "data_ptr" not in encoded
    assert prepared.receipt.stable_hash() == prepared.receipt.admission_hash
    prepared.close()


def test_tensor_free_walker_accepts_immutable_dag_sharing(formal_fixture) -> None:
    _, prepared = _prepare(formal_fixture)
    shared_shape = prepared.receipt.slots[0].alpha_active_shape
    changed_slot = replace(
        prepared.receipt.slots[0],
        alpha_active_shape=shared_shape,
        alpha_preserved_shape=shared_shape,
    )
    provisional = replace(
        prepared.receipt,
        slots=(changed_slot,) + prepared.receipt.slots[1:],
        admission_hash="0" * 64,
    )
    resigned = replace(
        provisional,
        admission_hash=hashlib.sha256(
            json.dumps(
                provisional._payload_without_hash(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
    )
    resigned.validate()
    prepared.close()


def test_topology_and_snapshot_storage_order_do_not_change_receipt(
    formal_fixture,
) -> None:
    _, baseline = _prepare(formal_fixture)
    permuted_snapshot = replace(
        formal_fixture.snapshot,
        tensors=tuple(reversed(formal_fixture.snapshot.tensors)),
    )
    _, permuted = _prepare(
        formal_fixture,
        topology=tuple(reversed(TOPOLOGY)),
        snapshot=permuted_snapshot,
    )
    assert baseline.receipt.to_dict() == permuted.receipt.to_dict()
    baseline.close()
    permuted.close()


def test_exact_call_identity_changes_only_identity_bound_receipt(
    formal_fixture,
) -> None:
    _, first = _prepare(formal_fixture, call_id="s4-call-a")
    _, second = _prepare(formal_fixture, call_id="s4-call-b")
    left = first.receipt.to_dict()
    right = second.receipt.to_dict()
    assert left.pop("exact_call_identity_hash") != right.pop("exact_call_identity_hash")
    assert left.pop("admission_hash") != right.pop("admission_hash")
    assert left == right
    first.close()
    second.close()


@pytest.mark.parametrize(
    "invalid",
    ["", " bad", "bad/id", "bad id", "bad\nid", "*", "汉字", "a" * 257],
)
def test_illegal_exact_call_identity_fails_closed(formal_fixture, invalid) -> None:
    live = _live_sources(formal_fixture)
    _assert_error(
        "EXACT_CALL_IDENTITY_INVALID",
        VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
        lambda: prepare_s4_mutable_state_admission_v1(
            formal_fixture.snapshot,
            TOPOLOGY,
            formal_fixture.plan,
            live,
            exact_call_id=invalid,
        ),
    )


def test_live_mapping_requires_exact_builtin_dict(formal_fixture) -> None:
    class DictSubclass(dict):
        pass

    live = DictSubclass(_live_sources(formal_fixture))
    _assert_error(
        "LIVE_SOURCE_CONTAINER_TYPE_MISMATCH",
        VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
        lambda: prepare_s4_mutable_state_admission_v1(
            formal_fixture.snapshot,
            TOPOLOGY,
            formal_fixture.plan,
            live,
            exact_call_id=CALL_ID,
        ),
    )


@pytest.mark.parametrize(
    ("mode", "detail"),
    [
        ("snapshot-schema", "SNAPSHOT_SCHEMA_VERSION_MISMATCH"),
        ("snapshot-id", "SNAPSHOT_SCHEMA_VERSION_MISMATCH"),
        ("plan-schema", "PLAN_SCHEMA_VERSION_MISMATCH"),
    ],
)
def test_snapshot_and_plan_envelopes_fail_closed(formal_fixture, mode, detail) -> None:
    live = _live_sources(formal_fixture)
    if mode == "snapshot-schema":
        snapshot = replace(formal_fixture.snapshot, schema_version="wrong")
        operation = lambda: _prepare(formal_fixture, live, snapshot=snapshot)
    elif mode == "snapshot-id":
        snapshot = replace(formal_fixture.snapshot, snapshot_id="")
        operation = lambda: _prepare(formal_fixture, live, snapshot=snapshot)
    else:
        plan = replace(formal_fixture.plan, schema_version="wrong")
        operation = lambda: prepare_s4_mutable_state_admission_v1(
            formal_fixture.snapshot,
            TOPOLOGY,
            plan,
            live,
            exact_call_id=CALL_ID,
        )
    _assert_error(
        detail,
        VerificationRejectionReason.STATE_VERSION_MISMATCH,
        operation,
    )


@pytest.mark.parametrize(
    "policy_changes",
    [
        {"bound_lower": False, "bound_upper": True},
        {"bound_lower": True, "bound_upper": True},
        {"fix_intermediate_bounds": False},
    ],
)
def test_non_lower_only_optimizer_policy_is_rejected(
    formal_fixture, policy_changes
) -> None:
    live = _live_sources(formal_fixture)
    policy = replace(
        formal_fixture.snapshot.optimizer_policy,
        **policy_changes,
    )
    snapshot = replace(formal_fixture.snapshot, optimizer_policy=policy)
    _assert_error(
        "OPTIMIZER_POLICY_MISMATCH",
        VerificationRejectionReason.BOUND_POLARITY_MISMATCH,
        lambda: _prepare(formal_fixture, live, snapshot=snapshot),
    )


@pytest.mark.parametrize("mode", ["missing", "duplicate", "wrong-native"])
def test_topology_identity_mismatch_is_rejected(formal_fixture, mode) -> None:
    live = _live_sources(formal_fixture)
    topology = list(TOPOLOGY)
    if mode == "missing":
        topology.pop()
    elif mode == "duplicate":
        topology[-1] = topology[0]
    else:
        topology[-1] = replace(topology[-1], native_preactivation="not-a-node")
    _assert_error(
        "TOPOLOGY_IDENTITY_MISMATCH",
        VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
        lambda: _prepare(formal_fixture, live, topology=tuple(topology)),
    )


@pytest.mark.parametrize("mode", ["missing", "extra", "non_string"])
def test_live_mapping_coverage_fails_closed(formal_fixture, mode) -> None:
    live = _live_sources(formal_fixture)
    if mode == "missing":
        live.pop(next(iter(live)))
        detail = "LIVE_SOURCE_COVERAGE_MISMATCH"
    elif mode == "extra":
        live["alpha/extra/path"] = next(iter(live.values())).clone().detach()
        detail = "LIVE_SOURCE_COVERAGE_MISMATCH"
    else:
        live[1] = live.pop(next(iter(live)))  # type: ignore[index]
        detail = "LIVE_SOURCE_CONTAINER_TYPE_MISMATCH"
    _assert_error(
        detail,
        (
            VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH
            if mode == "non_string"
            else VerificationRejectionReason.VJP_OWNER_OR_SAVED_STATE_MISMATCH
        ),
        lambda: prepare_s4_mutable_state_admission_v1(
            formal_fixture.snapshot,
            TOPOLOGY,
            formal_fixture.plan,
            live,
            exact_call_id=CALL_ID,
        ),
    )


@pytest.mark.parametrize("mode", ["tensor-subclass", "object-alias"])
def test_tensor_subclass_and_object_alias_fail_closed(formal_fixture, mode) -> None:
    live = _live_sources(formal_fixture)
    if mode == "tensor-subclass":
        first = next(iter(live))
        live[first] = torch.nn.Parameter(live[first].detach())
        detail = "LIVE_SOURCE_TENSOR_SUBCLASS_UNSUPPORTED"
        reason = VerificationRejectionReason.DTYPE_OR_DEVICE_MISMATCH
    else:
        paths = list(live)
        live[paths[1]] = live[paths[0]]
        detail = "LIVE_SOURCE_OBJECT_ALIAS_CONFLICT"
        reason = VerificationRejectionReason.UNSAFE_ALIAS_OR_LIFETIME
    _assert_error(
        detail,
        reason,
        lambda: _prepare(formal_fixture, live),
    )


def test_nonempty_shared_storage_alias_fails_closed(formal_fixture) -> None:
    live = _live_sources(formal_fixture)
    nonempty = [path for path, value in live.items() if value.numel()]
    left_path, right_path = nonempty[:2]
    left_shape, right_shape = live[left_path].shape, live[right_path].shape
    left_count, right_count = live[left_path].numel(), live[right_path].numel()
    base = torch.empty(left_count + right_count, device="cuda:0")
    live[left_path] = base[:left_count].view(left_shape).detach().requires_grad_(True)
    live[right_path] = base[left_count:].view(right_shape).detach().requires_grad_(True)
    _assert_error(
        "LIVE_SOURCE_STORAGE_ALIAS_CONFLICT",
        VerificationRejectionReason.UNSAFE_ALIAS_OR_LIFETIME,
        lambda: _prepare(formal_fixture, live),
    )


@pytest.mark.parametrize(
    ("mode", "detail", "reason"),
    [
        (
            "clone",
            "LIVE_SOURCE_OBJECT_REPLACED",
            VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
        ),
        (
            "storage",
            "LIVE_SOURCE_STORAGE_REPLACED",
            VerificationRejectionReason.UNSAFE_ALIAS_OR_LIFETIME,
        ),
        (
            "version",
            "LIVE_TENSOR_VERSION_MISMATCH",
            VerificationRejectionReason.STATE_VERSION_MISMATCH,
        ),
        (
            "content",
            "LIVE_SOURCE_CONTENT_MISMATCH",
            VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
        ),
    ],
)
def test_post_admission_live_drift_fails_closed(
    formal_fixture, mode, detail, reason
) -> None:
    live, prepared = _prepare(formal_fixture)
    path = next(path for path, value in live.items() if value.numel())
    value = live[path]
    if mode == "clone":
        live[path] = value.detach().clone().requires_grad_(True)
    elif mode == "storage":
        value.data = value.detach().clone()
    elif mode == "version":
        with torch.no_grad():
            value.add_(1.0)
    else:
        value.data.add_(1.0)
    _assert_error(
        detail,
        reason,
        lambda: prepared.begin_buffer_prepare(live, exact_call_id=CALL_ID),
    )


@pytest.mark.parametrize("mode", ["exact-call", "owner-thread", "cuda-stream"])
def test_exact_call_thread_and_stream_identity_fail_closed(
    formal_fixture, mode
) -> None:
    live, prepared = _prepare(formal_fixture)
    if mode == "exact-call":
        _assert_error(
            "EXACT_CALL_IDENTITY_MISMATCH",
            VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
            lambda: prepared.begin_buffer_prepare(live, exact_call_id="another-call"),
        )
    elif mode == "owner-thread":
        caught: list[BaseException] = []

        def cross_thread() -> None:
            try:
                prepared.begin_buffer_prepare(live, exact_call_id=CALL_ID)
            except BaseException as error:  # pragma: no branch - required capture
                caught.append(error)

        thread = threading.Thread(target=cross_thread)
        thread.start()
        thread.join()
        assert len(caught) == 1
        assert isinstance(caught[0], S4MutableStateAdmissionError)
        assert caught[0].detail_code == "LIVE_SOURCE_OWNER_THREAD_MISMATCH"
        assert caught[0].verification_reason == (
            VerificationRejectionReason.UNSAFE_ALIAS_OR_LIFETIME
        )
    else:
        other_stream = torch.cuda.Stream()
        with torch.cuda.stream(other_stream):
            _assert_error(
                "LIVE_SOURCE_STREAM_MISMATCH",
                VerificationRejectionReason.UNSAFE_ALIAS_OR_LIFETIME,
                lambda: prepared.begin_buffer_prepare(live, exact_call_id=CALL_ID),
            )
    prepared.close()


@pytest.mark.parametrize(
    "mode",
    [
        "copy",
        "deepcopy",
        "pickle",
        "second-begin",
        "second-transfer",
        "after-close",
    ],
)
def test_lease_is_single_transfer_and_nonserializable(formal_fixture, mode) -> None:
    live, prepared = _prepare(formal_fixture)
    if mode in {"copy", "deepcopy", "pickle"}:
        operations = {
            "copy": lambda: copy.copy(prepared),
            "deepcopy": lambda: copy.deepcopy(prepared),
            "pickle": lambda: pickle.dumps(prepared),
        }
        _assert_error(
            "LIVE_LEASE_SERIALIZATION_FORBIDDEN",
            VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
            operations[mode],
        )
        prepared.close()
        return
    adoption = prepared.begin_buffer_prepare(live, exact_call_id=CALL_ID)
    if mode == "second-begin":
        operation = lambda: prepared.begin_buffer_prepare(live, exact_call_id=CALL_ID)
        detail = "LIVE_LEASE_ALREADY_TRANSFERRED"
    elif mode == "second-transfer":
        operation = lambda: adoption._lease.transfer_to_prepared_runtime(
            expected_admission_hash=adoption.receipt.admission_hash,
            exact_call_id=CALL_ID,
        )
        detail = "LIVE_LEASE_ALREADY_TRANSFERRED"
    else:
        adoption._lease.close()
        operation = lambda: adoption._lease.mark_commit_started(exact_call_id=CALL_ID)
        detail = "LIVE_LEASE_ALREADY_CLOSED"
    _assert_error(
        detail,
        VerificationRejectionReason.UNSAFE_ALIAS_OR_LIFETIME,
        operation,
    )
    adoption._lease.close()


def test_lease_holds_strong_tensor_ownership_until_close(formal_fixture) -> None:
    live, prepared = _prepare(formal_fixture)
    path = next(iter(live))
    reference = weakref.ref(live[path])
    del live[path]
    gc.collect()
    assert reference() is not None
    prepared.close()
    gc.collect()
    assert reference() is None


@pytest.mark.parametrize(
    "phase",
    [
        "after_input_envelope",
        "after_snapshot_validation",
        "after_first_live_capture",
        "after_receipt_validation",
        "before_second_live_capture",
        "before_lease_publish",
    ],
)
def test_failure_injection_never_publishes_partial_owner(formal_fixture, phase) -> None:
    live = _live_sources(formal_fixture)
    versions = {path: value._version for path, value in live.items()}
    contents = {path: value.detach().cpu().clone() for path, value in live.items()}

    def fail_at(observed: str) -> None:
        if observed == phase:
            raise RuntimeError(f"injected:{phase}")

    with pytest.raises(RuntimeError, match=f"injected:{phase}"):
        _prepare_with_failure_hook(
            formal_fixture.snapshot,
            TOPOLOGY,
            formal_fixture.plan,
            live,
            exact_call_id=CALL_ID,
            failure_hook=fail_at,
        )
    assert all(value._version == versions[path] for path, value in live.items())
    assert all(
        torch.equal(value.detach().cpu(), contents[path])
        for path, value in live.items()
    )


@pytest.mark.parametrize("mode", ["object", "storage", "version", "content"])
def test_double_capture_race_fails_with_one_stable_reason(formal_fixture, mode) -> None:
    live = _live_sources(formal_fixture)
    path = next(path for path, value in live.items() if value.numel())

    def mutate_after_first_capture(phase: str) -> None:
        if phase != "after_first_live_capture":
            return
        value = live[path]
        if mode == "object":
            live[path] = value.detach().clone().requires_grad_(True)
        elif mode == "storage":
            value.data = value.detach().clone()
        elif mode == "version":
            with torch.no_grad():
                value.add_(1.0)
        else:
            value.data.add_(1.0)

    _assert_error(
        "LIVE_SOURCE_READ_RACE",
        VerificationRejectionReason.STATE_VERSION_MISMATCH,
        lambda: _prepare_with_failure_hook(
            formal_fixture.snapshot,
            TOPOLOGY,
            formal_fixture.plan,
            live,
            exact_call_id=CALL_ID,
            failure_hook=mutate_after_first_capture,
        ),
    )


def test_double_capture_stream_drift_fails_as_read_race(formal_fixture) -> None:
    live = _live_sources(formal_fixture)
    original = torch.cuda.current_stream()
    other = torch.cuda.Stream()

    def switch_stream(phase: str) -> None:
        if phase == "after_first_live_capture":
            torch.cuda.set_stream(other)

    try:
        _assert_error(
            "LIVE_SOURCE_READ_RACE",
            VerificationRejectionReason.STATE_VERSION_MISMATCH,
            lambda: _prepare_with_failure_hook(
                formal_fixture.snapshot,
                TOPOLOGY,
                formal_fixture.plan,
                live,
                exact_call_id=CALL_ID,
                failure_hook=switch_stream,
            ),
        )
    finally:
        torch.cuda.set_stream(original)


def test_fully_resigned_receipt_still_cannot_cross_live_lease(formal_fixture) -> None:
    live, prepared = _prepare(formal_fixture)
    changed = replace(prepared.receipt, exact_call_identity_hash="1" * 64)
    payload = changed._payload_without_hash()
    resigned_hash = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    changed = replace(changed, admission_hash=resigned_hash)
    changed.validate()
    prepared.receipt = changed
    _assert_error(
        "LIVE_LEASE_ADMISSION_MISMATCH",
        VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
        lambda: prepared.begin_buffer_prepare(live, exact_call_id=CALL_ID),
    )


@pytest.mark.parametrize(
    ("field", "value", "detail"),
    [
        ("admission_hash", "0" * 64, "RECEIPT_IDENTITY_MISMATCH"),
        ("live_tensor_count", 13, "RECEIPT_LIVE_COPY_ACCOUNTING_MISMATCH"),
        ("live_bytes_per_pass", 34_012, "RECEIPT_LIVE_COPY_ACCOUNTING_MISMATCH"),
        (
            "device_to_host_validation_copy_count",
            23,
            "RECEIPT_LIVE_COPY_ACCOUNTING_MISMATCH",
        ),
        (
            "device_to_host_validation_bytes",
            68_020,
            "RECEIPT_LIVE_COPY_ACCOUNTING_MISMATCH",
        ),
        ("candidate_kernel_launch_count", 1, "CLAIM_FLAG_TRUE_BEFORE_FORMAL"),
        ("candidate_cuda_allocation_count", 1, "CLAIM_FLAG_TRUE_BEFORE_FORMAL"),
        ("dense_materialization_observed", True, "CLAIM_FLAG_TRUE_BEFORE_FORMAL"),
        ("timing_recorded", True, "CLAIM_FLAG_TRUE_BEFORE_FORMAL"),
        ("performance_claimed", True, "CLAIM_FLAG_TRUE_BEFORE_FORMAL"),
        (
            "process_global_query_exclusivity_validated",
            True,
            "S4_0_CROSS_QUERY_EXCLUSIVITY_UNPROVEN",
        ),
    ],
)
def test_receipt_tamper_fails_closed(formal_fixture, field, value, detail) -> None:
    _, prepared = _prepare(formal_fixture)
    tampered = replace(prepared.receipt, **{field: value})
    expected_reason = (
        VerificationRejectionReason.QUEUE_OR_TERMINATION_EFFECT_CROSSED
        if detail == "S4_0_CROSS_QUERY_EXCLUSIVITY_UNPROVEN"
        else VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH
    )
    _assert_error(detail, expected_reason, tampered.validate)
    prepared.close()


def test_mutable_slot_order_tamper_has_specific_reason(formal_fixture) -> None:
    _, prepared = _prepare(formal_fixture)
    slots = list(prepared.receipt.slots)
    slots[0], slots[1] = slots[1], slots[0]
    tampered = replace(prepared.receipt, slots=tuple(slots))
    _assert_error(
        "MUTABLE_SLOT_ORDER_MISMATCH",
        VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
        tampered.validate,
    )
    prepared.close()


class _DictSubclass(dict):
    pass


def _provider_result(live: dict[str, torch.Tensor], fixture: _FormalFixture):
    alpha_data: dict[str, dict[str, torch.Tensor]] = {}
    beta_data: dict[str, list[object]] = {}
    for layout, link in zip(fixture.plan.relu_layouts, TOPOLOGY):
        alpha_data[link.provider_activation] = {
            link.provider_start_node: live[layout.alpha_path]
        }
        beta_data[link.provider_preactivation] = [
            SimpleNamespace(val=live[layout.beta_path])
        ]
    return SimpleNamespace(
        alphas_by_layer=SimpleNamespace(_data=alpha_data),
        betas_by_layer=SimpleNamespace(_data=beta_data),
    )


def test_provider_adapter_extracts_exact_builtin_structure(formal_fixture) -> None:
    live = _live_sources(formal_fixture)
    extracted = extract_s4_live_mutable_sources_v1(
        _provider_result(live, formal_fixture), TOPOLOGY
    )
    assert tuple(sorted(extracted)) == tuple(sorted(live))
    assert all(extracted[path] is live[path] for path in live)


def test_provider_adapter_accepts_exact_stdlib_defaultdict_alpha_owner(
    formal_fixture,
) -> None:
    live = _live_sources(formal_fixture)
    result = _provider_result(live, formal_fixture)
    result.alphas_by_layer._data = defaultdict(dict, result.alphas_by_layer._data)
    extracted = extract_s4_live_mutable_sources_v1(result, TOPOLOGY)
    assert tuple(sorted(extracted)) == tuple(sorted(live))
    assert all(extracted[path] is live[path] for path in live)


@pytest.mark.parametrize(
    ("mode", "detail"),
    [
        ("alpha_data_subclass", "LIVE_SOURCE_NESTED_CONTAINER_TYPE_MISMATCH"),
        ("alpha_default_factory", "LIVE_SOURCE_NESTED_CONTAINER_TYPE_MISMATCH"),
        ("inner_alpha_subclass", "LIVE_SOURCE_NESTED_CONTAINER_TYPE_MISMATCH"),
        ("beta_tuple", "LIVE_SOURCE_NESTED_CONTAINER_TYPE_MISMATCH"),
        ("missing_sparse_val", "LIVE_SOURCE_NESTED_CONTAINER_TYPE_MISMATCH"),
        ("parameter", "LIVE_SOURCE_TENSOR_SUBCLASS_UNSUPPORTED"),
        ("missing_key", "LIVE_SOURCE_COVERAGE_MISMATCH"),
    ],
)
def test_provider_adapter_rejects_unpinned_structure(
    formal_fixture, mode, detail
) -> None:
    live = _live_sources(formal_fixture)
    result = _provider_result(live, formal_fixture)
    alpha_data = result.alphas_by_layer._data
    beta_data = result.betas_by_layer._data
    first = TOPOLOGY[0]
    if mode == "alpha_data_subclass":
        result.alphas_by_layer._data = _DictSubclass(alpha_data)
    elif mode == "alpha_default_factory":
        result.alphas_by_layer._data = defaultdict(list, alpha_data)
    elif mode == "inner_alpha_subclass":
        alpha_data[first.provider_activation] = _DictSubclass(
            alpha_data[first.provider_activation]
        )
    elif mode == "beta_tuple":
        beta_data[first.provider_preactivation] = tuple(
            beta_data[first.provider_preactivation]
        )
    elif mode == "missing_sparse_val":
        beta_data[first.provider_preactivation] = [SimpleNamespace()]
    elif mode == "parameter":
        alpha_data[first.provider_activation][first.provider_start_node] = (
            torch.nn.Parameter(
                alpha_data[first.provider_activation][first.provider_start_node]
            )
        )
    else:
        alpha_data.pop(first.provider_activation)
    expected_reason = (
        VerificationRejectionReason.DTYPE_OR_DEVICE_MISMATCH
        if mode == "parameter"
        else (
            VerificationRejectionReason.VJP_OWNER_OR_SAVED_STATE_MISMATCH
            if mode == "missing_key"
            else VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH
        )
    )
    _assert_error(
        detail,
        expected_reason,
        lambda: extract_s4_live_mutable_sources_v1(result, TOPOLOGY),
    )
