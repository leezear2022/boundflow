"""RVIR-v3 executable payload and independent replacement contracts."""

# pylint: disable=missing-function-docstring,too-few-public-methods

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from boundflow.runtime.rvir_v3_replacement import (
    DomainSlice,
    ExecutableVerifierPayload,
    OwnedVerifierTensor,
    ReplacementStateUpdate,
    TorchAffineRVIRV3Backend,
    VerifierPhase,
    VerifierTensorRole,
    execute_rvir_v3_replacement,
)


def _tensor(
    tensor_id: str, role: VerifierTensorRole, value: torch.Tensor
) -> OwnedVerifierTensor:
    return OwnedVerifierTensor.own(tensor_id, role, value)


def _payload(
    *,
    phase: VerifierPhase = VerifierPhase.INITIAL_CROWN,
    polarities: tuple[str, ...] = ("lower", "upper"),
    ragged: bool = False,
    copy_out: bool = False,
) -> ExecutableVerifierPayload:
    lower = torch.tensor([[-1.0, 0.0], [0.5, -0.5], [-0.25, 0.25]])
    upper = lower + 1.0
    values = [
        _tensor("input.lower", VerifierTensorRole.INPUT_LOWER, lower),
        _tensor("input.upper", VerifierTensorRole.INPUT_UPPER, upper),
        _tensor(
            "objective",
            VerifierTensorRole.LINEAR_SPEC,
            torch.tensor([[1.0, -1.0], [-0.5, 2.0]]),
        ),
        _tensor(
            "program.weight",
            VerifierTensorRole.PROGRAM_WEIGHT,
            torch.tensor([[2.0, -1.0], [0.5, 3.0]]),
        ),
        _tensor(
            "program.bias", VerifierTensorRole.PROGRAM_BIAS, torch.tensor([0.25, -0.5])
        ),
    ]
    if phase in {VerifierPhase.ALPHA_OPTIMIZE, VerifierPhase.BETA_SPLIT}:
        values.append(
            _tensor("state.alpha", VerifierTensorRole.ALPHA_STATE, torch.ones(2))
        )
    if phase == VerifierPhase.BETA_SPLIT:
        values.extend(
            (
                _tensor("state.beta", VerifierTensorRole.BETA_STATE, torch.zeros(2)),
                _tensor("split.lower", VerifierTensorRole.SPLIT_LOWER, lower + 0.1),
                _tensor("split.upper", VerifierTensorRole.SPLIT_UPPER, upper - 0.1),
            )
        )
    return ExecutableVerifierPayload(
        query_id="rvir-v3-00000000",
        sequence_number=0,
        parent_query_id=None,
        phase=phase,
        method="CROWN-Optimized" if phase != VerifierPhase.INITIAL_CROWN else "CROWN",
        requested_polarities=polarities,
        tensors=tuple(values),
        expected_result_shape=(3, 2),
        ragged_slices=(
            (DomainSlice("ragged-0", 0, 1), DomainSlice("ragged-1", 1, 3))
            if ragged
            else ()
        ),
        mutable_state_ids=(("state.alpha",) if copy_out else ()),
        copy_out_state_ids=(("state.alpha",) if copy_out else ()),
    )


@pytest.mark.parametrize(
    ("phase", "polarities"),
    [
        (VerifierPhase.INITIAL_CROWN, ("lower",)),
        (VerifierPhase.ALPHA_OPTIMIZE, ("upper",)),
        (VerifierPhase.BETA_SPLIT, ("lower", "upper")),
    ],
)
def test_independent_backend_executes_phase_and_polarity(
    phase: VerifierPhase, polarities: tuple[str, ...]
) -> None:
    execution = execute_rvir_v3_replacement(
        _payload(phase=phase, polarities=polarities), TorchAffineRVIRV3Backend()
    )

    assert (execution.lower is not None) == ("lower" in polarities)
    assert (execution.upper is not None) == ("upper" in polarities)
    assert execution.replacement_dispatch_count == 1
    assert execution.original_callback_count == 0
    assert execution.fallback_dispatch_count == 0
    assert execution.performance_claimed is False


def test_dense_and_ragged_batches_are_semantically_exact() -> None:
    dense = execute_rvir_v3_replacement(_payload(), TorchAffineRVIRV3Backend())
    ragged = execute_rvir_v3_replacement(
        _payload(ragged=True), TorchAffineRVIRV3Backend()
    )

    torch.testing.assert_close(dense.lower, ragged.lower)
    torch.testing.assert_close(dense.upper, ragged.upper)
    assert dense.result_hash == ragged.result_hash
    assert dense.payload_hash != ragged.payload_hash


def test_payload_owns_tensors_and_detects_later_payload_mutation() -> None:
    caller = torch.tensor([[-1.0, 0.0], [0.5, -0.5], [-0.25, 0.25]])
    owned = OwnedVerifierTensor.own(
        "input.lower", VerifierTensorRole.INPUT_LOWER, caller
    )
    caller.add_(100)
    assert float(owned.value.max()) < 2

    owned.value.add_(1)
    with pytest.raises(ValueError, match="content differs"):
        owned.validate()


@pytest.mark.parametrize(
    "slices",
    [
        (DomainSlice("a", 0, 1), DomainSlice("b", 2, 3)),
        (DomainSlice("a", 0, 2), DomainSlice("b", 1, 3)),
        (DomainSlice("a", 0, 2),),
    ],
)
def test_ragged_gap_overlap_and_incomplete_coverage_fail_closed(
    slices: tuple[DomainSlice, ...],
) -> None:
    with pytest.raises(ValueError, match="gap or overlap|do not cover"):
        replace(_payload(), ragged_slices=slices).validate()


def test_alpha_and_beta_phases_require_executable_state() -> None:
    initial = _payload()
    with pytest.raises(ValueError, match="omits alpha state"):
        replace(initial, phase=VerifierPhase.ALPHA_OPTIMIZE).validate()
    with pytest.raises(ValueError, match="omits executable state"):
        replace(initial, phase=VerifierPhase.BETA_SPLIT).validate()


class _ExternalBackend:
    backend_id = "external_abcrown_exact_call/v1"

    def execute(self, payload, tensors):  # type: ignore[no-untyped-def]
        raise AssertionError("external backend must be rejected before dispatch")


def test_external_provider_backend_is_rejected_before_dispatch() -> None:
    with pytest.raises(ValueError, match="backend identity differs"):
        execute_rvir_v3_replacement(_payload(), _ExternalBackend())


class _MutatingBackend(TorchAffineRVIRV3Backend):
    def execute(self, payload, tensors):  # type: ignore[no-untyped-def]
        base = super().execute(payload, tensors)
        return replace(
            base,
            state_updates=(
                ReplacementStateUpdate("state.alpha", tensors["state.alpha"] + 2),
            ),
        )


def test_declared_state_mutation_is_atomic_and_receipted() -> None:
    payload = _payload(phase=VerifierPhase.ALPHA_OPTIMIZE, copy_out=True)
    live = torch.ones(2)
    execution = execute_rvir_v3_replacement(
        payload,
        _MutatingBackend(),
        copy_out_targets={"state.alpha": live},
    )

    torch.testing.assert_close(live, torch.full((2,), 3.0))
    assert len(execution.mutations) == 1
    assert execution.mutations[0].before_sha256 != execution.mutations[0].after_sha256
    assert execution.mutations[0].copied_out is True


def test_undeclared_or_stale_state_mutation_fails_before_copy_out() -> None:
    payload = _payload(phase=VerifierPhase.ALPHA_OPTIMIZE, copy_out=True)
    stale = torch.zeros(2)
    with pytest.raises(ValueError, match="live copy-out state differs"):
        execute_rvir_v3_replacement(
            payload,
            _MutatingBackend(),
            copy_out_targets={"state.alpha": stale},
        )
    torch.testing.assert_close(stale, torch.zeros(2))

    no_mutation = replace(payload, mutable_state_ids=(), copy_out_state_ids=())
    with pytest.raises(ValueError, match="undeclared state mutation"):
        execute_rvir_v3_replacement(no_mutation, _MutatingBackend())


class _InvalidResultBackend(TorchAffineRVIRV3Backend):
    def execute(self, payload, tensors):  # type: ignore[no-untyped-def]
        base = super().execute(payload, tensors)
        return replace(base, lower=torch.full((1,), float("nan")))


def test_invalid_result_fails_closed_without_state_commit() -> None:
    payload = _payload(phase=VerifierPhase.ALPHA_OPTIMIZE, copy_out=True)
    live = torch.ones(2)
    with pytest.raises(ValueError, match="result tensor differs"):
        execute_rvir_v3_replacement(
            payload,
            _InvalidResultBackend(),
            copy_out_targets={"state.alpha": live},
        )
    torch.testing.assert_close(live, torch.ones(2))


def test_payload_hash_binds_parent_phase_polarity_and_tensor_content() -> None:
    payload = _payload()
    variants = (
        replace(payload, parent_query_id="parent-0"),
        _payload(phase=VerifierPhase.ALPHA_OPTIMIZE),
        _payload(polarities=("lower",)),
    )
    assert len({payload.stable_hash(), *(item.stable_hash() for item in variants)}) == 4
