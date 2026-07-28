"""Exact-version runtime payloads for Schedule IR state actions."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Optional

import torch

from ..ir.bound import BFBoundModule, BoundRepresentation
from ..ir.plan import StateValidity
from ..ir.schedule import StateLoadAction, StateStoreAction
from .bound_ir_interpreter import PlainCrownBoundIRSession, runtime_value_hash


@dataclass(frozen=True)
class BoundRuntimeStatePayload:
    """One owned dense Bound value tied to exact compiler semantics."""

    state_id: str
    source_value_id: str
    state_version: str
    bound_module_hash: str
    value_hash: str
    value: torch.Tensor

    @classmethod
    def create(
        cls,
        *,
        state_id: str,
        source_value_id: str,
        state_version: str,
        bound_module_hash: str,
        value: torch.Tensor,
    ) -> "BoundRuntimeStatePayload":
        """Own a detached tensor and bind its immutable content hash."""

        owned = value.detach().clone()
        payload = cls(
            state_id=state_id,
            source_value_id=source_value_id,
            state_version=state_version,
            bound_module_hash=bound_module_hash,
            value_hash=runtime_value_hash(owned),
            value=owned,
        )
        payload.validate()
        return payload

    def validate(self) -> None:
        """Reject incomplete, mutable-gradient, or tampered payloads."""

        for name in (
            "state_id",
            "source_value_id",
            "state_version",
            "bound_module_hash",
            "value_hash",
        ):
            if not getattr(self, name):
                raise ValueError(f"runtime state payload {name} must be non-empty")
        if len(self.bound_module_hash) != 64 or len(self.value_hash) != 64:
            raise ValueError("runtime state payload hashes must be SHA-256")
        if not torch.is_tensor(self.value):
            raise TypeError("runtime state payload value must be a tensor")
        if self.value.requires_grad:
            raise ValueError("runtime state payload must not retain autograd state")
        if runtime_value_hash(self.value) != self.value_hash:
            raise ValueError("runtime state payload content hash mismatch")

    def stable_hash(self) -> str:
        """Return an identity hash without serializing tensor bytes twice."""

        self.validate()
        identity = "|".join(
            (
                self.state_id,
                self.source_value_id,
                self.state_version,
                self.bound_module_hash,
                self.value_hash,
            )
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()


class BoundRuntimeStateStore:
    """In-memory exact-match store consumed by typed Schedule execution."""

    def __init__(self) -> None:
        self._payloads: dict[str, BoundRuntimeStatePayload] = {}
        self.load_hits = 0
        self.load_misses = 0
        self.stores = 0
        self.invalidations = 0

    def put(self, payload: BoundRuntimeStatePayload) -> None:
        """Insert an already validated owned payload."""

        payload.validate()
        self._payloads[payload.state_id] = BoundRuntimeStatePayload.create(
            state_id=payload.state_id,
            source_value_id=payload.source_value_id,
            state_version=payload.state_version,
            bound_module_hash=payload.bound_module_hash,
            value=payload.value,
        )

    def validity(
        self,
        *,
        state_id: str,
        source_value_id: str,
        state_version: str,
        bound_module_hash: str,
    ) -> StateValidity:
        """Describe exact availability for Plan selection without loading."""

        payload = self._payloads.get(state_id)
        valid = (
            payload is not None
            and payload.source_value_id == source_value_id
            and payload.state_version == state_version
            and payload.bound_module_hash == bound_module_hash
        )
        return StateValidity(
            state_id=state_id,
            source_value_id=source_value_id,
            state_version=state_version,
            valid=valid,
            invalidation_reason=None if valid else "runtime_state_exact_match_miss",
        )

    def load(
        self,
        action: StateLoadAction,
        *,
        bound_module: BFBoundModule,
        session: PlainCrownBoundIRSession,
    ) -> BoundRuntimeStatePayload:
        """Load only an exact module/value/version payload into the session."""

        payload = self._payloads.get(action.state_id)
        expected_module_hash = bound_module.stable_hash()
        if (
            payload is None
            or payload.source_value_id != action.source_value_id
            or payload.state_version != action.state_version
            or payload.bound_module_hash != expected_module_hash
        ):
            self.load_misses += 1
            raise ValueError(
                "Schedule IR state load has no exact runtime payload: "
                f"{action.state_id}"
            )
        payload.validate()
        session.load_state_value(
            action.source_value_id,
            state_version=action.state_version,
            value=payload.value,
        )
        self.load_hits += 1
        return payload

    def store(
        self,
        action: StateStoreAction,
        *,
        bound_module: BFBoundModule,
        session: PlainCrownBoundIRSession,
    ) -> BoundRuntimeStatePayload:
        """Export one computed dense value under the Schedule identity."""

        value = session.export_state_value(
            action.source_value_id,
            state_version=action.state_version,
        )
        payload = BoundRuntimeStatePayload.create(
            state_id=action.state_id,
            source_value_id=action.source_value_id,
            state_version=action.state_version,
            bound_module_hash=bound_module.stable_hash(),
            value=value,
        )
        self._payloads[action.state_id] = payload
        self.stores += 1
        return payload

    def invalidate(self, state_id: str) -> Optional[BoundRuntimeStatePayload]:
        """Remove one exact state; unknown IDs remain a deterministic no-op."""

        payload = self._payloads.pop(state_id, None)
        self.invalidations += 1
        return payload

    def audit(self) -> dict[str, int]:
        """Return deterministic state-runtime counters."""

        return {
            "entries": len(self._payloads),
            "load_hits": self.load_hits,
            "load_misses": self.load_misses,
            "stores": self.stores,
            "invalidations": self.invalidations,
        }


def validate_state_value_capability(
    bound_module: BFBoundModule, source_value_id: str
) -> None:
    """Fail closed until structured state payload serialization is specified."""

    value = next(
        (
            candidate
            for candidate in bound_module.graph.values
            if candidate.value_id == source_value_id
        ),
        None,
    )
    if value is None:
        raise ValueError(f"unknown Bound state value: {source_value_id}")
    if value.representation != BoundRepresentation.DENSE:
        raise NotImplementedError(
            "runtime state payload v1 supports dense Bound values only"
        )


__all__ = [
    "BoundRuntimeStatePayload",
    "BoundRuntimeStateStore",
    "validate_state_value_capability",
]
