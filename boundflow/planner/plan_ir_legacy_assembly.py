"""Deterministic assembly and audit report for legacy Plan IR migrations."""

# Compact report dataclasses use self-describing method names.
# pylint: disable=missing-function-docstring

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import json
from typing import Tuple

from ..ir.bound import BFBoundModule
from ..ir.plan import PlanProvenance, PlanTemplate
from .plan_ir_legacy import (
    LegacyMigrationStatus,
    LegacyPlanKind,
    LegacyPlanMigration,
)


class LegacyAssemblyDisposition(Enum):
    """How one migration affected the consolidated PlanTemplate."""

    ACCEPTED = "accepted"
    CLASSIFIED_UNSUPPORTED = "classified_unsupported"
    REJECTED = "rejected"


@dataclass(frozen=True)
class LegacyAssemblyEntry:
    """One deterministic assembly outcome."""

    source_kind: LegacyPlanKind
    source_hash: str
    migration_status: LegacyMigrationStatus
    disposition: LegacyAssemblyDisposition
    candidate_ids: Tuple[str, ...]
    selected_candidate_ids: Tuple[str, ...]
    reasons: Tuple[str, ...]

    def validate(self) -> None:
        if len(self.source_hash) != 64:
            raise ValueError("legacy assembly source hash is invalid")
        if len(self.candidate_ids) != len(set(self.candidate_ids)):
            raise ValueError("legacy assembly entry contains duplicate candidates")
        if not set(self.selected_candidate_ids).issubset(set(self.candidate_ids)):
            raise ValueError("legacy assembly selects an unreported candidate")
        if self.disposition == LegacyAssemblyDisposition.ACCEPTED and self.reasons:
            raise ValueError("accepted legacy assembly entry cannot have reasons")
        if self.disposition != LegacyAssemblyDisposition.ACCEPTED and not self.reasons:
            raise ValueError("non-accepted legacy assembly entry requires reasons")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "source_kind": self.source_kind.value,
            "source_hash": self.source_hash,
            "migration_status": self.migration_status.value,
            "disposition": self.disposition.value,
            "candidate_ids": list(self.candidate_ids),
            "selected_candidate_ids": list(self.selected_candidate_ids),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class LegacyAssemblyReport:
    """Auditable mapping from legacy records to one validated candidate space."""

    base_template_hash: str
    result_template_hash: str
    entries: Tuple[LegacyAssemblyEntry, ...]

    def validate(self) -> None:
        if len(self.base_template_hash) != 64 or len(self.result_template_hash) != 64:
            raise ValueError("legacy assembly template hashes are invalid")
        identities = tuple(
            (entry.source_kind, entry.source_hash) for entry in self.entries
        )
        if len(identities) != len(set(identities)):
            raise ValueError("legacy assembly report contains duplicate sources")
        for entry in self.entries:
            entry.validate()

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": "boundflow.plan-ir-legacy-assembly/v1",
            "base_template_hash": self.base_template_hash,
            "result_template_hash": self.result_template_hash,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    def stable_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LegacyAssemblyResult:
    """Validated consolidated template plus its deterministic report."""

    template: PlanTemplate
    report: LegacyAssemblyReport


def assemble_legacy_migrations(
    base_template: PlanTemplate,
    *,
    bound_module: BFBoundModule,
    migrations: Tuple[LegacyPlanMigration, ...],
) -> LegacyAssemblyResult:
    """Append valid migration groups atomically; classify every rejected group."""

    base_template.validate(bound_module=bound_module)
    for migration in migrations:
        migration.validate()
    ordered = tuple(
        sorted(
            migrations,
            key=lambda item: (item.source_kind.value, item.source_hash),
        )
    )
    source_identities = tuple(
        (migration.source_kind, migration.source_hash) for migration in ordered
    )
    if len(source_identities) != len(set(source_identities)):
        raise ValueError("legacy migration batch contains duplicate source records")

    current = base_template
    entries: list[LegacyAssemblyEntry] = []
    for migration in ordered:
        candidate_ids = tuple(
            candidate.candidate_id for candidate in migration.all_candidates()
        )
        if migration.status == LegacyMigrationStatus.UNSUPPORTED:
            entries.append(
                LegacyAssemblyEntry(
                    source_kind=migration.source_kind,
                    source_hash=migration.source_hash,
                    migration_status=migration.status,
                    disposition=LegacyAssemblyDisposition.CLASSIFIED_UNSUPPORTED,
                    candidate_ids=(),
                    selected_candidate_ids=(),
                    reasons=tuple(
                        f"{issue.field}:{issue.reason}" for issue in migration.issues
                    ),
                )
            )
            continue
        duplicates = tuple(sorted(set(candidate_ids) & set(current.candidate_map())))
        if duplicates:
            entries.append(
                _rejected_entry(
                    migration,
                    candidate_ids=candidate_ids,
                    reason="duplicate_candidate_ids:" + ",".join(duplicates),
                )
            )
            continue
        proposed = replace(
            current,
            representation_candidates=(
                *current.representation_candidates,
                *migration.representation_candidates,
            ),
            materialization_candidates=(
                *current.materialization_candidates,
                *migration.materialization_candidates,
            ),
            backend_candidates=(
                *current.backend_candidates,
                *migration.backend_candidates,
            ),
            batch_candidates=(
                *current.batch_candidates,
                *migration.batch_candidates,
            ),
            storage_candidates=(
                *current.storage_candidates,
                *migration.storage_candidates,
            ),
            provenance=(
                *current.provenance,
                PlanProvenance(
                    key=f"legacy_migration_{len(entries):04d}",
                    value=(
                        f"{migration.source_kind.value}:{migration.source_hash}:"
                        f"{migration.status.value}"
                    ),
                ),
            ),
        )
        try:
            proposed.validate(bound_module=bound_module)
        except ValueError as error:
            entries.append(
                _rejected_entry(
                    migration,
                    candidate_ids=candidate_ids,
                    reason=f"template_validation_failed:{error}",
                )
            )
            continue
        current = proposed
        entries.append(
            LegacyAssemblyEntry(
                source_kind=migration.source_kind,
                source_hash=migration.source_hash,
                migration_status=migration.status,
                disposition=LegacyAssemblyDisposition.ACCEPTED,
                candidate_ids=candidate_ids,
                selected_candidate_ids=migration.selected_candidate_ids,
                reasons=(),
            )
        )

    report = LegacyAssemblyReport(
        base_template_hash=base_template.stable_hash(bound_module=bound_module),
        result_template_hash=current.stable_hash(bound_module=bound_module),
        entries=tuple(entries),
    )
    report.validate()
    return LegacyAssemblyResult(template=current, report=report)


def _rejected_entry(
    migration: LegacyPlanMigration,
    *,
    candidate_ids: Tuple[str, ...],
    reason: str,
) -> LegacyAssemblyEntry:
    return LegacyAssemblyEntry(
        source_kind=migration.source_kind,
        source_hash=migration.source_hash,
        migration_status=migration.status,
        disposition=LegacyAssemblyDisposition.REJECTED,
        candidate_ids=candidate_ids,
        selected_candidate_ids=migration.selected_candidate_ids,
        reasons=(reason,),
    )
