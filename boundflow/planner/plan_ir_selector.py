"""Deterministic reference selector for Plan IR v1 instances."""

# Exhaustive cross-axis reference search is intentionally centralized here.
# pylint: disable=too-many-instance-attributes,too-many-locals,too-many-nested-blocks,too-many-branches,too-many-statements,missing-function-docstring

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
from typing import Iterable, Iterator, Optional, Sequence, Tuple

from ..ir.bound import BFBoundModule
from ..ir.plan import (
    BackendCandidate,
    BackendDecision,
    BatchCandidate,
    BatchDecision,
    MaterializationCandidate,
    MaterializationDecision,
    PlanCandidate,
    PlanCost,
    PlanInstance,
    PlanProvenance,
    PlanTemplate,
    RegionCandidate,
    RegionDecision,
    RejectedCandidate,
    RepresentationCandidate,
    RepresentationDecision,
    StateCandidate,
    StateAction,
    StateDecision,
    StateValidity,
    StorageCandidate,
    StorageDecision,
)


@dataclass(frozen=True)
class PlanSelectionFailure:
    """Auditable count of one reason candidate combinations were rejected."""

    reason: str
    count: int


class NoFeasiblePlanError(ValueError):
    """Raised only after the bounded reference search exhausts legal choices."""

    def __init__(self, failures: Tuple[PlanSelectionFailure, ...]) -> None:
        self.failures = failures
        summary = ", ".join(f"{failure.reason}={failure.count}" for failure in failures)
        super().__init__(f"no feasible PlanInstance: {summary}")


@dataclass(frozen=True)
class PlanSelectionContext:
    """Query-distribution and compile-cache facts used by adaptive selection."""

    query_distribution_id: str = "single-query"
    expected_query_count: int = 1
    cached_artifact_keys: Tuple[str, ...] = ()
    max_domain_batch_size: Optional[int] = None
    max_spec_batch_size: Optional[int] = None
    max_sample_batch_size: Optional[int] = None
    required_storage_candidate_id: Optional[str] = None

    def validate(self) -> None:
        if not self.query_distribution_id:
            raise ValueError("query distribution ID must be non-empty")
        if self.expected_query_count <= 0:
            raise ValueError("expected query count must be positive")
        for name in (
            "max_domain_batch_size",
            "max_spec_batch_size",
            "max_sample_batch_size",
        ):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"selection {name} must be positive when present")
        if self.required_storage_candidate_id == "":
            raise ValueError("required storage candidate ID must be non-empty")
        if (
            any(not key for key in self.cached_artifact_keys)
            or len(self.cached_artifact_keys) != len(set(self.cached_artifact_keys))
            or self.cached_artifact_keys != tuple(sorted(self.cached_artifact_keys))
        ):
            raise ValueError(
                "cached artifact keys must be sorted, unique, and non-empty"
            )


@dataclass(frozen=True)
class _Selection:
    regions: Tuple[RegionCandidate, ...]
    representations: Tuple[RepresentationCandidate, ...]
    transitions: Tuple[MaterializationCandidate, ...]
    backends: Tuple[BackendCandidate, ...]
    batch: BatchCandidate
    storage: StorageCandidate
    states: Tuple[StateCandidate, ...]
    cost: PlanCost

    @property
    def candidate_ids(self) -> Tuple[str, ...]:
        return (
            *(candidate.candidate_id for candidate in self.regions),
            *(candidate.candidate_id for candidate in self.representations),
            *(candidate.candidate_id for candidate in self.transitions),
            *(candidate.candidate_id for candidate in self.backends),
            self.batch.candidate_id,
            self.storage.candidate_id,
            *(candidate.candidate_id for candidate in self.states),
        )

    def score(
        self, context: PlanSelectionContext
    ) -> tuple[float, int, float, Tuple[str, ...]]:
        """Rank by amortized latency under exact compile-cache facts."""

        cached_keys = set(context.cached_artifact_keys)
        candidates: tuple[PlanCandidate, ...] = (
            *self.regions,
            *self.representations,
            *self.transitions,
            *self.backends,
            self.batch,
            self.storage,
            *self.states,
        )
        uncached_compile_ms = sum(
            candidate.cost.compile_cost_ms
            for candidate in candidates
            if not (
                isinstance(candidate, BackendCandidate)
                and candidate.compiled_artifact_key in cached_keys
            )
        )
        amortized_setup_ms = (uncached_compile_ms + self.cost.setup_cost_ms) / float(
            context.expected_query_count
        )
        return (
            self.cost.predicted_latency_ms + amortized_setup_ms,
            self.cost.predicted_peak_bytes,
            uncached_compile_ms,
            self.candidate_ids,
        )


def select_plan_instance(  # pylint: disable=too-many-arguments
    template: PlanTemplate,
    *,
    bound_module: BFBoundModule,
    query_bucket_id: str,
    available_memory_bytes: int,
    memory_budget_bytes: int,
    deadline_us: Optional[int] = None,
    state_validities: Tuple[StateValidity, ...] = (),
    selection_context: Optional[PlanSelectionContext] = None,
    max_evaluated_combinations: int = 100_000,
) -> PlanInstance:
    """Select the lowest-latency feasible, fully verified PlanInstance."""

    template.validate(bound_module=bound_module)
    if not query_bucket_id:
        raise ValueError("query_bucket_id must be non-empty")
    if available_memory_bytes <= 0 or memory_budget_bytes <= 0:
        raise ValueError("selection memory limits must be positive")
    if deadline_us is not None and deadline_us <= 0:
        raise ValueError("deadline_us must be positive when present")
    if max_evaluated_combinations <= 0:
        raise ValueError("max_evaluated_combinations must be positive")
    context = selection_context or PlanSelectionContext()
    context.validate()
    for validity in state_validities:
        validity.validate()
    if len({validity.state_id for validity in state_validities}) != len(
        state_validities
    ):
        raise ValueError("query-time state validities must have unique state IDs")

    effective_budget = min(
        available_memory_bytes,
        memory_budget_bytes,
        template.hardware.total_memory_bytes,
    )
    failures: dict[str, int] = {}
    feasible: list[_Selection] = []
    evaluated = 0
    state_groups = _state_candidate_groups(
        template,
        state_validities=state_validities,
    )
    if any(not group for group in state_groups):
        raise NoFeasiblePlanError(
            (PlanSelectionFailure("state_without_valid_candidate", 1),)
        )
    for partition in _exact_region_partitions(template, bound_module=bound_module):
        representation_groups = tuple(
            tuple(
                candidate
                for candidate in template.representation_candidates
                if candidate.region_id == region.region_id and candidate.static_legal
            )
            for region in partition
        )
        if any(not group for group in representation_groups):
            _increment(failures, "region_without_legal_representation")
            continue
        for representations in _storage_compatible_representation_products(
            representation_groups,
            storage_candidates=template.storage_candidates,
        ):
            transition_ids = {
                candidate_id
                for representation in representations
                for candidate_id in representation.required_transition_candidate_ids
            }
            transitions = tuple(
                candidate
                for candidate in template.materialization_candidates
                if candidate.candidate_id in transition_ids
            )
            if len(transitions) != len(transition_ids) or any(
                not transition.static_legal for transition in transitions
            ):
                _increment(failures, "required_transition_illegal_or_missing")
                continue
            backend_groups = tuple(
                tuple(
                    backend
                    for backend in template.backend_candidates
                    if backend.region_id == region.region_id
                    and backend.static_legal
                    and representation.candidate_id
                    in backend.compatible_representation_candidate_ids
                )
                for region, representation in zip(partition, representations)
            )
            if any(not group for group in backend_groups):
                _increment(failures, "representation_without_legal_backend")
                continue
            for backends in itertools.product(*backend_groups):
                for batch in template.batch_candidates:
                    if not batch.static_legal:
                        continue
                    if _batch_exceeds_selection_limit(batch, context=context):
                        _increment(failures, "batch_exceeds_runtime_limit")
                        continue
                    if (
                        batch.domain_batch_size > template.workload.domain_batch_size
                        or batch.spec_batch_size > template.workload.spec_batch_size
                        or batch.sample_batch_size > template.workload.sample_batch_size
                    ):
                        _increment(failures, "batch_exceeds_workload_bucket")
                        continue
                    for states in _state_products(state_groups):
                        for storage in template.storage_candidates:
                            evaluated += 1
                            if evaluated > max_evaluated_combinations:
                                raise ValueError(
                                    "Plan IR reference search exceeded "
                                    "max_evaluated_combinations"
                                )
                            if not storage.static_legal:
                                continue
                            if (
                                context.required_storage_candidate_id is not None
                                and storage.candidate_id
                                != context.required_storage_candidate_id
                            ):
                                _increment(failures, "storage_not_required_policy")
                                continue
                            if (
                                batch.candidate_id
                                not in storage.compatible_batch_candidate_ids
                            ):
                                _increment(failures, "storage_batch_incompatible")
                                continue
                            if any(
                                representation.candidate_id
                                not in (storage.compatible_representation_candidate_ids)
                                for representation in representations
                            ):
                                _increment(
                                    failures,
                                    "storage_representation_incompatible",
                                )
                                continue
                            if storage.cost.predicted_peak_bytes > effective_budget:
                                _increment(failures, "memory_budget_exceeded")
                                continue
                            selected_candidates: tuple[PlanCandidate, ...] = (
                                *partition,
                                *representations,
                                *transitions,
                                *backends,
                                batch,
                                storage,
                                *states,
                            )
                            cost = _aggregate_cost(
                                selected_candidates,
                                storage_peak=storage.cost.predicted_peak_bytes,
                            )
                            selection = _Selection(
                                regions=partition,
                                representations=representations,
                                transitions=transitions,
                                backends=backends,
                                batch=batch,
                                storage=storage,
                                states=states,
                                cost=cost,
                            )
                            if deadline_us is not None:
                                amortized_latency_ms = float(
                                    selection.score(context)[0]
                                )
                                if amortized_latency_ms * 1_000.0 > deadline_us:
                                    _increment(failures, "deadline_exceeded")
                                    continue
                            feasible.append(selection)
    if not feasible:
        if not failures:
            failures["no_legal_candidate_combination"] = 1
        raise NoFeasiblePlanError(
            tuple(
                PlanSelectionFailure(reason=reason, count=count)
                for reason, count in sorted(failures.items())
            )
        )
    selected = min(feasible, key=lambda choice: choice.score(context))
    return _build_instance(
        selected,
        template=template,
        bound_module=bound_module,
        query_bucket_id=query_bucket_id,
        available_memory_bytes=available_memory_bytes,
        memory_budget_bytes=memory_budget_bytes,
        deadline_us=deadline_us,
        state_validities=state_validities,
        selection_context=context,
        evaluated_combinations=evaluated,
    )


def _exact_region_partitions(
    template: PlanTemplate, *, bound_module: BFBoundModule
) -> Iterator[Tuple[RegionCandidate, ...]]:
    op_order = tuple(op.op_id for op in bound_module.graph.ops)
    op_position = {op_id: index for index, op_id in enumerate(op_order)}
    candidates_by_op: dict[str, list[RegionCandidate]] = {
        op_id: [] for op_id in op_order
    }
    for candidate in template.region_candidates:
        if not candidate.op_ids:
            continue
        first = min(candidate.op_ids, key=op_position.__getitem__)
        candidates_by_op[first].append(candidate)
    for candidates in candidates_by_op.values():
        candidates.sort(key=lambda candidate: candidate.candidate_id)

    def visit(
        covered: frozenset[str],
        chosen: Tuple[RegionCandidate, ...],
    ) -> Iterator[Tuple[RegionCandidate, ...]]:
        if len(covered) == len(op_order):
            yield chosen
            return
        next_op = next((op_id for op_id in op_order if op_id not in covered), None)
        if next_op is None:
            return
        for candidate in candidates_by_op.get(next_op, []):
            candidate_ops = frozenset(candidate.op_ids)
            if candidate_ops & covered:
                continue
            yield from visit(covered | candidate_ops, (*chosen, candidate))

    yield from visit(frozenset(), ())


def _storage_compatible_representation_products(
    groups: Tuple[Tuple[RepresentationCandidate, ...], ...],
    *,
    storage_candidates: Tuple[StorageCandidate, ...],
) -> Iterator[Tuple[RepresentationCandidate, ...]]:
    """Enumerate only prefixes that at least one legal storage can complete.

    This is semantically identical to the later whole-plan compatibility check,
    but avoids exponential mixed-policy enumeration when storage candidates
    encode globally coherent representation families.
    """

    compatible_sets = tuple(
        frozenset(candidate.compatible_representation_candidate_ids)
        for candidate in storage_candidates
        if candidate.static_legal
    )
    if not compatible_sets:
        return

    def visit(
        index: int,
        selected: Tuple[RepresentationCandidate, ...],
        possible_storage_sets: Tuple[frozenset[str], ...],
    ) -> Iterator[Tuple[RepresentationCandidate, ...]]:
        if index == len(groups):
            yield selected
            return
        for candidate in groups[index]:
            next_storage_sets = tuple(
                compatible
                for compatible in possible_storage_sets
                if candidate.candidate_id in compatible
            )
            if not next_storage_sets:
                continue
            yield from visit(
                index + 1,
                (*selected, candidate),
                next_storage_sets,
            )

    yield from visit(0, (), compatible_sets)


def _state_candidate_groups(
    template: PlanTemplate,
    *,
    state_validities: Tuple[StateValidity, ...],
) -> Tuple[Tuple[StateCandidate, ...], ...]:
    validity_by_state = {validity.state_id: validity for validity in state_validities}
    grouped: dict[str, list[StateCandidate]] = {}
    for candidate in template.state_candidates:
        grouped.setdefault(candidate.state_id, [])
        if not candidate.static_legal:
            continue
        if candidate.action == StateAction.REUSE:
            validity = validity_by_state.get(candidate.state_id)
            if (
                validity is None
                or not validity.valid
                or validity.source_value_id != candidate.source_value_id
                or validity.state_version != candidate.state_version
            ):
                continue
        grouped[candidate.state_id].append(candidate)
    return tuple(
        tuple(sorted(candidates, key=lambda candidate: candidate.candidate_id))
        for _state_id, candidates in sorted(grouped.items())
    )


def _state_products(
    groups: Tuple[Tuple[StateCandidate, ...], ...],
) -> Iterable[Tuple[StateCandidate, ...]]:
    if not groups:
        return ((),)
    return itertools.product(*groups)


def _aggregate_cost(
    selected: Sequence[PlanCandidate], *, storage_peak: int
) -> PlanCost:
    risks = tuple(
        sorted({risk for candidate in selected for risk in candidate.cost.risk_tags})
    )
    return PlanCost(
        predicted_latency_ms=sum(
            candidate.cost.predicted_latency_ms for candidate in selected
        ),
        predicted_peak_bytes=storage_peak,
        compile_cost_ms=sum(candidate.cost.compile_cost_ms for candidate in selected),
        setup_cost_ms=sum(candidate.cost.setup_cost_ms for candidate in selected),
        confidence=min(candidate.cost.confidence for candidate in selected),
        risk_tags=risks,
    )


def _build_instance(  # pylint: disable=too-many-arguments
    selected: _Selection,
    *,
    template: PlanTemplate,
    bound_module: BFBoundModule,
    query_bucket_id: str,
    available_memory_bytes: int,
    memory_budget_bytes: int,
    deadline_us: Optional[int],
    state_validities: Tuple[StateValidity, ...],
    selection_context: PlanSelectionContext,
    evaluated_combinations: int,
) -> PlanInstance:
    selected_ids = set(selected.candidate_ids)
    rejected = tuple(
        RejectedCandidate(
            candidate_id=candidate.candidate_id,
            reasons=_rejection_reasons(candidate),
        )
        for candidate in template.all_candidates()
        if candidate.candidate_id not in selected_ids
    )
    template_hash = template.stable_hash(bound_module=bound_module)
    identity_payload = "|".join(
        (
            template_hash,
            query_bucket_id,
            str(available_memory_bytes),
            str(memory_budget_bytes),
            str(deadline_us),
            selection_context.query_distribution_id,
            str(selection_context.expected_query_count),
            *selection_context.cached_artifact_keys,
            *(
                (
                    f"max_domain_batch_size={selection_context.max_domain_batch_size}",
                    f"max_spec_batch_size={selection_context.max_spec_batch_size}",
                    f"max_sample_batch_size={selection_context.max_sample_batch_size}",
                )
                if _has_batch_limit(selection_context)
                else ()
            ),
            *(
                (
                    "required_storage_candidate_id="
                    f"{selection_context.required_storage_candidate_id}",
                )
                if selection_context.required_storage_candidate_id is not None
                else ()
            ),
            *(
                "|".join(
                    (
                        validity.state_id,
                        validity.source_value_id,
                        validity.state_version,
                        str(validity.valid),
                        str(validity.invalidation_reason),
                    )
                )
                for validity in state_validities
            ),
            *selected.candidate_ids,
        )
    )
    instance_id = (
        "plan-instance:"
        + hashlib.sha256(identity_payload.encode("utf-8")).hexdigest()[:24]
    )
    instance = PlanInstance(
        instance_id=instance_id,
        template_hash=template_hash,
        query_bucket_id=query_bucket_id,
        available_memory_bytes=available_memory_bytes,
        memory_budget_bytes=memory_budget_bytes,
        deadline_us=deadline_us,
        region_decisions=tuple(
            RegionDecision(candidate.region_id, candidate.candidate_id)
            for candidate in selected.regions
        ),
        representation_decisions=tuple(
            RepresentationDecision(candidate.region_id, candidate.candidate_id)
            for candidate in selected.representations
        ),
        materialization_decisions=tuple(
            MaterializationDecision(candidate.candidate_id)
            for candidate in selected.transitions
        ),
        backend_decisions=tuple(
            BackendDecision(candidate.region_id, candidate.candidate_id)
            for candidate in selected.backends
        ),
        batch_decision=BatchDecision(selected.batch.candidate_id),
        storage_decision=StorageDecision(selected.storage.candidate_id),
        state_decisions=tuple(
            StateDecision(candidate.state_id, candidate.candidate_id)
            for candidate in selected.states
        ),
        state_validities=state_validities,
        rejected_candidates=rejected,
        cost_summary=selected.cost,
        provenance=(
            PlanProvenance(
                "selector",
                "amortized_latency_then_peak_compile_lexical_v2",
            ),
            PlanProvenance("evaluated_combinations", str(evaluated_combinations)),
            PlanProvenance(
                "query_distribution_id",
                selection_context.query_distribution_id,
            ),
            PlanProvenance(
                "expected_query_count",
                str(selection_context.expected_query_count),
            ),
            PlanProvenance(
                "cached_artifact_keys",
                ",".join(selection_context.cached_artifact_keys) or "none",
            ),
            *(
                (
                    PlanProvenance(
                        "max_domain_batch_size",
                        str(selection_context.max_domain_batch_size or "none"),
                    ),
                    PlanProvenance(
                        "max_spec_batch_size",
                        str(selection_context.max_spec_batch_size or "none"),
                    ),
                    PlanProvenance(
                        "max_sample_batch_size",
                        str(selection_context.max_sample_batch_size or "none"),
                    ),
                )
                if _has_batch_limit(selection_context)
                else ()
            ),
            *(
                (
                    PlanProvenance(
                        "required_storage_candidate_id",
                        selection_context.required_storage_candidate_id,
                    ),
                )
                if selection_context.required_storage_candidate_id is not None
                else ()
            ),
            PlanProvenance(
                "amortized_selection_latency_ms",
                format(float(selected.score(selection_context)[0]), ".12g"),
            ),
            PlanProvenance(
                "uncached_compile_cost_ms",
                format(float(selected.score(selection_context)[2]), ".12g"),
            ),
        ),
    )
    instance.validate(template=template, bound_module=bound_module)
    return instance


def _rejection_reasons(candidate: PlanCandidate) -> Tuple[str, ...]:
    static_legal = getattr(candidate, "static_legal", True)
    static_reasons = getattr(candidate, "rejection_reasons", ())
    if not static_legal and static_reasons:
        return tuple(static_reasons)
    if isinstance(candidate, RegionCandidate):
        return ("partition_not_selected",)
    if isinstance(candidate, MaterializationCandidate):
        return ("transition_not_required_by_selected_representation",)
    return ("not_selected_by_reference_selector",)


def _increment(counts: dict[str, int], reason: str) -> None:
    counts[reason] = counts.get(reason, 0) + 1


def _has_batch_limit(context: PlanSelectionContext) -> bool:
    return any(
        value is not None
        for value in (
            context.max_domain_batch_size,
            context.max_spec_batch_size,
            context.max_sample_batch_size,
        )
    )


def _batch_exceeds_selection_limit(
    batch: BatchCandidate, *, context: PlanSelectionContext
) -> bool:
    return any(
        limit is not None and size > limit
        for size, limit in (
            (batch.domain_batch_size, context.max_domain_batch_size),
            (batch.spec_batch_size, context.max_spec_batch_size),
            (batch.sample_batch_size, context.max_sample_batch_size),
        )
    )
