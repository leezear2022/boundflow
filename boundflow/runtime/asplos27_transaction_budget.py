"""Derive a research 10x budget from explicit solver transaction evidence."""

# pylint: disable=missing-function-docstring,too-many-locals,too-many-branches
# pylint: disable=too-many-statements

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from boundflow.runtime.gpu_attribution import canonical_hash
from boundflow.runtime.solver_transaction_observer import (
    host_transaction_span_from_dict,
    summarize_solver_transactions,
)

TRANSACTION_BUDGET_SCHEMA_VERSION = "boundflow.asplos27-transaction-budget/v1"


@dataclass(frozen=True)
class OptimizationAxisPolicy:
    """One mutually exclusive optimization family and its research target."""

    axis_id: str
    target_speedup: float
    mechanism: str
    exact_categories: tuple[str, ...]
    category_prefixes: tuple[str, ...] = ()
    evidence_note: str = "research target; not measured"

    def validate(self) -> None:
        if not self.axis_id or not self.mechanism or not self.evidence_note:
            raise ValueError("transaction budget axis identity differs")
        if not math.isfinite(self.target_speedup) or self.target_speedup < 1.0:
            raise ValueError("transaction budget target speedup differs")
        if not self.exact_categories and not self.category_prefixes:
            raise ValueError("transaction budget axis categories are empty")
        if any(not value for value in self.exact_categories + self.category_prefixes):
            raise ValueError("transaction budget category identity is empty")

    def matches(self, category: str) -> bool:
        return category in self.exact_categories or any(
            category.startswith(prefix) for prefix in self.category_prefixes
        )

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "axis_id": self.axis_id,
            "target_speedup": self.target_speedup,
            "mechanism": self.mechanism,
            "exact_categories": list(self.exact_categories),
            "category_prefixes": list(self.category_prefixes),
            "evidence_note": self.evidence_note,
            "target_validated": False,
            "performance_claimed": False,
        }


DEFAULT_AXIS_POLICIES = (
    OptimizationAxisPolicy(
        axis_id="O1_coarse_bound_regions",
        target_speedup=16.0,
        mechanism=(
            "CIBC horizontal lower/upper fusion plus coarse CROWN forward/backward "
            "TIR regions"
        ),
        exact_categories=("bound_core",),
        category_prefixes=("bound_compute:",),
        evidence_note=(
            "stretch target; local anchors are CIBC 12.795x and B4-B2 4.898x at "
            "different scopes"
        ),
    ),
    OptimizationAxisPolicy(
        axis_id="O2_structured_state_and_batching",
        target_speedup=8.0,
        mechanism=(
            "structured alpha/beta/split/history ownership, minimal saved state, "
            "domain/spec batching and fused prepare/commit"
        ),
        exact_categories=(
            "bound_prepare",
            "bound_postprocess",
            "domain_preprocess",
            "domain_solve",
            "domain_postprocess",
            "bab_bootstrap",
            "bab_scope",
            "spec_handoff",
        ),
    ),
    OptimizationAxisPolicy(
        axis_id="O3_compiled_admission_and_prepared_runtime",
        target_speedup=12.0,
        mechanism=(
            "canonical compile/prepare entry, cached model/spec lowering, static "
            "receipts and O(1) warm-run guards"
        ),
        exact_categories=(
            "frontend_setup",
            "constraint_import",
            "environment_setup",
            "model_prepare",
            "spec_prepare",
            "incomplete_verification",
        ),
    ),
    OptimizationAxisPolicy(
        axis_id="O4_memory_lifetime_and_reclamation",
        target_speedup=20.0,
        mechanism=(
            "persistent arena, bounded lifetimes, no repeated hot-path GC or CUDA "
            "cache flush"
        ),
        exact_categories=("host_garbage_collection", "device_cache_release"),
    ),
    OptimizationAxisPolicy(
        axis_id="O5_result_and_termination",
        target_speedup=4.0,
        mechanism="typed compact result publication and termination receipt",
        exact_categories=("result_publish", "solver_termination"),
    ),
)


def _strict_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    return value


def _profile_category_ns(record: Mapping[str, Any]) -> Mapping[str, int]:
    if record.get("mode") != "profile":
        raise ValueError("transaction budget only accepts profile workers")
    if record.get("performance_claimed") is not False:
        raise ValueError("transaction budget source cannot claim performance")
    scope_ns = _strict_int(record.get("scope_ns"), "transaction budget scope")
    calls = record.get("compute_calls")
    transactions = record.get("transactions")
    if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes)):
        raise TypeError("transaction budget compute calls differ")
    if not isinstance(transactions, Sequence) or isinstance(transactions, (str, bytes)):
        raise TypeError("transaction budget transactions differ")
    spans = tuple(
        host_transaction_span_from_dict(transaction)
        for transaction in transactions
        if isinstance(transaction, Mapping)
    )
    if len(spans) != len(transactions):
        raise TypeError("transaction budget transaction row differs")
    summary = summarize_solver_transactions(
        spans,
        compute_calls=[call for call in calls if isinstance(call, Mapping)],
        scope_ns=scope_ns,
    )
    if len(calls) != summary["compute_call_count"]:
        raise TypeError("transaction budget compute call row differs")
    if record.get("transaction_summary") != summary:
        raise ValueError("transaction budget semantic summary differs")
    if summary["mechanism_admitted"] is not True:
        raise ValueError("transaction budget mechanism attribution is not admitted")
    category_ns = summary["category_ns"]
    if not isinstance(category_ns, Mapping):
        raise TypeError("transaction budget category payload differs")
    normalized: dict[str, int] = {}
    for category, duration in category_ns.items():
        if not isinstance(category, str):
            raise TypeError("transaction budget category identity differs")
        normalized[category] = _strict_int(duration, "transaction category duration")
    if sum(normalized.values()) != scope_ns:
        raise ValueError("transaction budget categories do not close")
    return normalized


def _classify_category(
    category: str, policies: Sequence[OptimizationAxisPolicy]
) -> str:
    if category.startswith("mechanism_unresolved:"):
        return "U_unresolved_immutable"
    matches = [policy.axis_id for policy in policies if policy.matches(category)]
    if len(matches) != 1:
        raise ValueError(
            f"transaction budget category has {len(matches)} owners: {category}"
        )
    return matches[0]


def derive_transaction_budgets(
    records: Sequence[Mapping[str, Any]],
    *,
    repeats: int,
    target_speedup: float = 10.0,
    integration_overhead_share: float = 0.0,
    policies: Sequence[OptimizationAxisPolicy] = DEFAULT_AXIS_POLICIES,
) -> dict[str, object]:
    """Pool fresh profiles by workload and derive a no-claim research budget."""

    if repeats < 1:
        raise ValueError("transaction budget repeat count differs")
    if not math.isfinite(target_speedup) or target_speedup <= 1.0:
        raise ValueError("transaction budget headline target differs")
    if (
        not math.isfinite(integration_overhead_share)
        or not 0.0 <= integration_overhead_share < 1.0
    ):
        raise ValueError("transaction budget integration overhead differs")
    if not policies:
        raise ValueError("transaction budget policies are empty")
    for policy in policies:
        policy.validate()
    axis_ids = [policy.axis_id for policy in policies]
    if len(axis_ids) != len(set(axis_ids)):
        raise ValueError("transaction budget axis IDs duplicate")
    profiles = [record for record in records if record.get("mode") == "profile"]
    workloads = sorted({str(record.get("workload_id")) for record in profiles})
    if not workloads or len(profiles) != len(workloads) * repeats:
        raise ValueError("transaction budget profile matrix differs")
    rows: list[dict[str, object]] = []
    all_feasible = True
    for workload_id in workloads:
        workload_records = [
            record for record in profiles if record.get("workload_id") == workload_id
        ]
        if len(workload_records) != repeats:
            raise ValueError("transaction budget workload repeats differ")
        pooled_scope_ns = 0
        pooled_category_ns: dict[str, int] = {}
        summary_hashes: list[str] = []
        for record in workload_records:
            category_ns = _profile_category_ns(record)
            scope_ns = _strict_int(record.get("scope_ns"), "transaction budget scope")
            pooled_scope_ns += scope_ns
            for category, duration in category_ns.items():
                pooled_category_ns[category] = (
                    pooled_category_ns.get(category, 0) + duration
                )
            summary = record["transaction_summary"]
            if not isinstance(summary, Mapping) or not isinstance(
                summary.get("summary_hash"), str
            ):
                raise TypeError("transaction budget source summary hash differs")
            summary_hashes.append(str(summary["summary_hash"]))
        axis_ns = {policy.axis_id: 0 for policy in policies}
        axis_ns["U_unresolved_immutable"] = 0
        for category, duration in pooled_category_ns.items():
            axis_ns[_classify_category(category, policies)] += duration
        if sum(axis_ns.values()) != pooled_scope_ns:
            raise ValueError("transaction budget axes do not close")
        axis_rows: list[dict[str, object]] = []
        projected_fraction = integration_overhead_share
        for policy in policies:
            share = axis_ns[policy.axis_id] / pooled_scope_ns
            contribution = share / policy.target_speedup
            projected_fraction += contribution
            axis_rows.append(
                {
                    **policy.to_dict(),
                    "pooled_ns": axis_ns[policy.axis_id],
                    "baseline_share": share,
                    "projected_residual_share": contribution,
                }
            )
        unresolved_share = axis_ns["U_unresolved_immutable"] / pooled_scope_ns
        projected_fraction += unresolved_share
        projected_speedup = 1.0 / projected_fraction
        denominator = (
            1.0 / target_speedup - unresolved_share - integration_overhead_share
        )
        required_uniform_resolved_speedup = (
            None if denominator <= 0.0 else (1.0 - unresolved_share) / denominator
        )
        feasible = projected_speedup >= target_speedup
        all_feasible = all_feasible and feasible
        row: dict[str, object] = {
            "workload_id": workload_id,
            "repeat_count": len(workload_records),
            "pooled_scope_ns": pooled_scope_ns,
            "source_transaction_summary_hashes": summary_hashes,
            "category_ns": dict(sorted(pooled_category_ns.items())),
            "axes": axis_rows,
            "unresolved_ns": axis_ns["U_unresolved_immutable"],
            "unresolved_share": unresolved_share,
            "integration_overhead_share": integration_overhead_share,
            "target_speedup": target_speedup,
            "required_uniform_resolved_speedup": required_uniform_resolved_speedup,
            "projected_runtime_fraction": projected_fraction,
            "projected_speedup_hypothesis": projected_speedup,
            "tenx_feasible_hypothesis": feasible,
            "all_axis_targets_validated": False,
            "performance_claimed": False,
        }
        row["budget_hash"] = canonical_hash(row)
        rows.append(row)
    report: dict[str, object] = {
        "schema_version": TRANSACTION_BUDGET_SCHEMA_VERSION,
        "status": (
            "s0-transaction-budget-research-route-open"
            if all_feasible
            else "s0-transaction-budget-research-route-closed"
        ),
        "target_speedup": target_speedup,
        "integration_overhead_share": integration_overhead_share,
        "workload_count": len(rows),
        "profile_count": len(profiles),
        "policies": [policy.to_dict() for policy in policies],
        "workloads": rows,
        "all_workloads_tenx_feasible_hypothesis": all_feasible,
        "s1_implementation_open": all_feasible,
        "s1_performance_gate_open": False,
        "performance_claimed": False,
    }
    report["report_hash"] = canonical_hash(report)
    return report
