"""ASPLOS'27 full-stack attribution admission and 10x budget contracts."""

# pylint: disable=missing-function-docstring,too-many-locals,too-many-branches
# pylint: disable=too-many-statements

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Mapping, Sequence

from boundflow.runtime.gpu_attribution import canonical_hash

ASPLOS27_TENX_BUDGET_SCHEMA_VERSION = "boundflow.asplos27-tenx-budget/v1"


class ClaimMode(str, Enum):
    """Paper claim scopes that must never be mixed in one speedup."""

    FIXED_TRAJECTORY_SYSTEMS = "fixed_trajectory_systems"
    SOLVED_QUERY_TTV = "solved_query_ttv"


class EvidenceScope(str, Enum):
    """Physical scope at which one mechanism has actually been measured."""

    LOCAL_OPERATOR = "local_operator"
    STANDALONE_GRAPH = "standalone_graph"
    SAME_SOLVER_REGION = "same_solver_region"
    COMPLETE_QUERY = "complete_query"
    FIXED_PREFIX = "fixed_prefix"
    HYPOTHESIS = "hypothesis"


@dataclass(frozen=True)
class BudgetBucket:
    """One exclusive B0 critical-path bucket and its cumulative target."""

    bucket_id: str
    baseline_share: float
    target_speedup: float
    mechanism: str
    evidence_scope: EvidenceScope

    def validate(self) -> None:
        if not self.bucket_id:
            raise ValueError("10x budget bucket ID must be non-empty")
        if not 0.0 <= self.baseline_share <= 1.0:
            raise ValueError("10x budget bucket share is outside [0, 1]")
        if not math.isfinite(self.target_speedup) or self.target_speedup < 1.0:
            raise ValueError("10x budget bucket speedup must be finite and >= 1")
        if not self.mechanism:
            raise ValueError("10x budget bucket mechanism must be non-empty")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "bucket_id": self.bucket_id,
            "baseline_share": self.baseline_share,
            "target_speedup": self.target_speedup,
            "mechanism": self.mechanism,
            "evidence_scope": self.evidence_scope.value,
        }


@dataclass(frozen=True)
class TenXBudget:  # pylint: disable=too-many-instance-attributes
    """Amdahl budget tied to one exact B0 scope and semantic coverage gate."""

    scope_id: str
    claim_mode: ClaimMode
    target_speedup: float
    integration_overhead_share: float
    semantic_coverage_share: float
    semantic_unclassified_share: float
    fixed_trajectory_complete: bool
    solved_query_complete: bool
    buckets: tuple[BudgetBucket, ...]

    def validate(self) -> None:
        if not self.scope_id:
            raise ValueError("10x budget scope ID must be non-empty")
        if not math.isfinite(self.target_speedup) or self.target_speedup <= 1.0:
            raise ValueError("10x target speedup must be finite and > 1")
        if not 0.0 <= self.integration_overhead_share < 1.0:
            raise ValueError("10x integration overhead is outside [0, 1)")
        if not 0.0 <= self.semantic_coverage_share <= 1.0:
            raise ValueError("10x semantic coverage is outside [0, 1]")
        if not 0.0 <= self.semantic_unclassified_share <= 1.0:
            raise ValueError("10x semantic unclassified share is outside [0, 1]")
        if not math.isclose(
            self.semantic_coverage_share + self.semantic_unclassified_share,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("10x semantic coverage does not close")
        if not self.buckets:
            raise ValueError("10x budget requires at least one exclusive bucket")
        bucket_ids = [bucket.bucket_id for bucket in self.buckets]
        if len(bucket_ids) != len(set(bucket_ids)):
            raise ValueError("10x budget bucket IDs duplicate")
        for bucket in self.buckets:
            bucket.validate()
        if not math.isclose(
            sum(bucket.baseline_share for bucket in self.buckets),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("10x exclusive baseline buckets do not close")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "scope_id": self.scope_id,
            "claim_mode": self.claim_mode.value,
            "target_speedup": self.target_speedup,
            "integration_overhead_share": self.integration_overhead_share,
            "semantic_coverage_share": self.semantic_coverage_share,
            "semantic_unclassified_share": self.semantic_unclassified_share,
            "fixed_trajectory_complete": self.fixed_trajectory_complete,
            "solved_query_complete": self.solved_query_complete,
            "buckets": [bucket.to_dict() for bucket in self.buckets],
        }


@dataclass(frozen=True)
class DirectCumulativeObservation:  # pylint: disable=too-many-instance-attributes
    """One already measured direct ratio; ratios from different scopes stay separate."""

    observation_id: str
    scope_id: str
    evidence_scope: EvidenceScope
    baseline_id: str
    candidate_id: str
    baseline_over_candidate: float
    source_digest: str
    semantic_passed: bool
    performance_claimed: bool = False

    def validate(self) -> None:
        for label, value in (
            ("observation ID", self.observation_id),
            ("scope ID", self.scope_id),
            ("baseline ID", self.baseline_id),
            ("candidate ID", self.candidate_id),
            ("source digest", self.source_digest),
        ):
            if not value:
                raise ValueError(f"direct cumulative {label} must be non-empty")
        if (
            not math.isfinite(self.baseline_over_candidate)
            or self.baseline_over_candidate <= 0.0
        ):
            raise ValueError("direct cumulative ratio must be finite and positive")
        if self.performance_claimed:
            raise ValueError(
                "S0 direct cumulative observation cannot claim performance"
            )

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "observation_id": self.observation_id,
            "scope_id": self.scope_id,
            "evidence_scope": self.evidence_scope.value,
            "baseline_id": self.baseline_id,
            "candidate_id": self.candidate_id,
            "baseline_over_candidate": self.baseline_over_candidate,
            "source_digest": self.source_digest,
            "semantic_passed": self.semantic_passed,
            "performance_claimed": False,
        }


def _float_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def summarize_tenx_budget(
    budget: TenXBudget,
    *,
    minimum_semantic_coverage: float = 0.97,
    maximum_semantic_unclassified: float = 0.03,
) -> dict[str, object]:
    """Evaluate admission separately from the Amdahl feasibility hypothesis."""

    budget.validate()
    if not 0.0 <= minimum_semantic_coverage <= 1.0:
        raise ValueError("minimum semantic coverage is outside [0, 1]")
    if not 0.0 <= maximum_semantic_unclassified <= 1.0:
        raise ValueError("maximum semantic unclassified share is outside [0, 1]")
    coverage_passed = budget.semantic_coverage_share >= minimum_semantic_coverage
    unclassified_passed = (
        budget.semantic_unclassified_share <= maximum_semantic_unclassified
    )
    scope_complete = (
        budget.fixed_trajectory_complete
        if budget.claim_mode == ClaimMode.FIXED_TRAJECTORY_SYSTEMS
        else budget.solved_query_complete
    )
    attribution_admitted = coverage_passed and unclassified_passed and scope_complete
    projected_fraction = budget.integration_overhead_share + sum(
        bucket.baseline_share / bucket.target_speedup for bucket in budget.buckets
    )
    projected_speedup = (
        math.inf if projected_fraction == 0.0 else 1.0 / projected_fraction
    )
    target_fraction = 1.0 / budget.target_speedup
    tenx_feasible_hypothesis = projected_fraction <= target_fraction
    immutable_share = sum(
        bucket.baseline_share
        for bucket in budget.buckets
        if math.isclose(bucket.target_speedup, 1.0, rel_tol=0.0, abs_tol=1e-12)
    )
    optimizable_share = 1.0 - immutable_share
    denominator = target_fraction - immutable_share - budget.integration_overhead_share
    if denominator <= 0.0:
        required_uniform_speedup: float | None = None
    else:
        required_uniform_speedup = optimizable_share / denominator
    result: dict[str, object] = {
        "schema_version": ASPLOS27_TENX_BUDGET_SCHEMA_VERSION,
        "scope_id": budget.scope_id,
        "claim_mode": budget.claim_mode.value,
        "target_speedup": budget.target_speedup,
        "target_fraction": target_fraction,
        "semantic_coverage_share": budget.semantic_coverage_share,
        "semantic_unclassified_share": budget.semantic_unclassified_share,
        "coverage_gate": minimum_semantic_coverage,
        "unclassified_gate": maximum_semantic_unclassified,
        "coverage_passed": coverage_passed,
        "unclassified_passed": unclassified_passed,
        "scope_complete": scope_complete,
        "attribution_admitted": attribution_admitted,
        "integration_overhead_share": budget.integration_overhead_share,
        "immutable_share": immutable_share,
        "optimizable_share": optimizable_share,
        "projected_runtime_fraction": projected_fraction,
        "projected_speedup": projected_speedup,
        "required_uniform_speedup": required_uniform_speedup,
        "tenx_feasible_hypothesis": tenx_feasible_hypothesis,
        "budget_buckets": [bucket.to_dict() for bucket in budget.buckets],
        "performance_claimed": False,
    }
    result["budget_hash"] = canonical_hash(result)
    return result


def derive_fsg1_diagnostic_budgets(
    closure: Mapping[str, Any],
    *,
    operator_target_speedup: float,
    target_speedup: float = 10.0,
) -> dict[str, object]:
    """Convert the frozen FSG1 B0 prefix into conservative S0 diagnostics."""

    runs_value = closure.get("runs")
    if not isinstance(runs_value, Sequence) or isinstance(runs_value, (str, bytes)):
        raise TypeError("FSG1 closure runs must be a sequence")
    budgets: list[dict[str, object]] = []
    for run_value in runs_value:
        run = _float_mapping(run_value, "FSG1 closure run")
        if run.get("configuration_id") != "B0":
            raise ValueError("S0 diagnostic only accepts official B0 runs")
        if run.get("performance_claimed") is not False:
            raise ValueError("FSG1 diagnostic source cannot claim performance")
        scope_ns = int(run["scope_ns"])
        if scope_ns <= 0:
            raise ValueError("FSG1 diagnostic scope must be positive")
        layer_ns = _float_mapping(run["layer_ns"], "FSG1 layer_ns")
        phase_ns = _float_mapping(run["phase_ns"], "FSG1 phase_ns")
        operator_share = float(layer_ns["operator_execution"]) / scope_ns
        solver_share = 1.0 - operator_share
        semantic_unclassified = float(phase_ns["unclassified"]) / scope_ns
        budget = TenXBudget(
            scope_id=f"fsg1:{run['run_id']}:fixed-16-iteration-prefix",
            claim_mode=ClaimMode.FIXED_TRAJECTORY_SYSTEMS,
            target_speedup=target_speedup,
            integration_overhead_share=0.0,
            semantic_coverage_share=1.0 - semantic_unclassified,
            semantic_unclassified_share=semantic_unclassified,
            fixed_trajectory_complete=True,
            solved_query_complete=False,
            buckets=(
                BudgetBucket(
                    bucket_id="operator_execution",
                    baseline_share=operator_share,
                    target_speedup=operator_target_speedup,
                    mechanism="diagnostic_existing_cibc_operator_ceiling",
                    evidence_scope=EvidenceScope.LOCAL_OPERATOR,
                ),
                BudgetBucket(
                    bucket_id="all_non_operator",
                    baseline_share=solver_share,
                    target_speedup=1.0,
                    mechanism="unchanged_host_solver_runtime",
                    evidence_scope=EvidenceScope.FIXED_PREFIX,
                ),
            ),
        )
        summary = summarize_tenx_budget(budget)
        summary["source_run_hash"] = run["run_hash"]
        summary["operator_infinite_speedup_ceiling"] = (
            math.inf if solver_share == 0.0 else 1.0 / solver_share
        )
        summary["diagnostic_only"] = True
        summary["performance_claimed"] = False
        summary["budget_hash"] = canonical_hash(
            {key: value for key, value in summary.items() if key != "budget_hash"}
        )
        budgets.append(summary)
    report: dict[str, object] = {
        "schema_version": ASPLOS27_TENX_BUDGET_SCHEMA_VERSION,
        "status": (
            "s0-attribution-admitted"
            if budgets and all(item["attribution_admitted"] for item in budgets)
            else "s0-attribution-not-admitted"
        ),
        "source_scope": "official-b0-fixed-16-iteration-prefix",
        "target_speedup": target_speedup,
        "operator_target_speedup": operator_target_speedup,
        "run_count": len(budgets),
        "admitted_run_count": sum(
            item["attribution_admitted"] is True for item in budgets
        ),
        "tenx_feasible_run_count": sum(
            item["tenx_feasible_hypothesis"] is True for item in budgets
        ),
        "runs": budgets,
        "claim_limitations": [
            "fixed_iteration_prefix_not_complete_query",
            "local_operator_speedup_is_hypothesis_not_scope_matched_measurement",
            "no_boundflow_candidate_was_timed_in_fsg1",
            "cross_scope_speedups_must_not_be_multiplied",
        ],
        "performance_claimed": False,
    }
    report["report_hash"] = canonical_hash(report)
    return report


def derive_fsg1_transaction_inventory(
    worker_records: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    """Subdivide FSG1 critical-path ownership into solver transaction contexts.

    This is deliberately not a mechanism profiler. Inter-call gaps receive stable
    neighboring phase context, while remaining unresolved for optimization admission.
    """

    profile_records = [
        record for record in worker_records if record.get("mode") == "profile"
    ]
    if not profile_records:
        raise ValueError("FSG1 transaction inventory requires profile records")
    rows: list[dict[str, object]] = []
    for record in profile_records:
        if record.get("configuration_id") != "B0":
            raise ValueError("FSG1 transaction inventory only accepts B0")
        if record.get("performance_claimed") is not False:
            raise ValueError("FSG1 transaction source cannot claim performance")
        calls_value = record.get("calls")
        if not isinstance(calls_value, Sequence) or isinstance(
            calls_value, (str, bytes)
        ):
            raise TypeError("FSG1 transaction calls must be a sequence")
        calls = [_float_mapping(call, "FSG1 transaction call") for call in calls_value]
        if not calls:
            raise ValueError("FSG1 transaction profile calls are empty")
        scope_ns = int(record["scope_ns"])
        boundaries = sorted(
            {0, scope_ns}
            | {int(call["host_start_ns"]) for call in calls}
            | {int(call["host_end_ns"]) for call in calls}
        )
        first_start = min(int(call["host_start_ns"]) for call in calls)
        last_end = max(int(call["host_end_ns"]) for call in calls)
        category_ns: dict[str, int] = {}
        unresolved_ns = 0
        topology_unclassified_ns = 0
        for start, end in zip(boundaries, boundaries[1:]):
            if end <= start:
                continue
            active = [
                call
                for call in calls
                if int(call["host_start_ns"]) <= start
                and int(call["host_end_ns"]) >= end
            ]
            mechanism_resolved = True
            if active:
                owner = max(
                    active,
                    key=lambda call: (int(call["depth"]), int(call["call_id"])),
                )
                category = f"bound_call:{owner['phase']}:{owner['external_phase']}"
            elif end <= first_start:
                category = "solver_control:setup"
            elif start >= last_end:
                category = "solver_control:termination"
            else:
                previous = [call for call in calls if int(call["host_end_ns"]) <= start]
                following = [
                    call for call in calls if int(call["host_start_ns"]) >= end
                ]
                mechanism_resolved = False
                unresolved_ns += end - start
                if previous and following:
                    left = max(
                        previous,
                        key=lambda call: (
                            int(call["host_end_ns"]),
                            int(call["depth"]),
                            int(call["call_id"]),
                        ),
                    )
                    right = min(
                        following,
                        key=lambda call: (
                            int(call["host_start_ns"]),
                            -int(call["depth"]),
                            int(call["call_id"]),
                        ),
                    )
                    if left["phase"] == right["phase"]:
                        category = f"solver_control:within:{left['phase']}"
                    else:
                        category = (
                            f"solver_control:transition:{left['phase']}"
                            f"->{right['phase']}"
                        )
                else:
                    category = "solver_control:topology_unclassified"
                    topology_unclassified_ns += end - start
            category_ns[category] = category_ns.get(category, 0) + end - start
            if mechanism_resolved and category.startswith("solver_control:topology"):
                raise AssertionError("S0 topology classification invariant differs")
        if sum(category_ns.values()) != scope_ns:
            raise ValueError("FSG1 transaction inventory does not close")
        row: dict[str, object] = {
            "run_id": record["run_id"],
            "workload_id": record["workload_id"],
            "scope_ns": scope_ns,
            "transaction_ns": dict(sorted(category_ns.items())),
            "transaction_share": {
                key: value / scope_ns for key, value in sorted(category_ns.items())
            },
            "topology_unclassified_share": topology_unclassified_ns / scope_ns,
            "mechanism_unresolved_share": unresolved_ns / scope_ns,
            "topology_context_closed": topology_unclassified_ns == 0,
            "mechanism_admitted": unresolved_ns / scope_ns <= 0.03,
            "performance_claimed": False,
        }
        row["transaction_hash"] = canonical_hash(row)
        rows.append(row)
    inventory: dict[str, object] = {
        "schema_version": ASPLOS27_TENX_BUDGET_SCHEMA_VERSION,
        "status": (
            "s0-transaction-mechanism-admitted"
            if all(row["mechanism_admitted"] is True for row in rows)
            else "s0-transaction-mechanism-not-admitted"
        ),
        "run_count": len(rows),
        "topology_context_closed_count": sum(
            row["topology_context_closed"] is True for row in rows
        ),
        "mechanism_admitted_count": sum(
            row["mechanism_admitted"] is True for row in rows
        ),
        "runs": rows,
        "interpretation": (
            "neighboring phase context is classified; inter-call host mechanisms "
            "remain unresolved until explicit solver transaction markers exist"
        ),
        "performance_claimed": False,
    }
    inventory["inventory_hash"] = canonical_hash(inventory)
    return inventory


def validate_direct_observation_ledger(
    observations: Sequence[DirectCumulativeObservation],
) -> dict[str, object]:
    """Validate and serialize direct observations without cross-scope aggregation."""

    if not observations:
        raise ValueError("direct cumulative ledger must be non-empty")
    ids = [observation.observation_id for observation in observations]
    if len(ids) != len(set(ids)):
        raise ValueError("direct cumulative observation IDs duplicate")
    rows = []
    for observation in observations:
        observation.validate()
        rows.append(observation.to_dict())
    ledger: dict[str, object] = {
        "schema_version": ASPLOS27_TENX_BUDGET_SCHEMA_VERSION,
        "aggregation": "forbidden_across_distinct_scope_id_or_evidence_scope",
        "observation_count": len(rows),
        "observations": rows,
        "all_semantic_passed": all(row["semantic_passed"] is True for row in rows),
        "performance_claimed": False,
    }
    ledger["ledger_hash"] = canonical_hash(ledger)
    return ledger
