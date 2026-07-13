"""Frozen three-level benchmark contract tests for PR-12H."""

from boundflow.benchmarks.contracts import (
    BENCHMARK_CONTRACT_SCHEMA_VERSION,
    END_TO_END_FINAL_BOUND_CONTRACT,
    KERNEL_CONTRACT,
    REGION_RUNTIME_CONTRACT,
    BenchmarkContractLevel,
    contract_from_payload,
)
from scripts.benchmark_phase7a_pr12_fused_sanity import LEGACY_KERNEL_DISCLOSURE
from scripts.benchmark_phase7a_pr12_runtime_pareto import (
    LEGACY_CANDIDATE_DISCLOSURE,
)


def test_three_contract_levels_are_distinct_and_valid() -> None:
    contracts = (
        KERNEL_CONTRACT,
        REGION_RUNTIME_CONTRACT,
        END_TO_END_FINAL_BOUND_CONTRACT,
    )

    for contract in contracts:
        contract.validate()

    assert {contract.level for contract in contracts} == set(BenchmarkContractLevel)
    assert len({contract.contract_id for contract in contracts}) == 3
    assert KERNEL_CONTRACT.requires_preallocated_outputs
    assert not KERNEL_CONTRACT.measures_allocator_peak
    assert REGION_RUNTIME_CONTRACT.includes_backend_dispatch
    assert not REGION_RUNTIME_CONTRACT.includes_planner
    assert END_TO_END_FINAL_BOUND_CONTRACT.includes_planner
    assert END_TO_END_FINAL_BOUND_CONTRACT.includes_concretization


def test_contract_payload_round_trip_and_hash_are_stable() -> None:
    for contract in (
        KERNEL_CONTRACT,
        REGION_RUNTIME_CONTRACT,
        END_TO_END_FINAL_BOUND_CONTRACT,
    ):
        payload = contract.to_dict()
        assert payload["schema_version"] == BENCHMARK_CONTRACT_SCHEMA_VERSION
        assert contract_from_payload(payload) == contract
        assert contract.sha256() == contract_from_payload(payload).sha256()
        assert payload["timing"]["timed_global_synchronize"] is False


def test_historical_pr12_benchmarks_disclose_contract_gaps() -> None:
    assert LEGACY_KERNEL_DISCLOSURE["compliant"] is False
    assert LEGACY_CANDIDATE_DISCLOSURE["compliant"] is False
    assert "planner" in LEGACY_CANDIDATE_DISCLOSURE["excluded_from_timed_region"]
    assert (
        "region_matching" in LEGACY_CANDIDATE_DISCLOSURE["excluded_from_timed_region"]
    )
