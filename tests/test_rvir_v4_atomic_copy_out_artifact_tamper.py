"""Tests for the formal RVIR-v4 atomic copy-out tamper suite."""

# pylint: disable=missing-function-docstring

from pathlib import Path

from scripts import probe_rvir_v4_atomic_copy_out_artifact_tamper as probe

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/rvir-v4-atomic-copy-out/resnet2b-core-copy-out-v1"


def test_six_synchronized_rehash_attacks_fail_closed() -> None:
    report = probe.run_probe_suite(ARTIFACT)

    assert report["probe_count"] == 6
    assert report["all_outer_provenance_gates_rejected"] is True
    assert report["all_semantic_mutation_gates_rejected"] is True
    assert report["performance_claimed"] is False
    assert {row["name"] for row in report["probes"]} == {
        "topology-internal-rehash",
        "initial-upper-alpha-internal-rehash",
        "expected-post-alpha-internal-rehash",
        "final-production-lower-cross-resign",
        "recorded-copy_out-full-resign",
        "recorded-commit-full-resign",
    }
