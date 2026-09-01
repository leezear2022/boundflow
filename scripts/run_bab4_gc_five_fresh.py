#!/usr/bin/env python3
"""Generate or replay five symmetric prepared-GC B4-A/BAB4 fresh pairs."""

from __future__ import annotations

from scripts import run_asplos27_s4_same_solver_five_fresh as implementation

ARTIFACT_SCHEMA = "boundflow.bab4-gc-five-fresh/v1"
CONTROL_CONFIGURATION = "B4-A-GC"
CANDIDATE_CONFIGURATION = "BAB4-GC"
PAIR_ORDERS = (
    (CONTROL_CONFIGURATION, CANDIDATE_CONFIGURATION),
    (CANDIDATE_CONFIGURATION, CONTROL_CONFIGURATION),
    (CONTROL_CONFIGURATION, CANDIDATE_CONFIGURATION),
    (CANDIDATE_CONFIGURATION, CONTROL_CONFIGURATION),
    (CONTROL_CONFIGURATION, CANDIDATE_CONFIGURATION),
)
CODE_PATHS = (
    "boundflow/backends/tvm/bab_input_domain.py",
    "boundflow/backends/tvm/bab_terminal_linear.py",
    "boundflow/backends/tvm/root_crown_projection.py",
    "boundflow/backends/tvm/root_crown_residual.py",
    "boundflow/runtime/bab_four_segment_exact_call_bridge.py",
    "boundflow/runtime/bab_four_segment_optimizer.py",
    "boundflow/runtime/bab_full_region_owner.py",
    "boundflow/runtime/bab_input_domain_tir.py",
    "boundflow/runtime/bab_terminal_tir.py",
    "boundflow/runtime/prepared_gc_isolation.py",
    "boundflow/runtime/prepared_root_optimizer_warmup.py",
    "boundflow/runtime/root_crown_projection_tir.py",
    "boundflow/runtime/root_crown_residual_tir.py",
    "scripts/run_asplos27_s4_same_solver_five_fresh.py",
    "scripts/run_asplos27_s4_same_solver_worker.py",
    "scripts/run_bab4_gc_five_fresh.py",
    "scripts/run_fsg3_same_solver_timing.py",
    "scripts/run_fsg4_b3_same_solver_timing.py",
    "scripts/run_fsg4_b4a_same_solver_worker.py",
    "artifacts/asplos27-s4-exact-call-plan/resnet2b-prop0-v1/plan_template.json",
)


def configure() -> None:
    """Install the symmetric prepared-GC protocol into the shared runner."""

    implementation.ARTIFACT_SCHEMA = ARTIFACT_SCHEMA
    implementation.CONTROL_CONFIGURATION = CONTROL_CONFIGURATION
    implementation.CANDIDATE_CONFIGURATION = CANDIDATE_CONFIGURATION
    implementation.PAIR_ORDERS = PAIR_ORDERS
    implementation.CODE_PATHS = CODE_PATHS


def main() -> None:
    """Dispatch generation or replay using the prepared-GC protocol."""

    configure()
    implementation.main()


if __name__ == "__main__":
    main()
