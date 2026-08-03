"""Deterministic P0 audit for production Schedule IR ownership and memory value."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from ..ir.bound import BoundOpKind
from ..ir.plan import BackendKind, PlanInstance
from ..ir.schedule import lower_plan_instance_to_reference_schedule
from ..runtime.verifier_ir_integration import (
    ExternalVerifierCallSpec,
    compile_external_verifier_call,
)
from .plan_ir_selector import NoFeasiblePlanError, select_plan_instance
from .typed_benchmark_workloads import build_residual_cnn_candidate

PRODUCTION_SCHEDULE_COVERAGE_SCHEMA = "boundflow.production-schedule-coverage-audit/v1"
IR5_SCHEMA = "boundflow.ir5-residual-final-artifact/v3"
RVIR_SCHEMA = "boundflow.real-verifier-ir-artifact/v2"
HIGH_MEMORY_BUDGET_BYTES = 512 * 1024 * 1024
LOW_MEMORY_BUDGET_BYTES = 64 * 1024 * 1024
RESIDUAL_WORKLOADS: tuple[dict[str, Any], ...] = (
    {
        "workload_id": "final-residual-gray-v3",
        "batch": 4,
        "input_channels": 1,
        "image_size": 14,
        "block_channels": 5,
        "output_dim": 12,
        "seed": 7501,
    },
    {
        "workload_id": "final-residual-color-v3",
        "batch": 4,
        "input_channels": 3,
        "image_size": 18,
        "block_channels": 7,
        "output_dim": 12,
        "seed": 7502,
    },
)
AUDITED_BACKENDS = (
    BackendKind.REFERENCE,
    BackendKind.PYTORCH_DENSE,
    BackendKind.PYTORCH_CHUNKED,
    BackendKind.TVM_FUSED_TIR,
)
ARTIFACT_FILES = ("coverage.json",)


def canonical_json(value: object, *, indent: int | None = None) -> str:
    """Encode one artifact value with deterministic JSON ordering."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
        allow_nan=False,
    )


def sha256_path(path: Path) -> str:
    """Return the SHA256 digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_production_schedule_coverage(
    *, ir5_artifact_dir: Path, rvir_artifact_dir: Path
) -> dict[str, object]:
    """Recompute the P0 gate from frozen evidence and current compiler lowering."""

    ir5_source = _verify_source_artifact(ir5_artifact_dir, IR5_SCHEMA)
    rvir_source = _verify_source_artifact(rvir_artifact_dir, RVIR_SCHEMA)
    ir5_summary = _read_json(ir5_artifact_dir / "summary.json")
    ir5_measurements = _read_jsonl(ir5_artifact_dir / "heldout_compiler.jsonl")
    residual = _residual_schedule_coverage(ir5_summary, ir5_measurements)
    real_resnet = _real_resnet_schedule_coverage(rvir_artifact_dir)

    gates = {
        "frozen_source_digests_verified": True,
        "residual_schedule_owns_full_bound_graph": bool(
            residual["all_bound_ops_schedule_owned"]
        ),
        "residual_schedule_owns_arena_lifecycle": bool(
            residual["all_arena_lifecycle_actions_present"]
        ),
        "residual_schedule_exercises_materialization_transition": bool(
            residual["any_materialization_action_present"]
        ),
        "residual_template_has_storage_choice": bool(
            residual["any_alternative_storage_candidate"]
        ),
        "real_resnet_is_native_multi_region_bound_ir": not bool(
            real_resnet["main_compute_is_opaque_external_call"]
        ),
        "real_resnet_schedule_owns_main_compute_lifecycle": not bool(
            real_resnet["main_compute_is_opaque_external_call"]
        ),
        "frozen_multi_budget_plan_switch_demonstrated": bool(
            residual["frozen_measurement_gates"]["any_multi_budget_global_switch"]
        ),
        "current_plan_decisions_change_with_budget": bool(
            residual["any_structural_budget_decision_switch"]
        ),
        "compiler_latency_memory_pareto_all_workloads": bool(
            residual["frozen_measurement_gates"][
                "compiler_latency_memory_pareto_all_workloads"
            ]
        ),
        "baseline_oom_rescue_demonstrated": False,
        "prior_ir5_global_regret_gate_passed": bool(
            residual["frozen_measurement_gates"]["global_p90_regret_lte_1_20"]
        ),
    }
    admission_gate_ids = (
        "residual_schedule_exercises_materialization_transition",
        "residual_template_has_storage_choice",
        "real_resnet_is_native_multi_region_bound_ir",
        "real_resnet_schedule_owns_main_compute_lifecycle",
        "frozen_multi_budget_plan_switch_demonstrated",
        "current_plan_decisions_change_with_budget",
        "compiler_latency_memory_pareto_all_workloads",
        "baseline_oom_rescue_demonstrated",
    )
    admitted = all(gates[gate_id] for gate_id in admission_gate_ids)
    reasons = [gate_id for gate_id in admission_gate_ids if not gates[gate_id]]
    return {
        "schema_version": PRODUCTION_SCHEDULE_COVERAGE_SCHEMA,
        "gate_id": "production-schedule-memory-p0",
        "claim_boundary": (
            "structural ownership and frozen-evidence audit only; no new latency, "
            "memory, correctness, or performance claim"
        ),
        "source_artifacts": {
            "ir5_residual_final_v3": ir5_source,
            "rvir_cpu_correctness_v2": rvir_source,
        },
        "residual_schedule_path": residual,
        "real_resnet_schedule_path": real_resnet,
        "gates": gates,
        "admission_gate_ids": list(admission_gate_ids),
        "verdict": "GO" if admitted else "NO_GO",
        "failed_gate_ids": reasons,
        "next_workstream": {
            "branch": "feat/native-real-network-bound-ir-v1",
            "objective": (
                "lower one frozen real residual network into a non-opaque native "
                "Bound IR graph before reopening Schedule IR memory optimization"
            ),
            "reopen_memory_workstream_only_if": [
                "the real-network main compute is a multi-region native Bound IR graph",
                "Schedule IR owns allocate/free/batch/materialize/launch for that graph",
                "two legal memory budgets select different batch/storage/region decisions",
                "a frozen baseline OOM is rescued or a reproducible memory Pareto point exists",
            ],
        },
        "limitations": [
            (
                "residual-final-v3 is a reduced synthetic residual CNN and its frozen "
                "measurements were produced on CUDA"
            ),
            (
                "current structural regeneration uses CPU because this host exposes no "
                "CUDA device; it does not replace the frozen CUDA measurements"
            ),
            (
                "the real ResNet path is audited through 51 frozen VNN-COMP activation "
                "calls, but their numerical execution remains externally owned"
            ),
            "absence of an OOM-rescue artifact is reported as not demonstrated, not impossible",
        ],
    }


def generate_production_schedule_coverage_artifact(
    artifact_dir: Path, *, ir5_artifact_dir: Path, rvir_artifact_dir: Path
) -> dict[str, object]:
    """Generate a deterministic P0 audit artifact from frozen sources."""

    if artifact_dir.exists() and any(artifact_dir.iterdir()):
        raise ValueError(f"artifact output directory is not empty: {artifact_dir}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    coverage = build_production_schedule_coverage(
        ir5_artifact_dir=ir5_artifact_dir,
        rvir_artifact_dir=rvir_artifact_dir,
    )
    coverage_path = artifact_dir / "coverage.json"
    coverage_path.write_text(
        canonical_json(coverage, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": PRODUCTION_SCHEDULE_COVERAGE_SCHEMA,
        "gate_id": "production-schedule-memory-p0",
        "verdict": coverage["verdict"],
        "files": {name: sha256_path(artifact_dir / name) for name in ARTIFACT_FILES},
    }
    (artifact_dir / "manifest.json").write_text(
        canonical_json(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return coverage


def replay_production_schedule_coverage_artifact(
    artifact_dir: Path, *, ir5_artifact_dir: Path, rvir_artifact_dir: Path
) -> dict[str, object]:
    """Reject digest or semantic drift by recomputing the complete audit."""

    manifest = _read_json(artifact_dir / "manifest.json")
    if manifest.get("schema_version") != PRODUCTION_SCHEDULE_COVERAGE_SCHEMA:
        raise ValueError("production Schedule coverage manifest schema mismatch")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(ARTIFACT_FILES):
        raise ValueError("production Schedule coverage manifest file set mismatch")
    for name in ARTIFACT_FILES:
        if sha256_path(artifact_dir / name) != files[name]:
            raise ValueError(f"production Schedule coverage digest mismatch: {name}")
    stored = _read_json(artifact_dir / "coverage.json")
    expected = build_production_schedule_coverage(
        ir5_artifact_dir=ir5_artifact_dir,
        rvir_artifact_dir=rvir_artifact_dir,
    )
    if canonical_json(stored) != canonical_json(expected):
        raise ValueError("production Schedule coverage semantic replay mismatch")
    if manifest.get("verdict") != expected["verdict"]:
        raise ValueError("production Schedule coverage verdict mismatch")
    return expected


def _verify_source_artifact(
    artifact_dir: Path, expected_schema: str
) -> dict[str, object]:
    manifest_path = artifact_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != expected_schema:
        raise ValueError(f"source artifact schema mismatch: {artifact_dir}")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ValueError(f"source artifact manifest has no files: {artifact_dir}")
    for name, expected_digest in sorted(files.items()):
        path = artifact_dir / str(name)
        if not path.is_file() or sha256_path(path) != expected_digest:
            raise ValueError(f"source artifact digest mismatch: {path}")
    return {
        "schema_version": expected_schema,
        "manifest_sha256": sha256_path(manifest_path),
        "verified_file_count": len(files),
    }


def _residual_schedule_coverage(
    summary: Mapping[str, Any], measurements: list[dict[str, Any]]
) -> dict[str, Any]:
    gates = summary.get("gates")
    if not isinstance(gates, Mapping):
        raise ValueError("IR-5 summary omits gates")
    expected_measurements = len(RESIDUAL_WORKLOADS) * len(AUDITED_BACKENDS)
    if len(measurements) != expected_measurements:
        raise ValueError("IR-5 held-out measurement count drift")
    frozen_hashes = {
        f"{row['workload']['workload_id']}:{row['backend']}": str(row["schedule_hash"])
        for row in measurements
    }
    if len(frozen_hashes) != expected_measurements or any(
        len(value) != 64 for value in frozen_hashes.values()
    ):
        raise ValueError("IR-5 frozen Schedule IR identities are incomplete")

    cases = []
    for workload in RESIDUAL_WORKLOADS:
        for backend in AUDITED_BACKENDS:
            prepared = build_residual_cnn_candidate(
                backend=backend,
                device="cpu",
                **workload,
            )
            cases.append(_prepared_budget_case(prepared))
    return {
        "evidence_scope": "reduced_synthetic_residual_cnn",
        "structural_regeneration_device": "cpu",
        "frozen_measurement_schedule_hashes": dict(sorted(frozen_hashes.items())),
        "frozen_measurement_gates": {
            "all_compiler_candidates_semantic_allclose": bool(
                gates.get("all_compiler_candidates_semantic_allclose")
            ),
            "any_multi_budget_global_switch": bool(
                gates.get("any_multi_budget_global_switch")
            ),
            "compiler_latency_memory_pareto_all_workloads": bool(
                gates.get("compiler_latency_memory_pareto_all_workloads")
            ),
            "global_p90_regret_lte_1_20": bool(gates.get("global_p90_regret_lte_1_20")),
        },
        "case_count": len(cases),
        "all_bound_ops_schedule_owned": all(
            case["covered_bound_op_count"] == case["bound_op_count"] for case in cases
        ),
        "all_arena_lifecycle_actions_present": all(
            case["arena_lifecycle_actions_present"] for case in cases
        ),
        "any_materialization_action_present": any(
            case["schedule_action_counts"].get("materialize", 0) > 0 for case in cases
        ),
        "any_alternative_storage_candidate": any(
            case["storage_candidate_count"] > 1 for case in cases
        ),
        "any_structural_budget_decision_switch": any(
            case["budget_probe"]["decisions_switched"] for case in cases
        ),
        "cases": cases,
    }


def _prepared_budget_case(prepared) -> dict[str, Any]:
    high = _select_for_budget(prepared, HIGH_MEMORY_BUDGET_BYTES)
    low = _select_for_budget(prepared, LOW_MEMORY_BUDGET_BYTES)
    peak_bytes = high.cost_summary.predicted_peak_bytes
    below_peak_rejected = False
    rejection_reasons: list[str] = []
    try:
        _select_for_budget(prepared, peak_bytes - 1)
    except NoFeasiblePlanError as error:
        below_peak_rejected = True
        rejection_reasons = [failure.reason for failure in error.failures]
    high_schedule = lower_plan_instance_to_reference_schedule(
        prepared.bound_module,
        template=prepared.template,
        instance=high,
        query_ids=(f"query:{prepared.workload_id}",),
    )
    low_schedule = lower_plan_instance_to_reference_schedule(
        prepared.bound_module,
        template=prepared.template,
        instance=low,
        query_ids=(f"query:{prepared.workload_id}",),
    )
    regions = {
        candidate.candidate_id: candidate
        for candidate in prepared.template.region_candidates
    }
    covered_ops = {
        op_id
        for decision in high.region_decisions
        for op_id in regions[decision.candidate_id].op_ids
    }
    action_counts = _action_counts(high_schedule.actions)
    lifecycle = all(
        action_counts.get(kind, 0) > 0
        for kind in ("check_budget", "allocate", "batch_loop", "launch", "free")
    )
    high_signature = _decision_signature(high)
    low_signature = _decision_signature(low)
    return {
        "workload_id": prepared.workload_id,
        "backend": prepared.backend.value,
        "bound_op_count": len(prepared.bound_module.graph.ops),
        "external_bound_op_count": sum(
            op.kind == BoundOpKind.EXTERNAL_VERIFIER_CALL
            for op in prepared.bound_module.graph.ops
        ),
        "covered_bound_op_count": len(covered_ops),
        "selected_region_count": len(high.region_decisions),
        "schedule_buffer_count": len(high_schedule.buffers),
        "schedule_action_counts": action_counts,
        "arena_lifecycle_actions_present": lifecycle,
        "batch_candidate_count": len(prepared.template.batch_candidates),
        "storage_candidate_count": len(prepared.template.storage_candidates),
        "predicted_peak_bytes": peak_bytes,
        "budget_probe": {
            "high_memory_budget_bytes": HIGH_MEMORY_BUDGET_BYTES,
            "low_memory_budget_bytes": LOW_MEMORY_BUDGET_BYTES,
            "both_plan_instances_legal": True,
            "plan_instance_hashes_differ": high.stable_hash(
                template=prepared.template, bound_module=prepared.bound_module
            )
            != low.stable_hash(
                template=prepared.template, bound_module=prepared.bound_module
            ),
            "decisions_switched": high_signature != low_signature,
            "high_decisions": high_signature,
            "low_decisions": low_signature,
            "schedule_action_profiles_differ": _action_counts(high_schedule.actions)
            != _action_counts(low_schedule.actions),
            "below_peak_budget_bytes": peak_bytes - 1,
            "below_peak_rejected": below_peak_rejected,
            "below_peak_rejection_reasons": rejection_reasons,
        },
    }


def _select_for_budget(prepared, budget_bytes: int) -> PlanInstance:
    return select_plan_instance(
        prepared.template,
        bound_module=prepared.bound_module,
        query_bucket_id=f"p0:{prepared.workload_id}:{prepared.backend.value}",
        available_memory_bytes=HIGH_MEMORY_BUDGET_BYTES,
        memory_budget_bytes=budget_bytes,
    )


def _decision_signature(instance: PlanInstance) -> dict[str, object]:
    return {
        "regions": [
            [item.region_id, item.candidate_id] for item in instance.region_decisions
        ],
        "representations": [
            [item.region_id, item.candidate_id]
            for item in instance.representation_decisions
        ],
        "materializations": [
            item.candidate_id for item in instance.materialization_decisions
        ],
        "backends": [
            [item.region_id, item.candidate_id] for item in instance.backend_decisions
        ],
        "batch": instance.batch_decision.candidate_id,
        "storage": instance.storage_decision.candidate_id,
        "states": [
            [item.state_id, item.candidate_id] for item in instance.state_decisions
        ],
    }


def _real_resnet_schedule_coverage(artifact_dir: Path) -> dict[str, Any]:
    rows = [
        row
        for row in _read_jsonl(artifact_dir / "activation_calls.jsonl")
        if row.get("source_workload") == "vnncomp21-resnet2b-prop0"
    ]
    if len(rows) != 51:
        raise ValueError("RVIR ResNet activation-call count drift")
    profiles: Counter[str] = Counter()
    representative: dict[str, object] | None = None
    for row in rows:
        query = row.get("query")
        if not isinstance(query, Mapping):
            raise ValueError("RVIR activation row omits query")
        compiled = compile_external_verifier_call(
            ExternalVerifierCallSpec.from_query_dict(query)
        )
        if compiled.hashes() != row.get("ir_hashes"):
            raise ValueError("RVIR ResNet IR hash replay mismatch")
        action_counts = _action_counts(compiled.schedule.actions)
        profile = canonical_json(action_counts)
        profiles[profile] += 1
        external_ops = sum(
            op.kind == BoundOpKind.EXTERNAL_VERIFIER_CALL
            for op in compiled.bound_module.graph.ops
        )
        if representative is None:
            representative = {
                "query_id": compiled.call_spec.query_id,
                "domain_count": compiled.call_spec.domain_count,
                "spec_count": compiled.call_spec.spec_count,
                "bound_op_count": len(compiled.bound_module.graph.ops),
                "external_bound_op_count": external_ops,
                "selected_region_count": len(compiled.instance.region_decisions),
                "schedule_buffer_count": len(compiled.schedule.buffers),
                "schedule_action_counts": action_counts,
                "predicted_peak_bytes": (
                    compiled.instance.cost_summary.predicted_peak_bytes
                ),
                "backend": str(row.get("backend")),
                "semantics_owner": str(row.get("semantics_owner")),
                "performance_claimed": row.get("performance_claimed"),
            }
        if external_ops != len(compiled.bound_module.graph.ops):
            raise ValueError("RVIR ResNet unexpectedly contains native Bound IR ops")
    if representative is None:
        raise AssertionError("RVIR ResNet representative was not constructed")
    return {
        "evidence_scope": "vnncomp21-resnet2b-prop0",
        "activation_call_count": len(rows),
        "all_ir_hashes_recompiled_exactly": True,
        "unique_schedule_action_profile_count": len(profiles),
        "schedule_action_profiles": [
            {"action_counts": json.loads(profile), "call_count": count}
            for profile, count in sorted(profiles.items())
        ],
        "main_compute_is_opaque_external_call": True,
        "representative": representative,
    }


def _action_counts(actions) -> dict[str, int]:
    return dict(sorted(Counter(action.kind.value for action in actions).items()))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"expected JSON object at {path}:{line_number}")
        rows.append(value)
    return rows


__all__ = [
    "PRODUCTION_SCHEDULE_COVERAGE_SCHEMA",
    "build_production_schedule_coverage",
    "canonical_json",
    "generate_production_schedule_coverage_artifact",
    "replay_production_schedule_coverage_artifact",
    "sha256_path",
]
