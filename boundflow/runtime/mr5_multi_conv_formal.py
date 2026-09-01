"""Mechanical gates for the MR5 five-pair multi-Conv correctness artifact."""

# pylint: disable=missing-function-docstring,too-many-locals,too-many-statements
# pylint: disable=too-many-branches,too-many-boolean-expressions
# pylint: disable=duplicate-code

from __future__ import annotations

from typing import Any, Mapping, Sequence, cast

from boundflow.runtime.mr3_production_bridge_formal import _compare_payload
from boundflow.runtime.mr3_provider_hook_feasibility import canonical_hash

FORMAL_SCHEMA = "boundflow.mr5-multi-conv-production-bridge-formal/v1"
WORKER_SCHEMA = "boundflow.mr5-multi-conv-production-bridge-worker/v1"
STATUS = "VALIDATED-MR5-MULTI-CONV-PRODUCTION-BRIDGE-CORRECTNESS"
SOURCE_COMMIT = "3e1a70933910c009019c59de4f44d233a75f7950"
ABCROWN_COMMIT = "e5c7e17bf0488843acb77b7519f59876717a49f4"
AUTO_LIRPA_COMMIT = "5a098e8f9fb5786a428a024981d833d303921f2d"
VNNCOMP_COMMIT = "90419aadcf06cf543ce5c1706cae1059dc9fa6cf"
MODEL_SHA256 = "791aa24d77917ecda16809fbbd48e7739616f88ebf74cf358b2d1bf911dc4a6d"
PROPERTY_SHA256 = "89edf0665d74397670d0562d513db694a49a84edaf5cf3d64c9c6fa63c3769ff"
SITE_ORDER = ("C2", "C1", "C0")
EXPECTED_RUNS = (
    (0, 0, "provider"),
    (0, 1, "bridge"),
    (1, 0, "bridge"),
    (1, 1, "provider"),
    (2, 0, "provider"),
    (2, 1, "bridge"),
    (3, 0, "bridge"),
    (3, 1, "provider"),
    (4, 0, "provider"),
    (4, 1, "bridge"),
)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _expect_site_map(
    value: object, expected: int, *, allow_range: bool = False
) -> bool:
    if not isinstance(value, Mapping) or set(value) != set(SITE_ORDER):
        return False
    if allow_range:
        return all(
            isinstance(item, int) and item in range(expected + 1)
            for item in value.values()
        )
    return all(item == expected for item in value.values())


def _validate_module_receipts(value: object, signatures: Mapping[str, object]) -> None:
    if not isinstance(value, Mapping) or set(value) != set(SITE_ORDER):
        raise ValueError("MR5 module receipt inventory differs")
    expected_workspace = {
        "C0": {"adjoint_conv": [6, 1, 8, 16, 16], "bias_delta": [6, 1]},
        "C1": {"adjoint_conv": [6, 1, 16, 8, 8], "bias_delta": [6, 1]},
        "C2": {"adjoint_conv": [6, 1, 16, 8, 8], "bias_delta": [6, 1]},
    }
    identities: set[tuple[str, str, str, str]] = set()
    for site, receipt in value.items():
        if not isinstance(receipt, Mapping):
            raise ValueError("MR5 module receipt differs")
        hashes = tuple(
            receipt.get(name)
            for name in (
                "signature_hash",
                "unscheduled_tir_hash",
                "scheduled_tir_hash",
                "device_source_hash",
            )
        )
        workspace = receipt.get("workspace_inventory")
        if (
            receipt.get("site_id") != site
            or receipt.get("signature_hash") != signatures.get(site)
            or any(not _is_sha256(item) for item in hashes)
            or not isinstance(receipt.get("tvm_version"), str)
            or not isinstance(receipt.get("torch_version"), str)
            or not isinstance(workspace, list)
            or {
                item.get("name"): item.get("shape")
                for item in workspace
                if isinstance(item, Mapping)
            }
            != expected_workspace[cast(str, site)]
        ):
            raise ValueError(f"MR5 {site} module receipt differs")
        identities.add(cast(tuple[str, str, str, str], hashes))
    if len(identities) != 3:
        raise ValueError("MR5 module receipts are not site-specific")


def _validate_bridge_receipt(value: object) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("MR5 bridge receipt is absent")
    signatures = value.get("signature_hashes")
    if (
        value.get("evaluation_count") != 10
        or value.get("site_order_count") != 30
        or not _expect_site_map(value.get("forward_launches"), 10)
        or not _expect_site_map(value.get("backward_launches"), 9)
        or not _expect_site_map(value.get("beta_tensor_count"), 10)
        or not _expect_site_map(value.get("beta_numel"), 0)
        or not _expect_site_map(value.get("handoff_content_count"), 10)
        or not _expect_site_map(
            value.get("handoff_pointer_count"), 10, allow_range=True
        )
        or not _expect_site_map(value.get("cache_miss_count"), 1)
        or not _expect_site_map(value.get("cache_hit_count"), 9)
        or not isinstance(signatures, Mapping)
        or set(signatures) != set(SITE_ORDER)
        or any(not _is_sha256(item) for item in signatures.values())
        or len(set(signatures.values())) != 3
        or value.get("pending_site_count") != 0
        or value.get("fallback_count") != 0
        or value.get("eager_count") != 0
        or value.get("native_shadow_count") != 0
        or value.get("timing_recorded") is not False
        or value.get("performance_claimed") is not False
    ):
        raise ValueError("MR5 bridge receipt differs")
    _validate_module_receipts(value.get("module_receipts"), signatures)


def _validate_worker(worker: Mapping[str, Any], *, mode: str) -> None:
    unsigned = dict(worker)
    worker_hash = unsigned.pop("worker_hash", None)
    source = worker.get("source")
    expected_atomic = {
        "exact_call_launch_count": 1,
        "staged_emit_count": 1,
        "atomic_commit_count": 1,
        "rollback_count": 0,
    }
    if (
        worker.get("schema_version") != WORKER_SCHEMA
        or worker.get("mode") != mode
        or worker_hash != canonical_hash(unsigned)
        or not isinstance(source, Mapping)
        or source.get("abcrown_commit") != ABCROWN_COMMIT
        or source.get("auto_lirpa_commit") != AUTO_LIRPA_COMMIT
        or source.get("vnncomp_commit") != VNNCOMP_COMMIT
        or source.get("model_sha256") != MODEL_SHA256
        or source.get("property_sha256") != PROPERTY_SHA256
        or worker.get("protocol")
        != {
            "device": "cuda",
            "seed": 100,
            "max_iterations": 1,
            "batch_size": 64,
            "alpha_steps": 5,
            "beta_steps": 10,
            "site_order": list(SITE_ORDER),
        }
        or worker.get("solver_result")
        != {"status": "verified", "success": True, "visited_domains": [6]}
        or worker.get("timing_recorded") is not False
        or worker.get("performance_claimed") is not False
        or worker.get("atomic_receipt") != expected_atomic
        or len(worker.get("region_states", [])) != 30
        or len(worker.get("evaluation_trajectory", [])) != 10
        or len(worker.get("mutation_trajectory", [])) != 9
        or not isinstance(worker.get("final_clip_state"), Mapping)
    ):
        raise ValueError("MR5 formal worker provenance/count differs")
    if mode == "provider":
        if worker.get("bridge_receipt") is not None:
            raise ValueError("MR5 provider unexpectedly has bridge receipt")
    else:
        _validate_bridge_receipt(worker.get("bridge_receipt"))
    for ordinal, row in enumerate(worker["evaluation_trajectory"]):
        if row.get("evaluation_ordinal") != ordinal:
            raise ValueError("MR5 evaluation order differs")
    for ordinal, row in enumerate(worker["mutation_trajectory"]):
        if (
            row.get("mutation_ordinal") != ordinal
            or row.get("optimizer_step") != float(ordinal + 1)
            or not isinstance(row.get("clamp_mask"), Mapping)
        ):
            raise ValueError("MR5 mutation order differs")
    expected_regions = [
        (evaluation, site) for evaluation in range(10) for site in SITE_ORDER
    ]
    observed_regions = [
        (row.get("evaluation_ordinal"), row.get("site"))
        for row in worker["region_states"]
    ]
    if observed_regions != expected_regions:
        raise ValueError("MR5 site/evaluation region order differs")


def _validate_rollback(rollback: Mapping[str, Any]) -> None:
    unsigned = dict(rollback)
    worker_hash = unsigned.pop("worker_hash", None)
    receipt = rollback.get("atomic_receipt")
    if (
        rollback.get("schema_version") != WORKER_SCHEMA
        or rollback.get("mode") != "bridge"
        or rollback.get("injected_failure_evaluation") != 5
        or rollback.get("injected_failure_site") != "C1"
        or rollback.get("caught_failure") != "MR5 injected candidate failure"
        or rollback.get("timing_recorded") is not False
        or rollback.get("performance_claimed") is not False
        or worker_hash != canonical_hash(unsigned)
        or not isinstance(receipt, Mapping)
        or receipt.get("exact_call_launch_count") != 1
        or receipt.get("staged_emit_count") != 0
        or receipt.get("atomic_commit_count") != 0
        or receipt.get("rollback_count") != 1
        or receipt.get("owner_tensor_count") != 12
        or receipt.get("owner_content_hash_before")
        != receipt.get("owner_content_hash_after")
        or receipt.get("owner_pointer_hash_before")
        != receipt.get("owner_pointer_hash_after")
        or not _is_sha256(receipt.get("owner_content_hash_before"))
        or not _is_sha256(receipt.get("owner_pointer_hash_before"))
        or not isinstance(receipt.get("version_delta_min"), int)
        or receipt.get("version_delta_min", 0) < 1
        or receipt.get("version_delta_max", 0) < receipt.get("version_delta_min", 0)
    ):
        raise ValueError("MR5 atomic rollback receipt differs")


def _pair_metric(
    pair_index: int, provider: Mapping[str, Any], bridge: Mapping[str, Any]
) -> dict[str, object]:
    general_fields: Sequence[str] = (
        "solver_result",
        "region_states",
        "evaluation_trajectory",
        "inner_result_states",
        "outer_result_state",
        "final_target_alpha_state",
        "final_module_state",
        "final_clip_state",
    )
    general_maximum = 0.0
    general_count = 0
    for field in general_fields:
        maximum, count = _compare_payload(
            provider[field],
            bridge[field],
            atol=2.0e-4,
            rtol=2.0e-4,
            path=field,
        )
        general_maximum = max(general_maximum, maximum)
        general_count += count
    optimizer_maximum = 0.0
    optimizer_count = 0
    for ordinal, (provider_row, bridge_row) in enumerate(
        zip(provider["mutation_trajectory"], bridge["mutation_trajectory"])
    ):
        for field in (
            "gradient",
            "alpha_pre_clamp",
            "exp_avg",
            "exp_avg_sq",
            "alpha_post_clamp",
        ):
            maximum, count = _compare_payload(
                provider_row[field],
                bridge_row[field],
                atol=2.0e-5,
                rtol=2.0e-5,
                path=f"mutation[{ordinal}].{field}",
            )
            optimizer_maximum = max(optimizer_maximum, maximum)
            optimizer_count += count
        for field in ("mutation_ordinal", "optimizer_step", "lr_used", "clamp_mask"):
            if provider_row[field] != bridge_row[field]:
                raise ValueError(
                    f"MR5 optimizer discrete drift at mutation[{ordinal}].{field}"
                )
    metric: dict[str, object] = {
        "pair_index": pair_index,
        "general_maximum_absolute_difference": general_maximum,
        "general_element_count": general_count,
        "optimizer_maximum_absolute_difference": optimizer_maximum,
        "optimizer_element_count": optimizer_count,
        "sign_exact": True,
        "allclose": True,
    }
    metric["metric_hash"] = canonical_hash(metric)
    return metric


def derive_summary(raw: Mapping[str, Any]) -> dict[str, object]:
    unsigned = dict(raw)
    raw_hash = unsigned.pop("raw_hash", None)
    runs_value = raw.get("runs")
    rollback = raw.get("rollback_probe")
    if (
        raw.get("schema_version") != FORMAL_SCHEMA
        or raw.get("source_commit") != SOURCE_COMMIT
        or raw.get("run_order") != [list(run) for run in EXPECTED_RUNS]
        or raw_hash != canonical_hash(unsigned)
        or not isinstance(runs_value, list)
        or len(runs_value) != 10
        or not isinstance(rollback, Mapping)
    ):
        raise ValueError("MR5 formal raw provenance differs")
    _validate_rollback(rollback)
    runs: list[Mapping[str, Any]] = []
    for expected, wrapper in zip(EXPECTED_RUNS, runs_value):
        if (
            not isinstance(wrapper, Mapping)
            or (wrapper.get("pair_index"), wrapper.get("position"), wrapper.get("mode"))
            != expected
        ):
            raise ValueError("MR5 formal run order differs")
        worker = wrapper.get("worker")
        if not isinstance(worker, Mapping):
            raise ValueError("MR5 formal worker is absent")
        _validate_worker(worker, mode=expected[2])
        runs.append(wrapper)
    bridge_receipts = [
        run["worker"]["bridge_receipt"] for run in runs if run["mode"] == "bridge"
    ]
    if any(receipt != bridge_receipts[0] for receipt in bridge_receipts[1:]):
        raise ValueError("MR5 bridge compiler receipt drifted across fresh runs")
    metrics: list[dict[str, object]] = []
    for pair_index in range(5):
        pair = [run for run in runs if run["pair_index"] == pair_index]
        provider = next(run["worker"] for run in pair if run["mode"] == "provider")
        bridge = next(run["worker"] for run in pair if run["mode"] == "bridge")
        metrics.append(_pair_metric(pair_index, provider, bridge))
    summary: dict[str, object] = {
        "schema_version": FORMAL_SCHEMA,
        "status": STATUS,
        "source_commit": SOURCE_COMMIT,
        "pair_count": 5,
        "fresh_process_count": 10,
        "run_order": [list(run) for run in EXPECTED_RUNS],
        "site_count": 3,
        "candidate_forward_count": 150,
        "candidate_backward_count": 135,
        "atomic_rollback_probe_count": 1,
        "pair_metrics": metrics,
        "general_maximum_absolute_difference": max(
            cast(float, metric["general_maximum_absolute_difference"])
            for metric in metrics
        ),
        "optimizer_maximum_absolute_difference": max(
            cast(float, metric["optimizer_maximum_absolute_difference"])
            for metric in metrics
        ),
        "timing_preregistration_open": True,
        "timing_recorded": False,
        "performance_claimed": False,
    }
    summary["summary_hash"] = canonical_hash(summary)
    return summary


__all__ = [
    "EXPECTED_RUNS",
    "FORMAL_SCHEMA",
    "SOURCE_COMMIT",
    "STATUS",
    "derive_summary",
]
