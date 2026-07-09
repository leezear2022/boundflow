import json

import torch

from boundflow.runtime.bound_planner import (
    phase7a_capability_table_jsonable,
    phase7b_cost_model_rules_jsonable,
    plan_phase7b_shared_crown,
)
from boundflow.runtime import linear_operator as linear_op_mod
from scripts.bench_phase7a_shared_crown_path_attribution import (
    _build_case,
    _collect_counts,
    _run_variant_once,
    main,
)


def test_phase7a_pr11_relu_dense_baseline_matches_structured_bounds_and_counters() -> None:
    device = torch.device("cpu")
    dtype = torch.float32
    workloads = ("relu_heavy_mlp", "residual_relu_mlp", "concat_relu_mlp")

    for idx, workload in enumerate(workloads):
        _target, module, spec = _build_case(
            workload,
            device=device,
            dtype=dtype,
            profile="smoke",
            seed=idx,
        )
        structured = _run_variant_once(module, spec, variant="structured")
        baseline = _run_variant_once(module, spec, variant="dense_relu")
        structured_counts = _collect_counts(module, spec, variant="structured")
        baseline_counts = _collect_counts(module, spec, variant="dense_relu")

        assert torch.allclose(structured.lower, baseline.lower, atol=1e-5, rtol=1e-5)
        assert torch.allclose(structured.upper, baseline.upper, atol=1e-5, rtol=1e-5)
        assert structured_counts["relu_backward_calls"] > 0
        assert baseline_counts["dense_relu_barrier_calls"] > 0
        assert structured_counts["dense_relu_barrier_calls"] == 0

    relu_target, relu_module, relu_spec = _build_case(
        "relu_heavy_mlp",
        device=device,
        dtype=dtype,
        profile="smoke",
        seed=7,
    )
    assert relu_target == "relu_barrier"
    relu_counts = _collect_counts(relu_module, relu_spec, variant="structured")
    assert relu_counts["split_pos_neg_dense_total"] == 0
    assert relu_counts["split_pos_neg_dense_by_op"].get("RightMatmulLinearOperator", 0) == 0
    assert relu_counts["split_pos_neg_dense_by_op"].get("SliceInputLinearOperator", 0) == 0

    concat_target, concat_module, concat_spec = _build_case(
        "concat_relu_mlp",
        device=device,
        dtype=dtype,
        profile="smoke",
        seed=11,
    )
    assert concat_target == "relu_barrier"
    concat_counts = _collect_counts(concat_module, concat_spec, variant="structured")
    assert concat_counts["split_pos_neg_dense_by_op"].get("RightMatmulLinearOperator", 0) == 0
    assert concat_counts["split_pos_neg_dense_by_op"].get("SliceInputLinearOperator", 0) == 0


def test_phase7a_pr15_operator_attribution_is_opt_in_and_side_effect_free() -> None:
    _target, module, spec = _build_case(
        "relu_heavy_mlp",
        device=torch.device("cpu"),
        dtype=torch.float32,
        profile="smoke",
        seed=17,
    )
    without_attr = _run_variant_once(module, spec, variant="structured")
    with linear_op_mod.collect_operator_attribution(path_kind="structured", phase="structured_execution") as trace:
        with_attr = _run_variant_once(module, spec, variant="structured")

    assert torch.allclose(with_attr.lower, without_attr.lower, atol=1e-5, rtol=1e-5)
    assert torch.allclose(with_attr.upper, without_attr.upper, atol=1e-5, rtol=1e-5)

    payload = trace.to_jsonable()
    assert payload["schema_version"] == 1
    assert payload["path_kind"] == "structured"
    assert payload["relu_pullback"]["total_calls"] > 0


def test_phase7a_pr11_layout_dense_baseline_matches_structured_bounds_and_counters() -> None:
    compare_target, module, spec = _build_case(
        "permute_reshape_linear",
        device=torch.device("cpu"),
        dtype=torch.float32,
        profile="smoke",
        seed=0,
    )
    structured = _run_variant_once(module, spec, variant="structured")
    baseline = _run_variant_once(module, spec, variant="dense_layout")
    structured_counts = _collect_counts(module, spec, variant="structured")
    baseline_counts = _collect_counts(module, spec, variant="dense_layout")

    assert compare_target == "layout_only"
    assert torch.allclose(structured.lower, baseline.lower, atol=1e-5, rtol=1e-5)
    assert torch.allclose(structured.upper, baseline.upper, atol=1e-5, rtol=1e-5)
    assert structured_counts["permute_backward_calls"] > 0
    assert baseline_counts["dense_layout_barrier_calls"] > 0
    assert structured_counts["dense_layout_barrier_calls"] == 0


def test_phase7a_pr11_bench_script_schema_smoke(capsys) -> None:
    rc = main(
        [
            "--device",
            "cpu",
            "--profile",
            "smoke",
            "--workloads",
            "all",
            "--warmup",
            "1",
            "--iters",
            "1",
        ]
    )
    assert rc == 0

    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert "meta" in payload
    assert "rows" in payload
    assert payload["meta"]["schema_version"] == "phase7a_shared_crown_path_attribution.v1"
    assert payload["meta"]["profile"] == "smoke"
    assert payload["meta"]["device"] == "cpu"
    assert "device_meta" in payload["meta"]
    assert "mps_built" in payload["meta"]["device_meta"]
    assert payload["meta"]["final_concretization_policy"] == "structured"
    assert payload["meta"]["capability_table"]["schema_version"] == 1
    assert payload["meta"]["cost_model_rules"]["schema_version"] == 1
    assert "RightMatmulLinearOperator" in payload["meta"]["capability_table"]["operators"]
    assert len(payload["rows"]) == 4

    required_count_keys = [
        "relu_backward_calls",
        "permute_backward_calls",
        "split_pos_neg_dense_total",
        "split_pos_neg_dense_by_op",
        "dense_relu_barrier_calls",
        "dense_layout_barrier_calls",
    ]
    for row in payload["rows"]:
        for key in (
            "workload",
            "compare_target",
            "structured_ms_p50",
            "baseline_ms_p50",
            "speedup",
            "counts_structured",
            "counts_baseline",
            "planner_decision",
        ):
            assert key in row, key
        assert row["planner_decision"]["schema_version"] == 1
        assert row["planner_decision"]["final_concretization_policy"] == "structured"
        assert row["compare_target"] in {"relu_barrier", "layout_only"}
        for counts_key in required_count_keys:
            assert counts_key in row["counts_structured"], counts_key
            assert counts_key in row["counts_baseline"], counts_key
        for counts, path_kind in (
            (row["counts_structured"], "structured"),
            (row["counts_baseline"], "dense_baseline"),
        ):
            attr = counts["operator_attribution"]
            assert attr["schema_version"] == 1
            assert attr["path_kind"] == path_kind
            assert "by_phase" in attr["materialization"]
            assert "by_op" in attr["materialization"]
            assert "by_reason" in attr["materialization"]
            assert "by_reason" in attr["fallback"]
            assert "cache" in attr
            assert "hits" in attr["cache"]
            assert "misses" in attr["cache"]
            assert "by_op" in attr["cache"]
            assert "by_reason" in attr["cache"]
            assert attr["materialization"]["by_reason"].get("unknown_materialization", {}).get("calls", 0) == 0
            for event in attr["materialization"]["events"]:
                for key in ("op", "shape", "numel", "bytes", "phase", "reason"):
                    assert key in event, key
                assert isinstance(event["shape"], list)
                assert event["reason"] != ""


def test_phase7a_pr18_capability_table_and_auto_planner_policy() -> None:
    table = phase7a_capability_table_jsonable()
    assert table["schema_version"] == 1
    right = table["operators"]["RightMatmulLinearOperator"]
    assert right["relu_pullback"] == "exact_requires_dense_sign_split"
    assert right["planner_action"] == "cached_dense_do_not_fake_structured_sign_split"

    rules = phase7b_cost_model_rules_jsonable()
    assert rules["schema_version"] == 1
    assert any(rule["workload"] == "permute_reshape_linear" for rule in rules["rules"])

    relu_decision = plan_phase7b_shared_crown(
        compare_target="relu_barrier",
        workload="relu_heavy_mlp",
        scale_id="smoke",
        device="cpu",
        requested_final_concretization_policy="auto",
    )
    assert relu_decision.final_concretization_policy == "structured"
    assert relu_decision.use_dense_cache is True
    assert "right_matmul_cached_dense_exact_sign_split" in relu_decision.selected_rules

    layout_smoke_decision = plan_phase7b_shared_crown(
        compare_target="layout_only",
        workload="permute_reshape_linear",
        scale_id="smoke",
        device="cpu",
        requested_final_concretization_policy="auto",
    )
    assert layout_smoke_decision.final_concretization_policy == "dense_barrier"
    assert layout_smoke_decision.use_dense_cache is True
    assert "layout_only_final_dense_barrier" in layout_smoke_decision.selected_rules

    layout_small_decision = plan_phase7b_shared_crown(
        compare_target="layout_only",
        workload="permute_reshape_linear",
        scale_id="small",
        device="cpu",
        requested_final_concretization_policy="auto",
    )
    assert layout_small_decision.planner == "phase7b_cost_model_v1"
    assert layout_small_decision.final_concretization_policy == "structured"
    assert layout_small_decision.confidence == "high"


def test_phase7a_pr18_bench_script_auto_planner_schema_smoke(capsys) -> None:
    rc = main(
        [
            "--device",
            "cpu",
            "--profile",
            "smoke",
            "--workloads",
            "relu_heavy_mlp,permute_reshape_linear",
            "--warmup",
            "1",
            "--iters",
            "1",
            "--final-concretization-policy",
            "auto",
        ]
    )
    assert rc == 0

    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["meta"]["final_concretization_policy"] == "auto"
    by_workload = {row["workload"]: row["planner_decision"] for row in payload["rows"]}
    assert by_workload["relu_heavy_mlp"]["final_concretization_policy"] == "structured"
    assert by_workload["permute_reshape_linear"]["final_concretization_policy"] == "dense_barrier"


def test_phase7a_pr15_right_matmul_relu_pullback_materialization_is_attributed() -> None:
    _target, module, spec = _build_case(
        "relu_heavy_mlp",
        device=torch.device("cpu"),
        dtype=torch.float32,
        profile="smoke",
        seed=23,
    )
    counts = _collect_counts(module, spec, variant="structured")
    attr = counts["operator_attribution"]
    right_events = [
        event
        for event in attr["materialization"]["events"]
        if event["op"] == "RightMatmulLinearOperator"
        and event["reason"] == "right_matmul_exact_sign_split_required"
    ]
    if right_events:
        assert all(event["numel"] > 0 for event in right_events)
        assert all(event["bytes"] > 0 for event in right_events)
    assert counts["split_pos_neg_dense_total"] == 0


def test_phase7a_pr16_dense_cache_is_run_local_and_semantics_preserving() -> None:
    _target, module, spec = _build_case(
        "relu_heavy_mlp",
        device=torch.device("cpu"),
        dtype=torch.float32,
        profile="smoke",
        seed=29,
    )
    with linear_op_mod.operator_dense_cache(enabled=False):
        without_cache = _run_variant_once(module, spec, variant="structured")
    with linear_op_mod.operator_dense_cache(enabled=True):
        with_cache = _run_variant_once(module, spec, variant="structured")

    assert torch.allclose(with_cache.lower, without_cache.lower, atol=1e-5, rtol=1e-5)
    assert torch.allclose(with_cache.upper, without_cache.upper, atol=1e-5, rtol=1e-5)


def test_phase7a_pr16_dense_cache_records_right_matmul_hits() -> None:
    torch.manual_seed(0)
    base = linear_op_mod.DenseLinearOperator(torch.randn(2, 3, 4, dtype=torch.float32))
    op = base.matmul_right(torch.randn(4, 5, dtype=torch.float32))

    with linear_op_mod.collect_operator_attribution(path_kind="unit", phase="structured_execution") as trace:
        with linear_op_mod.operator_dense_cache(enabled=True):
            first = op.to_dense()
            second = op.to_dense()

    assert torch.allclose(first, second, atol=0.0, rtol=0.0)
    payload = trace.to_jsonable()
    right_cache = payload["cache"]["by_op"]["RightMatmulLinearOperator"]
    assert right_cache["misses"] == 1
    assert right_cache["hits"] == 1
    right_mat = payload["materialization"]["by_op"]["RightMatmulLinearOperator"]
    assert right_mat["calls"] == 1


def test_phase7a_pr16_cache_enabled_materialization_is_not_higher_than_disabled() -> None:
    _target, module, spec = _build_case(
        "concat_relu_mlp",
        device=torch.device("cpu"),
        dtype=torch.float32,
        profile="smoke",
        seed=31,
    )
    enabled = _collect_counts(module, spec, variant="structured", use_dense_cache=True)
    disabled = _collect_counts(module, spec, variant="structured", use_dense_cache=False)

    enabled_reason = enabled["operator_attribution"]["materialization"]["by_reason"]
    disabled_reason = disabled["operator_attribution"]["materialization"]["by_reason"]
    for reason in ("right_matmul_exact_sign_split_required", "final_bound_concretization"):
        assert enabled_reason[reason]["calls"] <= disabled_reason[reason]["calls"]
        assert enabled_reason[reason]["total_bytes"] <= disabled_reason[reason]["total_bytes"]
    right_cache = enabled["operator_attribution"]["cache"]["by_op"]["RightMatmulLinearOperator"]
    assert right_cache["misses"] > 0
    assert right_cache["hits"] > 0
    assert enabled["operator_attribution"]["materialization"]["by_reason"].get("unknown_materialization", {}).get("calls", 0) == 0
