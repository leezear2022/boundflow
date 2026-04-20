import json

import torch

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
        ):
            assert key in row, key
        assert row["compare_target"] in {"relu_barrier", "layout_only"}
        for counts_key in required_count_keys:
            assert counts_key in row["counts_structured"], counts_key
            assert counts_key in row["counts_baseline"], counts_key
