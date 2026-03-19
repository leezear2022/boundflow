import json


def test_phase6h_bench_outputs_8_rows_and_fixed_counters(capsys) -> None:
    from scripts.bench_phase6h_bab_e2e_time_to_verify import main

    rc = main(
        [
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--workload",
            "1d_relu",
            "--oracle",
            "alpha_beta",
            "--steps",
            "0",
            "--max-nodes",
            "64",
            "--node-batch-size",
            "8",
            "--warmup",
            "1",
            "--iters",
            "1",
            "--enable-node-eval-cache",
            "0,1",
            "--use-branch-hint",
            "0,1",
            "--enable-batch-infeasible-prune",
            "0,1",
        ]
    )
    assert rc == 0

    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert "rows" in payload
    assert "meta" in payload
    assert len(payload["rows"]) == 8

    required_meta = [
        "schema_version",
        "device",
        "device_name",
        "dtype",
        "node_batch_size",
        "specs",
        "max_nodes",
        "timer",
        "torch_version",
        "git_sha",
        "timeout_s",
    ]
    for k in required_meta:
        assert k in payload["meta"], k

    required_counters = [
        "oracle_calls",
        "forward_trace_calls",
        "cache_hits",
        "cache_misses",
        "cache_hit_rate",
        "pruned_infeasible_count",
        "evaluated_nodes_count",
    ]
    for row in payload["rows"]:
        for k in (
            "batch_ms_p50",
            "batch_ms_p90",
            "serial_ms_p50",
            "serial_ms_p90",
            "speedup",
            "speedup_p90",
            "batch_runs_count",
            "batch_valid_runs_count",
            "batch_timeouts_count",
            "serial_runs_count",
            "serial_valid_runs_count",
            "serial_timeouts_count",
        ):
            assert k in row, k
        assert "counts_batch" in row
        assert "counts_serial" in row
        for k in required_counters:
            assert k in row["counts_batch"], k
            assert k in row["counts_serial"], k
