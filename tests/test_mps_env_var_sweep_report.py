import json

from scripts.report_mps_env_var_sweep import main


def test_mps_env_var_sweep_dry_run_schema(capsys) -> None:
    rc = main(
        [
            "--cases",
            "default,prefer_metal,fast_math,both",
            "--workloads",
            "permute_reshape_linear",
            "--scales",
            "smoke",
            "--policies",
            "auto",
            "--warmup",
            "0",
            "--iters",
            "1",
            "--set-kmp-duplicate-lib-ok",
            "--omp-num-threads",
            "1",
            "--dry-run",
        ]
    )
    assert rc == 0

    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["meta"]["schema_version"] == "mps_env_var_sweep.v1"
    assert payload["summary"]["dry_run"] == 4
    assert payload["summary"]["fail"] == 0

    by_case = {result["case"]: result for result in payload["results"]}
    assert by_case["default"]["env"]["PYTORCH_MPS_PREFER_METAL"] is None
    assert by_case["default"]["env"]["PYTORCH_MPS_FAST_MATH"] is None
    assert by_case["prefer_metal"]["env"]["PYTORCH_MPS_PREFER_METAL"] == "1"
    assert by_case["prefer_metal"]["env"]["PYTORCH_MPS_FAST_MATH"] is None
    assert by_case["fast_math"]["env"]["PYTORCH_MPS_PREFER_METAL"] is None
    assert by_case["fast_math"]["env"]["PYTORCH_MPS_FAST_MATH"] == "1"
    assert by_case["both"]["env"]["PYTORCH_MPS_PREFER_METAL"] == "1"
    assert by_case["both"]["env"]["PYTORCH_MPS_FAST_MATH"] == "1"

    for result in payload["results"]:
        assert result["status"] == "dry_run"
        assert result["env"]["PYTORCH_ENABLE_MPS_FALLBACK"] is None
        assert result["env"]["KMP_DUPLICATE_LIB_OK"] == "TRUE"
        assert result["env"]["OMP_NUM_THREADS"] == "1"
        command = " ".join(result["command"])
        assert "bench_phase7b_crossover_matrix.py" in command
        assert "--device mps" in command
