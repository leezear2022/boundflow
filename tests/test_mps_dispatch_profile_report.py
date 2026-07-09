import json

from scripts.report_mps_dispatch_profile import main


def test_mps_dispatch_profile_cpu_schema_smoke(capsys) -> None:
    rc = main(
        [
            "--device",
            "cpu",
            "--workloads",
            "permute_reshape_linear",
            "--scales",
            "smoke",
            "--policy",
            "auto",
            "--warmup",
            "0",
            "--iters",
            "1",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["meta"]["schema_version"] == "mps_dispatch_profile.v1"
    assert payload["summary"]["rows"] == 1
    assert payload["summary"]["unknown_materialization_total"] == 0
    row = payload["rows"][0]
    assert row["dispatch"]["materialization_total_bytes"] >= 0
    assert "cache" in row["dispatch"]
