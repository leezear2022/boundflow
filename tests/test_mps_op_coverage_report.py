import json

import pytest
import torch

from scripts.report_mps_op_coverage import main


@pytest.mark.skipif(
    not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    reason="MPS is not available",
)
def test_mps_op_coverage_report_schema_smoke(capsys) -> None:
    rc = main(
        [
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
    assert payload["meta"]["schema_version"] == "mps_op_coverage.v1"
    assert payload["meta"]["device"] == "mps"
    assert payload["meta"]["device_meta"]["mps_available"] is True
    assert payload["summary"]["ok"] == 1
    assert payload["summary"]["fail"] == 0
    row = payload["rows"][0]
    assert row["status"] == "ok"
    assert row["planner_decision"]["final_concretization_policy"] in {"structured", "dense_barrier"}
