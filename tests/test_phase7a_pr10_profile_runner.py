import csv
import json
from pathlib import Path

from scripts.profile_phase7a_pr10_materialization import main


def test_pr10_profile_runner_writes_trace_on_and_timing_off_evidence(
    tmp_path: Path,
) -> None:
    rc = main(
        [
            "--run-id",
            "smoke",
            "--out-root",
            str(tmp_path),
            "--device",
            "cpu",
            "--workloads",
            "mlp_chain",
            "--methods",
            "CROWN",
            "--spec-sizes",
            "1",
            "--domain-batches",
            "1",
            "--optimization-steps",
            "0",
            "--warmup",
            "0",
            "--repeats",
            "1",
        ]
    )
    assert rc == 0
    profile = tmp_path / "smoke" / "profile"
    row = json.loads((profile / "raw.jsonl").read_text(encoding="utf-8"))
    assert row["status"] == "ok"
    assert row["domain_source"] == "fixed_batch_replay"
    assert row["timing_trace_off"]["trace_enabled"] is False
    assert row["trace_on"]["schema_version"] == "boundflow.materialization/v1"
    assert row["trace_on"]["materialization"]["event_count"] == 4
    assert row["correctness"] == {"finite": True, "lower_le_upper": True}

    with (profile / "normalized.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["workload"] == "mlp_chain"
    assert int(rows[0]["event_count"]) == 4

    manifest = json.loads((profile / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["rows"] == 1
    assert manifest["status_counts"]["ok"] == 1


def test_pr10_profile_runner_alpha_beta_uses_per_domain_split_state(
    tmp_path: Path,
) -> None:
    rc = main(
        [
            "--run-id",
            "alpha-beta-smoke",
            "--out-root",
            str(tmp_path),
            "--device",
            "cpu",
            "--workloads",
            "mlp_chain",
            "--methods",
            "alpha-beta-CROWN",
            "--spec-sizes",
            "1",
            "--domain-batches",
            "2",
            "--optimization-steps",
            "0",
            "--warmup",
            "0",
            "--repeats",
            "1",
        ]
    )
    assert rc == 0
    raw = tmp_path / "alpha-beta-smoke" / "profile" / "raw.jsonl"
    row = json.loads(raw.read_text(encoding="utf-8"))
    assert row["status"] == "ok"
    assert row["domain_batch"] == 2
    assert row["trace_on"]["state_bytes"]["beta_state_bytes"] > 0
    assert all(event["beta_related"] for event in row["trace_on"]["events"])
