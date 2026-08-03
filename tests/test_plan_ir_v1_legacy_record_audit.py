"""Contracts for locating serialized PR-11/12 migration inputs."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_plan_ir_v1_legacy_records import audit_legacy_records


def test_legacy_record_audit_finds_nested_known_schemas(tmp_path: Path) -> None:
    (tmp_path / "records.json").write_text(
        json.dumps(
            {
                "schema_version": "wrapper/v1",
                "records": [
                    {"schema_version": "boundflow.materialization_plan/v1"},
                    {"schema_version": "boundflow.backend_candidate/v1.0"},
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "records.jsonl").write_text(
        '{"schema_version":"boundflow.materialization_placement/v1"}\n'
        '{"schema_version":"other/v1"}\n',
        encoding="utf-8",
    )
    report = audit_legacy_records(tmp_path)
    assert report["recognized_legacy_record_count"] == 3
    assert report["raw_legacy_records_present"] is True
    records = report["recognized_legacy_records"]
    assert isinstance(records, list)
    assert {record["legacy_kind"] for record in records} == {
        "materialization_plan",
        "materialization_placement_plan",
        "execution_candidate",
    }


def test_legacy_record_audit_reports_absent_raw_records(tmp_path: Path) -> None:
    (tmp_path / "unrelated.json").write_text(
        '{"schema_version":"unrelated/v1"}\n',
        encoding="utf-8",
    )
    report = audit_legacy_records(tmp_path)
    assert report["recognized_legacy_record_count"] == 0
    assert report["raw_legacy_records_present"] is False
