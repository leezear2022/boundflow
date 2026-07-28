"""Inventory serialized PR-11/12 records available for Plan IR migration."""

# The private parse marker and CLI entrypoint are intentionally compact.
# pylint: disable=missing-function-docstring,too-few-public-methods

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Iterable, Sequence

from boundflow.planner.execution_candidate import (
    BACKEND_CANDIDATE_SCHEMA_VERSION,
)
from boundflow.planner.materialization import PLAN_SCHEMA_VERSION
from boundflow.planner.materialization_placement import (
    PLACEMENT_SCHEMA_VERSION,
)

LEGACY_SCHEMAS = {
    PLAN_SCHEMA_VERSION: "materialization_plan",
    PLACEMENT_SCHEMA_VERSION: "materialization_placement_plan",
    BACKEND_CANDIDATE_SCHEMA_VERSION: "execution_candidate",
}


def audit_legacy_records(root: Path) -> dict[str, object]:
    """Return a deterministic inventory without guessing missing typed context."""

    files = tuple(
        sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix in {".json", ".jsonl"}
        )
    )
    recognized: list[dict[str, str]] = []
    schema_counts: Counter[str] = Counter()
    parse_failures: list[dict[str, str]] = []
    objects_scanned = 0
    for path in files:
        for line_number, value in _read_objects(path):
            if isinstance(value, _ParseFailure):
                parse_failures.append(
                    {
                        "path": str(path),
                        "record": line_number,
                        "reason": value.reason,
                    }
                )
                continue
            for payload in _walk_objects(value):
                objects_scanned += 1
                schema = payload.get("schema_version")
                if isinstance(schema, str):
                    schema_counts[schema] += 1
                if isinstance(schema, str) and schema in LEGACY_SCHEMAS:
                    kind = LEGACY_SCHEMAS[schema]
                    recognized.append(
                        {
                            "path": str(path),
                            "record": line_number,
                            "schema_version": schema,
                            "legacy_kind": kind,
                            "migration_status": "requires_typed_bound_cost_context",
                        }
                    )
    return {
        "schema_version": "boundflow.plan-ir-legacy-record-audit/v1",
        "root": str(root),
        "json_jsonl_files_scanned": len(files),
        "json_objects_scanned": objects_scanned,
        "recognized_legacy_records": recognized,
        "recognized_legacy_record_count": len(recognized),
        "raw_legacy_records_present": bool(recognized),
        "observed_schema_counts": dict(sorted(schema_counts.items())),
        "parse_failures": parse_failures,
        "interpretation": (
            "zero recognized records means serialized PR-11/12 planner decisions "
            "are unavailable under this root; code-level adapters remain auditable"
        ),
    }


class _ParseFailure:
    def __init__(self, reason: str) -> None:
        self.reason = reason


def _read_objects(path: Path) -> Iterable[tuple[str, object | _ParseFailure]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        try:
            yield "document", json.loads(text)
        except json.JSONDecodeError as error:
            yield "document", _ParseFailure(str(error))
        return
    for index, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            yield str(index), json.loads(line)
        except json.JSONDecodeError as error:
            yield str(index), _ParseFailure(str(error))


def _walk_objects(value: object) -> Iterable[dict[str, object]]:
    if isinstance(value, dict):
        payload = {
            str(key): item for key, item in value.items() if isinstance(key, str)
        }
        yield payload
        for item in payload.values():
            yield from _walk_objects(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_objects(item)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    report = audit_legacy_records(args.root)
    encoded = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
