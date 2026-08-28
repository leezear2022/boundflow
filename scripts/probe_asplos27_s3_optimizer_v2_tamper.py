#!/usr/bin/env python3
"""Apply ten outer-resigned attacks to an S3 v2 optimizer artifact."""

# pylint: disable=missing-function-docstring,too-many-locals,too-many-branches
# pylint: disable=line-too-long

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import tempfile

from scripts import run_asplos27_s3_optimizer_artifact as v1
from scripts import run_asplos27_s3_optimizer_artifact_v2 as v2


def _write_json(path: Path, value: object) -> None:
    path.write_text(v1.canonical(value) + "\n", encoding="utf-8")


def _load_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(v1.canonical(row) + "\n" for row in rows), encoding="utf-8")


def probe(source: Path) -> dict[str, object]:
    cases = []
    names = (
        "latency",
        "step-lower",
        "optimizer-moment",
        "replicate-index",
        "execution-counter",
        "estimator",
        "protocol-gate",
        "claim-flag",
        "code-revision",
        "summary-status",
    )
    for name in names:
        with tempfile.TemporaryDirectory(prefix=f"s3-v2-tamper-{name}-") as directory:
            target = Path(directory) / "artifact"
            shutil.copytree(source, target)
            rows_path = target / "raw/workers.jsonl"
            rows = _load_rows(rows_path)
            protocol = v1.load_json(target / "protocol.json")
            summary = v1.load_json(target / "summary.json")
            if name == "latency":
                rows[0]["latency_ns"]["P"] = [  # type: ignore[index]
                    max(1, value // 2)
                    for value in rows[0]["latency_ns"]["P"]  # type: ignore[index]
                ]
            elif name == "step-lower":
                rows[0]["semantic"]["P"][0]["lower"]["values"][0] += 1.0  # type: ignore[index]
            elif name == "optimizer-moment":
                rows[0]["semantic"]["P"][0]["optimizer_exp_avg"]["values"][0] += 1.0  # type: ignore[index]
            elif name == "replicate-index":
                rows[0]["replicate_index"] = 2
            elif name == "execution-counter":
                rows[0]["candidate_receipt"]["host_policy_cut_count"] = 9  # type: ignore[index]
            elif name == "estimator":
                protocol["headline_estimator"] = "best-of-three"
            elif name == "protocol-gate":
                protocol["p_over_n_worst_min"] = 0.0
            elif name == "claim-flag":
                rows[0]["same_solver_claimed"] = True
            elif name == "code-revision":
                protocol["code_revision"] = copy.deepcopy(protocol["code_revision"])
                protocol["code_revision"][next(iter(protocol["code_revision"]))] = "0" * 64  # type: ignore[index]
            else:
                summary["status"] = "validated-s3-v2-forged"
            _write_rows(rows_path, rows)
            _write_json(target / "protocol.json", protocol)
            _write_json(target / "summary.json", summary)
            v2._write_manifest(target)  # pylint: disable=protected-access
            rejected = False
            error = ""
            try:
                v2.replay(target, require_validated_3x=False)
            except (ValueError, TypeError, KeyError, OverflowError) as exception:
                rejected = True
                error = str(exception)
            cases.append(
                {
                    "name": name,
                    "outer_resigned": True,
                    "rejected": rejected,
                    "error": error,
                }
            )
    report = {
        "schema_version": "boundflow.asplos27-s3-optimizer-tamper/v2",
        "case_count": len(cases),
        "rejected_count": sum(1 for row in cases if row["rejected"] is True),
        "rows": cases,
        "performance_claimed": False,
    }
    if report["case_count"] != 10 or report["rejected_count"] != 10:
        accepted = [row["name"] for row in cases if not row["rejected"]]
        raise RuntimeError(f"S3 v2 tamper probe accepted cases: {accepted}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = probe(args.artifact.resolve())
    payload = v1.canonical(report) + "\n"
    if args.output is not None:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
