#!/usr/bin/env python
"""Aggregate independent PR-11 exhaustive profiles by query/pattern median."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
from typing import Any, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_SCHEMA_VERSION = "boundflow.pr11-barrier-placement-profile/v3"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=_REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _load(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"empty profile: {path}")
    if {row.get("schema_version") for row in rows} != {PROFILE_SCHEMA_VERSION}:
        raise ValueError(f"profile schema mismatch: {path}")
    if any(row.get("status") != "ok" for row in rows):
        raise ValueError(f"profile contains non-ok rows: {path}")
    identities = [str(row["query_id"]) for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError(f"profile contains duplicate query ids: {path}")
    return rows


def aggregate_profiles(
    profiles: Sequence[tuple[Path, Sequence[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    """Validate aligned profile coverage and aggregate timing/memory targets."""

    if len(profiles) < 2:
        raise ValueError("profile aggregation requires at least two replicates")
    indexed = [{str(row["query_id"]): row for row in rows} for _path, rows in profiles]
    identities = set(indexed[0])
    if any(set(rows) != identities for rows in indexed[1:]):
        raise ValueError("profile replicates do not have identical query coverage")

    output: list[dict[str, Any]] = []
    for query_id in sorted(identities):
        replicates = [rows[query_id] for rows in indexed]
        first = replicates[0]
        for row in replicates[1:]:
            for key in (
                "workload",
                "method",
                "spec_size",
                "domain_batch_size",
                "barrier_ids",
                "static_barriers",
                "placement",
            ):
                if row[key] != first[key]:
                    raise ValueError(f"replicate metadata mismatch: {query_id}:{key}")
        latency = [
            float(row["timing_trace_off"]["latency_ms_median"]) for row in replicates
        ]
        peak_allocated = [
            int(row["timing_trace_off"]["peak_cuda_allocated_bytes"])
            for row in replicates
        ]
        peak_reserved = [
            int(row["timing_trace_off"]["peak_cuda_reserved_bytes"])
            for row in replicates
        ]
        row = copy.deepcopy(first)
        row["run_id"] = "pr11-replicated-profile-aggregate"
        row["timing_trace_off"]["latency_ms_median"] = statistics.median(latency)
        row["timing_trace_off"]["latency_ms_p90"] = _percentile(latency, 0.9)
        row["timing_trace_off"]["peak_cuda_allocated_bytes"] = round(
            statistics.median(peak_allocated)
        )
        row["timing_trace_off"]["peak_cuda_reserved_bytes"] = round(
            statistics.median(peak_reserved)
        )
        row["replicate_aggregation"] = {
            "kind": "independent_profile_median_v1",
            "replicate_count": len(replicates),
            "latency_ms_min": min(latency),
            "latency_ms_max": max(latency),
            "latency_ms_p90": _percentile(latency, 0.9),
            "peak_allocated_bytes_min": min(peak_allocated),
            "peak_allocated_bytes_max": max(peak_allocated),
            "sources": [str(path) for path, _rows in profiles],
        }
        output.append(row)
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Write aggregate JSONL and a source-hashed manifest."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    profiles = [(path, _load(path)) for path in args.profile]
    rows = aggregate_profiles(profiles)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.out_dir / "raw.jsonl"
    raw_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "boundflow.pr11-replicated-profile-manifest/v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": _git_value("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(_git_value("status", "--porcelain")),
        "aggregation": "independent_profile_median_v1",
        "replicate_count": len(profiles),
        "sources": [
            {"path": str(path), "sha256": _sha256(path)} for path, _rows in profiles
        ],
        "row_count": len(rows),
        "outputs": {"raw.jsonl": _sha256(raw_path)},
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
