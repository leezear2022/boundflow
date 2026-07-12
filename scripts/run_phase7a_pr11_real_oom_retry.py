#!/usr/bin/env python
"""Repeat the real-CUDA-OOM placement retry smoke in isolated processes."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CHILD = _REPO_ROOT / "scripts" / "smoke_phase7a_pr11_real_oom_retry.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=_REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _is_stable_retry(row: dict[str, object]) -> bool:
    stats = row.get("stats")
    if not isinstance(stats, dict):
        return False
    return bool(
        row.get("status") == "ok"
        and row.get("child_returncode") == 0
        and stats.get("oom_failures") == 1
        and stats.get("selected_index") == 1
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run isolated repetitions and write JSONL plus a hashed manifest."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--cap-mib", type=int, default=380)
    parser.add_argument("--spec-size", type=int, default=128)
    parser.add_argument("--domain-batch", type=int, default=32)
    args = parser.parse_args(argv)
    if min(args.repetitions, args.cap_mib, args.spec_size, args.domain_batch) <= 0:
        parser.error("all numeric arguments must be positive")

    rows: list[dict[str, object]] = []
    for repetition in range(args.repetitions):
        command = [
            sys.executable,
            str(_CHILD),
            "--cap-mib",
            str(args.cap_mib),
            "--spec-size",
            str(args.spec_size),
            "--domain-batch",
            str(args.domain_batch),
        ]
        completed = subprocess.run(
            command,
            cwd=_REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        stdout_lines = [line for line in completed.stdout.splitlines() if line.strip()]
        if not stdout_lines:
            payload: dict[str, object] = {
                "status": "missing_json",
                "error": "child emitted no stdout JSON",
            }
        else:
            try:
                payload = json.loads(stdout_lines[-1])
            except json.JSONDecodeError as error:
                payload = {
                    "status": "invalid_json",
                    "error": str(error),
                    "stdout_tail": stdout_lines[-1],
                }
        payload.update(
            {
                "repetition": repetition,
                "child_returncode": completed.returncode,
                "child_stderr_sha256": hashlib.sha256(
                    completed.stderr.encode()
                ).hexdigest(),
            }
        )
        rows.append(payload)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.out_dir / "raw.jsonl"
    raw_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    stable = all(_is_stable_retry(row) for row in rows)
    manifest = {
        "schema_version": "boundflow.pr11-real-oom-retry-manifest/v2",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": _git_value("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(_git_value("status", "--porcelain")),
        "command": {
            "cap_mib": args.cap_mib,
            "spec_size": args.spec_size,
            "domain_batch": args.domain_batch,
            "repetitions": args.repetitions,
        },
        "row_count": len(rows),
        "stable_dense_oom_structured_success": stable,
        "retry_strategy": "latency_topology_density_stratified_v3",
        "outputs": {"raw.jsonl": _sha256(raw_path)},
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if stable else 1


if __name__ == "__main__":
    raise SystemExit(main())
