#!/usr/bin/env python3
"""Auditor-built coherent full-resign attack for S4-0 (assurance boundary probe).

Forges the abcrown commit string across raw+protocol, re-derives summary,
re-signs manifest. Question: does the offline self-check accept it?
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path("/home/lee/Codes/boundflow")
sys.path.insert(0, str(ROOT))

from scripts import replay_asplos27_s4_0_admission_stdlib as replay_tool  # noqa: E402

SRC = ROOT / "artifacts/asplos27-s4-admission/resnet2b-prop0-v1"
DST = Path("/tmp/boundflow-s4-audit-attack-src")


def main() -> None:
    if DST.exists():
        shutil.rmtree(DST)
    shutil.copytree(SRC, DST)

    rows = replay_tool._load_rows(DST / "raw/workers.jsonl")  # pylint: disable=protected-access
    forged = "f" * 40
    for row in rows:
        row["source"]["abcrown_commit"] = forged
        row["source_hash"] = replay_tool.canonical_hash(row["source"])
        row["raw_hash"] = replay_tool._hash_payload(  # pylint: disable=protected-access
            row, "raw_hash"
        )
    (DST / "raw/workers.jsonl").write_text(
        "".join(replay_tool.canonical(r) + "\n" for r in rows), encoding="utf-8"
    )

    protocol = replay_tool.load_json(DST / "protocol.json")
    protocol["source"]["abcrown_commit"] = forged
    protocol["source_hash"] = replay_tool.canonical_hash(protocol["source"])
    protocol["workers_jsonl_sha256"] = replay_tool.file_sha256(
        DST / "raw/workers.jsonl"
    )
    protocol["protocol_hash"] = replay_tool._hash_payload(  # pylint: disable=protected-access
        protocol, "protocol_hash"
    )
    (DST / "protocol.json").write_text(
        replay_tool.canonical(protocol) + "\n", encoding="utf-8"
    )

    negative = replay_tool.load_json(DST / "negative_registry.json")
    summary = replay_tool._derive_summary(rows, protocol, negative)  # pylint: disable=protected-access
    (DST / "summary.json").write_text(
        replay_tool.canonical(summary) + "\n", encoding="utf-8"
    )

    # re-sign manifest
    files = {
        p.relative_to(DST).as_posix(): replay_tool.file_sha256(p)
        for p in sorted(DST.rglob("*"))
        if p.is_file() and p.name != "manifest.json"
    }
    manifest = {
        "schema_version": replay_tool.MANIFEST_SCHEMA,
        "artifact_schema": replay_tool.ARTIFACT_SCHEMA,
        "files": files,
        "performance_claimed": False,
    }
    manifest["manifest_hash"] = replay_tool.canonical_hash(manifest)
    (DST / "manifest.json").write_text(
        replay_tool.canonical(manifest) + "\n", encoding="utf-8"
    )

    try:
        result = replay_tool.replay(DST)
    except Exception as exc:  # noqa: BLE001
        print(f"coherent full resign: REJECTED ({type(exc).__name__}: {exc})")
        return
    print(f"coherent full resign: ACCEPTED status={result['status']}")
    print(f"forged abcrown_commit now reads: {forged}")

    orig = hashlib.sha256(
        (SRC / "raw/workers.jsonl").read_bytes()
    ).hexdigest()
    print(f"original raw sha256 after attack: {orig}")


if __name__ == "__main__":
    main()
