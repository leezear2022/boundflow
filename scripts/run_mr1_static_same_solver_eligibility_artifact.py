#!/usr/bin/env python3
"""Generate or replay the MR1 static same-solver eligibility artifact."""

# pylint: disable=too-many-locals,too-many-statements,duplicate-code,wrong-import-position
# pylint: disable=missing-function-docstring

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/measurement-recovery/mr1-static-same-solver-eligibility-v1"
INPUTS = {
    "activation_calls.jsonl": (
        ROOT / "artifacts/rvir/rvir-cpu-correctness-v2-20260803/activation_calls.jsonl",
        "b8dc6652d487dbe3fd2a00933443a1f20221221babc22ae2f4f4f32a58462c4d",
    ),
    "rvir_manifest.json": (
        ROOT / "artifacts/rvir/rvir-cpu-correctness-v2-20260803/manifest.json",
        "0f8927c5b1909b7a0b671f1c2cda28835956ca259ff088c358fd0121f96979f6",
    ),
    "inventory.json": (
        ROOT
        / "artifacts/fsg2-rvir-v3/resnet2b-production-state-inventory-v2/inventory.json",
        "ab0595bb002b79d80be8b78abd7a795a8aa634b17c9a8524df5a1b9b5fe19e06",
    ),
    "inventory_manifest.json": (
        ROOT
        / "artifacts/fsg2-rvir-v3/resnet2b-production-state-inventory-v2/manifest.json",
        "a4bc22f52163b4b668a4753587c50a25e3d45e4fe484f07b226f714b6f31fde3",
    ),
    "b3_manifest.json": (
        ROOT / "artifacts/fsg4-b3-same-solver-timing/resnet2b-prop0-v1/manifest.json",
        "d88eeecafcd6a7a9394cdf9654962a36497b1c7afd15d7862048b1c3ccd7db4a",
    ),
    "cibc_manifest.json": (
        ROOT / "artifacts/cibc-ibp-horizontal-formal/resnet2b-prop0-v1/manifest.json",
        "b260fa6a49e77e3b8b1ff9502e6cc6bc27c6ddfcd5c01ba3bcd73b249d6dd807",
    ),
}
CODE_PATHS = (
    "boundflow/runtime/mr1_static_same_solver_eligibility.py",
    "scripts/run_mr1_static_same_solver_eligibility_artifact.py",
    "scripts/probe_mr1_static_same_solver_eligibility_tamper.py",
    "gemini_doc/BOUNDFLOW_MR1_STATIC_SAME_SOLVER_ELIGIBILITY_AUDIT_PLAN_2026_08_26.md",
)

from boundflow.runtime.mr1_static_same_solver_eligibility import (  # noqa: E402
    EXPECTED_CALL_COUNT,
    REASON_ORDER,
    TARGET_MODEL_HASH,
    TARGET_TOPOLOGY,
    canonical_hash,
    classify_call,
    derive_coverage,
    derive_summary,
)


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("MR1 JSON root differs")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError("MR1 JSONL row differs")
        rows.append(value)
    return rows


def _write(path: Path, value: object) -> None:
    path.write_text(_canonical(value) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text("".join(_canonical(row) + "\n" for row in rows), encoding="utf-8")


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _protocol(revision: str, source: Path) -> dict[str, object]:
    input_hashes = {name: _file_hash(source / name) for name in sorted(INPUTS)}
    expected = {name: digest for name, (_, digest) in INPUTS.items()}
    if input_hashes != expected:
        raise ValueError("MR1 frozen input digest differs")
    result: dict[str, object] = {
        "schema_version": "boundflow.mr1-static-same-solver-eligibility-protocol/v1",
        "source_revision": revision,
        "input_sha256": input_hashes,
        "activation_call_count": EXPECTED_CALL_COUNT,
        "target_model_hash": TARGET_MODEL_HASH,
        "target_topology": TARGET_TOPOLOGY,
        "reason_order": list(REASON_ORDER),
        "read_only": True,
        "solver_executed": False,
        "gpu_executed": False,
        "timing_collected": False,
        "same_solver_timing_open": False,
        "r2_open": False,
        "performance_claimed": False,
        "code_revision": {path: _file_hash(ROOT / path) for path in CODE_PATHS},
    }
    result["protocol_hash"] = canonical_hash(result)
    return result


def _derive(
    source: Path,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    rows = _jsonl(source / "activation_calls.jsonl")
    coverage = derive_coverage(rows)
    ledger = [
        classify_call(row)
        for row in rows
        if row["query"]["model_structure_hash"] == TARGET_MODEL_HASH
    ]
    summary = derive_summary(coverage=coverage, target_ledger=ledger)
    return ledger, coverage, summary


def generate(output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"MR1 artifact exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix="mr1-static-eligibility-", dir=output.parent)
    )
    try:
        source = temporary / "source"
        source.mkdir()
        for name, (path, digest) in INPUTS.items():
            if _file_hash(path) != digest:
                raise ValueError(f"MR1 source input differs: {name}")
            shutil.copyfile(path, source / name)
        protocol = _protocol(_git("rev-parse", "HEAD"), source)
        ledger, coverage, summary = _derive(source)
        _write(temporary / "protocol.json", protocol)
        _write_jsonl(temporary / "ledger.jsonl", ledger)
        _write(temporary / "coverage.json", coverage)
        _write(temporary / "summary.json", summary)
        (temporary / "replay_stdout.txt").write_text(
            "MR1 replay PASS: "
            f"eligible={summary['eligible_target_model_call_count']}/"
            f"{summary['target_model_call_count']} verdict={summary['verdict']}\n",
            encoding="utf-8",
        )
        files = {
            str(path.relative_to(temporary)): _file_hash(path)
            for path in sorted(temporary.rglob("*"))
            if path.is_file()
        }
        manifest: dict[str, object] = {
            "schema_version": "boundflow.mr1-static-same-solver-eligibility-manifest/v1",
            "source_revision": protocol["source_revision"],
            "protocol_hash": protocol["protocol_hash"],
            "coverage_hash": coverage["coverage_hash"],
            "summary_hash": summary["summary_hash"],
            "files": files,
        }
        manifest["manifest_hash"] = canonical_hash(manifest)
        _write(temporary / "manifest.json", manifest)
        replay(temporary)
        temporary.rename(output)
        return summary
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def replay(artifact: Path) -> dict[str, object]:
    manifest = _json(artifact / "manifest.json")
    unsigned = dict(manifest)
    manifest_hash = unsigned.pop("manifest_hash", None)
    schema_ok = (
        manifest.get("schema_version")
        == "boundflow.mr1-static-same-solver-eligibility-manifest/v1"
    )
    if not schema_ok or manifest_hash != canonical_hash(unsigned):
        raise ValueError("MR1 manifest differs")
    files = manifest.get("files")
    if not isinstance(files, dict) or any(
        _file_hash(artifact / str(name)) != digest for name, digest in files.items()
    ):
        raise ValueError("MR1 file digest differs")
    protocol = _json(artifact / "protocol.json")
    protocol_unsigned = dict(protocol)
    protocol_hash = protocol_unsigned.pop("protocol_hash", None)
    if protocol_hash != canonical_hash(
        protocol_unsigned
    ) or protocol_hash != manifest.get("protocol_hash"):
        raise ValueError("MR1 protocol differs")
    if protocol_unsigned.get("input_sha256") != {
        name: digest for name, (_, digest) in INPUTS.items()
    }:
        raise ValueError("MR1 protocol input binding differs")
    source = artifact / "source"
    if {
        name: _file_hash(source / name) for name in sorted(INPUTS)
    } != protocol_unsigned["input_sha256"]:
        raise ValueError("MR1 copied input digest differs")
    expected_ledger, expected_coverage, expected_summary = _derive(source)
    if _jsonl(artifact / "ledger.jsonl") != expected_ledger:
        raise ValueError("MR1 ledger derivation differs")
    if _json(artifact / "coverage.json") != expected_coverage:
        raise ValueError("MR1 coverage derivation differs")
    if _json(artifact / "summary.json") != expected_summary:
        raise ValueError("MR1 summary derivation differs")
    if (
        manifest.get("coverage_hash") != expected_coverage["coverage_hash"]
        or manifest.get("summary_hash") != expected_summary["summary_hash"]
    ):
        raise ValueError("MR1 manifest semantic binding differs")
    return expected_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args()
    if args.replay is not None:
        summary = replay(args.replay.resolve())
        print(
            "MR1 replay PASS: "
            f"eligible={summary['eligible_target_model_call_count']}/"
            f"{summary['target_model_call_count']} verdict={summary['verdict']}"
        )
        return
    summary = generate(args.output.resolve())
    print(
        "MR1 generated: "
        f"eligible={summary['eligible_target_model_call_count']}/"
        f"{summary['target_model_call_count']} verdict={summary['verdict']}"
    )


if __name__ == "__main__":
    main()
