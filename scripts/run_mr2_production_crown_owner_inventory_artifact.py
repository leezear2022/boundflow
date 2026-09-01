#!/usr/bin/env python3
"""Generate or replay the MR2 production CROWN owner inventory artifact."""

# pylint: disable=duplicate-code,missing-function-docstring,wrong-import-position
# pylint: disable=too-many-locals,too-many-statements

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, cast, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "artifacts/measurement-recovery/mr2-production-crown-subgraph-owner-inventory-v1"
)
INPUTS = {
    "inventory.json": (
        ROOT
        / "artifacts/fsg2-rvir-v3/resnet2b-production-state-inventory-v2/inventory.json",
        "ab0595bb002b79d80be8b78abd7a795a8aa634b17c9a8524df5a1b9b5fe19e06",
    ),
    "p_bundle.json": (
        ROOT / "artifacts/r3-structured-owner/r3-0-contract-v2/bundle.json",
        "2f6aaca66be142db2a72c99f601085026da6e0b7a9d36c72f486b998c1584a25",
    ),
    "p_contract_manifest.json": (
        ROOT / "artifacts/r3-structured-owner/r3-0-contract-v2/manifest.json",
        "a059599a1a589ea1c660191b276cfc6a0162d6c1460c87370d03b70a6fada92e",
    ),
    "p_trajectory_summary.json": (
        ROOT / "artifacts/r3-structured-owner/r3-d2b-correctness-v1/summary.json",
        "d78c043e8e4185626f0b1f95fd3a40c99196bd1a051bf4f406aa2d023193f894",
    ),
    "p_trajectory_manifest.json": (
        ROOT / "artifacts/r3-structured-owner/r3-d2b-correctness-v1/manifest.json",
        "d43db41d3a6b16dc9a73306581499fdea521bd8508c351d325d27536ab28359f",
    ),
    "s_correctness_summary.json": (
        ROOT
        / "artifacts/r3-structured-owner/r3-3-active-beta-correctness-v1/summary.json",
        "819612e49cafd8ebfdfdf038dea23aa50d8fb14fcea689b10c11a22372711fc1",
    ),
    "s_correctness_manifest.json": (
        ROOT
        / "artifacts/r3-structured-owner/r3-3-active-beta-correctness-v1/manifest.json",
        "3413c578ca3dae83f6d5d18683124a758c8fb12f8e6801b4f8b719d1539f0a4c",
    ),
    "p_cibc_summary.json": (
        ROOT / "artifacts/fsg4-b4b2-v2-cibc-formal/resnet2b-prop0-v1/summary.json",
        "8e76c2115b169a2b6469e9ca87111bba09ceef53e52e7a18fdaeeef7c238a70a",
    ),
    "p_cibc_manifest.json": (
        ROOT / "artifacts/fsg4-b4b2-v2-cibc-formal/resnet2b-prop0-v1/manifest.json",
        "372651d3c0f9f516d6af6812a537ae9d33e7323e4827d9d6de473225afae2511",
    ),
    "p_v1_summary.json": (
        ROOT
        / "artifacts/fsg4-b4b2-b2-5-formal-microphysics/resnet2b-prop0-v1/summary.json",
        "b58c6b4e26d41ac94c6b173a8e7e5c8e84f404384a4af8171ed90805cf52b995",
    ),
    "p_v1_manifest.json": (
        ROOT
        / "artifacts/fsg4-b4b2-b2-5-formal-microphysics/resnet2b-prop0-v1/manifest.json",
        "b84d74d9b8398ca7593b78481da613bb8fb0acb6196dd2dd5fa26b364ac813d7",
    ),
    "mr1_summary.json": (
        ROOT
        / "artifacts/measurement-recovery/mr1-static-same-solver-eligibility-v1/summary.json",
        "43ff5ee6bc7c967bab8f0dd5affbf29d0c8dd3bbcca2eae71bb614e18fea18d3",
    ),
    "mr1_manifest.json": (
        ROOT
        / "artifacts/measurement-recovery/mr1-static-same-solver-eligibility-v1/manifest.json",
        "adb7ba6e303a67466f66cd40ae0cd3e02a0f4f02331904f13977832a666ffd7f",
    ),
}
CODE_PATHS = (
    "boundflow/runtime/mr2_production_crown_owner_inventory.py",
    "scripts/run_mr2_production_crown_owner_inventory_artifact.py",
    "scripts/probe_mr2_production_crown_owner_inventory_tamper.py",
    "gemini_doc/BOUNDFLOW_MR2_PRODUCTION_CROWN_SUBGRAPH_OWNER_INVENTORY_PLAN_2026_08_26.md",
)

from boundflow.runtime.mr2_production_crown_owner_inventory import (  # noqa: E402
    GATE_ORDER,
    canonical_hash,
    derive_site_ledger,
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
        raise TypeError("MR2 JSON root differs")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write(path: Path, value: object) -> None:
    path.write_text(_canonical(value) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text("".join(_canonical(row) + "\n" for row in rows), encoding="utf-8")


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _inputs(source: Path) -> dict[str, Mapping[str, Any]]:
    return {
        "inventory": _json(source / "inventory.json"),
        "p_bundle": _json(source / "p_bundle.json"),
        "p_trajectory": _json(source / "p_trajectory_summary.json"),
        "s_correctness": _json(source / "s_correctness_summary.json"),
        "p_cibc": _json(source / "p_cibc_summary.json"),
        "p_v1": _json(source / "p_v1_summary.json"),
        "mr1": _json(source / "mr1_summary.json"),
    }


def _protocol(revision: str, source: Path) -> dict[str, object]:
    input_hashes = {name: _file_hash(source / name) for name in sorted(INPUTS)}
    expected = {name: digest for name, (_, digest) in INPUTS.items()}
    if input_hashes != expected:
        raise ValueError("MR2 frozen input digest differs")
    result: dict[str, object] = {
        "schema_version": "boundflow.mr2-production-crown-owner-inventory-protocol/v1",
        "source_revision": revision,
        "input_sha256": input_hashes,
        "site_order": ["P:25/Conv_8", "S:31/Gemm_14"],
        "gate_order": list(GATE_ORDER),
        "selection_order": ["P:25/Conv_8", "S:31/Gemm_14"],
        "read_only": True,
        "solver_executed": False,
        "gpu_executed": False,
        "timing_collected": False,
        "same_solver_open": False,
        "r2_open": False,
        "performance_claimed": False,
        "code_revision": {path: _file_hash(ROOT / path) for path in CODE_PATHS},
    }
    result["protocol_hash"] = canonical_hash(result)
    return result


def _derive(source: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    ledger = derive_site_ledger(_inputs(source))
    return ledger, derive_summary(ledger)


def _gap_matrix(ledger: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    matrix: dict[str, object] = {}
    for row in ledger:
        gates = cast(Mapping[str, Mapping[str, Any]], row["gates"])
        matrix[str(row["site_id"])] = {
            gate: gates[gate]["status"] for gate in GATE_ORDER
        }
    matrix["matrix_hash"] = canonical_hash(matrix)
    return matrix


def generate(output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"MR2 artifact exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="mr2-owner-inventory-", dir=output.parent))
    try:
        source = temporary / "source"
        source.mkdir()
        for name, (path, digest) in INPUTS.items():
            if _file_hash(path) != digest:
                raise ValueError(f"MR2 source input differs: {name}")
            shutil.copyfile(path, source / name)
        protocol = _protocol(_git("rev-parse", "HEAD"), source)
        ledger, summary = _derive(source)
        gap_matrix = _gap_matrix(ledger)
        _write(temporary / "protocol.json", protocol)
        _write_jsonl(temporary / "site_ledger.jsonl", ledger)
        _write(temporary / "gap_matrix.json", gap_matrix)
        _write(temporary / "summary.json", summary)
        (temporary / "replay_stdout.txt").write_text(
            f"MR2 replay PASS: selected={summary['selected_site']} route={summary['route']}\n",
            encoding="utf-8",
        )
        files = {
            str(path.relative_to(temporary)): _file_hash(path)
            for path in sorted(temporary.rglob("*"))
            if path.is_file()
        }
        manifest: dict[str, object] = {
            "schema_version": "boundflow.mr2-production-crown-owner-inventory-manifest/v1",
            "source_revision": protocol["source_revision"],
            "protocol_hash": protocol["protocol_hash"],
            "matrix_hash": gap_matrix["matrix_hash"],
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
        == "boundflow.mr2-production-crown-owner-inventory-manifest/v1"
    )
    if not schema_ok or manifest_hash != canonical_hash(unsigned):
        raise ValueError("MR2 manifest differs")
    files = manifest.get("files")
    if not isinstance(files, dict) or any(
        _file_hash(artifact / str(name)) != digest for name, digest in files.items()
    ):
        raise ValueError("MR2 file digest differs")
    protocol = _json(artifact / "protocol.json")
    protocol_unsigned = dict(protocol)
    protocol_hash = protocol_unsigned.pop("protocol_hash", None)
    if protocol_hash != canonical_hash(
        protocol_unsigned
    ) or protocol_hash != manifest.get("protocol_hash"):
        raise ValueError("MR2 protocol differs")
    expected_inputs = {name: digest for name, (_, digest) in INPUTS.items()}
    if protocol_unsigned.get("input_sha256") != expected_inputs:
        raise ValueError("MR2 protocol input binding differs")
    source = artifact / "source"
    if {name: _file_hash(source / name) for name in sorted(INPUTS)} != expected_inputs:
        raise ValueError("MR2 copied input digest differs")
    ledger, summary = _derive(source)
    if _jsonl(artifact / "site_ledger.jsonl") != ledger:
        raise ValueError("MR2 site ledger derivation differs")
    matrix = _gap_matrix(ledger)
    if _json(artifact / "gap_matrix.json") != matrix:
        raise ValueError("MR2 gap matrix derivation differs")
    if _json(artifact / "summary.json") != summary:
        raise ValueError("MR2 summary derivation differs")
    if (
        manifest.get("matrix_hash") != matrix["matrix_hash"]
        or manifest.get("summary_hash") != summary["summary_hash"]
    ):
        raise ValueError("MR2 manifest semantic binding differs")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args()
    if args.replay is not None:
        summary = replay(args.replay.resolve())
        print(
            f"MR2 replay PASS: selected={summary['selected_site']} route={summary['route']}"
        )
        return
    summary = generate(args.output.resolve())
    print(
        f"MR2 generated: selected={summary['selected_site']} route={summary['route']}"
    )


if __name__ == "__main__":
    main()
