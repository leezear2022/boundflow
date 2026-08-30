"""S4-1A formal artifact replay, tamper, and hygiene gates."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, cast

from scripts import probe_asplos27_s4_1a_buffer_tamper as tamper_tool
from scripts import replay_asplos27_s4_1a_buffer_stdlib as replay_tool

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/asplos27-s4-1a-buffer/resnet2b-prop0-v1"


def test_s4_1a_buffer_artifact_semantic_replay() -> None:
    summary = replay_tool.replay(ARTIFACT)
    assert summary["status"] == replay_tool.FORMAL_STATUS
    assert (
        summary["fresh_process_count"],
        summary["positive_process_count"],
        summary["isolated_fault_process_count"],
    ) == (12, 5, 7)
    assert summary["negative_case_count"] >= 68
    assert summary["source_candidate_binary_pair_count"] == 40
    assert summary["source_candidate_binary_exact_count"] == 40
    assert summary["isolated_fault_clean_count"] == 7
    assert summary["candidate_kernel_launch_count"] == 0
    assert summary["fallback_count"] == 0
    assert summary["retry_count"] == 0
    assert summary["mutation_count"] == 0
    assert summary["timing_recorded"] is False
    assert summary["performance_claimed"] is False


def test_s4_1a_buffer_artifact_outer_resigned_tamper_probe() -> None:
    report = tamper_tool.probe(ARTIFACT)
    cases = cast(list[dict[str, Any]], report["cases"])
    assert report["case_count"] == 10
    assert report["fully_outer_resigned_attack_count"] == 10
    assert report["rejected_count"] == 10
    assert all(item["outer_resign_completed"] for item in cases)
    assert all(item["semantic_recompute_rejected"] for item in cases)


def test_s4_1a_replay_has_stdlib_only_imports() -> None:
    source = (ROOT / "scripts/replay_asplos27_s4_1a_buffer_stdlib.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module.split(".")[0])
    assert imported <= {
        "__future__",
        "argparse",
        "hashlib",
        "json",
        "math",
        "pathlib",
        "typing",
    }


def test_s4_1a_artifact_contains_no_machine_local_paths() -> None:
    for path in ARTIFACT.rglob("*"):
        if not path.is_file():
            continue
        payload = path.read_bytes()
        assert b"/home/lee" not in payload
        assert b"/tmp/" not in payload
