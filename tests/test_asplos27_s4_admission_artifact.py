"""S4-0 formal artifact replay and tamper closure tests."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, cast

from scripts import probe_asplos27_s4_0_admission_tamper as tamper_tool
from scripts import replay_asplos27_s4_0_admission_stdlib as replay_tool

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/asplos27-s4-admission/resnet2b-prop0-v1"


def test_s4_admission_artifact_semantic_replay() -> None:
    summary = replay_tool.replay(ARTIFACT)
    assert summary["status"] == replay_tool.FORMAL_STATUS
    assert summary["fresh_process_count"] == 5
    assert summary["negative_case_count"] == 63
    assert summary["formal_counts"] == replay_tool.EXPECTED_COUNTS
    assert summary["candidate_kernel_launch_count"] == 0
    assert summary["candidate_cuda_allocation_count"] == 0
    assert summary["provider_bound_callback_count"] == 0
    assert summary["buffer_prepare_count"] == 0
    assert summary["mutation_count"] == 0
    assert summary["timing_recorded"] is False
    assert summary["performance_claimed"] is False
    assert summary["process_global_query_exclusivity_validated"] is False


def test_s4_admission_artifact_fully_resigned_tamper_probe() -> None:
    report = tamper_tool.probe(ARTIFACT)
    cases = cast(list[dict[str, Any]], report["cases"])
    assert report["case_count"] == 10
    assert report["fully_outer_resigned_attack_count"] == 10
    assert report["rejected_count"] == 10
    assert all(row["outer_resign_completed"] for row in cases)
    assert all(row["rejected"] for row in cases)


def test_s4_replay_script_has_stdlib_only_imports() -> None:
    source = (ROOT / "scripts/replay_asplos27_s4_0_admission_stdlib.py").read_text(
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


def test_s4_artifact_contains_no_machine_local_paths() -> None:
    for path in ARTIFACT.rglob("*"):
        if not path.is_file():
            continue
        payload = path.read_bytes()
        assert b"/home/lee" not in payload
        assert b"/tmp/" not in payload
