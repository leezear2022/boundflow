"""Frozen FSG4/B4-A formal timing artifact gates."""

# pylint: disable=missing-function-docstring,protected-access

import json
from pathlib import Path

from scripts import run_fsg4_b4a_formal_timing as timing

ARTIFACT = Path("artifacts/fsg4-b4a-formal-timing/resnet2b-prop0-v5")
TAMPER_REPORT = Path(
    "artifacts/fsg4-b4a-formal-timing/resnet2b-prop0-v5-tamper-report.json"
)


def test_b4a_formal_artifact_replays_to_preregistered_no_go() -> None:
    summary, replay = timing._verify(ARTIFACT)
    assert replay == {
        "status": "validated-no-go-b4a-performance",
        "run_count": 24,
        "summary_hash": (
            "46360e41e87917f2f5f801733fe6f13f10591cb63eff9b2675db37320a0bc3d7"
        ),
        "core_wall_geomean": 1.0189949992169265,
        "query_wall_worst_pair": 0.996947022444439,
        "performance_candidate_admitted": False,
        "performance_claimed": False,
    }
    assert summary["correctness_passed"] is True
    assert summary["environment_passed"] is True
    assert summary["profile_closed"] is True
    assert summary["core_wall_geomean_passed"] is False
    assert summary["query_wall_worst_pair_passed"] is True


def test_b4a_formal_artifact_freezes_pairs_semantics_and_memory() -> None:
    pairs = [
        json.loads(line)
        for line in (ARTIFACT / "paired_runs.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(pairs) == 6
    assert [row["ratios"]["core_wall_ns"] for row in pairs] == [
        1.021616992365494,
        1.0186565141565094,
        1.0378662482148024,
        1.0165780728306277,
        1.0001952774637546,
        1.0194114684882396,
    ]
    assert (
        max(row["export_pair"]["maximum_absolute_difference"] for row in pairs)
        == 4.410743713378906e-06
    )
    assert all(
        row["environment_admitted"] is True
        and row["semantic_failures"] == []
        and row["export_pair"]["all_sign_exact"] is True
        and row["export_pair"]["tensor_count"] == 19
        and row["ratios"]["peak_allocated_bytes"] == 1.0
        and row["ratios"]["peak_reserved_bytes"] == 1.0
        for row in pairs
    )


def test_b4a_formal_artifact_freezes_outer_resigned_tamper_report() -> None:
    report = json.loads(TAMPER_REPORT.read_text(encoding="utf-8"))
    assert report["artifact_source_git_head"] == (
        "46a8493557c49f327df4e70d7cdd7649227b14b9"
    )
    assert report["clean_summary_hash"] == (
        "46360e41e87917f2f5f801733fe6f13f10591cb63eff9b2675db37320a0bc3d7"
    )
    assert report["attack_count"] == 14
    assert report["outer_resigned_attack_count"] == 14
    assert report["all_rejected"] is True
    assert all(row["rejected"] is True for row in report["attacks"])
    assert report["performance_claimed"] is False
