"""Frozen assertions for the B4-B2 B2-5 formal artifact."""

from pathlib import Path

from scripts.run_fsg4_b4b2_b2_5_artifact import load_json, replay

ARTIFACT = Path(
    "artifacts/fsg4-b4b2-b2-5-formal-microphysics/resnet2b-prop0-v1"
)


def test_b2_5_formal_artifact_replays_to_v1_physics_no_go() -> None:
    result = replay(ARTIFACT, recompile=False)
    summary = load_json(ARTIFACT / "summary.json")
    assert result == {
        "status": "replay-pass",
        "manifest_hash": (
            "c2d3d30dd7606f1fe777ed209bb5d89559ffd38e4497749c67a91da57046f518"
        ),
        "summary_hash": (
            "f873d4717b7029aa045836ed276d5cd4c752696b82100ba5f1f739cd99f4b163"
        ),
        "timing_admitted": False,
        "result_status": "validated-no-go-b4-b2-v1-physics",
        "recompile_receipt": None,
    }
    assert summary["winner_ordinal"] == 11
    assert summary["paired_speedup_geomean"] == 0.42484238749783887
    assert summary["bootstrap_95_lower"] == 0.4031569161542472
    assert summary["worst_worker_speedup"] == 0.3776925294408135
    assert summary["maximum_allocated_ratio"] == 0.4746376811594203
    assert summary["maximum_reserved_ratio"] == 1.0
    assert summary["b4b3_open"] is False


def test_b2_5_formal_artifact_freezes_real_kernel_inventory() -> None:
    summary = load_json(ARTIFACT / "summary.json")
    inventory = summary["kernel_inventory"]
    assert inventory["forward_kernel_count"] == 3
    assert inventory["backward_kernel_count"] == 3
    assert inventory["total_kernel_count"] == 6
    assert inventory["shared_memory_token_count"] == 0
    assert inventory["vector_token_count"] == 0
    assert inventory["half_token_count"] == 0
