"""Static protocol gates for five fresh FSG4/B3 correctness pairs."""

# pylint: disable=missing-function-docstring,protected-access,duplicate-code

from argparse import Namespace
from pathlib import Path

import pytest

from scripts import run_fsg4_b3_correctness_pairs as pairs
from scripts import probe_fsg4_b3_correctness_pairs_tamper as tamper

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT.parent / "vnncomp2021/benchmarks/cifar10_resnet/onnx/resnet_2b.onnx"
PROPERTY = (
    ROOT.parent
    / "vnncomp2021/benchmarks/cifar10_resnet/vnnlib_properties_pgd_filtered"
    / "resnet2b_pgd_filtered/prop_0_eps_0.008.vnnlib"
)


def _args() -> Namespace:
    return Namespace(
        benchmark_root=ROOT.parent / "vnncomp2021",
        abcrown_root=ROOT.parent / "alpha-beta-CROWN",
        model=MODEL,
        property=PROPERTY,
    )


def test_five_pair_schedule_is_frozen_and_balanced() -> None:
    assert pairs.PAIR_SCHEDULE == (
        ("B2", "B3-C"),
        ("B3-C", "B2"),
        ("B2", "B3-C"),
        ("B3-C", "B2"),
        ("B2", "B3-C"),
    )
    assert (
        sum(
            configuration == "B2"
            for row in pairs.PAIR_SCHEDULE
            for configuration in row
        )
        == 5
    )
    assert (
        sum(
            configuration == "B3-C"
            for row in pairs.PAIR_SCHEDULE
            for configuration in row
        )
        == 5
    )


def test_protocol_binds_content_without_local_absolute_paths() -> None:
    protocol = pairs._protocol(_args())
    pairs._validate_protocol(protocol)

    assert protocol["pair_count"] == 5
    assert protocol["worker_count"] == 10
    assert protocol["timing_admitted"] is False
    assert protocol["performance_claimed"] is False
    encoded = pairs._canonical_json(protocol)
    assert "/home/" not in encoded
    assert "query_wall" not in encoded
    assert "speedup" not in encoded


def test_protocol_outer_resign_cannot_change_schedule() -> None:
    protocol = pairs._protocol(_args())
    protocol["schedule"][0]["positions"] = ["B3-C", "B2"]
    payload = dict(protocol)
    payload.pop("protocol_hash")
    protocol["protocol_hash"] = pairs.canonical_hash(payload)

    with pytest.raises(ValueError, match="protocol differs"):
        pairs._validate_protocol(protocol)


def test_root_file_inventory_includes_nested_manifests_only(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text("root", encoding="utf-8")
    nested = tmp_path / "runs/pair-00/position-0-b2"
    nested.mkdir(parents=True)
    (nested / "manifest.json").write_text("nested", encoding="utf-8")
    (tmp_path / "protocol.json").write_text("protocol", encoding="utf-8")

    files = pairs._all_files(tmp_path)

    assert "manifest.json" not in files
    assert "runs/pair-00/position-0-b2/manifest.json" in files
    assert "protocol.json" in files


def test_outer_resigned_attack_inventory_covers_all_evidence_layers() -> None:
    names = tuple(name for name, _attack in tamper.ATTACKS)

    assert len(names) == 7
    assert any("protocol" in name for name in names)
    assert any("counter" in name for name in names)
    assert any("semantic" in name for name in names)
    assert any("audit" in name for name in names)
    assert any("swap" in name for name in names)
    assert any("delete" in name for name in names)
