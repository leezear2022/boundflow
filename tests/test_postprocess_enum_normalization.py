from __future__ import annotations

from scripts.postprocess_ablation_jsonl import _normalize_enum_repr


def test_normalize_enum_repr_parses_value() -> None:
    assert _normalize_enum_repr("<MemoryPlanMode.DEFAULT: 'default'>") == "default"


def test_normalize_enum_repr_passthrough() -> None:
    assert _normalize_enum_repr("default") == "default"

