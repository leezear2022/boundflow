"""R3-0 artifact preflight and integrity helper tests."""

# pylint: disable=missing-function-docstring,protected-access

from scripts.run_r3_structured_owner_artifact import _parse_porcelain


def test_porcelain_parser_preserves_dot_prefixed_path() -> None:
    assert _parse_porcelain(" M .docops/ev.jsonl\n") == (".docops/ev.jsonl",)


def test_porcelain_parser_uses_rename_destination() -> None:
    assert _parse_porcelain("R  old.md -> .hidden/new.md\n") == (".hidden/new.md",)
