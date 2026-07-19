"""Deterministic PR-12 split fixtures that do not depend on ignored artifacts."""

import json
from pathlib import Path
from typing import Any

from scripts.start_phase7a_pr12 import _heldout_split
from scripts.start_phase7a_pr12_v2 import build_split


def pr12_v1_split() -> dict[str, Any]:
    """Recreate the code-frozen v1 split."""

    return _heldout_split()


def pr12_v2_split() -> dict[str, Any]:
    """Recreate the code-frozen v2 split from its parent."""

    return build_split(pr12_v1_split())


def write_pr12_v2_split(directory: Path) -> Path:
    """Write a process-shareable split for runner smoke tests."""

    path = directory / "pr12-v2-heldout-split.json"
    path.write_text(
        json.dumps(pr12_v2_split(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
