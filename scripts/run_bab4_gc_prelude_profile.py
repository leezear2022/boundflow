#!/usr/bin/env python3
"""Run three fresh prepared-GC BAB4 complete-prelude attribution pairs."""

from __future__ import annotations

from scripts import run_bab4_complete_prelude_profile as implementation

CONTROL = "B4-A-GC"
CANDIDATE = "BAB4-GC"
PAIR_ORDERS = (
    (CONTROL, CANDIDATE),
    (CANDIDATE, CONTROL),
    (CONTROL, CANDIDATE),
)


def configure() -> None:
    """Install prepared-GC configurations into the shared attribution runner."""

    implementation.CONTROL = CONTROL
    implementation.CANDIDATE = CANDIDATE
    implementation.PAIR_ORDERS = PAIR_ORDERS


def main() -> None:
    """Run the prepared-GC attribution protocol."""

    configure()
    implementation.main()


if __name__ == "__main__":
    main()
