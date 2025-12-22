#!/usr/bin/env bash
set -euo pipefail

# Thin wrapper; keep logic in Python for testability.
python scripts/run_phase5d_artifact.py "$@"
