#!/usr/bin/env bash
set -euo pipefail

if ! command -v npm >/dev/null 2>&1; then
  echo "Error: npm not found. Please install Node.js/npm first." >&2
  exit 1
fi

for pkg in \
  "@google/gemini-cli@latest" \
  "@anthropic-ai/claude-code@latest" \
  "@openai/codex@latest"; do
  echo "Updating ${pkg}..."
  npm install -g "${pkg}"
done

echo "Done."
