#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

uv run python scripts/build_index.py

echo "=== Eval preview ==="
uv run python scripts/eval.py --preview

echo "Done: build_index + eval preview"
