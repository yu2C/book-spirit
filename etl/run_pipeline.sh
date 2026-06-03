#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== Extract: PDF → Markdown ==="
python3 1_convert_pdf_to_md.py

echo "=== Transform + Load: chunk → embed → Qdrant ==="
python3 3_build_qdrant.py

echo "=== Eval preview ==="
python3 4_test_search_quality.py --preview

echo "✅ ETL pipeline 完成"
