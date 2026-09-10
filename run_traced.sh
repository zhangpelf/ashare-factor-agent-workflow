#!/bin/bash
# LangSmith 追踪版因子挖掘流水线
# 用法: ./run_traced.sh [--stocks N] [--source akshare|yfinance]

set -e

BASE="/Users/zhangpeifu/Library/Mobile Documents/com~apple~CloudDocs/my all memory/factors"
PYTHON="$BASE/.venv/bin/python"

export LANGSMITH_PROJECT="${LANGSMITH_PROJECT:-ashare-factors}"
export LANGSMITH_TRACING="true"

echo "=== LangSmith Traced Factor Pipeline ==="
echo "  Project: $LANGSMITH_PROJECT"
echo "  API Key: ${LANGSMITH_API_KEY:+✅已设置}"
echo ""

$PYTHON "$BASE/src/run_traced_pipeline.py" "$@"

echo ""
echo "查看 trace:"
echo "  langsmith trace list --project $LANGSMITH_PROJECT --limit 5 --full"
echo "  langsmith trace list --project $LANGSMITH_PROJECT --error"
